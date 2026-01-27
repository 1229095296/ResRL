import logging
import os

import numpy as np
import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, compute_policy_loss, get_policy_loss_fn, kl_penalty
from verl.utils.device import get_device_name, is_cuda_available, is_npu_available
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outputs_and_unpad, ulysses_pad, ulysses_pad_and_slice_inputs
from verl.workers.actor import BasePPOActor

if is_cuda_available:
    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
elif is_npu_available:
    from transformers.integrations.npu_flash_attention import index_first_axis, pad_input, rearrange, unpad_input


__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class DataParallelPPOActor(BasePPOActor):
    def __init__(self, config, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer

        self.use_remove_padding = self.config.get("use_remove_padding", False)
        if torch.distributed.get_rank() == 0:
            print(f"Actor use_remove_padding={self.use_remove_padding}")
        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        if torch.distributed.get_rank() == 0:
            print(f"Actor use_fused_kernels={self.use_fused_kernels}")

        self.use_svd_token_weighting = self.config.get("use_svd_token_weighting", False)
        self.svd_rank = self.config.get("svd_rank", 64)
        self.svd_max_pos_tokens = self.config.get("svd_max_pos_tokens", 8192)
        self.rollout_n = self.config.get("rollout_n", 8)

        if torch.distributed.get_rank() == 0:
            print(f"Actor use_svd_token_weighting={self.use_svd_token_weighting}")
            if self.use_svd_token_weighting:
                print(f"Actor svd_rank={self.svd_rank}, rollout_n={self.rollout_n}")

        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        if self.config.entropy_from_logits_with_chunking:
            entropy_from_logits = verl_F.entropy_from_logits_with_chunking
        else:
            entropy_from_logits = verl_F.entropy_from_logits

        self.compute_entropy_from_logits = (
            torch.compile(entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)
            else entropy_from_logits
        )
        self.device_name = get_device_name()

        self.think_token_ids = self.config.get("think_token_ids", [151648, 151649, 151667, 151668])

    def _create_think_only_mask(self, input_ids: torch.Tensor, response_start_idx: int) -> torch.Tensor:
        bsz, seqlen = input_ids.shape
        device = input_ids.device

        think_only_mask = torch.zeros((bsz, seqlen), dtype=torch.bool, device=device)

        think_start_ids = torch.tensor(
            [self.think_token_ids[0], self.think_token_ids[2]], device=device
        )
        think_end_ids = torch.tensor(
            [self.think_token_ids[1], self.think_token_ids[3]], device=device
        )

        for b in range(bsz):
            in_think = False
            for i in range(response_start_idx, seqlen):
                token_id = input_ids[b, i].item()

                if token_id in think_start_ids:
                    in_think = True
                    think_only_mask[b, i] = True
                elif token_id in think_end_ids:
                    think_only_mask[b, i] = True
                    in_think = False
                elif in_think:
                    think_only_mask[b, i] = True

        return think_only_mask

    def _forward_micro_batch(
        self, micro_batch, temperature, calculate_entropy=False) -> tuple[torch.Tensor, torch.Tensor]:
        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            if "image_bound" in micro_batch["multi_modal_inputs"][0]:
                for key in micro_batch["multi_modal_inputs"][0].keys():
                    multi_modal_inputs[key] = [inputs[key] for inputs in micro_batch["multi_modal_inputs"]]
            else:
                for key in micro_batch["multi_modal_inputs"][0].keys():
                    multi_modal_inputs[key] = torch.cat(
                        [inputs[key] for inputs in micro_batch["multi_modal_inputs"]], dim=0
                    )

        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            entropy = None
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 3, seqlen) -> (3, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, cu_seqlens, *_ = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask
                )
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                if position_ids.dim() == 3:
                    position_ids_rmpad = (
                        index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                        .transpose(0, 1)
                        .unsqueeze(1)
                    )  # (3, bsz, seqlen) -> (3, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(
                        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                    ).transpose(0, 1)

                if "image_bound" in multi_modal_inputs:
                    from verl.utils.dataset.vision_utils import process_multi_modal_inputs_for_minicpmo

                    multi_modal_inputs = process_multi_modal_inputs_for_minicpmo(
                        input_ids, attention_mask, position_ids, cu_seqlens, multi_modal_inputs
                    )

                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                if self.use_ulysses_sp:
                    is_vlm_model = "multi_modal_inputs" in micro_batch.keys()
                    if is_vlm_model:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                extra_args["output_hidden_states"] = True

                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                    entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)

                else:
                    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                    logits_rmpad.div_(temperature)

                    inplace_backward = True
                    if calculate_entropy:
                        inplace_backward = False
                    log_probs = logprobs_from_logits(
                        logits=logits_rmpad,
                        labels=input_ids_rmpad_rolled,
                        inplace_backward=inplace_backward,
                    )

                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)
                        else:
                            entropy_rmpad = torch.utils.checkpoint.checkpoint(
                                self.compute_entropy_from_logits, logits_rmpad
                            )

                if self.use_ulysses_sp:
                    log_probs = gather_outputs_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    if calculate_entropy:
                        entropy_rmpad = gather_outputs_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )

                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)

            else:  # not using rmpad and no ulysses sp
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                extra_args["output_hidden_states"] = True

                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                ) 

                if self.use_fused_kernels:
                    log_probs = output.log_probs[:, -response_length - 1 : -1]
                    entropy = output.entropy[:, -response_length - 1 : -1]  # (bsz, response_length)

                else:
                    logits = output.logits

                    logits.div_(temperature)
                    logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                    log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)
                        else:
                            entropy = torch.utils.checkpoint.checkpoint(verl_F.entropy_from_logits, logits)
            

            if output.hidden_states is None:
                raise RuntimeError("Please ensure output_hidden_states=True is passed to the model.")
            last_hidden = output.hidden_states[-2].squeeze(0) # (total_nnz, hidden size)
            if self.use_ulysses_sp:
                    last_hidden = gather_outputs_and_unpad(
                        last_hidden,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size, 
                    )
            full_last_hidden = pad_input(
                hidden_states=last_hidden,
                indices=indices,
                batch=batch_size,
                seqlen=seqlen,
            )
            response_last_hidden = full_last_hidden[:, -response_length - 1: -1]  # (bsz, response_length, hidden_size)
            prompt_last_hidden = full_last_hidden[:, : -response_length - 1]  # (bsz, prompt_length, hidden_size)
            
            return entropy, log_probs, response_last_hidden, prompt_last_hidden


    def _compute_svd_token_weights(
        self, 
        response_hidden: torch.Tensor,  # (bsz, response_length, hidden_size)
        response_mask: torch.Tensor,    # (bsz, response_length)
        is_positive_sample: torch.Tensor,  # (bsz,)
        prompt_ids: torch.Tensor = None,  # (bsz,) optional prompt group IDs
        input_ids: torch.Tensor = None,  # (bsz, seqlen) token ids, used to detect think labels
        response_length: int = None,  # response length, used to locate think label range
    ) -> torch.Tensor:
        response_hidden = response_hidden.detach()

        bsz, response_length, hidden_size = response_hidden.shape
        device = response_hidden.device
        eps = 1e-6
        min_w = 0.1
        q_low = float(getattr(self, "svd_q_low", 0.2))
        q_high = float(getattr(self, "svd_q_high", 0.8))
        max_pos_tokens = int(getattr(self, "svd_max_pos_tokens", 8192))
        niter = int(getattr(self, "svd_pca_niter", 4))

        token_weights = torch.ones_like(response_mask, dtype=torch.float32, device=device)

        think_mask = None
        if input_ids is not None and response_length is not None:
            response_start_idx = input_ids.shape[1] - response_length - 1
            think_mask_full = self._create_think_only_mask(input_ids, response_start_idx)
            think_mask = think_mask_full[:, -response_length - 1: -1]

        if not is_positive_sample.any():
            return token_weights

        if prompt_ids is None:
            prompt_ids = torch.arange(bsz, device=device) // self.rollout_n
        else:
            if not isinstance(prompt_ids, torch.Tensor):
                prompt_ids = torch.tensor(prompt_ids, device=device)
            else:
                prompt_ids = prompt_ids.to(device)

        unique_prompts = torch.unique(prompt_ids)

        response_mask_bool = response_mask > 0

        for pid in unique_prompts:
            group_mask = (prompt_ids == pid)
            pos_mask = group_mask & is_positive_sample
            neg_mask = group_mask & (~is_positive_sample)
            num_pos = int(pos_mask.sum())
            num_neg = int(neg_mask.sum())
            print(f"[SVD] prompt_id={int(pid)}: {num_pos}个正样本, {num_neg}个负样本", end="")
            
            if not pos_mask.any():
                print(" NO positive samples")
                continue
            if not neg_mask.any():
                print(" NO negative samples")
                continue

            pos_hidden = response_hidden[pos_mask]                 # (Bp, T, H)
            pos_m = response_mask_bool[pos_mask]                   # (Bp, T)
            if pos_hidden.numel() == 0:
                print(f"  [SVD] prompt_id={int(pid)}: NO pos_hidden")
                continue

            if think_mask is not None:
                think_mask_pos = think_mask[pos_mask]  # (Bp, T)
                pos_m = pos_m & (~think_mask_pos)

            num_total_tokens = pos_hidden.shape[0] * pos_hidden.shape[1]
            num_valid_tokens = pos_m.sum().item()
            pos_hidden_valid = pos_hidden[pos_m]
            
            if pos_hidden_valid.shape[0] < 4:
                print(f"  [SVD] prompt_id={int(pid)}: NO valid positive samples ({pos_hidden_valid.shape[0]} < 4, total tokens={num_total_tokens}, valid tokens={num_valid_tokens})")
                continue
            
            assert pos_hidden_valid.shape[0] == num_valid_tokens, \
                f"mask error: pos_hidden_valid.shape[0]={pos_hidden_valid.shape[0]}, num_valid_tokens={num_valid_tokens}"

            M = pos_hidden_valid.shape[0]
            if M > max_pos_tokens:
                indices = torch.linspace(0, M - 1, steps=max_pos_tokens, device=pos_hidden_valid.device).long()
                pos_hidden_valid = pos_hidden_valid[indices]
            X = pos_hidden_valid.float()
            
            X_norm = torch.nn.functional.layer_norm(X, normalized_shape=(X.shape[-1],))  # (M, H)

            mu = X_norm.mean(dim=0, keepdim=True)                  # (1, H)
            Xc = X_norm - mu                                       # centered

            k = min(int(self.svd_rank), Xc.shape[0] - 1, Xc.shape[1])
            if k < 1:
                print(f"  [SVD] prompt_id={int(pid)}: NO - k={k} < 1 (rank too small)")
                continue

            try:
                _, _, V = torch.pca_lowrank(Xc, q=k, niter=niter)
                V_k = V  # (H, k)
                print(f"  [SVD] prompt_id={int(pid)}: PCA success - rank={k}, positive samples={pos_hidden_valid.shape[0]}")
            except Exception as e:
                print(f"  [SVD] prompt_id={int(pid)}: PCA failed - {str(e)}")
                continue

            neg_indices = torch.where(neg_mask)[0]
            neg_hidden = response_hidden[neg_mask].float()         # (Bn, T, H)
            neg_m = response_mask_bool[neg_mask]                   # (Bn, T)

            if neg_hidden.numel() == 0:
                print(f"  [SVD] prompt_id={int(pid)}: NO - neg_hidden is empty")
                continue

            neg_hidden_norm = torch.nn.functional.layer_norm(neg_hidden, normalized_shape=(neg_hidden.shape[-1],))  # (Bn, T, H)

            Hc = neg_hidden_norm - mu                              # (Bn, T, H)
            proj = (Hc @ V_k) @ V_k.T                               # (Bn, T, H)
            resid = Hc - proj                                       # (Bn, T, H)

            if think_mask is not None:
                think_mask_neg = think_mask[neg_mask]  # (Bn, T)
                non_think_valid_mask = neg_m & (~think_mask_neg)
            
                resid_non_think = resid[non_think_valid_mask]
                distances_non_think = (resid_non_think * resid_non_think).sum(dim=-1) / float(hidden_size)
                vd_non_think = distances_non_think  # (M,)

                if vd_non_think.numel() == 0:
                    print(f"  [SVD] prompt_id={int(pid)}: NO - non-think part has no valid token distance")
                    continue

                print(f"  [SVD] prompt_id={int(pid)}: compute weights - {neg_hidden.shape[0]} negative samples, {vd_non_think.numel()} non-think tokens")
                think_count = think_mask_neg.sum().item()
                print(f"  [SVD] prompt_id={int(pid)}: {think_count} think tokens (weights set to -1, fully masked)")
            else:
                distances = (resid * resid).sum(dim=-1) / float(hidden_size)  # (Bn, T)
                vd = distances[neg_m]
                if vd.numel() == 0:
                    print(f"  [SVD] prompt_id={int(pid)}: NO - no valid token distance")
                    continue
                vd_non_think = vd
                print(f"  [SVD] prompt_id={int(pid)}: compute weights - {neg_hidden.shape[0]} negative samples, {vd.numel()} valid tokens")

            weights = torch.ones((neg_hidden.shape[0], neg_hidden.shape[1]), dtype=torch.float32, device=device)

            if think_mask is not None:
                if vd_non_think.numel() >= 32:
                    lo = torch.quantile(vd_non_think, q_low)
                    hi = torch.quantile(vd_non_think, q_high)
                    range_val = (hi - lo).clamp_min(eps)

                    if range_val < eps * 10:
                        print(f"    all distances are almost the same (range={range_val.item():.6f}), non-think part uses fixed weight 0.55")
                        weights[non_think_valid_mask] = (min_w + 1.0) / 2.0  # 0.55
                    else:
                        u = ((vd_non_think - lo) / range_val).clamp(0.0, 1.0)  # (M,)
                        w_non_think = min_w + (1.0 - min_w) * u  # (M,)
                        weights[non_think_valid_mask] = w_non_think

                        num_zero_dist = (vd_non_think < 1e-6).sum().item()
                        num_small_dist = ((vd_non_think >= 1e-6) & (vd_non_think < lo)).sum().item()
                        num_mid_dist = ((vd_non_think >= lo) & (vd_non_think <= hi)).sum().item()
                        num_large_dist = (vd_non_think > hi).sum().item()
                        print(f"    non-think distance distribution: ~0({num_zero_dist}), <lo({num_small_dist}), [lo,hi]({num_mid_dist}), >hi({num_large_dist})")
                else:
                    lo = torch.quantile(vd_non_think, q_low)
                    hi = torch.quantile(vd_non_think, q_high)
                    range_val = (hi - lo).clamp_min(eps)

                    u = ((vd_non_think - lo) / range_val).clamp(0.0, 1.0)  # (M,)
                    w_non_think = min_w + (1.0 - min_w) * u  # (M,)
                    weights[non_think_valid_mask] = w_non_think


                print(f"    non-think part: compute weights based on SVD distance | think part: weights set to -1 (fully masked)")
            else:
                distances = (resid * resid).sum(dim=-1) / float(hidden_size)  # (Bn, T)
                
                if vd_non_think.numel() >= 32:
                    lo = torch.quantile(vd_non_think, q_low)
                    hi = torch.quantile(vd_non_think, q_high)
                    range_val = (hi - lo).clamp_min(eps)

                    if range_val < eps * 10:
                        print(f"    all distances are almost the same (range={range_val.item():.6f}), non-think part uses fixed weight 0.55")
                        weights = torch.full_like(distances, (min_w + 1.0) / 2.0)
                    else:
                        u = ((distances - lo) / range_val).clamp(0.0, 1.0)
                        weights = min_w + (1.0 - min_w) * u

                else:
                    lo = torch.quantile(vd_non_think, q_low)
                    hi = torch.quantile(vd_non_think, q_high)
                    range_val = (hi - lo).clamp_min(eps)
            
                    u = ((distances - lo) / range_val).clamp(0.0, 1.0)
                    weights = min_w + (1.0 - min_w) * u


            weights = weights.clamp(min_w, 1.0)
            
            neg_m_float = neg_m.to(weights.dtype)
            weights = weights * neg_m_float + (1.0 - neg_m_float) 

            if think_mask is not None:
                think_mask_neg = think_mask[neg_mask]  # (Bn, T)
                weights[think_mask_neg] = -1.0 

            for neg_idx_local, neg_idx_global in enumerate(neg_indices):
                neg_idx_global_item = neg_idx_global.item()
                sample_weights = weights[neg_idx_local].detach().cpu()  # (T,)
                sample_think_mask = think_mask_neg[neg_idx_local].cpu() if think_mask is not None else None
                valid_mask = neg_m[neg_idx_local].cpu()
                
                if think_mask is not None:
                    think_valid_mask = valid_mask & sample_think_mask
                    non_think_valid_mask = valid_mask & (~sample_think_mask)
                    valid_weights_non_think = sample_weights[non_think_valid_mask]
                    valid_weights_think = sample_weights[think_valid_mask]

                    valid_weights_all = sample_weights[valid_mask]

                    if valid_weights_non_think.numel() > 0:
                        num_print = min(5, valid_weights_non_think.numel())
                        print(f"    negative sample {neg_idx_global_item}: non-think weights [{valid_weights_non_think.min().item():.3f}, {valid_weights_non_think.max().item():.3f}], mean={valid_weights_non_think.mean().item():.3f}, number={valid_weights_non_think.numel()}", end="")
                        if num_print > 0:
                            print(f" | samples: ", end="")
                            for i in range(num_print):
                                print(f"w={valid_weights_non_think[i].item():.3f}", end="  ")
                        print(f" | think weights number={valid_weights_think.numel()}")
                    else:
                        print(f"    negative sample {neg_idx_global_item}: [NOTE] no non-think tokens, all {valid_weights_all.numel()} tokens are think (weights fixed to -1.0)")
                else:
                    sample_distances = distances[neg_idx_local].detach().cpu()  # (T,)
                    valid_weights = sample_weights[valid_mask]
                    valid_distances = sample_distances[valid_mask]

                    if valid_weights.numel() > 0:
                        num_print = min(5, valid_weights.numel())
                        print(f"    negative sample {neg_idx_global_item}: weights [{valid_weights.min().item():.3f}, {valid_weights.max().item():.3f}], mean={valid_weights.mean().item():.3f}", end=" | ")
                        for i in range(num_print):
                            print(f"dist={valid_distances[i].item():.6f}→w={valid_weights[i].item():.3f}", end="  ")
                        print()

            token_weights[neg_indices] = weights

        if think_mask is not None:
            for i in range(bsz):
                if not is_positive_sample[i]:
                    token_weights[i, think_mask[i]] = -1.0

        return token_weights


    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        if not torch.isfinite(grad_norm):
            print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
            self.actor_optimizer.zero_grad()
        else:
            self.actor_optimizer.step()
        return grad_norm

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy=False) -> torch.Tensor:
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []
        for micro_batch in micro_batches:
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                entropy, log_probs = self._forward_micro_batch(
                    model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                )
            log_probs_lst.append(log_probs)
            if calculate_entropy:
                entropy_lst.append(entropy)

        log_probs = torch.concat(log_probs_lst, dim=0)
        entropys = None
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)

        if use_dynamic_bsz:
            log_probs = restore_dynamic_batch(log_probs, batch_idx_list)
            if calculate_entropy:
                entropys = restore_dynamic_batch(entropys, batch_idx_list)

        return log_probs, entropys

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        self.actor_module.train()

        temperature = data.meta_info["temperature"]

        select_keys = [
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
            "old_log_probs",
            "advantages",
            "token_level_scores",
        ]
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")

        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []
        
        if self.use_svd_token_weighting and "uid" in data.non_tensor_batch.keys():
            non_tensor_select_keys.append("uid")

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        mini_batches = data.split(self.config.ppo_mini_batch_size)

        metrics = {}
        for _ in range(self.config.ppo_epochs):
            for batch_idx, mini_batch in enumerate(mini_batches):
                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                    micro_batches = list(micro_batches)
                else:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    micro_batches = list(mini_batch.split(self.config.ppo_micro_batch_size_per_gpu))

                self.actor_optimizer.zero_grad()

                for micro_batch in micro_batches:
                    micro_batch_metrics = {}
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    response_mask = model_inputs["response_mask"]
                    old_log_prob = model_inputs["old_log_probs"]
                    advantages = model_inputs["advantages"]
                    token_level_scores = model_inputs["token_level_scores"]
                    seq_level_scores = (token_level_scores * response_mask).sum(dim=-1)
                    is_positive_sample = seq_level_scores > 0
                    
                    if "uid" in micro_batch.non_tensor_batch:
                        u, c = np.unique(micro_batch.non_tensor_batch["uid"], return_counts=True)
                        print("micro_batch size", len(micro_batch.non_tensor_batch["uid"]), "max uid count", c.max())

                    clip_ratio = self.config.clip_ratio
                    clip_ratio_low = (
                        self.config.clip_ratio_low if self.config.clip_ratio_low is not None else clip_ratio
                    )
                    clip_ratio_high = (
                        self.config.clip_ratio_high if self.config.clip_ratio_high is not None else clip_ratio
                    )
                    clip_ratio_c = self.config.get("clip_ratio_c", 3.0)
                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    calculate_entropy = False
                    if entropy_coeff != 0:
                        calculate_entropy = True
                    entropy, log_prob, response_hidden, prompt_hidden = self._forward_micro_batch(
                        model_inputs, temperature=temperature, calculate_entropy=calculate_entropy)

                    loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
                    
                    bsz = seq_level_scores.shape[0]
                    
                    advantages = advantages * response_mask
                    weighted_advantages = advantages.clone()
                    
                    if bsz >= 1:
                        if self.use_svd_token_weighting:
                            svd_token_weights = self._compute_svd_token_weights(
                                response_hidden=response_hidden,
                                response_mask=response_mask,
                                is_positive_sample=is_positive_sample,
                                prompt_ids=None,
                                input_ids=model_inputs["input_ids"],
                                response_length=model_inputs["responses"].size(-1),
                            )

                            for i in range(bsz):
                                if is_positive_sample[i]:
                                    weighted_advantages[i] = advantages[i] * 0.1
                                else:
                                    weighted_advantages[i] = advantages[i] * svd_token_weights[i]
                        else:
                            for i in range(bsz):
                                if is_positive_sample[i]:
                                    weighted_advantages[i] = advantages[i] * 1.0
                        
                        num_positive = is_positive_sample.sum().item()
                        num_negative = (~is_positive_sample).sum().item()
                        micro_batch_metrics["actor/num_positive_samples"] = num_positive
                        micro_batch_metrics["actor/num_negative_samples"] = num_negative
                        micro_batch_metrics["actor/positive_sample_ratio"] = num_positive / max(bsz, 1)
                        if num_positive > 0:
                            micro_batch_metrics["actor/positive_score_mean"] = seq_level_scores[is_positive_sample].mean().item()
                        if num_negative > 0:
                            micro_batch_metrics["actor/negative_score_mean"] = seq_level_scores[~is_positive_sample].mean().item()
                    
                    if self.config.policy_loss.loss_mode == "vanilla":
                        pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = compute_policy_loss(
                            old_log_prob=old_log_prob,
                            log_prob=log_prob,
                            advantages=weighted_advantages,
                            response_mask=response_mask,
                            cliprange=clip_ratio,
                            cliprange_low=clip_ratio_low,
                            cliprange_high=clip_ratio_high,
                            clip_ratio_c=clip_ratio_c,
                            loss_agg_mode=loss_agg_mode,
                        )
                    else:
                        policy_loss_fn = get_policy_loss_fn(loss_mode)
                        pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = policy_loss_fn(
                            old_log_prob=old_log_prob,
                            log_prob=log_prob,
                            advantages=weighted_advantages,
                            response_mask=response_mask,
                            loss_agg_mode=loss_agg_mode,
                            config=self.config,
                        )
                    
                    if entropy_coeff != 0:
                        entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
                        policy_loss = pg_loss - entropy_loss * entropy_coeff
                    else:
                        policy_loss = pg_loss
                    
                    if self.config.use_kl_loss:
                        ref_log_prob = model_inputs["ref_log_prob"]
                        kld = kl_penalty(
                            logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type
                        )
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        micro_batch_metrics["actor/kl_loss"] = kl_loss.detach().item()
                        micro_batch_metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    if self.config.use_dynamic_bsz:
                        loss = policy_loss * (response_mask.shape[0] / self.config.ppo_mini_batch_size)
                    else:
                        loss = policy_loss / self.gradient_accumulation
                    loss.backward()

                    micro_batch_metrics.update(
                        {
                            "actor/pg_loss": pg_loss.detach().item(),
                            "actor/pg_clipfrac": pg_clipfrac.detach().item(),
                            "actor/ppo_kl": ppo_kl.detach().item(),
                            "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
                        }
                    )
                    append_to_dict(metrics, micro_batch_metrics)

                grad_norm = self._optimizer_step()
                mini_batch_metrics = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, mini_batch_metrics)
        self.actor_optimizer.zero_grad()
        return metrics
