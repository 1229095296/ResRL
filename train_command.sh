#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

PRETRAINED_MODEL="${PRETRAINED_MODEL:-Qwen/Qwen3-4B}"
TRAIN_FILES="${TRAIN_FILES:-/data/dapo-math-17k.parquet}"
VAL_FILES="${VAL_FILES:-/data/validation.parquet}"

PROJECT_NAME="${PROJECT_NAME:-ResRL}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-resrl-qwen3-4b-math}"
OUTPUT_DIR="${OUTPUT_DIR:-/ckpt/${EXPERIMENT_NAME}}"

NNODES="${NNODES:-1}"
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-8}"
TENSOR_MODEL_PARALLEL_SIZE="${TENSOR_MODEL_PARALLEL_SIZE:-8}"

MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-4096}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-256}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-64}"
ROLLOUT_N="${ROLLOUT_N:-4}"
VAL_ROLLOUT_N="${VAL_ROLLOUT_N:-1}"

LEARNING_RATE="${LEARNING_RATE:-1e-6}"
LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-10}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.65}"
SAVE_FREQ="${SAVE_FREQ:-50}"
TEST_FREQ="${TEST_FREQ:-5}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"

ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-0.6}"
ROLLOUT_TOP_P="${ROLLOUT_TOP_P:-1.0}"
ROLLOUT_TOP_K="${ROLLOUT_TOP_K:--1}"

export VERL_ENABLE_LENGTH_PENALTY="${VERL_ENABLE_LENGTH_PENALTY:-true}"
export MAX_RESPONSE_LENGTH
export LENGTH_PENALTY_START="${LENGTH_PENALTY_START:-3500}"
export LENGTH_PENALTY_END_SCALE="${LENGTH_PENALTY_END_SCALE:-0.7}"

set -x
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_files="${TRAIN_FILES}" \
    data.val_files="${VAL_FILES}" \
    data.train_batch_size="${TRAIN_BATCH_SIZE}" \
    data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
    data.max_response_length="${MAX_RESPONSE_LENGTH}" \
    data.truncation='left' \
    +data.apply_chat_template_kwargs.enable_thinking=False \
    actor_rollout_ref.model.path="${PRETRAINED_MODEL}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr="${LEARNING_RATE}" \
    actor_rollout_ref.actor.optim.lr_warmup_steps="${LR_WARMUP_STEPS}" \
    actor_rollout_ref.actor.optim.weight_decay="${WEIGHT_DECAY}" \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${ROLLOUT_N}" \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))" \
    actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.use_svd_token_weighting=true \
    actor_rollout_ref.actor.rollout_n="${ROLLOUT_N}" \
    actor_rollout_ref.actor.svd_rank=64 \
    actor_rollout_ref.actor.svd_max_pos_tokens=4096 \
    actor_rollout_ref.actor.svd_q_low=0.2 \
    actor_rollout_ref.actor.svd_q_high=0.8 \
    actor_rollout_ref.actor.svd_min_weight=0.1 \
    actor_rollout_ref.actor.svd_pos_weight=0.1 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${ROLLOUT_N}" \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))" \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${ROLLOUT_N}" \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))" \
    actor_rollout_ref.rollout.max_num_batched_tokens="$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))" \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${TENSOR_MODEL_PARALLEL_SIZE}" \
    actor_rollout_ref.rollout.gpu_memory_utilization="${GPU_MEMORY_UTILIZATION}" \
    actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
    actor_rollout_ref.rollout.val_kwargs.n="${VAL_ROLLOUT_N}" \
    actor_rollout_ref.rollout.val_kwargs.temperature="${ROLLOUT_TEMPERATURE}" \
    actor_rollout_ref.rollout.val_kwargs.top_p="${ROLLOUT_TOP_P}" \
    actor_rollout_ref.rollout.val_kwargs.top_k="${ROLLOUT_TOP_K}" \
    actor_rollout_ref.rollout.val_kwargs.do_sample=true \
    trainer.critic_warmup=0 \
    trainer.logger="['console','tensorboard']" \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.n_gpus_per_node="${N_GPUS_PER_NODE}" \
    trainer.nnodes="${NNODES}" \
    trainer.resume_mode=auto \
    trainer.save_freq="${SAVE_FREQ}" \
    trainer.default_local_dir="${OUTPUT_DIR}" \
    trainer.test_freq="${TEST_FREQ}" \
    trainer.total_epochs="${TOTAL_EPOCHS}" \
    "$@"
