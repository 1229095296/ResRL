#!/bin/bash
PRETRAINED_MODEL=/huggingface.co/Qwen/Qwen3-1.7B
n_nodes=1
n_gpus_per_node=8
tensor_model_parallel_size=4
save_freq=50

dapo_train_path=/data/dapo-math-17k.parquet
r1_test_path=/data/validation.parquet

experiment_name="ResRL"
max_prompt_length=2048
max_response_length=4096
OUTPUT_DIR=/ckpt/$experiment_name

set -x
rollout_n=4
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$dapo_train_path \
    data.val_files=$r1_test_path \
    data.train_batch_size=512 \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    data.truncation='left' \
    +data.apply_chat_template_kwargs.enable_thinking=False \
    +actor_rollout_ref.actor.tb_type=tempered_important_sampling \
    +actor_rollout_ref.actor.porj_layer=0 \
    actor_rollout_ref.model.path=$PRETRAINED_MODEL \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=False \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$rollout_n \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$rollout_n \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$rollout_n \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.rollout.val_kwargs.n=2 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$tensor_model_parallel_size \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.65 \
    actor_rollout_ref.rollout.n=$rollout_n \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.use_svd_token_weighting=true \
    actor_rollout_ref.actor.rollout_n=$rollout_n \
    actor_rollout_ref.actor.svd_rank=64 \
    actor_rollout_ref.actor.svd_tau=1.0 \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','tensorboard'] \
    trainer.project_name='FlowRL' \
    trainer.experiment_name=$experiment_name \
    trainer.n_gpus_per_node=$n_gpus_per_node \
    trainer.nnodes=$n_nodes \
    trainer.resume_mode=auto \
    trainer.save_freq=$save_freq \
    trainer.default_local_dir=$OUTPUT_DIR \
    trainer.test_freq=5 \
    trainer.total_epochs=1 \
    $@