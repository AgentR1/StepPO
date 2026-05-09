set -x

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1,2,3,4,5}
export HOTPOTQA_EMBEDDING_DEVICE=${HOTPOTQA_EMBEDDING_DEVICE:-cuda:4}
export VLLM_USE_V1=1
export HYDRA_FULL_ERROR=1
export MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI:-http://172.17.0.1:5000}

PROJECT_DIR="$(pwd)"
CONFIG_PATH="$PROJECT_DIR/recipe/hotpotqa/base_faiss_cpu.yaml"

HOTPOTQA_MODEL_PATH=${HOTPOTQA_MODEL_PATH:-Qwen/Qwen2.5-3B-Instruct}

# 长度预算（与 Agent-R1-legacy `run_ppo_hotpotqa.sh` 对照，语义不同）：
# - Legacy：多轮 token 拼一条轨迹 → data.max_prompt_length=8192、整段 response=8192、单轮 max_response_length_single_turn=1024。
# - 本脚本（ARFT AgentFlow）：每步重拼 prompt + 单步一次 generate；user 侧多了「Recent tool / format issues」段落，
#   故适当加大 prompt；单步 response 对齐 legacy 的 1024，减少 <tool_call> JSON 被 max_tokens 截断。
HOTPOTQA_MAX_PROMPT_LEN=${HOTPOTQA_MAX_PROMPT_LEN:-10240}
HOTPOTQA_MAX_RESPONSE_LEN=${HOTPOTQA_MAX_RESPONSE_LEN:-1024}

TRAIN_PATH="$PROJECT_DIR/data/corpus/hotpotqa/train.parquet"
VAL_PATH="$PROJECT_DIR/data/corpus/hotpotqa/validation.parquet"

PROJECT_NAME='HotpotQA_ARFT'
EXP_NAME='hotpotqa_token_level_0.99_adv_mlflow_4gpu'

python3 -m arft.main_agent_ppo \
    algorithm.adv_estimator=token_gae \
    data.train_files="$TRAIN_PATH" \
    data.val_files="$VAL_PATH" \
    data.train_batch_size=256 \
    data.max_prompt_length="$HOTPOTQA_MAX_PROMPT_LEN" \
    data.max_response_length="$HOTPOTQA_MAX_RESPONSE_LEN" \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path="$HOTPOTQA_MODEL_PATH" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.agent.agent_flow_config_path="$CONFIG_PATH" \
    actor_rollout_ref.rollout.agent.num_workers=4 \
    actor_rollout_ref.rollout.agent.default_agent_flow=hotpotqa_agent \
    actor_rollout_ref.rollout.trace.backend=mlflow \
    actor_rollout_ref.rollout.trace.token2text=True \
    actor_rollout_ref.rollout.trace.max_samples_per_step_per_worker=5 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    critic.model.path="$HOTPOTQA_MODEL_PATH" \
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding=True \
    critic.model.enable_gradient_checkpointing=True \
    critic.ppo_micro_batch_size_per_gpu=4 \
    critic.model.fsdp_config.param_offload=True \
    critic.model.fsdp_config.optimizer_offload=True \
    algorithm.use_kl_in_reward=False \
    algorithm.gamma=0.99 \
    reward_model.enable=False \
    custom_reward_function.path=recipe/hotpotqa/reward_fn.py \
    custom_reward_function.name=compute_score \
    trainer.critic_warmup=0 \
    trainer.logger='["console","swanlab","mlflow"]' \
    trainer.project_name="$PROJECT_NAME" \
    trainer.experiment_name="$EXP_NAME" \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.val_before_train=True \
    trainer.save_freq=100 \
    trainer.test_freq=10 \
    trainer.max_actor_ckpt_to_keep=3 \
    trainer.max_critic_ckpt_to_keep=3 \
    trainer.total_epochs=5 "$@"