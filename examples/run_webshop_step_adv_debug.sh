#!/usr/bin/env bash
set -euo pipefail

SCRIPT_NAME="$(basename "$0" .sh)"
LOG_ROOT="${LOG_ROOT:-$(pwd)/logs}"
LOG_DIR="${LOG_DIR:-$LOG_ROOT/webshop}"
mkdir -p "$LOG_DIR"
TIMESTAMP="$(date -u +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_FILE:-$LOG_DIR/${SCRIPT_NAME}_${TIMESTAMP}.log}"

exec > >(tee -a "$LOG_FILE") 2>&1
echo "Logging to $LOG_FILE"
set -x

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2}"
export VLLM_USE_V1=1
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export HYDRA_FULL_ERROR=1
export WEBSHOP_ENV_BASE_URL="${WEBSHOP_ENV_BASE_URL:-http://127.0.0.1:4111}"

PROJECT_DIR="$(pwd)"
CONFIG_PATH="$PROJECT_DIR/recipe/webshop/base.yaml"

WEBSHOP_MODEL_PATH="${WEBSHOP_MODEL_PATH:-/data/wdy/Downloads/models/Qwen/Qwen2.5-0.5B-Instruct}"
WEBSHOP_MAX_PROMPT_LEN="${WEBSHOP_MAX_PROMPT_LEN:-4096}"
WEBSHOP_MAX_RESPONSE_LEN="${WEBSHOP_MAX_RESPONSE_LEN:-512}"
WEBSHOP_TRAIN_PATH="${WEBSHOP_TRAIN_PATH:-$PROJECT_DIR/data/webshop/train.parquet}"
WEBSHOP_VAL_PATH="${WEBSHOP_VAL_PATH:-$PROJECT_DIR/data/webshop/test.parquet}"
VAL_DUMP_DIR="${WEBSHOP_VAL_DUMP_DIR:-$PROJECT_DIR/outputs/webshop_validation/debug}"

PROJECT_NAME="${PROJECT_NAME:-WebShop_ARFT_Debug}"
EXP_NAME="${EXP_NAME:-webshop_step_adv_debug}"

python3 - <<'PY'
import os
import sys

import httpx

base_url = os.environ["WEBSHOP_ENV_BASE_URL"].rstrip("/")
try:
    resp = httpx.get(f"{base_url}/health", timeout=10.0)
    resp.raise_for_status()
except Exception as exc:
    print(f"WebShop env service is not ready at {base_url}: {exc}", file=sys.stderr)
    print("Start it first, for example:", file=sys.stderr)
    print("  WEBSHOP_ENV_PORT=4100 bash recipe/webshop/env/run_env_server.sh", file=sys.stderr)
    raise SystemExit(1)
print(f"WebShop env health: {resp.json()}")
PY

python3 -m arft.main_agent_ppo \
    algorithm.adv_estimator=gae \
    data.train_files="$WEBSHOP_TRAIN_PATH" \
    data.val_files="$WEBSHOP_VAL_PATH" \
    data.train_batch_size=32 \
    data.val_batch_size=16 \
    data.max_prompt_length="$WEBSHOP_MAX_PROMPT_LEN" \
    data.max_response_length="$WEBSHOP_MAX_RESPONSE_LEN" \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path="$WEBSHOP_MODEL_PATH" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.clip_ratio_low=3e-4 \
    actor_rollout_ref.actor.clip_ratio_high=4e-4 \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.policy_loss.loss_mode=gspo \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.agent.agent_flow_config_path="$CONFIG_PATH" \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.agent.num_workers=2 \
    actor_rollout_ref.rollout.agent.default_agent_flow=webshop_agent \
    reward_model.enable=False \
    custom_reward_function.path=recipe/webshop/reward_fn.py \
    custom_reward_function.name=compute_score \
    critic.model.path="$WEBSHOP_MODEL_PATH" \
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding=True \
    critic.model.enable_gradient_checkpointing=True \
    critic.ppo_micro_batch_size_per_gpu=8 \
    critic.model.fsdp_config.param_offload=True \
    critic.model.fsdp_config.optimizer_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console"]' \
    trainer.project_name="$PROJECT_NAME" \
    trainer.experiment_name="$EXP_NAME" \
    trainer.validation_data_dir="$VAL_DUMP_DIR" \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.val_before_train=True \
    trainer.save_freq=20 \
    trainer.test_freq=10 \
    trainer.max_actor_ckpt_to_keep=2 \
    trainer.max_critic_ckpt_to_keep=2 \
    trainer.total_epochs=2 "$@"

