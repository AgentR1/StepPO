#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

export EXP_NAME="${EXP_NAME:-papersearch_gigpo}"
export ARFT_GRPO_ROLLOUT_N="${ARFT_GIGPO_ROLLOUT_N:-${ARFT_GRPO_ROLLOUT_N:-8}}"

exec bash "$ROOT_DIR/examples/run_papersearch_grpo.sh" \
    algorithm.adv_estimator=gigpo \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef="${ARFT_GIGPO_KL_COEF:-0.001}" \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    algorithm.use_kl_in_reward=False \
    ++algorithm.gigpo.step_advantage_w="${ARFT_GIGPO_STEP_ADVANTAGE_W:-1.0}" \
    ++algorithm.gigpo.mode="${ARFT_GIGPO_MODE:-mean_std_norm}" \
    ++algorithm.gigpo.enable_similarity="${ARFT_GIGPO_ENABLE_SIMILARITY:-False}" \
    ++algorithm.gigpo.similarity_thresh="${ARFT_GIGPO_SIMILARITY_THRESH:-0.95}" \
    "$@"
