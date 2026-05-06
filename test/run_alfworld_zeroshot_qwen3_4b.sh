#!/usr/bin/env bash
set -euo pipefail

SCRIPT_NAME="$(basename "$0" .sh)"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
LOG_ROOT="${LOG_ROOT:-$PROJECT_DIR/logs/alfworld_zeroshot}"
mkdir -p "$LOG_ROOT"
TIMESTAMP="$(date -u +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_FILE:-$LOG_ROOT/${SCRIPT_NAME}_${TIMESTAMP}.log}"

exec > >(tee -a "$LOG_FILE") 2>&1
echo "Logging to $LOG_FILE"
set -x

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export ALFWORLD_DATA_ROOT="${ALFWORLD_DATA_ROOT:-$PROJECT_DIR/data/alfworld}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

MODEL_PATH="${ALFWORLD_ZEROSHOT_MODEL_PATH:-/data/wdy/Downloads/models/Qwen/Qwen3-4B-Instruct-2507}"
OUTPUT_DIR="${ALFWORLD_ZEROSHOT_OUTPUT_DIR:-$PROJECT_DIR/outputs/alfworld_zeroshot/qwen3_4b}"

python "$PROJECT_DIR/test/alfworld_zeroshot_eval.py" \
    --model_path "$MODEL_PATH" \
    --data_root "$PROJECT_DIR/data/alfworld" \
    --output_dir "$OUTPUT_DIR" \
    --splits valid_seen valid_unseen \
    --max_samples "${ALFWORLD_ZEROSHOT_MAX_SAMPLES:--1}" \
    --sample_mode "${ALFWORLD_ZEROSHOT_SAMPLE_MODE:-stratified}" \
    --seed "${ALFWORLD_ZEROSHOT_SEED:-0}" \
    --max_steps "${ALFWORLD_ZEROSHOT_MAX_STEPS:-50}" \
    --max_episode_steps "${ALFWORLD_ZEROSHOT_MAX_EPISODE_STEPS:-50}" \
    --max_new_tokens "${ALFWORLD_ZEROSHOT_MAX_NEW_TOKENS:-256}" \
    --temperature "${ALFWORLD_ZEROSHOT_TEMPERATURE:-0.4}" \
    --top_p "${ALFWORLD_ZEROSHOT_TOP_P:-1.0}" \
    --dtype "${ALFWORLD_ZEROSHOT_DTYPE:-auto}" \
    --device_map "${ALFWORLD_ZEROSHOT_DEVICE_MAP:-auto}" \
    --parser_mode "${ALFWORLD_ZEROSHOT_PARSER_MODE:-strict}" \
    ${ALFWORLD_ZEROSHOT_PRINT_STEPS:+--print_steps} \
    "$@"
