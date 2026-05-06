#!/usr/bin/env bash
set -euo pipefail

SCRIPT_NAME="$(basename "$0" .sh)"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
LOG_ROOT="${LOG_ROOT:-$PROJECT_DIR/logs/webshop_zeroshot}"
mkdir -p "$LOG_ROOT"
TIMESTAMP="$(date -u +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_FILE:-$LOG_ROOT/${SCRIPT_NAME}_${TIMESTAMP}.log}"

exec > >(tee -a "$LOG_FILE") 2>&1
echo "Logging to $LOG_FILE"
set -x

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export WEBSHOP_ENV_BASE_URL="${WEBSHOP_ENV_BASE_URL:-http://127.0.0.1:4111}"

MODEL_PATH="${WEBSHOP_ZEROSHOT_MODEL_PATH:-/data/wdy/Downloads/models/Qwen/Qwen3-4B-Instruct-2507}"
OUTPUT_DIR="${WEBSHOP_ZEROSHOT_OUTPUT_DIR:-$PROJECT_DIR/outputs/webshop_zeroshot/qwen3_4b}"

python "$PROJECT_DIR/test/webshop_zeroshot_eval.py" \
    --model_path "$MODEL_PATH" \
    --data_root "$PROJECT_DIR/data/webshop" \
    --env_base_url "$WEBSHOP_ENV_BASE_URL" \
    --output_dir "$OUTPUT_DIR" \
    --splits test \
    --max_samples "${WEBSHOP_ZEROSHOT_MAX_SAMPLES:--1}" \
    --sample_mode "${WEBSHOP_ZEROSHOT_SAMPLE_MODE:-category_stratified}" \
    --seed "${WEBSHOP_ZEROSHOT_SEED:-0}" \
    --max_steps "${WEBSHOP_ZEROSHOT_MAX_STEPS:-15}" \
    --max_new_tokens "${WEBSHOP_ZEROSHOT_MAX_NEW_TOKENS:-256}" \
    --temperature "${WEBSHOP_ZEROSHOT_TEMPERATURE:-0.4}" \
    --top_p "${WEBSHOP_ZEROSHOT_TOP_P:-1.0}" \
    --dtype "${WEBSHOP_ZEROSHOT_DTYPE:-auto}" \
    --device_map "${WEBSHOP_ZEROSHOT_DEVICE_MAP:-auto}" \
    --parser_mode "${WEBSHOP_ZEROSHOT_PARSER_MODE:-strict}" \
    ${WEBSHOP_ZEROSHOT_DISABLE_THINKING:+--disable_thinking} \
    ${WEBSHOP_ZEROSHOT_PRINT_STEPS:+--print_steps} \
    "$@"
