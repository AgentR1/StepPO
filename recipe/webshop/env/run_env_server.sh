#!/usr/bin/env bash
set -euo pipefail

HOST="${WEBSHOP_ENV_HOST:-127.0.0.1}"
PORT="${WEBSHOP_ENV_PORT:-4111}"
WORKERS="${WEBSHOP_ENV_WORKERS:-8}"

export WEBSHOP_DATA_DIR="${WEBSHOP_DATA_DIR:-$(pwd)/webshop_data}"
export WEBSHOP_INDEX_DIR="${WEBSHOP_INDEX_DIR:-$(pwd)/data/webshop/index}"
export WEBSHOP_SEARCH_TOP_K="${WEBSHOP_SEARCH_TOP_K:-10}"

exec gunicorn \
  -w "$WORKERS" \
  -k uvicorn.workers.UvicornWorker \
  recipe.webshop.env.server:app \
  -b "$HOST:$PORT" \
  --timeout "${WEBSHOP_ENV_TIMEOUT:-120}"

