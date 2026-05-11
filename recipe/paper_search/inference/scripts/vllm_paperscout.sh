#!/usr/bin/env bash
# Optional: serve the policy with ``vllm serve`` for manual HTTP experiments only.
# Batch inference uses in-process vLLM via ``run_paper_agent.py`` (see inference/.env).

export CUDA_VISIBLE_DEVICES=6
export VLLM_USE_MODELSCOPE=1
export CUDA_HOME=/usr/local/cuda

vllm serve /data/tingyue/.cache/modelscope/hub/models/Qwen/Qwen3-4B-Instruct-2507 \
  --max-model-len 10240 \
  --api-key Qwen3-4b-instruct \
  --gpu-memory-utilization 0.9 \
  --port 8998 \
  --served-model-name Qwen3-4b-instruct
