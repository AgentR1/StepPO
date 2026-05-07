export CUDA_VISIBLE_DEVICES=6

vllm serve /data/tingyue/workspace/oyj/Falcon-workflow-12-27/checkpoints/selector_convert_to_seq_cls \
    --served-model-name selector \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 2048 \
    --host 0.0.0.0 \
    --port 8993 \
    --task classify