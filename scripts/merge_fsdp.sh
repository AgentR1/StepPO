cd ../../verl

BASE_DIR=/data/tingyue/workspace/oyj/AgentRFT-PaperSearch/checkpoints
SRC_ROOT=${BASE_DIR}/FALCON/falcon-v3-force-think-gspo-lock
DST_ROOT=${BASE_DIR}/Convert/falcon-v3-force-think-gspo-lock

for step in $(seq 20 20 200); do
    echo "=== Merging checkpoint at step ${step} ==="

    python scripts/legacy_model_merger.py merge \
        --backend fsdp \
        --local_dir ${SRC_ROOT}/global_step_${step}/actor \
        --target_dir ${DST_ROOT}/actor_${step}
done
