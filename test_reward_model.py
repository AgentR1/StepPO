import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["VLLM_LOGGING_LEVEL"] = "DEBUG"

model_path = "/root/workspace/StepPO/recipe/paper_search/selector-qwen3-8b/Melmaphother/selector-qwen-8b"

print(f"Testing vLLM with: {model_path}")
print("=" * 60)

from vllm import LLM

try:
    llm = LLM(
        model=model_path,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.3,
        max_model_len=1024,
        trust_remote_code=True,
        dtype="bfloat16",
    )
    print("✓ SUCCESS: vLLM loaded the model!")
    
    # 测试推理
    outputs = llm.generate(["Test"])
    print("✓ Inference successful")
    
except Exception as e:
    print(f"✗ FAILED: {e}")
    import traceback
    traceback.print_exc()