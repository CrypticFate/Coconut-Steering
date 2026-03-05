import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["CUDA_MODULE_LOADING"] = "LAZY"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import functools
print = functools.partial(print, flush=True)

print("=== Diagnosing model loading segfault ===")

import torch
print(f"torch OK (CUDA={torch.cuda.is_available()})")

from transformers import AutoModelForCausalLM, AutoTokenizer
print("transformers OK")

model_id = "Qwen/Qwen2.5-3B-Instruct"

# --- Test A: No trust_remote_code, no device_map ---
print(f"\n--- Test A: bare minimum from_pretrained ---")
print(f"Loading {model_id} (torch_dtype=bfloat16, no trust_remote_code, no device_map)...", end=" ")
try:
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
    )
    print("OK")
    del model
    torch.cuda.empty_cache()
except Exception as e:
    print(f"FAILED: {e}")

    # --- Test B: float32 instead ---
    print(f"\n--- Test B: float32, no trust_remote_code ---")
    print(f"Loading {model_id} (no dtype specified)...", end=" ")
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id)
        print("OK")
        del model
    except Exception as e2:
        print(f"FAILED: {e2}")

print("\n=== Done ===")
