import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["CUDA_MODULE_LOADING"] = "LAZY"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import functools
print = functools.partial(print, flush=True)

print("=== Diagnosing from_pretrained segfault ===\n")

import torch
from transformers import AutoModelForCausalLM, AutoConfig

model_id = "Qwen/Qwen2.5-3B-Instruct"

# --- Test 1: Can we instantiate the model from config (no weights)? ---
print("Test 1: Instantiate model from config (no weights)...", end=" ")
config = AutoConfig.from_pretrained(model_id)
model = AutoModelForCausalLM.from_config(config)
print(f"OK ({sum(p.numel() for p in model.parameters())/1e9:.2f}B params)")
del model
import gc; gc.collect()

# --- Test 2: Check safetensors ---
print("\nTest 2: Check safetensors library...", end=" ")
try:
    import safetensors
    print(f"version={safetensors.__version__}")
    from safetensors.torch import load_file
    print("  load_file import OK")
except Exception as e:
    print(f"FAILED: {e}")

# --- Test 3: Try loading WITHOUT safetensors ---
print("\nTest 3: from_pretrained with use_safetensors=False...", end=" ")
try:
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        use_safetensors=False,
    )
    print("OK")
    del model; gc.collect()
except Exception as e:
    print(f"FAILED: {e}")

    # --- Test 4: Try with low_cpu_mem_usage=False ---
    print("\nTest 4: from_pretrained with low_cpu_mem_usage=False...", end=" ")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=False,
        )
        print("OK")
        del model; gc.collect()
    except Exception as e2:
        print(f"FAILED: {e2}")

print("\n=== Done ===")
