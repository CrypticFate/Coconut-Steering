import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["CUDA_MODULE_LOADING"] = "LAZY"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import functools
print = functools.partial(print, flush=True)

print("=== Diagnosing model __init__ segfault ===\n")

import torch
from transformers import AutoModelForCausalLM, AutoConfig

model_id = "Qwen/Qwen2.5-3B-Instruct"

# --- Test 1: Eager attention (no Flash/SDPA) ---
print("Test 1: from_config with attn_implementation='eager'...", end=" ")
try:
    config = AutoConfig.from_pretrained(model_id)
    config._attn_implementation = "eager"
    model = AutoModelForCausalLM.from_config(config)
    print(f"OK ({sum(p.numel() for p in model.parameters())/1e9:.2f}B params)")
    del model
except Exception as e:
    print(f"FAILED: {e}")

import gc; gc.collect(); torch.cuda.empty_cache()

# --- Test 2: Full from_pretrained with eager attention ---
print("\nTest 2: from_pretrained with attn_implementation='eager'...", end=" ")
try:
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    )
    print(f"OK ({sum(p.numel() for p in model.parameters())/1e9:.2f}B params)")
    del model
except Exception as e:
    print(f"FAILED: {e}")

gc.collect(); torch.cuda.empty_cache()

print("\n=== Done ===")
