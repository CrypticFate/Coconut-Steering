import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["CUDA_MODULE_LOADING"] = "LAZY"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import functools
print = functools.partial(print, flush=True)

print("=== Diagnosing model loading segfault ===")

print("1. Importing torch...", end=" ")
import torch
print(f"OK (CUDA={torch.cuda.is_available()})")

print("2. Importing transformers...", end=" ")
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
print("OK")

model_id = "Qwen/Qwen2.5-3B-Instruct"

print(f"3. Loading config for {model_id}...", end=" ")
model_config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
print(f"OK (hidden_size={model_config.hidden_size})")

print("4. Loading tokenizer...", end=" ")
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
print(f"OK (vocab_size={tokenizer.vocab_size})")

print("5. Loading model with from_pretrained (CPU, bf16)...", end=" ")
try:
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
    )
    print("OK")
except TypeError as e:
    print(f"TypeError with torch_dtype: {e}")
    print("5b. Retrying without torch_dtype, using config override...", end=" ")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="cpu",
        trust_remote_code=True,
    )
    print("OK")

print(f"6. Model type: {type(model).__name__}")
print(f"7. Model params: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")

print("\n=== Model loading succeeded. No segfault. ===")
