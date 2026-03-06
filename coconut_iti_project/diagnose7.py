import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["CUDA_MODULE_LOADING"] = "LAZY"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import functools
print = functools.partial(print, flush=True)

print("=== Diagnosing: is it Qwen-specific or all models? ===\n")

import torch
print(f"torch: {torch.__version__}")
import transformers
print(f"transformers: {transformers.__version__}")

from transformers import AutoModelForCausalLM, AutoConfig

# --- Test 1: Can we create GPT-2 (tiny, well-tested)? ---
print("\nTest 1: Create GPT-2 from config...", end=" ")
try:
    cfg = AutoConfig.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_config(cfg)
    print(f"OK ({sum(p.numel() for p in model.parameters())/1e6:.0f}M params)")
    del model
except Exception as e:
    print(f"FAILED: {e}")

import gc; gc.collect()

# --- Test 2: Can we create Qwen2 manually? ---
print("\nTest 2: Import Qwen2 modeling module...", end=" ")
try:
    from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM
    print("OK")
except Exception as e:
    print(f"FAILED: {e}")

print("\nTest 3: Create Qwen2 config only...", end=" ")
try:
    qwen_cfg = AutoConfig.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
    print(f"OK (type={type(qwen_cfg).__name__})")
    print(f"  hidden_size={qwen_cfg.hidden_size}, num_layers={qwen_cfg.num_hidden_layers}")
    print(f"  num_heads={qwen_cfg.num_attention_heads}, vocab={qwen_cfg.vocab_size}")
except Exception as e:
    print(f"FAILED: {e}")

# --- Test 4: Create a TINY Qwen2 model to test construction ---
print("\nTest 4: Create tiny Qwen2 (2 layers, 256 hidden)...", end=" ")
try:
    from transformers import Qwen2Config, Qwen2ForCausalLM
    tiny_cfg = Qwen2Config(
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        vocab_size=1000,
    )
    tiny_model = Qwen2ForCausalLM(tiny_cfg)
    print(f"OK ({sum(p.numel() for p in tiny_model.parameters())/1e6:.1f}M params)")
    del tiny_model
except Exception as e:
    print(f"FAILED: {e}")

gc.collect()

# --- Test 5: Create full-size Qwen2 from config ---
print("\nTest 5: Create full Qwen2.5-3B from config (random weights)...", end=" ")
try:
    full_cfg = AutoConfig.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
    full_cfg._attn_implementation = "eager"
    model = Qwen2ForCausalLM(full_cfg)
    print(f"OK ({sum(p.numel() for p in model.parameters())/1e9:.2f}B params)")
    del model
except Exception as e:
    print(f"FAILED: {e}")

gc.collect()

print("\n=== Done ===")
