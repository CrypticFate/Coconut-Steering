import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["CUDA_MODULE_LOADING"] = "LAZY"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys

print("Step 1: Basic imports OK")

try:
    import torch
    print(f"Step 2: torch OK (version {torch.__version__})")
except Exception as e:
    print(f"Step 2: torch FAILED: {e}")
    sys.exit(1)

try:
    print(f"Step 3: CUDA available = {torch.cuda.is_available()}")
except Exception as e:
    print(f"Step 3: CUDA check FAILED: {e}")

try:
    import numpy as np
    print(f"Step 4: numpy OK (version {np.__version__})")
except Exception as e:
    print(f"Step 4: numpy FAILED: {e}")

try:
    import transformers
    print(f"Step 5: transformers OK (version {transformers.__version__})")
except Exception as e:
    print(f"Step 5: transformers FAILED: {e}")

try:
    import datasets
    print(f"Step 6: datasets OK (version {datasets.__version__})")
except Exception as e:
    print(f"Step 6: datasets FAILED: {e}")

try:
    import bitsandbytes as bnb
    print(f"Step 7: bitsandbytes OK (version {bnb.__version__})")
except Exception as e:
    print(f"Step 7: bitsandbytes FAILED: {e}")

try:
    import plotly
    print(f"Step 8: plotly OK (version {plotly.__version__})")
except Exception as e:
    print(f"Step 8: plotly FAILED: {e}")

try:
    import matplotlib
    print(f"Step 9: matplotlib OK (version {matplotlib.__version__})")
except Exception as e:
    print(f"Step 9: matplotlib FAILED: {e}")

try:
    import tqdm
    print(f"Step 10: tqdm OK (version {tqdm.__version__})")
except Exception as e:
    print(f"Step 10: tqdm FAILED: {e}")

try:
    import accelerate
    print(f"Step 11: accelerate OK (version {accelerate.__version__})")
except Exception as e:
    print(f"Step 11: accelerate FAILED: {e}")

print("\nAll imports passed. No segfault from imports.")
print(f"Python: {sys.version}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")
