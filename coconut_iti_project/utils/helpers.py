import gc
import os
import random
from datetime import datetime

import numpy as np
import torch


def set_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)


def clear_memory():
    torch.cuda.empty_cache()
    gc.collect()


def save_phase_log(save_path, phase_num, phase_name, content):
    """Write a log file for a completed phase to the checkpoints directory."""
    os.makedirs(save_path, exist_ok=True)
    log_file = os.path.join(save_path, f"phase{phase_num}_log.txt")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, "w") as f:
        f.write(f"Phase {phase_num}: {phase_name}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write("=" * 60 + "\n\n")
        f.write(content)
    print(f"[LOG] Phase {phase_num} log saved to {log_file}")
