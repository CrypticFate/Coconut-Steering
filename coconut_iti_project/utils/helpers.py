import gc
import os
import csv
import json
import random
import re
import sys
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
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def normalize_answer(value):
    """Normalize a GSM-style final answer for exact-match comparisons."""
    if value is None:
        return ""
    text = str(value).replace(",", "").strip()
    numbers = _NUMBER_RE.findall(text)
    if numbers:
        return numbers[-1].rstrip("0").rstrip(".") if "." in numbers[-1] else numbers[-1]
    return re.sub(r"\s+", " ", text.lower()).strip().rstrip(".")


def extract_final_answer(output_text):
    """Extract the final answer from generated text using GSM8K conventions."""
    text = str(output_text)
    if "####" in text:
        text = text.split("####")[-1]
    elif "Answer:" in text:
        text = text.split("Answer:")[-1]
    return normalize_answer(text)


def answers_match(predicted, gold):
    """Return True when normalized predicted and gold answers match exactly."""
    return normalize_answer(predicted) == normalize_answer(gold)


class TeeLogger:
    """Writes all output to both the terminal and a log file simultaneously.

    Can be used for both stdout and stderr so that *every* piece of output
    (including tqdm progress bars, warnings, and tracebacks) is captured.
    """
    def __init__(self, filepath, original_stream):
        self.original = original_stream
        self.log = open(filepath, "a")

    def write(self, message):
        self.original.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.original.flush()
        self.log.flush()

    def close(self):
        self.log.close()

    def isatty(self):
        return self.original.isatty()

    # Support for tqdm and other libraries that check these attributes
    @property
    def encoding(self):
        return getattr(self.original, "encoding", "utf-8")

    def fileno(self):
        return self.original.fileno()


def setup_run_directory(base_save_path):
    """Create a timestamped run directory inside the base save path.

    Directory layout:
        base_save_path/
            run_20260430_154300/
                logs/
                    pipeline_full.log   (master stdout + stderr)
                    training_loss.csv   (per-epoch loss)
                    extraction.log      (Phase 2 stats)
                    phase4_evaluation.log
                plots/
                    loss_curve.png
                    pca_latents.html
                checkpoints/
                    coconut_phase1.pt
                    truth_vector.pt
                    stage_N_epoch_M.pt  (per-stage snapshots)
                config_snapshot.json
                phase1_log.txt
                phase2_log.txt
                ...

    Returns the absolute path to the run directory.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_save_path, f"run_{timestamp}")

    subdirs = ["logs", "plots", "checkpoints", "results"]
    for subdir in subdirs:
        os.makedirs(os.path.join(run_dir, subdir), exist_ok=True)

    return run_dir


def activate_logging(run_dir):
    """Redirect both stdout and stderr to log files inside `run_dir/logs/`.

    Returns (stdout_tee, stderr_tee) so they can be closed at the end.
    """
    log_path = os.path.join(run_dir, "logs", "pipeline_full.log")

    stdout_tee = TeeLogger(log_path, sys.stdout)
    stderr_tee = TeeLogger(log_path, sys.stderr)
    sys.stdout = stdout_tee
    sys.stderr = stderr_tee

    return stdout_tee, stderr_tee


def deactivate_logging(stdout_tee, stderr_tee):
    """Restore the original stdout/stderr streams and close the log files."""
    sys.stdout = stdout_tee.original
    sys.stderr = stderr_tee.original
    stdout_tee.close()
    stderr_tee.close()


def save_config_snapshot(run_dir, config):
    """Persist every Config attribute to a JSON file for reproducibility."""
    snapshot = {}
    for attr in sorted(vars(config)):
        val = getattr(config, attr)
        # Convert non-serializable types
        if isinstance(val, torch.device):
            val = str(val)
        elif isinstance(val, (torch.dtype,)):
            val = str(val)
        snapshot[attr] = val

    path = os.path.join(run_dir, "config_snapshot.json")
    with open(path, "w") as f:
        json.dump(snapshot, f, indent=2, default=str)
    print(f"[LOG] Config snapshot saved to {path}")


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


# ---------- CSV helpers for structured numeric logs ----------

def init_training_csv(run_dir):
    """Create the training loss CSV with headers. Returns the file path."""
    csv_path = os.path.join(run_dir, "logs", "training_loss.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "stage", "avg_loss", "lr", "timestamp"])
    return csv_path


def append_training_csv(csv_path, epoch, stage, avg_loss, lr):
    """Append a row to the training loss CSV."""
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            epoch, stage, f"{avg_loss:.6f}", f"{lr:.2e}",
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        ])


def save_extraction_log(run_dir, n_correct, n_wrong, vector_shape, l2_norm, duration_s):
    """Write the Phase 2 extraction summary to a dedicated log file."""
    path = os.path.join(run_dir, "logs", "extraction.log")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(path, "w") as f:
        f.write(f"Phase 2: Truth Vector Extraction\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Correct Latent Paths: {n_correct}\n")
        f.write(f"Incorrect Latent Paths: {n_wrong}\n")
        f.write(f"Truth Vector Shape: {vector_shape}\n")
        f.write(f"Truth Vector L2 Norm: {l2_norm:.6f}\n")
        f.write(f"Duration: {duration_s / 60:.1f} minutes\n")
    print(f"[LOG] Extraction log saved to {path}")


def save_json_log(run_dir, filename, payload):
    """Save structured protocol metadata under the run log directory."""
    log_dir = os.path.join(run_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, filename)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"[LOG] JSON log saved to {path}")
    return path
