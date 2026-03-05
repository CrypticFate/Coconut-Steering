import torch
import random
import numpy as np

class Config:
    def __init__(self):
        # --- UPGRADED MODEL ---
        self.model_id = "Qwen/Qwen2.5-3B-Instruct"
        self.save_path = "./checkpoints_coconut_steered"

        # --- DATASET SPLIT (GSM8k) ---
        self.train_split_1_ratio = 0.60
        self.train_split_2_ratio = 0.10
        self.train_split_3_ratio = 0.30

        # --- RTX 3090 (24GB VRAM) OPTIMIZATIONS ---
        # With 3B parameters, full-parameter training requires keeping physical
        # batches minimal to prevent CUDA Out-Of-Memory errors.
        self.batch_size_training = 1
        self.gradient_accumulation_steps = 128  # Matches paper's effective batch size
        self.max_seq_len = 512                  # Trimmed to save activation memory

        # --- STABILITY FIXES ---
        self.lr = 2e-5  # Safe finetuning LR for 3B parameter models
        self.weight_decay = 0.01

        # --- PAPER EXACT CURRICULUM (PHASE 1) ---
        self.num_epochs_phase1 = 18
        self.c_thought = 2
        self.max_latent_stage = 3

        # --- GLOBAL EXTRACTION (PHASE 2) ---
        self.num_generations_per_sample = 1
        self.generation_temperature = 0.0

        # --- INFERENCE-TIME INTERVENTION (PHASE 4) ---
        self.alpha_sweep = [0.0, 5.0, 10.0, 20.0, 50.0]
        self.alpha_decay = 0.7

        # --- ENVIRONMENT ---
        self.seed = 42
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Deferred: bf16 check can segfault if called during module import
        self._bf16 = None

    @property
    def bf16(self):
        if self._bf16 is None:
            try:
                self._bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            except Exception:
                self._bf16 = False
        return self._bf16

def set_seed(seed_value=42):
    """Sets the seed across all stochastic modules for reproducibility."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
