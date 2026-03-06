import torch
import random
import numpy as np

class Config:
    def __init__(self):
        # --- 1. THE LLAMA 3.2 UPGRADE ---
        self.model_id = "meta-llama/Llama-3.2-3B"
        self.save_path = "./checkpoints_coconut_steered"

        self.train_split_1_ratio = 0.60
        self.train_split_2_ratio = 0.10
        self.train_split_3_ratio = 0.30

        # --- 2. STRICT RTX 3090 (24GB) SAFEGUARDS ---
        # Physical batch size MUST be 1. Do not increase this or you will OOM.
        self.batch_size_training = 1
        self.gradient_accumulation_steps = 128  # 1 * 128 = 128 Effective Batch Size
        self.max_seq_len = 512

        self.lr = 2e-5  # Lowered from GPT-2's 1e-4 to prevent weight shattering
        self.weight_decay = 0.01

        # --- 3. HYBRID & FINER-GRAINED CURRICULUM ---
        self.hybrid_mode = True
        self.drop_tokens_per_stage = 4
        self.c_thought = 1
        self.max_latent_tokens = 5
        self.num_epochs_phase1 = 24

        self.num_generations_per_sample = 1
        self.generation_temperature = 0.0

        self.alpha_sweep = [0.0, 5.0, 10.0, 20.0, 50.0]
        self.alpha_decay = 0.95

        self.seed = 42
        self.bf16 = True # STRICTLY ENABLED: Llama 3 requires bfloat16
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

configs = Config()

def set_seed(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)

set_seed(configs.seed)
