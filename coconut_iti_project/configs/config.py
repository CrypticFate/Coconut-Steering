import torch
import random
import numpy as np

class Config:
    def __init__(self):
        # 1. BASE MODEL SWITCH: Using pre-trained GPT-2 (124M parameters)
        self.model_id = "gpt2"
        self.save_path = "./checkpoints_coconut_steered"

        self.train_split_1_ratio = 0.60
        self.train_split_2_ratio = 0.10
        self.train_split_3_ratio = 0.30

        # RTX 3090 Optimizations for GPT-2
        # GPT-2 is tiny. We can massively increase physical batch size.
        self.batch_size_training = 16
        self.gradient_accumulation_steps = 8  # 16 x 8 = 128 Effective Batch Size
        self.max_seq_len = 512

        self.lr = 1e-4  # The exact paper learning rate for GPT-2
        self.weight_decay = 0.01

        # 2. HYBRID & FINER-GRAINED CURRICULUM
        self.hybrid_mode = True           # Forces Step 1 to remain in English
        self.drop_tokens_per_stage = 4    # Drops 4 words per stage instead of whole sentences
        self.c_thought = 1                # Adds 1 latent token per stage
        self.max_latent_tokens = 6        # Cap the total silent thoughts
        self.num_epochs_phase1 = 24       # Extended epochs for the finer-grained steps

        self.num_generations_per_sample = 1
        self.generation_temperature = 0.0

        # 3. ITI DECAY TUNING
        self.alpha_sweep = [0.0, 5.0, 10.0, 20.0, 50.0]
        self.alpha_decay = 0.95  # Increased from 0.7 to maintain sustained steering pressure

        self.seed = 42
        # GPT-2 trains best in standard float32 on the 3090
        self.bf16 = False
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

configs = Config()

def set_seed(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)

set_seed(configs.seed)
