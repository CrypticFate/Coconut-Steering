import torch


class Config:
    def __init__(self):
        self.model_id = "Qwen/Qwen2.5-1.5B-Instruct"
        self.save_path = "./checkpoints"

        self.train_split_1_ratio = 0.60
        self.train_split_2_ratio = 0.10
        self.train_split_3_ratio = 0.30

        # --- RTX 3090 Optimization: batch=8, grad_accum=16 (effective=128) ---
        self.batch_size_training = 8
        self.gradient_accumulation_steps = 16

        # --- STABILITY FIX: Lowered LR for 1.5B Model ---
        self.lr = 2e-5

        self.weight_decay = 0.01

        # --- MEMORY OPTIMIZATION: Trimmed to 512 ---
        self.max_seq_len = 512

        self.num_epochs_phase1 = 18
        self.c_thought = 2
        self.max_latent_stage = 3

        self.num_generations_per_sample = 1
        self.generation_temperature = 0.0

        self.alpha_sweep = [0.0, 5.0, 10.0, 20.0, 50.0]
        self.alpha_decay = 0.7

        self.seed = 42
        self.bf16 = True if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else False
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
