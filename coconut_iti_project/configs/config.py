import torch
import random
import numpy as np
import os

class Config:
    def __init__(self):
        # --- THE 24GB VRAM ENGINE ---
        self.model_id = "./qwen-3b-local" 
        self.save_path = "./checkpoints_coconut_qwen3b_full"
        
        # Keep physical batch at 1 to prevent OOM during full-parameter backprop
        self.batch_size_training = 1       
        self.gradient_accumulation_steps = 128
        self.max_seq_len = 512             
        
        # Lower LR for full-parameter tuning stability
        self.lr = 5e-6
        self.weight_decay = 0.01
        
        # --- COCONUT CURRICULUM ---
        self.hybrid_mode = True           
        self.c_thought = 2                 
        self.max_latent_tokens = 6         
        self.num_epochs_total = 50         
        
        self.num_generations_per_sample = 1 
        self.generation_temperature = 0.0   
        self.alpha_sweep = [0.0, 5.0, 10.0, 20.0, 50.0]
        self.alpha_decay = 0.95  
        
        self.seed = 42
        self.bf16 = True 
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

configs = Config()

def set_seed(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)

set_seed(configs.seed)
os.makedirs(configs.save_path, exist_ok=True)
