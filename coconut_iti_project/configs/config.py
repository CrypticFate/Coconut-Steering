import torch
import random
import numpy as np
import os

class Config:
    def __init__(self):
        # 1. Point to the new 1.5B Math model
        self.model_id = "./qwen-1.5b-math-local" 
        self.save_path = "./checkpoints_coconut_qwen1.5b_math_full"
        
        self.batch_size_training = 1       
        self.gradient_accumulation_steps = 128
        self.max_seq_len = 512             
        
        # 2. Golden LR for 1.5B Full-Parameter Tune
        self.lr = 1e-5  
        self.weight_decay = 0.01
        
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
