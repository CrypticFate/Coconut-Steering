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
        
        # Phase 2/3 protocol data. The reserved tail size stays at 1000,
        # preserving the existing Phase 1 training split.
        self.protocol_reserved_examples = 1000
        self.phase2_steer_examples = 500

        # Phase 2: frozen stochastic extraction from D_steer.
        self.num_generations_per_sample = 10
        self.vector_extraction_temperature = 1.0
        self.min_vector_class_count = 100
        self.skip_non_contrastive_questions = True
        self.vector_method = "dom"
        self.compute_cpca = True
        self.cpca_beta = 0.5
        self.cpca_k_values = [1, 2, 5, 10]
        self.linear_probe_test_size = 0.2

        # Phase 3: gradient-based alpha tuning on D_val.
        self.alpha_max = 50.0
        self.alpha_initial = 1.0
        self.alpha_lr = 5e-2
        self.alpha_max_epochs = 3
        self.alpha_patience = 5
        self.alpha_tune_fraction = 0.9
        self.lambda_align = 0.1
        self.lambda_mag = 0.01
        self.gradient_check_epsilon = 1e-2
        self.gradient_check_max_rel_error = 0.05
        self.enforce_gradient_check = True
        self.alpha_sweep = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
        self.alpha_decay = 1.0
        self.alpha_star_path = None
        self.truth_vector_path = None
        self.run_alpha_diagnostic_sweep = True

        # Phase 4: final locked-test evaluation.
        self.random_noise_seed = 1234
        self.max_new_tokens_no_cot = 32
        self.max_new_tokens_text_cot = 128
        self.max_new_tokens_ccot = 64
        
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
