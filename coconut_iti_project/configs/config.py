import torch
import random
import numpy as np
import os

class Config:
    def __init__(self):
        # 1. Point to the new 1.5B Math model
        # Use the HF repo id by default. If you have a local copy, replace with
        # an absolute path like "/root/Coconut-Steering/coconut_iti_project/qwen-1.5b-math-local".
        self.model_id = "Qwen/Qwen2.5-Math-1.5B"
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
        
        # Full-run protocol split on a 7,473-example train pool:
        # - D_train = 4,484 (60%)
        # - D_steer =   747 (10%)
        # - D_val   = 2,242 (30%)
        self.train_pool_size = 7473
        self.protocol_reserved_examples = 2989  # D_steer + D_val
        self.phase2_steer_examples = 747
        # Internal split for D_train checkpoint selection:
        # - D_train-actual = 90%
        # - D_train-val    = 10%
        self.phase1_train_fraction = 0.9

        # Phase 2: frozen stochastic extraction from D_steer.
        # More samples per question → more contrastive (H⁺/H⁻) paths when Phase 1 is weak.
        self.num_generations_per_sample = 20
        self.vector_extraction_temperature = 1.0
        self.min_vector_class_count = 100
        self.skip_non_contrastive_questions = True
        self.vector_method = "dom"
        self.compute_cpca = True
        self.cpca_beta = 0.5
        self.cpca_beta_sweep = [0.0, 0.3, 0.5, 0.7, 1.0]
        self.cpca_k_values = [1, 2, 5, 10]
        self.cpca_probe_k = 10
        self.linear_probe_test_size = 0.2

        # Phase 3: gradient-based alpha tuning on D_val.
        self.alpha_max = 50.0
        self.alpha_initial = 1.0
        self.alpha_lr = 5e-2
        self.alpha_max_epochs = 3
        self.alpha_patience = 5
        self.alpha_tune_fraction = 0.9
        # Keep alignment regularizer disabled; optimize L_ans + λ_m * L_mag.
        self.lambda_align = 0.0
        self.lambda_mag = 0.01
        self.gradient_check_epsilon = 1e-2
        self.gradient_check_max_rel_error = 0.05
        # Failed check means α gradients are wrong; tuning must not continue silently.
        self.enforce_gradient_check = True
        self.alpha_sweep = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
        self.alpha_decay = 1.0
        self.alpha_star_path = None
        self.truth_vector_path = None
        self.run_alpha_diagnostic_sweep = True

        # Phase 4: final locked-test evaluation.
        self.test_pool_size = 1319
        # After the main 5-condition table, also run CCoT+truth (and optionally +random)
        # across these alphas, always merging in the learned alpha* from Phase 3.
        self.phase4_run_alpha_sweep = True
        self.phase4_alpha_sweep = None  # None -> use every alpha in alpha_sweep with alpha > 0
        self.phase4_sweep_random = False  # If True, also sweep random noise at each α (2× cost)
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
os.makedirs(os.path.join(configs.save_path, "log"), exist_ok=True)
