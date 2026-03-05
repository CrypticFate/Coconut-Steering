import os

# --- Must be set BEFORE any library imports to prevent TF/PyTorch CUDA conflict ---
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"           # Suppress TF logging entirely
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"           # Disable oneDNN ops
os.environ["CUDA_MODULE_LOADING"] = "LAZY"           # Lazy CUDA module loading
os.environ["TOKENIZERS_PARALLELISM"] = "false"        # Prevent tokenizer deadlocks

import argparse
import time

import torch

from configs.config import Config, set_seed
from core.evaluator import analyze_confidence, evaluate_with_iti, print_sample_outputs
from core.extractor import extract_truth_vector
from core.trainer import train_phase1
from data.data_loader import prepare_datasets
from models.coconut import initialize_qwen_model
from utils.helpers import clear_memory
from utils.visualizer import plot_latent_pca, plot_loss_curve


def log_phase(phase_num, phase_name):
    print("\n" + "=" * 60)
    print(f"  PHASE {phase_num}: {phase_name}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="COCONUT ITI Pipeline")
    parser.add_argument(
        "--skip-phase1", action="store_true",
        help="Skip Phase 1 training and load checkpoint from checkpoints/coconut_phase1.pt",
    )
    parser.add_argument(
        "--skip-phase2", action="store_true",
        help="Skip Phase 2 extraction and load truth_vector from checkpoints/truth_vector.pt",
    )
    args = parser.parse_args()

    pipeline_start = time.time()

    print("=" * 60)
    print("  COCONUT + ITI Steering Pipeline")
    print("=" * 60)

    config = Config()
    set_seed(config.seed)
    os.makedirs(config.save_path, exist_ok=True)

    print(f"\nDevice: {config.device}")
    print(f"Model: {config.model_id}")
    print(f"BF16: {config.bf16}")
    print(f"Batch size: {config.batch_size_training} x {config.gradient_accumulation_steps} "
          f"= {config.batch_size_training * config.gradient_accumulation_steps} effective")
    print(f"Epochs: {config.num_epochs_phase1}")
    print(f"Alpha sweep: {config.alpha_sweep}")

    # --- Data ---
    print("\n" + "-" * 60)
    print("  Loading Datasets")
    print("-" * 60)
    data_phase1, data_phase2, data_phase3, test_data = prepare_datasets(config)

    # --- Model ---
    print("\n" + "-" * 60)
    print("  Initializing Model")
    print("-" * 60)
    coconut_model, tokenizer, latent_id, start_id, end_id = initialize_qwen_model(config)

    # =========================================================
    # Phase 1: Silent Thinking (Base Training)
    # =========================================================
    log_phase(1, "SILENT THINKING (Base COCONUT Training)")

    if not args.skip_phase1:
        phase1_start = time.time()
        loss_history = train_phase1(
            coconut_model, data_phase1, tokenizer, config, latent_id, start_id, end_id
        )
        phase1_time = time.time() - phase1_start
        print(f"\nPhase 1 completed in {phase1_time / 60:.1f} minutes")
        print(f"Final training loss: {loss_history[-1]:.4f}")

        plot_loss_curve(loss_history, os.path.join(config.save_path, "loss_curve.png"))
        print_sample_outputs(coconut_model, tokenizer, test_data, config, phase_name="PHASE 1 (BASE)")
    else:
        checkpoint_path = os.path.join(config.save_path, "coconut_phase1.pt")
        print(f"[SKIP] Loading Phase 1 checkpoint from {checkpoint_path}...")
        coconut_model.load_state_dict(torch.load(checkpoint_path, map_location=config.device))
        print("[SKIP] Checkpoint loaded successfully.")

    clear_memory()

    # =========================================================
    # Phase 2: Mind Reading (Truth Vector Extraction)
    # =========================================================
    log_phase(2, "MIND READING (Global Truth Vector Extraction)")

    if not args.skip_phase2:
        phase2_start = time.time()
        truth_vector, correct_latents, wrong_latents = extract_truth_vector(
            coconut_model, data_phase2, tokenizer, config, latent_id, start_id, end_id
        )
        phase2_time = time.time() - phase2_start
        print(f"\nPhase 2 completed in {phase2_time / 60:.1f} minutes")

        plot_latent_pca(correct_latents, wrong_latents, os.path.join(config.save_path, "pca_latents.html"))
    else:
        vector_path = os.path.join(config.save_path, "truth_vector.pt")
        print(f"[SKIP] Loading truth vector from {vector_path}...")
        truth_vector = torch.load(vector_path, map_location=config.device)
        print(f"[SKIP] Truth vector loaded. Shape: {truth_vector.shape}")

    clear_memory()

    # =========================================================
    # Phase 3: Mind Control (Intervention Setup)
    # =========================================================
    log_phase(3, "MIND CONTROL (Intervention Setup)")
    print("Steering vector is embedded in the COCONUT forward pass.")
    print(f"Alpha decay (gamma): {config.alpha_decay}")
    print(f"Intervention formula: h_new = h_old + (alpha * sigma * v_truth)")
    print("Phase 3 logic is built into the model architecture -- no separate step needed.")

    # =========================================================
    # Phase 4: Final Exam (Evaluation with ITI)
    # =========================================================
    log_phase(4, "FINAL EXAM (Inference-Time Intervention Evaluation)")

    phase4_start = time.time()
    experiment_results = evaluate_with_iti(
        coconut_model, test_data, tokenizer, config, truth_vector, latent_id
    )
    phase4_time = time.time() - phase4_start
    print(f"\nPhase 4 evaluation completed in {phase4_time / 60:.1f} minutes")

    # --- Confidence Deep Dive ---
    print("\n" + "-" * 60)
    print("  Confidence Deep Dive (Logit Analysis)")
    print("-" * 60)
    analyze_confidence(coconut_model, test_data, tokenizer, config, truth_vector, latent_id)

    # --- Summary ---
    total_time = time.time() - pipeline_start
    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Total runtime: {total_time / 60:.1f} minutes")
    print(f"Checkpoints saved to: {os.path.abspath(config.save_path)}")
    print(f"Loss curve: {os.path.join(config.save_path, 'loss_curve.png')}")
    print(f"PCA plot: {os.path.join(config.save_path, 'pca_latents.html')}")


if __name__ == "__main__":
    main()
