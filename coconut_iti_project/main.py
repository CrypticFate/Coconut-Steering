import os
import sys

# --- Must be set BEFORE any library imports to prevent TF/PyTorch CUDA conflict ---
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"           # Suppress TF logging entirely
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"           # Disable oneDNN ops
os.environ["CUDA_MODULE_LOADING"] = "LAZY"           # Lazy CUDA module loading
os.environ["TOKENIZERS_PARALLELISM"] = "false"        # Prevent tokenizer deadlocks

import argparse
import time

import torch

from configs.config import Config, set_seed
from core.evaluator import run_full_evaluation, print_sample_outputs
from core.extractor import extract_truth_vector
from core.trainer import train_phase1
from data.data_loader import prepare_datasets
from models.coconut import initialize_model
from utils.helpers import clear_memory, save_phase_log
from utils.visualizer import plot_latent_pca, plot_loss_curve


class TeeLogger:
    """Writes all stdout to both the terminal and a log file simultaneously."""
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()
        sys.stdout = self.terminal


def log_phase(phase_num, phase_name):
    print("\n" + "=" * 60)
    print(f"  PHASE {phase_num}: {phase_name}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="COCONUT ITI Pipeline (Qwen 1.5B Math Full Parameter)")
    parser.add_argument(
        "--skip-phase1", action="store_true",
        help="Skip Phase 1 training and load checkpoint from checkpoints/coconut_phase1.pt",
    )
    parser.add_argument(
        "--skip-phase2", action="store_true",
        help="Skip Phase 2 extraction and load truth_vector from checkpoints/truth_vector.pt",
    )
    args = parser.parse_args()

    config = Config()
    set_seed(config.seed)
    os.makedirs(config.save_path, exist_ok=True)

    # --- Pipeline-level log: capture ALL stdout to a master log file ---
    master_log_path = os.path.join(config.save_path, "pipeline_full.log")
    tee = TeeLogger(master_log_path)
    sys.stdout = tee

    pipeline_start = time.time()

    print("=" * 60)
    print("  COCONUT + ITI Steering Pipeline (Qwen 1.5B Math Full Parameter)")
    print("=" * 60)

    print(f"\nDevice: {config.device}")
    print(f"Model: {config.model_id}")
    print(f"BF16: {config.bf16}")
    print(f"Batch size: {config.batch_size_training} x {config.gradient_accumulation_steps} "
          f"= {config.batch_size_training * config.gradient_accumulation_steps} effective")
    print(f"Epochs: {config.num_epochs_total}")
    print(f"Alpha sweep: {config.alpha_sweep}")

    # --- Data ---
    print("\n" + "-" * 60)
    print("  Loading Datasets")
    print("-" * 60)
    data_phase1, data_phase2, test_data = prepare_datasets(config)

    # --- Model ---
    print("\n" + "-" * 60)
    print("  Initializing Model")
    print("-" * 60)
    coconut_model, tokenizer, latent_id, start_id, end_id = initialize_model(config)

    # =========================================================
    # Phase 1: Silent Thinking (Base Training)
    # =========================================================
    log_phase(1, "SILENT THINKING (Full Parameter COCONUT Training)")

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

        save_phase_log(config.save_path, 1, "SILENT THINKING (Full Parameter COCONUT Training)", (
            f"Model: {config.model_id}\n"
            f"Training Mode: Full Parameter (Native 32-bit AdamW)\n"
            f"Epochs: {config.num_epochs_total}\n"
            f"Learning Rate: {config.lr}\n"
            f"Batch Size: {config.batch_size_training} x {config.gradient_accumulation_steps} "
            f"= {config.batch_size_training * config.gradient_accumulation_steps} effective\n"
            f"Max Seq Length: {config.max_seq_len}\n"
            f"BF16: {config.bf16}\n"
            f"Training Samples: {len(data_phase1)}\n"
            f"Final Loss: {loss_history[-1]:.6f}\n"
            f"Min Loss: {min(loss_history):.6f}\n"
            f"Duration: {phase1_time / 60:.1f} minutes\n"
            f"Checkpoint: {os.path.join(config.save_path, 'coconut_phase1.pt')}\n"
        ))
    else:
        checkpoint_path = os.path.join(config.save_path, "coconut_phase1.pt")
        print(f"[SKIP] Loading Phase 1 checkpoint from {checkpoint_path}...")
        coconut_model.load_state_dict(torch.load(checkpoint_path, map_location=config.device))
        print("[SKIP] Checkpoint loaded successfully.")

        save_phase_log(config.save_path, 1, "SILENT THINKING (Full Parameter COCONUT Training)", (
            f"SKIPPED: Loaded from checkpoint\n"
            f"Checkpoint: {checkpoint_path}\n"
        ))

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

        save_phase_log(config.save_path, 2, "MIND READING (Global Truth Vector Extraction)", (
            f"Extraction Samples: {len(data_phase2)}\n"
            f"Correct Latent Paths: {len(correct_latents)}\n"
            f"Incorrect Latent Paths: {len(wrong_latents)}\n"
            f"Truth Vector Shape: {truth_vector.shape}\n"
            f"Truth Vector L2 Norm: {torch.norm(truth_vector).item():.6f}\n"
            f"Duration: {phase2_time / 60:.1f} minutes\n"
            f"Vector Saved: {os.path.join(config.save_path, 'truth_vector.pt')}\n"
        ))
    else:
        vector_path = os.path.join(config.save_path, "truth_vector.pt")
        print(f"[SKIP] Loading truth vector from {vector_path}...")
        truth_vector = torch.load(vector_path, map_location=config.device)
        print(f"[SKIP] Truth vector loaded. Shape: {truth_vector.shape}")

        save_phase_log(config.save_path, 2, "MIND READING (Global Truth Vector Extraction)", (
            f"SKIPPED: Loaded from file\n"
            f"Vector Path: {vector_path}\n"
            f"Truth Vector Shape: {truth_vector.shape}\n"
        ))

    clear_memory()

    # =========================================================
    # Phase 3: Mind Control (Intervention Setup)
    # =========================================================
    log_phase(3, "MIND CONTROL (Intervention Setup)")
    print("Steering vector is embedded in the COCONUT forward pass.")
    print(f"Alpha decay (gamma): {config.alpha_decay}")
    print(f"Intervention formula: h_new = h_old + (alpha * sigma * v_truth)")
    print("Phase 3 logic is built into the model architecture -- no separate step needed.")

    save_phase_log(config.save_path, 3, "MIND CONTROL (Intervention Setup)", (
        f"Alpha Decay (gamma): {config.alpha_decay}\n"
        f"Alpha Sweep Values: {config.alpha_sweep}\n"
        f"Intervention Formula: h_new = h_old + (alpha * sigma * v_truth)\n"
        f"Note: Steering logic is embedded in the Coconut forward pass.\n"
    ))

    # =========================================================
    # Phase 4: Final Exam (Full Evaluation)
    # =========================================================
    log_phase(4, "FINAL EXAM (Full Pipeline Evaluation)")

    phase4_start = time.time()
    experiment_results, ablation_results = run_full_evaluation(
        coconut_model, test_data, tokenizer, config, truth_vector, latent_id
    )
    phase4_time = time.time() - phase4_start
    print(f"\nPhase 4 completed in {phase4_time / 60:.1f} minutes")

    # Build Phase 4 log content
    phase4_lines = [
        f"Test Samples: {len(test_data)}\n",
        f"Alpha Sweep: {config.alpha_sweep}\n",
        f"Alpha Decay (gamma): {config.alpha_decay}\n",
        f"Duration: {phase4_time / 60:.1f} minutes\n\n",
        "--- STRUCTURAL ABLATIONS ---\n",
        f"No CoT:              Accuracy {ablation_results['no_cot'][0]:.2%} | Avg Tokens: {ablation_results['no_cot'][1]:.1f}\n",
        f"Text-based CoT:      Accuracy {ablation_results['text_cot'][0]:.2%} | Avg Tokens: {ablation_results['text_cot'][1]:.1f}\n",
        f"Just CCoT:           Accuracy {ablation_results['ccot'][0]:.2%} | Avg Tokens: {ablation_results['ccot'][1]:.1f}\n",
        f"Random Noise:        Accuracy {ablation_results['random'][0]:.2%} | Avg Tokens: {ablation_results['random'][1]:.1f}\n\n",
        "--- ITI SWEEP ---\n",
        f"{'Alpha':<8} | {'Accuracy':<10} | {'Flip Rate':<10} | {'Faithfulness'}\n",
        "-" * 50 + "\n",
    ]
    for res in experiment_results:
        phase4_lines.append(
            f"{res['alpha']:<8} | {res['accuracy']:.2%}     | "
            f"{res['flip_rate']:.2%}     | {res['trajectory_faithfulness']:.4f}\n"
        )
    save_phase_log(config.save_path, 4, "FINAL EXAM (Full Pipeline Evaluation)",
                   "".join(phase4_lines))

    # --- Summary ---
    total_time = time.time() - pipeline_start
    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Total runtime: {total_time / 60:.1f} minutes")
    print(f"Checkpoints saved to: {os.path.abspath(config.save_path)}")
    print(f"Loss curve: {os.path.join(config.save_path, 'loss_curve.png')}")
    print(f"PCA plot: {os.path.join(config.save_path, 'pca_latents.html')}")
    print(f"Master log: {master_log_path}")
    print(f"Evaluation log: {os.path.join(config.save_path, 'phase4_evaluation.log')}")

    # Pipeline summary log
    save_phase_log(config.save_path, 0, "PIPELINE SUMMARY", (
        f"Model: {config.model_id}\n"
        f"Training Mode: Full Parameter (Native 32-bit AdamW)\n"
        f"Device: {config.device}\n"
        f"BF16: {config.bf16}\n"
        f"Total Runtime: {total_time / 60:.1f} minutes\n"
        f"Checkpoints Directory: {os.path.abspath(config.save_path)}\n\n"
        f"Log Files:\n"
        f"  - {os.path.join(config.save_path, 'pipeline_full.log')} (master)\n"
        f"  - {os.path.join(config.save_path, 'phase4_evaluation.log')}\n"
        f"  - {os.path.join(config.save_path, 'phase1_log.txt')}\n"
        f"  - {os.path.join(config.save_path, 'phase2_log.txt')}\n"
        f"  - {os.path.join(config.save_path, 'phase3_log.txt')}\n"
        f"  - {os.path.join(config.save_path, 'phase4_log.txt')}\n"
    ))

    # Restore stdout and close log
    tee.close()


if __name__ == "__main__":
    main()
