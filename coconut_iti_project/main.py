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

from core.alpha_tuner import tune_alpha
from configs.config import Config, set_seed
from core.evaluator import run_full_evaluation, print_sample_outputs
from core.extractor import extract_truth_vector
from core.trainer import train_phase1
from data.data_loader import prepare_datasets
from models.coconut import initialize_model
from utils.helpers import (
    clear_memory, save_phase_log, save_config_snapshot,
    setup_run_directory, activate_logging, deactivate_logging,
)
from utils.visualizer import (
    plot_decoded_latents,
    plot_latent_pca,
    plot_loss_curve,
    plot_pipeline_summary,
)


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
    parser.add_argument(
        "--skip-phase3", action="store_true",
        help="Skip Phase 3 alpha tuning and load alpha_star from checkpoints/alpha_star.pt",
    )
    args = parser.parse_args()

    config = Config()
    set_seed(config.seed)
    os.makedirs(config.save_path, exist_ok=True)

    # =========================================================
    # Create a timestamped run directory for ALL outputs
    # =========================================================
    run_dir = setup_run_directory(config.save_path)
    plots_dir = os.path.join(run_dir, "plots")
    ckpt_dir = os.path.join(run_dir, "checkpoints")

    # Capture BOTH stdout AND stderr to the master log
    stdout_tee, stderr_tee = activate_logging(run_dir)

    pipeline_start = time.time()

    print("=" * 60)
    print("  COCONUT + ITI Steering Pipeline (Qwen 1.5B Math Full Parameter)")
    print("=" * 60)
    print(f"\nRun directory: {os.path.abspath(run_dir)}")
    print(f"(Runs are stored under: {os.path.abspath(os.path.join(config.save_path, 'log'))}/)")

    print(f"\nDevice: {config.device}")
    print(f"Model: {config.model_id}")
    print(f"BF16: {config.bf16}")
    print(f"Batch size: {config.batch_size_training} x {config.gradient_accumulation_steps} "
          f"= {config.batch_size_training * config.gradient_accumulation_steps} effective")
    print(f"Epochs: {config.num_epochs_total}")
    print(f"Alpha sweep: {config.alpha_sweep}")

    # Save config snapshot for reproducibility
    save_config_snapshot(run_dir, config)

    # --- Data ---
    print("\n" + "-" * 60)
    print("  Loading Datasets")
    print("-" * 60)
    data_phase1, data_phase1_val, data_phase2, data_val, test_data = prepare_datasets(config, include_val=True)

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
        phase1_artifacts = train_phase1(
            coconut_model, data_phase1, tokenizer, config, latent_id, start_id, end_id,
            run_dir=run_dir,
        )
        loss_history = phase1_artifacts["loss_history"]
        phase1_time = time.time() - phase1_start
        print(f"\nPhase 1 completed in {phase1_time / 60:.1f} minutes")
        print(f"Final training loss: {loss_history[-1]:.4f}")

        plot_loss_curve(loss_history, os.path.join(plots_dir, "phase1", "loss_curve.png"))
        # P1-D decoded latent interpretations
        decoded_examples = []
        for sample in test_data[:5]:
            prompt = (
                sample["question"] + "\n<|start-latent|>"
                + "<|latent|>" * config.max_latent_tokens + "<|end-latent|>"
            )
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(config.device)
            with torch.no_grad():
                _generated_ids, _, _ = coconut_model.generate_with_latents(
                    input_ids,
                    max_new_tokens=config.max_new_tokens_ccot,
                    temperature=0.0,
                )
                lm_head = coconut_model.base_causallm.lm_head
                steps = []
                for t, h in enumerate(coconut_model.last_generation_latents):
                    logits = lm_head(h.to(config.device))
                    probs = torch.softmax(logits, dim=-1)[0]
                    topv, topi = torch.topk(probs, k=3)
                    top3 = []
                    for p, idx in zip(topv.tolist(), topi.tolist()):
                        tok = tokenizer.decode([idx]).strip().replace("\n", " ")
                        top3.append((tok if tok else f"<id:{idx}>", p))
                    steps.append({"t": t, "top3": top3})
            decoded_examples.append({"question": sample["question"], "steps": steps})
        plot_decoded_latents(decoded_examples, os.path.join(plots_dir, "phase1", "decoded_latents.png"))
        print_sample_outputs(coconut_model, tokenizer, test_data, config, phase_name="PHASE 1 (BASE)")

        save_phase_log(run_dir, 1, "SILENT THINKING (Full Parameter COCONUT Training)", (
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
            f"Checkpoint: {os.path.join(ckpt_dir, 'coconut_phase1.pt')}\n"
        ))
    else:
        checkpoint_path = os.path.join(config.save_path, "coconut_phase1.pt")
        print(f"[SKIP] Loading Phase 1 checkpoint from {checkpoint_path}...")
        coconut_model.load_state_dict(torch.load(checkpoint_path, map_location=config.device))
        print("[SKIP] Checkpoint loaded successfully.")

        save_phase_log(run_dir, 1, "SILENT THINKING (Full Parameter COCONUT Training)", (
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
            coconut_model, data_phase2, tokenizer, config, latent_id, start_id, end_id,
            run_dir=run_dir,
        )
        phase2_time = time.time() - phase2_start
        print(f"\nPhase 2 completed in {phase2_time / 60:.1f} minutes")

        plot_latent_pca(correct_latents, wrong_latents, os.path.join(plots_dir, "phase2", "pca_latents.html"))

        save_phase_log(run_dir, 2, "MIND READING (Global Truth Vector Extraction)", (
            f"Extraction Samples: {len(data_phase2)}\n"
            f"Correct Latent Paths: {len(correct_latents)}\n"
            f"Incorrect Latent Paths: {len(wrong_latents)}\n"
            f"Truth Vector Shape: {truth_vector.shape}\n"
            f"Truth Vector L2 Norm: {torch.norm(truth_vector).item():.6f}\n"
            f"Duration: {phase2_time / 60:.1f} minutes\n"
            f"Vector Saved: {os.path.join(ckpt_dir, 'truth_vector.pt')}\n"
        ))
    else:
        vector_path = (
            config.truth_vector_path
            or os.path.join(config.save_path, "truth_vector.pt")
        )
        print(f"[SKIP] Loading truth vector from {vector_path}...")
        truth_vector = torch.load(vector_path, map_location=config.device)
        print(f"[SKIP] Truth vector loaded. Shape: {truth_vector.shape}")

        save_phase_log(run_dir, 2, "MIND READING (Global Truth Vector Extraction)", (
            f"SKIPPED: Loaded from file\n"
            f"Vector Path: {vector_path}\n"
            f"Truth Vector Shape: {truth_vector.shape}\n"
        ))

    clear_memory()

    # =========================================================
    # Phase 3: Mind Control (Alpha Tuning)
    # =========================================================
    log_phase(3, "MIND CONTROL (Gradient-Based Alpha Tuning)")

    if not args.skip_phase3:
        phase3_start = time.time()
        alpha_star, alpha_metadata = tune_alpha(
            coconut_model, data_val, tokenizer, config, truth_vector, run_dir=run_dir
        )
        phase3_time = time.time() - phase3_start
        gradient_check = alpha_metadata.get("gradient_check", {})
        print(f"\nPhase 3 completed in {phase3_time / 60:.1f} minutes")

        save_phase_log(run_dir, 3, "MIND CONTROL (Gradient-Based Alpha Tuning)", (
            f"Validation Samples: {len(data_val)}\n"
            f"Alpha*: {float(alpha_star.detach().cpu()):.6f}\n"
            f"Alpha Max: {config.alpha_max}\n"
            "Sigma definition: σ = h.std(dim=-1, keepdim=True) across hidden units (Li et al. 2023 ITI)\n"
            f"Alpha Decay (gamma): {config.alpha_decay}\n"
            "Gamma note: default γ=0.95 (configurable); report γ=1.0 ablation separately.\n"
            f"Lambda Align: {config.lambda_align}\n"
            f"Lambda Mag: {config.lambda_mag}\n"
            f"Best D_val_es Accuracy: {alpha_metadata.get('best_early_stop_accuracy')}\n"
            f"Finite Difference Rel Error: {gradient_check.get('relative_error')}\n"
            f"Finite Difference Passed: {gradient_check.get('passed')}\n"
            f"Duration: {phase3_time / 60:.1f} minutes\n"
            f"Alpha Saved: {os.path.join(ckpt_dir, 'alpha_star.pt')}\n"
        ))
    else:
        alpha_path = (
            config.alpha_star_path
            or os.path.join(config.save_path, "alpha_star.pt")
        )
        print(f"[SKIP] Loading alpha* from {alpha_path}...")
        alpha_star = torch.load(alpha_path, map_location=config.device)
        print(f"[SKIP] alpha* loaded: {float(alpha_star.detach().cpu()):.6f}")

        save_phase_log(run_dir, 3, "MIND CONTROL (Gradient-Based Alpha Tuning)", (
            f"SKIPPED: Loaded from file\n"
            f"Alpha Path: {alpha_path}\n"
            f"Alpha*: {float(alpha_star.detach().cpu()):.6f}\n"
        ))

    # =========================================================
    # Phase 4: Final Exam (Full Evaluation)
    # =========================================================
    log_phase(4, "FINAL EXAM (Full Pipeline Evaluation)")

    phase4_start = time.time()
    experiment_results, _results_by_key, sweep_truth_rows, sweep_random_rows = run_full_evaluation(
        coconut_model, test_data, tokenizer, config, truth_vector, latent_id,
        alpha_star=alpha_star,
        run_dir=run_dir,
    )
    phase4_time = time.time() - phase4_start
    print(f"\nPhase 4 completed in {phase4_time / 60:.1f} minutes")

    # Build Phase 4 log content
    phase4_lines = [
        f"Test Samples: {len(test_data)}\n",
        f"Alpha*: {float(alpha_star.detach().cpu()):.6f}\n",
        f"Alpha Decay (gamma): {config.alpha_decay}\n",
        "Evaluation split note: This Phase 4 report is locked-test only. "
        "Validation alpha sweeps are logged in Phase 3 (alpha_tuning.json).\n",
        "Compression Ratio definition: num_latent_tokens / mean_text_cot_token_count.\n",
        f"Duration: {phase4_time / 60:.1f} minutes\n\n",
        "--- FINAL CONDITIONS ---\n",
        f"{'Condition':<30} | {'Accuracy':<10} | {'Flip Rate':<10} | {'Tokens':<8} | {'Latency'}\n",
        "-" * 86 + "\n",
    ]
    for res in experiment_results:
        phase4_lines.append(
            f"{res['condition']:<30} | {res['accuracy']:.2%}     | "
            f"{res['flip_rate']:.2%}     | {res['token_count']:<8.1f} | "
            f"{res['latency']:.3f}s\n"
        )
    if sweep_truth_rows:
        phase4_lines.append("\n--- STEERING SWEEP (Truth vector) ---\n")
        phase4_lines.append(
            f"{'Condition':<42} | {'Accuracy':<10} | {'Flip':<10} | {'Tokens':<8} | {'Latency'}\n"
        )
        phase4_lines.append("-" * 92 + "\n")
        for res in sweep_truth_rows:
            phase4_lines.append(
                f"{res['condition']:<42} | {res['accuracy']:.2%}     | "
                f"{res['flip_rate']:.2%}     | {res['token_count']:<8.1f} | "
                f"{res['latency']:.3f}s\n"
            )
    if sweep_random_rows:
        phase4_lines.append("\n--- STEERING SWEEP (Random noise) ---\n")
        phase4_lines.append(
            f"{'Condition':<42} | {'Accuracy':<10} | {'Flip':<10} | {'Tokens':<8} | {'Latency'}\n"
        )
        phase4_lines.append("-" * 92 + "\n")
        for res in sweep_random_rows:
            phase4_lines.append(
                f"{res['condition']:<42} | {res['accuracy']:.2%}     | "
                f"{res['flip_rate']:.2%}     | {res['token_count']:<8.1f} | "
                f"{res['latency']:.3f}s\n"
            )
    save_phase_log(run_dir, 4, "FINAL EXAM (Full Pipeline Evaluation)",
                   "".join(phase4_lines))

    # --- Summary ---
    total_time = time.time() - pipeline_start
    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Total runtime: {total_time / 60:.1f} minutes")
    print(f"Run directory (under log/): {os.path.abspath(run_dir)}")
    print(f"  Checkpoints:  {ckpt_dir}/")
    print(f"  Plots:        {plots_dir}/")
    print(f"  Logs & JSON: {os.path.join(run_dir, 'logs')}/")

    # Cross-phase summary plot
    by_key = {r["key"]: r for r in experiment_results}
    stage_labels = [
        "Phase1 COCONUT",
        "Phase4 α=0",
        "Phase4 α* Random",
        "Phase4 α* Truth",
    ]
    stage_accs = [
        by_key["ccot"]["accuracy"],
        by_key["ccot"]["accuracy"],
        by_key["random"]["accuracy"],
        by_key["truth"]["accuracy"],
    ]
    plot_pipeline_summary(stage_labels, stage_accs, os.path.join(plots_dir, "pipeline_summary.png"))

    # Pipeline summary log
    save_phase_log(run_dir, 0, "PIPELINE SUMMARY", (
        f"Model: {config.model_id}\n"
        f"Training Mode: Full Parameter (Native 32-bit AdamW)\n"
        f"Device: {config.device}\n"
        f"BF16: {config.bf16}\n"
        f"Total Runtime: {total_time / 60:.1f} minutes\n"
        f"Run Directory: {os.path.abspath(run_dir)}\n\n"
        f"All artifacts for this run are under: <save_path>/log/run_<timestamp>/\n\n"
        f"Output Structure:\n"
        f"  {run_dir}/\n"
        f"    logs/\n"
        f"      pipeline_full.log       (master stdout + stderr)\n"
        f"      training_loss.csv       (per-epoch loss)\n"
        f"      extraction.log          (Phase 2 stats)\n"
        f"      phase4_evaluation.log   (evaluation results)\n"
        f"      config_snapshot.json\n"
        f"      alpha_tuning.json, truth_vector_metadata.json\n"
        f"      phase0_log.txt … phase4_log.txt (phase summaries)\n"
        f"    plots/\n"
        f"      loss_curve.png\n"
        f"      pca_latents.html\n"
        f"    checkpoints/\n"
        f"      coconut_phase1.pt\n"
        f"      truth_vector.pt\n"
        f"      alpha_star.pt\n"
        f"      stage*_epoch*.pt        (per-stage snapshots)\n"
        f"    results/\n"
        f"      metrics.json\n"
    ))

    # Restore stdout/stderr and close log
    deactivate_logging(stdout_tee, stderr_tee)


if __name__ == "__main__":
    main()
