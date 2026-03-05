import argparse
import os

import torch

from configs.config import Config
from core.evaluator import analyze_confidence, evaluate_with_iti, print_sample_outputs
from core.extractor import extract_truth_vector
from core.trainer import train_phase1
from data.data_loader import prepare_datasets
from models.coconut import initialize_model
from utils.helpers import clear_memory, set_seed
from utils.visualizer import plot_latent_pca, plot_loss_curve


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

    # Disable tokenizer parallelism to prevent deadlocks
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    config = Config()
    set_seed(config.seed)
    os.makedirs(config.save_path, exist_ok=True)

    # --- Data ---
    data_phase1, data_phase2, data_phase3, test_data = prepare_datasets(config)

    # --- Model ---
    coconut_model, tokenizer, latent_id, start_id, end_id = initialize_model(config)

    # --- Phase 1: Train ---
    if not args.skip_phase1:
        loss_history = train_phase1(
            coconut_model, data_phase1, tokenizer, config, latent_id, start_id, end_id
        )
        plot_loss_curve(loss_history, os.path.join(config.save_path, "loss_curve.png"))
        print_sample_outputs(coconut_model, tokenizer, test_data, config, phase_name="PHASE 1 (BASE)")
    else:
        checkpoint_path = os.path.join(config.save_path, "coconut_phase1.pt")
        print(f"Loading Phase 1 checkpoint from {checkpoint_path}...")
        coconut_model.load_state_dict(torch.load(checkpoint_path, map_location=config.device))
        print("Checkpoint loaded.")

    clear_memory()

    # --- Phase 2: Extract Truth Vector ---
    if not args.skip_phase2:
        truth_vector, correct_latents, wrong_latents = extract_truth_vector(
            coconut_model, data_phase2, tokenizer, config, latent_id, start_id, end_id
        )
        plot_latent_pca(correct_latents, wrong_latents, os.path.join(config.save_path, "pca_latents.html"))
    else:
        vector_path = os.path.join(config.save_path, "truth_vector.pt")
        print(f"Loading truth vector from {vector_path}...")
        truth_vector = torch.load(vector_path, map_location=config.device)
        print(f"Truth vector loaded. Shape: {truth_vector.shape}")

    clear_memory()

    # --- Phase 4: Evaluate with ITI ---
    experiment_results = evaluate_with_iti(
        coconut_model, test_data, tokenizer, config, truth_vector, latent_id
    )

    # --- Confidence Analysis ---
    analyze_confidence(coconut_model, test_data, tokenizer, config, truth_vector, latent_id)

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
