import os

import torch
from tqdm.auto import tqdm

from data.data_loader import get_hf_dataset
from utils.helpers import clear_memory


def extract_truth_vector(coconut_model, data_phase2, tokenizer, config, latent_id, start_id, end_id):
    print("Starting Phase 2: Calculating Global Truth Vector...")
    coconut_model.eval()

    ds_phase2 = get_hf_dataset(data_phase2, tokenizer)
    correct_latents = []
    wrong_latents = []
    n_latents_infer = config.max_latent_tokens

    for sample in tqdm(ds_phase2, desc="Phase 2 Inference"):
        prompt = (
            sample["question"]
            + "\n<|start-latent|>"
            + "<|latent|>" * n_latents_infer
            + "<|end-latent|>"
        )
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(config.device)
        ground_truth = sample["ground_truth"].replace(",", "").strip()

        for _ in range(config.num_generations_per_sample):
            generated_ids, mean_latent, _ = coconut_model.generate_with_latents(
                input_ids,
                max_new_tokens=64,
                temperature=config.generation_temperature,
            )

            output_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            pred = (
                output_text.split("####")[-1].strip()
                if "####" in output_text
                else output_text.split(" ")[-1].strip()
            )
            pred_clean = pred.replace(",", "").replace(".", "").split(" ")[0]

            if ground_truth in pred_clean or pred_clean in ground_truth:
                correct_latents.append(mean_latent)
            else:
                wrong_latents.append(mean_latent)

    print(f"Total Correct Paths: {len(correct_latents)}")
    print(f"Total Incorrect Paths: {len(wrong_latents)}")

    if len(correct_latents) > 0 and len(wrong_latents) > 0:
        mean_correct = torch.mean(torch.stack(correct_latents), dim=0)
        mean_wrong = torch.mean(torch.stack(wrong_latents), dim=0)
        truth_vector = (mean_correct - mean_wrong).to(config.device)
    else:
        print("WARNING: Missing either correct or wrong examples. Vector will be zero.")
        hidden_size = coconut_model.base_causallm.config.hidden_size
        truth_vector = torch.zeros((1, hidden_size)).to(config.device)

    print(f"Computed Global Truth Vector. Shape: {truth_vector.shape}")
    l2_norm = torch.norm(truth_vector).item()
    print(f"Truth Vector L2 Norm (Magnitude): {l2_norm:.6f}")

    if l2_norm < 1e-3:
        print("WARNING: Vector magnitude is extremely small.")

    # Save truth vector
    vector_path = os.path.join(config.save_path, "truth_vector.pt")
    torch.save(truth_vector, vector_path)
    print(f"Truth vector saved to {vector_path}")

    clear_memory()
    return truth_vector, correct_latents, wrong_latents
