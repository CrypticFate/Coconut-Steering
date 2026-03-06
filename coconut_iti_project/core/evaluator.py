import gc

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm


def print_sample_outputs(model, tokenizer, dataset, config, num_samples=2, phase_name=""):
    print(f"\n{'=' * 20} {phase_name} OUTPUT CHECK {'=' * 20}")
    model.eval()
    n_latents = config.max_latent_tokens

    sample_indices = [0, 1] if len(dataset) >= 2 else range(len(dataset))

    for idx in sample_indices:
        sample = dataset[idx]
        prompt = (
            sample["question"]
            + "\n<|start-latent|>"
            + "<|latent|>" * n_latents
            + "<|end-latent|>"
        )
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(config.device)

        with torch.no_grad():
            generated_ids, _, _ = model.generate_with_latents(
                input_ids,
                max_new_tokens=64,
                temperature=0.0,
            )

        output_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        prompt_decoded = tokenizer.decode(input_ids[0], skip_special_tokens=True)
        new_generation = output_text.replace(prompt_decoded, "").strip()

        print(f"\n[Question {idx + 1}]: {sample['question']}")
        print(f"[Ground Truth]: {sample['answer']}")
        print(f"[Model Output]: {new_generation}")
        print("-" * 60)

    del input_ids, generated_ids
    torch.cuda.empty_cache()
    gc.collect()


def evaluate_with_iti(coconut_model, test_data, tokenizer, config, truth_vector, latent_id):
    print("Starting Phase 4 Evaluation on Full Test Set...")
    coconut_model.eval()

    eval_subset = test_data
    n_latents_infer = config.max_latent_tokens

    baseline_correctness = []
    experiment_results = []

    for alpha in config.alpha_sweep:
        print(f"\nEvaluating Alpha = {alpha}")
        correct = 0
        total = len(eval_subset)
        flips = 0
        faithfulness_list = []

        for idx, sample in enumerate(tqdm(eval_subset, desc=f"Eval \u03b1={alpha}")):
            prompt = (
                sample["question"]
                + "\n<|start-latent|>"
                + "<|latent|>" * n_latents_infer
                + "<|end-latent|>"
            )
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(config.device)
            ground_truth = sample["answer"].replace(",", "").strip()

            with torch.no_grad():
                generated_ids, _, faith_score = coconut_model.generate_with_latents(
                    input_ids,
                    max_new_tokens=64,
                    temperature=0.0,
                    steering_vector=truth_vector if alpha > 0 else None,
                    alpha=alpha,
                    gamma=config.alpha_decay,
                )

            output_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            pred = (
                output_text.split("####")[-1].strip()
                if "####" in output_text
                else output_text.split(" ")[-1].strip()
            )
            pred_clean = pred.replace(",", "").replace(".", "").split(" ")[0]

            is_correct = ground_truth in pred_clean or pred_clean in ground_truth

            if is_correct:
                correct += 1

            if alpha > 0:
                faithfulness_list.append(faith_score)

            if alpha == 0.0:
                baseline_correctness.append(is_correct)
            else:
                if not baseline_correctness[idx] and is_correct:
                    flips += 1

        acc = correct / total
        baseline_errors = total - sum(baseline_correctness)
        flip_rate = (flips / baseline_errors) if baseline_errors > 0 else 0.0
        avg_faith = np.mean(faithfulness_list) if faithfulness_list else 0.0

        experiment_results.append({
            "alpha": alpha,
            "accuracy": acc,
            "flip_rate": flip_rate,
            "trajectory_faithfulness": avg_faith,
        })

    print("\n" + "=" * 50)
    print(f"{'Alpha':<8} | {'Accuracy':<10} | {'Flip Rate':<10} | {'Faithfulness'}")
    print("-" * 50)
    for res in experiment_results:
        print(
            f"{res['alpha']:<8} | {res['accuracy']:.2%}     | "
            f"{res['flip_rate']:.2%}     | {res['trajectory_faithfulness']:.4f}"
        )
    print("=" * 50)

    return experiment_results


def analyze_confidence(coconut_model, test_data, tokenizer, config, truth_vector, latent_id):
    print("Running Single-Sample Confidence Analysis...")
    coconut_model.eval()

    n_latents_infer = config.max_latent_tokens
    sample = test_data[0]
    prompt = (
        sample["question"]
        + "\n<|start-latent|>"
        + "<|latent|>" * n_latents_infer
        + "<|end-latent|>"
    )
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(config.device)
    ground_truth = sample["answer"].replace(",", "").strip()

    target_token_ids = tokenizer.encode(ground_truth, add_special_tokens=False)
    if len(target_token_ids) > 0:
        first_target_token = target_token_ids[0]
        target_token_string = tokenizer.decode([first_target_token])

        print(f"Question: {sample['question']}")
        print(f"Tracking Probability for target token: '{target_token_string}' (ID: {first_target_token})")
        print("-" * 50)

        for alpha in config.alpha_sweep:
            with torch.no_grad():
                outputs = coconut_model(
                    input_ids,
                    attention_mask=torch.ones_like(input_ids),
                    steering_vector=truth_vector if alpha > 0 else None,
                    alpha=alpha,
                    gamma=config.alpha_decay,
                )

                next_token_logits = outputs.logits[:, -1, :]
                probs = F.softmax(next_token_logits, dim=-1)

                target_prob = probs[0, first_target_token].item()

                top_token_id = torch.argmax(next_token_logits, dim=-1).item()
                top_token_prob = probs[0, top_token_id].item()
                top_token_str = tokenizer.decode([top_token_id])

                print(
                    f"Alpha {alpha:>4.1f} | Target Token Prob: {target_prob:>6.2%} | "
                    f"Top Pred: '{top_token_str}' ({top_token_prob:.2%})"
                )
    else:
        print("Could not parse ground truth for token tracking.")
