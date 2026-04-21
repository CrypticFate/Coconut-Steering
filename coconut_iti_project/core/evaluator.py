import gc
import os

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm


def _extract_answer(output_text):
    """Extract the final numerical answer from model output."""
    if "####" in output_text:
        pred = output_text.split("####")[-1].strip()
    else:
        pred = output_text.split(" ")[-1].strip()
    return pred.replace(",", "").replace(".", "").split(" ")[0]


def _check_correct(pred_clean, ground_truth):
    """Check if prediction matches ground truth."""
    return ground_truth in pred_clean or pred_clean in ground_truth


def print_sample_outputs(model, tokenizer, dataset, config, num_samples=2, phase_name=""):
    """Quick output check after training phases."""
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


def run_full_evaluation(coconut_model, test_data, tokenizer, config, truth_vector, latent_id):
    """
    Complete Phase 4 evaluation with three parts:
      Part 1: Structural Ablation Comparisons (No CoT, Text CoT, CCoT, Random Noise)
      Part 2: ITI Truth Vector Alpha Sweep
      Part 3: Confidence Deep Dive (Logit Analysis)

    All output is printed AND saved to phase4_evaluation.log.
    Returns (experiment_results, ablation_results).
    """
    coconut_model.eval()
    base_model = coconut_model.base_causallm
    n_latents = config.max_latent_tokens
    log_lines = []

    def log(msg=""):
        print(msg)
        log_lines.append(msg)

    log("Starting Phase 4: Full Pipeline Evaluation...")
    log("")

    # ==================================================================
    # PART 1: STRUCTURAL ABLATION COMPARISONS
    # ==================================================================
    log("=" * 65)
    log("PART 1: STRUCTURAL ABLATION COMPARISONS")
    log("=" * 65)

    ablation_results = {}

    # --- 1a. No CoT: question -> answer directly ---
    correct, total_tokens = 0, 0
    for sample in tqdm(test_data, desc="No CoT"):
        prompt = sample["question"] + "\n#### "
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(config.device)
        gt = sample["answer"].replace(",", "").strip()
        with torch.no_grad():
            out_ids = base_model.generate(
                input_ids, max_new_tokens=32, do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        total_tokens += out_ids.shape[1] - input_ids.shape[1]
        pred = _extract_answer(tokenizer.decode(out_ids[0], skip_special_tokens=True))
        if _check_correct(pred, gt):
            correct += 1
    ablation_results["no_cot"] = (correct / len(test_data), total_tokens / len(test_data))
    torch.cuda.empty_cache()

    # --- 1b. Text-based CoT: question + reasoning prompt ---
    correct, total_tokens = 0, 0
    for sample in tqdm(test_data, desc="Text based CoT"):
        prompt = sample["question"] + "\nLet's solve this step by step.\n"
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(config.device)
        gt = sample["answer"].replace(",", "").strip()
        with torch.no_grad():
            out_ids = base_model.generate(
                input_ids, max_new_tokens=128, do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        total_tokens += out_ids.shape[1] - input_ids.shape[1]
        pred = _extract_answer(tokenizer.decode(out_ids[0], skip_special_tokens=True))
        if _check_correct(pred, gt):
            correct += 1
    ablation_results["text_cot"] = (correct / len(test_data), total_tokens / len(test_data))
    torch.cuda.empty_cache()

    # --- 1c. Just CCoT (no steering, α=0) ---
    correct, total_tokens = 0, 0
    for sample in tqdm(test_data, desc="Just CCoT"):
        prompt = (
            sample["question"] + "\n<|start-latent|>"
            + "<|latent|>" * n_latents + "<|end-latent|>"
        )
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(config.device)
        gt = sample["answer"].replace(",", "").strip()
        with torch.no_grad():
            gen_ids, _, _ = coconut_model.generate_with_latents(
                input_ids, max_new_tokens=64, temperature=0.0,
            )
        total_tokens += gen_ids.shape[1] - input_ids.shape[1]
        pred = _extract_answer(tokenizer.decode(gen_ids[0], skip_special_tokens=True))
        if _check_correct(pred, gt):
            correct += 1
    ablation_results["ccot"] = (correct / len(test_data), total_tokens / len(test_data))
    torch.cuda.empty_cache()

    # --- 1d. Random Noise Steering (α=20.0) ---
    random_alpha = 20.0
    hidden_size = base_model.config.hidden_size
    random_vector = torch.randn(1, hidden_size).to(config.device)
    correct, total_tokens = 0, 0
    for sample in tqdm(test_data, desc=f"Random Noise (α={random_alpha})"):
        prompt = (
            sample["question"] + "\n<|start-latent|>"
            + "<|latent|>" * n_latents + "<|end-latent|>"
        )
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(config.device)
        gt = sample["answer"].replace(",", "").strip()
        with torch.no_grad():
            gen_ids, _, _ = coconut_model.generate_with_latents(
                input_ids, max_new_tokens=64, temperature=0.0,
                steering_vector=random_vector, alpha=random_alpha, gamma=config.alpha_decay,
            )
        total_tokens += gen_ids.shape[1] - input_ids.shape[1]
        pred = _extract_answer(tokenizer.decode(gen_ids[0], skip_special_tokens=True))
        if _check_correct(pred, gt):
            correct += 1
    ablation_results["random"] = (correct / len(test_data), total_tokens / len(test_data))
    torch.cuda.empty_cache()

    # Print ablation comparison table
    log(f"Compare against no cot                : Accuracy {ablation_results['no_cot'][0]:>5.2%} | Avg Tokens: {ablation_results['no_cot'][1]:.1f}")
    log(f"Compare against text based cot        : Accuracy {ablation_results['text_cot'][0]:>5.2%} | Avg Tokens: {ablation_results['text_cot'][1]:.1f}")
    log(f"Compare against just ccot             : Accuracy {ablation_results['ccot'][0]:>5.2%} | Avg Tokens: {ablation_results['ccot'][1]:.1f}")
    log(f"Compare against random noise (\u03b1={random_alpha}) : Accuracy {ablation_results['random'][0]:>5.2%} | Avg Tokens: {ablation_results['random'][1]:.1f}")
    log("")

    # ==================================================================
    # PART 2: ITI TRUTH VECTOR SWEEP
    # ==================================================================
    log("=" * 65)
    log("PART 2: ITI TRUTH VECTOR SWEEP")
    log("=" * 65)

    baseline_correctness = []
    experiment_results = []

    for alpha in config.alpha_sweep:
        correct = 0
        total = len(test_data)
        flips = 0
        faithfulness_list = []

        for idx, sample in enumerate(tqdm(test_data, desc=f"Sweeping Alpha={alpha}")):
            prompt = (
                sample["question"] + "\n<|start-latent|>"
                + "<|latent|>" * n_latents + "<|end-latent|>"
            )
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(config.device)
            gt = sample["answer"].replace(",", "").strip()

            with torch.no_grad():
                gen_ids, _, faith_score = coconut_model.generate_with_latents(
                    input_ids, max_new_tokens=64, temperature=0.0,
                    steering_vector=truth_vector if alpha > 0 else None,
                    alpha=alpha, gamma=config.alpha_decay,
                )

            pred = _extract_answer(tokenizer.decode(gen_ids[0], skip_special_tokens=True))
            is_correct = _check_correct(pred, gt)
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
        torch.cuda.empty_cache()

    # Print ITI sweep results
    log(f"{'Alpha':<6}| {'Accuracy':>9} | {'Flip Rate':>10} | {'Faithfulness'}")
    for res in experiment_results:
        log(
            f"{res['alpha']:<6}| {res['accuracy']:>8.2%} | "
            f"{res['flip_rate']:>9.2%} | {res['trajectory_faithfulness']:.4f}"
        )
    log("")

    # ==================================================================
    # PART 3: CONFIDENCE DEEP DIVE (Logit Analysis)
    # ==================================================================
    log("=" * 65)
    log("PART 3: CONFIDENCE DEEP DIVE (Logit Analysis)")
    log("=" * 65)

    sample = test_data[0]
    prompt = (
        sample["question"] + "\n<|start-latent|>"
        + "<|latent|>" * n_latents + "<|end-latent|>"
    )
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(config.device)
    gt = sample["answer"].replace(",", "").strip()

    target_token_ids = tokenizer.encode(gt, add_special_tokens=False)
    if len(target_token_ids) > 0:
        first_target = target_token_ids[0]
        target_str = tokenizer.decode([first_target])

        log(f"Question: {sample['question']}")
        log(f"Tracking Probability for target token: '{target_str}' (ID: {first_target})")
        log("-" * 50)

        for alpha in config.alpha_sweep:
            with torch.no_grad():
                outputs = coconut_model(
                    input_ids,
                    attention_mask=torch.ones_like(input_ids),
                    steering_vector=truth_vector if alpha > 0 else None,
                    alpha=alpha,
                    gamma=config.alpha_decay,
                )
                logits = outputs.logits[:, -1, :]
                probs = F.softmax(logits, dim=-1)
                target_prob = probs[0, first_target].item()
                top_id = torch.argmax(logits, dim=-1).item()
                top_prob = probs[0, top_id].item()
                top_str = tokenizer.decode([top_id])

            log(
                f"Alpha {alpha:>4.1f} | Target Token Prob: {target_prob:>6.2%} | "
                f"Top Pred: '{top_str}' ({top_prob:.2%})"
            )
    else:
        log("Could not parse ground truth for token tracking.")

    # ==================================================================
    # Save comprehensive log file
    # ==================================================================
    log_path = os.path.join(config.save_path, "phase4_evaluation.log")
    with open(log_path, "w") as f:
        f.write("\n".join(log_lines) + "\n")
    print(f"\n[LOG] Full evaluation log saved to {log_path}")

    return experiment_results, ablation_results
