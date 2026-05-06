import gc
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from utils.helpers import answers_match, extract_final_answer
from utils.visualizer import (
    plot_alignment_coherence,
    plot_alpha_sweep_test,
    plot_efficiency,
    plot_flip_examples,
    plot_flip_rate,
    plot_main_comparison,
)


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


def _legacy_run_full_evaluation(coconut_model, test_data, tokenizer, config, truth_vector, latent_id, run_dir=None):
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
                pad_token_id=tokenizer.pad_token_id,
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
                pad_token_id=tokenizer.pad_token_id,
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
    log_dir = os.path.join(run_dir, "logs") if run_dir else config.save_path
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "phase4_evaluation.log")
    with open(log_path, "w") as f:
        f.write("\n".join(log_lines) + "\n")
    print(f"\n[LOG] Full evaluation log saved to {log_path}")

    return experiment_results, ablation_results


def _latent_prompt_v2(sample, n_latents):
    return (
        sample["question"]
        + "\n<|start-latent|>"
        + "<|latent|>" * n_latents
        + "<|end-latent|>"
    )


def _reasoning_token_count_v2(sample, tokenizer):
    reasoning = "\n".join(sample.get("steps", []))
    if not reasoning.strip():
        return 0
    return len(tokenizer.encode(reasoning, add_special_tokens=False))


def _compression_ratio_v2(sample, tokenizer, compressed_tokens):
    original_tokens = _reasoning_token_count_v2(sample, tokenizer)
    if original_tokens == 0:
        return 0.0
    return compressed_tokens / original_tokens


def _summarize_condition_v2(key, label, correctness, token_counts, latencies,
                            coherence_scores, alignment_scores, compression_ratios,
                            baseline_correctness=None):
    total = len(correctness)
    correct = sum(correctness)
    baseline_errors = 0
    flips = 0
    if baseline_correctness is not None:
        for base_ok, cond_ok in zip(baseline_correctness, correctness):
            if not base_ok:
                baseline_errors += 1
                if cond_ok:
                    flips += 1

    return {
        "key": key,
        "condition": label,
        "accuracy": correct / total if total else 0.0,
        "flip_rate": flips / baseline_errors if baseline_errors else 0.0,
        "token_count": float(np.mean(token_counts)) if token_counts else 0.0,
        "latency": float(np.mean(latencies)) if latencies else 0.0,
        "trajectory_coherence": float(np.mean(coherence_scores)) if coherence_scores else 0.0,
        "truth_alignment": float(np.mean(alignment_scores)) if alignment_scores else 0.0,
        "compression_ratio": float(np.mean(compression_ratios)) if compression_ratios else 0.0,
        "correct": correct,
        "total": total,
    }


def _evaluate_direct_v2(base_model, test_data, tokenizer, config, mode):
    correctness, token_counts, latencies, coherence, alignment, compression_ratios = [], [], [], [], [], []
    if mode == "no_cot":
        desc = "No CoT"
        max_new_tokens = config.max_new_tokens_no_cot
    else:
        desc = "Text CoT"
        max_new_tokens = config.max_new_tokens_text_cot

    for sample in tqdm(test_data, desc=desc):
        if mode == "no_cot":
            prompt = sample["question"] + "\n#### "
            reasoning_tokens = 0
            ratio = 0.0
        else:
            prompt = sample["question"] + "\nLet's solve this step by step.\n"
            reasoning_tokens = None
            ratio = 1.0

        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(config.device)
        start = time.perf_counter()
        with torch.no_grad():
            out_ids = base_model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        latencies.append(time.perf_counter() - start)
        generated_tokens = out_ids.shape[1] - input_ids.shape[1]
        token_counts.append(float(generated_tokens if reasoning_tokens is None else reasoning_tokens))
        compression_ratios.append(ratio)
        coherence.append(0.0)
        alignment.append(0.0)
        predicted = extract_final_answer(tokenizer.decode(out_ids[0], skip_special_tokens=True))
        correctness.append(answers_match(predicted, sample["answer"]))

    return correctness, token_counts, latencies, coherence, alignment, compression_ratios


def _evaluate_ccot_v2(coconut_model, test_data, tokenizer, config, steering_vector=None,
                      alpha=0.0, desc="CCoT", steering_mode="vector", truth_reference=None,
                      include_details=False):
    correctness, token_counts, latencies, coherence, alignment, compression_ratios = [], [], [], [], [], []
    n_latents = config.max_latent_tokens
    truth_ref_cpu = None
    if truth_reference is not None:
        truth_ref_cpu = F.normalize(truth_reference.detach().cpu().float(), p=2, dim=-1)

    details = []
    for sample in tqdm(test_data, desc=desc):
        prompt = _latent_prompt_v2(sample, n_latents)
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(config.device)
        start = time.perf_counter()
        with torch.no_grad():
            gen_ids, mean_latent, coherence_score = coconut_model.generate_with_latents(
                input_ids,
                max_new_tokens=config.max_new_tokens_ccot,
                temperature=0.0,
                steering_vector=steering_vector,
                alpha=alpha,
                gamma=config.alpha_decay,
                steering_mode=steering_mode,
            )
        latencies.append(time.perf_counter() - start)
        token_counts.append(float(n_latents))
        compression_ratios.append(_compression_ratio_v2(sample, tokenizer, n_latents))
        coherence.append(float(coherence_score))
        if mean_latent is not None and truth_ref_cpu is not None:
            align = F.cosine_similarity(
                F.normalize(mean_latent.float(), p=2, dim=-1),
                truth_ref_cpu,
                dim=-1,
            ).mean().item()
        else:
            align = 0.0
        alignment.append(float(align))
        predicted = extract_final_answer(tokenizer.decode(gen_ids[0], skip_special_tokens=True))
        is_ok = answers_match(predicted, sample["answer"])
        correctness.append(is_ok)
        if include_details:
            details.append({
                "qid": sample.get("qid"),
                "question": sample["question"],
                "gold": sample["answer"],
                "pred": predicted,
                "correct": bool(is_ok),
            })

    if include_details:
        return correctness, token_counts, latencies, coherence, alignment, compression_ratios, details
    return correctness, token_counts, latencies, coherence, alignment, compression_ratios


def _phase4_steering_alpha_grid(config, alpha_star_value):
    """Positive alphas to evaluate in Phase 4 sweep, always including optimized alpha*."""
    sweep = getattr(config, "phase4_alpha_sweep", None)
    if sweep is None:
        sweep = [a for a in config.alpha_sweep if a > 0]
    alphas = {float(a) for a in sweep}
    alphas.add(float(alpha_star_value))
    return sorted(alphas)


def _format_alpha_label(alpha, alpha_star_value):
    if math.isclose(alpha, alpha_star_value, rel_tol=0.0, abs_tol=1e-5):
        return f"{alpha:.4f} (α* optimized)"
    s = f"{alpha:.4f}".rstrip("0").rstrip(".")
    return s if s else "0"


def _format_alpha_sweep_cell(alpha, alpha_star_value):
    if math.isclose(alpha, alpha_star_value, rel_tol=0.0, abs_tol=1e-5):
        return f"{alpha:.4f} (best α*)"
    return _format_alpha_label(alpha, alpha_star_value)


def _load_alpha_star_v2(config, run_dir):
    if config.alpha_star_path:
        path = config.alpha_star_path
    elif run_dir:
        path = os.path.join(run_dir, "checkpoints", "alpha_star.pt")
    else:
        path = os.path.join(config.save_path, "alpha_star.pt")

    if os.path.exists(path):
        alpha = torch.load(path, map_location=config.device)
        return float(alpha.detach().cpu()) if torch.is_tensor(alpha) else float(alpha)

    print("WARNING: alpha_star.pt not found; using configured alpha_initial.")
    return float(config.alpha_initial)


def run_full_evaluation(coconut_model, test_data, tokenizer, config, truth_vector, latent_id,
                        alpha_star=None, run_dir=None):
    """
    Phase 4 locked-test evaluation across the five protocol conditions:
    No CoT, Text CoT, CCoT baseline, CCoT + random noise, and CCoT + truth vector.
    """
    coconut_model.eval()
    base_model = coconut_model.base_causallm
    log_lines = []
    alpha_value = float(alpha_star.detach().cpu()) if torch.is_tensor(alpha_star) else (
        float(alpha_star) if alpha_star is not None else _load_alpha_star_v2(config, run_dir)
    )
    truth_vector = F.normalize(truth_vector.to(config.device), p=2, dim=-1)

    def log(msg=""):
        print(msg)
        log_lines.append(msg)

    log("Starting Phase 4: Locked-Test Evaluation...")
    log(f"Using learned alpha*: {alpha_value:.4f}")
    log("Note: Phase 4 tables are on the locked test set; validation alpha sweep lives in Phase 3 outputs.")
    log("")

    no_cot = _evaluate_direct_v2(base_model, test_data, tokenizer, config, mode="no_cot")
    text_cot = _evaluate_direct_v2(base_model, test_data, tokenizer, config, mode="text_cot")
    ccot = _evaluate_ccot_v2(
        coconut_model, test_data, tokenizer, config, desc="CCoT baseline", truth_reference=truth_vector,
        include_details=True,
    )

    generator = torch.Generator(device=config.device).manual_seed(config.random_noise_seed)
    random_vector = torch.randn(
        1,
        base_model.config.hidden_size,
        generator=generator,
        device=config.device,
    )
    random_vector = F.normalize(random_vector, p=2, dim=-1)

    random_noise = _evaluate_ccot_v2(
        coconut_model,
        test_data,
        tokenizer,
        config,
        steering_vector=random_vector,
        alpha=alpha_value,
        desc=f"Random noise α={alpha_value:.4f}",
        truth_reference=truth_vector,
        include_details=True,
    )
    truth_steered = _evaluate_ccot_v2(
        coconut_model,
        test_data,
        tokenizer,
        config,
        steering_vector=truth_vector,
        alpha=alpha_value,
        desc=f"Truth vector α={alpha_value:.4f}",
        truth_reference=truth_vector,
        include_details=True,
    )

    ccot_correctness = ccot[0]
    ccot_details = ccot[-1]
    random_details = random_noise[-1]
    truth_details = truth_steered[-1]
    alpha_lbl = _format_alpha_label(alpha_value, alpha_value)
    rows = [
        _summarize_condition_v2("no_cot", "No CoT", *no_cot),
        _summarize_condition_v2("text_cot", "Text CoT", *text_cot),
        _summarize_condition_v2("ccot", "CCoT (baseline, alpha=0)", *ccot[:-1]),
        _summarize_condition_v2(
            "random",
            f"CCoT + Random Noise (α={alpha_lbl})",
            *random_noise[:-1],
            baseline_correctness=ccot_correctness,
        ),
        _summarize_condition_v2(
            "truth",
            f"CCoT + Truth Vector (α={alpha_lbl})",
            *truth_steered[:-1],
            baseline_correctness=ccot_correctness,
        ),
    ]
    for row in rows:
        row.setdefault("alpha", None)
        row.setdefault("steering", None)
    rows[3]["alpha"] = alpha_value
    rows[3]["steering"] = "random"
    rows[4]["alpha"] = alpha_value
    rows[4]["steering"] = "truth"
    results_by_key = {row["key"]: row for row in rows}

    sweep_truth_rows = []
    sweep_random_rows = []
    grid = _phase4_steering_alpha_grid(config, alpha_value) if getattr(config, "phase4_run_alpha_sweep", True) else []
    if getattr(config, "phase4_run_alpha_sweep", True):
        sweep_random = getattr(config, "phase4_sweep_random", True)
        cached_truth = {}
        cached_random = {}
        for a in grid:
            a = float(a)
            if math.isclose(a, alpha_value, rel_tol=0.0, abs_tol=1e-5):
                cached_truth[a] = truth_steered
                cached_random[a] = random_noise
        log("")
        log("=" * 96)
        log("STEERING SWEEP ON LOCKED TEST (multiple α; α* row = Phase 3 optimum)")
        log("=" * 96)
        log(
            f"{'Steering':<14} | {'α':>14} | {'Accuracy':>8} | {'Flip':>8} | "
            f"{'Tokens':>8} | {'Comp.R':>7} | {'Coh.':>7} | {'Align':>7} | {'Latency'}"
        )
        log("-" * 96)
        for a in grid:
            a = float(a)
            alpha_disp = _format_alpha_label(a, alpha_value)
            truth_a = cached_truth.get(a)
            if truth_a is None:
                truth_a = _evaluate_ccot_v2(
                    coconut_model,
                    test_data,
                    tokenizer,
                    config,
                    steering_vector=truth_vector,
                    alpha=a,
                    desc=f"Truth sweep α={alpha_disp}",
                    truth_reference=truth_vector,
                )
            t_row = _summarize_condition_v2(
                f"truth_a_{a}",
                f"Truth ({alpha_disp})",
                *truth_a,
                baseline_correctness=ccot_correctness,
            )
            t_row["alpha"] = a
            t_row["steering"] = "truth"
            sweep_truth_rows.append(t_row)
            log(
                f"{'Truth vector':<14} | {alpha_disp:>14} | {t_row['accuracy']:>7.2%} | "
                f"{t_row['flip_rate']:>7.2%} | {t_row['token_count']:>8.1f} | "
                f"{t_row['compression_ratio']:>7.3f} | {t_row['trajectory_coherence']:>7.4f} | "
                f"{t_row['truth_alignment']:>7.4f} | "
                f"{t_row['latency']:.3f}s"
            )
            if sweep_random:
                rand_a = cached_random.get(a)
                if rand_a is None:
                    rand_a = _evaluate_ccot_v2(
                        coconut_model,
                        test_data,
                        tokenizer,
                        config,
                        steering_vector=random_vector,
                        alpha=a,
                        desc=f"Random sweep α={alpha_disp}",
                        truth_reference=truth_vector,
                    )
                r_row = _summarize_condition_v2(
                    f"random_a_{a}",
                    f"Random ({alpha_disp})",
                    *rand_a,
                    baseline_correctness=ccot_correctness,
                )
                r_row["alpha"] = a
                r_row["steering"] = "random"
                sweep_random_rows.append(r_row)
                log(
                    f"{'Random noise':<14} | {alpha_disp:>14} | {r_row['accuracy']:>7.2%} | "
                    f"{r_row['flip_rate']:>7.2%} | {r_row['token_count']:>8.1f} | "
                    f"{r_row['compression_ratio']:>7.3f} | {r_row['trajectory_coherence']:>7.4f} | "
                    f"{r_row['truth_alignment']:>7.4f} | "
                    f"{r_row['latency']:.3f}s"
                )

    # Table 1: compact alpha sweep (truth vector), includes best α* explicitly.
    if sweep_truth_rows:
        log("")
        log("=" * 96)
        log("ALPHA SWEEP TABLE (Truth Vector on locked test; includes best α*)")
        log("=" * 96)
        log(f"{'Alpha':<22} | {'Accuracy':>9} | {'Flip Rate':>10} | {'Faithfulness'}")
        log("-" * 96)
        for row in sweep_truth_rows:
            a_cell = _format_alpha_sweep_cell(row["alpha"], alpha_value)
            log(
                f"{a_cell:<22} | {row['accuracy']:>8.2%} | {row['flip_rate']:>9.2%} | "
                f"{row['trajectory_coherence']:.4f}"
            )

    # Table 2: final protocol comparison at best α*.
    log("")
    log("=" * 96)
    log(f"FINAL COMPARISON TABLE (best α*={alpha_value:.4f} used)")
    log("Random-noise condition uses a random unit vector at best α*.")
    log("=" * 96)
    log(
        f"{'Condition':<30} | {'Accuracy':>8} | {'Flip':>8} | "
        f"{'Tokens':>8} | {'Comp.R':>7} | {'Coh.':>7} | {'Align':>7} | {'Latency'}"
    )
    log("-" * 96)
    for row in rows:
        log(
            f"{row['condition']:<30} | {row['accuracy']:>7.2%} | "
            f"{row['flip_rate']:>7.2%} | {row['token_count']:>8.1f} | "
            f"{row['compression_ratio']:>7.3f} | {row['trajectory_coherence']:>7.4f} | "
            f"{row['truth_alignment']:>7.4f} | "
            f"{row['latency']:.3f}s"
        )

    log_dir = os.path.join(run_dir, "logs") if run_dir else config.save_path
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "phase4_evaluation.log")
    with open(log_path, "w") as f:
        f.write("\n".join(log_lines) + "\n")
    print(f"\n[LOG] Full evaluation log saved to {log_path}")

    results_dir = os.path.join(run_dir, "results") if run_dir else config.save_path
    os.makedirs(results_dir, exist_ok=True)
    metrics_path = os.path.join(results_dir, "metrics.json")
    metrics_payload = {
        "alpha_star": alpha_value,
        "phase4_alpha_sweep_grid": grid,
        "conditions": rows,
        "alpha_sweep_truth": sweep_truth_rows,
        "alpha_sweep_random": sweep_random_rows,
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics_payload, f, indent=2)
    print(f"[LOG] Metrics saved to {metrics_path}")

    if run_dir:
        phase4_plot_dir = os.path.join(run_dir, "plots", "phase4")
        os.makedirs(phase4_plot_dir, exist_ok=True)
        plot_main_comparison(rows, os.path.join(phase4_plot_dir, "main_comparison.png"))
        plot_flip_rate(rows, os.path.join(phase4_plot_dir, "flip_rate.png"))
        plot_alpha_sweep_test(
            sweep_truth_rows,
            sweep_random_rows,
            baseline_acc=results_by_key["ccot"]["accuracy"],
            alpha_star=alpha_value,
            save_path=os.path.join(phase4_plot_dir, "alpha_sweep_test.png"),
        )
        plot_efficiency(rows, os.path.join(phase4_plot_dir, "accuracy_vs_tokens.png"))
        plot_alignment_coherence(rows, os.path.join(phase4_plot_dir, "alignment_vs_coherence.png"))
        flip_examples = []
        random_by_qid = {d.get("qid"): d for d in random_details}
        truth_by_qid = {d.get("qid"): d for d in truth_details}
        for base in ccot_details:
            qid = base.get("qid")
            r = random_by_qid.get(qid)
            t = truth_by_qid.get(qid)
            if not r or not t:
                continue
            if (not base["correct"]) and t["correct"]:
                flip_examples.append({
                    "qid": qid,
                    "question": base["question"],
                    "gold": base["gold"],
                    "base_pred": base["pred"],
                    "random_pred": r["pred"],
                    "truth_pred": t["pred"],
                    "fixed_by_random": bool(r["correct"]),
                })
        flip_examples.sort(key=lambda x: (not x["fixed_by_random"], x["qid"] or ""))
        plot_flip_examples(flip_examples, os.path.join(phase4_plot_dir, "flip_examples.png"))

    return rows, results_by_key, sweep_truth_rows, sweep_random_rows
