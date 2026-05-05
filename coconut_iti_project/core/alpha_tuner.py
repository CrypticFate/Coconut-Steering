import math
import os
import random
import time
from contextlib import nullcontext

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from utils.helpers import answers_match, clear_memory, extract_final_answer, save_json_log


def _latent_prompt(sample, n_latents):
    return (
        sample["question"]
        + "\n<|start-latent|>"
        + "<|latent|>" * n_latents
        + "<|end-latent|>"
    )


def _build_teacher_forced_batch(sample, tokenizer, config):
    prompt = _latent_prompt(sample, config.max_latent_tokens)
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
    answer_ids = tokenizer.encode("#### " + sample["answer"], add_special_tokens=False)
    answer_ids = answer_ids + [tokenizer.eos_token_id]
    input_ids = torch.tensor([prompt_ids + answer_ids], device=config.device)
    labels = torch.tensor([[-100] * len(prompt_ids) + answer_ids], device=config.device)
    attention_mask = torch.ones_like(input_ids)
    return input_ids, attention_mask, labels


def _autocast_context(config):
    if str(config.device).startswith("cuda") and torch.cuda.is_available() and config.bf16:
        return torch.amp.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


def _steering_regularizers(coconut_model, truth_vector):
    stats = coconut_model.last_steering_stats
    if not stats:
        zero = truth_vector.new_tensor(0.0)
        return zero, zero

    h_before = torch.cat([entry["h_before"] for entry in stats], dim=0)
    h_after = torch.cat([entry["h_after"] for entry in stats], dim=0)
    intervention = torch.cat([entry["intervention"] for entry in stats], dim=0)
    direction = F.normalize(truth_vector.to(h_after.device, h_after.dtype), p=2, dim=-1)
    if direction.dim() == 2:
        direction = direction[0]
    direction = direction.unsqueeze(0).expand_as(h_after)

    align_loss = -F.cosine_similarity(h_after.float(), direction.float(), dim=-1).mean()
    mag_loss = (
        intervention.float().norm(dim=-1)
        / h_before.float().norm(dim=-1).clamp_min(1e-8)
    ).pow(2).mean()
    return align_loss, mag_loss


def _loss_for_sample(coconut_model, sample, tokenizer, config, truth_vector, alpha,
                     include_regularizers=True):
    input_ids, attention_mask, labels = _build_teacher_forced_batch(sample, tokenizer, config)
    with _autocast_context(config):
        outputs = coconut_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            steering_vector=truth_vector,
            alpha=alpha,
            gamma=config.alpha_decay,
            collect_steering_stats=True,
            detach_latents=False,
        )
        answer_loss = outputs.loss.float()
        align_loss, mag_loss = _steering_regularizers(coconut_model, truth_vector)
        if include_regularizers:
            loss = answer_loss + config.lambda_align * align_loss + config.lambda_mag * mag_loss
        else:
            loss = answer_loss

    return loss, {
        "answer_loss": float(answer_loss.detach().cpu()),
        "align_loss": float(align_loss.detach().cpu()),
        "mag_loss": float(mag_loss.detach().cpu()),
        "alpha": float(alpha.detach().cpu()) if torch.is_tensor(alpha) else float(alpha),
    }


def _evaluate_loss(coconut_model, dataset, tokenizer, config, truth_vector, alpha):
    if not dataset:
        return float("inf")
    losses = []
    for sample in dataset:
        with torch.no_grad():
            loss, _ = _loss_for_sample(
                coconut_model,
                sample,
                tokenizer,
                config,
                truth_vector,
                alpha,
                include_regularizers=True,
            )
        losses.append(float(loss.detach().cpu()))
    return sum(losses) / len(losses)


def _gradient_check(coconut_model, sample, tokenizer, config, truth_vector, alpha_value):
    eps = config.gradient_check_epsilon
    alpha = torch.tensor(float(alpha_value), device=config.device, dtype=torch.float32, requires_grad=True)
    loss, _ = _loss_for_sample(
        coconut_model,
        sample,
        tokenizer,
        config,
        truth_vector,
        alpha,
        include_regularizers=False,
    )
    loss.backward()
    analytic = float(alpha.grad.detach().cpu())

    with torch.no_grad():
        alpha_plus = torch.tensor(float(alpha_value + eps), device=config.device, dtype=torch.float32)
        alpha_minus = torch.tensor(float(max(alpha_value - eps, 0.0)), device=config.device, dtype=torch.float32)
        loss_plus, _ = _loss_for_sample(
            coconut_model, sample, tokenizer, config, truth_vector, alpha_plus,
            include_regularizers=False,
        )
        loss_minus, _ = _loss_for_sample(
            coconut_model, sample, tokenizer, config, truth_vector, alpha_minus,
            include_regularizers=False,
        )

    denom_eps = float(alpha_plus.detach().cpu() - alpha_minus.detach().cpu())
    finite_diff = float((loss_plus - loss_minus).detach().cpu()) / max(denom_eps, 1e-12)
    rel_error = abs(analytic - finite_diff) / max(abs(analytic), abs(finite_diff), 1e-8)
    return {
        "alpha": float(alpha_value),
        "epsilon": eps,
        "analytic_grad": analytic,
        "finite_difference_grad": finite_diff,
        "relative_error": rel_error,
        "passed": rel_error < config.gradient_check_max_rel_error,
    }


def _split_val(data_val, tune_fraction, seed):
    data = list(data_val)
    random.Random(seed).shuffle(data)
    if len(data) <= 1:
        return data, data
    n_tune = max(1, int(len(data) * tune_fraction))
    n_tune = min(n_tune, len(data) - 1)
    return data[:n_tune], data[n_tune:]


def _alpha_from_theta(theta, alpha_max):
    return alpha_max * torch.sigmoid(theta)


def _initial_theta(alpha_initial, alpha_max):
    prob = min(max(alpha_initial / alpha_max, 1e-6), 1.0 - 1e-6)
    return math.log(prob / (1.0 - prob))


def run_alpha_diagnostic_sweep(coconut_model, data_val, tokenizer, config, truth_vector):
    results = []
    n_latents = config.max_latent_tokens
    for alpha in config.alpha_sweep:
        correct = 0
        total = len(data_val)
        for sample in tqdm(data_val, desc=f"Val alpha sweep {alpha}"):
            prompt = _latent_prompt(sample, n_latents)
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(config.device)
            with torch.no_grad():
                gen_ids, _, _ = coconut_model.generate_with_latents(
                    input_ids,
                    max_new_tokens=config.max_new_tokens_ccot,
                    temperature=0.0,
                    steering_vector=truth_vector if alpha > 0 else None,
                    alpha=alpha,
                    gamma=config.alpha_decay,
                )
            predicted = extract_final_answer(tokenizer.decode(gen_ids[0], skip_special_tokens=True))
            if answers_match(predicted, sample["answer"]):
                correct += 1
        results.append({
            "alpha": alpha,
            "accuracy": correct / total if total else 0.0,
        })
    return results


def tune_alpha(coconut_model, data_val, tokenizer, config, truth_vector, run_dir=None):
    print("Starting Phase 3: Gradient-based alpha tuning...")
    phase_start = time.time()

    ckpt_dir = os.path.join(run_dir, "checkpoints") if run_dir else config.save_path
    os.makedirs(ckpt_dir, exist_ok=True)

    if not data_val:
        alpha_star = torch.tensor(config.alpha_initial, device=config.device)
        torch.save(alpha_star, os.path.join(ckpt_dir, "alpha_star.pt"))
        print("WARNING: Empty validation set; using configured alpha_initial.")
        return alpha_star, {"warning": "empty_validation_set"}

    previous_trainable = [param.requires_grad for param in coconut_model.parameters()]
    for param in coconut_model.parameters():
        param.requires_grad = False
    coconut_model.eval()
    truth_vector = F.normalize(truth_vector.to(config.device), p=2, dim=-1)

    tune_data, early_stop_data = _split_val(
        data_val,
        tune_fraction=config.alpha_tune_fraction,
        seed=config.seed,
    )

    theta = torch.nn.Parameter(torch.tensor(
        _initial_theta(config.alpha_initial, config.alpha_max),
        device=config.device,
        dtype=torch.float32,
    ))
    optimizer = torch.optim.AdamW([theta], lr=config.alpha_lr)

    gradient_check = _gradient_check(
        coconut_model,
        tune_data[0],
        tokenizer,
        config,
        truth_vector,
        alpha_value=config.alpha_initial,
    )
    print(
        "Finite-difference alpha gradient check: "
        f"rel_error={gradient_check['relative_error']:.6f} "
        f"passed={gradient_check['passed']}"
    )
    if getattr(config, "enforce_gradient_check", True) and not gradient_check["passed"]:
        raise RuntimeError(
            "Finite-difference alpha gradient check failed: "
            f"relative_error={gradient_check['relative_error']:.6f}"
        )

    best_es_loss = float("inf")
    best_theta = theta.detach().clone()
    stale_epochs = 0
    training_log = []

    for epoch in range(config.alpha_max_epochs):
        epoch_losses = []
        for sample in tqdm(tune_data, desc=f"Alpha tuning epoch {epoch + 1}"):
            optimizer.zero_grad()
            alpha = _alpha_from_theta(theta, config.alpha_max)
            loss, components = _loss_for_sample(
                coconut_model,
                sample,
                tokenizer,
                config,
                truth_vector,
                alpha,
                include_regularizers=True,
            )
            loss.backward()
            optimizer.step()

            loss_value = float(loss.detach().cpu())
            components.update({"epoch": epoch, "loss": loss_value})
            training_log.append(components)
            epoch_losses.append(loss_value)

        with torch.no_grad():
            alpha_now = _alpha_from_theta(theta, config.alpha_max).detach()
            es_loss = _evaluate_loss(
                coconut_model,
                early_stop_data,
                tokenizer,
                config,
                truth_vector,
                alpha_now,
            )

        mean_epoch_loss = sum(epoch_losses) / max(len(epoch_losses), 1)
        print(
            f"Epoch {epoch + 1}: tune_loss={mean_epoch_loss:.6f} "
            f"early_stop_loss={es_loss:.6f} alpha={float(alpha_now.cpu()):.4f}"
        )

        if es_loss < best_es_loss:
            best_es_loss = es_loss
            best_theta = theta.detach().clone()
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= config.alpha_patience:
                print("Alpha tuning early-stopped.")
                break

    with torch.no_grad():
        alpha_star = _alpha_from_theta(best_theta, config.alpha_max).detach()

    alpha_path = os.path.join(ckpt_dir, "alpha_star.pt")
    torch.save(alpha_star, alpha_path)
    print(f"Learned alpha*: {float(alpha_star.cpu()):.4f}")
    print(f"[LOG] alpha* saved to {alpha_path}")

    diagnostic_sweep = []
    if getattr(config, "run_alpha_diagnostic_sweep", True):
        diagnostic_sweep = run_alpha_diagnostic_sweep(
            coconut_model,
            data_val,
            tokenizer,
            config,
            truth_vector,
        )

    metadata = {
        "alpha_star": float(alpha_star.cpu()),
        "alpha_path": alpha_path,
        "alpha_max": config.alpha_max,
        "alpha_initial": config.alpha_initial,
        "lambda_align": config.lambda_align,
        "lambda_mag": config.lambda_mag,
        "tune_examples": len(tune_data),
        "early_stop_examples": len(early_stop_data),
        "best_early_stop_loss": best_es_loss,
        "gradient_check": gradient_check,
        "training_log": training_log,
        "diagnostic_sweep": diagnostic_sweep,
        "duration_s": time.time() - phase_start,
    }

    if run_dir:
        save_json_log(run_dir, "alpha_tuning.json", metadata)

    for param, requires_grad in zip(coconut_model.parameters(), previous_trainable):
        param.requires_grad = requires_grad

    clear_memory()
    return alpha_star.to(config.device), metadata
