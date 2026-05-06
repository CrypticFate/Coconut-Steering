import math
import os
import random
import time
from contextlib import nullcontext

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from utils.helpers import answers_match, clear_memory, extract_final_answer, save_json_log
from utils.visualizer import (
    plot_alpha_convergence,
    plot_alpha_sweep_val,
    plot_gradient_check,
    plot_loss_components,
    plot_val_accuracy_tuning,
)


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


def _tokenize_for_alpha_tuning(sample, tokenizer, device):
    """Question through ``<|start-latent|>`` only; answer ids separate (matches steered layout)."""
    q_text = sample["question"] + "\n<|start-latent|>"
    question_ids = tokenizer.encode(q_text, add_special_tokens=True, return_tensors="pt").to(device)
    ans = tokenizer.encode("#### " + sample["answer"], add_special_tokens=False)
    ans = ans + [tokenizer.eos_token_id]
    answer_ids = torch.tensor([ans], device=device, dtype=torch.long)
    return question_ids, answer_ids


def _tuning_regularizers(h_steered_seq, truth_vector, alpha_scalar):
    """Magnitude penalty at current α (align regularizer intentionally disabled)."""
    h_stack = torch.stack(h_steered_seq, dim=0)
    direction = F.normalize(truth_vector.float(), p=2, dim=-1)
    if direction.dim() == 2:
        direction = direction.squeeze(0)
    align = h_stack.new_tensor(0.0)
    sig = h_stack.float().std(dim=-1, keepdim=True)
    a = alpha_scalar.float()
    perturb = a * sig * direction
    mag = (
        perturb.norm(dim=-1) / h_stack.float().norm(dim=-1).clamp_min(1e-8)
    ).pow(2).mean()
    return align, mag


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
                     include_regularizers=True, use_kv_cache=True):
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
            use_kv_cache=use_kv_cache,
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


def _evaluate_steered_accuracy(coconut_model, dataset, tokenizer, config, truth_vector, alpha):
    """Greedy CCoT accuracy on dataset at scalar/tensor alpha (no gradients)."""
    if not dataset:
        return 0.0
    n_latents = config.max_latent_tokens
    alpha_f = float(alpha.detach().cpu()) if torch.is_tensor(alpha) else float(alpha)
    correct = 0
    for sample in dataset:
        prompt = _latent_prompt(sample, n_latents)
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(config.device)
        with torch.no_grad():
            gen_ids, _, _ = coconut_model.generate_with_latents(
                input_ids,
                max_new_tokens=config.max_new_tokens_ccot,
                temperature=0.0,
                steering_vector=truth_vector if alpha_f > 0 else None,
                alpha=alpha,
                gamma=config.alpha_decay,
            )
        predicted = extract_final_answer(tokenizer.decode(gen_ids[0], skip_special_tokens=True))
        if answers_match(predicted, sample["answer"]):
            correct += 1
    return correct / len(dataset)


def _evaluate_loss(coconut_model, dataset, tokenizer, config, truth_vector, alpha):
    """Mean tuning objective (answer CE + regularizers) at fixed α, using ``steered_forward_for_tuning``."""
    if not dataset:
        return float("inf")
    alpha_t = torch.tensor(
        float(alpha.detach().cpu()) if torch.is_tensor(alpha) else float(alpha),
        device=config.device,
        dtype=torch.float32,
    )
    losses = []
    for sample in dataset:
        with torch.no_grad():
            with _autocast_context(config):
                q_ids, a_ids = _tokenize_for_alpha_tuning(sample, tokenizer, config.device)
                L_ans, h_seq = coconut_model.steered_forward_for_tuning(
                    q_ids,
                    a_ids,
                    truth_vector,
                    config.max_latent_tokens,
                    gamma=config.alpha_decay,
                    alpha_tensor=alpha_t,
                )
                _, mag = _tuning_regularizers(h_seq, truth_vector, alpha_t)
                loss = L_ans.float() + config.lambda_mag * mag
        losses.append(float(loss.detach().cpu()))
    return sum(losses) / len(losses)


def _gradient_check_steered(coconut_model, sample, tokenizer, config, truth_vector):
    """
    Finite-difference check on θ where α = α_max · σ(θ), using only L_ans from
    ``steered_forward_for_tuning`` (symmetric difference).
    """
    eps = config.gradient_check_epsilon
    theta = torch.nn.Parameter(
        torch.tensor(
            _initial_theta(config.alpha_initial, config.alpha_max),
            device=config.device,
            dtype=torch.float32,
        )
    )

    def forward_L_ans():
        with _autocast_context(config):
            q_ids, a_ids = _tokenize_for_alpha_tuning(sample, tokenizer, config.device)

            def alpha_fn():
                return _alpha_from_theta(theta, config.alpha_max)

            L_ans, _ = coconut_model.steered_forward_for_tuning(
                q_ids,
                a_ids,
                truth_vector,
                config.max_latent_tokens,
                gamma=config.alpha_decay,
                alpha_fn=alpha_fn,
            )
        return L_ans

    theta.grad = None
    L = forward_L_ans()
    L.backward()
    analytic = float(theta.grad.detach().cpu())

    with torch.no_grad():
        orig = theta.data.clone()
        theta.data.copy_(orig + eps)
        L_p = forward_L_ans()
        theta.data.copy_(orig - eps)
        L_m = forward_L_ans()
        theta.data.copy_(orig)

    grad_fd = float((L_p - L_m).detach().cpu()) / (2.0 * eps)
    rel_error = abs(analytic - grad_fd) / max(abs(analytic), abs(grad_fd), 1e-8)
    passed = rel_error < config.gradient_check_max_rel_error
    return {
        "theta_initial": float(orig.detach().cpu()),
        "epsilon": eps,
        "analytic_grad": analytic,
        "finite_difference_grad": grad_fd,
        "relative_error": rel_error,
        "passed": passed,
    }


def _split_val(data_val, tune_fraction, seed):
    data = list(data_val)
    random.Random(seed).shuffle(data)
    if len(data) <= 1:
        return data, data
    n_tune = max(1, int(round(len(data) * tune_fraction)))
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
    truth_vector = F.normalize(
        truth_vector.to(config.device), p=2, dim=-1
    ).to(dtype=torch.float32)

    tune_data, early_stop_data = _split_val(
        data_val,
        tune_fraction=config.alpha_tune_fraction,
        seed=config.seed,
    )

    theta = torch.nn.Parameter(
        torch.tensor(
            _initial_theta(config.alpha_initial, config.alpha_max),
            device=config.device,
            dtype=torch.float32,
        )
    )
    assert theta.dtype == torch.float32, "theta_alpha must be float32 for stable α gradients"
    optimizer = torch.optim.AdamW([theta], lr=config.alpha_lr, weight_decay=0.0)

    gradient_check = _gradient_check_steered(
        coconut_model,
        tune_data[0],
        tokenizer,
        config,
        truth_vector,
    )
    print(
        "Gradient check (θ → α = α_max·σ(θ), L_ans only): "
        f"analytic={gradient_check['analytic_grad']:.4e} "
        f"numeric={gradient_check['finite_difference_grad']:.4e} "
        f"rel_err={gradient_check['relative_error']:.4f} "
        f"passed={gradient_check['passed']}"
    )
    if not gradient_check["passed"]:
        msg = (
            f"Gradient check FAILED (rel_error={gradient_check['relative_error']:.4f} "
            f"> {config.gradient_check_max_rel_error}). "
            "Alpha tuning would be noise. Fix gradient flow first. "
            f"theta.dtype={theta.dtype}"
        )
        if getattr(config, "enforce_gradient_check", True):
            raise AssertionError(msg)
        print(f"WARNING: {msg} Continuing because enforce_gradient_check=False.")
    else:
        print("Gradient check passed — proceeding to alpha tuning.")

    best_es_accuracy = -1.0
    best_es_loss_at_acc = float("inf")
    best_theta = theta.detach().clone()
    stale_epochs = 0
    training_log = []
    val_es_acc_history = []
    alpha_epoch_history = []

    for epoch in range(config.alpha_max_epochs):
        epoch_losses = []
        for sample in tqdm(tune_data, desc=f"Alpha tuning epoch {epoch + 1}"):
            optimizer.zero_grad()
            q_ids, a_ids = _tokenize_for_alpha_tuning(sample, tokenizer, config.device)

            def alpha_fn():
                return _alpha_from_theta(theta, config.alpha_max)

            with _autocast_context(config):
                L_ans, h_seq = coconut_model.steered_forward_for_tuning(
                    q_ids,
                    a_ids,
                    truth_vector,
                    config.max_latent_tokens,
                    gamma=config.alpha_decay,
                    alpha_fn=alpha_fn,
                )
                alpha_now_step = _alpha_from_theta(theta, config.alpha_max)
                _, mag = _tuning_regularizers(h_seq, truth_vector, alpha_now_step)
                loss = (
                    L_ans.float()
                    + config.lambda_mag * mag
                )

            loss.backward()
            torch.nn.utils.clip_grad_norm_([theta], max_norm=1.0)
            optimizer.step()

            loss_value = float(loss.detach().cpu())
            components = {
                "epoch": epoch,
                "loss": loss_value,
                "answer_loss": float(L_ans.detach().cpu()),
                "align_loss": 0.0,
                "mag_loss": float(mag.detach().cpu()),
                "alpha": float(alpha_now_step.detach().cpu()),
            }
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
            es_accuracy = _evaluate_steered_accuracy(
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
            f"D_val_es_acc={es_accuracy:.2%} early_stop_loss={es_loss:.6f} "
            f"alpha={float(alpha_now.cpu()):.4f}"
        )
        val_es_acc_history.append(float(es_accuracy))
        alpha_epoch_history.append(float(alpha_now.cpu()))

        if es_accuracy > best_es_accuracy:
            best_es_accuracy = es_accuracy
            best_es_loss_at_acc = es_loss
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
        print("Running validation-only alpha diagnostic sweep on D_val (analysis only).")
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
        "best_early_stop_accuracy": best_es_accuracy,
        "best_early_stop_loss_at_best_acc": best_es_loss_at_acc,
        "gradient_check": gradient_check,
        "training_log": training_log,
        "diagnostic_sweep_split": "validation",
        "diagnostic_sweep": diagnostic_sweep,
        "val_es_acc_history": val_es_acc_history,
        "alpha_epoch_history": alpha_epoch_history,
        "duration_s": time.time() - phase_start,
    }

    if run_dir:
        save_json_log(run_dir, "alpha_tuning.json", metadata)
        phase3_plot_dir = os.path.join(run_dir, "plots", "phase3")
        os.makedirs(phase3_plot_dir, exist_ok=True)
        alpha_hist = [entry["alpha"] for entry in training_log]
        ans_hist = [entry["answer_loss"] for entry in training_log]
        mag_hist = [entry["mag_loss"] for entry in training_log]
        total_hist = [entry["loss"] for entry in training_log]
        plot_alpha_convergence(
            alpha_hist,
            float(alpha_star.detach().cpu()),
            os.path.join(phase3_plot_dir, "alpha_convergence.png"),
        )
        plot_loss_components(
            ans_hist, mag_hist, total_hist,
            os.path.join(phase3_plot_dir, "loss_components.png"),
        )
        plot_val_accuracy_tuning(
            val_es_acc_history,
            os.path.join(phase3_plot_dir, "val_es_accuracy.png"),
        )
        if diagnostic_sweep:
            plot_alpha_sweep_val(
                [x["alpha"] for x in diagnostic_sweep],
                [x["accuracy"] for x in diagnostic_sweep],
                float(alpha_star.detach().cpu()),
                os.path.join(phase3_plot_dir, "alpha_sweep_val.png"),
            )
        plot_gradient_check(
            [gradient_check["analytic_grad"]],
            [gradient_check["finite_difference_grad"]],
            gradient_check["relative_error"],
            os.path.join(phase3_plot_dir, "gradient_check.png"),
        )

    for param, requires_grad in zip(coconut_model.parameters(), previous_trainable):
        param.requires_grad = requires_grad

    clear_memory()
    return alpha_star.to(config.device), metadata
