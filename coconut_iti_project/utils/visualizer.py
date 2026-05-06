import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import torch


def plot_loss_curve(loss_history, save_path):
    if not loss_history:
        print("No loss history to plot.")
        return
    _ensure_parent(save_path)

    plt.figure(figsize=(10, 5))

    window = max(1, len(loss_history) // 50)
    smoothed_loss = np.convolve(loss_history, np.ones(window) / window, mode="valid")

    plt.plot(smoothed_loss, color="blue", linewidth=2)
    plt.title("Phase 1: Continuous Thought Training Loss (Smoothed)")
    plt.xlabel("Training Steps")
    plt.ylabel("Cross Entropy Loss")
    plt.grid(True, linestyle="--", alpha=0.7)

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Loss curve saved to {save_path}")


def plot_latent_pca(correct_latents, wrong_latents, save_path):
    if len(correct_latents) < 3 or len(wrong_latents) < 3:
        print("Not enough latents generated to perform PCA.")
        return

    print("Analyzing Latent Space Geometry using PyTorch...")

    flat_correct = [vec.cpu().float().view(-1) for vec in correct_latents]
    flat_wrong = [vec.cpu().float().view(-1) for vec in wrong_latents]

    all_latents = torch.stack(flat_correct + flat_wrong)
    labels = [1] * len(flat_correct) + [0] * len(flat_wrong)

    all_latents_centered = all_latents - all_latents.mean(dim=0)
    U, S, V = torch.pca_lowrank(all_latents_centered, q=2)
    reduced_latents = torch.matmul(all_latents_centered, V[:, :2]).numpy()

    fig = go.Figure()

    wrong_mask = np.array(labels) == 0
    fig.add_trace(go.Scatter(
        x=reduced_latents[wrong_mask, 0],
        y=reduced_latents[wrong_mask, 1],
        mode="markers",
        name="Incorrect Paths",
        marker=dict(color="red", symbol="x", opacity=0.6, size=8),
    ))

    correct_mask = np.array(labels) == 1
    fig.add_trace(go.Scatter(
        x=reduced_latents[correct_mask, 0],
        y=reduced_latents[correct_mask, 1],
        mode="markers",
        name="Correct Paths",
        marker=dict(color="blue", symbol="circle", opacity=0.6, size=8),
    ))

    fig.update_layout(
        title="Latent Space PCA: Correct vs. Incorrect Thoughts",
        xaxis_title="Principal Component 1",
        yaxis_title="Principal Component 2",
        template="plotly_white",
        width=800,
        height=600,
    )

    fig.write_html(save_path)
    print(f"PCA plot saved to {save_path}")


def _ensure_parent(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)


def plot_stage_losses(losses_per_stage, stage_transition_epochs, save_path):
    _ensure_parent(save_path)
    fig, ax = plt.subplots(figsize=(10, 4))
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63", "#9C27B0"]
    cursor = 0
    for stage_idx, stage_losses in sorted(losses_per_stage.items()):
        if not stage_losses:
            continue
        xs = list(range(cursor, cursor + len(stage_losses)))
        ax.plot(xs, stage_losses, label=f"Stage {stage_idx}", color=colors[stage_idx % len(colors)])
        cursor += len(stage_losses)
    for t in stage_transition_epochs:
        ax.axvline(x=t, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Phase 1: Training Loss per Curriculum Stage")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_val_accuracy(val_accuracies, best_epoch, best_acc, save_path):
    _ensure_parent(save_path)
    if not val_accuracies:
        return
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(val_accuracies, color="#4CAF50", marker="o")
    ax.axvline(x=best_epoch, color="red", linestyle="--", label=f"Best epoch={best_epoch+1}, acc={best_acc:.2%}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation accuracy")
    ax.set_title("Phase 1: Validation Accuracy (D_train-val)")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_embedding_drift(drift_by_token, save_path):
    _ensure_parent(save_path)
    fig, ax = plt.subplots(figsize=(10, 4))
    for token_name, values in drift_by_token.items():
        if values:
            ax.plot(values, label=token_name)
    ax.axhline(y=0.99, color="red", linestyle="--", label="Danger zone")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cosine similarity to init")
    ax.set_title("Phase 1: Latent Token Embedding Drift")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_decoded_latents(decoded_examples, save_path):
    _ensure_parent(save_path)
    if not decoded_examples:
        return
    fig_h = max(4, 1.6 * len(decoded_examples))
    fig, ax = plt.subplots(figsize=(13, fig_h))
    ax.axis("off")
    lines = ["Phase 1: Decoded Latent Interpretations (top-3 tokens per latent step)", ""]
    for i, ex in enumerate(decoded_examples, start=1):
        lines.append(f"[Q{i}] {ex['question'][:110]}")
        for step in ex["steps"]:
            top_txt = ", ".join([f"{tok} ({p:.2%})" for tok, p in step["top3"]])
            lines.append(f"  latent t={step['t']}: {top_txt}")
        lines.append("")
    ax.text(0.0, 1.0, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=9)
    fig.tight_layout()
    fig.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_source_comparison(source_accs, save_path):
    _ensure_parent(save_path)
    labels = list(source_accs.keys())
    vals = [source_accs[k] if source_accs[k] is not None else 0.0 for k in labels]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(labels, vals, color=["#2196F3", "#4CAF50", "#FF9800"][: len(labels)])
    ax.axhline(y=0.5, color="red", linestyle="--", alpha=0.7, label="Chance")
    ax.set_ylabel("Linear probe accuracy")
    ax.set_title("Phase 2: Source A vs Source B")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_cpca_sweep(cpca_probe_sweep, dom_probe_acc, save_path):
    _ensure_parent(save_path)
    if not cpca_probe_sweep:
        return
    pts = [(x["beta"], x["probe_accuracy"]) for x in cpca_probe_sweep if x.get("probe_accuracy") is not None]
    if not pts:
        return
    pts.sort(key=lambda x: x[0])
    betas = [x[0] for x in pts]
    accs = [x[1] for x in pts]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(betas, accs, marker="o", color="#2196F3", label="cPCA")
    if dom_probe_acc is not None:
        ax.axhline(y=dom_probe_acc, color="black", linestyle="--", label="DoM baseline")
    ax.set_xlabel("β")
    ax.set_ylabel("Probe accuracy")
    ax.set_title("Phase 2: cPCA β Sweep")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_position_probe_acc(position_probe_acc, save_path):
    _ensure_parent(save_path)
    if not position_probe_acc:
        return
    xs = sorted(position_probe_acc.keys())
    ys = [position_probe_acc[x] for x in xs]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(xs, ys, color="#2196F3")
    ax.axhline(y=0.5, color="red", linestyle="--", label="Chance")
    ax.set_xlabel("Latent position t")
    ax.set_ylabel("Probe accuracy")
    ax.set_title("Phase 2: H+/H- Separation per Latent Position")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_vector_similarity(sim_matrix, save_path):
    _ensure_parent(save_path)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(sim_matrix, cmap="viridis", vmin=-1, vmax=1)
    ax.set_title("Phase 2: Cross-position Truth Vector Similarity")
    ax.set_xlabel("Position j")
    ax.set_ylabel("Position i")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_alpha_convergence(alpha_history, alpha_star, save_path):
    _ensure_parent(save_path)
    if not alpha_history:
        return
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(alpha_history, color="#2196F3", label="α during tuning")
    ax.axhline(y=alpha_star, color="red", linestyle="--", label=f"α*={alpha_star:.3f}")
    ax.set_xlabel("Training step")
    ax.set_ylabel("α")
    ax.set_title("Phase 3: Alpha Convergence")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_loss_components(ans_hist, mag_hist, total_hist, save_path):
    _ensure_parent(save_path)
    if not total_hist:
        return
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].plot(ans_hist, color="#F44336")
    axes[0].set_title("L_ans")
    axes[1].plot(mag_hist, color="#FF9800")
    axes[1].set_title("L_mag")
    axes[2].plot(total_hist, color="#2196F3")
    axes[2].set_title("Total")
    for ax in axes:
        ax.set_xlabel("Step")
        ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_val_accuracy_tuning(val_es_acc_history, save_path):
    _ensure_parent(save_path)
    if not val_es_acc_history:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(val_es_acc_history, marker="o", color="#4CAF50")
    best_idx = int(np.argmax(val_es_acc_history))
    ax.axvline(x=best_idx, color="red", linestyle="--", label=f"Best={val_es_acc_history[best_idx]:.3f}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("D_val_es accuracy")
    ax.set_title("Phase 3: Early-stop Accuracy")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_alpha_sweep_val(alpha_grid, val_accs, alpha_star, save_path):
    _ensure_parent(save_path)
    if not alpha_grid:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(alpha_grid, val_accs, marker="o", color="#2196F3", label="D_val grid")
    ax.axvline(x=alpha_star, color="red", linestyle="--", label=f"α*={alpha_star:.3f}")
    ax.set_xlabel("α")
    ax.set_ylabel("D_val accuracy")
    ax.set_title("Phase 3: Validation Accuracy vs α")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_gradient_check(analytic_grads, numeric_grads, rel_error, save_path):
    _ensure_parent(save_path)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(analytic_grads, numeric_grads, alpha=0.8, color="#2196F3")
    lim = max(abs(min(analytic_grads + numeric_grads)), abs(max(analytic_grads + numeric_grads)), 1e-6)
    ax.plot([-lim, lim], [-lim, lim], "r--")
    ax.set_xlabel("Analytic grad")
    ax.set_ylabel("Numeric grad")
    ax.set_title(f"Phase 3: Gradient Check (rel_error={rel_error:.4f})")
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_main_comparison(rows, save_path):
    _ensure_parent(save_path)
    labels = [r["condition"] for r in rows]
    accs = [100.0 * r["accuracy"] for r in rows]
    colors = ["#9E9E9E", "#607D8B", "#2196F3", "#FF9800", "#4CAF50"][: len(rows)]
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, accs, color=colors)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3, f"{acc:.1f}%", ha="center", va="bottom")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Phase 4: Final Comparison (Locked Test)")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_flip_rate(rows, save_path):
    _ensure_parent(save_path)
    subset = [r for r in rows if r.get("key") in {"random", "truth"}]
    if not subset:
        return
    labels = [r["condition"] for r in subset]
    vals = [100.0 * r["flip_rate"] for r in subset]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(labels, vals, color=["#FF9800", "#4CAF50"][: len(subset)])
    ax.set_ylabel("Flip Rate (%)")
    ax.set_title("Phase 4: Flip Rate")
    ax.tick_params(axis="x", rotation=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_alpha_sweep_test(truth_rows, random_rows, baseline_acc, alpha_star, save_path):
    _ensure_parent(save_path)
    if not truth_rows:
        return
    truth_rows = sorted(truth_rows, key=lambda x: x["alpha"])
    tx = [r["alpha"] for r in truth_rows]
    ty = [r["accuracy"] for r in truth_rows]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(tx, ty, marker="o", color="#4CAF50", label="Truth vector")
    if random_rows:
        random_rows = sorted(random_rows, key=lambda x: x["alpha"])
        rx = [r["alpha"] for r in random_rows]
        ry = [r["accuracy"] for r in random_rows]
        ax.plot(rx, ry, marker="s", linestyle="--", color="#FF9800", label="Random noise")
    ax.axhline(y=baseline_acc, color="#2196F3", linestyle=":", label="CCoT baseline")
    ax.axvline(x=alpha_star, color="red", linestyle="--", label=f"α*={alpha_star:.3f}")
    ax.set_xlabel("α")
    ax.set_ylabel("Test accuracy")
    ax.set_title("Phase 4: Test Accuracy vs α")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_efficiency(rows, save_path):
    _ensure_parent(save_path)
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#9E9E9E", "#607D8B", "#2196F3", "#FF9800", "#4CAF50"]
    for i, r in enumerate(rows):
        x = r["token_count"]
        y = 100.0 * r["accuracy"]
        ax.scatter(x, y, color=colors[i % len(colors)], s=140, zorder=3)
        ax.annotate(r["condition"], (x, y), textcoords="offset points", xytext=(5, 5), fontsize=8)
    ax.set_xlabel("Mean token count")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Phase 4: Accuracy vs Token Count")
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_alignment_coherence(rows, save_path):
    _ensure_parent(save_path)
    subset = [r for r in rows if r.get("key") in {"ccot", "random", "truth"}]
    if not subset:
        return
    labels = [r["condition"] for r in subset]
    align = [r.get("truth_alignment", 0.0) for r in subset]
    coh = [r.get("trajectory_coherence", 0.0) for r in subset]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.bar(labels, align, color=["#2196F3", "#FF9800", "#4CAF50"][: len(subset)])
    ax1.set_title("Truth Alignment")
    ax1.tick_params(axis="x", rotation=15)
    ax2.bar(labels, coh, color=["#2196F3", "#FF9800", "#4CAF50"][: len(subset)])
    ax2.set_title("Trajectory Coherence")
    ax2.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_flip_examples(flip_examples, save_path, top_k=10):
    _ensure_parent(save_path)
    if not flip_examples:
        return
    picks = flip_examples[:top_k]
    fig_h = max(4, 0.9 * len(picks) + 1.5)
    fig, ax = plt.subplots(figsize=(14, fig_h))
    ax.axis("off")
    lines = ["Phase 4: Per-example flip analysis (top truth-only/random-supported fixes)", ""]
    for i, ex in enumerate(picks, start=1):
        tag = "random+truth" if ex.get("fixed_by_random") else "truth-only"
        lines.append(
            f"{i:02d}. [{tag}] {ex['question'][:90]} | gt={ex['gold']} | base={ex['base_pred']} | random={ex['random_pred']} | truth={ex['truth_pred']}"
        )
    ax.text(0.0, 1.0, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_pipeline_summary(stage_labels, accuracies, save_path):
    _ensure_parent(save_path)
    fig, ax = plt.subplots(figsize=(9, 4))
    colors = ["#9E9E9E", "#607D8B", "#2196F3", "#FF9800", "#4CAF50"][: len(stage_labels)]
    bars = ax.bar(stage_labels, [100.0 * a for a in accuracies], color=colors)
    for b, a in zip(bars, accuracies):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.2, f"{100*a:.1f}%", ha="center", va="bottom")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("End-to-End Accuracy Progression")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
