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
