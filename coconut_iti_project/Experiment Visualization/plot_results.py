import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

# Set output directory
out_dir = "/home/cryptic/Study/Thesis/Experiment/Final/coconut_iti_project/Experiment Visualization/plots"
os.makedirs(out_dir, exist_ok=True)

# Data
models = ['Qwen2.5-0.5B-Instruct', 'Qwen2.5-3B-Instruct', 'Llama-3.2-3B', 'microsoft/phi-2']
alphas = [0.0, 5.0, 10.0, 20.0, 50.0]

data = {
    'Qwen2.5-0.5B-Instruct': {
        'Accuracy': [14.50, 14.50, 15.00, 15.00, 15.50],
        'Flip Rate': [0.00, 0.00, 0.58, 0.58, 1.75],
        'Faithfulness': [0.0000, 0.2290, 0.3665, 0.5907, 0.8772]
    },
    'Qwen2.5-3B-Instruct': {
        'Accuracy': [12.51, 12.51, 12.51, 12.43, 12.59],
        'Flip Rate': [0.00, 0.00, 0.00, 0.00, 1.39],
        'Faithfulness': [0.0000, -0.1517, -0.0904, 0.0502, 0.4850]
    },
    'Llama-3.2-3B': {
        'Accuracy': [9.86, 9.55, 9.78, 10.77, 10.69],
        'Flip Rate': [0.00, 1.01, 2.02, 4.29, 4.54],
        'Faithfulness': [0.0000, 0.1882, 0.3537, 0.5975, 0.8730]
    },
    'microsoft/phi-2': {
        'Accuracy': [14.78, 14.94, 15.62, 15.54, 13.87],
        'Flip Rate': [0.00, 2.22, 3.65, 5.25, 8.72],
        'Faithfulness': [0.0000, 0.1436, 0.2448, 0.4304, 0.7800]
    }
}

markers = ['o', 's', '^', 'D']
colors = sns.color_palette("deep", len(models))

def setup_academic_style():
    plt.rcParams.update({
        'font.size': 14,
        'font.family': 'serif',
        'axes.labelsize': 16,
        'axes.titlesize': 18,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 12,
        'lines.linewidth': 2.5,
        'lines.markersize': 8,
        'figure.autolayout': True,
    })
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.5)

setup_academic_style()

metrics = ['Accuracy', 'Flip Rate', 'Faithfulness']
ylabels = {'Accuracy': 'Accuracy (%)', 'Flip Rate': 'Flip Rate (%)', 'Faithfulness': 'Faithfulness'}

# 1. Line Plots
for metric in metrics:
    plt.figure(figsize=(8, 6))
    for i, model in enumerate(models):
        plt.plot(alphas, data[model][metric], marker=markers[i], label=model, color=colors[i])
    
    plt.xlabel(r'Intervention Strength ($\alpha$)')
    plt.ylabel(ylabels[metric])
    plt.title(f'Model {metric} vs. Intervention Strength')
    plt.legend(title="Models", loc='best')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'line_{metric.lower().replace(" ", "_")}.png'), dpi=300)
    plt.savefig(os.path.join(out_dir, f'line_{metric.lower().replace(" ", "_")}.pdf'), dpi=300)
    plt.close()

# 2. Grouped Bar Charts
x = np.arange(len(alphas))  # the label locations
width = 0.2  # the width of the bars

for metric in metrics:
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, model in enumerate(models):
        offset = (i - 1.5) * width
        ax.bar(x + offset, data[model][metric], width, label=model, color=colors[i], edgecolor='black', linewidth=0.5)

    ax.set_xlabel(r'Intervention Strength ($\alpha$)')
    ax.set_ylabel(ylabels[metric])
    ax.set_title(f'Comparison of {metric} across Models')
    ax.set_xticks(x)
    ax.set_xticklabels(alphas)
    ax.legend(title="Models", loc='best')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'bar_{metric.lower().replace(" ", "_")}.png'), dpi=300)
    plt.savefig(os.path.join(out_dir, f'bar_{metric.lower().replace(" ", "_")}.pdf'), dpi=300)
    plt.close()

# 3. Comprehensive Subplot (1x3)
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
for idx, metric in enumerate(metrics):
    ax = axes[idx]
    for i, model in enumerate(models):
        ax.plot(alphas, data[model][metric], marker=markers[i], label=model, color=colors[i])
    ax.set_xlabel(r'Intervention Strength ($\alpha$)')
    ax.set_ylabel(ylabels[metric])
    ax.set_title(metric)
    ax.grid(True, linestyle='--', alpha=0.7)
    if idx == 0:
        ax.legend(title="Models", loc='best')

plt.suptitle('Overall Intervention Effects across Models', fontsize=20, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'combined_line_plots.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(out_dir, 'combined_line_plots.pdf'), dpi=300, bbox_inches='tight')
plt.close()

print(f"All plots saved to {out_dir}")
