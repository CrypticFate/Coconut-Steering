
# Research Protocol: Steering Continuous Reasoning via Latent Intervention

## Abstract & Hypothesis
**Objective**: To investigate whether reasoning in a continuous latent space (as seen in Coconut/CCoT architectures) can be "steered" toward factuality using a linear intervention vector derived from the model's own activations.

**Hypothesis**: Truthfulness and logical validity are encoded as linear directions in the high-dimensional latent space of the model.
Injecting this "Truth Vector" ($\mathbf{v}_{truth}$) during the continuous reasoning phase will reduce "reasoning drift" (hallucination) and improve final answer accuracy.

## Dataset Preparation (GSM8K)
Strict data isolation is required to prevent data leakage. The GSM8K dataset will be split into two distinct subsets: Training and Testing. The training set is further split into three subsets:

### The Splits
| Subset Name | Size | Purpose |
| --- | --- | --- |
| **(Base Training)** | 6,000 | Train the base CCoT model to reason.
| **(Vector Extraction)** | 500 | Compute the  vector.
| **(Further Training)** | 1,500 | Final testing of the hypothesis.

Critical Rule: The Final Set ($\mathcal{D}_{final train}$) must never be used to calculate the Truth Vector. Doing so invalidates the results.

## Model Architecture
We utilize a Continuous Chain-of-Thought (CCoT) architecture (e.g., Coconut on a modified Llama-3).
- **Input**: Text tokens $x$.
- **Reasoning Phase**: A sequence of $k$ continuous hidden states $h_1, h_2, \dots, h_k$ where $h_t \in \mathbb{R}^d$. These are not decoded into text during training.
- **Output**: Final answer tokens $y$.
- **Mechanism**: The model operates in "Latent Mode," feeding the hidden state $h_t$ back into the input for step $t+1$ without passing through a discrete softmax layer.

## Phase 1: Base Model Training (Coconut paper)
- Goal: Create a model capable of continuous reasoning, even if imperfect.
- Format: Convert GSM8K examples into the format: Question -> [RAT_START] -> (Reasoning Steps) -> [RAT_END] -> Answer.
- Curriculum Learning:
    - Stage A: Train on standard text CoT (discrete tokens) to ground the reasoning.
    - Stage B: Gradually replace discrete reasoning tokens with continuous vectors (as per Coconut paper).
- Loss Function: Standard Next-Token Prediction (Cross-Entropy) on the final answer only. The latent steps are optimized end-to-end via backpropagation.
- Stopping Criterion: Stop when accuracy on a held-out validation set plateaus. Save Checkpoint.

## Phase 2: Extracting the Truth Vector ($\mathbf{v}_{truth}$)
- Goal: Identify the direction of "correct reasoning" in the latent space using $\mathcal{D}_{steer}$.
- Data Collection:
    - For each question $x_i$ in $\mathcal{D}_{steer}$: Run the frozen model $N$ times with temperature $T=1.0$.
    - Classify the traces:
        - Positive Set ($H^+$): Latent trajectories where the final answer $y$ matches the ground truth.
        - Negative Set ($H^-$): Latent trajectories where the final answer $y$ is incorrect.
    - Store the continuous vectors $h_{i,t}$ for each step $t$.
- Vector Calculation (Difference-of-Means): Compute the "Truth Direction" by subtracting the mean "Wrong" state from the mean "Right" state:
$$\mathbf{v}_{truth} = \frac{1}{|H^+|} \sum_{h \in H^+} h - \frac{1}{|H^-|} \sum_{h \in H^-} h$$

Note: Perform this calculation separately for each reasoning depth $t$ (yielding $\mathbf{v}_{truth}^t$) or average across all steps for a single global vector.

## Phase 3: Inference-Time Intervention (The Experiment)
Goal: Test if injecting $\mathbf{v}_{truth}$ improves performance on $\mathcal{D}_{eval}$.

### The Steering Equation
During the continuous reasoning loop on the Evaluation Set:
$$h_{t+1} = \text{Model}(h_t) + \alpha \cdot \sigma_{l} \cdot \frac{\mathbf{v}_{truth}}{|\mathbf{v}_{truth}|}$$

$\alpha$ (Steering Strength): A scalar hyperparameter controlling the intervention intensity. $\sigma_l$ is the standard deviation of the activations at layer $l$ of the model.

### Hyperparameter Sweep
Run the evaluation loop for different values of $\alpha$:
- $\alpha \in \{0.0, 0.1, 0.5, 1.0, 2.0, 5.0\}$
- $\alpha = 0.0$ is the Baseline (Control Group).

## Phase 4: Evaluation Metrics
We assess performance using three distinct lenses.
- Accuracy (Performance)
- Flip Rate: % of CCoT failures that are corrected after steering.
- Trajectory Faithfulness (Geometric): For every step $t$, calculate cosine similarity between $h_t$ and $\mathbf{v}_{truth}$.
- Interpretability Check (Optional): Train a linear probe to project $h_t$ back to the vocabulary. Decode the steered vs. unsteered vectors into text. Use GPT-4 to judge if the steered text represents more coherent logic.

