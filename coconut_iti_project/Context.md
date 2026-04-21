This is the perfect time to zoom out. Now that you are migrating this massive pipeline to your local RTX 3090, having a crystal-clear mental model of the entire thesis is essential.

Your project is a cutting-edge hybrid. It combines **COCONUT (Chain of Continuous Thought)**, which changes *how* a model thinks, with **Inference-Time Intervention (ITI)**, which changes *what* a model thinks.

Here is the comprehensive breakdown of the entire COCONUT Steering Thesis, explaining the theory of each phase and exactly what your code cells are doing to execute it.

---

### Phase 1: The "Silent Thinking" Phase (Base Training)

**The Theory:** Standard Large Language Models (LLMs) are restricted to reasoning in the "language space," generating text token-by-token (Chain-of-Thought). This is inefficient because language is optimized for communication, not complex mathematical planning. Phase 1 teaches the model to bypass the English language. Instead of decoding hidden states into words, it feeds the last hidden state directly back into the network as a "continuous thought" vector. To teach the model this new trick, we use a multi-stage curriculum that gradually deletes English reasoning steps and replaces them with silent `<|latent|>` tokens.

**How the Cells Work:**

* **Cell 1 (Environment):** Purges memory and installs specific versions of PyTorch and `bitsandbytes` to ensure the massive 1.5B parameter Qwen model can train efficiently.
* **Cell 2 (Configuration):** Sets the global rules. It establishes the 60% / 10% / 30% data split, the learning rate (2e-5 for stability), the effective batch size (128), and the exact 18-epoch paper curriculum.
* **Cell 3 (Dataset Prep):** Loads the GSM8k math dataset and rigidly isolates 60% of it specifically for this training phase, ensuring no data leaks into the extraction or testing phases.
* **Cell 5 (Training Loop):** The heavy lifter. It loops through the dataset using a custom scheduler. In early epochs, it leaves most English text intact. By the final epochs, it triggers the "Drop Remaining Text" rule, forcing the model to solve the math entirely inside the `<|latent|>` tokens without outputting any intermediate text. It uses Paged AdamW and Gradient Clipping to prevent Catastrophic Forgetting.



---

### Phase 2: The "Mind Reading" Phase (Vector Extraction)

**The Theory:**
Now that the model knows how to think in silent math vectors, we need to map the geometry of its mind. We want to find the exact mathematical direction of "Truth" or "Logical Correctness." We run the model on fresh problems, collect the continuous thoughts where it got the math *right*, and subtract the thoughts where it got the math *wrong*. The resulting average difference is a 2048-dimensional arrow pointing directly toward correct logic: the Global Truth Vector.

**How the Cells Work:**

* **Cell 6 (Global Extraction):** Takes the isolated 10% of your dataset. It feeds the questions to the newly trained model from Phase 1 and lets it generate answers deterministically (Temperature = 0.0). It separates the `<|latent|>` representations into `correct_latents` and `wrong_latents`. It then calculates the mean of both groups and subtracts them to generate the `truth_vector`.
* **Cell 6.2 (PCA Visualization):** A critical diagnostic tool. It takes those 2048-dimensional thoughts and uses Principal Component Analysis (PCA) to squash them down to 2D. It plots them as blue dots (correct) and red dots (wrong) to visually prove if the model is actually organizing concepts logically in its latent space.

---

### Phase 3: The "Mind Control" Phase (Intervention Setup)

**The Theory:**
This is where ITI comes in. We have the Truth Vector, and we want to actively inject it into the model's brain while it is trying to solve a brand-new, unseen problem. However, if we push the model too hard, it forgets how to speak English. We use a formula with an Alpha multiplier (how hard we push) and a Gamma decay (letting go near the end of the thought) to hit the perfect balance.

**How the Cells Work:**

* **Cell 4 (The COCONUT Wrapper):** This is the custom neural network architecture block. It overrides the standard `generate` function. During the forward pass, specifically when the model is processing a `<|latent|>` token, this cell intercepts the hidden state ($h_{old}$) and applies the steering math: $h_{new} = h_{old} + (\alpha \times \mathbf{v}_{truth})$. It actively hijacks the thought process.

---

### Phase 4: The "Final Exam" Phase (Evaluation)

**The Theory:**
We must scientifically prove that injecting the Truth Vector actually causes the model to become smarter, rather than just scrambling its outputs. We run the official test set through the model multiple times, sweeping through different intervention strengths ($\alpha$) from 0.0 (no intervention) up to 50.0 (aggressive intervention), measuring if wrong answers flip to right answers.

**How the Cells Work:**

* **Cell 7 (Evaluation Loop):** Takes the official, untouched GSM8k test set. It runs a loop for each $\alpha$ value in `[0.0, 5.0, 10.0, 20.0, 50.0]`. It measures three things:
1. **Accuracy:** Total correct answers.
2. **Flip Rate:** How many answers were wrong at $\alpha=0.0$ but became correct when pushed by the Truth Vector.
3. **Faithfulness:** A cosine similarity check to ensure the model's thoughts actually moved in the direction of the steering vector.


* **Cell 8 (Logit Confidence Deep Dive):** A microscopic view. It takes a single question and checks the exact probability percentage of the *correct target number* token. This proves if the intervention is mathematically working under the hood, even if the final English output string didn't fully flip.

