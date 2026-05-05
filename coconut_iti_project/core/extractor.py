import json
import os
import time
from collections import defaultdict

import torch
from tqdm.auto import tqdm

from utils.helpers import (
    answers_match,
    clear_memory,
    extract_final_answer,
    save_extraction_log,
    save_json_log,
)


def _unit_normalize(vector, eps=1e-12):
    return vector / vector.norm(dim=-1, keepdim=True).clamp_min(eps)


def _compute_difference_of_means(h_pos, h_neg):
    raw = h_pos.mean(dim=0) - h_neg.mean(dim=0)
    return _unit_normalize(raw)


def _compute_cpca_subspace(h_pos, h_neg, k, beta):
    h_pos_c = h_pos - h_pos.mean(dim=0, keepdim=True)
    h_neg_c = h_neg - h_neg.mean(dim=0, keepdim=True)
    c_pos = (h_pos_c.T @ h_pos_c) / max(len(h_pos) - 1, 1)
    c_neg = (h_neg_c.T @ h_neg_c) / max(len(h_neg) - 1, 1)
    c_contrast = c_pos - beta * c_neg
    eigenvalues, eigenvectors = torch.linalg.eigh(c_contrast.float())
    k = min(k, eigenvectors.shape[1])
    subspace = eigenvectors[:, -k:]
    selected = eigenvalues[-k:]
    denom = eigenvalues.abs().sum().clamp_min(1e-12)
    explained = (selected.abs().sum() / denom).item()
    return subspace, selected, explained


def _linear_probe_accuracy(records, seed, test_size):
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score
    except Exception as exc:
        print(f"WARNING: Could not run linear probe: {exc}")
        return None

    qids = sorted({r["qid"] for r in records})
    if len(qids) < 2:
        return None

    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(qids), generator=generator).tolist()
    n_test = max(1, int(round(len(qids) * test_size)))
    test_qids = {qids[i] for i in order[:n_test]}

    train_records = [r for r in records if r["qid"] not in test_qids]
    test_records = [r for r in records if r["qid"] in test_qids]
    if not train_records or not test_records:
        return None
    if len({r["label"] for r in train_records}) < 2 or len({r["label"] for r in test_records}) < 2:
        return None

    x_train = torch.stack([r["latent"].float().cpu() for r in train_records]).numpy()
    y_train = [r["label"] for r in train_records]
    x_test = torch.stack([r["latent"].float().cpu() for r in test_records]).numpy()
    y_test = [r["label"] for r in test_records]

    probe = LogisticRegression(max_iter=1000)
    probe.fit(x_train, y_train)
    return float(accuracy_score(y_test, probe.predict(x_test)))


def _build_latent_prompt(sample, n_latents):
    return (
        sample["question"]
        + "\n<|start-latent|>"
        + "<|latent|>" * n_latents
        + "<|end-latent|>"
    )


def _save_tensor(tensor, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(tensor, path)
    print(f"[LOG] Saved tensor to {path}")


def extract_truth_vector(coconut_model, data_phase2, tokenizer, config, latent_id, start_id, end_id, run_dir=None):
    print("Starting Phase 2: Extracting frozen-model truth vectors...")
    phase2_start = time.time()

    previous_trainable = [param.requires_grad for param in coconut_model.parameters()]
    for param in coconut_model.parameters():
        param.requires_grad = False
    coconut_model.eval()

    ckpt_dir = os.path.join(run_dir, "checkpoints") if run_dir else config.save_path
    os.makedirs(ckpt_dir, exist_ok=True)

    correct_latents = []
    wrong_latents = []
    probe_records = []
    position_latents = defaultdict(lambda: {"pos": [], "neg": []})
    generated_paths = 0
    skipped_non_contrastive = 0
    n_latents_infer = config.max_latent_tokens

    try:
        for sample in tqdm(data_phase2, desc="Phase 2 Extraction"):
            prompt = _build_latent_prompt(sample, n_latents_infer)
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(config.device)
            ground_truth = sample["ground_truth"]
            qid = sample.get("qid") or sample["question"]
            sample_records = []

            for _ in range(config.num_generations_per_sample):
                with torch.no_grad():
                    generated_ids, mean_latent, _ = coconut_model.generate_with_latents(
                        input_ids,
                        max_new_tokens=config.max_new_tokens_ccot,
                        temperature=config.vector_extraction_temperature,
                    )

                if mean_latent is None:
                    continue

                output_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                predicted_answer = extract_final_answer(output_text)
                is_correct = answers_match(predicted_answer, ground_truth)
                latent_steps = [h.squeeze(0).float() for h in coconut_model.last_generation_latents]
                sample_records.append({
                    "qid": qid,
                    "label": int(is_correct),
                    "latent": mean_latent.squeeze(0).float(),
                    "steps": latent_steps,
                })
                generated_paths += 1

            labels = {record["label"] for record in sample_records}
            if (
                config.skip_non_contrastive_questions
                and config.num_generations_per_sample > 1
                and len(labels) < 2
            ):
                skipped_non_contrastive += 1
                continue

            for record in sample_records:
                if record["label"] == 1:
                    correct_latents.append(record["latent"])
                    bucket = "pos"
                else:
                    wrong_latents.append(record["latent"])
                    bucket = "neg"
                probe_records.append({
                    "qid": record["qid"],
                    "label": record["label"],
                    "latent": record["latent"],
                })
                for pos, latent in enumerate(record["steps"]):
                    position_latents[pos][bucket].append(latent)
    finally:
        for param, requires_grad in zip(coconut_model.parameters(), previous_trainable):
            param.requires_grad = requires_grad

    print(f"Generated Paths: {generated_paths}")
    print(f"Skipped Non-Contrastive Questions: {skipped_non_contrastive}")
    print(f"Total Correct Paths: {len(correct_latents)}")
    print(f"Total Incorrect Paths: {len(wrong_latents)}")

    metadata = {
        "source": "ccot_finetuned",
        "generated_paths": generated_paths,
        "skipped_non_contrastive_questions": skipped_non_contrastive,
        "n_correct": len(correct_latents),
        "n_wrong": len(wrong_latents),
        "num_generations_per_sample": config.num_generations_per_sample,
        "temperature": config.vector_extraction_temperature,
        "min_vector_class_count": config.min_vector_class_count,
        "position_counts": {},
        "cpca": {},
    }

    hidden_size = coconut_model.base_causallm.config.hidden_size
    if len(correct_latents) > 0 and len(wrong_latents) > 0:
        h_pos = torch.stack(correct_latents).float()
        h_neg = torch.stack(wrong_latents).float()
        truth_vector = _compute_difference_of_means(h_pos, h_neg).unsqueeze(0).to(config.device)
        if min(len(correct_latents), len(wrong_latents)) < config.min_vector_class_count:
            print(
                "WARNING: One vector class is below the configured minimum "
                f"({config.min_vector_class_count})."
            )
    else:
        print("WARNING: Missing either correct or wrong examples. Vector will be zero.")
        h_pos, h_neg = None, None
        truth_vector = torch.zeros((1, hidden_size), device=config.device)

    vector_path = os.path.join(ckpt_dir, "truth_vector.pt")
    dom_path = os.path.join(ckpt_dir, "truth_vector_dom.pt")
    _save_tensor(truth_vector, vector_path)
    _save_tensor(truth_vector, dom_path)

    if h_pos is not None and h_neg is not None:
        for pos, buckets in sorted(position_latents.items()):
            metadata["position_counts"][str(pos)] = {
                "correct": len(buckets["pos"]),
                "wrong": len(buckets["neg"]),
            }
            if buckets["pos"] and buckets["neg"]:
                pos_vector = _compute_difference_of_means(
                    torch.stack(buckets["pos"]).float(),
                    torch.stack(buckets["neg"]).float(),
                ).unsqueeze(0)
                _save_tensor(pos_vector, os.path.join(ckpt_dir, f"truth_vector_pos{pos}.pt"))

        if config.compute_cpca:
            for k in config.cpca_k_values:
                subspace, eigenvalues, explained = _compute_cpca_subspace(
                    h_pos, h_neg, k=k, beta=config.cpca_beta
                )
                cpca_path = os.path.join(ckpt_dir, f"truth_subspace_cpca_k{k}.pt")
                _save_tensor(subspace, cpca_path)
                metadata["cpca"][str(k)] = {
                    "path": cpca_path,
                    "explained_contrastive_variance": explained,
                    "eigenvalues": [float(v) for v in eigenvalues.cpu()],
                }

    metadata["linear_probe_accuracy"] = _linear_probe_accuracy(
        probe_records,
        seed=config.seed,
        test_size=config.linear_probe_test_size,
    )
    if metadata["linear_probe_accuracy"] is not None:
        print(f"Linear probe accuracy: {metadata['linear_probe_accuracy']:.3f}")
    else:
        print("Linear probe accuracy: unavailable")

    l2_norm = torch.norm(truth_vector).item()
    print(f"Computed Truth Vector. Shape: {truth_vector.shape}")
    print(f"Truth Vector L2 Norm: {l2_norm:.6f}")
    if l2_norm < 1e-3:
        print("WARNING: Vector magnitude is extremely small.")

    duration_s = time.time() - phase2_start
    metadata["duration_s"] = duration_s

    if run_dir:
        save_extraction_log(
            run_dir,
            n_correct=len(correct_latents),
            n_wrong=len(wrong_latents),
            vector_shape=truth_vector.shape,
            l2_norm=l2_norm,
            duration_s=duration_s,
        )
        save_json_log(run_dir, "truth_vector_metadata.json", metadata)
    else:
        metadata_path = os.path.join(ckpt_dir, "truth_vector_metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)
        print(f"[LOG] Truth vector metadata saved to {metadata_path}")

    clear_memory()
    return truth_vector, correct_latents, wrong_latents
