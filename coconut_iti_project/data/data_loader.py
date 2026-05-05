import json
import os

import torch


# Paths to local JSONL files (relative to project root)
TRAIN_JSONL = os.path.join(os.path.dirname(os.path.dirname(__file__)), "train.jsonl")
TEST_JSONL = os.path.join(os.path.dirname(os.path.dirname(__file__)), "test.jsonl")


def _load_jsonl(filepath):
    """Load a JSONL file and return a list of dicts."""
    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def parse_gsm(ex, qid=None):
    """Parse a GSM8K example into question, steps, answer, and ground_truth."""
    raw_answer = ex["answer"]
    if "####" in raw_answer:
        reasoning, final_ans = raw_answer.split("####")
        final_ans = final_ans.strip()
    else:
        reasoning = raw_answer
        final_ans = ""
    steps = [s.strip() for s in reasoning.split("\n") if s.strip()]
    return {
        "qid": qid if qid is not None else ex.get("qid"),
        "question": ex["question"],
        "steps": steps,
        "answer": final_ans,
        "ground_truth": final_ans,
    }


def get_hf_dataset(raw_data, tokenizer):
    """
    Tokenize raw data samples into a list of dicts with a .map()-compatible
    wrapper class, without requiring the `datasets` library.
    """
    tokenized = []
    for sample in raw_data:
        q_tok = tokenizer.encode(sample["question"] + "\n", add_special_tokens=True)
        s_tok = [tokenizer.encode(s + "\n", add_special_tokens=False) for s in sample["steps"]]
        a_tok = tokenizer.encode("#### " + sample["answer"], add_special_tokens=False) + [tokenizer.eos_token_id]
        tokenized.append({
            "qid": sample.get("qid"),
            "question": sample["question"],
            "question_tokenized": q_tok,
            "steps_tokenized": s_tok,
            "answer_tokenized": a_tok,
            "ground_truth": sample["answer"],
        })
    return SimpleDataset(tokenized)


class SimpleDataset:
    """
    Lightweight dataset wrapper that mimics the HuggingFace Dataset interface
    (.map(), .shuffle(), .features, indexing) using plain Python lists.
    No external dependency required.
    """
    def __init__(self, data):
        self._data = data

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        return self._data[idx]

    def __iter__(self):
        return iter(self._data)

    @property
    def features(self):
        """Return the keys of the first element (used by trainer to strip columns)."""
        if self._data:
            return {k: None for k in self._data[0].keys()}
        return {}

    def map(self, func, remove_columns=None, num_proc=None):
        """Apply a function to each element, optionally removing columns."""
        mapped = []
        for item in self._data:
            result = func(item)
            if remove_columns:
                result = {k: v for k, v in result.items() if k not in remove_columns}
            mapped.append(result)
        return SimpleDataset(mapped)

    def shuffle(self, seed=42):
        """Return a shuffled copy of the dataset."""
        import random
        rng = random.Random(seed)
        shuffled = list(self._data)
        rng.shuffle(shuffled)
        return SimpleDataset(shuffled)


def prepare_datasets(config, include_val=False):
    """
    Load and prepare all datasets from local JSONL files:
    - Phase 1 (Training): train set minus the reserved protocol tail
    - Phase 2 (Truth Vector Extraction): first slice of the reserved tail
    - Phase 3 (Alpha Tuning): remaining reserved examples
    - Phase 4 (Evaluation): All 1,319 examples from test.jsonl

    By default this preserves the historical 3-tuple return. Pass
    include_val=True to get the v2 protocol split:
        data_phase1, data_phase2, data_val, test_data
    """
    print(f"Loading training data from {TRAIN_JSONL}...")
    raw_train = _load_jsonl(TRAIN_JSONL)
    print(f"Loading test data from {TEST_JSONL}...")
    raw_test = _load_jsonl(TEST_JSONL)

    # Parse all examples
    all_train_parsed = [parse_gsm(ex, qid=f"train_{i:05d}") for i, ex in enumerate(raw_train)]
    test_data = [parse_gsm(ex, qid=f"test_{i:05d}") for i, ex in enumerate(raw_test)]

    # Split train: preserve the existing Phase 1 split point, then carve
    # disjoint D_steer and D_val from the reserved tail.
    reserved = min(config.protocol_reserved_examples, len(all_train_parsed))
    split_point = len(all_train_parsed) - reserved
    data_phase1 = all_train_parsed[:split_point]
    protocol_tail = all_train_parsed[split_point:]
    steer_count = min(config.phase2_steer_examples, len(protocol_tail))
    data_phase2 = protocol_tail[:steer_count]
    data_val = protocol_tail[steer_count:]

    train_qids = {ex["qid"] for ex in data_phase1}
    steer_qids = {ex["qid"] for ex in data_phase2}
    val_qids = {ex["qid"] for ex in data_val}
    test_qids = {ex["qid"] for ex in test_data}
    assert train_qids.isdisjoint(steer_qids)
    assert train_qids.isdisjoint(val_qids)
    assert steer_qids.isdisjoint(val_qids)
    assert test_qids.isdisjoint(train_qids | steer_qids | val_qids)

    print("\n" + "=" * 50)
    print(f"Phase 1 (Training):                {len(data_phase1):,} examples")
    print(f"Phase 2 (Truth Vector Extraction): {len(data_phase2):,} examples")
    print(f"Phase 3 (Alpha Tuning):            {len(data_val):,} examples")
    print(f"Phase 4 (Locked Test Set):         {len(test_data):,} examples")
    print("=" * 50 + "\n")

    if include_val:
        return data_phase1, data_phase2, data_val, test_data
    return data_phase1, data_phase2, test_data
