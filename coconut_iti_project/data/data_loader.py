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


def parse_gsm(ex):
    """Parse a GSM8K example into question, steps, answer, and ground_truth."""
    raw_answer = ex["answer"]
    if "####" in raw_answer:
        reasoning, final_ans = raw_answer.split("####")
        final_ans = final_ans.strip()
    else:
        reasoning = raw_answer
        final_ans = ""
    steps = [s.strip() for s in reasoning.split("\n") if s.strip()]
    return {"question": ex["question"], "steps": steps, "answer": final_ans, "ground_truth": final_ans}


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


def prepare_datasets(config):
    """
    Load and prepare all datasets from local JSONL files:
    - Phase 1 (Training): First 6,473 examples from train.jsonl
    - Phase 2 (Truth Vector Extraction): Last 1,000 examples from train.jsonl
    - Phase 4 (Evaluation): All 1,319 examples from test.jsonl
    """
    print(f"Loading training data from {TRAIN_JSONL}...")
    raw_train = _load_jsonl(TRAIN_JSONL)
    print(f"Loading test data from {TEST_JSONL}...")
    raw_test = _load_jsonl(TEST_JSONL)

    # Parse all examples
    all_train_parsed = [parse_gsm(ex) for ex in raw_train]
    test_data = [parse_gsm(ex) for ex in raw_test]

    # Split train: bulk for Phase 1, last 1000 for Phase 2
    split_point = len(all_train_parsed) - 1000
    data_phase1 = all_train_parsed[:split_point]
    data_phase2 = all_train_parsed[split_point:]

    print("\n" + "=" * 50)
    print(f"Phase 1 (Training):                {len(data_phase1):,} examples")
    print(f"Phase 2 (Truth Vector Extraction): {len(data_phase2):,} examples")
    print(f"Phase 4 (Ablation Test Set):       {len(test_data):,} examples")
    print("=" * 50 + "\n")

    return data_phase1, data_phase2, test_data
