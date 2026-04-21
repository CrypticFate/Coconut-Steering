import random

from datasets import load_dataset
from datasets import Dataset as HFDataset


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


def parse_metamath(ex):
    """Parse a MetaMathQA example into question, steps, answer, and ground_truth."""
    response = ex["response"]
    if "The answer is:" in response:
        reasoning, ans = response.split("The answer is:")
        ans = ans.strip()
    else:
        reasoning = response
        ans = ""
        
    steps = [s.strip() for s in reasoning.split("\n") if s.strip()]
    if len(steps) <= 1:
        steps = [s.strip() + "." for s in reasoning.split(". ") if s.strip()]
    return {"question": ex["query"], "steps": steps, "answer": ans, "ground_truth": ans}


def get_hf_dataset(raw_data, tokenizer):
    """Tokenize raw data samples into a HuggingFace Dataset."""
    def tokenize_sample(sample):
        q_tok = tokenizer.encode(sample["question"] + "\n", add_special_tokens=True)
        s_tok = [tokenizer.encode(s + "\n", add_special_tokens=False) for s in sample["steps"]]
        a_tok = tokenizer.encode("#### " + sample["answer"], add_special_tokens=False) + [tokenizer.eos_token_id]
        return {
            "question_tokenized": q_tok,
            "steps_tokenized": s_tok,
            "answer_tokenized": a_tok,
            "ground_truth": sample["answer"],
        }

    ds = HFDataset.from_list(raw_data)
    return ds.map(tokenize_sample, num_proc=4)


def prepare_datasets(config):
    """
    Load and prepare all datasets for the pipeline:
    - Phase 1: 50,000 synthetic MetaMathQA examples (GSM-augmented subset)
    - Phase 2: 1,000 original GSM8K train examples for truth vector extraction
    - Phase 4: Full GSM8K test set for ablation evaluation
    """
    print("Loading Original GSM8K for Evaluation and Vector Extraction...")
    raw_gsm = load_dataset("gsm8k", "main")

    print("Loading Augmented Synthetic Dataset (MetaMathQA) for Phase 1...")
    raw_metamath = load_dataset("meta-math/MetaMathQA", split="train")
    gsm_augmented = raw_metamath.filter(lambda x: x["type"].startswith("GSM"))

    print("Parsing dataset...")
    data_phase1 = [parse_metamath(ex) for ex in gsm_augmented]

    # Prune to 50,000 examples for realistic local GPU training time
    data_phase1 = random.sample(data_phase1, min(50000, len(data_phase1)))

    train_gsm_parsed = [parse_gsm(ex) for ex in raw_gsm["train"]]
    data_phase2 = train_gsm_parsed[:1000]
    test_data = [parse_gsm(ex) for ex in raw_gsm["test"]]

    print("\n" + "=" * 50)
    print(f"Phase 1 (Training - Synthetic Subset): {len(data_phase1):,} examples")
    print(f"Phase 2 (Truth Vector Extraction): {len(data_phase2):,} examples")
    print(f"Phase 4 (Ablation Test Set): {len(test_data):,} examples")
    print("=" * 50 + "\n")

    return data_phase1, data_phase2, test_data
