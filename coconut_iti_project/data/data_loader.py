from datasets import load_dataset, Dataset as HFDataset


def parse_example(ex):
    raw_answer = ex["answer"]
    if "####" in raw_answer:
        reasoning, final_ans = raw_answer.split("####")
        final_ans = final_ans.strip()
    else:
        reasoning = raw_answer
        final_ans = ""
    steps = [s.strip() for s in reasoning.split("\n") if s.strip()]
    return {"question": ex["question"], "steps": steps, "answer": final_ans}


def get_hf_dataset(raw_data, tokenizer):
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
    print("Loading and splitting GSM8K dataset...")
    raw_dataset = load_dataset("gsm8k", "main")

    train_full = [parse_example(ex) for ex in raw_dataset["train"]]

    total_samples = len(train_full)
    idx_phase1_end = int(total_samples * config.train_split_1_ratio)
    idx_phase2_end = idx_phase1_end + int(total_samples * config.train_split_2_ratio)

    data_phase1 = train_full[:idx_phase1_end]
    data_phase2 = train_full[idx_phase1_end:idx_phase2_end]
    data_phase3 = train_full[idx_phase2_end:]

    test_data = [parse_example(ex) for ex in raw_dataset["test"]]

    print(f"Total Train Sub-Dataset Used: {total_samples}")
    print(f"Phase 1 (Base Train - 60%): {len(data_phase1)}")
    print(f"Phase 2 (Vector Extraction - 10%): {len(data_phase2)}")
    print(f"Phase 3 (Validation - 30%): {len(data_phase3)}")
    print(f"Phase 4 (Full Test Set): {len(test_data)}")

    return data_phase1, data_phase2, data_phase3, test_data
