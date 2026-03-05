import itertools
import os
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from data.data_loader import get_hf_dataset
from utils.helpers import clear_memory


def get_stage_info(epoch):
    if epoch < 6:
        return 0, False, (epoch == 0)
    elif epoch < 9:
        return 1, False, (epoch == 6)
    elif epoch < 12:
        return 2, False, (epoch == 9)
    elif epoch < 15:
        return 3, False, (epoch == 12)
    else:
        return 3, True, (epoch == 15)


def get_cot_latent_dataset(scheduled_stage, drop_remaining, base_dataset, config,
                           start_id, latent_id, end_id, shuffle=False):
    def process_dataset(sample):
        n_steps_total = len(sample["steps_tokenized"])
        n_steps_to_latentize = min(scheduled_stage, n_steps_total)
        if n_steps_to_latentize > config.max_latent_stage:
            n_steps_to_latentize = config.max_latent_stage

        n_latent_tokens = n_steps_to_latentize * config.c_thought

        if drop_remaining:
            remaining_steps = []
        else:
            remaining_steps = list(itertools.chain.from_iterable(
                sample["steps_tokenized"][n_steps_to_latentize:]
            ))

        tokens = (
            sample["question_tokenized"]
            + [start_id]
            + [latent_id] * n_latent_tokens
            + [end_id]
            + remaining_steps
            + sample["answer_tokenized"]
        )

        mask_len = len(sample["question_tokenized"]) + n_latent_tokens + 2
        labels = [-100] * mask_len + tokens[mask_len:]

        tokens = tokens[:config.max_seq_len]
        labels = labels[:config.max_seq_len]

        return {"input_ids": tokens, "labels": labels, "attention_mask": [1] * len(tokens)}

    dataset = base_dataset.map(process_dataset, remove_columns=list(base_dataset.features))
    if shuffle:
        dataset = dataset.shuffle(seed=config.seed)
    return dataset


@dataclass
class MyCollator:
    tokenizer: object
    latent_id: int
    label_pad_token_id: int = -100

    def __call__(self, features):
        earliest_latent = [
            f["input_ids"].index(self.latent_id)
            for f in features
            if self.latent_id in f["input_ids"]
        ]
        if earliest_latent:
            latest_earliest = max(earliest_latent)
            for feature in features:
                pad = (
                    latest_earliest - feature["input_ids"].index(self.latent_id)
                    if self.latent_id in feature["input_ids"]
                    else 0
                )
                feature["input_ids"] = [self.tokenizer.pad_token_id] * pad + feature["input_ids"]
                feature["attention_mask"] = [0] * pad + feature["attention_mask"]
                if "labels" in feature:
                    feature["labels"] = [self.label_pad_token_id] * pad + feature["labels"]

        labels = [f.pop("labels") for f in features] if "labels" in features[0] else None

        # Manual padding (replaces pad_without_fast_tokenizer_warning)
        max_len = max(len(f["input_ids"]) for f in features)
        padded_input_ids = []
        padded_attention_mask = []
        for f in features:
            pad_len = max_len - len(f["input_ids"])
            padded_input_ids.append(f["input_ids"] + [self.tokenizer.pad_token_id] * pad_len)
            padded_attention_mask.append(f["attention_mask"] + [0] * pad_len)

        batch = {
            "input_ids": torch.tensor(padded_input_ids),
            "attention_mask": torch.tensor(padded_attention_mask),
        }

        if labels:
            batch["labels"] = torch.tensor(
                [l + [self.label_pad_token_id] * (max_len - len(l)) for l in labels]
            )
        return batch


def train_phase1(coconut_model, data_phase1, tokenizer, config, latent_id, start_id, end_id):
    import bitsandbytes as bnb  # Lazy import to prevent segfault at module load time

    collator = MyCollator(tokenizer, latent_id=latent_id)
    ds_phase1 = get_hf_dataset(data_phase1, tokenizer)

    optimizer = None
    loss_history = []

    print("Starting Phase 1 Base COCONUT Training...")
    for epoch in range(config.num_epochs_phase1):

        current_stage, drop_remaining, requires_reset = get_stage_info(epoch)

        if requires_reset or optimizer is None:
            print(f"\n[Epoch {epoch}] Stage shifted to {current_stage}. "
                  f"Hard resetting PagedAdamW8bit Optimizer...")
            optimizer = bnb.optim.PagedAdamW8bit(
                coconut_model.parameters(), lr=config.lr, weight_decay=config.weight_decay
            )
            torch.cuda.empty_cache()

        train_ds = get_cot_latent_dataset(
            current_stage, drop_remaining, ds_phase1, config, start_id, latent_id, end_id, shuffle=True
        )
        train_loader = DataLoader(
            train_ds, batch_size=config.batch_size_training, collate_fn=collator, shuffle=True
        )

        coconut_model.train()
        total_loss = 0
        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{config.num_epochs_phase1} | "
                 f"Stage {current_stage} | Drop Text: {drop_remaining}",
        )

        for step, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(config.device)
            attention_mask = batch["attention_mask"].to(config.device)
            labels = batch["labels"].to(config.device)

            outputs = coconut_model(input_ids, attention_mask, labels)

            loss_history.append(outputs.loss.item())

            loss = outputs.loss / config.gradient_accumulation_steps
            loss.backward()

            if (step + 1) % config.gradient_accumulation_steps == 0:
                nn.utils.clip_grad_norm_(coconut_model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                torch.cuda.empty_cache()

            total_loss += loss.item() * config.gradient_accumulation_steps
            pbar.set_postfix({"loss": total_loss / (step + 1)})

    # Save checkpoint
    checkpoint_path = os.path.join(config.save_path, "coconut_phase1.pt")
    torch.save(coconut_model.state_dict(), checkpoint_path)
    print(f"Phase 1 checkpoint saved to {checkpoint_path}")

    clear_memory()
    return loss_history
