import itertools
import os
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase
from transformers.data.data_collator import pad_without_fast_tokenizer_warning
from tqdm.auto import tqdm

from data.data_loader import get_hf_dataset
from utils.helpers import clear_memory


def get_stage_info(epoch):
    """
    5-stage COCONUT curriculum for 50 epochs:
    - Epochs  0-5:  Stage 0 (full CoT, no latent drops)
    - Epochs  6-8:  Stage 1 (drop 1 step, replace with latents)
    - Epochs  9-11: Stage 2 (drop 2 steps)
    - Epochs 12-14: Stage 3 (drop 3 steps)
    - Epochs 15-49: Stage 4 (drop ALL remaining text, full latent mode)
    """
    if epoch < 6:
        return 0, False, (epoch == 0)
    elif epoch < 9:
        return 1, False, (epoch == 6)
    elif epoch < 12:
        return 2, False, (epoch == 9)
    elif epoch < 15:
        return 3, False, (epoch == 12)
    else:
        return 4, True, (epoch == 15)


def get_cot_latent_dataset(scheduled_stage, drop_remaining, base_dataset, configs,
                           start_id, latent_id, end_id, shuffle=False):
    """
    Build the COCONUT curriculum dataset for a given stage.
    
    In hybrid mode, the first reasoning step is always kept as the
    'reasoning skeleton'. Remaining steps are progressively replaced
    with latent tokens across stages.
    """
    def process_dataset(sample):
        # --- HYBRID ARCHITECTURE ---
        # Extract Step 1 as the "Reasoning Skeleton" (always kept in English)
        if len(sample["steps_tokenized"]) > 0 and configs.hybrid_mode:
            skeleton_text = sample["steps_tokenized"][0]
            remaining_steps = sample["steps_tokenized"][1:]
        else:
            skeleton_text = []
            remaining_steps = sample["steps_tokenized"]

        # Drop whole steps according to the current curriculum stage
        steps_to_drop = min(scheduled_stage, len(remaining_steps))
        
        if drop_remaining:
            kept_remaining_steps = []
            n_latent_tokens = configs.max_latent_tokens
        else:
            kept_remaining_steps = remaining_steps[steps_to_drop:]
            n_latent_tokens = steps_to_drop * configs.c_thought
            
        kept_remaining_text = list(itertools.chain.from_iterable(kept_remaining_steps))

        # Build: [Question] -> [Skeleton] -> <bot> [Latents] <eot> -> [Remaining] -> [Answer]
        tokens = (sample["question_tokenized"] + skeleton_text + 
                  [start_id] + [latent_id] * n_latent_tokens + [end_id] +
                  kept_remaining_text + sample["answer_tokenized"])
        
        # Mask: loss only on text AFTER the latent section
        mask_len = len(sample["question_tokenized"]) + len(skeleton_text) + n_latent_tokens + 2
        labels = [-100] * mask_len + tokens[mask_len:]
        
        tokens = tokens[:configs.max_seq_len]
        labels = labels[:configs.max_seq_len]
        return {"input_ids": tokens, "labels": labels, "attention_mask": [1] * len(tokens)}
    
    dataset = base_dataset.map(process_dataset, remove_columns=list(base_dataset.features))
    if shuffle:
        dataset = dataset.shuffle(seed=configs.seed)
    return dataset


@dataclass
class MyCollator:
    """
    Custom data collator that aligns latent token positions across
    batch elements by left-padding shorter sequences.
    """
    tokenizer: PreTrainedTokenizerBase
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
        batch = pad_without_fast_tokenizer_warning(self.tokenizer, features, padding=True, return_tensors="pt")
        if labels:
             max_len = batch["input_ids"].shape[1]
             batch["labels"] = torch.tensor([l + [self.label_pad_token_id]*(max_len-len(l)) for l in labels])
        return batch


def train_phase1(coconut_model, data_phase1, tokenizer, config, latent_id, start_id, end_id):
    """
    Full-parameter multi-stage COCONUT training.
    
    Uses AdamW8bit (not PagedAdamW8bit) for full-parameter optimization,
    with torch.amp.autocast for bfloat16 mixed precision.
    """
    import bitsandbytes as bnb  # Lazy import to prevent segfault at module load time

    collator = MyCollator(tokenizer, latent_id=latent_id)
    ds_phase1 = get_hf_dataset(data_phase1, tokenizer)

    optimizer = None
    loss_history = []

    print("Starting FULL PARAMETER Multi-Stage COCONUT Training...")
    for epoch in range(config.num_epochs_total):

        current_stage, drop_remaining, requires_reset = get_stage_info(epoch)

        if requires_reset or optimizer is None:
            print(f"\n[Epoch {epoch}] Stage shifted to {current_stage}. "
                  f"Hard resetting AdamW Optimizer...")
            # Full-parameter optimizer: ALL parameters, not just LoRA adapters
            optimizer = bnb.optim.AdamW8bit(
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
            desc=f"Epoch {epoch + 1}/{config.num_epochs_total} | Stage {current_stage}",
        )

        for step, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(config.device)
            attention_mask = batch["attention_mask"].to(config.device)
            labels = batch["labels"].to(config.device)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = coconut_model(input_ids, attention_mask, labels)
                loss = outputs.loss / config.gradient_accumulation_steps

            loss.backward()

            if (step + 1) % config.gradient_accumulation_steps == 0:
                nn.utils.clip_grad_norm_(coconut_model.parameters(), max_norm=1.0)
                
                optimizer.step()
                optimizer.zero_grad()
                
                del outputs
                del loss
                torch.cuda.empty_cache()

            total_loss += loss.item() * config.gradient_accumulation_steps if 'loss' in locals() else total_loss
            pbar.set_postfix({"loss": total_loss / (step + 1)})

        # Track epoch-level average loss
        epoch_avg_loss = total_loss / max(len(train_loader), 1)
        loss_history.append(epoch_avg_loss)

    # Save checkpoint
    checkpoint_path = os.path.join(config.save_path, "coconut_phase1.pt")
    torch.save(coconut_model.state_dict(), checkpoint_path)
    print(f"Phase 1 checkpoint saved to {checkpoint_path}")

    clear_memory()
    return loss_history
