import torch
from configs.config import Config
from models.coconut import initialize_model
from data.data_loader import prepare_datasets
from core.trainer import MyCollator, get_stage_info, get_cot_latent_dataset
from torch.utils.data import DataLoader

config = Config()
config.batch_size_training = 1
config.gradient_accumulation_steps = 1
coconut_model, tokenizer, latent_id, start_id, end_id = initialize_model(config)

data_phase1, _, _ = prepare_datasets(config)
ds_phase1 = data_phase1

train_ds = get_cot_latent_dataset(
    0, False, ds_phase1, config, start_id, latent_id, end_id, shuffle=False
)
collator = MyCollator(tokenizer, latent_id=latent_id)
train_loader = DataLoader(
    train_ds, batch_size=config.batch_size_training, collate_fn=collator, shuffle=False
)

batch = next(iter(train_loader))
input_ids = batch["input_ids"].to(config.device)
attention_mask = batch["attention_mask"].to(config.device)
labels = batch["labels"].to(config.device)

print("Input ids:", input_ids.shape)
print("Labels:", labels.shape)
valid_labels = labels[labels != -100]
print("Valid labels:", valid_labels)

with torch.amp.autocast("cuda", dtype=torch.bfloat16):
    outputs = coconut_model(input_ids, attention_mask, labels)
    loss = outputs.loss.to(torch.float32)
    print("Initial Loss:", loss.item())

    # Do a dummy backward
    loss.backward()

print("Grad norm:", torch.nn.utils.clip_grad_norm_(coconut_model.parameters(), 1.0).item())

