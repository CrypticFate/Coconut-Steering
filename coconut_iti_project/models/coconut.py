from collections import namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from torch.nn import CrossEntropyLoss


Outputs = namedtuple("Outputs", ["loss", "inputs_embeds", "logits", "latent_sequence"])


class Coconut(nn.Module):
    """
    The custom wrapper that intercepts the forward pass to enable
    Continuous Chain-of-Thought and Inference-Time Intervention.
    
    Implements the Qwen RoPE position_ids fix and float32 loss casting
    for numerical stability during full-parameter training.
    """
    def __init__(self, base_causallm, latent_token_id, start_latent_id, end_latent_id, eos_token_id):
        super().__init__()
        self.base_causallm = base_causallm
        self.latent_token_id = latent_token_id
        self.eos_token_id = eos_token_id
        self.embedding = self.base_causallm.get_input_embeddings()

    def _process_kv(self, kv_cache, keep_len):
        if kv_cache is None:
            return None
        new_cache = DynamicCache()
        num_layers = len(kv_cache.key_cache) if hasattr(kv_cache, "key_cache") else len(kv_cache)
        for i in range(num_layers):
            try:
                if hasattr(kv_cache, "key_cache"):
                    k, v = kv_cache.key_cache[i], kv_cache.value_cache[i]
                else:
                    k, v = kv_cache[i]
            except:
                k, v = kv_cache[i]
            new_cache.update(k[..., :keep_len, :], v[..., :keep_len, :], layer_idx=i)
        return new_cache

    def forward(self, input_ids, attention_mask, labels=None, steering_vector=None, alpha=0.0, gamma=1.0):
        """
        Custom forward pass that routes hidden states through continuous thought
        and applies ITI steering: h_new = h_old + (alpha * sigma * v_truth)
        
        Includes Qwen RoPE position_ids fix for correct rotary position encoding.
        """
        logits = []
        latent_sequence = []
        
        latent_indices = (input_ids == self.latent_token_id).nonzero()
        latent_lists = [
            [idx[1].item() for idx in latent_indices if idx[0] == i]
            for i in range(input_ids.shape[0])
        ]
        max_n_latents = max([len(l) for l in latent_lists]) if latent_lists else 0

        inputs_embeds = self.embedding(input_ids)
        # Qwen RoPE fix: explicit position_ids for correct rotary encoding
        position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device).unsqueeze(0)
        
        next_compute_range = (0, input_ids.shape[1])
        if max_n_latents > 0:
            next_compute_range = (0, latent_indices[:, 1].min().item())

        kv_cache = None

        for pass_idx in range(max_n_latents):
            curr_cache = self._process_kv(kv_cache, next_compute_range[0])
            outputs = self.base_causallm(
                inputs_embeds=inputs_embeds[:, next_compute_range[0] : next_compute_range[1]],
                attention_mask=attention_mask[:, :next_compute_range[1]],
                position_ids=position_ids[:, next_compute_range[0] : next_compute_range[1]], 
                past_key_values=curr_cache,
                output_hidden_states=True,
                use_cache=True 
            )
            logits.append(outputs.logits)

            next_end = input_ids.shape[1] if pass_idx + 1 >= max_n_latents else next_compute_range[1] + 1
            next_compute_range = (next_compute_range[1], next_end)

            hidden_states = outputs.hidden_states[-1]
            
            # --- Dynamic Alpha Decay ITI ---
            current_alpha = alpha * (gamma ** pass_idx)
            if steering_vector is not None and current_alpha > 0:
                sigma_l = hidden_states.std(dim=-1, keepdim=True)
                norm_dir = F.normalize(steering_vector, p=2, dim=-1).to(hidden_states.device)
                hidden_states = hidden_states + (current_alpha * sigma_l * norm_dir)

            latent_sequence.append(hidden_states.detach())
            
            kv_cache = outputs.past_key_values
            inputs_embeds = inputs_embeds.clone()

            filling_indices = [(i, l[pass_idx]) for i, l in enumerate(latent_lists) if len(l) > pass_idx]
            for batch_idx, token_idx in filling_indices:
                inputs_embeds[batch_idx, token_idx] = hidden_states[batch_idx, -1]

        final_cache = self._process_kv(kv_cache, next_compute_range[0])
        outputs = self.base_causallm(
            inputs_embeds=inputs_embeds[:, next_compute_range[0] : next_compute_range[1]],
            attention_mask=attention_mask[:, :next_compute_range[1]],
            position_ids=position_ids[:, next_compute_range[0] : next_compute_range[1]], 
            past_key_values=final_cache,
            use_cache=True 
        )
        logits.append(outputs.logits)
        logits = torch.cat(logits, dim=-2)
        
        # FP32 loss cast for numerical stability during full-parameter bfloat16 training
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = CrossEntropyLoss()(shift_logits.view(-1, shift_logits.size(-1)).to(torch.float32), shift_labels.view(-1))

        return Outputs(loss, inputs_embeds, logits, latent_sequence)

    def generate_with_latents(self, input_ids, max_new_tokens=128, temperature=0.0,
                              steering_vector=None, alpha=0.0, gamma=1.0):
        """Deterministic generation with continuous thought and optional ITI steering."""
        self.eval()
        tokens = input_ids.tolist()[0]

        with torch.no_grad():
            outputs = self.forward(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                steering_vector=steering_vector,
                alpha=alpha,
                gamma=gamma,
            )
            
        mean_latent = (
            torch.mean(torch.stack([h[:, -1, :] for h in outputs.latent_sequence]), dim=0).cpu()
            if outputs.latent_sequence
            else None
        )
        
        faithfulness_scores = []
        if steering_vector is not None and outputs.latent_sequence and alpha > 0.0:
            for h in outputs.latent_sequence:
                sim = F.cosine_similarity(h[:, -1, :], steering_vector.unsqueeze(0), dim=-1)
                faithfulness_scores.append(sim.item())
        avg_faithfulness = np.mean(faithfulness_scores) if faithfulness_scores else 0.0

        if temperature > 0:
            scaled_logits = outputs.logits[:, -1, :] / temperature
            probs = torch.nn.functional.softmax(scaled_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
        else:
            next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1).item()
            
        tokens.append(next_token)
        curr_input_ids = torch.tensor([tokens], device=input_ids.device)
        
        for _ in range(max_new_tokens):
            with torch.no_grad():
                outputs = self.base_causallm(input_ids=curr_input_ids)
                
            if temperature > 0:
                scaled_logits = outputs.logits[:, -1, :] / temperature
                probs = torch.nn.functional.softmax(scaled_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()
            else:
                next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1).item()
                
            if next_token == self.eos_token_id:
                break
            tokens.append(next_token)
            curr_input_ids = torch.tensor([tokens], device=input_ids.device)

        return torch.tensor([tokens]), mean_latent, avg_faithfulness


def initialize_model(config):
    """
    Initialize Qwen 3B for full-parameter training.
    
    No LoRA, no gradient checkpointing. Uses .to(device) instead of device_map="auto"
    to keep all parameters on a single GPU for full backprop.
    """
    print(f"Initializing {config.model_id} for Full-Parameter Phase 1...")

    model = AutoModelForCausalLM.from_pretrained(
        config.model_id, 
        torch_dtype=torch.bfloat16,
        attn_implementation="eager"
    ).to(config.device)

    tokenizer = AutoTokenizer.from_pretrained(config.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Inject Latent Vocabulary
    tokenizer.add_tokens(["<|start-latent|>", "<|end-latent|>", "<|latent|>"])
    latent_id, start_id, end_id = tokenizer.convert_tokens_to_ids(
        ["<|latent|>", "<|start-latent|>", "<|end-latent|>"]
    )
    model.resize_token_embeddings(len(tokenizer))

    # Initialize ALL custom tokens to prevent initial hidden state corruption
    with torch.no_grad():
        input_embeds = model.get_input_embeddings()
        init_id = tokenizer.encode("The", add_special_tokens=False)[0] 

        for new_token_id in [latent_id, start_id, end_id]:
            input_embeds.weight.data[new_token_id] = input_embeds.weight.data[init_id].clone()
            if hasattr(model, 'lm_head') and model.lm_head is not None:
                model.lm_head.weight.data[new_token_id] = model.lm_head.weight.data[init_id].clone()

        input_embeds.weight.requires_grad = True

    coconut_model = Coconut(model, latent_id, start_id, end_id, tokenizer.eos_token_id).to(config.device)

    return coconut_model, tokenizer, latent_id, start_id, end_id
