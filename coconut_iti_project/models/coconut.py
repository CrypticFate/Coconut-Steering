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
    for numerical stability during LoRA fine-tuning.
    """
    def __init__(self, base_causallm, latent_token_id, start_latent_id, end_latent_id, eos_token_id):
        super().__init__()
        self.base_causallm = base_causallm
        self.latent_token_id = latent_token_id
        self.start_latent_id = start_latent_id
        self.end_latent_id = end_latent_id
        self.eos_token_id = eos_token_id
        self.embedding = self.base_causallm.get_input_embeddings()
        self.last_steering_stats = []
        self.last_generation_latents = []
        self.last_trajectory_faithfulness = 0.0

    def _process_kv(self, kv_cache, keep_len):
        if kv_cache is None:
            return None

        # Newer transformers DynamicCache variants may not expose indexable
        # per-layer tuples (or key_cache/value_cache lists). In that case,
        # the cache is already in the expected opaque format for the model
        # forward call, so we pass it through unchanged.
        if not hasattr(kv_cache, "key_cache") and not isinstance(kv_cache, (list, tuple)):
            return kv_cache

        new_cache = DynamicCache()
        num_layers = len(kv_cache.key_cache) if hasattr(kv_cache, "key_cache") else len(kv_cache)
        for i in range(num_layers):
            try:
                if hasattr(kv_cache, "key_cache"):
                    k, v = kv_cache.key_cache[i], kv_cache.value_cache[i]
                else:
                    k, v = kv_cache[i]
            except (TypeError, IndexError, KeyError, AttributeError):
                k, v = kv_cache[i]
            new_cache.update(k[..., :keep_len, :], v[..., :keep_len, :], layer_idx=i)
        return new_cache

    def _alpha_tensor(self, alpha, device, dtype):
        if torch.is_tensor(alpha):
            return alpha.to(device=device, dtype=dtype)
        return torch.tensor(float(alpha), device=device, dtype=dtype)

    def _prepare_direction(self, steering_vector, hidden_states, mode):
        vector = steering_vector.to(device=hidden_states.device, dtype=hidden_states.dtype)
        if mode == "subspace":
            if vector.dim() != 2:
                raise ValueError("Subspace steering expects a [hidden, k] matrix.")
            return vector
        if vector.dim() == 1:
            vector = vector.unsqueeze(0)
        return F.normalize(vector, p=2, dim=-1)

    def _apply_steering(self, hidden_states, steering_vector, alpha, gamma, pass_idx,
                        steering_mode, collect_steering_stats):
        if steering_vector is None:
            return hidden_states

        h_t = hidden_states[:, -1, :]
        d_model = h_t.shape[-1]
        alpha_t = self._alpha_tensor(alpha, h_t.device, h_t.dtype) * (gamma ** pass_idx)
        sigma_t = h_t.norm(dim=-1, keepdim=True) / (d_model ** 0.5)
        direction = self._prepare_direction(steering_vector, hidden_states, steering_mode)

        if steering_mode == "subspace":
            h_unit = F.normalize(h_t, p=2, dim=-1)
            projection = h_unit @ direction @ direction.T
            intervention = alpha_t * sigma_t * projection
        else:
            intervention = alpha_t * sigma_t * direction

        steered_h = h_t + intervention
        hidden_states = hidden_states.clone()
        hidden_states[:, -1, :] = steered_h

        if collect_steering_stats:
            self.last_steering_stats.append({
                "h_before": h_t,
                "h_after": steered_h,
                "intervention": intervention,
                "direction": direction,
            })

        return hidden_states

    def steered_forward_for_tuning(
        self,
        question_ids,
        answer_ids,
        truth_vector,
        k,
        gamma=1.0,
        alpha_fn=None,
        alpha_tensor=None,
        steering_mode="vector",
    ):
        """
        Gradient-safe steered forward for Phase-3 α tuning.

        Prefix embeddings (question + ``<|start-latent|>``) are built under ``torch.no_grad()``;
        each of ``k`` latent steps recomputes the full prefix + steered slots with
        ``past_key_values=None`` so no detached KV cache breaks the graph to ``alpha``.

        Pass exactly one of ``alpha_fn`` (trainable, e.g. sigmoid(theta)) or ``alpha_tensor``
        (fixed scalar α for eval).

        Steering uses float32 math then casts back to the model dtype (stable α gradients under autocast).
        """
        if (alpha_fn is None) == (alpha_tensor is None):
            raise ValueError("Pass exactly one of alpha_fn or alpha_tensor.")
        if steering_mode != "vector":
            raise ValueError("steered_forward_for_tuning only supports steering_mode='vector'.")

        device = question_ids.device
        model_dtype = self.base_causallm.dtype
        embed = self.embedding

        v_truth = F.normalize(truth_vector.float().to(device), p=2, dim=-1)
        if v_truth.dim() == 2:
            v_truth = v_truth.squeeze(0)

        with torch.no_grad():
            pref = embed(question_ids)

        p = question_ids.shape[1]
        steered_slots = []
        h_steered_seq = []

        for t in range(k):
            inputs_embeds = torch.cat([pref] + steered_slots, dim=1)
            attn = torch.ones(
                1, inputs_embeds.shape[1], device=device, dtype=torch.long
            )
            pos = torch.arange(
                0, inputs_embeds.shape[1], dtype=torch.long, device=device
            ).unsqueeze(0)

            outputs = self.base_causallm(
                inputs_embeds=inputs_embeds,
                attention_mask=attn,
                position_ids=pos,
                past_key_values=None,
                output_hidden_states=True,
                use_cache=False,
            )
            h_t = outputs.hidden_states[-1][:, -1, :]

            h32 = h_t.float()
            sig = h32.std(dim=-1, keepdim=True)
            if alpha_tensor is not None:
                a = alpha_tensor.float()
            else:
                a = alpha_fn().float()
            steer = a * (gamma ** t) * sig * v_truth
            h_steered = (h32 + steer).to(model_dtype)
            h_steered_seq.append(h_steered.squeeze(0))
            steered_slots.append(h_steered.unsqueeze(1))

        end_id = torch.tensor([[self.end_latent_id]], device=device, dtype=torch.long)
        end_e = embed(end_id)
        ans_e = embed(answer_ids)
        full_emb = torch.cat([pref] + steered_slots + [end_e, ans_e], dim=1)

        prompt_len = p + k + 1
        ans_len = answer_ids.shape[1]
        labels = torch.full(
            (1, prompt_len + ans_len),
            -100,
            dtype=torch.long,
            device=device,
        )
        labels[0, prompt_len : prompt_len + ans_len] = answer_ids[0]

        attn_full = torch.ones(1, full_emb.shape[1], device=device, dtype=torch.long)
        pos_full = torch.arange(0, full_emb.shape[1], dtype=torch.long, device=device).unsqueeze(0)

        out_final = self.base_causallm(
            inputs_embeds=full_emb,
            attention_mask=attn_full,
            position_ids=pos_full,
            past_key_values=None,
            output_hidden_states=False,
            use_cache=False,
        )
        logits = out_final.logits

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = CrossEntropyLoss()(
            shift_logits.view(-1, shift_logits.size(-1)).float(),
            shift_labels.view(-1),
        )
        return loss, h_steered_seq

    def forward(self, input_ids, attention_mask, labels=None, steering_vector=None, alpha=0.0,
                gamma=1.0, steering_mode="vector", collect_steering_stats=False,
                detach_latents=True, use_kv_cache=True):
        """
        Custom forward pass that routes hidden states through continuous thought
        and applies ITI steering: h_new = h_old + (alpha * sigma * v_truth)
        
        Includes Qwen RoPE position_ids fix for correct rotary position encoding.

        use_kv_cache: When False, each latent step recomputes activations from position 0 with
        past_key_values=None. This preserves gradients through alpha during Phase 3 tuning; the
        cached incremental path can detach the graph across latent steps.
        """
        latent_sequence = []
        self.last_steering_stats = []

        latent_indices = (input_ids == self.latent_token_id).nonzero()
        latent_lists = [
            [idx[1].item() for idx in latent_indices if idx[0] == i]
            for i in range(input_ids.shape[0])
        ]
        max_n_latents = max([len(l) for l in latent_lists]) if latent_lists else 0

        inputs_embeds = self.embedding(input_ids)
        position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device).unsqueeze(0)

        next_compute_range = (0, input_ids.shape[1])
        if max_n_latents > 0:
            next_compute_range = (0, latent_indices[:, 1].min().item())

        if not use_kv_cache:
            for pass_idx in range(max_n_latents):
                end = next_compute_range[1]
                outputs = self.base_causallm(
                    inputs_embeds=inputs_embeds[:, :end],
                    attention_mask=attention_mask[:, :end],
                    position_ids=position_ids[:, :end],
                    past_key_values=None,
                    output_hidden_states=True,
                    use_cache=False,
                )
                next_end = input_ids.shape[1] if pass_idx + 1 >= max_n_latents else next_compute_range[1] + 1
                next_compute_range = (next_compute_range[1], next_end)

                hidden_states = outputs.hidden_states[-1]
                hidden_states = self._apply_steering(
                    hidden_states=hidden_states,
                    steering_vector=steering_vector,
                    alpha=alpha,
                    gamma=gamma,
                    pass_idx=pass_idx,
                    steering_mode=steering_mode,
                    collect_steering_stats=collect_steering_stats,
                )
                latent_sequence.append(hidden_states.detach() if detach_latents else hidden_states)
                inputs_embeds = inputs_embeds.clone()
                filling_indices = [(i, l[pass_idx]) for i, l in enumerate(latent_lists) if len(l) > pass_idx]
                for batch_idx, token_idx in filling_indices:
                    inputs_embeds[batch_idx, token_idx] = hidden_states[batch_idx, -1]

            outputs = self.base_causallm(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=None,
                output_hidden_states=False,
                use_cache=False,
            )
            logits = outputs.logits
        else:
            logits = []
            kv_cache = None
            for pass_idx in range(max_n_latents):
                curr_cache = self._process_kv(kv_cache, next_compute_range[0])
                outputs = self.base_causallm(
                    inputs_embeds=inputs_embeds[:, next_compute_range[0] : next_compute_range[1]],
                    attention_mask=attention_mask[:, :next_compute_range[1]],
                    position_ids=position_ids[:, next_compute_range[0] : next_compute_range[1]],
                    past_key_values=curr_cache,
                    output_hidden_states=True,
                    use_cache=True,
                )
                logits.append(outputs.logits)

                next_end = input_ids.shape[1] if pass_idx + 1 >= max_n_latents else next_compute_range[1] + 1
                next_compute_range = (next_compute_range[1], next_end)

                hidden_states = outputs.hidden_states[-1]
                hidden_states = self._apply_steering(
                    hidden_states=hidden_states,
                    steering_vector=steering_vector,
                    alpha=alpha,
                    gamma=gamma,
                    pass_idx=pass_idx,
                    steering_mode=steering_mode,
                    collect_steering_stats=collect_steering_stats,
                )
                latent_sequence.append(hidden_states.detach() if detach_latents else hidden_states)

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
                use_cache=True,
            )
            logits.append(outputs.logits)
            logits = torch.cat(logits, dim=-2)
        
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Strict FP32 upcasting to prevent CrossEntropy explosion
            loss = CrossEntropyLoss()(shift_logits.view(-1, shift_logits.size(-1)).to(torch.float32), shift_labels.view(-1))

        return Outputs(loss, inputs_embeds, logits, latent_sequence)

    def generate_with_latents(self, input_ids, max_new_tokens=128, temperature=0.0,
                              steering_vector=None, alpha=0.0, gamma=1.0,
                              steering_mode="vector"):
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
                steering_mode=steering_mode,
            )

        latent_steps = [h[:, -1, :].detach().cpu() for h in outputs.latent_sequence]
        self.last_generation_latents = latent_steps

        mean_latent = (
            torch.mean(torch.stack(latent_steps), dim=0)
            if latent_steps
            else None
        )

        trajectory_scores = []
        if len(latent_steps) > 1:
            for h_prev, h_next in zip(latent_steps[:-1], latent_steps[1:]):
                trajectory_scores.append(F.cosine_similarity(h_prev, h_next, dim=-1).item())
        avg_faithfulness = float(np.mean(trajectory_scores)) if trajectory_scores else 0.0
        self.last_trajectory_faithfulness = avg_faithfulness

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
    Initialize Qwen2.5-Math-1.5B for full-parameter fine-tuning.
    
    All 1.5B parameters are trainable. The smaller model fits easily 
    in 24GB VRAM with a native 32-bit AdamW optimizer, eliminating the 
    need for 8-bit quantization or FP32 upcasting hacks.
    """
    print(f"Initializing {config.model_id} for Full-Parameter Tuning...")

    model = AutoModelForCausalLM.from_pretrained(
        config.model_id, 
        torch_dtype=torch.bfloat16,
        attn_implementation="eager"
    ).to(config.device)

    tokenizer = AutoTokenizer.from_pretrained(config.model_id)
    # pad_token_id must differ from eos for generate(); pad==eos often stops after one token.
    added_pad_token = False
    if tokenizer.pad_token is None or tokenizer.pad_token_id == tokenizer.eos_token_id:
        if tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
            added_pad_token = True

    # Inject Latent Vocabulary
    tokenizer.add_tokens(["<|start-latent|>", "<|end-latent|>", "<|latent|>"])
    latent_id, start_id, end_id = tokenizer.convert_tokens_to_ids(
        ["<|latent|>", "<|start-latent|>", "<|end-latent|>"]
    )
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id

    # Initialize ALL custom tokens to prevent initial hidden state corruption
    with torch.no_grad():
        input_embeds = model.get_input_embeddings()
        init_id = tokenizer.encode("The", add_special_tokens=False)[0] 

        new_ids = [latent_id, start_id, end_id]
        if added_pad_token:
            new_ids.append(tokenizer.pad_token_id)
        for new_token_id in new_ids:
            input_embeds.weight.data[new_token_id] = input_embeds.weight.data[init_id].clone()
            if hasattr(model, 'lm_head') and model.lm_head is not None:
                model.lm_head.weight.data[new_token_id] = model.lm_head.weight.data[init_id].clone()
                
        # CRITICAL: Expose embeddings to the optimizer
        input_embeds.weight.requires_grad = True

    coconut_model = Coconut(model, latent_id, start_id, end_id, tokenizer.eos_token_id).to(config.device)

    return coconut_model, tokenizer, latent_id, start_id, end_id
