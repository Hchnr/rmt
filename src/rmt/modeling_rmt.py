"""Recurrent Qwen3: token-local expert projections with global causal attention."""
from dataclasses import dataclass
from functools import partial
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from transformers import GenerationMixin
from transformers.cache_utils import Cache
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3PreTrainedModel, Qwen3RMSNorm, Qwen3RotaryEmbedding, apply_rotary_pos_emb,
)
from .configuration_rmt import RmtConfig
from .routing import BoundRouter, selected_probability_st
from .experts import BoundExpertBank
from .attention import causal_mask, attend
from .cache import RmtCache


@dataclass
class RmtCausalLMOutput(CausalLMOutputWithPast):
    last_hidden_state: Optional[torch.Tensor] = None
    router_indices: Optional[tuple] = None
    router_probabilities: Optional[tuple] = None


class RmtRecurrentCell(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.bank = BoundExpertBank(config)
        self.router = BoundRouter(config)

    def forward(self, hidden, step, mask, position_embeddings, mode,
                forced=None, cache=None):
        b, s, _ = hidden.shape
        uniform = None
        probabilities = hidden.new_empty(0, dtype=torch.float32)
        selected = None
        if mode == "layer_order":
            uniform = step % self.config.num_experts
            indices = torch.full((b, s), uniform, dtype=torch.long, device=hidden.device)
        elif mode == "forced":
            indices = forced[..., step]
        else:
            indices, selected, probabilities = self.router(hidden, step)
        groups = None if uniform is not None else self.bank.groups(indices)
        q, k, v = self.bank.qkv(hidden, groups, uniform)
        d = self.config.head_dim
        q = q.view(b, s, -1, d).transpose(1, 2)
        k = k.view(b, s, -1, d).transpose(1, 2)
        v = v.view(b, s, -1, d).transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k, *position_embeddings)
        if cache is not None:
            k, v = cache.update(k, v, step)
        attended = attend(q, k, v, mask, self.config._attn_implementation,
                         self.config.attention_dropout if self.training else 0.0)
        output = self.bank.output(attended.reshape(b, s, -1), hidden, groups, uniform)
        if selected is not None:
            output = selected_probability_st(output, hidden, selected)
        return output, indices, probabilities


class RmtModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.cell = RmtRecurrentCell(config)
        self.norm = Qwen3RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config=config)
        self.gradient_checkpointing = False

    def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                past_key_values=None, inputs_embeds=None, use_cache=False,
                output_hidden_states=False, cache_position=None, segment_ids=None,
                forced_routes=None, routing_mode=None, output_router_trace=False):
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids and inputs_embeds")
        hidden = self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        b, s, _ = hidden.shape
        if self.training and (use_cache or past_key_values is not None):
            raise ValueError("Mutable inference cache is disabled during training")
        if past_key_values is not None and not isinstance(past_key_values, Cache):
            raise TypeError("Use an HF Cache, not a legacy tuple")
        if past_key_values is not None and not use_cache:
            raise ValueError("past_key_values requires use_cache=True")
        cache = (RmtCache() if past_key_values is None else past_key_values) if use_cache else None
        past = 0 if cache is None else cache.get_seq_length()
        if segment_ids is not None and use_cache:
            raise ValueError("Packed inference caching is not supported")
        if position_ids is None:
            if segment_ids is not None:
                starts = torch.ones_like(segment_ids, dtype=torch.bool)
                starts[:, 1:] = segment_ids[:, 1:] != segment_ids[:, :-1]
                positions = torch.arange(s, device=hidden.device).expand(b, s)
                last_start = torch.where(starts, positions, 0).cummax(-1).values
                position_ids = positions - last_start
            elif attention_mask is not None:
                position_ids = (attention_mask.long().cumsum(-1) - 1).clamp_min(0)[:, -s:]
            else:
                position_ids = torch.arange(past, past + s, device=hidden.device).unsqueeze(0)
        mask = causal_mask(hidden, attention_mask, past, segment_ids)
        pos = self.rotary_emb(hidden, position_ids)
        mode = routing_mode or self.config.routing_mode
        if mode not in ("layer_order", "forced", "learned"):
            raise ValueError("Invalid routing_mode")
        if mode == "forced":
            if forced_routes is None or forced_routes.shape != (b, s, self.config.num_recurrences):
                raise ValueError("forced_routes must have shape [batch, sequence, recurrence]")
            if forced_routes.dtype != torch.long or (forced_routes < 0).any() or (forced_routes >= self.config.num_experts).any():
                raise ValueError("forced_routes contains invalid expert indices")
        states, routes, probabilities = [], [], []
        for step in range(self.config.num_recurrences):
            if output_hidden_states:
                states.append(hidden)
            call = partial(self.cell, step=step, mask=mask, position_embeddings=pos,
                           mode=mode, forced=forced_routes, cache=cache)
            if self.gradient_checkpointing and self.training:
                hidden, indices, probs = checkpoint(call, hidden, use_reentrant=False)
            else:
                hidden, indices, probs = call(hidden)
            if output_router_trace:
                routes.append(indices)
                probabilities.append(probs)
        hidden = self.norm(hidden)
        if output_hidden_states:
            states.append(hidden)
        return hidden, cache, tuple(states) if output_hidden_states else None, tuple(routes), tuple(probabilities)


class RmtForCausalLM(Qwen3PreTrainedModel, GenerationMixin):
    config_class = RmtConfig
    _tied_weights_keys = ["lm_head.weight"]
    _no_split_modules = ["RmtRecurrentCell"]
    _supports_cache_class = True
    _supports_static_cache = False
    _supports_sdpa = True
    _supports_flash_attn_2 = False

    def __init__(self, config):
        super().__init__(config)
        self.model = RmtModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()
        with torch.no_grad():
            self.model.cell.router.weight.zero_()
            self.model.cell.router.step_bias.zero_()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, value):
        self.lm_head = value

    def get_decoder(self):
        return self.model

    def set_decoder(self, value):
        self.model = value

    def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                past_key_values=None, inputs_embeds=None, labels=None, use_cache=None,
                output_attentions=False, output_hidden_states=False, return_dict=True,
                cache_position=None, logits_to_keep=0, segment_ids=None, forced_routes=None,
                routing_mode=None, output_router_trace=False, return_hidden_only=False, **kwargs):
        if output_attentions:
            raise ValueError("Attention weight materialization is not exposed")
        if use_cache is None:
            use_cache = self.config.use_cache and not self.training
        hidden, cache, states, routes, probs = self.model(
            input_ids, attention_mask, position_ids, past_key_values, inputs_embeds, use_cache,
            output_hidden_states, cache_position, segment_ids, forced_routes, routing_mode, output_router_trace)
        loss = None
        logits = None
        if not return_hidden_only:
            if labels is not None and logits_to_keep != 0:
                raise ValueError("Training labels require all sequence logits")
            selected = hidden[:, -logits_to_keep:, :] if isinstance(logits_to_keep, int) else hidden[:, logits_to_keep, :]
            logits = self.lm_head(selected)
            if labels is not None:
                targets = labels[:, 1:].clone()
                if attention_mask is not None:
                    targets.masked_fill_(~(attention_mask[:, 1:].bool() & attention_mask[:, :-1].bool()), -100)
                if segment_ids is not None:
                    targets.masked_fill_((segment_ids[:, 1:] != segment_ids[:, :-1]) | (segment_ids[:, 1:] < 0), -100)
                if (targets != -100).any():
                    loss = F.cross_entropy(logits[:, :-1].float().reshape(-1, self.config.vocab_size), targets.reshape(-1))
                else:
                    loss = logits.sum() * 0
        output = RmtCausalLMOutput(loss=loss, logits=logits, past_key_values=cache,
                                  hidden_states=states, last_hidden_state=hidden,
                                  router_indices=routes if output_router_trace else None,
                                  router_probabilities=probs if output_router_trace else None)
        return output if return_dict else output.to_tuple()

    def generate(self, inputs=None, **kwargs):
        # HF decoder-only generation uses the last prompt column. Canonicalize
        # right padding to left padding before entering the standard HF loop.
        ids = kwargs.get("input_ids", inputs)
        mask = kwargs.get("attention_mask")
        if ids is not None and mask is not None and kwargs.get("past_key_values") is None:
            if ids.ndim != 2 or mask.shape != ids.shape or (mask.sum(-1) == 0).any():
                raise ValueError("Generation requires a nonempty prompt per row")
            if not mask[:, -1].bool().all():
                if kwargs.get("position_ids") is not None or kwargs.get("forced_routes") is not None:
                    raise ValueError("Use left padding with explicit positions or forced routing")
                # Stable partition: padding first, valid tokens retain their order.
                order = torch.argsort(mask.long(), dim=-1, stable=True)
                ids = ids.gather(1, order)
                kwargs["attention_mask"] = mask.gather(1, order)
                if "input_ids" in kwargs:
                    kwargs["input_ids"] = ids
                else:
                    inputs = ids
        return super().generate(inputs=inputs, **kwargs)
