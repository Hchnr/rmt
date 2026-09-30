"""Recurrent Qwen3: token-local expert projections with global causal attention."""
from dataclasses import dataclass
from functools import partial
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from transformers import GenerationMixin
from transformers.cache_utils import DynamicCache
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3PreTrainedModel, Qwen3RMSNorm, Qwen3RotaryEmbedding, apply_rotary_pos_emb,
)
from .configuration_rmt import RmtConfig
from .routing import BoundRouter, selected_probability_st
from .experts import BoundExpertBank
from .attention import causal_mask, attend
from .cache import RmtCache
from .halting import prior_expert, hidden_stable, probability_stable


@dataclass
class RmtCausalLMOutput(CausalLMOutputWithPast):
    last_hidden_state: Optional[torch.Tensor] = None
    router_indices: Optional[tuple] = None
    router_probabilities: Optional[tuple] = None
    ce_loss: Optional[torch.Tensor] = None
    kd_loss: Optional[torch.Tensor] = None
    exit_depths: Optional[torch.Tensor] = None
    exit_reasons: Optional[torch.Tensor] = None


class RmtRecurrentCell(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.bank = BoundExpertBank(config)
        self.router = BoundRouter(config)

    def forward(self, hidden, step, mask, position_embeddings, mode,
                forced=None, cache=None, active=None, previous_k=None, previous_v=None):
        b, s, _ = hidden.shape
        if active is not None and previous_k is not None and not bool(active.any()):
            if cache is not None:
                cache.update(previous_k, previous_v, step)
            return hidden, torch.full((b,s), -1, dtype=torch.long, device=hidden.device), hidden.new_empty(0, dtype=torch.float32), previous_k, previous_v
        uniform = None
        probabilities = hidden.new_empty(0, dtype=torch.float32)
        selected = None
        if mode == "layer_order":
            uniform = prior_expert(self.config, step)
            indices = torch.full((b, s), uniform, dtype=torch.long, device=hidden.device)
        elif mode == "forced":
            indices = forced[..., step]
        else:
            indices, selected, probabilities = self.router(hidden, step, prior_expert(self.config, step))
        if active is not None:
            indices = torch.where(active, indices, -1)
            if not bool(active.all()):
                uniform = None
        groups = None if uniform is not None else self.bank.groups(indices)
        if groups is not None:
            groups = [(e, positions) for e, positions in groups if e >= 0]
        q, k, v = self.bank.qkv(hidden, groups, uniform)
        d = self.config.head_dim
        q = q.view(b, s, -1, d).transpose(1, 2)
        k = k.view(b, s, -1, d).transpose(1, 2)
        v = v.view(b, s, -1, d).transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k, *position_embeddings)
        if active is not None and previous_k is not None:
            keep = active[:, None, :, None]
            k = torch.where(keep, k, previous_k)
            v = torch.where(keep, v, previous_v)
        if active is not None and self.bank.compiled:
            k, v = k.clone(), v.clone()
        local_k, local_v = k, v
        if cache is not None:
            k, v = cache.update(k, v, step)
        attended = attend(q, k, v, mask, self.config._attn_implementation,
                         self.config.attention_dropout if self.training else 0.0)
        output = self.bank.output(attended.reshape(b, s, -1), hidden, groups, uniform)
        if selected is not None:
            output = selected_probability_st(output, hidden, selected)
        if active is not None:
            output = torch.where(active[..., None], output, hidden)
            return output, indices, probabilities, local_k, local_v
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
                forced_routes=None, routing_mode=None, output_router_trace=False,
                recurrence_limit=None, halting_policy=None, forced_exit_depths=None, head_weight=None):
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids and inputs_embeds")
        hidden = self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        if self.cell.bank.compiled and not torch.is_grad_enabled():
            if inputs_embeds is not None:
                hidden = hidden.clone()
            torch.compiler.cudagraph_mark_step_begin()
        b, s, _ = hidden.shape
        if self.training and (use_cache or past_key_values is not None):
            raise ValueError("Mutable inference cache is disabled during training")
        if past_key_values is not None and not isinstance(past_key_values, DynamicCache):
            raise TypeError("Only append-only HF DynamicCache is supported")
        if past_key_values is not None and not use_cache:
            raise ValueError("past_key_values requires use_cache=True")
        cache = (RmtCache() if past_key_values is None else past_key_values) if use_cache else None
        past = 0 if cache is None else cache.get_seq_length()
        if cache_position is not None:
            expected = torch.arange(past, past + s, device=hidden.device)
            if cache_position.shape != expected.shape or not torch.equal(cache_position.to(hidden.device), expected):
                raise ValueError("cache_position must append contiguous physical positions")
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
        policy = self.config.halting_policy if halting_policy is None else halting_policy
        if policy not in ("fixed", "hidden", "probability", "hybrid"):
            raise ValueError("Unknown halting policy")
        limit = (self.config.num_recurrences if policy == "fixed" else self.config.max_recurrences) if recurrence_limit is None else recurrence_limit
        if not isinstance(limit, int) or not 1 <= limit <= self.config.max_recurrences:
            raise ValueError("recurrence_limit exceeds configured bounds")
        if policy != "fixed" and limit < self.config.min_recurrences:
            raise ValueError("Dynamic recurrence limit is below minimum")
        if cache is not None and past and any(cache.get_seq_length(i) != past for i in range(limit)):
            raise ValueError("Incomplete recurrence cache history; reset cache before changing fixed depth")
        dynamic = policy != "fixed" or forced_exit_depths is not None
        if forced_exit_depths is not None:
            if forced_exit_depths.shape != (b,s) or forced_exit_depths.dtype != torch.long or (forced_exit_depths < 1).any() or (forced_exit_depths > limit).any():
                raise ValueError("forced_exit_depths must be integer [batch, sequence] within recurrence limit")
        if mode == "forced":
            if forced_routes is None or forced_routes.shape != (b, s, limit):
                raise ValueError("forced_routes must have shape [batch, sequence, recurrence]")
            if forced_routes.dtype != torch.long or (forced_routes < 0).any() or (forced_routes >= self.config.num_experts).any():
                raise ValueError("forced_routes contains invalid expert indices")
        states, routes, probabilities = [], [], []
        valid = torch.ones((b,s), dtype=torch.bool, device=hidden.device)
        if attention_mask is not None:
            valid = valid & attention_mask[:, -s:].bool()
        if segment_ids is not None:
            valid = valid & (segment_ids >= 0)
        depths = torch.zeros((b,s), dtype=torch.long, device=hidden.device)
        reasons = torch.zeros_like(depths)  # 0 padding, 1 convergence, 2 cap, 3 forced
        active = valid
        streak = torch.zeros_like(depths)
        h_streak = torch.zeros_like(depths)
        p_previous = hidden.detach()
        p_ready = torch.zeros_like(valid)
        previous_k = previous_v = None
        for step in range(limit):
            if output_hidden_states:
                states.append(hidden.clone() if self.cell.bank.compiled else hidden)
            call = partial(self.cell, step=step, mask=mask, position_embeddings=pos,
                           mode=mode, forced=forced_routes, cache=cache)
            if dynamic:
                call = partial(call, active=active, previous_k=previous_k, previous_v=previous_v)
            previous = hidden
            values = checkpoint(call, hidden, use_reentrant=False) if self.gradient_checkpointing and self.training else call(hidden)
            hidden, indices, probs = values[:3]
            if dynamic:
                previous_k, previous_v = values[3:]
                # The decision is detached, never the carried hidden/KV graph.
                hs = hidden_stable(previous, hidden, self.config)
                h_streak = torch.where(active & hs, h_streak + 1, 0)
                stable = hs
                if policy == "probability":
                    stable = probability_stable(previous, hidden, active, self.norm, head_weight,
                                                self.config.halt_probability_threshold)
                elif policy == "hybrid":
                    eligible = active & (h_streak >= self.config.halt_patience)
                    stable = probability_stable(p_previous, hidden, eligible & p_ready, self.norm,
                                                head_weight, self.config.halt_probability_threshold)
                    p_previous = torch.where(eligible[...,None], hidden.detach(), p_previous)
                    p_ready = eligible
                streak = torch.where(active & stable, streak + 1, 0)
                stop = active & (streak >= self.config.halt_patience) & (step+1 >= self.config.min_recurrences)
                reason = 1
                if forced_exit_depths is not None:
                    stop = active & (forced_exit_depths <= step+1)
                    reason = 3
                if step+1 == limit:
                    reasons = torch.where(active & ~stop, 2, reasons)
                    stop = active
                depths = torch.where(stop, step+1, depths)
                reasons = torch.where(stop & (reasons == 0), reason, reasons)
                active = active & ~stop
            if output_router_trace:
                routes.append(indices)
                probabilities.append(probs)
        if not dynamic:
            depths = torch.where(valid, limit, 0)
            reasons = torch.where(valid, 2, 0)
        hidden = self.norm(hidden)
        if output_hidden_states:
            states.append(hidden.clone() if self.cell.bank.compiled else hidden)
        return hidden, cache, tuple(states) if output_hidden_states else None, tuple(routes), tuple(probabilities), depths, reasons


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
        if config._attn_implementation not in ("eager", "sdpa"):
            raise ValueError("Bootstrap supports eager and SDPA attention only")
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
                routing_mode=None, output_router_trace=False, return_hidden_only=False,
                loss_chunk_size=0, teacher_hidden_states=None, teacher_head_weight=None,
                kd_weight=0.0, ce_weight=1.0, temperature=1.0,
                recurrence_limit=None, halting_policy=None, forced_exit_depths=None, **kwargs):
        if output_attentions:
            raise ValueError("Attention weight materialization is not exposed")
        if use_cache is None:
            use_cache = self.config.use_cache and not self.training
        hidden, cache, states, routes, probs, depths, reasons = self.model(
            input_ids, attention_mask, position_ids, past_key_values, inputs_embeds, use_cache,
            output_hidden_states, cache_position, segment_ids, forced_routes, routing_mode, output_router_trace,
            recurrence_limit, halting_policy, forced_exit_depths, self.lm_head.weight)
        loss = None
        ce_loss = kd_loss = None
        logits = None
        if loss_chunk_size:
            if labels is None:
                raise ValueError("Chunked loss requires labels")
            from .losses import distillation_loss
            loss, ce_loss, kd_loss = distillation_loss(
                hidden, self.lm_head.weight, labels, attention_mask, segment_ids,
                teacher_hidden_states, teacher_head_weight, loss_chunk_size,
                temperature, ce_weight, kd_weight)
        elif not return_hidden_only:
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
        if self.training and loss is not None:
            # Conditional token dispatch can leave parameters unused on one rank.
            # FSDP2 needs a local zero gradient for every shared collective member;
            # otherwise a rank may retain None despite another rank using it.
            # Define optimizer semantics consistently: unused parameters get zero
            # gradients (so momentum/weight decay still apply), on every backend.
            zero = sum(p.reshape(-1)[0] * 0 for p in self.parameters() if p.requires_grad)
            loss = loss + zero.to(loss.dtype)
        output = RmtCausalLMOutput(loss=loss, logits=logits, past_key_values=cache,
                                  hidden_states=states, last_hidden_state=hidden,
                                  ce_loss=ce_loss, kd_loss=kd_loss, exit_depths=depths, exit_reasons=reasons,
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
