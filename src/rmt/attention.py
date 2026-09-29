"""Global causal GQA: expert grouping must never partition sequence attention."""
import torch
from torch.nn import functional as F
from transformers.models.qwen3.modeling_qwen3 import repeat_kv


def causal_mask(hidden, attention_mask, past_length=0, segment_ids=None):
    b, q, _ = hidden.shape
    k = past_length + q
    qp = torch.arange(past_length, k, device=hidden.device)
    kp = torch.arange(k, device=hidden.device)
    allowed = (kp[None, :] <= qp[:, None]).expand(b, q, k)
    if attention_mask is not None:
        if attention_mask.ndim != 2 or attention_mask.shape != (b, k):
            raise ValueError("attention_mask must include all cached and current token positions")
        allowed = allowed & attention_mask[:, None, :].bool()
    if segment_ids is not None:
        if past_length:
            raise ValueError("Packed segment_ids with inference cache are unsupported")
        if segment_ids.shape != (b, q):
            raise ValueError("segment_ids must match the input shape")
        allowed = allowed & (segment_ids[:, :, None] == segment_ids[:, None, :])
        allowed = allowed & (segment_ids[:, None, :] >= 0)
    mask = torch.zeros((b, 1, q, k), device=hidden.device, dtype=hidden.dtype)
    return mask.masked_fill(~allowed[:, None], torch.finfo(hidden.dtype).min)


def attend(q, k, v, mask, implementation="eager", dropout=0.0):
    k = repeat_kv(k, q.shape[1] // k.shape[1])
    v = repeat_kv(v, q.shape[1] // v.shape[1])
    if implementation == "sdpa":
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=dropout,
                                             is_causal=False, scale=q.shape[-1] ** -0.5)
    elif implementation == "eager":
        scores = (q @ k.transpose(-1, -2)) * (q.shape[-1] ** -0.5)
        weights = (scores + mask).softmax(-1, dtype=torch.float32).to(q.dtype)
        out = F.dropout(weights, dropout, training=dropout > 0) @ v
    else:
        raise ValueError(f"Unsupported attention implementation: {implementation}")
    return out.transpose(1, 2).contiguous()
