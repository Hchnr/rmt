"""Bound expert parameters, with regrouping only around token-local operations."""
import torch
from torch import nn
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer


def project_qkv(expert, x):
    attn = expert.self_attn
    y = expert.input_layernorm(x)
    q = attn.q_norm(attn.q_proj(y).view(*x.shape[:-1], -1, attn.head_dim))
    k = attn.k_norm(attn.k_proj(y).view(*x.shape[:-1], -1, attn.head_dim))
    v = attn.v_proj(y).view(*x.shape[:-1], -1, attn.head_dim)
    return q.flatten(-2), k.flatten(-2), v.flatten(-2)


def project_output(expert, attended, residual):
    h = residual + expert.self_attn.o_proj(attended)
    return h + expert.mlp(expert.post_attention_layernorm(h))


class BoundExpertBank(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Decoder layers are parameter containers only; their HF cache index
        # is unused. Cache depth belongs to the recurrent controller, so E and
        # R remain independent even when the expert count exceeds recurrence.
        self.experts = nn.ModuleList([Qwen3DecoderLayer(config, 0) for _ in range(config.num_experts)])
        self.q_width = config.num_attention_heads * config.head_dim
        self.kv_width = config.num_key_value_heads * config.head_dim
        self._project_qkv = project_qkv
        self._project_output = project_output
        self.compiled = False

    def compile_projections(self):
        # Compile the numerical kernels, leaving variable-length dispatch in Python.
        # Shared functions avoid constructing one compiled graph wrapper per expert.
        self._project_qkv = torch.compile(project_qkv, dynamic=True, fullgraph=True)
        self._project_output = torch.compile(project_output, dynamic=True, fullgraph=True)
        self.compiled = True

    def groups(self, indices):
        flat = indices.reshape(-1)
        return [(e, (flat == e).nonzero(as_tuple=True)[0]) for e in range(len(self.experts))]

    def qkv(self, hidden, groups, uniform_expert=None):
        if uniform_expert is not None:
            return self._project_qkv(self.experts[uniform_expert], hidden)
        flat = hidden.reshape(-1, hidden.shape[-1])
        outputs = [flat.new_zeros((flat.shape[0], width)) for width in (self.q_width, self.kv_width, self.kv_width)]
        for e, positions in groups:
            if positions.numel() == 0:
                continue
            values = self._project_qkv(self.experts[e], flat.index_select(0, positions))
            outputs = [out.index_copy(0, positions, value) for out, value in zip(outputs, values)]
        return tuple(out.view(*hidden.shape[:-1], -1) for out in outputs)

    def output(self, attended, hidden, groups, uniform_expert=None):
        if uniform_expert is not None:
            return self._project_output(self.experts[uniform_expert], attended, hidden)
        flat = hidden.reshape(-1, hidden.shape[-1])
        attn_flat = attended.reshape(-1, attended.shape[-1])
        result = torch.zeros_like(flat)
        for e, positions in groups:
            if positions.numel() == 0:
                continue
            value = self._project_output(self.experts[e], attn_flat.index_select(0, positions), flat.index_select(0, positions))
            result = result.index_copy(0, positions, value)
        return result.view_as(hidden)
