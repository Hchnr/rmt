"""Bound expert parameters, with regrouping only around token-local operations."""
import torch
from torch import nn
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer


@torch.library.custom_op("rmt::reference_rms_norm", mutates_args=())
def reference_rms_norm(x: torch.Tensor, weight: torch.Tensor, epsilon: float) -> torch.Tensor:
    # An opaque inference boundary preserves the exact HF CUDA reduction and
    # BF16 casts; Triton reduction order can otherwise amplify across 36 steps.
    dtype = x.dtype
    h = x.float()
    variance = h.pow(2).mean(-1, keepdim=True)
    h = h * torch.rsqrt(variance + epsilon)
    return weight * h.to(dtype)


@reference_rms_norm.register_fake
def _reference_rms_norm_fake(x, weight, epsilon):
    return torch.empty_like(x, dtype=torch.promote_types(x.dtype, weight.dtype))


def inference_norm(norm, x):
    if torch.compiler.is_compiling() and not torch.is_grad_enabled():
        return reference_rms_norm(x, norm.weight, norm.variance_epsilon)
    return norm(x)


def project_qkv(expert, x):
    attn = expert.self_attn
    y = inference_norm(expert.input_layernorm, x)
    q = inference_norm(attn.q_norm, attn.q_proj(y).view(*x.shape[:-1], -1, attn.head_dim))
    k = inference_norm(attn.k_norm, attn.k_proj(y).view(*x.shape[:-1], -1, attn.head_dim))
    v = attn.v_proj(y).view(*x.shape[:-1], -1, attn.head_dim)
    return q.flatten(-2), k.flatten(-2), v.flatten(-2)


def project_output(expert, attended, residual):
    h = residual + expert.self_attn.o_proj(attended)
    return h + expert.mlp(inference_norm(expert.post_attention_layernorm, h))


def project_linear_qkv(expert, normalized):
    attn = expert.self_attn
    return attn.q_proj(normalized), attn.k_proj(normalized), attn.v_proj(normalized)


def project_mlp(expert, normalized):
    return expert.mlp(normalized)


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
        # Disable addmm pattern fusion: on regrouped 2-D BF16 tensors it changes
        # the rounding of residual additions, even with emulate_precision_casts.
        # Shared functions avoid constructing one compiled graph wrapper per expert.
        self._project_qkv = torch.compile(project_qkv, dynamic=True, fullgraph=True, options={"emulate_precision_casts": True, "pattern_matcher": False})
        self._project_output = torch.compile(project_output, dynamic=True, fullgraph=True, options={"emulate_precision_casts": True, "pattern_matcher": False})
        self._mlp = torch.compile(project_mlp, dynamic=True, fullgraph=True,
            options={"emulate_precision_casts": True, "pattern_matcher": False})
        self.compiled = True

    def mixed_qkv(self, expert, x):
        # Norm reductions stay eager for numerical parity. Compiling only the
        # remaining three GEMMs adds wrapper overhead without fusion benefits.
        return project_qkv(expert, x)

    def mixed_output(self, expert, attended, residual):
        if not self.compiled:
            return project_output(expert, attended, residual)
        h = residual + expert.self_attn.o_proj(attended)
        return h + self._mlp(expert, expert.post_attention_layernorm(h))

    def groups(self, indices):
        flat = indices.reshape(-1)
        if flat.numel() == 0:
            return []
        positions = torch.argsort(flat, stable=True)
        experts, counts = torch.unique_consecutive(flat[positions], return_counts=True)
        # One compact device-to-host transfer rather than one nonzero sync per
        # possible expert. Stable sorting preserves token order within each group.
        layout = torch.stack((experts, counts), dim=-1).tolist()
        groups = []
        start = 0
        for expert, count in layout:
            groups.append((expert, positions[start:start + count]))
            start += count
        return groups

    def qkv(self, hidden, groups, uniform_expert=None):
        if uniform_expert is not None:
            return self._project_qkv(self.experts[uniform_expert], hidden)
        flat = hidden.reshape(-1, hidden.shape[-1])
        outputs = [flat.new_zeros((flat.shape[0], width)) for width in (self.q_width, self.kv_width, self.kv_width)]
        for e, positions in groups:
            if positions.numel() == 0:
                continue
            values = self.mixed_qkv(self.experts[e], flat.index_select(0, positions))
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
            value = self.mixed_output(self.experts[e], attn_flat.index_select(0, positions), flat.index_select(0, positions))
            result = result.index_copy(0, positions, value)
        return result.view_as(hidden)
