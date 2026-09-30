"""Bound expert parameters, with regrouping only around token-local operations."""
import torch
from torch import nn
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer


# A low-level library registration avoids custom_op's Python alias-checking
# wrapper on every norm call. The schema is functional and the kernel allocates
# a fresh tensor; keep the library alive for the lifetime of this module.
_VARIANCE_LIBRARY = torch.library.Library("rmt", "FRAGMENT")
if not hasattr(torch.ops.rmt, "reference_variance"):
    _VARIANCE_LIBRARY.define("reference_variance(Tensor x) -> Tensor")

    def _reference_variance(x):
        return x.float().pow(2).mean(-1, keepdim=True)

    _VARIANCE_LIBRARY.impl("reference_variance", _reference_variance, "CUDA")
    _VARIANCE_LIBRARY.impl("reference_variance", _reference_variance, "CPU")

    @torch.library.register_fake("rmt::reference_variance")
    def _reference_variance_fake(x):
        return x.new_empty((*x.shape[:-1], 1), dtype=torch.float32)

reference_variance = torch.ops.rmt.reference_variance.default


def inference_norm(norm, x):
    if torch.compiler.is_compiling() and not torch.is_grad_enabled():
        variance = reference_variance(x)
        h = x.float() * torch.rsqrt(variance + norm.variance_epsilon)
        return norm.weight * h.to(x.dtype)
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

    def compile_projections(self, training=False):
        # RMS epsilon is a configuration constant. Avoid CPU scalar tensors and
        # device-copy partitions inside otherwise capturable projection graphs.
        torch._dynamo.config.specialize_float = True
        # Compile the numerical kernels, leaving variable-length dispatch in Python.
        # Disable addmm pattern fusion: on regrouped 2-D BF16 tensors it changes
        # the rounding of residual additions, even with emulate_precision_casts.
        # Shared functions avoid constructing one compiled graph wrapper per expert.
        self._project_qkv = torch.compile(project_qkv, dynamic=True, fullgraph=True, options={"emulate_precision_casts": True, "pattern_matcher": False, "triton.cudagraphs": not training})
        self._project_output = torch.compile(project_output, dynamic=True, fullgraph=True, options={"emulate_precision_casts": True, "pattern_matcher": False, "triton.cudagraphs": not training})
        prefill_options = {"emulate_precision_casts": True, "pattern_matcher": False}
        # Arbitrary prompt lengths must not create an unbounded CUDA-graph pool.
        # Prefill still uses Inductor; replay is reserved for small decode shapes.
        self._prefill_qkv = torch.compile(project_qkv, dynamic=True, fullgraph=True, options=prefill_options)
        self._prefill_output = torch.compile(project_output, dynamic=True, fullgraph=True, options=prefill_options)
        self._prefill_mlp = torch.compile(project_mlp, dynamic=True, fullgraph=True, options=prefill_options)
        self._mlp = torch.compile(project_mlp, dynamic=True, fullgraph=True,
            options={"emulate_precision_casts": True, "pattern_matcher": False, "triton.cudagraphs": not training})
        self.compiled = True

    def mixed_qkv(self, expert, x):
        # Norm reductions stay eager for numerical parity. Compiling only the
        # remaining three GEMMs adds wrapper overhead without fusion benefits.
        return project_qkv(expert, x)

    def mixed_output(self, expert, attended, residual):
        if not self.compiled:
            return project_output(expert, attended, residual)
        h = residual + expert.self_attn.o_proj(attended)
        mlp = self._mlp if h.shape[0] <= 8 else self._prefill_mlp
        return h + mlp(expert, expert.post_attention_layernorm(h))

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
            kernel = self._prefill_qkv if self.compiled and hidden.shape[-2] > 1 else self._project_qkv
            return kernel(self.experts[uniform_expert], hidden)
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
            kernel = self._prefill_output if self.compiled and hidden.shape[-2] > 1 else self._project_output
            return kernel(self.experts[uniform_expert], attended, hidden)
        flat = hidden.reshape(-1, hidden.shape[-1])
        attn_flat = attended.reshape(-1, attended.shape[-1])
        result = torch.zeros_like(flat)
        for e, positions in groups:
            if positions.numel() == 0:
                continue
            value = self.mixed_output(self.experts[e], attn_flat.index_select(0, positions), flat.index_select(0, positions))
            result = result.index_copy(0, positions, value)
        return result.view_as(hidden)
