"""Test opaque reference projections under Inductor/CUDA graphs for exact rounding."""
import runpy
import sys
import torch
from torch.nn import functional as F
import rmt.experts as experts

library=torch.library.Library('rmt_v005_projection_probe','DEF')
library.define('qkv(Tensor x, Tensor nw, Tensor qw, Tensor kw, Tensor vw, Tensor qn, Tensor kn, int d, float eps) -> (Tensor, Tensor, Tensor)')
library.define('output(Tensor a, Tensor h, Tensor ow, Tensor nw, Tensor gw, Tensor uw, Tensor dw, float eps) -> Tensor')

def norm(x,w,eps):
    h=x.float();h=h*torch.rsqrt(h.pow(2).mean(-1,keepdim=True)+eps)
    return w*h.to(x.dtype)

def qkv(x,nw,qw,kw,vw,qn,kn,d,eps):
    y=norm(x,nw,eps)
    q=norm(F.linear(y,qw).view(*x.shape[:-1],-1,d),qn,eps)
    k=norm(F.linear(y,kw).view(*x.shape[:-1],-1,d),kn,eps)
    return q.flatten(-2),k.flatten(-2),F.linear(y,vw)

def output(a,h,ow,nw,gw,uw,dw,eps):
    h=h+F.linear(a,ow);n=norm(h,nw,eps)
    return h+F.linear(F.silu(F.linear(n,gw))*F.linear(n,uw),dw)

for device in ('CPU','CUDA'):
    library.impl('qkv',qkv,device);library.impl('output',output,device)

@torch.library.register_fake('rmt_v005_projection_probe::qkv')
def fake_qkv(x,nw,qw,kw,vw,qn,kn,d,eps):
    return tuple(x.new_empty((*x.shape[:-1],w.shape[0])) for w in (qw,kw,vw))

@torch.library.register_fake('rmt_v005_projection_probe::output')
def fake_output(a,h,ow,nw,gw,uw,dw,eps):return torch.empty_like(h)

original_qkv=experts.project_qkv;original_output=experts.project_output

def project_qkv(expert,x):
    if torch.compiler.is_compiling() and not torch.is_grad_enabled():
        a=expert.self_attn
        return torch.ops.rmt_v005_projection_probe.qkv(x,expert.input_layernorm.weight,
            a.q_proj.weight,a.k_proj.weight,a.v_proj.weight,a.q_norm.weight,a.k_norm.weight,
            a.head_dim,expert.input_layernorm.variance_epsilon)
    return original_qkv(expert,x)

def project_output(expert,a,h):
    if torch.compiler.is_compiling() and not torch.is_grad_enabled():
        m=expert.mlp
        return torch.ops.rmt_v005_projection_probe.output(a,h,expert.self_attn.o_proj.weight,
            expert.post_attention_layernorm.weight,m.gate_proj.weight,m.up_proj.weight,
            m.down_proj.weight,expert.post_attention_layernorm.variance_epsilon)
    return original_output(expert,a,h)

experts.project_qkv=project_qkv;experts.project_output=project_output
sys.argv=['verify_06b_compiled_decode.py','--output','reports/v0.0.5/compiled_decode_reference_projections.json']
runpy.run_path('scripts/verify_06b_compiled_decode.py')['main']()
