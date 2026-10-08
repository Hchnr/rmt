"""Isolate compiled RMSNorm rounding without changing the running trainer's code."""
import runpy
import sys
import torch
import rmt.experts as experts

library=torch.library.Library('rmt_v005_probe','DEF')
library.define('norm(Tensor x, Tensor weight, float eps) -> Tensor')

def reference_norm(x,weight,eps):
    h=x.float()
    h=h*torch.rsqrt(h.pow(2).mean(-1,keepdim=True)+eps)
    return weight*h.to(x.dtype)

library.impl('norm',reference_norm,'CUDA')
library.impl('norm',reference_norm,'CPU')

@torch.library.register_fake('rmt_v005_probe::norm')
def fake_norm(x,weight,eps):
    return x.new_empty(x.shape,dtype=torch.promote_types(x.dtype,weight.dtype))

def inference_norm(norm,x):
    if torch.compiler.is_compiling() and not torch.is_grad_enabled():
        return torch.ops.rmt_v005_probe.norm(x,norm.weight,norm.variance_epsilon)
    return norm(x)

experts.inference_norm=inference_norm
sys.argv=['verify_06b_compiled_decode.py','--output','reports/v0.0.5/compiled_decode_reference_norm.json']
runpy.run_path('scripts/verify_06b_compiled_decode.py')['main']()
