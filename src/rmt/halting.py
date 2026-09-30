"""Fixed, causal stopping rules. No learned halting parameters or label access."""
import torch
from torch.nn import functional as F


def prior_expert(config, step):
    if config.recurrence_schedule == "tail" and step >= config.num_experts:
        return config.num_experts - config.tail_experts + (step - config.num_experts) % config.tail_experts
    return step % config.num_experts


@torch.no_grad()
def hidden_stable(previous, current, config):
    a, b = previous.float(), current.float()
    if not torch.isfinite(b).all():
        raise FloatingPointError("Nonfinite recurrent hidden state")
    ar = a.square().mean(-1).sqrt()
    br = b.square().mean(-1).sqrt()
    u = a * torch.rsqrt(a.square().mean(-1, keepdim=True) + 1e-6)
    v = b * torch.rsqrt(b.square().mean(-1, keepdim=True) + 1e-6)
    direction = (u - v).square().mean(-1).sqrt()
    relative = (a - b).square().mean(-1).sqrt() / ar.clamp_min(1e-6)
    return (direction < config.halt_threshold) & (relative < config.halt_relative_threshold)


@torch.no_grad()
def probability_stable(previous, current, eligible, norm, head_weight, threshold, chunk_size=32):
    """Exact vocabulary JS in bounded chunks; both readouts are charged as work."""
    result = torch.zeros_like(eligible)
    positions = eligible.flatten().nonzero().flatten()
    a = previous.reshape(-1, previous.shape[-1])
    b = current.reshape_as(a)
    flat = result.flatten()
    for index in positions.split(chunk_size):
        if index.numel() == 0:
            continue
        p = F.linear(norm(a[index]), head_weight).float().log_softmax(-1)
        q = F.linear(norm(b[index]), head_weight).float().log_softmax(-1)
        m = torch.logaddexp(p, q) - 0.6931471805599453
        js = 0.5 * ((p.exp() * (p-m)).sum(-1) + (q.exp() * (q-m)).sum(-1))
        if not torch.isfinite(js).all():
            raise FloatingPointError("Nonfinite halting probability divergence")
        flat[index] = js < threshold
    return result
