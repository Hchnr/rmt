"""One causal choice per token and recurrence, shared by all seven projections."""
import torch
from torch import nn
from torch.nn import functional as F


class BoundRouter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(config.num_experts, config.hidden_size))
        self.step_bias = nn.Parameter(torch.zeros(config.num_recurrences, config.num_experts))
        self.num_experts = config.num_experts
        self.prior_strength = float(config.router_prior_strength)

    def forward(self, hidden, step):
        x = hidden.float()
        x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)
        scores = F.linear(x, self.weight.float()) + self.step_bias[step].float()
        prior = F.one_hot(torch.tensor(step % self.num_experts, device=x.device), self.num_experts)
        probabilities = (scores + self.prior_strength * prior).softmax(-1)
        indices = probabilities.argmax(-1)
        selected = probabilities.gather(-1, indices.unsqueeze(-1))
        return indices, selected, probabilities


def selected_probability_st(output, residual, selected_probability):
    """Biased surrogate: exactly hard forward; selected-probability task gradient."""
    correction = (selected_probability - selected_probability.detach()).to(output.dtype)
    return output + correction * (output - residual)
