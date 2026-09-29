"""Configuration for recurrent, fully bound Qwen3 experts."""
from transformers import Qwen3Config


class RmtConfig(Qwen3Config):
    model_type = "bound_rmt"

    def __init__(self, num_experts=None, num_recurrences=None, binding_mode="full_block",
                 routing_mode="layer_order", router_top_k=1, router_prior_strength=4.0,
                 router_gradient="selected_probability_st", norm_policy="expert_bound",
                 cache_layout_version=1, **kwargs):
        depth = kwargs.pop("num_hidden_layers", 36)
        self.num_experts = depth if num_experts is None else num_experts
        self.num_recurrences = depth if num_recurrences is None else num_recurrences
        if self.num_experts < 1 or self.num_recurrences < 1:
            raise ValueError("Expert count and recurrence count must be positive")
        if binding_mode != "full_block" or router_top_k != 1:
            raise ValueError("Bootstrap supports full_block binding and top-1 only")
        if routing_mode not in ("layer_order", "forced", "learned"):
            raise ValueError(f"Unknown routing mode: {routing_mode}")
        if router_gradient != "selected_probability_st" or norm_policy != "expert_bound":
            raise ValueError("Unsupported router gradient or norm policy")
        if cache_layout_version != 1:
            raise ValueError("Unsupported cache layout")
        self.binding_mode = binding_mode
        self.routing_mode = routing_mode
        self.router_top_k = router_top_k
        self.router_prior_strength = router_prior_strength
        self.router_gradient = router_gradient
        self.norm_policy = norm_policy
        self.cache_layout_version = cache_layout_version
        layer_types = kwargs.pop("layer_types", None)
        if layer_types is not None and any(t != "full_attention" for t in layer_types):
            raise ValueError("Only full attention is supported")
        super().__init__(num_hidden_layers=self.num_recurrences,
                         layer_types=["full_attention"] * self.num_recurrences, **kwargs)
        if self.use_sliding_window:
            raise ValueError("Sliding-window attention is outside bootstrap scope")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("Q heads must be divisible by KV heads")
        self.architectures = ["RmtForCausalLM"]

    @classmethod
    def from_qwen(cls, config, **overrides):
        values = config.to_dict()
        values.pop("model_type", None)
        values.update(overrides)
        return cls(**values)
