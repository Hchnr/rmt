"""Import rmt before AutoModel loading to register the local HF implementation."""
from transformers import AutoConfig, AutoModelForCausalLM
from .configuration_rmt import RmtConfig
from .modeling_rmt import RmtForCausalLM

AutoConfig.register(RmtConfig.model_type, RmtConfig, exist_ok=True)
AutoModelForCausalLM.register(RmtConfig, RmtForCausalLM, exist_ok=True)
__all__ = ["RmtConfig", "RmtForCausalLM"]
