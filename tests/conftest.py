import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM
from rmt.checkpoint import from_qwen_model


@pytest.fixture
def pair():
    torch.manual_seed(17)
    torch.set_num_threads(2)
    config = Qwen3Config(vocab_size=97, hidden_size=32, intermediate_size=48,
                         num_hidden_layers=3, num_attention_heads=4,
                         num_key_value_heads=2, head_dim=8, max_position_embeddings=128,
                         attention_dropout=0.0, tie_word_embeddings=True,
                         bos_token_id=1, eos_token_id=2, pad_token_id=0)
    config._attn_implementation = "eager"
    original = Qwen3ForCausalLM(config)
    model = from_qwen_model(original)
    return original, model
