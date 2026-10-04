"""Auditable local Qwen conversion and HF export. Never modifies source weights."""
import argparse
import copy
import json
from pathlib import Path

import torch
from accelerate import init_empty_weights
from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer, GenerationConfig
from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding
from .configuration_rmt import RmtConfig
from .modeling_rmt import RmtForCausalLM


def map_key(key):
    return key.replace("model.layers.", "model.cell.bank.experts.", 1)


def from_qwen_model(source, **overrides):
    config = RmtConfig.from_qwen(source.config, **overrides)
    config._attn_implementation = source.config._attn_implementation
    model = RmtForCausalLM(config).to(device=source.device, dtype=source.dtype)
    missing, unexpected = model.load_state_dict({map_key(k): v for k, v in source.state_dict().items()}, strict=False)
    if unexpected or set(missing) != {"model.cell.router.weight", "model.cell.router.step_bias"}:
        raise ValueError(f"Unmapped parameters: {missing}, {unexpected}")
    model.tie_weights()
    return model


def load_qwen_as_rmt(path, dtype=torch.bfloat16, **overrides):
    path = Path(path)
    original = AutoConfig.from_pretrained(path, local_files_only=True)
    if original.model_type != "qwen3":
        raise ValueError("Expected a dense Qwen3 checkpoint")
    config = RmtConfig.from_qwen(original, **overrides)
    config._attn_implementation = overrides.get("attn_implementation", "eager")
    with init_empty_weights():
        model = RmtForCausalLM(config).to(dtype=dtype)
    model.to_empty(device="cpu")
    # Non-persistent RoPE buffers cannot be restored from state_dict.
    model.model.rotary_emb = Qwen3RotaryEmbedding(config=config)
    model.model.cell.router.weight.data.zero_()
    model.model.cell.router.step_bias.data.zero_()
    model.tie_weights()
    index_path = path / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text())
        shard_names = sorted(set(index["weight_map"].values()))
    else:
        index = None
        shard_names = ["model.safetensors"]
    params = dict(model.named_parameters(remove_duplicate=False))
    expected = set(params) - {"model.cell.router.weight", "model.cell.router.step_bias"}
    if config.tie_word_embeddings:
        expected.discard("lm_head.weight")
    seen, mappings = set(), []
    with torch.no_grad():
        for filename in shard_names:
            with safe_open(path / filename, framework="pt", device="cpu") as shard:
                for key in shard.keys():
                    if index and index["weight_map"].get(key) != filename:
                        raise ValueError(f"Shard index mismatch: {key}")
                    target = map_key(key)
                    if target not in params or target in seen:
                        raise ValueError(f"Unexpected or duplicate source key: {key}")
                    value = shard.get_tensor(key)
                    if tuple(value.shape) != tuple(params[target].shape):
                        raise ValueError(f"Shape mismatch: {key}")
                    params[target].copy_(value)
                    seen.add(target)
                    mappings.append({"source": key, "target": target, "shape": list(value.shape), "dtype": str(value.dtype)})
    if expected - seen:
        raise ValueError(f"Missing source weights: {expected - seen}")
    model.tie_weights()
    model.generation_config = GenerationConfig.from_pretrained(path, local_files_only=True)
    report = {"source": str(path.resolve()), "source_tensor_count": len(mappings),
              "base_parameters": sum(p.numel() for n, p in model.named_parameters() if ".router." not in n),
              "router_parameters": sum(p.numel() for n, p in model.named_parameters() if ".router." in n),
              "mapping": mappings}
    return model, report


def export_hf(model, output, source=None, generation_config=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    RmtConfig.register_for_auto_class()
    RmtForCausalLM.register_for_auto_class("AutoModelForCausalLM")
    if generation_config is not None:
        model.generation_config = copy.deepcopy(generation_config)
    model.save_pretrained(output, safe_serialization=True, max_shard_size="4GB")
    if source:
        AutoTokenizer.from_pretrained(source, local_files_only=True).save_pretrained(output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    model, report = load_qwen_as_rmt(args.base_model)
    export_hf(model, args.output, args.base_model)
    Path(args.output, "conversion.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k:v for k,v in report.items() if k != "mapping"}))


if __name__ == "__main__":
    main()
