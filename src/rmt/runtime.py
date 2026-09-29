"""Run metadata and this workspace's explicit physical GPU allocation."""
import json
import os
from pathlib import Path


def enforce_gpu_scope():
    visible = os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7")
    if visible and any(x.strip() not in {str(i) for i in range(8)} for x in visible.split(",")):
        raise RuntimeError("Only physical GPUs 0–7 are authorized in this bootstrap run")
    return visible


def write_report(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def versions():
    import platform
    import torch
    import transformers
    return {"python": platform.python_version(), "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda, "transformers": transformers.__version__,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "tf32_override": os.environ.get("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"),
            "nvidia_tf32_override": os.environ.get("NVIDIA_TF32_OVERRIDE")}
