import pytest
from rmt.runtime import enforce_gpu_scope


def test_explicit_gpu_allowlist(monkeypatch):
    monkeypatch.setenv('RMT_ALLOWED_GPUS', '4,5,6,7')
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES', raising=False)
    assert enforce_gpu_scope() == '4,5,6,7'
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '5,7')
    assert enforce_gpu_scope() == '5,7'
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0,4')
    with pytest.raises(RuntimeError, match='exceeds'):
        enforce_gpu_scope()
