# Bound RMT

基于 Qwen3 的循环 Transformer 实验项目。将原始 Transformer 层作为完整绑定的专家，由同一个 recurrent cell 在多个循环步中调用；每个 token 在每一步选择一个专家，注意力仍覆盖因果范围内的历史 token。

## 当前实现

- 支持 `layer_order`（按原层顺序）、`forced`（指定路径）和 `learned`（可学习 top-1）三种路由。
- 支持按循环步保存 KV cache、文本生成、独立文档 packing 和梯度检查点。
- 支持将本地 Qwen3 safetensors 权重转换为 RMT，并导出 Hugging Face 格式。
- 已编写小模型测试，覆盖固定路径前向／反向一致性、因果性、packing、cache、路由梯度和模型导出／加载。

目前处于 bootstrap 阶段，仅支持完整 block 绑定和 top-1 路由。验证进展见[实施记录](plans/v0.0.2_bound_rmt_bootstrap/record.md)。

## 开发环境与使用

主开发目录：`/share/project/eai_pwm/home/hcr/repos/test/rmt`。
当前仓库的 `.venv` 软链接指向主开发目录的 `.venv`，共用同一套依赖。
以下命令在主开发目录执行，首次使用前需完成环境安装。

依赖要求：Python 3.11–3.13、PyTorch 2.7.1、Transformers 4.51.3，完整依赖见 [pyproject.toml](pyproject.toml)。

```bash
cd /share/project/eai_pwm/home/hcr/repos/test/rmt
source .venv/bin/activate

# 运行小模型测试
PYTHONPATH=src python -m pytest -q

# 转换本地 Qwen3 权重，保存模型及 conversion.json 映射报告
PYTHONPATH=src python -m rmt.checkpoint \
  --base-model /share/project/eai_pwm/models/Qwen/Qwen3-4B \
  --output artifacts/qwen3-4b-rmt
```

主开发目录中的 `src/rmt/` 包含模型、路由、注意力、cache 和权重转换实现，`tests/` 保存测试；设计与实施计划见 [plans/](plans/)。
