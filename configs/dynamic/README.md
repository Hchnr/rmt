# 动态循环实验

本组配置使用 36 套绑定专家；前 36 次原层序，之后基于尾部 8 套先验开放路由。
H、P、H+P 是固定判据，不引入可学习退出头。默认最小 36、最大 48、连续两次稳定。
每个 token 独立停止，后续深度保留其最后实际计算的 K/V，训练梯度保留。

## 顺序与复现

1. `tiny.yaml` 验证 FSDP2、重计算、动态梯度、保存和独立进程恢复。
2. `tiny_compile_safe.yaml` 额外验证真实 Inductor 投影编译。
   `tiny_compile.yaml` 保留最初 CUDA graph 训练失败的实验参数；现在的代码在训练时关闭 replay，仍编译 kernel。
3. `warmup.yaml` 从原生 4B 做 32-step 随机深度预热。
4. `fixed36.yaml`、`fixed40.yaml`、`hidden.yaml`、`probability.yaml`、`hybrid.yaml`
   都从相同预热 checkpoint、相同数据和 seed 开始，各 64 steps × 4 ranks。
   动态配置包含预设 10% 强制 48 次批次；默认阈值来自训练前的独立小校准集。

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=src .venv-cached/bin/python scripts/run_training_stage.py \
  --gpus 0,1,2,3 --verify-restart --report-root reports/v0.0.4_dynamic_recurr \
  --configs configs/dynamic/hidden.yaml
```

不要覆盖已有实验输出后仍沿用旧报告；变更配置或代码后使用新 artifact/report 路径。
恢复会核对训练源代码和数据身份，旧实验需检出对应提交复验。

## 校准、质量与性能

校准只能使用 `corpus/halt_calibration.jsonl`；dev 选择候选，test 在冻结配置后评估。
初始四个 pack 只是诊断，不代表完整校准通过。全量文档评估不截断上下文：

```bash
CUDA_VISIBLE_DEVICES=4 PYTHONPATH=src .venv-cached/bin/python scripts/evaluate_dynamic_holdout.py \
  --model artifacts/v0.0.4_dynamic_recurr/train/hidden/hf \
  --data artifacts/v0.0.4_dynamic_recurr/corpus/halt_calibration.jsonl \
  --output reports/v0.0.4_dynamic_recurr/hidden_calibration_full.json
CUDA_VISIBLE_DEVICES=4 PYTHONPATH=src .venv-cached/bin/python scripts/benchmark_dynamic_inference.py \
  --model artifacts/v0.0.4_dynamic_recurr/train/hidden/hf \
  --output reports/v0.0.4_dynamic_recurr/hidden_inference.json
```

`calibrate_dynamic.py --grid ... --packs 0` 可覆盖整个 packed 校准集；
完整文档与 packed 口径有上下文差异，最终预算以报告注明的口径为准。
编译仅覆盖投影数值 kernel，动态分发、退出判断仍在图外；矩形 KV 的存储没有按退出步压缩。
吞吐包含这些开销，额外循环／P 的完整词表 JS 都不是免费计算。

## 教师答案

教师固定 revision，完成全部 LFS SHA256 校验后写入 download_manifest，才允许加载。
`generate_dynamic_teacher.py` 按冻结的训练 prompt 分片，保存采样种子、batch 和身份。
`prepare_teacher_distillation.py` 只接收所有分片完成、ID 完全匹配的输出；排除截断、
thinking、超长和不可核验数学答案。代码语法检查与正确性测试明确区分。

主实验的学生模型环境为 `.venv-cached`；官方评测为 `.venv-eval`。
数学筛选的可选依赖单独安装在 artifacts 下，通过 PYTHONPATH 导入，避免修改官方评测环境。

```bash
PYTHONPATH=artifacts/v0.0.4_dynamic_recurr/math_verifier:src .venv-eval/bin/python \
  scripts/prepare_teacher_distillation.py
```

实验进度、失败、实际资源与结论以 `plans/v0.0.4_dynamic_recurr_record.md` 和报告为准。

## 32B 配对阶段

`teacher32_fixed36`、`teacher32_fixed40`、`teacher32_anchor40`、`teacher32_hybrid`
使用同一份筛选后的 32B 序列答案；`matched_source_fixed36` 使用完全相同接受 prompt 池的原来源答案。
五组从相同 warmup HF 起点训练，64 steps × 4 ranks、长度 2048、compile 开启。
`anchor40` 推理固定 40，训练与 hybrid 使用相同 seed/step 的 10% 强制 48 批次，
用于区分实际动态退出与该保底训练日程的影响；普通 fixed40 对照同时保留。

教师与源答案长度不同，相同步数不保证相同有效 targets 或相同 prompt 曝光次数；
报告单列实际 token 与覆盖，源答案对照只解释替换答案后的整体训练配方，不能孤立归因到教师参数规模。
本轮 1k 教师生成采用 2048 输出上限、八副本、每副本 batch 8；更大的数据扩容等待收益证据。
