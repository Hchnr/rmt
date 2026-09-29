# Bound RMT bootstrap

Qwen3-4B 的 36 层迁移为 36 套绑定专家，由同一个 recurrent cell 循环调用 36 次。每个 token 每次循环选择一整套 Q/K/V/O、FFN 和套内 norm。attention 始终保留原序列的因果关系；KV cache 按循环深度保存。

本项目当前是 **v0.0.2 正确性与系统兼容性验证**。训练使用少量固定样例，不能据此判断是否超过 Qwen3-4B。完整实现过程、失败定位和保守决策见 [record](plans/v0.0.2_bound_rmt_bootstrap_record.md)，设计见 [plan](plans/v0.0.2_bound_rmt_bootstrap.md)。

## 环境

当前已验证环境：Python 3.13.14、PyTorch 2.10.0+cu130、Transformers 4.57.6、8×H100。使用 `.venv-cached`；依赖精确版本在 `requirements.lock.txt`。项目代码用 editable 安装。

```bash
source .venv-cached/bin/activate
python -m pytest -q
```

若需要在本机重建，已有 wheelhouse 位于 `/tmp/rmt-wheelhouse`。缓存重打包脚本仅适用于本机现有完整 uv wheel archives，不从不完整缓存下载或修补文件：

```bash
/opt/venv/bin/python scripts/cached_environment.py --output /tmp/rmt-wheelhouse
uv venv --python /opt/venv/bin/python .venv-cached
uv pip install --python .venv-cached/bin/python --offline --no-index --find-links /tmp/rmt-wheelhouse -r requirements.lock.txt
uv pip install --python .venv-cached/bin/python --no-deps --no-build-isolation -e .
```

换机器时需根据 CUDA／Python ABI 提供相应 wheel 来源，并重新运行环境探针。曾尝试的 torch 2.7.1/cu126 在线路线因下载超时未采用，未将其记为验证环境。实际缓存来源见 `reports/bootstrap/environment_cache_manifest.json`。

## 验证命令

```bash
# 只读资产审计，逐字节 SHA-256
python -m rmt.audit --base-model /share/project/eai_pwm/models/Qwen/Qwen3-4B --full-hash --output reports/bootstrap/assets.json

# BF16、SDPA、Inductor 和 NCCL
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 -m rmt.audit --cuda --distributed --output reports/bootstrap/gpu_probe.json

# 398 tensor 迁移、真实 4B BF16 等价、HF 导出与 batched generation
CUDA_VISIBLE_DEVICES=0 python -m rmt.verify
# 独立进程重载已导出的 HF checkpoint
CUDA_VISIBLE_DEVICES=0 python -m rmt.verify --reload --output reports/bootstrap/hf_reload_4b.json

# 两卡更新与同全局 batch 单卡参考；入口关闭 TF32
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 -m rmt.fsdp_reference

# packing + compile + FSDP2 + 重计算，20 步及精确恢复
CUDA_VISIBLE_DEVICES=0,1 TORCHINDUCTOR_COMPILE_THREADS=2 torchrun --standalone --nproc_per_node=2 -m rmt.smoke_train --config configs/bootstrap/tiny_integration.yaml

# 当前授权下的 4 卡 4B，5 步及保存恢复
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 -m rmt.smoke_train --config configs/bootstrap/qwen3_4b_four_gpu.yaml
# 退出训练进程后，恢复下一步并逐参数 SHA-256 对照
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 -m rmt.smoke_train --config configs/bootstrap/qwen3_4b_four_gpu.yaml --verify-saved-resume
```

运行需自行保证所选 GPU 空闲。v0.0.2 后期测试使用物理 GPU 0–3；用户在 v0.0.3 已重新授权物理 GPU 0–7。历史训练配置仍保留原卡数，执行前按当时可用资源选择。`CUDA_VISIBLE_DEVICES` 可缩小范围。测试与性能报告中的有效 token 是输入 token，loss 只统计文档内有效 next-token targets。

报告在 `reports/bootstrap/*.json`，大 checkpoint 在被 git 忽略的 `artifacts/bootstrap/`。重复 smoke 会覆盖该配置的训练 checkpoint；请为正式实验设置独立 artifact 路径。训练 checkpoint 使用 PyTorch Distributed Checkpoint，额外按 rank 保存 scheduler、Python／CPU／CUDA RNG、数据 cursor、先验日程配置；当前 smoke 会自动对比恢复后下一步与不中断分支的 loss 及全部参数。

## HF 推理

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

path = 'artifacts/bootstrap/rmt-bound-4b'
tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
tokenizer.padding_side = 'left'
model = AutoModelForCausalLM.from_pretrained(
    path, trust_remote_code=True, local_files_only=True,
    dtype=torch.bfloat16, attn_implementation='eager',
).cuda().eval()
batch = tokenizer(['Explain recurrent models.', '用一句话介绍循环模型。'],
                  padding=True, return_tensors='pt').to('cuda')
with torch.no_grad():
    result = model.generate(**batch, max_new_tokens=32, do_sample=False)
print(tokenizer.batch_decode(result, skip_special_tokens=True))
```

`trust_remote_code=True` 加载的是导出目录中的本项目 Python 文件。转换 checkpoint 默认 `layer_order`，严格重现原层路径；它还没有经过能力训练。右 padding 的普通 token 输入会在 `generate` 内整理成左 padding；有显式 position_ids／forced_routes 时应由调用者提供左 padding。当前支持 append-only DynamicCache；`cache.batch_select_indices(indices)` 可用于重排或剔除请求。

## 训练与效率边界

- `layer_order` 用于数值基线；`forced` 接受 `[B,S,R]` 整数专家表；`learned` 使用 token 级 top-1。七类投影共享选择，norm 随专家绑定。
- router 用 FP32 score 和选中概率的直通代理梯度。代理有偏；forward 仍严格选择一套专家。无辅助 loss 的小实验已验证任务梯度和路由改变。
- 局部未使用专家显式产生零梯度，保证 FSDP2 各 rank 的梯度归约一致。因此未选中参数仍受 optimizer momentum／weight decay 影响；这与将 grad 保持 None 的跳过语义不同。
- KL 为 `KL(teacher || student)`，精确全词表、按 token 分块并重计算 head；教师冻结。teacher 与 student 使用相同 packing 和位置。
- FSDP2 以整个模型为稳定通信边界，持有 unsharded 参数穿过全部循环；不条件式 wrap 单个专家。BF16 计算配 FP32 master／reduction，避免 bootstrap 首轮引入额外精度问题。
- 固定逐层路径 compile 覆盖 QKV/norm、O/FFN 数值函数；v0.0.3 混合路径仅编译 MLP，以保留 BF16 reduction 数值行为。Python 分组调度保持 eager。两卡组合测试用真实 Inductor；八卡默认 eager 投影，分别验证实际权重规模下的训练与恢复。
- 当前 gather/scatter、逐专家 Python 循环和显式平方 attention mask 优先保证可检查性。它们是下一阶段吞吐优化对象，不是生产级 grouped GEMM、varlen attention 或 continuous batching。
- 完整 Qwen 技术报告评测集、语料工程、24h 公平质量实验属于后续 plan；本版结果仅说明实现语义和训练系统可行。

## 共享开发目录

主开发目录为 `/share/project/eai_pwm/home/hcr/repos/test/rmt`。其他 worktree 若通过 `.venv` 软链接共享环境，应指向此目录已验证的 `.venv-cached`，并从各自源码目录使用 `PYTHONPATH=src` 运行，避免 editable 安装指向另一 worktree。`.gitignore` 同时忽略 `.venv` 目录和软链接。源码位于 `src/rmt/`，测试位于 `tests/`；可用 `python -m rmt.checkpoint --base-model /share/project/eai_pwm/models/Qwen/Qwen3-4B --output artifacts/bootstrap/rmt-bound-4b` 单独转换权重。

## v0.0.3 推理与评测

新增 recurrence KV cache、默认编译的离线／HTTP 批量推理、Qwen HF 对照，以及使用 EvalScope 的五类固定样本回归。复现命令和边界见 [使用说明](plans/v0.0.3_usage.md)，过程、问题及验收结论见 [实施记录](plans/v0.0.3_infer_eval_record.md)。

## v0.0.4 训练稳定性与真实数据对照

新增确定性正式训练入口、真实指令语料 packing、精确教师 KL、FSDP2、重计算、可选训练 compile、DCP 恢复和 HF 导出。两组真实 4B／128 步候选已通过独立进程精确恢复；短程开放路由没有显示优于固定路径的质量证据。完整 IFEval 为 80.41%，数学全量和独占八卡吞吐仍在收尾。

结果与限制见 [阶段报告](reports/v0.0.4/summary.md)，复现命令见 [使用说明](plans/v0.0.4_usage.md)，持续记录见 [实施记录](plans/v0.0.4_train_stability_record.md)。本轮是训练链路和路由稳定性诊断，未宣称实现 24h 同计算预算超过原 Qwen。
