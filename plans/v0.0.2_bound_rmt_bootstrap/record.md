# v0.0.2 实施记录

## 范围与执行原则

按 [实现计划](../v0.0.2_bound_rmt_bootstrap.md)完成 B0–B5：36 套绑定专家、36 次循环、HF 转换与生成、正确性测试、可学习路由、训练和分布式组合验证。只做 bootstrap 与短跑，不执行 24h 能力实验。用户授权自主采用保守方案并持续提交；决策、失败与实际验证结果在此记录。

## 进度

| 阶段 | 状态 |
| --- | --- |
| B0 环境与资产 | 进行中 |
| B1 小模型固定路径 | 待实现 |
| B2 本地 4B 迁移／等价 | 待实现 |
| B3 路由／cache／packing／HF | 待实现 |
| B4 可学习路由与恢复 | 待实现 |
| B5 compile／FSDP2／重计算组合 | 待实现 |

## 2026-09-29：开始实施

- 读取现有计划；仓库已有 Git，v0.0.2 计划尚未提交，纳入本次基线提交。
- 本地基座：`/share/project/eai_pwm/models/Qwen/Qwen3-4B`。
- 当前 8 张 H100 均被已有任务占用约 73GB／80GB，GPU 利用率 100%。不停止或修改已有任务；先完成环境、CPU 正确性实现，并随进展检查可用显存。GPU 验证按实际资源执行，未执行项绝不记为通过。
- 采用独立 `.venv`、固定依赖；产物 checkpoint 不进入 Git，报告和代码进入 Git。

### 资源与环境决策更新

- 用户明确只授权物理 GPU 0–3；全部运行脚本和验证命令固定 `CUDA_VISIBLE_DEVICES=0,1,2,3` 或其子集。GPU 4–7 不使用。当前前四卡已经空闲。
- 本轮分布式验收改为两卡小模型和四卡 4B；八卡验证列为 `not_run: waiting_for_user_resource_release`，不会自行扩大到八卡。
- Python 3.11 下载长时间无进展，终止自己启动的下载，保守改用系统已有 Python 3.12 创建隔离环境；模型算法与依赖版本不因这个选择改变。

### B1 首轮实现（待依赖安装完成后运行测试）

- 已实现单个 recurrent cell、36 套参数组织、layer_order／forced／learned 三种路由、全序列 GQA、按循环索引的 HF cache，以及同套 norm 和七投影绑定。
- 编译边界选为数值投影函数，动态 token 分组保持在 Python，避免把部分编译错误表述成全图编译。
- router 采用计划中的选中概率直通代理，明确是有偏估计；固定路径绕过代理以校验真实梯度。
- 已加入固定路径 forward/backward、因果性、packing 梯度、混合路由 cache、重计算与 HF 生成／导出的初始测试；此时尚未执行，不能视为通过。

### 资源再次更新与离线依赖方案

- 用户随后开放全部 8 卡；运行入口默认范围更新为 GPU 0–7，恢复八卡 4B 验收。之前的前四卡限制只对应当时授权窗口。
- torch 2.7.1／CUDA 12.6 在线下载反复超时，两个 registry 路线均无法及时完成；没有继续无限重试。
- 系统自带 Python 3.12／NGC torch 2.12 nightly 的 BF16 CUDA matmul 可运行，但为避免使用 nightly，选择本机 uv 缓存中的稳定 torch 2.10.0+cu130、Transformers 4.57.6、Python 3.13.14，在独立 `.venv-cached` 中离线安装。计划版本因此调整，数值基准也使用该锁定的 HF 版本。
- 增加 `scripts/cached_environment.py`：只读缓存，按 wheel ABI 和依赖版本重建本地 wheelhouse，保留来源 manifest；不修改系统 Python 或共享缓存。后续用实际 CUDA／compile／NCCL 探针验证兼容性，不凭驱动版本号推断通过。

### B0/B1 运行结果与 B2 对照修正

- 环境安装完成，锁定 Python 3.13.14、torch 2.10.0+cu130、Transformers 4.57.6；离线缓存来源已保存到报告目录。
- CUDA BF16 matmul/backward、SDPA/backward、真实 Inductor 编译、两卡 NCCL collective 全部通过（`reports/bootstrap/gpu_probe.json`）。
- 小模型 10 项测试通过，含逐参数梯度等价、token 分组慢速参考、任务驱动 router 更新、精确分块 CE/KL 梯度、packing、cache 和 HF roundtrip。新版 DynamicCache 的存储接口由 key_cache 改为 layers，测试据实际接口更新。
- 真实 4B 首轮发现对照输入位置不一致：HF 原模型直接 forward 默认物理 arange，而 RMT 左 padding 默认有效 token cumsum。已在教师对照显式传入相同 position_ids，保留原阈值重跑，不放宽误差阈值。
- 增加真实模型验收入口和两卡／八卡短训练入口；当前开始运行组合探针，尚未声明通过。

### B2/B3 通过；B5 发现并处理局部未使用专家梯度

- 相同显式 position_ids 后，真实 4B 的 398 个源 tensor 逐项相等，37 个 hidden 检查点、有效 token logits、ΔNLL 均为 **0**；迁移后基座参数 4,022,468,096，新增 router 参数 93,456。
- `artifacts/bootstrap/rmt-bound-4b` 已完成 HF 导出。独立进程 AutoModel＋trust_remote_code 重载后 logits 逐位一致、生成一致、embedding/head 仍绑定。
- 左右 padding batched generation、EOS 独立结束、SDPA 混合路由 cache 重排／剔除新增覆盖，小模型合计 12 项测试通过（后续修复后重新执行）。
- 两卡 packing＋Inductor＋FSDP2＋重计算完成 20 步，包括混合学习路由和受控重复路由；同进程保存恢复后的下一步 loss 与全部参数完全一致。
- 额外的同全局 batch 分布式梯度对照发现：某 rank 完全未调用某专家时，FSDP2 在该 rank 上的对应梯度可能为 None，即使其他 rank 使用了该专家。20 步能运行并不足以证明梯度正确。
- 暂停刚启动的八卡短跑（仅终止本任务 torchrun），先修复：训练 loss 加入每个可训练参数的零值依赖，使局部未使用参数显式产生零梯度并参与同一个 reduce-scatter；单卡／分布式一致采用此语义。对应 optimizer 对未使用参数仍执行 momentum／weight decay，明确记录此决策。修复后重新验收分布式对照和两卡组合，再运行八卡。

### 分布式修复确认与数值环境决策

- 零梯度依赖修复后，分布式对照通过：forced 路径（rank 0 仅专家 0、rank 1 仅专家 1，专家 2 全局未使用）最大梯度误差 5.96e-8，learned 路径 1.19e-7；SGD 更新最大误差 7.45e-9。
- 本机环境预设 `TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1`，不同 GEMM batch 形状下 TF32 使 FP32 梯度对照出现约 1.2e-4 差异。严格数值对照显式关闭 TF32（包括 NVIDIA override），保留原容差；性能短跑保留原环境，并在版本信息中记录。
- 重新运行已修复的两卡组合短跑，然后恢复八卡 4B；增加 append-only cache_position 验证，拒绝未实现的静态／任意位置 cache 写入。

### 八卡真实 4B 首轮验收通过与训练观察

- 八卡 4B（E=R=36）完成 5 个 optimizer steps；开启 packing、FSDP2、循环重计算、分块 CE/KL，投影采用 eager。前两步恢复原层路径，第三／四步已出现学习路由变化，第五步强制反复混用专家 0／1。
- 保存完整模型与 AdamW 状态后，恢复的下一步 loss 与不中断分支一致，全部参数最大差异为 0。报告为 `reports/bootstrap/qwen3_4b_integration.json`，训练 checkpoint 在 `artifacts/bootstrap/qwen3_4b_integration`（约 47GB）。
- 首轮 rank 0 训练峰值约 35.8GiB；短序列 16、全局 128 input tokens 下热运行约 137–184 input tokens/s。此时序列／dispatch 极小，不能外推正式训练吞吐。后续报告改为记录所有 rank 的最大峰值。
- **训练观察**：先验 4.0 时 CE≈2.80；在数步内降到 1.6 后 CE≈9.38，受控异常路径 CE≈17.25。说明原层权重对随意替换路径不具备即用兼容性。当前日程是触发路由变化的压力测试，不适合直接作为正式蒸馏日程；后续应维持强先验并慢慢开放路由，增加稳定性对照。
- 进一步完善不同 rank 有效 targets 数量不一致时的全局 loss 归一化，对照测试也改为不同长度文档；保存每 rank 下一步参数 SHA-256，增加独立进程恢复验收入口。由此会重跑受影响的训练检查。

### 恢复能力与验收加固

- 小模型目前 **14 项 pytest 通过**，新增直通代理的解析梯度检查（forward 精确保持选中 expert 输出）。
- 不等长 rank 文档的全局有效 target 归一化对照通过：最大梯度误差 1.79e-7、更新误差 7.45e-9。
- 两卡重新训练、保存后，**退出进程并重新 torchrun** 恢复：下一步 loss 一致，每个 rank 全部参数 SHA-256 一致。新增 `--verify-saved-resume` 可复现实验；对比包括 optimizer／scheduler／RNG／数据 cursor 恢复。
- 增加 `tiny_reshard.yaml`，比较 root FSDP 保留 unsharded 与 forward 后 reshard 的语义和开销；仅在稳定 root 边界比较，不引入专家条件 collectives。
- 正在以最终训练入口重新完成八卡验收并做独立进程恢复。README 和最终锁文件已整理，原 plan 顶部注明当前实施状态及实际依赖版本。

### 合并冲突解决与再次限制 GPU

- 用户要求先解决 merge 冲突。冲突仅在 `.gitignore` 和 README：保留当前已验证的环境／验收说明，合入对方关于主开发目录、共享 worktree 环境和独立转换命令的内容；`.venv` 忽略规则同时覆盖目录与软链接。已提交 merge，没有覆盖现有测试报告。
- 当前授权再次缩小为物理 GPU 0–3。已停止本任务仍在运行的八卡独立进程恢复 torchrun；没有终止其他任务。运行入口恢复前四卡限制。
- 最新八卡 5 步及同进程恢复已完整通过，报告保留；八卡独立进程恢复因资源授权变化中止，不记为通过。新增独立产物路径的四卡配置，继续实际 4B 训练和独立进程恢复验收，避免覆盖八卡 checkpoint。

### 最后接口检查与通信边界对照

- 修复新版 HF 的 `layer_types` 长度对 E/R 解耦的影响：配置按循环次数保存 attention 类型；expert 中的原 HF 层编号仅作参数容器占位，cache 仍由 recurrent controller 按循环深度索引。补测 E=3、R=2／5 的 forward、cache 和权重迁移。
- root `reshard_after_forward=True` 的两卡 20 步及恢复也通过，与持有权重版本的 loss／路由轨迹一致。本次小模型热步中位数约 86ms，对照持有权重版本约 77ms；显存差异被小模型／通信缓冲掩盖，不据此外推 4B。默认继续采用保留权重策略。
- HF 导出目录的 Python 源码已更新到当前实现，权重不变；后续重载检查将校验一致性。

- 合并后 15 项单测全部通过，`git ls-files -u` 为空、`git diff --check` 通过。
- 四卡真实模型已完成 5 步，峰值约 41.4GiB／卡；正在保存和恢复验证。报告索引与下一阶段建议已写入 `reports/bootstrap/README.md`。


### v0.0.2 最终交接

- **代码冲突已解决并提交**：merge commit `3dd87b2`；GPU 范围始终以最新授权为准，目前默认／允许物理 GPU 0–3。
- **B0–B5 本版验收完成**：资产完整指纹、15 项单测、真实 4B BF16 零误差等价、混合路由／packing／cache／EOS、HF 独立进程重载、两卡四项组合 20 步、分布式全局 batch 梯度对照、八卡 4B 5 步及同进程恢复均通过。
- 四卡 4B 完成 5 步和同进程恢复；随后退出进程，重新 torchrun 恢复，下一步 loss = 24.785930633544922，与保存前参考一致，每个 rank 全部参数 SHA-256 一致。最终报告：`reports/bootstrap/qwen3_4b_four_gpu_fresh_resume.json`。
- 更新后的 HF 源码已重新通过独立进程重载／生成检查；推理产物：`artifacts/bootstrap/rmt-bound-4b`。此 checkpoint 是严格迁移的 layer_order 基线，不是质量提升模型。
- 八卡独立进程恢复属于追加检查，因授权变化中止，保留 `interrupted` 状态；四卡独立进程恢复补足真实规模重启验证，没有把八卡未完成项伪记通过。
- 当前训练 smoke checkpoint 支持同配置、同 world size 的精确重启验收；变更 world size 的恢复／数据重分片没有验证。当前 mask、dispatch、compile 范围和生产性能限制详见 README。
- 完整复现命令见仓库 README；报告索引与下一阶段建议见 `reports/bootstrap/README.md`。本次没有运行正式评测或 24h 质量实验，也没有声称超过 Qwen3-4B。
- 下一阶段优先解决路由开放的训练稳定性，并使用真实序列长度做吞吐和原 Qwen 对照，再确定正式蒸馏语料及公平质量评测。当前数步内快速移除原层先验的日程只用于压力测试，不建议直接用于正式训练。
