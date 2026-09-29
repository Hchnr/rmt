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
