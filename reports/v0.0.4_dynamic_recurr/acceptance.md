# v0.0.4 验收索引

当前工程实现和独立 CE 测试完成，修正 EOS 后的六项完整官方评测仍在运行。最终结论见 [summary](summary.md)，实现选择与失败过程见 [record](../../plans/v0.0.4_dynamic_recurr_record.md)。

| 验收项 | 证据 | 边界 |
|---|---|---|
| 原层序兼容、动态 KV、缓存与 HF 重载 | [真实 4B 验证](model_verification.json)、tests/test_halting.py | 已测形状的数值证据，不推广到所有形状 bitwise |
| H／P／H+P 动态训练 | hidden.json、probability.json、hybrid.json | 固定判据，学习专家路由；P 部署平均深度未达预算 |
| FSDP2 混合停止、空活跃 rank 梯度 | [空 rank](empty_rank_fsdp.json)、[混合深度](mixed_rank_fsdp.json) | 真实分布式执行与参考梯度对照 |
| compile／重计算／保存与新进程恢复 | [最终导出回归](compile_4b_generation_fix.json)、[五组恢复](teacher_fresh_recovery.json) | 训练禁用 CUDA graph replay；数值 kernel 仍编译 |
| 延迟概率检查的等价性 | [4B 验证](deferred_readout_verification.json) | 保留 reference 路径，未编译 Python 动态控制 |
| 数据冻结与跨阶段泄漏复核 | [语料](corpus_manifest.json)、corpus_audit.json、warmup_overlap_audit.json | 规范化／包含匹配，不保证语义或预训练去污染 |
| 32B 离线教师与失败恢复 | [教师结果](teacher_distillation.json)、teacher_recovery_audit.json | 接受 804/1000；代码仅语法检查，数学核验有选择偏差 |
| 实际训练曝光与成本 | teacher_training_exposure.json、teacher_training_cost.json | 同 prompt 池不等于同消费样本；未作等 GPU-hours 再训练 |
| 冻结独立测试与配对区间 | [冻结协议](frozen_test_protocol.json)、[与固定 40 配对](paired_test_vs_teacher40.json) | 单训练 seed，H+P 未超过固定 40 |
| 深度分布与预算 | teacher_depth_distribution.json、teacher32_hybrid_eos_decode_calibration.json | H+P 高度集中于 40，不宣称有效内容自适应 |
| 原生 EOS 与采样配置 | [修正协议](corrected_generation_protocol.json)、generation_repair_verification.json | 原旧生成分数不作有效原生对比；CE 不受影响 |
| 编译推理吞吐 | [修复后性能](teacher32_hybrid_eos_performance.json) | 32-token 短测，共享机器；不是长输出端到端成本 |
| 官方评分可复现性 | ifeval_seeded_baseline.json、durable_recovery_verification.json | 沿用官方规则；逐题固定评分 RNG，非论文逐字节协议 |
| 完整 IFEval／MATH-500 | 六个 `full_eos_*_runs.json`（完成后生成） | 当前等待全部 541／500 唯一回答与评分，不能提前验收 |

未纳入本轮：压缩 KV、完全稀疏 attention、可学习退出头、投影解绑、14B 教师、多 seed 大规模复验、50M–100M 数据扩容。以上不是已实现功能。H+P 未显示优于固定 40 的质量证据，因此不启动盲目扩容。
