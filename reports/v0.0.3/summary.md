# v0.0.3 最终验收结果

2026-09-29，8×H100。默认 RMT 为 Qwen3-4B 映射的 36-expert／36-recurrence `layer_order` HF checkpoint，BF16、non-thinking、capacity KV cache、真实 Inductor 编译。Qwen 后端为 Transformers。完整过程含失败候选见 [实施记录](../../plans/v0.0.3_infer_eval_record.md)，运行命令见 [使用说明](../../plans/v0.0.3_usage.md)。

## 质量回归

每项 32 个固定样本，max_new_tokens=2048，两模型 prompt／seed 相同，EvalScope 1.0.0 官方 adapter／scorer。共 320 次生成，无缺题／空评分／逐题分数差异。当前是等价初始化的工程回归，不是训练收益验证或技术报告完整复现。

| 任务 | RMT | Qwen | 生成文本完全相同比例 |
| --- | ---: | ---: | ---: |
| MMLU-Redux（四学科） | 75.00% | 75.00% | 87.5% |
| C-Eval（四学科，zero-shot） | 71.875% | 71.875% | 100% |
| IFEval（prompt strict） | 71.875% | 71.875% | 87.5% |
| MATH-500（五难度分层） | 84.375% | 84.375% | 100% |
| LiveCodeBench v5 增量日期子集 | 12.50% | 12.50% | 100% |

LCB 4/32、数学 3/32 达到输出上限，两模型一致；没有删除这些题目。样本量小且为固定子集，不能用此表证明模型优于 Qwen。官方 report 部分数值仅保留四位小数；此处百分比来自逐题统计。

- [最终质量与逐题成对审计](quality.json)：各 metric、Wilson 区间、截断率、token 数和原始 artifact 路径。
- [唯一正式运行索引](representative_runs.json)、[数据版本与抽样哈希](dataset_manifest.json)。旧 pilot／旧失败正式运行不进入本表。
- [160 条实际输入分词核对](tokenization.json)：两套 tokenizer 的渲染和 token IDs 一致，匹配真实 HTTP prompt hash。
- [隔离探针](sandbox.json)：正确／错误／超时判题、文件／网络限制、降权。加固后复评 LCB，结果不变。
- [重启恢复](resume.json)：相同服务元数据及数据集指纹，零新增模型调用、9 个输出文件哈希不变。

## 正确性和性能

| 检查 | 结果 | 证据 |
| --- | --- | --- |
| HF Qwen eager vs RMT eager，batch prefill＋4 decode | 已测 BF16 logits 逐字节一致 | [validation.json](validation.json) |
| 20 个真实 prompt × prefill＋4 decode，compile vs eager | max abs／relative RMSE／ΔNLL 均为 0 | [real_prompt_numerics.json](real_prompt_numerics.json) |
| fixed compile vs 同 cache eager | 三个 shape 热运行几何平均 **1.446×**，通过 1.10× 门槛 | [performance_final.json](performance_final.json) |
| mixed compile vs 同实现 eager | 数值通过；速度 **0.896×，未达加速目标** | [performance_mixed_bounded.json](performance_mixed_bounded.json) |
| cached 生产路径 vs uncached compiled 全前缀 | 输入 129／513／2049，输出 16：1.638×／1.655×／5.284× | [cache_and_profile.json](cache_and_profile.json) |
| 四副本 vs 单副本吞吐 | RMT 3.777×；Qwen 3.846× | [replica_scaling.json](replica_scaling.json) |
| HTTP／并发微批／stop／thinking smoke | 通过 | [service_contract.json](service_contract.json) |
| pytest（含真实 CUDA graph 专项） | 25 passed | 实施记录及本地 `tests.log` |

fixed 性能矩阵为输入 129／513／2049 token、batch 1／4／1、固定输出 32，三轮图预热后测三次。首次调用可能命中磁盘编译缓存，不代表完全冷启动。KV 性能比较包含 decode CUDA replay 的影响。四副本指标按后端分别测量，不用于归因 RMT/Qwen 架构速度。

逐字节一致仅覆盖已测严格参考路径；编译数值小样例全零也不保证所有输入或长采样文本一致。随机 mixed 路由只做受控数值／性能压力测试，不将未训练路由作为质量候选。

## 交付边界

默认 cached compiled 推理、离线 batch、HTTP 微批、八卡独立副本、官方五类代表性评测与结果追溯已交付。mixed compile 性能仍需后续 grouped kernel／调度优化。完整技术报告数据、完整 thinking、judge／工具交互／长上下文未运行，范围和原因见 [评测清单](../../configs/eval/inventory.json)。vLLM 在有限依赖探针内未完成接入，本版使用既定 Transformers fallback。

正式任务原始 wall time 求和约 0.609 小时，仅作为该轮生成与原评分的单卡 GPU 时间上界；不包括环境安装、下载、预热、调试或后续重评分。本任务启动的服务已停止。
