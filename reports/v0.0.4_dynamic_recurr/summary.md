# 动态循环阶段报告（实施中）

32B 序列蒸馏与独立 CE 盲测已经完成；修复生成元数据后的官方全量评测仍在运行。

## 工程结果

- 36 套绑定专家，最大 48 次，前 36 次原层序、追加循环允许学习路由；相对旧固定 RMT 增加 432 个 step bias；相对原生增加 93,888 个 router 参数，总参数 4,022,561,984。
- H、P、H+P 固定退出，逐 token 独立状态；沿深度保留最后实际 K/V，训练不截断相关梯度。
- 固定深度兼容、因果性、混合退出、padding/packing、重计算、KV 重排与重载已测试。
- 真实 4B 已测轨迹的 full／cached／compile／HF remote-code reload logits 差为零；不推广为所有形状逐字节相等。
- 训练 compile 保留 Inductor，关闭 CUDA graph replay，避免已复现的反传生命周期错误。真实 4B 四卡 compile／重计算／独立进程精确恢复通过；全套 47 项测试通过。

## 数据

四个来源桶的独立 train 20,109 条、10,498,188 input tokens、7,459,764 targets；
calibration/dev/test 为 682/658/917 条。源 SHA、页面哈希与实际筛选见 corpus_manifest。
OpenCoder 来源桶混有普通问答，domain 不是严格语义标签；正式教师的 code 配额额外筛选编程线索。中文来源不足按实际记录；只声称已实现的规范化／包含匹配及来源标签筛查，不声称语义或预训练污染已消除。

## 已完成质量证据

同一个预热模型，在完整新 dev 的固定 36／40／48 CE 分别为 0.89915／0.92954／1.09903，追加深度尚无收益。

| 模型 | 完整 dev assistant CE | 平均深度 |
|---|---:|---:|
| 原生 Qwen3-4B | 1.40185 | 36 |
| 同预热起点固定 36 继续训练 | 0.81032 | 36 |
| 同预热起点固定 40 继续训练 | 0.82993 | 40 |
| H 动态训练，完整文档后校准 | 0.84032 | 39.559 |
| P 动态训练，后校准 | 0.85531 | 39.058 |
| 同预热起点 H+P 动态训练 | 0.83020 | 40.014 |

五个继续训练组均为 64 steps × 4 ranks、201,355 inputs／143,914 targets；并非相同 GPU-hours。首轮实际覆盖 373 个来源 ID，10.5M 是准备池大小，不是已经消费的训练量。
H+P 完整 calibration 平均 40.014，预算合格，但退出高度集中于 40，不能称为有效的内容自适应分配。
上述 CE 下降不等于官方任务能力提高，也不等于架构超越原生；固定 36 对照目前更好。

## 历史编译推理（EOS 修复前，复测中）

H+P、SDPA、KV cache、生成预算 32 token，预热后重复三次：

| batch | eager token/s | compile token/s | 加速 | 峰值 allocated GB |
|---|---:|---:|---:|---:|
| 1 | 12.47 | 13.61 | 1.092× | 8.25 |
| 4 | 50.25 | 51.34 | 1.022× | 8.29 |

已测 greedy token 一致。这里只编译投影 kernel；Python 路由／退出及矩形 KV 的开销均包含。
测试与其他实验共享机器，未保证独占全机带宽。只按存活 token 数比例推算加速是不成立的。

## 32B 蒸馏与独立测试

1,000 条教师输出接受 804 条；数学接受 128/300，存在截断与核验筛选偏差。
四个教师答案组实际均消费 401,335 inputs／335,019 targets，覆盖 395 个 ID。
源答案对照使用相同接受 prompt 池，但实际覆盖 804 个 ID，不能孤立归因为教师大小。
五组 compile＋FSDP2＋重计算及独立精确恢复全部通过。

| 模型 | 独立 test CE | 平均深度 |
|---|---:|---:|
| 原生 Qwen3-4B | 1.42016 | 36 |
| 首轮源语料固定 36 | 0.83233 | 36 |
| 首轮源语料 H+P | 0.85224 | 40.013 |
| 32B 答案固定 36 | 1.19693 | 36 |
| 32B 答案固定 40 | 1.14637 | 40 |
| 32B 答案 anchor40 | 1.15098 | 40 |
| 32B 答案 H+P | 1.14746 | 40.009 |
| 匹配来源答案固定 36 | 0.82230 | 36 |

32B H+P 相对固定 40 的 CE 差 +0.001092，成对文档 bootstrap 95% 区间
[+0.000351,+0.001765]；当前没有动态退出胜过固定 40 的证据。
相同教师数据下追加深度比固定 36 好，但这些 CE 不替代官方能力分数。

32B H+P 原阈值完整校准平均 40.010；修复 EOS 后生成校准 prefill／decode 为 40.057／40.018。
修复后编译推理 batch1／4 为 13.96／54.65 token/s，加速 1.059／1.045×，已测 greedy token 与 eager 一致。
H 修复后 decode 均值 40.067；P 为 38.604，低于 39–41 预算区间；初轮 H+P 为 40.024。P 的预算偏离保留，不基于 test 调整。

## 尚在执行

- 三个冻结 32B 学生的完整 IFEval／MATH-500 官方回归；IFEval 固定逐题评分 RNG，原生同一批答案重评为 436/541，旧 435/541 单独保留。
- 最终成本汇总、验收清单和结论。

## Generation-metadata correction (full rerun pending)

Earlier trained-checkpoint generation scores and performance used a single EOS instead of the native two-EOS configuration and are historical diagnostics, not valid final native comparisons. Corrected immutable exports preserve identical weights/configuration and real-4B forward logits. Teacher-forced CE remains valid. All six official tasks will be rerun from scratch; thresholds and sampling remain frozen. See generation_repair_verification.json and *_eos_repair.json. The four-prompt stopping probe did not observe the omitted EOS and does not explain all long outputs.
