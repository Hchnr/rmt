# Bootstrap 验收报告索引

本目录记录工程正确性；fixture loss 不作为能力评测成绩。全部命令见仓库 README，异常定位与决策见 `plans/v0.0.2_bound_rmt_bootstrap/record.md`。

| 报告 | 内容 |
| --- | --- |
| `assets.json` | 398 tensor、4,022,468,096 基座参数，三个分片完整 SHA-256 及 tokenizer／配置指纹 |
| `gpu_probe.json` | BF16、SDPA backward、Inductor、两卡 NCCL |
| `unit_tests.xml` | 15 项 pytest，包括前后向、因果性、packing、cache、EOS、代理梯度及 E/R 解耦 |
| `equivalence_4b.json` | 有效 token logits、37 个 hidden 检查点和 ΔNLL 均为 0；新增 93,456 router 参数 |
| `hf_reload_4b.json` | HF 自定义模型独立进程重载、绑定权重、生成一致 |
| `fsdp_reference.json` | 不等长 rank 文档、不同局部专家使用；最大梯度误差 1.79e-7、更新误差 7.45e-9 |
| `tiny_integration.json` | 两卡 packing＋Inductor＋FSDP2＋重计算，20 步及完整恢复 |
| `tiny_integration_fresh_resume.json` | 两卡独立进程恢复，下一步所有参数 SHA-256 一致 |
| `tiny_reshard.json` | root forward 后 reshard 对照；20 步及恢复 |
| `qwen3_4b_integration.json` | 已授权窗口内的八卡 4B，5 步及同进程恢复 |
| `qwen3_4b_integration_fresh_resume.json` | 八卡独立进程恢复因授权缩至前四卡而中止；不是通过结果 |

四卡 `qwen3_4b_four_gpu.json` 已通过 5 步及同进程恢复；`qwen3_4b_four_gpu_fresh_resume.json` 已通过独立进程恢复，下一步 loss 和每个 rank 全部参数 SHA-256 相同。热步约 67–88 input tokens/s，训练峰值约 41.4GiB／卡。checkpoint 路径均与配置对应，八卡历史结果保留。

## 性能读法

- 全部实际 4B smoke 为每 rank 一行 16 tokens，冻结完整教师；CE＋精确全词表 KL，非代表性短序列负载。
- 八卡第 1–4 步热运行约 134–186 input tokens/s，最大分配显存约 35.8GiB／卡。去掉原层先验后，混合学习路径约 46–49 input tokens/s；动态专家分组数量明显影响吞吐。
- 小模型 forward 后 reshard 热步中位数约 86ms，保留 unsharded 权重约 77ms。两者 loss／路由轨迹一致；小模型显存由缓冲等开销主导，不用这个结果估计真实模型显存收益。
- 编译范围仅为投影／norm／FFN，Python dispatch 保持 eager；小模型捕获 7 个图。4B smoke 没有开启投影 compile。编译时间包含缓存状态，首次 step 也含其它初始化，不能视作单独编译耗时。
- 输入 tokens/s 与有效 next-token targets/s 不同；报告给出 `valid_targets`。所有 rank 按有效 target 数量加权训练 loss。

## 后续路线建议

先开训练稳定性与吞吐实验：保持强原层先验，逐渐开放路由；比较固定原层路径、缓慢开放路由和原 Qwen 继续训练。当前压力日程数步内放开路由会使 CE 从约 2.80 升至 9.38，说明路径切换尚未被训练适配。配合真实长度 profiling 决定 grouped GEMM、varlen packing 和通信边界优化，再制定 24h 公平质量实验。完整技术报告评测、正式语料与质量收益尚不属于本次验证结论。
