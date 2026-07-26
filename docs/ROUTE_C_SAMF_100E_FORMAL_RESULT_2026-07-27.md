# Route C SAMF 100 轮正式结果

日期：2026-07-27

结论：SAMF 在三类 IoU、Fire recall 和边界 F1 上均有小幅正收益，并满足 RTX 2060 的 30 FPS 硬门槛，但未达到预注册的内部方法成立阈值。SAMF 保留为正向消融和设计证据，不单独作为已经成立的论文主创新；TGM 按既定顺序解锁。

## 1. 实验完整性

- 运行：`route_c_pidnet_s_samf/route_c_samf_100e_label_fix_seed200`；
- 100/100 轮，seed 200，AMP，RTX 2060；
- 从 ImageNet 预训练重新初始化，匹配 301 个张量，未续接 30 轮 checkpoint；
- 用时 3188.83 秒，峰值分配显存 237.80 MB；
- stderr 为空，测试集未使用；
- 按验证集 mIoU 保存 best，SAMF best 为 epoch 95，Fusion best 为 epoch 86。

## 2. best checkpoint 对比

| 指标 | Fusion 基线 | SAMF | 变化 |
|---|---:|---:|---:|
| mIoU | 0.806728034 | 0.808613050 | +0.001885015 |
| Background IoU | 0.908875792 | 0.909280513 | +0.000404721 |
| Smoke IoU | 0.841997893 | 0.843180155 | +0.001182263 |
| Fire IoU | 0.669310419 | 0.673378481 | +0.004068062 |
| Fire precision | 0.794331430 | 0.777911415 | -0.016420016 |
| Fire recall | 0.809615180 | 0.833642109 | +0.024026930 |
| Fire boundary F1 | 0.909687971 | 0.910664101 | +0.000976130 |

SAMF 的主要变化是以少量 precision 换取 Fire recall，三类 IoU 同时保持小幅正增益。这说明“烟概率门控热注入”没有造成类别退化，并具有方向正确的信号，但收益量级不足以宣称方法成立。

## 3. 正式门槛判定

预注册内部规则：

1. mIoU 至少 `+0.005` 且 Fire IoU 不下降；或
2. Fire IoU 至少 `+0.01` 且 mIoU 下降不超过 `0.003`。

本次 mIoU 仅 `+0.001885`，Fire IoU 仅 `+0.004068`，两条均未满足。因此不补 3 seed、不使用测试集，也不把 SAMF 单项列为最终主方法。

## 4. 部署结果

正式 best checkpoint、256x256、batch 1、AMP、RTX 2060 同一进程交替成对测速；每模型预热 100 次，每组 200 次，共 10 组：

| 模型 | 参数量 | FLOPs | 中位延迟 | FPS |
|---|---:|---:|---:|---:|
| Fusion PIDNet-S | 7,623,939 | 2.927702 G | 27.589912 ms | 36.245132 |
| Fusion + SAMF | 7,705,271 | 3.1545 G | 28.929772 ms | 34.566467 |

SAMF 延迟增加 4.8563%，但旧“速度下降不超过 3%”约束已废止。SAMF 仍高于 30 FPS，满足当前 Route C 实时硬门槛。

## 5. 决策

- SAMF 保留为正向消融、病因诊断到结构设计的证据链；
- 本结果归档时的旧组合限制已被 2026-07-27 的 Amendment 5 取代；若
  TGM 亦呈正向，可按新规则筛选 ABL+SAMF 与 ABL+SAMF+TGM；
- TGM 现在解锁，按工程验收 → 单一 30 轮筛选 → 通过才 100 轮的顺序执行；
- 测试集继续封存。

证据文件包括同目录的 `metrics.jsonl`、`environment.json`、`resolved_config.json`、`run_summary.json`、`formal_method_decision.json`、`complexity.json` 和 `paired_latency_rtx2060.json`。best/last checkpoint 不提交 Git。

注：本页原始正式结论保留为历史证据；组合授权的最新有效文本以
`docs/EXPERIMENT_RULE_AMENDMENTS.md` 的 Amendment 5 为准。
