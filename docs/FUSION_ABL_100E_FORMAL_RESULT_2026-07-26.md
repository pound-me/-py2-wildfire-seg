# Route A Fusion-ABL 100轮正式结果

日期：2026-07-26

## 1. 实验身份

- 模型：Fusion PIDNet-S + Active Boundary Loss（训练期only）；
- 运行名：`route_a_fusion_abl_100e_label_fix_seed200`；
- seed：200；100轮训练与100轮学习率horizon；AMP；RTX 2060；
- 从ImageNet预训练权重重新开始，未续接30轮checkpoint；
- checkpoint按验证集mIoU保存；测试集未使用。

训练耗时`4248.77 s`，峰值分配显存`232.15 MB`。官方源码、许可证、适配和工程验收
见`docs/ABL_SOURCE_METHOD_AUDIT_2026-07-26.md`。

## 2. mIoU-selected best checkpoint

Fusion基线best为epoch 86，ABL best为epoch 93：

| 指标 | Fusion基线 | Fusion-ABL | 变化 |
|---|---:|---:|---:|
| mIoU | 0.806728034 | 0.808422710 | +0.001694676 |
| Background IoU | 0.908875792 | 0.909832057 | +0.000956265 |
| Smoke IoU | 0.841997893 | 0.843313296 | +0.001315403 |
| Fire IoU | 0.669310419 | 0.672122777 | +0.002812359 |
| Fire precision | 0.794331430 | 0.766599101 | -0.027732330 |
| Fire recall | 0.809615180 | 0.845050937 | +0.035435758 |
| Fire boundary F1 | 0.909687971 | 0.912402012 | +0.002714042 |

ABL提高了Fire recall，但伴随precision下降；总体mIoU、Fire IoU和边界F1均只有小幅
正变化，不能把该结果表述为显著改善。

## 3. 正式门槛判定

内部方法成立规则：

1. mIoU至少`+0.005`且Fire不下降；或
2. Fire IoU至少`+0.01`且mIoU下降不超过`0.003`。

本次只有mIoU `+0.001695`、Fire `+0.002812`，两条规则均未满足。因此：

- B路线正式失败并关闭；
- ABL保留为训练期边界监督消融，不作为论文主创新；
- 不自动启动CBL或第二个B路线损失；
- 测试集继续封存，暂不补3 seed。

## 4. 轻量化属性

ABL不改变推理模型，部署参数、FLOPs和推理延迟与同一Fusion PIDNet-S checkpoint架构
一致。在后续统一mIoU–延迟Pareto图中，ABL与Fusion基线使用相同架构延迟坐标，但分别
保留各自验证集mIoU。

## 5. 证据

- `experiments/route_a_pidnet_s_fusion_abl/route_a_fusion_abl_100e_label_fix_seed200/metrics.jsonl`
- 同目录`environment.json`、`resolved_config.json`、`run_summary.json`；
- best checkpoint：同目录`best.pth`（不提交Git）；
- `experiments/route_a_pidnet_s_fusion_abl/route_a_fusion_abl_100e_label_fix_seed200/formal_method_decision.json`。
