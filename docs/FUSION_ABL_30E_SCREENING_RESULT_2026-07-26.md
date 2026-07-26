# Route A Fusion-ABL 30轮筛选结果

日期：2026-07-26

## 1. 实验身份与纪律

- B路线唯一候选：Active Boundary Loss；
- 模型：四通道Fusion PIDNet-S，网络结构不变；
- 运行名：`route_a_fusion_abl_30e_label_fix_seed200`；
- seed：200；训练30轮，学习率horizon 100轮；AMP；RTX 2060；
- ABL权重1.0，没有权重扫描；
- 测试集未使用。

候选依据、官方源码、Apache-2.0许可证、固定commit和兼容适配见
`docs/ABL_SOURCE_METHOD_AUDIT_2026-07-26.md`。训练前官方源码数值一致性、P/I/D/DFM
梯度、AMP、checkpoint恢复及推理隔离均已通过。

## 2. 固定第26--30轮结果

| 指标 | Fusion基线 | Fusion-ABL | 变化 |
|---|---:|---:|---:|
| mIoU | 0.786140956 | 0.792377067 | **+0.006236112** |
| Background IoU | 0.891985251 | 0.901545347 | +0.009560096 |
| Smoke IoU | 0.819727442 | 0.830255424 | +0.010527983 |
| Fire IoU | 0.646710174 | 0.645330430 | -0.001379743 |
| Fire precision | 0.750454037 | 0.760741992 | +0.010287955 |
| Fire recall | 0.824636898 | 0.811627365 | -0.013009533 |
| Fire boundary F1 | 0.888580114 | 0.886509115 | -0.002070999 |

Fusion基线Fire保守噪声带为`max-min=0.014250943`。ABL的Fire变化
`-0.001379743`位于该范围内，只能解释为中性，不能宣称提高或降低Fire。

候选窗口内mIoU为：

`0.790758, 0.787553, 0.795958, 0.791501, 0.796115`

Fire IoU为：

`0.645370, 0.640594, 0.648606, 0.642902, 0.649180`

Smoke和Fire均未发生类别崩溃。

## 3. 固定筛选判定

规则一要求mIoU至少`+0.005`，且Fire不低于基线减去保守噪声带。ABL满足：

- mIoU `+0.006236112 ≥ +0.005`；
- Fire `-0.001379743 ≥ -0.014250943`；
- 类别健康。

因此`passes_screening=true`，ABL进入100轮正式实验。没有启用第二个B候选，也没有改变
权重、seed、增强或筛选门槛。

## 4. 结果解释边界

30轮收益主要来自Background和Smoke IoU，Fire整体中性。Fire precision略升，但recall
和边界F1略降，因此当前不能写成“ABL已经改善火焰边界召回”。更稳妥的阶段性解释是：
ABL改善了多类语义边界附近的总体判别，尤其是Background/Smoke，但对Fire的净收益尚未
建立。是否能成为最终方法必须看100轮mIoU-selected best checkpoint和正式错误画像。

## 5. 下一步

启动唯一的100轮正式运行：

`route_a_fusion_abl_100e_label_fix_seed200`

配置、ABL超参数和训练协议完全不变。100轮仍按验证集mIoU保存best，并与Fusion正式
基线best比较内部方法成立门槛。测试集继续封存；在100轮正式结论冻结前不补三seed。

证据：

- `experiments/route_a_pidnet_s_fusion_abl/route_a_fusion_abl_30e_label_fix_seed200/metrics.jsonl`
- `experiments/route_a_pidnet_s_fusion_abl/route_a_fusion_abl_30e_label_fix_seed200/screening_vs_fusion.json`
- 同目录`resolved_config.json`、`environment.json`和`run_summary.json`
