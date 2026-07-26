# Route A Fusion正式基线验证集错误画像

日期：2026-07-26

## 1. 诊断对象与纪律

- 模型：Fusion PIDNet-S正式基线；
- checkpoint：`route_a_fusion_100e_label_fix_seed200/best.pth`，epoch 86；
- 数据：固定验证集240张；
- 测试集：未使用；
- 目的：在选择B路线唯一候选前，判断剩余错误是边界问题还是区域内部问题。

诊断直接读取正式 `val_best/predictions_raw`，不重新选择checkpoint。重算的三类混淆
矩阵与 `val_best/metrics.json` 完全一致；Smoke和Fire边界计数也与正式评估逐项一致。

## 2. 边界/内部定义

真实语义边界由标签中上下或左右相邻像素类别变化构成，同时标记边界两侧；图像外框不
视为边界。边界误差定义为距真实语义边界Chebyshev距离不超过3像素的误分类像素；其余
误分类像素为区域内部误差。

该分解与类别边界F1是两套互补统计：误差分解使用统一的多类真实语义边界；边界F1仍
完全沿用 `evaluate_baseline.py` 的逐类二值边界和3像素容差。

## 3. 三类混淆矩阵

像素计数，行为真实类别、列为预测类别：

| True \\ Pred | Background | Smoke | Fire |
|---|---:|---:|---:|
| Background | 8,214,593 | 299,312 | 107,496 |
| Smoke | 318,549 | 5,194,690 | 187,370 |
| Fire | 98,241 | 169,560 | 1,138,829 |

按真实类别归一化：

| True \\ Pred | Background | Smoke | Fire |
|---|---:|---:|---:|
| Background | 95.2814% | 3.4717% | 1.2469% |
| Smoke | 5.5880% | 91.1252% | 3.2868% |
| Fire | 6.9841% | 12.0543% | 80.9615% |

最大错误对依次为：

1. Smoke → Background：318,549像素，占全部错误26.98%；
2. Background → Smoke：299,312像素，占25.35%；
3. Smoke → Fire：187,370像素，占15.87%；
4. Fire → Smoke：169,560像素，占14.36%；
5. Background → Fire：107,496像素，占9.11%；
6. Fire → Background：98,241像素，占8.32%。

## 4. 边界与内部错误分解

全验证集共有1,180,528个误分类像素：

- 真实边界3像素带内：925,917，占78.4324%；
- 区域内部：254,611，占21.5676%；
- 边界带像素错误率：21.5143%；
- 内部像素错误率：2.2286%。

按真实类别：

| 类别 | 全部错误 | 边界错误占本类错误 | 边界带错误率 | 内部错误率 |
|---|---:|---:|---:|---:|
| Background | 406,808 | 72.5856% | 24.6666% | 1.5021% |
| Smoke | 505,919 | 71.7202% | 18.9531% | 3.7788% |
| Fire | 267,801 | 99.9948% | 22.4617% | 0.0065% |

Fire的267,801个错误中只有14个距离真实语义边界超过3像素。需要注意，84.7553%的
Fire真值像素本身就在3像素边界带内，这与FLAME2中大量细小、狭长Fire区域一致；因此
“Fire错误几乎都在边界”不能单独证明Fire边界头失效，但证明继续优化大块Fire内部区域
没有收益空间。

## 5. Smoke与Fire边界指标

| 类别 | Boundary precision | Boundary recall | Boundary F1 |
|---|---:|---:|---:|
| Smoke | 0.884192 | 0.747883 | 0.810345 |
| Fire | 0.998368 | 0.835477 | 0.909688 |

Fire边界precision已接近1，但recall仍为0.8355；剩余Fire漏分中63.32%被分成Smoke，
36.68%被分成Background。Smoke边界F1明显低于Fire，主要短板是boundary recall。

## 6. 对B路线候选选择的约束

诊断不支持继续做区域内部增强、额外上下文模块或原型分离：内部只占21.57%的总错误，
Fire内部几乎没有错误。B路线唯一候选必须满足：

1. 直接监督主分割logits的语义边界，而不只监督PIDNet现有D分支；
2. 能提高边界召回或边界位置对齐，重点覆盖Smoke/Background与Smoke/Fire转换；
3. 训练期only，部署参数、FLOPs和延迟完全等于Fusion基线；
4. 有可确认的论文、作者官方代码和许可证；
5. 只允许一个候选进入30轮。

下一步只在满足以上诊断约束的方法中审计并选择候选。不会从CBL/ABL中任意挑选，也不
因为Fire边界F1较高就笼统宣称“所有边界已经解决”。

## 7. 证据文件

- `src/profile_fusion_errors.py`
- `experiments/route_a_pidnet_s_fusion/route_a_fusion_100e_label_fix_seed200/val_best/error_profile/error_profile.json`
- `experiments/route_a_pidnet_s_fusion/route_a_fusion_100e_label_fix_seed200/val_best/error_profile/confusion_matrix_counts.csv`
- `experiments/route_a_pidnet_s_fusion/route_a_fusion_100e_label_fix_seed200/val_best/error_profile/confusion_matrix_row_normalized.csv`
- `experiments/route_a_pidnet_s_fusion/route_a_fusion_100e_label_fix_seed200/val_best/error_profile/region_error_summary.csv`
- `experiments/route_a_pidnet_s_fusion/route_a_fusion_100e_label_fix_seed200/val_best/error_profile/per_image_error_profile.csv`

## 8. Route C追加空间归因

在B路线完成后，Route C又对同一固定Fusion验证预测完成Fire FN/FP的小目标与Smoke
空间归因。严格下四分位规则因Q1=1像素而退化为空小目标集合；Smoke邻域并集覆盖
69.90%的FN和72.18%的FP。完整定义、敏感性分析与证据见
`docs/FUSION_FIRE_SPATIAL_ATTRIBUTION_2026-07-26.md`。
