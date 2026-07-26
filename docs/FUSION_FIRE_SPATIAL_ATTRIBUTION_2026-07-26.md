# Fusion基线Fire错误空间归因

日期：2026-07-26
范围：Fusion PIDNet-S正式基线epoch 86，固定验证集240张；测试集未使用。

## 1. 定义与验证

- 直接读取正式`val_best/predictions_raw`，不重新选择checkpoint；
- Fire FN：GT=Fire且预测不为Fire，共267,801像素；
- Fire FP：预测=Fire且GT不为Fire，共294,866像素；
- 两个总数与已保存混淆矩阵逐项一致；
- Fire连通域在评估分辨率GT掩膜上按8邻域计算；
- Smoke相关：位于GT Smoke或预测Smoke的3像素Chebyshev邻域并集，区域本身包含在内；
- 3像素半径沿用现有Fusion错误画像和边界指标协议。

FN可以直接归属GT Fire连通域。FP不可能属于GT Fire连通域，因此同时报告两种透明定义：

1. 主定义：FP所属的“预测Fire连通域”面积小于GT Fire面积阈值；
2. 次级邻近代理：FP距小GT Fire连通域不超过3像素。

这样不把FN的GT归属定义暗中套到FP，也不把邻近误写成所属。

## 2. 下四分位阈值退化

全验证集共有31,315个GT Fire连通域，面积分布为：min=1、Q1=1、median=3、
mean=44.9187、max=6570像素。原指令规定“小目标=面积严格小于Q1”，因此正式阈值为
`area < 1`，没有任何合法Fire连通域满足，主结果的小目标计数必然为0。

本项目不把规则偷改为`<=1`。为避免丢失单像素火点信息，另外报告一个明确标注为
“描述性敏感性分析”的`area <= Q1`结果；它不替换正式主定义，也不用于改变筛选门槛。
在该敏感性定义下，单像素GT Fire连通域有9,338个，占全部连通域29.82%。

## 3. 正式主结果：严格`area < Q1`

| 错误类型 | 总像素 | 小目标相关 | Smoke相关 | Smoke相关占比 | 两者均非 |
|---|---:|---:|---:|---:|---:|
| Fire FN | 267,801 | 0 | 187,196 | 69.9012% | 80,605 |
| Fire FP | 294,866 | 0 | 212,836 | 72.1806% | 82,030 |

Smoke相关并非只靠邻域推断：

- FN中169,560像素被模型直接预测为Smoke，占全部FN 63.3157%；
- FP中187,370像素的GT本身就是Smoke，占全部FP 63.5441%；
- 使用GT/预测Smoke 3像素邻域并集后，分别覆盖FN 69.90%和FP 72.18%。

因此Fire漏检和误检都主要集中在Smoke/Fire混淆及其空间邻域。这个结论与worst-20的
20/20 heavy烟遮挡人工结论一致，直接支持SAMF先于一般边界或小目标模块。

## 4. `area <= 1`描述性敏感性分析

| 描述性指标 | 像素 | 占对应错误比例 |
|---|---:|---:|
| FN属于单像素GT Fire连通域 | 8,749 | 3.2670% |
| 上述FN同时Smoke相关 | 5,805 | 2.1677% |
| FP属于单像素预测Fire连通域 | 73 | 0.0248% |
| FP距单像素GT Fire不超过3像素 | 25,912 | 8.7877% |

即使采用更宽松的`<=Q1`描述，小目标只解释少量FN；FP主要不是孤立的单像素预测噪声。
Smoke相关比例仍远高于小目标相关比例。小火点可以作为次要困难保留在论文动机中，但
不能替代烟遮挡作为C路线首要病因。

## 5. 证据文件

- `src/attribute_fusion_fire_errors.py`
- `experiments/route_a_pidnet_s_fusion/route_a_fusion_100e_label_fix_seed200/val_best/fire_spatial_attribution/fire_error_spatial_attribution.json`
- 同目录`fire_error_spatial_attribution_per_image.csv`

脚本会校验FN/FP总数与正式混淆矩阵一致，并校验small-only、smoke-only、both、neither
四个互斥分区之和等于对应错误总数。相同输入连续重跑的输出哈希完全一致：JSON
`981C766BFDAA76EBC078FA1CA6C4DA3406B2F14FC5990851C3CE82A1CAE86B00`，CSV
`B7907B518AED8EEB3280B8E87949762FA4D6A099BDEDF5F1A0E9679025DFD155`。
