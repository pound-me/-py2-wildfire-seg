# Route A Fusion PIDNet-S 100轮正式基线记录

日期：2026-07-26

## 实验身份

- 模型：PIDNet-S
- 输入：RGB 三通道 + IR 一通道直接拼接，合计四通道
- 数据：FLAME2 三分类 Background / Smoke / Fire
- 协议：采用 A1（关闭强亮度变换），随机缩放 0.5--1.5
- Seed：200
- GPU：NVIDIA GeForce RTX 2060（`cuda:0`）
- 训练轮数与 LR horizon：100 / 100
- 测试集：未使用

配置：

`G:\py2\configs\route_a\pidnet_s_fusion_100e_label_fix.yaml`

运行目录：

`G:\py2\experiments\route_a_pidnet_s_fusion\route_a_fusion_100e_label_fix_seed200`

## 预训练与输入层

官方 RoboFireFuseNet 的 PIDNet-S 数据用法将 RGB 与 IR 直接拼成四通道，并令
首层 `channels=4`。ImageNet 权重按官方 shape-match-only 规则加载，共匹配
301 个张量；四通道首层卷积不加载三通道预训练权重。该策略已写入 resolved
config，未使用未声明的权重扩展或替代初始化。

## 100轮结果

- Best epoch：86
- Best validation mIoU：0.8067280343
- Background IoU：0.9088757916
- Smoke IoU：0.8419978925
- Fire IoU：0.6693104186
- Fire precision：0.7943314303
- Fire recall：0.8096151795
- Fire Boundary F1@3px：0.9096879709
- 第100轮 mIoU：0.8064
- 第100轮 Smoke / Fire IoU：0.8412 / 0.6700
- 训练耗时：3951.5秒
- 峰值训练显存：228.0 MB

正式 best 评估：

`G:\py2\experiments\route_a_pidnet_s_fusion\route_a_fusion_100e_label_fix_seed200\val_best\metrics.json`

## 新基线窗口与门槛

第 26--30 轮固定窗口：

- mIoU 均值：0.7861409556
- Fire IoU 均值：0.6467101738
- Fire `max-min` 保守噪声带：0.0142509431
- Fire 样本标准差：0.0060898514
- Smoke/Fire 无类别崩溃

归档文件：

`G:\py2\experiments\route_a_pidnet_s_fusion\route_a_fusion_100e_label_fix_seed200\baseline_window_26_30.json`

后续所有 Fusion-D/P 30轮筛选均使用上述窗口。进入100轮仍满足任一：

1. mIoU 提高至少0.005，Fire 不低于基线减去 `Delta_fire`；
2. Fire 提高超过 `Delta_fire`，mIoU 下降不超过0.005。

## 复杂度与速度

- 推理参数量：7,623,939（7.624M）
- 前向 FLOPs：2.956013568 GFLOPs
- 估算 MACs：1.478006784 GMACs
- RTX 2060 单图独立评估延迟：25.870958 ms
- FPS：38.7

四通道 Fusion 只比三通道 RGB PIDNet-S 增加首层的288个权重；参数增量很小。
独立评估延迟不作为严格配对速度差结论，后续结构候选必须与本 Fusion 基线在
同一进程、相同四通道输入上做成对测速。

复杂度：

`G:\py2\experiments\route_a_pidnet_s_fusion\route_a_fusion_100e_label_fix_seed200\val_best\complexity.json`

## 相对原 RGB 正式基线

| 指标 | RGB best | Fusion best | 变化 |
|---|---:|---:|---:|
| mIoU | 0.613671 | 0.806728 | +0.193057 |
| Background IoU | 0.867944 | 0.908876 | +0.040932 |
| Smoke IoU | 0.755838 | 0.841998 | +0.086159 |
| Fire IoU | 0.217230 | 0.669310 | +0.452081 |

结论：Route A 的输入模态调整带来的收益远大于此前 RGB-only 结构模块收益。
Fusion PIDNet-S 正式成为新基线；RGB 基线保留为模态消融和研究动机材料。

## 下一步

保留 D2 DEConv 与 P1 EMA prototype，实现不删除。以本 Fusion 正式基线为
参照，分别重新进行30轮单因素筛选；不得直接沿用 RGB 下的模块结论。通过者
再进入100轮，并在四通道输入下完成参数/FLOPs和RTX 2060成对速度验收。
