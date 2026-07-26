# Route A Fusion-DySample备用路线触发记录

日期：2026-07-26

## 1. 触发依据

最终实验计划规定：DEConv与mproto均未通过筛选时，启用备用路线，先评估
DySample，再考虑FreqFusion。Fusion正式基线确立后，旧RGB模块结论不直接沿用，
因此已在相同Fusion协议下重新筛选此前两条主路线的优胜者：

- D2：mIoU +0.0047978975，Fire +0.0035788852，未通过；
- P1：mIoU +0.0037798046，Fire -0.0060304533，未通过。

两项均完成30轮、无类别崩溃，P1原型健康，但都未满足进入100轮的既定门槛。
因此“DEConv与mproto均未通过”的备用路线触发条件在Fusion协议下成立。

## 2. 本轮候选范围

本轮只筛选一个候选：Fusion-DySample Pag4。

选择理由：

1. Pag4是最后一次I分支语义特征向P分支细节特征的直接融合，和Fire边界及小目标
   对齐问题最直接相关；
2. RGB阶段的Pag4候选已经完成30轮，并按当时相同协议基线通过筛选，提供了先验
   证据；
3. Context位置仅完成过工程检查，没有30轮结果，因此不把它描述为失败，也不把
   两个位置混在同一轮组合实验中；
4. 先用单一、理论依据最强且已有筛选证据的位置，符合限制备用路线变体数量的
   既定要求。

若Fusion-Pag4未通过，不自动启动Context或FreqFusion；后续路线需根据本轮结果
另行落档。若Fusion-Pag4通过，则进入100轮正式训练，并与Fusion基线进行参数量、
FLOPs及RTX 2060成对速度验收。

## 3. 固定实现与协议

- 模型：`pidnet_s_dysample`
- 插入点：`pag4`
- 输入：RGB三通道与IR一通道直接拼接，共四通道
- 数据协议：Fusion正式基线的A1协议，关闭强亮度变换，缩放0.5--1.5
- Seed：200
- 训练轮数：30
- 学习率horizon：100轮
- GPU：`cuda:0`，RTX 2060
- 实验组：`route_a_pidnet_s_fusion_dysample`
- 运行名：`route_a_fusion_dysample_pag4_30e_label_fix_seed200`
- 官方仓库：https://github.com/tiny-smart/dysample
- 固定commit：`81a1de5caa95d55a0f5488425fa53ec7ef47f8f0`
- LICENSE：MIT
- 官方设置：`style=lp`、`groups=4`、`dyscope=False`
- 测试集：继续封存

配置：

`configs/route_a/pidnet_s_fusion_dysample_pag4_30e_label_fix.yaml`

## 4. 筛选规则

候选第26--30轮均值与Fusion正式基线同一窗口比较：

- 基线mIoU：0.7861409556
- 基线Fire IoU：0.6467101738
- Fire `max-min` 保守噪声带：0.0142509431

进入100轮需满足任一：

1. mIoU提高至少0.005，且Fire不低于基线减去保守噪声带；
2. Fire提高超过保守噪声带，且mIoU下降不超过0.005。

同时要求Smoke和Fire无类别崩溃。筛选门槛不会因为结果接近而放宽。

## 5. 训练前工程验收

工程检查已通过：

- 官方checkout实际commit与配置一致，工作区干净，LICENSE为MIT；
- Fusion-DySample与Fusion基线均匹配301个ImageNet预训练张量；四通道stem按
  `official_shape_match_only`保持随机初始化；
- 模型包含两个官方DySample模块：`pag4.query_upsampler`与
  `pag4.value_upsampler`；
- AMP前向和反向成功，两个offset卷积梯度范数分别为0.006775和0.008643；
- 训练输出保持PIDNet三个张量，推理只返回主分割张量；
- 训练参数量从7,717,095增至7,729,639，增加12,544；
- 推理参数量从7,623,939增至7,636,483，增加12,544，约0.165%；
- 256×256四通道前向FLOPs从2.956013568G增至2.957635872G，增加
  0.001622304G，约0.055%。

复杂度数字使用架构模式计算，未冒充checkpoint评估；`measure_complexity.py`
新增显式 `--architecture-only` 选项，并已用Fusion best checkpoint回归验证原有
checkpoint加载路径未受影响。

证据：

- `experiments/route_a_pidnet_s_fusion_dysample/route_a_fusion_dysample_pag4_engineering/pipeline_check.json`
- `experiments/route_a_pidnet_s_fusion_dysample/route_a_fusion_dysample_pag4_engineering/complexity_architecture_only.json`
- `experiments/route_a_pidnet_s_fusion_dysample/route_a_fusion_dysample_pag4_engineering/fusion_baseline_complexity_architecture_only.json`
