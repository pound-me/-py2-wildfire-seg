# LSCM v2 实验记录

记录日期：2026-07-24  
当前状态：完成 v2 两组权重与 v2.1 梯度隔离版的公平 10 轮筛选；二值 smoke-vs-rest 门控路线停止，不进入 100 轮长训练，不使用测试集挑选模型。

## 1. 研究问题

LSCM v1 的空间门控没有明确的烟雾语义约束，100 轮训练中残差系数由 0.1 降到约 0.05，最终未超过 PIDNet-S 基线。

LSCM v2 保持 v1 的轻量多尺度上下文结构，同时增加训练期二值烟雾监督：

```text
I 分支多尺度上下文
  -> 1 通道 smoke logits
  -> sigmoid smoke gate
  -> 残差增强

smoke_target = (label == 1)
auxiliary loss = BCEWithLogits + Soft Dice
label == 255 的像素不参与辅助损失
```

第三方 `third_party` 代码未修改。自定义实现：

- `G:\py2\src\custom_models\pidnet_lscm.py`
- `G:\py2\src\custom_losses.py`

## 2. 稳定初始化

真实批次检查发现，单通道 1x1 gate 头若使用 `fan_out` Kaiming 初始化，初始 logits 幅度过大：原始辅助损失约 4.01，其中 BCE 约 3.24。

v2 将 gate 头改为：

- 卷积权重初始化为 0
- bias 初始化为 -1
- 初始 gate 为 sigmoid(-1)，约 0.269

稳定初始化后，同一真实批次的辅助损失降至 1.3685，其中 BCE 0.5098、Dice 0.8587；smoke head 梯度范数为 0.2959，前向和反向均正常。

## 3. 公平筛选协议

- 数据：552 train / 240 validation
- 输入：RGB，256 x 256
- Seed：200
- 独立 `torch.Generator(seed=200)` 控制训练 shuffle
- cuDNN：deterministic=True，benchmark=False
- ImageNet 预训练：302 个张量匹配
- Optimizer：SGD
- 初始学习率：0.001
- 实际训练轮数：10
- Polynomial LR 调度总跨度：100 轮
- AMP：开启
- 模型选择：验证集 mIoU
- 测试集：未使用

该协议使 10 轮筛选复现正式 100 轮训练的前 10 轮学习率轨迹，而不是在第 10 轮把学习率提前降到零。

## 4. 参数量与计算量

输入：1 x 3 x 256 x 256。统计方法与正式基线一致。

| 指标 | PIDNet-S | LSCM v2 | 增量 | 增幅 |
|---|---:|---:|---:|---:|
| 推理参数量 | 7,623,651 | 7,641,541 | 17,890 | 0.2347% |
| Forward GFLOPs | 2.946576 | 2.982295 | 0.035718 | 1.2122% |
| GMACs | 1.473288 | 1.491147 | 0.017859 | 1.2122% |

辅助损失只在训练期计算，不增加部署模型的额外损失计算。

## 5. 十轮结果

基线使用正式公平 100 轮实验 `baseline_100e_fair` 的前 10 轮。

| 模型 | 最佳轮次 | mIoU | Background IoU | Smoke IoU | Fire IoU |
|---|---:|---:|---:|---:|---:|
| PIDNet-S 前10轮 | 10 | 0.576613 | 0.836530 | 0.698519 | 0.194790 |
| LSCM v2, aux=0.4 | 10 | 0.582413 | 0.848904 | 0.739355 | 0.158980 |
| LSCM v2, aux=0.2 | 5 | 0.555111 | 0.795111 | 0.727238 | 0.142983 |
| LSCM v2.1, 梯度隔离, aux=0.4 | 6 | 0.568525 | 0.812862 | 0.739448 | 0.153264 |

相对基线最佳结果：

| 模型 | mIoU 变化 | Smoke IoU 变化 | Fire IoU 变化 |
|---|---:|---:|---:|
| aux=0.4 | +0.005800 | +0.040836 | -0.035810 |
| aux=0.2 | -0.021502 | +0.028719 | -0.051807 |
| v2.1 梯度隔离 | -0.008088 | +0.040929 | -0.041526 |

十轮逐轮平均：

| 模型 | 平均 mIoU | 平均 Smoke IoU | 平均 Fire IoU |
|---|---:|---:|---:|
| PIDNet-S | 0.526800 | 0.610501 | 0.163962 |
| LSCM v2, aux=0.4 | 0.515289 | 0.632953 | 0.141731 |
| LSCM v2, aux=0.2 | 0.488370 | 0.536485 | 0.150174 |
| LSCM v2.1, 梯度隔离 | 0.516023 | 0.590495 | 0.152738 |

`aux=0.4` 在 10 轮中有 5 轮 mIoU 高于同轮基线。其 Smoke IoU 平均提高 0.022452，但 Fire IoU 平均下降 0.022231，平均 mIoU 下降 0.011511。

## 6. 辅助学习状态

`aux=0.4`：

- validation smoke auxiliary loss：1.3341 -> 0.8231
- 最佳 checkpoint 残差系数：0.096782（初始 0.1）
- smoke head bias：-0.966782
- smoke head weight norm：0.857884

`aux=0.2`：

- validation smoke auxiliary loss：1.3160 -> 0.8637
- 最佳 checkpoint 残差系数：0.100069
- smoke head bias：-0.964525
- smoke head weight norm：0.467294

两组辅助损失均正常下降，说明烟雾监督确实被学习。主要问题不是辅助头失效，而是烟雾特化与三分类主任务之间出现类别竞争。

## 7. v2.1 梯度隔离实验

为验证类别竞争是否由辅助梯度直接改写共享上下文特征引起，v2.1 使用独立文件：

- `G:\py2\src\custom_models\pidnet_lscm_v21.py`

其 smoke head 接收 `context.detach()`：

- 二值烟雾辅助损失只训练 smoke head。
- 三分类主损失仍训练共享多尺度上下文、残差路径和 smoke head。
- 推理结构、参数量与 FLOPs 不变。

真实批次梯度验证：

- 仅辅助损失时 smoke head 梯度范数：0.586328
- 仅辅助损失时共享上下文梯度范数：0.000000
- 完整损失时共享上下文梯度范数：0.143755

说明梯度隔离实现正确。

公平 10 轮结果：最佳 epoch 6，mIoU 0.568525，Smoke IoU 0.739448，Fire IoU 0.153264。相对基线，Smoke IoU 提高 0.040929，但 Fire IoU 下降 0.041526，mIoU 下降 0.008088。

最佳 checkpoint 状态：

- 残差系数：0.112343
- smoke head bias：-0.820807
- smoke head weight norm：0.672314

梯度隔离未消除 Smoke 与 Fire 的性能交换，说明核心问题不是辅助梯度泄漏，而是二值 `smoke-vs-rest` 目标本身没有显式维护 smoke / fire / background 三类之间的距离。

## 8. 判定

- `aux=0.2`：淘汰，不进入长训练。
- `aux=0.4`：证明显式烟雾监督能够提高 Smoke IoU，但 Fire IoU 损失过大，且十轮平均 mIoU 低于基线，因此暂不进入 100 轮训练。
- v2.1 梯度隔离：淘汰，不进入长训练。
- 二值 smoke-vs-rest 门控路线到此停止；当前不能把 v2 或 v2.1 宣称为最终改进模型。
- 不使用测试集为 v2 选择权重或版本。

下一版转向训练期三类别原型分离约束：同时建模 background / smoke / fire，并明确拉开 smoke-fire 距离；不再继续盲调 `SMOKE_AUX_WEIGHT`。可结合的轻量结构方案：

1. 引入训练期轻量类别原型分离约束，显式拉开 background / smoke / fire 特征。
2. 将空洞卷积分支替换或补充为轻量方向长核，增强烟羽方向与内部纹理建模。
3. 原型或辅助损失仅训练时使用，确保推理参数和 FLOPs 基本不增加。

## 9. 文件与校验

- `aux=0.4` 指标：`G:\py2\experiments\pidnet_s_lscm_v2\lscm_v2_10e_h100_w04\metrics.jsonl`
- `aux=0.2` 指标：`G:\py2\experiments\pidnet_s_lscm_v2\lscm_v2_10e_h100_w02\metrics.jsonl`
- v2.1 指标：`G:\py2\experiments\pidnet_s_lscm_v21\lscm_v21_10e_h100_w04\metrics.jsonl`
- 复杂度：`G:\py2\experiments\pidnet_s_lscm_v2\pipeline_check\complexity.json`
- `aux=0.4` 最佳 checkpoint SHA256：`0B8954EC06E649D35DC8CA9D4CE707FD7EAC25B40F6ECE8E4A49EBF0CB7B7EB6`
- `aux=0.2` 最佳 checkpoint SHA256：`AE0762368BEC3F87EED32019DCEA3DC7D0B0D7A677A693CEB1B339BF960D5B8B`
- 模型源码 SHA256：`6175E6DCC731B28A32DDE0A52B5E2791B3AFFD209F8B1AA0703C9325946E15A8`
- 损失源码 SHA256：`15D4C3DE8B5C72FF373CCC91CA9B9760CC4113F0E8D9EFE9D115A37422B93BAA`
- 训练脚本 SHA256：`462387FF1B01875019483376CFD91AFC13E59BE4398227557B8851536DE73784`
- 配置 SHA256：`925FFECF44D4A845F5065473EA0B87B48837B66F67B764EE1161689436A3568D`
- v2.1 最佳 checkpoint SHA256：`DDF8F76ED6BA399E9C29FE5E52154045810FCB6695F9DAE9C97C958451068C21`
- v2.1 模型源码 SHA256：`0305558BB6AF3631EBDFC732D1D8C1BB2C7A4722104D500B4805082521499BE1`
- v2.1 配置 SHA256：`FD9BC28DA5725C991B29D4BBAC14B0C87DFA2D18472F220BFC779327132D6D9E`
