# LSCM v1 十轮筛选实验档案

记录日期：2026-07-24  
状态：十轮筛选通过，但公平100轮实验未超过基线；v1 不进入测试集和论文最终模型。

## 1. 模块设计

模型名称：PIDNet-S-LSCM  
模块名称：Lightweight Smoke Context Module  
插入位置：PIDNet I 分支的 PAPPM 上采样输出之后、Light-Bag 三分支融合之前。  
输入特征：128 通道，约为输入图像的 1/8 分辨率。

结构：

```text
128通道I分支特征
  -> 1x1降维到32通道
  -> 并行深度卷积，dilation=1/2/3
  -> 拼接与1x1融合
  -> 单通道烟雾空间门控
  -> 可学习缩放残差
```

官方 `third_party` 代码未修改。自定义实现：

`G:\py2\src\custom_models\pidnet_lscm.py`

## 2. 轻量化开销

输入尺寸：1 x 3 x 256 x 256  
FLOPs 统计：`torch.profiler(with_flops=True)`，乘法和加法分别计数。

| 指标 | PIDNet-S | PIDNet-S-LSCM | 增量 | 增幅 |
|---|---:|---:|---:|---:|
| 部署参数量 | 7,623,651 | 7,641,541 | 17,890 | 0.2347% |
| Forward GFLOPs | 2.946576 | 2.982295 | 0.035718 | 1.2122% |
| GMACs | 1.473288 | 1.491147 | 0.017859 | 1.2122% |

该模块满足参数和 FLOPs 增幅均低于 5% 的筛选预算。

## 3. 公平训练协议

- 数据划分：552 train / 240 validation
- 输入：RGB，256 x 256
- ImageNet 预训练：相同，均匹配 302 个张量
- Seed：200
- Batch size：4
- Epochs：10
- Optimizer：SGD
- 初始学习率：0.001
- Scheduler：Polynomial decay，power=0.9
- AMP：开启
- 数据增强：相同
- 训练集 shuffle：独立 `torch.Generator(seed=200)`
- cuDNN：benchmark=False，deterministic=True
- 最佳模型选择：验证集 mIoU

旧的 `baseline_10e` 和 `lscm_10e` 使用全局随机状态控制 shuffle，不作为严格消融对照。正式十轮对照只使用带 `_fair` 后缀的两次实验。

## 4. 各自最佳 mIoU 结果

| 模型 | 最佳轮次 | mIoU | Background IoU | Smoke IoU | Fire IoU |
|---|---:|---:|---:|---:|---:|
| PIDNet-S fair | 7 | 0.499423 | 0.765930 | 0.578466 | 0.153872 |
| PIDNet-S-LSCM fair | 6 | 0.562332 | 0.819172 | 0.722542 | 0.145281 |
| 绝对变化 | - | +0.062909 | +0.053242 | +0.144075 | -0.008591 |

相对基线 mIoU 提升约 12.60%。

## 5. 相同第6轮结果

| 模型 | Validation Loss | mIoU | Background IoU | Smoke IoU | Fire IoU |
|---|---:|---:|---:|---:|---:|
| PIDNet-S fair | 6.850248 | 0.481741 | 0.682038 | 0.623498 | 0.139686 |
| PIDNet-S-LSCM fair | 6.259971 | 0.562332 | 0.819172 | 0.722542 | 0.145281 |
| 绝对变化 | -0.590277 | +0.080591 | +0.137133 | +0.099044 | +0.005595 |

相同轮次下，LSCM 的三类 IoU 均未下降。

## 6. 类别峰值

| 指标 | PIDNet-S fair | PIDNet-S-LSCM fair | 变化 |
|---|---:|---:|---:|
| 最大 Smoke IoU | 0.634357，epoch 3 | 0.722542，epoch 6 | +0.088185 |
| 最大 Fire IoU | 0.153872，epoch 7 | 0.167032，epoch 8 | +0.013160 |

LSCM 最佳 checkpoint 中可学习残差缩放值为 0.085345，初始值为 0.1。

## 7. 十轮阶段结论

LSCM v1 在严格相同的十轮训练协议下显著提高了烟雾 IoU 和总体 mIoU，同时参数量仅增加 0.2347%，FLOPs 仅增加 1.2122%。因此当时通过筛选并进入正式长训练。该结论只适用于十轮筛选，最终取舍以公平100轮实验为准。

当前不能直接与旧 `baseline_100e` 做严格消融，因为旧实验的 shuffle 尚未使用独立 generator。正式论文对比需要在固定协议下分别训练 PIDNet-S 和 PIDNet-S-LSCM。

## 8. 公平100轮结果

| 模型 | 最佳轮次 | mIoU | Background IoU | Smoke IoU | Fire IoU |
|---|---:|---:|---:|---:|---:|
| PIDNet-S fair | 40 | 0.601258 | 0.849239 | 0.753362 | 0.201175 |
| PIDNet-S-LSCM v1 fair | 71 | 0.592956 | 0.849803 | 0.745791 | 0.183275 |
| LSCM v1 变化 | - | -0.008302 | +0.000564 | -0.007571 | -0.017900 |

类别峰值：

| 指标 | PIDNet-S fair | PIDNet-S-LSCM v1 fair | 变化 |
|---|---:|---:|---:|
| 最大 Smoke IoU | 0.757929，epoch 19 | 0.766067，epoch 22 | +0.008138 |
| 最大 Fire IoU | 0.221641，epoch 78 | 0.209866，epoch 78 | -0.011774 |

100轮逐轮平均差异，LSCM v1 减去基线：

- mIoU：-0.014970
- Smoke IoU：-0.020512
- Fire IoU：-0.010861
- 100轮中仅33轮的 mIoU 高于同轮基线

可学习残差缩放值从初始 0.1 降至：

- 最佳 checkpoint：0.052744
- 最后一轮 checkpoint：0.050701

这说明网络在长训练中主动减弱该模块。LSCM v1 的烟雾峰值偶尔较高，但综合性能、火焰性能和逐轮平均性能均未超过基线。

## 9. 最终判定

LSCM v1 被淘汰，不使用测试集挑选结果，也不作为论文最终创新模块。十轮筛选的提升属于短调度下的早期现象，不能替代正式长训练结论。

下一版需要解决“门控没有明确烟雾约束”的问题。建议 v2 使用显式 smoke-aware 辅助监督，使门控预测二值烟雾区域，并将辅助损失仅用于训练；推理阶段仍保持轻量。

## 10. 文件与校验

- 公平基线指标：`G:\py2\experiments\pidnet_s_rgb_baseline\baseline_10e_fair\metrics.jsonl`
- 公平 LSCM 指标：`G:\py2\experiments\pidnet_s_lscm\lscm_10e_fair\metrics.jsonl`
- 公平基线最佳模型 SHA256：`27F150F51FE3103A237D5B79B8E49E27DEBB765DE013BAC3CDB903F932521885`
- 公平 LSCM 最佳模型 SHA256：`06E0164B41CA8460A9B8B4BF0079D8895EA8DB644FE0B3109E59378E641408CD`
- LSCM 源码 SHA256：`A224660BE3914A96096E8BD89293E9045DEEC333B5A810C8CA013653F3BCC9DE`
- LSCM 配置 SHA256：`CA840E7669C2708051B5F43E0626F6B140CD463BE2A2FF7B84DE0989A61EAE4A`
- 公平100轮基线指标：`G:\py2\experiments\pidnet_s_rgb_baseline\baseline_100e_fair\metrics.jsonl`
- 公平100轮 LSCM v1 指标：`G:\py2\experiments\pidnet_s_lscm\lscm_100e_fair\metrics.jsonl`
- 公平100轮基线最佳模型 SHA256：`1823E4FB1E979C869240C7264B6DF8BFF43DC4092636704DE1F8228F2318B347`
- 公平100轮 LSCM v1 最佳模型 SHA256：`DAD9B3172D8A1201FAB9C8D403B6594E2732C924360B266891C4D7219F1101A1`
