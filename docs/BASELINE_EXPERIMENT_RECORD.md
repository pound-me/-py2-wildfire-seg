# PIDNet-S RGB FLAME2 基线实验档案

记录日期：2026-07-24  
用途：论文基线、消融实验和复现实验的统一取数来源。

## 1. 任务与数据

- 任务：无人机 RGB 森林火灾三分类语义分割
- 类别：background、smoke、fire
- 数据集：FLAME2
- 样本总数：992
- 训练集：552
- 验证集：240
- 测试集：200
- RGB、IR、GT 缺失文件：0
- 输入尺寸：256 x 256
- 测试集在本次基线模型冻结后评估，不参与训练和最佳轮次选择。

## 2. 模型与训练配置

- 基线：PIDNet-S
- 输入模态：RGB
- ImageNet 预训练权重：`G:\py2\weights\PIDNet_S_ImageNet.pth.tar`
- 成功匹配的预训练张量：302
- 随机种子：200
- 正式训练轮数：100
- Batch size：4
- 优化器：SGD
- 初始学习率：0.001
- 学习率策略：Polynomial decay，power=0.9
- Momentum：0.9
- Weight decay：0.00001
- AMP：开启
- 训练增强：多尺度、水平翻转、亮度变化
- 类别权重：background=2.27681343，smoke=3.04257527，fire=8.8657764
- 最佳模型选择标准：验证集 mIoU
- 最佳轮次：54
- 正式训练命令：

```powershell
& "F:\anaconda3\envs\pytorch\python.exe" "G:\py2\src\train_baseline.py" --config "G:\py2\configs\pidnet_s_rgb_baseline.yaml" --epochs 100 --run-name baseline_100e --amp
```

说明：这次已经生成的 checkpoint 内嵌配置中 `EPOCHS` 仍为旧值 1，因为当时命令行覆盖值没有回写到配置字典。实际训练为 100 轮，可由 `metrics.jsonl` 的 100 条记录和 `last.pth` 的 epoch=100 共同确认。训练程序现已修正，后续 checkpoint 会保存实际轮数。

## 3. 验证集最佳结果

最佳模型：`G:\py2\experiments\pidnet_s_rgb_baseline\baseline_100e\best.pth`

| 指标 | 数值 |
|---|---:|
| mIoU | 0.593317 |
| Mean Dice | 0.693058 |
| Pixel Accuracy | 0.833276 |
| Background IoU | 0.851162 |
| Smoke IoU | 0.751151 |
| Fire IoU | 0.177637 |

## 4. 测试集结果

测试样本数：200

| 类别 | IoU | Dice/F1 | Precision | Recall |
|---|---:|---:|---:|---:|
| Background | 0.883805 | 0.938319 | 0.971283 | 0.907519 |
| Smoke | 0.757950 | 0.862312 | 0.855632 | 0.869097 |
| Fire | 0.120525 | 0.215122 | 0.173230 | 0.283737 |

| 汇总指标 | 数值 |
|---|---:|
| mIoU | 0.587427 |
| Mean Dice | 0.671918 |
| Pixel Accuracy | 0.858816 |
| Macro Precision | 0.666715 |
| Macro Recall | 0.686784 |

测试集混淆矩阵，行为真实类别，列为预测类别：

```text
[[7785438,   84902, 708475],
 [ 116660, 3193444, 364337],
 [ 113522,  453919, 224783]]
```

## 5. 轻量化与速度

测试设备：NVIDIA GeForce RTX 2060  
输入：1 x 3 x 256 x 256  
精度模式：AMP  
预热：30 次  
每组计时：100 次  
独立测速：2 次，每次 5 组，共 10 组  
范围：输入已在 GPU，仅统计主分割输出的网络前向，不含磁盘读取、预处理和后处理。

| 指标 | 数值 |
|---|---:|
| 训练模型参数量，含辅助头 | 7,716,807 / 7.717M |
| 部署模型参数量，仅主输出 | 7,623,651 / 7.624M |
| 单张前向延迟中位数 | 28.971 ms |
| 单张前向延迟均值 | 28.842 ± 2.544 ms |
| FPS 中位数 | 34.52 |
| FPS 均值 | 34.93 ± 2.97 |
| 峰值 GPU allocated memory | 58.824 MB |

| 计算量指标 | 数值 |
|---|---:|
| PyTorch forward FLOPs | 2.9466 GFLOPs |
| 对应估算 MACs | 1.4733 GMACs |

FLOPs 使用 `torch.profiler(with_flops=True)` 测量，乘法和加法分别计为一次浮点运算。之后所有模型必须使用相同工具和相同输入尺寸测量。

## 6. 边界质量

边界指标使用 256 x 256 标签和 3 像素匹配容差，统计整个测试集。

| 类别 | Boundary Precision | Boundary Recall | Boundary F1@3px |
|---|---:|---:|---:|
| Smoke | 0.727752 | 0.251199 | 0.373482 |
| Fire | 0.364872 | 0.186045 | 0.246435 |
| Mean | - | - | 0.309959 |

## 7. 软件与代码版本

- Python：3.9.23
- PyTorch：2.8.0+cu126
- CUDA runtime：12.6
- cuDNN：9.1.0.2
- NumPy：1.26.4
- Pillow：11.3.0
- PIDNet 官方仓库 commit：`4c158cf24ce432f0a8cb43364fae38d93cee0dc3`
- RoboFireFuseNet 官方仓库 commit：`0d8ec502da0bafea7c388a989650aa53d1ecf278`

## 8. 文件校验值

- ImageNet 预训练权重 SHA256：`F96E2C96B1ACA1400A6F54AC41093D98B4817F6008A5D869ABE6248551A5F359`
- 最佳基线模型 SHA256：`A47AF6B710A53BD562BE774101BBE6BA8B0504677FC90C7B3F8638961F266489`

## 9. 结果文件

- 训练曲线原始数据：`G:\py2\experiments\pidnet_s_rgb_baseline\baseline_100e\metrics.jsonl`
- 测试总体指标：`G:\py2\experiments\pidnet_s_rgb_baseline\baseline_100e\test_best\metrics.json`
- 测试逐图指标：`G:\py2\experiments\pidnet_s_rgb_baseline\baseline_100e\test_best\per_image_metrics.csv`
- 测试对比图：`G:\py2\experiments\pidnet_s_rgb_baseline\baseline_100e\test_best\comparisons`
- 测试彩色预测：`G:\py2\experiments\pidnet_s_rgb_baseline\baseline_100e\test_best\predictions_color`
- 测试原始类别掩膜：`G:\py2\experiments\pidnet_s_rgb_baseline\baseline_100e\test_best\predictions_raw`
- FLOPs/MACs：`G:\py2\experiments\pidnet_s_rgb_baseline\baseline_100e\test_best\complexity.json`
- 两次独立测速历史：`G:\py2\experiments\pidnet_s_rgb_baseline\baseline_100e\test_best\speed_benchmark_history.json`

## 10. 当前结论与待补数据

当前基线能够较好分割烟雾，但火焰精度、召回率和边界质量明显不足。后续模块应围绕烟雾上下文和火焰边界设计，同时严格记录新增参数量、FLOPs、FPS和延迟。

论文定稿前仍需补充：

1. 至少 3 个随机种子的均值与标准差。
2. 基线、单模块和双模块的统一消融表。
3. 与其他轻量分割网络在相同数据划分和训练协议下的公平对比。
