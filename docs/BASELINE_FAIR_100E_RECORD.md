# PIDNet-S 公平100轮正式基线

记录日期：2026-07-24  
状态：后续所有消融实验的正式对照组。

## 公平协议

- Seed：200
- Train/Validation/Test：552 / 240 / 200
- 输入：RGB，256 x 256
- Batch size：4
- Epochs：100
- SGD，初始学习率0.001，Polynomial decay
- AMP：开启
- 训练集 shuffle 使用独立 `torch.Generator(seed=200)`
- cuDNN benchmark=False，deterministic=True
- 最佳模型由验证集 mIoU 选择
- 测试集只在模型冻结后评估

## 验证集最佳结果

最佳轮次：40

| 指标 | 数值 |
|---|---:|
| mIoU | 0.601258 |
| Background IoU | 0.849239 |
| Smoke IoU | 0.753362 |
| Fire IoU | 0.201175 |

## 测试集结果

| 类别 | IoU | Dice/F1 | Precision | Recall |
|---|---:|---:|---:|---:|
| Background | 0.890328 | 0.941982 | 0.969867 | 0.915656 |
| Smoke | 0.771610 | 0.871084 | 0.858771 | 0.883754 |
| Fire | 0.156410 | 0.270509 | 0.227243 | 0.334125 |

| 汇总指标 | 数值 |
|---|---:|
| mIoU | 0.606116 |
| Mean Dice | 0.694525 |
| Pixel Accuracy | 0.871356 |
| Macro Precision | 0.685294 |
| Macro Recall | 0.711179 |

## Boundary F1@3px

| 类别 | Boundary Precision | Boundary Recall | Boundary F1 |
|---|---:|---:|---:|
| Smoke | 0.705542 | 0.286417 | 0.407435 |
| Fire | 0.393692 | 0.208197 | 0.272361 |
| Mean | - | - | 0.339898 |

## 轻量化指标

- 部署参数量：7.623651M
- Forward FLOPs：2.946576 GFLOPs
- 估算 MACs：1.473288 GMACs
- RTX 2060，batch=1，AMP，256 x 256
- 延迟中位数：31.710 ms
- FPS 中位数：31.54
- 峰值 allocated memory：58.824 MB

速度仅统计输入已在 GPU 后的主分割网络前向，不含文件读取、预处理和后处理。

## 文件

- 最佳模型：`G:\py2\experiments\pidnet_s_rgb_baseline\baseline_100e_fair\best.pth`
- 最佳模型 SHA256：`1823E4FB1E979C869240C7264B6DF8BFF43DC4092636704DE1F8228F2318B347`
- 训练指标：`G:\py2\experiments\pidnet_s_rgb_baseline\baseline_100e_fair\metrics.jsonl`
- 测试指标：`G:\py2\experiments\pidnet_s_rgb_baseline\baseline_100e_fair\test_best\metrics.json`
- 逐图指标：`G:\py2\experiments\pidnet_s_rgb_baseline\baseline_100e_fair\test_best\per_image_metrics.csv`
- 对比图：`G:\py2\experiments\pidnet_s_rgb_baseline\baseline_100e_fair\test_best\comparisons`

## 使用规则

后续模型必须采用相同数据划分、随机种子、独立 shuffle generator、训练轮数和评估工具。先在验证集完成模型选择；没有超过本基线的候选模型不得通过测试集挑选数字。
