# Fire Boundary-Aware Loss 实验记录

## 1. 目的

在不改变 PIDNet-S 推理结构的前提下，复用已有 D 分支的一通道边界头，增加针对火焰类别的训练专用边界监督，尝试提高 FLAME2 RGB 三分类任务中的 Fire IoU。

类别定义：背景 / 烟雾 / 火焰。测试集未参与模型选择。

## 2. 方法

- 基础网络：PIDNet-S。
- 从语义标签中提取类别 2 的内部火焰边界。
- 火焰边界辅助损失：batch 内正负平衡 BCE + 0.5 × soft Dice。
- 总损失中的辅助权重：0.2。
- 仅训练时使用；推理网络、参数量和 FLOPs 与基线完全相同。

## 3. 真实批次检查

- ImageNet 预训练匹配张量：302。
- 总损失：16.192389。
- 火焰边界辅助损失：0.678894。
- BCE：0.211729。
- Dice：0.934328。
- 火焰边界正像素：7,282。
- 仅辅助损失的边界头梯度范数：0.064876。
- 完整损失的边界头梯度范数：0.203099。
- 完整损失的语义头梯度范数：2.802979。

结论：损失有限，火焰边界目标存在，辅助梯度和主语义梯度均正常。

## 4. 完整管线检查

- 训练和验证各 1 个 batch。
- AMP、优化器、验证指标、best/last checkpoint 和 metrics.jsonl 均正常。
- 峰值显存：192.4 MB。
- 验证 batch 没有火焰边界时，辅助损失按设计安全归零。

## 5. 复杂度

输入尺寸统一为 1 × 3 × 256 × 256，使用 `torch.profiler(with_flops=True)`。

| 方法 | 推理参数量 | Forward GFLOPs | GMACs |
|---|---:|---:|---:|
| PIDNet-S baseline | 7,623,651 | 2.946576 | 1.473288 |
| Fire Boundary-Aware Loss | 7,623,651 | 2.946576 | 1.473288 |
| 增量 | 0 | 0 | 0 |

## 6. 公平 10 轮筛选

协议：seed=200，batch size=4，552/240 train/val，AMP 开启，训练 10 轮，Poly 学习率总周期固定为 100 轮。

实验目录：

`G:\py2\experiments\pidnet_s_fire_boundary\fire_boundary_10e_h100_w02`

### 6.1 按最佳 mIoU 选取

| 方法 | Epoch | mIoU | Background IoU | Smoke IoU | Fire IoU |
|---|---:|---:|---:|---:|---:|
| PIDNet-S baseline 前 10 轮 | 10 | 0.576613 | 0.836530 | 0.698519 | 0.194790 |
| Fire Boundary-Aware Loss | 8 | 0.578028 | 0.826092 | 0.761369 | 0.146622 |
| 差值 | - | +0.001415 | -0.010438 | +0.062850 | -0.048168 |

### 6.2 本方法的火焰最佳轮次

第 10 轮 Fire IoU 最高：

- mIoU：0.548358。
- Smoke IoU：0.625471。
- Fire IoU：0.180571。

仍低于基线第 10 轮 Fire IoU 0.194790，差值为 -0.014219。

训练总用时 493.6 秒，峰值显存 232.3 MB。

## 7. 结论

该方法实现了严格的零推理开销，并将最佳 mIoU 略微提高 0.001415，但提升完全来自 Smoke IoU；最佳 mIoU 轮次的 Fire IoU 反而下降 0.048168。即使按本方法火焰表现最好的第 10 轮计算，Fire IoU 仍未超过基线。

因此：

- 工程实现正确。
- 零推理开销结论成立。
- 未实现提高 Fire IoU 的核心目标。
- 不进入 100 轮正式训练，保留为阴性消融实验。

结果说明一通道、类别无关的 D 分支主要学习“是否存在边界”。额外强调火焰边界并不能充分迫使主语义头将边界内部像素识别为火焰。下一步应直接约束主语义输出中的火焰区域，例如采用 fire-vs-rest 的区域 Tversky/Focal 目标，同时继续保持训练专用和零推理开销。

## 8. SHA256

| 文件 | SHA256 |
|---|---|
| `src/custom_losses.py` | `298FEFDA7C608E484BC49613F6B979EC2ECD13681172C596502F2DEB674EE0E3` |
| `src/train_baseline.py` | `BBAAD454E23AB679A3C977AC94540F52615C50AACF0761DBEB6F0580C33B55C2` |
| `configs/pidnet_s_fire_boundary.yaml` | `51488766FE0FFECAC8C16DFF1AF1711C5958B7EBE284F655B8F6C6415126BE55` |
| `src/check_fire_boundary_training.py` | `30B9FA5B659586DCDB1C1714382C1750D376975F8E27DB7FAA45EB3D47F55E95` |
| `metrics.jsonl` | `DA113ECEA00F7DD0EAC5AA5CDB6F71514302765A3EC30B819CC4A9C391B7F0F1` |
| `best.pth` | `78FB21940E75ED6F786B55F68BC7AC5B263C615A78798C8F4ACC5F3913F95CD5` |
| `last.pth` | `2DF4A8D88BCE1C64D3C9AEA73B36FD1026A57FCE2F22029D90A4E85CCDD90982` |
