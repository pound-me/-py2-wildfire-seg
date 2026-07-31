# FLAME3 输入模态消融与公平Stem初始化预注册

日期：2026-07-31  
状态：用户批准；在RGB-only/IR-only精度训练前冻结

## 目的

补齐同协议的RGB-only、IR-only与RGB+IR Fusion对照，判断Fusion收益是否来自真实模态互补。该消融是输入基线，不作为结构创新。

## 公平初始化规则

官方PIDNet ImageNet权重的输入stem为3通道。默认按形状加载会导致RGB匹配302个张量，而IR/Fusion只匹配301个，产生首层预训练不公平。

冻结规则为：

```text
PRETRAIN_SKIP_KEYS = [conv1.0.weight]
```

- RGB、IR、Fusion的输入stem全部保留PIDNet构造时的Kaiming随机初始化；
- 其余301个骨干张量加载同一份官方ImageNet权重；
- 三种模式必须通过逐元素检查：预训练加载前后stem完全不变；
- 至少一个非stem骨干张量必须与官方checkpoint逐元素一致；
- 三种模式加载计数必须全部等于301。

当前Fusion正式基线虽未在旧配置中显式列出skip key，但其4通道stem因形状不匹配已自动跳过，实际同样只加载301个张量。因此新增显式字段是行为等价的审计增强，不触发Fusion重跑。

## 固定训练协议

- split v2；train/val 493/134；测试集封存；
- 部分标签损失及checkpoint选择规则不变；
- 640×512、物理batch 8、seed 200、AMP初始scale 128；
- 强亮度关闭，随机缩放0.8–1.5，水平翻转开启；
- 30轮使用100轮学习率周期；
- RTX 4090单任务顺序执行，先RGB、后IR；两者不得并行。

RGB与IR都是必须报告的输入基线，不依据某一方精度决定是否运行另一方。30轮只用于确认训练稳定性与形成阶段记录；只要无NaN/OOM、无实现错误，二者均从各自last checkpoint精确续训到100轮，避免性能选择偏差。

## 输出

- `flame3_rgb_partial_30e_seed200`；
- `flame3_ir_partial_30e_seed200`；
- 每组保存逐轮Fire IoU、precision/recall/F1、边界F1、空Fire/No Fire误报、环境与checkpoint；
- 100轮结束后与Fusion `best.pth`组成输入模态表；
- 不报告Smoke IoU或三分类mIoU，因为FLAME3没有完整Smoke真值。
