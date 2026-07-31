# FLAME3 PIDNet-S Fusion 部分标签30轮筛选结果

日期：2026-07-31  
状态：30轮管线通过；批准按原100轮学习率周期从epoch 30精确续训  
测试集状态：封存，未读取  
源码：`2af80a2c79a8504de8175325aab4af954c5b3523`

## 固定协议

- 模型：PIDNet-S fusion，RGB+IR四通道；
- 数据划分：FLAME3 split v2，train/val为493/134；
- 分辨率：`640×512`；物理batch：8；seed：200；
- 训练目标：预注册部分标签损失；
- checkpoint选择：validation Fire IoU；
- 30轮筛选沿用100轮 polynomial 学习率周期；
- GPU：RTX 4090；AMP初始scale 128。

## 结果

- 最佳epoch：28；
- 最佳validation Fire IoU：`0.584992`；
- Fire precision / recall / F1：`0.600124 / 0.958678 / 0.738164`；
- Fire boundary F1：`0.774017`；
- No Fire图片Fire误报像素率：`0.000000`；
- 空伪标签Fire图片Fire预测像素率：`0.048982`；
- 第26–30轮Fire IoU：均值`0.564110`，样本标准差`0.015862`，范围`0.036009`；
- 总耗时：`1059.73 s`（约17.7分钟）；
- 峰值allocated显存：`1220.35 MiB`。

作为管线可行性参照，split v2冻结Fusion模型的零样本Fire IoU为`0.1985`；训练后最佳值提高约`0.3865`。该差异只证明FLAME3域内训练有效，不作为方法创新收益。

## 波动与过拟合诊断

最佳epoch 28时，train/validation Fire IoU约为`0.6958/0.5850`；epoch 30约为`0.7159/0.5490`。训练指标继续上升而验证指标回落，表明已经出现早期泛化间隙和较明显的逐轮波动。

这不构成中止100轮的理由：30轮筛选预先固定为100轮学习率周期，epoch 30时学习率尚未完成衰减；`best.pth`持续按validation Fire IoU保存，后续退化不会覆盖当前最佳模型。最终结论必须基于完整曲线和冻结选择规则，不能改用末轮指标。

## 决策

1. 30轮训练管线、部分标签目标、AMP与batch工程验收通过；
2. 从`last.pth`恢复模型、优化器、AMP scaler、DataLoader generator和Python/NumPy/Torch/CUDA随机状态，继续epoch 31–100；
3. 使用同一run目录追加指标，保持seed、物理batch、数据增强、学习率周期和选择指标不变；
4. 不重新随机起跑，不依据测试集作任何决定；
5. 100轮结束后分析最佳epoch、后50轮稳定性、过拟合间隙和Fire误报，再决定后续模块筛选。

## 数据迁移记录

训练完成后，完整bundle由C盘迁移到：

```text
D:\qianpengcheng\7.31\flame3_4090_bundle_v1_20260731
```

基础清单3172个文件中，除8个被后续split v2/代码更新有意覆盖的文件外，其余全部通过；最新更新清单97个文件全部通过SHA256校验。C盘旧副本随后由用户删除。历史`run_summary.json`中的C盘路径仅记录训练发生时的绝对路径，不影响D盘checkpoint；续训时以命令行覆盖为D盘根目录。
