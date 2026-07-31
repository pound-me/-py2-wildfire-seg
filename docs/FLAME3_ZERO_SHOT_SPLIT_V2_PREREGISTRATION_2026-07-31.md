# FLAME3 split v2 零样本评估预注册

日期：2026-07-31  
状态：在 v2 推理前冻结

## 固定对象

只评估以下三个已冻结 FLAME2 seed 200、验证集 mIoU-best checkpoint：

1. Fusion baseline；
2. SAMF；
3. ABL+SAMF。

checkpoint、配置和权重哈希沿用 2026-07-30 的冻结 manifest，不因 split v2 改变。

## 固定协议

- 数据：`flame3_splits_v2_preregistered/val.csv`，99 张 Fire + 35 张 No Fire；
- 推理：640×512、物理 batch 1、AMP、单卡 RTX 2060；
- 只读推理，不训练、不更新 checkpoint、不读取 test；
- 标签只表示温度支持的活动火区：0=非伪火区、2=活动火区核心、255=忽略边界；
- 无 Smoke 像素真值，因此禁止报告 Smoke IoU 或三分类 mIoU；
- 主指标：Fire IoU、precision、recall、F1；Smoke 预测比例仅作诊断；
- 输出到新目录 `experiments/flame3_zero_shot_split_v2_seed200`，不得覆盖 v1 输出；
- v1 结果保留但标记为被 v2 替代。

## 解释纪律

split v2 validation 新增 16 张空温度伪标签 Fire 帧，因此聚合指标可能与 v1 变化。该变化首先解释为验证样本状态覆盖改变，而不是模型权重或方法性能改变。模型间比较只在同一个 v2 validation 内进行。
