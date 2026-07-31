# FLAME3 split v2 冻结模型零样本评估结果

日期：2026-07-31  
状态：validation 只读评估完成；未训练；test 未读取  
正式输出：`G:\py2\experiments\flame3_zero_shot_split_v2_seed200`

## 1. v2 划分审计

- validation：99 张 Fire + 35 张 No Fire，共 134 张；
- 99 张 Fire 中：83 张具有非空温度活动火区伪标签，16 张为空伪标签；
- train/val/test 空伪标签 Fire 计数：62/16/2；
- 原 block 0 在最大 199 秒内部间隔处分成 `0a=16` 与 `0b=66`；
- 边界 train 侧 4 帧 `00499、00447、00089、00028` 被排除，保留的 `0b` 为 62 张；
- v1→v2：99 张 train→val、99 张 val→train、4 张 train→排除缓冲；
- No Fire 空间簇逐样本不变，test 成员与 v1 完全一致；
- 同一 300 秒时间块内，保留样本的跨集合相邻帧对为 0；
- 全局 EXIF 连续记录有 7 个集合切换对，但间隔全部大于 300 秒，均为自然时间块边界。

## 2. 固定评估协议

- 三个模型均为冻结的 FLAME2 seed 200、原验证集 mIoU-best checkpoint；
- 输入 640×512，batch 1，AMP，RTX 2060；
- Python 3.9.23，PyTorch 2.8.0+cu126；
- checkpoint 严格加载，不更新参数；
- `training_performed=false`，`test_touched=false`；
- FLAME3 当前无 Smoke 像素真值，禁止把 Smoke 预测比例解释为 Smoke IoU 或准确率。

## 3. 聚合结果

| 冻结 FLAME2 模型 | Fire IoU | Precision | Recall | F1 | No Fire 的 Fire 误报像素率 |
|---|---:|---:|---:|---:|---:|
| Fusion baseline seed200 | 0.1985 | 0.5281 | 0.2413 | 0.3312 | 0.0276% |
| SAMF seed200 | 0.2117 | 0.4595 | **0.2820** | 0.3495 | 0.1009% |
| ABL+SAMF seed200 | **0.2199** | **0.5443** | 0.2695 | **0.3605** | 0.1187% |

相对 Fusion baseline：

- SAMF：Fire IoU `+0.01325`，Recall `+0.04069`，但 Precision `-0.06867`；
- ABL+SAMF：Fire IoU `+0.02139`，Precision `+0.01618`，Recall `+0.02820`，F1 `+0.02926`；
- ABL+SAMF 在 v2 聚合 Fire IoU/F1 上排名第一；SAMF 的 Recall 最高，但误报更多。

## 4. 16 张空伪标签 Fire 帧诊断

| 模型 | Fire 预测像素率 | Smoke 预测像素率 |
|---|---:|---:|
| Fusion baseline | 1.8834% | 7.2401% |
| SAMF | 2.7354% | 16.4412% |
| ABL+SAMF | **1.7753%** | 15.4847% |

这里的 Fire 预测可作为空温度活动火区帧上的误报诊断。Smoke 没有像素真值，只记录模型响应，不下正确/错误结论。ABL+SAMF 在三者中具有最低的空伪标签 Fire 预测率，同时保持最高的聚合 Fire IoU。

## 5. 与 v1 的替代关系

v1 validation 的 99 张 Fire 全部具有非空伪标签；v2 则包含 83 张非空和 16 张空伪标签，并更换了完整时间块。因此 v1 与 v2 的绝对数值不能解释为同一验证集上的模型退化或提升。

- v1 输出保留：`G:\py2\experiments\flame3_zero_shot_seed200_modules_retry2`；
- v1 标记为 `superseded by split v2`，只用于划分修订审计；
- 从本记录起，正式 FLAME3 零样本引用统一使用 v2 结果；
- 模型权重、配置和推理脚本未因 v2 改动，变化来源仅为 validation 成员更新。

## 6. 结论

split v2 零样本闭环完成。三个冻结模型均显示明显的跨 FLAME2→FLAME3 域差异；在更完整覆盖空伪标签状态的 v2 validation 上，ABL+SAMF 是三者中最稳健的冻结模型。该结论只用于初始化诊断，不替代后续 FLAME3 训练基线。
