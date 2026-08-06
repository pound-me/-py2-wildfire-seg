# FLAME3 CMRC 前置核验与准入记录

日期：2026-08-06

## 冻结范围

- 正式输入保持 Corrected FOV RGB 与 Raw Thermal JPG 灰度，不启用 Z-score 新基线。
- Celsius TIFF 仅用于诊断，不作为模型输入。
- 测试集封存；本阶段只读取 split v2 的 train/validation。
- 不恢复 MRFF，不启用 NTS，不叠加 ABL、SAMF 或其他结构。
- 在输入消融与人工错误审查同时通过前，不实现 `pidnet_s_cmrc`。

## 自动数据核验结果

### 审计范围修正

- 首次临时配准运行暴露出旧扫描器未受 split 白名单约束，其中包含 14 对 test 图像。
- 该临时结果已立即判为无效并删除，未用于训练、参数选择、模型筛选或正式结论。
- 正式脚本现强制传入非测试 split CSV，显式拒绝 `test.csv`，并记录 split 哈希。
- 下述配准结果已使用 train+validation 独立重跑，正式结果标记 `test_split_touched=false`。

### 预处理与增强

- 训练样本数：493。
- RGB 均值：`[0.516655, 0.516714, 0.500814]`。
- RGB 标准差：`[0.195641, 0.205705, 0.239463]`。
- Thermal JPG 均值/标准差：`0.109287 / 0.168490`。
- Thermal JPG 零值比例：`5.048%`；零值超过 50% 的图像：`9/493 (1.83%)`；255 饱和比例为 0。
- 100 张抽样中，JPG 灰度与 Celsius TIFF 的单帧 Pearson 中位数为 `0.993190`，跨帧合并 Pearson 为 `0.116050`。
- 32 次固定随机增强检查：RGB/IR 坐标最大误差为 0；标签 marker 质心最大偏移 `1.414 px`，最小 IoU `0.944751`。
- 结论：预处理与同步增强通过冻结门槛。

### 100 对配准审计（仅 train+validation）

- 固定 seed `20260806`，Fire/No Fire 各 50 对。
- train/val CSV 哈希已写入结果，`test_split_touched=false`。
- 可用估计 `54/100`，低于最低 60 对要求。
- 可用估计的平移 MAD：`dx=3.680 px`、`dy=1.675 px`。
- 旋转 MAD：`0.195°`；尺度 MAD：`x=0.00640`、`y=0.00682`。
- 相似度增益中位数：`0.08045`。
- 虽然相似度有提升，但可用样本数、平移稳定性和尺度稳定性均未通过冻结阈值。
- 结论：数值规则拒绝统一全局仿射；继续使用官方 Corrected FOV，不修改源数据。

### RGB/IR/Fusion 三 seed 输入消融

- 全部使用 epoch 26--30 validation Fire IoU 算术均值。
- RGB-only 三 seed 均值：`0.115512`。
- IR-only 三 seed均值：`0.509269`。
- Fusion 三 seed均值：`0.562542`。
- Fusion 相对 RGB 的同 seed 平均增益：`+0.447030`，`3/3` seed提高。
- Fusion 相对 IR 的同 seed 平均增益：`+0.053273`，`3/3` seed提高。
- split哈希、增强、loss、batch、LR总长度和ImageNet权重哈希协议核验通过。
- IR seed201曾在完成epoch 23后因窗口关闭中断，随后从 `last.pth` 精确恢复模型、优化器、AMP scaler、DataLoader generator及全部随机状态至epoch 30。
- 续训覆盖了 `environment.json` 中的pretrained字段；中断checkpoint的config、原始日志及权重SHA256共同证明其使用相同ImageNet初始化，证据写入 `resume_provenance.json`。
- 结论：输入互补性门槛通过。

## 待完成门槛

1. 完成冻结的 30 FN、20 FP、20 TP 人工清单：FN 配准错位少于 9 项，且至少 15 项 FN 属于小火/弱火/遮挡/边缘等 CMRC 假设可解释类型。
2. 自动输入互补性门槛已经通过；人工门槛通过后，才允许实现唯一冻结 CMRC 结构。

## 可复现实用工具

- `src/audit_flame3_registration.py`
- `src/audit_flame3_preprocessing.py`
- `src/export_flame3_manual_error_audit.py`
- `src/evaluate_flame3_manual_error_audit.py`
- `src/summarize_flame3_input_ablation.py`

当前状态：`input complementarity passed; CMRC awaits manual-review gate`。
