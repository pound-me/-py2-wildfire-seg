# FLAME3 输入模态消融30轮结果

日期：2026-07-31  
状态：RGB、IR与Fusion均完成30轮；批准RGB与IR精确续训到100轮  
测试集：封存，未读取

## 公平协议

- 三模态输入stem均为Kaiming随机初始化；
- 三者各加载相同301个ImageNet骨干张量；
- split v2、部分标签损失、640×512、batch 8、seed 200、AMP与100轮学习率周期相同；
- checkpoint均按validation Fire IoU选择；
- Smoke无完整真值，不报告Smoke IoU或三分类mIoU。

## 30轮结果

| 输入 | 最佳epoch | Fire IoU | Precision | Recall | Fire F1 | Boundary F1 | epoch 26–30均值±std |
|---|---:|---:|---:|---:|---:|---:|---:|
| RGB | 28 | 0.118029 | 0.124614 | 0.690747 | 0.211138 | 0.378758 | 0.111535 ± 0.005073 |
| IR | 22 | 0.548908 | 0.595627 | 0.874971 | 0.708768 | 0.784984 | 0.499280 ± 0.024079 |
| Fusion | 28 | 0.584992 | 0.600124 | 0.958678 | 0.738164 | 0.774017 | 0.564110 ± 0.015862 |

30轮阶段：

- IR相对RGB Fire IoU提高`0.430879`，证明当前火场中热信息是活动火区识别的主要来源；
- Fusion相对IR提高`0.036083`，主要同时体现为更高recall（`+0.083706`）和更高Fire F1（`+0.029396`）；
- Fusion的Boundary F1略低于IR约`0.010967`，需要在100轮最佳checkpoint上重新比较，不能用30轮单点定论；
- RGB结果不是“模块失败”，而是输入模态动机证据。

## Checkpoint审计

RGB正式30轮run：`flame3_rgb_partial_30e_seed200_retry3`

```text
best.pth SHA256 = 7BC8FC6285220587D861B3BE61C3B129861543DE1AD32B15661DCCC40C635A7F
last.pth SHA256 = 1B442BD68219E12C62F7E7CA6D5B56A9DE792BED789A5447168B7AD00A9B81C8
```

IR正式30轮run：`flame3_ir_partial_30e_seed200_retry1`

```text
best.pth SHA256 = 96D95D0243FAC22607D8FB5BB51CBACB49B787BC57297018E8D1C264F2ABC5B5
last.pth SHA256 = 23390F6F404F3F6DCB432A66997A015D2AECB1FADDF53E97CD309BE19F3F7C77
```

## 决策

RGB与IR均无NaN/OOM、无类别实现崩溃并完整保存30轮。依据预注册，二者必须分别从各自`last.pth`恢复模型、优化器、AMP与全部随机状态，续训epoch 31–100。100轮完成前不把30轮差异作为最终论文数值。
