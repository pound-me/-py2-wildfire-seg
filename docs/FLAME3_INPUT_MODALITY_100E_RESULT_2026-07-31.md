# FLAME3 输入模态消融100轮最终结果

日期：2026-07-31  
状态：RGB、IR、Fusion三模态公平对照完成  
测试集：封存，未读取

## 固定协议

- split v2、部分标签损失、640×512、物理batch 8、seed 200；
- 三种输入stem均Kaiming随机初始化，其余301个骨干张量加载同一ImageNet权重；
- 100轮polynomial学习率周期；
- validation Fire IoU选择`best.pth`；
- 单独占用RTX 4090，正式结果均来自用户可见的前台PowerShell；
- Smoke无完整真值，不报告Smoke IoU或三分类mIoU。

## 最佳checkpoint对比

| 输入 | 最佳epoch | Fire IoU | Precision | Recall | Fire F1 | Boundary F1 | 后10轮IoU均值±std |
|---|---:|---:|---:|---:|---:|---:|---:|
| RGB | 95 | 0.151406 | 0.164655 | 0.652967 | 0.262993 | 0.393481 | 0.147751 ± 0.003143 |
| IR | 47 | 0.573053 | 0.609146 | 0.906292 | 0.728587 | 0.809503 | 0.480808 ± 0.042336 |
| Fusion | 63 | **0.639680** | **0.669306** | **0.935283** | **0.780250** | **0.819273** | **0.607448 ± 0.007177** |

Fusion相对IR：

- Fire IoU：`+0.066628`；
- precision：`+0.060160`；
- recall：`+0.028991`；
- Fire F1：`+0.051663`；
- Boundary F1：`+0.009770`；
- 后10轮均值：`+0.126640`；
- 后10轮波动显著更小。

IR相对RGB Fire IoU提高`0.421647`，说明热信息是当前烟遮挡火场识别活动火区的主要来源。Fusion又稳定优于IR，说明RGB虽然单独识别能力弱，但与IR联合时仍提供有用的空间、纹理或上下文补充。

## 泛化与过拟合

- RGB最佳epoch 95，train Fire IoU约`0.0936`而validation约`0.1514`；模型总体欠拟合/可分性不足，不能仅靠延长训练解决；
- IR最佳epoch 47，train/validation间隙约`0.2150`，后10轮标准差`0.0423`，存在明显过拟合和不稳定；
- Fusion最佳epoch 63，train/validation间隙约`0.1507`，后10轮标准差仅`0.0072`，性能和稳定性均优于IR。

正式比较一律使用各自`best.pth`，不得使用末轮权重。

## Checkpoint SHA256

RGB：

```text
452D312BBDE6A28A31A011E5CC6F41D18610106121BEDD24844585AFA78AA6DF
```

IR：

```text
FC39F33F75A9BF4DDABBF9D6687C29E18D24E47E858A0586DED85270FACB8FFC
```

Fusion：

```text
2A3911573C067620270EF5AC185DD100B9463BD2929FC20B750E30D0A707EFB3
```

## 论文动机结论

在当前单场次FLAME3 split v2上，可以形成以下证据链：

1. 大面积烟遮挡使RGB-only难以分离活动火区；
2. IR显著恢复活动火区可见性，是主要信息来源；
3. RGB+IR Fusion不仅超过RGB，也稳定超过IR，证明双模态融合具有实际价值；
4. 当前Fusion仍是简单四通道拼接，因而“如何在轻量条件下更有效地利用RGB上下文与热信息”可以作为后续方法创新入口。

该结论限于当前validation和单场次数据；测试集及跨场次泛化结论继续封存。

## 下一步

在设计新模块前，对Fusion最佳checkpoint执行只读验证集错误画像：

- Fire FN/FP及precision-recall分解；
- 小连通域、边界与区域内部误差；
- 空伪标签Fire图和No Fire图误报；
- RGB低可见、IR高温区域中的错误分布；
- 按错误类型确定后续融合模块，而不是先选模块再解释。
