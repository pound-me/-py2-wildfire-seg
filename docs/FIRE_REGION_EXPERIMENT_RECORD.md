# Fire Region-Aware Loss 实验记录

## 1. 目的

火焰边界监督未能提高 Fire IoU，因此改为直接约束 PIDNet-S 主语义输出中的 fire-vs-rest 概率。该辅助目标仅训练时使用，不改变推理结构。

类别：背景 / 烟雾 / 火焰。测试集未用于模型选择。

## 2. 方法

- 基础网络：PIDNet-S。
- 二值 fire-vs-rest logit：火焰 logit 减去其他类别的 log-sum-exp。
- 平衡 Focal：火焰像素与非火焰像素分别求均值后等权平均。
- Tversky：alpha=0.3，beta=0.7，提高漏检火焰的代价。
- 辅助总损失：Focal + Tversky。
- 对比权重：0.1 和 0.05。
- 推理参数量、FLOPs 与原始 PIDNet-S 完全相同。

## 3. 真实批次检查

- ImageNet 预训练匹配张量：302。
- 总损失：16.323214。
- 火焰区域辅助损失：2.666037。
- Focal：1.895692。
- Tversky：0.770345。
- 火焰正像素：19,187。
- 仅辅助损失的语义头梯度范数：1.398468。
- 仅辅助损失的边界头梯度：0，隔离正确。
- 完整损失语义头梯度范数：2.914748。
- 完整损失边界头梯度范数：0.190249。

## 4. 管线与复杂度

- AMP、优化器、验证、checkpoint 和 metrics.jsonl 均通过。
- 单批次检查峰值显存：202.0 MB。
- 推理参数量：7,623,651。
- Forward FLOPs：2.946576 GFLOPs。
- GMACs：1.473288。
- 相对基线推理开销增量：0。

## 5. 公平 10 轮筛选

统一协议：seed=200，batch size=4，552/240 train/val，AMP，10 个实际训练 epoch，Poly 学习率总周期固定为 100 epoch。

### 5.1 按最佳 mIoU 比较

| 方法 | 权重 | Epoch | mIoU | Background IoU | Smoke IoU | Fire IoU |
|---|---:|---:|---:|---:|---:|---:|
| PIDNet-S baseline | - | 10 | 0.576613 | 0.836530 | 0.698519 | 0.194790 |
| Fire Region | 0.1 | 8 | 0.580931 | 0.837021 | 0.737184 | 0.168586 |
| Fire Region | 0.05 | 9 | 0.554957 | 0.824204 | 0.680723 | 0.159943 |

权重 0.1 相对基线最佳 mIoU：

- mIoU：+0.004318。
- Background IoU：+0.000491。
- Smoke IoU：+0.038665。
- Fire IoU：-0.026204。

### 5.2 按本方法最佳 Fire IoU 比较

| 权重 | Fire 最佳 Epoch | mIoU | Smoke IoU | Fire IoU |
|---:|---:|---:|---:|---:|
| 0.1 | 10 | 0.548984 | 0.613133 | 0.186392 |
| 0.05 | 8 | 0.528674 | 0.591735 | 0.173772 |

权重 0.1 的最佳 Fire IoU 比基线低 0.008398；权重 0.05 比基线低 0.021018。

权重 0.1 用时 439.0 秒，权重 0.05 用时 431.5 秒；两者峰值显存均为 240.7 MB。

实验目录：

- `G:\py2\experiments\pidnet_s_fire_region\fire_region_10e_h100_w01`
- `G:\py2\experiments\pidnet_s_fire_region\fire_region_10e_h100_w005`

## 6. 结论

权重 0.1 是两个设置中更好的版本。它在零推理开销条件下将最佳 mIoU 提高 0.004318，并同时提高 Background 和 Smoke IoU。Fire IoU 从第 1 轮的 0.1119 上升到第 10 轮的 0.1864，呈现继续收敛趋势，但在前 10 轮内仍未超过基线 0.1948。

权重减小到 0.05 后，总体 mIoU 和 Fire IoU 都变差，因此不再继续减小权重。

当前判断：

- 0.05：淘汰，保留为权重消融。
- 0.1：通过总体 mIoU 的初筛，且 Fire IoU 在第 10 轮仍处于上升阶段；可作为唯一候选进入 100 轮验证。
- 最终是否采用必须由公平 100 轮验证决定，不使用测试集挑选。

## 7. 公平 100 轮正式验证

正式运行目录：

`G:\py2\experiments\pidnet_s_fire_region\fire_region_100e_fair_w01`

训练完成 100 轮，无 stderr，耗时 4,339.0 秒，峰值显存 241.5 MB。

### 7.1 按最佳 mIoU 比较

| 方法 | Epoch | mIoU | Background IoU | Smoke IoU | Fire IoU |
|---|---:|---:|---:|---:|---:|
| PIDNet-S baseline | 40 | 0.601258 | 0.849239 | 0.753362 | 0.201175 |
| Fire Region w=0.1 | 74 | 0.579154 | 0.815687 | 0.744521 | 0.177253 |
| 差值 | - | -0.022105 | -0.033551 | -0.008840 | -0.023922 |

### 7.2 按最佳 Fire IoU 比较

| 方法 | Epoch | mIoU | Smoke IoU | Fire IoU |
|---|---:|---:|---:|---:|
| PIDNet-S baseline | 78 | 0.579510 | 0.676403 | 0.221641 |
| Fire Region w=0.1 | 71 | 0.539911 | 0.590032 | 0.195972 |
| 差值 | - | -0.039599 | -0.086371 | -0.025669 |

第 100 轮 Fire Region 指标：mIoU 0.569679，Smoke IoU 0.692221，Fire IoU 0.191186。

### 7.3 正式结论

10 轮筛选中的小幅 mIoU 优势没有在 100 轮正式训练中保持。正式基线在总体 mIoU、Background、Smoke 和 Fire 指标上均更好；本方法的最佳 Fire IoU 也低于基线 0.025669。

因此 Fire Region w=0.1 正式淘汰，不作为最终模型，不进入测试集评估。该实验保留为重要阴性结果：仅增加 fire-vs-rest 区域损失会改变烟雾与火焰的竞争关系，但不能稳定提高三分类语义分割性能。

## 8. SHA256

| 文件 | SHA256 |
|---|---|
| `src/custom_losses.py` | `CAB6F8E40FE25FFE85077535F59E03BE006FA0A78EF958F070C4EEF9DC55BD3D` |
| `src/train_baseline.py` | `9260D7822A47E27FF5ED07D335EB39038386354D6A64E74E2685F09F087CBCCA` |
| `configs/pidnet_s_fire_region.yaml` | `1AC41CDAD93F78DA24C71507A95CBC910BDCFF4FF361A450A5A8A41E32E31B76` |
| `configs/pidnet_s_fire_region_w005.yaml` | `64CEAD2D60360E70EC70F336766CC318F209ED437CA1DC89CB71B1C439513244` |
| `src/check_fire_region_training.py` | `7E89873AC90587139067AB13FC124C8108238FCB1AA839EA9F21AB0BDF45462D` |
| `w=0.1 metrics.jsonl` | `2A839ED56364056860EC968FBDC6343CD82968860D96C5D55DA27926413FA52E` |
| `w=0.1 best.pth` | `B3FCDFCB4510563AF9AF6F9F6878A19F44B71985600D5B513F02CAA90152A520` |
| `w=0.1 last.pth` | `A08D8B0B58BDD9BD48F14798721607362C770243FC13FFAB34CA9201F470349D` |
| `w=0.05 metrics.jsonl` | `D49EC201BAFB8B45BEA5C4417A3E6EAD85E5B10D6B02C37CF77A0B195460B869` |
| `w=0.05 best.pth` | `DA4ABC064DE2BD4D69A6B287141BAADC6280C80709C046331647ABE67FE943C6` |
| `w=0.05 last.pth` | `0D65ECFA999AF24611F376FCF7DA909FA30C6BE903DD6542F561683B9D724645` |
| `100e metrics.jsonl` | `972EBD798CDD99373549AA40035575B0A04837E2D976710EB7196BF4D80D1F06` |
| `100e best.pth` | `19903EC3C9F91FC1DEF23C3ED037ACC7FF2E036D7EA9CAAF4F6F642D2984CD3D` |
| `100e last.pth` | `EF035C95B1EA1A91B0DFA7E0A95297CC913878800382606A84E85D647249E47F` |
| `100e resolved_config.json` | `A61B35CC94F98D963A5470D919C7E2CC9E4015CCFC300454F82306FA53196F1B` |
