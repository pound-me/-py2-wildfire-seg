# Route C 模块二：TGM 设计与工程验收

日期：2026-07-27

状态：工程验收通过；允许唯一主变体进入30轮筛选。

## 1. 路线边界

TGM是独立Fusion PIDNet-S候选，不包含SAMF烟门，也不实现SAMF+TGM路径。ABL+SAMF及ABL+SAMF+TGM组合仅受Amendment 5的条件授权，当前不得提前运行。

## 2. 用户确认的冻结定义

```text
IR -> ThermalStem -> T(1/4, 32ch)
T -> adaptive_avg_pool2d -> T_aligned(1/8, 32ch)
g_s = sigmoid(Conv1x1 32->1 no-bias(DWConv3x3 32 no-bias(T_aligned)))
g_c = sigmoid(Linear 32->64 bias(GAP(original T)))
F_out = F_pag3 + alpha * g_c * g_s * Conv1x1_64->64_no-bias(F_pag3)
```

- ThermalStem与SAMF已确认定义相同；
- 插入点只有Pag3输出；alpha可学习且零初始化；
- 训练辅助P头读取注入后的`F_out`；
- 推理只返回最终分割，不保留训练辅助头；
- 不增加第二插入点、门控变体或权重扫描。

## 3. 正向触发定义

TGM第26–30轮均值相对Fusion基线至少一个mIoU/Fire增益严格大于0；另一指标必须保持在原容忍范围内：mIoU不低于`-0.005`，Fire不低于`-Delta_fire`。同时不得类别崩溃，且架构须达到RTX 2060成对实测30 FPS。该定义只触发Amendment 5组合筛选，不替代TGM自身进入100轮的原门槛。

## 4. 工程准入清单

1. alpha=0时三个训练输出与Fusion正式checkpoint逐位一致；
2. AMP损失有限，alpha=0首步只有alpha获得TGM路径梯度；
3. 临时alpha=0.1时热支路、空间门、通道门、特征投影及P/I/D分支均有非零梯度；
4. 推理只返回Tensor且不存在`seghead_p/seghead_d`；
5. 报告参数、FLOPs和RTX 2060同会话成对延迟，候选必须至少30 FPS；
6. 全部通过后才启动`route_c_tgm_30e_label_fix_seed200`。

## 5. 验收结果

- alpha=0时三个训练输出与Fusion正式checkpoint逐位一致，最大绝对误差均为0；
- TGM与Fusion均匹配301个ImageNet预训练张量；
- alpha=0首步alpha梯度为0.200865，热支路、空间门、通道门与特征投影梯度均为0；
- 临时alpha=0.1时上述模块及P/I/D分支和最终头梯度均非零；
- 推理只返回最终Tensor，不存在`seghead_p/seghead_d`；
- 推理参数量7,635,252，比Fusion增加11,313；
- FLOPs为3.0077 G，MACs为1.5038 G；
- RTX 2060成对架构测速：Fusion 21.0642 ms / 47.47 FPS，TGM 22.5512 ms / 44.34 FPS；
- TGM满足30 FPS硬门槛，工程准入通过。

证据文件：

- `experiments/route_c_pidnet_s_tgm/route_c_tgm_engineering/pipeline_check.json`；
- 同目录`complexity_architecture.json`；
- 同目录`paired_latency_rtx2060_architecture.json`。

复杂度和延迟属于训练前architecture-only准入证据，不冒充checkpoint精度评估。若进入100轮，将在正式best checkpoint上重新测量。
