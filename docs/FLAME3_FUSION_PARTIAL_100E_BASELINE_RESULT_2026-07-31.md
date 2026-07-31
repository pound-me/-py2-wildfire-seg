# FLAME3 PIDNet-S Fusion 部分标签100轮正式基线结果

日期：2026-07-31  
状态：100轮完整结束，正式Fusion基线成立  
测试集状态：封存，未读取

## 协议与完整性

- 模型：PIDNet-S fusion，RGB+IR四通道；
- split v2：train/val为493/134；
- 原生分辨率：`640×512`；物理batch：8；seed：200；
- 训练目标：预注册部分标签损失；
- checkpoint选择：validation Fire IoU；
- 学习率：100轮polynomial周期；
- epoch 1–30完成筛选后，从`last.pth`精确恢复模型、优化器、AMP scaler、DataLoader generator及全部随机状态，继续epoch 31–100；
- 最终`metrics.jsonl`恰好100行，Windows计划任务结果码为0，无残留训练进程。

epoch 1–30使用提交`2af80a2`；续训启动与记录提交为`236c063`。续训提交只新增文档、打包器和启动器，训练生效的模型、损失、dataset及配置未改变；配置SHA256始终为：

```text
dc2ff3159bdb3212cc7232190addef06d98fe5b51efbe3b95f416269eded62c5
```

## 最佳checkpoint

- 最佳epoch：63；
- validation Fire IoU：`0.639680`；
- Fire precision / recall / F1：`0.669306 / 0.935283 / 0.780250`；
- Fire boundary F1：`0.819273`；
- No Fire图片Fire误报像素率：`0.000000087`；
- 空伪标签Fire图片Fire预测像素率：`0.038545`；
- train Fire IoU：`0.790410`；
- train/validation Fire IoU间隙：`0.150729`。

正式权重：

```text
best.pth SHA256 = 2A3911573C067620270EF5AC185DD100B9463BD2929FC20B750E30D0A707EFB3
```

`last.pth`只用于恢复与审计，不作为正式评估权重：

```text
last.pth SHA256 = 1A4959C1A4EF5FDC45C3CEFCF45FDA764067A8EBF6E557C2E6829607D32573EB
```

## 后期稳定性与过拟合

| 窗口 | Fire IoU均值 | 样本标准差 | 最小值 | 最大值 |
|---|---:|---:|---:|---:|
| epoch 26–30 | 0.564110 | 0.015862 | 0.548982 | 0.584992 |
| epoch 81–100 | 0.602448 | 0.008654 | 0.586940 | 0.619071 |
| epoch 91–100 | 0.607448 | 0.007177 | 0.599103 | 0.619071 |
| epoch 96–100 | 0.607953 | 0.008043 | 0.599103 | 0.619071 |

epoch 100的validation Fire IoU为`0.613020`，train Fire IoU为`0.816990`，间隙扩大到`0.203970`。因此模型存在明确过拟合，但后20轮验证结果已形成稳定平台，没有类别崩溃。正式结果必须使用epoch 63的`best.pth`，不得用末轮模型替代。

相对30轮最佳`0.584992`，100轮最佳提高`0.054689`；相对split v2零样本Fusion的`0.1985`提高约`0.4412`。后者只证明域内训练必要性，不作为结构创新收益。

## 时间和资源

- epoch 1–30：`1059.73 s`；
- epoch 31–100：`2389.67 s`；
- 总计约：`3449.40 s`（57.5分钟）；
- 峰值allocated显存约：`1.22 GiB`；
- 训练参数量：`7,717,095`。

## 决策

1. 将该checkpoint冻结为FLAME3 split v2、seed 200的正式PIDNet-S Fusion基线；
2. 后续模块必须使用相同split、部分标签公式、分辨率、batch、seed和100轮学习率协议；
3. 目前不能宣称RGB+IR优于单模态，因为尚缺同协议训练的RGB-only与IR-only对照；
4. 在模块创新筛选前，优先补齐RGB-only与IR-only输入消融，形成公平的输入模态动机表；
5. 测试集继续封存，待结构和超参数冻结后再按预注册规则评估。

## 保存位置

远端正式实验：

```text
D:\qianpengcheng\7.31\flame3_4090_bundle_v1_20260731\project_support\experiments\flame3_pidnet_s_fusion_partial\flame3_fusion_partial_30e_seed200_retry1
```

本地已同步`metrics.jsonl`、`run_summary.json`、`resolved_config.json`和`environment.json`至同名实验目录；大体积checkpoint继续保存在远端D盘，并以SHA256固定。
