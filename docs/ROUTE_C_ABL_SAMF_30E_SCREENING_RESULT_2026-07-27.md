# Route C ABL+SAMF 30轮组合筛选结果

日期：2026-07-27

结论：ABL+SAMF 在 mIoU、Smoke IoU 和 Fire IoU 上均呈正向，但未达到预注册的组合门槛，因此不进入 100 轮。该实验补齐了此前遗漏的条件分支，Route C 至此真正闭环。测试集未使用。

## 1. 实验完整性

- 配置：`configs/route_c/pidnet_s_fusion_abl_samf_30e_label_fix.yaml`；
- 运行：`route_c_pidnet_s_abl_samf/route_c_abl_samf_30e_label_fix_seed200`；
- 30/30 轮，seed 200，AMP，100轮学习率 horizon；
- 从 PIDNet-S ImageNet 权重 fresh 初始化，匹配 301 个张量；
- 采用冻结 SAMF 结构和冻结 ABL 损失，没有新增组合模型类或推理层；
- 用时 1431.6 秒，峰值分配显存 242.4 MB，stderr 为空；
- 测试集封存。

## 2. 第26--30轮窗口

| 指标 | plain Fusion | ABL+SAMF | 增益 |
|---|---:|---:|---:|
| mIoU | 0.786140956 | 0.790729560 | +0.004588605 |
| Smoke IoU | 0.819727442 | 0.824281070 | +0.004553628 |
| Fire IoU | 0.646710174 | 0.649644881 | +0.002934707 |

五轮中 Smoke 与 Fire 均未崩溃。mIoU、Smoke 和 Fire 三项方向均为正，说明 ABL 与 SAMF 没有发生明显负交互；但筛选条件按精确值判定，不能把 `+0.004588605` 四舍五入为 `+0.005`。

## 3. 门槛判定

预注册组合门槛为相对 plain Fusion 满足任一：

- mIoU 增益至少 `+0.005`；或
- Fire IoU 增益至少 `+0.01`。

本结果距离 mIoU 门槛还差 `0.000411395`，Fire 增益也未达到 `+0.01`。因此：

- `passes_miou_rule: false`；
- `passes_fire_rule: false`；
- `passes_30_epoch_combination_screen: false`；
- `promote_to_fresh_100_epochs: false`。

best 单点为 epoch 30，mIoU 0.795030524、Smoke IoU 0.833448800、Fire IoU 0.650603031；只作运行描述，不替代五轮均值规则。

## 4. RTX 2060正式成对测速

双方 best checkpoint、256x256、batch 1、AMP、预热100次、每组200次、10组交替顺序：

| 模型 | 参数量 | FLOPs | 中位延迟 | FPS |
|---|---:|---:|---:|---:|
| Fusion PIDNet-S | 7,623,939 | 2.927702 G | 20.786136 ms | 48.108991 |
| ABL+SAMF | 7,705,271 | 3.1545 G | 22.022326 ms | 45.408464 |

ABL+SAMF 满足 30 FPS 工程门槛。ABL 是训练专用损失，因此其部署结构与 standalone SAMF 完全相同；精度门槛而非实时门槛导致本次淘汰。

## 5. 最终处理

- 保留 ABL+SAMF 为正向近失消融；
- 不运行 100 轮、不补 3 seed、不使用测试集；
- 不启用 ABL+SAMF+TGM，因为 TGM 正向条件为 false；
- 不扫描权重或追加组合变体；
- Route C 全部有效预注册分支闭环。

机器证据：运行目录中的 `metrics.jsonl`、`run_summary.json`、`screening_26_30.json`、`complexity.json`、`paired_latency_rtx2060_best.json`、`environment.json` 与 `resolved_config.json`。checkpoint 不提交 Git。
