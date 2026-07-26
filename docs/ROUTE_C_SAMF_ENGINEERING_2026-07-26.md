# Route C 模块一：SAMF 设计与工程验收

日期：2026-07-26
状态：工程验收通过；允许进入唯一主变体的 30 轮筛选。

## 1. 证据与原创边界

SAMF 是本项目依据自身诊断独立设计的轻量融合适配，不复制或声称复现 CMX、UTFNet、SGFNet 或 RoboFireFuseNet 模块。设计依据为：

- worst-20 中 RGB 不可见、heavy 烟遮挡、IR 更清晰均为 20/20；
- Smoke 区域或 3 像素邻域覆盖 Fire FN 69.90%、Fire FP 72.18%；
- IR-only Fire IoU 为 0.673819，略高于 Fusion 的 0.669310；Fusion Smoke IoU 比 IR-only 高 0.129299。

因此模块目标是在模型判断为 Smoke 的位置选择性注入热特征，同时保留 Fusion 对 Smoke 和 Background 的优势。

## 2. 已冻结的实现定义

```text
IR -> ThermalStem -> T(1/4, 32ch)
Pag3 注入前特征 -> 现有 seghead_p -> smoke softmax G(1/8)
T -> adaptive_avg_pool2d(1/8) -> psi 1x1(32->64)
F_out = F_pag3 + beta * G_smoke * psi(T)
```

- ThermalStem：Conv3x3 1→16 s2、BN、ReLU、Conv3x3 16→32 s2；
- 插入点：Pag3 输出，1/8 分辨率；
- beta：可学习标量，零初始化；
- smoke gate 不 detach，端到端训练；
- gate 使用注入前 Pag3 特征，避免循环依赖；
- 推理保留 `seghead_p` 作为内部烟概率预测器，但只返回最终分割张量；
- 推理不保留无用的 `seghead_d`。

熵门不实现。它仅在烟概率门本身工程上无法成立时作为备用，不得因精度结果不佳自动追加。TGM 在 SAMF 筛选结论冻结前不得动工。

## 3. 工程验收结果

验收命令使用 Fusion 正式 best checkpoint：

```text
experiments/route_a_pidnet_s_fusion/route_a_fusion_100e_label_fix_seed200/best.pth
```

结果文件：

- `experiments/route_c_pidnet_s_samf/route_c_samf_engineering/pipeline_check.json`
- `experiments/route_c_pidnet_s_samf/route_c_samf_engineering/complexity_architecture.json`
- `experiments/route_c_pidnet_s_samf/route_c_samf_engineering/paired_latency_rtx2060_architecture.json`

通过项：

- beta=0 时，三个训练输出与 Fusion 基线逐位一致，最大绝对误差均为 0；
- ThermalStem、投影、Pag3、烟 logits 形状分别为 32x64x64、64x32x32、64x32x32、3x32x32；
- SAMF 与 Fusion 基线均匹配 301 个 ImageNet 预训练张量；
- AMP 下损失有限；beta=0 首步 beta 梯度非零，热支路与投影梯度为 0；
- 临时 beta=0.1 时，ThermalStem、投影、烟门、P/I/D 三分支和最终头梯度均非零；
- 推理只返回一个 Tensor，内部 `seghead_p` 实际调用一次，不存在 `seghead_d`。

## 4. 复杂度与 RTX 2060 准入

256x256、batch=1、AMP、同一进程交替顺序成对测速；每个模型预热 100 次，每组计时 200 次，共 10 组，中位数用于判定。

| 模型 | 推理参数量 | FLOPs | 中位延迟 | FPS |
|---|---:|---:|---:|---:|
| Fusion PIDNet-S | 7,623,939 | 2.927702 G | 29.6173 ms | 33.7640 |
| Fusion PIDNet-S + SAMF | 7,705,271 | 3.1545 G | 30.9367 ms | 32.3241 |

SAMF 推理参数增加 81,332（1.0668%），延迟相对增加 4.4547%。旧“速度下降不超过 3%”规则已经废止，不能作为淘汰依据。新 Route C 硬门槛为同会话实测至少 30 FPS；SAMF 为 32.3241 FPS，因此通过工程准入。

复杂度和测速均为训练前的 architecture-only 工程准入数据，不冒充 checkpoint 精度评估。若 SAMF 进入最终方法，将在正式 checkpoint 上重新测量并纳入统一 mIoU–延迟 Pareto 图。

## 5. 下一步纪律

只运行配置 `configs/route_c/pidnet_s_fusion_samf_30e_label_fix.yaml` 对应的一个烟概率门主变体，实验名为 `route_c_samf_30e_label_fix_seed200`。第 26–30 轮均值与相同 label_fix 协议的 Fusion 基线窗口比较；通过预注册门槛才进入 100 轮，失败则归档 SAMF，不自动启用熵门或 TGM。
