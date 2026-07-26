# Route A Fusion-DySample Pag4 30轮筛选结果

日期：2026-07-26

## 实验身份

- 模型：`pidnet_s_dysample`
- 变体：Pag4
- 输入：RGB+IR四通道Fusion
- Seed：200
- 训练轮数 / 学习率horizon：30 / 100
- GPU：RTX 2060，`cuda:0`
- 测试集：未使用
- 运行目录：
  `experiments/route_a_pidnet_s_fusion_dysample/route_a_fusion_dysample_pag4_30e_label_fix_seed200`

## 训练状态

- 30/30轮完整结束
- Best validation mIoU：0.7920576028（第28轮）
- 总耗时：1397.3秒
- 峰值训练显存：228.5 MB
- Smoke/Fire无类别崩溃

Best单点不用于筛选，正式判断只使用第26--30轮固定窗口。

## 固定窗口结果

Fusion正式基线：

- mIoU均值：0.7861409556
- Fire IoU均值：0.6467101738
- Fire `max-min` 保守噪声带：0.0142509431

Fusion-DySample Pag4：

- mIoU均值：0.7863853198，相对基线 +0.0002443642
- Fire IoU均值：0.6409149274，相对基线 -0.0057952464
- Smoke IoU均值：0.8220981309
- Fire变化位于保守噪声带内，判为中性

## 正式判定

- mIoU规则：不通过；增益未达到 +0.005。
- Fire规则：不通过；Fire未提高超过保守噪声带。
- 健康与类别条件：通过。
- 总判定：`passes_screening = false`。

因此Fusion-DySample Pag4不进入100轮训练，不进行Pag4与其他插入位置组合，也不因
单点best或结果接近而放宽门槛。DySample代码、官方checkout、配置、checkpoint和
历史结果保留。

## 轻量化证据

相对Fusion PIDNet-S：

- 推理参数增加12,544，约0.165%；
- 256×256四通道FLOPs增加0.001622304G，约0.055%。

轻量化准入本身通过，但性能筛选未通过，所以不进入RTX 2060正式成对速度验收。

## 后续边界

最终计划中的另一个备用方向FreqFusion仍具备启动资格，但必须先单独完成官方来源、
固定commit、LICENSE、PIDNet插入位置和轻量化预算审计，并形成新的决策记录；本结果
不自动授权直接训练。Context DySample仅有历史工程检查，不被描述为失败，也不在本轮
继续追加实验。

证据：

- `metrics.jsonl`
- `screening_vs_fusion.json`
- `run_summary.json`
- `docs/FUSION_DYSAMPLE_TRIGGER_2026-07-26.md`
