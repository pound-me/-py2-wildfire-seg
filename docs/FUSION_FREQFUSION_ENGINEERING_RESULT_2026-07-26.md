# Route A Fusion-FreqFusion工程准入结果

日期：2026-07-26

## 1. 实验目的与边界

在Fusion PIDNet-S正式基线上评估备用模块FreqFusion是否满足训练前工程准入条件。
本轮只比较架构、复杂度和RTX 2060推理速度，不使用测试集，也不把未训练模型的结果
解释为精度结论。总门槛保持不变：部署参数量和FLOPs不高于基线，RTX 2060速度下降
不超过3%。

## 2. 来源与授权

- 论文：*Frequency-Aware Feature Fusion for Dense Image Prediction*，IEEE TPAMI 2024；
- 官方仓库：`https://github.com/Linwei-Chen/FreqFusion`；
- 固定commit：`3fb0c70637a3c194fb74294d3ce4681958b26241`；
- 使用文件：`SegNeXt/mmseg/models/decode_heads/FreqFusion.py`；
- 目录级许可证：SegNeXt `Apache-2.0`；
- 源文件SHA256：`14a5af3a614a721cc01777102974e0cbfb4f29a19a45df8b883be7b0d1f38d91`。

官方仓库根目录的无许可证 `FreqFusion.py` 未使用。完整官方checkout位于被父仓库
忽略的 `third_party/FreqFusion`，不会提交到 `origin/main`。父仓库只保存PIDNet适配层、
来源校验以及按公开公式独立实现的纯PyTorch CARAFE兼容层。

若该方法未来因约束变化而进入最终方法，投稿前必须执行
`docs/FREQFUSION_SOURCE_LICENSE_AUDIT_2026-07-26.md` 第6节预案：优先联系作者取得发布
许可；否则按论文公式独立重实现，或只发布固定commit拉取与SHA256校验脚本；论文、
README和代码中规范引用论文与官方仓库。

## 3. 结构选择

完整FreqFusion要求高、低分辨率特征为2倍关系。PIDNet-S中只有Pag3自然满足：

```text
P detail: 64 x 32 x 32
I semantic: 64 x 16 x 16
```

本轮只实现Pag3单点，启用完整ALPF、AHPF和offset重采样，
`feature_resample=True`、`compressed_channels=16`。FreqFusion增强结果仍经过原PagFM
门控，保留PIDNet融合逻辑和原有预训练参数名。Pag4的4倍比例和SPP到DFM的8倍比例
不拆成级联模块，避免改变官方用法并扩大计算预算。

## 4. 工程检查

`pipeline_check.json`记录：

- 固定commit、checkout洁净状态、源文件及许可证SHA256均通过；
- 独立CARAFE兼容层与公式对照最大绝对误差为`1.1920929e-7`；
- 候选和基线均匹配PIDNet预训练参数301项；
- 仅存在一个FreqFusion模块：`pag3.freqfusion`；
- 六个核心组件梯度均非零，AMP训练输出和推理输出接口正确；
- 训练参数为基线7,717,095、候选7,732,889，增加15,794。

新增的 `--architecture-only` 测速模式没有破坏原checkpoint加载路径：使用Fusion正式
基线best checkpoint完成了最小回归运行，输出的 `weights_source` 为`checkpoints`，模型
严格从checkpoint载入并完成AMP推理。

## 5. 复杂度与速度结果

| 项目 | Fusion基线 | Fusion-FreqFusion Pag3 | 变化 |
|---|---:|---:|---:|
| 部署参数 | 7,623,939 | 7,639,733 | +15,794（+0.207%） |
| 前向FLOPs | 2.9560G | 2.9851G | 约+0.98% |
| FP32中位延迟 | 16.6855 ms | 24.8460 ms | +48.907% |
| AMP中位延迟 | 35.0886 ms | 42.7023 ms | +21.699% |

FP32和AMP均采用同进程、基线/候选交替顺序的成对架构测速。AMP正式工程判定使用100次
预热、每组200次计时、10组重复。尽管通用FLOPs profiler可能低估`unfold`和
`grid_sample`的真实开销，实测延迟已经足以作出门槛判断。

## 6. 决策

FreqFusion Pag3同时违反以下轻量化条件：

1. 部署参数量不高于基线；
2. FLOPs不高于基线；
3. RTX 2060速度下降不超过3%。

因此该候选在训练前工程准入阶段失败，不启动30轮筛选，不生成训练checkpoint，也不
为了通过预算而删去ALPF、AHPF或offset后冒充完整FreqFusion。本结果只说明当前完整
FreqFusion-Pag3适配不符合本论文的轻量化约束，不能解释为FreqFusion方法本身无效。

归档证据：

- `experiments/route_a_pidnet_s_fusion_freqfusion/route_a_fusion_freqfusion_pag3_engineering/pipeline_check.json`
- `experiments/route_a_pidnet_s_fusion_freqfusion/route_a_fusion_freqfusion_pag3_engineering/complexity_architecture_only.json`
- `experiments/route_a_pidnet_s_fusion_freqfusion/route_a_fusion_freqfusion_pag3_engineering/paired_speed_architecture_only.json`
- `experiments/route_a_pidnet_s_fusion_freqfusion/route_a_fusion_freqfusion_pag3_engineering/paired_speed_architecture_only_amp.json`
- `experiments/route_a_pidnet_s_fusion_freqfusion/route_a_fusion_freqfusion_pag3_engineering/checkpoint_mode_regression.json`

测试集继续封存。
