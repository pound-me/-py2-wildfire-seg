# Route C ABL+SAMF 组合执行记录

日期：2026-07-27

状态：预注册执行遗漏已纠正；组合配置与工程验收通过，允许进入唯一一次 30 轮筛选。

## 1. 授权依据

ABL 与 SAMF 的 100 轮正式结果相对 plain Fusion 均呈正向。按 SAMF 归档当日已澄清的规则，两者的两两组合不以 TGM 结果为前提；TGM 只控制是否加入三元组合。此前闭环结论错误跳过本分支，现按原预注册语义补执行，不属于事后新增候选。

## 2. 冻结定义

- 模型仍为 `pidnet_s_samf`，完全复用已冻结 SAMF 结构；
- 训练目标使用已冻结 ABL，权重与全部超参数不变；
- ABL 只存在于训练 criterion，不增加模型层、参数、FLOPs或推理输出；
- 从 ImageNet 预训练全新初始化，不续接 ABL 或 SAMF checkpoint；
- label_fix A1，seed 200，30 轮，100 轮 LR horizon，AMP；
- 实验目录：`route_c_pidnet_s_abl_samf/route_c_abl_samf_30e_label_fix_seed200`；
- 测试集封存。

## 3. 门槛

第 26--30 轮均值直接与 plain Fusion 同窗口比较，满足任一精度条件且无 Smoke/Fire 类别崩溃：

- mIoU 增益不少于 `+0.005`；或
- Fire IoU 增益不少于 `+0.01`。

同时必须在 RTX 2060 与 Fusion 同会话成对实测不少于 30 FPS。通过后进入一次全新初始化的 100 轮正式实验；未通过时 Route C 方真正闭环。测试集不参与任何判定。

## 4. 工程验收结果

验收配置：`configs/route_c/pidnet_s_fusion_abl_samf_30e_label_fix.yaml`。

- 工厂只创建一个现有 `PIDNetSAMF`，没有新增组合模型类；
- 从 ImageNet 预训练匹配 301 个张量，`CHECKPOINT: null`；
- beta=0 时三个训练输出与 Fusion best 逐位一致，最大绝对误差均为 0；
- ABL 在 AMP 下保持 FP32，单批次 ABL 为 0.904531，总损失有限；
- beta=0 时 beta、DFM、P/I/D 分支和最终头梯度均非零，热支路梯度严格为 0；
- 临时 beta=0.1 时，仅 ABL 反传即可到达 ThermalStem、热投影、烟头、DFM、P/I/D 分支和最终头；
- 推理类、state signature 和参数量与 standalone SAMF 完全相同；推理只返回分割 Tensor，不含 ABL 对象或 D 辅助头。

证据：

- `experiments/route_c_pidnet_s_abl_samf/route_c_abl_samf_engineering/pipeline_check.json`；
- 同目录 `complexity_architecture.json`；
- 同目录 `paired_latency_rtx2060_architecture.json`。

## 5. 复杂度与 RTX 2060 准入

256x256、batch 1、AMP、同一进程交替顺序成对测速；每模型预热 100 次，每组 200 次，共 10 组：

| 模型 | 推理参数量 | FLOPs | 中位延迟 | FPS |
|---|---:|---:|---:|---:|
| Fusion PIDNet-S | 7,623,939 | 2.927702 G | 27.067042 ms | 36.945300 |
| ABL+SAMF | 7,705,271 | 3.1545 G | 27.856984 ms | 35.897641 |

ABL 是训练专用损失，因此 ABL+SAMF 的推理参数和 FLOPs与 standalone SAMF 完全相同。组合相对本次 Fusion 配对延迟增加 2.9185%，并达到 35.90 FPS，通过 Route C 实时准入。以上为训练前 architecture-only 工程证据，不冒充 checkpoint 精度或正式速度结果；30 轮完成后使用双方 best checkpoint 重测。

## 6. 唯一训练节点

只运行 `route_c_pidnet_s_abl_samf/route_c_abl_samf_30e_label_fix_seed200`。不扫描 ABL 权重、SAMF 门控或插入点，不启用三元组合。第 26--30 轮由专用脚本按本页第 3 节的组合门槛判定。
