# Route A Fusion-DySample Context最终备用候选触发记录

日期：2026-07-26

## 1. 为什么仍属于既定计划

原DySample筛选设计把候选限制为两个理论驱动的单点位置：

1. Pag4语义到细节融合；
2. SPP context到DFM的最终语义上下文上采样。

Fusion-Pag4已经完成30轮并未通过；FreqFusion Pag3也因RTX 2060速度和复杂度门槛
在训练前失败。Context位置此前只有RGB协议下的工程检查，没有Fusion 30轮结果，不能
被描述为已经失败。现在启用Context是完成原先“最多两个DySample位置”的第二个、也是
最后一个候选，不是新增第三个位置，也不引入新的外部模块。

## 2. 失败归因与候选匹配

Fusion-Pag4第26--30轮相对基线mIoU仅+0.000244，Fire -0.005795，说明在最后一次
PagFM中改进语义/细节对齐没有形成稳定收益。FreqFusion完整机制虽直接建模频率融合，
但AMP延迟增加21.699%，不满足实时约束。

Context候选只替换SPP输出从4×4到32×32的单次双线性上采样。该位置直接决定进入DFM
的128通道全局语义上下文如何恢复空间结构，和此前P路线归因中的“小火与边界在粗语义
上下文中被稀释”相对应，同时只插入一个官方DySample模块，预期比Pag4双模块和
FreqFusion更轻。

## 3. 固定实现与协议

- 模型：`pidnet_s_dysample`；
- 变体：`context`；
- 输入：RGB+IR四通道Fusion；
- 唯一替换：SPP context `128×4×4 → 128×32×32`；
- Pag3、Pag4、D分支插值和最终logit resize保持基线；
- 官方仓库：`https://github.com/tiny-smart/dysample`；
- 固定commit：`81a1de5caa95d55a0f5488425fa53ec7ef47f8f0`；
- LICENSE：MIT；
- 设置：`style=lp`、`groups=4`、`dyscope=False`；
- Seed：200；训练30轮；学习率horizon 100轮；
- 数据协议：Fusion正式基线A1协议；
- GPU：RTX 2060，`cuda:0`；
- 测试集继续封存。

配置：

`configs/route_a/pidnet_s_fusion_dysample_context_30e_label_fix.yaml`

## 4. 训练前准入规则

必须先完成：

1. 固定官方commit、MIT许可证和checkout洁净状态检查；
2. 预训练匹配数、AMP前反向、offset梯度和训练/推理接口检查；
3. 参数量与FLOPs比较；
4. RTX 2060同进程、交替顺序、AMP成对速度测试。

部署参数量和FLOPs不得高于原PIDNet-S、速度下降不得超过3%。若工程门槛失败，直接
归档，不启动30轮；不会删减模块组件来绕过门槛。

## 5. 30轮筛选规则

只有工程准入通过才运行30轮。第26--30轮与Fusion正式基线比较：

- 基线mIoU：0.7861409556；
- 基线Fire IoU：0.6467101738；
- Fire `max-min`保守噪声带：0.0142509431。

进入100轮必须满足任一：

1. mIoU提高至少0.005，Fire不低于基线减去保守噪声带；
2. Fire提高超过保守噪声带，mIoU下降不超过0.005。

同时要求Smoke和Fire均无类别崩溃。门槛不因单点best或结果接近而放宽。

Context完成后，原DySample设计的两个插入点即全部闭环；不自动扩展Pag3、组合位置或
其他新模块。若仍无候选通过，必须另行决定是冻结Fusion基线进入论文材料阶段，还是
授权新的研究路线。

## 6. 工程准入结果

工程检查与RTX 2060 AMP成对测速已经完成：

- 官方commit、MIT许可证、checkout洁净状态和官方源码导入均通过；
- Fusion候选与基线均匹配301个PIDNet预训练张量；
- 模型仅包含一个官方DySample模块：`context_upsampler`；
- AMP前反向、offset非零梯度、训练三输出与推理单输出均通过；
- 部署参数：7,623,939 → 7,689,987，增加66,048（+0.8663%）；
- 前向FLOPs：2.956013568G → 2.958135336G，增加0.002121768G（约+0.0718%）；
- AMP成对测速中位延迟：32.8206 ms → 31.1089 ms，变化-5.2155%，速度门槛通过。

速度结果组间波动较大，只用于确认没有超过3%下降上限，不宣称Context带来真实加速。
单个Context模块比Pag4双模块参数更多，是因为官方`lp`实现的offset输出通道随
`2 × groups × scale²`增长；本位置scale=8，产生512个offset输出通道。

尽管功能与速度检查通过，参数量和FLOPs均高于Fusion PIDNet-S，违反最终计划的硬性
轻量化总原则。因此正式判定为工程准入失败，不启动30轮，不生成训练checkpoint。

证据：

- `experiments/route_a_pidnet_s_fusion_dysample/route_a_fusion_dysample_context_engineering/pipeline_check.json`
- `experiments/route_a_pidnet_s_fusion_dysample/route_a_fusion_dysample_context_engineering/fusion_baseline_complexity_architecture_only.json`
- `experiments/route_a_pidnet_s_fusion_dysample/route_a_fusion_dysample_context_engineering/complexity_architecture_only.json`
- `experiments/route_a_pidnet_s_fusion_dysample/route_a_fusion_dysample_context_engineering/paired_speed_architecture_only_amp.json`

至此两个DySample插入点均已闭环：Pag4未通过30轮性能筛选，Context未通过训练前
轻量化门槛。测试集未使用。
