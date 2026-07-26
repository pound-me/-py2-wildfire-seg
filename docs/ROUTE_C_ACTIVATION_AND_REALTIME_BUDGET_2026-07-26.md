# Route C正式启动与RTX 2060实测实时预算

日期：2026-07-26
状态：**用户已批准，立即生效。**

## 1. 协议修订

废止“部署参数量和FLOPs不得高于Fusion PIDNet-S基线”的零增量硬约束，也不再使用
“相对基线速度下降不超过3%”作为结构准入条件。新规则为：

- 基线与候选必须在同一RTX 2060进程/会话内成对测量；
- 使用相同输入尺寸、batch size、精度模式、预热次数和重复次数；
- 保存每组原始延迟，准入统计量取预热后多轮中位数；
- 候选AMP中位延迟不超过`33.33 ms/image`，即至少`30 FPS`；
- 达到实时门槛后，允许部署参数量和FLOPs有限增长，但二者必须完整报告；
- 通用FLOPs profiler可能漏计部分算子，因此实测延迟优先于理论FLOPs。

本次批准没有设置固定`+5% FLOPs`上限。参数/FLOPs明显增加但仍达到30 FPS的候选不会
仅因高于基线而在训练前自动淘汰；其效率优劣由统一Pareto比较呈现。

## 2. 不变的科研纪律

- 30轮筛选和100轮内部方法成立门槛维持原预注册数值；
- Fire变化仍使用相同协议基线的保守噪声带解释；
- 不使用测试集选择结构、超参数或checkpoint；
- 只有进入论文主表的行补3 seed；
- 结构与超参数冻结后，每个最终seed只评估一次测试集。

## 3. 统一mIoU–延迟Pareto登记

后续建立同一张验证集mIoU–RTX 2060中位延迟Pareto图，至少登记：

1. Fusion PIDNet-S正式基线；
2. 五个已闭环Route A模块：DEConv D2、mproto P1、DySample Pag4、
   DySample Context、FreqFusion Pag3；
3. Route B ABL；
4. Route C的SAMF、后续获准的TGM及可能组合；
5. RoboFireFuseNet。

没有完成训练的模块只登记参数、FLOPs、延迟和“未训练/历史工程拒绝”状态，不伪造或
借用mIoU。ABL属于训练期only，部署架构与Fusion基线相同，因此二者共享架构延迟坐标但
保留各自验证mIoU。所有历史结论保留其当时规则语境，不因新规则自动变成通过结果。

## 4. Route C顺序与命名

1. 收尾人工与机器诊断，并完成IR-only 100轮动机实验；
2. 实现并验收SAMF烟概率门控热注入；
3. SAMF只放一个主变体进入30轮，达门槛后才跑100轮；
4. SAMF筛选结果冻结后才允许动工TGM；
5. 两者分别通过100轮门槛后，才追加组合配置。

新实验统一使用`route_c_*`前缀。每个模块每个阶段单独形成决策文档、commit并推送；正在
写入的日志与checkpoint不提交。测试集继续封存。

## 5. SAMF优先级证据

人工核查表为
`experiments/route_a_diagnostics/rgb_visibility_val/worst20_manual_checklist_filled.csv`。
用户已完成20张最差Fire-IoU验证样本审查：

- RGB中火焰不可见：20/20；
- heavy烟遮挡：20/20；
- IR更清晰：20/20；
- 小火点：4/20。

因此烟遮挡下的热信息注入是当前最直接的病因对应机制，SAMF优先于TGM和更通用的复杂
融合模块。节点1a已经读取原CSV并汇总全部列；完整分布及其与
`smoke_proximity_ratio`机器统计的互证见
`docs/RGB_VISIBILITY_MANUAL_DIAGNOSTIC_2026-07-26.md`。
