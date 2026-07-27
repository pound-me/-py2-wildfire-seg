# ABL+SAMF 终局三 Seed 评估预注册

日期：2026-07-27

状态：本规则在任何新增终局训练启动前冻结并提交。它只建立一个新的终局评估阶段，不追溯修改 Route C 的30轮未过筛结论、已有100轮单项结论或任何历史门槛。

## 1. 目的与冻结方法

终局问题只有一个：依据原计划“3 seed 均值差异大于标准差”的要求，判断 `Fusion + ABL + SAMF` 的小幅正收益是否超过随机种子波动，足以作为论文主方法行。

冻结方法为：

- 基线：plain Fusion PIDNet-S；
- 候选：同一冻结 `pidnet_s_samf` 推理结构，加训练专用 ABL；
- ABL 不增加推理参数、FLOPs或输出；SAMF 为冻结的烟概率门控热注入；
- label_fix A1、256x256、batch 4、AMP、100轮、100轮 polynomial LR horizon；
- 每个新运行均从 PIDNet-S ImageNet 权重 fresh 初始化，`CHECKPOINT: null`；
- seeds 固定为 `200, 201, 202`，不替换失败或异常但数据/代码正确的 seed；
- 测试集在验证集裁决冻结前继续封存。

## 2. 现有 seed 200 审计与运行矩阵

审计结果：

- Fusion seed 200 已完成100/100轮，可进入终局统计；
- ABL+SAMF seed 200 只有30轮，不能与100轮 seed 201/202混合为“三个100轮seed”。

因此，为满足用户要求的两组各3个100轮seed，新增运行不是4个而是5个：

| 方法 | seed | 100轮状态 | 终局动作 |
|---|---:|---|---|
| Fusion | 200 | 已完成 | 复用冻结结果 |
| Fusion | 201 | 未运行 | fresh 100轮 |
| Fusion | 202 | 未运行 | fresh 100轮 |
| ABL+SAMF | 200 | 仅有30轮 | 另起目录 fresh 100轮，不续训 |
| ABL+SAMF | 201 | 未运行 | fresh 100轮 |
| ABL+SAMF | 202 | 未运行 | fresh 100轮 |

原30轮 ABL+SAMF seed 200继续保留为筛选证据，不冒充100轮seed，也不被覆盖。

## 3. 每个 seed 的指标取值规则

每个100轮运行按验证集 mIoU 严格大于历史best时保存 `best.pth`；若数值完全相同，保留更早epoch，与现有训练代码一致。

对每个 seed：

- `mIoU` 取 mIoU-selected best checkpoint 的验证 mIoU；
- `Fire IoU` 必须取同一个 mIoU-selected best checkpoint，禁止为 Fire 单独挑epoch；
- 同时保存 Background/Smoke IoU、Fire precision/recall、Fire boundary F1，但它们不参与终局通过判定；
- 不使用第26--30轮均值、last checkpoint或测试集选择任何终局checkpoint。

## 4. 精确统计公式

对指标 `q`（分别为 mIoU 和 Fire IoU），设基线三个值为
`b_1,b_2,b_3`，候选三个值为 `m_1,m_2,m_3`。

两组均值：

```text
mean_b(q) = (b_1 + b_2 + b_3) / 3
mean_m(q) = (m_1 + m_2 + m_3) / 3
delta(q)  = mean_m(q) - mean_b(q)
```

两组样本标准差均使用 `ddof=1`：

```text
s_b(q) = sqrt(sum_i (b_i - mean_b)^2 / (3 - 1))
s_m(q) = sqrt(sum_i (m_i - mean_m)^2 / (3 - 1))
```

等样本量两组的合并组内样本标准差：

```text
s_p(q) = sqrt(
  ((3 - 1) * s_b(q)^2 + (3 - 1) * s_m(q)^2)
  / (3 + 3 - 2)
)
       = sqrt((s_b(q)^2 + s_m(q)^2) / 2)
```

每个指标的预注册通过条件为严格不等式：

```text
pass_q = delta(q) > s_p(q)
```

相等视为未通过。另报告标准化比值 `R_q = delta(q) / s_p(q)`；若 `s_p=0`，正增益记为无穷大比值，零或负增益不通过。

这是项目内部预注册的“一倍合并标准差”可分辨性/效应量门槛，不冒充小样本正式假设检验，不把它写成 p 值或统计学显著性检验。

## 5. ABL+SAMF 最终裁决规则

mIoU 与 Fire IoU 分别完整报告：三seed原值、均值、样本标准差、均值差、合并标准差、`R_q`和单指标通过状态。

ABL+SAMF 成为论文主方法行，当且仅当：

```text
(pass_mIoU OR pass_Fire)
AND delta_mIoU >= 0
AND delta_Fire >= 0
AND RTX2060 paired FPS >= 30
```

- 通过：定位为“零推理开销边界监督 + 轻量烟感知热注入”，表述为小幅但超过预注册种子波动门槛；速度沿用冻结结构的同会话实测并重测正式checkpoint，目标约45 FPS；
- 未通过：ABL+SAMF只保留为正向消融。论文定量主张转为“诊断证据 + 模态不对称 + 系统性阴性/近失对照”，不得声称主方法显著提升。

不因某个seed结果不理想而更换seed、改变权重、延长轮数、换best规则或追加变体。

## 6. 基线噪声地板

Fusion三seed的 `s_b(mIoU)` 与 `s_b(Fire)` 定义为全项目相应指标的经验噪声地板，并写入最终报告。

它只用于回溯解释，不追溯改变历史录用结论：

- 绝对增益小于对应 `s_b`：在本项目seed波动下难以分辨；
- 绝对增益约等于 `s_b`：处于经验噪声地板附近；
- 绝对增益大于 `s_b`：超过基线自身seed波动，但终局主方法仍必须按第4、5节的合并标准差规则裁决。

## 7. 测试集纪律

在六个验证集best checkpoint、统计脚本、方法身份和终局裁决全部冻结前，不运行任何测试集评估。

冻结后，无论验证集裁决通过或未通过，Fusion与ABL+SAMF的六个冻结checkpoint各在测试集评估一次，以避免根据测试结果决定是否查看另一组。测试结果只作一次性最终报告，不得改变方法身份、超参数、checkpoint或本文预注册裁决。每组报告3 seed均值±样本标准差。

## 8. 执行与归档纪律

- 新实验统一前缀 `terminal_eval_*`，不得覆盖历史 `route_a_*` 或 `route_c_*`；
- 每个运行保存 resolved config、environment、逐轮metrics、best/last、耗时、显存和哈希；
- 训练中日志不提交；每个运行完成、验证和归档后再单独commit/push；
- 聚合脚本必须拒绝混入30轮结果、重复seed、非200/201/202 seed、不同数据列表或不同冻结方法配置；
- checkpoint不提交Git；测试集锁继续由 `--confirm-frozen-test` 保护。

本预注册由用户于2026-07-27批准，记录发生在任何新增终局训练之前。
