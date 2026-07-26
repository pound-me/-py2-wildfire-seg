# Route A Fusion-D2/P1 30轮筛选决策记录

日期：2026-07-26

## 1. 决策范围

Route A 的四通道 Fusion PIDNet-S 已成为正式基线。本轮仅重新筛选此前
RGB 路线中胜出的两个结构候选：

- D2：在 P 分支 `layer3_` 与 `layer4_` 使用可重参数化 DEConv。
- P1：DFM 后每类一个 EMA prototype，prototype detach，仅在训练期使用。

两项实验均使用同一 Fusion 数据协议、seed 200、100轮学习率 horizon，并以
第26--30轮固定窗口比较。测试集未使用。

## 2. Fusion正式基线窗口

- mIoU均值：0.7861409556
- Fire IoU均值：0.6467101738
- Fire `max-min` 保守噪声带：0.0142509431
- Fire样本标准差：0.0060898514
- Smoke/Fire无类别崩溃

进入100轮需满足任一：

1. mIoU提高至少0.005，且Fire不低于基线减去保守噪声带；
2. Fire提高超过保守噪声带，且mIoU下降不超过0.005。

## 3. D2结果

运行目录：

`experiments/route_a_pidnet_s_fusion_deconv/route_a_fusion_deconv_d2_30e_label_fix_seed200`

第26--30轮均值：

- mIoU：0.7909388531，相对基线 +0.0047978975
- Fire IoU：0.6502890590，相对基线 +0.0035788852
- Fire变化位于保守噪声带内，判为中性
- Smoke/Fire无类别崩溃

判定：两条进入100轮规则均未满足。mIoU距离第一条规则还差
0.0002021025，但门槛不因接近而放宽。D2不进入Fusion 100轮。

## 4. P1结果

运行目录：

`experiments/route_a_pidnet_s_fusion_mproto/route_a_fusion_mproto_p1_30e_label_fix_seed200`

第26--30轮均值：

- mIoU：0.7899207602，相对基线 +0.0037798046
- Fire IoU：0.6406797204，相对基线 -0.0060304533
- Fire变化位于保守噪声带内，判为中性
- Smoke/Fire无类别崩溃
- EMA prototype全程已初始化、范数正常、命中非零，无持续死亡或塌缩

判定：两条进入100轮规则均未满足。P1不进入Fusion 100轮。

## 5. 联合决策

1. D2与P1在Fusion正式基线下均未通过30轮筛选，不运行对应100轮，也不运行
   D2+P1组合。
2. Fusion PIDNet-S继续作为当前正式基线；本轮没有产生可替代它的新结构。
3. DEConv与mproto实现、配置、checkpoint和历史结果保留，不删除；失败仅表示
   它们在当前Fusion协议和既定门槛下未获录用。
4. 本轮不自动扩展为D1或P2--P4重筛。Route A既有计划明确只重筛此前路线
   优胜者D2与P1，额外变体属于新的实验决策。
5. DySample Pag4的RGB 30轮筛选已经完成并归档，满足原备用路线闭环要求；
   是否在Fusion基线上重新筛选DySample，需要单独记录触发依据后再执行。
6. 测试集继续封存，不依据测试结果修改结构、超参数或checkpoint。

## 6. 证据文件

- Fusion基线：
  `experiments/route_a_pidnet_s_fusion/route_a_fusion_100e_label_fix_seed200/metrics.jsonl`
- D2筛选：
  `experiments/route_a_pidnet_s_fusion_deconv/route_a_fusion_deconv_d2_30e_label_fix_seed200/screening_vs_fusion.json`
- P1筛选：
  `experiments/route_a_pidnet_s_fusion_mproto/route_a_fusion_mproto_p1_30e_label_fix_seed200/screening_vs_fusion.json`

## 7. 后续备用路线触发

D2与P1均未通过后，最终计划中的备用路线触发条件已经满足。下一项实验固定为
Fusion-DySample Pag4单候选30轮筛选；触发依据、候选边界和协议见
`docs/FUSION_DYSAMPLE_TRIGGER_2026-07-26.md`。
