# FLAME3 温度伪标签人工抽验结论

日期：2026-07-30  
抽验对象：预注册固定30张（24张 Fire、6张 No Fire）  
查看材料：`manual_review_contact_sheet.jpg` 及其五联图说明

## 用户人工结论

- 24张 Fire 样本目前未发现明显的温度火区位置错误；
- 6张 No Fire 纯背景样本的最右侧 `0/2/255 label` 均为全黑，即没有生成 Fire 伪标签；
- 总体结论：本轮人工抽验通过，80℃候选、200℃种子、8邻域滞后连接、连通域清理及2像素边界忽略带规则保持冻结，不因模型结果调整。

## 状态边界

- `pseudolabel_manual_quality_gate = passed`；
- 原始 `pseudolabel_summary.json` 作为生成时的不可变记录继续保留 `awaiting_manual_review`，不回写覆盖；本文件作为后续人工结论增量记录；
- `formal_training_authorized = false` 仍保持，因为正式训练还需先冻结 Fire 图片非伪火区域的 Smoke/Background 监督方式，并等待RTX 4070上线；
- 测试集继续封存。

## 审计说明

本次用户给出的是对固定拼图的总体审核结论，原逐图 `manual_review_checklist.csv` 的详细字段仍为空。若后续投稿或复现实验需要逐图审计，可在不改变温度规则的前提下补填逐图字段；不得依据模型预测反向修改人工结论。
