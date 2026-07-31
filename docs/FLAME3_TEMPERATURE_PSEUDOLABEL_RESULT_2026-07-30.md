# FLAME3 温度伪标注首轮结果

日期：2026-07-30  
状态：`awaiting_manual_review`，禁止进入正式训练

## 1. 执行规则

严格采用预注册主方案：`80℃`候选、`200℃`高置信种子、8邻域滞后连接、
小连通域与小孔洞清理、2像素对称边界ignore带。源图未修改，Smoke未自动赋值。

## 2. 全量统计

- 图像四元组：738（Fire 622，No Fire 116）；
- No Fire最高温度：中位数24.47℃，最大47.19℃；
- No Fire产生伪火区的图像：0；
- Fire最高温度：中位数514.03℃，最小108.25℃；
- Fire中没有200℃高置信种子的图像：84/622（13.50%）；
- 上述84张按冻结滞后规则得到空伪标签；
- Fire伪火区面积比例：中位数1.095%，P90为5.562%，最大9.998%。

## 3. 当前判断边界

No Fire负样本在温度规则下表现干净。另一方面，200℃种子对部分官方Fire图像明显
偏保守；首批叠图中可见若干最高温度约187℃、含烟或燃烧迹象的Fire图像得到空mask。
该现象已完整记录，不在本轮偷偷降低阈值。

本轮不做阈值结论。用户需检查固定30张清单后填写接受/拒绝：

- 若达到预注册通过条件，冻结80/200℃规则；
- 若未达到条件，先新增正式修订文档，再生成新版本；
- 任何修订只依据人工标注质量和温度证据，不依据模型精度。

## 4. 产物

- `pseudolabel_summary.json`：全量统计与脚本哈希；
- `pseudolabel_statistics.csv`：738张逐图统计；
- `fire_binary_masks/`：温度活动火区二值mask；
- `train_mask_templates/`：取值0/2/255的训练模板；
- `manual_review_30/manual_review_contact_sheet.jpg`：固定30张总览；
- `manual_review_30/manual_review_checklist.csv`：待用户填写的人工抽验清单；
- `manual_review_30/visuals/`：30张独立大图。

`training_authorized`保持为`false`。
