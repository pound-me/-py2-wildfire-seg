# FLAME3 冻结模型零样本评估预注册

日期：2026-07-30  
设备：RTX 2060  
状态：在任何FLAME3模型推理结果前冻结

## 1. 评估范围

- 只使用预注册 `val.csv` 的134张样本；
- `test.csv`继续封存；
- 恰好评估三个用户确认的FLAME2冻结checkpoint；
- 不训练、不更新权重、不选择checkpoint、不修改温度阈值；
- 每个模型使用其原训练config重建网络，并严格加载`model_state_dict`。

## 2. 输入适配

旧FLAME2模型训练时的IR输入是8位灰度图并缩放到`[0,1]`。因此零样本评估固定为：

- RGB：FLAME3 `RGB/Corrected FOV`，640×512，RGB顺序，除以255；
- IR：FLAME3 `Thermal/Raw JPG`转灰度，除以255；
- Fusion：RGB三通道与IR灰度一通道拼接；
- 不把Celsius TIFF直接喂给旧checkpoint，因为这会改变旧模型的输入分布；
- Celsius TIFF只用于温度分桶和伪标签诊断。

FLAME3正式重训阶段将另行预注册Celsius TIFF的物理温度归一化，本零样本结论不替代
新基线。

## 3. 可报告指标

当前伪标签只包含Background候选、活动Fire和边界ignore，没有Smoke真值。因此禁止
报告三分类mIoU或Smoke IoU。只报告：

- 温度支持Fire的像素IoU、precision、recall和F1；
- No Fire纯背景图中的Fire误报率；
- 模型预测Smoke的像素比例（仅诊断，不称为准确率）；
- `<50℃`、`50–80℃`、`80–200℃`、`>=200℃`温度桶中的Fire/Smoke预测比例；
- 有非空伪火区图像的逐图Fire IoU及最差样本；
- 84张200℃种子缺席Fire图像单独统计，不将其空mask解释成官方像素真值。

所有像素指标排除`255`边界ignore带。

## 4. 错误画像

每个模型保存：

- 逐图指标和原始预测；
- 温度分桶统计；
- 非空伪火区中Fire IoU最差样本；
- No Fire中Fire误报率最高样本；
- Fire文件夹中空伪标签样本的预测Fire/Smoke比例；
- 固定可视化，面板包含RGB、热图、伪标签、三分类预测和叠图。

错误画像只决定后续人工烟雾小集合的抽样重点和规模，不得用于回调温度阈值、
改变当前test划分或选择FLAME2 checkpoint。

## 5. 工程纪律

- 使用RTX 2060、batch 1、AMP；
- 保存GPU/软件环境、config/checkpoint SHA256和脚本SHA256；
- 三个模型清单必须由用户明确确认后写入manifest；
- 输出统一前缀`flame3_zero_shot_*`；
- 任何加载不严格、输入尺寸不一致或非val split请求均直接报错。
