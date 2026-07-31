# FLAME3 部分标签损失预注册

日期：2026-07-31  
状态：用户批准；在任何 FLAME3 训练前冻结  
适用对象：`DATASET_TYPE=flame3_csv`、三分类 PIDNet-S RGB+IR fusion

## 1. 标签含义

FLAME3 温度伪标签只提供“温度支持的活动火区”，不提供完整 Smoke 像素真值：

- `No Fire` 图片：全部有效像素是硬 Background；
- `Fire` 图片中 `label=2`：硬 Fire 核心；
- `Fire` 图片中 `label=255`：温度边界不确定带，语义损失忽略；
- `Fire` 图片中 `label=0`：部分标签集合 `{Background, Smoke}`，不得强迫为 Background。

Fire/No Fire 身份必须来自 split CSV 的 `sample_class`，不能根据伪标签是否为空推断。空伪标签 Fire 图仍采用 `{Background, Smoke}` 部分监督。

## 2. 像素损失

设三类 softmax 概率为 `p_bg, p_smoke, p_fire`。

硬 Background：

\[
L_{bg}=-\log p_{bg}
\]

硬 Fire 核心：

\[
L_{fire}=-\log p_{fire}
\]

Fire 图片非核心区域：

\[
L_{partial}=-\log(p_{bg}+p_{smoke})
\]

该式允许 Background 与 Smoke 之间自由分配概率，只惩罚错误的 Fire 概率。AMP 下 `log_softmax/logsumexp` 强制 FP32。

## 3. 图像、batch与双头归一化

- No Fire 图像：对全部有效 Background 像素取均值；
- Fire 图像：分别计算“部分非火区域均值”和“Fire 核心均值”，对当前图像中实际存在的项做等权平均；
- Fire 图像或随机裁剪中没有 Fire 核心时，只计算部分非火项，不伪造正样本；
- batch loss 是各图像 loss 的等权平均，避免大面积区域或某一类图片仅凭像素数量支配梯度；
- PIDNet 辅助语义头与主语义头都使用同一部分标签损失，权重固定为 `0.4/1.0`。

## 4. 边界分支

- D 分支边界真值由“增强后的 `label==2` Fire 核心二值图”通过 Canny 和 4 像素膨胀生成；
- 禁止直接对含 `255` 的三值标签做 Canny，避免把 ignore 带的两侧都错误当成语义边界；
- 原 PIDNet boundary BCE 保留，`BD_WEIGHT=1.0`；
- 主语义头在预测边界概率 `>0.8` 的像素上追加同一套部分标签集合损失，`SB_WEIGHTS=1.0`；
- 总损失为：

\[
L=0.4L_{aux}+1.0L_{main}+L_{boundary}+L_{semantic-boundary}
\]

原始 Lovasz/OHEM 语义项不直接使用，因为它们要求每个像素只有一个确定类别，无法表达 `{Background, Smoke}` 集合标签。

## 5. 验证指标与checkpoint

- Smoke 没有真值，禁止报告 Smoke IoU、三分类 mIoU或将 Background/Smoke 分配解释为准确率；
- 将 Background 与 Smoke 合并为“非 Fire”，主指标为 Fire-vs-rest IoU、precision、recall、F1；
- `best.pth` 固定按 validation `fire_iou` 保存；
- 额外记录空伪标签 Fire 图和 No Fire 图的 Fire 预测像素率；
- Smoke 预测比例仅作响应诊断，不作性能结论。

## 6. 工程验收

- 零 logits 下，部分像素损失必须等于 `-log(2/3)`；
- 部分像素梯度必须同时提高 Background/Smoke、降低 Fire，且 Background/Smoke 梯度对称；
- Fire 核心梯度必须提高 Fire；
- No Fire 样本只能返回硬 Background；
- Fire/No Fire 标志在缩放、裁剪和翻转后保持正确；
- 640×512 fusion 输入、AMP前后向、有限loss、非零梯度和checkpoint恢复通过；
- test 不参与公式、超参、batch或checkpoint选择。

## 7. 固定配置

- 模型：PIDNet-S fusion，4通道输入；
- AMP动态缩放初值固定为1024，后续由GradScaler自动增长/回退；
- 原生裁剪：高512、宽640；
- 强亮度变换关闭；
- 随机缩放候选协议固定为0.8–1.5，水平翻转开启；
- seed 200；30轮筛选的学习率日程长度保持100轮；
- 正式GPU为RTX 4090，物理batch由4/8工程测试后另行冻结。
