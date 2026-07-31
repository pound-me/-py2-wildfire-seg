# FLAME3 ERCTC 30轮筛选预注册（2026-07-31）

## 动机

Fusion best错误画像和用户人工判断一致：16张空Fire-core帧中的大片连续红色响应多数是
烟雾覆盖、高温地表或余燃，不应整体算Fire；少量燃烧带边缘可能属于正在蔓延的火线。
因此候选同时需要抑制区域性热响应并保留热前沿信息。

## 单变量结构

模型名：`pidnet_s_erctc`，含义为 Edge-aware RGB-Conditioned Thermal
Calibration。

- 保留原PIDNet-S Fusion四通道主干；
- 在Pag3输出的1/8特征处插入一个校准模块；
- RGB和IR先无参数平均池化到1/4，再用1×1投影形成16通道上下文；
- 对池化IR使用固定Sobel得到热前沿，不增加可训练梯度核；
- 区域路径使用RGB/IR/Pag特征产生有符号校准；前沿路径单独提供热梯度信息；
- 两条残差各有一个零初始化标量，初始输出与Fusion基线逐位一致；
- 本次不加ABL、SAMF或TGM，不增加新损失。

相关工作定位：SGFNet、CAINet、LASNet等已有语义引导、上下文交互和锐化思想。
ERCTC只主张为当前火场错误画像设计的轻量任务适配，不宣称通用RGB-T融合原理首创。

## 固定训练协议

- split v2；640×512；RTX 4090；物理batch 8；seed 200；AMP；
- 与Fusion基线相同的增强、部分标签损失、301个ImageNet骨干张量和100轮学习率周期；
- 训练30轮，按validation Fire IoU保存best；测试集不读取；
- 每轮额外记录region/frontier两个标量。

## 30轮准入条件

同30轮Fusion基线比较：best Fire IoU `0.584992`、recall `0.958678`、空
Fire-core预测Fire率`0.048982`。

进入100轮必须同时满足：

1. best Fire IoU不低于`0.594992`（至少+0.01）；
2. Fire recall不低于`0.938678`（下降不超过0.02）；
3. 空Fire-core预测Fire率不高于`0.048982`；
4. No Fire没有明显误报崩溃。

若空Fire-core率下降至少20%（≤`0.039186`），可作为校准机制证据；否则即使性能过线，
也只能解释为一般结构收益。

## 工程准入结果

- 零初始化三路输出逐位一致，最大绝对误差0；
- 301个预训练张量与Fusion基线一致；
- AMP、P/I/D梯度、推理接口和完整1轮训练管线通过；
- 推理参数：7,626,742，仅增加2,803；
- 512×640 FLOPs：14.8101 GFLOPs，基线14.7795 GFLOPs；
- RTX 2060同会话成对中位速度：基线43.99 FPS，ERCTC 39.44 FPS；两者均超过30 FPS。

