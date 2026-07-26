# Route A（RGB+IR 融合）正式路线决策

日期：2026-07-26

## 1. 决策背景

原主线以 RGB 输入的 PIDNet-S 为基线，依次评估 mproto、DEConv 与备用
DySample。P1 与 D2 的 100 轮正式训练均未达到内部方法成立门槛。DySample
Pag4 在 30 轮筛选中通过，但本轮新诊断显示，当前性能上限首先受输入模态
限制，而不只是网络结构或损失设计限制。

用户在诊断与 Fusion 训练开始前明确批准转向 Route A：先验证 RGB 可见性与
IR 单模态价值，再用官方 RoboFireFuseNet 数据用法建立四通道 PIDNet-S
Fusion 基线。

## 2. 数据与可见性证据

诊断仅使用验证集与已经保存的 RGB 基线验证预测，未读取测试集。

- 验证集：240 张，其中 180 张含 Fire 标注。
- Fire 总像素：1,406,630。
- RGB 低亮度 Fire 像素比例：
  - Y <= 60：0.098089
  - Y <= 80：0.180189（主阈值）
  - Y <= 100：0.292358
- Fire 像素处于 5 像素 Smoke 邻域内的比例：0.598542。
- Fire 与 Smoke 精确重叠比例：0。三分类标签互斥，因此精确重叠在定义上
  必然为零；本项目将 5 像素 Smoke 邻域明确称为遮挡/邻近代理，不冒充真实
  像素重叠。
- RGB 基线 Fire IoU 最差 20 张（仅选择含真实 Fire 的验证图）已经导出 RGB、
  IR、GT 与预测拼图和人工核查清单。人工观察显示，多数失败图包含大面积烟幕、
  小而碎片化的 Fire 区域；IR 中地表与热点结构通常更清楚。

证据目录：

`G:\py2\experiments\route_a_diagnostics\rgb_visibility_val`

## 3. IR 单通道 30 轮证据

运行：

`G:\py2\experiments\route_a_pidnet_s_ir\route_a_ir_30e_label_fix_seed200`

固定第 26--30 轮均值：

- mIoU：0.695315（相对 RGB +0.149334）
- Background IoU：0.761705
- Smoke IoU：0.666275
- Fire IoU：0.657964（相对 RGB +0.460124）
- Fire 保守波动范围：0.010669

结论：IR 对 Fire 的辨识能力远强于 RGB，证明新增模态具有直接价值；但 IR
单模态 Smoke IoU 低于 RGB 最佳模型，适合作为诊断证据，不作为最终输入方案。

## 4. RGB+IR Fusion 30 轮证据

运行：

`G:\py2\experiments\route_a_pidnet_s_fusion\route_a_fusion_30e_label_fix_seed200`

固定第 26--30 轮均值：

- mIoU：0.792495（相对 RGB +0.246514）
- Background IoU：0.901540
- Smoke IoU：0.829422
- Fire IoU：0.646524（相对 RGB +0.448684）
- Fire 保守波动范围：0.006796
- 无 Smoke/Fire 类别崩溃。

Fusion 相对 IR：mIoU +0.097181、Smoke +0.163147、Fire -0.011440。它保留了
绝大部分 IR Fire 优势，并显著恢复/提高 RGB 擅长的 Smoke 与 Background，符合
多模态互补预期。

## 5. 工程实现边界

- 输入数据严格沿用官方 RoboFireFuseNet 用法：RGB 三通道与 IR 一通道直接
  拼接为四通道，PIDNet-S 首层 `channels=4`，未引入额外融合模块。
- 官方来源：`third_party/RoboFireFuseNet/utils/training_tools.py` 中的
  `channels = {'rgb': 3, 'ir': 1, 'fusion': 4}`。
- ImageNet 预训练沿用官方“仅加载名称和形状均匹配的张量”规则。IR/Fusion
  都匹配 301 个张量；通道数发生变化的首层卷积不加载 RGB 三通道权重，保持
  官方行为并在配置中记录为 `official_shape_match_only`。
- 已修复 IR dataset 的通道维问题：`images[3]` 改为 `images[3:4]`，确保 batch
  为 `B x 1 x H x W`。该修复不改变像素值或标签。
- 所有新实验使用 `route_a_` 前缀，未覆盖任何既有目录。

## 6. DySample 备用路线闭环

DySample Pag4 已完成 30 轮并归档：

`G:\py2\experiments\pidnet_s_dysample\dysample_pag4_30e_label_fix`

第 26--30 轮均值为 mIoU 0.560060、Fire IoU 0.199149；相对同协议 RGB
基线 mIoU +0.014079，Fire 在保守噪声带内中性，按原备用筛选规则通过。
鉴于 Route A 已获得远大于该改进量的直接证据，DySample 暂不进入 100 轮，
其结果作为备用路线规则的闭环记录保留。

## 7. 正式决策与后续规则

1. Route A（RGB+IR Fusion）正式成为当前主路线。
2. Fusion PIDNet-S 立即进入 100 轮正式基线训练；完成后重新计算其第 26--30
   轮保守噪声带，并以验证集 mIoU 保存 best。
3. 只有 Fusion 100 轮结果完成并验收后，它才替代 RGB 成为论文主表正式基线；
   在此之前称为“新正式基线候选”。
4. RGB 基线不删除，保留为研究动机与模态消融材料。
5. DEConv 与 mproto 代码和历史结果不删除。Fusion 正式基线确立后，在完全相同
   的 Fusion 协议上重新进行 30 轮筛选，旧 RGB 结果不直接作为 Fusion 下的模块
   结论。
6. 后续结构门槛、噪声带和 100 轮比较全部以 Fusion 正式基线为参照。
7. 测试集继续封存；本次路线决策、结构选择与 checkpoint 选择均未使用测试集。

批准依据：用户在 2026-07-26 明确下达 Route A 执行顺序，并要求诊断与 Fusion
30 轮证据完成后将路线变更正式写入决策日志。本文件在上述证据全部生成后写入。
