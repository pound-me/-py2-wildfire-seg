# FLAME3 Fusion 验证集错误画像协议（2026-07-31）

## 目的

三模态 100 轮输入消融已经完成。Fusion 相对 IR-only 的 Fire IoU 提升为
`+0.066628`，证明 RGB 与热信息联合有效；但现有 Fusion 只是四通道直接拼接。
在选择新的融合模块以前，先对冻结的 Fusion seed-200 最佳 checkpoint 做一次只读
验证集错误画像，以确定剩余误差究竟来自小火区、边界、远离火区的误报、低可见 RGB
区域还是特定温度区间。

## 数据与纪律

- 只接受冻结的 split-v2 `val.csv`，固定 134 张。
- 只读取 Fusion seed-200 的 `best.pth`，不训练、不更新参数。
- 测试集继续封存；脚本拒绝非 `val.csv` 输入。
- FLAME3 温度掩膜是部分标签，不是完整三分类真值。因此只报告温度伪标签定义的
  active-Fire 二分类指标，不报告 Smoke IoU 或三分类 mIoU。
- Fire 文件夹非 Fire-core 区域仍按已注册的部分标签协议用于 Fire 误报统计；不得把
  其中的 Background/Smoke 预测差异解释为完整语义混淆。

## 固定诊断定义

### 连通域

- 在验证分辨率 640×512 的 Fire core 上使用 8 邻域连通域。
- 用全验证集 GT Fire 连通域面积分布的 Q1、Q3 形成三档：
  `area <= Q1`、`Q1 < area <= Q3`、`area > Q3`。
- 逐连通域保存面积、TP、FN、召回率和外接框。
- 预测 Fire 连通域使用同一组 GT 阈值分桶，保存面积、TP、FP和精确率。

### 边界与内部

- 二值 Fire 边界定义为 Fire mask 减去 3×3 腐蚀结果。
- 默认将 GT 边界膨胀 3 像素形成边界带。
- FN 分为边界带内与 GT Fire 内部；FP 分为 GT 边界带附近与远离 GT 边界。
- 两组分解必须分别与全局 FN、FP 像素数严格守恒。

### 温度与 RGB 可见性

- Celsius TIFF 按固定温度区间统计 TP、FN、FP、TN 的像素数、比例、均值和标准差。
- RGB 亮度采用 BT.709 luma，范围 `[0,1]`。
- 局部对比度采用 15×15 邻域内 luma 标准差。
- 描述性低可见阈值固定为 `luma < 0.20` 或局部对比度 `< 0.05`；同时保留连续分桶，
  避免结论只依赖单一阈值。

### 空 Fire-core 与 No Fire

- 分别统计：有 Fire core 的 Fire 文件夹、空 Fire-core 的 Fire 文件夹、No Fire 文件夹。
- 每组保存图像数、有效像素数、预测 Fire 像素数与误报比例。

## 输出

- `summary.json`：全局指标、连通域、边界/内部、温度、亮度和误报组汇总。
- `per_image_metrics.csv`：逐图 Fire IoU、precision、recall、FN/FP 分解。
- `gt_fire_components.csv` 与 `predicted_fire_components.csv`：逐连通域记录。
- `predictions_raw/`：验证集原始类别预测。
- 最差 20 张有 Fire-core 图、最差 5 张空 Fire-core 图、最差 5 张 No Fire 图的
  单图五联图和 contact sheet。
- 全部 16 张空 Fire-core 图的审核拼图与 `empty_fire_core_manual_review_checklist.csv`。
- 额外给出仅限“非空 Fire-core 图”的描述性敏感性指标，但它不替换预注册的全验证集
  选择指标，也不得用于选择 checkpoint。

空 Fire-core 只表示没有达到冻结规则要求的 200℃ 高置信种子，不是人工像素级负标签。
因此该组的“FP”只能解释为模型与温度伪标签规则不一致，必须结合人工语义审核，不能直接
宣称模型把真实背景误判成火。

## 决策用途

错误画像完成前不实现新融合模块。后续候选必须直接对应主要剩余误差：

- 若 FN 主要集中在小连通域：优先多尺度/小目标热特征保留；
- 若 FN/FP 主要集中在边界：优先轻量边界对齐或热引导上采样；
- 若 FN 主要位于高温但 RGB 低可见区域：优先热主导、RGB 条件校正式融合；
- 若 FP 主要远离 GT 且处于低温区：优先温度置信门控或跨模态一致性约束；
- 若误差没有单一结构性来源，再考虑更通用的轻量双流融合，而不是随意堆模块。
