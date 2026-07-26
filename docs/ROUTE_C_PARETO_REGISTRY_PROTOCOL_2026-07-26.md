# Route C统一mIoU–延迟Pareto登记协议

日期：2026-07-26

## 1. 必须登记的方法

模板`configs/route_c/pareto_registry_template.csv`已经列出：

- Fusion PIDNet-S正式基线；
- 五个已闭环Route A模块：DEConv D2、mproto P1、DySample Pag4、
  DySample Context、FreqFusion Pag3；
- Route B ABL；
- Route C SAMF、TGM及可能组合；
- RoboFireFuseNet。

## 2. 不伪造坐标

Pareto图只绘制同时具有以下两项的行：

1. 相同`flame2_val_label_fix_seed200`验证协议的mIoU；
2. RTX 2060新协议下的同会话预热后中位延迟。

工程阶段未训练的DySample Context和FreqFusion Pag3没有mIoU，保留为pending；
RoboFireFuseNet在本项目同一验证划分重新评估前，不能使用论文其他划分上的报告值。训练期
only的mproto与ABL可以共享对应部署架构的延迟，但各自保留自己的验证mIoU。

不同阶段的结果必须在`stage`中显式标注，例如`30e screen`或`100e formal`，避免把筛选
结果包装成正式100轮结果。

## 3. 绘图与门槛

`src/build_route_c_pareto.py`会：

- 拒绝任何split名称含`test`的选择行；
- 拒绝重复method id、越界mIoU或非正延迟；
- 把缺失mIoU/延迟的行输出到`pending_rows.csv`而不是填0；
- 计算“延迟最小化、mIoU最大化”的Pareto前沿；
- 画出33.33 ms（30 FPS）实时门槛线；
- 输出绘图行、pending行、summary JSON和PNG。

该图是效率/精度关系证据，不改变30轮和100轮预注册精度门槛，也不允许依据测试集调整
结构或超参数。
