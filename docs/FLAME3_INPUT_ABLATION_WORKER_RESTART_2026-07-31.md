# FLAME3 输入消融Windows Worker重启审计

日期：2026-07-31  
状态：精度结果产生前的工程异常；正式协议不变

## 事件

首次计划任务按RGB→IR顺序启动，RGB进程创建了环境、resolved config与空的`metrics.jsonl`，但数分钟内：

- metrics保持0行；
- GPU连续10秒约1%利用率；
- 主进程CPU时间在5秒窗口内不增长；
- 4个DataLoader worker存在但无有效进度；
- 未生成`best.pth`、`last.pth`或任何epoch精度记录。

因此判定为Windows multiprocessing worker一次性挂起。停止计划任务及其进程树，保留原空目录，不删除、不复用。

## 诊断

使用同一RGB配置、物理batch 8、AMP、seed 200与`num_workers=4`执行：

1. 单train batch + 单validation batch完整管线：通过；
2. 完整单epoch（61 train batches + 17 validation batches）：通过，耗时38.7秒；
3. 两次均匹配301个预训练骨干张量，stem显式跳过；
4. loss和梯度有限，无OOM；未读取测试集。

由此不能认定`num_workers=4`或RGB实现存在可复现错误，正式worker协议保持不变。上述两个诊断run仅用于工程定位，不进入筛选或论文表格。

## 重启规则

- RGB与IR正式30轮统一使用后缀`_retry1`；
- 仍按RGB→IR顺序串行；
- 所有模型、数据、增强、batch、seed、学习率与初始化规则不变；
- 原空目录作为失败审计保留；
- 若同一worker挂起在`_retry1`再次出现，才登记协议修订并评估`num_workers=0`，不得静默修改。
