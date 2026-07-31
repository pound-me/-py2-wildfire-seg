# FLAME3 输入消融执行方式最终审计

日期：2026-07-31  
状态：前台PowerShell执行方式冻结，训练超参数不变

## 被排除的工程run

- `flame3_rgb_partial_30e_seed200`：计划任务嵌套串行脚本，0个epoch；
- `flame3_rgb_partial_30e_seed200_retry1`：同类嵌套启动，0个epoch；
- `flame3_rgb_partial_30e_seed200_retry2`：单独计划任务可训练，但DataLoader周期性停顿，只完成2个epoch且每轮约217秒；
- `flame3_rgb_workers4_pipeline_check`：单batch工程诊断；
- `flame3_rgb_workers4_epoch_check`：单epoch工程诊断。

以上run不进入筛选、续训或论文表格，均保留作审计。

## 正式执行方式

同配置在用户可见的交互式PowerShell前台运行时，`num_workers=4`完整epoch约32–39秒，无周期性停顿。因此正式30轮采用：

- RGB：`flame3_rgb_partial_30e_seed200_retry3`；
- IR：`flame3_ir_partial_30e_seed200_retry1`；
- 前台PowerShell直接调用单模态启动器；
- 100轮续训沿用同一执行方式；
- 不修改模型、seed、batch、worker、数据增强或学习率协议。

该调整仅改变Windows进程承载方式，不改变训练算法或数据。测试集始终未读取。
