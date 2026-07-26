# Route C Table 1复杂度与速度测量协议

日期：2026-07-26
状态：工具已准备；等待IR-only 100轮best checkpoint后执行。

## 1. Table 1行与列

固定三行：RGB-only PIDNet-S、IR-only PIDNet-S、Fusion PIDNet-S。固定列为：

- Background / Smoke / Fire IoU；
- mIoU；
- 部署参数量；
- 256×256前向GFLOPs；
- RTX 2060 AMP batch-1中位FPS。

三类IoU与mIoU均来自各自100轮训练中按验证集mIoU选择的best checkpoint。测试集不用于
该表，也不因Table 1结果改变结构或超参数。

## 2. 同会话速度协议

使用`src/benchmark_route_c_input_modes.py`在同一Python进程中同时加载三种输入模型：

1. 每个模型独立使用对应通道数的`1×C×256×256`输入；
2. 每个模型先预热100次；
3. 每个trial前向200次，默认重复12组；
4. 三模型测量顺序按RGB→IR→Fusion、IR→Fusion→RGB、Fusion→RGB→IR循环，降低
   GPU时钟和系统负载漂移；
5. 以每个模型12组trial的中位延迟换算FPS，同时保存mean/std/P95和全部原始值；
6. 至少30 FPS（中位延迟不超过33.33 ms）为实时硬门槛。

不同输入通道数需要不同随机张量，因此不能使用旧的“强制同通道”双模型脚本。新脚本仍
保持同进程、同会话、相同预热/重复次数和轮换顺序，满足当前正式协议。

## 3. 参数与FLOPs

参数和FLOPs继续使用`src/measure_complexity.py`、相同256×256输入和相同profiler定义。
理论FLOPs不能替代实测延迟；两者都进入最终Table 1与后续Pareto登记。

## 4. 执行纪律

- 必须等IR 100轮训练完成并在验证集评估best checkpoint；
- 测速时不同时运行其他训练；
- 保存GPU、PyTorch、CUDA、cuDNN、配置哈希、原始trial和门槛判定；
- 正在写入的训练日志/checkpoint不提交，测速JSON与Table 1归档文档完成后再commit/push。
