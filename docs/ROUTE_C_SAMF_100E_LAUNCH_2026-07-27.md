# Route C SAMF 100 轮正式训练启动记录

启动时间：2026-07-27 00:13（Asia/Shanghai）

状态：运行中；运行日志与指标在写入完成前不提交。

## 冻结内容

- 模型：`pidnet_s_samf`；
- 配置：`configs/route_c/pidnet_s_fusion_samf_100e_label_fix.yaml`；
- 运行名：`route_c_samf_100e_label_fix_seed200`；
- 输入：Fusion（RGB + IR，第 4 通道为 IR）；
- seed：200；GPU：RTX 2060 / `cuda:0`；
- epoch：100；batch size：4；AMP：开启；
- 学习率调度总长度：100 epoch；
- 初始化：PIDNet-S ImageNet 预训练，匹配 301 个张量；
- 不从 30 轮 checkpoint 续训；
- 测试集继续封存。

## 进程与路径

- PID：5572；
- 实验目录：`experiments/route_c_pidnet_s_samf/route_c_samf_100e_label_fix_seed200`；
- stdout：`launch_stdout.log`；
- stderr：`launch_stderr.log`。

启动后已确认进入 `Epoch 1/100`，训练 loss 正常下降，stderr 为空。按 30 轮实测速度估计总时长约 60–65 分钟。

## 完成后的固定动作

1. 核对 100/100 轮、stderr、环境/config/source hash 与运行摘要；
2. 按验证集 mIoU 的 best checkpoint 计算三类 IoU、Fire precision/recall 与边界 F1；
3. 与 Fusion 正式基线应用预注册方法成立门槛；
4. 正式 checkpoint 上重测参数、FLOPs及 RTX 2060 同会话成对延迟，更新 Pareto 数据；
5. 归档结果并 commit/push；
6. 只有 SAMF 正式结论归档后才决定是否启动 TGM。
