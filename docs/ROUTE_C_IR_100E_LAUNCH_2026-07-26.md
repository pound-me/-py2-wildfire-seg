# Route C节点1c：IR-only 100轮启动记录

日期：2026-07-26
状态：配置冻结，等待/开始100轮训练。

## 1. 目的

该实验不是新结构候选，而是Table 1的输入模态动机行：在完全相同的label_fix A1训练
协议、seed 200和PIDNet-S骨干下，对RGB-only、IR-only、Fusion三种输入进行100轮比较。
测试集保持封存，checkpoint仍只按验证集mIoU保存。

## 2. 冻结身份

- 配置：`configs/route_c/pidnet_s_ir_100e_label_fix.yaml`；
- 模式：`MODE: ir`，单通道输入；
- 运行名：`route_c_ir_100e_label_fix_seed200`；
- 实验组：`route_c_pidnet_s_ir`；
- seed：200；epoch：100；LR horizon：100；AMP；
- GPU：RTX 2060，`cuda:0`；
- 预训练：PIDNet-S ImageNet权重，按官方shape-match-only规则加载；
- 数据：严格三分类label_fix解码；A1协议关闭强亮度变换，scale 0.5--1.5；
- 不从既有30轮checkpoint续训，从ImageNet初始化重新开始。

## 3. 完成后动作

1. 保存文本日志、resolved config、environment、逐轮metrics与run summary；
2. 不提交`best.pth`、`last.pth`或正在写入的日志；
3. 用best checkpoint只评估验证集；
4. 与RGB-only和Fusion正式best组成Table 1：三类IoU、Params、GFLOPs、FPS；
5. FPS按新规则在RTX 2060同一会话中成对测量，预热后取多轮中位数；
6. 结果归档后单独commit并push，测试集不使用。

## 4. 启动命令

```powershell
& "F:\anaconda3\envs\pytorch\python.exe" `
  "G:\py2\src\train_baseline.py" `
  --config "G:\py2\configs\route_c\pidnet_s_ir_100e_label_fix.yaml" `
  --run-name route_c_ir_100e_label_fix_seed200 `
  --amp
```
