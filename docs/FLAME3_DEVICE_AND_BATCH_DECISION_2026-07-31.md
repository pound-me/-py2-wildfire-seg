# FLAME3 RTX 4090 设备与物理Batch修订预注册

日期：2026-07-31  
状态：用户批准，立即生效  
替代：`FLAME3_DEVICE_AND_BATCH_DECISION_2026-07-30.md` 中的 RTX 4070 规则

## 1. 设备分工

- RTX 4090 24GB：承担全部 FLAME3 30轮筛选、100轮正式训练和论文主表seed；
- RTX 2060：仅用于冻结模型零样本评估、轻量管线检查及最终部署FPS成对测量；
- 禁止将同一组正式对比拆到不同训练GPU；
- 禁止4090正式训练期间启动另一项GPU训练或测速；
- 最终部署速度仍在RTX 2060同会话成对测量，4090吞吐不替代部署指标。

## 2. Batch候选与选择规则

候选仅为物理batch `4` 和 `8`，使用相同的PIDNet-S fusion部分标签配置、640×512、AMP、seed 200。

每个候选在独立Python进程中执行：

1. 2步预热；
2. 至少20个连续前向、反向和optimizer step；
3. 记录loss有限性、梯度有限性、峰值allocated/reserved显存、每步中位时间和samples/s；
4. 不计算或比较验证精度。

冻结规则：

- batch 8 无OOM、无非有限loss/梯度，且峰值allocated显存不超过总显存80%，则固定batch 8；
- 否则batch 4满足同样稳定性条件时固定batch 4；
- 两者均失败则停止，不得自行启用梯度累积冒充相同物理batch；
- 结果写入 `flame3_4090_batch_preregistered.json` 后，30轮训练启动器才解除阻塞。

## 3. 公平性

- 一旦冻结，Fusion基线及后续所有进入正式比较的模块使用相同物理batch；
- 学习率、BatchNorm和优化器不因模型结果临时修改；
- 如未来模块无法在冻结batch运行，必须在产生该模块精度结果前登记协议修订，并重跑需要公平比较的基线；
- batch选择只依据工程稳定性和预注册显存余量，不依据任何模型精度。

## 4. 环境记录

首次运行前保存：主机名、GPU/显存、驱动、Python、PyTorch、CUDA、cuDNN、关键依赖、配置/源码/split SHA256和Git commit。环境文件、冒烟结果与batch结果进入独立审计目录，不写入训练实验目录。
