# Route C轻量RGB+IR融合候选预调研（不实施）

日期：2026-07-26
状态：**仅调研。预算制提案尚未获导师批准，不下载实现、不修改网络、不启动训练。**

## 1. 调研边界

Route C只在以下条件同时满足时才可能启动：B路线ABL关闭或完成、导师批准FLOPs≤+5%
预算制、用户再次授权具体候选。候选必须改善当前“RGB与IR仅在输入层直接拼接”的融合，
且优先具有作者官方代码和清晰许可证。

本轮只核实方法方向、官方仓库、许可证和与PIDNet-S的理论匹配，不把整网结果直接当成
本项目预期收益，也不提前移植源码。

## 2. 候选审计

| 优先级 | 方法 | 论文/年份 | 官方仓库与固定审计点 | 许可证 | 对本项目的判断 |
|---|---|---|---|---|---|
| 1 | CMX的Cross-Modal Feature Rectification与Feature Fusion思想 | *CMX: Cross-Modal Fusion for RGB-X Semantic Segmentation with Transformers*，IEEE T-ITS 2023 | `https://github.com/huaaaliu/RGBX_Semantic_Segmentation`，审计HEAD `e251d860aebc2f583a6c4919877e6bebe7f1aff3` | MIT | 官方结果明确包含RGB-T MFNet，模态最匹配。完整双Transformer整网不符合PIDNet轻量目标；若C获批，只研究低通道、单位置的简化FRM/FFM思想，不能声称复现完整CMX。 |
| 2 | EAEF显式注意力融合 | *EAEFNet: Explicit Attention-Enhanced Fusion for RGB-Thermal Perception Tasks* | `https://github.com/FreeformRobotics/EAEFNet`；README含MFNet/PST900分割入口 | **未通过授权审计**：README有MIT徽章，但仓库根目录无LICENSE，GitHub许可证元数据为空 | 任务与模态高度匹配、模块化潜力较好，但在取得作者许可或找到明确授权版本前不能移植或发布源码。 |
| 3 | DFormer/DFormerv2几何注意力 | *DFormer*，ICLR 2024；*DFormerv2*，CVPR 2025 | `https://github.com/VCIP-RGBD/DFormer`，审计HEAD `814799bb1f39eb380f72fdea1cd591f2cc27b6aa` | MIT | 官方、近期且代码完整，但核心先验面向RGB-D几何；FLAME2的IR不是深度。除非后续能提出热红外可靠性替代几何先验，否则理论错配，不作为首选。 |
| 4 | TokenFusion | *Multimodal Token Fusion for Vision Transformers*，CVPR 2022 | `https://github.com/yikaiw/TokenFusion`，审计HEAD `3834ccf7765bb0bd50ea729069ad5adbd6de288d` | MIT | 通过token替换进行跨模态交互，但依赖Transformer token结构，年份也早于前三者；对CNN式PIDNet适配成本较高，仅保留为机制参考。 |

## 3. 初步结论

若未来正式启动C路线，第一研究对象应是“CMX启发的低通道、单位置跨模态校正与融合”，
原因是它有MIT官方代码、直接覆盖RGB-T分割，且研究问题与当前直接拼接短板一致。

这只是方向优先级，不是实现许可。正式动工前必须另外完成：

1. 明确插入点，只允许1个理论最强位置，避免排列组合；
2. 独立计算简化模块参数/FLOPs，先证明≤+5%；
3. RTX 2060同会话成对测速，证明AMP中位延迟≤33.33 ms；
4. 明确哪些代码来自MIT官方仓库、哪些是本项目独立适配；
5. 重新写候选触发与30轮筛选文档，再由用户批准训练。

EAEFNet在许可证澄清前不得进入实现；DFormer和TokenFusion暂不作为首个C候选。当前仍只
执行B路线ABL，本调研不改变任何已冻结门槛或测试集纪律。
