# Route C轻量RGB+IR融合候选预调研（历史候选背景）

日期：2026-07-26
状态：**Route C已正式启动；本文件保留为启动前候选调研背景。首个获批实现不是完整
CMX移植，而是诊断驱动的极轻SAMF。**

## 1. 调研边界

Route C的先决条件已经满足：B路线ABL正式失败并关闭，用户批准以RTX 2060至少30 FPS
为硬门槛的实测实时预算，并明确授权SAMF作为首个候选。候选必须改善当前“RGB与IR仅在
输入层直接拼接”的融合，且外部模块仍优先使用作者官方代码和清晰许可证。

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

启动前调研曾把“CMX启发的低通道、单位置跨模态校正与融合”列为第一研究对象。正式
启动决定进一步使用了项目自身诊断证据：worst-20中RGB火焰不可见、heavy烟遮挡和IR更
清晰均为20/20，因此当前首个研究对象改为烟概率门控热注入SAMF。CMX继续作为跨模态
融合相关工作和后备机制参考，不能把SAMF表述为CMX复现。

上表四个外部方法仍只是方向优先级，不构成移植许可；SAMF则已由用户单独批准实施。
外部方法若后续动工仍必须另外完成：

1. 明确插入点，只允许1个理论最强位置，避免排列组合；
2. 独立计算简化模块参数/FLOPs并完整报告；
3. RTX 2060同会话成对测速，证明预热后多轮AMP中位延迟≤33.33 ms；
4. 明确哪些代码来自MIT官方仓库、哪些是本项目独立适配；
5. 重新写候选触发与30轮筛选文档，再由用户批准训练。

EAEFNet在许可证澄清前不得进入实现；DFormer和TokenFusion不作为首个C候选。当前按
SAMF工程验收、30轮筛选、达标后100轮的顺序执行，TGM在SAMF筛选冻结前不得动工。
