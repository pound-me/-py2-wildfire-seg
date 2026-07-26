# Route C verified bibliography

核实日期：2026-07-26

本表只收录能够核实正式标题、出版来源/年份与原文或正式元数据的条目。论文正文使用正式
出版年；PSL的DOI在2025年登记、正式卷期为2026年，RoboFireFuseNet正式发表于2026年。
FLAME 2是IEEE DataPort数据集条目，不写成期刊论文。

对应BibTeX：`docs/verified_references.bib`。

| Key | 正式标题 | 出处/年份 | DOI | 官方代码/数据 | 代码复用状态 |
|---|---|---|---|---|---|
| `wang2023utfnet` | UTFNet: Uncertainty-Guided Trustworthy Fusion Network for RGB-Thermal Semantic Segmentation | IEEE GRSL, 2023 | 10.1109/LGRS.2023.3322452 | `KustTeamWQW/UTFNet` | MIT |
| `zhang2023cmx` | CMX: Cross-Modal Fusion for RGB-X Semantic Segmentation With Transformers | IEEE T-ITS, 2023 | 10.1109/TITS.2023.3300537 | `huaaaliu/RGBX_Semantic_Segmentation` | MIT |
| `zhang2023cmnext` | Delivering Arbitrary-Modal Semantic Segmentation（CMNeXt） | CVPR, 2023 | 10.1109/CVPR52729.2023.00116 | `InSAI-Lab/DELIVER` | Apache-2.0 |
| `wang2023sgfnet` | SGFNet: Semantic-Guided Fusion Network for RGB-Thermal Semantic Segmentation | IEEE TCSVT, 2023 | 10.1109/TCSVT.2023.3281419 | `kw717/SGFNet` | **NO LICENSE—reference only** |
| `yin2025dformerv2` | DFormerv2: Geometry Self-Attention for RGBD Semantic Segmentation | CVPR, 2025 | 10.1109/CVPR52734.2025.01802 | `VCIP-RGBD/DFormer` | MIT |
| `fotiou2026robofirefusenet` | RoboFireFuseNet: Robust Fusion of Visible and Infrared Wildfire Imaging for Real-Time Flame and Smoke Segmentation | Pattern Recognition Letters, 2026 | 10.1016/j.patrec.2026.04.024 | `dimfot3/RoboFireFuseNet` | 代码/分割标注MIT；原始图像按FLAME 2 CC BY 4.0 |
| `hopkins2022flame2` | FLAME 2: Fire detection and modeLing: Aerial Multi-spectral imagE dataset | IEEE DataPort Dataset, 2022 | 10.21227/swyw-6j78 | IEEE DataPort | CC BY 4.0 |
| `wang2022abl` | Active Boundary Loss for Semantic Segmentation | AAAI, 2022 Oral | 10.1609/aaai.v36i2.20139 | `wangchi95/active-boundary-loss` | Apache-2.0 |
| `wu2023cbl` | Conditional Boundary Loss for Semantic Segmentation | IEEE TIP, 2023 | 10.1109/TIP.2023.3290519 | `dywu98/CBL-Conditional-Boundary-Loss` | **NO LICENSE—reference only** |
| `liu2023dysample` | Learning to Upsample by Learning to Sample | ICCV, 2023 | 10.1109/ICCV51070.2023.00554 | `tiny-smart/dysample` | MIT |
| `chen2024freqfusion` | Frequency-Aware Feature Fusion for Dense Image Prediction | IEEE TPAMI, 2024 | 10.1109/TPAMI.2024.3449959 | `Linwei-Chen/FreqFusion` | 顶层NO LICENSE；仅SegNeXt子树Apache-2.0 |
| `xu2023pidnet` | PIDNet: A Real-time Semantic Segmentation Network Inspired by PID Controllers | CVPR, 2023 | 10.1109/CVPR52729.2023.01871 | `XuJiacong/PIDNet` | MIT |
| `yao2026psl` | Prototype-based Scatter Learning for Smoke Segmentation | Pattern Recognition 172, 112605, 2026 | 10.1016/j.patcog.2025.112605 | `LujianYao/psl` | **NO LICENSE—reference only** |

## 可安全陈述的核心结论

- UTFNet：以Dirichlet证据估计RGB/热红外模态不确定性，并用Dempster--Shafer理论融合，
  适合支撑“模态可靠性会随场景变化”的相关工作论述。
- CMX：CM-FRM进行跨模态校正，FFM在融合前交换长距离上下文；覆盖RGB-T等多种X模态，
  是RGB+IR融合机制的重要参考，但SAMF不是CMX复现。
- CMNeXt：SQ-Hub与Parallel Pooling Mixer支持任意数量模态；适合相关工作，不作为当前
  单点轻量SAMF的直接源码来源。
- SGFNet：多模态语义信息引导RGB/热红外融合，并在MFNet/PST900验证。因无法开放核对
  全文细节且代码无LICENSE，不写更细的模块断言，不复制源码。
- DFormerv2：从深度与图像块的空间距离构造几何注意力先验。IR不是深度，不能把RGB-D
  几何结论直接套到本项目。
- RoboFireFuseNet：面向可见光/红外野火分割，论述烟遮挡、小而稀疏火焰和实时部署；
  本项目只按其官方数据输入用法建立RGB/IR/Fusion基线，不冒充复现其完整网络。
- FLAME 2：2021年计划燃烧的配准RGB/IR无人机数据及相关辅助资料；正式引用采用DataCite/
  IEEE DataPort七位创作者元数据，不采用README中多出的作者。
- ABL：训练期将预测边界向真实边界逐步对齐，不增加推理结构。
- CBL：对边界像素构造局部条件优化，拉近本类中心并推离异类邻居；无LICENSE，只参考。
- DySample：以点采样方式实现动态上采样，避免动态卷积和自定义CUDA。
- FreqFusion：以低通、偏移重采样和高通机制改善融合特征一致性与边界；只允许使用官方
  仓库内带Apache-2.0的SegNeXt集成来源。
- PIDNet：P/I/D三分支分别建模细节、上下文和边界，并通过边界注意力融合。
- PSL：BES通过Bottom-K特征值增强可分性，PUO在Stiefel流形优化原型防止退化；本项目
  使用的是独立的feature-side decorrelation surrogate，不能声称复现原PUO。

## 授权纪律

SGFNet、CBL、FreqFusion顶层源码和PSL官方仓库均缺少顶层LICENSE。它们可以用于阅读、
内部数值核验和思想参考，但不能默认复制后公开发布。FreqFusion当前内部实验只采用官方
仓库中明确带Apache-2.0的SegNeXt子树；若最终方法需要其他未授权代码，投稿前必须取得
作者许可、依据论文独立重实现，或改为固定commit的拉取校验脚本。
