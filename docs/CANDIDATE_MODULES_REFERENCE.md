# 候选模块与论文出处参考表

整理日期：2026-07-25
用途：PIDNet-S 火焰/烟雾三分类改进的可移植模块清单，按插入位置分组。
状态标记：[已采纳] 进入当前实验计划；[备用] 主方法未通过筛选时启用；[参考] 仅借鉴思想，不直接移植。

代码仓库核实说明：标注"已核实"的仓库与论文作者对应关系比较可靠；其余使用前请自行核对
仓库归属（README、论文链接、作者名）。PSL 仓库以 `docs/RECENT_SMOKE_SEGMENTATION_PAPERS.md`
中 2026-07-24 的核实结果为准。

---

## 一、P 分支：小目标细节增强

| 模块 | 论文 | 出处 / 年份 | 机制简述 | 推理开销 | 代码仓库 | 状态 |
|---|---|---|---|---|---|---|
| DEConv（Detail-Enhanced Convolution） | DEA-Net: Single Image Dehazing Based on Detail-Enhanced Convolution and Content-Guided Attention | IEEE TIP, 2024 | 普通卷积 + 中心差分/角差分/水平差分/垂直差分卷积五路并联；卷积对核线性，训练可先合核再做一次卷积；部署重参数化为单个 3×3 | 零（重参数化后与基线完全一致） | https://github.com/cecret3350/DEA-Net （已核实） | [已采纳] D1/D2 变体 |
| WTConv（小波卷积） | Wavelet Convolutions for Large Receptive Fields | ECCV, 2024 | Haar 小波分解后在低/高频子带分别做 depthwise 卷积，保高频同时扩大感受野 | 很小（depthwise 级） | https://github.com/BGU-CS-VIL/WTConv （已核实） | [参考] |
| HWD（Haar 小波下采样） | Haar Wavelet Downsampling: A Simple but Effective Downsampling Module for Semantic Segmentation | Pattern Recognition, 2023 | 用 Haar 变换替换 stride-2 卷积/池化，下采样不丢高频 | 约等于原下采样 | https://github.com/apple1986/HWD （已核实） | [参考] |
| SPD-Conv | No More Strided Convolutions or Pooling: A New CNN Building Block for Low-Resolution Images and Small Objects | ECML-PKDD, 2022 | space-to-depth 替换 stride 下采样，小目标信息无损转入通道维 | 小 | https://github.com/LabSAINT/SPD-Conv （已核实） | [参考] |

## 二、烟/火特征分离（训练期约束）

| 方法 | 论文 | 出处 / 年份 | 机制简述 | 推理开销 | 代码仓库 | 状态 |
|---|---|---|---|---|---|---|
| PSL（BES + PUO） | Prototype-based Scatter Learning for Smoke Segmentation | Pattern Recognition, Vol.172, 2026, Article 112605, DOI: 10.1016/j.patcog.2025.112605 | 每类多原型 + Bottom-K 特征值散度损失（类内聚合）+ 原型去相关（防退化） | 零（训练期 only） | https://github.com/LujianYao/psl （以项目文档核实结果为准） | [已采纳] dfm 后多原型分离的主要参照 |
| 跨图像像素对比 + 记忆库 | Exploring Cross-Image Pixel Contrast for Semantic Segmentation | ICCV, 2021 | 像素级 InfoNCE，EMA 记忆库跨 batch 积累各类样本，缓解小 batch 下稀有类样本不足 | 零（训练期 only） | https://github.com/tfzhou/ContrastiveSeg （已核实） | [参考] EMA 跨批次机制的参照 |
| FoSp | FoSp: Focus and Separation Network for Early Smoke Segmentation | arXiv:2306.04474, 2023 | 透射率引导，把烟雾从背景域显式分离；整体结构偏重 | 较大 | 见论文页 | [参考] 只借鉴分离思想 |

## 三、D 分支 / 边界（损失级）

| 方法 | 论文 | 出处 / 年份 | 机制简述 | 推理开销 | 代码仓库 | 状态 |
|---|---|---|---|---|---|---|
| CBL（Conditional Boundary Loss） | Conditional Boundary Loss for Semantic Segmentation | IEEE TIP, 2023 | 边界像素拉向本类局部均值、推离邻近他类；同时做边界锐化与边界处类别分离 | 零（损失 only） | 使用前自行核对官方仓库 | [参考] 可作 mproto 的替代消融 |
| ABL（Active Boundary Loss） | Active Boundary Loss for Semantic Segmentation | AAAI, 2022 | 预测边界向真值边界法向逐步对齐 | 零（损失 only） | 使用前自行核对官方仓库 | [参考] |

## 四、Light-Bag / 融合与上采样

| 模块 | 论文 | 出处 / 年份 | 机制简述 | 推理开销 | 代码仓库 | 状态 |
|---|---|---|---|---|---|---|
| DySample | Learning to Upsample by Learning to Sample | ICCV, 2023 | 点采样式动态上采样，替换 bilinear `F.interpolate`；参数量仅数 K | 极小 | https://github.com/tiny-smart/dysample （已核实） | [备用] |
| FreqFusion | Frequency-Aware Feature Fusion for Dense Image Prediction | IEEE TPAMI, 2024 | 融合时对低分辨率特征自适应低通、高分辨率特征补高频，针对上采样边界模糊与小目标被粗特征淹没 | 小 | https://github.com/Linwei-Chen/FreqFusion （采用官方仓库内 Apache-2.0 的 SegNeXt 集成版；顶层文件不使用） | [备用，内部学术实验已授权] |
| CAA（Context Anchor Attention） | Poly Kernel Inception Network for Remote Sensing Detection (PKINet) | CVPR, 2024 | 条带卷积（1×k / k×1）构成的轻量方向注意力，适合细长烟羽 | 小 | https://github.com/NUST-Machine-Intelligence-Laboratory/PKINet （使用前自行核对） | [参考] |

## 五、领域专用网络（整网参考，不直接移植）

| 网络 | 论文 | 出处 / 年份 | 可借鉴点 | 代码 | 状态 |
|---|---|---|---|---|---|
| TANet | Texture-Aware Network for Enhancing Inner Smoke Representation in Visual Smoke Density Estimation | IET Computer Vision, 2025, DOI: 10.1049/cvi2.70023 | 方向长卷积核（1×5/5×1 depthwise）建模烟羽方向与内部纹理 | https://github.com/xia-xx-cv/TANet_smoke | [参考] |
| SmokeNet | SmokeNet: Efficient Smoke Segmentation Leveraging Multiscale Convolutions and Multiview Attention Mechanisms | arXiv:2502.12258, 2025 | 矩形多尺度卷积、逐层监督；未找到官方代码 | 无官方实现（截至 2026-07-24 核实） | [参考] |
| EGNL-FAT | An Edge-Guided Non-Local Network with Frequency-Aware Transformer for Smoke Segmentation | Expert Systems with Applications, Vol.280, 2025, Article 127621 | 边缘引导非局部 + 频率感知；结构偏重，仅理论参考 | 未找到官方代码（截至 2026-07-24 核实） | [参考] |

---

## 与当前实验计划的对应关系

1. [已采纳] P 分支 DEConv（pidnet_s_deconv，D1/D2 变体，合核训练 + 零初始化等价 + 部署重参数化）。
2. [已采纳] dfm 后多原型分离（pidnet_s_dfm_mproto，PSL 式多原型 + EMA 0.99 跨批次 + 类内去相关；
   Fire 采用混合点采样：256 均匀 + 256 按连通域均匀，基于与标签同步增强的连通域 ID 图）。
3. [备用] DySample / FreqFusion：仅当 DEConv 与多原型均未通过 30 轮筛选时启用。
4. 已被本项目阴性实验排除的路线：单独增加烟雾二值辅助（LSCM v2/v2.1）、单独火焰边界损失
   （fire_boundary）、单独火焰区域损失（fire_region）。详见 docs/ 下各实验记录。
