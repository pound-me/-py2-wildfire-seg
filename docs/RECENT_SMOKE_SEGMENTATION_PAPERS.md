# 近期烟雾分割方法筛选

核实日期：2026-07-24  
目的：为 UAV RGB 三分类 PIDNet-S 的轻量烟雾上下文模块提供可实施参考。

## 1. 最高优先级：Prototype-based Scatter Learning

论文：Lujian Yao et al., **Prototype-based Scatter Learning for Smoke Segmentation**, Pattern Recognition, Volume 172, 2026, Article 112605.  
DOI：<https://doi.org/10.1016/j.patcog.2025.112605>  
作者开源代码：<https://github.com/LujianYao/psl>

关键方法：

- Bottom-K Eigenvalues Scatter（BES）loss，提高困难烟雾模式的特征可分性。
- Prototype Uncorrelated Optimization（PUO），保持多原型多样性，避免原型退化。
- 针对模糊边缘和多样烟雾形态进行类内聚合、类间分离。

与本项目关系：当前 LSCM v2 出现 Smoke IoU 提高、Fire IoU 下降，说明 smoke-vs-rest 辅助任务可能压缩了 smoke 与 fire 的区分空间。PSL 的“多类别原型分离”比继续调二值辅助权重更对症。

建议阅读：Abstract、PSL Overview、BES Loss、PUO；不需要先复现整套 MMSegmentation 工程。

## 2. 高优先级：TANet

论文：Xue Xia et al., **Texture-Aware Network for Enhancing Inner Smoke Representation in Visual Smoke Density Estimation**, IET Computer Vision, 2025.  
DOI：<https://doi.org/10.1049/cvi2.70023>  
作者开源代码：<https://github.com/xia-xx-cv/TANet_smoke>

关键方法：

- 场景信息与纹理信息分路径、自适应融合。
- 使用方向长卷积核捕获烟雾的方向依赖与内部纹理。
- 定位与密度恢复双任务解码。
- 最终阶段使用频域对齐保存内部细节。

与本项目关系：FLAME2 只有三分类硬标签，不能直接照搬烟雾密度回归；但方向长核可轻量移植到 LSCM，将 `3x3 dilation` 分支改为或补充为 depthwise `1x5`、`5x1` 分支。

建议阅读：Abstract、Section 3.2 Texture-Aware Head、双任务解码器、Table 6 长程与短程纹理模块消融。

## 3. 方法参考：SmokeNet

论文：Xuesong Liu and Emmett J. Ientilucci, **SmokeNet: Efficient Smoke Segmentation Leveraging Multiscale Convolutions and Multiview Attention Mechanisms**, arXiv:2502.12258, 2025.  
论文：<https://arxiv.org/abs/2502.12258>

关键方法：

- 标准核与矩形核组成多尺度卷积，适应细长或横向扩散烟羽。
- 轻量 multiview linear attention，同时建模空间和通道信息。
- layer-specific losses 做中间层监督。
- 论文报告约 0.34M 参数、0.07 GFLOPs、77.05 FPS（其统计输入和硬件不可直接与本项目横比）。

截至核实日期，未找到由论文作者明确发布的官方实现。搜索结果中出现的 ERFNet GitHub 是引用的基础网络，不是 SmokeNet 官方代码。

建议阅读：Section 3 Methodology 和 Table 1 消融；重点借鉴矩形多尺度卷积，不直接更换 PIDNet 主干。

## 4. 方法参考：EGNL-FAT

论文：Yitong Fu et al., **EGNL-FAT: An Edge-Guided Non-Local Network with Frequency-Aware Transformer for Smoke Segmentation**, Expert Systems with Applications, Volume 280, 2025, Article 127621.  
DOI：<https://doi.org/10.1016/j.eswa.2025.127621>

关键方法：边缘引导的非局部建模与频率感知 Transformer。

与本项目关系：更适合作为后续边界/细节模块的理论参考；完整非局部与 Transformer 结构可能破坏当前轻量化预算。截至核实日期未找到明确的作者官方代码。

## 5. 基础必读：2023 轻量烟雾网络

论文：Feiniu Yuan et al., **A Lightweight Network for Smoke Semantic Segmentation**, Pattern Recognition, Volume 137, 2023, Article 109289.  
DOI：<https://doi.org/10.1016/j.patcog.2022.109289>

注意：搜索结果常把 <https://github.com/rekon/Smoke-semantic-segmentation> 与该论文关联，但该仓库实际是 2018–2021 年的 Kaggle U-Net/LinkNet 部分实现，早于论文，且 README 明确说明是通用 U-Net 烟雾示例，因此不能标为该论文的官方复现代码。

## 6. 当前阅读顺序

如果只看三篇，按以下顺序：

1. PSL：重点解决 smoke / fire / background 可分性。
2. TANet：重点看方向长核和纹理表示。
3. SmokeNet：重点看轻量矩形多尺度卷积与逐层监督。

当前不建议先完整复现这些网络。更高效的路线是从作者代码中核对损失或模块细节，再以小型消融形式接入 PIDNet-S。

