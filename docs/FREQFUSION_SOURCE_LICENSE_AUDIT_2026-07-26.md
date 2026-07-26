# FreqFusion官方来源、授权与PIDNet适配审计

日期：2026-07-26

## 1. 官方来源

- 论文：*Frequency-Aware Feature Fusion for Dense Image Prediction*
- 出处：IEEE TPAMI，2024
- 官方仓库：https://github.com/Linwei-Chen/FreqFusion
- 本地checkout：`third_party/FreqFusion`
- 固定commit：`3fb0c70637a3c194fb74294d3ce4681958b26241`
- commit日期：2025-11-25
- 核心文件：`FreqFusion.py`
- checkout状态：下载后干净，未进行本地修改

README提到的 `ying-fu/FreqFusion` 是上述仓库的fork，不是具有独立授权条款的
替代来源。

## 2. LICENSE核查结果

官方仓库根目录没有 `LICENSE`、`COPYING` 或其他核心代码授权文件。Git历史中也
没有发现仓库级LICENSE。GitHub license API对主仓库返回404，仓库元数据的
`license`字段为`null`；README提到的fork同样为`null`。

仓库内的 `Mask2Former/LICENSE`、`mmdetection/LICENSE` 和`SegNeXt/LICENSE`
只对应各自上游子项目，不能推定根目录 `FreqFusion.py` 采用相同许可证。

因此当前不能把核心代码标记为MIT、Apache或其他开源许可证，也不能在父项目中
复制或重新分发。checkout已加入父项目 `.gitignore`，只在本机作为来源审计材料，
不会进入 `origin/main`。

## 3. PIDNet-S结构适配结论

官方核心实现的高分辨率输入 `hr_feat` 必须是低分辨率输入 `lr_feat` 的2倍，内部
CARAFE路径固定按2倍处理。PIDNet-S候选位置如下：

- Pag3：P分支细节特征为64×32×32，压缩后的I分支语义特征为64×16×16，通道相同、
  分辨率比为2，天然符合官方接口。
- Pag4：细节为64×32×32，语义为64×8×8，是4倍，不符合单个官方模块接口。
- SPP到DFM：128×4×4到128×32×32，是8倍，不符合单个官方模块接口。

所以若获准继续，首个且唯一的单点候选应为Pag3。把Pag4或SPP路径拆成多级模块会
改变官方用法、增加混杂因素和计算开销，不作为本轮方案。

## 4. 官方配置与轻量化边界

论文提供的正式配置通常启用 `feature_resample=True`，即同时使用ALPF、AHPF和
offset重采样；仅使用默认的 `feature_resample=False` 会删去方法的重要组成部分，
不能透明地称为完整FreqFusion。

PIDNet适配若获准，应先按官方完整机制进行工程测量，并使用官方集成中常见的通道
压缩思路降低 `compressed_channels`。任何压缩值必须在配置和论文中明确，不冒充
原始默认64通道。若参数、FLOPs或RTX 2060速度明显违反轻量化总原则，应在训练前
停止该路线，而不是为了通过预算偷偷删减方法组件。

## 5. 当前决策状态

- 官方代码已取得并固定commit。
- 来源、接口和结构位置审计完成。
- LICENSE缺失是当前未决事项。
- 尚未复制、导入或修改FreqFusion核心代码。
- 尚未新增模型、配置、checkpoint或训练实验。
- 测试集未使用。

继续实施前需要用户明确决定：是否接受在记录“上游未声明LICENSE、仅作内部学术
实验、官方源码不进入父仓库”的前提下使用该checkout；否则应暂停并向作者申请明确
许可，或终止FreqFusion路线。
