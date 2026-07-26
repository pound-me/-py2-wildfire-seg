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

因此顶层 `FreqFusion.py` 不能标记为MIT、Apache或其他开源许可证，也不能在父项目
中复制或重新分发。checkout已加入父项目 `.gitignore`，不会进入 `origin/main`。

进一步核查发现，官方仓库包含三个带目录级许可证的框架集成版本：

- `SegNeXt/`：Apache-2.0；
- `mmdetection/`：Apache-2.0；
- `Mask2Former/`：MIT。

本项目是语义分割任务，因此切换到
`SegNeXt/mmseg/models/decode_heads/FreqFusion.py`。该文件位于Apache-2.0覆盖的
SegNeXt子项目中，SHA256为
`14a5af3a614a721cc01777102974e0cbfb4f29a19a45df8b883be7b0d1f38d91`。
顶层无许可证版本不作为实现来源。

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

## 5. 用户授权与实现边界

用户于2026-07-26明确授权：在“仅内部学术实验、官方源码不提交主仓库”的条件下
使用FreqFusion。结合上述带许可证框架版本，后续实现固定为：

- 从被忽略的官方checkout动态加载SegNeXt Apache-2.0集成版；
- 不把官方 `FreqFusion.py` 复制进父项目；
- 保留官方仓库、commit、源文件SHA256和许可证SHA256记录；
- 本地适配代码只负责PIDNet接口、来源校验及环境兼容；
- 不把顶层无许可证fallback当作代码来源。

SegNeXt集成版依赖 `mmcv.ops.carafe`，而当前PyTorch 2.8/CUDA 12.6环境没有兼容
MMCV扩展。适配时将使用按CARAFE公开公式独立编写的纯PyTorch算子，并限定本实验
`up_group=1`、2倍上采样接口；不得复制顶层无许可证fallback代码。工程检查必须验证
张量形状、梯度和数值有限性。

## 6. 投稿前代码发布预案

如果FreqFusion通过筛选并进入最终方法，投稿前必须解决代码发布授权与复现交付：

1. 优先邮件联系FreqFusion作者，取得对核心实现及本项目适配发布的明确许可，并保存
   邮件或公开回复作为授权记录；
2. 若未取得许可，依据论文公式独立重实现所需组件，并进行等价性与来源隔离审计；
3. 若投稿时间不允许完成独立重实现，则主仓库只发布固定commit拉取脚本、SHA256校验、
   本地适配层和安装说明，不重新分发官方源码；
4. 论文、README和代码文档必须规范引用FreqFusion论文及官方仓库，清楚区分官方方法、
   带许可证框架集成来源、本项目PIDNet适配和独立CARAFE兼容实现；
5. 在上述事项完成前，不宣称发布了可自由再分发的完整FreqFusion源码。

## 7. 当前决策状态

- 官方代码已取得并固定commit。
- 来源、接口和结构位置审计完成。
- 顶层LICENSE缺失已通过切换到SegNeXt Apache-2.0集成版规避；顶层文件不使用。
- 用户已批准仅内部学术实验，官方源码保持在被忽略checkout中。
- 未复制或修改FreqFusion官方核心代码；父仓库只保存PIDNet适配层和独立CARAFE兼容实现。
- Pag3单点适配、配置、来源校验、AMP梯度检查和复杂度测量已完成。
- RTX 2060成对测速中，FP32中位延迟增加48.907%，AMP中位延迟增加21.699%，均超过3%上限。
- 部署参数由7,623,939增至7,639,733，FLOPs由2.9560G增至2.9851G，也不满足“不高于基线”的总原则。
- 因工程轻量化门槛失败，不启动30轮筛选，不生成FreqFusion训练checkpoint；该路线归档为工程阴性结果。
- 测试集未使用。

完整工程结果见 `docs/FUSION_FREQFUSION_ENGINEERING_RESULT_2026-07-26.md`。若未来修改
轻量化约束并重新启用该路线，第6节投稿前代码发布预案仍为强制要求。
