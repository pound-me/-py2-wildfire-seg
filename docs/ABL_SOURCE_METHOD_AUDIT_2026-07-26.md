# Active Boundary Loss来源、方法与Route B适配审计

日期：2026-07-26

## 1. 选择依据

Fusion正式基线的固定验证集错误画像表明：全部误分类中78.4324%位于真实语义边界
3像素带内；Smoke和Fire边界recall分别为0.747883和0.835477；Fire错误中63.32%
被判为Smoke。B路线因此只接受直接监督主分割logits、改善语义边界位置或召回、且推理
零开销的方法。

ABL满足上述诊断约束。CBL未被选择：其作者实现与OCRHead/MaskFormer接口耦合，且
审计时没有发现仓库级LICENSE。B路线只允许一个候选，不能把二者都跑一遍再挑。

## 2. 官方来源

- 论文：*Active Boundary Loss for Semantic Segmentation*；
- 出处：AAAI 2022 Oral；
- 作者官方仓库：https://github.com/wangchi95/active-boundary-loss；
- 固定commit：`1511507533ad98f04ea26e3648360a6c1d477d37`；
- 本地checkout：`third_party/active-boundary-loss`，父仓库忽略，不重复分发；
- LICENSE：Apache-2.0；
- `abl.py` SHA256：`07fc49d923a2420db7316eeb1650e95b2ea415bc85233937df99b239e3c6fe87`；
- LICENSE SHA256：`1eb85fc97224598dad1852b5d6483bbcf0aa8608790dcc657a5a2a761ae9c8c6`。

官方`abl.py`链接到CoinCheung/pytorch-loss commit
`af876e43218694dc8599cc4711d9a5c5e043b1b2`中的LSSCE-V1；核对文件SHA256为
`0367fcad3d1b62283db0a4bb40fe7b5e08095f4ad17db4e6b96b323162ac2251`。

## 3. 官方源码不能原样运行的兼容点

固定官方源码包含以下环境绑定问题：

1. 中间张量硬编码`.cuda()`，不能安全跟随输入device；
2. 使用已从NumPy 1.24移除的`np.bool`；
3. checkout中没有随仓库保存其相对导入的`label_smooth.py`；
4. AMP关键相似度与损失没有统一的显式FP32作用域；
5. 论文表述为每张输入图最多1%的预测边界种子像素，而源码对整个batch共用一个
   `H×W×1%`上限。

因此本项目不声称“未经修改地直接运行官方文件”，而声明为：基于作者Apache-2.0官方
实现的device-safe PIDNet适配。工程验收保留`source_batch`模式，与固定官方算法在同一
输入上的loss和gradient进行数值比对；正式实验使用符合论文逐图表述的`per_image`模式。

## 4. 固定方法与插入位置

ABL只作用于PIDNet训练输出中的主语义logits `outputs[1]`。原PIDNet总损失已经对两个
语义输出使用Lovasz-Softmax，因此不额外叠加另一种IoU/区域损失。D分支原有边界损失
继续保持不变。

固定设置如下，不做超参数搜索：

- ABL权重1.0；
- 预测边界种子像素上限：每图1%，随后按官方方法做3×3膨胀；
- 邻域概率detach；
- LSSCE-V1标签平滑0.2，保留官方源文件的目标权重行为；
- 距离图保留官方`one_hot2dist`整数缓冲导致的欧氏距离截断行为；
- 最大距离权重20；
- ABL相似度、距离权重和损失在AMP下强制FP32。

权重1.0采用论文ADE20K设置；本轮不在1.0/1.5间挑选，避免把唯一候选变成隐性
多配置筛选。

## 5. 轻量化与推理边界

ABL无可学习参数、无buffer，只存在于训练criterion。模型仍为原四通道Fusion
PIDNet-S，推理时只返回原分割tensor：

- 部署参数量增量：0；
- 部署FLOPs增量：0；
- 部署延迟增量：0；
- checkpoint不保存ABL原型、状态或训练专用输出。

训练时间会因CPU距离变换和边界方向损失增加，必须实测记录，但不属于部署开销。

## 6. 筛选纪律

- 唯一实验：`route_a_fusion_abl_30e_label_fix_seed200`；
- 30轮、seed 200、100轮学习率horizon、RTX 2060、AMP；
- 第26--30轮对比Fusion基线mIoU `0.7861409556`、Fire `0.6467101738`、
  保守噪声带 `0.0142509431`；
- 测试集继续封存；
- 若结果不是“近失且原因明确”，B路线直接关闭，不自动换CBL或增加第二候选。

本文件记录的是可复现适配与候选选择依据，不把适配代码冒充为作者仓库的逐字复刻。

## 7. 工程验收结果

`src/check_active_boundary_pipeline.py`已在RTX 2060上通过，证据保存于
`experiments/route_a_pidnet_s_fusion_abl/engineering_check.json`：

- `source_batch`固定输入下，官方源码与适配版loss误差0、logits梯度最大误差0、
  预测边界掩码逐像素一致；
- `per_image`固定输入的两个种子像素数为8和5，均不超过32×32×1%=10.24，并且重复
  计算完全一致；
- 只反传ABL时，主分割头、DFM及P/I/D三分支梯度范数均大于0，两个辅助输出头均为0；
- 完整总损失仍保留D分支边界头梯度；
- ABL推理参数增量为0，推理只返回原单tensor；
- AMP优化器检查中，ABL与原Fusion基线在同一批次都于动态scale降至2048后首次成功
  更新，ABL没有额外增加scale退避次数；
- 模型checkpoint恢复最大误差0，optimizer和GradScaler状态均成功恢复，criterion无额外
  持久化状态。
