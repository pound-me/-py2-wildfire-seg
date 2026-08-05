# FLAME3 MRFF 与条件 NTS 预注册（2026-08-06）

状态：用户批准，MRFF 第一阶段立即执行；测试集继续封存。

## 1. 固定结构

- 模型名：`pidnet_s_mrff`；输入固定为 RGB+IR 四通道。
- RGB Stem：`3→16→32`，IR Stem：`1→16→32`；每层均为
  `3×3 stride=2 + BN + ReLU`。
- Gate 输入为 `[F_rgb,F_ir,|F_rgb-F_ir|]`，使用
  `1×1 96→16 + BN + ReLU + 1×1 16→2 + Softmax`。
- Gate 最后一层 weight/bias 全零，初始权重严格为 `0.5/0.5`。
- 融合特征保持32通道、1/4分辨率，直接进入原PIDNet-S `layer1`。
- 默认接口为 `model(images)`；显式诊断接口为
  `model(images, return_aux=True)`，只返回
  `aux["modality_weights"]=[B,2,H/4,W/4]`，不缓存 `last_gate`。
- RGB/IR Stem统一Kaiming随机初始化；ImageNet checkpoint中与新模型共享的
  288个post-stem PIDNet主体张量全部加载。历史Fusion的301个匹配张量包含
  13个原单Stem张量，因此不能机械要求MRFF仍匹配301。

## 2. NTS固定定义

第一阶段MRFF关闭NTS。只有MRFF正向或边缘正向且出现预注册误报现象时，才从
相同初始化重新训练MRFF+NTS。

NTS仅选择No Fire图像中主头实际预测为Fire的像素：

`mean(stopgrad(p_fire) * W_thermal)`。

- `p_fire`与预测类别来自主分割logits；`p_fire.detach()`；
- thermal门控图双线性对齐，`align_corners=False`；
- 无选中像素时返回与门控计算图连接的零；
- 梯度允许进入Gate与双Stem，不进入主分割logits；
- epoch 1权重0，epoch 2按batch线性升至0.02，epoch 3起保持0.02。

## 3. 固定训练与筛选

- split v2、640×512、物理batch 8、AMP、RTX 4090；
- seeds 200/201/202；LR polynomial总周期100轮；
- 与Fusion保持相同增强、部分标签损失、边界损失和数据路径；
- 统一比较epoch 26–30 validation Fire IoU算术均值；
- 三seed平均提升至少0.005且至少2/3 seed提高：通过；
- 平均提升0至0.005：三个seed从last精确恢复至50轮，再比较46–50；
- 平均提升不大于0或只有一个seed提高：停止；
- RTX 2060固定256×256、batch 1、AMP、同会话交替顺序测速，候选必须至少30 FPS。

测试集在结构、损失、超参数和checkpoint选择全部冻结前不得读取。

## 4. 工程准入证据

本机RTX 2060检查结果：

- FP32/AMP固定平均融合误差均为0；权重和误差为0；
- 第一次反传：Gate末层、RGB Stem、IR Stem、PIDNet主体梯度非零，Gate前层为0；
- 一次optimizer step后第二次反传：Gate前后层、双Stem和主体梯度均非零；
- NTS-only反传进入Gate与双Stem，PIDNet主体和合成分割logits无梯度；
- checkpoint恢复后的分割输出及aux逐位一致；
- AMP loss和全部有效梯度有限；默认推理只返回分割Tensor；
- 训练参数7,718,121；未融合部署参数7,624,965；相对Fusion仅增加1,026个训练参数；
- 256×256未融合MRFF约2.9501 GFLOPs，Fusion约2.9560 GFLOPs；
- 最终门控精度路径的正式architecture-only成对测速：Fusion 41.51 FPS，MRFF
  39.67 FPS，均通过30 FPS。

## 5. 部署Stem融合的否决记录

曾审计仅部署使用的等价融合：第一层采用`4→32`块稀疏普通卷积（前16输出
只读RGB，后16输出只读IR），第二层采用`32→64, groups=2`；Conv+BN只用eval
running statistics折叠，两层之间的ReLU均保留。

FP32真实样本上，Stem/门控/融合特征误差小于`1e-5`且argmax一致；但AMP真实
样本上RGB Stem与融合特征最大误差达到`0.0078125`，最终logits误差达到
`0.140625`并出现临界像素类别变化。因此该部署转换不满足AMP等价要求，相关
部署实现已从正式代码路径撤回；正式训练与部署均继续使用同一原双Stem实现。
