# FLAME3 AMP与Batch工程测试修订记录

日期：2026-07-31  
状态：在任何FLAME3精度训练前修订，旧工程JSON保留

## 原测试

- 配置：AMP初始scale 1024；batch 4/8；每个候选重复同一个确定性混合batch；
- batch 4：首步loss有限，但反向梯度出现NaN；
- batch 8：22步通过，峰值allocated显存约4.95%，152.17 samples/s；
- 旧输出：`audit/flame3_4090_batch`。

## 发现的问题

重复同一batch不能代表正式训练的随机增强与样本组成。batch 8在固定batch上通过，不能排除后续随机batch再次触发AMP溢出，因此旧冻结结果不用于启动精度训练。

## 修订

1. AMP初始scale由1024降为128，后续仍由GradScaler自动调整；
2. batch 4和8分别在独立进程中运行2步预热+20步测量；
3. 每一步从完整train split中读取新的随机打乱、随机缩放、裁剪和翻转batch，不再重复同一张量；
4. 必须累计覆盖Fire与No Fire图片；
5. 最终输出改为`audit/flame3_4090_batch_final`；
6. batch选择规则仍为：batch 8稳定且allocated显存≤80%则选8，否则检查4；不使用精度或test。

该修订发生在30轮训练启动前，没有产生任何FLAME3训练精度结果。
