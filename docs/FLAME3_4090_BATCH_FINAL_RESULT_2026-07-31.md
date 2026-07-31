# FLAME3 RTX 4090 物理 Batch 最终冻结结果

日期：2026-07-31  
状态：在任何 FLAME3 精度训练前冻结  
对应源码：`2af80a2c79a8504de8175325aab4af954c5b3523`

## 最终工程协议

- GPU：NVIDIA GeForce RTX 4090 24GB；
- 输入：RGB+IR 四通道，`640×512`；
- AMP 初始 scale：`128`；
- 候选物理 batch：`4`、`8`；
- 每个候选使用2步预热和20步测量，共22个不同的随机打乱、随机缩放、裁剪和翻转 batch；
- 必须累计覆盖 Fire 与 No Fire 图片；
- 不计算验证精度，不读取测试图片或标签。

## 结果

| Batch | 状态 | 样本吞吐 | 峰值 allocated | 峰值 reserved | Fire/No Fire累计覆盖 |
|---:|---|---:|---:|---:|---:|
| 4 | passed | 82.868 samples/s | 2.632% | 3.281% | 79 / 9 |
| 8 | passed | 122.861 samples/s | 4.965% | 5.691% | 148 / 28 |

两组均完成全部步骤，loss与梯度有限，无OOM。依据预注册规则，batch 8通过且峰值 allocated 显存不超过80%，因此在任何FLAME3精度结果产生前冻结：

```text
physical_batch = 8
```

Fusion基线及后续所有正式比较模型均使用相同物理batch 8；若未来结构无法运行该batch，必须在产生其精度结果前登记协议修订。

审计文件：`audit/flame3_4090_batch_final/flame3_4090_batch_preregistered.json`。
