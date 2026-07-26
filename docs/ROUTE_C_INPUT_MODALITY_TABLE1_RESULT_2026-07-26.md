# Route C节点1c：输入模态Table 1正式结果

日期：2026-07-26
范围：验证集；测试集未使用。

## 1. IR-only 100轮身份

- 配置：`configs/route_c/pidnet_s_ir_100e_label_fix.yaml`；
- 运行：`route_c_pidnet_s_ir/route_c_ir_100e_label_fix_seed200`；
- seed 200，100轮，LR horizon 100，AMP，RTX 2060；
- MODE=ir，单通道；label_fix A1协议；
- 从PIDNet-S ImageNet权重重新开始，shape-match-only加载301个张量；
- 训练耗时`3241.43 s`，峰值分配显存`224.53 MB`；
- best checkpoint按验证集mIoU保存于epoch 82；stderr为空；测试集未使用。

## 2. Table 1

三行均使用各自100轮训练的mIoU-selected best checkpoint。参数/FLOPs采用同一256×256
profiler；FPS采用RTX 2060同进程、100次预热、每trial 200次前向、12组轮换顺序的AMP
中位数。

| Input | Background IoU | Smoke IoU | Fire IoU | mIoU | Params (M) | GFLOPs | FPS |
|---|---:|---:|---:|---:|---:|---:|---:|
| RGB-only PIDNet-S | 0.867944 | 0.755838 | 0.217230 | 0.613671 | 7.624 | 2.9466 | 49.26 |
| IR-only PIDNet-S | 0.814379 | 0.712699 | 0.673819 | 0.733632 | 7.623 | 2.9277 | 48.61 |
| RGB+IR Fusion PIDNet-S | 0.908876 | 0.841998 | 0.669310 | 0.806728 | 7.624 | 2.9560 | 49.56 |

三种输入均超过30 FPS实时硬门槛。FPS差异在约1 FPS以内，不能解释精度差异。

## 3. 动机结论

IR相对RGB：

- mIoU `+0.119962`；
- Fire IoU `+0.456590`；
- Smoke IoU `-0.043140`。

说明热红外是Fire识别的决定性模态，但单独IR对Smoke与Background的表征不如RGB。

Fusion相对IR：

- mIoU `+0.073096`；
- Smoke IoU `+0.129299`；
- Fire IoU `-0.004509`。

直接拼接Fusion显著恢复Smoke/Background并取得最高总体mIoU，但Fire相对纯IR略降。这与
前两项诊断一致：模型需要在Smoke遮挡场景有选择地保留/注入热信息，而不是简单扩大通用
融合模块。它构成SAMF“烟概率门控热注入”的直接动机，但不提前证明SAMF有效。

Fusion相对RGB的mIoU提升为`+0.193057`、Fire提升为`+0.452081`，确认RGB+IR路线变更
具有强证据基础。

## 4. 速度协议结果

三模型在同一进程中按三种循环顺序测量，每个模型得到12个原始trial：

- RGB：median `20.3006 ms`，P95 `22.2660 ms`，`49.2596 FPS`；
- IR：median `20.5733 ms`，P95 `22.1308 ms`，`48.6066 FPS`；
- Fusion：median `20.1783 ms`，P95 `20.8317 ms`，`49.5582 FPS`。

结果只覆盖batch-1模型前向，不包含磁盘读取和后处理；三者采用完全相同的测量范围。

## 5. 证据文件

- IR训练：同运行目录`environment.json`、`resolved_config.json`、`metrics.jsonl`、
  `run_summary.json`及stdout/stderr日志；
- IR验证：同目录`val_best/metrics.json`、`per_image_metrics.csv`、`complexity.json`；
- 同会话测速：`experiments/route_c_input_motivation/table1_paired_latency_rtx2060.json`；
- Table 1：同目录`table1_input_modalities.json`与`table1_input_modalities.csv`；
- `best.pth`、`last.pth`、预测图和比较图不提交Git。
