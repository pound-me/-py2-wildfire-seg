# 终局三 Seed 执行状态

日期：2026-07-27

## 当前状态

- 统计预注册已在 commit `0b5b1c5b8b02efa2154eacbfb81c9fc11886bb6f` 单独落库并推送，早于任何新增训练；
- 五个新增100轮训练均已完成、审计并逐运行归档，六个验证集 best checkpoint 已冻结；
- mIoU 与 Fire IoU 均严格满足 `mean gain > pooled sample SD`；
- RTX 2060 正式成对测速为 Fusion `46.35 FPS`、ABL+SAMF `43.75 FPS`，候选通过30 FPS门槛；
- ABL+SAMF 已冻结为论文主方法行；
- 验证集裁决完成后，六个唯一 checkpoint 已各在测试集评估一次并完成三 seed 聚合；
- 不再依据测试结果修改方法、超参数、checkpoint或seed。

## 冻结执行顺序

1. `terminal_eval_fusion_100e_seed201`；
2. `terminal_eval_fusion_100e_seed202`；
3. `terminal_eval_abl_samf_100e_seed200`；
4. `terminal_eval_abl_samf_100e_seed201`；
5. `terminal_eval_abl_samf_100e_seed202`。

上述顺序已完整执行。每个运行完成后均独立验证、归档并 commit/push；未并行争用 RTX 2060，未覆盖旧目录，未依据中间结果改变后续seed或配置。

最终机器证据：

- 验证裁决：`experiments/terminal_eval/terminal_three_seed_decision.json`；
- 正式测速：`experiments/terminal_eval/paired_latency_rtx2060_best_seed200.json`；
- 测试汇总：`experiments/terminal_eval/terminal_test_once_summary.json`；
- 人类可读结论：`docs/TERMINAL_THREE_SEED_FINAL_DECISION_2026-07-27.md`。
