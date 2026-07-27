# 终局三 Seed 执行状态

日期：2026-07-27

## 当前状态

- 统计预注册已在 commit `0b5b1c5b8b02efa2154eacbfb81c9fc11886bb6f` 单独落库并推送，早于任何新增训练；
- Fusion seed 200 的100轮历史结果已审计可用；
- ABL+SAMF seed 200 历史结果只有30轮，终局阶段将另起目录 fresh 跑100轮；
- 五个新增100轮配置、六行运行manifest、配置验收脚本和终局聚合脚本已准备；
- `experiments/terminal_eval/preregistration_config_check.json` 验证通过：seeds、轮数、冻结字段、fresh初始化、目标目录和测试集锁均正确；
- 测试集未使用。

## 冻结执行顺序

1. `terminal_eval_fusion_100e_seed201`；
2. `terminal_eval_fusion_100e_seed202`；
3. `terminal_eval_abl_samf_100e_seed200`；
4. `terminal_eval_abl_samf_100e_seed201`；
5. `terminal_eval_abl_samf_100e_seed202`。

先完成两个新Fusion seed，以便尽早冻结 mIoU 与 Fire 的基线噪声地板。每个运行完成后独立验证、归档、commit/push；不并行争用 RTX 2060，不覆盖旧目录，不依据中间结果改变后续seed或配置。

六个验证集best与正式checkpoint成对延迟全部冻结后，运行 `src/aggregate_terminal_three_seed.py`。该脚本拒绝30轮结果、非200/201/202 seed、不同训练关键配置、不同数据列表或不同预训练权重，并严格执行预注册的 `mean gain > pooled sample SD` 规则。
