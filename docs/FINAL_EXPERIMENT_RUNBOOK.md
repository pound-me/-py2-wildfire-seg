# PIDNet-S final experiment runbook

All commands use:

```powershell
$py = "F:\anaconda3\envs\pytorch\python.exe"
Set-Location "G:\py2"
```

Do not evaluate the test set until the structure and hyperparameters are
frozen. The engineering-check runs under `experiments/engineering_checks` and
the one-batch pipeline runs are not scientific results.

Current hardware rule: run every scientific experiment **serially** on
`cuda:0`, the RTX 2060. Do not use the RTX 4060 or RTX 5070 while they are
allocated to other users. Results produced before the strict FLAME2 label
decoder fix are archive-only and must not enter the final paper tables.

## 1. Augmentation protocol screening

Run A1 and A2 for 30 epochs, serially on the RTX 2060:

```powershell
& $py src\train_baseline.py --config configs\pidnet_s_protocol_a1.yaml --run-name protocol_a1_30e --device cuda:0 --amp
& $py src\train_baseline.py --config configs\pidnet_s_protocol_a2.yaml --run-name protocol_a2_30e --device cuda:0 --amp
```

Screen against the existing 100-epoch baseline window:

```powershell
& $py src\screen_experiment.py `
  --baseline experiments\pidnet_s_rgb_baseline\baseline_100e_fair\metrics.jsonl `
  --candidate experiments\pidnet_s_protocol\protocol_a1_30e\metrics.jsonl `
  --candidate experiments\pidnet_s_protocol\protocol_a2_30e\metrics.jsonl `
  --output experiments\pidnet_s_protocol\a1_a2_screening.json
```

Only if A1 and A2 both pass, run A3 and screen all three candidates.

After choosing one protocol, materialize a new 100-epoch baseline and all
candidate configs with exactly the same augmentation settings:

```powershell
& $py src\materialize_adopted_protocol.py `
  --protocol configs\pidnet_s_protocol_a1.yaml `
  --output-dir configs\adopted_protocol_label_fix
```

Replace A1 with A2 or A3 if that protocol wins. Then run the generated
`pidnet_s_rgb_baseline_adopted_100e.yaml` first. Recompute its epoch 26-30
conservative Fire noise band before screening any module. Archive the old
baseline and do not mix it into the final main table.

## 2. DEConv screening

Use the generated strict-label adopted-protocol configs, serially:

```powershell
& $py src\train_baseline.py --config configs\adopted_protocol_label_fix\pidnet_s_deconv_d1.yaml --run-name deconv_d1_30e_label_fix --device cuda:0 --amp
& $py src\train_baseline.py --config configs\adopted_protocol_label_fix\pidnet_s_deconv_d2.yaml --run-name deconv_d2_30e_label_fix --device cuda:0 --amp
```

Use `screen_experiment.py` with the adopted baseline as comparison. For a
passing checkpoint, run the full deployment equivalence check:

```powershell
& $py src\check_deconv_reparameterization.py `
  --config configs\adopted_protocol_label_fix\pidnet_s_deconv_d1.yaml `
  --checkpoint experiments\pidnet_s_deconv\deconv_d1_30e_label_fix\best.pth `
  --output experiments\pidnet_s_deconv\deconv_d1_30e_label_fix\reparameterization.json
```

## 3. EMA multi-prototype ablations

Run P1, then P2, then P3, then P4. Do not skip the order because each row
isolates one factor:

```powershell
& $py src\train_baseline.py --config configs\adopted_protocol_label_fix\pidnet_s_dfm_mproto_p1.yaml --run-name p1_30e_label_fix --device cuda:0 --amp
& $py src\train_baseline.py --config configs\adopted_protocol_label_fix\pidnet_s_dfm_mproto_p2.yaml --run-name p2_30e_label_fix --device cuda:0 --amp
& $py src\train_baseline.py --config configs\adopted_protocol_label_fix\pidnet_s_dfm_mproto_p3.yaml --run-name p3_30e_label_fix --device cuda:0 --amp
& $py src\train_baseline.py --config configs\adopted_protocol_label_fix\pidnet_s_dfm_mproto_p4.yaml --run-name p4_30e_label_fix --device cuda:0 --amp
```

Run them strictly in P1, P2, P3, P4 order. A model with persistent
dead/collapsed prototypes is automatically marked invalid in `metrics.jsonl`
and cannot pass screening even if its mIoU is high.

## 4. Combination and 100-epoch runs

Put the winning DEConv variant and winning P1-P4 settings into the generated
combination config, then run 30 epochs. If the combination passes, run 100
epochs. If it degrades, keep the better individual method and report the
combination only as an ablation.

Use three seeds for every local method/major ablation row that enters the main
table:

```powershell
& $py src\train_baseline.py --config <final-config.yaml> --epochs 100 --seed 200 --run-name final_seed200 --device cuda:0 --amp
& $py src\train_baseline.py --config <final-config.yaml> --epochs 100 --seed 3407 --run-name final_seed3407 --device cuda:0 --amp
& $py src\train_baseline.py --config <final-config.yaml> --epochs 100 --seed 12007 --run-name final_seed12007 --device cuda:0 --amp
```

Run the three seeds one after another. Do not combine heterogeneous GPUs in
one DDP run, and do not occupy the RTX 4060/5070 without renewed permission.

After evaluating the three validation runs, aggregate them without any new
checkpoint selection:

```powershell
& $py src\aggregate_seed_evaluations.py `
  --evaluation <seed200-val>\metrics.json `
  --evaluation <seed3407-val>\metrics.json `
  --evaluation <seed12007-val>\metrics.json `
  --output <method-dir>\validation_3seed_summary.json
```

## 5. Best-checkpoint paper materials

Complexity (DEConv is fused automatically before counting):

```powershell
& $py src\measure_complexity.py --config <final-config.yaml> --checkpoint <best.pth> --output <run-dir>\complexity.json
```

Validation metrics, Fire precision/recall, boundary F1, prediction files, and
RTX 2060 speed:

```powershell
& $py src\evaluate_baseline.py --config <final-config.yaml> --checkpoint <best.pth> --split val --amp --output-dir <run-dir>\val_best
```

Fused-feature evidence for an mproto model:

```powershell
& $py src\analyze_fused_features.py --config <final-config.yaml> --checkpoint <best.pth> --output-dir <run-dir>\fused_feature_analysis
```

The t-SNE plot is qualitative only. Use IoU, Fire precision/recall, boundary
F1, cosine distances, silhouette score, and prototype allocation statistics as
the quantitative evidence.

## 6. Final test protocol

After architecture, augmentation, loss weights, epoch selection, and all
hyperparameters are frozen, evaluate each final seed on the test set exactly
once. Report mean and standard deviation across the three seeds. Do not modify
the method based on test results.

The evaluator enforces an explicit confirmation for test runs:

```powershell
& $py src\evaluate_baseline.py --config <frozen-config.yaml> `
  --checkpoint <final-seed-best.pth> --split test --confirm-frozen-test `
  --amp --output-dir <final-seed-dir>\test_once
```
