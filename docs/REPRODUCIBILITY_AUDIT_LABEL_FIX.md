# Reproducibility audit after the strict FLAME2 label fix

Audit date: 2026-07-26

Compared runs:

- `experiments/pidnet_s_protocol/protocol_a1_30e_label_fix`
- `experiments/pidnet_s_rgb_baseline_adopted/baseline_a1_100e_label_fix`

The two runs use seed 200, RTX 2060, brightness disabled, scale 0.5--1.5,
batch size 4, and a 100-epoch polynomial learning-rate horizon. Their recorded
`source_sha256` values are identical:

`de886001af762d76823e526ea705863fcf58da7f7fd353df94f4ed97b4dcbfc2`

Resolved-config differences are limited to `EPOCHS` (30 versus 100), naming
and experiment-output fields, and adopted-protocol provenance metadata. No
model, optimizer, augmentation, dataset-list, seed, or LR-horizon setting
differs.

Nevertheless, the validation trajectories diverge from epoch 1. The A1
screening run has epoch-26--30 mean mIoU `0.5810447343` and Fire IoU
`0.2079233164`; the adopted 100-epoch run has mean mIoU `0.5459809667` and
Fire IoU `0.1978396030` over the same window. Exact rerun identity is therefore
not supported by current evidence. CUDA backward operators in this network,
including bilinear interpolation paths, can remain numerically non-identical
despite fixed seeds, cuDNN deterministic mode and benchmark=False; the audit
does not claim a single unproven operator as the definitive cause.

Decision required by the final experiment plan:

1. keep the selected A1 protocol unchanged;
2. use only the adopted 100-epoch baseline as the reference for D1/D2 and
   P1--P4;
3. call max-minus-min the conservative Fire noise band, not sigma;
4. treat 30-epoch single-seed results as screening evidence only;
5. base paper claims on the required three-seed mean plus standard deviation.

The adopted baseline epoch-26--30 reference is saved in
`reference_epoch26_30.json` in its run directory.
