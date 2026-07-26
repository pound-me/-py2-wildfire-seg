# Formal 100-epoch route decision

Date: 2026-07-26

All numbers below use the adopted A1 protocol, seed 200, the fixed validation
set, and the mIoU-selected best checkpoint. The test set remains untouched.

## Baseline

- Best epoch: 83
- mIoU: 0.6136705456
- Fire IoU: 0.2172295512
- Fire precision / recall: 0.3167596710 / 0.4087542566
- Fire boundary F1@3 px: 0.3685805170
- Deployment parameters: 7,623,651
- FLOPs at 256x256: 2.946576 GFLOPs

## P1: one EMA prototype per class

- Best epoch: 40
- mIoU: 0.6070303828 (baseline -0.0066401627)
- Fire IoU: 0.2181775577 (baseline +0.0009480065)
- Internal mIoU rule: fail
- Internal Fire rule: fail
- Deployment parameters/FLOPs: identical to baseline
- Feature evidence: Smoke-Fire class-mean cosine similarity 0.8902794123;
  cosine silhouette score 0.0955862403
- Decision: preserve as a complete 100-epoch negative ablation; do not promote
  directly to three seeds.

## D2: DEConv in P-branch layer3_ and layer4_

- Best epoch: 69
- mIoU: 0.6166547253 (baseline +0.0029841797)
- Fire IoU: 0.2192201883 (baseline +0.0019906371)
- Internal mIoU rule: fail
- Internal Fire rule: fail
- FP32 reparameterization maximum error: 0.0
- Full-validation mismatched pixels after fusion: 0 / 245,760
- Deployment parameters/FLOPs: identical to baseline
- Paired RTX 2060 latency change after fusion: -5.4055% (no speed loss)
- Decision: engineering-valid and slightly positive, but below the predefined
  internal method threshold; do not claim the method as established.

## Combination D2+P1

- Epoch-26--30 mean mIoU: 0.5542379345
- Epoch-26--30 mean Fire IoU: 0.2038516514
- It passes the baseline screening threshold, but is worse than D2 on both
  metrics and loses 0.0214621692 mIoU versus P1.
- Decision: retain only as a combination ablation.

## Backup-route activation

The user approved a formal rule amendment on 2026-07-26: both primary routes
failing either their 30-epoch screening gate or their promoted winner's
100-epoch internal method gate triggers the backup route. The full original
wording, revised wording, rationale, evidence and approval are recorded in
`docs/EXPERIMENT_RULE_AMENDMENTS.md`.

The revised trigger is satisfied because P1 and D2 are the promoted route
winners and both fail the 100-epoch internal method threshold. DySample is
activated before FreqFusion because it has the smaller expected deployment
overhead.

## P-route failure attribution before DySample

P1 completed all 100 epochs without dead prototypes, invalid health records,
data-decoding errors or training-pipeline failures. Its failure is therefore
not explained by instability or a broken label/component pipeline.

The best checkpoint shows a small Fire-recall increase (+0.0142162473), but
lower Fire precision (-0.0061226950), lower Fire boundary F1
(-0.0159234165), and lower overall mIoU (-0.0066401627). On the fixed
validation feature subset, Smoke and Fire class means retain cosine similarity
0.8902794123 and the cosine silhouette score is only 0.0955862403. The
single-prototype objective did not produce strong Smoke/Fire semantic
separation and did not improve the Fire boundary.

DySample directly targets learned upsampling, boundary placement and small
detail recovery, so it is technically relevant to the observed boundary loss.
However, it does not directly solve the high Smoke/Fire semantic similarity.
The expected benefit should therefore be treated as moderate rather than
assumed: a positive result would likely come from better spatial reconstruction
instead of the class-separation mechanism attempted by P1.
