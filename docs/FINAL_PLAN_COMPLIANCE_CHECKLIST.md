# PIDNet-S final-plan compliance checklist

Last updated: 2026-07-26

This checklist separates implemented code from scientific results. A passing
engineering check is not treated as evidence that a method improves accuracy.

## Data and protocol

- [x] Runtime training and cache construction share the strict black/gray/white
  FLAME2 decoder.
- [x] Unknown colors and missing labels raise; no all-background fallback.
- [x] 992 labels and uint16 8-connected Fire caches verified.
- [x] Runtime and augmented component-map/Fire alignment assertions exist.
- [x] A1/A2/A3 rerun after the label fix; A1 adopted by the specified rule.
- [x] Adopted A1 configs materialized in `configs/adopted_protocol_label_fix`.
- [x] Adopted baseline epoch-26--30 reference and conservative Fire noise band
  saved.
- [x] Adopted A1 baseline completes all 100 epochs and best checkpoint is
  evaluated on validation.

## DEConv

- [x] Official DEA-Net source pinned at commit
  `599d4d5533325aec0ceca81e661f3eeb2b0619e8`.
- [x] CDC/ADC/HDC/VDC mappings checked against official source.
- [x] Five kernels are summed before one convolution; no branch BN or bias.
- [x] Vanilla weight loads the PIDNet checkpoint and differential weights start
  at zero.
- [x] Formal checker covers initial main and three training outputs, deployment
  FP32 error, validation predictions and parameter equality.
- [x] Rerun the updated D1/D2 engineering checker and save JSON evidence.
- [x] D1 and D2 strict-label 30-epoch runs and screening; D2 selected.

## EMA multi-prototype supervision

- [x] Official PSL source pinned at commit
  `39e6838315ff33222bf23011a793f4b7e72e79f6`.
- [x] DFM 128-channel feature sampling uses full-resolution coordinates and
  `grid_sample(..., align_corners=False)` with pixel-center normalization.
- [x] P1--P4 configurations isolate K, Fire component sampling and
  feature-side decorrelation.
- [x] Epoch-1 EMA-only, epoch-2 linear loss warmup, EMA=0.99, detached
  prototypes and forced-FP32 prototype operations implemented.
- [x] Fire mixed sampling is without replacement and records small/medium/large
  component-area buckets.
- [x] Health records hits, norms, cosine matrices, bucket assignments, dead and
  collapse streaks. Uninitialized zero-hit prototypes cannot evade the
  five-epoch death gate.
- [x] Screening checks prototype-health failures over epochs 1--30, not only
  the final five-epoch metric window.
- [x] Rerun the updated AMP/gradient/checkpoint/inference engineering checker.
- [x] P1, P2, P3 and P4 strict-label 30-epoch runs and screening in order;
  P1 selected among the prototype variants.

## Combination, deployment and paper evidence

- [x] Select passing DEConv and mproto variants and run the D2+P1 combination
  for 30 epochs. The combination passes the baseline screening threshold but
  is dominated by the successful individual variants, so it is retained only
  as an ablation.
- [x] Run P1 and D2 for 100 epochs and verify the internal method threshold.
  Neither method reaches either internal threshold on its mIoU-selected best
  checkpoint; both are preserved as formal negative/near-neutral ablations.
- [ ] Complete required three-seed rows and aggregate mean plus sample standard
  deviation with unique-seed/config checks.
- [x] Measure P1 and fused-D2 deployment parameters/FLOPs and RTX 2060 speed.
  Both have baseline-identical deployment parameters/FLOPs. A same-process,
  alternating-order paired benchmark verifies that fused D2 has no speed loss.
- [ ] Export validation IoU, Fire precision/recall, boundary F1, prototype
  statistics, fixed-sample fused features, cosine distances, silhouette score
  and qualitative t-SNE.
- [x] Test evaluation is code-locked behind `--confirm-frozen-test`.
- [ ] After all choices are frozen, evaluate each final seed on the test set
  once and aggregate without further method changes.

## Backup route

- [x] Formally amend and log the backup trigger so that both primary routes
  failing either 30-epoch screening or their promoted winner's 100-epoch
  internal threshold activates the backup route.
- [x] Confirm that the mproto route winner P1 and the DEConv route winner D2
  both completed 100 epochs and failed the internal method threshold.
- [x] Record the P-route failure attribution before implementing DySample.
- [x] Pin the official DySample repository at commit
  `81a1de5caa95d55a0f5488425fa53ec7ef47f8f0` and record its MIT LICENSE.
- [x] Limit initial DySample screening to two theory-driven single-position
  variants: late Pag4 semantic fusion and SPP-context upsampling before DFM.
- [x] Close both theory-driven DySample positions under the Route A Fusion
  protocol: Pag4 passes engineering but fails its 30-epoch accuracy screen;
  Context passes functional/AMP checks but exceeds baseline deployment
  parameters and FLOPs, so it is correctly stopped before training.
- [x] Audit FreqFusion licensing and switch from the unlicensed repository-root
  file to the official repository's Apache-2.0 SegNeXt integration; keep the
  complete upstream checkout ignored by the parent repository.
- [x] Record the publication contingency: obtain author permission before
  release, otherwise independently reimplement from the paper or publish only
  a pinned-commit fetch-and-verify script, with proper paper/repository citation.
- [x] Complete Pag3 FreqFusion engineering admission. FP32 latency increases
  48.907% and AMP latency increases 21.699% on RTX 2060; deployment parameters
  and FLOPs also increase, so the 3% lightweight gate fails and no 30-epoch
  screening run is started.
- [x] Record that every explicitly named Route A primary/backup candidate is
  now closed and that no candidate qualifies for a Fusion 100-epoch run.
- [x] Obtain the user's explicit choice: Route B is authorized because plain
  direct-concatenation Fusion leaves the method-innovation section too thin.
- [x] Complete the single diagnosis-driven Route B candidate (ABL) at 30
  epochs. It passes through mIoU +0.006236 with Fire neutral inside the
  conservative noise band.
- [x] Complete the promoted ABL 100-epoch formal run. Its best checkpoint gains
  mIoU +0.001695 and Fire IoU +0.002812, so it fails both internal method
  thresholds and is retained only as a B-route ablation.
- [x] Keep the test set sealed and close Route B without automatically adding
  a second loss candidate.

## Route C

- [x] Replace the zero-increase parameter/FLOP rule with the approved measured
  real-time gate: RTX 2060 paired same-session AMP benchmark, warmed up,
  repeated median latency no greater than 33.33 ms (at least 30 FPS).
- [x] Keep all accuracy thresholds, validation-only selection, seed discipline
  and the sealed test set unchanged.
- [x] Require one validation-mIoU versus latency Pareto registry for the five
  closed Route A modules, ABL, Route C, Fusion PIDNet-S and RoboFireFuseNet.
- [x] Formally activate Route C and freeze the order: evidence closeout, SAMF
  engineering/screening/formal run, then and only then TGM implementation.
- [x] Complete and archive the filled worst-20 manual-check summary: RGB
  invisible 20/20, heavy smoke 20/20, IR clearer 20/20, small fire 4/20,
  cross-checked against the Smoke-proximity proxy.
- [x] Complete full-validation Fire FN/FP spatial attribution. The strict
  lower-quartile rule degenerates at Q1=1 pixel and is preserved as written;
  Smoke neighbourhoods cover 69.90% of FN and 72.18% of FP.
- [x] Complete the IR-only 100-epoch motivation row and RGB/IR/Fusion paired
  efficiency measurements for Table 1; all three exceed 30 FPS.
- [x] Freeze a non-overwriting `route_c_*` IR-only 100-epoch config and launch
  record using label_fix A1, seed 200 and a fresh ImageNet initialization.
- [x] Prepare a same-process rotating-order RGB/IR/Fusion latency benchmark for
  Table 1; execution remains blocked until the IR best checkpoint is frozen.
- [x] Prepare a Table 1 builder that rejects split, input-channel or checkpoint
  mismatches across validation metrics, complexity and latency artifacts.
- [x] Create the unified Pareto registry template and a plotter that leaves
  untrained or unmeasured methods pending instead of inventing coordinates.
- [x] Complete SAMF engineering acceptance and its single 30-epoch screen.
  Engineering acceptance passed: beta-zero bitwise equivalence, AMP/two-stage
  gradients, inference interface, 7.705M parameters, 3.1545 GFLOPs and paired
  RTX 2060 median 32.32 FPS. The 26--30 window gained 0.005667 mIoU while Fire
  remained neutral inside the Fusion conservative band, so the screen passed.
- [x] Run the frozen SAMF definition for 100 epochs and apply the formal method
  threshold before unfreezing TGM. SAMF gained 0.001885 mIoU and 0.004068 Fire
  IoU, passed the 30 FPS gate at 34.57 FPS, but failed both internal accuracy
  thresholds; retain it as a positive ablation and unfreeze TGM.
- [x] Implement and engineering-check the single frozen TGM variant using the
  confirmed 1/4-to-1/8 pre-gate alignment, explicit spatial/channel gates,
  post-injection auxiliary P head and alpha-zero identity. Bitwise equivalence,
  AMP gradients and inference interface passed; 7.635M parameters, 3.0077
  GFLOPs and paired RTX 2060 44.34 FPS pass engineering admission.
- [x] Run the single TGM 30-epoch screen and evaluate both decisions. TGM
  gained -0.003078 mIoU and -0.006146 Fire, so it failed its own promotion and
  the Amendment 5 positive trigger; no TGM 100e or combination run is allowed.
- [x] Close Route C after auditing commit chronology, all conditional branches,
  paired latency gates and the sealed test set. Any new main-innovation search
  requires a separately approved route or preregistration amendment.
- [x] Before any TGM result, preregister Amendment 5: if TGM is also positive,
  allow one ABL+SAMF and one ABL+SAMF+TGM 30-epoch screen against the plain
  Fusion baseline; promote at mIoU +0.005 or Fire +0.01, subject to the paired
  RTX 2060 >=30 FPS gate and no class collapse.
- [x] Before TGM implementation, numerically define a positive TGM trigger as
  at least one strictly positive mIoU/Fire gain while the other remains inside
  the original -0.005/-Delta_fire tolerance and no class collapses.
- [x] Verify the Route C bibliography against formal titles, venues/years and
  DOI/original metadata; archive BibTeX and code-reuse status. Mark SGFNet,
  CBL, top-level FreqFusion and PSL as no-license reference-only sources.
