# Experiment-rule amendments

This log records changes to experiment-selection rules after the final plan was
frozen. Each amendment must be recorded before the newly enabled experiment is
implemented or trained.

## Amendment 2026-07-26: backup-route trigger

### Original rule

> DySample and FreqFusion are backup routes and are enabled only when both
> DEConv and mproto fail 30-epoch screening.

### Revised rule

> The backup route is triggered when both primary routes fail their applicable
> gate at either stage: a route fails if it does not pass 30-epoch screening,
> or if its screening winner proceeds to 100-epoch formal training but does
> not meet either predefined internal method-establishment threshold on the
> mIoU-selected best checkpoint. Both the DEConv route and the mproto route
> must be failed before DySample/FreqFusion is enabled.

Chinese operative wording:

> 两条主路线在30轮筛选或100轮正式训练中均未达对应门槛，即触发备用路线。

### Reason for the amendment

The original rule intended to delay backup modules until both primary routes
were shown not to work. A 100-epoch formal failure is stronger evidence of a
route not working than a 30-epoch screening failure, but the original wording
only named the earlier gate. The revised wording closes that real rule gap; it
does not retroactively change any metric, threshold, checkpoint-selection
rule, seed, augmentation protocol, or dataset split.

### Evidence that the revised trigger is satisfied

#### mproto route

- P1--P4 completed the prescribed 30-epoch sequence.
- P1 was the route winner and therefore the only prototype variant promoted to
  100-epoch formal training under the existing selection rule.
- P1 completed 100/100 epochs with healthy EMA state and no runtime/data error.
- On the mIoU-selected best checkpoint, P1 versus baseline is:
  - mIoU: -0.0066401627
  - Fire IoU: +0.0009480065
- It fails both internal method-establishment rules. P2--P4 were already
  rejected at the 30-epoch route-selection stage and are not independently
  promoted to 100 epochs.

#### DEConv route

- D1 and D2 completed 30-epoch screening; D2 was promoted.
- D2 completed 100/100 epochs.
- On the mIoU-selected best checkpoint, D2 versus baseline is:
  - mIoU: +0.0029841797
  - Fire IoU: +0.0019906371
- It fails both internal method-establishment rules.
- Reparameterization, deployment parameter/FLOP equality and the RTX 2060
  speed constraint all pass, so this is an accuracy-gate failure rather than
  an engineering failure.

Both primary routes therefore meet the revised failure condition. The backup
route is formally activated, with DySample evaluated before FreqFusion because
it has the smaller expected deployment overhead.

### Approval

Approved by the user on 2026-07-26 before downloading or implementing
DySample.

## Amendment 2: Route A RGB+IR fusion promotion

Date: 2026-07-26

The project route is changed from RGB-only structural optimization to an
RGB+IR fusion baseline after the prescribed visibility diagnostics, IR-only
screening and fusion screening supplied direct evidence that the RGB input is
the dominant limitation. The complete evidence, numerical thresholds and
follow-up rules are recorded in `docs/ROUTE_A_RGB_IR_DECISION_2026-07-26.md`.

The fusion model is promoted to a 100-epoch formal-baseline run. The existing
RGB baseline is retained as motivation evidence, and DEConv/mproto code is
retained for fresh 30-epoch screening only after the fusion baseline is
established. No test image or test metric was used in this amendment.

## Amendment 3: diagnosis-driven Route B single candidate

Date: 2026-07-26

### Previous decision point

After all named Route A structural and backup candidates were closed, the
project required an explicit choice between freezing plain Fusion PIDNet-S and
authorizing one tightly scoped additional research candidate.

### User authorization

The user selected Route B because direct RGB+IR concatenation alone leaves the
paper's method-innovation section too thin. Route B is limited to one 30-epoch
candidate; the test set remains sealed and the Fusion baseline, seed,
augmentation protocol, learning-rate horizon and screening thresholds remain
unchanged.

### Evidence-driven candidate selection

The fixed validation error profile shows that 78.4324% of all errors lie
within the three-pixel semantic-boundary band. Smoke boundary recall is
0.747883, Fire boundary recall is 0.835477, and 63.32% of Fire errors are
Fire-to-Smoke confusions. Consequently, Route B must directly supervise the
main semantic logits at class boundaries and must add no inference parameter,
FLOP or output.

Active Boundary Loss (ABL, AAAI 2022 Oral) is selected as the only Route B
candidate. The author repository is Apache-2.0 and pinned at commit
`1511507533ad98f04ea26e3648360a6c1d477d37`. Conditional Boundary Loss is not
selected because its official implementation is coupled to OCRHead/MaskFormer
and the audited repository does not declare a repository-level LICENSE.

### Screening discipline

- Run exactly one ABL configuration for 30 epochs, seed 200, with a 100-epoch
  polynomial-learning-rate horizon.
- Use ABL weight 1.0 and the paper/source defaults without a weight sweep.
- Compare epochs 26--30 against the frozen Fusion baseline window.
- A second Route B candidate is prohibited unless ABL is a documented near
  miss with a specific, correctable cause and the user separately approves it.
- Otherwise Route B closes after ABL. No test metric may be consulted.

### Lightweight-budget proposal remains separate

A possible future change from zero-overhead structural gating to a deployment
budget is only a proposal for Route C. It does not change the active rule and
does not retroactively admit any failed candidate unless the user and adviser
approve it in writing.

### Screening outcome

The single ABL run completed 30 epochs on 2026-07-26. Against the frozen
Fusion epochs 26--30 window it achieved:

- mIoU mean: `0.7923770674`, gain `+0.0062361118`;
- Fire IoU mean: `0.6453304305`, gain `-0.0013797433`;
- Fire interpretation: neutral inside the baseline conservative band
  `±0.0142509431`;
- no Smoke/Fire class collapse.

It therefore passes the predefined mIoU rule and is promoted, without any
hyperparameter change, to one 100-epoch formal run. The test set remains
sealed.
