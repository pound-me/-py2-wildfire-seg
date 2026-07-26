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
