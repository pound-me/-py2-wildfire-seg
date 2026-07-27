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

### Formal 100-epoch outcome

The promoted run completed 100 epochs. On each model's mIoU-selected best
validation checkpoint, ABL versus the Fusion baseline is:

- mIoU: `0.8084227100` versus `0.8067280343`, gain `+0.0016946757`;
- Fire IoU: `0.6721227772` versus `0.6693104186`, gain `+0.0028123586`.

It fails both preregistered internal method-establishment thresholds. Route B
is therefore closed and ABL is retained only as a zero-inference-overhead
ablation. No test image or test metric was used.

## Amendment 4: Route C activation and measured real-time budget

Date: 2026-07-26

### Superseded rule

> Deployment parameters and FLOPs must not exceed the Fusion PIDNet-S
> baseline, with any structural increase rejected before training.

### Operative rule, approved and effective immediately

> A candidate may have a finite increase in deployment parameters and FLOPs
> provided that it reaches at least 30 FPS on the RTX 2060. Baseline and
> candidate must be benchmarked as a same-session pair, after warm-up, using
> repeated measurements whose median latency is the decision statistic.

Thirty FPS is equivalent to a median per-image latency no greater than
`33.33 ms`. Parameters and FLOPs remain mandatory reported quantities, but
neither an increase above the Fusion baseline nor the former 3% relative
slowdown rule is an automatic rejection. Measured RTX 2060 latency is the
hard engineering gate. The previously proposed fixed `FLOPs <= +5%` cap is
not the operative rule.

The 30-epoch screening rules, conservative Fire noise-band interpretation,
100-epoch internal method thresholds, validation-only selection, three-seed
discipline and sealed test set are unchanged.

### Unified Pareto accounting

All comparable methods must be placed in one validation-mIoU versus measured
latency Pareto registry and plot under the new paired benchmark protocol. The
required historical rows are the five closed Route A modules (DEConv D2,
mproto P1, DySample Pag4, DySample Context and FreqFusion Pag3), Route B ABL,
all Route C candidates, the plain Fusion PIDNet-S baseline and
RoboFireFuseNet. A method that was not trained keeps its engineering status
and has no invented mIoU value. Historical rejection decisions remain valid
records of the rule active at the time; the Pareto remeasurement does not
retroactively promote them into formal methods.

### Route C authorization

Route C is formally activated. Its first candidate is SAMF, selected from the
completed visibility diagnosis: all 20 manually reviewed worst-Fire-IoU
validation samples have invisible RGB fire, heavy smoke occlusion and clearer
IR. TGM is explicitly deferred until the SAMF screening result is frozen.

Approved by the user on 2026-07-26 before SAMF implementation or training.

## Amendment 5: positive-component combination screening

Date: 2026-07-27

Chronology: approved after the ABL and SAMF 100-epoch results were frozen, and
before any TGM implementation, engineering measurement or training result was
produced.

### Superseded combination rule

> A combination may be screened only when every individual structural method
> has first passed the 100-epoch internal method-establishment threshold.

This rule is retired for the specifically named combinations below. It remains
the historical rule under which the earlier SAMF-only formal decision was
written.

### Operative combination rule, approved and effective immediately

ABL and SAMF both show positive validation direction while acting on different
error sources: ABL changes training-time boundary supervision without inference
overhead, whereas SAMF performs smoke-conditioned thermal injection. Their two-
component combination is therefore independently authorized. If the standalone
TGM result is also positive, TGM does not need to pass the existing single-
method promotion or 100-epoch establishment threshold before it is added to the
three-component screen:

- Fusion + ABL + SAMF;
- Fusion + ABL + SAMF + TGM.

These are the only newly authorized combination configurations; the amendment
does not authorize an unrestricted combination or hyperparameter sweep.

Each combination receives one 30-epoch run under the same label-fix protocol,
seed 200 and 100-epoch learning-rate horizon. Its gain is always computed
against the plain Fusion PIDNet-S baseline, not against an individual component
or an additive estimate. A combination passes the 30-epoch screen when either:

- mIoU gain versus Fusion is at least `+0.005`; or
- Fire IoU gain versus Fusion is at least `+0.01`.

Smoke and Fire class collapse remains prohibited. A passing combination is
promoted to one 100-epoch formal run. Every combination architecture must first
pass the RTX 2060 same-session paired deployment gate of at least 30 FPS.
Parameters, FLOPs and latency remain mandatory reported quantities. The test set
remains sealed, and no three-seed expansion occurs before a 100-epoch result
meets the formal method threshold.

The TGM-positive condition applies only to `Fusion + ABL + SAMF + TGM`; it is
not a prerequisite for `Fusion + ABL + SAMF`. The user's phrase "TGM is also
positive" is operationally frozen as follows, using the same epochs 26--30
means against the plain Fusion baseline:

- at least one of TGM mIoU gain or Fire IoU gain must be strictly greater than
  zero;
- if mIoU is not the positive metric, its gain must remain at least `-0.005`;
- if Fire is not the positive metric, its gain must remain no lower than the
  negative Fusion conservative Fire band `-Delta_fire`;
- Smoke and Fire class collapse is prohibited; and
- the standalone TGM architecture must pass the paired RTX 2060 `>=30 FPS`
  engineering gate.

This sign criterion is only the trigger for adding TGM to the three-component
screen; it does not control the ABL+SAMF pair and does not promote TGM itself to
a 100-epoch method. TGM's own promotion still uses the existing single-method
screening rules.

Approved and numerically confirmed by the user on 2026-07-27 before TGM
implementation, engineering output or training result existed.

### Amendment 5 trigger outcome and execution correction

The frozen standalone TGM run completed 30 epochs after this amendment was
committed. Against the Fusion epochs 26--30 window it obtained mIoU gain
`-0.0030784552` and Fire IoU gain `-0.0061460136`. Both primary gains are
negative. Although the other tolerance, class-stability and paired 44.78 FPS
engineering conditions are satisfied, the required strictly positive TGM gain
does not exist. Therefore `Fusion+ABL+SAMF+TGM` is not authorized.

The earlier closure audit incorrectly treated TGM positivity as a prerequisite
for both combinations. The user's same-day clarification confirms that this was
an execution omission: because standalone ABL and standalone SAMF are both
positive, `Fusion+ABL+SAMF` is independently required and remains pending. Its
30-epoch gain is evaluated directly against plain Fusion at mIoU `+0.005` or
Fire IoU `+0.01`, subject to no class collapse and the paired RTX 2060
`>=30 FPS` gate. Passing promotes one fresh-initialization 100-epoch run;
failure genuinely closes Route C. No threshold or observed result is changed.

The required ABL+SAMF run subsequently completed all 30 epochs. Its epochs
26--30 mean gains against plain Fusion were mIoU `+0.0045886046` and Fire IoU
`+0.0029347071`, with no class collapse and checkpoint-paired RTX 2060 speed of
45.41 FPS. It therefore misses both `+0.005` / `+0.01` combination thresholds
and is not promoted to 100 epochs. The triple remains unauthorized because TGM
was negative. With the omitted pair now resolved, Route C is genuinely closed.

No test image or test metric was used in this trigger decision.
