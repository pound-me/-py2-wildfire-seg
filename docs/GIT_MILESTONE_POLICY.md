# Git milestone policy for experiments

Effective date: 2026-07-26

The repository remote is `origin` and the working branch is `main`. From this
date forward, every completed key node is committed and pushed after its
outputs are closed and stable.

Key nodes include:

- a completed screening or formal experiment;
- a finalized experiment configuration or engineering implementation;
- a formal route, threshold or model-selection decision.

Commit rules:

1. Stage explicit files or closed experiment directories only. Do not use a
   broad stage command while a training process is active.
2. Never commit an actively written `metrics.jsonl`, log, checkpoint or other
   partial output.
3. Checkpoints and heavy binary artifacts remain excluded by `.gitignore`.
4. Commit messages must name the completed node and its purpose/result.
5. Push each milestone commit to `origin/main` after local verification.

## Policy activation snapshot

At activation, Route A diagnostics, the IR 30-epoch screening, the Fusion
30-epoch screening, the DySample Pag4 30-epoch archive, the finalized Route A
configs and the formal route decision were already present in the repository's
initial snapshot. Their evidence is recorded in:

- `docs/ROUTE_A_RGB_IR_DECISION_2026-07-26.md`
- `experiments/route_a_diagnostics/rgb_visibility_val/summary.json`
- `experiments/route_a_pidnet_s_ir/route_a_ir_30e_label_fix_seed200/`
- `experiments/route_a_pidnet_s_fusion/route_a_fusion_30e_label_fix_seed200/`
- `experiments/pidnet_s_dysample/dysample_pag4_30e_label_fix/`

The Fusion 100-epoch run
`experiments/route_a_pidnet_s_fusion/route_a_fusion_100e_label_fix_seed200/`
was still active when this policy was activated. Its changing files are
explicitly excluded from this policy-activation commit and will be committed
only after training and post-run evaluation finish.
