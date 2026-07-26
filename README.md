# UAV Wildfire Smoke and Flame Segmentation

This project studies lightweight three-class segmentation of UAV RGB imagery:
background, smoke, and fire.

## Environment

- VS Code interpreter: `F:\anaconda3\envs\pytorch\python.exe`
- Verified: PyTorch 2.8.0, CUDA 12.6, RTX 2060

## Baseline

- Main model: PIDNet-S
- Main annotated dataset: the FLAME2 subset released with RoboFireFuseNet
- External smoke validation: Boreal Forest Fire

Implemented experiment routes:

- configurable augmentation screening A1/A2/A3;
- P-branch DEA-Net DEConv D1/D2 with deployment reparameterization;
- DFM-feature EMA multi-prototype ablations P1-P4;
- DEConv + multi-prototype combination;
- synchronized uint16 Fire connected-component cache;
- reproducible screening, engineering checks, checkpoint resume, complexity,
  speed/evaluation, and fused-feature analysis.

The exact experiment order and commands are in
`docs/FINAL_EXPERIMENT_RUNBOOK.md`. Paper-derived source mappings and local
adaptations are recorded in `docs/IMPLEMENTATION_SOURCES.md`.
