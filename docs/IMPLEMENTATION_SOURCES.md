# Implementation sources and adaptations

This file records the upstream implementation used for each paper-derived
module. A module is not considered verified until its formulas are checked
against the listed source.

## DEConv

- Paper: *DEA-Net: Single Image Dehazing Based on Detail-Enhanced Convolution
  and Content-Guided Attention*, IEEE TIP 2024.
- Official repository: https://github.com/cecret3350/DEA-Net
- Verified commit: `599d4d5533325aec0ceca81e661f3eeb2b0619e8`
- Pinned local checkout: `third_party/DEA-Net`
- Upstream source: `code/model/modules/deconv.py`
- Upstream deployment reference: `code/reparam.py`
- Local implementation: `src/custom_models/pidnet_deconv.py`

The local CDC, ADC, HDC and VDC kernel mappings follow the official source.
PIDNet-specific adaptations required by the experiment plan are:

1. all five branches use `bias=False` because the replaced PIDNet BasicBlock
   convolution is bias-free and keeps its original BatchNorm after DEConv;
2. the vanilla branch parameter remains named `weight`, allowing the original
   PIDNet ImageNet checkpoint key to load directly;
3. the four differential parameters are zero-initialized so replacement is
   initially exactly equivalent to the original convolution;
4. kernels are summed before one `F.conv2d`, and deployment replaces the
   module with one ordinary 3x3 `nn.Conv2d`.

## EMA multi-prototype supervision

- Primary method reference: https://github.com/LujianYao/psl
- Verified PSL commit: `39e6838315ff33222bf23011a793f4b7e72e79f6`
- Pinned local checkout: `third_party/PSL`
- PSL source inspected: `mmseg/models/decode_heads/PSL_main.py`
- EMA memory reference: https://github.com/tfzhou/ContrastiveSeg
- Local implementation: `src/prototype_learning.py`

The local method is an experiment-specific adaptation described by the final
plan: supervision is attached to PIDNet's 128-channel DFM fused feature, uses
an EMA-only prototype bank, and samples Fire with the synchronized connected-
component map. It is not represented as an unchanged copy of PSL. In
particular, original PSL PUO updates prototypes on a Stiefel manifold, while
the final PIDNet plan requires EMA-only detached prototypes. P4 therefore
computes within-class decorrelation on differentiable per-batch cluster
centers formed after hard assignment to detached EMA prototypes. This keeps
the gradient on fused features and leaves the EMA bank gradient-free. This is
described as a **feature-side decorrelation surrogate adapted from the PSL PUO
idea**, not as a reproduction of the original Stiefel-manifold PUO. A cluster
center is used only when at least two sampled features are assigned to it; a
class needs at least two valid centers, and valid class losses are averaged
equally while absent/undersampled classes are skipped.

## DySample

- Paper: *Learning to Upsample by Learning to Sample*, ICCV 2023.
- Official repository: https://github.com/tiny-smart/dysample
- Verified commit: `81a1de5caa95d55a0f5488425fa53ec7ef47f8f0`
- Pinned local checkout: `third_party/dysample`
- Upstream source used directly: `third_party/dysample/dysample.py`
- Upstream license: MIT License, Copyright (c) 2023 Wenze Liu
- Citation: Liu, Wenze; Lu, Hao; Fu, Hongtao; Cao, Zhiguo. ICCV 2023.

The PIDNet adaptation imports the official `DySample` class from the pinned
checkout rather than reimplementing its sampling equations. Initial screening
uses the official lightweight defaults `style="lp"`, `groups=4`, and
`dyscope=False`. PIDNet-specific wrappers only select the insertion point and
preserve the surrounding PIDNet modules and pretrained parameter names.
