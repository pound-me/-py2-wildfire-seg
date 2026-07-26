# DySample screening design

Date: 2026-07-26

The backup route is enabled by the formally logged rule amendment. All
screening runs use the adopted A1 protocol, seed 200, strict label-fix data,
30 training epochs and the 100-epoch learning-rate horizon.

## Observed PIDNet-S tensor paths at 256x256 input

- Pag3 semantic input: 64x16x16 -> 64x32x32
- Pag4 semantic input: 64x8x8 -> 64x32x32
- SPP context output: 128x4x4 -> 128x32x32 before DFM
- Main segmentation logits: 3x32x32 (resized outside the model for loss and
  evaluation)

## Variant U1: late Pag4 semantic-to-detail fusion

Model variant name: `pag4`

Pag4 is the last direct I-branch-to-P-branch semantic fusion before the final
P-branch refinement and DFM. It upsamples the 8x8 semantic value and its 32-
channel projected query to the 32x32 P-branch resolution. U1 replaces both
bilinear operations inside Pag4 with separate official DySample modules at
scale 4. Replacing both is required to keep the PagFM similarity query and the
mixed semantic value spatially consistent. Pag3 remains unchanged so this is
a single late-fusion intervention rather than a multi-stage combination.

Expected strength: improved late semantic/detail alignment and Fire boundary
placement with small parameter/FLOP growth.

## Variant U2: SPP context-to-DFM upsampling

Model variant name: `context`

The 128-channel SPP semantic context is upsampled from 4x4 to 32x32 immediately
before DFM combines the P, I and D branches. U2 replaces only this scale-8
bilinear interpolation with one official DySample module. Pag3, Pag4, D-branch
interpolation and the final external logit resize remain unchanged.

Expected strength: learned spatial reconstruction of the final semantic
context supplied to DFM, directly targeting the boundary/detail dilution seen
in the P-route failure analysis.

## Deliberately excluded from initial screening

- PAPPM internal scale1--scale4 interpolation: changing four heterogeneous
  paths at once would confound the first backup ablation.
- Pag3 plus Pag4 together: this would mix early and late semantic fusion.
- Final 3-channel logit upsampling: it operates outside the current model
  interface, lacks high-resolution guidance, and its 256x256 grid-sampling
  latency could dominate the strict real-time budget.
- Any U1+U2 combination: only considered after an individual variant passes.

## Fixed implementation settings

- Official checkout commit:
  `81a1de5caa95d55a0f5488425fa53ec7ef47f8f0`
- License: MIT
- DySample style: `lp`
- Groups: 4
- Dynamic scope branch: disabled (`dyscope=False`)
- No customized CUDA extension

## Admission checks before a 30-epoch run

1. Official module is imported from the pinned checkout and the checkout is
   clean.
2. Pretrained PIDNet tensors still match the baseline count.
3. AMP forward/backward succeeds and every inserted offset convolution has a
   finite, nonzero gradient.
4. Training returns the original three PIDNet segmentation outputs; inference
   returns only the main tensor.
5. Output shapes match baseline PIDNet at every unchanged interface.
6. Parameter/FLOP increments are recorded before training.

## Screening rule

Each variant is run independently for 30 epochs. Epochs 26--30 are compared
with the adopted A1 baseline using the existing conservative Fire noise band
and screening rules. No U1/U2 combination or 100-epoch run is started unless
an individual variant passes. Deployment latency is measured on RTX 2060 and
must not be more than 3% slower than the paired baseline benchmark.
