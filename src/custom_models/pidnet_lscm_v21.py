from __future__ import annotations

import torch

from custom_models.pidnet_lscm import (
    LightweightSmokeContextV2,
    PIDNetLSCMV2,
)


class GradientIsolatedSmokeContext(LightweightSmokeContextV2):
    """Supervise the smoke gate without pushing aux gradients into context."""

    def forward(
        self,
        feature: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        reduced = self.reduce(feature)
        contexts = [branch(reduced) for branch in self.context_branches]
        context = self.fuse(torch.cat(contexts, dim=1))

        # The gate head learns from the smoke auxiliary target, while shared
        # context features remain governed by the three-class segmentation
        # objective. Main-task gradients can still update the gate head.
        smoke_logits = self.smoke_gate[0](context.detach())
        gate = torch.sigmoid(smoke_logits)
        enhanced = feature + self.residual_scale * gate * context
        return enhanced, smoke_logits


class PIDNetLSCMV21(PIDNetLSCMV2):
    """LSCM v2.1 with gradient-isolated smoke auxiliary supervision."""

    def __init__(
        self,
        m: int = 2,
        n: int = 3,
        num_classes: int = 3,
        planes: int = 32,
        ppm_planes: int = 96,
        head_planes: int = 128,
        augment: bool = True,
        channels: int = 3,
    ) -> None:
        super().__init__(
            m=m,
            n=n,
            num_classes=num_classes,
            planes=planes,
            ppm_planes=ppm_planes,
            head_planes=head_planes,
            augment=augment,
            channels=channels,
        )
        self.smoke_context = GradientIsolatedSmokeContext(planes * 4)


__all__ = ["GradientIsolatedSmokeContext", "PIDNetLSCMV21"]
