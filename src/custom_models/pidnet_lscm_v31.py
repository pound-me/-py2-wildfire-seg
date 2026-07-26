from __future__ import annotations

import torch

from custom_models.pidnet_lscm_v3 import (
    ClassAwarePrototypeContext,
    PIDNetLSCMV3,
)


class TrainingOnlyPrototypeContext(ClassAwarePrototypeContext):
    """Prototype-supervised context without auxiliary inference gating."""

    def __init__(
        self,
        channels: int,
        num_classes: int = 3,
        reduction: int = 4,
        dilations: tuple[int, ...] = (1, 2, 3),
        auxiliary_head: bool = True,
    ) -> None:
        super().__init__(
            channels=channels,
            num_classes=num_classes,
            smoke_class=1,
            reduction=reduction,
            dilations=dilations,
        )
        self.auxiliary_head = auxiliary_head
        self.residual_scale.data.fill_(0.05)
        if not auxiliary_head:
            self.class_head = None

    def forward(
        self,
        feature: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        reduced = self.reduce(feature)
        contexts = [branch(reduced) for branch in self.context_branches]
        context = self.fuse(torch.cat(contexts, dim=1))
        enhanced = feature + self.residual_scale * context
        if self.class_head is None:
            class_logits = context.new_zeros(
                context.shape[0],
                0,
                context.shape[2],
                context.shape[3],
            )
        else:
            class_logits = self.class_head(context)
        return enhanced, class_logits, context


class PIDNetLSCMV31(PIDNetLSCMV3):
    """LSCM v3.1 with training-only balanced class/prototype supervision."""

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
        self.smoke_context = TrainingOnlyPrototypeContext(
            channels=planes * 4,
            num_classes=num_classes,
            auxiliary_head=augment,
        )


__all__ = ["TrainingOnlyPrototypeContext", "PIDNetLSCMV31"]
