from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from custom_models.pidnet_lscm import LightweightSmokeContext, PIDNetLSCM


class ClassAwarePrototypeContext(LightweightSmokeContext):
    """Three-class context gate whose features support prototype learning."""

    def __init__(
        self,
        channels: int,
        num_classes: int = 3,
        smoke_class: int = 1,
        reduction: int = 4,
        dilations: tuple[int, ...] = (1, 2, 3),
    ) -> None:
        super().__init__(
            channels=channels,
            reduction=reduction,
            dilations=dilations,
        )
        self.num_classes = num_classes
        self.smoke_class = smoke_class
        del self.smoke_gate
        self.class_head = nn.Conv2d(
            channels,
            num_classes,
            kernel_size=1,
            bias=True,
        )
        nn.init.zeros_(self.class_head.weight)
        nn.init.zeros_(self.class_head.bias)

    def forward(
        self,
        feature: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        reduced = self.reduce(feature)
        contexts = [branch(reduced) for branch in self.context_branches]
        context = self.fuse(torch.cat(contexts, dim=1))
        class_logits = self.class_head(context)
        class_probabilities = torch.softmax(class_logits, dim=1)
        smoke_gate = class_probabilities[
            :,
            self.smoke_class : self.smoke_class + 1,
        ]
        enhanced = feature + self.residual_scale * smoke_gate * context
        return enhanced, class_logits, context


class PIDNetLSCMV3(PIDNetLSCM):
    """PIDNet-S with three-class context gating and prototype supervision."""

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
        self.smoke_context = ClassAwarePrototypeContext(
            channels=planes * 4,
            num_classes=num_classes,
            smoke_class=1,
        )

    def forward(self, image: torch.Tensor):
        feature = self.conv1(image)
        feature = self.layer1(feature)
        feature = self.relu(self.layer2(self.relu(feature)))
        detail_feature = self.layer3_(feature)
        boundary_feature = self.layer3_d(feature)

        output_size = boundary_feature.shape[-2:]

        feature = self.relu(self.layer3(feature))
        detail_feature = self.pag3(
            detail_feature,
            self.compression3(feature),
        )
        boundary_feature = boundary_feature + F.interpolate(
            self.diff3(feature),
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )
        if self.augment:
            auxiliary_detail = detail_feature

        feature = self.relu(self.layer4(feature))
        detail_feature = self.layer4_(self.relu(detail_feature))
        boundary_feature = self.layer4_d(self.relu(boundary_feature))

        detail_feature = self.pag4(
            detail_feature,
            self.compression4(feature),
        )
        boundary_feature = boundary_feature + F.interpolate(
            self.diff4(feature),
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )
        if self.augment:
            auxiliary_boundary = boundary_feature

        detail_feature = self.layer5_(self.relu(detail_feature))
        boundary_feature = self.layer5_d(self.relu(boundary_feature))
        context_feature = F.interpolate(
            self.spp(self.layer5(feature)),
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )
        context_feature, class_logits, prototype_features = self.smoke_context(
            context_feature
        )

        segmentation = self.final_layer(
            self.dfm(
                detail_feature,
                context_feature,
                boundary_feature,
            )
        )

        if self.augment:
            detail_output = self.seghead_p(auxiliary_detail)
            boundary_output = self.seghead_d(auxiliary_boundary)
            return [
                detail_output,
                segmentation,
                boundary_output,
                class_logits,
                prototype_features,
            ]
        return segmentation


__all__ = ["ClassAwarePrototypeContext", "PIDNetLSCMV3"]
