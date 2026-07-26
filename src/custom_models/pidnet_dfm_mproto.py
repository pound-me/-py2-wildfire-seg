from __future__ import annotations

import torch
import torch.nn.functional as F

from models.pidnet import PIDNet

from custom_models.pidnet_deconv import apply_p_branch_deconv


class PIDNetDFMMProto(PIDNet):
    """PIDNet-S exposing the 128-channel DFM feature only during training."""

    def forward(self, image: torch.Tensor):
        feature = self.conv1(image)
        feature = self.layer1(feature)
        feature = self.relu(self.layer2(self.relu(feature)))
        detail_feature = self.layer3_(feature)
        boundary_feature = self.layer3_d(feature)
        output_size = boundary_feature.shape[-2:]

        feature = self.relu(self.layer3(feature))
        detail_feature = self.pag3(detail_feature, self.compression3(feature))
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
        detail_feature = self.pag4(detail_feature, self.compression4(feature))
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
        fused_feature = self.dfm(
            detail_feature,
            context_feature,
            boundary_feature,
        )
        segmentation = self.final_layer(fused_feature)

        if self.augment:
            return [
                self.seghead_p(auxiliary_detail),
                segmentation,
                self.seghead_d(auxiliary_boundary),
                fused_feature,
            ]
        return segmentation


class PIDNetDEConvMProto(PIDNetDFMMProto):
    """Training feature model combining P-branch DEConv and EMA prototypes."""

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
        deconv_variant: str = "D1",
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
        replacements = apply_p_branch_deconv(self, deconv_variant)
        if replacements == 0:
            raise RuntimeError("No P-branch 3x3 convolutions were replaced.")


__all__ = ["PIDNetDFMMProto", "PIDNetDEConvMProto"]
