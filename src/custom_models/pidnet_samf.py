from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.pidnet import PIDNet
from models.pidnet_utils import segmenthead


class ThermalStem(nn.Module):
    """The confirmed two-layer 1/4-resolution IR feature stem."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            1,
            16,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(16, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            16,
            32,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )

    def forward(self, infrared: torch.Tensor) -> torch.Tensor:
        return self.conv2(self.relu(self.bn1(self.conv1(infrared))))


class PIDNetSAMF(PIDNet):
    """Smoke-aware thermal injection at the 1/8 Pag3 fusion output.

    The gate is computed from the pre-injection Pag3 feature by the existing
    PIDNet auxiliary semantic head. The head remains internally active during
    deployment, while inference still returns only the final segmentation.
    """

    def __init__(
        self,
        *args,
        smoke_class: int = 1,
        thermal_channel: int = 3,
        **kwargs,
    ) -> None:
        requested_augment = bool(kwargs.get("augment", True))
        channels = int(kwargs.get("channels", 3))
        if channels <= thermal_channel:
            raise ValueError(
                "SAMF requires a Fusion input containing the configured IR "
                f"channel {thermal_channel}; received {channels} channels."
            )
        num_classes = int(kwargs.get("num_classes", 19))
        planes = int(kwargs.get("planes", 64))
        head_planes = int(kwargs.get("head_planes", 128))
        if not 0 <= smoke_class < num_classes:
            raise ValueError(
                f"Invalid SAMF smoke class {smoke_class} for {num_classes} classes."
            )
        super().__init__(*args, **kwargs)
        self.smoke_class = int(smoke_class)
        self.thermal_channel = int(thermal_channel)
        self.thermal_stem = ThermalStem()
        self.thermal_projection = nn.Conv2d(
            32,
            planes * 2,
            kernel_size=1,
            bias=False,
        )
        self.samf_beta = nn.Parameter(torch.zeros(1, dtype=torch.float32))

        # Official PIDNet removes auxiliary heads for augment=False. SAMF uses
        # seghead_p as an internal deployment gate, so retain only that head.
        if not requested_augment:
            self.seghead_p = segmenthead(
                planes * 2,
                head_planes,
                num_classes,
            )

        self._initialize_samf_modules()

    def _initialize_samf_modules(self) -> None:
        for module in (self.thermal_stem, self.thermal_projection):
            for layer in module.modules():
                if isinstance(layer, nn.Conv2d):
                    nn.init.kaiming_normal_(
                        layer.weight,
                        mode="fan_out",
                        nonlinearity="relu",
                    )
                elif isinstance(layer, nn.BatchNorm2d):
                    nn.init.constant_(layer.weight, 1.0)
                    nn.init.constant_(layer.bias, 0.0)
        nn.init.zeros_(self.samf_beta)

    def _inject_thermal(
        self,
        detail_feature: torch.Tensor,
        thermal_feature: torch.Tensor,
        smoke_logits: torch.Tensor,
    ) -> torch.Tensor:
        aligned_thermal = F.adaptive_avg_pool2d(
            thermal_feature,
            detail_feature.shape[-2:],
        )
        aligned_thermal = self.thermal_projection(aligned_thermal)
        smoke_probability = torch.softmax(
            smoke_logits.float(),
            dim=1,
        )[:, self.smoke_class : self.smoke_class + 1]
        smoke_probability = smoke_probability.to(dtype=detail_feature.dtype)
        beta = self.samf_beta.to(dtype=detail_feature.dtype)
        return detail_feature + beta * smoke_probability * aligned_thermal

    def forward(self, image: torch.Tensor):
        if image.ndim != 4 or image.shape[1] <= self.thermal_channel:
            raise ValueError(
                "SAMF expected a BCHW Fusion tensor containing the IR channel."
            )
        infrared = image[:, self.thermal_channel : self.thermal_channel + 1]
        thermal_feature = self.thermal_stem(infrared)

        feature = self.conv1(image)
        feature = self.layer1(feature)
        feature = self.relu(self.layer2(self.relu(feature)))
        detail_feature = self.layer3_(feature)
        boundary_feature = self.layer3_d(feature)
        output_size = boundary_feature.shape[-2:]

        feature = self.relu(self.layer3(feature))
        detail_feature = self.pag3(detail_feature, self.compression3(feature))
        smoke_logits = self.seghead_p(detail_feature)
        detail_feature = self._inject_thermal(
            detail_feature,
            thermal_feature,
            smoke_logits,
        )
        boundary_feature = boundary_feature + F.interpolate(
            self.diff3(feature),
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )

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
        segmentation = self.final_layer(
            self.dfm(detail_feature, context_feature, boundary_feature)
        )
        if self.augment:
            return [
                smoke_logits,
                segmentation,
                self.seghead_d(auxiliary_boundary),
            ]
        return segmentation


def count_samf_modules(model: nn.Module) -> int:
    return sum(isinstance(module, PIDNetSAMF) for module in model.modules())


__all__ = ["PIDNetSAMF", "ThermalStem", "count_samf_modules"]
