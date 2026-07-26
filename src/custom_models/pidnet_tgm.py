from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.pidnet import PIDNet

from custom_models.pidnet_samf import ThermalStem


class PIDNetTGM(PIDNet):
    """Standalone thermal-guided Pag3 gating for Route C."""

    def __init__(
        self,
        *args,
        thermal_channel: int = 3,
        **kwargs,
    ) -> None:
        channels = int(kwargs.get("channels", 3))
        if channels <= thermal_channel:
            raise ValueError(
                "TGM requires a Fusion input containing the configured IR "
                f"channel {thermal_channel}; received {channels} channels."
            )
        planes = int(kwargs.get("planes", 64))
        super().__init__(*args, **kwargs)
        pag_channels = planes * 2
        self.thermal_channel = int(thermal_channel)
        self.thermal_stem = ThermalStem()
        self.tgm_spatial_depthwise = nn.Conv2d(
            32,
            32,
            kernel_size=3,
            padding=1,
            groups=32,
            bias=False,
        )
        self.tgm_spatial_pointwise = nn.Conv2d(
            32,
            1,
            kernel_size=1,
            bias=False,
        )
        self.tgm_channel_fc = nn.Linear(32, pag_channels, bias=True)
        self.tgm_feature_projection = nn.Conv2d(
            pag_channels,
            pag_channels,
            kernel_size=1,
            bias=False,
        )
        self.tgm_alpha = nn.Parameter(torch.zeros(1, dtype=torch.float32))
        self._initialize_tgm_modules()

    def _initialize_tgm_modules(self) -> None:
        convolution_modules = (
            self.thermal_stem,
            self.tgm_spatial_depthwise,
            self.tgm_spatial_pointwise,
            self.tgm_feature_projection,
        )
        for module in convolution_modules:
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
        nn.init.xavier_uniform_(self.tgm_channel_fc.weight)
        nn.init.zeros_(self.tgm_channel_fc.bias)
        nn.init.zeros_(self.tgm_alpha)

    def _apply_tgm(
        self,
        detail_feature: torch.Tensor,
        thermal_feature: torch.Tensor,
    ) -> torch.Tensor:
        aligned_thermal = F.adaptive_avg_pool2d(
            thermal_feature,
            detail_feature.shape[-2:],
        )
        spatial_logits = self.tgm_spatial_pointwise(
            self.tgm_spatial_depthwise(aligned_thermal)
        )
        spatial_gate = torch.sigmoid(spatial_logits.float()).to(
            dtype=detail_feature.dtype
        )

        channel_descriptor = F.adaptive_avg_pool2d(
            thermal_feature,
            output_size=1,
        ).flatten(1)
        channel_logits = self.tgm_channel_fc(channel_descriptor.float())
        channel_gate = torch.sigmoid(channel_logits.float()).to(
            dtype=detail_feature.dtype
        )[:, :, None, None]

        transformed_feature = self.tgm_feature_projection(detail_feature)
        alpha = self.tgm_alpha.to(dtype=detail_feature.dtype)
        return (
            detail_feature
            + alpha * channel_gate * spatial_gate * transformed_feature
        )

    def forward(self, image: torch.Tensor):
        if image.ndim != 4 or image.shape[1] <= self.thermal_channel:
            raise ValueError(
                "TGM expected a BCHW Fusion tensor containing the IR channel."
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
        detail_feature = self._apply_tgm(detail_feature, thermal_feature)
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
        segmentation = self.final_layer(
            self.dfm(detail_feature, context_feature, boundary_feature)
        )
        if self.augment:
            return [
                self.seghead_p(auxiliary_detail),
                segmentation,
                self.seghead_d(auxiliary_boundary),
            ]
        return segmentation


def count_tgm_modules(model: nn.Module) -> int:
    return sum(isinstance(module, PIDNetTGM) for module in model.modules())


__all__ = ["PIDNetTGM", "count_tgm_modules"]
