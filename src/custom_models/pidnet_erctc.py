from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.pidnet import PIDNet


class FixedThermalFrontier(nn.Module):
    """Extract a normalized, non-trainable Sobel frontier from pooled IR."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer(
            "sobel_x",
            torch.tensor(
                [[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]],
                dtype=torch.float32,
            ).unsqueeze(0),
            persistent=True,
        )
        self.register_buffer(
            "sobel_y",
            torch.tensor(
                [[[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]],
                dtype=torch.float32,
            ).unsqueeze(0),
            persistent=True,
        )

    def forward(self, infrared: torch.Tensor) -> torch.Tensor:
        if infrared.ndim != 4 or infrared.shape[1] != 1:
            raise ValueError("FixedThermalFrontier expects a BCHW one-channel IR tensor.")
        with torch.autocast(device_type=infrared.device.type, enabled=False):
            infrared_fp32 = infrared.float()
            gradient_x = F.conv2d(infrared_fp32, self.sobel_x, padding=1)
            gradient_y = F.conv2d(infrared_fp32, self.sobel_y, padding=1)
            frontier = 0.25 * (gradient_x.abs() + gradient_y.abs())
            return frontier.clamp_(0.0, 1.0)


class PooledModalityProjection(nn.Module):
    """Project an already pooled modality tensor with a cheap 1x1 convolution."""

    def __init__(self, input_channels: int) -> None:
        super().__init__()
        self.projection = nn.Conv2d(
            input_channels,
            16,
            kernel_size=1,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(16, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.projection(image)))


class PIDNetERCTC(PIDNet):
    """Edge-aware RGB-conditioned thermal calibration at Pag3.

    The baseline four-channel PIDNet stem remains unchanged.  A lightweight
    side branch observes RGB context, IR appearance and a fixed Sobel thermal
    frontier.  Two zero-initialized scalar residuals independently learn broad
    region calibration and frontier-aware correction.  With both scalars at zero,
    the complete network is bitwise equivalent to the Fusion baseline.
    """

    def __init__(
        self,
        *args,
        thermal_channel: int = 3,
        compressed_channels: int = 16,
        **kwargs,
    ) -> None:
        channels = int(kwargs.get("channels", 3))
        if channels < 4 or channels <= thermal_channel:
            raise ValueError(
                "ERCTC requires a Fusion tensor with RGB channels 0:3 and "
                f"thermal channel {thermal_channel}; received {channels} channels."
            )
        if compressed_channels <= 0:
            raise ValueError("compressed_channels must be positive")
        planes = int(kwargs.get("planes", 64))
        super().__init__(*args, **kwargs)

        pag_channels = planes * 2
        self.thermal_channel = int(thermal_channel)
        self.compressed_channels = int(compressed_channels)
        self.thermal_frontier = FixedThermalFrontier()
        self.rgb_context_stem = PooledModalityProjection(3)
        self.thermal_context_stem = PooledModalityProjection(2)
        self.erctc_detail_compression = nn.Sequential(
            nn.Conv2d(
                pag_channels,
                compressed_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(compressed_channels, momentum=0.1),
            nn.ReLU(inplace=True),
        )
        joint_channels = compressed_channels + 16 + 16
        self.erctc_context_depthwise = nn.Sequential(
            nn.Conv2d(
                joint_channels,
                joint_channels,
                kernel_size=3,
                padding=1,
                groups=joint_channels,
                bias=False,
            ),
            nn.BatchNorm2d(joint_channels, momentum=0.1),
            nn.ReLU(inplace=True),
        )
        self.erctc_region_logits = nn.Conv2d(
            joint_channels,
            1,
            kernel_size=1,
            bias=True,
        )
        self.erctc_thermal_projection = nn.Conv2d(
            16,
            pag_channels,
            kernel_size=1,
            bias=False,
        )
        self.erctc_region_scale = nn.Parameter(torch.zeros(1, dtype=torch.float32))
        self.erctc_frontier_scale = nn.Parameter(torch.zeros(1, dtype=torch.float32))
        self._initialize_erctc_modules()

    def _initialize_erctc_modules(self) -> None:
        modules = (
            self.rgb_context_stem,
            self.thermal_context_stem,
            self.erctc_detail_compression,
            self.erctc_context_depthwise,
            self.erctc_region_logits,
            self.erctc_thermal_projection,
        )
        for module in modules:
            for layer in module.modules():
                if isinstance(layer, nn.Conv2d):
                    nn.init.kaiming_normal_(
                        layer.weight,
                        mode="fan_out",
                        nonlinearity="relu",
                    )
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
                elif isinstance(layer, nn.BatchNorm2d):
                    nn.init.ones_(layer.weight)
                    nn.init.zeros_(layer.bias)
        nn.init.zeros_(self.erctc_region_scale)
        nn.init.zeros_(self.erctc_frontier_scale)

    def _calibrate_detail(
        self,
        detail_feature: torch.Tensor,
        rgb_feature: torch.Tensor,
        thermal_feature: torch.Tensor,
        thermal_frontier: torch.Tensor,
    ) -> torch.Tensor:
        target_size = detail_feature.shape[-2:]
        aligned_rgb = F.adaptive_avg_pool2d(rgb_feature, target_size)
        aligned_thermal = F.adaptive_avg_pool2d(thermal_feature, target_size)
        aligned_frontier = F.adaptive_max_pool2d(thermal_frontier, target_size)
        compressed_detail = self.erctc_detail_compression(detail_feature)
        context = self.erctc_context_depthwise(
            torch.cat((compressed_detail, aligned_rgb, aligned_thermal), dim=1)
        )
        signed_region = (
            2.0 * torch.sigmoid(self.erctc_region_logits(context).float()) - 1.0
        ).to(dtype=detail_feature.dtype)
        projected_thermal = self.erctc_thermal_projection(aligned_thermal)
        region_scale = self.erctc_region_scale.to(dtype=detail_feature.dtype)
        frontier_scale = self.erctc_frontier_scale.to(dtype=detail_feature.dtype)
        return (
            detail_feature
            + region_scale * signed_region * projected_thermal
            + frontier_scale
            * aligned_frontier.to(dtype=detail_feature.dtype)
            * projected_thermal
        )

    def forward(self, image: torch.Tensor):
        if image.ndim != 4 or image.shape[1] <= self.thermal_channel:
            raise ValueError(
                "ERCTC expected a BCHW Fusion tensor containing RGB and IR channels."
            )
        rgb = image[:, :3]
        infrared = image[:, self.thermal_channel : self.thermal_channel + 1]
        pooled_rgb = F.avg_pool2d(rgb, kernel_size=4, stride=4)
        pooled_infrared = F.avg_pool2d(infrared, kernel_size=4, stride=4)
        thermal_frontier = self.thermal_frontier(pooled_infrared)
        rgb_feature = self.rgb_context_stem(pooled_rgb)
        thermal_feature = self.thermal_context_stem(
            torch.cat(
                (
                    pooled_infrared,
                    thermal_frontier.to(dtype=pooled_infrared.dtype),
                ),
                dim=1,
            )
        )

        feature = self.conv1(image)
        feature = self.layer1(feature)
        feature = self.relu(self.layer2(self.relu(feature)))
        detail_feature = self.layer3_(feature)
        boundary_feature = self.layer3_d(feature)
        output_size = boundary_feature.shape[-2:]

        feature = self.relu(self.layer3(feature))
        detail_feature = self.pag3(detail_feature, self.compression3(feature))
        detail_feature = self._calibrate_detail(
            detail_feature,
            rgb_feature,
            thermal_feature,
            thermal_frontier,
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


def count_erctc_modules(model: nn.Module) -> int:
    return sum(isinstance(module, PIDNetERCTC) for module in model.modules())


__all__ = [
    "FixedThermalFrontier",
    "PooledModalityProjection",
    "PIDNetERCTC",
    "count_erctc_modules",
]
