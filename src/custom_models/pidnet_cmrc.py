from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.pidnet import PIDNet


class ModalityHintStem(nn.Module):
    """Extract a lightweight quarter-resolution hint from one modality."""

    def __init__(
        self,
        input_channels: int,
        stem_channels: int = 8,
        output_channels: int = 16,
    ) -> None:
        super().__init__()
        if min(input_channels, stem_channels, output_channels) <= 0:
            raise ValueError("CMRC hint channels must be positive.")
        self.layers = nn.Sequential(
            nn.Conv2d(
                input_channels,
                stem_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(stem_channels, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                stem_channels,
                stem_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                groups=stem_channels,
                bias=False,
            ),
            nn.BatchNorm2d(stem_channels, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                stem_channels,
                output_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(output_channels, momentum=0.1),
            nn.ReLU(inplace=True),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.layers(image)


class PIDNetCMRC(PIDNet):
    """PIDNet-S with bounded cross-modal residual correction before branching.

    The original four-channel Fusion stem and all PIDNet branches remain intact.
    RGB and IR side hints can only add a bounded residual to the shared layer2
    feature.  The final correction convolution is zero-initialized, making the
    complete model exactly equivalent to the Fusion baseline at initialization.
    """

    def __init__(
        self,
        *args,
        thermal_channel: int = 3,
        hint_stem_channels: int = 8,
        hint_channels: int = 16,
        context_channels: int = 16,
        correction_hidden_channels: int = 16,
        residual_limit: float = 0.1,
        **kwargs,
    ) -> None:
        input_channels = int(kwargs.get("channels", 3))
        planes = int(kwargs.get("planes", 64))
        if input_channels != 4 or thermal_channel != 3:
            raise ValueError(
                "CMRC requires four-channel Fusion input ordered as RGB then IR."
            )
        if min(
            hint_stem_channels,
            hint_channels,
            context_channels,
            correction_hidden_channels,
        ) <= 0:
            raise ValueError("CMRC channel counts must be positive.")
        if residual_limit <= 0.0:
            raise ValueError("CMRC residual_limit must be positive.")
        super().__init__(*args, **kwargs)

        shared_channels = planes * 2
        joint_channels = context_channels + hint_channels * 3
        self.thermal_channel = int(thermal_channel)
        self.residual_limit = float(residual_limit)
        self.rgb_hint_stem = ModalityHintStem(
            3,
            stem_channels=int(hint_stem_channels),
            output_channels=int(hint_channels),
        )
        self.thermal_hint_stem = ModalityHintStem(
            1,
            stem_channels=int(hint_stem_channels),
            output_channels=int(hint_channels),
        )
        self.cmrc_context_projection = nn.Sequential(
            nn.Conv2d(
                shared_channels,
                context_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(context_channels, momentum=0.1),
            nn.ReLU(inplace=True),
        )
        self.cmrc_correction_head = nn.Sequential(
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
            nn.Conv2d(
                joint_channels,
                correction_hidden_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(correction_hidden_channels, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                correction_hidden_channels,
                shared_channels,
                kernel_size=1,
                bias=True,
            ),
        )
        self._initialize_cmrc_modules()

    @property
    def final_correction(self) -> nn.Conv2d:
        layer = self.cmrc_correction_head[-1]
        if not isinstance(layer, nn.Conv2d):
            raise RuntimeError("CMRC correction head has an invalid final layer.")
        return layer

    def _initialize_cmrc_modules(self) -> None:
        modules = (
            self.rgb_hint_stem,
            self.thermal_hint_stem,
            self.cmrc_context_projection,
            self.cmrc_correction_head,
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
        nn.init.zeros_(self.final_correction.weight)
        nn.init.zeros_(self.final_correction.bias)

    def _correct_shared_feature(
        self,
        shared_feature: torch.Tensor,
        rgb: torch.Tensor,
        thermal: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        target_size = shared_feature.shape[-2:]
        rgb_hint = F.adaptive_avg_pool2d(
            self.rgb_hint_stem(rgb),
            target_size,
        )
        thermal_hint = F.adaptive_avg_pool2d(
            self.thermal_hint_stem(thermal),
            target_size,
        )
        context = self.cmrc_context_projection(shared_feature)
        correction_input = torch.cat(
            (
                context,
                rgb_hint,
                thermal_hint,
                (rgb_hint - thermal_hint).abs(),
            ),
            dim=1,
        )
        correction_logits = self.cmrc_correction_head(correction_input)
        residual = self.residual_limit * torch.tanh(correction_logits)
        return shared_feature + residual, residual

    @staticmethod
    def _residual_aux(residual: torch.Tensor) -> dict[str, torch.Tensor]:
        detached = residual.detach().float().abs()
        return {
            "residual_abs_mean": detached.mean(),
            "residual_abs_max": detached.max(),
            "residual_saturation_ratio": detached.ge(0.095).float().mean(),
        }

    def forward(self, image: torch.Tensor, return_aux: bool = False):
        if image.ndim != 4 or image.shape[1] != 4:
            raise ValueError(
                "CMRC expected BCHW Fusion input with four channels, got "
                f"{tuple(image.shape)}."
            )
        rgb = image[:, :3]
        thermal = image[:, self.thermal_channel : self.thermal_channel + 1]

        feature = self.conv1(image)
        feature = self.layer1(feature)
        feature = self.relu(self.layer2(self.relu(feature)))
        feature, residual = self._correct_shared_feature(feature, rgb, thermal)

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
            outputs = [
                self.seghead_p(auxiliary_detail),
                segmentation,
                self.seghead_d(auxiliary_boundary),
            ]
        else:
            outputs = segmentation
        if return_aux:
            return outputs, self._residual_aux(residual)
        return outputs


def count_cmrc_modules(model: nn.Module) -> int:
    return sum(isinstance(module, PIDNetCMRC) for module in model.modules())


__all__ = [
    "ModalityHintStem",
    "PIDNetCMRC",
    "count_cmrc_modules",
]
