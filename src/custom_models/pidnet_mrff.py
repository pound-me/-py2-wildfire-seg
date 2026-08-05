from __future__ import annotations

import torch
import torch.nn as nn

from models.pidnet import PIDNet


class ModalityStem(nn.Sequential):
    """Two-layer quarter-resolution stem used by one input modality."""

    def __init__(self, input_channels: int, output_channels: int = 32) -> None:
        if input_channels <= 0 or output_channels <= 0:
            raise ValueError("Stem channel counts must be positive.")
        super().__init__(
            nn.Conv2d(
                input_channels,
                16,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(16, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                16,
                output_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(output_channels, momentum=0.1),
            nn.ReLU(inplace=True),
        )


class ModalityReliabilityGate(nn.Module):
    """Predict per-pixel RGB/thermal weights from paired stem features."""

    def __init__(self, feature_channels: int = 32, hidden_channels: int = 16) -> None:
        super().__init__()
        if feature_channels <= 0 or hidden_channels <= 0:
            raise ValueError("Gate channel counts must be positive.")
        self.context = nn.Sequential(
            nn.Conv2d(
                feature_channels * 3,
                hidden_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(hidden_channels, momentum=0.1),
            nn.ReLU(inplace=True),
        )
        self.logits = nn.Conv2d(
            hidden_channels,
            2,
            kernel_size=1,
            bias=True,
        )

    def forward(
        self,
        rgb_feature: torch.Tensor,
        thermal_feature: torch.Tensor,
    ) -> torch.Tensor:
        if rgb_feature.shape != thermal_feature.shape:
            raise ValueError(
                "RGB and thermal stem features must have identical shapes, got "
                f"{tuple(rgb_feature.shape)} and {tuple(thermal_feature.shape)}."
            )
        context = torch.cat(
            (rgb_feature, thermal_feature, (rgb_feature - thermal_feature).abs()),
            dim=1,
        )
        gate_logits = self.logits(self.context(context))
        # Compute Softmax in the active autocast dtype, then promote before the
        # complementary construction so exported weights sum exactly to one.
        # Feature fusion casts them back to the Stem dtype; training and inference
        # therefore use the same precision path.
        probabilities = torch.softmax(
            gate_logits,
            dim=1,
            dtype=gate_logits.dtype,
        )
        thermal_weight = probabilities[:, 1:2].float()
        rgb_weight = torch.ones_like(thermal_weight) - thermal_weight
        return torch.cat((rgb_weight, thermal_weight), dim=1)


class PIDNetMRFF(PIDNet):
    """PIDNet-S with modality-reliability feature fusion before layer1.

    RGB and radiometric-thermal inputs are encoded independently.  A lightweight
    gate predicts convex per-pixel modality weights.  The original PIDNet body is
    unchanged after the quarter-resolution fused feature.
    """

    def __init__(
        self,
        *args,
        thermal_channel: int = 3,
        gate_hidden_channels: int = 16,
        **kwargs,
    ) -> None:
        channels = int(kwargs.get("channels", 4))
        planes = int(kwargs.get("planes", 32))
        if channels != 4 or thermal_channel != 3:
            raise ValueError(
                "MRFF requires a four-channel Fusion tensor ordered as RGB then IR."
            )
        if planes != 32:
            raise ValueError("The preregistered PIDNet-S MRFF stem requires planes=32.")
        super().__init__(*args, **kwargs)

        # The inherited four-channel stem is not part of MRFF.  Identity lets the
        # unmodified official forward consume the already fused 32-channel feature.
        self.conv1 = nn.Identity()
        self.thermal_channel = int(thermal_channel)
        self.rgb_stem = ModalityStem(3, planes)
        self.thermal_stem = ModalityStem(1, planes)
        self.modality_gate = ModalityReliabilityGate(
            feature_channels=planes,
            hidden_channels=int(gate_hidden_channels),
        )
        self._initialize_mrff_modules()

    def _initialize_mrff_modules(self) -> None:
        for module in (self.rgb_stem, self.thermal_stem, self.modality_gate):
            for child in module.modules():
                if isinstance(child, nn.Conv2d):
                    nn.init.kaiming_normal_(
                        child.weight,
                        mode="fan_out",
                        nonlinearity="relu",
                    )
                    if child.bias is not None:
                        nn.init.zeros_(child.bias)
                elif isinstance(child, nn.BatchNorm2d):
                    nn.init.ones_(child.weight)
                    nn.init.zeros_(child.bias)
        # This is applied after the generic initialization so Softmax is exactly
        # 0.5/0.5 at step zero while the earlier gate remains trainable.
        nn.init.zeros_(self.modality_gate.logits.weight)
        nn.init.zeros_(self.modality_gate.logits.bias)

    def extract_modality_features(
        self,
        image: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if image.ndim != 4 or image.shape[1] != 4:
            raise ValueError(
                "MRFF expected BCHW Fusion input with four channels, got "
                f"{tuple(image.shape)}."
            )
        rgb = image[:, :3]
        thermal = image[:, self.thermal_channel : self.thermal_channel + 1]
        return self.rgb_stem(rgb), self.thermal_stem(thermal)

    def fuse_modalities(
        self,
        rgb_feature: torch.Tensor,
        thermal_feature: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        weights = self.modality_gate(rgb_feature, thermal_feature)
        feature_weights = weights.to(dtype=rgb_feature.dtype)
        fused = (
            feature_weights[:, 0:1] * rgb_feature
            + feature_weights[:, 1:2] * thermal_feature
        )
        return fused, weights

    def forward(self, image: torch.Tensor, return_aux: bool = False):
        rgb_feature, thermal_feature = self.extract_modality_features(image)
        fused, weights = self.fuse_modalities(rgb_feature, thermal_feature)
        outputs = super().forward(fused)
        if return_aux:
            return outputs, {"modality_weights": weights}
        return outputs


def count_mrff_modules(model: nn.Module) -> int:
    return sum(isinstance(module, PIDNetMRFF) for module in model.modules())


__all__ = [
    "ModalityReliabilityGate",
    "ModalityStem",
    "PIDNetMRFF",
    "count_mrff_modules",
]
