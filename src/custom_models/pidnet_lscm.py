from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.pidnet import PIDNet


class LightweightSmokeContext(nn.Module):
    """Lightweight multi-scale context and spatial gating for diffuse smoke."""

    def __init__(
        self,
        channels: int,
        reduction: int = 4,
        dilations: tuple[int, ...] = (1, 2, 3),
    ) -> None:
        super().__init__()
        hidden_channels = max(channels // reduction, 16)
        self.reduce = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_channels, momentum=0.1),
            nn.ReLU(inplace=True),
        )
        self.context_branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        hidden_channels,
                        hidden_channels,
                        kernel_size=3,
                        padding=dilation,
                        dilation=dilation,
                        groups=hidden_channels,
                        bias=False,
                    ),
                    nn.BatchNorm2d(hidden_channels, momentum=0.1),
                    nn.ReLU(inplace=True),
                )
                for dilation in dilations
            ]
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(
                hidden_channels * len(dilations),
                channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(channels, momentum=0.1),
            nn.ReLU(inplace=True),
        )
        self.smoke_gate = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.residual_scale = nn.Parameter(torch.tensor(0.1))
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )
                if module.bias is not None:
                    nn.init.constant_(module.bias, -1.0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0.0)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        reduced = self.reduce(feature)
        contexts = [branch(reduced) for branch in self.context_branches]
        context = self.fuse(torch.cat(contexts, dim=1))
        gate = self.smoke_gate(context)
        return feature + self.residual_scale * gate * context


class PIDNetLSCM(PIDNet):
    """PIDNet-S with smoke context enhancement in the I branch."""

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
        context_channels = planes * 4
        self.smoke_context = LightweightSmokeContext(context_channels)

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
        context_feature = self.smoke_context(context_feature)

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
            return [detail_output, segmentation, boundary_output]
        return segmentation


class LightweightSmokeContextV2(LightweightSmokeContext):
    """LSCM with an explicitly supervised smoke-logit prediction."""

    def __init__(
        self,
        channels: int,
        reduction: int = 4,
        dilations: tuple[int, ...] = (1, 2, 3),
    ) -> None:
        super().__init__(
            channels=channels,
            reduction=reduction,
            dilations=dilations,
        )
        # A one-channel 1x1 head is unstable with fan-out Kaiming
        # initialization. Start from a mild, spatially uniform gate and let
        # the binary smoke supervision learn its spatial structure.
        nn.init.zeros_(self.smoke_gate[0].weight)
        if self.smoke_gate[0].bias is not None:
            nn.init.constant_(self.smoke_gate[0].bias, -1.0)

    def forward(
        self,
        feature: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        reduced = self.reduce(feature)
        contexts = [branch(reduced) for branch in self.context_branches]
        context = self.fuse(torch.cat(contexts, dim=1))
        smoke_logits = self.smoke_gate[0](context)
        gate = torch.sigmoid(smoke_logits)
        enhanced = feature + self.residual_scale * gate * context
        return enhanced, smoke_logits


class PIDNetLSCMV2(PIDNetLSCM):
    """PIDNet-S LSCM v2 with training-only binary smoke supervision."""

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
        self.smoke_context = LightweightSmokeContextV2(planes * 4)

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
        context_feature, smoke_logits = self.smoke_context(context_feature)

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
                smoke_logits,
            ]
        return segmentation
