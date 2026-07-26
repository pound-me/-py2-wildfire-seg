from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.pidnet import PIDNet


PROJECT_ROOT = Path(__file__).resolve().parents[2]
UPSTREAM_SOURCE = PROJECT_ROOT / "third_party" / "dysample" / "dysample.py"
UPSTREAM_MODULE_NAME = "pidnet_official_dysample_upstream"


def _load_official_dysample_class():
    if not UPSTREAM_SOURCE.is_file():
        raise FileNotFoundError(
            f"Pinned official DySample source not found: {UPSTREAM_SOURCE}"
        )
    module = sys.modules.get(UPSTREAM_MODULE_NAME)
    if module is None:
        spec = importlib.util.spec_from_file_location(
            UPSTREAM_MODULE_NAME,
            UPSTREAM_SOURCE,
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot import official DySample: {UPSTREAM_SOURCE}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[UPSTREAM_MODULE_NAME] = module
        previous_dont_write_bytecode = sys.dont_write_bytecode
        sys.dont_write_bytecode = True
        try:
            spec.loader.exec_module(module)
        finally:
            sys.dont_write_bytecode = previous_dont_write_bytecode
    return module.DySample


OfficialDySample = _load_official_dysample_class()


class PagFMDySample(nn.Module):
    """PagFM with only its late semantic upsampling replaced by DySample."""

    def __init__(self, original: nn.Module, scale: int = 4) -> None:
        super().__init__()
        self.with_channel = original.with_channel
        self.after_relu = original.after_relu
        self.f_x = original.f_x
        self.f_y = original.f_y
        if hasattr(original, "up"):
            self.up = original.up
        if hasattr(original, "relu"):
            self.relu = original.relu

        value_channels = self.f_y[0].in_channels
        query_channels = self.f_y[0].out_channels
        self.query_upsampler = OfficialDySample(
            query_channels,
            scale=scale,
            style="lp",
            groups=4,
            dyscope=False,
        )
        self.value_upsampler = OfficialDySample(
            value_channels,
            scale=scale,
            style="lp",
            groups=4,
            dyscope=False,
        )

    @staticmethod
    def _require_shape(tensor: torch.Tensor, target: tuple[int, int]) -> None:
        if tensor.shape[-2:] != target:
            raise RuntimeError(
                "DySample Pag4 output does not match the P-branch size: "
                f"{tensor.shape[-2:]} vs {target}."
            )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        target = (x.shape[-2], x.shape[-1])
        if self.after_relu:
            y = self.relu(y)
            x = self.relu(x)

        y_query = self.query_upsampler(self.f_y(y))
        self._require_shape(y_query, target)
        x_key = self.f_x(x)
        if self.with_channel:
            similarity = torch.sigmoid(self.up(x_key * y_query))
        else:
            similarity = torch.sigmoid(
                torch.sum(x_key * y_query, dim=1).unsqueeze(1)
            )

        y_value = self.value_upsampler(y)
        self._require_shape(y_value, target)
        return (1.0 - similarity) * x + similarity * y_value


class PIDNetDySample(PIDNet):
    """PIDNet-S with one theory-selected official DySample insertion."""

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
        dysample_variant: str = "pag4",
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
        variant = dysample_variant.lower()
        if variant == "pag4":
            self.pag4 = PagFMDySample(self.pag4, scale=4)
            self.context_upsampler = None
        elif variant == "context":
            self.context_upsampler = OfficialDySample(
                planes * 4,
                scale=8,
                style="lp",
                groups=4,
                dyscope=False,
            )
        else:
            raise ValueError(f"Unsupported DySample variant: {dysample_variant}")
        self.dysample_variant = variant

    @staticmethod
    def _require_context_shape(
        tensor: torch.Tensor,
        target: tuple[int, int],
    ) -> None:
        if tensor.shape[-2:] != target:
            raise RuntimeError(
                "DySample context output does not match DFM resolution: "
                f"{tensor.shape[-2:]} vs {target}."
            )

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
        context_feature = self.spp(self.layer5(feature))
        if self.dysample_variant == "context":
            context_feature = self.context_upsampler(context_feature)
            self._require_context_shape(context_feature, output_size)
        else:
            context_feature = F.interpolate(
                context_feature,
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


def count_dysample_modules(model: nn.Module) -> int:
    return sum(isinstance(module, OfficialDySample) for module in model.modules())


__all__ = [
    "OfficialDySample",
    "PIDNetDySample",
    "PagFMDySample",
    "UPSTREAM_SOURCE",
    "count_dysample_modules",
]
