from __future__ import annotations

import hashlib
import importlib.util
import sys
import types
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.pidnet import PIDNet


PROJECT_ROOT = Path(__file__).resolve().parents[2]
UPSTREAM_REPOSITORY = PROJECT_ROOT / "third_party" / "FreqFusion"
UPSTREAM_SOURCE = (
    UPSTREAM_REPOSITORY
    / "SegNeXt"
    / "mmseg"
    / "models"
    / "decode_heads"
    / "FreqFusion.py"
)
UPSTREAM_LICENSE = UPSTREAM_REPOSITORY / "SegNeXt" / "LICENSE"
UPSTREAM_COMMIT = "3fb0c70637a3c194fb74294d3ce4681958b26241"
UPSTREAM_SOURCE_SHA256 = (
    "14a5af3a614a721cc01777102974e0cbfb4f29a19a45df8b883be7b0d1f38d91"
)
UPSTREAM_LICENSE_SHA256 = (
    "73cbe2ff56ba66bc9c3e4049253dac54618588883e9f188e6d07686e0e5e13fe"
)
UPSTREAM_MODULE_NAME = "pidnet_freqfusion_segnext_upstream"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pytorch_carafe(
    features: torch.Tensor,
    normalized_mask: torch.Tensor,
    kernel_size: int,
    group: int = 1,
    up: int = 1,
) -> torch.Tensor:
    """Independent PyTorch implementation of the public CARAFE formula."""
    if features.ndim != 4 or normalized_mask.ndim != 4:
        raise ValueError("CARAFE features and mask must both be 4D tensors.")
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError("CARAFE kernel_size must be a positive odd integer.")
    if group <= 0 or up <= 0:
        raise ValueError("CARAFE group and up must be positive integers.")

    batch, channels, height, width = features.shape
    if channels % group != 0:
        raise ValueError("CARAFE feature channels must be divisible by group.")
    output_height = height * up
    output_width = width * up
    expected_mask_channels = group * kernel_size * kernel_size
    if normalized_mask.shape != (
        batch,
        expected_mask_channels,
        output_height,
        output_width,
    ):
        raise ValueError(
            "CARAFE mask shape mismatch: "
            f"{tuple(normalized_mask.shape)} vs "
            f"{(batch, expected_mask_channels, output_height, output_width)}."
        )

    padding = kernel_size // 2
    padded = F.pad(features, [padding] * 4, mode="reflect")
    patches = F.unfold(padded, kernel_size=kernel_size)
    patches = patches.view(
        batch,
        group,
        channels // group,
        kernel_size * kernel_size,
        height,
        width,
    )
    if up != 1:
        patches = patches.reshape(
            batch,
            channels * kernel_size * kernel_size,
            height,
            width,
        )
        patches = F.interpolate(patches, scale_factor=up, mode="nearest")
        patches = patches.view(
            batch,
            group,
            channels // group,
            kernel_size * kernel_size,
            output_height,
            output_width,
        )
    mask = normalized_mask.view(
        batch,
        group,
        1,
        kernel_size * kernel_size,
        output_height,
        output_width,
    )
    return (patches * mask).sum(dim=3).reshape(
        batch,
        channels,
        output_height,
        output_width,
    )


def _xavier_init(
    module: nn.Module,
    gain: float = 1.0,
    bias: float = 0.0,
    distribution: str = "normal",
) -> None:
    if distribution not in {"uniform", "normal"}:
        raise ValueError(f"Unsupported Xavier distribution: {distribution}")
    if hasattr(module, "weight") and module.weight is not None:
        if distribution == "uniform":
            nn.init.xavier_uniform_(module.weight, gain=gain)
        else:
            nn.init.xavier_normal_(module.weight, gain=gain)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def _normal_init(
    module: nn.Module,
    mean: float = 0.0,
    std: float = 1.0,
    bias: float = 0.0,
) -> None:
    if hasattr(module, "weight") and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def _load_licensed_freqfusion_class():
    if not UPSTREAM_SOURCE.is_file():
        raise FileNotFoundError(
            "Licensed FreqFusion SegNeXt source is missing. Clone the pinned "
            f"official repository into: {UPSTREAM_REPOSITORY}"
        )
    if not UPSTREAM_LICENSE.is_file():
        raise FileNotFoundError(
            f"FreqFusion SegNeXt Apache-2.0 LICENSE is missing: {UPSTREAM_LICENSE}"
        )
    actual_source_hash = file_sha256(UPSTREAM_SOURCE)
    if actual_source_hash != UPSTREAM_SOURCE_SHA256:
        raise RuntimeError(
            "FreqFusion SegNeXt source hash mismatch: "
            f"{actual_source_hash} vs {UPSTREAM_SOURCE_SHA256}."
        )
    actual_license_hash = file_sha256(UPSTREAM_LICENSE)
    if actual_license_hash != UPSTREAM_LICENSE_SHA256:
        raise RuntimeError(
            "FreqFusion SegNeXt LICENSE hash mismatch: "
            f"{actual_license_hash} vs {UPSTREAM_LICENSE_SHA256}."
        )

    loaded = sys.modules.get(UPSTREAM_MODULE_NAME)
    if loaded is not None:
        return loaded.FreqFusion

    shim_modules = {
        "mmcv": types.ModuleType("mmcv"),
        "mmcv.ops": types.ModuleType("mmcv.ops"),
        "mmcv.ops.carafe": types.ModuleType("mmcv.ops.carafe"),
    }
    shim_modules["mmcv"].__path__ = []
    shim_modules["mmcv.ops"].__path__ = []
    shim_modules["mmcv"].ops = shim_modules["mmcv.ops"]
    shim_modules["mmcv.ops"].carafe = shim_modules["mmcv.ops.carafe"]
    shim_modules["mmcv.ops.carafe"].carafe = pytorch_carafe
    shim_modules["mmcv.ops.carafe"].normal_init = _normal_init
    shim_modules["mmcv.ops.carafe"].xavier_init = _xavier_init

    previous_modules = {
        name: sys.modules.get(name)
        for name in shim_modules
    }
    spec = importlib.util.spec_from_file_location(
        UPSTREAM_MODULE_NAME,
        UPSTREAM_SOURCE,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import licensed FreqFusion: {UPSTREAM_SOURCE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[UPSTREAM_MODULE_NAME] = module
    previous_dont_write_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        sys.modules.update(shim_modules)
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous_dont_write_bytecode
        for name, previous in previous_modules.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous
    return module.FreqFusion


OfficialFreqFusion = _load_licensed_freqfusion_class()


class PagFMFreqFusion(nn.Module):
    """FreqFusion-enhanced Pag3 while preserving the original PagFM gate."""

    def __init__(
        self,
        original: nn.Module,
        channels: int,
        compressed_channels: int = 16,
        feature_resample_group: int = 4,
    ) -> None:
        super().__init__()
        if compressed_channels % feature_resample_group != 0:
            raise ValueError(
                "FreqFusion compressed channels must be divisible by the "
                "feature-resample group count."
            )
        self.with_channel = original.with_channel
        self.after_relu = original.after_relu
        self.f_x = original.f_x
        self.f_y = original.f_y
        if hasattr(original, "up"):
            self.up = original.up
        if hasattr(original, "relu"):
            self.relu = original.relu
        self.freqfusion = OfficialFreqFusion(
            hr_channels=channels,
            lr_channels=channels,
            scale_factor=1,
            lowpass_kernel=5,
            highpass_kernel=3,
            up_group=1,
            encoder_kernel=3,
            encoder_dilation=1,
            compressed_channels=compressed_channels,
            align_corners=False,
            upsample_mode="nearest",
            feature_resample=True,
            feature_resample_group=feature_resample_group,
            comp_feat_upsample=True,
            use_high_pass=True,
            use_low_pass=True,
            hr_residual=True,
            semi_conv=True,
            hamming_window=False,
            feature_resample_norm=True,
        )
        self.compressed_channels = compressed_channels
        self.feature_resample_group = feature_resample_group

    def forward(self, detail: torch.Tensor, semantic: torch.Tensor) -> torch.Tensor:
        if detail.shape[1] != semantic.shape[1]:
            raise RuntimeError(
                "FreqFusion Pag3 requires equal detail/semantic channels: "
                f"{detail.shape[1]} vs {semantic.shape[1]}."
            )
        if detail.shape[-2] != semantic.shape[-2] * 2 or detail.shape[-1] != semantic.shape[-1] * 2:
            raise RuntimeError(
                "FreqFusion Pag3 requires a 2x spatial ratio: "
                f"{tuple(detail.shape[-2:])} vs {tuple(semantic.shape[-2:])}."
            )
        if self.after_relu:
            detail = self.relu(detail)
            semantic = self.relu(semantic)

        _, refined_detail, upsampled_semantic = self.freqfusion(
            hr_feat=detail,
            lr_feat=semantic,
        )
        if refined_detail.shape != upsampled_semantic.shape:
            raise RuntimeError(
                "FreqFusion Pag3 outputs must have identical shapes: "
                f"{tuple(refined_detail.shape)} vs "
                f"{tuple(upsampled_semantic.shape)}."
            )

        semantic_query = self.f_y(upsampled_semantic)
        detail_key = self.f_x(refined_detail)
        if self.with_channel:
            similarity = torch.sigmoid(self.up(detail_key * semantic_query))
        else:
            similarity = torch.sigmoid(
                torch.sum(detail_key * semantic_query, dim=1).unsqueeze(1)
            )
        return (
            (1.0 - similarity) * refined_detail
            + similarity * upsampled_semantic
        )


class PIDNetFreqFusion(PIDNet):
    """PIDNet-S with one licensed FreqFusion module at Pag3."""

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
        freqfusion_variant: str = "pag3",
        compressed_channels: int = 16,
        feature_resample_group: int = 4,
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
        variant = freqfusion_variant.lower()
        if variant != "pag3":
            raise ValueError(f"Unsupported FreqFusion variant: {freqfusion_variant}")
        self.pag3 = PagFMFreqFusion(
            self.pag3,
            channels=planes * 2,
            compressed_channels=compressed_channels,
            feature_resample_group=feature_resample_group,
        )
        self.freqfusion_variant = variant


def count_freqfusion_modules(model: nn.Module) -> int:
    return sum(
        isinstance(module, OfficialFreqFusion)
        for module in model.modules()
    )


__all__ = [
    "OfficialFreqFusion",
    "PIDNetFreqFusion",
    "PagFMFreqFusion",
    "UPSTREAM_COMMIT",
    "UPSTREAM_LICENSE",
    "UPSTREAM_LICENSE_SHA256",
    "UPSTREAM_REPOSITORY",
    "UPSTREAM_SOURCE",
    "UPSTREAM_SOURCE_SHA256",
    "count_freqfusion_modules",
    "file_sha256",
    "pytorch_carafe",
]
