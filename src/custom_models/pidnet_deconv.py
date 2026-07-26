from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.pidnet import PIDNet


class DEConv2d(nn.Module):
    """DEA-Net style DEConv: vanilla + CDC + ADC + HDC + VDC.

    The five effective kernels are summed before a single spatial convolution.
    The vanilla parameter is named ``weight`` so PIDNet ImageNet checkpoints
    keep matching the original BasicBlock convolution keys.
    """

    angular_permutation = (3, 0, 1, 6, 4, 2, 7, 8, 5)

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 1,
        dilation: int | tuple[int, int] = 1,
        groups: int = 1,
    ) -> None:
        super().__init__()
        if in_channels % groups != 0 or out_channels % groups != 0:
            raise ValueError("DEConv channels must be divisible by groups.")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        input_per_group = in_channels // groups
        self.weight = nn.Parameter(
            torch.empty(out_channels, input_per_group, 3, 3)
        )
        self.weight_cd = nn.Parameter(torch.zeros_like(self.weight))
        self.weight_ad = nn.Parameter(torch.zeros_like(self.weight))
        self.weight_hd = nn.Parameter(
            torch.zeros(out_channels, input_per_group, 1, 3)
        )
        self.weight_vd = nn.Parameter(
            torch.zeros(out_channels, input_per_group, 3, 1)
        )
        nn.init.kaiming_normal_(self.weight, mode="fan_out", nonlinearity="relu")

    @classmethod
    def from_conv(cls, convolution: nn.Conv2d) -> "DEConv2d":
        if convolution.kernel_size != (3, 3) or convolution.bias is not None:
            raise ValueError("DEConv replacement requires a bias-free 3x3 Conv2d.")
        module = cls(
            in_channels=convolution.in_channels,
            out_channels=convolution.out_channels,
            stride=convolution.stride,
            padding=convolution.padding,
            dilation=convolution.dilation,
            groups=convolution.groups,
        )
        with torch.no_grad():
            module.weight.copy_(convolution.weight)
        return module

    def center_difference_kernel(self) -> torch.Tensor:
        kernel = self.weight_cd.reshape(
            self.out_channels,
            self.in_channels // self.groups,
            9,
        ).clone()
        kernel[:, :, 4] = kernel[:, :, 4] - kernel.sum(dim=2)
        return kernel.reshape_as(self.weight_cd)

    def angular_difference_kernel(self) -> torch.Tensor:
        kernel = self.weight_ad.reshape(
            self.out_channels,
            self.in_channels // self.groups,
            9,
        )
        rotated = kernel[:, :, self.angular_permutation]
        return (kernel - rotated).reshape_as(self.weight_ad)

    def horizontal_difference_kernel(self) -> torch.Tensor:
        kernel = self.weight.new_zeros(self.weight.shape)
        kernel[:, :, :, 0] = self.weight_hd[:, :, 0, :]
        kernel[:, :, :, 2] = -self.weight_hd[:, :, 0, :]
        return kernel

    def vertical_difference_kernel(self) -> torch.Tensor:
        kernel = self.weight.new_zeros(self.weight.shape)
        kernel[:, :, 0, :] = self.weight_vd[:, :, :, 0]
        kernel[:, :, 2, :] = -self.weight_vd[:, :, :, 0]
        return kernel

    def effective_kernel(self) -> torch.Tensor:
        return (
            self.weight
            + self.center_difference_kernel()
            + self.angular_difference_kernel()
            + self.horizontal_difference_kernel()
            + self.vertical_difference_kernel()
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.conv2d(
            inputs,
            self.effective_kernel(),
            bias=None,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )

    @torch.no_grad()
    def to_conv2d(self) -> nn.Conv2d:
        convolution = nn.Conv2d(
            self.in_channels,
            self.out_channels,
            kernel_size=3,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
            bias=False,
            device=self.weight.device,
            dtype=self.weight.dtype,
        )
        convolution.weight.copy_(self.effective_kernel())
        return convolution


def _replace_basicblock_convolutions(layer: nn.Module) -> int:
    replacements = 0
    for block in layer.modules():
        for attribute in ("conv1", "conv2"):
            convolution = getattr(block, attribute, None)
            if isinstance(convolution, nn.Conv2d) and convolution.kernel_size == (3, 3):
                setattr(block, attribute, DEConv2d.from_conv(convolution))
                replacements += 1
    return replacements


def apply_p_branch_deconv(model: PIDNet, variant: str) -> int:
    normalized = variant.upper()
    if normalized not in {"D1", "D2"}:
        raise ValueError(f"Unsupported DEConv variant: {variant}")
    replacements = 0
    if normalized == "D2":
        replacements += _replace_basicblock_convolutions(model.layer3_)
    replacements += _replace_basicblock_convolutions(model.layer4_)
    model.deconv_variant = normalized
    model.deconv_replacements = replacements
    return replacements


def reparameterize_deconv_model(
    model: nn.Module,
    inplace: bool = True,
) -> nn.Module:
    deployed = model if inplace else copy.deepcopy(model)

    def replace(parent: nn.Module) -> None:
        for name, child in list(parent.named_children()):
            if isinstance(child, DEConv2d):
                setattr(parent, name, child.to_conv2d())
            else:
                replace(child)

    replace(deployed)
    return deployed


def count_deconv_modules(model: nn.Module) -> int:
    return sum(isinstance(module, DEConv2d) for module in model.modules())


class PIDNetDEConv(PIDNet):
    """PIDNet-S with reparameterizable DEConv only in the P branch."""

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


__all__ = [
    "DEConv2d",
    "PIDNetDEConv",
    "apply_p_branch_deconv",
    "count_deconv_modules",
    "reparameterize_deconv_model",
]
