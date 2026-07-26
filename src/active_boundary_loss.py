from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_REPOSITORY = PROJECT_ROOT / "third_party" / "active-boundary-loss"
UPSTREAM_SOURCE = UPSTREAM_REPOSITORY / "abl.py"
UPSTREAM_LICENSE = UPSTREAM_REPOSITORY / "LICENSE"
UPSTREAM_COMMIT = "1511507533ad98f04ea26e3648360a6c1d477d37"
UPSTREAM_SOURCE_SHA256 = (
    "07fc49d923a2420db7316eeb1650e95b2ea415bc85233937df99b239e3c6fe87"
)
UPSTREAM_LICENSE_SHA256 = (
    "1eb85fc97224598dad1852b5d6483bbcf0aa8608790dcc657a5a2a761ae9c8c6"
)


def kl_divergence(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    """Match the official source's KL(second || first) convention."""
    return F.softmax(second, dim=1) * (
        F.log_softmax(second, dim=1) - F.log_softmax(first, dim=1)
    )


def official_label_smoothing_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    smoothing: float,
) -> torch.Tensor:
    """Reproduce the exact LSSCE-V1 target weights linked by the ABL repo."""
    if logits.ndim != 2:
        raise ValueError("Direction logits must have shape [pixels, directions].")
    if labels.ndim != 1 or labels.shape[0] != logits.shape[0]:
        raise ValueError("Direction labels must have shape [pixels].")
    num_classes = logits.shape[1]
    with torch.no_grad():
        positive = 1.0 - smoothing
        negative = smoothing / num_classes
        targets = torch.full_like(logits, negative)
        targets.scatter_(1, labels.unsqueeze(1), positive)
    return -(F.log_softmax(logits.float(), dim=1) * targets).sum(dim=1)


class ActiveBoundaryLoss(nn.Module):
    """Device-safe ABL adaptation based on the official Apache-2.0 source.

    The selected ``per_image`` threshold scope follows the paper statement
    that predicted boundary pixels are capped at 1% per input image. The
    official source's batch-global behavior remains available as
    ``source_batch`` for numerical provenance checks.
    """

    def __init__(
        self,
        ignore_label: int = 255,
        detach_neighbors: bool = True,
        max_boundary_ratio: float = 0.01,
        label_smoothing: float = 0.2,
        max_clip_distance: float = 20.0,
        threshold_scope: str = "per_image",
    ) -> None:
        super().__init__()
        if not 0.0 < max_boundary_ratio <= 1.0:
            raise ValueError("max_boundary_ratio must be in (0, 1].")
        if not 0.0 <= label_smoothing < 1.0:
            raise ValueError("label_smoothing must be in [0, 1).")
        if max_clip_distance <= 0.0:
            raise ValueError("max_clip_distance must be positive.")
        if threshold_scope not in {"per_image", "source_batch"}:
            raise ValueError(
                "threshold_scope must be 'per_image' or 'source_batch'."
            )
        self.ignore_label = int(ignore_label)
        self.detach_neighbors = bool(detach_neighbors)
        self.max_boundary_ratio = float(max_boundary_ratio)
        self.label_smoothing = float(label_smoothing)
        self.max_clip_distance = float(max_clip_distance)
        self.threshold_scope = threshold_scope

    @staticmethod
    def ground_truth_boundary(
        target: torch.Tensor,
        ignore_label: int,
    ) -> torch.Tensor:
        if target.ndim != 3:
            raise ValueError("ABL target must have shape [batch, height, width].")
        vertical = target[:, 1:, :] - target[:, :-1, :]
        horizontal = target[:, :, 1:] - target[:, :, :-1]
        vertical = F.pad(vertical, (0, 0, 0, 1), value=0) != 0
        horizontal = F.pad(horizontal, (0, 1, 0, 0), value=0) != 0
        boundary = vertical | horizontal
        boundary |= target == ignore_label
        return boundary

    @staticmethod
    def boundary_distance_maps(boundary: torch.Tensor) -> torch.Tensor:
        maps = []
        for sample in boundary.detach().cpu().numpy().astype(bool):
            if not sample.any():
                maps.append(np.zeros(sample.shape, dtype=np.float32))
                continue
            distance = distance_transform_edt(~sample).astype(np.float32)
            # The pinned source builds ``one_hot2dist`` with an integer
            # ``zeros_like`` buffer, so Euclidean distances are truncated
            # before conversion back to FP32. Preserve that numerical behavior.
            maps.append(np.floor(np.maximum(distance - 1.0, 0.0)))
        return torch.from_numpy(np.stack(maps, axis=0)).to(
            device=boundary.device,
            dtype=torch.float32,
        )

    def adaptive_boundary_seed(self, logits: torch.Tensor) -> torch.Tensor:
        """Return the pre-dilation PDB mask used by the adaptive 1% cap."""
        _, _, height, width = logits.shape
        vertical = kl_divergence(
            logits[:, :, 1:, :],
            logits[:, :, :-1, :],
        ).sum(1, keepdim=True)
        horizontal = kl_divergence(
            logits[:, :, :, 1:],
            logits[:, :, :, :-1],
        ).sum(1, keepdim=True)
        vertical = F.pad(vertical, (0, 0, 0, 1), value=0)
        horizontal = F.pad(horizontal, (0, 1, 0, 0), value=0)
        combined = vertical + horizontal
        threshold = 1e-5
        maximum = height * width * self.max_boundary_ratio
        for _ in range(256):
            boundary = combined > threshold
            if int(boundary.sum()) <= maximum:
                break
            threshold *= 1.2
        else:
            raise RuntimeError("ABL adaptive boundary threshold did not converge.")
        return boundary[:, 0]

    @staticmethod
    def dilate_boundary(boundary: torch.Tensor) -> torch.Tensor:
        if boundary.ndim != 3:
            raise ValueError("Boundary masks must have shape [batch, height, width].")
        kernel = torch.ones(
            (1, 1, 3, 3),
            dtype=torch.float32,
            device=boundary.device,
        )
        dilated = F.conv2d(
            boundary.unsqueeze(1).to(dtype=torch.float32),
            kernel,
            stride=1,
            padding=1,
        )
        return dilated[:, 0] > 0

    def predicted_boundary(self, logits: torch.Tensor) -> torch.Tensor:
        if self.threshold_scope == "source_batch":
            return self.dilate_boundary(self.adaptive_boundary_seed(logits))
        return torch.cat(
            [
                self.dilate_boundary(
                    self.adaptive_boundary_seed(logits[index : index + 1])
                )
                for index in range(logits.shape[0])
            ],
            dim=0,
        )

    def direction_targets_and_logits(
        self,
        distance_maps: torch.Tensor,
        predicted_boundary: torch.Tensor,
        logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        locations = torch.nonzero(predicted_boundary, as_tuple=False)
        if locations.numel() == 0:
            empty_target = torch.empty(
                0,
                dtype=torch.long,
                device=logits.device,
            )
            empty_logits = torch.empty(
                0,
                8,
                dtype=torch.float32,
                device=logits.device,
            )
            empty_weight = torch.empty(
                0,
                dtype=torch.float32,
                device=logits.device,
            )
            return empty_target, empty_logits, empty_weight

        batch_index, row, column = locations.unbind(dim=1)
        maximum_distance = 1e5
        padded_distance = F.pad(
            distance_maps,
            (1, 1, 1, 1),
            mode="constant",
            value=maximum_distance,
        )
        padded_logits = F.pad(
            logits,
            (1, 1, 1, 1),
            mode="replicate",
        ).permute(0, 2, 3, 1)
        logits_nhwc = logits.permute(0, 2, 3, 1)
        center_logits = logits_nhwc[batch_index, row, column]
        row_offsets = (1, -1, 0, 0, -1, 1, -1, 1, 0)
        column_offsets = (0, 0, -1, 1, 1, 1, -1, -1, 0)
        local_distances = []
        local_kl = []
        for row_offset, column_offset in zip(row_offsets, column_offsets):
            neighbour_row = row + row_offset + 1
            neighbour_column = column + column_offset + 1
            local_distances.append(
                padded_distance[
                    batch_index,
                    neighbour_row,
                    neighbour_column,
                ]
            )
            if row_offset == 0 and column_offset == 0:
                continue
            neighbour_logits = padded_logits[
                batch_index,
                neighbour_row,
                neighbour_column,
            ]
            if self.detach_neighbors:
                neighbour_logits = neighbour_logits.detach()
            local_kl.append(
                kl_divergence(center_logits, neighbour_logits).sum(dim=1)
            )

        distance_matrix = torch.stack(local_distances, dim=0)
        direction_target = torch.argmin(distance_matrix, dim=0)
        keep = direction_target != 8
        direction_logits = torch.stack(local_kl, dim=1)
        weights = distance_maps[batch_index, row, column]
        return (
            direction_target[keep],
            direction_logits[keep],
            weights[keep],
        )

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if logits.ndim != 4 or target.ndim != 3:
            raise ValueError(
                "ABL expects logits [N,C,H,W] and target [N,H,W]."
            )
        if logits.shape[0] != target.shape[0]:
            raise ValueError("ABL logits and target batch sizes differ.")
        if logits.shape[-2:] != target.shape[-2:]:
            logits = F.interpolate(
                logits,
                size=target.shape[-2:],
                mode="bilinear",
                align_corners=True,
            )
        device_type = logits.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            logits_fp32 = logits.float()
            target_long = target.long()
            gt_boundary = self.ground_truth_boundary(
                target_long,
                self.ignore_label,
            )
            distance_maps = self.boundary_distance_maps(gt_boundary)
            pred_boundary = self.predicted_boundary(logits_fp32)
            (
                direction_target,
                direction_logits,
                distance_weights,
            ) = self.direction_targets_and_logits(
                distance_maps,
                pred_boundary,
                logits_fp32,
            )
            zero = logits_fp32.sum() * 0.0
            predicted_count = pred_boundary.sum().to(dtype=torch.float32)
            supervised_count = torch.tensor(
                float(direction_target.numel()),
                device=logits.device,
            )
            if direction_target.numel() == 0:
                diagnostics = {
                    "predicted_boundary_pixels": predicted_count,
                    "supervised_boundary_pixels": supervised_count,
                    "mean_boundary_distance": zero.detach(),
                    "skipped": torch.ones((), device=logits.device),
                }
                return zero, diagnostics

            per_pixel = official_label_smoothing_cross_entropy(
                direction_logits,
                direction_target,
                self.label_smoothing,
            )
            normalized_weights = torch.clamp(
                distance_weights,
                max=self.max_clip_distance,
            ) / self.max_clip_distance
            loss = (per_pixel * normalized_weights).mean()
            diagnostics = {
                "predicted_boundary_pixels": predicted_count,
                "supervised_boundary_pixels": supervised_count,
                "mean_boundary_distance": distance_weights.mean().detach(),
                "skipped": torch.zeros((), device=logits.device),
            }
            return loss, diagnostics


__all__ = [
    "ActiveBoundaryLoss",
    "UPSTREAM_COMMIT",
    "UPSTREAM_LICENSE",
    "UPSTREAM_LICENSE_SHA256",
    "UPSTREAM_REPOSITORY",
    "UPSTREAM_SOURCE",
    "UPSTREAM_SOURCE_SHA256",
    "kl_divergence",
    "official_label_smoothing_cross_entropy",
]
