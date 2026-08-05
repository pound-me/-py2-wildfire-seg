from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from baseline_runtime import (
    PROJECT_ROOT,
    TotalLoss,
    build_dataset,
    build_model,
    load_config,
    load_pretrained_if_available,
    seed_everything,
)
from custom_losses import build_training_criterion


def create_cuda_grad_scaler(enabled: bool, init_scale: float):
    amp_namespace = getattr(torch, "amp", None)
    scaler_class = getattr(amp_namespace, "GradScaler", None)
    if scaler_class is not None:
        try:
            return scaler_class("cuda", enabled=enabled, init_scale=init_scale)
        except TypeError:
            pass
    return torch.cuda.amp.GradScaler(enabled=enabled, init_scale=init_scale)


def collect_model_diagnostics(model: torch.nn.Module) -> dict | None:
    core_model = model.module if hasattr(model, "module") else model
    if hasattr(core_model, "erctc_region_scale") and hasattr(
        core_model, "erctc_frontier_scale"
    ):
        return {
            "module": "erctc",
            "region_scale": float(
                core_model.erctc_region_scale.detach().float().cpu()
            ),
            "frontier_scale": float(
                core_model.erctc_frontier_scale.detach().float().cpu()
            ),
        }
    return None


def model_uses_modality_aux(config: dict) -> bool:
    return str(config.get("MODEL", "")) == "pidnet_s_mrff"


def extract_main_logits(outputs) -> torch.Tensor:
    if isinstance(outputs, (list, tuple)):
        if len(outputs) < 2:
            raise RuntimeError("PIDNet training outputs do not contain main logits.")
        return outputs[1]
    if isinstance(outputs, torch.Tensor):
        return outputs
    raise TypeError(f"Unsupported model output type: {type(outputs)!r}")


def nts_weight_for_step(
    config: dict,
    epoch: int,
    batch_index: int,
    epoch_batches: int,
    training: bool,
) -> float:
    if not bool(config.get("MRFF_NTS_ENABLED", False)):
        return 0.0
    target = float(config.get("MRFF_NTS_WEIGHT", 0.02))
    if target <= 0.0:
        raise ValueError("MRFF_NTS_WEIGHT must be positive when NTS is enabled.")
    if epoch <= 0:
        return 0.0
    if epoch == 1:
        if not training:
            return target
        return target * min((batch_index + 1) / max(epoch_batches, 1), 1.0)
    return target


def compute_nts_loss(
    main_logits: torch.Tensor,
    modality_weights: torch.Tensor,
    fire_folder_flags: torch.Tensor,
    fire_class: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Suppress thermal reliance only at No-Fire pixels predicted as Fire."""
    if modality_weights.ndim != 4 or modality_weights.shape[1] != 2:
        raise ValueError(
            "MRFF modality_weights must be [batch, 2, height, width], got "
            f"{tuple(modality_weights.shape)}."
        )
    if main_logits.ndim != 4 or main_logits.shape[0] != modality_weights.shape[0]:
        raise ValueError("Main logits and MRFF weights use incompatible batches.")
    flags = fire_folder_flags.reshape(-1).to(
        device=main_logits.device,
        dtype=torch.bool,
    )
    if flags.numel() != main_logits.shape[0]:
        raise ValueError("Fire-folder flags do not match the MRFF batch size.")

    probabilities = torch.softmax(main_logits.float(), dim=1)
    p_fire = probabilities[:, fire_class]
    predicted_fire = main_logits.argmax(dim=1).eq(fire_class)
    thermal_weight = F.interpolate(
        modality_weights[:, 1:2].float(),
        size=main_logits.shape[-2:],
        mode="bilinear",
        align_corners=False,
    ).squeeze(1)
    no_fire_images = (~flags).view(-1, 1, 1)
    selected = no_fire_images & predicted_fire
    zero = thermal_weight.sum() * 0.0
    loss = (
        (p_fire.detach() * thermal_weight)[selected].mean()
        if bool(selected.any())
        else zero
    )
    return loss, selected.sum().to(dtype=torch.float32)


class ModalityGateAccumulator:
    """Streaming MRFF gate statistics with fixed-bin distributions."""

    def __init__(self, bins: int = 20) -> None:
        if bins <= 0:
            raise ValueError("Gate histogram bins must be positive.")
        self.bins = int(bins)
        self.count = [0, 0]
        self.total: list[torch.Tensor | None] = [None, None]
        self.square_total: list[torch.Tensor | None] = [None, None]
        self.histograms = [torch.zeros(self.bins, dtype=torch.int64) for _ in range(2)]
        self.maximum_sum_error = 0.0
        self.thermal_groups = {
            "fire_core": {"count": 0, "total": None},
            "no_fire": {"count": 0, "total": None},
            "fire_file_noncore": {"count": 0, "total": None},
        }

    def update(
        self,
        weights: torch.Tensor,
        labels: torch.Tensor,
        fire_folder_flags: torch.Tensor,
        fire_class: int,
        background_class: int,
        ignore_label: int,
    ) -> None:
        weights_fp32 = weights.detach().float()
        if weights_fp32.ndim != 4 or weights_fp32.shape[1] != 2:
            raise ValueError("MRFF gate weights must have shape [B,2,H,W].")
        sum_error = float(
            (weights_fp32.sum(dim=1) - 1.0).abs().max().cpu()
        )
        self.maximum_sum_error = max(self.maximum_sum_error, sum_error)
        if sum_error > 1e-5:
            raise RuntimeError(f"MRFF modality weights do not sum to one: {sum_error}")
        if self.histograms[0].device != weights_fp32.device:
            self.histograms = [
                histogram.to(device=weights_fp32.device)
                for histogram in self.histograms
            ]

        resized_labels = F.interpolate(
            labels.unsqueeze(1).float(),
            size=weights_fp32.shape[-2:],
            mode="nearest",
        )[:, 0].long()
        valid = resized_labels.ne(ignore_label)
        for modality in range(2):
            values = weights_fp32[:, modality][valid]
            if values.numel() == 0:
                continue
            self.count[modality] += int(values.numel())
            value_sum = values.sum()
            square_sum = values.square().sum()
            self.total[modality] = (
                value_sum
                if self.total[modality] is None
                else self.total[modality] + value_sum
            )
            self.square_total[modality] = (
                square_sum
                if self.square_total[modality] is None
                else self.square_total[modality] + square_sum
            )
            histogram = torch.histc(
                values,
                bins=self.bins,
                min=0.0,
                max=1.0,
            ).to(dtype=torch.int64)
            self.histograms[modality] += histogram

        flags = fire_folder_flags.reshape(-1).to(
            device=labels.device,
            dtype=torch.bool,
        )
        if flags.numel() != labels.shape[0]:
            raise ValueError("Fire-folder flags do not match gate statistics batch.")
        thermal = weights_fp32[:, 1]
        group_masks = {
            "fire_core": resized_labels.eq(fire_class) & valid,
            "no_fire": (~flags).view(-1, 1, 1) & valid,
            "fire_file_noncore": (
                flags.view(-1, 1, 1)
                & resized_labels.eq(background_class)
                & valid
            ),
        }
        for name, mask in group_masks.items():
            values = thermal[mask]
            self.thermal_groups[name]["count"] += int(values.numel())
            value_sum = values.sum()
            existing = self.thermal_groups[name]["total"]
            self.thermal_groups[name]["total"] = (
                value_sum if existing is None else existing + value_sum
            )

    def _quantile_from_histogram(self, modality: int, quantile: float) -> float:
        histogram = self.histograms[modality]
        count = int(histogram.sum())
        if count == 0:
            return 0.0
        target = max(int(np.ceil(quantile * count)), 1)
        index = int(torch.searchsorted(histogram.cumsum(0), target).item())
        index = min(index, self.bins - 1)
        return (index + 0.5) / self.bins

    def finalize(self) -> dict:
        modality_names = ("rgb", "thermal")
        modalities: dict[str, dict] = {}
        for index, name in enumerate(modality_names):
            count = max(self.count[index], 1)
            total = (
                float(self.total[index].detach().cpu())
                if self.total[index] is not None
                else 0.0
            )
            square_total = (
                float(self.square_total[index].detach().cpu())
                if self.square_total[index] is not None
                else 0.0
            )
            mean = total / count
            variance = max(square_total / count - mean**2, 0.0)
            modalities[name] = {
                "count": self.count[index],
                "mean": mean,
                "std": variance**0.5,
                "quantiles_approx": {
                    "p05": self._quantile_from_histogram(index, 0.05),
                    "p25": self._quantile_from_histogram(index, 0.25),
                    "p50": self._quantile_from_histogram(index, 0.50),
                    "p75": self._quantile_from_histogram(index, 0.75),
                    "p95": self._quantile_from_histogram(index, 0.95),
                },
                "histogram_0_to_1_20_bins": self.histograms[index].cpu().tolist(),
            }
        thermal_groups = {
            name: {
                "count": values["count"],
                "mean": (
                    float(values["total"].detach().cpu())
                    if values["total"] is not None
                    else 0.0
                ) / max(values["count"], 1),
            }
            for name, values in self.thermal_groups.items()
        }
        thermal = modalities["thermal"]
        return {
            "modalities": modalities,
            "thermal_weight_by_region": thermal_groups,
            "weight_sum_max_abs_error": self.maximum_sum_error,
            "histogram_bin_edges": [
                index / self.bins for index in range(self.bins + 1)
            ],
            "collapse_diagnostic": {
                "all_rgb_like": (
                    thermal["mean"] < 0.05
                    and thermal["quantiles_approx"]["p95"] < 0.10
                ),
                "all_thermal_like": (
                    thermal["mean"] > 0.95
                    and thermal["quantiles_approx"]["p05"] > 0.90
                ),
            },
        }


def update_confusion_matrix(
    confusion: torch.Tensor,
    logits: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    ignore_label: int,
) -> None:
    if logits.shape[-2:] != labels.shape[-2:]:
        logits = F.interpolate(
            logits,
            size=labels.shape[-2:],
            mode="bilinear",
            align_corners=True,
        )
    predictions = logits.argmax(dim=1)
    valid = labels != ignore_label
    indices = labels[valid] * num_classes + predictions[valid]
    bins = torch.bincount(indices, minlength=num_classes**2)
    confusion += bins.reshape(num_classes, num_classes).cpu()


def metrics_from_confusion(confusion: torch.Tensor, class_names: list[str]) -> dict:
    matrix = confusion.to(torch.float64)
    true_positive = matrix.diag()
    false_positive = matrix.sum(dim=0) - true_positive
    false_negative = matrix.sum(dim=1) - true_positive
    denominator = true_positive + false_positive + false_negative
    iou = torch.where(
        denominator > 0,
        true_positive / denominator,
        torch.zeros_like(denominator),
    )
    dice_denominator = 2 * true_positive + false_positive + false_negative
    dice = torch.where(
        dice_denominator > 0,
        2 * true_positive / dice_denominator,
        torch.zeros_like(dice_denominator),
    )
    total = matrix.sum()
    pixel_accuracy = true_positive.sum() / total if total > 0 else torch.tensor(0.0)
    result = {
        "miou": float(iou.mean()),
        "mean_dice": float(dice.mean()),
        "pixel_accuracy": float(pixel_accuracy),
    }
    for index, name in enumerate(class_names):
        result[f"iou_{name}"] = float(iou[index])
        result[f"dice_{name}"] = float(dice[index])
        result[f"precision_{name}"] = float(
            true_positive[index] / (true_positive[index] + false_positive[index])
        ) if true_positive[index] + false_positive[index] > 0 else 0.0
        result[f"recall_{name}"] = float(
            true_positive[index] / (true_positive[index] + false_negative[index])
        ) if true_positive[index] + false_negative[index] > 0 else 0.0
    return result


def new_partial_fire_statistics() -> dict[str, int]:
    return {
        "true_positive": 0,
        "false_positive": 0,
        "false_negative": 0,
        "true_negative": 0,
        "empty_fire_valid_pixels": 0,
        "empty_fire_predicted_fire_pixels": 0,
        "no_fire_valid_pixels": 0,
        "no_fire_predicted_fire_pixels": 0,
        "partial_nonfire_pixels": 0,
        "partial_predicted_background_pixels": 0,
        "partial_predicted_smoke_pixels": 0,
        "partial_predicted_fire_pixels": 0,
        "fire_folder_images": 0,
        "empty_fire_folder_images": 0,
        "no_fire_images": 0,
    }


def update_partial_fire_statistics(
    statistics: dict[str, int],
    logits: torch.Tensor,
    labels: torch.Tensor,
    fire_folder_flags: torch.Tensor,
    fire_class: int,
    smoke_class: int,
    ignore_label: int,
) -> None:
    if logits.shape[-2:] != labels.shape[-2:]:
        logits = F.interpolate(
            logits,
            size=labels.shape[-2:],
            mode="bilinear",
            align_corners=True,
        )
    predictions = logits.argmax(dim=1)
    flags = fire_folder_flags.reshape(-1).to(
        device=labels.device, dtype=torch.bool
    )
    valid = labels != ignore_label
    target_fire = labels == fire_class
    predicted_fire = predictions == fire_class
    statistics["true_positive"] += int((target_fire & predicted_fire & valid).sum())
    statistics["false_positive"] += int((~target_fire & predicted_fire & valid).sum())
    statistics["false_negative"] += int((target_fire & ~predicted_fire & valid).sum())
    statistics["true_negative"] += int((~target_fire & ~predicted_fire & valid).sum())

    for index in range(labels.shape[0]):
        image_valid = valid[index]
        if bool(flags[index]):
            statistics["fire_folder_images"] += 1
            partial = labels[index].eq(0) & image_valid
            statistics["partial_nonfire_pixels"] += int(partial.sum())
            statistics["partial_predicted_background_pixels"] += int(
                (predictions[index].eq(0) & partial).sum()
            )
            statistics["partial_predicted_smoke_pixels"] += int(
                (predictions[index].eq(smoke_class) & partial).sum()
            )
            statistics["partial_predicted_fire_pixels"] += int(
                (predictions[index].eq(fire_class) & partial).sum()
            )
            if not bool((target_fire[index] & image_valid).any()):
                statistics["empty_fire_folder_images"] += 1
                statistics["empty_fire_valid_pixels"] += int(image_valid.sum())
                statistics["empty_fire_predicted_fire_pixels"] += int(
                    (predicted_fire[index] & image_valid).sum()
                )
        else:
            statistics["no_fire_images"] += 1
            statistics["no_fire_valid_pixels"] += int(image_valid.sum())
            statistics["no_fire_predicted_fire_pixels"] += int(
                (predicted_fire[index] & image_valid).sum()
            )


def finalize_partial_fire_statistics(statistics: dict[str, int]) -> dict:
    tp = statistics["true_positive"]
    fp = statistics["false_positive"]
    fn = statistics["false_negative"]
    tn = statistics["true_negative"]
    partial_pixels = statistics["partial_nonfire_pixels"]
    return {
        "metric_protocol": "flame3_active_fire_partial_label",
        "fire_iou": tp / max(tp + fp + fn, 1),
        "fire_precision": tp / max(tp + fp, 1),
        "fire_recall": tp / max(tp + fn, 1),
        "fire_f1": (2 * tp) / max(2 * tp + fp + fn, 1),
        "binary_accuracy": (tp + tn) / max(tp + fp + fn + tn, 1),
        "empty_fire_predicted_fire_ratio": statistics[
            "empty_fire_predicted_fire_pixels"
        ] / max(statistics["empty_fire_valid_pixels"], 1),
        "no_fire_predicted_fire_ratio": statistics[
            "no_fire_predicted_fire_pixels"
        ] / max(statistics["no_fire_valid_pixels"], 1),
        "partial_predicted_background_ratio": statistics[
            "partial_predicted_background_pixels"
        ] / max(partial_pixels, 1),
        "partial_predicted_smoke_ratio": statistics[
            "partial_predicted_smoke_pixels"
        ] / max(partial_pixels, 1),
        "partial_predicted_fire_ratio": statistics[
            "partial_predicted_fire_pixels"
        ] / max(partial_pixels, 1),
        **statistics,
    }


def update_fire_boundary_statistics(
    statistics: dict[str, int],
    logits: torch.Tensor,
    labels: torch.Tensor,
    fire_class: int,
    ignore_label: int,
    tolerance: int,
) -> None:
    if logits.shape[-2:] != labels.shape[-2:]:
        logits = F.interpolate(
            logits,
            size=labels.shape[-2:],
            mode="bilinear",
            align_corners=True,
        )
    predictions = logits.argmax(dim=1)
    valid = labels != ignore_label

    def boundary(mask: torch.Tensor) -> torch.Tensor:
        values = mask.unsqueeze(1).to(dtype=torch.float32)
        eroded = F.conv2d(
            values,
            torch.ones((1, 1, 3, 3), device=mask.device),
            padding=1,
        ) >= 9.0
        return mask & ~eroded[:, 0]

    predicted_boundary = boundary((predictions == fire_class) & valid)
    target_boundary = boundary((labels == fire_class) & valid)
    dilation_size = 2 * tolerance + 1
    dilated_prediction = F.max_pool2d(
        predicted_boundary.unsqueeze(1).float(),
        kernel_size=dilation_size,
        stride=1,
        padding=tolerance,
    )[:, 0] > 0
    dilated_target = F.max_pool2d(
        target_boundary.unsqueeze(1).float(),
        kernel_size=dilation_size,
        stride=1,
        padding=tolerance,
    )[:, 0] > 0
    statistics["matched_prediction"] += int(
        (predicted_boundary & dilated_target).sum()
    )
    statistics["prediction"] += int(predicted_boundary.sum())
    statistics["matched_target"] += int(
        (target_boundary & dilated_prediction).sum()
    )
    statistics["target"] += int(target_boundary.sum())


def finalize_fire_boundary_statistics(statistics: dict[str, int]) -> dict:
    precision = (
        statistics["matched_prediction"] / statistics["prediction"]
        if statistics["prediction"]
        else 0.0
    )
    recall = (
        statistics["matched_target"] / statistics["target"]
        if statistics["target"]
        else 0.0
    )
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )
    return {
        "boundary_precision_fire": precision,
        "boundary_recall_fire": recall,
        "boundary_f1_fire": f1,
        **statistics,
    }


def assert_component_alignment(
    labels: torch.Tensor,
    component_maps: torch.Tensor,
    fire_class: int,
) -> None:
    if labels.shape != component_maps.shape:
        raise RuntimeError(
            f"Component-map batch shape {tuple(component_maps.shape)} does not "
            f"match labels {tuple(labels.shape)}."
        )
    if not torch.equal(
        component_maps.to(dtype=torch.int32) > 0,
        labels == fire_class,
    ):
        raise RuntimeError(
            "Augmented component maps and Fire labels are not strictly aligned."
        )


def hash_files(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(str(path).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def set_polynomial_lr(
    optimizer: torch.optim.Optimizer,
    base_lr: float,
    current_step: int,
    total_steps: int,
    power: float = 0.9,
) -> float:
    progress = min(current_step / max(total_steps, 1), 1.0)
    learning_rate = max(base_lr * (1.0 - progress) ** power, 1e-8)
    for group in optimizer.param_groups:
        group["lr"] = learning_rate
    return learning_rate


def run_training_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: object,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    config: dict,
    epoch: int,
    lr_total_epochs: int,
    max_batches: int | None,
    use_amp: bool,
) -> tuple[dict, float, dict[str, float], dict | None, dict | None]:
    model.train()
    device = torch.device(config["DEVICE"])
    confusion = torch.zeros(
        config["NUM_CLASSES"], config["NUM_CLASSES"], dtype=torch.int64
    )
    partial_statistics = new_partial_fire_statistics()
    loss_sum = 0.0
    component_sums: dict[str, float] = {}
    boundary_statistics = {
        "matched_prediction": 0,
        "prediction": 0,
        "matched_target": 0,
        "target": 0,
    }
    completed_batches = 0
    epoch_batches = min(len(loader), max_batches) if max_batches else len(loader)
    total_steps = max(lr_total_epochs * epoch_batches, 1)
    gate_accumulator = (
        ModalityGateAccumulator()
        if model_uses_modality_aux(config)
        else None
    )
    if criterion.objective_name == "ema_mproto":
        criterion.begin_epoch()

    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        images, labels, edges = batch[0], batch[1], batch[2]
        fire_folder_flags = None
        if criterion.objective_name == "partial_label":
            if len(batch) < 5:
                raise RuntimeError(
                    "FLAME3 partial-label training requires Fire-folder flags at batch[4]."
                )
            fire_folder_flags = batch[4]
        sampling_labels = labels
        component_maps = None
        if criterion.objective_name == "ema_mproto":
            if len(batch) < 5:
                raise RuntimeError(
                    "EMA multi-prototype training requires a component map at batch[4]."
                )
            component_maps = batch[4]
            assert_component_alignment(
                sampling_labels,
                component_maps,
                int(config.get("FIRE_CLASS_INDEX", 2)),
            )
        images = images.to(device=device, dtype=torch.float, non_blocking=True)
        labels = labels.to(device=device, dtype=torch.long, non_blocking=True)
        edges = edges.to(device=device, dtype=torch.float, non_blocking=True)
        if fire_folder_flags is not None:
            fire_folder_flags = fire_folder_flags.to(
                device=device, dtype=torch.bool, non_blocking=True
            )
        global_step = epoch * epoch_batches + batch_index
        learning_rate = set_polynomial_lr(
            optimizer,
            config["LR"],
            global_step,
            total_steps,
        )

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=use_amp,
        ):
            modality_aux = None
            if gate_accumulator is not None:
                outputs, modality_aux = model(images, return_aux=True)
            else:
                outputs = model(images)
            if criterion.objective_name == "partial_label":
                losses, metric_outputs, _, loss_components = criterion.get_loss(
                    outputs,
                    labels,
                    edges,
                    fire_folder_flags=fire_folder_flags,
                )
            elif criterion.objective_name == "ema_mproto":
                losses, metric_outputs, _, loss_components = criterion.get_loss(
                    outputs,
                    labels,
                    edges,
                    component_maps=component_maps,
                    sampling_labels=sampling_labels,
                    epoch=epoch,
                    batch_index=batch_index,
                    epoch_batches=epoch_batches,
                    training=True,
                )
            else:
                losses, metric_outputs, _, loss_components = criterion.get_loss(
                    outputs,
                    labels,
                    edges,
                )
            if bool(config.get("MRFF_NTS_ENABLED", False)):
                if modality_aux is None or fire_folder_flags is None:
                    raise RuntimeError(
                        "NTS requires MRFF auxiliary weights and FLAME3 No-Fire flags."
                    )
                nts_loss, nts_selected_pixels = compute_nts_loss(
                    extract_main_logits(outputs),
                    modality_aux["modality_weights"],
                    fire_folder_flags,
                    int(config.get("FIRE_CLASS_INDEX", 2)),
                )
                nts_weight = nts_weight_for_step(
                    config,
                    epoch,
                    batch_index,
                    epoch_batches,
                    training=True,
                )
                weighted_nts = nts_loss * nts_weight
                losses = losses + weighted_nts
                loss_components = {
                    **loss_components,
                    "nts_loss": nts_loss,
                    "nts_weight": torch.tensor(
                        nts_weight,
                        device=losses.device,
                        dtype=torch.float32,
                    ),
                    "weighted_nts": weighted_nts,
                    "nts_selected_pixels": nts_selected_pixels,
                }
            loss = losses.mean()

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite training loss at epoch {epoch + 1}, "
                f"batch {batch_index + 1}: {loss.item()}"
            )
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if gate_accumulator is not None:
            if modality_aux is None or fire_folder_flags is None:
                raise RuntimeError("MRFF gate diagnostics require partial-label flags.")
            gate_accumulator.update(
                modality_aux["modality_weights"],
                labels,
                fire_folder_flags,
                int(config.get("FIRE_CLASS_INDEX", 2)),
                int(config.get("BACKGROUND_CLASS_INDEX", 0)),
                int(config["IGNORE_LABEL"]),
            )

        if criterion.objective_name == "partial_label":
            update_partial_fire_statistics(
                partial_statistics,
                metric_outputs[1].detach(),
                labels,
                fire_folder_flags,
                int(config.get("FIRE_CLASS_INDEX", 2)),
                int(config.get("SMOKE_CLASS_INDEX", 1)),
                int(config["IGNORE_LABEL"]),
            )
        else:
            update_confusion_matrix(
                confusion,
                metric_outputs[1].detach(),
                labels,
                config["NUM_CLASSES"],
                config["IGNORE_LABEL"],
            )
        update_fire_boundary_statistics(
            boundary_statistics,
            metric_outputs[1].detach(),
            labels,
            int(config.get("FIRE_CLASS_INDEX", 2)),
            int(config["IGNORE_LABEL"]),
            int(config.get("BOUNDARY_TOLERANCE", 3)),
        )
        loss_sum += float(loss.detach())
        for name, value in loss_components.items():
            component_sums[name] = component_sums.get(name, 0.0) + float(
                value.detach()
            )
        completed_batches += 1
        if completed_batches == 1 or completed_batches % 10 == 0:
            print(
                f"  train batch {completed_batches}/{epoch_batches}: "
                f"loss={float(loss.detach()):.5f}, lr={learning_rate:.8f}"
            )

    metrics = (
        finalize_partial_fire_statistics(partial_statistics)
        if criterion.objective_name == "partial_label"
        else metrics_from_confusion(confusion, config["CLS_NAMES"])
    )
    metrics.update(finalize_fire_boundary_statistics(boundary_statistics))
    denominator = max(completed_batches, 1)
    component_averages = {
        name: value / denominator for name, value in component_sums.items()
    }
    prototype_health = (
        criterion.finalize_epoch_health()
        if criterion.objective_name == "ema_mproto"
        else None
    )
    gate_statistics = (
        gate_accumulator.finalize()
        if gate_accumulator is not None
        else None
    )
    return (
        metrics,
        loss_sum / denominator,
        component_averages,
        prototype_health,
        gate_statistics,
    )


@torch.inference_mode()
def run_validation(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: object,
    config: dict,
    epoch: int,
    max_batches: int | None,
    use_amp: bool,
) -> tuple[dict, float, dict[str, float], dict | None]:
    model.eval()
    device = torch.device(config["DEVICE"])
    confusion = torch.zeros(
        config["NUM_CLASSES"], config["NUM_CLASSES"], dtype=torch.int64
    )
    partial_statistics = new_partial_fire_statistics()
    loss_sum = 0.0
    component_sums: dict[str, float] = {}
    boundary_statistics = {
        "matched_prediction": 0,
        "prediction": 0,
        "matched_target": 0,
        "target": 0,
    }
    completed_batches = 0
    gate_accumulator = (
        ModalityGateAccumulator()
        if model_uses_modality_aux(config)
        else None
    )

    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        images, labels, edges = batch[0], batch[1], batch[2]
        fire_folder_flags = None
        if criterion.objective_name == "partial_label":
            if len(batch) < 5:
                raise RuntimeError(
                    "FLAME3 partial-label validation requires Fire-folder flags at batch[4]."
                )
            fire_folder_flags = batch[4]
        sampling_labels = labels
        component_maps = None
        if criterion.objective_name == "ema_mproto":
            if len(batch) < 5:
                raise RuntimeError(
                    "EMA multi-prototype validation requires a component map at batch[4]."
                )
            component_maps = batch[4]
            assert_component_alignment(
                sampling_labels,
                component_maps,
                int(config.get("FIRE_CLASS_INDEX", 2)),
            )
        images = images.to(device=device, dtype=torch.float, non_blocking=True)
        labels = labels.to(device=device, dtype=torch.long, non_blocking=True)
        edges = edges.to(device=device, dtype=torch.float, non_blocking=True)
        if fire_folder_flags is not None:
            fire_folder_flags = fire_folder_flags.to(
                device=device, dtype=torch.bool, non_blocking=True
            )

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=use_amp,
        ):
            modality_aux = None
            if gate_accumulator is not None:
                outputs, modality_aux = model(images, return_aux=True)
            else:
                outputs = model(images)
            if criterion.objective_name == "partial_label":
                losses, metric_outputs, _, loss_components = criterion.get_loss(
                    outputs,
                    labels,
                    edges,
                    fire_folder_flags=fire_folder_flags,
                )
            elif criterion.objective_name == "ema_mproto":
                losses, metric_outputs, _, loss_components = criterion.get_loss(
                    outputs,
                    labels,
                    edges,
                    component_maps=component_maps,
                    sampling_labels=sampling_labels,
                    epoch=epoch,
                    batch_index=batch_index,
                    epoch_batches=len(loader),
                    training=False,
                )
            else:
                losses, metric_outputs, _, loss_components = criterion.get_loss(
                    outputs,
                    labels,
                    edges,
                )
            if bool(config.get("MRFF_NTS_ENABLED", False)):
                if modality_aux is None or fire_folder_flags is None:
                    raise RuntimeError(
                        "NTS validation requires MRFF weights and No-Fire flags."
                    )
                nts_loss, nts_selected_pixels = compute_nts_loss(
                    extract_main_logits(outputs),
                    modality_aux["modality_weights"],
                    fire_folder_flags,
                    int(config.get("FIRE_CLASS_INDEX", 2)),
                )
                nts_weight = nts_weight_for_step(
                    config,
                    epoch,
                    batch_index,
                    len(loader),
                    training=False,
                )
                weighted_nts = nts_loss * nts_weight
                losses = losses + weighted_nts
                loss_components = {
                    **loss_components,
                    "nts_loss": nts_loss,
                    "nts_weight": torch.tensor(
                        nts_weight,
                        device=losses.device,
                        dtype=torch.float32,
                    ),
                    "weighted_nts": weighted_nts,
                    "nts_selected_pixels": nts_selected_pixels,
                }
            loss = losses.mean()
        if gate_accumulator is not None:
            if modality_aux is None or fire_folder_flags is None:
                raise RuntimeError("MRFF gate diagnostics require partial-label flags.")
            gate_accumulator.update(
                modality_aux["modality_weights"],
                labels,
                fire_folder_flags,
                int(config.get("FIRE_CLASS_INDEX", 2)),
                int(config.get("BACKGROUND_CLASS_INDEX", 0)),
                int(config["IGNORE_LABEL"]),
            )
        if criterion.objective_name == "partial_label":
            update_partial_fire_statistics(
                partial_statistics,
                metric_outputs[1],
                labels,
                fire_folder_flags,
                int(config.get("FIRE_CLASS_INDEX", 2)),
                int(config.get("SMOKE_CLASS_INDEX", 1)),
                int(config["IGNORE_LABEL"]),
            )
        else:
            update_confusion_matrix(
                confusion,
                metric_outputs[1],
                labels,
                config["NUM_CLASSES"],
                config["IGNORE_LABEL"],
            )
        update_fire_boundary_statistics(
            boundary_statistics,
            metric_outputs[1],
            labels,
            int(config.get("FIRE_CLASS_INDEX", 2)),
            int(config["IGNORE_LABEL"]),
            int(config.get("BOUNDARY_TOLERANCE", 3)),
        )
        loss_sum += float(loss.detach())
        for name, value in loss_components.items():
            component_sums[name] = component_sums.get(name, 0.0) + float(
                value.detach()
            )
        completed_batches += 1

    metrics = (
        finalize_partial_fire_statistics(partial_statistics)
        if criterion.objective_name == "partial_label"
        else metrics_from_confusion(confusion, config["CLS_NAMES"])
    )
    metrics.update(finalize_fire_boundary_statistics(boundary_statistics))
    denominator = max(completed_batches, 1)
    component_averages = {
        name: value / denominator for name, value in component_sums.items()
    }
    gate_statistics = (
        gate_accumulator.finalize()
        if gate_accumulator is not None
        else None
    )
    return metrics, loss_sum / denominator, component_averages, gate_statistics


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    criterion: object,
    train_generator: torch.Generator,
    config: dict,
    epoch: int,
    validation_metrics: dict,
    best_selection_metric: float,
    selection_metric_name: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "criterion_state_dict": (
                criterion.state_dict()
                if hasattr(criterion, "state_dict")
                else None
            ),
            "train_generator_state": train_generator.get_state(),
            "python_random_state": __import__("random").getstate(),
            "numpy_random_state": np.random.get_state(),
            "torch_random_state": torch.get_rng_state(),
            "cuda_random_states": torch.cuda.get_rng_state_all(),
            "config": config,
            "validation_metrics": validation_metrics,
            "selection_metric_name": selection_metric_name,
            "best_selection_metric": best_selection_metric,
            "best_miou": (
                best_selection_metric if selection_metric_name == "miou" else -1.0
            ),
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the RGB PIDNet-S baseline.")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "pidnet_s_rgb_baseline.yaml",
    )
    parser.add_argument("--epochs", type=int)
    parser.add_argument(
        "--lr-total-epochs",
        type=int,
        help=(
            "Polynomial learning-rate horizon. Use 100 while training only "
            "10 screening epochs to match the first 10 epochs of a 100-epoch run."
        ),
    )
    parser.add_argument(
        "--smoke-aux-weight",
        type=float,
        help="Override SMOKE_AUX_WEIGHT for a controlled ablation run.",
    )
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-val-batches", type=int)
    parser.add_argument(
        "--device",
        help="Override DEVICE, for example cuda:0, cuda:1, or cuda:2.",
    )
    parser.add_argument(
        "--root-dataset",
        type=Path,
        help="Override ROOTDATASET, used by the portable FLAME3 bundle.",
    )
    parser.add_argument(
        "--pretrained",
        type=Path,
        help="Override PRETRAINED initialization path.",
    )
    parser.add_argument("--trainset", type=Path, help="Override TRAINSET path.")
    parser.add_argument("--validset", type=Path, help="Override VALIDSET path.")
    parser.add_argument("--batch-size", type=int, help="Override physical batch size.")
    parser.add_argument("--num-workers", type=int, help="Override data-loader workers.")
    parser.add_argument("--seed", type=int, help="Override the config seed.")
    parser.add_argument("--run-name", default="baseline")
    parser.add_argument(
        "--resume",
        type=Path,
        help="Resume model, optimizer, AMP, prototype bank, and RNG states.",
    )
    parser.add_argument("--amp", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config.resolve())
    if args.device:
        config["DEVICE"] = args.device
    if args.root_dataset is not None:
        config["ROOTDATASET"] = str(args.root_dataset.resolve())
    if args.pretrained is not None:
        config["PRETRAINED"] = str(args.pretrained.resolve())
    if args.trainset is not None:
        config["TRAINSET"] = str(args.trainset.resolve())
    if args.validset is not None:
        config["VALIDSET"] = str(args.validset.resolve())
    if args.batch_size is not None:
        if args.batch_size <= 0:
            raise ValueError("batch-size must be positive")
        config["BATCHSIZE"] = args.batch_size
    if args.num_workers is not None:
        if args.num_workers < 0:
            raise ValueError("num-workers must be non-negative")
        config["NUM_WORKERS"] = args.num_workers
    if args.seed is not None:
        config["SEED"] = args.seed
    epochs = args.epochs if args.epochs is not None else config["EPOCHS"]
    lr_total_epochs = (
        args.lr_total_epochs
        if args.lr_total_epochs is not None
        else config.get("LR_TOTAL_EPOCHS", epochs)
    )
    if epochs <= 0:
        raise ValueError("epochs must be positive.")
    if lr_total_epochs < epochs:
        raise ValueError("lr-total-epochs must be greater than or equal to epochs.")
    config["EPOCHS"] = epochs
    config["LR_TOTAL_EPOCHS"] = lr_total_epochs
    if args.smoke_aux_weight is not None:
        if args.smoke_aux_weight < 0.0:
            raise ValueError("smoke-aux-weight must be non-negative.")
        config["SMOKE_AUX_WEIGHT"] = args.smoke_aux_weight
    seed_everything(config["SEED"])
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable.")

    train_dataset = build_dataset(config, split="train")
    validation_dataset = build_dataset(config, split="val")
    train_generator = torch.Generator()
    train_generator.manual_seed(config["SEED"])
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["BATCHSIZE"],
        shuffle=True,
        num_workers=config["NUM_WORKERS"],
        pin_memory=True,
        drop_last=True,
        generator=train_generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config["BATCHSIZE"],
        shuffle=False,
        num_workers=config["NUM_WORKERS"],
        pin_memory=True,
    )

    model = build_model(config)
    if bool(config.get("MRFF_NTS_ENABLED", False)) and not model_uses_modality_aux(
        config
    ):
        raise ValueError("MRFF_NTS_ENABLED requires MODEL=pidnet_s_mrff.")
    matched_pretrained = (
        0 if args.resume else load_pretrained_if_available(model, config)
    )
    device = torch.device(config["DEVICE"])
    model = model.to(device)
    criterion = build_training_criterion(TotalLoss(config), config)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config["LR"],
        momentum=config["MOMENTUM"],
        weight_decay=config["WD"],
    )
    scaler = create_cuda_grad_scaler(
        enabled=args.amp,
        init_scale=float(config.get("AMP_INIT_SCALE", 65536.0)),
    )
    experiment_group = config.get(
        "EXPERIMENT_GROUP",
        "pidnet_s_rgb_baseline",
    )
    run_directory = PROJECT_ROOT / "experiments" / experiment_group / args.run_name
    if run_directory.exists() and not args.resume:
        existing = list(run_directory.iterdir())
        if existing:
            raise FileExistsError(
                f"Refusing to overwrite non-empty run directory: {run_directory}"
            )
    run_directory.mkdir(parents=True, exist_ok=True)
    metrics_path = run_directory / "metrics.jsonl"
    with (run_directory / "resolved_config.json").open(
        "w",
        encoding="utf-8",
    ) as config_file:
        json.dump(config, config_file, ensure_ascii=False, indent=2)
    selection_metric_name = str(config.get("SELECTION_METRIC", "miou"))
    if criterion.objective_name == "partial_label" and selection_metric_name != "fire_iou":
        raise ValueError(
            "FLAME3 partial-label training must select checkpoints by fire_iou"
        )
    best_selection_metric = -1.0
    start_epoch = 0
    if args.resume:
        resume_path = args.resume.resolve()
        checkpoint = torch.load(
            resume_path,
            map_location="cpu",
            weights_only=False,
        )
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        criterion_state = checkpoint.get("criterion_state_dict")
        if criterion_state is not None:
            if not hasattr(criterion, "load_state_dict"):
                raise RuntimeError(
                    "Checkpoint has criterion state, but this objective cannot restore it."
                )
            criterion.load_state_dict(criterion_state)
        train_generator.set_state(checkpoint["train_generator_state"])
        __import__("random").setstate(checkpoint["python_random_state"])
        np.random.set_state(checkpoint["numpy_random_state"])
        torch.set_rng_state(checkpoint["torch_random_state"])
        torch.cuda.set_rng_state_all(checkpoint["cuda_random_states"])
        start_epoch = int(checkpoint["epoch"])
        checkpoint_metric_name = str(
            checkpoint.get("selection_metric_name", "miou")
        )
        if checkpoint_metric_name != selection_metric_name:
            raise RuntimeError(
                f"Resume selection metric changed: {checkpoint_metric_name} "
                f"!= {selection_metric_name}"
            )
        best_selection_metric = float(
            checkpoint.get(
                "best_selection_metric",
                checkpoint.get("best_miou", -1.0),
            )
        )
        if start_epoch >= epochs:
            raise ValueError(
                f"Resume epoch {start_epoch} is not below requested epochs {epochs}."
            )

    print(f"Run directory: {run_directory}")
    print(f"Train/val samples: {len(train_dataset)}/{len(validation_dataset)}")
    print(f"Epochs: {epochs}, batch size: {config['BATCHSIZE']}, AMP: {args.amp}")
    print(f"Learning-rate schedule horizon: {lr_total_epochs} epochs")
    if criterion.objective_name == "partial_label":
        print(
            "FLAME3 partial-label objective: Fire core=hard Fire, "
            "Fire-folder non-core={Background,Smoke}, No Fire=hard Background"
        )
        print(f"Checkpoint selection metric: {selection_metric_name}")
        if model_uses_modality_aux(config):
            print(
                "MRFF: separate RGB/IR stems with explicit per-pixel modality "
                "weights and epoch gate diagnostics"
            )
            print(
                "NTS: "
                + (
                    f"enabled, target weight={float(config.get('MRFF_NTS_WEIGHT', 0.02)):.4f}"
                    if bool(config.get("MRFF_NTS_ENABLED", False))
                    else "disabled for standalone MRFF screening"
                )
            )
    elif criterion.objective_name == "smoke_binary":
        print(f"Smoke auxiliary loss weight: {criterion.auxiliary_weight:.4f}")
    elif criterion.objective_name == "fire_boundary":
        print(f"Fire-boundary loss weight: {criterion.auxiliary_weight:.4f}")
    elif criterion.objective_name == "fire_region":
        print(f"Fire-region loss weight: {criterion.auxiliary_weight:.4f}")
    elif criterion.objective_name == "active_boundary":
        print(f"Active-boundary loss weight: {criterion.auxiliary_weight:.4f}")
    elif criterion.objective_name == "class_prototype":
        print(
            "Class/prototype loss weights: "
            f"{criterion.class_auxiliary_weight:.4f}/"
            f"{criterion.prototype_weight:.4f}"
        )
    print(f"Matched pretrained tensors: {matched_pretrained}")
    if args.resume:
        print(f"Resumed from: {args.resume.resolve()} at epoch {start_epoch}")
    source_files = list((PROJECT_ROOT / "src").rglob("*.py"))
    source_files.extend(
        (PROJECT_ROOT / "third_party" / "RoboFireFuseNet" / "datasets").glob("*.py")
    )
    source_files.extend(
        (PROJECT_ROOT / "third_party" / "RoboFireFuseNet" / "models").glob("pidnet*.py")
    )
    source_files.append(
        PROJECT_ROOT / "third_party" / "RoboFireFuseNet" / "utils" / "total_loss.py"
    )
    data_root = Path(config["ROOTDATASET"])
    dataset_list_paths = {
        key: data_root / config[key]
        for key in ("TRAINSET", "VALIDSET", "TESTSET")
    }
    pretrained_path = (
        Path(config["PRETRAINED"]) if config.get("PRETRAINED") else None
    )
    component_manifest = (
        Path(config["FIRE_COMPONENT_CACHE"]) / "manifest.json"
        if config.get("FIRE_COMPONENT_CACHE")
        else None
    )
    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(device),
        "gpu_capability": list(torch.cuda.get_device_capability(device)),
        "seed": config["SEED"],
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "deterministic_algorithms_enabled": (
            torch.are_deterministic_algorithms_enabled()
        ),
        "config_sha256": hashlib.sha256(
            args.config.resolve().read_bytes()
        ).hexdigest(),
        "source_sha256": hash_files(source_files),
        "dataset_list_sha256": {
            key: hash_file(path) for key, path in dataset_list_paths.items()
        },
        "pretrained_sha256": (
            hash_file(pretrained_path)
            if pretrained_path is not None and pretrained_path.is_file()
            else None
        ),
        "fire_component_manifest_sha256": (
            hash_file(component_manifest)
            if component_manifest is not None and component_manifest.is_file()
            else None
        ),
        "training_parameters": sum(
            parameter.numel() for parameter in model.parameters()
        ),
    }
    with (run_directory / "environment.json").open("w", encoding="utf-8") as stream:
        json.dump(environment, stream, ensure_ascii=False, indent=2)
    start_time = time.perf_counter()

    metrics_mode = "a" if args.resume else "w"
    with metrics_path.open(metrics_mode, encoding="utf-8") as metrics_file:
        for epoch in range(start_epoch, epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            epoch_start = time.perf_counter()
            torch.cuda.reset_peak_memory_stats(device)
            (
                train_metrics,
                train_loss,
                train_loss_components,
                prototype_health,
                train_gate_statistics,
            ) = run_training_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                scaler,
                config,
                epoch,
                lr_total_epochs,
                args.max_train_batches,
                args.amp,
            )
            (
                validation_metrics,
                validation_loss,
                validation_loss_components,
                validation_gate_statistics,
            ) = (
                run_validation(
                model,
                validation_loader,
                criterion,
                config,
                epoch,
                args.max_val_batches,
                args.amp,
                )
            )
            record = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "train": train_metrics,
                "validation": validation_metrics,
                "loss_components": {
                    "train": train_loss_components,
                    "validation": validation_loss_components,
                },
                "learning_rate_schedule_total_epochs": lr_total_epochs,
                "epoch_elapsed_seconds": time.perf_counter() - epoch_start,
                "peak_allocated_gpu_memory_mb": (
                    torch.cuda.max_memory_allocated(device) / 1024**2
                ),
                "prototype_health": prototype_health,
                "modality_gate": (
                    {
                        "train": train_gate_statistics,
                        "validation": validation_gate_statistics,
                    }
                    if train_gate_statistics is not None
                    else None
                ),
                "model_diagnostics": collect_model_diagnostics(model),
                "selection_metric_name": selection_metric_name,
                "selection_metric_value": validation_metrics[
                    selection_metric_name
                ],
            }
            metrics_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            metrics_file.flush()

            if criterion.objective_name == "partial_label":
                print(
                    f"  train loss={train_loss:.5f}, "
                    f"Fire IoU={train_metrics['fire_iou']:.4f}, "
                    f"precision/recall={train_metrics['fire_precision']:.4f}/"
                    f"{train_metrics['fire_recall']:.4f}"
                )
                print(
                    f"  val loss={validation_loss:.5f}, "
                    f"Fire IoU={validation_metrics['fire_iou']:.4f}, "
                    f"precision/recall={validation_metrics['fire_precision']:.4f}/"
                    f"{validation_metrics['fire_recall']:.4f}, "
                    "empty-Fire FP="
                    f"{validation_metrics['empty_fire_predicted_fire_ratio']:.4%}, "
                    "No-Fire FP="
                    f"{validation_metrics['no_fire_predicted_fire_ratio']:.4%}"
                )
            else:
                print(
                    f"  train loss={train_loss:.5f}, mIoU={train_metrics['miou']:.4f}"
                )
                print(
                    f"  val loss={validation_loss:.5f}, "
                    f"mIoU={validation_metrics['miou']:.4f}, "
                    f"smoke IoU={validation_metrics['iou_smoke']:.4f}, "
                    f"fire IoU={validation_metrics['iou_fire']:.4f}"
                )
            if criterion.objective_name == "partial_label":
                print(
                    "  partial-label semantic main/boundary: "
                    f"train={train_loss_components['semantic_main']:.5f}/"
                    f"{train_loss_components['boundary']:.5f}, "
                    f"val={validation_loss_components['semantic_main']:.5f}/"
                    f"{validation_loss_components['boundary']:.5f}"
                )
                model_diagnostics = record["model_diagnostics"]
                if model_diagnostics is not None:
                    print(
                        "  ERCTC scales: "
                        f"region={model_diagnostics['region_scale']:.6f}, "
                        f"frontier={model_diagnostics['frontier_scale']:.6f}"
                    )
                if train_gate_statistics is not None:
                    train_thermal = train_gate_statistics["modalities"]["thermal"]
                    val_thermal = validation_gate_statistics["modalities"]["thermal"]
                    print(
                        "  MRFF thermal gate mean/std: "
                        f"train={train_thermal['mean']:.4f}/{train_thermal['std']:.4f}, "
                        f"val={val_thermal['mean']:.4f}/{val_thermal['std']:.4f}"
                    )
                    if bool(config.get("MRFF_NTS_ENABLED", False)):
                        print(
                            "  NTS loss/weight/selected px: "
                            f"train={train_loss_components['nts_loss']:.6f}/"
                            f"{train_loss_components['nts_weight']:.6f}/"
                            f"{train_loss_components['nts_selected_pixels']:.1f}, "
                            f"val={validation_loss_components['nts_loss']:.6f}/"
                            f"{validation_loss_components['nts_weight']:.6f}/"
                            f"{validation_loss_components['nts_selected_pixels']:.1f}"
                        )
            elif criterion.objective_name == "smoke_binary":
                print(
                    "  smoke auxiliary: "
                    f"train={train_loss_components['smoke_auxiliary']:.5f}, "
                    f"val={validation_loss_components['smoke_auxiliary']:.5f}"
                )
            elif criterion.objective_name == "fire_boundary":
                print(
                    "  fire-boundary auxiliary: "
                    f"train={train_loss_components['fire_boundary_auxiliary']:.5f}, "
                    f"val={validation_loss_components['fire_boundary_auxiliary']:.5f}"
                )
            elif criterion.objective_name == "fire_region":
                print(
                    "  fire-region auxiliary: "
                    f"train={train_loss_components['fire_region_auxiliary']:.5f}, "
                    f"val={validation_loss_components['fire_region_auxiliary']:.5f}"
                )
            elif criterion.objective_name == "active_boundary":
                print(
                    "  active boundary: "
                    f"train={train_loss_components['active_boundary']:.5f}, "
                    f"val={validation_loss_components['active_boundary']:.5f}, "
                    "supervised px="
                    f"{train_loss_components['abl_supervised_boundary_pixels']:.1f}"
                )
            elif criterion.objective_name == "class_prototype":
                print(
                    "  class/prototype auxiliary: "
                    f"train={train_loss_components['class_auxiliary']:.5f}/"
                    f"{train_loss_components['prototype_total']:.5f}, "
                    f"val={validation_loss_components['class_auxiliary']:.5f}/"
                    f"{validation_loss_components['prototype_total']:.5f}"
                )
            elif criterion.objective_name == "ema_mproto":
                print(
                    "  EMA prototype: "
                    f"train={train_loss_components['prototype_total']:.5f}, "
                    f"val={validation_loss_components['prototype_total']:.5f}, "
                    f"weight={train_loss_components['prototype_weight']:.5f}"
                )
                print(
                    "  prototype health valid: "
                    f"{prototype_health['valid_multi_prototype_method']}"
                )
            current_selection_metric = float(
                validation_metrics[selection_metric_name]
            )
            save_checkpoint(
                run_directory / "last.pth",
                model,
                optimizer,
                scaler,
                criterion,
                train_generator,
                config,
                epoch + 1,
                validation_metrics,
                max(best_selection_metric, current_selection_metric),
                selection_metric_name,
            )
            if current_selection_metric > best_selection_metric:
                best_selection_metric = current_selection_metric
                save_checkpoint(
                    run_directory / "best.pth",
                    model,
                    optimizer,
                    scaler,
                    criterion,
                    train_generator,
                    config,
                    epoch + 1,
                    validation_metrics,
                    best_selection_metric,
                    selection_metric_name,
                )

    elapsed = time.perf_counter() - start_time
    peak_memory_mb = torch.cuda.max_memory_allocated(device) / 1024**2
    print("Training run completed")
    print(
        f"Best validation {selection_metric_name}: "
        f"{best_selection_metric:.4f}"
    )
    print(f"Elapsed time: {elapsed:.1f} seconds")
    print(f"Peak allocated GPU memory: {peak_memory_mb:.1f} MB")
    print(f"Metrics: {metrics_path}")
    summary = {
        "selection_metric_name": selection_metric_name,
        "best_validation_selection_metric": best_selection_metric,
        "elapsed_seconds": elapsed,
        "peak_allocated_gpu_memory_mb": peak_memory_mb,
        "metrics": str(metrics_path),
        "environment": environment,
    }
    with (run_directory / "run_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
