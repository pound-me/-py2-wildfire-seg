from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from prototype_learning import EMAPrototypeTotalLoss


class SmokeAuxiliaryLoss(nn.Module):
    """Binary smoke supervision using masked BCE and soft Dice losses."""

    def __init__(
        self,
        ignore_label: int,
        smoke_class: int = 1,
        bce_weight: float = 1.0,
        dice_weight: float = 1.0,
        smooth: float = 1.0,
    ) -> None:
        super().__init__()
        self.ignore_label = ignore_label
        self.smoke_class = smoke_class
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth

    def forward(
        self,
        smoke_logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if smoke_logits.ndim != 4 or smoke_logits.shape[1] != 1:
            raise ValueError(
                "smoke_logits must have shape [batch, 1, height, width], "
                f"got {tuple(smoke_logits.shape)}"
            )
        if smoke_logits.shape[-2:] != labels.shape[-2:]:
            smoke_logits = F.interpolate(
                smoke_logits,
                size=labels.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        logits = smoke_logits[:, 0].float()
        valid = labels != self.ignore_label
        target = (labels == self.smoke_class).to(dtype=torch.float32)
        zero = logits.sum() * 0.0
        if not bool(valid.any()):
            return zero, zero, zero

        pixel_bce = F.binary_cross_entropy_with_logits(
            logits,
            target,
            reduction="none",
        )
        bce = pixel_bce[valid].mean()

        probabilities = torch.sigmoid(logits)
        valid_float = valid.to(dtype=torch.float32)
        intersection = (
            probabilities * target * valid_float
        ).flatten(1).sum(dim=1)
        denominator = (
            (probabilities * valid_float).flatten(1).sum(dim=1)
            + (target * valid_float).flatten(1).sum(dim=1)
        )
        dice_per_image = 1.0 - (
            2.0 * intersection + self.smooth
        ) / (denominator + self.smooth)
        valid_images = valid.flatten(1).any(dim=1)
        dice = dice_per_image[valid_images].mean()
        total = self.bce_weight * bce + self.dice_weight * dice
        return total, bce, dice


class SmokeAwareTotalLoss:
    """Wrap the original PIDNet loss without modifying third-party code."""

    def __init__(self, base_criterion: object, config: dict) -> None:
        self.base_criterion = base_criterion
        self.auxiliary_weight = float(config.get("SMOKE_AUX_WEIGHT", 0.0))
        self.objective_name = (
            "smoke_binary" if self.auxiliary_weight > 0.0 else "base"
        )
        self.smoke_criterion = SmokeAuxiliaryLoss(
            ignore_label=int(config["IGNORE_LABEL"]),
            smoke_class=int(config.get("SMOKE_CLASS_INDEX", 1)),
            bce_weight=float(config.get("SMOKE_AUX_BCE_WEIGHT", 1.0)),
            dice_weight=float(config.get("SMOKE_AUX_DICE_WEIGHT", 1.0)),
        )

    @staticmethod
    def split_outputs(
        outputs: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...],
    ) -> tuple[
        torch.Tensor | list[torch.Tensor],
        torch.Tensor | None,
    ]:
        if isinstance(outputs, (list, tuple)) and len(outputs) == 4:
            return list(outputs[:3]), outputs[3]
        return outputs, None

    def get_loss(
        self,
        outputs: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...],
        labels: torch.Tensor,
        edges: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor], object, dict[str, torch.Tensor]]:
        base_outputs, smoke_logits = self.split_outputs(outputs)
        losses, _, accuracy, base_parts = self.base_criterion.get_loss(
            base_outputs,
            labels,
            edges,
        )
        zero = losses.mean() * 0.0
        smoke_auxiliary = zero
        smoke_bce = zero
        smoke_dice = zero
        if self.auxiliary_weight > 0.0:
            if smoke_logits is None:
                raise RuntimeError(
                    "SMOKE_AUX_WEIGHT is positive, but the model did not "
                    "return smoke logits."
                )
            smoke_auxiliary, smoke_bce, smoke_dice = self.smoke_criterion(
                smoke_logits,
                labels,
            )
            losses = losses + self.auxiliary_weight * smoke_auxiliary

        if not isinstance(base_outputs, list):
            base_outputs = [base_outputs]
        components = {
            "base_total": losses.mean() - self.auxiliary_weight * smoke_auxiliary,
            "semantic": base_parts[0].mean(),
            "boundary": base_parts[1].mean(),
            "smoke_auxiliary": smoke_auxiliary,
            "smoke_bce": smoke_bce,
            "smoke_dice": smoke_dice,
            "weighted_smoke_auxiliary": (
                self.auxiliary_weight * smoke_auxiliary
            ),
        }
        return losses, base_outputs, accuracy, components


class FireBoundaryAuxiliaryLoss(nn.Module):
    """Balanced auxiliary supervision for the fire-class boundary."""

    def __init__(
        self,
        ignore_label: int,
        fire_class: int = 2,
        bce_weight: float = 1.0,
        dice_weight: float = 0.5,
        smooth: float = 1.0,
    ) -> None:
        super().__init__()
        self.ignore_label = ignore_label
        self.fire_class = fire_class
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth

    @staticmethod
    def _interior_boundary(mask: torch.Tensor) -> torch.Tensor:
        values = mask.to(dtype=torch.float32).unsqueeze(1)
        eroded = -F.max_pool2d(
            -values,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        return (values[:, 0] > 0.5) & (eroded[:, 0] < 0.5)

    def forward(
        self,
        boundary_logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if boundary_logits.ndim != 4 or boundary_logits.shape[1] != 1:
            raise ValueError(
                "boundary_logits must have shape [batch, 1, height, width], "
                f"got {tuple(boundary_logits.shape)}"
            )
        if boundary_logits.shape[-2:] != labels.shape[-2:]:
            boundary_logits = F.interpolate(
                boundary_logits,
                size=labels.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        logits = boundary_logits[:, 0].float()
        valid = labels != self.ignore_label
        target = self._interior_boundary(labels == self.fire_class)
        valid_target = target & valid
        zero = logits.sum() * 0.0
        positive_count = valid_target.sum().to(dtype=torch.float32)
        if not bool(valid.any()) or not bool(valid_target.any()):
            return zero, zero, zero, positive_count

        valid_logits = logits[valid]
        valid_targets = target[valid].to(dtype=torch.float32)
        positive = valid_targets.sum()
        negative = valid_targets.numel() - positive
        total = positive + negative
        weights = torch.where(
            valid_targets > 0.5,
            negative / total,
            positive / total,
        )
        bce = F.binary_cross_entropy_with_logits(
            valid_logits,
            valid_targets,
            weight=weights,
            reduction="mean",
        )

        probabilities = torch.sigmoid(logits)
        valid_float = valid.to(dtype=torch.float32)
        target_float = target.to(dtype=torch.float32)
        intersection = (
            probabilities * target_float * valid_float
        ).flatten(1).sum(dim=1)
        denominator = (
            (probabilities * valid_float).flatten(1).sum(dim=1)
            + (target_float * valid_float).flatten(1).sum(dim=1)
        )
        dice_per_image = 1.0 - (
            2.0 * intersection + self.smooth
        ) / (denominator + self.smooth)
        valid_images = valid.flatten(1).any(dim=1)
        dice = dice_per_image[valid_images].mean()
        total_loss = self.bce_weight * bce + self.dice_weight * dice
        return total_loss, bce, dice, positive_count


class FireBoundaryTotalLoss:
    """Original PIDNet loss plus training-only fire-boundary emphasis."""

    def __init__(self, base_criterion: object, config: dict) -> None:
        self.base_criterion = base_criterion
        self.objective_name = "fire_boundary"
        self.auxiliary_weight = float(config.get("FIRE_BOUNDARY_WEIGHT", 0.0))
        self.fire_criterion = FireBoundaryAuxiliaryLoss(
            ignore_label=int(config["IGNORE_LABEL"]),
            fire_class=int(config.get("FIRE_CLASS_INDEX", 2)),
            bce_weight=float(config.get("FIRE_BOUNDARY_BCE_WEIGHT", 1.0)),
            dice_weight=float(config.get("FIRE_BOUNDARY_DICE_WEIGHT", 0.5)),
        )

    def get_loss(
        self,
        outputs: list[torch.Tensor] | tuple[torch.Tensor, ...],
        labels: torch.Tensor,
        edges: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor], object, dict[str, torch.Tensor]]:
        if not isinstance(outputs, (list, tuple)) or len(outputs) < 3:
            raise RuntimeError(
                "Fire-boundary training expects PIDNet detail, semantic, "
                "and boundary outputs."
            )
        raw_outputs = list(outputs)
        losses, metric_outputs, accuracy, base_parts = self.base_criterion.get_loss(
            raw_outputs,
            labels,
            edges,
        )
        fire_auxiliary, fire_bce, fire_dice, positive_count = self.fire_criterion(
            raw_outputs[-1],
            labels,
        )
        weighted_fire = self.auxiliary_weight * fire_auxiliary
        losses = losses + weighted_fire
        components = {
            "base_total": losses.mean() - weighted_fire,
            "semantic": base_parts[0].mean(),
            "boundary": base_parts[1].mean(),
            "fire_boundary_auxiliary": fire_auxiliary,
            "fire_boundary_bce": fire_bce,
            "fire_boundary_dice": fire_dice,
            "fire_boundary_positive_pixels": positive_count,
            "weighted_fire_boundary": weighted_fire,
        }
        return losses, metric_outputs, accuracy, components


class FireRegionAuxiliaryLoss(nn.Module):
    """Fire-vs-rest balanced focal and recall-oriented Tversky objective."""

    def __init__(
        self,
        ignore_label: int,
        fire_class: int = 2,
        focal_alpha: float = 0.75,
        focal_gamma: float = 2.0,
        tversky_alpha: float = 0.3,
        tversky_beta: float = 0.7,
        focal_weight: float = 1.0,
        tversky_weight: float = 1.0,
        smooth: float = 1.0,
    ) -> None:
        super().__init__()
        self.ignore_label = ignore_label
        self.fire_class = fire_class
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.tversky_alpha = tversky_alpha
        self.tversky_beta = tversky_beta
        self.focal_weight = focal_weight
        self.tversky_weight = tversky_weight
        self.smooth = smooth

    def forward(
        self,
        semantic_logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if semantic_logits.ndim != 4:
            raise ValueError(
                "semantic_logits must have shape [batch, classes, height, width], "
                f"got {tuple(semantic_logits.shape)}"
            )
        if not 0 <= self.fire_class < semantic_logits.shape[1]:
            raise ValueError(
                f"fire class {self.fire_class} is outside the semantic logits."
            )
        if semantic_logits.shape[-2:] != labels.shape[-2:]:
            semantic_logits = F.interpolate(
                semantic_logits,
                size=labels.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        logits = semantic_logits.float()
        fire_logits = logits[:, self.fire_class]
        other_indices = [
            index for index in range(logits.shape[1]) if index != self.fire_class
        ]
        rest_logits = torch.logsumexp(logits[:, other_indices], dim=1)
        binary_logits = fire_logits - rest_logits
        probabilities = torch.sigmoid(binary_logits)
        valid = labels != self.ignore_label
        target = labels == self.fire_class
        target_float = target.to(dtype=torch.float32)
        zero = binary_logits.sum() * 0.0
        positive_count = (target & valid).sum().to(dtype=torch.float32)
        if not bool(valid.any()):
            return zero, zero, zero, positive_count

        bce = F.binary_cross_entropy_with_logits(
            binary_logits,
            target_float,
            reduction="none",
        )
        probability_target = torch.where(target, probabilities, 1.0 - probabilities)
        alpha_target = torch.where(
            target,
            torch.full_like(probabilities, self.focal_alpha),
            torch.full_like(probabilities, 1.0 - self.focal_alpha),
        )
        focal_values = (
            alpha_target
            * (1.0 - probability_target).pow(self.focal_gamma)
            * bce
        )
        positive_mask = target & valid
        negative_mask = (~target) & valid
        positive_focal = (
            focal_values[positive_mask].mean()
            if bool(positive_mask.any())
            else zero
        )
        negative_focal = (
            focal_values[negative_mask].mean()
            if bool(negative_mask.any())
            else zero
        )
        if bool(positive_mask.any()) and bool(negative_mask.any()):
            focal = 0.5 * (positive_focal + negative_focal)
        else:
            focal = positive_focal + negative_focal

        valid_float = valid.to(dtype=torch.float32)
        true_positive = (
            probabilities * target_float * valid_float
        ).flatten(1).sum(dim=1)
        false_positive = (
            probabilities * (1.0 - target_float) * valid_float
        ).flatten(1).sum(dim=1)
        false_negative = (
            (1.0 - probabilities) * target_float * valid_float
        ).flatten(1).sum(dim=1)
        tversky_per_image = 1.0 - (
            true_positive + self.smooth
        ) / (
            true_positive
            + self.tversky_alpha * false_positive
            + self.tversky_beta * false_negative
            + self.smooth
        )
        fire_images = positive_mask.flatten(1).any(dim=1)
        tversky = (
            tversky_per_image[fire_images].mean()
            if bool(fire_images.any())
            else zero
        )
        total_loss = (
            self.focal_weight * focal
            + self.tversky_weight * tversky
        )
        return total_loss, focal, tversky, positive_count


class FireRegionTotalLoss:
    """Original PIDNet loss plus direct fire-region semantic supervision."""

    def __init__(self, base_criterion: object, config: dict) -> None:
        self.base_criterion = base_criterion
        self.objective_name = "fire_region"
        self.auxiliary_weight = float(config.get("FIRE_REGION_WEIGHT", 0.0))
        self.fire_criterion = FireRegionAuxiliaryLoss(
            ignore_label=int(config["IGNORE_LABEL"]),
            fire_class=int(config.get("FIRE_CLASS_INDEX", 2)),
            focal_alpha=float(config.get("FIRE_FOCAL_ALPHA", 0.75)),
            focal_gamma=float(config.get("FIRE_FOCAL_GAMMA", 2.0)),
            tversky_alpha=float(config.get("FIRE_TVERSKY_ALPHA", 0.3)),
            tversky_beta=float(config.get("FIRE_TVERSKY_BETA", 0.7)),
            focal_weight=float(config.get("FIRE_FOCAL_WEIGHT", 1.0)),
            tversky_weight=float(config.get("FIRE_TVERSKY_WEIGHT", 1.0)),
        )

    def get_loss(
        self,
        outputs: list[torch.Tensor] | tuple[torch.Tensor, ...],
        labels: torch.Tensor,
        edges: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor], object, dict[str, torch.Tensor]]:
        if not isinstance(outputs, (list, tuple)) or len(outputs) < 3:
            raise RuntimeError(
                "Fire-region training expects PIDNet detail, semantic, "
                "and boundary outputs."
            )
        raw_outputs = list(outputs)
        losses, metric_outputs, accuracy, base_parts = self.base_criterion.get_loss(
            raw_outputs,
            labels,
            edges,
        )
        fire_auxiliary, fire_focal, fire_tversky, positive_count = (
            self.fire_criterion(raw_outputs[1], labels)
        )
        weighted_fire = self.auxiliary_weight * fire_auxiliary
        losses = losses + weighted_fire
        components = {
            "base_total": losses.mean() - weighted_fire,
            "semantic": base_parts[0].mean(),
            "boundary": base_parts[1].mean(),
            "fire_region_auxiliary": fire_auxiliary,
            "fire_region_focal": fire_focal,
            "fire_region_tversky": fire_tversky,
            "fire_region_positive_pixels": positive_count,
            "weighted_fire_region": weighted_fire,
        }
        return losses, metric_outputs, accuracy, components


class ClassPrototypeLoss(nn.Module):
    """Class-balanced pixel-to-prototype loss with prototype separation."""

    def __init__(
        self,
        num_classes: int,
        ignore_label: int,
        temperature: float = 0.2,
        separation_margin: float = 0.2,
        separation_weight: float = 0.5,
        minimum_pixels: int = 4,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.ignore_label = ignore_label
        self.temperature = temperature
        self.separation_margin = separation_margin
        self.separation_weight = separation_weight
        self.minimum_pixels = minimum_pixels

    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        if features.ndim != 4:
            raise ValueError(
                f"features must be [batch, channels, height, width], got "
                f"{tuple(features.shape)}"
            )
        resized_labels = F.interpolate(
            labels.unsqueeze(1).to(dtype=torch.float32),
            size=features.shape[-2:],
            mode="nearest",
        )[:, 0].to(dtype=torch.long)
        normalized = F.normalize(features.float(), dim=1)
        flat_features = normalized.permute(0, 2, 3, 1).reshape(
            -1,
            normalized.shape[1],
        )
        flat_labels = resized_labels.reshape(-1)

        class_ids: list[int] = []
        live_prototypes: list[torch.Tensor] = []
        class_feature_sets: list[torch.Tensor] = []
        for class_index in range(self.num_classes):
            class_features = flat_features[flat_labels == class_index]
            if class_features.shape[0] < self.minimum_pixels:
                continue
            prototype = F.normalize(class_features.mean(dim=0), dim=0)
            class_ids.append(class_index)
            live_prototypes.append(prototype)
            class_feature_sets.append(class_features)

        zero = features.float().sum() * 0.0
        if len(class_ids) < 2:
            return zero, zero, zero, len(class_ids)

        prototype_matrix = torch.stack(live_prototypes, dim=0)
        detached_prototypes = prototype_matrix.detach()
        class_losses = []
        for prototype_index, class_features in enumerate(class_feature_sets):
            similarity_logits = (
                class_features @ detached_prototypes.T
            ) / self.temperature
            targets = torch.full(
                (class_features.shape[0],),
                prototype_index,
                dtype=torch.long,
                device=features.device,
            )
            class_losses.append(F.cross_entropy(similarity_logits, targets))
        pixel_to_prototype = torch.stack(class_losses).mean()

        cosine_matrix = prototype_matrix @ prototype_matrix.T
        off_diagonal = ~torch.eye(
            len(class_ids),
            dtype=torch.bool,
            device=features.device,
        )
        separation = F.relu(
            cosine_matrix[off_diagonal] - self.separation_margin
        ).mean()
        total = pixel_to_prototype + self.separation_weight * separation
        return total, pixel_to_prototype, separation, len(class_ids)


class ClassPrototypeTotalLoss:
    """Original PIDNet loss plus three-class auxiliary and prototype losses."""

    def __init__(self, base_criterion: object, config: dict) -> None:
        self.base_criterion = base_criterion
        self.objective_name = "class_prototype"
        self.class_auxiliary_weight = float(config["CLASS_AUX_WEIGHT"])
        self.prototype_weight = float(config["PROTOTYPE_LOSS_WEIGHT"])
        self.auxiliary_weight = (
            self.class_auxiliary_weight + self.prototype_weight
        )
        self.ignore_label = int(config["IGNORE_LABEL"])
        self.align_corners = bool(config["ALIGN_CORNERS"])
        self.balanced_class_auxiliary = bool(
            config.get("CLASS_AUX_BALANCED", False)
        )
        self.class_weights = torch.tensor(
            config["CLASS_WEIGHTS"],
            dtype=torch.float32,
            device=config["DEVICE"],
        )
        self.prototype_criterion = ClassPrototypeLoss(
            num_classes=int(config["NUM_CLASSES"]),
            ignore_label=self.ignore_label,
            temperature=float(config.get("PROTOTYPE_TEMPERATURE", 0.2)),
            separation_margin=float(
                config.get("PROTOTYPE_SEPARATION_MARGIN", 0.2)
            ),
            separation_weight=float(
                config.get("PROTOTYPE_SEPARATION_WEIGHT", 0.5)
            ),
            minimum_pixels=int(config.get("PROTOTYPE_MIN_PIXELS", 4)),
        )

    def get_loss(
        self,
        outputs: list[torch.Tensor] | tuple[torch.Tensor, ...],
        labels: torch.Tensor,
        edges: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor], object, dict[str, torch.Tensor]]:
        if not isinstance(outputs, (list, tuple)) or len(outputs) != 5:
            raise RuntimeError(
                "Class-prototype training expects three PIDNet outputs, "
                "class logits, and prototype features."
            )
        base_outputs = list(outputs[:3])
        class_logits = outputs[3]
        prototype_features = outputs[4]
        losses, _, accuracy, base_parts = self.base_criterion.get_loss(
            base_outputs,
            labels,
            edges,
        )
        base_total = losses.mean()

        resized_labels = F.interpolate(
            labels.unsqueeze(1).to(dtype=torch.float32),
            size=class_logits.shape[-2:],
            mode="nearest",
        )[:, 0].to(dtype=torch.long)
        if self.balanced_class_auxiliary:
            log_probabilities = F.log_softmax(class_logits.float(), dim=1)
            per_class_losses = []
            for class_index in range(class_logits.shape[1]):
                class_mask = resized_labels == class_index
                if not bool(class_mask.any()):
                    continue
                per_class_losses.append(
                    -log_probabilities[:, class_index][class_mask].mean()
                )
            if per_class_losses:
                class_auxiliary = torch.stack(per_class_losses).mean()
            else:
                class_auxiliary = class_logits.float().sum() * 0.0
        else:
            class_auxiliary = F.cross_entropy(
                class_logits.float(),
                resized_labels,
                weight=self.class_weights,
                ignore_index=self.ignore_label,
            )
        (
            prototype_total,
            prototype_pixel,
            prototype_separation,
            present_classes,
        ) = self.prototype_criterion(prototype_features, labels)

        weighted_class_auxiliary = (
            self.class_auxiliary_weight * class_auxiliary
        )
        weighted_prototype = self.prototype_weight * prototype_total
        losses = losses + weighted_class_auxiliary + weighted_prototype
        components = {
            "base_total": base_total,
            "semantic": base_parts[0].mean(),
            "boundary": base_parts[1].mean(),
            "class_auxiliary": class_auxiliary,
            "prototype_total": prototype_total,
            "prototype_pixel": prototype_pixel,
            "prototype_separation": prototype_separation,
            "prototype_present_classes": torch.tensor(
                float(present_classes),
                device=labels.device,
            ),
            "weighted_class_auxiliary": weighted_class_auxiliary,
            "weighted_prototype": weighted_prototype,
        }
        return losses, base_outputs, accuracy, components


def build_training_criterion(base_criterion: object, config: dict):
    if config.get("TRAINING_OBJECTIVE") == "fire_boundary":
        return FireBoundaryTotalLoss(base_criterion, config)
    if config.get("TRAINING_OBJECTIVE") == "fire_region":
        return FireRegionTotalLoss(base_criterion, config)
    if config.get("TRAINING_OBJECTIVE") == "class_prototype":
        return ClassPrototypeTotalLoss(base_criterion, config)
    if config.get("TRAINING_OBJECTIVE") == "ema_mproto":
        return EMAPrototypeTotalLoss(base_criterion, config)
    return SmokeAwareTotalLoss(base_criterion, config)


__all__ = [
    "ClassPrototypeLoss",
    "ClassPrototypeTotalLoss",
    "EMAPrototypeTotalLoss",
    "FireBoundaryAuxiliaryLoss",
    "FireBoundaryTotalLoss",
    "FireRegionAuxiliaryLoss",
    "FireRegionTotalLoss",
    "SmokeAuxiliaryLoss",
    "SmokeAwareTotalLoss",
    "build_training_criterion",
]
