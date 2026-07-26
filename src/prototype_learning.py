from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SampledFeatures:
    features: torch.Tensor
    class_ids: torch.Tensor
    fire_bucket_ids: torch.Tensor


class EMAPrototypeBank(nn.Module):
    """Class-aware EMA prototype memory with full-resolution point sampling."""

    def __init__(
        self,
        num_classes: int,
        feature_channels: int,
        prototypes_per_class: int,
        temperature: float,
        momentum: float,
        max_samples_per_class: int,
        fire_class: int,
        fire_sampling: str,
        decorrelation: bool,
        decorrelation_weight: float,
        decorrelation_min_samples: int,
        seed: int,
        collapse_threshold: float = 0.95,
        persistence_epochs: int = 5,
    ) -> None:
        super().__init__()
        if prototypes_per_class < 1:
            raise ValueError("prototypes_per_class must be positive.")
        if fire_sampling not in {"uniform", "component_mixed"}:
            raise ValueError(f"Unsupported fire sampling: {fire_sampling}")
        self.num_classes = num_classes
        self.feature_channels = feature_channels
        self.prototypes_per_class = prototypes_per_class
        self.temperature = float(temperature)
        self.momentum = float(momentum)
        self.max_samples_per_class = int(max_samples_per_class)
        self.fire_class = int(fire_class)
        self.fire_sampling = fire_sampling
        self.use_decorrelation = bool(decorrelation)
        self.decorrelation_weight = float(decorrelation_weight)
        self.decorrelation_min_samples = int(decorrelation_min_samples)
        self.seed = int(seed)
        self.collapse_threshold = float(collapse_threshold)
        self.persistence_epochs = int(persistence_epochs)

        shape = (num_classes, prototypes_per_class, feature_channels)
        self.register_buffer("prototypes", torch.zeros(shape, dtype=torch.float32))
        self.register_buffer(
            "initialized",
            torch.zeros(num_classes, prototypes_per_class, dtype=torch.bool),
        )
        self.register_buffer(
            "ema_update_counts",
            torch.zeros(num_classes, prototypes_per_class, dtype=torch.long),
        )
        self.register_buffer(
            "dead_streak",
            torch.zeros(num_classes, prototypes_per_class, dtype=torch.long),
        )
        self.register_buffer(
            "collapse_streak",
            torch.zeros(
                num_classes,
                prototypes_per_class,
                prototypes_per_class,
                dtype=torch.long,
            ),
        )
        self._epoch_hits = torch.zeros(
            num_classes, prototypes_per_class, dtype=torch.long
        )
        self._epoch_fire_buckets = torch.zeros(
            3, prototypes_per_class, dtype=torch.long
        )

    def begin_epoch(self) -> None:
        self._epoch_hits.zero_()
        self._epoch_fire_buckets.zero_()

    @staticmethod
    def _uniform_indices(
        count: int,
        requested: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        if count <= requested:
            return np.arange(count, dtype=np.int64)
        return rng.choice(count, size=requested, replace=False)

    def _sample_fire_mixed(
        self,
        coordinates: np.ndarray,
        component_maps: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        total = coordinates.shape[0]
        requested = min(total, self.max_samples_per_class)
        if total <= self.max_samples_per_class:
            return np.arange(total, dtype=np.int64)

        uniform_target = requested // 2
        component_target = requested - uniform_target
        uniform = self._uniform_indices(total, uniform_target, rng)
        used = set(int(index) for index in uniform.tolist())

        pools: dict[tuple[int, int], list[int]] = {}
        for index, (batch_index, y, x) in enumerate(coordinates):
            if index in used:
                continue
            component_id = int(component_maps[batch_index, y, x])
            if component_id <= 0:
                raise RuntimeError("A sampled fire pixel has component id zero.")
            pools.setdefault((int(batch_index), component_id), []).append(index)
        for values in pools.values():
            rng.shuffle(values)

        selected = list(int(index) for index in uniform.tolist())
        active_keys = list(pools)
        while active_keys and len(selected) < uniform_target + component_target:
            rng.shuffle(active_keys)
            next_active = []
            for key in active_keys:
                values = pools[key]
                if values:
                    index = values.pop()
                    selected.append(index)
                    used.add(index)
                    if len(selected) >= uniform_target + component_target:
                        break
                if values:
                    next_active.append(key)
            active_keys = next_active

        if len(selected) < requested:
            remaining = np.asarray(
                [index for index in range(total) if index not in used],
                dtype=np.int64,
            )
            deficit = requested - len(selected)
            if remaining.size:
                fill = remaining[
                    self._uniform_indices(remaining.size, deficit, rng)
                ]
                selected.extend(int(index) for index in fill.tolist())
        result = np.asarray(selected, dtype=np.int64)
        if result.size != requested or np.unique(result).size != result.size:
            raise RuntimeError("Fire mixed sampling failed its no-replacement rule.")
        return result

    @staticmethod
    def _component_area_buckets(
        coordinates: np.ndarray,
        component_maps: np.ndarray,
    ) -> np.ndarray:
        areas: dict[tuple[int, int], int] = {}
        for batch_index in range(component_maps.shape[0]):
            ids, counts = np.unique(
                component_maps[batch_index], return_counts=True
            )
            for component_id, count in zip(ids.tolist(), counts.tolist()):
                if component_id > 0:
                    areas[(batch_index, int(component_id))] = int(count)
        buckets = np.full(coordinates.shape[0], -1, dtype=np.int64)
        for index, (batch_index, y, x) in enumerate(coordinates):
            component_id = int(component_maps[batch_index, y, x])
            area = areas[(int(batch_index), component_id)]
            buckets[index] = 0 if area <= 4 else (1 if area <= 64 else 2)
        return buckets

    def _sample_coordinates(
        self,
        labels: np.ndarray,
        component_maps: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        coordinate_sets = []
        class_sets = []
        bucket_sets = []
        for class_index in range(self.num_classes):
            coordinates = np.argwhere(labels == class_index).astype(
                np.int64, copy=False
            )
            if coordinates.size == 0:
                continue
            if (
                class_index == self.fire_class
                and self.fire_sampling == "component_mixed"
            ):
                selected = self._sample_fire_mixed(
                    coordinates, component_maps, rng
                )
            else:
                selected = self._uniform_indices(
                    coordinates.shape[0],
                    self.max_samples_per_class,
                    rng,
                )
            coordinates = coordinates[selected]
            coordinate_sets.append(coordinates)
            class_sets.append(
                np.full(coordinates.shape[0], class_index, dtype=np.int64)
            )
            if class_index == self.fire_class:
                bucket_sets.append(
                    self._component_area_buckets(coordinates, component_maps)
                )
            else:
                bucket_sets.append(
                    np.full(coordinates.shape[0], -1, dtype=np.int64)
                )
        if not coordinate_sets:
            empty = np.empty((0,), dtype=np.int64)
            return np.empty((0, 3), dtype=np.int64), empty, empty
        return (
            np.concatenate(coordinate_sets, axis=0),
            np.concatenate(class_sets, axis=0),
            np.concatenate(bucket_sets, axis=0),
        )

    def sample_features(
        self,
        fused_feature: torch.Tensor,
        labels: torch.Tensor,
        component_maps: torch.Tensor,
        sampling_seed: int,
    ) -> SampledFeatures:
        labels_np = labels.detach().cpu().numpy()
        components_np = component_maps.detach().cpu().numpy()
        if labels_np.shape != components_np.shape:
            raise ValueError("Labels and component maps must share a shape.")
        if not np.array_equal(components_np > 0, labels_np == self.fire_class):
            raise RuntimeError(
                "Batch fire component maps are not aligned with augmented labels."
            )
        rng = np.random.default_rng(sampling_seed)
        coordinates, class_ids, bucket_ids = self._sample_coordinates(
            labels_np,
            components_np,
            rng,
        )
        device = fused_feature.device
        if coordinates.shape[0] == 0:
            return SampledFeatures(
                features=fused_feature.float().new_empty(
                    (0, fused_feature.shape[1])
                ),
                class_ids=torch.empty(0, dtype=torch.long, device=device),
                fire_bucket_ids=torch.empty(
                    0, dtype=torch.long, device=device
                ),
            )

        sampled_parts = []
        class_parts = []
        bucket_parts = []
        height, width = labels_np.shape[-2:]
        for batch_index in np.unique(coordinates[:, 0]):
            mask = coordinates[:, 0] == batch_index
            image_coordinates = coordinates[mask]
            x = torch.as_tensor(
                image_coordinates[:, 2], dtype=torch.float32, device=device
            )
            y = torch.as_tensor(
                image_coordinates[:, 1], dtype=torch.float32, device=device
            )
            x_norm = 2.0 * (x + 0.5) / width - 1.0
            y_norm = 2.0 * (y + 0.5) / height - 1.0
            grid = torch.stack((x_norm, y_norm), dim=1).view(1, -1, 1, 2)
            sampled = F.grid_sample(
                fused_feature[int(batch_index) : int(batch_index) + 1].float(),
                grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )
            sampled_parts.append(sampled[0, :, :, 0].transpose(0, 1))
            class_parts.append(
                torch.as_tensor(class_ids[mask], dtype=torch.long, device=device)
            )
            bucket_parts.append(
                torch.as_tensor(bucket_ids[mask], dtype=torch.long, device=device)
            )
        return SampledFeatures(
            features=torch.cat(sampled_parts, dim=0),
            class_ids=torch.cat(class_parts, dim=0),
            fire_bucket_ids=torch.cat(bucket_parts, dim=0),
        )

    @staticmethod
    def _farthest_initialization(
        features: torch.Tensor,
        count: int,
    ) -> torch.Tensor:
        normalized = F.normalize(features.float(), dim=1)
        if normalized.shape[0] < count:
            raise ValueError("Not enough features to initialize all prototypes.")
        mean = F.normalize(normalized.mean(dim=0), dim=0)
        selected = [int(torch.argmin(normalized @ mean).item())]
        while len(selected) < count:
            similarity = normalized @ normalized[selected].T
            nearest = similarity.max(dim=1).values
            nearest[selected] = 1.0
            selected.append(int(torch.argmin(nearest).item()))
        return normalized[selected]

    @torch.no_grad()
    def _initialize_and_update(
        self,
        features: torch.Tensor,
        class_ids: torch.Tensor,
    ) -> torch.Tensor:
        assignments = torch.full_like(class_ids, -1)
        normalized = F.normalize(features.float(), dim=1)
        for class_index in range(self.num_classes):
            class_mask = class_ids == class_index
            class_features = normalized[class_mask]
            if class_features.shape[0] == 0:
                continue
            if not bool(self.initialized[class_index].all()):
                if class_features.shape[0] < self.prototypes_per_class:
                    continue
                initial = self._farthest_initialization(
                    class_features, self.prototypes_per_class
                )
                self.prototypes[class_index].copy_(initial)
                self.initialized[class_index].fill_(True)

            prototypes = self.prototypes[class_index]
            local_assignments = torch.argmax(
                class_features @ prototypes.T,
                dim=1,
            )
            assignments[class_mask] = local_assignments
            for prototype_index in range(self.prototypes_per_class):
                assigned = class_features[
                    local_assignments == prototype_index
                ]
                if assigned.shape[0] == 0:
                    continue
                mean = F.normalize(assigned.mean(dim=0), dim=0)
                updated = F.normalize(
                    self.momentum * self.prototypes[class_index, prototype_index]
                    + (1.0 - self.momentum) * mean,
                    dim=0,
                )
                self.prototypes[class_index, prototype_index].copy_(updated)
                self.ema_update_counts[class_index, prototype_index] += 1
        return assignments

    def _contrastive_loss(
        self,
        features: torch.Tensor,
        class_ids: torch.Tensor,
        assignments: torch.Tensor,
    ) -> torch.Tensor:
        normalized = F.normalize(features.float(), dim=1)
        valid = assignments >= 0
        zero = normalized.sum() * 0.0
        if not bool(valid.any()):
            return zero
        flat_initialized = self.initialized.reshape(-1)
        flat_prototypes = self.prototypes.reshape(
            -1, self.feature_channels
        )[flat_initialized].detach()
        global_indices = (
            class_ids[valid] * self.prototypes_per_class + assignments[valid]
        )
        initialized_indices = torch.nonzero(
            flat_initialized, as_tuple=False
        )[:, 0]
        target_lookup = torch.full(
            (self.num_classes * self.prototypes_per_class,),
            -1,
            dtype=torch.long,
            device=features.device,
        )
        target_lookup[initialized_indices] = torch.arange(
            initialized_indices.numel(), device=features.device
        )
        targets = target_lookup[global_indices]
        logits = normalized[valid] @ flat_prototypes.T / self.temperature
        class_losses = []
        valid_classes = class_ids[valid]
        for class_index in range(self.num_classes):
            class_mask = valid_classes == class_index
            if bool(class_mask.any()):
                class_losses.append(
                    F.cross_entropy(logits[class_mask], targets[class_mask])
                )
        return torch.stack(class_losses).mean() if class_losses else zero

    def _decorrelation_loss(
        self,
        features: torch.Tensor,
        class_ids: torch.Tensor,
        assignments: torch.Tensor,
    ) -> torch.Tensor:
        zero = features.float().sum() * 0.0
        if not self.use_decorrelation or self.prototypes_per_class < 2:
            return zero
        losses = []
        for class_index in range(self.num_classes):
            centers = []
            for prototype_index in range(self.prototypes_per_class):
                mask = (
                    (class_ids == class_index)
                    & (assignments == prototype_index)
                )
                if int(mask.sum().item()) >= self.decorrelation_min_samples:
                    centers.append(
                        F.normalize(features[mask].float().mean(dim=0), dim=0)
                    )
            if len(centers) < 2:
                continue
            centers_tensor = torch.stack(centers, dim=0)
            cosine = centers_tensor @ centers_tensor.T
            upper = torch.triu(
                torch.ones_like(cosine, dtype=torch.bool), diagonal=1
            )
            losses.append(cosine[upper].pow(2).mean())
        return torch.stack(losses).mean() if losses else zero

    def _collect_health(
        self,
        class_ids: torch.Tensor,
        assignments: torch.Tensor,
        fire_bucket_ids: torch.Tensor,
    ) -> None:
        valid = assignments >= 0
        for class_index in range(self.num_classes):
            class_mask = valid & (class_ids == class_index)
            for prototype_index in range(self.prototypes_per_class):
                count = int(
                    (class_mask & (assignments == prototype_index)).sum().item()
                )
                self._epoch_hits[class_index, prototype_index] += count
        fire_mask = valid & (class_ids == self.fire_class)
        for bucket_index in range(3):
            bucket_mask = fire_mask & (fire_bucket_ids == bucket_index)
            for prototype_index in range(self.prototypes_per_class):
                count = int(
                    (bucket_mask & (assignments == prototype_index)).sum().item()
                )
                self._epoch_fire_buckets[bucket_index, prototype_index] += count

    def forward(
        self,
        fused_feature: torch.Tensor,
        labels: torch.Tensor,
        component_maps: torch.Tensor,
        sampling_seed: int,
        update_ema: bool,
        collect_health: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]]:
        device_type = fused_feature.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            sampled = self.sample_features(
                fused_feature,
                labels,
                component_maps,
                sampling_seed,
            )
            if update_ema:
                assignments = self._initialize_and_update(
                    sampled.features.detach(), sampled.class_ids
                )
            else:
                assignments = torch.full_like(sampled.class_ids, -1)
                normalized = F.normalize(sampled.features.float(), dim=1)
                for class_index in range(self.num_classes):
                    class_mask = sampled.class_ids == class_index
                    if not bool(class_mask.any()) or not bool(
                        self.initialized[class_index].all()
                    ):
                        continue
                    assignments[class_mask] = torch.argmax(
                        normalized[class_mask]
                        @ self.prototypes[class_index].detach().T,
                        dim=1,
                    )
            contrastive = self._contrastive_loss(
                sampled.features,
                sampled.class_ids,
                assignments,
            )
            decorrelation = self._decorrelation_loss(
                sampled.features,
                sampled.class_ids,
                assignments,
            )
            total = contrastive + self.decorrelation_weight * decorrelation
            if collect_health:
                self._collect_health(
                    sampled.class_ids,
                    assignments,
                    sampled.fire_bucket_ids,
                )
            diagnostics = {
                "sampled_total": float(sampled.features.shape[0]),
                "sampled_background": float(
                    (sampled.class_ids == 0).sum().item()
                ),
                "sampled_smoke": float(
                    (sampled.class_ids == 1).sum().item()
                ),
                "sampled_fire": float(
                    (sampled.class_ids == self.fire_class).sum().item()
                ),
            }
            return total, contrastive, decorrelation, diagnostics

    @torch.no_grad()
    def finalize_epoch_health(self) -> dict:
        hits = self._epoch_hits.to(self.prototypes.device)
        initialized = self.initialized
        # The experiment rule is based on observed assignments, not on whether
        # a prototype ever managed to initialize.  An uninitialized prototype
        # also has zero hits and must not silently evade the five-epoch dead
        # prototype gate.
        dead_now = hits == 0
        self.dead_streak.copy_(
            torch.where(dead_now, self.dead_streak + 1, torch.zeros_like(self.dead_streak))
        )

        for class_index in range(self.num_classes):
            if not bool(initialized[class_index].all()):
                self.collapse_streak[class_index].zero_()
                continue
            cosine = self.prototypes[class_index] @ self.prototypes[class_index].T
            collapsed = cosine > self.collapse_threshold
            collapsed.fill_diagonal_(False)
            self.collapse_streak[class_index].copy_(
                torch.where(
                    collapsed,
                    self.collapse_streak[class_index] + 1,
                    torch.zeros_like(self.collapse_streak[class_index]),
                )
            )

        flat = self.prototypes.reshape(-1, self.feature_channels)
        cosine_all = flat @ flat.T
        norms = torch.linalg.vector_norm(self.prototypes, dim=2)
        bucket_counts = self._epoch_fire_buckets.clone()
        bucket_totals = bucket_counts.sum(dim=1, keepdim=True)
        bucket_ratios = torch.where(
            bucket_totals > 0,
            bucket_counts.float() / bucket_totals.clamp_min(1),
            torch.zeros_like(bucket_counts, dtype=torch.float32),
        )
        persistent_dead = self.dead_streak >= self.persistence_epochs
        persistent_collapse = self.collapse_streak >= self.persistence_epochs
        return {
            "prototype_hits": self._epoch_hits.tolist(),
            "prototype_norms": norms.cpu().tolist(),
            "initialized": self.initialized.cpu().tolist(),
            "ema_update_counts": self.ema_update_counts.cpu().tolist(),
            "cosine_matrix_all": cosine_all.cpu().tolist(),
            "cosine_matrix_intra_class": [
                (
                    self.prototypes[class_index]
                    @ self.prototypes[class_index].T
                ).cpu().tolist()
                for class_index in range(self.num_classes)
            ],
            "fire_area_bucket_names": ["small_le_4", "medium_5_64", "large_gt_64"],
            "fire_area_bucket_assignment_counts": bucket_counts.tolist(),
            "fire_area_bucket_assignment_ratios": bucket_ratios.tolist(),
            "dead_streak": self.dead_streak.cpu().tolist(),
            "collapse_streak": self.collapse_streak.cpu().tolist(),
            "persistent_dead": persistent_dead.cpu().tolist(),
            "persistent_collapse": persistent_collapse.cpu().tolist(),
            "valid_multi_prototype_method": not bool(
                persistent_dead.any() or persistent_collapse.any()
            ),
        }


class EMAPrototypeTotalLoss:
    """Original PIDNet objective plus a training-only EMA prototype bank."""

    objective_name = "ema_mproto"

    def __init__(self, base_criterion: object, config: dict) -> None:
        self.base_criterion = base_criterion
        self.prototype_weight = float(config.get("PROTOTYPE_LOSS_WEIGHT", 0.05))
        self.validation_seed = int(config.get("PROTOTYPE_VALIDATION_SEED", 12007))
        self.training_seed = int(config["SEED"])
        self.bank = EMAPrototypeBank(
            num_classes=int(config["NUM_CLASSES"]),
            feature_channels=int(config.get("PROTOTYPE_FEATURE_CHANNELS", 128)),
            prototypes_per_class=int(config.get("PROTOTYPES_PER_CLASS", 3)),
            temperature=float(config.get("PROTOTYPE_TEMPERATURE", 0.2)),
            momentum=float(config.get("PROTOTYPE_EMA_MOMENTUM", 0.99)),
            max_samples_per_class=int(
                config.get("PROTOTYPE_MAX_SAMPLES_PER_CLASS", 512)
            ),
            fire_class=int(config.get("FIRE_CLASS_INDEX", 2)),
            fire_sampling=str(config.get("FIRE_PROTOTYPE_SAMPLING", "uniform")),
            decorrelation=bool(config.get("PROTOTYPE_DECORRELATION", False)),
            decorrelation_weight=float(
                config.get("PROTOTYPE_DECORRELATION_WEIGHT", 0.1)
            ),
            decorrelation_min_samples=int(
                config.get("PROTOTYPE_DECORRELATION_MIN_SAMPLES", 2)
            ),
            seed=self.training_seed,
        ).to(torch.device(config["DEVICE"]))

    def state_dict(self) -> dict:
        return self.bank.state_dict()

    def load_state_dict(self, state_dict: dict) -> object:
        return self.bank.load_state_dict(state_dict, strict=True)

    def begin_epoch(self) -> None:
        self.bank.begin_epoch()

    def finalize_epoch_health(self) -> dict:
        return self.bank.finalize_epoch_health()

    def current_weight(
        self,
        epoch: int,
        batch_index: int,
        epoch_batches: int,
        training: bool,
    ) -> float:
        if epoch <= 0:
            return 0.0
        if epoch == 1 and training:
            return self.prototype_weight * (
                batch_index / max(epoch_batches - 1, 1)
            )
        return self.prototype_weight

    def get_loss(
        self,
        outputs: list[torch.Tensor] | tuple[torch.Tensor, ...],
        labels: torch.Tensor,
        edges: torch.Tensor,
        *,
        component_maps: torch.Tensor,
        sampling_labels: torch.Tensor,
        epoch: int,
        batch_index: int,
        epoch_batches: int,
        training: bool,
    ) -> tuple[torch.Tensor, list[torch.Tensor], object, dict[str, torch.Tensor]]:
        if not isinstance(outputs, (list, tuple)) or len(outputs) != 4:
            raise RuntimeError(
                "EMA multi-prototype training expects three PIDNet outputs "
                "and the DFM fused feature."
            )
        base_outputs = list(outputs[:3])
        fused_feature = outputs[3]
        losses, metric_outputs, accuracy, base_parts = self.base_criterion.get_loss(
            base_outputs,
            labels,
            edges,
        )
        sampling_seed = (
            self.training_seed + epoch * 1_000_003 + batch_index
            if training
            else self.validation_seed + batch_index
        )
        total, contrastive, decorrelation, diagnostics = self.bank(
            fused_feature,
            sampling_labels,
            component_maps,
            sampling_seed=sampling_seed,
            update_ema=training,
            collect_health=training,
        )
        weight = self.current_weight(
            epoch, batch_index, epoch_batches, training
        )
        weighted = weight * total
        losses = losses + weighted
        components = {
            "base_total": losses.mean() - weighted,
            "semantic": base_parts[0].mean(),
            "boundary": base_parts[1].mean(),
            "prototype_total": total,
            "prototype_contrastive": contrastive,
            "prototype_decorrelation": decorrelation,
            "prototype_weight": fused_feature.new_tensor(weight),
            "weighted_prototype": weighted,
        }
        components.update(
            {
                name: fused_feature.new_tensor(value)
                for name, value in diagnostics.items()
            }
        )
        return losses, metric_outputs, accuracy, components


__all__ = ["EMAPrototypeBank", "EMAPrototypeTotalLoss", "SampledFeatures"]
