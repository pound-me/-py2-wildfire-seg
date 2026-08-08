from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterator, Sequence

import torch
from torch.utils.data import Sampler


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _read_csv(path: Path) -> list[dict[str, str]]:
    last_error: UnicodeDecodeError | None = None
    for encoding in ("utf-8-sig", "gb18030"):
        try:
            with path.open("r", encoding=encoding, newline="") as handle:
                return list(csv.DictReader(handle))
        except UnicodeDecodeError as error:
            last_error = error
    assert last_error is not None
    raise last_error


def _resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


class FixedRatioSubsetSampler(Sampler[int]):
    """Draw an exact target/non-target ratio with replacement each epoch."""

    def __init__(
        self,
        target_indices: Sequence[int],
        other_indices: Sequence[int],
        target_ratio: float,
        num_samples: int,
        generator: torch.Generator,
    ) -> None:
        self.target_indices = torch.as_tensor(list(target_indices), dtype=torch.int64)
        self.other_indices = torch.as_tensor(list(other_indices), dtype=torch.int64)
        self.target_ratio = float(target_ratio)
        self.num_samples = int(num_samples)
        self.generator = generator
        self.dataset_size = self.num_samples
        self.batch_size = 1
        if self.target_indices.numel() == 0:
            raise ValueError("FixedRatioSubsetSampler requires target indices")
        if self.other_indices.numel() == 0:
            raise ValueError("FixedRatioSubsetSampler requires non-target indices")
        if not 0.0 < self.target_ratio < 1.0:
            raise ValueError("target_ratio must be strictly between 0 and 1")
        if self.num_samples <= 1:
            raise ValueError("num_samples must be greater than 1")
        self.target_draws = min(
            max(int(round(self.num_samples * self.target_ratio)), 1),
            self.num_samples - 1,
        )
        self.other_draws = self.num_samples - self.target_draws

    def __iter__(self) -> Iterator[int]:
        target_local = torch.randint(
            self.target_indices.numel(),
            (self.target_draws,),
            generator=self.generator,
        )
        other_local = torch.randint(
            self.other_indices.numel(),
            (self.other_draws,),
            generator=self.generator,
        )
        combined = torch.cat(
            (self.target_indices[target_local], self.other_indices[other_local])
        )
        order = torch.randperm(self.num_samples, generator=self.generator)
        yield from combined[order].tolist()

    def __len__(self) -> int:
        return self.num_samples

    def summary(self) -> dict[str, object]:
        return {
            "sampler": "fixed_ratio_subset_with_replacement",
            "dataset_images": self.dataset_size,
            "batch_size": self.batch_size,
            "batches_per_epoch": self.num_samples // self.batch_size,
            "samples_drawn_per_epoch": self.num_samples,
            "drop_last_remainder_excluded_before_sampling": (
                self.dataset_size - self.num_samples
            ),
            "target_pool_images": int(self.target_indices.numel()),
            "non_target_pool_images": int(self.other_indices.numel()),
            "target_draws_per_epoch": self.target_draws,
            "non_target_draws_per_epoch": self.other_draws,
            "requested_target_ratio": self.target_ratio,
            "realized_target_ratio": self.target_draws / self.num_samples,
        }


def build_flame3_train_sampler(
    dataset,
    config: dict,
    generator: torch.Generator,
) -> FixedRatioSubsetSampler | None:
    if not bool(config.get("FLAME3_TARGETED_SAMPLING_ENABLED", False)):
        return None
    if str(config.get("DATASET_TYPE", "")).lower() != "flame3_csv":
        raise ValueError("FLAME3 targeted sampling requires DATASET_TYPE=flame3_csv")
    manifest_value = config.get("FLAME3_TARGETED_SAMPLE_KEYS_CSV")
    if not manifest_value:
        raise ValueError("FLAME3_TARGETED_SAMPLE_KEYS_CSV is required")
    target_categories = {
        str(value)
        for value in config.get("FLAME3_TARGETED_SAMPLE_CATEGORIES", [])
    }
    if not target_categories:
        raise ValueError("FLAME3_TARGETED_SAMPLE_CATEGORIES cannot be empty")
    target_ratio = float(config.get("FLAME3_TARGETED_SAMPLING_RATIO", 0.0))
    manifest_path = _resolve_project_path(str(manifest_value))
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    rows = _read_csv(manifest_path)
    required_fields = {"sample_key", "split", "selection_category"}
    if not rows or not required_fields.issubset(rows[0]):
        raise RuntimeError(
            f"Targeted sampling CSV must contain {sorted(required_fields)}"
        )
    target_keys = {
        row["sample_key"]
        for row in rows
        if row["split"] == "train"
        and row["selection_category"] in target_categories
    }
    if not target_keys:
        raise RuntimeError("No train target keys matched the requested categories")
    dataset_rows = getattr(dataset, "rows", None)
    if dataset_rows is None:
        raise TypeError("FLAME3 targeted sampling requires dataset.rows")
    dataset_keys = [row["sample_key"] for row in dataset_rows]
    if len(dataset_keys) != len(set(dataset_keys)):
        raise RuntimeError("Dataset sample keys are not unique")
    unknown = target_keys.difference(dataset_keys)
    if unknown:
        raise RuntimeError(
            f"Targeted sampling keys are outside the train dataset: {sorted(unknown)[:5]}"
        )
    target_indices = [
        index for index, sample_key in enumerate(dataset_keys) if sample_key in target_keys
    ]
    other_indices = [
        index for index, sample_key in enumerate(dataset_keys) if sample_key not in target_keys
    ]
    batch_size = int(config.get("BATCHSIZE", 0))
    if batch_size <= 0:
        raise ValueError("BATCHSIZE must be positive for fixed-ratio sampling")
    samples_per_epoch = (len(dataset) // batch_size) * batch_size
    if samples_per_epoch <= 1:
        raise ValueError("Dataset is too small for the configured batch size")
    sampler = FixedRatioSubsetSampler(
        target_indices=target_indices,
        other_indices=other_indices,
        target_ratio=target_ratio,
        num_samples=samples_per_epoch,
        generator=generator,
    )
    sampler.dataset_size = len(dataset)
    sampler.batch_size = batch_size
    sampler.target_manifest_path = manifest_path
    sampler.target_categories = sorted(target_categories)
    sampler.target_sample_keys = sorted(target_keys)
    return sampler


def sampler_audit_summary(sampler: FixedRatioSubsetSampler | None) -> dict | None:
    if sampler is None:
        return None
    return {
        **sampler.summary(),
        "target_manifest_path": str(sampler.target_manifest_path),
        "target_categories": sampler.target_categories,
        "target_sample_key_count": len(sampler.target_sample_keys),
        "test_images_or_labels_read": False,
    }


__all__ = [
    "FixedRatioSubsetSampler",
    "build_flame3_train_sampler",
    "sampler_audit_summary",
]
