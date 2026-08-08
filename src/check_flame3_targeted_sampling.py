from __future__ import annotations

import argparse
import copy
import csv
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from baseline_runtime import build_dataset, load_config
from flame3_sampling import build_flame3_train_sampler, sampler_audit_summary


TARGET_CATEGORIES = [
    "small_or_weak_fire_core",
    "boundary_complex_fire_core",
    "suspected_label_gap_hot_noncore",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Engineer-check the single FLAME3 data-side candidate without training."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root-dataset", type=Path, required=True)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--val-csv", type=Path, required=True)
    parser.add_argument("--target-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=200)
    parser.add_argument("--crop-trials", type=int, default=100)
    parser.add_argument("--engineering-target-ratio", type=float, default=0.30)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    args = parse_args()
    for path in (
        args.config,
        args.root_dataset,
        args.train_csv,
        args.val_csv,
        args.target_csv,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.crop_trials <= 0:
        raise ValueError("crop-trials must be positive")

    base = load_config(args.config.resolve())
    base["ROOTDATASET"] = str(args.root_dataset.resolve())
    base["TRAINSET"] = str(args.train_csv.resolve())
    base["VALIDSET"] = str(args.val_csv.resolve())
    base["NUM_WORKERS"] = 0
    base["BRIGHTNESS"] = False
    base.pop("FLAME3_PROTECT_FIRE_CORE_CROP", None)
    base.pop("FLAME3_TARGETED_SAMPLING_ENABLED", None)
    baseline_dataset = build_dataset(base, split="train")
    if baseline_dataset.protect_fire_core_crop:
        raise RuntimeError("Baseline dataset unexpectedly enables protected cropping")

    candidate = copy.deepcopy(base)
    candidate.update(
        {
            "BATCHSIZE": 8,
            "FLAME3_PROTECT_FIRE_CORE_CROP": True,
            "FLAME3_FIRE_CORE_CROP_MIN_PIXELS": 32,
            "FLAME3_FIRE_CORE_CROP_ATTEMPTS": 64,
            "FLAME3_TARGETED_SAMPLING_ENABLED": True,
            "FLAME3_TARGETED_SAMPLE_KEYS_CSV": str(args.target_csv.resolve()),
            "FLAME3_TARGETED_SAMPLE_CATEGORIES": TARGET_CATEGORIES,
            "FLAME3_TARGETED_SAMPLING_RATIO": args.engineering_target_ratio,
        }
    )
    train_dataset = build_dataset(candidate, split="train")
    validation_dataset = build_dataset(candidate, split="val")
    if not train_dataset.protect_fire_core_crop:
        raise RuntimeError("Candidate training dataset did not enable protected cropping")
    if validation_dataset.protect_fire_core_crop:
        raise RuntimeError("Validation must not use protected random cropping")

    target_rows = read_csv(args.target_csv.resolve())
    weak_keys = [
        row["sample_key"]
        for row in target_rows
        if row["split"] == "train"
        and row["selection_category"] == "small_or_weak_fire_core"
    ]
    dataset_index = {
        row["sample_key"]: index for index, row in enumerate(train_dataset.rows)
    }
    sample_key = next(key for key in weak_keys if key in dataset_index)
    sample_index = dataset_index[sample_key]
    images, label, _, _ = train_dataset._load_raw(sample_index)
    edge = cv2.Canny(label, 0.1, 0.2)

    scaled_images = [
        cv2.resize(image, (960, 768), interpolation=cv2.INTER_LINEAR).reshape(
            768, 960, image.shape[-1]
        )
        for image in images
    ]
    scaled_label = cv2.resize(label, (960, 768), interpolation=cv2.INTER_NEAREST)
    scaled_edge = cv2.resize(edge, (960, 768), interpolation=cv2.INTER_NEAREST)
    random.seed(args.seed)
    delegated = baseline_dataset.rand_crop(
        [image.copy() for image in scaled_images],
        scaled_label.copy(),
        scaled_edge.copy(),
    )
    random.seed(args.seed)
    reference = super(type(baseline_dataset), baseline_dataset).rand_crop(
        [image.copy() for image in scaled_images],
        scaled_label.copy(),
        scaled_edge.copy(),
    )
    baseline_delegate_identical = all(
        np.array_equal(left, right)
        for left, right in zip(delegated[0], reference[0])
    ) and np.array_equal(delegated[1], reference[1]) and np.array_equal(
        delegated[2], reference[2]
    )
    if not baseline_delegate_identical:
        raise RuntimeError("Disabled protected-crop path changed baseline random cropping")
    retained_counts: list[int] = []
    for trial in range(args.crop_trials):
        random.seed(args.seed + trial)
        np.random.seed(args.seed + trial)
        transformed = train_dataset.multi_scale_aug(
            [image.copy() for image in images],
            label.copy(),
            edge.copy(),
            rand_scale=1.5,
            rand_crop=True,
        )
        _, cropped_label, _ = transformed
        retained = int(np.count_nonzero(cropped_label == 2))
        if retained <= 0:
            raise RuntimeError(
                f"Protected crop lost all Fire-core pixels on trial {trial}"
            )
        retained_counts.append(retained)

    generator = torch.Generator().manual_seed(args.seed)
    sampler = build_flame3_train_sampler(train_dataset, candidate, generator)
    if sampler is None:
        raise RuntimeError("Targeted sampler was not constructed")
    target_index_set = set(sampler.target_indices.tolist())
    epoch_target_draws: list[int] = []
    for _ in range(5):
        indices = list(iter(sampler))
        epoch_target_draws.append(sum(index in target_index_set for index in indices))
    if any(count != sampler.target_draws for count in epoch_target_draws):
        raise RuntimeError("Fixed-ratio sampler did not realize the exact target draw count")

    generator_a = torch.Generator().manual_seed(args.seed)
    generator_b = torch.Generator().manual_seed(args.seed)
    sampler_a = build_flame3_train_sampler(train_dataset, candidate, generator_a)
    sampler_b = build_flame3_train_sampler(train_dataset, candidate, generator_b)
    deterministic_first_epoch = list(iter(sampler_a)) == list(iter(sampler_b))
    if not deterministic_first_epoch:
        raise RuntimeError("Fixed-ratio sampler is not deterministic for the same seed")

    resume_source_generator = torch.Generator().manual_seed(args.seed)
    resume_source_sampler = build_flame3_train_sampler(
        train_dataset,
        candidate,
        resume_source_generator,
    )
    list(iter(resume_source_sampler))
    saved_generator_state = resume_source_generator.get_state()
    expected_resumed_epoch = list(iter(resume_source_sampler))
    resumed_generator = torch.Generator()
    resumed_generator.set_state(saved_generator_state)
    resumed_sampler = build_flame3_train_sampler(
        train_dataset,
        candidate,
        resumed_generator,
    )
    resumed_epoch = list(iter(resumed_sampler))
    exact_generator_state_resume = expected_resumed_epoch == resumed_epoch
    if not exact_generator_state_resume:
        raise RuntimeError("Sampler sequence did not restore from generator state")

    loader_generator = torch.Generator().manual_seed(args.seed)
    loader_sampler = build_flame3_train_sampler(
        train_dataset,
        candidate,
        loader_generator,
    )
    loader = DataLoader(
        train_dataset,
        batch_size=candidate["BATCHSIZE"],
        shuffle=False,
        sampler=loader_sampler,
        num_workers=0,
        drop_last=True,
        generator=loader_generator,
    )
    images_batch, labels_batch, edges_batch, sample_keys, is_fire_folder = next(
        iter(loader)
    )
    if tuple(images_batch.shape) != (8, 4, 512, 640):
        raise RuntimeError(f"Unexpected image batch shape: {tuple(images_batch.shape)}")

    sample_key_batch_items = (
        len(sample_keys[0])
        if len(sample_keys) == 1 and isinstance(sample_keys[0], (list, tuple))
        else len(sample_keys)
    )
    if sample_key_batch_items != candidate["BATCHSIZE"]:
        raise RuntimeError(f"Unexpected sample-key batch collation: {sample_keys!r}")

    result = {
        "protocol": "flame3_single_data_side_candidate_engineering_only",
        "formal_training_started": False,
        "ratio_status": "provisional_engineering_value_pending_manual_semantic_audit",
        "baseline_default_path_unchanged": baseline_delegate_identical,
        "validation_augmentation_unchanged": True,
        "protected_crop": {
            "sample_key": sample_key,
            "trials": args.crop_trials,
            "minimum_retained_fire_pixels": min(retained_counts),
            "maximum_retained_fire_pixels": max(retained_counts),
            "all_trials_retain_fire": True,
            "configured_min_pixels": candidate["FLAME3_FIRE_CORE_CROP_MIN_PIXELS"],
            "configured_attempts": candidate["FLAME3_FIRE_CORE_CROP_ATTEMPTS"],
        },
        "targeted_sampling": {
            **sampler_audit_summary(sampler),
            "five_epoch_target_draws": epoch_target_draws,
            "same_seed_first_epoch_identical": deterministic_first_epoch,
            "exact_generator_state_resume": exact_generator_state_resume,
        },
        "batch_smoke": {
            "images": list(images_batch.shape),
            "labels": list(labels_batch.shape),
            "edges": list(edges_batch.shape),
            "sample_key_batch_items": sample_key_batch_items,
            "is_fire_folder": list(is_fire_folder.shape),
        },
        "other_data_changes_enabled": {
            "rgb_ir_separate_standardization": False,
            "thermal_dynamic_range_enhancement": False,
            "registration_perturbation": False,
        },
        "test_images_or_labels_read": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
