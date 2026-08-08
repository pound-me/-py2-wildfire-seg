from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import yaml


TARGET_CATEGORIES = {
    "small_or_weak_fire_core",
    "boundary_complex_fire_core",
    "suspected_label_gap_hot_noncore",
}
CONFIRMED_CATEGORY = "confirmed_weak_or_edge_active_fire"
SEMANTIC_CLASSES = {
    "active_fire",
    "residual_heat",
    "smoke_only",
    "no_fire",
    "mixed",
    "uncertain",
}
YES_NO_UNCERTAIN = {"yes", "no", "uncertain"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the completed FLAME3 semantic checklist and, only with an "
            "explicit ratio, freeze the single data-side candidate."
        )
    )
    parser.add_argument("--completed-checklist", type=Path, required=True)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-ratio", type=float)
    parser.add_argument("--dataset-size", type=int, default=493)
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    last_error: UnicodeDecodeError | None = None
    for encoding in ("utf-8-sig", "gb18030"):
        try:
            with path.open("r", encoding=encoding, newline="") as handle:
                return list(csv.DictReader(handle))
        except UnicodeDecodeError as error:
            last_error = error
    assert last_error is not None
    raise last_error


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized(row: dict[str, str], field: str) -> str:
    return str(row.get(field, "")).strip().lower()


def validate_choice(
    row: dict[str, str], field: str, allowed: set[str], errors: list[dict[str, str]]
) -> str:
    value = normalized(row, field)
    if not value:
        errors.append(
            {
                "annotation_id": row.get("annotation_id", ""),
                "sample_key": row.get("sample_key", ""),
                "field": field,
                "issue": "blank",
            }
        )
    elif value not in allowed:
        errors.append(
            {
                "annotation_id": row.get("annotation_id", ""),
                "sample_key": row.get("sample_key", ""),
                "field": field,
                "issue": f"invalid_value:{value}",
            }
        )
    return value


def main() -> None:
    args = parse_args()
    checklist = args.completed_checklist.resolve()
    base_config = args.base_config.resolve()
    output = args.output_dir.resolve()
    for path in (checklist, base_config):
        if not path.is_file():
            raise FileNotFoundError(path)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite finalization output: {output}")
    if args.dataset_size <= 1 or args.batch_size <= 0:
        raise ValueError("Invalid dataset-size or batch-size")
    if args.target_ratio is not None and not 0.0 < args.target_ratio < 1.0:
        raise ValueError("target-ratio must be strictly between 0 and 1")

    rows = read_csv(checklist)
    if len(rows) != 150:
        raise RuntimeError(f"Expected 150 checklist rows, got {len(rows)}")
    required = {
        "annotation_id",
        "split",
        "selection_category",
        "sample_key",
        "semantic_class",
        "counts_as_active_fire_yes_no_uncertain",
        "residual_heat_or_hot_ground_yes_no_uncertain",
        "pixel_mask_status",
    }
    if not required.issubset(rows[0]):
        raise RuntimeError(f"Checklist is missing fields: {sorted(required - rows[0].keys())}")
    if any(normalized(row, "split") == "test" for row in rows):
        raise RuntimeError("Completed checklist must not contain test rows")
    train_rows = [row for row in rows if normalized(row, "split") == "train"]
    val_rows = [row for row in rows if normalized(row, "split") == "val"]
    if len(train_rows) != 103 or len(val_rows) != 47:
        raise RuntimeError(
            f"Expected train/val review counts 103/47, got {len(train_rows)}/{len(val_rows)}"
        )
    target_candidates = [
        row
        for row in train_rows
        if row["selection_category"] in TARGET_CATEGORIES
    ]
    if len(target_candidates) != 60:
        raise RuntimeError(f"Expected 60 target candidates, got {len(target_candidates)}")

    missing_or_invalid: list[dict[str, str]] = []
    conflicts: list[dict[str, str]] = []
    confirmed_targets: list[dict[str, str]] = []
    reviewed_train = 0
    pixel_masks_completed = 0
    for row in train_rows:
        semantic = validate_choice(
            row,
            "semantic_class",
            SEMANTIC_CLASSES,
            missing_or_invalid,
        )
        active = validate_choice(
            row,
            "counts_as_active_fire_yes_no_uncertain",
            YES_NO_UNCERTAIN,
            missing_or_invalid,
        )
        residual = validate_choice(
            row,
            "residual_heat_or_hot_ground_yes_no_uncertain",
            YES_NO_UNCERTAIN,
            missing_or_invalid,
        )
        if semantic and active and residual:
            reviewed_train += 1
        if normalized(row, "pixel_mask_status") == "completed":
            pixel_masks_completed += 1
        if active == "yes" and semantic in {"residual_heat", "no_fire"}:
            conflicts.append(
                {
                    "annotation_id": row["annotation_id"],
                    "sample_key": row["sample_key"],
                    "issue": "active_yes_conflicts_with_semantic_class",
                }
            )
        if active == "no" and semantic in {"active_fire", "mixed"}:
            conflicts.append(
                {
                    "annotation_id": row["annotation_id"],
                    "sample_key": row["sample_key"],
                    "issue": "active_no_conflicts_with_semantic_class",
                }
            )
        if active == "yes" and residual == "yes":
            conflicts.append(
                {
                    "annotation_id": row["annotation_id"],
                    "sample_key": row["sample_key"],
                    "issue": "active_yes_and_residual_heat_yes",
                }
            )
        if (
            row["selection_category"] in TARGET_CATEGORIES
            and active == "yes"
            and residual == "no"
            and semantic in {"active_fire", "mixed"}
        ):
            confirmed_targets.append(row)

    samples_per_epoch = (args.dataset_size // args.batch_size) * args.batch_size
    ratio_candidates: list[dict[str, object]] = []
    for ratio in (0.20, 0.25, 0.30, 0.35):
        draws = int(round(samples_per_epoch * ratio))
        ratio_candidates.append(
            {
                "candidate_ratio": ratio,
                "target_draws_per_epoch": draws,
                "non_target_draws_per_epoch": samples_per_epoch - draws,
                "realized_ratio": draws / samples_per_epoch,
                "mean_draws_per_confirmed_target": (
                    draws / len(confirmed_targets) if confirmed_targets else None
                ),
            }
        )

    output.mkdir(parents=True)
    if missing_or_invalid:
        write_csv(output / "missing_or_invalid_fields.csv", missing_or_invalid)
    if conflicts:
        write_csv(output / "semantic_conflicts.csv", conflicts)
    write_csv(output / "sampling_ratio_candidates.csv", ratio_candidates)
    ready_for_freeze = (
        reviewed_train == len(train_rows)
        and not missing_or_invalid
        and not conflicts
        and bool(confirmed_targets)
    )
    summary = {
        "protocol": "flame3_semantic_review_to_single_data_candidate",
        "checklist": str(checklist),
        "checklist_sha256": sha256(checklist),
        "train_rows": len(train_rows),
        "validation_rows": len(val_rows),
        "reviewed_train_rows": reviewed_train,
        "target_candidate_rows": len(target_candidates),
        "confirmed_target_rows": len(confirmed_targets),
        "missing_or_invalid_field_count": len(missing_or_invalid),
        "semantic_conflict_count": len(conflicts),
        "pixel_masks_completed": pixel_masks_completed,
        "sampling_ratio_status": (
            "explicit_ratio_provided" if args.target_ratio is not None else "not_frozen"
        ),
        "ready_for_freeze": ready_for_freeze,
        "formal_training_started": False,
        "test_images_or_labels_read": False,
    }
    (output / "review_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    if args.target_ratio is None:
        print(json.dumps(summary, ensure_ascii=False))
        return
    if not ready_for_freeze:
        raise RuntimeError(
            "An explicit target ratio was provided, but the semantic review is not ready "
            "for freezing. Inspect the generated error CSV files."
        )

    target_pool_rows = [
        {
            "sample_key": row["sample_key"],
            "split": "train",
            "selection_category": CONFIRMED_CATEGORY,
            "source_annotation_id": row["annotation_id"],
            "original_selection_category": row["selection_category"],
            "semantic_class": normalized(row, "semantic_class"),
            "counts_as_active_fire": normalized(
                row, "counts_as_active_fire_yes_no_uncertain"
            ),
            "residual_heat_or_hot_ground": normalized(
                row, "residual_heat_or_hot_ground_yes_no_uncertain"
            ),
        }
        for row in confirmed_targets
    ]
    target_pool_path = output / "confirmed_target_pool.csv"
    write_csv(target_pool_path, target_pool_rows)

    with base_config.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config.update(
        {
            "PROJECTNAME": "FLAME3_PIDNETS_FUSION_TARGETED_SAMPLING",
            "SESSIONAME": "flame3_fusion_targeted_sampling_screening",
            "EXPERIMENT_GROUP": "flame3_pidnet_s_fusion_targeted_sampling",
            "BATCHSIZE": args.batch_size,
            "EPOCHS": 30,
            "LR_TOTAL_EPOCHS": 100,
            "FLAME3_PROTECT_FIRE_CORE_CROP": True,
            "FLAME3_FIRE_CORE_CROP_MIN_PIXELS": 32,
            "FLAME3_FIRE_CORE_CROP_ATTEMPTS": 64,
            "FLAME3_TARGETED_SAMPLING_ENABLED": True,
            "FLAME3_TARGETED_SAMPLE_KEYS_CSV": str(target_pool_path),
            "FLAME3_TARGETED_SAMPLE_CATEGORIES": [CONFIRMED_CATEGORY],
            "FLAME3_TARGETED_SAMPLING_RATIO": args.target_ratio,
            "FLAME3_RGB_IR_SEPARATE_STANDARDIZATION": False,
            "FLAME3_THERMAL_DYNAMIC_RANGE_ENHANCEMENT": False,
            "FLAME3_REGISTRATION_PERTURBATION": False,
            "TEST_SET_POLICY": "sealed_until_final_method_freeze",
        }
    )
    config_path = output / "pidnet_s_fusion_targeted_sampling_partial_30e.yaml"
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)

    target_draws = int(round(samples_per_epoch * args.target_ratio))
    preregistration = {
        **summary,
        "status": "frozen_before_formal_training",
        "target_ratio": args.target_ratio,
        "dataset_images": args.dataset_size,
        "batch_size": args.batch_size,
        "samples_drawn_per_epoch": samples_per_epoch,
        "target_draws_per_epoch": target_draws,
        "non_target_draws_per_epoch": samples_per_epoch - target_draws,
        "realized_target_ratio": target_draws / samples_per_epoch,
        "confirmed_target_pool": str(target_pool_path),
        "confirmed_target_pool_sha256": sha256(target_pool_path),
        "frozen_config": str(config_path),
        "base_config": str(base_config),
        "base_config_sha256": sha256(base_config),
        "comparison": "Fusion baseline versus this single data-side candidate only",
        "test_images_or_labels_read": False,
    }
    (output / "sampling_preregistration.json").write_text(
        json.dumps(preregistration, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(preregistration, ensure_ascii=False))


if __name__ == "__main__":
    main()
