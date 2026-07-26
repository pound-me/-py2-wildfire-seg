from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from baseline_runtime import PROJECT_ROOT, build_dataset, load_config
from evaluate_baseline import (
    finalize_boundary_statistics,
    update_boundary_statistics,
)
from train_baseline import metrics_from_confusion, seed_everything


def semantic_boundary(labels: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Return two-sided GT semantic boundaries without treating image edges as boundaries."""
    boundary = np.zeros(labels.shape, dtype=bool)
    vertical = (
        (labels[1:, :] != labels[:-1, :])
        & valid[1:, :]
        & valid[:-1, :]
    )
    horizontal = (
        (labels[:, 1:] != labels[:, :-1])
        & valid[:, 1:]
        & valid[:, :-1]
    )
    boundary[1:, :] |= vertical
    boundary[:-1, :] |= vertical
    boundary[:, 1:] |= horizontal
    boundary[:, :-1] |= horizontal
    return boundary


def dilate_square(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.copy()
    size = 2 * radius + 1
    kernel = np.ones((size, size), dtype=np.uint8)
    return cv2.dilate(mask.astype(np.uint8), kernel, iterations=1) > 0


def confusion_for_mask(
    labels: np.ndarray,
    predictions: np.ndarray,
    mask: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    indices = labels[mask].astype(np.int64) * num_classes
    indices += predictions[mask].astype(np.int64)
    return np.bincount(
        indices,
        minlength=num_classes * num_classes,
    ).reshape(num_classes, num_classes)


def safe_ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def write_matrix_csv(
    path: Path,
    matrix: np.ndarray,
    class_names: list[str],
    value_format: str,
) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["true\\predicted", *class_names])
        for class_name, row in zip(class_names, matrix):
            writer.writerow(
                [class_name, *[value_format.format(value) for value in row]]
            )


def write_rows(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def extract_name(raw_name: object) -> str:
    if isinstance(raw_name, (list, tuple)) and raw_name:
        return str(raw_name[0])
    return str(raw_name)


def prediction_path(evaluation_dir: Path, name: str) -> Path:
    safe_stem = Path(name).stem.replace("XXX", "rgb")
    return evaluation_dir / "predictions_raw" / f"{safe_stem}_pred.png"


def class_profile(
    class_index: int,
    class_names: list[str],
    confusion: np.ndarray,
    boundary_confusion: np.ndarray,
    interior_confusion: np.ndarray,
    distance_errors: dict[str, np.ndarray],
) -> dict:
    class_name = class_names[class_index]
    total = int(confusion[class_index].sum())
    correct = int(confusion[class_index, class_index])
    errors = total - correct
    boundary_total = int(boundary_confusion[class_index].sum())
    boundary_correct = int(boundary_confusion[class_index, class_index])
    boundary_errors = boundary_total - boundary_correct
    interior_total = int(interior_confusion[class_index].sum())
    interior_correct = int(interior_confusion[class_index, class_index])
    interior_errors = interior_total - interior_correct
    predicted_as = {
        class_names[predicted_index]: {
            "pixels": int(confusion[class_index, predicted_index]),
            "fraction_of_true_class": safe_ratio(
                int(confusion[class_index, predicted_index]),
                total,
            ),
            "fraction_of_class_errors": (
                safe_ratio(int(confusion[class_index, predicted_index]), errors)
                if predicted_index != class_index
                else 0.0
            ),
        }
        for predicted_index in range(len(class_names))
    }
    return {
        "class_index": class_index,
        "class_name": class_name,
        "true_pixels": total,
        "correct_pixels": correct,
        "error_pixels": errors,
        "error_rate": safe_ratio(errors, total),
        "recall": safe_ratio(correct, total),
        "boundary_band": {
            "pixels": boundary_total,
            "fraction_of_true_class": safe_ratio(boundary_total, total),
            "error_pixels": boundary_errors,
            "error_rate": safe_ratio(boundary_errors, boundary_total),
            "share_of_class_errors": safe_ratio(boundary_errors, errors),
        },
        "interior": {
            "pixels": interior_total,
            "fraction_of_true_class": safe_ratio(interior_total, total),
            "error_pixels": interior_errors,
            "error_rate": safe_ratio(interior_errors, interior_total),
            "share_of_class_errors": safe_ratio(interior_errors, errors),
        },
        "error_distance_from_gt_boundary": {
            bin_name: int(values[class_index])
            for bin_name, values in distance_errors.items()
        },
        "predicted_as": predicted_as,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Profile the frozen Fusion validation predictions into confusion, "
            "GT-boundary-band and region-interior errors."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=(
            PROJECT_ROOT
            / "configs"
            / "route_a"
            / "pidnet_s_fusion_100e_label_fix.yaml"
        ),
    )
    parser.add_argument(
        "--evaluation-dir",
        type=Path,
        default=(
            PROJECT_ROOT
            / "experiments"
            / "route_a_pidnet_s_fusion"
            / "route_a_fusion_100e_label_fix_seed200"
            / "val_best"
        ),
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--boundary-tolerance", type=int)
    args = parser.parse_args()

    config_path = args.config.resolve()
    evaluation_dir = args.evaluation_dir.resolve()
    config = load_config(config_path)
    if config["MODE"] != "fusion":
        raise ValueError("Fusion error profiling requires MODE: fusion.")
    seed_everything(int(config["SEED"]))
    dataset = build_dataset(config, "val")
    existing_metrics_path = evaluation_dir / "metrics.json"
    existing_metrics = json.loads(
        existing_metrics_path.read_text(encoding="utf-8")
    )
    if existing_metrics.get("split") != "val":
        raise ValueError("The reference evaluation must be the validation split.")
    if int(existing_metrics.get("sample_count", -1)) != len(dataset):
        raise RuntimeError("Reference evaluation sample count does not match dataset.")

    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else evaluation_dir / "error_profile"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    tolerance = (
        max(int(args.boundary_tolerance), 0)
        if args.boundary_tolerance is not None
        else max(int(config.get("BOUNDARY_TOLERANCE", 3)), 0)
    )
    num_classes = int(config["NUM_CLASSES"])
    class_names = [str(name) for name in config["CLS_NAMES"]]
    ignore_label = int(config["IGNORE_LABEL"])

    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    boundary_confusion = np.zeros_like(confusion)
    interior_confusion = np.zeros_like(confusion)
    distance_errors = {
        "distance_0_to_1_px": np.zeros(num_classes, dtype=np.int64),
        "distance_2_to_3_px": np.zeros(num_classes, dtype=np.int64),
        "distance_4_to_8_px": np.zeros(num_classes, dtype=np.int64),
        "distance_over_8_px": np.zeros(num_classes, dtype=np.int64),
    }
    boundary_statistics = {
        class_name: {
            "matched_prediction": 0,
            "prediction": 0,
            "matched_target": 0,
            "target": 0,
        }
        for class_index, class_name in enumerate(class_names)
        if class_index > 0
    }
    per_image_rows: list[dict] = []

    for index in range(len(dataset)):
        sample = dataset[index]
        labels = np.asarray(sample[1], dtype=np.int64)
        name = extract_name(sample[3])
        saved_prediction_path = prediction_path(evaluation_dir, name)
        if not saved_prediction_path.is_file():
            raise FileNotFoundError(
                f"Saved validation prediction missing: {saved_prediction_path}"
            )
        predictions = np.asarray(Image.open(saved_prediction_path), dtype=np.int64)
        if predictions.shape != labels.shape:
            raise RuntimeError(
                f"Prediction/label shape mismatch for {name}: "
                f"{predictions.shape} vs {labels.shape}"
            )
        valid = labels != ignore_label
        if ((labels[valid] < 0) | (labels[valid] >= num_classes)).any():
            raise RuntimeError(f"Invalid class id in validation label: {name}")
        if ((predictions < 0) | (predictions >= num_classes)).any():
            raise RuntimeError(f"Invalid class id in prediction: {name}")

        image_confusion = confusion_for_mask(
            labels,
            predictions,
            valid,
            num_classes,
        )
        confusion += image_confusion
        gt_boundary = semantic_boundary(labels, valid)
        band_1 = dilate_square(gt_boundary, 1) & valid
        band_3 = dilate_square(gt_boundary, tolerance) & valid
        band_8 = dilate_square(gt_boundary, 8) & valid
        interior = valid & ~band_3
        image_boundary_confusion = confusion_for_mask(
            labels,
            predictions,
            band_3,
            num_classes,
        )
        image_interior_confusion = confusion_for_mask(
            labels,
            predictions,
            interior,
            num_classes,
        )
        boundary_confusion += image_boundary_confusion
        interior_confusion += image_interior_confusion
        if not np.array_equal(
            image_confusion,
            image_boundary_confusion + image_interior_confusion,
        ):
            raise RuntimeError(f"Boundary/interior partition failed for: {name}")

        error = valid & (predictions != labels)
        distance_masks = {
            "distance_0_to_1_px": error & band_1,
            "distance_2_to_3_px": error & band_3 & ~band_1,
            "distance_4_to_8_px": error & band_8 & ~band_3,
            "distance_over_8_px": error & ~band_8,
        }
        for bin_name, mask in distance_masks.items():
            distance_errors[bin_name] += np.bincount(
                labels[mask],
                minlength=num_classes,
            )[:num_classes]

        prediction_tensor = torch.from_numpy(predictions).unsqueeze(0)
        label_tensor = torch.from_numpy(labels).unsqueeze(0)
        update_boundary_statistics(
            boundary_statistics,
            prediction_tensor,
            label_tensor,
            class_names,
            ignore_label,
            tolerance,
        )
        per_image_rows.append(
            {
                "index": index,
                "name": name,
                "valid_pixels": int(valid.sum()),
                "error_pixels": int(error.sum()),
                "boundary_error_pixels": int((error & band_3).sum()),
                "interior_error_pixels": int((error & ~band_3).sum()),
                "smoke_to_background": int(
                    ((labels == 1) & (predictions == 0) & valid).sum()
                ),
                "smoke_to_fire": int(
                    ((labels == 1) & (predictions == 2) & valid).sum()
                ),
                "fire_to_background": int(
                    ((labels == 2) & (predictions == 0) & valid).sum()
                ),
                "fire_to_smoke": int(
                    ((labels == 2) & (predictions == 1) & valid).sum()
                ),
            }
        )

    reference_confusion = np.asarray(
        existing_metrics["confusion_matrix"],
        dtype=np.int64,
    )
    if not np.array_equal(confusion, reference_confusion):
        raise RuntimeError("Recomputed confusion matrix differs from val_best metrics.")
    boundary_metrics = finalize_boundary_statistics(
        boundary_statistics,
        tolerance,
    )
    for class_name in boundary_statistics:
        reference = existing_metrics["boundary_metrics"][class_name]
        recomputed = boundary_metrics[class_name]
        for count_name in (
            "matched_prediction",
            "prediction",
            "matched_target",
            "target",
        ):
            if int(recomputed[count_name]) != int(reference[count_name]):
                raise RuntimeError(
                    f"Recomputed {class_name} boundary count differs: {count_name}"
                )

    row_totals = confusion.sum(axis=1, keepdims=True)
    row_normalized = np.divide(
        confusion,
        row_totals,
        out=np.zeros(confusion.shape, dtype=np.float64),
        where=row_totals > 0,
    )
    all_errors = int(confusion.sum() - np.trace(confusion))
    boundary_errors = int(
        boundary_confusion.sum() - np.trace(boundary_confusion)
    )
    interior_errors = int(
        interior_confusion.sum() - np.trace(interior_confusion)
    )
    class_profiles = {
        class_names[class_index]: class_profile(
            class_index,
            class_names,
            confusion,
            boundary_confusion,
            interior_confusion,
            distance_errors,
        )
        for class_index in range(num_classes)
    }
    confusion_pairs = sorted(
        (
            {
                "true_class": class_names[true_index],
                "predicted_class": class_names[predicted_index],
                "pixels": int(confusion[true_index, predicted_index]),
                "share_of_all_errors": safe_ratio(
                    int(confusion[true_index, predicted_index]),
                    all_errors,
                ),
            }
            for true_index in range(num_classes)
            for predicted_index in range(num_classes)
            if true_index != predicted_index
        ),
        key=lambda row: (-row["pixels"], row["true_class"], row["predicted_class"]),
    )
    profile = {
        "config": str(config_path),
        "evaluation_dir": str(evaluation_dir),
        "checkpoint": existing_metrics["checkpoint"],
        "checkpoint_epoch": existing_metrics["checkpoint_epoch"],
        "split": "val",
        "sample_count": len(dataset),
        "class_names": class_names,
        "boundary_tolerance_pixels": tolerance,
        "methodology": {
            "confusion_orientation": "rows=true class, columns=predicted class",
            "semantic_boundary": (
                "two-sided 4-neighbour class transitions in the ground-truth; "
                "image borders are not boundaries"
            ),
            "boundary_error": (
                "misclassified valid pixel within Chebyshev distance <= tolerance "
                "of a ground-truth semantic boundary"
            ),
            "interior_error": (
                "misclassified valid pixel farther than tolerance from every "
                "ground-truth semantic boundary"
            ),
            "class_boundary_f1": (
                "same per-class binary-boundary implementation and tolerance as "
                "evaluate_baseline.py"
            ),
        },
        "global": {
            "valid_pixels": int(confusion.sum()),
            "correct_pixels": int(np.trace(confusion)),
            "error_pixels": all_errors,
            "pixel_error_rate": safe_ratio(all_errors, int(confusion.sum())),
            "boundary_band_pixels": int(boundary_confusion.sum()),
            "boundary_error_pixels": boundary_errors,
            "boundary_error_rate": safe_ratio(
                boundary_errors,
                int(boundary_confusion.sum()),
            ),
            "boundary_share_of_all_errors": safe_ratio(
                boundary_errors,
                all_errors,
            ),
            "interior_pixels": int(interior_confusion.sum()),
            "interior_error_pixels": interior_errors,
            "interior_error_rate": safe_ratio(
                interior_errors,
                int(interior_confusion.sum()),
            ),
            "interior_share_of_all_errors": safe_ratio(
                interior_errors,
                all_errors,
            ),
        },
        "metrics_from_confusion": metrics_from_confusion(
            torch.from_numpy(confusion),
            class_names,
        ),
        "boundary_metrics": boundary_metrics,
        "confusion_matrix": confusion.tolist(),
        "confusion_matrix_row_normalized": row_normalized.tolist(),
        "boundary_confusion_matrix": boundary_confusion.tolist(),
        "interior_confusion_matrix": interior_confusion.tolist(),
        "class_profiles": class_profiles,
        "largest_confusion_pairs": confusion_pairs,
        "validation_checks": {
            "confusion_matches_saved_evaluation": True,
            "boundary_counts_match_saved_evaluation": True,
            "boundary_and_interior_partition_all_valid_pixels": True,
            "test_set_used": False,
        },
    }

    (output_dir / "error_profile.json").write_text(
        json.dumps(profile, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_matrix_csv(
        output_dir / "confusion_matrix_counts.csv",
        confusion,
        class_names,
        "{:d}",
    )
    write_matrix_csv(
        output_dir / "confusion_matrix_row_normalized.csv",
        row_normalized,
        class_names,
        "{:.9f}",
    )
    region_rows = []
    for class_name, values in class_profiles.items():
        region_rows.append(
            {
                "class": class_name,
                "true_pixels": values["true_pixels"],
                "error_pixels": values["error_pixels"],
                "error_rate": values["error_rate"],
                "boundary_pixels": values["boundary_band"]["pixels"],
                "boundary_error_pixels": values["boundary_band"]["error_pixels"],
                "boundary_error_rate": values["boundary_band"]["error_rate"],
                "boundary_share_of_class_errors": values["boundary_band"][
                    "share_of_class_errors"
                ],
                "interior_pixels": values["interior"]["pixels"],
                "interior_error_pixels": values["interior"]["error_pixels"],
                "interior_error_rate": values["interior"]["error_rate"],
                "interior_share_of_class_errors": values["interior"][
                    "share_of_class_errors"
                ],
            }
        )
    write_rows(
        output_dir / "region_error_summary.csv",
        region_rows,
        [
            "class",
            "true_pixels",
            "error_pixels",
            "error_rate",
            "boundary_pixels",
            "boundary_error_pixels",
            "boundary_error_rate",
            "boundary_share_of_class_errors",
            "interior_pixels",
            "interior_error_pixels",
            "interior_error_rate",
            "interior_share_of_class_errors",
        ],
    )
    per_image_rows.sort(
        key=lambda row: (-int(row["error_pixels"]), str(row["name"]))
    )
    write_rows(
        output_dir / "per_image_error_profile.csv",
        per_image_rows,
        [
            "index",
            "name",
            "valid_pixels",
            "error_pixels",
            "boundary_error_pixels",
            "interior_error_pixels",
            "smoke_to_background",
            "smoke_to_fire",
            "fire_to_background",
            "fire_to_smoke",
        ],
    )
    print(json.dumps(profile["global"], ensure_ascii=False, indent=2))
    print(
        "Boundary F1: "
        f"smoke={boundary_metrics['smoke']['boundary_f1']:.6f}, "
        f"fire={boundary_metrics['fire']['boundary_f1']:.6f}"
    )
    print(f"Result: {output_dir / 'error_profile.json'}")


if __name__ == "__main__":
    main()
