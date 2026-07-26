from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from baseline_runtime import PROJECT_ROOT, build_dataset, load_config
from profile_fusion_errors import dilate_square, extract_name, prediction_path, safe_ratio
from train_baseline import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Attribute frozen Fusion validation Fire FN/FP pixels to small "
            "components and Smoke regions/neighbourhoods."
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
    parser.add_argument("--neighbourhood-radius", type=int, default=3)
    return parser.parse_args()


def connected_components(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    count, component_map = cv2.connectedComponents(
        mask.astype(np.uint8), connectivity=8
    )
    areas = np.bincount(component_map.reshape(-1), minlength=count).astype(np.int64)
    return component_map, areas


def strict_lower_quartile(values: list[int]) -> float:
    if not values:
        raise RuntimeError("No GT Fire connected component exists in validation.")
    array = np.asarray(values, dtype=np.float64)
    try:
        return float(np.quantile(array, 0.25, method="linear"))
    except TypeError:
        return float(np.quantile(array, 0.25, interpolation="linear"))


def mask_component_areas(
    component_map: np.ndarray,
    areas: np.ndarray,
    threshold: float,
    *,
    inclusive: bool = False,
) -> np.ndarray:
    comparison = (
        areas.astype(np.float64) <= threshold
        if inclusive
        else areas.astype(np.float64) < threshold
    )
    small_ids = np.flatnonzero((np.arange(len(areas)) > 0) & comparison)
    if not len(small_ids):
        return np.zeros(component_map.shape, dtype=bool)
    return np.isin(component_map, small_ids)


def count(mask: np.ndarray) -> int:
    return int(np.count_nonzero(mask))


def attribution(error: np.ndarray, small: np.ndarray, smoke: np.ndarray) -> dict:
    small_error = error & small
    smoke_error = error & smoke
    both = small_error & smoke
    small_only = small_error & ~smoke
    smoke_only = smoke_error & ~small
    neither = error & ~small & ~smoke
    total = count(error)
    result = {
        "total": total,
        "small_nonexclusive": count(small_error),
        "smoke_nonexclusive": count(smoke_error),
        "both": count(both),
        "small_only": count(small_only),
        "smoke_only": count(smoke_only),
        "neither": count(neither),
    }
    result["fractions"] = {
        key: safe_ratio(value, total)
        for key, value in result.items()
        if key != "total"
    }
    if result["small_only"] + result["smoke_only"] + result["both"] + result[
        "neither"
    ] != total:
        raise RuntimeError("Exclusive attribution partition does not sum to total.")
    return result


def write_rows(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def main() -> None:
    args = parse_args()
    if args.neighbourhood_radius < 0:
        raise ValueError("Neighbourhood radius must be non-negative.")
    config_path = args.config.resolve()
    evaluation_dir = args.evaluation_dir.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else evaluation_dir / "fire_spatial_attribution"
    )
    config = load_config(config_path)
    if config["MODE"] != "fusion":
        raise ValueError("Fire spatial attribution requires MODE: fusion.")
    seed_everything(int(config["SEED"]))
    dataset = build_dataset(config, "val")
    metrics = json.loads((evaluation_dir / "metrics.json").read_text(encoding="utf-8"))
    if metrics.get("split") != "val" or int(metrics.get("sample_count", -1)) != len(
        dataset
    ):
        raise RuntimeError("Reference evaluation is not the matching validation split.")

    ignore_label = int(config["IGNORE_LABEL"])
    fire_class = 2
    smoke_class = 1
    samples: list[tuple[int, str, np.ndarray, np.ndarray]] = []
    gt_component_areas: list[int] = []
    for index in range(len(dataset)):
        sample = dataset[index]
        labels = np.asarray(sample[1], dtype=np.int64)
        name = extract_name(sample[3])
        pred_path = prediction_path(evaluation_dir, name)
        if not pred_path.is_file():
            raise FileNotFoundError(f"Saved validation prediction missing: {pred_path}")
        predictions = np.asarray(Image.open(pred_path), dtype=np.int64)
        if predictions.shape != labels.shape:
            raise RuntimeError(f"Prediction/label shape mismatch for {name}")
        valid = labels != ignore_label
        if ((labels[valid] < 0) | (labels[valid] >= 3)).any():
            raise RuntimeError(f"Invalid label class in {name}")
        if ((predictions < 0) | (predictions >= 3)).any():
            raise RuntimeError(f"Invalid prediction class in {name}")
        _, gt_areas = connected_components((labels == fire_class) & valid)
        gt_component_areas.extend(int(area) for area in gt_areas[1:])
        samples.append((index, name, labels, predictions))

    q1 = strict_lower_quartile(gt_component_areas)
    radius = int(args.neighbourhood_radius)
    aggregate = {
        "fn": {
            "total": 0,
            "small_primary": 0,
            "smoke_union": 0,
            "both": 0,
            "small_only": 0,
            "smoke_only": 0,
            "neither": 0,
            "predicted_smoke_exact": 0,
            "gt_smoke_neighbourhood": 0,
            "predicted_smoke_neighbourhood": 0,
        },
        "fp": {
            "total": 0,
            "small_primary": 0,
            "smoke_union": 0,
            "both": 0,
            "small_only": 0,
            "smoke_only": 0,
            "neither": 0,
            "gt_smoke_exact": 0,
            "gt_smoke_neighbourhood": 0,
            "predicted_smoke_neighbourhood": 0,
            "near_small_gt_fire": 0,
        },
    }
    sensitivity_inclusive_q1 = {
        "definition": (
            "descriptive sensitivity only: component area <= q1; this does not "
            "replace the preregistered strict area < q1 primary rule"
        ),
        "fn_small_gt_component": 0,
        "fn_small_and_smoke": 0,
        "fp_small_predicted_component": 0,
        "fp_small_and_smoke": 0,
        "fp_near_small_gt_fire": 0,
    }
    per_image_rows: list[dict] = []

    for index, name, labels, predictions in samples:
        valid = labels != ignore_label
        gt_fire = (labels == fire_class) & valid
        pred_fire = (predictions == fire_class) & valid
        fn = gt_fire & ~pred_fire
        fp = pred_fire & ~gt_fire

        gt_components, gt_areas = connected_components(gt_fire)
        pred_components, pred_areas = connected_components(pred_fire)
        small_gt = mask_component_areas(gt_components, gt_areas, q1)
        small_pred = mask_component_areas(pred_components, pred_areas, q1)
        near_small_gt = dilate_square(small_gt, radius) & valid
        small_gt_inclusive = mask_component_areas(
            gt_components, gt_areas, q1, inclusive=True
        )
        small_pred_inclusive = mask_component_areas(
            pred_components, pred_areas, q1, inclusive=True
        )
        near_small_gt_inclusive = dilate_square(small_gt_inclusive, radius) & valid

        gt_smoke_exact = (labels == smoke_class) & valid
        pred_smoke_exact = (predictions == smoke_class) & valid
        gt_smoke_neighbourhood = dilate_square(gt_smoke_exact, radius) & valid
        pred_smoke_neighbourhood = dilate_square(pred_smoke_exact, radius) & valid
        smoke_union = gt_smoke_neighbourhood | pred_smoke_neighbourhood

        sensitivity_inclusive_q1["fn_small_gt_component"] += count(
            fn & small_gt_inclusive
        )
        sensitivity_inclusive_q1["fn_small_and_smoke"] += count(
            fn & small_gt_inclusive & smoke_union
        )
        sensitivity_inclusive_q1["fp_small_predicted_component"] += count(
            fp & small_pred_inclusive
        )
        sensitivity_inclusive_q1["fp_small_and_smoke"] += count(
            fp & small_pred_inclusive & smoke_union
        )
        sensitivity_inclusive_q1["fp_near_small_gt_fire"] += count(
            fp & near_small_gt_inclusive
        )

        fn_attr = attribution(fn, small_gt, smoke_union)
        fp_attr = attribution(fp, small_pred, smoke_union)
        for key, value in (
            ("total", fn_attr["total"]),
            ("small_primary", fn_attr["small_nonexclusive"]),
            ("smoke_union", fn_attr["smoke_nonexclusive"]),
            ("both", fn_attr["both"]),
            ("small_only", fn_attr["small_only"]),
            ("smoke_only", fn_attr["smoke_only"]),
            ("neither", fn_attr["neither"]),
            ("predicted_smoke_exact", count(fn & pred_smoke_exact)),
            ("gt_smoke_neighbourhood", count(fn & gt_smoke_neighbourhood)),
            (
                "predicted_smoke_neighbourhood",
                count(fn & pred_smoke_neighbourhood),
            ),
        ):
            aggregate["fn"][key] += int(value)
        for key, value in (
            ("total", fp_attr["total"]),
            ("small_primary", fp_attr["small_nonexclusive"]),
            ("smoke_union", fp_attr["smoke_nonexclusive"]),
            ("both", fp_attr["both"]),
            ("small_only", fp_attr["small_only"]),
            ("smoke_only", fp_attr["smoke_only"]),
            ("neither", fp_attr["neither"]),
            ("gt_smoke_exact", count(fp & gt_smoke_exact)),
            ("gt_smoke_neighbourhood", count(fp & gt_smoke_neighbourhood)),
            (
                "predicted_smoke_neighbourhood",
                count(fp & pred_smoke_neighbourhood),
            ),
            ("near_small_gt_fire", count(fp & near_small_gt)),
        ):
            aggregate["fp"][key] += int(value)

        per_image_rows.append(
            {
                "index": index,
                "name": name,
                "fn_total": fn_attr["total"],
                "fn_small_gt_component": fn_attr["small_nonexclusive"],
                "fn_smoke_union": fn_attr["smoke_nonexclusive"],
                "fn_both": fn_attr["both"],
                "fn_neither": fn_attr["neither"],
                "fp_total": fp_attr["total"],
                "fp_small_predicted_component": fp_attr["small_nonexclusive"],
                "fp_near_small_gt_fire": count(fp & near_small_gt),
                "fp_smoke_union": fp_attr["smoke_nonexclusive"],
                "fp_both": fp_attr["both"],
                "fp_neither": fp_attr["neither"],
                "fn_small_gt_component_leq_q1_sensitivity": count(
                    fn & small_gt_inclusive
                ),
                "fp_small_predicted_component_leq_q1_sensitivity": count(
                    fp & small_pred_inclusive
                ),
                "fp_near_small_gt_fire_leq_q1_sensitivity": count(
                    fp & near_small_gt_inclusive
                ),
            }
        )

    reference_confusion = np.asarray(metrics["confusion_matrix"], dtype=np.int64)
    reference_fn = int(reference_confusion[fire_class].sum() - reference_confusion[fire_class, fire_class])
    reference_fp = int(reference_confusion[:, fire_class].sum() - reference_confusion[fire_class, fire_class])
    if aggregate["fn"]["total"] != reference_fn:
        raise RuntimeError("FN total does not match the saved validation confusion matrix.")
    if aggregate["fp"]["total"] != reference_fp:
        raise RuntimeError("FP total does not match the saved validation confusion matrix.")

    for error_name in ("fn", "fp"):
        total = aggregate[error_name]["total"]
        aggregate[error_name]["fractions"] = {
            key: safe_ratio(value, total)
            for key, value in aggregate[error_name].items()
            if key not in {"total", "fractions"}
        }
        if (
            aggregate[error_name]["small_only"]
            + aggregate[error_name]["smoke_only"]
            + aggregate[error_name]["both"]
            + aggregate[error_name]["neither"]
            != total
        ):
            raise RuntimeError(f"{error_name.upper()} partition does not sum.")

    component_array = np.asarray(gt_component_areas, dtype=np.float64)
    sensitivity_inclusive_q1["fractions"] = {
        "fn_small_gt_component": safe_ratio(
            sensitivity_inclusive_q1["fn_small_gt_component"],
            aggregate["fn"]["total"],
        ),
        "fn_small_and_smoke": safe_ratio(
            sensitivity_inclusive_q1["fn_small_and_smoke"],
            aggregate["fn"]["total"],
        ),
        "fp_small_predicted_component": safe_ratio(
            sensitivity_inclusive_q1["fp_small_predicted_component"],
            aggregate["fp"]["total"],
        ),
        "fp_small_and_smoke": safe_ratio(
            sensitivity_inclusive_q1["fp_small_and_smoke"],
            aggregate["fp"]["total"],
        ),
        "fp_near_small_gt_fire": safe_ratio(
            sensitivity_inclusive_q1["fp_near_small_gt_fire"],
            aggregate["fp"]["total"],
        ),
    }
    result = {
        "config": str(config_path),
        "evaluation_dir": str(evaluation_dir),
        "checkpoint": metrics["checkpoint"],
        "checkpoint_epoch": metrics["checkpoint_epoch"],
        "split": "val",
        "sample_count": len(dataset),
        "test_set_used": False,
        "fire_component_definition": "8-connected components on evaluation-resolution GT Fire masks",
        "gt_fire_component_area_distribution": {
            "component_count": len(gt_component_areas),
            "min": int(component_array.min()),
            "q1_linear": q1,
            "median": float(np.median(component_array)),
            "mean": float(component_array.mean()),
            "max": int(component_array.max()),
            "small_rule": "component area < q1_linear (strict inequality)",
            "small_component_count_lt_q1": int(
                np.count_nonzero(component_array < q1)
            ),
            "component_count_leq_q1_sensitivity": int(
                np.count_nonzero(component_array <= q1)
            ),
        },
        "neighbourhood": {
            "radius_pixels": radius,
            "metric": "Chebyshev / square dilation",
            "smoke_union": "within radius of GT Smoke or predicted Smoke, including the region itself",
        },
        "attribution_definitions": {
            "fn_small_primary": "FN pixel belongs to a small GT Fire component",
            "fp_small_primary": (
                "FP pixel belongs to a predicted-Fire component whose area is below "
                "the GT-Fire q1 threshold"
            ),
            "fp_near_small_gt_fire_secondary": (
                "FP pixel lies within the same radius of a small GT Fire component; "
                "reported separately because an FP cannot belong to a GT Fire component"
            ),
            "smoke_related": (
                "error pixel lies in the union of GT-Smoke and predicted-Smoke "
                "Chebyshev neighbourhoods"
            ),
            "exclusive_partition": "small_only, smoke_only, both, neither",
        },
        "false_negative": aggregate["fn"],
        "false_positive": aggregate["fp"],
        "inclusive_q1_sensitivity": sensitivity_inclusive_q1,
        "validation_checks": {
            "fn_matches_saved_confusion": True,
            "fp_matches_saved_confusion": True,
            "exclusive_partitions_sum": True,
            "test_set_used": False,
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "fire_error_spatial_attribution.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    per_image_rows.sort(key=lambda row: (-int(row["fn_total"] + row["fp_total"]), str(row["name"])))
    write_rows(
        output_dir / "fire_error_spatial_attribution_per_image.csv",
        per_image_rows,
        [
            "index",
            "name",
            "fn_total",
            "fn_small_gt_component",
            "fn_smoke_union",
            "fn_both",
            "fn_neither",
            "fp_total",
            "fp_small_predicted_component",
            "fp_near_small_gt_fire",
            "fp_smoke_union",
            "fp_both",
            "fp_neither",
            "fn_small_gt_component_leq_q1_sensitivity",
            "fp_small_predicted_component_leq_q1_sensitivity",
            "fp_near_small_gt_fire_leq_q1_sensitivity",
        ],
    )
    print(f"GT Fire component q1 area: {q1:.6f} pixels")
    print(json.dumps({"false_negative": aggregate["fn"], "false_positive": aggregate["fp"]}, ensure_ascii=False, indent=2))
    print(f"Result: {output_dir / 'fire_error_spatial_attribution.json'}")


if __name__ == "__main__":
    main()
