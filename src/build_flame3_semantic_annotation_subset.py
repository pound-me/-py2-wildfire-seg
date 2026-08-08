from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


FIRE_ID = 2
IGNORE_ID = 255
MAX_ITEMS = 150


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the frozen FLAME3 train/validation-only semantic and pixel-annotation "
            "candidate subset after the CMRC route is closed."
        )
    )
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--val-csv", type=Path, required=True)
    parser.add_argument("--statistics-csv", type=Path, required=True)
    parser.add_argument("--prior-audit-checklist", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
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
        raise ValueError("Cannot write an empty annotation checklist")
    path.parent.mkdir(parents=True, exist_ok=True)
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


def as_int(row: dict[str, str], key: str) -> int:
    return int(float(row[key]))


def as_float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def stats_key(row: dict[str, str]) -> str:
    prefix = "fire" if row["sample_class"] == "Fire" else "no_fire"
    return f"{prefix}/{row['sample_id']}"


def resolve_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise ValueError(f"Expected an absolute frozen data path, got: {value}")
    return path.resolve()


def build_features(
    split_row: dict[str, str], statistics: dict[str, str]
) -> dict[str, object]:
    pixels = max(as_int(statistics, "height") * as_int(statistics, "width"), 1)
    train_fire_pixels = as_int(statistics, "train_fire_pixels")
    ignore_pixels = as_int(statistics, "ignore_pixels")
    low_pixels = as_int(statistics, "low_threshold_pixels")
    kept_pixels = as_int(statistics, "kept_fire_pixels_before_ignore")
    noncore_hot_pixels = max(low_pixels - kept_pixels, 0)
    return {
        **split_row,
        "min_temperature_c": as_float(statistics, "min_temperature_c"),
        "max_temperature_c": as_float(statistics, "max_temperature_c"),
        "mean_temperature_c": as_float(statistics, "mean_temperature_c"),
        "low_threshold_pixels": low_pixels,
        "high_seed_pixels": as_int(statistics, "high_seed_pixels"),
        "kept_fire_pixels_before_ignore": kept_pixels,
        "train_fire_pixels": train_fire_pixels,
        "ignore_pixels": ignore_pixels,
        "component_count_after_cleanup": as_int(
            statistics, "component_count_after_cleanup"
        ),
        "fire_ratio": train_fire_pixels / pixels,
        "ignore_ratio": ignore_pixels / pixels,
        "noncore_hot_pixels": noncore_hot_pixels,
        "noncore_hot_ratio": noncore_hot_pixels / pixels,
        "boundary_to_fire_ratio": ignore_pixels / max(train_fire_pixels, 1),
        "low_threshold_ratio": low_pixels / pixels,
    }


def diversity_group(row: dict[str, object]) -> str:
    if row["sample_class"] == "Fire":
        return f"time-{row.get('time_block_id', '')}"
    return f"space-{row.get('space_cluster_id', '')}"


def diverse_take(
    rows: list[dict[str, object]],
    count: int,
    score,
    min_time_gap_seconds: float = 0.0,
) -> list[dict[str, object]]:
    ordered = sorted(
        rows,
        key=lambda row: (float(score(row)), str(row["sample_key"])),
        reverse=True,
    )
    groups = {diversity_group(row) for row in ordered}
    cap = max(2, math.ceil(count / max(len(groups), 1)) + 2)
    selected: list[dict[str, object]] = []
    selected_keys: set[str] = set()
    group_counts: defaultdict[str, int] = defaultdict(int)
    group_times: defaultdict[str, list[float]] = defaultdict(list)
    for row in ordered:
        group = diversity_group(row)
        if group_counts[group] >= cap:
            continue
        timestamp = str(row.get("timestamp", "")).strip()
        time_value = datetime.fromisoformat(timestamp).timestamp() if timestamp else None
        if (
            min_time_gap_seconds > 0.0
            and time_value is not None
            and any(
                abs(time_value - previous) < min_time_gap_seconds
                for previous in group_times[group]
            )
        ):
            continue
        selected.append(row)
        selected_keys.add(str(row["sample_key"]))
        group_counts[group] += 1
        if time_value is not None:
            group_times[group].append(time_value)
        if len(selected) == count:
            return selected
    for row in ordered:
        key = str(row["sample_key"])
        if key in selected_keys:
            continue
        selected.append(row)
        if len(selected) == count:
            return selected
    raise RuntimeError(f"Only {len(selected)} candidates available for requested {count}")


def temporally_dispersed_take(
    rows: list[dict[str, object]], count: int, score
) -> list[dict[str, object]]:
    if len(rows) < count:
        raise RuntimeError(f"Only {len(rows)} candidates available for requested {count}")
    candidates = list(rows)
    scored = {str(row["sample_key"]): float(score(row)) for row in candidates}
    times = {
        str(row["sample_key"]): datetime.fromisoformat(str(row["timestamp"])).timestamp()
        for row in candidates
    }
    first = max(
        candidates,
        key=lambda row: (scored[str(row["sample_key"])], str(row["sample_key"])),
    )
    selected = [first]
    selected_keys = {str(first["sample_key"])}
    while len(selected) < count:
        selected_times = [times[str(row["sample_key"])] for row in selected]
        available = [
            row for row in candidates if str(row["sample_key"]) not in selected_keys
        ]
        chosen = max(
            available,
            key=lambda row: (
                min(
                    abs(times[str(row["sample_key"])] - selected_time)
                    for selected_time in selected_times
                ),
                scored[str(row["sample_key"])],
                str(row["sample_key"]),
            ),
        )
        selected.append(chosen)
        selected_keys.add(str(chosen["sample_key"]))
    return selected


def select_train_candidates(
    train_rows: list[dict[str, object]], required: int
) -> list[tuple[str, str, dict[str, object]]]:
    quotas = [
        ("empty_fire_hotspot", 30),
        ("suspected_label_gap_hot_noncore", 15),
        ("small_or_weak_fire_core", 25),
        ("boundary_complex_fire_core", 20),
        ("no_fire_hotspot_negative", 13),
    ]
    if sum(count for _, count in quotas) != required:
        raise RuntimeError(
            f"Frozen train quota is {sum(count for _, count in quotas)}, expected {required}"
        )
    remaining = {str(row["sample_key"]): row for row in train_rows}
    output: list[tuple[str, str, dict[str, object]]] = []

    candidates = [
        row
        for row in remaining.values()
        if row["sample_class"] == "Fire" and int(row["train_fire_pixels"]) == 0
    ]
    chosen = temporally_dispersed_take(
        candidates,
        30,
        lambda row: 8.0 * float(row["low_threshold_ratio"])
        + 0.002 * float(row["max_temperature_c"])
        + 0.00002 * float(row["low_threshold_pixels"]),
    )
    for row in chosen:
        remaining.pop(str(row["sample_key"]))
        output.append(
            (
                "empty_fire_hotspot",
                "Fire-folder frame with an empty pseudo Fire core but strong thermal activity; "
                "review residual heat versus obscured active combustion.",
                row,
            )
        )

    candidates = [
        row
        for row in remaining.values()
        if row["sample_class"] == "Fire" and int(row["train_fire_pixels"]) > 0
    ]
    chosen = diverse_take(
        candidates,
        15,
        lambda row: 12.0 * float(row["noncore_hot_ratio"])
        + 2.0 * float(row["ignore_ratio"])
        + 0.03 * float(row["component_count_after_cleanup"]),
        min_time_gap_seconds=12.0,
    )
    for row in chosen:
        remaining.pop(str(row["sample_key"]))
        output.append(
            (
                "suspected_label_gap_hot_noncore",
                "Large 80C-plus thermal region lies outside the retained pseudo Fire core; "
                "review possible label gaps under smoke or along the fire front.",
                row,
            )
        )

    candidates = [
        row
        for row in remaining.values()
        if row["sample_class"] == "Fire" and int(row["train_fire_pixels"]) > 0
    ]
    chosen = diverse_take(
        candidates,
        25,
        lambda row: 0.003 / max(float(row["fire_ratio"]), 1e-8)
        + max(350.0 - float(row["max_temperature_c"]), 0.0) / 350.0
        + float(row["boundary_to_fire_ratio"]),
        min_time_gap_seconds=12.0,
    )
    for row in chosen:
        remaining.pop(str(row["sample_key"]))
        output.append(
            (
                "small_or_weak_fire_core",
                "Small or comparatively low-temperature Fire core; review weak, distant, "
                "smoke-obscured and edge fire pixels.",
                row,
            )
        )

    candidates = [
        row
        for row in remaining.values()
        if row["sample_class"] == "Fire" and int(row["train_fire_pixels"]) > 0
    ]
    chosen = diverse_take(
        candidates,
        20,
        lambda row: float(row["boundary_to_fire_ratio"])
        + 0.06 * float(row["component_count_after_cleanup"])
        + 2.0 * float(row["ignore_ratio"]),
        min_time_gap_seconds=12.0,
    )
    for row in chosen:
        remaining.pop(str(row["sample_key"]))
        output.append(
            (
                "boundary_complex_fire_core",
                "Large ignored boundary relative to the Fire core or many small components; "
                "review narrow propagation fronts and pseudo-label boundary completeness.",
                row,
            )
        )

    candidates = [
        row for row in remaining.values() if row["sample_class"] == "No Fire"
    ]
    chosen = diverse_take(
        candidates,
        13,
        lambda row: 10.0 * float(row["low_threshold_ratio"])
        + 0.003 * float(row["max_temperature_c"])
        + 0.00002 * float(row["low_threshold_pixels"]),
    )
    for row in chosen:
        remaining.pop(str(row["sample_key"]))
        output.append(
            (
                "no_fire_hotspot_negative",
                "No-Fire frame with relatively strong thermal response; review hot ground, "
                "reflection and true hard-negative appearance.",
                row,
            )
        )
    return output


def caption_panel(image: np.ndarray, title: str) -> np.ndarray:
    result = np.asarray(image, dtype=np.uint8).copy()
    cv2.rectangle(result, (0, 0), (result.shape[1], 30), (0, 0, 0), -1)
    cv2.putText(
        result,
        title,
        (8, 21),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return result


def colorize_temperature(temperature: np.ndarray) -> np.ndarray:
    normalized = np.clip(temperature, 0.0, 500.0) / 500.0
    bgr = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def threshold_panel(temperature: np.ndarray) -> np.ndarray:
    result = np.zeros((*temperature.shape, 3), dtype=np.uint8)
    result[(temperature >= 80.0) & (temperature < 200.0)] = (255, 170, 0)
    result[temperature >= 200.0] = (255, 0, 0)
    return result


def label_overlay(rgb: np.ndarray, label: np.ndarray) -> np.ndarray:
    result = rgb.astype(np.float32).copy()
    for mask, color in (
        (label == FIRE_ID, (255, 90, 0)),
        (label == IGNORE_ID, (255, 0, 255)),
    ):
        result[mask] = 0.35 * result[mask] + 0.65 * np.asarray(color, dtype=np.float32)
    return np.clip(result, 0, 255).astype(np.uint8)


def build_visual(row: dict[str, object], output: Path) -> None:
    rgb = np.asarray(Image.open(resolve_path(str(row["corrected_rgb_path"]))).convert("RGB"))
    thermal_gray = np.asarray(
        Image.open(resolve_path(str(row["raw_thermal_path"]))).convert("L")
    )
    thermal_rgb = np.repeat(thermal_gray[..., None], 3, axis=2)
    temperature = np.asarray(
        Image.open(resolve_path(str(row["thermal_tiff_path"]))), dtype=np.float32
    )
    label = np.asarray(
        Image.open(resolve_path(str(row["temperature_mask_path"]))), dtype=np.uint8
    )
    if not (
        rgb.shape[:2]
        == thermal_gray.shape
        == temperature.shape
        == label.shape
    ):
        raise RuntimeError(f"Shape mismatch for {row['sample_key']}")
    panels = (
        caption_panel(rgb, "Corrected RGB"),
        caption_panel(thermal_rgb, "Thermal JPG as model gray"),
        caption_panel(colorize_temperature(temperature), "Celsius TIFF 0-500C"),
        caption_panel(threshold_panel(temperature), "80-200C orange / >=200C red"),
        caption_panel(label_overlay(rgb, label), "Pseudo Fire orange / ignore magenta"),
    )
    canvas = np.concatenate(panels, axis=1)
    footer_height = 42
    framed = np.zeros((canvas.shape[0] + footer_height, canvas.shape[1], 3), dtype=np.uint8)
    framed[: canvas.shape[0]] = canvas
    footer = (
        f"{row['annotation_id']} | {row['split']} | {row['selection_category']} | "
        f"{row['sample_key']} | Tmax={float(row['max_temperature_c']):.1f}C | "
        f"Fire={100.0 * float(row['fire_ratio']):.3f}% | "
        f"Hot-noncore={100.0 * float(row['noncore_hot_ratio']):.3f}%"
    )
    cv2.putText(
        framed,
        footer,
        (8, canvas.shape[0] + 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(framed).save(output, quality=91)


def save_contact_sheet(paths: list[Path], output: Path, columns: int = 2) -> None:
    thumbs: list[Image.Image] = []
    for path in paths:
        image = Image.open(path).convert("RGB")
        image.thumbnail((1500, 265), Image.Resampling.LANCZOS)
        thumbs.append(image.copy())
    if not thumbs:
        return
    width = max(image.width for image in thumbs)
    height = max(image.height for image in thumbs)
    rows = math.ceil(len(thumbs) / columns)
    sheet = Image.new("RGB", (columns * width, rows * height), color=(18, 18, 18))
    for index, image in enumerate(thumbs):
        row_index, column_index = divmod(index, columns)
        sheet.paste(image, (column_index * width, row_index * height))
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, quality=90)


def main() -> None:
    args = parse_args()
    train_csv = args.train_csv.resolve()
    val_csv = args.val_csv.resolve()
    statistics_csv = args.statistics_csv.resolve()
    prior_audit = args.prior_audit_checklist.resolve()
    output = args.output.resolve()
    for path in (train_csv, val_csv, statistics_csv, prior_audit):
        if not path.is_file():
            raise FileNotFoundError(path)
    if train_csv.name.lower() != "train.csv" or val_csv.name.lower() != "val.csv":
        raise ValueError("This builder accepts the frozen train.csv and val.csv only")
    if "test" in str(train_csv).lower() or "test" in str(val_csv).lower():
        raise ValueError("Test-set paths are forbidden")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing annotation subset: {output}")

    train_split = read_csv(train_csv)
    val_split = read_csv(val_csv)
    if len(train_split) != 493 or len(val_split) != 134:
        raise RuntimeError(
            f"Expected split-v2 train/val sizes 493/134, got {len(train_split)}/{len(val_split)}"
        )
    train_keys = {row["sample_key"] for row in train_split}
    val_keys = {row["sample_key"] for row in val_split}
    if train_keys & val_keys:
        raise RuntimeError("Train and validation membership overlap")

    statistics_rows = read_csv(statistics_csv)
    statistics_by_key = {stats_key(row): row for row in statistics_rows}
    train_features = [
        build_features(row, statistics_by_key[row["sample_key"]]) for row in train_split
    ]
    val_features = {
        row["sample_key"]: build_features(row, statistics_by_key[row["sample_key"]])
        for row in val_split
    }

    prior_rows = read_csv(prior_audit)
    prior_by_sample: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
    for row in prior_rows:
        if row["sample_key"] not in val_keys:
            raise RuntimeError(f"Prior audit item is not in frozen validation: {row['sample_key']}")
        prior_by_sample[row["sample_key"]].append(row)
    if len(prior_by_sample) != 47:
        raise RuntimeError(f"Expected 47 unique prior-reviewed validation images, got {len(prior_by_sample)}")

    selected: list[tuple[str, str, dict[str, object], list[dict[str, str]]]] = []
    for sample_key, reviews in sorted(
        prior_by_sample.items(), key=lambda item: min(row["audit_id"] for row in item[1])
    ):
        selected.append(
            (
                "reviewed_validation",
                "Previously reviewed Fusion FN/FP/TP validation image retained for full-image "
                "semantic adjudication and optional pixel annotation.",
                val_features[sample_key],
                reviews,
            )
        )
    for category, reason, row in select_train_candidates(
        train_features, MAX_ITEMS - len(selected)
    ):
        selected.append((category, reason, row, []))
    if len(selected) != MAX_ITEMS:
        raise RuntimeError(f"Expected exactly {MAX_ITEMS} unique images, got {len(selected)}")
    selected_keys = [str(item[2]["sample_key"]) for item in selected]
    if len(selected_keys) != len(set(selected_keys)):
        raise RuntimeError("Duplicate image selected for semantic annotation")

    output.mkdir(parents=True)
    visual_dir = output / "visuals"
    mask_seed_dir = output / "pseudo_mask_seeds"
    contact_dir = output / "contact_sheets"
    checklist_rows: list[dict[str, object]] = []
    manifest_items: list[dict[str, object]] = []
    category_visuals: defaultdict[str, list[Path]] = defaultdict(list)
    for index, (category, reason, row, reviews) in enumerate(selected, start=1):
        annotation_id = f"A{index:03d}"
        split_name = "val" if str(row["sample_key"]) in val_keys else "train"
        prior_ids = ";".join(review["audit_id"] for review in reviews)
        prior_causes = ";".join(
            sorted({review["primary_cause"] for review in reviews if review["primary_cause"]})
        )
        prior_counts = ";".join(
            sorted(
                {
                    review["counts_as_active_fire_yes_no_uncertain"]
                    for review in reviews
                    if review["counts_as_active_fire_yes_no_uncertain"]
                }
            )
        )
        enriched = {
            **row,
            "annotation_id": annotation_id,
            "split": split_name,
            "selection_category": category,
        }
        visual_path = visual_dir / f"{annotation_id}_{str(row['sample_key']).replace('/', '_')}.jpg"
        build_visual(enriched, visual_path)
        category_visuals[category].append(visual_path)
        mask_seed_path = mask_seed_dir / f"{annotation_id}.png"
        mask_seed_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(resolve_path(str(row["temperature_mask_path"])), mask_seed_path)
        checklist_rows.append(
            {
                "annotation_id": annotation_id,
                "split": split_name,
                "selection_category": category,
                "sample_key": row["sample_key"],
                "visual_path": str(visual_path),
                "sample_class": row["sample_class"],
                "sample_id": row["sample_id"],
                "semantic_class": "",
                "counts_as_active_fire_yes_no_uncertain": "",
                "visible_flame_yes_no_uncertain": "",
                "residual_heat_or_hot_ground_yes_no_uncertain": "",
                "smoke_occlusion_yes_no_uncertain": "",
                "registration_mismatch_yes_no_uncertain": "",
                "pseudo_label_semantic_ambiguity_yes_no": "",
                "pseudo_label_missing_active_fire_yes_no_uncertain": "",
                "pixel_mask_status": "not_started",
                "completed_mask_path": "",
                "notes": "",
                "timestamp": row["timestamp"],
                "time_block_id": row.get("time_block_id", ""),
                "space_cluster_id": row.get("space_cluster_id", ""),
                "selection_reason": reason,
                "previously_reviewed_yes_no": "yes" if reviews else "no",
                "prior_audit_ids": prior_ids,
                "prior_primary_causes": prior_causes,
                "prior_counts_as_active_fire": prior_counts,
                "max_temperature_c": float(row["max_temperature_c"]),
                "mean_temperature_c": float(row["mean_temperature_c"]),
                "fire_ratio": float(row["fire_ratio"]),
                "ignore_ratio": float(row["ignore_ratio"]),
                "noncore_hot_ratio": float(row["noncore_hot_ratio"]),
                "corrected_rgb_path": row["corrected_rgb_path"],
                "raw_thermal_path": row["raw_thermal_path"],
                "thermal_tiff_path": row["thermal_tiff_path"],
                "pseudo_label_path": row["temperature_mask_path"],
                "pseudo_mask_seed_path": str(mask_seed_path),
            }
        )
        manifest_items.append(
            {
                "annotation_id": annotation_id,
                "split": split_name,
                "selection_category": category,
                "sample_key": row["sample_key"],
                "visual": str(visual_path),
                "pseudo_mask_seed": str(mask_seed_path),
            }
        )

    checklist_path = output / "flame3_semantic_pixel_annotation_checklist_150.csv"
    write_csv(checklist_path, checklist_rows)
    for category, paths in category_visuals.items():
        save_contact_sheet(paths, contact_dir / f"{category}_contact_sheet.jpg")
    overview_paths: list[Path] = []
    for category in (
        "reviewed_validation",
        "empty_fire_hotspot",
        "suspected_label_gap_hot_noncore",
        "small_or_weak_fire_core",
        "boundary_complex_fire_core",
        "no_fire_hotspot_negative",
    ):
        overview_paths.extend(category_visuals[category][:4])
    save_contact_sheet(overview_paths, output / "overview_contact_sheet.jpg")

    category_counts = {
        category: len(paths) for category, paths in sorted(category_visuals.items())
    }
    manifest = {
        "protocol": "flame3_train_val_only_targeted_semantic_pixel_annotation_subset",
        "status": "candidate_subset_frozen_before_manual_pixel_annotation",
        "maximum_images": MAX_ITEMS,
        "selected_images": len(selected),
        "selection_counts": category_counts,
        "selection_rules": {
            "reviewed_validation": "all 47 unique validation images from the completed 70-component manual audit",
            "empty_fire_hotspot": "30 train Fire images with empty pseudo Fire core, selected by farthest-point timestamp dispersion with thermal score as tie-break",
            "suspected_label_gap_hot_noncore": "15 train Fire images ranked by 80C-plus non-core thermal extent, ignore extent and component count, with 12-second within-block spacing",
            "small_or_weak_fire_core": "25 train Fire images ranked by small core, lower peak temperature and boundary-to-core ratio, with 12-second within-block spacing",
            "boundary_complex_fire_core": "20 train Fire images ranked by ignored-boundary/core ratio and component count, with 12-second within-block spacing",
            "no_fire_hotspot_negative": "13 train No-Fire images ranked by strongest thermal response with spatial-cluster diversity",
        },
        "split_counts": {
            "train": sum(row["split"] == "train" for row in checklist_rows),
            "val": sum(row["split"] == "val" for row in checklist_rows),
        },
        "inputs": {
            "train_csv": str(train_csv),
            "train_csv_sha256": sha256(train_csv),
            "val_csv": str(val_csv),
            "val_csv_sha256": sha256(val_csv),
            "statistics_csv": str(statistics_csv),
            "statistics_csv_sha256": sha256(statistics_csv),
            "prior_audit_checklist": str(prior_audit),
            "prior_audit_checklist_sha256": sha256(prior_audit),
        },
        "mask_schema": {
            "0": "background_or_non_active_residual_heat",
            "1": "smoke",
            "2": "active_fire",
            "255": "uncertain_or_ignore",
        },
        "annotation_policy": {
            "purpose": "resolve active-fire versus residual-heat and smoke-occlusion ambiguity before any data-side training change",
            "pseudo_masks_are_seeds_only": True,
            "do_not_count_high_temperature_alone_as_active_fire": True,
            "uncertain_pixels_use_255": True,
            "not_a_representative_metric_sample": True,
        },
        "test_images_or_labels_read": False,
        "test_set_policy": "sealed_until_final_method_freeze",
        "items": manifest_items,
    }
    manifest_path = output / "flame3_semantic_pixel_annotation_manifest_150.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    readme = """# FLAME3语义与像素标注候选集（150张）

本候选集只读取冻结split v2的train和validation，测试集继续封存。

## 每张图的五列

1. Corrected RGB；
2. 模型实际使用的热JPG灰度；
3. 0–500°C固定色标温度TIFF；
4. 80–200°C橙色、≥200°C红色的阈值诊断；
5. RGB上的温度伪标签：火区橙色、边界忽略带紫色。

## 建议填写顺序

先判断是否属于活动火，再判断可见火焰、余热/热地面、烟雾遮挡、配准和伪标签争议。
高温本身不能单独证明存在活动火。肉眼和两种热图都无法确认的像素，完整掩码使用255忽略。

`pseudo_mask_seeds`仅作为编辑起点，不能直接当成人工真值。完成后的掩码另存并填写`completed_mask_path`。
"""
    (output / "README_REVIEW.md").write_text(readme, encoding="utf-8")
    print(json.dumps({"output": str(output), "counts": category_counts}, ensure_ascii=False))


if __name__ == "__main__":
    main()
