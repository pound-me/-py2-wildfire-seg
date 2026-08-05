from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


FIRE_ID = 2
IGNORE_ID = 255


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a frozen 30-FN/20-FP/20-TP FLAME3 validation review set."
    )
    parser.add_argument("--profile-dir", type=Path, required=True)
    parser.add_argument("--split-csv", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
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


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def resolve_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def connected_components(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    _, component_map, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    return component_map, stats


def robust_temperature_color(temperature: np.ndarray) -> np.ndarray:
    finite = temperature[np.isfinite(temperature)]
    low, high = np.percentile(finite, [1.0, 99.0])
    if high <= low:
        high = low + 1.0
    normalized = np.clip((temperature - low) / (high - low), 0.0, 1.0)
    bgr = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def colorize_label(label: np.ndarray) -> np.ndarray:
    result = np.zeros((*label.shape, 3), dtype=np.uint8)
    result[label == FIRE_ID] = (255, 180, 0)
    result[label == IGNORE_ID] = (255, 0, 255)
    return result


def colorize_prediction(prediction: np.ndarray) -> np.ndarray:
    result = np.zeros((*prediction.shape, 3), dtype=np.uint8)
    result[prediction == 1] = (155, 155, 155)
    result[prediction == FIRE_ID] = (255, 0, 0)
    return result


def error_overlay(
    rgb: np.ndarray, label: np.ndarray, prediction: np.ndarray, focus: np.ndarray
) -> np.ndarray:
    valid = label != IGNORE_ID
    target = label == FIRE_ID
    predicted = prediction == FIRE_ID
    result = rgb.astype(np.float32).copy()
    for mask, color in (
        (valid & target & predicted, (0, 255, 0)),
        (valid & target & ~predicted, (0, 120, 255)),
        (valid & ~target & predicted, (255, 0, 0)),
    ):
        result[mask] = 0.30 * result[mask] + 0.70 * np.asarray(color, dtype=np.float32)
    result = np.clip(result, 0, 255).astype(np.uint8)
    contours, _ = cv2.findContours(
        focus.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(result, contours, -1, (255, 255, 0), 2)
    return result


def caption_panel(panel: np.ndarray, title: str) -> np.ndarray:
    result = np.array(panel, dtype=np.uint8, copy=True)
    cv2.rectangle(result, (0, 0), (result.shape[1], 30), (0, 0, 0), -1)
    cv2.putText(
        result,
        title,
        (8, 21),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return result


def build_visual(
    rgb: np.ndarray,
    thermal_jpg: np.ndarray,
    temperature: np.ndarray,
    label: np.ndarray,
    prediction: np.ndarray,
    focus: np.ndarray,
    caption: str,
) -> np.ndarray:
    thermal_rgb = np.repeat(thermal_jpg[..., None], 3, axis=2)
    panels = (
        caption_panel(rgb, "Corrected RGB"),
        caption_panel(thermal_rgb, "Raw Thermal JPG gray"),
        caption_panel(robust_temperature_color(temperature), "Celsius TIFF diagnostic"),
        caption_panel(colorize_label(label), "Partial label"),
        caption_panel(colorize_prediction(prediction), "Fusion prediction"),
        caption_panel(error_overlay(rgb, label, prediction, focus), "TP green / FN blue / FP red"),
    )
    canvas = np.concatenate(panels, axis=1)
    cv2.rectangle(canvas, (0, canvas.shape[0] - 30), (canvas.shape[1], canvas.shape[0]), (0, 0, 0), -1)
    cv2.putText(
        canvas,
        caption[:220],
        (8, canvas.shape[0] - 9),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return canvas


def save_contact_sheet(paths: list[Path], output: Path, columns: int = 2) -> None:
    thumbs: list[Image.Image] = []
    for path in paths:
        image = Image.open(path).convert("RGB")
        image.thumbnail((1500, 235), Image.Resampling.LANCZOS)
        thumbs.append(image.copy())
    if not thumbs:
        return
    width = max(image.width for image in thumbs)
    height = max(image.height for image in thumbs)
    rows = math.ceil(len(thumbs) / columns)
    sheet = Image.new("RGB", (columns * width, rows * height), color=(18, 18, 18))
    for index, image in enumerate(thumbs):
        row, column = divmod(index, columns)
        sheet.paste(image, (column * width, row * height))
    sheet.save(output, quality=92)


def enrich_gt_components(
    rows: list[dict[str, str]], split_rows: dict[str, dict[str, str]], data_root: Path
) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row["sample_key"], []).append(row)
    enriched: list[dict[str, object]] = []
    for sample_key, components in grouped.items():
        source = split_rows[sample_key]
        label = np.asarray(
            Image.open(resolve_path(data_root, source["temperature_mask_path"])), dtype=np.uint8
        )
        temperature = np.asarray(
            Image.open(resolve_path(data_root, source["thermal_tiff_path"])), dtype=np.float32
        )
        component_map, _ = connected_components(label == FIRE_ID)
        for row in components:
            component_id = int(row["component_id"])
            mask = component_map == component_id
            if not mask.any():
                raise RuntimeError(f"Missing GT component {sample_key}/{component_id}")
            enriched.append(
                {
                    **row,
                    "temperature_mean_c": float(temperature[mask].mean()),
                    "temperature_max_c": float(temperature[mask].max()),
                }
            )
    return enriched


def select_even_temperature(rows: list[dict[str, object]], count: int) -> list[dict[str, object]]:
    eligible = sorted(rows, key=lambda row: float(row["temperature_mean_c"]))
    if len(eligible) <= count:
        return eligible
    indices = np.linspace(0, len(eligible) - 1, count, dtype=int)
    return [eligible[int(index)] for index in indices]


def main() -> None:
    args = parse_args()
    profile_dir = args.profile_dir.resolve()
    split_csv = args.split_csv.resolve()
    data_root = args.data_root.resolve()
    output = args.output.resolve()
    if split_csv.name.lower() != "val.csv":
        raise ValueError("Manual error audit permits val.csv only")
    required = (
        profile_dir / "per_image_metrics.csv",
        profile_dir / "gt_fire_components.csv",
        profile_dir / "predicted_fire_components.csv",
        profile_dir / "predictions_raw",
    )
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)
    split_list = read_csv(split_csv)
    if len(split_list) != 134:
        raise RuntimeError(f"Expected 134 validation samples, got {len(split_list)}")
    split_rows = {row["sample_key"]: row for row in split_list}
    per_image = {row["sample_key"]: row for row in read_csv(required[0])}
    gt_rows = enrich_gt_components(read_csv(required[1]), split_rows, data_root)
    pred_rows: list[dict[str, object]] = [dict(row) for row in read_csv(required[2])]

    fn_selected: list[dict[str, object]] = []
    for bucket in ("small_le_q1", "medium_q1_to_q3", "large_gt_q3"):
        candidates = [
            row
            for row in gt_rows
            if row["area_bucket"] == bucket and int(row["false_negative_pixels"]) > 0
        ]
        candidates.sort(key=lambda row: int(row["false_negative_pixels"]), reverse=True)
        fn_selected.extend(candidates[:10])
    if len(fn_selected) != 30:
        raise RuntimeError(f"Could not select frozen 30 FN components: {len(fn_selected)}")

    for row in pred_rows:
        metadata = per_image[str(row["sample_key"])]
        row["sample_class"] = metadata["sample_class"]
        row["has_fire_core"] = metadata["has_fire_core"].lower() == "true"
    fp_candidates = [row for row in pred_rows if int(row["false_positive_pixels"]) > 0]
    empty = sorted(
        (
            row
            for row in fp_candidates
            if row["sample_class"] == "Fire" and not bool(row["has_fire_core"])
        ),
        key=lambda row: int(row["false_positive_pixels"]),
        reverse=True,
    )
    nonempty = sorted(
        (
            row
            for row in fp_candidates
            if row["sample_class"] == "Fire" and bool(row["has_fire_core"])
        ),
        key=lambda row: int(row["false_positive_pixels"]),
        reverse=True,
    )
    fp_selected = empty[:15] + nonempty[:5]
    if len(fp_selected) < 20:
        selected_ids = {(row["sample_key"], row["component_id"]) for row in fp_selected}
        remaining = sorted(
            (
                row
                for row in fp_candidates
                if (row["sample_key"], row["component_id"]) not in selected_ids
            ),
            key=lambda row: int(row["false_positive_pixels"]),
            reverse=True,
        )
        fp_selected.extend(remaining[: 20 - len(fp_selected)])
    if len(fp_selected) != 20:
        raise RuntimeError(f"Could not select frozen 20 FP components: {len(fp_selected)}")

    tp_selected: list[dict[str, object]] = []
    for bucket, count in (("small_le_q1", 5), ("medium_q1_to_q3", 10), ("large_gt_q3", 5)):
        candidates = [
            row
            for row in gt_rows
            if row["area_bucket"] == bucket and int(row["true_positive_pixels"]) > 0
        ]
        tp_selected.extend(select_even_temperature(candidates, count))
    if len(tp_selected) != 20:
        raise RuntimeError(f"Could not select frozen 20 TP components: {len(tp_selected)}")

    groups = (("FN", fn_selected), ("FP", fp_selected), ("TP", tp_selected))
    output.mkdir(parents=True, exist_ok=True)
    visual_dir = output / "visuals"
    visual_dir.mkdir(parents=True, exist_ok=True)
    checklist: list[dict[str, object]] = []
    manifest_items: list[dict[str, object]] = []
    running_index = 1
    for category, rows in groups:
        group_paths: list[Path] = []
        for row in rows:
            sample_key = str(row["sample_key"])
            source = split_rows[sample_key]
            rgb = np.asarray(
                Image.open(resolve_path(data_root, source["corrected_rgb_path"])).convert("RGB"),
                dtype=np.uint8,
            )
            thermal_jpg = np.asarray(
                Image.open(resolve_path(data_root, source["raw_thermal_path"])).convert("L"),
                dtype=np.uint8,
            )
            temperature = np.asarray(
                Image.open(resolve_path(data_root, source["thermal_tiff_path"])), dtype=np.float32
            )
            label = np.asarray(
                Image.open(resolve_path(data_root, source["temperature_mask_path"])), dtype=np.uint8
            )
            prediction_path = required[3] / f"{safe_name(sample_key)}.png"
            prediction = np.asarray(Image.open(prediction_path), dtype=np.uint8)
            component_id = int(row["component_id"])
            if category in {"FN", "TP"}:
                component_map, _ = connected_components(label == FIRE_ID)
            else:
                # The frozen profiling CSV built predicted components only on
                # valid partial-label pixels.  Reapply the same mask here so
                # ignored boundary pixels cannot merge/split components and
                # change their recorded IDs.
                component_map, _ = connected_components(
                    (prediction == FIRE_ID) & (label != IGNORE_ID)
                )
            focus = component_map == component_id
            if not focus.any():
                raise RuntimeError(f"Missing {category} component {sample_key}/{component_id}")
            temperature_mean = float(temperature[focus].mean())
            temperature_max = float(temperature[focus].max())
            audit_id = f"{running_index:03d}_{category}"
            caption = (
                f"{audit_id} {sample_key} component={component_id} area={int(focus.sum())} "
                f"Tmean={temperature_mean:.1f}C Tmax={temperature_max:.1f}C"
            )
            visual = build_visual(
                rgb, thermal_jpg, temperature, label, prediction, focus, caption
            )
            visual_path = visual_dir / f"{audit_id}_{safe_name(sample_key)}_c{component_id}.jpg"
            Image.fromarray(visual).save(visual_path, quality=92)
            group_paths.append(visual_path)
            checklist.append(
                {
                    "audit_id": audit_id,
                    "category": category,
                    "sample_key": sample_key,
                    "component_id": component_id,
                    "area_bucket": row.get("area_bucket", row.get("area_bucket_using_gt_thresholds", "")),
                    "area_pixels": int(focus.sum()),
                    "temperature_mean_c": temperature_mean,
                    "temperature_max_c": temperature_max,
                    "primary_cause": "",
                    "visible_flame_yes_no_uncertain": "",
                    "registration_mismatch_yes_no_uncertain": "",
                    "label_uncertain_yes_no": "",
                    "counts_as_active_fire_yes_no_uncertain": "",
                    "notes": "",
                }
            )
            manifest_items.append(
                {
                    "audit_id": audit_id,
                    "category": category,
                    "sample_key": sample_key,
                    "component_id": component_id,
                    "visual": str(visual_path),
                }
            )
            running_index += 1
        save_contact_sheet(group_paths, output / f"{category.lower()}_contact_sheet.jpg")
    write_csv(output / "manual_error_audit_checklist.csv", checklist)
    readme = [
        "# Fusion基线人工错误审查",
        "",
        "固定内容：30个FN、20个FP、20个TP；只读取split v2 validation。",
        "",
        "`primary_cause`只允许填写以下一个主因：",
        "",
        "```text",
        "小火/远距离火",
        "低温弱火",
        "烟雾遮挡",
        "火焰边缘",
        "热地面或余热",
        "反光或高亮区域",
        "RGB/IR错位",
        "标签不确定",
        "其他",
        "```",
        "",
        "其余判断列填写 `yes`、`no` 或 `uncertain`，不要修改前七列。",
    ]
    (output / "README_MANUAL_REVIEW.md").write_text(
        "\n".join(readme) + "\n", encoding="utf-8"
    )
    manifest = {
        "audit": "flame3_fusion_30fn_20fp_20tp_manual_review",
        "validation_only": True,
        "test_touched": False,
        "selection": {
            "fn": "10 highest-FN components from each GT area bucket",
            "fp": "15 largest FP components from empty-Fire-core frames and 5 from nonempty frames",
            "tp": "5/10/5 small/medium/large components evenly sampled across component temperature",
        },
        "counts": {"FN": 30, "FP": 20, "TP": 20},
        "items": manifest_items,
        "integrity": {
            "split_csv_sha256": sha256(split_csv),
            "per_image_metrics_sha256": sha256(required[0]),
            "gt_components_sha256": sha256(required[1]),
            "predicted_components_sha256": sha256(required[2]),
        },
    }
    (output / "manual_error_audit_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
