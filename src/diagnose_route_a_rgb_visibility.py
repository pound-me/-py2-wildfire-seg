from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from baseline_runtime import PROJECT_ROOT, build_dataset, load_config, seed_everything


PALETTE = np.asarray(
    [[0, 0, 0], [255, 190, 0], [255, 35, 35]], dtype=np.uint8
)


def colorize(mask: np.ndarray) -> np.ndarray:
    result = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for class_index, color in enumerate(PALETTE):
        result[mask == class_index] = color
    return result


def overlay(image: np.ndarray, mask: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    result = image.astype(np.float32).copy()
    foreground = (mask > 0) & (mask < len(PALETTE))
    colors = colorize(mask).astype(np.float32)
    result[foreground] = (
        (1.0 - alpha) * result[foreground] + alpha * colors[foreground]
    )
    return np.clip(result, 0, 255).astype(np.uint8)


def make_panel(image: np.ndarray, title: str) -> Image.Image:
    title_height = 22
    height, width = image.shape[:2]
    panel = Image.new("RGB", (width, height + title_height), (18, 18, 18))
    panel.paste(Image.fromarray(image), (0, title_height))
    ImageDraw.Draw(panel).text((5, 4), title, fill=(255, 255, 255))
    return panel


def make_tile(record: dict) -> Image.Image:
    rgb = record["rgb"]
    ir = np.repeat(record["ir"][..., None], 3, axis=2)
    panels = [
        make_panel(rgb, "RGB"),
        make_panel(ir, "IR"),
        make_panel(overlay(rgb, record["label"]), "GT overlay"),
        make_panel(overlay(rgb, record["prediction"]), "Prediction overlay"),
    ]
    width = rgb.shape[1]
    header_height = 38
    tile = Image.new(
        "RGB", (width * 4, panels[0].height + header_height), (8, 8, 8)
    )
    draw = ImageDraw.Draw(tile)
    draw.text(
        (7, 5),
        (
            f"#{record['rank']:02d} {record['name']}  "
            f"FireIoU={record['iou_fire']:.3f}  "
            f"lowY={record['low_luma_ratio_80']:.2%}  "
            f"smoke-near={record['smoke_proximity_ratio']:.2%}"
        ),
        fill=(255, 255, 255),
    )
    draw.text(
        (7, 21),
        (
            f"GT fire px={record['gt_fire_pixels']}  "
            f"precision={record['precision_fire']:.3f}  "
            f"recall={record['recall_fire']:.3f}"
        ),
        fill=(190, 190, 190),
    )
    for index, panel in enumerate(panels):
        tile.paste(panel, (index * width, header_height))
    return tile


def save_contact_sheet(tiles: list[Image.Image], path: Path) -> None:
    thumbnail_width = 640
    thumbnails = []
    for tile in tiles:
        ratio = thumbnail_width / tile.width
        thumbnails.append(
            tile.resize(
                (thumbnail_width, int(round(tile.height * ratio))),
                Image.Resampling.LANCZOS,
            )
        )
    columns = 2
    rows = (len(thumbnails) + columns - 1) // columns
    cell_height = max(image.height for image in thumbnails)
    sheet = Image.new(
        "RGB", (columns * thumbnail_width, rows * cell_height), (28, 28, 28)
    )
    for index, tile in enumerate(thumbnails):
        sheet.paste(
            tile,
            ((index % columns) * thumbnail_width, (index // columns) * cell_height),
        )
    sheet.save(path)


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Route-A RGB/IR visibility diagnostics on validation data."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=(
            PROJECT_ROOT
            / "configs"
            / "adopted_protocol_label_fix"
            / "pidnet_s_rgb_baseline_adopted_100e.yaml"
        ),
    )
    parser.add_argument(
        "--evaluation-dir",
        type=Path,
        default=(
            PROJECT_ROOT
            / "experiments"
            / "pidnet_s_rgb_baseline_adopted"
            / "baseline_a1_100e_label_fix"
            / "val_best"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            PROJECT_ROOT
            / "experiments"
            / "route_a_diagnostics"
            / "rgb_visibility_val"
        ),
    )
    parser.add_argument("--worst-count", type=int, default=20)
    parser.add_argument("--low-luma-threshold", type=float, default=80.0)
    parser.add_argument("--smoke-radius", type=int, default=5)
    args = parser.parse_args()

    config = load_config(args.config.resolve())
    if config["MODE"] != "rgb":
        raise ValueError("The diagnostic reference config must be MODE: rgb.")
    seed_everything(int(config["SEED"]))
    fusion_config = dict(config)
    fusion_config["MODE"] = "fusion"
    dataset = build_dataset(fusion_config, "val")

    evaluation_dir = args.evaluation_dir.resolve()
    metrics_path = evaluation_dir / "per_image_metrics.csv"
    prediction_dir = evaluation_dir / "predictions_raw"
    with metrics_path.open("r", encoding="utf-8-sig", newline="") as stream:
        metric_rows = {row["name"]: row for row in csv.DictReader(stream)}

    radius = max(int(args.smoke_radius), 0)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
    )
    thresholds = (60.0, float(args.low_luma_threshold), 100.0)
    all_rows: list[dict] = []
    total_fire_pixels = 0
    total_smoke_near = 0
    total_exact_overlap = 0
    total_low = {threshold: 0 for threshold in thresholds}

    for index in range(len(dataset)):
        images, label, _, raw_name = dataset[index][:4]
        name = str(raw_name[0])
        if name not in metric_rows:
            raise KeyError(f"No saved validation metric for: {name}")
        rgb = np.clip(images[:3].transpose(1, 2, 0) * 255.0, 0, 255).astype(
            np.uint8
        )
        ir = np.clip(images[3] * 255.0, 0, 255).astype(np.uint8)
        label = np.asarray(label, dtype=np.uint8)
        prediction_path = prediction_dir / (
            Path(name).stem.replace("XXX", "rgb") + "_pred.png"
        )
        prediction = np.asarray(Image.open(prediction_path), dtype=np.uint8)
        if prediction.shape != label.shape:
            raise RuntimeError(
                f"Prediction/label shape mismatch for {name}: "
                f"{prediction.shape} vs {label.shape}"
            )

        fire = label == 2
        fire_pixels = int(fire.sum())
        if fire_pixels == 0:
            continue
        smoke = label == 1
        smoke_near = cv2.dilate(smoke.astype(np.uint8), kernel) > 0
        exact_overlap = int((fire & smoke).sum())
        smoke_near_count = int((fire & smoke_near).sum())
        luma = (
            0.2126 * rgb[..., 0]
            + 0.7152 * rgb[..., 1]
            + 0.0722 * rgb[..., 2]
        )
        low_counts = {
            threshold: int((fire & (luma <= threshold)).sum())
            for threshold in thresholds
        }
        saved = metric_rows[name]
        union = int(((prediction == 2) | fire).sum())
        intersection = int(((prediction == 2) & fire).sum())
        recomputed_iou = intersection / union if union else 0.0
        saved_iou = float(saved["iou_fire"])
        if abs(recomputed_iou - saved_iou) > 1e-12:
            raise RuntimeError(
                f"Saved/recomputed Fire IoU mismatch for {name}: "
                f"{saved_iou} vs {recomputed_iou}"
            )

        row = {
            "index": index,
            "name": name,
            "iou_fire": saved_iou,
            "precision_fire": float(saved["precision_fire"]),
            "recall_fire": float(saved["recall_fire"]),
            "gt_fire_pixels": fire_pixels,
            "mean_fire_luma": float(luma[fire].mean()),
            "low_luma_ratio_60": low_counts[60.0] / fire_pixels,
            "low_luma_ratio_80": low_counts[float(args.low_luma_threshold)]
            / fire_pixels,
            "low_luma_ratio_100": low_counts[100.0] / fire_pixels,
            "smoke_proximity_ratio": smoke_near_count / fire_pixels,
            "exact_fire_smoke_overlap_ratio": exact_overlap / fire_pixels,
            "rgb": rgb,
            "ir": ir,
            "label": label,
            "prediction": prediction,
        }
        all_rows.append(row)
        total_fire_pixels += fire_pixels
        total_smoke_near += smoke_near_count
        total_exact_overlap += exact_overlap
        for threshold in thresholds:
            total_low[threshold] += low_counts[threshold]

    if not all_rows or total_fire_pixels == 0:
        raise RuntimeError("The validation split contains no Fire pixels.")
    all_rows.sort(
        key=lambda row: (row["iou_fire"], row["recall_fire"], row["name"])
    )
    worst = all_rows[: max(int(args.worst_count), 0)]
    for rank, row in enumerate(worst, start=1):
        row["rank"] = rank

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    data_fields = [
        "index",
        "name",
        "iou_fire",
        "precision_fire",
        "recall_fire",
        "gt_fire_pixels",
        "mean_fire_luma",
        "low_luma_ratio_60",
        "low_luma_ratio_80",
        "low_luma_ratio_100",
        "smoke_proximity_ratio",
        "exact_fire_smoke_overlap_ratio",
    ]
    write_csv(
        output_dir / "fire_visibility_stats_per_image.csv", all_rows, data_fields
    )
    checklist_fields = [
        "rank",
        *data_fields[1:],
        "manual_rgb_visibility_visible_partial_invisible",
        "manual_smoke_occlusion_none_partial_heavy",
        "manual_ir_clearer_yes_no",
        "manual_small_fire_yes_no",
        "manual_notes",
    ]
    write_csv(
        output_dir / "worst20_manual_checklist.csv", worst, checklist_fields
    )

    tiles = [make_tile(row) for row in worst]
    save_contact_sheet(tiles, output_dir / "worst20_contact_sheet.png")
    for page_start in range(0, len(tiles), 5):
        page_tiles = tiles[page_start : page_start + 5]
        page = Image.new(
            "RGB",
            (page_tiles[0].width, sum(tile.height for tile in page_tiles)),
            (20, 20, 20),
        )
        y = 0
        for tile in page_tiles:
            page.paste(tile, (0, y))
            y += tile.height
        page.save(output_dir / f"worst20_page_{page_start // 5 + 1}.png")

    summary = {
        "split": "val",
        "test_split_used": False,
        "reference_checkpoint_evaluation": str(evaluation_dir),
        "validation_samples": len(dataset),
        "fire_containing_samples": len(all_rows),
        "total_fire_pixels": total_fire_pixels,
        "primary_low_luminance_threshold_0_255": float(
            args.low_luma_threshold
        ),
        "fire_pixel_low_luminance_ratio_60": total_low[60.0]
        / total_fire_pixels,
        "fire_pixel_low_luminance_ratio_80": total_low[
            float(args.low_luma_threshold)
        ]
        / total_fire_pixels,
        "fire_pixel_low_luminance_ratio_100": total_low[100.0]
        / total_fire_pixels,
        "smoke_proximity_radius_pixels": radius,
        "fire_pixel_smoke_proximity_ratio": total_smoke_near
        / total_fire_pixels,
        "exact_fire_smoke_overlap_ratio": total_exact_overlap
        / total_fire_pixels,
        "definitions": {
            "luminance": (
                "BT.709 Y = 0.2126R + 0.7152G + 0.0722B on the "
                "evaluation-resolution RGB image"
            ),
            "low_luminance": "Ground-truth Fire pixel with Y <= threshold",
            "smoke_proximity_proxy": (
                "Ground-truth Fire pixel within the configured radius of a "
                "ground-truth Smoke pixel"
            ),
            "exact_overlap_note": (
                "The three-class mask is mutually exclusive, so a pixel "
                "cannot be both Smoke and Fire; exact overlap is structurally zero."
            ),
        },
        "worst_selection": (
            "Lowest saved per-image Fire IoU among validation images containing "
            "at least one ground-truth Fire pixel; ties use Fire recall then filename."
        ),
        "worst_count": len(worst),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Checklist: {output_dir / 'worst20_manual_checklist.csv'}")
    print(f"Contact sheet: {output_dir / 'worst20_contact_sheet.png'}")


if __name__ == "__main__":
    main()
