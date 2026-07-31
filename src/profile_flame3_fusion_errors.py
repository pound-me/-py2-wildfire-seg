from __future__ import annotations

"""Read-only FLAME3 validation error profiling for a frozen checkpoint.

The FLAME3 temperature masks are partial labels, not complete three-class masks.
Accordingly this script evaluates only the temperature-derived active-Fire core and
never reports Smoke IoU or three-class mIoU.  It deliberately accepts val.csv only.
"""

import argparse
import csv
import hashlib
import json
import math
import platform
import re
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from baseline_runtime import build_model, load_config, seed_everything


BACKGROUND_ID = 0
SMOKE_ID = 1
FIRE_ID = 2
IGNORE_ID = 255
EXPECTED_VAL_SAMPLES = 134

TEMPERATURE_EDGES = (-math.inf, 0.0, 25.0, 50.0, 80.0, 100.0, 150.0, 200.0, 300.0, 400.0, 500.0, math.inf)
TEMPERATURE_LABELS = (
    "lt_0c",
    "0_to_25c",
    "25_to_50c",
    "50_to_80c",
    "80_to_100c",
    "100_to_150c",
    "150_to_200c",
    "200_to_300c",
    "300_to_400c",
    "400_to_500c",
    "ge_500c",
)
LUMA_EDGES = (-math.inf, 0.15, 0.30, 0.50, 0.70, math.inf)
LUMA_LABELS = ("lt_0.15", "0.15_to_0.30", "0.30_to_0.50", "0.50_to_0.70", "ge_0.70")
CONTRAST_EDGES = (-math.inf, 0.03, 0.08, 0.15, math.inf)
CONTRAST_LABELS = ("lt_0.03", "0.03_to_0.08", "0.08_to_0.15", "ge_0.15")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile a frozen FLAME3 Fusion checkpoint on split-v2 validation only."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split-csv", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--visualizations", type=int, default=30)
    parser.add_argument("--boundary-radius", type=int, default=3)
    parser.add_argument("--local-contrast-window", type=int, default=15)
    parser.add_argument("--low-luma-threshold", type=float, default=0.20)
    parser.add_argument("--low-contrast-threshold", type=float, default=0.05)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def resolve_data_path(root: Path, raw: str) -> Path:
    path = Path(raw)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


class Flame3ProfileDataset(Dataset):
    def __init__(self, split_csv: Path, data_root: Path, mode: str) -> None:
        self.split_csv = split_csv.resolve()
        self.data_root = data_root.resolve()
        if self.split_csv.name.lower() != "val.csv":
            raise ValueError(f"This diagnostic permits val.csv only: {self.split_csv}")
        self.mode = str(mode).lower()
        if self.mode not in {"rgb", "ir", "fusion"}:
            raise ValueError(f"Unsupported mode: {mode}")
        with self.split_csv.open("r", encoding="utf-8-sig", newline="") as handle:
            self.rows = list(csv.DictReader(handle))
        if len(self.rows) != EXPECTED_VAL_SAMPLES:
            raise RuntimeError(
                f"Expected split-v2 validation size {EXPECTED_VAL_SAMPLES}, got {len(self.rows)}"
            )
        keys = [row["sample_key"] for row in self.rows]
        if len(keys) != len(set(keys)):
            raise RuntimeError("Duplicate sample_key values in validation CSV")

    def __len__(self) -> int:
        return len(self.rows)

    def paths_for(self, row: dict[str, str]) -> dict[str, Path]:
        return {
            "rgb": resolve_data_path(self.data_root, row["corrected_rgb_path"]),
            "thermal_jpg": resolve_data_path(self.data_root, row["raw_thermal_path"]),
            "temperature": resolve_data_path(self.data_root, row["thermal_tiff_path"]),
            "label": resolve_data_path(self.data_root, row["temperature_mask_path"]),
        }

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.rows[index]
        paths = self.paths_for(row)
        for path in paths.values():
            if not path.is_file():
                raise FileNotFoundError(f"Missing FLAME3 validation file: {path}")
        rgb_u8 = np.asarray(Image.open(paths["rgb"]).convert("RGB"), dtype=np.uint8)
        ir_u8 = np.asarray(Image.open(paths["thermal_jpg"]).convert("L"), dtype=np.uint8)
        temperature = np.asarray(Image.open(paths["temperature"]), dtype=np.float32)
        label = np.asarray(Image.open(paths["label"]), dtype=np.uint8)
        if rgb_u8.shape[:2] != ir_u8.shape or ir_u8.shape != temperature.shape or ir_u8.shape != label.shape:
            raise RuntimeError(f"Shape mismatch for {row['sample_key']}")
        values = set(int(value) for value in np.unique(label).tolist())
        if not values.issubset({BACKGROUND_ID, FIRE_ID, IGNORE_ID}):
            raise RuntimeError(f"Unexpected pseudo-label values {sorted(values)} for {row['sample_key']}")
        if row["sample_class"] == "No Fire" and values != {BACKGROUND_ID}:
            raise RuntimeError(f"No Fire validation sample is not hard Background: {row['sample_key']}")
        rgb = rgb_u8.astype(np.float32) / 255.0
        ir = ir_u8.astype(np.float32) / 255.0
        rgb_chw = np.transpose(rgb, (2, 0, 1))
        if self.mode == "rgb":
            image = rgb_chw
        elif self.mode == "ir":
            image = ir[None, :, :]
        else:
            image = np.concatenate((rgb_chw, ir[None, :, :]), axis=0)
        return {
            "image": torch.from_numpy(image.copy()).float(),
            "rgb": torch.from_numpy(rgb_u8.copy()),
            "temperature": torch.from_numpy(temperature.copy()).float(),
            "label": torch.from_numpy(label.copy()).long(),
            "sample_key": row["sample_key"],
            "sample_class": row["sample_class"],
            "sample_id": row["sample_id"],
        }


@dataclass
class NumericDistribution:
    edges: tuple[float, ...]
    labels: tuple[str, ...]
    count: int = 0
    total: float = 0.0
    total_squared: float = 0.0
    minimum: float = math.inf
    maximum: float = -math.inf
    histogram: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        if len(self.edges) != len(self.labels) + 1:
            raise ValueError("Histogram edges and labels do not match")
        self.histogram = np.zeros(len(self.labels), dtype=np.int64)

    def update(self, image: np.ndarray, mask: np.ndarray) -> None:
        values = np.asarray(image[mask], dtype=np.float64)
        values = values[np.isfinite(values)]
        if values.size == 0:
            return
        self.count += int(values.size)
        self.total += float(values.sum(dtype=np.float64))
        self.total_squared += float(np.square(values).sum(dtype=np.float64))
        self.minimum = min(self.minimum, float(values.min()))
        self.maximum = max(self.maximum, float(values.max()))
        self.histogram += np.histogram(values, bins=np.asarray(self.edges, dtype=np.float64))[0]

    def summary(self) -> dict[str, object]:
        mean = self.total / self.count if self.count else 0.0
        variance = max(self.total_squared / self.count - mean * mean, 0.0) if self.count else 0.0
        return {
            "pixels": self.count,
            "mean": mean,
            "std": math.sqrt(variance),
            "min": self.minimum if self.count else None,
            "max": self.maximum if self.count else None,
            "bins": [
                {
                    "bin": label,
                    "pixels": int(value),
                    "ratio": int(value) / max(self.count, 1),
                }
                for label, value in zip(self.labels, self.histogram.tolist())
            ],
        }


@dataclass
class VisibilityCounter:
    pixels: int = 0
    low_luma: int = 0
    low_contrast: int = 0
    low_either: int = 0
    low_both: int = 0

    def update(
        self,
        mask: np.ndarray,
        luma: np.ndarray,
        contrast: np.ndarray,
        luma_threshold: float,
        contrast_threshold: float,
    ) -> None:
        low_luma = luma < luma_threshold
        low_contrast = contrast < contrast_threshold
        self.pixels += int(mask.sum())
        self.low_luma += int((mask & low_luma).sum())
        self.low_contrast += int((mask & low_contrast).sum())
        self.low_either += int((mask & (low_luma | low_contrast)).sum())
        self.low_both += int((mask & low_luma & low_contrast).sum())

    def summary(self) -> dict[str, float | int]:
        return {
            "pixels": self.pixels,
            "low_luma_pixels": self.low_luma,
            "low_luma_ratio": self.low_luma / max(self.pixels, 1),
            "low_contrast_pixels": self.low_contrast,
            "low_contrast_ratio": self.low_contrast / max(self.pixels, 1),
            "low_either_pixels": self.low_either,
            "low_either_ratio": self.low_either / max(self.pixels, 1),
            "low_both_pixels": self.low_both,
            "low_both_ratio": self.low_both / max(self.pixels, 1),
        }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def connected_components(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    count, component_map, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    return component_map, stats[:count]


def area_bucket(area: int, q1: float, q3: float) -> str:
    if area <= q1:
        return "small_le_q1"
    if area <= q3:
        return "medium_q1_to_q3"
    return "large_gt_q3"


def binary_boundary(mask: np.ndarray) -> np.ndarray:
    eroded = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1)
    return mask & (eroded == 0)


def dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.copy()
    size = 2 * radius + 1
    return cv2.dilate(mask.astype(np.uint8), np.ones((size, size), np.uint8), iterations=1) > 0


def local_rgb_statistics(rgb_u8: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    rgb = rgb_u8.astype(np.float32) / 255.0
    luma = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    mean = cv2.boxFilter(luma, cv2.CV_32F, (window, window), normalize=True, borderType=cv2.BORDER_REFLECT)
    squared_mean = cv2.boxFilter(luma * luma, cv2.CV_32F, (window, window), normalize=True, borderType=cv2.BORDER_REFLECT)
    contrast = np.sqrt(np.maximum(squared_mean - mean * mean, 0.0))
    return luma, contrast


def robust_temperature_color(temperature: np.ndarray) -> np.ndarray:
    finite = temperature[np.isfinite(temperature)]
    if finite.size == 0:
        return np.zeros((*temperature.shape, 3), dtype=np.uint8)
    low, high = np.percentile(finite, [1.0, 99.0])
    if high <= low:
        high = low + 1.0
    normalized = np.clip((temperature - low) / (high - low), 0.0, 1.0)
    bgr = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def colorize_label(label: np.ndarray) -> np.ndarray:
    panel = np.zeros((*label.shape, 3), dtype=np.uint8)
    panel[label == FIRE_ID] = (255, 180, 0)
    panel[label == IGNORE_ID] = (255, 0, 255)
    return panel


def colorize_prediction(prediction: np.ndarray) -> np.ndarray:
    panel = np.zeros((*prediction.shape, 3), dtype=np.uint8)
    panel[prediction == SMOKE_ID] = (155, 155, 155)
    panel[prediction == FIRE_ID] = (255, 0, 0)
    return panel


def error_overlay(rgb: np.ndarray, label: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    valid = label != IGNORE_ID
    target = label == FIRE_ID
    predicted = prediction == FIRE_ID
    tp = valid & target & predicted
    fn = valid & target & ~predicted
    fp = valid & ~target & predicted
    overlay = rgb.astype(np.float32).copy()
    for mask, color in ((tp, (0, 255, 0)), (fn, (0, 120, 255)), (fp, (255, 0, 0))):
        overlay[mask] = 0.30 * overlay[mask] + 0.70 * np.asarray(color, dtype=np.float32)
    return np.clip(overlay, 0, 255).astype(np.uint8)


def caption_panel(panel: np.ndarray, title: str) -> np.ndarray:
    output = np.array(panel, dtype=np.uint8, copy=True)
    cv2.rectangle(output, (0, 0), (output.shape[1], 30), (0, 0, 0), -1)
    cv2.putText(output, title, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.53, (255, 255, 255), 1, cv2.LINE_AA)
    return output


def save_visualization(
    path: Path,
    rgb: np.ndarray,
    temperature: np.ndarray,
    label: np.ndarray,
    prediction: np.ndarray,
    caption: str,
) -> None:
    panels = (
        caption_panel(rgb, "Corrected RGB"),
        caption_panel(robust_temperature_color(temperature), "Celsius TIFF (robust color)"),
        caption_panel(colorize_label(label), "Partial label: Fire=orange Ignore=magenta"),
        caption_panel(colorize_prediction(prediction), "Prediction: Smoke=gray Fire=red"),
        caption_panel(error_overlay(rgb, label, prediction), "TP=green FN=blue FP=red"),
    )
    canvas = np.concatenate(panels, axis=1)
    cv2.rectangle(canvas, (0, canvas.shape[0] - 28), (canvas.shape[1], canvas.shape[0]), (0, 0, 0), -1)
    cv2.putText(canvas, caption, (8, canvas.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (255, 255, 255), 1, cv2.LINE_AA)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas).save(path, quality=92)


def contact_tile(rgb: np.ndarray, overlay: np.ndarray, caption: str) -> np.ndarray:
    target_h, target_w = 192, 240
    left = cv2.resize(rgb, (target_w, target_h), interpolation=cv2.INTER_AREA)
    right = cv2.resize(overlay, (target_w, target_h), interpolation=cv2.INTER_AREA)
    tile = np.zeros((target_h + 48, target_w * 2, 3), dtype=np.uint8)
    tile[:target_h, :target_w] = left
    tile[:target_h, target_w:] = right
    cv2.putText(tile, "RGB", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(tile, "TP green / FN blue / FP red", (target_w + 5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.39, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(tile, caption[:72], (6, target_h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    if len(caption) > 72:
        cv2.putText(tile, caption[72:144], (6, target_h + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    return tile


def save_contact_sheet(path: Path, tiles: list[np.ndarray], columns: int = 4) -> None:
    if not tiles:
        return
    height, width = tiles[0].shape[:2]
    rows = math.ceil(len(tiles) / columns)
    canvas = np.zeros((rows * height, columns * width, 3), dtype=np.uint8)
    for index, tile in enumerate(tiles):
        row, column = divmod(index, columns)
        canvas[row * height:(row + 1) * height, column * width:(column + 1) * width] = tile
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas).save(path, quality=92)


def fire_metrics(tp: int, fp: int, fn: int, tn: int) -> dict[str, float | int]:
    return {
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
        "fire_iou": tp / max(tp + fp + fn, 1),
        "fire_precision": tp / max(tp + fp, 1),
        "fire_recall": tp / max(tp + fn, 1),
        "fire_f1": 2 * tp / max(2 * tp + fp + fn, 1),
    }


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    if args.boundary_radius < 0:
        raise ValueError("boundary-radius cannot be negative")
    if args.local_contrast_window <= 0 or args.local_contrast_window % 2 == 0:
        raise ValueError("local-contrast-window must be a positive odd integer")
    config_path = args.config.resolve()
    checkpoint_path = args.checkpoint.resolve()
    split_csv = args.split_csv.resolve()
    data_root = args.data_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not config_path.is_file() or not checkpoint_path.is_file() or not split_csv.is_file():
        raise FileNotFoundError("Config, checkpoint, or validation CSV is missing")
    config = load_config(config_path)
    if str(config.get("DATASET_TYPE", "")).lower() != "flame3_csv":
        raise ValueError("The config is not a FLAME3 CSV experiment")
    if str(config.get("TRAINING_OBJECTIVE", "")).lower() != "partial_label":
        raise ValueError("The config does not use the FLAME3 partial-label objective")
    if str(config.get("MODE", "")).lower() != "fusion":
        raise ValueError("This error profile is preregistered for the Fusion baseline only")
    if int(config.get("FIRE_CLASS_INDEX", FIRE_ID)) != FIRE_ID:
        raise ValueError("Unexpected Fire class index")

    seed_everything(int(config.get("SEED", 200)))
    dataset = Flame3ProfileDataset(split_csv, data_root, str(config["MODE"]))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    all_gt_areas: list[int] = []
    for row in dataset.rows:
        label_path = dataset.paths_for(row)["label"]
        label = np.asarray(Image.open(label_path), dtype=np.uint8)
        _, stats = connected_components(label == FIRE_ID)
        all_gt_areas.extend(int(value) for value in stats[1:, cv2.CC_STAT_AREA])
    if not all_gt_areas:
        raise RuntimeError("No active-Fire component exists in validation")
    q1, q3 = (float(value) for value in np.percentile(np.asarray(all_gt_areas, dtype=np.float64), [25.0, 75.0]))

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "model_state_dict" not in checkpoint:
        raise KeyError("Checkpoint lacks model_state_dict")
    model = build_model(config)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    model = model.to(device).eval()

    categories = ("true_positive", "false_negative", "false_positive", "true_negative")
    temperature_distributions = {
        category: NumericDistribution(TEMPERATURE_EDGES, TEMPERATURE_LABELS) for category in categories
    }
    luma_distributions = {
        category: NumericDistribution(LUMA_EDGES, LUMA_LABELS) for category in categories
    }
    contrast_distributions = {
        category: NumericDistribution(CONTRAST_EDGES, CONTRAST_LABELS) for category in categories
    }
    visibility = {category: VisibilityCounter() for category in categories}
    component_aggregate = {
        bucket: {"component_count": 0, "pixels": 0, "true_positive": 0, "false_negative": 0}
        for bucket in ("small_le_q1", "medium_q1_to_q3", "large_gt_q3")
    }
    predicted_component_aggregate = {
        bucket: {"component_count": 0, "pixels": 0, "true_positive": 0, "false_positive": 0}
        for bucket in ("small_le_q1", "medium_q1_to_q3", "large_gt_q3")
    }
    boundary_counts = {
        "fn_near_gt_boundary": 0,
        "fn_gt_interior": 0,
        "fp_near_gt_boundary": 0,
        "fp_far_from_gt_boundary": 0,
        "tp_near_gt_boundary": 0,
        "tp_gt_interior": 0,
    }
    fp_groups = {
        "fire_folder_nonempty_core": {
            "images": 0,
            "valid_pixels": 0,
            "nonfire_valid_pixels": 0,
            "false_positive_pixels": 0,
        },
        "fire_folder_empty_core": {
            "images": 0,
            "valid_pixels": 0,
            "nonfire_valid_pixels": 0,
            "false_positive_pixels": 0,
        },
        "no_fire": {
            "images": 0,
            "valid_pixels": 0,
            "nonfire_valid_pixels": 0,
            "false_positive_pixels": 0,
        },
    }
    totals = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    per_image: list[dict[str, object]] = []
    gt_component_rows: list[dict[str, object]] = []
    predicted_component_rows: list[dict[str, object]] = []
    prediction_dir = output_dir / "predictions_raw"
    prediction_dir.mkdir(parents=True, exist_ok=True)

    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].numpy().astype(np.uint8)
            rgbs = batch["rgb"].numpy().astype(np.uint8)
            temperatures = batch["temperature"].numpy().astype(np.float32)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=args.amp and device.type == "cuda"):
                outputs = model(images)
                logits = outputs[1] if isinstance(outputs, (tuple, list)) else outputs
                if logits.shape[-2:] != labels.shape[-2:]:
                    logits = F.interpolate(
                        logits,
                        size=labels.shape[-2:],
                        mode="bilinear",
                        align_corners=bool(config.get("ALIGN_CORNERS", True)),
                    )
            predictions = logits.argmax(dim=1).cpu().numpy().astype(np.uint8)
            for index in range(predictions.shape[0]):
                prediction = predictions[index]
                label = labels[index]
                rgb = rgbs[index]
                temperature = temperatures[index]
                sample_key = str(batch["sample_key"][index])
                sample_class = str(batch["sample_class"][index])
                sample_id = str(batch["sample_id"][index])
                if not set(int(value) for value in np.unique(prediction)).issubset({0, 1, 2}):
                    raise RuntimeError(f"Invalid prediction IDs for {sample_key}")
                valid = label != IGNORE_ID
                gt_fire = (label == FIRE_ID) & valid
                pred_fire = (prediction == FIRE_ID) & valid
                masks = {
                    "true_positive": gt_fire & pred_fire,
                    "false_negative": gt_fire & ~pred_fire,
                    "false_positive": ~gt_fire & pred_fire & valid,
                    "true_negative": ~gt_fire & ~pred_fire & valid,
                }
                image_counts = {name: int(mask.sum()) for name, mask in masks.items()}
                totals["tp"] += image_counts["true_positive"]
                totals["fn"] += image_counts["false_negative"]
                totals["fp"] += image_counts["false_positive"]
                totals["tn"] += image_counts["true_negative"]
                luma, contrast = local_rgb_statistics(rgb, args.local_contrast_window)
                for category, mask in masks.items():
                    temperature_distributions[category].update(temperature, mask)
                    luma_distributions[category].update(luma, mask)
                    contrast_distributions[category].update(contrast, mask)
                    visibility[category].update(
                        mask,
                        luma,
                        contrast,
                        args.low_luma_threshold,
                        args.low_contrast_threshold,
                    )

                gt_boundary = binary_boundary(gt_fire)
                gt_boundary_band = dilate(gt_boundary, args.boundary_radius) & valid
                boundary_counts["fn_near_gt_boundary"] += int((masks["false_negative"] & gt_boundary_band).sum())
                boundary_counts["fn_gt_interior"] += int((masks["false_negative"] & ~gt_boundary_band).sum())
                boundary_counts["fp_near_gt_boundary"] += int((masks["false_positive"] & gt_boundary_band).sum())
                boundary_counts["fp_far_from_gt_boundary"] += int((masks["false_positive"] & ~gt_boundary_band).sum())
                boundary_counts["tp_near_gt_boundary"] += int((masks["true_positive"] & gt_boundary_band).sum())
                boundary_counts["tp_gt_interior"] += int((masks["true_positive"] & ~gt_boundary_band).sum())

                gt_components, gt_stats = connected_components(gt_fire)
                for component_id in range(1, gt_stats.shape[0]):
                    area = int(gt_stats[component_id, cv2.CC_STAT_AREA])
                    component_mask = gt_components == component_id
                    tp_pixels = int((component_mask & pred_fire).sum())
                    fn_pixels = area - tp_pixels
                    bucket = area_bucket(area, q1, q3)
                    aggregate = component_aggregate[bucket]
                    aggregate["component_count"] += 1
                    aggregate["pixels"] += area
                    aggregate["true_positive"] += tp_pixels
                    aggregate["false_negative"] += fn_pixels
                    gt_component_rows.append({
                        "sample_key": sample_key,
                        "component_id": component_id,
                        "area_pixels": area,
                        "area_bucket": bucket,
                        "true_positive_pixels": tp_pixels,
                        "false_negative_pixels": fn_pixels,
                        "component_recall": tp_pixels / max(area, 1),
                        "x": int(gt_stats[component_id, cv2.CC_STAT_LEFT]),
                        "y": int(gt_stats[component_id, cv2.CC_STAT_TOP]),
                        "width": int(gt_stats[component_id, cv2.CC_STAT_WIDTH]),
                        "height": int(gt_stats[component_id, cv2.CC_STAT_HEIGHT]),
                    })

                pred_components, pred_stats = connected_components(pred_fire)
                for component_id in range(1, pred_stats.shape[0]):
                    area = int(pred_stats[component_id, cv2.CC_STAT_AREA])
                    component_mask = pred_components == component_id
                    fp_pixels = int((component_mask & ~gt_fire & valid).sum())
                    tp_pixels = area - fp_pixels
                    bucket = area_bucket(area, q1, q3)
                    aggregate = predicted_component_aggregate[bucket]
                    aggregate["component_count"] += 1
                    aggregate["pixels"] += area
                    aggregate["true_positive"] += tp_pixels
                    aggregate["false_positive"] += fp_pixels
                    predicted_component_rows.append({
                        "sample_key": sample_key,
                        "component_id": component_id,
                        "area_pixels": area,
                        "area_bucket_using_gt_thresholds": bucket,
                        "true_positive_pixels": tp_pixels,
                        "false_positive_pixels": fp_pixels,
                        "component_precision": tp_pixels / max(area, 1),
                        "x": int(pred_stats[component_id, cv2.CC_STAT_LEFT]),
                        "y": int(pred_stats[component_id, cv2.CC_STAT_TOP]),
                        "width": int(pred_stats[component_id, cv2.CC_STAT_WIDTH]),
                        "height": int(pred_stats[component_id, cv2.CC_STAT_HEIGHT]),
                    })

                has_fire_core = bool(gt_fire.any())
                if sample_class == "No Fire":
                    group = "no_fire"
                elif has_fire_core:
                    group = "fire_folder_nonempty_core"
                else:
                    group = "fire_folder_empty_core"
                fp_groups[group]["images"] += 1
                fp_groups[group]["valid_pixels"] += int(valid.sum())
                fp_groups[group]["nonfire_valid_pixels"] += int((~gt_fire & valid).sum())
                fp_groups[group]["false_positive_pixels"] += image_counts["false_positive"]

                metric = fire_metrics(
                    image_counts["true_positive"],
                    image_counts["false_positive"],
                    image_counts["false_negative"],
                    image_counts["true_negative"],
                )
                prediction_path = prediction_dir / f"{safe_name(sample_key)}.png"
                Image.fromarray(prediction).save(prediction_path)
                per_image.append({
                    "sample_key": sample_key,
                    "sample_class": sample_class,
                    "sample_id": sample_id,
                    "has_fire_core": has_fire_core,
                    "valid_pixels": int(valid.sum()),
                    "fire_core_pixels": int(gt_fire.sum()),
                    "predicted_fire_pixels": int(pred_fire.sum()),
                    "predicted_smoke_pixels": int(((prediction == SMOKE_ID) & valid).sum()),
                    "predicted_fire_ratio": int(pred_fire.sum()) / max(int(valid.sum()), 1),
                    "fn_boundary_pixels": int((masks["false_negative"] & gt_boundary_band).sum()),
                    "fn_interior_pixels": int((masks["false_negative"] & ~gt_boundary_band).sum()),
                    "fp_near_boundary_pixels": int((masks["false_positive"] & gt_boundary_band).sum()),
                    "fp_far_pixels": int((masks["false_positive"] & ~gt_boundary_band).sum()),
                    **metric,
                    "prediction_path": str(prediction_path),
                })

    overall = fire_metrics(totals["tp"], totals["fp"], totals["fn"], totals["tn"])
    if sum(int(row["true_positive"]) for row in per_image) != totals["tp"]:
        raise RuntimeError("Per-image TP total mismatch")
    if sum(int(row["false_positive"]) for row in per_image) != totals["fp"]:
        raise RuntimeError("Per-image FP total mismatch")
    if sum(int(row["false_negative"]) for row in per_image) != totals["fn"]:
        raise RuntimeError("Per-image FN total mismatch")
    if boundary_counts["fn_near_gt_boundary"] + boundary_counts["fn_gt_interior"] != totals["fn"]:
        raise RuntimeError("FN boundary/interior partition mismatch")
    if boundary_counts["fp_near_gt_boundary"] + boundary_counts["fp_far_from_gt_boundary"] != totals["fp"]:
        raise RuntimeError("FP boundary/far partition mismatch")

    for values in component_aggregate.values():
        values["component_recall"] = values["true_positive"] / max(values["pixels"], 1)
        values["fn_share"] = values["false_negative"] / max(totals["fn"], 1)
    for values in predicted_component_aggregate.values():
        values["component_precision"] = values["true_positive"] / max(values["pixels"], 1)
        values["fp_share"] = values["false_positive"] / max(totals["fp"], 1)
    for values in fp_groups.values():
        values["false_positive_ratio"] = values["false_positive_pixels"] / max(values["nonfire_valid_pixels"], 1)

    boundary_summary = {
        **boundary_counts,
        "fn_near_gt_boundary_share": boundary_counts["fn_near_gt_boundary"] / max(totals["fn"], 1),
        "fn_gt_interior_share": boundary_counts["fn_gt_interior"] / max(totals["fn"], 1),
        "fp_near_gt_boundary_share": boundary_counts["fp_near_gt_boundary"] / max(totals["fp"], 1),
        "fp_far_from_gt_boundary_share": boundary_counts["fp_far_from_gt_boundary"] / max(totals["fp"], 1),
    }
    feature_distributions = {
        category: {
            "temperature_celsius": temperature_distributions[category].summary(),
            "rgb_luma_0_to_1": luma_distributions[category].summary(),
            "rgb_local_contrast_std": contrast_distributions[category].summary(),
            "visibility_thresholds": visibility[category].summary(),
        }
        for category in categories
    }

    write_csv(output_dir / "per_image_metrics.csv", per_image)
    write_csv(output_dir / "gt_fire_components.csv", gt_component_rows)
    write_csv(output_dir / "predicted_fire_components.csv", predicted_component_rows)

    positive_rows = sorted(
        (row for row in per_image if bool(row["has_fire_core"])),
        key=lambda row: (float(row["fire_iou"]), -int(row["false_negative"])),
    )
    empty_rows = sorted(
        (row for row in per_image if row["sample_class"] == "Fire" and not bool(row["has_fire_core"])),
        key=lambda row: float(row["predicted_fire_ratio"]),
        reverse=True,
    )
    no_fire_rows = sorted(
        (row for row in per_image if row["sample_class"] == "No Fire"),
        key=lambda row: float(row["predicted_fire_ratio"]),
        reverse=True,
    )
    selected_groups = {
        "worst_positive": positive_rows[: min(20, len(positive_rows))],
        "worst_empty_fire_folder": empty_rows[: min(5, len(empty_rows))],
        "worst_no_fire": no_fire_rows[: min(5, len(no_fire_rows))],
    }
    selected_rows: list[dict[str, object]] = []
    for rows in selected_groups.values():
        selected_rows.extend(rows)
    selected_rows = selected_rows[: args.visualizations]
    selected_keys = {str(row["sample_key"]) for row in selected_rows}
    dataset_rows = {row["sample_key"]: row for row in dataset.rows}
    tiles_by_group: dict[str, list[np.ndarray]] = {name: [] for name in selected_groups}
    combined_tiles: list[np.ndarray] = []
    visual_dir = output_dir / "worst_visualizations"
    for rank, row in enumerate(selected_rows, start=1):
        key = str(row["sample_key"])
        source = dataset_rows[key]
        paths = dataset.paths_for(source)
        rgb = np.asarray(Image.open(paths["rgb"]).convert("RGB"), dtype=np.uint8)
        temperature = np.asarray(Image.open(paths["temperature"]), dtype=np.float32)
        label = np.asarray(Image.open(paths["label"]), dtype=np.uint8)
        prediction = np.asarray(Image.open(row["prediction_path"]), dtype=np.uint8)
        caption = (
            f"{key} | IoU={float(row['fire_iou']):.4f} P={float(row['fire_precision']):.4f} "
            f"R={float(row['fire_recall']):.4f} FP={int(row['false_positive'])} FN={int(row['false_negative'])}"
        )
        save_visualization(
            visual_dir / f"{rank:02d}_{safe_name(key)}.jpg",
            rgb,
            temperature,
            label,
            prediction,
            caption,
        )
        tile = contact_tile(rgb, error_overlay(rgb, label, prediction), caption)
        combined_tiles.append(tile)
        for group_name, rows in selected_groups.items():
            if any(str(candidate["sample_key"]) == key for candidate in rows):
                tiles_by_group[group_name].append(tile)
                break
    save_contact_sheet(output_dir / "worst_samples_contact_sheet.jpg", combined_tiles)
    for group_name, tiles in tiles_by_group.items():
        save_contact_sheet(output_dir / f"{group_name}_contact_sheet.jpg", tiles)

    summary = {
        "analysis_name": "flame3_fusion_validation_error_profile",
        "protocol": {
            "split": "val",
            "split_v2_expected_samples": EXPECTED_VAL_SAMPLES,
            "sample_count": len(dataset),
            "metric_scope": "temperature-derived active-Fire binary core under partial labels",
            "smoke_iou_reported": False,
            "three_class_miou_reported": False,
            "test_touched": False,
            "training_performed": False,
            "boundary_radius_pixels": args.boundary_radius,
            "component_connectivity": 8,
            "component_area_buckets": "GT Fire component area <= Q1, Q1<area<=Q3, area>Q3",
            "rgb_luma": "BT.709 luma on corrected RGB scaled to [0,1]",
            "rgb_local_contrast": f"{args.local_contrast_window}x{args.local_contrast_window} local luma standard deviation",
            "low_luma_threshold": args.low_luma_threshold,
            "low_contrast_threshold": args.low_contrast_threshold,
        },
        "model": {
            "config": str(config_path),
            "checkpoint": str(checkpoint_path),
            "checkpoint_epoch": checkpoint.get("epoch"),
            "checkpoint_best_metric": checkpoint.get("best_metric"),
            "mode": config["MODE"],
            "model": config["MODEL"],
            "input_resolution_hw": [512, 640],
        },
        "overall_active_fire": overall,
        "gt_component_area_distribution": {
            "component_count": len(all_gt_areas),
            "minimum": int(min(all_gt_areas)),
            "q1": q1,
            "median": float(np.median(all_gt_areas)),
            "q3": q3,
            "maximum": int(max(all_gt_areas)),
        },
        "gt_component_buckets": component_aggregate,
        "predicted_component_buckets_using_gt_thresholds": predicted_component_aggregate,
        "boundary_vs_interior": boundary_summary,
        "false_positive_groups": fp_groups,
        "error_feature_distributions": feature_distributions,
        "worst_sample_keys": {
            group: [str(row["sample_key"]) for row in rows] for group, rows in selected_groups.items()
        },
        "integrity": {
            "per_image_counts_match_global": True,
            "fn_boundary_partition_exact": True,
            "fp_boundary_partition_exact": True,
            "selected_keys_exist": selected_keys.issubset(dataset_rows),
            "config_sha256": sha256_file(config_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "split_csv_sha256": sha256_file(split_csv),
            "script_sha256": sha256_file(Path(__file__).resolve()),
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "amp": bool(args.amp),
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "overall_active_fire": overall,
        "gt_component_area_distribution": summary["gt_component_area_distribution"],
        "boundary_vs_interior": boundary_summary,
        "false_positive_groups": fp_groups,
        "output_dir": str(output_dir),
        "test_touched": False,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
