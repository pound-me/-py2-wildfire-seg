from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from baseline_runtime import build_model, load_config


IGNORE_ID = 255
FIRE_ID = 2
SMOKE_ID = 1
TEMPERATURE_BINS = (
    ("lt_50c", float("-inf"), 50.0),
    ("50_to_80c", 50.0, 80.0),
    ("80_to_200c", 80.0, 200.0),
    ("ge_200c", 200.0, float("inf")),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate exactly three frozen FLAME2 checkpoints on FLAME3 val."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--visualizations", type=int, default=30)
    parser.add_argument("--amp", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")


class Flame3ZeroShotDataset(Dataset):
    def __init__(self, split_csv: Path, mode: str) -> None:
        self.split_csv = split_csv.resolve()
        if self.split_csv.stem.lower() != "val":
            raise ValueError(
                f"Zero-shot preregistration permits val.csv only: {self.split_csv}"
            )
        self.mode = mode.lower()
        if self.mode not in {"rgb", "ir", "fusion"}:
            raise ValueError(f"Unsupported mode: {mode}")
        with self.split_csv.open("r", encoding="utf-8-sig", newline="") as handle:
            self.rows = list(csv.DictReader(handle))
        if len(self.rows) != 134:
            raise RuntimeError(f"Expected 134 validation samples, got {len(self.rows)}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.rows[index]
        rgb = np.asarray(
            Image.open(row["corrected_rgb_path"]).convert("RGB"), dtype=np.float32
        ) / 255.0
        ir = np.asarray(
            Image.open(row["raw_thermal_path"]).convert("L"), dtype=np.float32
        ) / 255.0
        temperature = np.asarray(
            Image.open(row["thermal_tiff_path"]), dtype=np.float32
        )
        label = np.asarray(
            Image.open(row["temperature_mask_path"]), dtype=np.uint8
        )
        if rgb.shape[:2] != ir.shape or ir.shape != temperature.shape or ir.shape != label.shape:
            raise RuntimeError(f"Shape mismatch for {row['sample_key']}")
        rgb_chw = np.transpose(rgb, (2, 0, 1))
        ir_chw = ir[None, :, :]
        if self.mode == "rgb":
            image = rgb_chw
        elif self.mode == "ir":
            image = ir_chw
        else:
            image = np.concatenate([rgb_chw, ir_chw], axis=0)
        return {
            "image": torch.from_numpy(image.copy()).float(),
            "label": torch.from_numpy(label.copy()).long(),
            "temperature": torch.from_numpy(temperature.copy()).float(),
            "sample_key": row["sample_key"],
            "sample_class": row["sample_class"],
            "sample_id": row["sample_id"],
            "corrected_rgb_path": row["corrected_rgb_path"],
            "raw_thermal_path": row["raw_thermal_path"],
        }


@dataclass
class BinaryCounts:
    true_positive: int = 0
    false_positive: int = 0
    false_negative: int = 0
    true_negative: int = 0

    def update(self, prediction: np.ndarray, label: np.ndarray, valid: np.ndarray) -> None:
        predicted_fire = prediction == FIRE_ID
        target_fire = label == FIRE_ID
        self.true_positive += int(np.sum(predicted_fire & target_fire & valid))
        self.false_positive += int(np.sum(predicted_fire & ~target_fire & valid))
        self.false_negative += int(np.sum(~predicted_fire & target_fire & valid))
        self.true_negative += int(np.sum(~predicted_fire & ~target_fire & valid))

    def metrics(self) -> dict[str, float | int]:
        tp = self.true_positive
        fp = self.false_positive
        fn = self.false_negative
        tn = self.true_negative
        return {
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
            "true_negative": tn,
            "fire_iou": tp / max(tp + fp + fn, 1),
            "fire_precision": tp / max(tp + fp, 1),
            "fire_recall": tp / max(tp + fn, 1),
            "fire_f1": (2 * tp) / max(2 * tp + fp + fn, 1),
            "binary_accuracy": (tp + tn) / max(tp + fp + fn + tn, 1),
        }


def load_manifest(path: Path) -> list[dict[str, str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    models = payload.get("models")
    if not isinstance(models, list) or len(models) != 3:
        raise ValueError("Manifest must contain exactly three models")
    normalized: list[dict[str, str]] = []
    names: set[str] = set()
    for raw in models:
        name = str(raw["name"])
        if name in names:
            raise ValueError(f"Duplicate model name: {name}")
        names.add(name)
        config = Path(str(raw["config"])).resolve()
        checkpoint = Path(str(raw["checkpoint"])).resolve()
        if not config.is_file() or not checkpoint.is_file():
            raise FileNotFoundError(f"Missing config/checkpoint for {name}")
        normalized.append(
            {"name": name, "config": str(config), "checkpoint": str(checkpoint)}
        )
    return normalized


def colorize_prediction(mask: np.ndarray) -> np.ndarray:
    output = np.zeros((*mask.shape, 3), dtype=np.uint8)
    output[mask == 0] = (0, 0, 0)
    output[mask == 1] = (160, 160, 160)
    output[mask == 2] = (255, 0, 0)
    return output


def colorize_pseudo_label(mask: np.ndarray) -> np.ndarray:
    output = np.zeros((*mask.shape, 3), dtype=np.uint8)
    output[mask == FIRE_ID] = (255, 0, 0)
    output[mask == IGNORE_ID] = (255, 0, 255)
    return output


def robust_thermal_color(temperature: np.ndarray) -> np.ndarray:
    finite = temperature[np.isfinite(temperature)]
    low, high = np.percentile(finite, [1.0, 99.0])
    if high <= low:
        high = low + 1.0
    normalized = np.clip((temperature - low) / (high - low), 0.0, 1.0)
    bgr = cv2.applyColorMap(
        (normalized * 255.0).astype(np.uint8), cv2.COLORMAP_INFERNO
    )
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def save_visualization(
    output_path: Path,
    rgb_path: str,
    temperature: np.ndarray,
    label: np.ndarray,
    prediction: np.ndarray,
    caption: str,
) -> None:
    # Pillow may expose a read-only NumPy view.  OpenCV writes the panel
    # captions in place, so materialize a writable array explicitly.
    rgb = np.array(Image.open(rgb_path).convert("RGB"), dtype=np.uint8, copy=True)
    thermal = robust_thermal_color(temperature)
    pseudo = colorize_pseudo_label(label)
    predicted = colorize_prediction(prediction)
    overlay = rgb.copy().astype(np.float32)
    predicted_fire = prediction == FIRE_ID
    predicted_smoke = prediction == SMOKE_ID
    overlay[predicted_fire] = overlay[predicted_fire] * 0.35 + np.array([255, 0, 0]) * 0.65
    overlay[predicted_smoke] = overlay[predicted_smoke] * 0.50 + np.array([180, 180, 180]) * 0.50
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    # Keep every visualization panel writable for the in-place OpenCV calls
    # below, including arrays returned by helper functions.
    panels = [np.array(panel, dtype=np.uint8, copy=True) for panel in (rgb, thermal, pseudo, predicted, overlay)]
    labels = ["RGB", "Temperature", "Pseudo 0/2/255", "Prediction 0/1/2", "Prediction overlay"]
    height, width = label.shape
    for panel, text in zip(panels, labels):
        cv2.rectangle(panel, (0, 0), (width, 29), (0, 0, 0), -1)
        cv2.putText(panel, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    canvas = np.concatenate(panels, axis=1)
    cv2.rectangle(canvas, (0, height - 27), (canvas.shape[1], height), (0, 0, 0), -1)
    cv2.putText(canvas, caption, (8, height - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1, cv2.LINE_AA)
    Image.fromarray(canvas).save(output_path, quality=92)


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_model(
    entry: dict[str, str],
    split_csv: Path,
    output_root: Path,
    args: argparse.Namespace,
) -> dict[str, object]:
    name = entry["name"]
    model_dir = output_root / safe_name(name)
    prediction_dir = model_dir / "predictions_raw"
    visual_dir = model_dir / "visuals"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    visual_dir.mkdir(parents=True, exist_ok=True)

    config_path = Path(entry["config"])
    checkpoint_path = Path(entry["checkpoint"])
    config = load_config(config_path)
    dataset = Flame3ZeroShotDataset(split_csv, str(config["MODE"]))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    device = torch.device(args.device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "model_state_dict" not in checkpoint:
        raise KeyError(f"model_state_dict missing: {checkpoint_path}")
    model = build_model(config)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model = model.to(device).eval()

    total_counts = BinaryCounts()
    per_image: list[dict[str, object]] = []
    temperature_counts = {
        name: {"pixels": 0, "pred_fire": 0, "pred_smoke": 0}
        for name, _, _ in TEMPERATURE_BINS
    }
    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].numpy()
            temperatures = batch["temperature"].numpy()
            with torch.autocast(
                device_type="cuda", dtype=torch.float16, enabled=args.amp
            ):
                outputs = model(images)
                logits = outputs[1] if isinstance(outputs, (list, tuple)) else outputs
                if logits.shape[-2:] != labels.shape[-2:]:
                    logits = F.interpolate(
                        logits,
                        size=labels.shape[-2:],
                        mode="bilinear",
                        align_corners=bool(config["ALIGN_CORNERS"]),
                    )
            predictions = logits.argmax(dim=1).cpu().numpy().astype(np.uint8)
            for index in range(predictions.shape[0]):
                prediction = predictions[index]
                label = labels[index]
                temperature = temperatures[index]
                valid = label != IGNORE_ID
                sample_counts = BinaryCounts()
                sample_counts.update(prediction, label, valid)
                total_counts.update(prediction, label, valid)
                metrics = sample_counts.metrics()
                target_fire_pixels = int(np.sum((label == FIRE_ID) & valid))
                valid_pixels = int(valid.sum())
                predicted_fire_pixels = int(np.sum((prediction == FIRE_ID) & valid))
                predicted_smoke_pixels = int(np.sum((prediction == SMOKE_ID) & valid))
                sample_class = str(batch["sample_class"][index])
                sample_id = str(batch["sample_id"][index])
                sample_key = str(batch["sample_key"][index])
                row = {
                    "sample_key": sample_key,
                    "sample_class": sample_class,
                    "sample_id": sample_id,
                    "has_pseudo_fire": target_fire_pixels > 0,
                    "valid_pixels": valid_pixels,
                    "pseudo_fire_pixels": target_fire_pixels,
                    "predicted_fire_pixels": predicted_fire_pixels,
                    "predicted_smoke_pixels": predicted_smoke_pixels,
                    "predicted_fire_ratio": predicted_fire_pixels / max(valid_pixels, 1),
                    "predicted_smoke_ratio": predicted_smoke_pixels / max(valid_pixels, 1),
                    **metrics,
                    "corrected_rgb_path": str(batch["corrected_rgb_path"][index]),
                    "prediction_path": str(
                        prediction_dir / f"{safe_name(sample_key)}.png"
                    ),
                }
                per_image.append(row)
                Image.fromarray(prediction).save(row["prediction_path"])
                for bin_name, lower, upper in TEMPERATURE_BINS:
                    bin_mask = (temperature >= lower) & (temperature < upper) & valid
                    count = int(bin_mask.sum())
                    temperature_counts[bin_name]["pixels"] += count
                    temperature_counts[bin_name]["pred_fire"] += int(
                        np.sum((prediction == FIRE_ID) & bin_mask)
                    )
                    temperature_counts[bin_name]["pred_smoke"] += int(
                        np.sum((prediction == SMOKE_ID) & bin_mask)
                    )

    write_csv(per_image, model_dir / "per_image_metrics.csv")
    temperature_rows: list[dict[str, object]] = []
    for bin_name, counts in temperature_counts.items():
        pixels = counts["pixels"]
        temperature_rows.append(
            {
                "temperature_bin": bin_name,
                **counts,
                "pred_fire_ratio": counts["pred_fire"] / max(pixels, 1),
                "pred_smoke_ratio": counts["pred_smoke"] / max(pixels, 1),
            }
        )
    write_csv(temperature_rows, model_dir / "temperature_bins.csv")

    positives = [row for row in per_image if bool(row["has_pseudo_fire"])]
    worst_positive = sorted(positives, key=lambda row: float(row["fire_iou"]))
    no_fire = [row for row in per_image if row["sample_class"] == "No Fire"]
    worst_no_fire = sorted(
        no_fire, key=lambda row: float(row["predicted_fire_ratio"]), reverse=True
    )
    empty_fire = [
        row
        for row in per_image
        if row["sample_class"] == "Fire" and not bool(row["has_pseudo_fire"])
    ]
    selected: list[dict[str, object]] = []
    selected.extend(worst_positive[: min(20, len(worst_positive))])
    selected.extend(worst_no_fire[: min(5, len(worst_no_fire))])
    selected.extend(empty_fire[: min(5, len(empty_fire))])
    seen: set[str] = set()
    selected = [
        row
        for row in selected
        if not (row["sample_key"] in seen or seen.add(str(row["sample_key"])))
    ][: args.visualizations]
    row_by_key = {row["sample_key"]: row for row in per_image}
    dataset_row_by_key = {row["sample_key"]: row for row in dataset.rows}
    for rank, row in enumerate(selected):
        source = dataset_row_by_key[row["sample_key"]]
        label = np.asarray(Image.open(source["temperature_mask_path"]), dtype=np.uint8)
        temperature = np.asarray(Image.open(source["thermal_tiff_path"]), dtype=np.float32)
        prediction = np.asarray(Image.open(row["prediction_path"]), dtype=np.uint8)
        caption = (
            f"{row['sample_key']} | IoU={float(row['fire_iou']):.4f} | "
            f"pred_fire={float(row['predicted_fire_ratio']) * 100:.3f}% | "
            f"pred_smoke={float(row['predicted_smoke_ratio']) * 100:.3f}%"
        )
        save_visualization(
            visual_dir / f"{rank:02d}_{safe_name(str(row['sample_key']))}.jpg",
            source["corrected_rgb_path"],
            temperature,
            label,
            prediction,
            caption,
        )

    total_metrics = total_counts.metrics()
    no_fire_valid = sum(int(row["valid_pixels"]) for row in no_fire)
    no_fire_predicted_fire = sum(int(row["predicted_fire_pixels"]) for row in no_fire)
    summary = {
        "model_name": name,
        "config": str(config_path),
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "input_mode": config["MODE"],
        "input_resolution": [640, 512],
        "ir_zero_shot_source": "Thermal/Raw JPG converted to L and divided by 255",
        "split": "val",
        "samples": len(dataset),
        **total_metrics,
        "images_with_nonempty_pseudo_fire": len(positives),
        "fire_folder_empty_pseudo_images": len(empty_fire),
        "no_fire_predicted_fire_ratio": no_fire_predicted_fire / max(no_fire_valid, 1),
        "mean_smoke_prediction_ratio_fire_folder": float(
            np.mean(
                [row["predicted_smoke_ratio"] for row in per_image if row["sample_class"] == "Fire"]
            )
        ),
        "mean_smoke_prediction_ratio_no_fire_folder": float(
            np.mean([row["predicted_smoke_ratio"] for row in no_fire])
        ),
        "temperature_bins": temperature_rows,
        "config_sha256": sha256_file(config_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "training_performed": False,
        "test_touched": False,
    }
    (model_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    del model
    torch.cuda.empty_cache()
    return summary


def main() -> None:
    args = parse_args()
    if args.batch_size != 1:
        raise ValueError("Zero-shot preregistration freezes physical batch size at 1")
    manifest_path = args.manifest.resolve()
    split_csv = args.split_csv.resolve()
    output_root = args.output.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    entries = load_manifest(manifest_path)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the preregistered zero-shot run")

    summaries = [
        run_model(entry, split_csv, output_root, args) for entry in entries
    ]
    comparison_rows = [
        {
            "model_name": item["model_name"],
            "fire_iou": item["fire_iou"],
            "fire_precision": item["fire_precision"],
            "fire_recall": item["fire_recall"],
            "fire_f1": item["fire_f1"],
            "no_fire_predicted_fire_ratio": item["no_fire_predicted_fire_ratio"],
            "mean_smoke_prediction_ratio_fire_folder": item[
                "mean_smoke_prediction_ratio_fire_folder"
            ],
            "mean_smoke_prediction_ratio_no_fire_folder": item[
                "mean_smoke_prediction_ratio_no_fire_folder"
            ],
        }
        for item in summaries
    ]
    write_csv(comparison_rows, output_root / "model_comparison.csv")
    environment = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(torch.device(args.device)),
        "amp": args.amp,
        "batch_size": args.batch_size,
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "split_csv": str(split_csv),
        "split_csv_sha256": sha256_file(split_csv),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "test_touched": False,
    }
    (output_root / "environment.json").write_text(
        json.dumps(environment, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_root / "comparison.json").write_text(
        json.dumps(summaries, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(comparison_rows, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
