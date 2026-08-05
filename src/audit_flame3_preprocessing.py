from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit frozen FLAME3 RGB/thermal preprocessing without training."
    )
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--split-csv", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--correlation-samples", type=int, default=100)
    parser.add_argument("--augmentation-trials", type=int, default=32)
    return parser.parse_args()


def resolve_path(root: Path, raw: str) -> Path:
    path = Path(raw)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(order.size, dtype=np.float64)
    return ranks


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    if left.size != right.size or left.size < 2:
        return 0.0
    left = left - left.mean()
    right = right - right.mean()
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator > 1e-12 else 0.0


def balanced_sample(rows: list[dict[str, str]], total: int, seed: int) -> list[dict[str, str]]:
    groups = {
        name: [row for row in rows if row["sample_class"] == name]
        for name in ("Fire", "No Fire")
    }
    generator = np.random.default_rng(seed)
    selected: list[dict[str, str]] = []
    target = total // 2
    for name in ("Fire", "No Fire"):
        count = min(target, len(groups[name]))
        indices = generator.choice(len(groups[name]), size=count, replace=False)
        selected.extend(groups[name][int(index)] for index in np.sort(indices))
    remaining = total - len(selected)
    if remaining > 0:
        unused = [row for row in rows if row not in selected]
        indices = generator.choice(len(unused), size=min(remaining, len(unused)), replace=False)
        selected.extend(unused[int(index)] for index in np.sort(indices))
    return selected


def load_config(path: Path) -> dict[str, object]:
    try:
        import yaml
    except ImportError as error:
        raise RuntimeError("PyYAML is required for the preprocessing audit") from error
    result = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise RuntimeError(f"Invalid config: {path}")
    return result


def dataset_statistics(
    rows: list[dict[str, str]], data_root: Path, correlation_rows: list[dict[str, str]]
) -> dict[str, object]:
    rgb_sum = np.zeros(3, dtype=np.float64)
    rgb_square = np.zeros(3, dtype=np.float64)
    rgb_count = 0
    rgb_min = np.full(3, np.inf, dtype=np.float64)
    rgb_max = np.full(3, -np.inf, dtype=np.float64)
    ir_sum = 0.0
    ir_square = 0.0
    ir_count = 0
    ir_min = np.inf
    ir_max = -np.inf
    zero_pixels = 0
    saturated_pixels = 0
    per_image: list[dict[str, float | str]] = []
    correlation_keys = {row["sample_key"] for row in correlation_rows}
    correlations: list[dict[str, float | str]] = []
    global_gray: list[np.ndarray] = []
    global_temperature: list[np.ndarray] = []

    for row in rows:
        rgb_path = resolve_path(data_root, row["corrected_rgb_path"])
        ir_path = resolve_path(data_root, row["raw_thermal_path"])
        tiff_path = resolve_path(data_root, row["thermal_tiff_path"])
        rgb = np.asarray(Image.open(rgb_path).convert("RGB"), dtype=np.float32) / 255.0
        ir = np.asarray(Image.open(ir_path).convert("L"), dtype=np.float32) / 255.0
        flat_rgb = rgb.reshape(-1, 3)
        rgb_sum += flat_rgb.sum(axis=0, dtype=np.float64)
        rgb_square += np.square(flat_rgb, dtype=np.float64).sum(axis=0)
        rgb_count += int(flat_rgb.shape[0])
        rgb_min = np.minimum(rgb_min, flat_rgb.min(axis=0))
        rgb_max = np.maximum(rgb_max, flat_rgb.max(axis=0))
        ir_sum += float(ir.sum(dtype=np.float64))
        ir_square += float(np.square(ir, dtype=np.float64).sum(dtype=np.float64))
        ir_count += int(ir.size)
        ir_min = min(ir_min, float(ir.min()))
        ir_max = max(ir_max, float(ir.max()))
        zero_ratio = float(np.mean(ir == 0.0))
        saturated_ratio = float(np.mean(ir == 1.0))
        zero_pixels += int(np.count_nonzero(ir == 0.0))
        saturated_pixels += int(np.count_nonzero(ir == 1.0))
        per_image.append(
            {
                "sample_key": row["sample_key"],
                "ir_mean": float(ir.mean()),
                "ir_std": float(ir.std()),
                "ir_zero_ratio": zero_ratio,
                "ir_255_ratio": saturated_ratio,
            }
        )
        if row["sample_key"] in correlation_keys:
            temperature = np.asarray(Image.open(tiff_path), dtype=np.float32)
            gray_values = ir[::8, ::8].reshape(-1)
            temperature_values = temperature[::8, ::8].reshape(-1)
            pearson = correlation(gray_values, temperature_values)
            spearman = correlation(rankdata(gray_values), rankdata(temperature_values))
            correlations.append(
                {
                    "sample_key": row["sample_key"],
                    "pearson": pearson,
                    "spearman": spearman,
                }
            )
            global_gray.append(gray_values)
            global_temperature.append(temperature_values)

    rgb_mean = rgb_sum / rgb_count
    rgb_std = np.sqrt(np.maximum(rgb_square / rgb_count - np.square(rgb_mean), 0.0))
    ir_mean = ir_sum / ir_count
    ir_std = float(np.sqrt(max(ir_square / ir_count - ir_mean * ir_mean, 0.0)))
    zero_ratios = np.asarray([float(item["ir_zero_ratio"]) for item in per_image])
    image_means = np.asarray([float(item["ir_mean"]) for item in per_image])
    image_stds = np.asarray([float(item["ir_std"]) for item in per_image])
    pearsons = np.asarray([float(item["pearson"]) for item in correlations])
    spearmans = np.asarray([float(item["spearman"]) for item in correlations])
    joined_gray = np.concatenate(global_gray)
    joined_temperature = np.concatenate(global_temperature)
    return {
        "sample_count": len(rows),
        "rgb_0_to_1": {
            "mean": rgb_mean.tolist(),
            "std": rgb_std.tolist(),
            "min": rgb_min.tolist(),
            "max": rgb_max.tolist(),
        },
        "thermal_jpg_gray_0_to_1": {
            "mean": ir_mean,
            "std": ir_std,
            "min": ir_min,
            "max": ir_max,
            "zero_ratio": zero_pixels / ir_count,
            "value_255_ratio": saturated_pixels / ir_count,
            "images_zero_ratio_gt_0_5": int(np.count_nonzero(zero_ratios > 0.5)),
            "images_zero_ratio_gt_0_5_ratio": float(np.mean(zero_ratios > 0.5)),
            "per_image_mean_p05_p50_p95": np.quantile(image_means, [0.05, 0.5, 0.95]).tolist(),
            "per_image_std_p05_p50_p95": np.quantile(image_stds, [0.05, 0.5, 0.95]).tolist(),
        },
        "thermal_jpg_vs_celsius_tiff": {
            "sample_count": len(correlations),
            "per_image_pearson_p05_p50_p95": np.quantile(pearsons, [0.05, 0.5, 0.95]).tolist(),
            "per_image_spearman_p05_p50_p95": np.quantile(spearmans, [0.05, 0.5, 0.95]).tolist(),
            "cross_image_global_pearson": correlation(joined_gray, joined_temperature),
            "cross_image_global_spearman": correlation(
                rankdata(joined_gray), rankdata(joined_temperature)
            ),
            "interpretation": (
                "Raw Thermal JPG preserves within-frame relative heat ordering but "
                "does not provide a shared physical temperature scale across frames."
            ),
        },
        "per_image": per_image,
        "correlations": correlations,
    }


def center_of_mass(mask: np.ndarray) -> tuple[float, float] | None:
    points = np.argwhere(mask)
    if points.size == 0:
        return None
    center = points.mean(axis=0)
    return float(center[0]), float(center[1])


def augmentation_alignment(
    project_root: Path, config: dict[str, object], trials: int, seed: int
) -> dict[str, object]:
    third_party = project_root / "third_party" / "RoboFireFuseNet"
    sys.path.insert(0, str(third_party))
    from datasets.base_dataset import BaseDataset

    crop_size = tuple(int(value) for value in config["CROP_SIZE"])
    dataset = BaseDataset(
        ignore_label=int(config["IGNORE_LABEL"]),
        base_size=int(config["BASE_SIZE"]),
        crop_size=crop_size,
        scale_factor=int(config["SCALE_FACTOR"]),
        mean=[0, 0, 0, 0],
        std=[1, 1, 1, 1],
        scale_min=float(config["SCALE_MIN"]),
        scale_max=float(config["SCALE_MAX"]),
    )
    height, width = crop_size
    pattern = np.zeros((height, width), dtype=np.uint8)
    pattern[height // 3 : height // 3 + 80, width // 3 : width // 3 + 96] = 255
    label = np.zeros((height, width), dtype=np.uint8)
    label[pattern > 0] = 2
    modality_errors: list[float] = []
    label_offsets: list[float] = []
    label_ious: list[float] = []
    for index in range(trials):
        trial_seed = int(seed) + index
        random.seed(trial_seed)
        np.random.seed(trial_seed)
        rgb = np.repeat(pattern[..., None], 3, axis=2)
        infrared = pattern[..., None]
        images, transformed_label, _ = dataset.gen_sample(
            [rgb.copy(), infrared.copy()],
            label.copy(),
            multi_scale=bool(config["MULTISCALE"]),
            is_flip=bool(config["FLIP"]),
            edge_pad=False,
            brightness=bool(config["BRIGHTNESS"]),
        )
        rgb_marker = images[0][0]
        ir_marker = images[1][0]
        modality_errors.append(float(np.max(np.abs(rgb_marker - ir_marker))))
        image_mask = ir_marker >= 0.5
        label_mask = transformed_label == 2
        intersection = int(np.count_nonzero(image_mask & label_mask))
        union = int(np.count_nonzero(image_mask | label_mask))
        label_ious.append(intersection / max(union, 1))
        image_center = center_of_mass(image_mask)
        label_center = center_of_mass(label_mask)
        if image_center is None or label_center is None:
            raise RuntimeError("Synthetic augmentation marker disappeared")
        label_offsets.append(
            float(np.hypot(image_center[0] - label_center[0], image_center[1] - label_center[1]))
        )
    return {
        "trials": trials,
        "brightness_enabled": bool(config["BRIGHTNESS"]),
        "rgb_ir_max_abs_error": max(modality_errors),
        "label_center_offset_max_px": max(label_offsets),
        "label_marker_iou_min": min(label_ious),
    }


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    split_csv = args.split_csv.resolve()
    data_root = args.data_root.resolve()
    config_path = args.config.resolve()
    output = args.output.resolve()
    with split_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"Empty split: {split_csv}")
    config = load_config(config_path)
    correlation_rows = balanced_sample(rows, args.correlation_samples, args.seed)
    statistics = dataset_statistics(rows, data_root, correlation_rows)
    alignment = augmentation_alignment(
        project_root, config, args.augmentation_trials, args.seed
    )
    thermal = statistics["thermal_jpg_gray_0_to_1"]
    correlation_summary = statistics["thermal_jpg_vs_celsius_tiff"]
    passes = bool(
        not alignment["brightness_enabled"]
        and alignment["rgb_ir_max_abs_error"] <= 1e-6
        and alignment["label_center_offset_max_px"] <= 1.5
        and alignment["label_marker_iou_min"] >= 0.94
        and thermal["zero_ratio"] <= 0.10
        and thermal["images_zero_ratio_gt_0_5_ratio"] <= 0.05
        and correlation_summary["per_image_pearson_p05_p50_p95"][1] >= 0.90
    )
    result = {
        "audit": "flame3_preprocessing_and_augmentation",
        "split_csv": str(split_csv),
        "data_root": str(data_root),
        "config": str(config_path),
        "seed": int(args.seed),
        "statistics": statistics,
        "augmentation_alignment": alignment,
        "formal_input_policy": (
            "Keep Raw Thermal JPG grayscale divided by 255. Celsius TIFF remains "
            "diagnostic-only because it generated the temperature pseudo-label."
        ),
        "passes_frozen_cmrc_precondition": passes,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "preprocessing_audit.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    markdown = [
        "# FLAME3预处理与增强对齐核验",
        "",
        f"- 训练样本：{statistics['sample_count']}。",
        f"- RGB均值：{statistics['rgb_0_to_1']['mean']}。",
        f"- IR均值/标准差：{thermal['mean']:.6f}/{thermal['std']:.6f}。",
        f"- IR零值比例：{thermal['zero_ratio']:.4%}。",
        f"- 单帧灰度—温度Pearson中位数：{correlation_summary['per_image_pearson_p05_p50_p95'][1]:.6f}。",
        f"- 跨帧合并Pearson：{correlation_summary['cross_image_global_pearson']:.6f}。",
        f"- RGB/IR增强后最大差值：{alignment['rgb_ir_max_abs_error']:.8f}。",
        f"- 图像/标签中心最大偏移：{alignment['label_center_offset_max_px']:.4f}px。",
        f"- CMRC预处理准入：`{passes}`。",
        "",
        "正式输入继续使用Raw Thermal JPG；Celsius TIFF只用于诊断。",
    ]
    (output / "FLAME3_PREPROCESSING_AUDIT.md").write_text(
        "\n".join(markdown) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
