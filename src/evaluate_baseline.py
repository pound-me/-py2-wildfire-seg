from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

from baseline_runtime import PROJECT_ROOT, build_dataset, build_model, load_config
from custom_models.pidnet_deconv import (
    count_deconv_modules,
    reparameterize_deconv_model,
)
from train_baseline import (
    metrics_from_confusion,
    seed_everything,
    update_confusion_matrix,
)


OFFICIAL_PALETTE = np.asarray(
    [
        [0, 0, 0],
        [125, 125, 125],
        [255, 255, 255],
    ],
    dtype=np.uint8,
)
OVERLAY_PALETTE = np.asarray(
    [
        [0, 0, 0],
        [255, 190, 0],
        [255, 35, 35],
    ],
    dtype=np.uint8,
)


def extract_batch_names(raw_names: object, batch_size: int) -> list[str]:
    if isinstance(raw_names, (list, tuple)) and len(raw_names) == 1:
        nested = raw_names[0]
        if isinstance(nested, (list, tuple)):
            return [str(name) for name in nested]
    if isinstance(raw_names, (list, tuple)):
        names = [str(name) for name in raw_names]
        if len(names) == batch_size:
            return names
    return [f"sample_{index:04d}.png" for index in range(batch_size)]


def colorize_mask(mask: np.ndarray, palette: np.ndarray) -> np.ndarray:
    color = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for class_index in range(len(palette)):
        color[mask == class_index] = palette[class_index]
    return color


def overlay_mask(image: np.ndarray, mask: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    result = image.astype(np.float32).copy()
    foreground = (mask > 0) & (mask < len(OVERLAY_PALETTE))
    colors = colorize_mask(mask, OVERLAY_PALETTE).astype(np.float32)
    result[foreground] = (
        (1.0 - alpha) * result[foreground] + alpha * colors[foreground]
    )
    return np.clip(result, 0, 255).astype(np.uint8)


def image_from_tensor(image: torch.Tensor) -> np.ndarray:
    array = image.detach().cpu().float().permute(1, 2, 0).numpy()
    if array.shape[2] == 1:
        array = np.repeat(array, 3, axis=2)
    elif array.shape[2] >= 3:
        # Fusion inputs are RGB+IR; use the RGB channels for overlays.
        array = array[:, :, :3]
    else:
        raise ValueError(f"Unsupported visualization channel count: {array.shape[2]}")
    return np.clip(array * 255.0, 0, 255).astype(np.uint8)


def extract_binary_boundary(mask: torch.Tensor) -> torch.Tensor:
    values = mask.unsqueeze(1).to(dtype=torch.float32)
    kernel = torch.ones((1, 1, 3, 3), device=mask.device)
    eroded = F.conv2d(values, kernel, padding=1) >= 9.0
    return mask & ~eroded[:, 0]


def update_boundary_statistics(
    statistics: dict[str, dict[str, int]],
    predictions: torch.Tensor,
    labels: torch.Tensor,
    class_names: list[str],
    ignore_label: int,
    tolerance: int,
) -> None:
    valid = labels != ignore_label
    invalid_band = F.max_pool2d(
        (~valid).unsqueeze(1).to(dtype=torch.float32),
        kernel_size=3,
        stride=1,
        padding=1,
    )[:, 0] > 0
    dilation_size = 2 * tolerance + 1

    for class_index, class_name in enumerate(class_names):
        if class_index == 0:
            continue
        predicted_boundary = extract_binary_boundary(
            (predictions == class_index) & valid
        ) & ~invalid_band
        target_boundary = extract_binary_boundary(
            (labels == class_index) & valid
        ) & ~invalid_band
        dilated_prediction = F.max_pool2d(
            predicted_boundary.unsqueeze(1).to(dtype=torch.float32),
            kernel_size=dilation_size,
            stride=1,
            padding=tolerance,
        )[:, 0] > 0
        dilated_target = F.max_pool2d(
            target_boundary.unsqueeze(1).to(dtype=torch.float32),
            kernel_size=dilation_size,
            stride=1,
            padding=tolerance,
        )[:, 0] > 0

        class_statistics = statistics[class_name]
        class_statistics["matched_prediction"] += int(
            (predicted_boundary & dilated_target).sum()
        )
        class_statistics["prediction"] += int(predicted_boundary.sum())
        class_statistics["matched_target"] += int(
            (target_boundary & dilated_prediction).sum()
        )
        class_statistics["target"] += int(target_boundary.sum())


def finalize_boundary_statistics(
    statistics: dict[str, dict[str, int]],
    tolerance: int,
) -> dict:
    result = {"tolerance_pixels": tolerance}
    f1_values = []
    for class_name, counts in statistics.items():
        precision = (
            counts["matched_prediction"] / counts["prediction"]
            if counts["prediction"] > 0
            else 0.0
        )
        recall = (
            counts["matched_target"] / counts["target"]
            if counts["target"] > 0
            else 0.0
        )
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall > 0
            else 0.0
        )
        result[class_name] = {
            "boundary_precision": precision,
            "boundary_recall": recall,
            "boundary_f1": f1,
            **counts,
        }
        f1_values.append(f1)
    result["mean_boundary_f1"] = (
        float(np.mean(f1_values)) if f1_values else 0.0
    )
    return result


def save_comparison(
    path: Path,
    image: np.ndarray,
    label: np.ndarray,
    prediction: np.ndarray,
) -> None:
    panels = [
        image,
        overlay_mask(image, label),
        overlay_mask(image, prediction),
    ]
    titles = ["Input view", "Ground truth", "Prediction"]
    height, width = image.shape[:2]
    title_height = 24
    canvas = Image.new("RGB", (width * 3, height + title_height), color=(20, 20, 20))
    draw = ImageDraw.Draw(canvas)
    for index, (panel, title) in enumerate(zip(panels, titles)):
        x = index * width
        canvas.paste(Image.fromarray(panel), (x, title_height))
        draw.text((x + 6, 5), title, fill=(255, 255, 255))
    canvas.save(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the frozen PIDNet-S RGB baseline."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "pidnet_s_rgb_baseline.yaml",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=(
            PROJECT_ROOT
            / "experiments"
            / "pidnet_s_rgb_baseline"
            / "baseline_100e"
            / "best.pth"
        ),
    )
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument(
        "--confirm-frozen-test",
        action="store_true",
        help=(
            "Required with --split test. Confirms architecture, protocol, "
            "hyperparameters, and checkpoint-selection rules are frozen."
        ),
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--visualizations", type=int, default=20)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--warmup-iterations", type=int, default=30)
    parser.add_argument("--speed-iterations", type=int, default=100)
    parser.add_argument("--speed-repeats", type=int, default=5)
    parser.add_argument("--boundary-tolerance", type=int, default=3)
    args = parser.parse_args()

    if args.split == "test" and not args.confirm_frozen_test:
        raise RuntimeError(
            "Test evaluation is locked. Freeze the architecture, protocol, "
            "hyperparameters, and checkpoint-selection rules, then rerun "
            "with --confirm-frozen-test."
        )

    config = load_config(args.config.resolve())
    checkpoint_path = args.checkpoint.resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable.")

    seed_everything(config["SEED"])
    device = torch.device(config["DEVICE"])
    dataset = build_dataset(config, args.split)
    loader = DataLoader(
        dataset,
        batch_size=config["BATCHSIZE"],
        shuffle=False,
        num_workers=config["NUM_WORKERS"],
        pin_memory=True,
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    if "model_state_dict" not in checkpoint:
        raise KeyError("The checkpoint does not contain model_state_dict.")
    model = build_model(config)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    training_parameter_count = sum(
        parameter.numel() for parameter in model.parameters()
    )
    training_deconv_modules = count_deconv_modules(model)
    if training_deconv_modules:
        model = reparameterize_deconv_model(model, inplace=True)
    model = model.to(device)
    model.eval()

    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else checkpoint_path.parent / f"{args.split}_best"
    )
    raw_prediction_dir = output_dir / "predictions_raw"
    color_prediction_dir = output_dir / "predictions_color"
    comparison_dir = output_dir / "comparisons"
    for directory in (
        output_dir,
        raw_prediction_dir,
        color_prediction_dir,
        comparison_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    visual_count = min(max(args.visualizations, 0), len(dataset))
    visual_indices = set(
        np.linspace(0, len(dataset) - 1, visual_count, dtype=int).tolist()
        if visual_count
        else []
    )
    confusion = torch.zeros(
        config["NUM_CLASSES"], config["NUM_CLASSES"], dtype=torch.int64
    )
    boundary_statistics = {
        class_name: {
            "matched_prediction": 0,
            "prediction": 0,
            "matched_target": 0,
            "target": 0,
        }
        for class_index, class_name in enumerate(config["CLS_NAMES"])
        if class_index > 0
    }
    per_image_rows: list[dict] = []
    benchmark_image: torch.Tensor | None = None
    global_index = 0

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"Split/samples: {args.split}/{len(dataset)}")
    print(f"Output directory: {output_dir}")

    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            images, labels = batch[0], batch[1]
            names = extract_batch_names(batch[3], images.shape[0])
            images = images.to(
                device=device,
                dtype=torch.float,
                non_blocking=True,
            )
            labels = labels.to(
                device=device,
                dtype=torch.long,
                non_blocking=True,
            )
            if benchmark_image is None:
                benchmark_image = images[:1].clone()

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=args.amp,
            ):
                outputs = model(images)
                logits = outputs[1] if isinstance(outputs, (list, tuple)) else outputs
                if logits.shape[-2:] != labels.shape[-2:]:
                    logits = F.interpolate(
                        logits,
                        size=labels.shape[-2:],
                        mode="bilinear",
                        align_corners=config["ALIGN_CORNERS"],
                    )

            update_confusion_matrix(
                confusion,
                logits,
                labels,
                config["NUM_CLASSES"],
                config["IGNORE_LABEL"],
            )
            predictions = logits.argmax(dim=1)
            update_boundary_statistics(
                boundary_statistics,
                predictions,
                labels,
                config["CLS_NAMES"],
                config["IGNORE_LABEL"],
                max(args.boundary_tolerance, 0),
            )

            for local_index in range(images.shape[0]):
                sample_confusion = torch.zeros_like(confusion)
                update_confusion_matrix(
                    sample_confusion,
                    logits[local_index : local_index + 1],
                    labels[local_index : local_index + 1],
                    config["NUM_CLASSES"],
                    config["IGNORE_LABEL"],
                )
                sample_metrics = metrics_from_confusion(
                    sample_confusion,
                    config["CLS_NAMES"],
                )
                name = names[local_index]
                safe_stem = Path(name).stem.replace("XXX", "rgb")
                prediction = predictions[local_index].detach().cpu().numpy().astype(
                    np.uint8
                )
                label = labels[local_index].detach().cpu().numpy().astype(np.uint8)

                Image.fromarray(prediction).save(
                    raw_prediction_dir / f"{safe_stem}_pred.png"
                )
                Image.fromarray(colorize_mask(prediction, OFFICIAL_PALETTE)).save(
                    color_prediction_dir / f"{safe_stem}_pred_color.png"
                )
                if global_index in visual_indices:
                    save_comparison(
                        comparison_dir
                        / f"{global_index:04d}_{safe_stem}_comparison.png",
                        image_from_tensor(images[local_index]),
                        label,
                        prediction,
                    )

                per_image_rows.append(
                    {
                        "index": global_index,
                        "name": name,
                        **sample_metrics,
                    }
                )
                global_index += 1

            if (batch_index + 1) % 10 == 0 or batch_index + 1 == len(loader):
                print(f"Evaluated batch {batch_index + 1}/{len(loader)}")

    metrics = metrics_from_confusion(confusion, config["CLS_NAMES"])
    boundary_metrics = finalize_boundary_statistics(
        boundary_statistics,
        max(args.boundary_tolerance, 0),
    )
    if benchmark_image is None:
        raise RuntimeError("The selected dataset is empty.")
    inference_model = build_model(config, augment=False)
    incompatible = inference_model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=False,
    )
    if incompatible.missing_keys:
        raise RuntimeError(
            "Missing inference-model weights: "
            + ", ".join(incompatible.missing_keys)
        )
    inference_deconv_modules = count_deconv_modules(inference_model)
    if inference_deconv_modules:
        inference_model = reparameterize_deconv_model(
            inference_model,
            inplace=True,
        )
    inference_model = inference_model.to(device)
    inference_model.eval()
    inference_parameter_count = sum(
        parameter.numel() for parameter in inference_model.parameters()
    )
    del model
    torch.cuda.empty_cache()

    repeat_count = max(args.speed_repeats, 1)
    latency_trials_ms: list[float] = []
    fps_trials: list[float] = []
    with torch.inference_mode():
        for _ in range(max(args.warmup_iterations, 0)):
            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=args.amp,
            ):
                inference_model(benchmark_image)
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        for _ in range(repeat_count):
            start = time.perf_counter()
            for _ in range(max(args.speed_iterations, 1)):
                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.float16,
                    enabled=args.amp,
                ):
                    inference_model(benchmark_image)
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - start
            trial_latency_ms = elapsed * 1000.0 / max(args.speed_iterations, 1)
            latency_trials_ms.append(trial_latency_ms)
            fps_trials.append(1000.0 / trial_latency_ms)
        peak_memory_mb = torch.cuda.max_memory_allocated(device) / 1024**2

    iterations = max(args.speed_iterations, 1)
    latency_ms = float(np.median(latency_trials_ms))
    fps = float(np.median(fps_trials))
    checkpoint_config = dict(checkpoint.get("config", config))
    method_config = dict(checkpoint_config)
    method_config.pop("SEED", None)
    method_config_sha256 = hashlib.sha256(
        json.dumps(
            method_config,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    results = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "seed": int(
            checkpoint.get("config", {}).get("SEED", config["SEED"])
        ),
        "config_sha256": hashlib.sha256(
            args.config.resolve().read_bytes()
        ).hexdigest(),
        "method_config_excluding_seed_sha256": method_config_sha256,
        "split": args.split,
        "sample_count": len(dataset),
        "metrics": metrics,
        "boundary_metrics": boundary_metrics,
        "confusion_matrix": confusion.tolist(),
        "model": {
            "name": config["MODEL"],
            "training_parameters_with_auxiliary_heads": training_parameter_count,
            "training_parameters_millions": training_parameter_count / 1_000_000,
            "inference_parameters_main_head": inference_parameter_count,
            "training_deconv_modules": training_deconv_modules,
            "inference_deconv_modules_before_reparameterization": (
                inference_deconv_modules
            ),
            "deployment_deconv_modules": count_deconv_modules(inference_model),
            "inference_parameters_millions": inference_parameter_count / 1_000_000,
        },
        "speed": {
            "device": torch.cuda.get_device_name(device),
            "input_shape": list(benchmark_image.shape),
            "batch_size": 1,
            "amp": args.amp,
            "warmup_iterations": max(args.warmup_iterations, 0),
            "timed_iterations": iterations,
            "repeats": repeat_count,
            "latency_ms": latency_ms,
            "latency_ms_mean": float(np.mean(latency_trials_ms)),
            "latency_ms_std": float(np.std(latency_trials_ms)),
            "latency_ms_trials": latency_trials_ms,
            "fps": fps,
            "fps_mean": float(np.mean(fps_trials)),
            "fps_std": float(np.std(fps_trials)),
            "fps_trials": fps_trials,
            "peak_allocated_memory_mb": peak_memory_mb,
            "scope": "model forward, main segmentation output only",
        },
    }

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as stream:
        json.dump(results, stream, ensure_ascii=False, indent=2)

    metric_fieldnames = [
        key for key in per_image_rows[0] if key not in {"index", "name"}
    ]
    fieldnames = ["index", "name", *metric_fieldnames]
    with (output_dir / "per_image_metrics.csv").open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_image_rows)

    print(f"{args.split.capitalize()} evaluation completed")
    print(
        f"mIoU={metrics['miou']:.4f}, "
        f"smoke IoU={metrics['iou_smoke']:.4f}, "
        f"fire IoU={metrics['iou_fire']:.4f}"
    )
    print(
        f"Boundary F1@{boundary_metrics['tolerance_pixels']}px: "
        f"smoke={boundary_metrics['smoke']['boundary_f1']:.4f}, "
        f"fire={boundary_metrics['fire']['boundary_f1']:.4f}"
    )
    print(
        f"Inference parameters={inference_parameter_count / 1_000_000:.3f}M, "
        f"latency={latency_ms:.3f} ms, FPS={fps:.1f}"
    )
    print(f"Metrics: {output_dir / 'metrics.json'}")
    print(f"Comparisons: {comparison_dir}")


if __name__ == "__main__":
    main()
