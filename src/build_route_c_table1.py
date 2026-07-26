from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


MODES = ("rgb", "ir", "fusion")
DISPLAY_NAMES = {
    "rgb": "RGB-only PIDNet-S",
    "ir": "IR-only PIDNet-S",
    "fusion": "RGB+IR Fusion PIDNet-S",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the frozen Route C input-modality motivation Table 1."
    )
    for mode in MODES:
        parser.add_argument(f"--{mode}-metrics", type=Path, required=True)
        parser.add_argument(f"--{mode}-complexity", type=Path, required=True)
    parser.add_argument("--latency", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.resolve().read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    latency_path = args.latency.resolve()
    latency = load_json(latency_path)
    if set(latency.get("models", {})) != set(MODES):
        raise RuntimeError("Latency JSON must contain exactly rgb, ir and fusion models.")

    rows = []
    sources = {"latency": str(latency_path), "models": {}}
    for mode in MODES:
        metrics_path = getattr(args, f"{mode}_metrics").resolve()
        complexity_path = getattr(args, f"{mode}_complexity").resolve()
        metrics_file = load_json(metrics_path)
        complexity = load_json(complexity_path)
        if metrics_file.get("split") != "val":
            raise RuntimeError(f"{mode} metrics are not validation metrics.")
        metrics = metrics_file["metrics"]
        latency_row = latency["models"][mode]
        complexity_checkpoint = str(complexity.get("checkpoint", ""))
        metrics_checkpoint = str(metrics_file.get("checkpoint", ""))
        latency_checkpoint = str(latency_row.get("checkpoint", ""))
        if not (
            Path(complexity_checkpoint).resolve() == Path(metrics_checkpoint).resolve()
            == Path(latency_checkpoint).resolve()
        ):
            raise RuntimeError(f"{mode} artifacts do not use the same checkpoint.")
        if list(complexity.get("input_shape", []))[1] != {"rgb": 3, "ir": 1, "fusion": 4}[mode]:
            raise RuntimeError(f"{mode} complexity input channel count is incorrect.")
        row = {
            "mode": mode,
            "method": DISPLAY_NAMES[mode],
            "best_epoch": int(metrics_file["checkpoint_epoch"]),
            "background_iou": float(metrics["iou_background"]),
            "smoke_iou": float(metrics["iou_smoke"]),
            "fire_iou": float(metrics["iou_fire"]),
            "miou": float(metrics["miou"]),
            "parameters": int(complexity["inference_parameters_main_head"]),
            "parameters_millions": float(
                complexity["inference_parameters_main_head"] / 1_000_000.0
            ),
            "gflops": float(complexity["forward_gflops"]),
            "latency_ms_median": float(latency_row["latency_ms_median"]),
            "fps": float(latency_row["fps_from_median"]),
            "passes_30_fps": bool(latency_row["passes_30_fps"]),
        }
        rows.append(row)
        sources["models"][mode] = {
            "metrics": str(metrics_path),
            "complexity": str(complexity_path),
            "checkpoint": metrics_checkpoint,
        }

    result = {
        "title": "Table 1: input-modality motivation on the validation split",
        "test_set_used": False,
        "selection": "each 100-epoch run's validation-mIoU-selected best checkpoint",
        "latency_protocol": latency.get("protocol"),
        "all_models_pass_30_fps": all(row["passes_30_fps"] for row in rows),
        "rows": rows,
        "sources": sources,
    }

    output_json = args.output_json.resolve()
    output_csv = args.output_csv.resolve()
    output_markdown = args.output_markdown.resolve()
    for output in (output_json, output_csv, output_markdown):
        output.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    fields = list(rows[0])
    with output_csv.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    markdown = [
        "# Table 1: input-modality motivation",
        "",
        "Validation split only; test set not used.",
        "",
        "| Input | Background IoU | Smoke IoU | Fire IoU | mIoU | Params (M) | GFLOPs | FPS |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        markdown.append(
            f"| {row['method']} | {row['background_iou']:.6f} | "
            f"{row['smoke_iou']:.6f} | {row['fire_iou']:.6f} | "
            f"{row['miou']:.6f} | {row['parameters_millions']:.3f} | "
            f"{row['gflops']:.4f} | {row['fps']:.2f} |"
        )
    markdown.extend(
        [
            "",
            "FPS uses the RTX 2060 same-process rotating-order median protocol.",
            "Every row must pass at least 30 FPS under the active Route C rule.",
        ]
    )
    output_markdown.write_text("\n".join(markdown) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
