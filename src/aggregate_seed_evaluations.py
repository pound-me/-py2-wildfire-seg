from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


METRIC_PATHS = {
    "miou": ("metrics", "miou"),
    "iou_background": ("metrics", "iou_background"),
    "iou_smoke": ("metrics", "iou_smoke"),
    "iou_fire": ("metrics", "iou_fire"),
    "precision_fire": ("metrics", "precision_fire"),
    "recall_fire": ("metrics", "recall_fire"),
    "boundary_f1_fire": ("boundary_metrics", "fire", "boundary_f1"),
    "latency_ms": ("speed", "latency_ms"),
    "fps": ("speed", "fps"),
}


def nested_value(payload: dict, path: tuple[str, ...]) -> float:
    value: object = payload
    for key in path:
        if not isinstance(value, dict) or key not in value:
            raise KeyError(f"Missing evaluation field: {'.'.join(path)}")
        value = value[key]
    return float(value)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate already-evaluated repeated seeds without selecting "
            "checkpoints or running a dataset."
        )
    )
    parser.add_argument(
        "--evaluation",
        type=Path,
        action="append",
        required=True,
        help="Path to one evaluate_baseline.py metrics.json file.",
    )
    parser.add_argument("--expected-seeds", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    paths = [path.resolve() for path in args.evaluation]
    if len(paths) != args.expected_seeds:
        raise ValueError(
            f"Expected {args.expected_seeds} evaluations, got {len(paths)}."
        )
    payloads = [
        json.loads(path.read_text(encoding="utf-8")) for path in paths
    ]
    splits = {payload.get("split") for payload in payloads}
    models = {
        payload.get("model", {}).get("name") for payload in payloads
    }
    if len(splits) != 1:
        raise ValueError(f"Mixed evaluation splits: {sorted(splits)}")
    if len(models) != 1:
        raise ValueError(f"Mixed model names: {sorted(models)}")
    method_hashes = {
        payload.get("method_config_excluding_seed_sha256")
        for payload in payloads
    }
    if None in method_hashes or len(method_hashes) != 1:
        raise ValueError(
            "Method configs differ across seeds or lack a normalized hash: "
            f"{sorted(str(value) for value in method_hashes)}"
        )

    seeds = [payload.get("seed") for payload in payloads]
    if any(seed is None for seed in seeds):
        raise KeyError(
            "An evaluation is missing its seed; regenerate it with the "
            "current evaluate_baseline.py."
        )
    if len(set(int(seed) for seed in seeds)) != len(seeds):
        raise ValueError(f"Duplicate seeds in aggregate: {seeds}")

    aggregates = {}
    per_seed = []
    for path, payload, seed in zip(paths, payloads, seeds):
        values = {
            name: nested_value(payload, field_path)
            for name, field_path in METRIC_PATHS.items()
        }
        per_seed.append(
            {
                "seed": int(seed),
                "evaluation": str(path),
                "checkpoint": payload.get("checkpoint"),
                "checkpoint_epoch": payload.get("checkpoint_epoch"),
                "values": values,
            }
        )

    for name in METRIC_PATHS:
        values = np.asarray(
            [item["values"][name] for item in per_seed],
            dtype=np.float64,
        )
        aggregates[name] = {
            "mean": float(values.mean()),
            "sample_standard_deviation": (
                float(values.std(ddof=1)) if values.size > 1 else None
            ),
            "values": values.tolist(),
        }

    inference_parameters = {
        int(payload["model"]["inference_parameters_main_head"])
        for payload in payloads
    }
    if len(inference_parameters) != 1:
        raise ValueError(
            "Inference parameter counts differ across seeds: "
            f"{sorted(inference_parameters)}"
        )

    report = {
        "split": next(iter(splits)),
        "model": next(iter(models)),
        "seed_count": len(seeds),
        "seeds": [int(seed) for seed in seeds],
        "checkpoint_selection": (
            "Checkpoints were selected by validation mIoU before this "
            "aggregation; this script performs no selection."
        ),
        "method_config_excluding_seed_sha256": next(iter(method_hashes)),
        "inference_parameters_main_head": next(iter(inference_parameters)),
        "per_seed": per_seed,
        "aggregate": aggregates,
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
