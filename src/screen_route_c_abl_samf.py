from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_window(path: Path, start: int, end: int) -> list[dict]:
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    selected = [record for record in records if start <= record["epoch"] <= end]
    expected = end - start + 1
    if len(selected) != expected:
        raise ValueError(f"Expected {expected} epochs in {path}, found {len(selected)}.")
    return selected


def summarize(records: list[dict]) -> dict:
    miou = np.asarray([item["validation"]["miou"] for item in records])
    smoke = np.asarray([item["validation"]["iou_smoke"] for item in records])
    fire = np.asarray([item["validation"]["iou_fire"] for item in records])
    return {
        "miou_mean": float(miou.mean()),
        "smoke_iou_mean": float(smoke.mean()),
        "fire_iou_mean": float(fire.mean()),
        "miou_values": miou.tolist(),
        "smoke_iou_values": smoke.tolist(),
        "fire_iou_values": fire.tolist(),
        "smoke_or_fire_class_collapse": bool((smoke <= 0.0).any() or (fire <= 0.0).any()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply the preregistered Route C ABL+SAMF combination gate."
    )
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--start-epoch", type=int, default=26)
    parser.add_argument("--end-epoch", type=int, default=30)
    parser.add_argument("--miou-gain", type=float, default=0.005)
    parser.add_argument("--fire-gain", type=float, default=0.01)
    parser.add_argument("--paired-latency", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    baseline = summarize(load_window(args.baseline.resolve(), args.start_epoch, args.end_epoch))
    candidate = summarize(load_window(args.candidate.resolve(), args.start_epoch, args.end_epoch))
    miou_gain = candidate["miou_mean"] - baseline["miou_mean"]
    fire_gain = candidate["fire_iou_mean"] - baseline["fire_iou_mean"]
    latency = json.loads(args.paired_latency.resolve().read_text(encoding="utf-8"))
    passes_latency = bool(latency.get("route_c_candidate_admitted", False))
    passes_miou = miou_gain >= args.miou_gain
    passes_fire = fire_gain >= args.fire_gain
    no_class_collapse = not candidate["smoke_or_fire_class_collapse"]
    passes = (passes_miou or passes_fire) and no_class_collapse and passes_latency
    report = {
        "test_set_used": False,
        "epoch_window": [args.start_epoch, args.end_epoch],
        "baseline_metrics": str(args.baseline.resolve()),
        "candidate_metrics": str(args.candidate.resolve()),
        "paired_latency": str(args.paired_latency.resolve()),
        "baseline": baseline,
        "candidate": candidate,
        "gains_vs_plain_fusion": {
            "miou": miou_gain,
            "fire_iou": fire_gain,
        },
        "preregistered_thresholds": {
            "miou_gain": args.miou_gain,
            "fire_iou_gain": args.fire_gain,
            "logic": "miou_gain >= threshold OR fire_iou_gain >= threshold",
        },
        "passes_miou_rule": passes_miou,
        "passes_fire_rule": passes_fire,
        "no_smoke_or_fire_class_collapse": no_class_collapse,
        "paired_rtx2060_fps": latency["candidate"]["fps_from_median"],
        "passes_30_fps": passes_latency,
        "passes_30_epoch_combination_screen": passes,
        "promote_to_fresh_100_epochs": passes,
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
