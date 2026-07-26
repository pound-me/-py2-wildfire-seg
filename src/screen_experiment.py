from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_records(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_window(path: Path, start: int, end: int) -> list[dict]:
    records = load_records(path)
    selected = [record for record in records if start <= record["epoch"] <= end]
    expected = end - start + 1
    if len(selected) != expected:
        raise ValueError(
            f"Expected {expected} epochs ({start}-{end}) in {path}, "
            f"found {len(selected)}."
        )
    return selected


def summarize(path: Path, start: int, end: int) -> dict:
    records = load_window(path, start, end)
    all_records = [
        record for record in load_records(path) if record["epoch"] <= end
    ]
    miou = np.asarray(
        [record["validation"]["miou"] for record in records], dtype=np.float64
    )
    fire = np.asarray(
        [record["validation"]["iou_fire"] for record in records],
        dtype=np.float64,
    )
    smoke = np.asarray(
        [record["validation"]["iou_smoke"] for record in records],
        dtype=np.float64,
    )
    prototype_invalid_epochs = [
        int(record["epoch"])
        for record in all_records
        if record.get("prototype_health") is not None
        and not record["prototype_health"].get(
            "valid_multi_prototype_method", True
        )
    ]
    prototype_health_valid = not prototype_invalid_epochs
    return {
        "metrics": str(path),
        "epoch_window": [start, end],
        "miou_mean": float(miou.mean()),
        "fire_iou_mean": float(fire.mean()),
        "fire_conservative_noise_band": float(fire.max() - fire.min()),
        "fire_sample_standard_deviation": float(fire.std(ddof=1)),
        "miou_values": miou.tolist(),
        "fire_iou_values": fire.tolist(),
        "smoke_iou_values": smoke.tolist(),
        "smoke_or_fire_class_collapse": bool(
            (smoke <= 0.0).any() or (fire <= 0.0).any()
        ),
        "prototype_health_valid": prototype_health_valid,
        "prototype_health_checked_epoch_range": [1, end],
        "prototype_invalid_epochs": prototype_invalid_epochs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Screen 30-epoch candidates using the fixed 26-30 protocol."
    )
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, action="append", required=True)
    parser.add_argument("--start-epoch", type=int, default=26)
    parser.add_argument("--end-epoch", type=int, default=30)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    baseline = summarize(
        args.baseline.resolve(), args.start_epoch, args.end_epoch
    )
    conservative_band = baseline["fire_conservative_noise_band"]
    results = []
    for candidate_path in args.candidate:
        candidate = summarize(
            candidate_path.resolve(), args.start_epoch, args.end_epoch
        )
        miou_gain = candidate["miou_mean"] - baseline["miou_mean"]
        fire_gain = candidate["fire_iou_mean"] - baseline["fire_iou_mean"]
        rule_miou = miou_gain >= 0.005 and fire_gain >= -conservative_band
        rule_fire = fire_gain > conservative_band and miou_gain >= -0.005
        if fire_gain > conservative_band:
            fire_interpretation = "significant_positive"
        elif fire_gain < -conservative_band:
            fire_interpretation = "significant_negative"
        else:
            fire_interpretation = "neutral_within_conservative_band"
        health_and_classes_valid = (
            not candidate["smoke_or_fire_class_collapse"]
            and candidate["prototype_health_valid"]
        )
        results.append(
            {
                **candidate,
                "miou_gain": miou_gain,
                "fire_iou_gain": fire_gain,
                "fire_interpretation": fire_interpretation,
                "passes_miou_rule": rule_miou,
                "passes_fire_rule": rule_fire,
                "health_and_classes_valid": health_and_classes_valid,
                "passes_screening": (
                    (rule_miou or rule_fire) and health_and_classes_valid
                ),
            }
        )

    passing = [item for item in results if item["passes_screening"]]
    passing.sort(
        key=lambda item: (item["miou_mean"], item["fire_iou_mean"]),
        reverse=True,
    )
    recommendation = None
    if passing:
        best_miou = passing[0]["miou_mean"]
        near_ties = [
            item for item in passing
            if best_miou - item["miou_mean"] < 0.002
        ]
        recommendation = max(
            near_ties,
            key=lambda item: item["fire_iou_mean"],
        )

    report = {
        "terminology": (
            "fire_conservative_noise_band is max-min, not a statistical sigma"
        ),
        "baseline": baseline,
        "candidates": results,
        "recommended_metrics": (
            recommendation["metrics"] if recommendation else None
        ),
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
