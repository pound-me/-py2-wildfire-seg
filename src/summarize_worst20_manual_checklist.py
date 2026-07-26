from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter
from pathlib import Path


MANUAL_CATEGORICAL_FIELDS = (
    "manual_rgb_visibility_visible_partial_invisible",
    "manual_smoke_occlusion_none_partial_heavy",
    "manual_ir_clearer_yes_no",
    "manual_small_fire_yes_no",
)

NUMERIC_FIELDS = (
    "rank",
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
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and summarize the filled worst-20 visibility checklist."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def numeric_summary(values: list[float]) -> dict[str, float]:
    return {
        "min": min(values),
        "q1_inclusive": statistics.quantiles(values, n=4, method="inclusive")[0],
        "median": statistics.median(values),
        "mean": statistics.fmean(values),
        "q3_inclusive": statistics.quantiles(values, n=4, method="inclusive")[2],
        "max": max(values),
    }


def main() -> None:
    args = parse_args()
    with args.input.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = reader.fieldnames or []

    required = {"name", "manual_notes", *NUMERIC_FIELDS, *MANUAL_CATEGORICAL_FIELDS}
    missing = sorted(required.difference(fieldnames))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if len(rows) != 20:
        raise ValueError(f"Expected exactly 20 reviewed rows, got {len(rows)}")

    ranks = [int(row["rank"]) for row in rows]
    if sorted(ranks) != list(range(1, 21)):
        raise ValueError(f"Ranks must be exactly 1..20, got {sorted(ranks)}")
    names = [row["name"].strip() for row in rows]
    if any(not name for name in names) or len(set(names)) != len(names):
        raise ValueError("Image names must be non-empty and unique")

    categorical: dict[str, dict[str, int]] = {}
    for field in MANUAL_CATEGORICAL_FIELDS:
        values = [row[field].strip().lower() for row in rows]
        if any(not value for value in values):
            raise ValueError(f"Every row must fill {field}")
        categorical[field] = dict(sorted(Counter(values).items()))

    numeric: dict[str, dict[str, float]] = {}
    for field in NUMERIC_FIELDS:
        values = [float(row[field]) for row in rows]
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"Non-finite value found in {field}")
        numeric[field] = numeric_summary(values)

    notes = [row["manual_notes"].strip() for row in rows]
    summary = {
        "source": str(args.input.resolve()),
        "rows": len(rows),
        "unique_images": len(set(names)),
        "rank_min": min(ranks),
        "rank_max": max(ranks),
        "all_manual_fields_complete": True,
        "categorical_counts": categorical,
        "manual_notes": {
            "nonempty": sum(bool(note) for note in notes),
            "exact_text_counts": dict(sorted(Counter(notes).items())),
        },
        "numeric_columns": numeric,
        "headline": {
            "rgb_invisible": categorical[
                "manual_rgb_visibility_visible_partial_invisible"
            ].get("invisible", 0),
            "smoke_heavy": categorical[
                "manual_smoke_occlusion_none_partial_heavy"
            ].get("heavy", 0),
            "ir_clearer_yes": categorical["manual_ir_clearer_yes_no"].get("yes", 0),
            "small_fire_yes": categorical["manual_small_fire_yes_no"].get("yes", 0),
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary["headline"], ensure_ascii=False))


if __name__ == "__main__":
    main()
