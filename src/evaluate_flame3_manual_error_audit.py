from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ALLOWED_CAUSES = {
    "小火/远距离火",
    "低温弱火",
    "烟雾遮挡",
    "火焰边缘",
    "热地面或余热",
    "反光或高亮区域",
    "RGB/IR错位",
    "标签不确定",
    "其他",
}
ADDRESSABLE_FN_CAUSES = {
    "小火/远距离火",
    "低温弱火",
    "烟雾遮挡",
    "火焰边缘",
}
TRISTATE = {"yes", "no", "uncertain"}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply the preregistered CMRC manual-review admission gate."
    )
    parser.add_argument("--checklist", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    checklist = args.checklist.resolve()
    with checklist.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    counts = {
        category: sum(row["category"] == category for row in rows)
        for category in ("FN", "FP", "TP")
    }
    if counts != {"FN": 30, "FP": 20, "TP": 20}:
        raise RuntimeError(f"Unexpected checklist counts: {counts}")
    for row in rows:
        cause = row["primary_cause"].strip()
        if cause not in ALLOWED_CAUSES:
            raise ValueError(f"Invalid or missing primary_cause for {row['audit_id']}: {cause!r}")
        for field in (
            "visible_flame_yes_no_uncertain",
            "registration_mismatch_yes_no_uncertain",
            "counts_as_active_fire_yes_no_uncertain",
        ):
            value = row[field].strip().lower()
            if value not in TRISTATE:
                raise ValueError(f"Invalid {field} for {row['audit_id']}: {value!r}")
        if row["label_uncertain_yes_no"].strip().lower() not in {"yes", "no"}:
            raise ValueError(f"Invalid label uncertainty for {row['audit_id']}")

    fn_rows = [row for row in rows if row["category"] == "FN"]
    fp_rows = [row for row in rows if row["category"] == "FP"]
    registration_fn = sum(
        row["registration_mismatch_yes_no_uncertain"].strip().lower() == "yes"
        for row in fn_rows
    )
    addressable_fn = sum(row["primary_cause"].strip() in ADDRESSABLE_FN_CAUSES for row in fn_rows)
    ambiguous_fp = sum(
        row["primary_cause"].strip() in {"热地面或余热", "标签不确定"}
        or row["label_uncertain_yes_no"].strip().lower() == "yes"
        for row in fp_rows
    )
    passes = registration_fn < 9 and addressable_fn >= 15
    result = {
        "audit": "flame3_cmrc_manual_review_gate",
        "counts": counts,
        "fn_registration_mismatch_yes": registration_fn,
        "fn_addressable_by_frozen_cmrc_hypothesis": addressable_fn,
        "fp_label_uncertain_or_hot_smoldering": ambiguous_fp,
        "fp_reduction_interpretation": (
            "pseudo_label_consistency_only"
            if ambiguous_fp >= 10
            else "ordinary_false_positive_diagnostic"
        ),
        "cmrc_manual_review_gate": "pass" if passes else "stop",
        "rule": {
            "registration_mismatch_fn_max_exclusive": 9,
            "addressable_fn_minimum": 15,
            "ambiguous_fp_semantic_caveat_threshold": 10,
        },
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
