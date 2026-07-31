from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

from generate_flame3_split import (
    assign_no_fire,
    collect_metadata,
    temporal_blocks,
    write_split,
)


FIRE_ID = 2
IGNORE_ID = 255
EXPECTED_ORIGINAL_FIRE_BLOCK_SIZES = [82, 126, 34, 16, 17, 99, 32, 106, 21, 89]
EXPECTED_COUNTS = {
    "train": {"Fire": 430, "No Fire": 63, "total": 493},
    "val": {"Fire": 99, "No Fire": 35, "total": 134},
    "test": {"Fire": 89, "No Fire": 18, "total": 107},
}
EXPECTED_EMPTY_FIRE_COUNTS = {
    "train": {"empty": 62, "fire_total": 430},
    "val": {"empty": 16, "fire_total": 99},
    "test": {"empty": 2, "fire_total": 89},
}
BUFFER_FRAMES = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate preregistered FLAME3 split v2 with temporally contiguous "
            "empty-pseudolabel balancing."
        )
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--pseudolabel-root", type=Path, required=True)
    parser.add_argument("--v1-split-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--time-gap-seconds", type=int, default=300)
    parser.add_argument("--kmeans-seed", type=int, default=200)
    parser.add_argument("--v1-validation-fire-count", type=int, default=99)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mask_has_pseudo_fire(path: str) -> bool:
    mask = np.asarray(Image.open(path), dtype=np.uint8)
    values = set(int(value) for value in np.unique(mask).tolist())
    allowed = {0, FIRE_ID, IGNORE_ID}
    if not values.issubset(allowed):
        raise RuntimeError(f"Unexpected pseudolabel values {sorted(values)} in {path}")
    return bool(np.any(mask == FIRE_ID))


def annotate_empty_status(records: list[dict[str, object]]) -> None:
    for item in records:
        if item["sample_class"] == "Fire":
            item["empty_pseudolabel"] = not mask_has_pseudo_fire(
                str(item["temperature_mask_path"])
            )
        else:
            item["empty_pseudolabel"] = True


def choose_validation_blocks(
    block_sizes: list[int], empty_subblock_size: int, target: int
) -> tuple[int, ...]:
    eligible = tuple(range(1, len(block_sizes) - 1))
    candidates: list[tuple[tuple[object, ...], tuple[int, ...]]] = []
    for count in range(1, len(eligible) + 1):
        for indices in itertools.combinations(eligible, count):
            total = empty_subblock_size + sum(block_sizes[index] for index in indices)
            score = (abs(total - target), len(indices), indices)
            candidates.append((score, indices))
    candidates.sort(key=lambda item: item[0])
    best_score, best_indices = candidates[0]
    if best_score[0] != 0:
        raise RuntimeError(
            "No exact whole-block combination preserves the v1 Fire validation count"
        )
    return best_indices


def assign_fire_v2(
    records: list[dict[str, object]], args: argparse.Namespace
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    dict[str, object],
]:
    blocks = temporal_blocks(records, args.time_gap_seconds)
    sizes = [len(block) for block in blocks]
    if sizes != EXPECTED_ORIGINAL_FIRE_BLOCK_SIZES:
        raise RuntimeError(
            f"Fire temporal blocks changed: {sizes} != {EXPECTED_ORIGINAL_FIRE_BLOCK_SIZES}"
        )

    block0 = sorted(blocks[0], key=lambda item: (item["timestamp"], item["sample_id"]))
    internal_gaps = [
        (
            (block0[index]["timestamp"] - block0[index - 1]["timestamp"]).total_seconds(),
            index,
        )
        for index in range(1, len(block0))
    ]
    largest_gap_seconds, split_index = max(internal_gaps, key=lambda item: (item[0], -item[1]))
    if split_index != 16 or largest_gap_seconds != 199.0:
        raise RuntimeError(
            "The preregistered block0 natural boundary changed: "
            f"index={split_index}, gap={largest_gap_seconds}"
        )
    block0a = block0[:split_index]
    block0b = block0[split_index:]
    buffer = block0b[:BUFFER_FRAMES]
    block0b_train = block0b[BUFFER_FRAMES:]
    if not all(bool(item["empty_pseudolabel"]) for item in block0):
        raise RuntimeError("Original Fire block0 is no longer entirely empty-pseudolabel")
    if len(buffer) != BUFFER_FRAMES or len(block0b_train) != 62:
        raise RuntimeError("The preregistered four-frame boundary buffer changed")

    validation_whole_blocks = choose_validation_blocks(
        sizes, len(block0a), args.v1_validation_fire_count
    )
    if validation_whole_blocks != (2, 4, 6):
        raise RuntimeError(
            f"Expected unique validation block combination (2, 4, 6), got {validation_whole_blocks}"
        )

    assigned: list[dict[str, object]] = []
    block_rows: list[dict[str, object]] = []

    def add_block(
        items: list[dict[str, object]],
        block_id: str,
        original_block_id: int,
        split: str,
        boundary_reason: str,
    ) -> None:
        ordered = sorted(items, key=lambda item: (item["timestamp"], item["sample_id"]))
        for item in ordered:
            copied = dict(item)
            copied["split"] = split
            copied["time_block_id"] = block_id
            copied["space_cluster_id"] = ""
            assigned.append(copied)
        block_rows.append(
            {
                "time_block_id": block_id,
                "original_block_id": original_block_id,
                "split": split,
                "count": len(ordered),
                "empty_pseudolabel_count": sum(
                    int(bool(item["empty_pseudolabel"])) for item in ordered
                ),
                "start_timestamp": ordered[0]["timestamp"].isoformat(sep=" "),
                "end_timestamp": ordered[-1]["timestamp"].isoformat(sep=" "),
                "first_sample_id": ordered[0]["sample_id"],
                "last_sample_id": ordered[-1]["sample_id"],
                "boundary_reason": boundary_reason,
            }
        )

    add_block(
        block0a,
        "0a",
        0,
        "val",
        "smaller side before the largest 199-second internal gap in original block0",
    )
    add_block(
        block0b_train,
        "0b",
        0,
        "train",
        "larger side after the four-frame excluded boundary buffer",
    )
    buffer_rows: list[dict[str, object]] = []
    for item in buffer:
        buffer_rows.append(
            {
                "sample_key": item["sample_key"],
                "sample_class": item["sample_class"],
                "sample_id": item["sample_id"],
                "timestamp": item["timestamp"].isoformat(sep=" "),
                "original_block_id": 0,
                "buffer_position": len(buffer_rows) + 1,
                "empty_pseudolabel": item["empty_pseudolabel"],
                "reason": "four-frame train-side buffer at the 0a/0b cross-split boundary",
            }
        )
    block_rows.append(
        {
            "time_block_id": "0buf",
            "original_block_id": 0,
            "split": "excluded_buffer",
            "count": len(buffer),
            "empty_pseudolabel_count": len(buffer),
            "start_timestamp": buffer[0]["timestamp"].isoformat(sep=" "),
            "end_timestamp": buffer[-1]["timestamp"].isoformat(sep=" "),
            "first_sample_id": buffer[0]["sample_id"],
            "last_sample_id": buffer[-1]["sample_id"],
            "boundary_reason": "four consecutive frames excluded after the natural 199-second gap",
        }
    )
    for block_index, block in enumerate(blocks[1:], start=1):
        if block_index == len(blocks) - 1:
            split = "test"
        elif block_index in validation_whole_blocks:
            split = "val"
        else:
            split = "train"
        add_block(
            block,
            str(block_index),
            block_index,
            split,
            "original >300-second temporal block retained intact",
        )

    decision = {
        "original_block0_size": len(block0),
        "block0_split_rule": "largest internal timestamp gap; smaller side assigned to validation",
        "block0_boundary": {
            "gap_seconds": largest_gap_seconds,
            "left_count": len(block0a),
            "right_count": len(block0b),
            "left_end": block0a[-1]["timestamp"].isoformat(sep=" "),
            "right_start": block0b[0]["timestamp"].isoformat(sep=" "),
            "left_last_sample_id": block0a[-1]["sample_id"],
            "right_first_sample_id": block0b[0]["sample_id"],
        },
        "boundary_buffer": {
            "policy": "exclude four consecutive frames from the train side of the 0a/0b boundary",
            "count": len(buffer),
            "start": buffer[0]["timestamp"].isoformat(sep=" "),
            "end": buffer[-1]["timestamp"].isoformat(sep=" "),
            "sample_ids": [item["sample_id"] for item in buffer],
            "retained_train_start": block0b_train[0]["timestamp"].isoformat(sep=" "),
            "retained_train_first_sample_id": block0b_train[0]["sample_id"],
        },
        "validation_whole_block_rule": (
            "among original non-test blocks 1-8, choose the fewest-block lexicographic "
            "combination that exactly preserves the v1 Fire validation count after adding 0a"
        ),
        "validation_whole_blocks": list(validation_whole_blocks),
        "validation_fire_count": args.v1_validation_fire_count,
    }
    return assigned, block_rows, buffer_rows, decision


def load_v1_assignments(root: Path) -> dict[str, str]:
    assignments: dict[str, str] = {}
    for split in ("train", "val", "test"):
        path = root / f"{split}.csv"
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                key = row["sample_key"]
                if key in assignments:
                    raise RuntimeError(f"Duplicate v1 sample key: {key}")
                assignments[key] = split
    return assignments


def validate_and_summarize(
    assigned: list[dict[str, object]],
    excluded_buffer: list[dict[str, object]],
    v1_assignments: dict[str, str],
) -> tuple[dict[str, dict[str, int]], dict[str, dict[str, float | int]]]:
    if len({str(item["sample_key"]) for item in assigned}) != len(assigned):
        raise RuntimeError("Duplicate sample keys detected in v2")
    assigned_keys = {str(item["sample_key"]) for item in assigned}
    excluded_keys = {str(item["sample_key"]) for item in excluded_buffer}
    if assigned_keys & excluded_keys:
        raise RuntimeError("Excluded buffer overlaps retained v2 samples")
    if set(v1_assignments) != assigned_keys | excluded_keys:
        raise RuntimeError("v1 universe differs from retained v2 plus its boundary buffer")
    if len(excluded_keys) != BUFFER_FRAMES:
        raise RuntimeError(f"Expected {BUFFER_FRAMES} excluded buffer samples")

    counts: dict[str, dict[str, int]] = {}
    empty_counts: dict[str, dict[str, float | int]] = {}
    for split in ("train", "val", "test"):
        items = [item for item in assigned if item["split"] == split]
        class_counts = Counter(str(item["sample_class"]) for item in items)
        counts[split] = {
            "Fire": class_counts.get("Fire", 0),
            "No Fire": class_counts.get("No Fire", 0),
            "total": len(items),
        }
        if counts[split] != EXPECTED_COUNTS[split]:
            raise RuntimeError(
                f"Count mismatch for {split}: {counts[split]} != {EXPECTED_COUNTS[split]}"
            )
        fire_items = [item for item in items if item["sample_class"] == "Fire"]
        empty = sum(int(bool(item["empty_pseudolabel"])) for item in fire_items)
        expected_empty = EXPECTED_EMPTY_FIRE_COUNTS[split]
        if empty != expected_empty["empty"] or len(fire_items) != expected_empty["fire_total"]:
            raise RuntimeError(
                f"Empty-pseudolabel mismatch for {split}: {empty}/{len(fire_items)}"
            )
        empty_counts[split] = {
            "empty": empty,
            "fire_total": len(fire_items),
            "ratio": empty / max(len(fire_items), 1),
        }

    no_fire_changes = [
        str(item["sample_key"])
        for item in assigned
        if item["sample_class"] == "No Fire"
        and item["split"] != v1_assignments[str(item["sample_key"])]
    ]
    if no_fire_changes:
        raise RuntimeError(f"No Fire assignments changed: {no_fire_changes[:5]}")

    v1_test = {key for key, split in v1_assignments.items() if split == "test"}
    v2_test = {str(item["sample_key"]) for item in assigned if item["split"] == "test"}
    if v1_test != v2_test:
        raise RuntimeError("Sealed test membership changed between v1 and v2")

    fire_block_splits: dict[str, set[str]] = defaultdict(set)
    for item in assigned:
        if item["sample_class"] == "Fire":
            fire_block_splits[str(item["time_block_id"])].add(str(item["split"]))
    crossed = {key: value for key, value in fire_block_splits.items() if len(value) != 1}
    if crossed:
        raise RuntimeError(f"Temporal subblocks cross splits: {crossed}")
    return counts, empty_counts


def adjacent_pair_report(
    fire_records: list[dict[str, object]],
    assigned: list[dict[str, object]],
    excluded_buffer: list[dict[str, object]],
    time_gap_seconds: int,
) -> dict[str, object]:
    state = {str(item["sample_key"]): str(item["split"]) for item in assigned}
    state.update({str(item["sample_key"]): "excluded_buffer" for item in excluded_buffer})
    ordered = sorted(fire_records, key=lambda item: (item["timestamp"], item["sample_id"]))
    retained_cross_pairs: list[dict[str, object]] = []
    for left, right in zip(ordered, ordered[1:]):
        left_state = state[str(left["sample_key"])]
        right_state = state[str(right["sample_key"])]
        gap = (right["timestamp"] - left["timestamp"]).total_seconds()
        retained_states = {"train", "val", "test"}
        if (
            left_state in retained_states
            and right_state in retained_states
            and left_state != right_state
        ):
            retained_cross_pairs.append(
                {
                    "left_sample_key": left["sample_key"],
                    "right_sample_key": right["sample_key"],
                    "left_split": left_state,
                    "right_split": right_state,
                    "gap_seconds": gap,
                    "within_300_second_temporal_block": gap <= time_gap_seconds,
                }
            )
    within_block = [
        item for item in retained_cross_pairs if item["within_300_second_temporal_block"]
    ]
    if within_block:
        raise RuntimeError(f"Adjacent retained frames cross splits within a block: {within_block}")
    return {
        "definition": "consecutive Fire captures after EXIF ordering; excluded-buffer frames are not retained",
        "retained_cross_split_pairs_within_300_second_block": len(within_block),
        "retained_cross_split_pairs_across_all_consecutive_records": len(retained_cross_pairs),
        "all_global_cross_pairs_are_natural_gaps_over_300_seconds": all(
            float(item["gap_seconds"]) > time_gap_seconds for item in retained_cross_pairs
        ),
        "global_cross_pairs": retained_cross_pairs,
    }


def write_rows(rows: list[dict[str, object]], path: Path) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    args.data_root = args.data_root.resolve()
    args.pseudolabel_root = args.pseudolabel_root.resolve()
    args.v1_split_root = args.v1_split_root.resolve()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)

    records = collect_metadata(args.data_root, args.pseudolabel_root)
    annotate_empty_status(records)
    fire = [item for item in records if item["sample_class"] == "Fire"]
    no_fire = [item for item in records if item["sample_class"] == "No Fire"]
    assigned_fire, block_rows, excluded_buffer, decision = assign_fire_v2(fire, args)
    assigned_no_fire = assign_no_fire(no_fire, args)
    for item in assigned_no_fire:
        item["empty_pseudolabel"] = True
    assigned = assigned_fire + assigned_no_fire

    v1_assignments = load_v1_assignments(args.v1_split_root)
    counts, empty_counts = validate_and_summarize(
        assigned, excluded_buffer, v1_assignments
    )
    adjacency = adjacent_pair_report(
        fire, assigned, excluded_buffer, args.time_gap_seconds
    )

    for split in ("train", "val", "test"):
        split_items = [item for item in assigned if item["split"] == split]
        write_split(split_items, args.output / f"{split}.csv")
    write_rows(block_rows, args.output / "fire_time_blocks_v2.csv")
    write_rows(excluded_buffer, args.output / "excluded_boundary_buffer.csv")

    movement_rows = []
    for item in sorted(assigned, key=lambda row: str(row["sample_key"])):
        key = str(item["sample_key"])
        if v1_assignments[key] != item["split"]:
            movement_rows.append(
                {
                    "sample_key": key,
                    "sample_class": item["sample_class"],
                    "timestamp": item["timestamp"].isoformat(sep=" ")
                    if isinstance(item["timestamp"], datetime)
                    else item["timestamp"],
                    "time_block_id_v2": item["time_block_id"],
                    "empty_pseudolabel": item["empty_pseudolabel"],
                    "v1_split": v1_assignments[key],
                    "v2_split": item["split"],
                }
            )
    for item in excluded_buffer:
        key = str(item["sample_key"])
        movement_rows.append(
            {
                "sample_key": key,
                "sample_class": item["sample_class"],
                "timestamp": item["timestamp"],
                "time_block_id_v2": "0buf",
                "empty_pseudolabel": item["empty_pseudolabel"],
                "v1_split": v1_assignments[key],
                "v2_split": "excluded_buffer",
            }
        )
    movement_rows.sort(key=lambda item: str(item["sample_key"]))
    write_rows(movement_rows, args.output / "v1_to_v2_movements.csv")

    output_hashes = {
        name: sha256_file(args.output / name)
        for name in (
            "train.csv",
            "val.csv",
            "test.csv",
            "fire_time_blocks_v2.csv",
            "excluded_boundary_buffer.csv",
            "v1_to_v2_movements.csv",
        )
    }
    manifest = {
        "status": "preregistered_before_flame3_training_test_sealed",
        "version": "split_v2",
        "date": "2026-07-31",
        "supersedes": str(args.v1_split_root),
        "revision_motivation": (
            "v1 placed 82 of 84 empty Fire pseudolabel frames in train and none in val; "
            "v2 balances this state across train and val without random frame-level splitting"
        ),
        "data_root": str(args.data_root),
        "pseudolabel_root": str(args.pseudolabel_root),
        "counts": counts,
        "retained_total": sum(item["total"] for item in counts.values()),
        "excluded_boundary_buffer_count": len(excluded_buffer),
        "empty_fire_pseudolabel_counts": empty_counts,
        "fire_assignment_decision": decision,
        "adjacent_frame_cross_split_report": adjacency,
        "no_fire_assignment": {
            "method": "unchanged v1 KMeans spatial clusters",
            "clusters": 3,
            "random_state": args.kmeans_seed,
            "n_init": 20,
            "verified_identical_to_v1": True,
        },
        "test_membership_identical_to_v1": True,
        "zero_shot_allowed_split": "val",
        "test_usage": "final_once_after_method_freeze",
        "output_sha256": output_hashes,
        "absolute_date_note": (
            "Official data card states 2023; Raw RGB EXIF states 2022. "
            "Only relative EXIF order and gaps are used."
        ),
    }
    (args.output / "split_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
