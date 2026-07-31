from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.cluster import KMeans


CLASS_ORDER = ("Fire", "No Fire")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the preregistered metadata-grouped FLAME3 split."
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--pseudolabel-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--time-gap-seconds", type=int, default=300)
    parser.add_argument("--validation-target-ratio", type=float, default=0.15)
    parser.add_argument("--kmeans-seed", type=int, default=200)
    return parser.parse_args()


def gps_decimal(gps: dict[int, object]) -> tuple[float, float]:
    lat_tuple = gps[2]
    lon_tuple = gps[4]
    latitude = (
        float(lat_tuple[0]) + float(lat_tuple[1]) / 60.0 + float(lat_tuple[2]) / 3600.0
    )
    longitude = (
        float(lon_tuple[0]) + float(lon_tuple[1]) / 60.0 + float(lon_tuple[2]) / 3600.0
    )
    if gps.get(1) == "S":
        latitude = -latitude
    if gps.get(3) == "W":
        longitude = -longitude
    return latitude, longitude


def collect_metadata(data_root: Path, pseudolabel_root: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for class_name in CLASS_ORDER:
        safe_class = class_name.replace(" ", "_").lower()
        raw_rgb_dir = data_root / class_name / "RGB" / "Raw"
        corrected_dir = data_root / class_name / "RGB" / "Corrected FOV"
        raw_thermal_dir = data_root / class_name / "Thermal" / "Raw JPG"
        tiff_dir = data_root / class_name / "Thermal" / "Celsius TIFF"
        mask_dir = pseudolabel_root / "train_mask_templates" / safe_class
        for raw_rgb_path in sorted(raw_rgb_dir.glob("*.JPG")):
            stem = raw_rgb_path.stem
            paths = {
                "raw_rgb_path": raw_rgb_path,
                "corrected_rgb_path": corrected_dir / f"{stem}.JPG",
                "raw_thermal_path": raw_thermal_dir / f"{stem}.JPG",
                "thermal_tiff_path": tiff_dir / f"{stem}.TIFF",
                "temperature_mask_path": mask_dir / f"{stem}.png",
            }
            missing = [str(path) for path in paths.values() if not path.is_file()]
            if missing:
                raise FileNotFoundError(f"Missing paired files for {class_name}/{stem}: {missing}")
            image = Image.open(raw_rgb_path)
            exif = image.getexif()
            date_text = exif.get(306)
            if not date_text:
                raise RuntimeError(f"Missing EXIF DateTime: {raw_rgb_path}")
            timestamp = datetime.strptime(str(date_text), "%Y:%m:%d %H:%M:%S")
            gps = exif.get_ifd(34853)
            if not gps:
                raise RuntimeError(f"Missing EXIF GPS: {raw_rgb_path}")
            latitude, longitude = gps_decimal(gps)
            records.append(
                {
                    "sample_key": f"{safe_class}/{stem}",
                    "sample_class": class_name,
                    "sample_id": stem,
                    "timestamp": timestamp,
                    "latitude": latitude,
                    "longitude": longitude,
                    **{name: str(path) for name, path in paths.items()},
                }
            )
    return records


def temporal_blocks(records: list[dict[str, object]], gap_seconds: int) -> list[list[dict[str, object]]]:
    ordered = sorted(records, key=lambda item: (item["timestamp"], item["sample_id"]))
    if not ordered:
        return []
    blocks: list[list[dict[str, object]]] = [[ordered[0]]]
    for item in ordered[1:]:
        gap = (item["timestamp"] - blocks[-1][-1]["timestamp"]).total_seconds()
        if gap > gap_seconds:
            blocks.append([item])
        else:
            blocks[-1].append(item)
    return blocks


def assign_fire(records: list[dict[str, object]], args: argparse.Namespace) -> list[dict[str, object]]:
    blocks = temporal_blocks(records, args.time_gap_seconds)
    expected_sizes = [82, 126, 34, 16, 17, 99, 32, 106, 21, 89]
    sizes = [len(block) for block in blocks]
    if sizes != expected_sizes:
        raise RuntimeError(
            f"Fire temporal blocks differ from preregistration: {sizes} != {expected_sizes}"
        )
    test_index = len(blocks) - 1
    target = len(records) * args.validation_target_ratio
    validation_index = min(
        (index for index in range(len(blocks)) if index != test_index),
        key=lambda index: (abs(len(blocks[index]) - target), index),
    )
    if validation_index != 5:
        raise RuntimeError(
            f"Expected validation fire block 5, got {validation_index}"
        )
    output: list[dict[str, object]] = []
    for block_index, block in enumerate(blocks):
        if block_index == test_index:
            split = "test"
        elif block_index == validation_index:
            split = "val"
        else:
            split = "train"
        for item in block:
            copied = dict(item)
            copied["split"] = split
            copied["time_block_id"] = block_index
            copied["space_cluster_id"] = ""
            output.append(copied)
    return output


def project_local_xy(records: list[dict[str, object]]) -> np.ndarray:
    latitudes = np.asarray([item["latitude"] for item in records], dtype=np.float64)
    longitudes = np.asarray([item["longitude"] for item in records], dtype=np.float64)
    latitude_origin = float(latitudes.mean())
    longitude_origin = float(longitudes.mean())
    x = (
        (longitudes - longitude_origin)
        * 111320.0
        * np.cos(np.deg2rad(latitude_origin))
    )
    y = (latitudes - latitude_origin) * 110540.0
    return np.stack([x, y], axis=1)


def assign_no_fire(records: list[dict[str, object]], args: argparse.Namespace) -> list[dict[str, object]]:
    ordered = sorted(records, key=lambda item: item["sample_key"])
    xy = project_local_xy(ordered)
    model = KMeans(n_clusters=3, random_state=args.kmeans_seed, n_init=20)
    labels = model.fit_predict(xy)
    cluster_counts = Counter(int(label) for label in labels.tolist())
    sorted_clusters = sorted(cluster_counts, key=lambda label: (-cluster_counts[label], label))
    expected_ranked_sizes = [63, 35, 18]
    ranked_sizes = [cluster_counts[label] for label in sorted_clusters]
    if ranked_sizes != expected_ranked_sizes:
        raise RuntimeError(
            f"No Fire cluster sizes differ from preregistration: "
            f"{ranked_sizes} != {expected_ranked_sizes}"
        )
    cluster_to_split = {
        sorted_clusters[0]: "train",
        sorted_clusters[1]: "val",
        sorted_clusters[2]: "test",
    }
    output: list[dict[str, object]] = []
    for item, label in zip(ordered, labels.tolist()):
        copied = dict(item)
        copied["split"] = cluster_to_split[int(label)]
        copied["time_block_id"] = ""
        copied["space_cluster_id"] = int(label)
        output.append(copied)
    return output


def csv_value(value: object) -> object:
    if isinstance(value, datetime):
        return value.isoformat(sep=" ")
    return value


def write_split(records: list[dict[str, object]], path: Path) -> None:
    fields = [
        "sample_key",
        "sample_class",
        "sample_id",
        "timestamp",
        "latitude",
        "longitude",
        "time_block_id",
        "space_cluster_id",
        "raw_rgb_path",
        "corrected_rgb_path",
        "raw_thermal_path",
        "thermal_tiff_path",
        "temperature_mask_path",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in sorted(records, key=lambda row: row["sample_key"]):
            writer.writerow({field: csv_value(item[field]) for field in fields})


def main() -> None:
    args = parse_args()
    args.data_root = args.data_root.resolve()
    args.pseudolabel_root = args.pseudolabel_root.resolve()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)

    records = collect_metadata(args.data_root, args.pseudolabel_root)
    fire = [item for item in records if item["sample_class"] == "Fire"]
    no_fire = [item for item in records if item["sample_class"] == "No Fire"]
    assigned = assign_fire(fire, args) + assign_no_fire(no_fire, args)

    if len({item["sample_key"] for item in assigned}) != len(assigned):
        raise RuntimeError("Duplicate sample keys detected")
    expected = {
        "train": {"Fire": 434, "No Fire": 63, "total": 497},
        "val": {"Fire": 99, "No Fire": 35, "total": 134},
        "test": {"Fire": 89, "No Fire": 18, "total": 107},
    }
    summary_counts: dict[str, dict[str, int]] = {}
    for split in ("train", "val", "test"):
        split_items = [item for item in assigned if item["split"] == split]
        counts = Counter(str(item["sample_class"]) for item in split_items)
        summary_counts[split] = {
            "Fire": counts.get("Fire", 0),
            "No Fire": counts.get("No Fire", 0),
            "total": len(split_items),
        }
        if summary_counts[split] != expected[split]:
            raise RuntimeError(
                f"Split count mismatch for {split}: "
                f"{summary_counts[split]} != {expected[split]}"
            )
        write_split(split_items, args.output / f"{split}.csv")

    summary = {
        "status": "preregistered_test_sealed",
        "data_root": str(args.data_root),
        "pseudolabel_root": str(args.pseudolabel_root),
        "counts": summary_counts,
        "fire_time_gap_seconds": args.time_gap_seconds,
        "fire_validation_target_ratio": args.validation_target_ratio,
        "no_fire_kmeans": {
            "clusters": 3,
            "random_state": args.kmeans_seed,
            "n_init": 20,
            "assignment": "largest=train, medium=val, smallest=test",
        },
        "zero_shot_allowed_split": "val",
        "test_usage": "final_once_after_method_freeze",
        "absolute_date_note": (
            "Official data card states 2023; Raw RGB EXIF states 2022. "
            "Only relative EXIF order is used for grouping."
        ),
        "future_multiburn_upgrade": (
            "Replace this split with complete burn-event grouping before any "
            "multi-burn model result is produced."
        ),
    }
    (args.output / "split_manifest.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
