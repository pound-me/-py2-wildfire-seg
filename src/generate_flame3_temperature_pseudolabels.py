from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


CLASS_ORDER = ("Fire", "No Fire")
BACKGROUND_ID = 0
FIRE_ID = 2
IGNORE_ID = 255


@dataclass
class ImageStatistics:
    sample_class: str
    sample_id: str
    rgb_path: str
    thermal_path: str
    height: int
    width: int
    min_temperature_c: float
    max_temperature_c: float
    mean_temperature_c: float
    low_threshold_pixels: int
    high_seed_pixels: int
    kept_fire_pixels_before_ignore: int
    train_fire_pixels: int
    ignore_pixels: int
    component_count_before_cleanup: int
    component_count_after_cleanup: int
    removed_small_component_count: int
    filled_hole_count: int
    fire_ratio_before_ignore: float
    train_fire_ratio: float
    ignore_ratio: float
    no_fire_hotspot_flag: bool
    output_binary_mask: str
    output_train_mask: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate preregistered temperature-derived active-fire pseudo labels "
            "for the FLAME 3 Sycan Marsh CV subset. Source data is never modified."
        )
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--low-threshold", type=float, default=80.0)
    parser.add_argument("--high-threshold", type=float, default=200.0)
    parser.add_argument("--tiny-area", type=int, default=4)
    parser.add_argument("--tiny-peak-keep", type=float, default=300.0)
    parser.add_argument("--max-hole-area", type=int, default=16)
    parser.add_argument("--boundary-radius", type=int, default=2)
    parser.add_argument("--review-count", type=int, default=30)
    parser.add_argument("--review-fire-count", type=int, default=24)
    parser.add_argument("--seed", type=int, default=200)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_pairs(data_root: Path) -> list[tuple[str, str, Path, Path]]:
    records: list[tuple[str, str, Path, Path]] = []
    for class_name in CLASS_ORDER:
        rgb_dir = data_root / class_name / "RGB" / "Corrected FOV"
        thermal_dir = data_root / class_name / "Thermal" / "Celsius TIFF"
        if not rgb_dir.is_dir() or not thermal_dir.is_dir():
            raise FileNotFoundError(
                f"Missing FLAME3 directory for {class_name}: {rgb_dir} / {thermal_dir}"
            )
        rgb_by_stem = {path.stem: path for path in rgb_dir.glob("*.JPG")}
        thermal_by_stem = {path.stem: path for path in thermal_dir.glob("*.TIFF")}
        if set(rgb_by_stem) != set(thermal_by_stem):
            raise RuntimeError(f"RGB/TIFF stem mismatch in {class_name}")
        records.extend(
            (class_name, stem, rgb_by_stem[stem], thermal_by_stem[stem])
            for stem in sorted(rgb_by_stem)
        )
    return records


def component_filter(
    candidate: np.ndarray,
    high_seed: np.ndarray,
    temperature: np.ndarray,
    tiny_area: int,
    tiny_peak_keep: float,
) -> tuple[np.ndarray, int, int, int]:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        candidate.astype(np.uint8), connectivity=8
    )
    kept = np.zeros(candidate.shape, dtype=bool)
    before = max(count - 1, 0)
    after = 0
    removed_small = 0
    for label_id in range(1, count):
        component = labels == label_id
        if not np.any(high_seed[component]):
            continue
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        peak = float(np.nanmax(temperature[component]))
        if area < tiny_area and peak < tiny_peak_keep:
            removed_small += 1
            continue
        kept[component] = True
        after += 1
    return kept, before, after, removed_small


def fill_small_holes(mask: np.ndarray, max_hole_area: int) -> tuple[np.ndarray, int]:
    inverse = (~mask).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(inverse, connectivity=8)
    filled = mask.copy()
    height, width = mask.shape
    filled_count = 0
    for label_id in range(1, count):
        left = int(stats[label_id, cv2.CC_STAT_LEFT])
        top = int(stats[label_id, cv2.CC_STAT_TOP])
        component_width = int(stats[label_id, cv2.CC_STAT_WIDTH])
        component_height = int(stats[label_id, cv2.CC_STAT_HEIGHT])
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        touches_border = (
            left == 0
            or top == 0
            or left + component_width >= width
            or top + component_height >= height
        )
        if not touches_border and area <= max_hole_area:
            filled[labels == label_id] = True
            filled_count += 1
    return filled, filled_count


def build_train_mask(
    fire_mask: np.ndarray,
    high_seed: np.ndarray,
    boundary_radius: int,
) -> tuple[np.ndarray, np.ndarray]:
    train_mask = np.full(fire_mask.shape, BACKGROUND_ID, dtype=np.uint8)
    train_mask[fire_mask] = FIRE_ID
    if boundary_radius <= 0:
        return train_mask, np.zeros(fire_mask.shape, dtype=bool)
    kernel_size = boundary_radius * 2 + 1
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )
    fire_u8 = fire_mask.astype(np.uint8)
    dilated = cv2.dilate(fire_u8, kernel, iterations=1).astype(bool)
    eroded = cv2.erode(fire_u8, kernel, iterations=1).astype(bool)
    boundary = np.logical_xor(dilated, eroded)
    train_mask[boundary] = IGNORE_ID
    # Preserve high-confidence seeds even when a tiny component lies entirely
    # inside the symmetric boundary band.
    train_mask[high_seed & fire_mask] = FIRE_ID
    return train_mask, boundary


def robust_colorize_temperature(temperature: np.ndarray) -> np.ndarray:
    finite = temperature[np.isfinite(temperature)]
    low, high = np.percentile(finite, [1.0, 99.0])
    if high <= low:
        high = low + 1.0
    normalized = np.clip((temperature - low) / (high - low), 0.0, 1.0)
    return cv2.applyColorMap(
        (normalized * 255.0).astype(np.uint8), cv2.COLORMAP_INFERNO
    )


def render_review_image(
    statistics: ImageStatistics,
    rgb_bgr: np.ndarray,
    temperature: np.ndarray,
    low_mask: np.ndarray,
    high_seed: np.ndarray,
    fire_mask: np.ndarray,
    train_mask: np.ndarray,
) -> np.ndarray:
    height, width = fire_mask.shape
    thermal_bgr = robust_colorize_temperature(temperature)

    thresholds = np.zeros_like(rgb_bgr)
    thresholds[low_mask] = (0, 180, 255)
    thresholds[high_seed] = (0, 0, 255)

    fire_overlay = rgb_bgr.copy()
    if np.any(fire_mask):
        source_pixels = rgb_bgr[fire_mask].astype(np.float32)
        fire_color = np.zeros_like(source_pixels)
        fire_color[:, 2] = 255.0
        blended = source_pixels * 0.35 + fire_color * 0.65
        fire_overlay[fire_mask] = np.clip(blended, 0.0, 255.0).astype(np.uint8)

    label_view = np.zeros_like(rgb_bgr)
    label_view[train_mask == BACKGROUND_ID] = (0, 0, 0)
    label_view[train_mask == FIRE_ID] = (0, 0, 255)
    label_view[train_mask == IGNORE_ID] = (128, 128, 128)

    panels = [rgb_bgr, thermal_bgr, thresholds, fire_overlay, label_view]
    labels = ["RGB", "Celsius TIFF", "80C / 200C", "Clean fire overlay", "0/2/255 label"]
    for panel, label in zip(panels, labels):
        cv2.rectangle(panel, (0, 0), (width, 29), (0, 0, 0), -1)
        cv2.putText(
            panel,
            label,
            (8, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    canvas = np.concatenate(panels, axis=1)
    caption = (
        f"{statistics.sample_class}/{statistics.sample_id} | "
        f"Tmax={statistics.max_temperature_c:.1f}C | "
        f"fire={statistics.fire_ratio_before_ignore * 100:.3f}% | "
        f"components={statistics.component_count_after_cleanup}"
    )
    cv2.rectangle(canvas, (0, height - 27), (canvas.shape[1], height), (0, 0, 0), -1)
    cv2.putText(
        canvas,
        caption,
        (8, height - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return canvas


def even_indices(length: int, count: int) -> list[int]:
    if count <= 0 or length <= 0:
        return []
    count = min(count, length)
    return np.linspace(0, length - 1, count, dtype=int).tolist()


def select_review_samples(
    statistics: list[ImageStatistics], fire_count: int, total_count: int
) -> list[ImageStatistics]:
    fire_items = sorted(
        (item for item in statistics if item.sample_class == "Fire"),
        key=lambda item: (item.fire_ratio_before_ignore, item.sample_id),
    )
    no_fire_items = sorted(
        (item for item in statistics if item.sample_class == "No Fire"),
        key=lambda item: item.sample_id,
    )
    no_fire_count = max(total_count - fire_count, 0)
    selected = [fire_items[index] for index in even_indices(len(fire_items), fire_count)]
    selected.extend(
        no_fire_items[index] for index in even_indices(len(no_fire_items), no_fire_count)
    )
    return selected


def write_statistics_csv(statistics: list[ImageStatistics], path: Path) -> None:
    rows = [asdict(item) for item in statistics]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_review_checklist(selected: list[ImageStatistics], path: Path) -> None:
    fixed_fields = list(asdict(selected[0]))
    manual_fields = [
        "rgb_visible_fire_yes_no_uncertain",
        "temperature_region_reasonable_yes_no",
        "hot_ground_false_positive_none_minor_severe",
        "visible_fire_missed_none_minor_severe",
        "registration_issue_none_minor_severe",
        "smoke_present_yes_no_uncertain",
        "overall_accept_yes_no",
        "review_notes",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fixed_fields + manual_fields)
        writer.writeheader()
        for item in selected:
            row = asdict(item)
            row.update({field: "" for field in manual_fields})
            writer.writerow(row)


def make_contact_sheet(review_paths: list[Path], output_path: Path) -> None:
    images: list[Image.Image] = []
    for path in review_paths:
        image = Image.open(path).convert("RGB")
        image.thumbnail((1200, 245), Image.Resampling.LANCZOS)
        images.append(image.copy())
    if not images:
        return
    width = max(image.width for image in images)
    height = sum(image.height for image in images)
    sheet = Image.new("RGB", (width, height), (18, 18, 18))
    y = 0
    for image in images:
        sheet.paste(image, (0, y))
        y += image.height
    sheet.save(output_path, quality=92)


def percentile_summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "minimum": float(array.min()),
        "p10": float(np.percentile(array, 10)),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "maximum": float(array.max()),
    }


def main() -> None:
    args = parse_args()
    if args.low_threshold >= args.high_threshold:
        raise ValueError("low-threshold must be less than high-threshold")
    if args.review_fire_count > args.review_count:
        raise ValueError("review-fire-count cannot exceed review-count")

    args.data_root = args.data_root.resolve()
    args.output = args.output.resolve()
    binary_root = args.output / "fire_binary_masks"
    train_root = args.output / "train_mask_templates"
    review_root = args.output / "manual_review_30"
    review_visual_root = review_root / "visuals"
    for directory in (binary_root, train_root, review_visual_root):
        directory.mkdir(parents=True, exist_ok=True)

    records = collect_pairs(args.data_root)
    statistics: list[ImageStatistics] = []
    cached_review_data: dict[tuple[str, str], tuple[np.ndarray, ...]] = {}

    for class_name, stem, rgb_path, thermal_path in records:
        temperature = np.asarray(Image.open(thermal_path), dtype=np.float32)
        if temperature.ndim != 2 or not np.all(np.isfinite(temperature)):
            raise RuntimeError(f"Invalid temperature TIFF: {thermal_path}")
        rgb_bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if rgb_bgr is None or rgb_bgr.shape[:2] != temperature.shape:
            raise RuntimeError(f"RGB/TIFF shape mismatch: {class_name}/{stem}")

        low_mask = temperature >= args.low_threshold
        high_seed = temperature >= args.high_threshold
        fire_mask, before_count, after_count, removed_small = component_filter(
            low_mask,
            high_seed,
            temperature,
            args.tiny_area,
            args.tiny_peak_keep,
        )
        fire_mask, filled_holes = fill_small_holes(fire_mask, args.max_hole_area)
        train_mask, boundary = build_train_mask(
            fire_mask, high_seed, args.boundary_radius
        )

        class_binary_dir = binary_root / class_name.replace(" ", "_").lower()
        class_train_dir = train_root / class_name.replace(" ", "_").lower()
        class_binary_dir.mkdir(parents=True, exist_ok=True)
        class_train_dir.mkdir(parents=True, exist_ok=True)
        binary_path = class_binary_dir / f"{stem}.png"
        train_path = class_train_dir / f"{stem}.png"
        Image.fromarray((fire_mask.astype(np.uint8) * 255)).save(binary_path)
        Image.fromarray(train_mask).save(train_path)

        pixel_count = int(temperature.size)
        item = ImageStatistics(
            sample_class=class_name,
            sample_id=stem,
            rgb_path=str(rgb_path),
            thermal_path=str(thermal_path),
            height=int(temperature.shape[0]),
            width=int(temperature.shape[1]),
            min_temperature_c=float(np.min(temperature)),
            max_temperature_c=float(np.max(temperature)),
            mean_temperature_c=float(np.mean(temperature)),
            low_threshold_pixels=int(low_mask.sum()),
            high_seed_pixels=int(high_seed.sum()),
            kept_fire_pixels_before_ignore=int(fire_mask.sum()),
            train_fire_pixels=int((train_mask == FIRE_ID).sum()),
            ignore_pixels=int((train_mask == IGNORE_ID).sum()),
            component_count_before_cleanup=before_count,
            component_count_after_cleanup=after_count,
            removed_small_component_count=removed_small,
            filled_hole_count=filled_holes,
            fire_ratio_before_ignore=float(fire_mask.sum() / pixel_count),
            train_fire_ratio=float((train_mask == FIRE_ID).sum() / pixel_count),
            ignore_ratio=float((train_mask == IGNORE_ID).sum() / pixel_count),
            no_fire_hotspot_flag=bool(class_name == "No Fire" and fire_mask.any()),
            output_binary_mask=str(binary_path),
            output_train_mask=str(train_path),
        )
        statistics.append(item)
        cached_review_data[(class_name, stem)] = (
            rgb_bgr,
            temperature,
            low_mask,
            high_seed,
            fire_mask,
            train_mask,
        )

    write_statistics_csv(statistics, args.output / "pseudolabel_statistics.csv")
    selected = select_review_samples(
        statistics, args.review_fire_count, args.review_count
    )
    write_review_checklist(selected, review_root / "manual_review_checklist.csv")
    review_paths: list[Path] = []
    for item in selected:
        data = cached_review_data[(item.sample_class, item.sample_id)]
        visual = render_review_image(item, *data)
        safe_class = item.sample_class.replace(" ", "_").lower()
        visual_path = review_visual_root / f"{safe_class}_{item.sample_id}.jpg"
        if not cv2.imwrite(str(visual_path), visual):
            raise RuntimeError(f"Failed to save review visual: {visual_path}")
        review_paths.append(visual_path)
    make_contact_sheet(review_paths, review_root / "manual_review_contact_sheet.jpg")

    fire_statistics = [item for item in statistics if item.sample_class == "Fire"]
    no_fire_statistics = [item for item in statistics if item.sample_class == "No Fire"]
    fire_without_high_seed = [
        item.sample_id for item in fire_statistics if item.high_seed_pixels == 0
    ]
    fire_with_empty_mask = [
        item.sample_id
        for item in fire_statistics
        if item.kept_fire_pixels_before_ignore == 0
    ]
    script_path = Path(__file__).resolve()
    summary = {
        "status": "awaiting_manual_review",
        "data_root": str(args.data_root),
        "image_count": len(statistics),
        "class_counts": {
            name: sum(item.sample_class == name for item in statistics)
            for name in CLASS_ORDER
        },
        "preregistered_parameters": {
            "low_threshold_c": args.low_threshold,
            "high_threshold_c": args.high_threshold,
            "connectivity": 8,
            "tiny_area_px": args.tiny_area,
            "tiny_peak_keep_c": args.tiny_peak_keep,
            "max_hole_area_px": args.max_hole_area,
            "boundary_radius_px": args.boundary_radius,
            "background_id": BACKGROUND_ID,
            "fire_id": FIRE_ID,
            "ignore_id": IGNORE_ID,
        },
        "fire_max_temperature_c": percentile_summary(
            [item.max_temperature_c for item in fire_statistics]
        ),
        "no_fire_max_temperature_c": percentile_summary(
            [item.max_temperature_c for item in no_fire_statistics]
        ),
        "fire_area_ratio": percentile_summary(
            [item.fire_ratio_before_ignore for item in fire_statistics]
        ),
        "fire_without_200c_seed_count": len(fire_without_high_seed),
        "fire_without_200c_seed_ids": fire_without_high_seed,
        "fire_empty_pseudolabel_count": len(fire_with_empty_mask),
        "fire_empty_pseudolabel_ids": fire_with_empty_mask,
        "no_fire_hotspot_images": [
            item.sample_id for item in no_fire_statistics if item.no_fire_hotspot_flag
        ],
        "review_selection": {
            "total": len(selected),
            "fire": sum(item.sample_class == "Fire" for item in selected),
            "no_fire": sum(item.sample_class == "No Fire" for item in selected),
            "selection_rule": "Fire by evenly spaced pseudo-area rank; No Fire by filename order",
            "seed": args.seed,
        },
        "software": {
            "python": platform.python_version(),
            "opencv": cv2.__version__,
            "numpy": np.__version__,
            "pillow": Image.__version__,
        },
        "script_sha256": sha256_file(script_path),
        "source_images_modified": False,
        "training_authorized": False,
    }
    (args.output / "pseudolabel_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Temperature pseudolabel generation completed: {args.output}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
