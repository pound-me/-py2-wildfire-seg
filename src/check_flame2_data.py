from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np


SPLIT_FILES = (
    "lists/train_flm.txt",
    "lists/val_flm.txt",
    "lists/test_flm.txt",
)
CLASS_NAMES = {
    0: "background",
    1: "smoke",
    2: "fire",
}
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROBOFIRE_ROOT = PROJECT_ROOT / "third_party" / "RoboFireFuseNet"
sys.path.insert(0, str(ROBOFIRE_ROOT))

from datasets.flame2_labels import load_three_class_label  # noqa: E402


def resolve_triplet(root: Path, template: str) -> tuple[Path, Path, Path]:
    return (
        root / template.replace("XXX", "rgb"),
        root / template.replace("XXX", "ir"),
        root / template.replace("XXX", "gt"),
    )


def label_counts(mask_path: Path) -> Counter[int]:
    values = load_three_class_label(mask_path)
    unique, counts = np.unique(values, return_counts=True)
    return Counter(dict(zip(unique.astype(int), counts.astype(int))))


def check_split(root: Path, split_file: str, sample_masks: int) -> bool:
    list_path = root / split_file
    if not list_path.is_file():
        print(f"[missing] {list_path}")
        return False

    templates = [
        line.strip()
        for line in list_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    masks_to_inspect = len(templates) if sample_masks <= 0 else sample_masks
    missing_rgb = 0
    missing_ir = 0
    missing_gt = 0
    class_pixels: Counter[int] = Counter()

    for index, template in enumerate(templates):
        rgb_path, ir_path, gt_path = resolve_triplet(root, template)
        missing_rgb += not rgb_path.is_file()
        missing_ir += not ir_path.is_file()
        missing_gt += not gt_path.is_file()
        if index < masks_to_inspect and gt_path.is_file():
            class_pixels.update(label_counts(gt_path))

    class_summary = ", ".join(
        f"{CLASS_NAMES.get(class_id, f'unknown-{class_id}')}={count}"
        for class_id, count in sorted(class_pixels.items())
    )
    print(
        f"{split_file}: samples={len(templates)}, "
        f"missing_rgb={missing_rgb}, missing_ir={missing_ir}, missing_gt={missing_gt}"
    )
    print(f"  sampled mask pixels: {class_summary or 'none'}")
    return bool(templates) and missing_rgb == 0 and missing_gt == 0


def discover_dataset_root(requested_root: Path) -> Path:
    candidates = (
        requested_root,
        PROJECT_ROOT / "data",
        requested_root.parent,
    )
    for candidate in candidates:
        candidate = candidate.resolve()
        if all((candidate / split_file).is_file() for split_file in SPLIT_FILES):
            return candidate
    checked = "\n".join(f"  - {candidate.resolve()}" for candidate in candidates)
    raise FileNotFoundError(
        "Could not find the FLAME2 split lists. Checked:\n" + checked
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate the RoboFireFuseNet FLAME2 subset for RGB training."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=PROJECT_ROOT / "data",
    )
    parser.add_argument(
        "--sample-masks",
        type=int,
        default=0,
        help="Maximum masks inspected per split; 0 checks every mask.",
    )
    args = parser.parse_args()
    root = discover_dataset_root(args.root)

    print(f"Dataset root: {root}")
    results = [
        check_split(root, split_file, args.sample_masks)
        for split_file in SPLIT_FILES
    ]
    if not all(results):
        raise SystemExit(
            "Dataset check failed. RGB images, masks, or split lists are incomplete."
        )

    print("FLAME2 RGB three-class dataset check passed")


if __name__ == "__main__":
    main()
