from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from baseline_runtime import PROJECT_ROOT, load_config
from datasets.flame2_labels import load_three_class_label


def component_cache_name(relative_sample: str) -> str:
    gt_name = Path(relative_sample.replace("XXX", "gt")).name
    return Path(gt_name).with_suffix(".npy").name


def source_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build uint16 8-connected fire-component maps for FLAME2."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "pidnet_s_dfm_mproto_p3.yaml",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config.resolve())
    root = Path(config["ROOTDATASET"])
    output = (
        args.output.resolve()
        if args.output
        else Path(config["FIRE_COMPONENT_CACHE"])
    )
    output.mkdir(parents=True, exist_ok=True)

    relative_samples: list[str] = []
    seen: set[str] = set()
    for key in ("TRAINSET", "VALIDSET", "TESTSET"):
        list_path = root / config[key]
        for line in list_path.read_text(encoding="utf-8").splitlines():
            sample = line.strip()
            if sample and sample not in seen:
                seen.add(sample)
                relative_samples.append(sample)

    maximum_components = 0
    total_components = 0
    fire_images = 0
    records = []
    for index, relative_sample in enumerate(relative_samples, start=1):
        label_path = root / relative_sample.replace("XXX", "gt")
        cache_path = output / component_cache_name(relative_sample)
        label = load_three_class_label(label_path)
        fire = (label == 2).astype(np.uint8)
        component_count, component_map = cv2.connectedComponents(
            fire,
            connectivity=8,
            ltype=cv2.CV_16U,
        )
        foreground_components = int(component_count - 1)
        if foreground_components > np.iinfo(np.uint16).max:
            raise OverflowError(
                f"Too many components for uint16 in {label_path}: "
                f"{foreground_components}"
            )
        component_map = component_map.astype(np.uint16, copy=False)
        if not np.array_equal(component_map > 0, label == 2):
            raise RuntimeError(f"Component/label mismatch: {label_path}")
        if cache_path.exists() and not args.overwrite:
            existing = np.load(cache_path, allow_pickle=False)
            if existing.dtype != np.uint16 or not np.array_equal(
                existing,
                component_map,
            ):
                raise RuntimeError(
                    f"Existing cache differs; rerun with --overwrite: {cache_path}"
                )
        else:
            np.save(cache_path, component_map, allow_pickle=False)

        maximum_components = max(maximum_components, foreground_components)
        total_components += foreground_components
        fire_images += int(foreground_components > 0)
        records.append(
            {
                "sample": relative_sample,
                "label": str(label_path),
                "cache": str(cache_path),
                "components": foreground_components,
                "fire_pixels": int(fire.sum()),
                "label_sha256": source_digest(label_path),
            }
        )
        if index == 1 or index % 100 == 0 or index == len(relative_samples):
            print(f"Processed {index}/{len(relative_samples)} labels")

    manifest = {
        "connectivity": 8,
        "dtype": "uint16",
        "fire_class": 2,
        "sample_count": len(relative_samples),
        "fire_image_count": fire_images,
        "total_components": total_components,
        "maximum_components_per_image": maximum_components,
        "invariant": "component_id > 0 iff label == fire",
        "records": records,
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("Fire component cache check passed")
    print(f"Samples: {len(relative_samples)}")
    print(f"Maximum components per image: {maximum_components}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
