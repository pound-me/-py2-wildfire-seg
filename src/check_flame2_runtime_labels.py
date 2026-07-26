from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from baseline_runtime import PROJECT_ROOT, build_dataset, load_config
from datasets.flame2_labels import (
    decode_three_class_label,
    load_three_class_label,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Verify that FLAME2 runtime labels and Fire-component caches use "
            "the same strict three-class decoder."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "pidnet_s_dfm_mproto_p1.yaml",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    config = load_config(args.config.resolve())
    root = Path(config["ROOTDATASET"])
    split_counts: dict[str, int] = {}
    total_samples = 0
    pure_background_fire_samples = 0
    seen_cache_names: dict[str, str] = {}

    for split in ("train", "val", "test"):
        dataset = build_dataset(config, split=split)
        split_counts[split] = len(dataset)
        total_samples += len(dataset)
        for index, template in enumerate(dataset.files):
            _, runtime_label = dataset.load_sample(index)
            label_path = root / template.replace("XXX", "gt")
            strict_label = load_three_class_label(label_path)
            if not np.array_equal(runtime_label, strict_label):
                mismatch = int((runtime_label != strict_label).sum())
                raise RuntimeError(
                    f"Runtime/strict label mismatch for {template}: "
                    f"{mismatch} pixels."
                )

            present = set(np.unique(strict_label).tolist())
            if 2 in present and present.issubset({0, 2}):
                pure_background_fire_samples += 1

            component_map = dataset.load_fire_component_map(index)
            if component_map is None:
                raise RuntimeError(
                    f"Component cache is disabled for runtime check: {template}"
                )
            if component_map.dtype != np.uint16:
                raise TypeError(
                    f"Component cache is not uint16 for {template}: "
                    f"{component_map.dtype}"
                )
            if component_map.shape != strict_label.shape:
                raise RuntimeError(
                    f"Component/label shape mismatch for {template}: "
                    f"{component_map.shape} vs {strict_label.shape}"
                )
            if not np.array_equal(component_map > 0, strict_label == 2):
                mismatch = int(
                    np.logical_xor(component_map > 0, strict_label == 2).sum()
                )
                raise RuntimeError(
                    f"Component/Fire mismatch for {template}: {mismatch} pixels."
                )

            cache_name = Path(template.replace("XXX", "gt")).with_suffix(
                ".npy"
            ).name
            previous = seen_cache_names.get(cache_name)
            if previous is not None and previous != template:
                raise RuntimeError(
                    f"Component-cache basename collision: {previous} and "
                    f"{template} both map to {cache_name}."
                )
            seen_cache_names[cache_name] = template

    all_fire = np.full((4, 5, 3), 255, dtype=np.uint8)
    if not np.all(
        decode_three_class_label(all_fire, source="synthetic all-fire mask") == 2
    ):
        raise RuntimeError("An all-Fire RGB mask did not decode entirely as class 2.")

    unknown_color_raises = False
    try:
        decode_three_class_label(
            np.array([[[1, 2, 3]]], dtype=np.uint8),
            source="synthetic unknown-color mask",
        )
    except ValueError:
        unknown_color_raises = True
    if not unknown_color_raises:
        raise RuntimeError("Unknown FLAME2 colors did not raise ValueError.")

    missing_file_raises = False
    missing_path = root / "__codex_missing_flame2_label__.png"
    try:
        load_three_class_label(missing_path)
    except FileNotFoundError:
        missing_file_raises = True
    if not missing_file_raises:
        raise RuntimeError("A missing FLAME2 label did not raise FileNotFoundError.")

    result = {
        "strict_decoder_shared_by_runtime_and_cache_builder": True,
        "runtime_label_matches_strict_decoder": True,
        "component_id_positive_iff_fire": True,
        "component_dtype": "uint16",
        "cache_basename_collisions": 0,
        "split_counts": split_counts,
        "sample_count": total_samples,
        "pure_background_fire_samples": pure_background_fire_samples,
        "all_fire_decodes_as_class_2": True,
        "unknown_color_raises_value_error": unknown_color_raises,
        "missing_label_raises_file_not_found": missing_file_raises,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
