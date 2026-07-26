from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml

from baseline_runtime import PROJECT_ROOT


DEFAULT_CANDIDATES = [
    "pidnet_s_deconv_d1.yaml",
    "pidnet_s_deconv_d2.yaml",
    "pidnet_s_dfm_mproto_p1.yaml",
    "pidnet_s_dfm_mproto_p2.yaml",
    "pidnet_s_dfm_mproto_p3.yaml",
    "pidnet_s_dfm_mproto_p4.yaml",
    "pidnet_s_deconv_mproto.yaml",
]
PROTOCOL_KEYS = ["BRIGHTNESS", "SCALE_MIN", "SCALE_MAX"]


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def save_yaml(path: Path, config: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Materialize one adopted augmentation protocol consistently."
    )
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, action="append")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    protocol_path = args.protocol.resolve()
    protocol = load_yaml(protocol_path)
    missing = [key for key in PROTOCOL_KEYS if key not in protocol]
    if missing:
        raise KeyError(f"Protocol is missing keys: {missing}")
    digest = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else PROJECT_ROOT / "configs" / f"adopted_{protocol_path.stem}"
    )

    metadata = {
        "ADOPTED_PROTOCOL_SOURCE": str(protocol_path),
        "ADOPTED_PROTOCOL_SHA256": digest,
    }
    protocol_values = {key: protocol[key] for key in PROTOCOL_KEYS}
    written = []

    baseline_path = PROJECT_ROOT / "configs" / "pidnet_s_rgb_baseline.yaml"
    baseline = load_yaml(baseline_path)
    baseline.update(protocol_values)
    baseline.update(metadata)
    baseline["EPOCHS"] = 100
    baseline["LR_TOTAL_EPOCHS"] = 100
    baseline["EXPERIMENT_GROUP"] = "pidnet_s_rgb_baseline_adopted"
    baseline_output = output_dir / "pidnet_s_rgb_baseline_adopted_100e.yaml"
    save_yaml(baseline_output, baseline)
    written.append(str(baseline_output))

    candidate_paths = (
        [path.resolve() for path in args.candidate]
        if args.candidate
        else [PROJECT_ROOT / "configs" / name for name in DEFAULT_CANDIDATES]
    )
    for candidate_path in candidate_paths:
        candidate = load_yaml(candidate_path)
        candidate.update(protocol_values)
        candidate.update(metadata)
        candidate["EPOCHS"] = 30
        candidate["LR_TOTAL_EPOCHS"] = 100
        output = output_dir / candidate_path.name
        save_yaml(output, candidate)
        written.append(str(output))

    manifest = {
        "protocol": str(protocol_path),
        "protocol_sha256": digest,
        "protocol_values": protocol_values,
        "written_configs": written,
        "rule": (
            "Run the adopted 100-epoch baseline first, recompute its 26-30 "
            "conservative Fire noise band, then compare every candidate only "
            "against that baseline."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
