from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from baseline_runtime import (
    build_model,
    load_config,
    load_pretrained_if_available,
    seed_everything,
)


EXPECTED_CHANNELS = {"rgb": 3, "ir": 1, "fusion": 4}
STEM_KEY = "conv1.0.weight"


def checkpoint_state(path: Path) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    if "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    return checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, action="append", required=True)
    parser.add_argument("--pretrained", type=Path, required=True)
    args = parser.parse_args()
    pretrained = args.pretrained.resolve()
    source_state = checkpoint_state(pretrained)
    if STEM_KEY not in source_state:
        raise KeyError(f"Official checkpoint is missing {STEM_KEY}")

    results = []
    for config_path in args.config:
        config = load_config(config_path.resolve())
        mode = str(config["MODE"]).lower()
        if mode not in EXPECTED_CHANNELS:
            raise ValueError(f"Unexpected mode in {config_path}: {mode}")
        if config.get("PRETRAIN_SKIP_KEYS") != [STEM_KEY]:
            raise RuntimeError(
                f"{config_path} must explicitly skip only {STEM_KEY}: "
                f"{config.get('PRETRAIN_SKIP_KEYS')}"
            )
        config["PRETRAINED"] = str(pretrained)
        seed_everything(int(config["SEED"]))
        model = build_model(config)
        stem_before = model.state_dict()[STEM_KEY].detach().clone()
        if stem_before.shape[1] != EXPECTED_CHANNELS[mode]:
            raise RuntimeError(
                f"{mode} stem channels mismatch: {tuple(stem_before.shape)}"
            )
        matched = load_pretrained_if_available(model, config)
        stem_after = model.state_dict()[STEM_KEY].detach()
        if not torch.equal(stem_before, stem_after):
            raise RuntimeError(f"{mode} stem changed during pretrained loading")
        if matched != 301:
            raise RuntimeError(f"{mode} expected 301 matched tensors, got {matched}")
        loaded_key = next(
            key
            for key, value in source_state.items()
            if not key.startswith("conv1.0.")
            and key in model.state_dict()
            and value.shape == model.state_dict()[key].shape
        )
        if not torch.equal(model.state_dict()[loaded_key], source_state[loaded_key]):
            raise RuntimeError(f"{mode} backbone tensor was not loaded: {loaded_key}")
        if not bool(torch.isfinite(stem_after).all()) or float(stem_after.std()) <= 0.0:
            raise RuntimeError(f"{mode} stem initialization is invalid")
        results.append(
            {
                "mode": mode,
                "stem_shape": list(stem_after.shape),
                "stem_mean": float(stem_after.mean()),
                "stem_std": float(stem_after.std()),
                "matched_pretrained_tensors": matched,
                "explicit_skip": STEM_KEY,
                "backbone_evidence_key": loaded_key,
                "test_images_or_labels_read": False,
            }
        )
    if {result["mode"] for result in results} != set(EXPECTED_CHANNELS):
        raise RuntimeError("Fair-stem check requires rgb, ir, and fusion configs")
    print("FLAME3 fair input-stem initialization check passed")
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
