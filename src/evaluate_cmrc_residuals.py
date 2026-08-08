from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from baseline_runtime import PROJECT_ROOT, build_dataset, build_model, load_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure CMRC residual statistics on the validation split only."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=(
            PROJECT_ROOT
            / "configs"
            / "flame3"
            / "pidnet_s_cmrc_partial_30e.yaml"
        ),
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--root-dataset", type=Path, required=True)
    parser.add_argument("--validation-csv", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for CMRC residual diagnostics.")
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive.")

    config = load_config(args.config.resolve())
    if config.get("MODEL") != "pidnet_s_cmrc" or config.get("MODE") != "fusion":
        raise ValueError("Residual diagnostics require the Fusion CMRC config.")
    config["ROOTDATASET"] = str(args.root_dataset.resolve())
    config["VALIDSET"] = str(args.validation_csv.resolve())
    config["BATCHSIZE"] = int(args.batch_size)
    config["NUM_WORKERS"] = 0
    dataset = build_dataset(config, "val")
    loader = DataLoader(
        dataset,
        batch_size=int(config["BATCHSIZE"]),
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    device = torch.device(config["DEVICE"])
    model = build_model(config, augment=False)
    checkpoint = torch.load(
        args.checkpoint.resolve(),
        map_location="cpu",
        weights_only=False,
    )
    state = checkpoint.get("model_state_dict", checkpoint)
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.missing_keys:
        raise RuntimeError(f"Missing CMRC checkpoint keys: {incompatible.missing_keys}")
    model = model.to(device).eval()

    image_count = 0
    weighted_abs_mean = 0.0
    weighted_saturation = 0.0
    maximum_abs = 0.0
    with torch.inference_mode():
        for batch in loader:
            images = batch[0].to(
                device=device,
                dtype=torch.float32,
                non_blocking=True,
            )
            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=True,
            ):
                _outputs, aux = model(images, return_aux=True)
            batch_size = int(images.shape[0])
            image_count += batch_size
            weighted_abs_mean += float(aux["residual_abs_mean"].cpu()) * batch_size
            weighted_saturation += (
                float(aux["residual_saturation_ratio"].cpu()) * batch_size
            )
            maximum_abs = max(maximum_abs, float(aux["residual_abs_max"].cpu()))

    denominator = max(image_count, 1)
    result = {
        "protocol": "validation_only_cmrc_residual_diagnostics",
        "config": str(args.config.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch": checkpoint.get("epoch") if isinstance(checkpoint, dict) else None,
        "validation_csv": str(args.validation_csv.resolve()),
        "validation_images": image_count,
        "residual_abs_mean": weighted_abs_mean / denominator,
        "residual_abs_max": maximum_abs,
        "residual_saturation_ratio": weighted_saturation / denominator,
        "residual_limit": float(config.get("CMRC_RESIDUAL_LIMIT", 0.1)),
        "test_images_or_labels_read": False,
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
