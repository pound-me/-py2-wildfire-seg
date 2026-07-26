from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from baseline_runtime import (
    PROJECT_ROOT,
    build_dataset,
    build_model,
    load_config,
    load_pretrained_if_available,
    seed_everything,
)
from custom_models.pidnet_deconv import (
    count_deconv_modules,
    reparameterize_deconv_model,
)


def load_weights(model: torch.nn.Module, config: dict, checkpoint: Path | None) -> str:
    if checkpoint is None:
        load_pretrained_if_available(model, config)
        return str(config["PRETRAINED"])
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model_state_dict"], strict=False)
    return str(checkpoint)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify DEConv FP32 deployment reparameterization."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "pidnet_s_deconv_d1.yaml",
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--max-val-batches", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    config = load_config(args.config.resolve())
    device = torch.device(config["DEVICE"] if torch.cuda.is_available() else "cpu")
    seed_everything(int(config["SEED"]))
    model = build_model(config, augment=False)
    source = load_weights(
        model,
        config,
        args.checkpoint.resolve() if args.checkpoint else None,
    )
    model = model.to(device).eval().float()
    deconv_count = count_deconv_modules(model)
    if deconv_count == 0:
        raise RuntimeError("The selected model does not contain DEConv modules.")
    deployed = reparameterize_deconv_model(model, inplace=False).eval().float()
    if count_deconv_modules(deployed) != 0:
        raise RuntimeError("Deployment model still contains DEConv modules.")

    sample = torch.randn(1, 3, 256, 256, device=device, dtype=torch.float32)
    with torch.inference_mode():
        before = model(sample)
        after = deployed(sample)
    maximum_error = float((before - after).abs().max().cpu())
    random_prediction_equal = torch.equal(before.argmax(1), after.argmax(1))

    initial_baseline_error = None
    initial_baseline_prediction_equal = None
    initial_baseline_matched_tensors = None
    initial_training_outputs_error = None
    initial_training_output_count = None
    initial_training_segmentation_output_count = None
    initial_training_feature_output_count = None
    if args.checkpoint is None:
        baseline_config = copy.deepcopy(config)
        baseline_config["MODEL"] = "pidnet_s"
        seed_everything(int(config["SEED"]))
        initial_baseline = build_model(baseline_config, augment=False)
        initial_baseline_matched_tensors = load_pretrained_if_available(
            initial_baseline,
            baseline_config,
        )
        initial_baseline = initial_baseline.to(device).eval().float()
        with torch.inference_mode():
            initial_baseline_output = initial_baseline(sample)
        initial_baseline_error = float(
            (before - initial_baseline_output).abs().max().cpu()
        )
        initial_baseline_prediction_equal = torch.equal(
            before.argmax(1),
            initial_baseline_output.argmax(1),
        )
        if initial_baseline_error >= 1e-7:
            raise RuntimeError(
                "Zero-initialized DEConv is not initially equivalent to "
                f"PIDNet-S: maximum error {initial_baseline_error}"
            )
        if not initial_baseline_prediction_equal:
            raise RuntimeError(
                "Zero-initialized DEConv changed initial pixel predictions."
            )

        seed_everything(int(config["SEED"]))
        initial_deconv_training = build_model(config, augment=True)
        load_pretrained_if_available(initial_deconv_training, config)
        initial_deconv_training = initial_deconv_training.to(device).eval().float()
        seed_everything(int(config["SEED"]))
        initial_baseline_training = build_model(
            baseline_config,
            augment=True,
        )
        load_pretrained_if_available(
            initial_baseline_training,
            baseline_config,
        )
        initial_baseline_training = (
            initial_baseline_training.to(device).eval().float()
        )
        with torch.inference_mode():
            deconv_training_outputs = initial_deconv_training(sample)
            baseline_training_outputs = initial_baseline_training(sample)
        if not isinstance(deconv_training_outputs, (list, tuple)) or not isinstance(
            baseline_training_outputs,
            (list, tuple),
        ):
            raise RuntimeError("Initial training models did not return auxiliary outputs.")
        initial_training_output_count = len(deconv_training_outputs)
        initial_training_segmentation_output_count = len(baseline_training_outputs)
        initial_training_feature_output_count = (
            initial_training_output_count
            - initial_training_segmentation_output_count
        )
        expected_feature_outputs = int(
            config["MODEL"] in {"pidnet_s_dfm_mproto", "pidnet_s_deconv_mproto"}
        )
        if initial_training_feature_output_count != expected_feature_outputs:
            raise RuntimeError(
                "Unexpected initial training-only feature output count: "
                f"{initial_training_feature_output_count} vs "
                f"{expected_feature_outputs}."
            )
        initial_training_outputs_error = max(
            float((left - right).abs().max().cpu())
            for left, right in zip(
                deconv_training_outputs[:initial_training_segmentation_output_count],
                baseline_training_outputs,
            )
        )
        if initial_training_outputs_error >= 1e-7:
            raise RuntimeError(
                "Zero-initialized DEConv training outputs differ from "
                f"PIDNet-S: maximum error {initial_training_outputs_error}"
            )

    validation_dataset = build_dataset(config, split="val")
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=int(config["BATCHSIZE"]),
        shuffle=False,
        num_workers=0,
    )
    mismatched_pixels = 0
    compared_pixels = 0
    with torch.inference_mode():
        for batch_index, batch in enumerate(validation_loader):
            if args.max_val_batches is not None and batch_index >= args.max_val_batches:
                break
            images = batch[0].to(device=device, dtype=torch.float32)
            prediction_before = model(images).argmax(dim=1)
            prediction_after = deployed(images).argmax(dim=1)
            mismatched_pixels += int(
                (prediction_before != prediction_after).sum().cpu()
            )
            compared_pixels += prediction_before.numel()

    baseline_config = copy.deepcopy(config)
    baseline_config["MODEL"] = "pidnet_s"
    baseline = build_model(baseline_config, augment=False)
    baseline_parameters = sum(parameter.numel() for parameter in baseline.parameters())
    deployment_parameters = sum(
        parameter.numel() for parameter in deployed.parameters()
    )
    if maximum_error >= 1e-5:
        raise RuntimeError(f"Reparameterization error too large: {maximum_error}")
    if not random_prediction_equal or mismatched_pixels != 0:
        raise RuntimeError("DEConv deployment changed pixel predictions.")
    if deployment_parameters != baseline_parameters:
        raise RuntimeError(
            "Deployment parameter count differs from baseline: "
            f"{deployment_parameters} vs {baseline_parameters}"
        )

    result = {
        "config": str(args.config.resolve()),
        "weight_source": source,
        "deconv_modules_before": deconv_count,
        "deconv_modules_after": count_deconv_modules(deployed),
        "fp32_maximum_absolute_error": maximum_error,
        "threshold": 1e-5,
        "random_prediction_equal": random_prediction_equal,
        "initial_baseline_fp32_maximum_absolute_error": (
            initial_baseline_error
        ),
        "initial_baseline_prediction_equal": (
            initial_baseline_prediction_equal
        ),
        "initial_baseline_matched_pretrained_tensors": (
            initial_baseline_matched_tensors
        ),
        "initial_training_output_count": initial_training_output_count,
        "initial_training_segmentation_output_count": (
            initial_training_segmentation_output_count
        ),
        "initial_training_feature_output_count": (
            initial_training_feature_output_count
        ),
        "initial_training_outputs_fp32_maximum_absolute_error": (
            initial_training_outputs_error
        ),
        "validation_pixels_compared": compared_pixels,
        "validation_mismatched_pixels": mismatched_pixels,
        "deployment_parameters": deployment_parameters,
        "baseline_parameters": baseline_parameters,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.output:
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print("DEConv reparameterization check passed")


if __name__ == "__main__":
    main()
