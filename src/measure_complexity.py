from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

from baseline_runtime import PROJECT_ROOT, build_model, load_config
from custom_models.pidnet_deconv import (
    count_deconv_modules,
    reparameterize_deconv_model,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure PIDNet inference parameters and forward FLOPs."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "pidnet_s_rgb_baseline.yaml",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=(
            PROJECT_ROOT
            / "experiments"
            / "pidnet_s_rgb_baseline"
            / "baseline_100e"
            / "best.pth"
        ),
    )
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--warmup-iterations", type=int, default=10)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable.")
    config = load_config(args.config.resolve())
    input_channels = {"rgb": 3, "ir": 1, "fusion": 4}
    mode = str(config["MODE"]).lower()
    if mode not in input_channels:
        raise ValueError(f"Unsupported input mode: {config['MODE']}")
    checkpoint_path = args.checkpoint.resolve()
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    if "model_state_dict" not in checkpoint:
        raise KeyError("The checkpoint does not contain model_state_dict.")

    device = torch.device(config["DEVICE"])
    training_model = build_model(config, augment=True)
    training_parameters = sum(
        parameter.numel() for parameter in training_model.parameters()
    )
    del training_model

    model = build_model(config, augment=False)
    incompatible = model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=False,
    )
    if incompatible.missing_keys:
        raise RuntimeError(
            "Missing inference-model weights: "
            + ", ".join(incompatible.missing_keys)
        )
    deconv_modules_before = count_deconv_modules(model)
    if deconv_modules_before:
        model = reparameterize_deconv_model(model, inplace=True)
    model = model.to(device)
    model.eval()
    inference_parameters = sum(
        parameter.numel() for parameter in model.parameters()
    )
    sample = torch.randn(
        1,
        input_channels[mode],
        args.height,
        args.width,
        device=device,
    )

    with torch.inference_mode():
        for _ in range(max(args.warmup_iterations, 0)):
            model(sample)
        torch.cuda.synchronize(device)
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            with_flops=True,
        ) as profiler:
            model(sample)
        torch.cuda.synchronize(device)

    operator_flops = {
        event.key: int(event.flops)
        for event in profiler.key_averages()
        if event.flops
    }
    total_flops = sum(operator_flops.values())
    results = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "model": config["MODEL"],
        "input_shape": [1, input_channels[mode], args.height, args.width],
        "training_parameters_with_auxiliary_heads": training_parameters,
        "inference_parameters_main_head": inference_parameters,
        "forward_flops": total_flops,
        "forward_gflops": total_flops / 1_000_000_000,
        "estimated_macs": total_flops / 2,
        "estimated_gmacs": total_flops / 2_000_000_000,
        "operator_flops": operator_flops,
        "deconv_modules_before_reparameterization": deconv_modules_before,
        "deconv_modules_in_deployment_model": count_deconv_modules(model),
        "counting_method": (
            "torch.profiler with_flops=True; multiply and add are counted "
            "as separate floating-point operations"
        ),
        "scope": "deployment model forward, main segmentation output only",
    }
    output_path = (
        args.output.resolve()
        if args.output
        else checkpoint_path.parent / "test_best" / "complexity.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        json.dump(results, stream, ensure_ascii=False, indent=2)

    print(f"Inference parameters: {inference_parameters / 1_000_000:.3f}M")
    print(f"Forward FLOPs: {total_flops / 1_000_000_000:.4f} GFLOPs")
    print(f"Estimated MACs: {total_flops / 2_000_000_000:.4f} GMACs")
    print(f"Result: {output_path}")


if __name__ == "__main__":
    main()
