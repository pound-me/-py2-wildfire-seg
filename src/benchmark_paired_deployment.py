from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from baseline_runtime import build_model, load_config, seed_everything
from custom_models.pidnet_deconv import (
    count_deconv_modules,
    reparameterize_deconv_model,
)


def load_inference_model(
    config: dict,
    checkpoint_path: Path | None,
    device: torch.device,
) -> torch.nn.Module:
    model = build_model(config, augment=False)
    if checkpoint_path is not None:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model = model.to(device).eval().float()
    if count_deconv_modules(model) > 0:
        model = reparameterize_deconv_model(model, inplace=False).to(device)
        model = model.eval().float()
    if count_deconv_modules(model) != 0:
        raise RuntimeError("Deployment model still contains DEConv modules.")
    return model


def timed_forward(
    model: torch.nn.Module,
    sample: torch.Tensor,
    iterations: int,
    amp: bool,
) -> float:
    torch.cuda.synchronize(sample.device)
    start = time.perf_counter()
    with torch.inference_mode():
        for _ in range(iterations):
            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=amp,
            ):
                model(sample)
    torch.cuda.synchronize(sample.device)
    return (time.perf_counter() - start) * 1000.0 / iterations


def warmup(
    model: torch.nn.Module,
    sample: torch.Tensor,
    iterations: int,
    amp: bool,
) -> None:
    with torch.inference_mode():
        for _ in range(iterations):
            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=amp,
            ):
                model(sample)
    torch.cuda.synchronize(sample.device)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark baseline and a fused candidate back-to-back in one "
            "process to control GPU clock and system-load variation."
        )
    )
    parser.add_argument("--baseline-config", type=Path, required=True)
    parser.add_argument("--baseline-checkpoint", type=Path)
    parser.add_argument("--candidate-config", type=Path, required=True)
    parser.add_argument("--candidate-checkpoint", type=Path)
    parser.add_argument(
        "--architecture-only",
        action="store_true",
        help=(
            "Benchmark randomly initialized architectures without loading "
            "checkpoints. Use only for pre-training deployment admission."
        ),
    )
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--warmup-iterations", type=int, default=100)
    parser.add_argument("--timed-iterations", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for deployment-speed measurement.")
    if args.architecture_only:
        if args.baseline_checkpoint or args.candidate_checkpoint:
            raise ValueError(
                "Do not provide checkpoints together with --architecture-only."
            )
    elif not args.baseline_checkpoint or not args.candidate_checkpoint:
        raise ValueError(
            "Both checkpoints are required unless --architecture-only is used."
        )
    baseline_config = load_config(args.baseline_config.resolve())
    candidate_config = load_config(args.candidate_config.resolve())
    if baseline_config["DEVICE"] != candidate_config["DEVICE"]:
        raise RuntimeError("Baseline and candidate configs use different devices.")
    input_channels = {"rgb": 3, "ir": 1, "fusion": 4}
    baseline_mode = str(baseline_config["MODE"]).lower()
    candidate_mode = str(candidate_config["MODE"]).lower()
    if baseline_mode not in input_channels or candidate_mode not in input_channels:
        raise ValueError(
            f"Unsupported paired modes: {baseline_config['MODE']} / "
            f"{candidate_config['MODE']}"
        )
    if input_channels[baseline_mode] != input_channels[candidate_mode]:
        raise RuntimeError(
            "Paired deployment timing requires equal input channel counts; "
            f"got {baseline_mode} and {candidate_mode}."
        )
    device = torch.device(candidate_config["DEVICE"])
    seed_everything(int(candidate_config["SEED"]))

    baseline = load_inference_model(
        baseline_config,
        (
            None
            if args.architecture_only
            else args.baseline_checkpoint.resolve()
        ),
        device,
    )
    candidate = load_inference_model(
        candidate_config,
        (
            None
            if args.architecture_only
            else args.candidate_checkpoint.resolve()
        ),
        device,
    )
    baseline_parameters = sum(parameter.numel() for parameter in baseline.parameters())
    candidate_parameters = sum(parameter.numel() for parameter in candidate.parameters())

    sample = torch.randn(
        1,
        input_channels[candidate_mode],
        args.height,
        args.width,
        device=device,
        dtype=torch.float32,
    )
    warmup_iterations = max(args.warmup_iterations, 0)
    timed_iterations = max(args.timed_iterations, 1)
    repeats = max(args.repeats, 1)
    warmup(baseline, sample, warmup_iterations, args.amp)
    warmup(candidate, sample, warmup_iterations, args.amp)

    baseline_trials: list[float] = []
    candidate_trials: list[float] = []
    trial_orders: list[list[str]] = []
    for repeat_index in range(repeats):
        order = (
            (("baseline", baseline), ("candidate", candidate))
            if repeat_index % 2 == 0
            else (("candidate", candidate), ("baseline", baseline))
        )
        trial_orders.append([name for name, _ in order])
        for name, model in order:
            latency = timed_forward(
                model,
                sample,
                timed_iterations,
                args.amp,
            )
            if name == "baseline":
                baseline_trials.append(latency)
            else:
                candidate_trials.append(latency)

    baseline_median = float(np.median(baseline_trials))
    candidate_median = float(np.median(candidate_trials))
    relative_change = candidate_median / baseline_median - 1.0
    result = {
        "device": torch.cuda.get_device_name(device),
        "input_shape": list(sample.shape),
        "weights_source": (
            "architecture_only_random_initialization"
            if args.architecture_only
            else "checkpoints"
        ),
        "amp": args.amp,
        "warmup_iterations_per_model": warmup_iterations,
        "timed_iterations_per_trial": timed_iterations,
        "repeats": repeats,
        "trial_orders": trial_orders,
        "baseline": {
            "config": str(args.baseline_config.resolve()),
            "checkpoint": (
                None
                if args.architecture_only
                else str(args.baseline_checkpoint.resolve())
            ),
            "deployment_parameters": baseline_parameters,
            "latency_ms_trials": baseline_trials,
            "latency_ms_median": baseline_median,
            "latency_ms_mean": float(np.mean(baseline_trials)),
            "latency_ms_std": float(np.std(baseline_trials)),
            "fps_from_median": 1000.0 / baseline_median,
        },
        "candidate": {
            "config": str(args.candidate_config.resolve()),
            "checkpoint": (
                None
                if args.architecture_only
                else str(args.candidate_checkpoint.resolve())
            ),
            "deployment_parameters": candidate_parameters,
            "latency_ms_trials": candidate_trials,
            "latency_ms_median": candidate_median,
            "latency_ms_mean": float(np.mean(candidate_trials)),
            "latency_ms_std": float(np.std(candidate_trials)),
            "fps_from_median": 1000.0 / candidate_median,
        },
        "candidate_latency_relative_change": relative_change,
        "candidate_latency_percent_change": relative_change * 100.0,
        "candidate_parameter_relative_change": (
            candidate_parameters / baseline_parameters - 1.0
        ),
        "speed_drop_at_most_3_percent": relative_change <= 0.03,
        "real_time_minimum_fps": 30.0,
        "real_time_maximum_median_latency_ms": 1000.0 / 30.0,
        "baseline_passes_30_fps": baseline_median <= 1000.0 / 30.0,
        "candidate_passes_30_fps": candidate_median <= 1000.0 / 30.0,
        "route_c_admission_rule": (
            "candidate median FPS >= 30 in paired RTX 2060 measurement"
        ),
        "route_c_candidate_admitted": candidate_median <= 1000.0 / 30.0,
        "deployment_parameter_counts_equal": (
            baseline_parameters == candidate_parameters
        ),
        "scope": "paired model-forward benchmark in one process",
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
