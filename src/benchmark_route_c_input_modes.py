from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from baseline_runtime import load_config, seed_everything
from benchmark_paired_deployment import load_inference_model, timed_forward, warmup


MODE_CHANNELS = {"rgb": 3, "ir": 1, "fusion": 4}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark the formal RGB-only, IR-only and Fusion PIDNet-S models "
            "in one RTX 2060 process with rotating measurement order."
        )
    )
    for mode in MODE_CHANNELS:
        parser.add_argument(f"--{mode}-config", type=Path, required=True)
        parser.add_argument(f"--{mode}-checkpoint", type=Path, required=True)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--warmup-iterations", type=int, default=100)
    parser.add_argument("--timed-iterations", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=12)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Route C latency measurement.")
    device = torch.device("cuda:0")
    configs: dict[str, dict] = {}
    config_paths: dict[str, Path] = {}
    checkpoint_paths: dict[str, Path] = {}
    models: dict[str, torch.nn.Module] = {}
    samples: dict[str, torch.Tensor] = {}

    for mode in MODE_CHANNELS:
        config_path = getattr(args, f"{mode}_config").resolve()
        checkpoint_path = getattr(args, f"{mode}_checkpoint").resolve()
        config = load_config(config_path)
        if str(config["MODE"]).lower() != mode:
            raise ValueError(
                f"Expected MODE: {mode} in {config_path}, got {config['MODE']}"
            )
        if str(config["DEVICE"]).lower() != "cuda:0":
            raise ValueError(f"Expected DEVICE: cuda:0 in {config_path}")
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        configs[mode] = config
        config_paths[mode] = config_path
        checkpoint_paths[mode] = checkpoint_path

    seeds = {int(config["SEED"]) for config in configs.values()}
    if len(seeds) != 1:
        raise RuntimeError(f"Input-mode configs use different seeds: {sorted(seeds)}")
    seed_everything(seeds.pop())

    for mode, channels in MODE_CHANNELS.items():
        config = configs[mode]
        checkpoint_path = checkpoint_paths[mode]
        models[mode] = load_inference_model(config, checkpoint_path, device)
        samples[mode] = torch.randn(
            1,
            channels,
            max(int(args.height), 1),
            max(int(args.width), 1),
            device=device,
            dtype=torch.float32,
        )
    warmup_iterations = max(int(args.warmup_iterations), 0)
    timed_iterations = max(int(args.timed_iterations), 1)
    repeats = max(int(args.repeats), 1)
    modes = list(MODE_CHANNELS)

    for mode in modes:
        warmup(
            models[mode],
            samples[mode],
            warmup_iterations,
            args.amp,
        )

    trials: dict[str, list[float]] = {mode: [] for mode in modes}
    trial_orders: list[list[str]] = []
    for repeat_index in range(repeats):
        offset = repeat_index % len(modes)
        order = modes[offset:] + modes[:offset]
        trial_orders.append(order)
        for mode in order:
            trials[mode].append(
                timed_forward(
                    models[mode],
                    samples[mode],
                    timed_iterations,
                    args.amp,
                )
            )

    model_results = {}
    for mode in modes:
        values = np.asarray(trials[mode], dtype=np.float64)
        median = float(np.median(values))
        model_results[mode] = {
            "config": str(config_paths[mode]),
            "config_sha256": sha256(config_paths[mode]),
            "checkpoint": str(checkpoint_paths[mode]),
            "input_shape": list(samples[mode].shape),
            "deployment_parameters": sum(
                parameter.numel() for parameter in models[mode].parameters()
            ),
            "latency_ms_trials": values.tolist(),
            "latency_ms_median": median,
            "latency_ms_mean": float(values.mean()),
            "latency_ms_std": float(values.std()),
            "latency_ms_p95": float(np.percentile(values, 95)),
            "fps_from_median": 1000.0 / median,
            "passes_30_fps": median <= 1000.0 / 30.0,
        }

    result = {
        "protocol": (
            "single-process same-session rotating-order model-forward benchmark; "
            "warm-up precedes repeated median latency"
        ),
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256(Path(__file__).resolve()),
        "device": torch.cuda.get_device_name(device),
        "torch_cuda_version": torch.version.cuda,
        "torch_version": torch.__version__,
        "cudnn_version": torch.backends.cudnn.version(),
        "amp": bool(args.amp),
        "warmup_iterations_per_model": warmup_iterations,
        "timed_iterations_per_trial": timed_iterations,
        "repeats": repeats,
        "trial_orders": trial_orders,
        "real_time_threshold": {
            "minimum_fps": 30.0,
            "maximum_median_latency_ms": 1000.0 / 30.0,
        },
        "models": model_results,
        "all_models_pass_30_fps": all(
            values["passes_30_fps"] for values in model_results.values()
        ),
        "scope": "model forward only; batch size 1; validation metrics are stored separately",
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
