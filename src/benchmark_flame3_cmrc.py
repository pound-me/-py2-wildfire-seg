from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from baseline_runtime import PROJECT_ROOT, build_model, load_config, seed_everything


def load_common_baseline_state(
    baseline: torch.nn.Module,
    candidate: torch.nn.Module,
) -> int:
    source = baseline.state_dict()
    target = candidate.state_dict()
    common = {
        key: value
        for key, value in source.items()
        if key in target and target[key].shape == value.shape
    }
    target.update(common)
    candidate.load_state_dict(target, strict=True)
    return len(common)


@torch.inference_mode()
def warmup(
    model: torch.nn.Module,
    sample: torch.Tensor,
    iterations: int,
) -> None:
    for _ in range(iterations):
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
            model(sample)
    torch.cuda.synchronize(sample.device)


@torch.inference_mode()
def timed_forward(
    model: torch.nn.Module,
    sample: torch.Tensor,
    iterations: int,
) -> float:
    start = time.perf_counter()
    for _ in range(iterations):
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
            model(sample)
    torch.cuda.synchronize(sample.device)
    return (time.perf_counter() - start) * 1000.0 / iterations


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Paired 640x512 AMP latency benchmark for Fusion and CMRC."
    )
    parser.add_argument(
        "--fusion-config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "flame3" / "pidnet_s_fusion_partial_30e.yaml",
    )
    parser.add_argument(
        "--cmrc-config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "flame3" / "pidnet_s_cmrc_partial_30e.yaml",
    )
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--warmup-iterations", type=int, default=100)
    parser.add_argument("--timed-iterations", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the CMRC latency benchmark.")

    fusion_config = load_config(args.fusion_config.resolve())
    cmrc_config = load_config(args.cmrc_config.resolve())
    if fusion_config.get("MODEL") != "pidnet_s":
        raise ValueError("Fusion benchmark config must use MODEL=pidnet_s.")
    if cmrc_config.get("MODEL") != "pidnet_s_cmrc":
        raise ValueError("CMRC benchmark config must use MODEL=pidnet_s_cmrc.")
    if fusion_config.get("MODE") != "fusion" or cmrc_config.get("MODE") != "fusion":
        raise ValueError("Both benchmark configs must use MODE=fusion.")

    seed_everything(int(cmrc_config["SEED"]))
    device = torch.device("cuda:0")
    fusion = build_model(fusion_config, augment=False)
    candidate = build_model(cmrc_config, augment=False)
    common_state_tensors = load_common_baseline_state(fusion, candidate)
    if common_state_tensors != len(fusion.state_dict()):
        raise RuntimeError("CMRC does not retain every Fusion baseline state tensor.")
    fusion = fusion.to(device).eval()
    candidate = candidate.to(device).eval()
    sample = torch.randn(
        1,
        4,
        max(int(args.height), 1),
        max(int(args.width), 1),
        device=device,
        dtype=torch.float32,
    )

    warmup_iterations = max(int(args.warmup_iterations), 0)
    timed_iterations = max(int(args.timed_iterations), 1)
    repeats = max(int(args.repeats), 1)
    warmup(fusion, sample, warmup_iterations)
    warmup(candidate, sample, warmup_iterations)

    trials = {"fusion": [], "cmrc": []}
    trial_orders = []
    for repeat in range(repeats):
        order = ["fusion", "cmrc"] if repeat % 2 == 0 else ["cmrc", "fusion"]
        trial_orders.append(order)
        for name in order:
            model = fusion if name == "fusion" else candidate
            trials[name].append(timed_forward(model, sample, timed_iterations))

    models = {}
    for name, model in (("fusion", fusion), ("cmrc", candidate)):
        values = np.asarray(trials[name], dtype=np.float64)
        median = float(np.median(values))
        models[name] = {
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "latency_ms_trials": values.tolist(),
            "latency_ms_median": median,
            "latency_ms_mean": float(values.mean()),
            "latency_ms_std": float(values.std()),
            "fps_from_median": 1000.0 / median,
            "passes_30_fps": median <= 1000.0 / 30.0,
        }
    result = {
        "protocol": "same-session paired alternating-order model-forward benchmark",
        "device": torch.cuda.get_device_name(device),
        "input_shape": list(sample.shape),
        "amp": True,
        "warmup_iterations_per_model": warmup_iterations,
        "timed_iterations_per_trial": timed_iterations,
        "repeats": repeats,
        "trial_orders": trial_orders,
        "common_baseline_state_tensors": common_state_tensors,
        "models": models,
        "cmrc_passes_30_fps": models["cmrc"]["passes_30_fps"],
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
