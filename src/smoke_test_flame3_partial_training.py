from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from baseline_runtime import (
    TotalLoss,
    build_dataset,
    build_model,
    load_config,
    load_pretrained_if_available,
    seed_everything,
)
from custom_losses import build_training_criterion


def create_cuda_grad_scaler(init_scale: float):
    amp_namespace = getattr(torch, "amp", None)
    scaler_class = getattr(amp_namespace, "GradScaler", None)
    if scaler_class is not None:
        try:
            return scaler_class("cuda", enabled=True, init_scale=init_scale)
        except TypeError:
            pass
    return torch.cuda.amp.GradScaler(enabled=True, init_scale=init_scale)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root-dataset", type=Path, required=True)
    parser.add_argument("--pretrained", type=Path, required=True)
    parser.add_argument("--train-csv", type=Path)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--measure-steps", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-non-4090", action="store_true")
    args = parser.parse_args()
    if args.batch_size <= 0 or args.measure_steps <= 0 or args.warmup_steps < 0:
        raise ValueError("Invalid batch/step arguments")
    config = load_config(args.config.resolve())
    config["ROOTDATASET"] = str(args.root_dataset.resolve())
    config["PRETRAINED"] = str(args.pretrained.resolve())
    if args.train_csv is not None:
        config["TRAINSET"] = str(args.train_csv.resolve())
    config["BATCHSIZE"] = args.batch_size
    config["NUM_WORKERS"] = 0
    config["DEVICE"] = "cuda:0"
    seed_everything(int(config["SEED"]))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    device = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(device)
    if "4090" not in gpu_name and not args.allow_non_4090:
        raise RuntimeError(f"Expected RTX 4090, got {gpu_name}")

    dataset = build_dataset(config, split="train")
    loader_generator = torch.Generator()
    loader_generator.manual_seed(int(config["SEED"]))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
        generator=loader_generator,
    )
    model = build_model(config).to(device).train()
    matched_pretrained = load_pretrained_if_available(model, config)
    criterion = build_training_criterion(TotalLoss(config), config)
    if criterion.objective_name != "partial_label":
        raise RuntimeError("Partial-label criterion is not active")
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=float(config["LR"]),
        momentum=float(config["MOMENTUM"]),
        weight_decay=float(config["WD"]),
    )
    scaler = create_cuda_grad_scaler(
        init_scale=float(config.get("AMP_INIT_SCALE", 65536.0)),
    )
    total_steps = args.warmup_steps + args.measure_steps
    measured_times: list[float] = []
    measured_losses: list[float] = []
    gradient_norms: list[float] = []
    total_fire_folder_images = 0
    total_no_fire_images = 0
    input_shape: list[int] | None = None
    iterator = iter(loader)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    for step in range(total_steps):
        try:
            images, labels, edges, names, flags = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            images, labels, edges, names, flags = next(iterator)
        images = images.to(device=device, dtype=torch.float, non_blocking=True)
        labels = labels.to(device=device, dtype=torch.long, non_blocking=True)
        edges = edges.to(device=device, dtype=torch.float, non_blocking=True)
        flags = flags.to(device=device, dtype=torch.bool, non_blocking=True)
        input_shape = list(images.shape)
        total_fire_folder_images += int(flags.sum())
        total_no_fire_images += int((~flags).sum())
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
            outputs = model(images)
            losses, _, _, components = criterion.get_loss(
                outputs,
                labels,
                edges,
                fire_folder_flags=flags,
            )
            loss = losses.mean()
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at step {step}: {float(loss)}")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=1e9
        )
        if not torch.isfinite(gradient_norm) or float(gradient_norm) <= 0.0:
            nonfinite = [
                name
                for name, parameter in model.named_parameters()
                if parameter.grad is not None
                and not bool(torch.isfinite(parameter.grad).all())
            ]
            component_values = {
                key: float(value.detach()) for key, value in components.items()
            }
            raise RuntimeError(
                f"Invalid gradient norm at step {step}: {gradient_norm}; "
                f"nonfinite_parameters={nonfinite[:20]}; "
                f"loss_components={component_values}; "
                f"amp_scale={scaler.get_scale()}"
            )
        scaler.step(optimizer)
        scaler.update()
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        if step >= args.warmup_steps:
            measured_times.append(elapsed)
            measured_losses.append(float(loss.detach()))
            gradient_norms.append(float(gradient_norm.detach()))

    total_memory = torch.cuda.get_device_properties(device).total_memory
    if total_fire_folder_images == 0 or total_no_fire_images == 0:
        raise RuntimeError(
            "Random-batch smoke test did not cover both Fire and No Fire images"
        )
    peak_allocated = torch.cuda.max_memory_allocated(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)
    median_step = statistics.median(measured_times)
    result = {
        "status": "passed",
        "gpu": gpu_name,
        "batch_size": args.batch_size,
        "input_shape": input_shape,
        "fire_folder_images_seen": total_fire_folder_images,
        "no_fire_images_seen": total_no_fire_images,
        "distinct_random_augmented_batches": total_steps,
        "warmup_steps": args.warmup_steps,
        "measured_steps": args.measure_steps,
        "median_step_seconds": median_step,
        "samples_per_second": args.batch_size / median_step,
        "loss_min": min(measured_losses),
        "loss_max": max(measured_losses),
        "gradient_norm_min": min(gradient_norms),
        "gradient_norm_max": max(gradient_norms),
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
        "gpu_total_memory_bytes": total_memory,
        "peak_allocated_ratio": peak_allocated / total_memory,
        "peak_reserved_ratio": peak_reserved / total_memory,
        "matched_pretrained_tensors": matched_pretrained,
        "last_loss_components": {
            key: float(value.detach()) for key, value in components.items()
        },
        "test_images_or_labels_read": False,
    }
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.resolve().write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
