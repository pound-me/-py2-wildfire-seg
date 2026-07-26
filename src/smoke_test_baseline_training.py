from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from baseline_runtime import (  # noqa: E402
    PROJECT_ROOT,
    TotalLoss,
    build_dataset,
    build_model,
    load_config,
    seed_everything,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one official PIDNet-S RGB training step on FLAME2."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "pidnet_s_rgb_baseline.yaml",
    )
    args = parser.parse_args()
    config = load_config(args.config.resolve())
    seed_everything(config["SEED"])

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable.")

    device = torch.device(config["DEVICE"])
    dataset = build_dataset(config, split="train")
    loader = DataLoader(
        dataset,
        batch_size=config["BATCHSIZE"],
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )
    images, labels, edges, names = next(iter(loader))
    images = images.to(device=device, dtype=torch.float, non_blocking=True)
    labels = labels.to(device=device, dtype=torch.long, non_blocking=True)
    edges = edges.to(device=device, dtype=torch.float, non_blocking=True)

    model = build_model(config).to(device).train()
    criterion = TotalLoss(config)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config["LR"],
        momentum=config["MOMENTUM"],
        weight_decay=config["WD"],
    )

    torch.cuda.reset_peak_memory_stats(device)
    optimizer.zero_grad(set_to_none=True)
    outputs = model(images)
    losses, _, accuracy, loss_parts = criterion.get_loss(outputs, labels, edges)
    loss = losses.mean()
    if not torch.isfinite(loss):
        raise RuntimeError(f"Non-finite loss detected: {loss.item()}")
    loss.backward()

    gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1e9)
    optimizer.step()
    torch.cuda.synchronize(device)

    peak_memory_mb = torch.cuda.max_memory_allocated(device) / 1024**2
    class_ids, class_counts = torch.unique(labels, return_counts=True)
    label_summary = {
        int(class_id): int(count)
        for class_id, count in zip(class_ids.cpu(), class_counts.cpu())
    }

    print("PIDNet-S RGB baseline training smoke test passed")
    print(f"Samples: {list(names)}")
    print(f"Input shape: {tuple(images.shape)}")
    print(f"Label shape: {tuple(labels.shape)}")
    print(f"Label pixels: {label_summary}")
    print(f"Output shapes: {[tuple(output.shape) for output in outputs]}")
    print(f"Total loss: {loss.item():.6f}")
    print(
        "Loss parts: "
        f"semantic={loss_parts[0].item():.6f}, "
        f"boundary={loss_parts[1].item():.6f}"
    )
    print(f"Pixel accuracy vector: {np.asarray(accuracy).tolist()}")
    print(f"Gradient norm: {float(gradient_norm):.6f}")
    print(f"Peak allocated GPU memory: {peak_memory_mb:.1f} MB")


if __name__ == "__main__":
    main()
