from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from baseline_runtime import (
    PROJECT_ROOT,
    TotalLoss,
    build_dataset,
    build_model,
    load_config,
    load_pretrained_if_available,
    seed_everything,
)
from custom_losses import build_training_criterion


def main() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "pidnet_s_fire_boundary.yaml")
    seed_everything(config["SEED"])
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable.")

    dataset = build_dataset(config, split="train")
    generator = torch.Generator().manual_seed(config["SEED"])
    loader = DataLoader(
        dataset,
        batch_size=config["BATCHSIZE"],
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
        generator=generator,
    )
    images, labels, edges = next(iter(loader))[:3]
    device = torch.device(config["DEVICE"])
    images = images.to(device=device, dtype=torch.float)
    labels = labels.to(device=device, dtype=torch.long)
    edges = edges.to(device=device, dtype=torch.float)

    model = build_model(config).to(device).train()
    matched = load_pretrained_if_available(model, config)
    criterion = build_training_criterion(TotalLoss(config), config)
    outputs = model(images)
    losses, _, _, components = criterion.get_loss(outputs, labels, edges)
    total_loss = losses.mean()
    fire_loss = components["fire_boundary_auxiliary"]
    positive_pixels = components["fire_boundary_positive_pixels"]

    if not torch.isfinite(total_loss) or not torch.isfinite(fire_loss):
        raise RuntimeError("Non-finite fire-boundary training loss.")
    if float(positive_pixels) <= 0.0:
        raise RuntimeError("The sampled batch contains no fire-boundary pixels.")

    model.zero_grad(set_to_none=True)
    fire_loss.backward(retain_graph=True)
    auxiliary_gradient = model.seghead_d.conv2.weight.grad
    if auxiliary_gradient is None or float(auxiliary_gradient.norm()) <= 0.0:
        raise RuntimeError("Fire-boundary loss has no boundary-head gradient.")
    auxiliary_gradient_norm = float(auxiliary_gradient.norm())

    model.zero_grad(set_to_none=True)
    total_loss.backward()
    boundary_gradient = model.seghead_d.conv2.weight.grad
    semantic_gradient = model.final_layer.conv2.weight.grad
    if boundary_gradient is None or float(boundary_gradient.norm()) <= 0.0:
        raise RuntimeError("Full loss has no boundary-head gradient.")
    if semantic_gradient is None or float(semantic_gradient.norm()) <= 0.0:
        raise RuntimeError("Full loss has no semantic-head gradient.")

    print("Fire-boundary auxiliary training check passed")
    print(f"Matched pretrained tensors: {matched}")
    print(f"Total loss: {float(total_loss.detach()):.6f}")
    print(
        "Fire-boundary components: "
        f"total={float(fire_loss.detach()):.6f}, "
        f"bce={float(components['fire_boundary_bce'].detach()):.6f}, "
        f"dice={float(components['fire_boundary_dice'].detach()):.6f}, "
        f"positive_pixels={int(float(positive_pixels.detach()))}"
    )
    print(f"Auxiliary boundary-head gradient: {auxiliary_gradient_norm:.6f}")
    print(f"Full boundary-head gradient: {float(boundary_gradient.norm()):.6f}")
    print(f"Full semantic-head gradient: {float(semantic_gradient.norm()):.6f}")
    print("Inference architecture: unchanged PIDNet-S")


if __name__ == "__main__":
    main()
