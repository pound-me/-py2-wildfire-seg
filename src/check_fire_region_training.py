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
    config = load_config(PROJECT_ROOT / "configs" / "pidnet_s_fire_region.yaml")
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
    fire_loss = components["fire_region_auxiliary"]
    positive_pixels = components["fire_region_positive_pixels"]

    if not torch.isfinite(total_loss) or not torch.isfinite(fire_loss):
        raise RuntimeError("Non-finite fire-region training loss.")
    if float(positive_pixels) <= 0.0:
        raise RuntimeError("The sampled batch contains no fire pixels.")

    model.zero_grad(set_to_none=True)
    fire_loss.backward(retain_graph=True)
    auxiliary_semantic_gradient = model.final_layer.conv2.weight.grad
    auxiliary_boundary_gradient = model.seghead_d.conv2.weight.grad
    if (
        auxiliary_semantic_gradient is None
        or float(auxiliary_semantic_gradient.norm()) <= 0.0
    ):
        raise RuntimeError("Fire-region loss has no semantic-head gradient.")
    if (
        auxiliary_boundary_gradient is not None
        and float(auxiliary_boundary_gradient.norm()) > 0.0
    ):
        raise RuntimeError("Fire-region auxiliary unexpectedly trains boundary head.")
    auxiliary_semantic_norm = float(auxiliary_semantic_gradient.norm())

    model.zero_grad(set_to_none=True)
    total_loss.backward()
    full_semantic_gradient = model.final_layer.conv2.weight.grad
    full_boundary_gradient = model.seghead_d.conv2.weight.grad
    if full_semantic_gradient is None or float(full_semantic_gradient.norm()) <= 0.0:
        raise RuntimeError("Full loss has no semantic-head gradient.")
    if full_boundary_gradient is None or float(full_boundary_gradient.norm()) <= 0.0:
        raise RuntimeError("Full loss has no boundary-head gradient.")

    print("Fire-region auxiliary training check passed")
    print(f"Matched pretrained tensors: {matched}")
    print(f"Total loss: {float(total_loss.detach()):.6f}")
    print(
        "Fire-region components: "
        f"total={float(fire_loss.detach()):.6f}, "
        f"focal={float(components['fire_region_focal'].detach()):.6f}, "
        f"tversky={float(components['fire_region_tversky'].detach()):.6f}, "
        f"positive_pixels={int(float(positive_pixels.detach()))}"
    )
    print(f"Auxiliary semantic-head gradient: {auxiliary_semantic_norm:.6f}")
    print("Auxiliary boundary-head gradient: isolated (zero)")
    print(f"Full semantic-head gradient: {float(full_semantic_gradient.norm()):.6f}")
    print(f"Full boundary-head gradient: {float(full_boundary_gradient.norm()):.6f}")
    print("Inference architecture: unchanged PIDNet-S")


if __name__ == "__main__":
    main()
