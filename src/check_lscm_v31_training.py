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
    config = load_config(PROJECT_ROOT / "configs" / "pidnet_s_lscm_v31.yaml")
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
    loss = losses.mean()
    model.zero_grad(set_to_none=True)
    loss.backward()

    head_gradient = model.smoke_context.class_head.weight.grad
    context_gradient = model.smoke_context.fuse[0].weight.grad
    if head_gradient is None or float(head_gradient.norm()) <= 0.0:
        raise RuntimeError("Training-only class head has no gradient.")
    if context_gradient is None or float(context_gradient.norm()) <= 0.0:
        raise RuntimeError("Shared context has no gradient.")

    inference_model = build_model(config, augment=False)
    if inference_model.smoke_context.class_head is not None:
        raise RuntimeError("Inference model unexpectedly contains the class head.")

    print("LSCM v3.1 training-only prototype check passed")
    print(f"Matched pretrained tensors: {matched}")
    print(f"Total loss: {float(loss.detach()):.6f}")
    print(
        "Auxiliary components: "
        f"balanced_ce={float(components['class_auxiliary'].detach()):.6f}, "
        f"prototype={float(components['prototype_total'].detach()):.6f}, "
        f"present_classes={int(float(components['prototype_present_classes']))}"
    )
    print(f"Class-head gradient norm: {float(head_gradient.norm()):.6f}")
    print(f"Context gradient norm: {float(context_gradient.norm()):.6f}")
    print(
        "Residual scale: "
        f"{float(model.smoke_context.residual_scale.detach()):.6f}"
    )
    print("Inference class head: removed")


if __name__ == "__main__":
    main()
