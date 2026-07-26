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
    config = load_config(PROJECT_ROOT / "configs" / "pidnet_s_lscm_v3.yaml")
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
    if not isinstance(outputs, list) or len(outputs) != 5:
        raise RuntimeError("LSCM v3 did not return the expected five outputs.")
    losses, metric_outputs, _, components = criterion.get_loss(
        outputs,
        labels,
        edges,
    )
    loss = losses.mean()
    if not torch.isfinite(loss):
        raise RuntimeError(f"Non-finite total loss: {float(loss)}")
    model.zero_grad(set_to_none=True)
    loss.backward()

    class_head_grad = model.smoke_context.class_head.weight.grad
    context_grad = model.smoke_context.fuse[0].weight.grad
    if class_head_grad is None or float(class_head_grad.norm()) <= 0.0:
        raise RuntimeError("Three-class auxiliary head has no gradient.")
    if context_grad is None or float(context_grad.norm()) <= 0.0:
        raise RuntimeError("Prototype context features have no gradient.")
    if float(components["prototype_present_classes"]) < 2.0:
        raise RuntimeError("The test batch contains fewer than two valid classes.")

    print("LSCM v3 class-prototype real-batch check passed")
    print(f"Matched pretrained tensors: {matched}")
    print(f"Input shape: {tuple(images.shape)}")
    print(f"PIDNet output shapes: {[tuple(x.shape) for x in metric_outputs]}")
    print(f"Class-logit shape: {tuple(outputs[3].shape)}")
    print(f"Prototype-feature shape: {tuple(outputs[4].shape)}")
    print(f"Total loss: {float(loss.detach()):.6f}")
    print(
        "Auxiliary components: "
        f"class_ce={float(components['class_auxiliary'].detach()):.6f}, "
        f"prototype={float(components['prototype_total'].detach()):.6f}, "
        f"pixel={float(components['prototype_pixel'].detach()):.6f}, "
        f"separation={float(components['prototype_separation'].detach()):.6f}, "
        f"present_classes={int(float(components['prototype_present_classes']))}"
    )
    print(f"Class-head gradient norm: {float(class_head_grad.norm()):.6f}")
    print(f"Context gradient norm: {float(context_grad.norm()):.6f}")
    print(
        "Initial residual scale: "
        f"{float(model.smoke_context.residual_scale.detach()):.6f}"
    )


if __name__ == "__main__":
    main()
