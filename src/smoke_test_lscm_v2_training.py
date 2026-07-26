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
from custom_losses import SmokeAwareTotalLoss


def main() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "pidnet_s_lscm_v2.yaml")
    seed_everything(config["SEED"])
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable.")

    dataset = build_dataset(config, split="train")
    generator = torch.Generator()
    generator.manual_seed(config["SEED"])
    loader = DataLoader(
        dataset,
        batch_size=2,
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
    matched_pretrained = load_pretrained_if_available(model, config)
    criterion = SmokeAwareTotalLoss(TotalLoss(config), config)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config["LR"],
        momentum=config["MOMENTUM"],
        weight_decay=config["WD"],
    )

    optimizer.zero_grad(set_to_none=True)
    outputs = model(images)
    if not isinstance(outputs, list) or len(outputs) != 4:
        raise RuntimeError("LSCM v2 must return three PIDNet outputs and smoke logits.")
    losses, metric_outputs, _, components = criterion.get_loss(
        outputs,
        labels,
        edges,
    )
    loss = losses.mean()
    if not torch.isfinite(loss):
        raise RuntimeError(f"Non-finite loss: {float(loss)}")
    loss.backward()

    smoke_head = model.smoke_context.smoke_gate[0]
    if smoke_head.weight.grad is None or not torch.isfinite(
        smoke_head.weight.grad
    ).all():
        raise RuntimeError("Smoke auxiliary head did not receive finite gradients.")
    smoke_gradient_norm = float(smoke_head.weight.grad.norm())
    if smoke_gradient_norm <= 0.0:
        raise RuntimeError("Smoke auxiliary head gradient is zero.")
    optimizer.step()
    torch.cuda.synchronize(device)

    valid = labels != config["IGNORE_LABEL"]
    smoke_pixels = int(((labels == config["SMOKE_CLASS_INDEX"]) & valid).sum())
    print("LSCM v2 real-batch forward/backward check passed")
    print(f"Matched pretrained tensors: {matched_pretrained}")
    print(f"Input shape: {tuple(images.shape)}")
    print(f"PIDNet output shapes: {[tuple(x.shape) for x in metric_outputs]}")
    print(f"Smoke-logit shape: {tuple(outputs[3].shape)}")
    print(f"Smoke target pixels: {smoke_pixels}")
    print(f"Total loss: {float(loss.detach()):.6f}")
    print(
        "Auxiliary losses: "
        f"raw={float(components['smoke_auxiliary'].detach()):.6f}, "
        f"bce={float(components['smoke_bce'].detach()):.6f}, "
        f"dice={float(components['smoke_dice'].detach()):.6f}, "
        f"weighted={float(components['weighted_smoke_auxiliary'].detach()):.6f}"
    )
    print(f"Smoke-head gradient norm: {smoke_gradient_norm:.6f}")
    print(
        "Residual scale after one step: "
        f"{float(model.smoke_context.residual_scale.detach()):.6f}"
    )


if __name__ == "__main__":
    main()
