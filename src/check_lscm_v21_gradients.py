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
from custom_losses import SmokeAuxiliaryLoss, SmokeAwareTotalLoss


def gradient_norm(parameter: torch.nn.Parameter) -> float:
    if parameter.grad is None:
        return 0.0
    return float(parameter.grad.norm())


def main() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "pidnet_s_lscm_v21.yaml")
    seed_everything(config["SEED"])
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable.")

    dataset = build_dataset(config, split="train")
    generator = torch.Generator().manual_seed(config["SEED"])
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
    matched = load_pretrained_if_available(model, config)
    outputs = model(images)
    smoke_logits = outputs[3]

    auxiliary = SmokeAuxiliaryLoss(
        ignore_label=config["IGNORE_LABEL"],
        smoke_class=config["SMOKE_CLASS_INDEX"],
    )
    aux_loss, _, _ = auxiliary(smoke_logits, labels)
    model.zero_grad(set_to_none=True)
    aux_loss.backward()

    head_parameter = model.smoke_context.smoke_gate[0].weight
    context_parameter = model.smoke_context.fuse[0].weight
    head_aux_norm = gradient_norm(head_parameter)
    context_aux_norm = gradient_norm(context_parameter)
    if head_aux_norm <= 0.0:
        raise RuntimeError("Auxiliary loss did not train the smoke head.")
    if context_aux_norm != 0.0:
        raise RuntimeError(
            "Auxiliary gradient leaked into shared context features: "
            f"{context_aux_norm}"
        )

    model.zero_grad(set_to_none=True)
    outputs = model(images)
    criterion = SmokeAwareTotalLoss(TotalLoss(config), config)
    losses, _, _, components = criterion.get_loss(outputs, labels, edges)
    full_loss = losses.mean()
    full_loss.backward()
    context_full_norm = gradient_norm(model.smoke_context.fuse[0].weight)
    if context_full_norm <= 0.0:
        raise RuntimeError("Main segmentation loss did not train context features.")

    print("LSCM v2.1 gradient-isolation check passed")
    print(f"Matched pretrained tensors: {matched}")
    print(f"Auxiliary loss: {float(aux_loss.detach()):.6f}")
    print(f"Smoke-head aux gradient norm: {head_aux_norm:.6f}")
    print(f"Shared-context aux gradient norm: {context_aux_norm:.6f}")
    print(f"Full loss: {float(full_loss.detach()):.6f}")
    print(f"Shared-context full gradient norm: {context_full_norm:.6f}")
    print(
        "Weighted smoke auxiliary: "
        f"{float(components['weighted_smoke_auxiliary'].detach()):.6f}"
    )


if __name__ == "__main__":
    main()
