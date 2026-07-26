from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
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
from prototype_learning import EMAPrototypeBank
from train_baseline import assert_component_alignment


def gradient_norm(module: torch.nn.Module) -> float:
    total = 0.0
    for parameter in module.parameters():
        if parameter.grad is not None:
            total += float(parameter.grad.detach().float().norm().cpu())
    return total


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check component sampling, gradients, AMP, state restore, and inference."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "pidnet_s_dfm_mproto_p4.yaml",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    config = load_config(args.config.resolve())
    seed_everything(int(config["SEED"]))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the AMP pipeline check.")
    device = torch.device(config["DEVICE"])
    dataset = build_dataset(config, split="train")
    loader = DataLoader(
        dataset,
        batch_size=int(config["BATCHSIZE"]),
        shuffle=False,
        num_workers=0,
        drop_last=True,
    )
    batch = None
    selected_batch_index = -1
    for candidate_index, candidate in enumerate(loader):
        present = set(torch.unique(candidate[1]).tolist())
        if {0, 1, 2}.issubset(present):
            batch = candidate
            selected_batch_index = candidate_index
            break
    if batch is None:
        raise RuntimeError(
            "Could not find a training batch containing background, smoke, and fire."
        )
    images, labels, edges, component_maps = batch[0], batch[1], batch[2], batch[4]
    assert_component_alignment(labels, component_maps, int(config["FIRE_CLASS_INDEX"]))

    labels_np = labels.numpy()
    components_np = component_maps.numpy()
    temporary_bank = EMAPrototypeBank(
        num_classes=int(config["NUM_CLASSES"]),
        feature_channels=int(config["PROTOTYPE_FEATURE_CHANNELS"]),
        prototypes_per_class=int(config["PROTOTYPES_PER_CLASS"]),
        temperature=float(config["PROTOTYPE_TEMPERATURE"]),
        momentum=float(config["PROTOTYPE_EMA_MOMENTUM"]),
        max_samples_per_class=int(config["PROTOTYPE_MAX_SAMPLES_PER_CLASS"]),
        fire_class=int(config["FIRE_CLASS_INDEX"]),
        fire_sampling=str(config["FIRE_PROTOTYPE_SAMPLING"]),
        decorrelation=bool(config["PROTOTYPE_DECORRELATION"]),
        decorrelation_weight=float(config["PROTOTYPE_DECORRELATION_WEIGHT"]),
        decorrelation_min_samples=int(
            config.get("PROTOTYPE_DECORRELATION_MIN_SAMPLES", 2)
        ),
        seed=int(config["SEED"]),
    )
    dead_gate_bank = copy.deepcopy(temporary_bank)
    dead_gate_report = None
    for _ in range(dead_gate_bank.persistence_epochs):
        dead_gate_bank.begin_epoch()
        dead_gate_report = dead_gate_bank.finalize_epoch_health()
    if dead_gate_report is None or dead_gate_report[
        "valid_multi_prototype_method"
    ]:
        raise RuntimeError(
            "Uninitialized zero-hit prototypes evaded the persistent-death gate."
        )
    first = temporary_bank._sample_coordinates(
        labels_np, components_np, np.random.default_rng(12345)
    )
    second = temporary_bank._sample_coordinates(
        labels_np, components_np, np.random.default_rng(12345)
    )
    if not all(np.array_equal(a, b) for a, b in zip(first, second)):
        raise RuntimeError("Fixed-seed point sampling is not reproducible.")
    coordinates, class_ids, _ = first
    if np.unique(coordinates, axis=0).shape[0] != coordinates.shape[0]:
        raise RuntimeError("Point sampling contains duplicate coordinates.")
    sample_counts = {
        str(class_index): int((class_ids == class_index).sum())
        for class_index in range(int(config["NUM_CLASSES"]))
    }
    if any(
        count > int(config["PROTOTYPE_MAX_SAMPLES_PER_CLASS"])
        for count in sample_counts.values()
    ):
        raise RuntimeError("A class exceeded its sampling limit.")

    synthetic_labels = np.zeros((1, 256, 256), dtype=np.uint8)
    synthetic_components = np.zeros((1, 256, 256), dtype=np.uint16)
    synthetic_labels[:, 100:180, 100:180] = 1
    synthetic_labels[:, 0:24, 0:24] = 2
    synthetic_components[:, 0:24, 0:24] = 1
    component_id = 2
    for y in range(30, 70, 4):
        for x in range(0, 80, 4):
            synthetic_labels[:, y:y + 2, x:x + 2] = 2
            synthetic_components[:, y:y + 2, x:x + 2] = component_id
            component_id += 1
    synthetic = temporary_bank._sample_coordinates(
        synthetic_labels,
        synthetic_components,
        np.random.default_rng(444),
    )
    synthetic_coordinates, synthetic_classes, synthetic_buckets = synthetic
    synthetic_fire = synthetic_coordinates[synthetic_classes == 2]
    if synthetic_fire.shape[0] != 512:
        raise RuntimeError(
            f"Synthetic mixed Fire sampling expected 512 points, got "
            f"{synthetic_fire.shape[0]}."
        )
    if np.unique(synthetic_fire, axis=0).shape[0] != 512:
        raise RuntimeError("Synthetic mixed Fire sampling repeated coordinates.")
    fire_buckets = synthetic_buckets[synthetic_classes == 2]
    if not {0, 2}.issubset(set(fire_buckets.tolist())):
        raise RuntimeError(
            "Synthetic Fire sampling did not cover small and large components."
        )

    model = build_model(config, augment=True).to(device)
    load_pretrained_if_available(model, config)
    criterion = build_training_criterion(TotalLoss(config), config)
    model.train()
    images = images.to(device=device, dtype=torch.float32)
    labels_device = labels.to(device=device, dtype=torch.long)
    edges_device = edges.to(device=device, dtype=torch.float32)

    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
        outputs = model(images)
    fused_feature = outputs[3]
    criterion.bank(
        fused_feature.detach(),
        labels,
        component_maps,
        sampling_seed=700,
        update_ema=True,
        collect_health=False,
    )
    model.zero_grad(set_to_none=True)
    prototype_total, _, _, _ = criterion.bank(
        fused_feature,
        labels,
        component_maps,
        sampling_seed=701,
        update_ema=False,
        collect_health=False,
    )
    prototype_total.backward()
    branch_gradients = {
        "dfm": gradient_norm(model.dfm),
        "p_branch": gradient_norm(model.layer5_),
        "i_branch": gradient_norm(model.spp),
        "d_branch": gradient_norm(model.layer5_d),
    }
    if any(value <= 0.0 for value in branch_gradients.values()):
        raise RuntimeError(f"A required branch has zero gradient: {branch_gradients}")
    if criterion.bank.prototypes.grad is not None:
        raise RuntimeError("EMA prototypes unexpectedly received gradients.")

    model.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
        outputs = model(images)
        losses, _, _, components = criterion.get_loss(
            outputs,
            labels_device,
            edges_device,
            component_maps=component_maps,
            sampling_labels=labels,
            epoch=1,
            batch_index=1,
            epoch_batches=10,
            training=True,
        )
        full_loss = losses.mean()
    if not torch.isfinite(full_loss):
        raise RuntimeError("AMP pipeline produced a non-finite loss.")
    full_loss.backward()

    restored = build_training_criterion(TotalLoss(config), config)
    restored.load_state_dict(copy.deepcopy(criterion.state_dict()))
    for key, value in criterion.state_dict().items():
        if not torch.equal(value, restored.state_dict()[key]):
            raise RuntimeError(f"Prototype state restore mismatch: {key}")
    for bank in (criterion.bank, restored.bank):
        bank(
            fused_feature.detach(),
            labels,
            component_maps,
            sampling_seed=1701,
            update_ema=True,
            collect_health=False,
        )
    for key, value in criterion.state_dict().items():
        if not torch.equal(value, restored.state_dict()[key]):
            raise RuntimeError(f"Next EMA update differs after restore: {key}")

    inference_model = build_model(config, augment=False).to(device).eval()
    incompatible = inference_model.load_state_dict(model.state_dict(), strict=False)
    if incompatible.missing_keys:
        raise RuntimeError(
            "Inference model is missing deployment weights: "
            + ", ".join(incompatible.missing_keys)
        )
    if any(isinstance(module, EMAPrototypeBank) for module in inference_model.modules()):
        raise RuntimeError("Inference model contains an EMA prototype bank.")
    with torch.inference_mode():
        inference_output = inference_model(images[:1])
    if not isinstance(inference_output, torch.Tensor):
        raise RuntimeError("Inference model returned training-only outputs.")

    result = {
        "component_alignment": True,
        "fixed_seed_sampling_reproducible": True,
        "sampling_without_replacement": True,
        "sample_counts": sample_counts,
        "selected_real_batch_index": selected_batch_index,
        "synthetic_mixed_fire_count": int(synthetic_fire.shape[0]),
        "synthetic_fire_bucket_counts": {
            "small_le_4": int((fire_buckets == 0).sum()),
            "medium_5_64": int((fire_buckets == 1).sum()),
            "large_gt_64": int((fire_buckets == 2).sum()),
        },
        "prototype_loss": float(prototype_total.detach().cpu()),
        "amp_full_loss": float(full_loss.detach().cpu()),
        "prototype_components": {
            name: float(value.detach().cpu()) for name, value in components.items()
        },
        "branch_gradient_norms": branch_gradients,
        "prototype_gradient_is_none": criterion.bank.prototypes.grad is None,
        "checkpoint_restore_exact": True,
        "next_ema_update_exact": True,
        "uninitialized_zero_hit_prototypes_fail_after_five_epochs": True,
        "inference_has_no_prototype_bank": True,
        "inference_returns_tensor_only": True,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.output:
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print("EMA multi-prototype pipeline check passed")


if __name__ == "__main__":
    main()
