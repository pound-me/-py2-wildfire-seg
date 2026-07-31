from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from baseline_runtime import build_dataset, load_config, seed_everything
from custom_losses import PartialLabelSetLoss


def check_formula_and_gradients() -> dict[str, float]:
    criterion = PartialLabelSetLoss(align_corners=False)
    logits = torch.zeros((2, 3, 1, 3), dtype=torch.float32, requires_grad=True)
    labels = torch.tensor(
        [
            [[0, 2, 255]],
            [[0, 0, 0]],
        ],
        dtype=torch.long,
    )
    flags = torch.tensor([1, 0], dtype=torch.bool)
    loss, diagnostics = criterion(logits, labels, flags)
    partial = -math.log(2.0 / 3.0)
    hard = -math.log(1.0 / 3.0)
    expected = (0.5 * (partial + hard) + hard) / 2.0
    if not math.isclose(
        float(loss.detach()), expected, rel_tol=0.0, abs_tol=1e-6
    ):
        raise RuntimeError(f"Partial-label formula mismatch: {float(loss)} != {expected}")
    loss.backward()
    gradient = logits.grad
    assert gradient is not None
    partial_gradient = gradient[0, :, 0, 0]
    if not (
        partial_gradient[0] < 0
        and partial_gradient[1] < 0
        and partial_gradient[2] > 0
    ):
        raise RuntimeError(
            f"Partial pixel gradient directions are wrong: {partial_gradient.tolist()}"
        )
    if not torch.allclose(
        partial_gradient[0], partial_gradient[1], rtol=0.0, atol=1e-7
    ):
        raise RuntimeError("Background and Smoke gradients must be symmetric")
    fire_gradient = gradient[0, :, 0, 1]
    if not (fire_gradient[2] < 0 and fire_gradient[0] > 0 and fire_gradient[1] > 0):
        raise RuntimeError(
            f"Fire-core gradient directions are wrong: {fire_gradient.tolist()}"
        )
    if int(diagnostics["partial_nonfire_pixels"]) != 1:
        raise RuntimeError("Unexpected partial pixel diagnostic")
    if int(diagnostics["fire_core_pixels"]) != 1:
        raise RuntimeError("Unexpected Fire-core pixel diagnostic")
    if int(diagnostics["hard_background_pixels"]) != 3:
        raise RuntimeError("Unexpected hard-Background pixel diagnostic")
    return {
        "loss": float(loss.detach()),
        "expected_loss": expected,
        "partial_fire_gradient": float(partial_gradient[2]),
        "fire_core_fire_gradient": float(fire_gradient[2]),
    }


def check_dataset(
    config_path: Path, root_dataset: Path, train_csv: Path | None = None
) -> dict[str, object]:
    config = load_config(config_path)
    config["ROOTDATASET"] = str(root_dataset.resolve())
    if train_csv is not None:
        config["TRAINSET"] = str(train_csv.resolve())
    config["DEVICE"] = "cpu"
    seed_everything(int(config["SEED"]))
    dataset = build_dataset(config, split="train")
    fire_index = next(
        index for index, row in enumerate(dataset.rows) if row["sample_class"] == "Fire"
    )
    no_fire_index = next(
        index
        for index, row in enumerate(dataset.rows)
        if row["sample_class"] == "No Fire"
    )
    loader = DataLoader(
        Subset(dataset, [fire_index, no_fire_index]),
        batch_size=2,
        shuffle=False,
        num_workers=0,
    )
    images, labels, edges, names, flags = next(iter(loader))
    if tuple(images.shape) != (2, 4, 512, 640):
        raise RuntimeError(f"Unexpected FLAME3 fusion batch shape: {tuple(images.shape)}")
    if flags.to(dtype=torch.int64).tolist() != [1, 0]:
        raise RuntimeError(f"Fire-folder flags are wrong: {flags.tolist()}")
    allowed = {0, 2, 255}
    values = set(int(value) for value in torch.unique(labels).tolist())
    if not values.issubset(allowed):
        raise RuntimeError(f"Unexpected augmented label values: {sorted(values)}")
    if bool(labels[1].eq(2).any()) or bool(labels[1].eq(255).any()):
        raise RuntimeError("No Fire sample lost hard-Background supervision")
    if not torch.isfinite(images).all():
        raise RuntimeError("Non-finite FLAME3 input values")
    if not set(int(value) for value in torch.unique(edges).tolist()).issubset({0, 1}):
        raise RuntimeError("Boundary map is not binary")
    return {
        "samples": [str(value) for value in names[0]],
        "input_shape": list(images.shape),
        "label_values": sorted(values),
        "flags": flags.to(dtype=torch.int64).tolist(),
        "edge_positive_pixels": [int(item.sum()) for item in edges],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root-dataset", type=Path, required=True)
    parser.add_argument("--train-csv", type=Path)
    args = parser.parse_args()
    formula = check_formula_and_gradients()
    dataset = check_dataset(
        args.config.resolve(), args.root_dataset.resolve(), args.train_csv
    )
    print("FLAME3 partial-label formula and dataset check passed")
    print({"formula": formula, "dataset": dataset})


if __name__ == "__main__":
    main()
