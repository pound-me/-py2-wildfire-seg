from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

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
from custom_models.pidnet_dysample import (
    OfficialDySample,
    UPSTREAM_SOURCE,
    count_dysample_modules,
)


def git_output(repository: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def jsonable(value):
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            return value.detach().cpu().tolist()
        return float(value.detach().cpu())
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Check pinned-official DySample import, pretrained loading, AMP "
            "training gradients, output interfaces and parameter increments."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=(
            PROJECT_ROOT
            / "configs"
            / "adopted_protocol_label_fix"
            / "pidnet_s_dysample_pag4.yaml"
        ),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the DySample AMP check.")
    config = load_config(args.config.resolve())
    if config["MODEL"] != "pidnet_s_dysample":
        raise ValueError("The selected config is not a DySample model.")
    seed_everything(int(config["SEED"]))
    device = torch.device(config["DEVICE"])

    repository = PROJECT_ROOT / "third_party" / "dysample"
    actual_commit = git_output(repository, "rev-parse", "HEAD")
    expected_commit = config["DYSAMPLE_UPSTREAM_COMMIT"]
    if actual_commit != expected_commit:
        raise RuntimeError(
            f"DySample commit mismatch: {actual_commit} vs {expected_commit}"
        )
    repository_status = git_output(repository, "status", "--porcelain")
    if repository_status:
        raise RuntimeError(
            "Pinned DySample checkout is not clean:\n" + repository_status
        )
    license_text = (repository / "LICENSE").read_text(encoding="utf-8")
    if not license_text.startswith("MIT License"):
        raise RuntimeError("The pinned DySample LICENSE is not the recorded MIT license.")
    if Path(UPSTREAM_SOURCE).resolve() != (repository / "dysample.py").resolve():
        raise RuntimeError("The model is not importing the pinned official source file.")

    model = build_model(config, augment=True)
    matched_pretrained = load_pretrained_if_available(model, config)
    baseline_config = dict(config)
    baseline_config["MODEL"] = "pidnet_s"
    baseline = build_model(baseline_config, augment=True)
    baseline_matched_pretrained = load_pretrained_if_available(
        baseline,
        baseline_config,
    )
    if matched_pretrained != baseline_matched_pretrained:
        raise RuntimeError(
            "DySample changed the number of matched PIDNet pretrained tensors: "
            f"{matched_pretrained} vs {baseline_matched_pretrained}."
        )

    training_parameters = sum(parameter.numel() for parameter in model.parameters())
    baseline_training_parameters = sum(
        parameter.numel() for parameter in baseline.parameters()
    )
    module_names = [
        name
        for name, module in model.named_modules()
        if isinstance(module, OfficialDySample)
    ]
    expected_modules = 2 if config["DYSAMPLE_VARIANT"] == "pag4" else 1
    if len(module_names) != expected_modules:
        raise RuntimeError(
            f"Unexpected DySample module count: {len(module_names)} vs "
            f"{expected_modules}."
        )

    dataset = build_dataset(config, split="train")
    loader = DataLoader(
        dataset,
        batch_size=int(config["BATCHSIZE"]),
        shuffle=False,
        num_workers=0,
    )
    batch = next(iter(loader))
    images = batch[0].to(device=device, dtype=torch.float32)
    labels = batch[1].to(device=device, dtype=torch.long)
    edges = batch[2].to(device=device, dtype=torch.float32)
    model = model.to(device).train()
    criterion = build_training_criterion(TotalLoss(config), config)
    model.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
        outputs = model(images)
        raw_training_output_shapes = [list(output.shape) for output in outputs]
        raw_main_output_shape = tuple(outputs[1].shape)
        losses, metric_outputs, _, loss_components = criterion.get_loss(
            outputs,
            labels,
            edges,
        )
        loss = losses.mean()
    if not torch.isfinite(loss):
        raise RuntimeError(f"Non-finite DySample AMP loss: {loss.item()}")
    loss.backward()

    if not isinstance(outputs, (list, tuple)) or len(outputs) != 3:
        raise RuntimeError("DySample training model changed PIDNet's output interface.")
    offset_gradient_norms: dict[str, float] = {}
    for name, module in model.named_modules():
        if not isinstance(module, OfficialDySample):
            continue
        gradient = module.offset.weight.grad
        if gradient is None:
            raise RuntimeError(f"No offset gradient for DySample module: {name}")
        norm = float(gradient.float().norm().detach().cpu())
        if not torch.isfinite(gradient).all() or norm <= 0.0:
            raise RuntimeError(
                f"Invalid offset gradient for DySample module {name}: {norm}"
            )
        offset_gradient_norms[name] = norm

    inference_model = build_model(config, augment=False)
    inference_model.load_state_dict(model.state_dict(), strict=False)
    inference_model = inference_model.to(device).eval()
    with torch.inference_mode(), torch.autocast(
        device_type="cuda",
        dtype=torch.float16,
        enabled=True,
    ):
        inference_output = inference_model(images[:1])
    if not isinstance(inference_output, torch.Tensor):
        raise RuntimeError("DySample inference model did not return one tensor.")
    expected_inference_shape = (1, *raw_main_output_shape[1:])
    if tuple(inference_output.shape) != expected_inference_shape:
        raise RuntimeError(
            "DySample training/inference main-output shapes differ: "
            f"{tuple(inference_output.shape)} vs {expected_inference_shape}."
        )

    result = {
        "config": str(args.config.resolve()),
        "variant": config["DYSAMPLE_VARIANT"],
        "official_repository": config["DYSAMPLE_UPSTREAM_REPOSITORY"],
        "official_commit_expected": expected_commit,
        "official_commit_actual": actual_commit,
        "official_checkout_clean": True,
        "official_license": "MIT",
        "official_source": str(Path(UPSTREAM_SOURCE).resolve()),
        "matched_pretrained_tensors": matched_pretrained,
        "baseline_matched_pretrained_tensors": baseline_matched_pretrained,
        "dysample_module_count": count_dysample_modules(model),
        "dysample_module_names": module_names,
        "training_parameters": training_parameters,
        "baseline_training_parameters": baseline_training_parameters,
        "parameter_increment": training_parameters - baseline_training_parameters,
        "amp_loss": float(loss.detach().cpu()),
        "loss_components": jsonable(loss_components),
        "training_output_shapes": raw_training_output_shapes,
        "inference_output_shape": list(inference_output.shape),
        "offset_gradient_norms": offset_gradient_norms,
        "inference_returns_tensor_only": True,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.output:
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    print("DySample pipeline check passed")


if __name__ == "__main__":
    main()
