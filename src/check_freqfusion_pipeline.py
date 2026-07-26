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
from custom_models.pidnet_freqfusion import (
    OfficialFreqFusion,
    UPSTREAM_COMMIT,
    UPSTREAM_LICENSE,
    UPSTREAM_LICENSE_SHA256,
    UPSTREAM_REPOSITORY,
    UPSTREAM_SOURCE,
    UPSTREAM_SOURCE_SHA256,
    count_freqfusion_modules,
    file_sha256,
    pytorch_carafe,
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


def reference_carafe(
    features: torch.Tensor,
    mask: torch.Tensor,
    kernel_size: int,
    up: int,
) -> torch.Tensor:
    batch, channels, height, width = features.shape
    padding = kernel_size // 2
    padded = torch.nn.functional.pad(
        features,
        [padding] * 4,
        mode="reflect",
    )
    output = features.new_zeros(
        batch,
        channels,
        height * up,
        width * up,
    )
    for output_y in range(height * up):
        source_y = output_y // up
        for output_x in range(width * up):
            source_x = output_x // up
            local = padded[
                :,
                :,
                source_y : source_y + kernel_size,
                source_x : source_x + kernel_size,
            ].reshape(batch, channels, kernel_size * kernel_size)
            weights = mask[:, :, output_y, output_x].unsqueeze(1)
            output[:, :, output_y, output_x] = (local * weights).sum(dim=-1)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Check the licensed SegNeXt FreqFusion source, independent "
            "CARAFE compatibility, AMP gradients and PIDNet interfaces."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=(
            PROJECT_ROOT
            / "configs"
            / "route_a"
            / "pidnet_s_fusion_freqfusion_pag3_30e_label_fix.yaml"
        ),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the FreqFusion AMP check.")
    config = load_config(args.config.resolve())
    if config["MODEL"] != "pidnet_s_freqfusion":
        raise ValueError("The selected config is not a FreqFusion model.")
    if config["FREQFUSION_VARIANT"] != "pag3":
        raise ValueError("Only the audited Pag3 FreqFusion variant is allowed.")
    required_true = (
        "FREQFUSION_FEATURE_RESAMPLE",
        "FREQFUSION_USE_HIGH_PASS",
        "FREQFUSION_USE_LOW_PASS",
        "FREQFUSION_HR_RESIDUAL",
        "FREQFUSION_COMP_FEAT_UPSAMPLE",
        "FREQFUSION_SEMI_CONV",
        "FREQFUSION_INTERNAL_RESEARCH_ONLY",
    )
    for key in required_true:
        if config.get(key) is not True:
            raise RuntimeError(f"Required full FreqFusion setting is disabled: {key}")
    if config.get("FREQFUSION_OFFICIAL_SOURCE_TRACKED_IN_PARENT") is not False:
        raise RuntimeError("Official FreqFusion source must remain untracked.")

    actual_commit = git_output(UPSTREAM_REPOSITORY, "rev-parse", "HEAD")
    if actual_commit != UPSTREAM_COMMIT:
        raise RuntimeError(
            f"FreqFusion commit mismatch: {actual_commit} vs {UPSTREAM_COMMIT}"
        )
    repository_status = git_output(UPSTREAM_REPOSITORY, "status", "--porcelain")
    if repository_status:
        raise RuntimeError(
            "Pinned FreqFusion checkout is not clean:\n" + repository_status
        )
    source_hash = file_sha256(UPSTREAM_SOURCE)
    license_hash = file_sha256(UPSTREAM_LICENSE)
    if source_hash != UPSTREAM_SOURCE_SHA256:
        raise RuntimeError("Licensed FreqFusion source SHA256 mismatch.")
    if license_hash != UPSTREAM_LICENSE_SHA256:
        raise RuntimeError("FreqFusion SegNeXt LICENSE SHA256 mismatch.")
    license_text = UPSTREAM_LICENSE.read_text(encoding="utf-8")
    if "Apache License" not in license_text or "Version 2.0" not in license_text:
        raise RuntimeError("Selected FreqFusion integration is not Apache-2.0.")

    seed_everything(int(config["SEED"]))
    device = torch.device(config["DEVICE"])

    torch.manual_seed(7)
    carafe_features = torch.randn(1, 4, 3, 4, dtype=torch.float32)
    raw_mask = torch.randn(1, 9, 6, 8, dtype=torch.float32)
    carafe_mask = raw_mask.view(1, 1, 9, 6, 8).softmax(dim=2).view(1, 9, 6, 8)
    carafe_result = pytorch_carafe(
        carafe_features,
        carafe_mask,
        kernel_size=3,
        group=1,
        up=2,
    )
    carafe_reference = reference_carafe(
        carafe_features,
        carafe_mask,
        kernel_size=3,
        up=2,
    )
    carafe_max_abs_error = float(
        (carafe_result - carafe_reference).abs().max().item()
    )
    if carafe_max_abs_error >= 1e-6:
        raise RuntimeError(
            f"Independent CARAFE formula check failed: {carafe_max_abs_error}"
        )

    model = build_model(config, augment=True)
    matched_pretrained = load_pretrained_if_available(model, config)
    baseline_config = dict(config)
    baseline_config["MODEL"] = "pidnet_s"
    baseline = build_model(baseline_config, augment=True)
    baseline_matched = load_pretrained_if_available(baseline, baseline_config)
    if matched_pretrained != baseline_matched:
        raise RuntimeError(
            "FreqFusion changed PIDNet pretrained matching: "
            f"{matched_pretrained} vs {baseline_matched}."
        )
    module_names = [
        name
        for name, module in model.named_modules()
        if isinstance(module, OfficialFreqFusion)
    ]
    if module_names != ["pag3.freqfusion"]:
        raise RuntimeError(f"Unexpected FreqFusion modules: {module_names}")
    if count_freqfusion_modules(model) != 1:
        raise RuntimeError("Expected exactly one FreqFusion module.")

    training_parameters = sum(parameter.numel() for parameter in model.parameters())
    baseline_training_parameters = sum(
        parameter.numel() for parameter in baseline.parameters()
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
        losses, _, _, loss_components = criterion.get_loss(outputs, labels, edges)
        loss = losses.mean()
    if not torch.isfinite(loss):
        raise RuntimeError(f"Non-finite FreqFusion AMP loss: {loss.item()}")
    loss.backward()
    if not isinstance(outputs, (list, tuple)) or len(outputs) != 3:
        raise RuntimeError("FreqFusion changed PIDNet's training output interface.")

    gradient_parameter_names = (
        "pag3.freqfusion.hr_channel_compressor.weight",
        "pag3.freqfusion.lr_channel_compressor.weight",
        "pag3.freqfusion.content_encoder.weight",
        "pag3.freqfusion.content_encoder2.weight",
        "pag3.freqfusion.dysampler.offset.weight",
        "pag3.freqfusion.dysampler.hr_offset.weight",
    )
    named_parameters = dict(model.named_parameters())
    gradient_norms: dict[str, float] = {}
    for name in gradient_parameter_names:
        parameter = named_parameters.get(name)
        if parameter is None or parameter.grad is None:
            raise RuntimeError(f"Missing FreqFusion gradient: {name}")
        norm = float(parameter.grad.float().norm().detach().cpu())
        if not torch.isfinite(parameter.grad).all() or norm <= 0.0:
            raise RuntimeError(f"Invalid FreqFusion gradient {name}: {norm}")
        gradient_norms[name] = norm

    inference_model = build_model(config, augment=False)
    incompatible = inference_model.load_state_dict(
        model.state_dict(),
        strict=False,
    )
    if incompatible.missing_keys:
        raise RuntimeError(
            "Missing FreqFusion inference weights: "
            + ", ".join(incompatible.missing_keys)
        )
    allowed_unexpected_prefixes = ("seghead_p.", "seghead_d.")
    invalid_unexpected = [
        key
        for key in incompatible.unexpected_keys
        if not key.startswith(allowed_unexpected_prefixes)
    ]
    if invalid_unexpected:
        raise RuntimeError(
            "Unexpected non-auxiliary FreqFusion weights: "
            + ", ".join(invalid_unexpected)
        )
    inference_model = inference_model.to(device).eval()
    with torch.inference_mode(), torch.autocast(
        device_type="cuda",
        dtype=torch.float16,
        enabled=True,
    ):
        inference_output = inference_model(images[:1])
    if not isinstance(inference_output, torch.Tensor):
        raise RuntimeError("FreqFusion inference did not return one tensor.")
    expected_inference_shape = (1, *raw_main_output_shape[1:])
    if tuple(inference_output.shape) != expected_inference_shape:
        raise RuntimeError(
            "FreqFusion training/inference main-output shapes differ: "
            f"{tuple(inference_output.shape)} vs {expected_inference_shape}."
        )

    result = {
        "config": str(args.config.resolve()),
        "variant": config["FREQFUSION_VARIANT"],
        "official_repository": config["FREQFUSION_UPSTREAM_REPOSITORY"],
        "official_commit_expected": UPSTREAM_COMMIT,
        "official_commit_actual": actual_commit,
        "official_checkout_clean": True,
        "selected_integration": str(UPSTREAM_SOURCE),
        "selected_integration_license": "Apache-2.0",
        "selected_source_sha256": source_hash,
        "selected_license_sha256": license_hash,
        "official_source_tracked_in_parent": False,
        "carafe_compatibility": "independent_pytorch_formula",
        "carafe_formula_max_abs_error": carafe_max_abs_error,
        "matched_pretrained_tensors": matched_pretrained,
        "baseline_matched_pretrained_tensors": baseline_matched,
        "freqfusion_module_names": module_names,
        "training_parameters": training_parameters,
        "baseline_training_parameters": baseline_training_parameters,
        "parameter_increment": training_parameters - baseline_training_parameters,
        "compressed_channels": int(config["FREQFUSION_COMPRESSED_CHANNELS"]),
        "feature_resample": True,
        "amp_loss": float(loss.detach().cpu()),
        "loss_components": jsonable(loss_components),
        "training_output_shapes": raw_training_output_shapes,
        "inference_output_shape": list(inference_output.shape),
        "gradient_norms": gradient_norms,
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
    print("FreqFusion pipeline check passed")


if __name__ == "__main__":
    main()
