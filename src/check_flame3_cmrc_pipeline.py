from __future__ import annotations

import argparse
import json
import tempfile
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
from custom_models.pidnet_cmrc import PIDNetCMRC, count_cmrc_modules


def gradient_norm(model: torch.nn.Module, prefix: str) -> float:
    total = 0.0
    for name, parameter in model.named_parameters():
        if not name.startswith(prefix) or parameter.grad is None:
            continue
        total += float(parameter.grad.detach().float().pow(2).sum().cpu())
    return total**0.5


def max_abs_difference(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.float() - right.float()).abs().max().detach().cpu())


def main_logits(outputs) -> torch.Tensor:
    if isinstance(outputs, (list, tuple)):
        return outputs[1]
    if isinstance(outputs, torch.Tensor):
        return outputs
    raise TypeError(f"Unsupported output type: {type(outputs)!r}")


def load_common_baseline_state(
    baseline: torch.nn.Module,
    candidate: torch.nn.Module,
) -> int:
    source = baseline.state_dict()
    target = candidate.state_dict()
    common = {
        key: value
        for key, value in source.items()
        if key in target and target[key].shape == value.shape
    }
    target.update(common)
    candidate.load_state_dict(target, strict=True)
    return len(common)


def assert_aux_finite(aux: dict[str, torch.Tensor]) -> None:
    expected = {
        "residual_abs_mean",
        "residual_abs_max",
        "residual_saturation_ratio",
    }
    if set(aux) != expected:
        raise RuntimeError(f"Unexpected CMRC aux keys: {sorted(aux)}")
    for name, value in aux.items():
        if value.numel() != 1 or not bool(torch.isfinite(value)):
            raise RuntimeError(f"Non-finite CMRC auxiliary statistic: {name}")
    if float(aux["residual_abs_max"].cpu()) > 0.10001:
        raise RuntimeError("CMRC residual exceeded its frozen 0.1 bound.")
    saturation = float(aux["residual_saturation_ratio"].cpu())
    if not 0.0 <= saturation <= 1.0:
        raise RuntimeError("CMRC saturation ratio is outside [0,1].")


def create_grad_scaler(enabled: bool):
    amp_namespace = getattr(torch, "amp", None)
    scaler_class = getattr(amp_namespace, "GradScaler", None)
    if scaler_class is not None:
        try:
            return scaler_class("cuda", enabled=enabled, init_scale=128.0)
        except TypeError:
            pass
    return torch.cuda.amp.GradScaler(enabled=enabled, init_scale=128.0)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Check CMRC zero-initialized Fusion equivalence, two-step gradients, "
            "AMP, checkpoint recovery and inference interfaces."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=(
            PROJECT_ROOT
            / "configs"
            / "flame3"
            / "pidnet_s_cmrc_partial_30e.yaml"
        ),
    )
    parser.add_argument(
        "--root-dataset",
        type=Path,
        default=(
            PROJECT_ROOT
            / "transfer"
            / "flame3_4090_bundle_v1_20260731"
        ),
    )
    parser.add_argument(
        "--trainset",
        type=Path,
        default=(
            PROJECT_ROOT
            / "transfer"
            / "flame3_4090_update_split_v2_partial_74bebee"
            / "splits"
            / "portable"
            / "train.csv"
        ),
    )
    parser.add_argument(
        "--pretrained",
        type=Path,
        default=PROJECT_ROOT / "weights" / "PIDNet_S_ImageNet.pth.tar",
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the CMRC engineering check.")
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive.")

    config_path = args.config.resolve()
    config = load_config(config_path)
    if config.get("MODEL") != "pidnet_s_cmrc" or config.get("MODE") != "fusion":
        raise ValueError("The checker requires the FLAME3 Fusion CMRC config.")
    expected_definition = {
        "CMRC_INSERTION": "layer2_output_before_pid_branches_1_8",
        "CMRC_HINT_STEM_CHANNELS": 8,
        "CMRC_HINT_CHANNELS": 16,
        "CMRC_CONTEXT_CHANNELS": 16,
        "CMRC_CORRECTION_HIDDEN_CHANNELS": 16,
        "CMRC_RESIDUAL_LIMIT": 0.1,
        "CMRC_ALIGNMENT": "adaptive_avg_pool_hint_1_4_to_layer2_1_8",
        "CMRC_FINAL_INITIALIZATION": "final_conv_weight_and_bias_zero",
        "CMRC_RESIDUAL": "x2_plus_0.1_tanh_head_without_post_relu",
    }
    for key, expected in expected_definition.items():
        if config.get(key) != expected:
            raise ValueError(f"Frozen CMRC definition differs for {key}.")
    if not bool(
        config.get("CMRC_STANDALONE_NO_MRFF_NTS_ABL_SAMF_BOUNDARY_MODULE", False)
    ):
        raise ValueError("CMRC must remain the only added experimental module.")
    for path in (args.root_dataset, args.trainset, args.pretrained):
        if not path.resolve().exists():
            raise FileNotFoundError(path.resolve())

    config["ROOTDATASET"] = str(args.root_dataset.resolve())
    config["TRAINSET"] = str(args.trainset.resolve())
    config["PRETRAINED"] = str(args.pretrained.resolve())
    config["BATCHSIZE"] = int(args.batch_size)
    config["NUM_WORKERS"] = 0
    seed = int(config["SEED"])
    seed_everything(seed)
    device = torch.device(config["DEVICE"])

    dataset = build_dataset(config, "train")
    loader = DataLoader(
        dataset,
        batch_size=int(config["BATCHSIZE"]),
        shuffle=False,
        num_workers=0,
    )
    images, labels, edges, _sample_keys, fire_folder_flags = next(iter(loader))
    images = images.to(device=device, dtype=torch.float32)
    labels = labels.to(device=device, dtype=torch.long)
    edges = edges.to(device=device, dtype=torch.float32)
    fire_folder_flags = fire_folder_flags.to(device=device, dtype=torch.bool)
    if list(images.shape[-2:]) != [512, 640]:
        raise RuntimeError(f"Engineering batch is not 640x512: {list(images.shape)}")

    baseline_config = dict(config)
    baseline_config["MODEL"] = "pidnet_s"
    seed_everything(seed)
    baseline = build_model(baseline_config, augment=True)
    baseline_matched = load_pretrained_if_available(baseline, baseline_config)
    seed_everything(seed)
    candidate = build_model(config, augment=True)
    candidate_matched = load_pretrained_if_available(candidate, config)
    if not isinstance(candidate, PIDNetCMRC) or count_cmrc_modules(candidate) != 1:
        raise RuntimeError("Model factory did not create exactly one CMRC model.")
    if baseline_matched != candidate_matched:
        raise RuntimeError("CMRC changed the matched pretrained tensor count.")
    common_state_tensors = load_common_baseline_state(baseline, candidate)
    if common_state_tensors != len(baseline.state_dict()):
        raise RuntimeError("CMRC does not preserve every Fusion state tensor.")
    if bool(candidate.final_correction.weight.detach().count_nonzero()):
        raise RuntimeError("CMRC final correction weight is not zero initialized.")
    if bool(candidate.final_correction.bias.detach().count_nonzero()):
        raise RuntimeError("CMRC final correction bias is not zero initialized.")

    shapes: dict[str, list[int]] = {}
    hooks = [
        candidate.rgb_hint_stem.register_forward_hook(
            lambda _m, _i, output: shapes.__setitem__("rgb_hint_1_4", list(output.shape))
        ),
        candidate.thermal_hint_stem.register_forward_hook(
            lambda _m, _i, output: shapes.__setitem__(
                "thermal_hint_1_4", list(output.shape)
            )
        ),
        candidate.cmrc_context_projection.register_forward_hook(
            lambda _m, _i, output: shapes.__setitem__("context_1_8", list(output.shape))
        ),
        candidate.cmrc_correction_head.register_forward_hook(
            lambda _m, _i, output: shapes.__setitem__(
                "correction_logits_1_8", list(output.shape)
            )
        ),
    ]
    baseline = baseline.to(device).eval().float()
    candidate = candidate.to(device).eval().float()
    with torch.inference_mode():
        baseline_outputs = baseline(images)
        candidate_outputs, zero_aux = candidate(images, return_aux=True)
    for hook in hooks:
        hook.remove()
    if not isinstance(baseline_outputs, list) or not isinstance(candidate_outputs, list):
        raise RuntimeError("CMRC changed the three-output training interface.")
    fp32_differences = [
        max_abs_difference(left, right)
        for left, right in zip(baseline_outputs, candidate_outputs)
    ]
    if max(fp32_differences) >= 1e-6:
        raise RuntimeError(f"CMRC FP32 equivalence failed: {fp32_differences}")
    if not torch.equal(main_logits(baseline_outputs).argmax(1), main_logits(candidate_outputs).argmax(1)):
        raise RuntimeError("CMRC FP32 argmax differs from Fusion at zero initialization.")
    assert_aux_finite(zero_aux)
    if float(zero_aux["residual_abs_max"].cpu()) != 0.0:
        raise RuntimeError("CMRC residual is not exactly zero at initialization.")
    expected_shapes = {
        "rgb_hint_1_4": [args.batch_size, 16, 128, 160],
        "thermal_hint_1_4": [args.batch_size, 16, 128, 160],
        "context_1_8": [args.batch_size, 16, 64, 80],
        "correction_logits_1_8": [args.batch_size, 64, 64, 80],
    }
    if shapes != expected_shapes:
        raise RuntimeError(f"CMRC insertion shapes differ: {shapes} vs {expected_shapes}")

    with torch.inference_mode(), torch.autocast(
        device_type="cuda", dtype=torch.float16, enabled=True
    ):
        baseline_amp = baseline(images)
        candidate_amp, amp_zero_aux = candidate(images, return_aux=True)
    amp_differences = [
        max_abs_difference(left, right)
        for left, right in zip(baseline_amp, candidate_amp)
    ]
    if max(amp_differences) >= 1e-3:
        raise RuntimeError(f"CMRC AMP equivalence failed: {amp_differences}")
    if not torch.equal(main_logits(baseline_amp).argmax(1), main_logits(candidate_amp).argmax(1)):
        raise RuntimeError("CMRC AMP argmax differs from Fusion at zero initialization.")
    assert_aux_finite(amp_zero_aux)

    seed_everything(seed)
    train_model = build_model(config, augment=True)
    train_matched = load_pretrained_if_available(train_model, config)
    if train_matched != baseline_matched:
        raise RuntimeError("CMRC training model changed pretrained matching.")
    train_model = train_model.to(device).train()
    criterion = build_training_criterion(TotalLoss(config), config)
    optimizer = torch.optim.SGD(
        train_model.parameters(),
        lr=float(config["LR"]),
        momentum=float(config["MOMENTUM"]),
        weight_decay=float(config["WD"]),
    )
    scaler = create_grad_scaler(True)

    def backward_step() -> tuple[float, dict[str, float], dict[str, torch.Tensor]]:
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
            outputs, aux = train_model(images, return_aux=True)
            losses, _, _, _ = criterion.get_loss(
                outputs,
                labels,
                edges,
                fire_folder_flags=fire_folder_flags,
            )
            loss = losses.mean()
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("Non-finite CMRC AMP smoke-test loss.")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        gradients = {
            "final_correction": gradient_norm(train_model, "cmrc_correction_head.6."),
            "correction_early": gradient_norm(train_model, "cmrc_correction_head.0."),
            "correction_hidden": gradient_norm(train_model, "cmrc_correction_head.3."),
            "rgb_hint": gradient_norm(train_model, "rgb_hint_stem."),
            "thermal_hint": gradient_norm(train_model, "thermal_hint_stem."),
            "context_projection": gradient_norm(train_model, "cmrc_context_projection."),
            "fusion_stem": gradient_norm(train_model, "conv1."),
            "pidnet_backbone": gradient_norm(train_model, "layer3."),
            "final_head": gradient_norm(train_model, "final_layer."),
        }
        assert_aux_finite(aux)
        return float(loss.detach().cpu()), gradients, aux

    torch.cuda.reset_peak_memory_stats(device)
    first_loss, first_gradients, first_aux = backward_step()
    if first_gradients["final_correction"] <= 0.0:
        raise RuntimeError("CMRC final correction layer received no first-step gradient.")
    blocked_first = {
        name: first_gradients[name]
        for name in (
            "correction_early",
            "correction_hidden",
            "rgb_hint",
            "thermal_hint",
            "context_projection",
        )
    }
    if any(value > 1e-12 for value in blocked_first.values()):
        raise RuntimeError(f"Zero final layer did not block first-step branch gradients: {blocked_first}")
    for name in ("fusion_stem", "pidnet_backbone", "final_head"):
        if first_gradients[name] <= 0.0:
            raise RuntimeError(f"Fusion main path gradient missing on first step: {name}")
    scaler.step(optimizer)
    scaler.update()

    second_loss, second_gradients, second_aux = backward_step()
    for name in (
        "final_correction",
        "correction_early",
        "correction_hidden",
        "rgb_hint",
        "thermal_hint",
        "context_projection",
        "fusion_stem",
        "pidnet_backbone",
        "final_head",
    ):
        if second_gradients[name] <= 0.0:
            raise RuntimeError(f"CMRC second-step gradient missing: {name}")
    scaler.step(optimizer)
    scaler.update()
    peak_memory_mb = torch.cuda.max_memory_allocated(device) / 1024**2

    train_model.eval()
    with torch.inference_mode():
        before_outputs, before_aux = train_model(images, return_aux=True)
    with tempfile.TemporaryDirectory(prefix="flame3_cmrc_check_") as directory:
        checkpoint_path = Path(directory) / "cmrc_state.pth"
        torch.save({"model_state_dict": train_model.state_dict()}, checkpoint_path)
        restored = build_model(config, augment=True).to(device).eval()
        state = torch.load(checkpoint_path, map_location=device, weights_only=False)
        restored.load_state_dict(state["model_state_dict"], strict=True)
        with torch.inference_mode():
            restored_outputs, restored_aux = restored(images, return_aux=True)
    recovery_differences = [
        max_abs_difference(left, right)
        for left, right in zip(before_outputs, restored_outputs)
    ]
    recovery_aux_differences = {
        key: max_abs_difference(before_aux[key], restored_aux[key])
        for key in before_aux
    }
    if max(recovery_differences) != 0.0 or max(recovery_aux_differences.values()) != 0.0:
        raise RuntimeError(
            "CMRC checkpoint recovery changed predictions or aux: "
            f"outputs={recovery_differences}, aux={recovery_aux_differences}"
        )
    default_outputs = train_model(images[:1])
    if isinstance(default_outputs, tuple):
        raise RuntimeError("Default CMRC forward unexpectedly returned aux.")

    inference_model = build_model(config, augment=False)
    inference_state = inference_model.load_state_dict(train_model.state_dict(), strict=False)
    if inference_state.missing_keys:
        raise RuntimeError(f"CMRC inference weights missing: {inference_state.missing_keys}")
    inference_model = inference_model.to(device).eval()
    with torch.inference_mode(), torch.autocast(
        device_type="cuda", dtype=torch.float16, enabled=True
    ):
        inference_output = inference_model(images[:1])
        inference_output_aux, inference_aux = inference_model(
            images[:1], return_aux=True
        )
    if not isinstance(inference_output, torch.Tensor):
        raise RuntimeError("Default CMRC inference must return only final logits.")
    if not isinstance(inference_output_aux, torch.Tensor):
        raise RuntimeError("CMRC inference with aux changed the logits type.")
    assert_aux_finite(inference_aux)

    baseline_training_parameters = sum(parameter.numel() for parameter in baseline.parameters())
    candidate_training_parameters = sum(parameter.numel() for parameter in train_model.parameters())
    baseline_inference = build_model(baseline_config, augment=False)
    candidate_inference = build_model(config, augment=False)
    result = {
        "config": str(config_path),
        "frozen_definition": expected_definition,
        "common_baseline_state_tensors": common_state_tensors,
        "baseline_state_tensor_count": len(baseline.state_dict()),
        "matched_pretrained_tensors": candidate_matched,
        "baseline_matched_pretrained_tensors": baseline_matched,
        "fp32_output_max_abs": fp32_differences,
        "fp32_argmax_identical": True,
        "amp_output_max_abs": amp_differences,
        "amp_argmax_identical": True,
        "zero_residual_aux": {key: float(value.cpu()) for key, value in zero_aux.items()},
        "insertion_shapes": shapes,
        "first_amp_loss": first_loss,
        "first_gradient_norms": first_gradients,
        "first_blocked_branch_gradient_norms": blocked_first,
        "second_amp_loss": second_loss,
        "second_gradient_norms": second_gradients,
        "first_aux": {key: float(value.cpu()) for key, value in first_aux.items()},
        "second_aux": {key: float(value.cpu()) for key, value in second_aux.items()},
        "peak_allocated_gpu_memory_mb": peak_memory_mb,
        "checkpoint_output_max_abs": recovery_differences,
        "checkpoint_aux_max_abs": recovery_aux_differences,
        "default_training_forward_has_aux": False,
        "default_inference_returns_tensor_only": True,
        "inference_aux_available_explicitly": True,
        "inference_output_shape": list(inference_output.shape),
        "training_parameters": candidate_training_parameters,
        "training_parameter_increment": (
            candidate_training_parameters - baseline_training_parameters
        ),
        "inference_parameters": sum(parameter.numel() for parameter in candidate_inference.parameters()),
        "inference_parameter_increment": (
            sum(parameter.numel() for parameter in candidate_inference.parameters())
            - sum(parameter.numel() for parameter in baseline_inference.parameters())
        ),
        "test_images_or_labels_read": False,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.output:
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print("CMRC engineering pipeline check passed")


if __name__ == "__main__":
    main()
