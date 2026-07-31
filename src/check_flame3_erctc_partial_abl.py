from __future__ import annotations

import argparse
import json
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
from custom_models.pidnet_erctc import PIDNetERCTC, count_erctc_modules


def gradient_norm(model: torch.nn.Module, prefix: str) -> float:
    total = 0.0
    for name, parameter in model.named_parameters():
        if not name.startswith(prefix) or parameter.grad is None:
            continue
        total += float(parameter.grad.detach().float().pow(2).sum().cpu())
    return total**0.5


def max_abs_difference(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.float() - right.float()).abs().max().detach().cpu())


def jsonable(value):
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return float(value.detach().cpu())
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Check partial-label-compatible ABL on ERCTC, including the "
            "Background+Smoke union, AMP gradients and inference neutrality."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=(
            PROJECT_ROOT
            / "configs"
            / "flame3"
            / "pidnet_s_erctc_partial_abl_30e.yaml"
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
        raise RuntimeError("CUDA is required for the ERCTC AMP engineering check.")

    config_path = args.config.resolve()
    config = load_config(config_path)
    if config["MODEL"] != "pidnet_s_erctc" or config["MODE"] != "fusion":
        raise ValueError("The checker requires the FLAME3 Fusion ERCTC config.")
    expected_definition = {
        "ERCTC_INSERTION": "pag3_output_1_8",
        "ERCTC_RGB_STEM": "avgpool4_conv1x1_3to16_bn_relu",
        "ERCTC_THERMAL_FRONTIER": (
            "avgpool4_then_fixed_sobel_abs_dx_plus_abs_dy_times_0.25_clamp_0_1"
        ),
        "ERCTC_THERMAL_STEM": (
            "concat_pooled_ir_frontier_conv1x1_2to16_bn_relu"
        ),
        "ERCTC_CONTEXT": "concat_detail16_rgb16_thermal16_dwconv3x3_bn_relu",
        "ERCTC_REGION_PATH": (
            "signed_2sigmoid_minus1_times_thermal_projection"
        ),
        "ERCTC_FRONTIER_PATH": (
            "adaptive_max_frontier_times_thermal_projection"
        ),
        "ERCTC_RESIDUAL_SCALES": (
            "independent_region_and_frontier_scalars_zero_initialized"
        ),
    }
    for key, expected in expected_definition.items():
        if config.get(key) != expected:
            raise ValueError(f"Frozen ERCTC definition differs for {key}.")
    if bool(config.get("ERCTC_STANDALONE_NO_ABL_SAMF_TGM", True)):
        raise ValueError("The ERCTC+partial-ABL check requires the combination config.")
    if not bool(config.get("PARTIAL_ABL_ENABLED", False)):
        raise ValueError("PARTIAL_ABL_ENABLED must be true.")
    if config.get("PARTIAL_ABL_BINARY_UNION") != "background_smoke_vs_fire":
        raise ValueError("Partial ABL must merge Background and Smoke as Non-fire.")
    if not bool(config.get("PARTIAL_ABL_FIRE_CORE_IMAGES_ONLY", False)):
        raise ValueError("Partial ABL must skip images without a Fire core.")
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    for path in (args.root_dataset, args.trainset, args.pretrained):
        if not path.resolve().exists():
            raise FileNotFoundError(path.resolve())
    config["ROOTDATASET"] = str(args.root_dataset.resolve())
    config["TRAINSET"] = str(args.trainset.resolve())
    config["PRETRAINED"] = str(args.pretrained.resolve())
    config["BATCHSIZE"] = int(args.batch_size)
    config["NUM_WORKERS"] = 0

    seed_everything(int(config["SEED"]))
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
    fire_folder_flags = fire_folder_flags.to(device=device)

    baseline_config = dict(config)
    baseline_config["MODEL"] = "pidnet_s"
    seed_everything(int(config["SEED"]))
    baseline = build_model(baseline_config, augment=True)
    seed_everything(int(config["SEED"]))
    candidate = build_model(config, augment=True)
    if not isinstance(candidate, PIDNetERCTC) or count_erctc_modules(candidate) != 1:
        raise RuntimeError("Model factory did not create exactly one ERCTC model.")
    common_state_tensors = load_common_baseline_state(baseline, candidate)
    if float(candidate.erctc_region_scale.detach()) != 0.0:
        raise RuntimeError("ERCTC region scale is not zero at initialization.")
    if float(candidate.erctc_frontier_scale.detach()) != 0.0:
        raise RuntimeError("ERCTC frontier scale is not zero at initialization.")
    if any(parameter.requires_grad for parameter in candidate.thermal_frontier.parameters()):
        raise RuntimeError("Fixed Sobel frontier unexpectedly has trainable parameters.")

    shape_records: dict[str, list[int]] = {}
    hooks = [
        candidate.thermal_frontier.register_forward_hook(
            lambda _m, _i, output: shape_records.__setitem__(
                "thermal_frontier", list(output.shape)
            )
        ),
        candidate.rgb_context_stem.register_forward_hook(
            lambda _m, _i, output: shape_records.__setitem__(
                "rgb_context", list(output.shape)
            )
        ),
        candidate.thermal_context_stem.register_forward_hook(
            lambda _m, _i, output: shape_records.__setitem__(
                "thermal_context", list(output.shape)
            )
        ),
        candidate.erctc_detail_compression.register_forward_hook(
            lambda _m, _i, output: shape_records.__setitem__(
                "detail_compression", list(output.shape)
            )
        ),
        candidate.erctc_context_depthwise.register_forward_hook(
            lambda _m, _i, output: shape_records.__setitem__(
                "joint_context", list(output.shape)
            )
        ),
        candidate.erctc_region_logits.register_forward_hook(
            lambda _m, _i, output: shape_records.__setitem__(
                "region_logits", list(output.shape)
            )
        ),
        candidate.erctc_thermal_projection.register_forward_hook(
            lambda _m, _i, output: shape_records.__setitem__(
                "thermal_projection", list(output.shape)
            )
        ),
        candidate.pag3.register_forward_hook(
            lambda _m, _i, output: shape_records.__setitem__(
                "pag3", list(output.shape)
            )
        ),
    ]
    baseline = baseline.to(device).eval().float()
    candidate = candidate.to(device).eval().float()
    with torch.inference_mode():
        baseline_outputs = baseline(images)
        candidate_outputs = candidate(images)
    for hook in hooks:
        hook.remove()
    if not isinstance(baseline_outputs, list) or not isinstance(candidate_outputs, list):
        raise RuntimeError("Training-interface equivalence requires three outputs.")
    if len(baseline_outputs) != 3 or len(candidate_outputs) != 3:
        raise RuntimeError("ERCTC changed the PIDNet training output count.")
    output_max_abs = [
        max_abs_difference(left, right)
        for left, right in zip(baseline_outputs, candidate_outputs)
    ]
    output_bitwise_equal = [
        bool(torch.equal(left, right))
        for left, right in zip(baseline_outputs, candidate_outputs)
    ]
    if not all(output_bitwise_equal):
        raise RuntimeError(
            "ERCTC zero scales are not bitwise equivalent to Fusion baseline: "
            f"{output_max_abs}"
        )
    expected_shapes = {
        "thermal_frontier": [args.batch_size, 1, 128, 160],
        "rgb_context": [args.batch_size, 16, 128, 160],
        "thermal_context": [args.batch_size, 16, 128, 160],
        "pag3": [args.batch_size, 64, 64, 80],
        "detail_compression": [args.batch_size, 16, 64, 80],
        "joint_context": [args.batch_size, 48, 64, 80],
        "region_logits": [args.batch_size, 1, 64, 80],
        "thermal_projection": [args.batch_size, 64, 64, 80],
    }
    if shape_records != expected_shapes:
        raise RuntimeError(
            f"ERCTC insertion shapes differ: {shape_records} vs {expected_shapes}"
        )

    seed_everything(int(config["SEED"]))
    amp_model = build_model(config, augment=True)
    matched_pretrained = load_pretrained_if_available(amp_model, config)
    seed_everything(int(config["SEED"]))
    pretrained_baseline = build_model(baseline_config, augment=True)
    baseline_matched_pretrained = load_pretrained_if_available(
        pretrained_baseline,
        baseline_config,
    )
    if matched_pretrained != baseline_matched_pretrained:
        raise RuntimeError("ERCTC changed the matched PIDNet pretrained tensor count.")
    amp_model = amp_model.to(device).train()
    criterion = build_training_criterion(TotalLoss(config), config)
    if criterion.objective_name != "partial_label":
        raise RuntimeError("ERCTC+partial-ABL must retain the partial-label objective.")
    if criterion.partial_active_boundary is None:
        raise RuntimeError("Partial ABL was not constructed.")
    if list(criterion.partial_active_boundary.parameters()):
        raise RuntimeError("Training-only partial ABL unexpectedly has parameters.")
    if list(criterion.partial_active_boundary.buffers()):
        raise RuntimeError("Training-only partial ABL unexpectedly has buffers.")

    amp_model.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
        outputs_zero = amp_model(images)
        losses_zero, _, _, components_zero = criterion.get_loss(
            outputs_zero,
            labels,
            edges,
            fire_folder_flags,
        )
        loss_zero = losses_zero.mean()
    if not torch.isfinite(loss_zero):
        raise RuntimeError("Non-finite ERCTC AMP loss at zero scales.")
    if not torch.isfinite(components_zero["partial_active_boundary"]):
        raise RuntimeError("Partial ABL produced a non-finite FP32 loss under AMP.")
    if float(
        components_zero["partial_abl_supervised_boundary_pixels"].detach().cpu()
    ) <= 0.0:
        raise RuntimeError("Partial ABL supervised no boundary pixels.")
    expected_present_fire_images = float(
        labels.eq(int(config["FIRE_CLASS_INDEX"])).flatten(1).any(dim=1).sum().cpu()
    )
    actual_present_fire_images = float(
        components_zero["partial_abl_present_fire_images"].detach().cpu()
    )
    if actual_present_fire_images != expected_present_fire_images:
        raise RuntimeError(
            "Partial ABL did not select exactly the images containing Fire core."
        )
    with torch.no_grad():
        no_fire_labels = torch.zeros_like(labels)
        no_fire_flags = torch.zeros_like(fire_folder_flags, dtype=torch.bool)
        _, _, _, no_fire_components = criterion.get_loss(
            outputs_zero,
            no_fire_labels,
            edges,
            no_fire_flags,
        )
    if float(no_fire_components["partial_active_boundary"].detach().cpu()) != 0.0:
        raise RuntimeError("Partial ABL did not skip a batch without Fire core.")
    if float(
        no_fire_components["partial_abl_present_fire_images"].detach().cpu()
    ) != 0.0:
        raise RuntimeError("No-Fire batch reported present Fire-core images.")

    with torch.no_grad():
        binary_logits, binary_target = criterion.build_partial_abl_inputs(
            outputs_zero[1], labels
        )
        swapped_logits = outputs_zero[1].detach().clone()
        swapped_logits[:, [0, 1]] = swapped_logits[:, [1, 0]]
        swapped_binary_logits, swapped_binary_target = (
            criterion.build_partial_abl_inputs(swapped_logits, labels)
        )
    union_swap_error = max_abs_difference(binary_logits, swapped_binary_logits)
    if union_swap_error != 0.0:
        raise RuntimeError(
            "Background/Smoke swapping changed the Non-fire union: "
            f"{union_swap_error}"
        )
    if not torch.equal(binary_target, swapped_binary_target):
        raise RuntimeError("Background/Smoke swapping changed the binary target.")
    binary_target_values = sorted(
        int(value) for value in torch.unique(binary_target).detach().cpu().tolist()
    )
    if not set(binary_target_values).issubset({0, 1, int(config["IGNORE_LABEL"])}):
        raise RuntimeError(f"Unexpected partial ABL target IDs: {binary_target_values}")
    loss_zero.backward()
    zero_scale_gradients = {
        "region_scale": gradient_norm(amp_model, "erctc_region_scale"),
        "frontier_scale": gradient_norm(amp_model, "erctc_frontier_scale"),
    }
    inactive_gradients = {
        "rgb_context": gradient_norm(amp_model, "rgb_context_stem."),
        "thermal_context": gradient_norm(amp_model, "thermal_context_stem."),
        "detail_compression": gradient_norm(
            amp_model, "erctc_detail_compression."
        ),
        "context_depthwise": gradient_norm(
            amp_model, "erctc_context_depthwise."
        ),
        "region_logits": gradient_norm(amp_model, "erctc_region_logits."),
        "thermal_projection": gradient_norm(
            amp_model, "erctc_thermal_projection."
        ),
    }
    if any(value <= 0.0 for value in zero_scale_gradients.values()):
        raise RuntimeError(
            f"ERCTC zero-initialized scales receive no gradient: {zero_scale_gradients}"
        )
    if any(value != 0.0 for value in inactive_gradients.values()):
        raise RuntimeError(
            "Zero scales should block first-step branch gradients: "
            f"{inactive_gradients}"
        )

    with torch.no_grad():
        amp_model.erctc_region_scale.fill_(0.1)
        amp_model.erctc_frontier_scale.fill_(0.1)
    amp_model.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
        outputs_active = amp_model(images)
        losses_active, _, _, components_active = criterion.get_loss(
            outputs_active,
            labels,
            edges,
            fire_folder_flags,
        )
        loss_active = losses_active.mean()
    if not torch.isfinite(loss_active):
        raise RuntimeError("Non-finite ERCTC AMP loss with active paths.")
    loss_active.backward()
    active_gradients = {
        "region_scale": gradient_norm(amp_model, "erctc_region_scale"),
        "frontier_scale": gradient_norm(amp_model, "erctc_frontier_scale"),
        "rgb_context": gradient_norm(amp_model, "rgb_context_stem."),
        "thermal_context": gradient_norm(amp_model, "thermal_context_stem."),
        "detail_compression": gradient_norm(
            amp_model, "erctc_detail_compression."
        ),
        "context_depthwise": gradient_norm(
            amp_model, "erctc_context_depthwise."
        ),
        "region_logits": gradient_norm(amp_model, "erctc_region_logits."),
        "thermal_projection": gradient_norm(
            amp_model, "erctc_thermal_projection."
        ),
        "p_branch": gradient_norm(amp_model, "layer3_."),
        "i_branch": gradient_norm(amp_model, "layer3."),
        "d_branch": gradient_norm(amp_model, "layer3_d."),
        "final_head": gradient_norm(amp_model, "final_layer."),
    }
    if any(value <= 0.0 for value in active_gradients.values()):
        raise RuntimeError(f"ERCTC active-path gradient missing: {active_gradients}")

    amp_model.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
        outputs_isolated = amp_model(images)
        isolated_binary_logits, isolated_binary_target = (
            criterion.build_partial_abl_inputs(outputs_isolated[1], labels)
        )
        isolated_partial_abl, isolated_diagnostics = (
            criterion.partial_active_boundary(
                isolated_binary_logits,
                isolated_binary_target,
            )
        )
    if not torch.isfinite(isolated_partial_abl):
        raise RuntimeError("Isolated partial ABL is non-finite.")
    isolated_partial_abl.backward()
    isolated_gradients = {
        "region_scale": gradient_norm(amp_model, "erctc_region_scale"),
        "frontier_scale": gradient_norm(amp_model, "erctc_frontier_scale"),
        "p_branch": gradient_norm(amp_model, "layer3_."),
        "i_branch": gradient_norm(amp_model, "layer3."),
        "d_branch": gradient_norm(amp_model, "layer3_d."),
        "dfm": gradient_norm(amp_model, "dfm."),
        "final_head": gradient_norm(amp_model, "final_layer."),
    }
    if any(value <= 0.0 for value in isolated_gradients.values()):
        raise RuntimeError(
            f"Isolated partial ABL gradient missing: {isolated_gradients}"
        )

    inference_model = build_model(config, augment=False)
    incompatible = inference_model.load_state_dict(
        amp_model.state_dict(),
        strict=False,
    )
    if incompatible.missing_keys:
        raise RuntimeError(
            f"ERCTC inference model missing weights: {incompatible.missing_keys}"
        )
    inference_model = inference_model.to(device).eval()
    with torch.inference_mode(), torch.autocast(
        device_type="cuda", dtype=torch.float16, enabled=True
    ):
        inference_output = inference_model(images[:1])
    if not isinstance(inference_output, torch.Tensor):
        raise RuntimeError("ERCTC inference must return only the final tensor.")
    if hasattr(inference_model, "seghead_p") or hasattr(inference_model, "seghead_d"):
        raise RuntimeError("ERCTC inference retained an auxiliary segmentation head.")

    baseline_training_parameters = sum(
        parameter.numel() for parameter in pretrained_baseline.parameters()
    )
    candidate_training_parameters = sum(
        parameter.numel() for parameter in amp_model.parameters()
    )
    baseline_inference = build_model(baseline_config, augment=False)
    baseline_inference_parameters = sum(
        parameter.numel() for parameter in baseline_inference.parameters()
    )
    candidate_inference_parameters = sum(
        parameter.numel() for parameter in inference_model.parameters()
    )
    result = {
        "config": str(config_path),
        "frozen_definition": {key: config[key] for key in expected_definition},
        "standalone_without_abl_samf_tgm": False,
        "partial_abl_binary_union": config["PARTIAL_ABL_BINARY_UNION"],
        "partial_abl_weight": float(config["PARTIAL_ABL_WEIGHT"]),
        "partial_abl_fire_core_images_only": True,
        "partial_abl_present_fire_images": actual_present_fire_images,
        "no_fire_only_partial_abl_loss": float(
            no_fire_components["partial_active_boundary"].detach().cpu()
        ),
        "partial_abl_trainable_parameters": 0,
        "partial_abl_persistent_buffers": 0,
        "nonfire_union_bg_smoke_swap_max_abs": union_swap_error,
        "partial_abl_binary_target_values": binary_target_values,
        "common_baseline_state_tensors": common_state_tensors,
        "zero_scale_output_bitwise_equal": output_bitwise_equal,
        "zero_scale_output_max_abs": output_max_abs,
        "insertion_shapes": shape_records,
        "fixed_sobel_trainable_parameters": 0,
        "matched_pretrained_tensors": matched_pretrained,
        "baseline_matched_pretrained_tensors": baseline_matched_pretrained,
        "amp_zero_scale_loss": float(loss_zero.detach().cpu()),
        "amp_zero_scale_components": jsonable(components_zero),
        "zero_scale_gradients": zero_scale_gradients,
        "inactive_branch_gradient_norms": inactive_gradients,
        "amp_active_loss": float(loss_active.detach().cpu()),
        "amp_active_components": jsonable(components_active),
        "active_gradient_norms": active_gradients,
        "isolated_partial_abl_loss": float(isolated_partial_abl.detach().cpu()),
        "isolated_partial_abl_diagnostics": jsonable(isolated_diagnostics),
        "isolated_partial_abl_gradient_norms": isolated_gradients,
        "training_output_shapes": [list(output.shape) for output in outputs_active],
        "inference_output_shape": list(inference_output.shape),
        "inference_returns_tensor_only": True,
        "inference_has_auxiliary_heads": False,
        "baseline_training_parameters": baseline_training_parameters,
        "candidate_training_parameters": candidate_training_parameters,
        "training_parameter_increment": (
            candidate_training_parameters - baseline_training_parameters
        ),
        "baseline_inference_parameters": baseline_inference_parameters,
        "candidate_inference_parameters": candidate_inference_parameters,
        "inference_parameter_increment": (
            candidate_inference_parameters - baseline_inference_parameters
        ),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.output:
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print("ERCTC partial-label ABL engineering pipeline check passed")


if __name__ == "__main__":
    main()
