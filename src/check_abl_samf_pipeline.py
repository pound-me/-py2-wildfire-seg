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
from custom_models.pidnet_samf import PIDNetSAMF, count_samf_modules


def gradient_norm(model: torch.nn.Module, prefix: str) -> float:
    squared = 0.0
    for name, parameter in model.named_parameters():
        if name.startswith(prefix) and parameter.grad is not None:
            squared += float(parameter.grad.detach().float().pow(2).sum().cpu())
    return squared**0.5


def max_abs_difference(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.float() - right.float()).abs().max().detach().cpu())


def load_checkpoint_state(path: Path) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    if "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    return checkpoint


def state_signature(model: torch.nn.Module) -> dict[str, tuple[int, ...]]:
    return {name: tuple(value.shape) for name, value in model.state_dict().items()}


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Check the preregistered ABL+SAMF combination without adding a "
            "new inference architecture."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=(
            PROJECT_ROOT
            / "configs"
            / "route_c"
            / "pidnet_s_fusion_abl_samf_30e_label_fix.yaml"
        ),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the ABL+SAMF AMP check.")
    config_path = args.config.resolve()
    config = load_config(config_path)
    required_identity = {
        "MODEL": "pidnet_s_samf",
        "MODE": "fusion",
        "TRAINING_OBJECTIVE": "active_boundary",
        "CHECKPOINT": None,
        "SEED": 200,
        "EPOCHS": 30,
        "LR_TOTAL_EPOCHS": 100,
        "TESTSET_SEALED": True,
        "ABL_SAMF_ONLY_ROUTE_C_COMBINATION": True,
        "ABL_SAMF_FRESH_IMAGENET_INITIALIZATION": True,
    }
    mismatches = {
        key: {"actual": config.get(key), "expected": expected}
        for key, expected in required_identity.items()
        if config.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"ABL+SAMF frozen-config mismatch: {mismatches}")
    if bool(config.get("SAMF_ONLY_ROUTE_C_VARIANT")):
        raise ValueError("Combination config is incorrectly marked SAMF-only.")
    if bool(config.get("ABL_ONLY_ROUTE_B_CANDIDATE")):
        raise ValueError("Combination config is incorrectly marked ABL-only.")

    standalone_samf_config = load_config(Path(config["SAMF_STANDALONE_CONFIG"]))
    frozen_samf_keys = (
        "MODEL",
        "SAMF_SMOKE_CLASS",
        "SAMF_THERMAL_CHANNEL",
        "SAMF_BETA_INIT",
        "SAMF_GATE_SOURCE",
        "SAMF_GATE_DETACH",
        "SAMF_THERMAL_ALIGNMENT",
        "SAMF_THERMAL_PROJECTION",
        "SAMF_INSERTION",
        "SAMF_INFERENCE_INTERNAL_HEAD",
    )
    samf_mismatches = {
        key: {"combination": config.get(key), "standalone": standalone_samf_config.get(key)}
        for key in frozen_samf_keys
        if config.get(key) != standalone_samf_config.get(key)
    }
    if samf_mismatches:
        raise RuntimeError(f"Standalone SAMF definition changed: {samf_mismatches}")

    standalone_abl_config = load_config(Path(config["ABL_STANDALONE_CONFIG"]))
    frozen_abl_keys = (
        "ABL_WEIGHT",
        "ABL_DETACH_NEIGHBORS",
        "ABL_MAX_BOUNDARY_RATIO",
        "ABL_LABEL_SMOOTHING",
        "ABL_LABEL_SMOOTHING_BEHAVIOR",
        "ABL_MAX_CLIP_DISTANCE",
        "ABL_THRESHOLD_SCOPE",
        "ABL_FP32_UNDER_AMP",
        "ABL_UPSTREAM_COMMIT",
        "ABL_LINKED_LSSCE_COMMIT",
    )
    abl_mismatches = {
        key: {"combination": config.get(key), "standalone": standalone_abl_config.get(key)}
        for key in frozen_abl_keys
        if config.get(key) != standalone_abl_config.get(key)
    }
    if abl_mismatches:
        raise RuntimeError(f"Standalone ABL definition changed: {abl_mismatches}")

    seed_everything(int(config["SEED"]))
    device = torch.device(config["DEVICE"])
    dataset = build_dataset(config, "train")
    loader = DataLoader(
        dataset,
        batch_size=int(config["BATCHSIZE"]),
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    images, labels, edges = next(iter(loader))[:3]
    images = images.to(device=device, dtype=torch.float32)
    labels = labels.to(device=device, dtype=torch.long)
    edges = edges.to(device=device, dtype=torch.float32)

    baseline_config = dict(config)
    baseline_config["MODEL"] = "pidnet_s"
    baseline_config.pop("TRAINING_OBJECTIVE", None)
    baseline = build_model(baseline_config, augment=True)
    equivalent_candidate = build_model(config, augment=True)
    if not isinstance(equivalent_candidate, PIDNetSAMF):
        raise RuntimeError("Factory did not create PIDNetSAMF.")
    if count_samf_modules(equivalent_candidate) != 1:
        raise RuntimeError("Combination must contain exactly one SAMF structure.")
    checkpoint_state = load_checkpoint_state(Path(config["SAMF_EQUIVALENCE_CHECKPOINT"]))
    baseline.load_state_dict(checkpoint_state, strict=False)
    equivalent_candidate.load_state_dict(checkpoint_state, strict=False)
    if float(equivalent_candidate.samf_beta.detach()) != 0.0:
        raise RuntimeError("SAMF beta is not zero before equivalence checking.")
    baseline = baseline.to(device).eval().float()
    equivalent_candidate = equivalent_candidate.to(device).eval().float()
    with torch.inference_mode():
        baseline_outputs = baseline(images[:2])
        candidate_outputs = equivalent_candidate(images[:2])
    if not isinstance(baseline_outputs, list) or not isinstance(candidate_outputs, list):
        raise RuntimeError("Beta-zero equivalence requires the training interface.")
    bitwise_equal = [
        bool(torch.equal(left, right))
        for left, right in zip(baseline_outputs, candidate_outputs)
    ]
    maximum_errors = [
        max_abs_difference(left, right)
        for left, right in zip(baseline_outputs, candidate_outputs)
    ]
    if len(bitwise_equal) != 3 or not all(bitwise_equal):
        raise RuntimeError(
            f"ABL+SAMF beta-zero equivalence failed: {maximum_errors}"
        )
    del baseline, equivalent_candidate
    torch.cuda.empty_cache()

    seed_everything(int(config["SEED"]))
    model = build_model(config, augment=True)
    matched_pretrained = load_pretrained_if_available(model, config)
    model = model.to(device).train()
    criterion = build_training_criterion(TotalLoss(config), config)
    if criterion.objective_name != "active_boundary":
        raise RuntimeError("ABL criterion is not active.")
    if sum(parameter.numel() for parameter in criterion.active_boundary.parameters()):
        raise RuntimeError("Training-only ABL unexpectedly has parameters.")
    if list(criterion.active_boundary.buffers()):
        raise RuntimeError("Training-only ABL unexpectedly has persistent buffers.")

    model.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
        outputs = model(images)
        losses, _, _, components = criterion.get_loss(outputs, labels, edges)
        full_loss = losses.mean()
        active_loss = components["active_boundary"]
    if active_loss.dtype != torch.float32:
        raise RuntimeError("ABL did not remain FP32 under AMP.")
    if not torch.isfinite(full_loss) or not torch.isfinite(active_loss):
        raise RuntimeError("ABL+SAMF AMP loss is non-finite.")
    if float(components["abl_supervised_boundary_pixels"].detach()) <= 0.0:
        raise RuntimeError("ABL supervised no pixels in the fixed batch.")
    full_loss.backward()
    beta_zero_gradients = {
        "samf_beta": gradient_norm(model, "samf_beta"),
        "thermal_stem": gradient_norm(model, "thermal_stem."),
        "thermal_projection": gradient_norm(model, "thermal_projection."),
        "final_layer": gradient_norm(model, "final_layer."),
        "dfm": gradient_norm(model, "dfm."),
        "p_branch": gradient_norm(model, "layer5_."),
        "i_branch": gradient_norm(model, "layer5."),
        "d_branch": gradient_norm(model, "layer5_d."),
    }
    for key in ("samf_beta", "final_layer", "dfm", "p_branch", "i_branch", "d_branch"):
        if beta_zero_gradients[key] <= 0.0:
            raise RuntimeError(f"Missing beta-zero combined gradient: {key}")
    if beta_zero_gradients["thermal_stem"] != 0.0 or beta_zero_gradients["thermal_projection"] != 0.0:
        raise RuntimeError("Zero beta must block thermal-path gradients.")

    with torch.no_grad():
        model.samf_beta.fill_(0.1)
    model.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
        active_outputs = model(images)
        _, _, _, active_components = criterion.get_loss(active_outputs, labels, edges)
        isolated_active_loss = active_components["active_boundary"]
    isolated_active_loss.backward()
    active_abl_gradients = {
        "thermal_stem": gradient_norm(model, "thermal_stem."),
        "thermal_projection": gradient_norm(model, "thermal_projection."),
        "smoke_head": gradient_norm(model, "seghead_p."),
        "final_layer": gradient_norm(model, "final_layer."),
        "dfm": gradient_norm(model, "dfm."),
        "p_branch": gradient_norm(model, "layer5_."),
        "i_branch": gradient_norm(model, "layer5."),
        "d_branch": gradient_norm(model, "layer5_d."),
    }
    if any(value <= 0.0 for value in active_abl_gradients.values()):
        raise RuntimeError(
            f"Isolated ABL does not reach the active SAMF segmentation graph: {active_abl_gradients}"
        )

    combination_inference = build_model(config, augment=False)
    standalone_inference = build_model(standalone_samf_config, augment=False)
    if type(combination_inference) is not type(standalone_inference):
        raise RuntimeError("Combination inference class differs from standalone SAMF.")
    if state_signature(combination_inference) != state_signature(standalone_inference):
        raise RuntimeError("Combination inference state signature differs from SAMF.")
    combination_parameters = sum(
        parameter.numel() for parameter in combination_inference.parameters()
    )
    standalone_parameters = sum(
        parameter.numel() for parameter in standalone_inference.parameters()
    )
    if combination_parameters != standalone_parameters:
        raise RuntimeError("Training-only ABL changed inference parameters.")
    if hasattr(combination_inference, "active_boundary"):
        raise RuntimeError("Inference model retained an ABL object.")
    combination_inference.load_state_dict(model.state_dict(), strict=False)
    combination_inference = combination_inference.to(device).eval()
    with torch.inference_mode(), torch.autocast(
        device_type="cuda", dtype=torch.float16, enabled=True
    ):
        inference_output = combination_inference(images[:1])
    if not isinstance(inference_output, torch.Tensor):
        raise RuntimeError("Combination inference must return one tensor only.")
    if hasattr(combination_inference, "seghead_d"):
        raise RuntimeError("Combination inference retained the unused D head.")

    result = {
        "config": str(config_path),
        "test_set_used": False,
        "fresh_imagenet_initialization": config["CHECKPOINT"] is None,
        "factory_model": type(model).__name__,
        "samf_module_count": count_samf_modules(model),
        "matched_pretrained_tensors": matched_pretrained,
        "beta_zero_output_bitwise_equal": bitwise_equal,
        "beta_zero_output_max_abs": maximum_errors,
        "amp_total_loss": float(full_loss.detach().cpu()),
        "amp_active_boundary_loss": float(active_loss.detach().cpu()),
        "amp_components": jsonable(components),
        "beta_zero_combined_gradient_norms": beta_zero_gradients,
        "beta_point_one_isolated_abl_gradient_norms": active_abl_gradients,
        "inference_class_equals_standalone_samf": True,
        "inference_state_signature_equals_standalone_samf": True,
        "combination_inference_parameters": combination_parameters,
        "standalone_samf_inference_parameters": standalone_parameters,
        "inference_parameter_increment_over_samf": 0,
        "inference_returns_tensor_only": True,
        "inference_contains_abl": False,
        "inference_has_boundary_auxiliary_head": False,
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    print("ABL+SAMF engineering pipeline check passed")


if __name__ == "__main__":
    main()
