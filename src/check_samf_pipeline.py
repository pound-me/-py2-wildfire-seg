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
from custom_models.pidnet_samf import PIDNetSAMF, ThermalStem, count_samf_modules


def gradient_norm(model: torch.nn.Module, prefix: str) -> float:
    total = 0.0
    for name, parameter in model.named_parameters():
        if not name.startswith(prefix) or parameter.grad is None:
            continue
        total += float(parameter.grad.detach().float().pow(2).sum().cpu())
    return total**0.5


def load_checkpoint_state(path: Path) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    if "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    return checkpoint


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Check SAMF zero-beta equivalence, confirmed insertion shapes, AMP "
            "gradients and inference-only interface."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=(
            PROJECT_ROOT
            / "configs"
            / "route_c"
            / "pidnet_s_fusion_samf_30e_label_fix.yaml"
        ),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the SAMF AMP engineering check.")

    config_path = args.config.resolve()
    config = load_config(config_path)
    if config["MODEL"] != "pidnet_s_samf" or config["MODE"] != "fusion":
        raise ValueError("The checker requires the Route C Fusion SAMF config.")
    if config.get("SAMF_GATE_SOURCE") != "pre_injection_pag3_seghead_p_softmax":
        raise ValueError("SAMF gate source differs from the user-confirmed definition.")
    if bool(config.get("SAMF_GATE_DETACH", True)):
        raise ValueError("The confirmed SAMF smoke gate must remain attached.")
    seed_everything(int(config["SEED"]))
    device = torch.device(config["DEVICE"])

    dataset = build_dataset(config, "train")
    loader = DataLoader(
        dataset,
        batch_size=int(config["BATCHSIZE"]),
        shuffle=False,
        num_workers=0,
    )
    images, labels, edges = next(iter(loader))[:3]
    images = images.to(device=device, dtype=torch.float32)
    labels = labels.to(device=device, dtype=torch.long)
    edges = edges.to(device=device, dtype=torch.float32)

    baseline_config = dict(config)
    baseline_config["MODEL"] = "pidnet_s"
    baseline = build_model(baseline_config, augment=True)
    candidate = build_model(config, augment=True)
    if not isinstance(candidate, PIDNetSAMF) or count_samf_modules(candidate) != 1:
        raise RuntimeError("SAMF model factory did not create exactly one SAMF model.")
    checkpoint_path = Path(config["SAMF_EQUIVALENCE_CHECKPOINT"])
    checkpoint_state = load_checkpoint_state(checkpoint_path)
    baseline.load_state_dict(checkpoint_state, strict=False)
    candidate.load_state_dict(checkpoint_state, strict=False)
    if float(candidate.samf_beta.detach()) != 0.0:
        raise RuntimeError("SAMF beta is not zero before equivalence checking.")

    shape_records: dict[str, list[int]] = {}
    hooks = [
        candidate.thermal_stem.register_forward_hook(
            lambda _m, _i, output: shape_records.__setitem__(
                "thermal_stem", list(output.shape)
            )
        ),
        candidate.thermal_projection.register_forward_hook(
            lambda _m, _i, output: shape_records.__setitem__(
                "thermal_projection", list(output.shape)
            )
        ),
        candidate.pag3.register_forward_hook(
            lambda _m, _i, output: shape_records.__setitem__(
                "pag3", list(output.shape)
            )
        ),
        candidate.seghead_p.register_forward_hook(
            lambda _m, _i, output: shape_records.__setitem__(
                "smoke_logits", list(output.shape)
            )
        ),
    ]
    baseline = baseline.to(device).eval().float()
    candidate = candidate.to(device).eval().float()
    with torch.inference_mode():
        baseline_outputs = baseline(images[:2])
        candidate_outputs = candidate(images[:2])
    for hook in hooks:
        hook.remove()
    if not isinstance(baseline_outputs, list) or not isinstance(candidate_outputs, list):
        raise RuntimeError("Training-interface equivalence requires three PIDNet outputs.")
    if len(baseline_outputs) != 3 or len(candidate_outputs) != 3:
        raise RuntimeError("SAMF changed the training output count.")
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
            "SAMF beta=0 is not bitwise equivalent to the Fusion baseline: "
            f"{output_max_abs}"
        )
    expected_shapes = {
        "thermal_stem": [2, 32, 64, 64],
        "thermal_projection": [2, 64, 32, 32],
        "pag3": [2, 64, 32, 32],
        "smoke_logits": [2, 3, 32, 32],
    }
    if shape_records != expected_shapes:
        raise RuntimeError(
            f"SAMF insertion shapes differ: {shape_records} vs {expected_shapes}"
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
        raise RuntimeError("SAMF changed the matched PIDNet pretrained tensor count.")
    amp_model = amp_model.to(device).train()
    criterion = build_training_criterion(TotalLoss(config), config)

    amp_model.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
        outputs_zero = amp_model(images)
        losses_zero, _, _, components_zero = criterion.get_loss(
            outputs_zero,
            labels,
            edges,
        )
        loss_zero = losses_zero.mean()
    if not torch.isfinite(loss_zero):
        raise RuntimeError("Non-finite SAMF AMP loss at beta=0.")
    loss_zero.backward()
    beta_gradient_zero = gradient_norm(amp_model, "samf_beta")
    thermal_gradient_zero = gradient_norm(amp_model, "thermal_stem.")
    projection_gradient_zero = gradient_norm(amp_model, "thermal_projection.")
    if beta_gradient_zero <= 0.0:
        raise RuntimeError("SAMF beta receives no first-step gradient at beta=0.")
    if thermal_gradient_zero != 0.0 or projection_gradient_zero != 0.0:
        raise RuntimeError(
            "Zero beta should block first-step thermal/projection gradients."
        )

    with torch.no_grad():
        amp_model.samf_beta.fill_(0.1)
    amp_model.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
        outputs_active = amp_model(images)
        losses_active, _, _, components_active = criterion.get_loss(
            outputs_active,
            labels,
            edges,
        )
        loss_active = losses_active.mean()
    if not torch.isfinite(loss_active):
        raise RuntimeError("Non-finite SAMF AMP loss at beta=0.1.")
    loss_active.backward()
    active_gradients = {
        "beta": gradient_norm(amp_model, "samf_beta"),
        "thermal_stem": gradient_norm(amp_model, "thermal_stem."),
        "thermal_projection": gradient_norm(amp_model, "thermal_projection."),
        "smoke_head": gradient_norm(amp_model, "seghead_p."),
        "p_branch": gradient_norm(amp_model, "layer3_."),
        "i_branch": gradient_norm(amp_model, "layer3."),
        "d_branch": gradient_norm(amp_model, "layer3_d."),
        "final_head": gradient_norm(amp_model, "final_layer."),
    }
    if any(value <= 0.0 for value in active_gradients.values()):
        raise RuntimeError(f"SAMF active-path gradient missing: {active_gradients}")

    inference_model = build_model(config, augment=False)
    inference_model.load_state_dict(amp_model.state_dict(), strict=False)
    inference_model = inference_model.to(device).eval()
    smoke_head_calls = {"count": 0}
    inference_hook = inference_model.seghead_p.register_forward_hook(
        lambda _m, _i, _o: smoke_head_calls.__setitem__(
            "count", smoke_head_calls["count"] + 1
        )
    )
    with torch.inference_mode(), torch.autocast(
        device_type="cuda", dtype=torch.float16, enabled=True
    ):
        inference_output = inference_model(images[:1])
    inference_hook.remove()
    if not isinstance(inference_output, torch.Tensor):
        raise RuntimeError("SAMF inference must return only the final tensor.")
    if smoke_head_calls["count"] != 1:
        raise RuntimeError("SAMF inference did not use exactly one internal smoke head.")
    if hasattr(inference_model, "seghead_d"):
        raise RuntimeError("SAMF inference retained the unused boundary auxiliary head.")

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
        "confirmed_definition": {
            "gate_source": config["SAMF_GATE_SOURCE"],
            "gate_detached": False,
            "thermal_alignment": config["SAMF_THERMAL_ALIGNMENT"],
            "inference_internal_head": config["SAMF_INFERENCE_INTERNAL_HEAD"],
        },
        "equivalence_checkpoint": str(checkpoint_path.resolve()),
        "beta_zero": float(candidate.samf_beta.detach().cpu()),
        "beta_zero_output_bitwise_equal": output_bitwise_equal,
        "beta_zero_output_max_abs": output_max_abs,
        "insertion_shapes": shape_records,
        "matched_pretrained_tensors": matched_pretrained,
        "baseline_matched_pretrained_tensors": baseline_matched_pretrained,
        "amp_beta_zero_loss": float(loss_zero.detach().cpu()),
        "amp_beta_zero_components": jsonable(components_zero),
        "beta_gradient_at_zero": beta_gradient_zero,
        "thermal_gradient_at_zero": thermal_gradient_zero,
        "projection_gradient_at_zero": projection_gradient_zero,
        "amp_beta_active_loss": float(loss_active.detach().cpu()),
        "amp_beta_active_components": jsonable(components_active),
        "active_gradient_norms": active_gradients,
        "training_output_shapes": [list(output.shape) for output in outputs_active],
        "inference_output_shape": list(inference_output.shape),
        "inference_returns_tensor_only": True,
        "inference_internal_smoke_head_calls": smoke_head_calls["count"],
        "inference_has_boundary_auxiliary_head": False,
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
    print("SAMF engineering pipeline check passed")


if __name__ == "__main__":
    main()
