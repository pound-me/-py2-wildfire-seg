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
from custom_models.pidnet_mrff import (
    PIDNetMRFF,
    count_mrff_modules,
)
from train_baseline import compute_nts_loss


def gradient_norm(model: torch.nn.Module, prefix: str) -> float:
    total = 0.0
    for name, parameter in model.named_parameters():
        if name.startswith(prefix) and parameter.grad is not None:
            total += float(parameter.grad.detach().float().square().sum().cpu())
    return total**0.5


def maximum_output_error(left, right) -> float:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return float((left.float() - right.float()).abs().max().cpu())
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        if len(left) != len(right):
            raise RuntimeError("Restored MRFF output counts differ.")
        return max(maximum_output_error(a, b) for a, b in zip(left, right))
    raise TypeError("MRFF output structures differ.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Check MRFF equal-gate equivalence, two-step gradients, NTS gradient "
            "isolation, AMP, checkpoint restore and inference interfaces."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=(
            PROJECT_ROOT
            / "configs"
            / "flame3"
            / "pidnet_s_mrff_partial_30e.yaml"
        ),
    )
    parser.add_argument("--root-dataset", type=Path, required=True)
    parser.add_argument("--trainset", type=Path, required=True)
    parser.add_argument(
        "--pretrained",
        type=Path,
        default=PROJECT_ROOT / "weights" / "PIDNet_S_ImageNet.pth.tar",
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the MRFF engineering check.")
    if args.batch_size < 2:
        raise ValueError("Use batch-size >=2 so training-mode BatchNorm is valid.")

    config_path = args.config.resolve()
    config = load_config(config_path)
    if config.get("MODEL") != "pidnet_s_mrff" or config.get("MODE") != "fusion":
        raise ValueError("The checker requires the FLAME3 MRFF Fusion config.")
    if bool(config.get("MRFF_NTS_ENABLED", False)):
        raise ValueError("The base MRFF checker must use the NTS-disabled config.")
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
    fire_folder_flags = fire_folder_flags.to(device=device, dtype=torch.bool)

    seed_everything(int(config["SEED"]))
    model = build_model(config, augment=True)
    if not isinstance(model, PIDNetMRFF) or count_mrff_modules(model) != 1:
        raise RuntimeError("Model factory did not create exactly one MRFF model.")
    if hasattr(model, "last_gate"):
        raise RuntimeError("MRFF must not cache mutable batch gate state.")
    if not torch.equal(
        model.modality_gate.logits.weight,
        torch.zeros_like(model.modality_gate.logits.weight),
    ) or not torch.equal(
        model.modality_gate.logits.bias,
        torch.zeros_like(model.modality_gate.logits.bias),
    ):
        raise RuntimeError("MRFF final gate convolution is not exactly zero initialized.")

    model = model.to(device).eval().float()
    with torch.inference_mode():
        rgb_feature, thermal_feature = model.extract_modality_features(images)
        fused, weights = model.fuse_modalities(rgb_feature, thermal_feature)
        reference = 0.5 * rgb_feature + 0.5 * thermal_feature
    fp32_equivalence_error = float((fused - reference).abs().max().cpu())
    fp32_weight_sum_error = float((weights.sum(dim=1) - 1.0).abs().max().cpu())
    if fp32_equivalence_error >= 1e-6:
        raise RuntimeError(f"MRFF FP32 equal-fusion error: {fp32_equivalence_error}")
    if fp32_weight_sum_error >= 1e-7:
        raise RuntimeError(f"MRFF FP32 weight-sum error: {fp32_weight_sum_error}")

    with torch.inference_mode(), torch.autocast(
        device_type="cuda", dtype=torch.float16, enabled=True
    ):
        rgb_amp, thermal_amp = model.extract_modality_features(images)
        fused_amp, weights_amp = model.fuse_modalities(rgb_amp, thermal_amp)
        reference_amp = 0.5 * rgb_amp + 0.5 * thermal_amp
    amp_equivalence_error = float(
        (fused_amp.float() - reference_amp.float()).abs().max().cpu()
    )
    amp_weight_sum_error = float(
        (weights_amp.sum(dim=1) - 1.0).abs().max().cpu()
    )
    if amp_equivalence_error >= 1e-3:
        raise RuntimeError(f"MRFF AMP equal-fusion error: {amp_equivalence_error}")
    if amp_weight_sum_error >= 1e-6:
        raise RuntimeError(f"MRFF AMP weight-sum error: {amp_weight_sum_error}")

    seed_everything(int(config["SEED"]))
    train_model = build_model(config, augment=True)
    matched_pretrained = load_pretrained_if_available(train_model, config)
    baseline_config = dict(config)
    baseline_config["MODEL"] = "pidnet_s"
    baseline_config["PRETRAIN_SKIP_KEYS"] = ["conv1.0.weight"]
    seed_everything(int(config["SEED"]))
    baseline = build_model(baseline_config, augment=True)
    baseline_matched_pretrained = load_pretrained_if_available(
        baseline,
        baseline_config,
    )
    checkpoint = torch.load(
        args.pretrained.resolve(),
        map_location="cpu",
        weights_only=False,
    )
    source_state = checkpoint.get("state_dict", checkpoint)
    candidate_state = train_model.state_dict()
    baseline_state = baseline.state_dict()
    shared_body_keys = sorted(
        key
        for key, value in source_state.items()
        if not key.startswith("conv1.")
        and key in candidate_state
        and key in baseline_state
        and candidate_state[key].shape == value.shape
        and baseline_state[key].shape == value.shape
    )
    if len(shared_body_keys) != matched_pretrained:
        raise RuntimeError(
            "MRFF pretrained matches must be exactly the shared post-stem PIDNet body: "
            f"matches={matched_pretrained}, shared_body={len(shared_body_keys)}."
        )
    for key in shared_body_keys:
        if not torch.equal(candidate_state[key].cpu(), source_state[key].cpu()):
            raise RuntimeError(f"MRFF did not load shared pretrained tensor: {key}")
        if not torch.equal(baseline_state[key].cpu(), source_state[key].cpu()):
            raise RuntimeError(f"Fusion baseline did not load shared tensor: {key}")
    train_model = train_model.to(device).train().float()
    criterion = build_training_criterion(TotalLoss(config), config)
    optimizer = torch.optim.SGD(train_model.parameters(), lr=0.01, momentum=0.0)

    def semantic_loss(current_model: torch.nn.Module) -> torch.Tensor:
        outputs, aux = current_model(images, return_aux=True)
        if set(aux) != {"modality_weights"}:
            raise RuntimeError(f"Unexpected MRFF aux keys: {sorted(aux)}")
        losses, _metric_outputs, _accuracy, _components = criterion.get_loss(
            outputs,
            labels,
            edges,
            fire_folder_flags,
        )
        return losses.mean()

    optimizer.zero_grad(set_to_none=True)
    first_loss = semantic_loss(train_model)
    if not torch.isfinite(first_loss):
        raise RuntimeError("MRFF first-step FP32 loss is non-finite.")
    first_loss.backward()
    first_gradients = {
        "gate_final": gradient_norm(train_model, "modality_gate.logits."),
        "gate_earlier": gradient_norm(train_model, "modality_gate.context."),
        "rgb_stem": gradient_norm(train_model, "rgb_stem."),
        "thermal_stem": gradient_norm(train_model, "thermal_stem."),
        "backbone": gradient_norm(train_model, "layer1."),
    }
    if first_gradients["gate_final"] <= 0.0:
        raise RuntimeError(f"MRFF final gate has no first-step gradient: {first_gradients}")
    if first_gradients["gate_earlier"] > 1e-12:
        raise RuntimeError(
            "MRFF earlier gate should be blocked at exact zero initialization: "
            f"{first_gradients}"
        )
    for name in ("rgb_stem", "thermal_stem", "backbone"):
        if first_gradients[name] <= 0.0:
            raise RuntimeError(f"MRFF first-step gradient missing: {first_gradients}")

    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    second_loss = semantic_loss(train_model)
    if not torch.isfinite(second_loss):
        raise RuntimeError("MRFF second-step FP32 loss is non-finite.")
    second_loss.backward()
    second_gradients = {
        "gate_final": gradient_norm(train_model, "modality_gate.logits."),
        "gate_earlier": gradient_norm(train_model, "modality_gate.context."),
        "rgb_stem": gradient_norm(train_model, "rgb_stem."),
        "thermal_stem": gradient_norm(train_model, "thermal_stem."),
        "backbone": gradient_norm(train_model, "layer1."),
    }
    if any(value <= 0.0 for value in second_gradients.values()):
        raise RuntimeError(f"MRFF second-step gradient missing: {second_gradients}")

    train_model.zero_grad(set_to_none=True)
    _rgb, _thermal = train_model.extract_modality_features(images)
    _fused, nts_weights = train_model.fuse_modalities(_rgb, _thermal)
    synthetic_logits = torch.zeros(
        images.shape[0],
        int(config["NUM_CLASSES"]),
        nts_weights.shape[-2],
        nts_weights.shape[-1],
        device=device,
        requires_grad=True,
    )
    synthetic_logits.data[:, int(config.get("FIRE_CLASS_INDEX", 2))].fill_(5.0)
    nts_loss, nts_selected = compute_nts_loss(
        synthetic_logits,
        nts_weights,
        torch.zeros_like(fire_folder_flags),
        int(config.get("FIRE_CLASS_INDEX", 2)),
    )
    nts_loss.backward()
    nts_gradients = {
        "gate_final": gradient_norm(train_model, "modality_gate.logits."),
        "gate_earlier": gradient_norm(train_model, "modality_gate.context."),
        "rgb_stem": gradient_norm(train_model, "rgb_stem."),
        "thermal_stem": gradient_norm(train_model, "thermal_stem."),
        "backbone": gradient_norm(train_model, "layer1."),
    }
    if any(nts_gradients[name] <= 0.0 for name in (
        "gate_final", "gate_earlier", "rgb_stem", "thermal_stem"
    )):
        raise RuntimeError(f"NTS did not reach the gate and both stems: {nts_gradients}")
    if nts_gradients["backbone"] != 0.0:
        raise RuntimeError(f"NTS unexpectedly reached PIDNet logits/body: {nts_gradients}")
    if synthetic_logits.grad is not None:
        raise RuntimeError("NTS must not backpropagate into segmentation logits.")
    if int(nts_selected) <= 0:
        raise RuntimeError("Synthetic NTS check selected no predicted-Fire pixels.")

    amp_model = build_model(config, augment=True).to(device).train()
    load_pretrained_if_available(amp_model, config)
    amp_model.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
        amp_outputs, amp_aux = amp_model(images, return_aux=True)
        amp_losses, _metric, _accuracy, _components = criterion.get_loss(
            amp_outputs,
            labels,
            edges,
            fire_folder_flags,
        )
        amp_loss = amp_losses.mean()
    if not torch.isfinite(amp_loss):
        raise RuntimeError("MRFF AMP loss is non-finite.")
    amp_loss.backward()
    for name, parameter in amp_model.named_parameters():
        if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()):
            raise RuntimeError(f"MRFF AMP gradient is non-finite: {name}")
    if not bool(torch.isfinite(amp_aux["modality_weights"]).all()):
        raise RuntimeError("MRFF AMP gate weights are non-finite.")

    train_model.eval()
    random_generator = torch.Generator(device=device)
    random_generator.manual_seed(int(config["SEED"]) + 1701)
    precision_inputs = [
        ("real_flame3_batch", images),
        (
            "random_128x160",
            torch.randn(
                1,
                4,
                128,
                160,
                device=device,
                generator=random_generator,
            ),
        ),
        (
            "random_256x256",
            torch.randn(
                1,
                4,
                256,
                256,
                device=device,
                generator=random_generator,
            ),
        ),
    ]

    def compare_gate_precision_case(
        name: str,
        sample: torch.Tensor,
        amp: bool,
    ) -> dict:
        with torch.inference_mode(), torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=amp,
        ):
            rgb_feature, thermal_feature = train_model.extract_modality_features(
                sample
            )
            native_fused, native_weights = train_model.fuse_modalities(
                rgb_feature,
                thermal_feature,
            )
            context = torch.cat(
                (
                    rgb_feature,
                    thermal_feature,
                    (rgb_feature - thermal_feature).abs(),
                ),
                dim=1,
            )
            gate_logits = train_model.modality_gate.logits(
                train_model.modality_gate.context(context)
            )
            reference_probabilities = torch.softmax(
                gate_logits.float(),
                dim=1,
            )
            reference_thermal = reference_probabilities[:, 1:2]
            reference_weights = torch.cat(
                (
                    torch.ones_like(reference_thermal) - reference_thermal,
                    reference_thermal,
                ),
                dim=1,
            )
            reference_feature_weights = reference_weights.to(
                dtype=rgb_feature.dtype
            )
            reference_fused = (
                reference_feature_weights[:, 0:1] * rgb_feature
                + reference_feature_weights[:, 1:2] * thermal_feature
            )
            native_outputs = super(PIDNetMRFF, train_model).forward(native_fused)
            reference_outputs = super(PIDNetMRFF, train_model).forward(
                reference_fused
            )
        native_sum_error = float(
            (native_weights.float().sum(dim=1) - 1.0).abs().max().cpu()
        )
        record = {
            "name": name,
            "precision": "amp_fp16" if amp else "fp32",
            "gate_weight_max_abs_error": float(
                (
                    native_weights.float()
                    - reference_weights.float()
                ).abs().max().cpu()
            ),
            "fused_feature_max_abs_error": float(
                (native_fused.float() - reference_fused.float()).abs().max().cpu()
            ),
            "final_outputs_max_abs_error": maximum_output_error(
                native_outputs,
                reference_outputs,
            ),
            "argmax_predictions_equal": bool(
                torch.equal(
                    native_outputs[1].argmax(dim=1),
                    reference_outputs[1].argmax(dim=1),
                )
            ),
            "native_weight_sum_max_abs_error": native_sum_error,
            "native_weights_finite": bool(torch.isfinite(native_weights).all()),
            "native_fused_feature_finite": bool(torch.isfinite(native_fused).all()),
            "native_outputs_finite": all(
                bool(torch.isfinite(output).all()) for output in native_outputs
            ),
        }
        if native_sum_error != 0.0:
            raise RuntimeError(
                f"MRFF native gate weights do not sum exactly to one: {record}"
            )
        if not all(
            record[key]
            for key in (
                "native_weights_finite",
                "native_fused_feature_finite",
                "native_outputs_finite",
            )
        ):
            raise RuntimeError(f"MRFF native gate path is non-finite: {record}")
        if name == "real_flame3_batch" and not record["argmax_predictions_equal"]:
            raise RuntimeError(
                "MRFF native AMP gate changed real-sample classes relative to "
                f"the FP32 gate reference: {record}"
            )
        return record

    gate_precision_cases = []
    for case_name, case_input in precision_inputs:
        gate_precision_cases.append(
            compare_gate_precision_case(case_name, case_input, amp=False)
        )
        gate_precision_cases.append(
            compare_gate_precision_case(case_name, case_input, amp=True)
        )
    with torch.inference_mode():
        before_outputs, before_aux = train_model(images, return_aux=True)
    with tempfile.TemporaryDirectory(prefix="pidnet_mrff_restore_") as directory:
        checkpoint_path = Path(directory) / "checkpoint.pth"
        torch.save({"model_state_dict": train_model.state_dict()}, checkpoint_path)
        restored = build_model(config, augment=True).to(device).eval()
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        restored.load_state_dict(checkpoint["model_state_dict"], strict=True)
        with torch.inference_mode():
            restored_outputs, restored_aux = restored(images, return_aux=True)
    checkpoint_output_error = maximum_output_error(before_outputs, restored_outputs)
    checkpoint_aux_error = float(
        (
            before_aux["modality_weights"]
            - restored_aux["modality_weights"]
        ).abs().max().cpu()
    )
    if checkpoint_output_error != 0.0 or checkpoint_aux_error != 0.0:
        raise RuntimeError(
            "MRFF checkpoint restore is not exact: "
            f"outputs={checkpoint_output_error}, aux={checkpoint_aux_error}"
        )

    inference_model = build_model(config, augment=False).to(device).eval()
    inference_model.load_state_dict(train_model.state_dict(), strict=False)
    with torch.inference_mode():
        default_inference = inference_model(images[:1])
        inference_with_aux = inference_model(images[:1], return_aux=True)
    if not isinstance(default_inference, torch.Tensor):
        raise RuntimeError("Default MRFF inference returned training-only outputs.")
    if not (
        isinstance(inference_with_aux, tuple)
        and len(inference_with_aux) == 2
        and set(inference_with_aux[1]) == {"modality_weights"}
    ):
        raise RuntimeError("Explicit MRFF inference aux interface is invalid.")
    if hasattr(inference_model, "seghead_p") or hasattr(inference_model, "seghead_d"):
        raise RuntimeError("MRFF deployment model retained PIDNet auxiliary heads.")

    baseline_parameters = sum(parameter.numel() for parameter in baseline.parameters())
    mrff_parameters = sum(parameter.numel() for parameter in train_model.parameters())
    mrff_deployment_parameters = sum(
        parameter.numel() for parameter in inference_model.parameters()
    )
    result = {
        "device": torch.cuda.get_device_name(device),
        "input_shape": list(images.shape),
        "matched_pretrained_tensors": matched_pretrained,
        "baseline_matched_pretrained_tensors": baseline_matched_pretrained,
        "shared_post_stem_pretrained_tensors": len(shared_body_keys),
        "fp32_equal_fusion_max_abs_error": fp32_equivalence_error,
        "amp_equal_fusion_max_abs_error": amp_equivalence_error,
        "fp32_weight_sum_max_abs_error": fp32_weight_sum_error,
        "amp_weight_sum_max_abs_error": amp_weight_sum_error,
        "first_step_gradients": first_gradients,
        "second_step_gradients": second_gradients,
        "nts_only_gradients": nts_gradients,
        "nts_selected_pixels": int(nts_selected),
        "amp_loss": float(amp_loss.detach().cpu()),
        "checkpoint_output_max_abs_error": checkpoint_output_error,
        "checkpoint_aux_max_abs_error": checkpoint_aux_error,
        "gate_precision_path_cases": gate_precision_cases,
        "baseline_training_parameters": baseline_parameters,
        "mrff_training_parameters": mrff_parameters,
        "mrff_deployment_parameters": mrff_deployment_parameters,
        "parameter_increase": mrff_parameters - baseline_parameters,
        "default_inference_tensor_only": True,
        "explicit_aux_shape": list(inference_with_aux[1]["modality_weights"].shape),
        "no_mutable_last_gate": not hasattr(train_model, "last_gate"),
        "status": "passed",
    }
    output = (
        args.output.resolve()
        if args.output
        else PROJECT_ROOT / "experiments" / "mrff_engineering_check.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Result: {output}")


if __name__ == "__main__":
    main()
