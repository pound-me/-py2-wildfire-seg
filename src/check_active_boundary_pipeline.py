from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from active_boundary_loss import (
    ActiveBoundaryLoss,
    UPSTREAM_COMMIT,
    UPSTREAM_LICENSE,
    UPSTREAM_LICENSE_SHA256,
    UPSTREAM_REPOSITORY,
    UPSTREAM_SOURCE,
    UPSTREAM_SOURCE_SHA256,
)
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


LINKED_LSSCE_COMMIT = "af876e43218694dc8599cc4711d9a5c5e043b1b2"
LINKED_LSSCE_SHA256 = (
    "0367fcad3d1b62283db0a4bb40fe7b5e08095f4ad17db4e6b96b323162ac2251"
)


class OfficialLinkedLabelSmoothSoftmaxCEV1(nn.Module):
    """Exact V1 behavior linked from the pinned official ABL source."""

    def __init__(
        self,
        lb_smooth: float = 0.1,
        reduction: str = "mean",
        ignore_index: int = -100,
    ) -> None:
        super().__init__()
        self.lb_smooth = lb_smooth
        self.reduction = reduction
        self.lb_ignore = ignore_index
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(
        self,
        logits: torch.Tensor,
        label: torch.Tensor,
    ) -> torch.Tensor:
        logits = logits.float()
        with torch.no_grad():
            num_classes = logits.size(1)
            label = label.clone().detach()
            ignore = label.eq(self.lb_ignore)
            n_valid = ignore.eq(0).sum()
            label[ignore] = 0
            positive = 1.0 - self.lb_smooth
            negative = self.lb_smooth / num_classes
            one_hot = (
                torch.empty_like(logits)
                .fill_(negative)
                .scatter_(1, label.unsqueeze(1), positive)
                .detach()
            )
        loss = -torch.sum(self.log_softmax(logits) * one_hot, dim=1)
        loss[ignore] = 0
        if self.reduction == "mean":
            loss = loss.sum() / n_valid
        elif self.reduction == "sum":
            loss = loss.sum()
        return loss


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_output(repository: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def load_pinned_official_abl():
    """Execute the pinned official abl.py with its exact linked V1 dependency."""
    package_name = "_pinned_active_boundary_loss"
    package = types.ModuleType(package_name)
    package.__path__ = [str(UPSTREAM_REPOSITORY)]
    sys.modules[package_name] = package

    label_module = types.ModuleType(f"{package_name}.label_smooth")
    label_module.LabelSmoothSoftmaxCEV1 = OfficialLinkedLabelSmoothSoftmaxCEV1
    sys.modules[label_module.__name__] = label_module

    module_name = f"{package_name}.abl"
    spec = importlib.util.spec_from_file_location(module_name, UPSTREAM_SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load the pinned official ABL source.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    if "bool" not in np.__dict__:
        np.bool = np.bool_  # type: ignore[attr-defined]
    spec.loader.exec_module(module)
    return module


def gradient_norm(model: nn.Module, prefix: str) -> float:
    squared = 0.0
    found = False
    for name, parameter in model.named_parameters():
        if not name.startswith(prefix) or parameter.grad is None:
            continue
        found = True
        squared += float(parameter.grad.float().square().sum().detach().cpu())
    if not found:
        return 0.0
    return squared**0.5


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


def fixed_source_equivalence(device: torch.device) -> dict:
    official_module = load_pinned_official_abl()
    generator = torch.Generator(device="cpu").manual_seed(20220726)
    base_logits = torch.randn(2, 3, 32, 32, generator=generator).to(device)
    target = torch.zeros(2, 32, 32, dtype=torch.long, device=device)
    target[0, 5:25, 6:18] = 1
    target[0, 14:21, 18:29] = 2
    target[1, 3:27, 4:14] = 1
    target[1, 8:29, 17:25] = 2

    official_logits = base_logits.clone().requires_grad_(True)
    official = official_module.ABL(
        isdetach=True,
        max_N_ratio=0.01,
        ignore_label=255,
        label_smoothing=0.2,
        max_clip_dist=20.0,
    ).to(device)
    official_loss = official(official_logits, target)
    if official_loss is None or not torch.isfinite(official_loss):
        raise RuntimeError("Pinned official ABL produced no finite fixed-input loss.")
    official_boundary = official.logits2boundary(official_logits.detach())
    official_loss.backward()
    official_gradient = official_logits.grad.detach().clone()

    adapted_logits = base_logits.clone().requires_grad_(True)
    adapted = ActiveBoundaryLoss(
        ignore_label=255,
        detach_neighbors=True,
        max_boundary_ratio=0.01,
        label_smoothing=0.2,
        max_clip_distance=20.0,
        threshold_scope="source_batch",
    ).to(device)
    adapted_loss, diagnostics = adapted(adapted_logits, target)
    adapted_boundary = adapted.predicted_boundary(adapted_logits.detach())
    adapted_loss.backward()
    adapted_gradient = adapted_logits.grad.detach().clone()

    loss_error = abs(float(official_loss.detach() - adapted_loss.detach()))
    gradient_error = float(
        (official_gradient - adapted_gradient).abs().max().detach().cpu()
    )
    boundary_equal = torch.equal(official_boundary, adapted_boundary)
    if loss_error >= 1e-6:
        raise RuntimeError(f"Official/adapted ABL loss mismatch: {loss_error}")
    if gradient_error >= 1e-6:
        raise RuntimeError(
            f"Official/adapted ABL gradient mismatch: {gradient_error}"
        )
    if not boundary_equal:
        raise RuntimeError("Official/adapted source-batch PDB masks differ.")

    per_image = ActiveBoundaryLoss(
        ignore_label=255,
        detach_neighbors=True,
        max_boundary_ratio=0.01,
        label_smoothing=0.2,
        max_clip_distance=20.0,
        threshold_scope="per_image",
    ).to(device)
    seed_masks = torch.cat(
        [
            per_image.adaptive_boundary_seed(base_logits[index : index + 1])
            for index in range(base_logits.shape[0])
        ],
        dim=0,
    )
    seed_counts = seed_masks.flatten(1).sum(dim=1)
    maximum = base_logits.shape[-2] * base_logits.shape[-1] * 0.01
    if bool((seed_counts > maximum).any()):
        raise RuntimeError(
            f"Per-image ABL seed cap failed: {seed_counts.tolist()} > {maximum}"
        )
    repeated_loss, _ = per_image(base_logits, target)
    repeated_loss_2, _ = per_image(base_logits, target)
    if not torch.equal(repeated_loss, repeated_loss_2):
        raise RuntimeError("Fixed-input per-image ABL is not deterministic.")

    return {
        "official_loss": float(official_loss.detach().cpu()),
        "adapted_source_batch_loss": float(adapted_loss.detach().cpu()),
        "loss_maximum_absolute_error": loss_error,
        "gradient_maximum_absolute_error": gradient_error,
        "predicted_boundary_masks_equal": boundary_equal,
        "source_batch_diagnostics": jsonable(diagnostics),
        "per_image_seed_counts": [int(value) for value in seed_counts.cpu()],
        "per_image_seed_maximum": maximum,
        "per_image_fixed_input_deterministic": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Check pinned ABL provenance, source equivalence, paper per-image "
            "cap, AMP gradients, zero deployment overhead and inference isolation."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=(
            PROJECT_ROOT
            / "configs"
            / "route_a"
            / "pidnet_s_fusion_abl_30e_label_fix.yaml"
        ),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the ABL AMP pipeline check.")
    config = load_config(args.config.resolve())
    if config.get("TRAINING_OBJECTIVE") != "active_boundary":
        raise ValueError("The selected config is not the active-boundary candidate.")
    if not bool(config.get("ABL_ONLY_ROUTE_B_CANDIDATE")):
        raise ValueError("The config is not marked as the sole Route B candidate.")
    if not bool(config.get("TESTSET_SEALED")):
        raise ValueError("The Route B config must explicitly keep the test set sealed.")
    seed_everything(int(config["SEED"]))
    device = torch.device(config["DEVICE"])

    actual_commit = git_output(UPSTREAM_REPOSITORY, "rev-parse", "HEAD")
    if actual_commit != UPSTREAM_COMMIT or actual_commit != config["ABL_UPSTREAM_COMMIT"]:
        raise RuntimeError(
            f"ABL commit mismatch: {actual_commit} vs {UPSTREAM_COMMIT}"
        )
    checkout_status = git_output(UPSTREAM_REPOSITORY, "status", "--porcelain")
    if checkout_status:
        raise RuntimeError("Pinned ABL checkout is not clean:\n" + checkout_status)
    source_hash = sha256(UPSTREAM_SOURCE)
    license_hash = sha256(UPSTREAM_LICENSE)
    if source_hash != UPSTREAM_SOURCE_SHA256:
        raise RuntimeError("Pinned official abl.py SHA256 mismatch.")
    if license_hash != UPSTREAM_LICENSE_SHA256:
        raise RuntimeError("Pinned official ABL LICENSE SHA256 mismatch.")
    if not UPSTREAM_LICENSE.read_text(encoding="utf-8").startswith(
        "                                 Apache License"
    ):
        raise RuntimeError("Pinned ABL checkout does not contain Apache-2.0 text.")
    if config["ABL_LINKED_LSSCE_COMMIT"] != LINKED_LSSCE_COMMIT:
        raise RuntimeError("Linked LSSCE commit metadata mismatch.")
    if config["ABL_LINKED_LSSCE_SHA256"] != LINKED_LSSCE_SHA256:
        raise RuntimeError("Linked LSSCE SHA256 metadata mismatch.")

    equivalence = fixed_source_equivalence(device)

    model = build_model(config, augment=True)
    matched_pretrained = load_pretrained_if_available(model, config)
    training_parameters = sum(parameter.numel() for parameter in model.parameters())
    model = model.to(device).train()
    criterion = build_training_criterion(TotalLoss(config), config)
    if sum(parameter.numel() for parameter in criterion.active_boundary.parameters()):
        raise RuntimeError("ABL unexpectedly contains learnable parameters.")
    if len(list(criterion.active_boundary.buffers())):
        raise RuntimeError("ABL unexpectedly contains persistent buffers.")
    if hasattr(criterion, "state_dict"):
        raise RuntimeError("Training-only ABL wrapper unexpectedly adds checkpoint state.")

    dataset = build_dataset(config, split="train")
    loader = DataLoader(
        dataset,
        batch_size=int(config["BATCHSIZE"]),
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    batch = None
    selected_batch_index = None
    for batch_index, candidate_batch in enumerate(loader):
        if bool((candidate_batch[2] == 1).any()):
            batch = candidate_batch
            selected_batch_index = batch_index
            break
    if batch is None:
        raise RuntimeError("No training batch contains a positive D-head boundary label.")
    images = batch[0].to(device=device, dtype=torch.float32)
    labels = batch[1].to(device=device, dtype=torch.long)
    edges = batch[2].to(device=device, dtype=torch.float32)

    model.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
        outputs = model(images)
        if not isinstance(outputs, (list, tuple)) or len(outputs) != 3:
            raise RuntimeError("ABL changed the PIDNet training output interface.")
        raw_main_output_shape = tuple(outputs[1].shape)
        losses, metric_outputs, _, components = criterion.get_loss(
            outputs,
            labels,
            edges,
        )
        total_loss = losses.mean()
        active_loss = components["active_boundary"]
    if active_loss.dtype != torch.float32:
        raise RuntimeError("ABL did not stay FP32 under AMP.")
    if not torch.isfinite(total_loss) or not torch.isfinite(active_loss):
        raise RuntimeError("ABL AMP pipeline produced a non-finite loss.")
    if float(components["abl_supervised_boundary_pixels"].detach()) <= 0.0:
        raise RuntimeError("ABL supervised no boundary pixels in the fixed batch.")

    active_loss.backward(retain_graph=True)
    isolated_gradient_norms = {
        "final_layer": gradient_norm(model, "final_layer."),
        "dfm": gradient_norm(model, "dfm."),
        "p_branch_layer5": gradient_norm(model, "layer5_."),
        "i_branch_layer5": gradient_norm(model, "layer5."),
        "d_branch_layer5": gradient_norm(model, "layer5_d."),
        "auxiliary_p_head": gradient_norm(model, "seghead_p."),
        "auxiliary_d_head": gradient_norm(model, "seghead_d."),
    }
    for name in (
        "final_layer",
        "dfm",
        "p_branch_layer5",
        "i_branch_layer5",
        "d_branch_layer5",
    ):
        if isolated_gradient_norms[name] <= 0.0:
            raise RuntimeError(f"Isolated ABL has no {name} gradient.")
    if isolated_gradient_norms["auxiliary_p_head"] != 0.0:
        raise RuntimeError("Isolated ABL unexpectedly trains the auxiliary P head.")
    if isolated_gradient_norms["auxiliary_d_head"] != 0.0:
        raise RuntimeError("Isolated ABL unexpectedly trains the boundary output head.")

    model.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
        full_outputs = model(images)
        full_losses, _, _, full_components = criterion.get_loss(
            full_outputs,
            labels,
            edges,
        )
        full_loss = full_losses.mean()
    full_loss.backward()
    full_boundary_head_gradient = gradient_norm(model, "seghead_d.")
    if full_boundary_head_gradient <= 0.0:
        raise RuntimeError(
            "The full PIDNet+ABL loss lost its D-head supervision: "
            f"boundary_component={float(full_components['boundary'].detach())}, "
            f"edge_values={torch.unique(edges).detach().cpu().tolist()}"
        )

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=float(config["LR"]),
        momentum=float(config["MOMENTUM"]),
        weight_decay=float(config["WD"]),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    probe_parameter = model.final_layer.conv2.weight
    scaler_attempts = []
    optimizer_step_delta = 0.0
    for attempt in range(8):
        probe_before = probe_parameter.detach().clone()
        optimizer.zero_grad(set_to_none=True)
        scale_before = float(scaler.get_scale())
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
            pipeline_outputs = model(images)
            pipeline_losses, _, _, _ = criterion.get_loss(
                pipeline_outputs,
                labels,
                edges,
            )
            pipeline_loss = pipeline_losses.mean()
        scaler.scale(pipeline_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scale_after = float(scaler.get_scale())
        optimizer_step_delta = float(
            (probe_parameter.detach() - probe_before).abs().max().cpu()
        )
        scaler_attempts.append(
            {
                "attempt": attempt + 1,
                "scale_before": scale_before,
                "scale_after": scale_after,
                "parameter_delta": optimizer_step_delta,
            }
        )
        if optimizer_step_delta > 0.0:
            break
    if optimizer_step_delta <= 0.0:
        raise RuntimeError("ABL AMP optimizer/scaler pipeline did not update the model.")

    baseline_amp_config = dict(config)
    baseline_amp_config.pop("TRAINING_OBJECTIVE", None)
    seed_everything(int(config["SEED"]))
    baseline_amp_model = build_model(baseline_amp_config, augment=True)
    baseline_amp_matched = load_pretrained_if_available(
        baseline_amp_model,
        baseline_amp_config,
    )
    if baseline_amp_matched != matched_pretrained:
        raise RuntimeError("ABL and baseline pretrained tensor counts differ.")
    baseline_amp_model = baseline_amp_model.to(device).train()
    baseline_amp_criterion = build_training_criterion(
        TotalLoss(baseline_amp_config),
        baseline_amp_config,
    )
    baseline_amp_optimizer = torch.optim.SGD(
        baseline_amp_model.parameters(),
        lr=float(config["LR"]),
        momentum=float(config["MOMENTUM"]),
        weight_decay=float(config["WD"]),
    )
    baseline_amp_scaler = torch.amp.GradScaler("cuda", enabled=True)
    baseline_probe = baseline_amp_model.final_layer.conv2.weight
    baseline_scaler_attempts = []
    baseline_optimizer_step_delta = 0.0
    for attempt in range(8):
        baseline_before = baseline_probe.detach().clone()
        baseline_amp_optimizer.zero_grad(set_to_none=True)
        scale_before = float(baseline_amp_scaler.get_scale())
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
            baseline_outputs = baseline_amp_model(images)
            baseline_losses, _, _, _ = baseline_amp_criterion.get_loss(
                baseline_outputs,
                labels,
                edges,
            )
            baseline_pipeline_loss = baseline_losses.mean()
        baseline_amp_scaler.scale(baseline_pipeline_loss).backward()
        baseline_amp_scaler.step(baseline_amp_optimizer)
        baseline_amp_scaler.update()
        scale_after = float(baseline_amp_scaler.get_scale())
        baseline_optimizer_step_delta = float(
            (baseline_probe.detach() - baseline_before).abs().max().cpu()
        )
        baseline_scaler_attempts.append(
            {
                "attempt": attempt + 1,
                "scale_before": scale_before,
                "scale_after": scale_after,
                "parameter_delta": baseline_optimizer_step_delta,
            }
        )
        if baseline_optimizer_step_delta > 0.0:
            break
    if baseline_optimizer_step_delta <= 0.0:
        raise RuntimeError("Baseline AMP optimizer/scaler comparison did not update.")
    del baseline_amp_model
    torch.cuda.empty_cache()

    with tempfile.TemporaryDirectory(prefix="pidnet_abl_resume_") as directory:
        checkpoint_path = Path(directory) / "checkpoint.pth"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "criterion_state_dict": None,
            },
            checkpoint_path,
        )
        restored_model = build_model(config, augment=True).to(device)
        restored_optimizer = torch.optim.SGD(
            restored_model.parameters(),
            lr=float(config["LR"]),
            momentum=float(config["MOMENTUM"]),
            weight_decay=float(config["WD"]),
        )
        restored_scaler = torch.amp.GradScaler("cuda", enabled=True)
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )
        restored_model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        restored_optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        restored_scaler.load_state_dict(checkpoint["scaler_state_dict"])
        maximum_restore_error = 0.0
        for name, value in model.state_dict().items():
            restored = restored_model.state_dict()[name]
            if value.is_floating_point():
                maximum_restore_error = max(
                    maximum_restore_error,
                    float((value - restored).abs().max().detach().cpu()),
                )
            elif not torch.equal(value, restored):
                raise RuntimeError(f"Non-floating checkpoint state differs: {name}")
        if maximum_restore_error != 0.0:
            raise RuntimeError(
                f"ABL checkpoint model restore error: {maximum_restore_error}"
            )
        if restored_scaler.state_dict() != scaler.state_dict():
            raise RuntimeError("ABL checkpoint GradScaler state did not restore exactly.")
        if len(restored_optimizer.state_dict()["state"]) != len(
            optimizer.state_dict()["state"]
        ):
            raise RuntimeError("ABL checkpoint optimizer state did not restore.")

    inference_model = build_model(config, augment=False)
    inference_parameters = sum(
        parameter.numel() for parameter in inference_model.parameters()
    )
    baseline_config = dict(config)
    baseline_config.pop("TRAINING_OBJECTIVE", None)
    baseline_inference_model = build_model(baseline_config, augment=False)
    baseline_inference_parameters = sum(
        parameter.numel() for parameter in baseline_inference_model.parameters()
    )
    if inference_parameters != baseline_inference_parameters:
        raise RuntimeError("ABL changed inference parameter count.")
    incompatible = inference_model.load_state_dict(model.state_dict(), strict=False)
    expected_unexpected = {
        key
        for key in model.state_dict()
        if key.startswith("seghead_p.") or key.startswith("seghead_d.")
    }
    if set(incompatible.unexpected_keys) != expected_unexpected:
        raise RuntimeError("ABL inference load produced unexpected state differences.")
    if incompatible.missing_keys:
        raise RuntimeError("ABL inference model has missing segmentation weights.")
    inference_model = inference_model.to(device).eval()
    with torch.inference_mode(), torch.autocast(
        device_type="cuda",
        dtype=torch.float16,
        enabled=True,
    ):
        inference_output = inference_model(images[:1])
    if not isinstance(inference_output, torch.Tensor):
        raise RuntimeError("ABL inference model did not return one tensor.")
    expected_shape = (1, *raw_main_output_shape[1:])
    if tuple(inference_output.shape) != expected_shape:
        raise RuntimeError(
            f"ABL inference shape mismatch: {tuple(inference_output.shape)} "
            f"vs {expected_shape}"
        )

    result = {
        "config": str(args.config.resolve()),
        "official_repository": config["ABL_UPSTREAM_REPOSITORY"],
        "official_commit_expected": UPSTREAM_COMMIT,
        "official_commit_actual": actual_commit,
        "official_checkout_clean": True,
        "official_license": "Apache-2.0",
        "official_source_sha256": source_hash,
        "official_license_sha256": license_hash,
        "linked_lssce_commit": LINKED_LSSCE_COMMIT,
        "linked_lssce_sha256": LINKED_LSSCE_SHA256,
        "source_equivalence": equivalence,
        "matched_pretrained_tensors": matched_pretrained,
        "selected_training_batch_index": selected_batch_index,
        "training_parameters": training_parameters,
        "inference_parameters": inference_parameters,
        "baseline_inference_parameters": baseline_inference_parameters,
        "inference_parameter_increment": (
            inference_parameters - baseline_inference_parameters
        ),
        "amp_total_loss": float(total_loss.detach().cpu()),
        "amp_active_boundary_loss": float(active_loss.detach().cpu()),
        "loss_components": jsonable(components),
        "isolated_abl_gradient_norms": isolated_gradient_norms,
        "full_boundary_head_gradient_norm": full_boundary_head_gradient,
        "amp_optimizer_step_parameter_delta": optimizer_step_delta,
        "amp_optimizer_scaler_attempts": scaler_attempts,
        "baseline_amp_optimizer_step_parameter_delta": (
            baseline_optimizer_step_delta
        ),
        "baseline_amp_optimizer_scaler_attempts": baseline_scaler_attempts,
        "checkpoint_model_maximum_restore_error": maximum_restore_error,
        "checkpoint_optimizer_state_entries": len(optimizer.state_dict()["state"]),
        "checkpoint_scaler_state_restored": True,
        "training_output_shapes": [list(output.shape) for output in outputs],
        "metric_output_shapes": [list(output.shape) for output in metric_outputs],
        "inference_output_shape": list(inference_output.shape),
        "checkpoint_criterion_state": None,
        "inference_returns_tensor_only": True,
        "inference_contains_abl_state": False,
        "testset_sealed": True,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.output:
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    print("Active Boundary Loss pipeline check passed")


if __name__ == "__main__":
    main()
