from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import yaml

from check_terminal_eval_preregistration import (
    CANDIDATE_CRITICAL_KEYS,
    COMMON_CRITICAL_KEYS,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_METRICS = (
    "miou",
    "iou_background",
    "iou_smoke",
    "iou_fire",
    "precision_fire",
    "recall_fire",
    "boundary_f1_fire",
)
DECISION_METRICS = ("miou", "iou_fire")


def sample_standard_deviation(values: list[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.size < 2:
        raise ValueError("Sample standard deviation requires at least two values.")
    return float(array.std(ddof=1))


def pooled_sample_standard_deviation(
    baseline_values: list[float], candidate_values: list[float]
) -> float:
    n_baseline = len(baseline_values)
    n_candidate = len(candidate_values)
    if n_baseline < 2 or n_candidate < 2:
        raise ValueError("Pooled standard deviation requires at least two per group.")
    baseline_sd = sample_standard_deviation(baseline_values)
    candidate_sd = sample_standard_deviation(candidate_values)
    numerator = (
        (n_baseline - 1) * baseline_sd**2
        + (n_candidate - 1) * candidate_sd**2
    )
    return math.sqrt(numerator / (n_baseline + n_candidate - 2))


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_metrics(path: Path) -> list[dict]:
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if [record["epoch"] for record in records] != list(range(1, 101)):
        raise RuntimeError(f"Run is not exactly epochs 1--100: {path}")
    return records


def best_record(records: list[dict]) -> dict:
    best = records[0]
    for record in records[1:]:
        if record["validation"]["miou"] > best["validation"]["miou"]:
            best = record
    return best


def extract_run(method_name: str, method: dict, run: dict) -> dict:
    run_dir = Path(run["RUN_DIR"])
    metrics_path = run_dir / "metrics.jsonl"
    summary_path = run_dir / "run_summary.json"
    environment_path = run_dir / "environment.json"
    resolved_config_path = run_dir / "resolved_config.json"
    checkpoint_path = run_dir / "best.pth"
    for path in (
        metrics_path,
        summary_path,
        environment_path,
        resolved_config_path,
        checkpoint_path,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    records = load_metrics(metrics_path)
    selected = best_record(records)
    summary = load_json(summary_path)
    environment = load_json(environment_path)
    config = load_json(resolved_config_path)
    seed = int(run["SEED"])
    if int(config["SEED"]) != seed or int(environment["seed"]) != seed:
        raise RuntimeError(f"Seed metadata mismatch in {run_dir}")
    if config["MODEL"] != method["MODEL"]:
        raise RuntimeError(f"Model mismatch in {run_dir}")
    if config.get("TRAINING_OBJECTIVE") != method["TRAINING_OBJECTIVE"]:
        raise RuntimeError(f"Objective mismatch in {run_dir}")
    if int(config["EPOCHS"]) != 100:
        raise RuntimeError(f"Non-100e resolved config in {run_dir}")
    if config.get("CHECKPOINT") is not None:
        raise RuntimeError(f"Non-fresh run in {run_dir}")
    selected_miou = float(selected["validation"]["miou"])
    if abs(float(summary["best_validation_miou"]) - selected_miou) > 1e-12:
        raise RuntimeError(f"Run summary best mismatch in {run_dir}")
    validation = selected["validation"]
    values = {metric: float(validation[metric]) for metric in REPORT_METRICS}
    identity_keys = COMMON_CRITICAL_KEYS
    if method_name == "abl_samf":
        identity_keys += CANDIDATE_CRITICAL_KEYS
    training_identity = {key: config.get(key) for key in identity_keys}
    return {
        "method": method_name,
        "seed": seed,
        "run_dir": str(run_dir.resolve()),
        "config": str(Path(run["CONFIG"]).resolve()),
        "checkpoint": str(checkpoint_path.resolve()),
        "best_epoch": int(selected["epoch"]),
        "values": values,
        "elapsed_seconds": float(summary["elapsed_seconds"]),
        "peak_allocated_gpu_memory_mb": float(
            summary["peak_allocated_gpu_memory_mb"]
        ),
        "dataset_list_sha256": environment["dataset_list_sha256"],
        "pretrained_sha256": environment["pretrained_sha256"],
        "config_sha256": environment["config_sha256"],
        "source_sha256": environment["source_sha256"],
        "training_identity": training_identity,
    }


def summarize_method(runs: list[dict]) -> dict:
    aggregate = {}
    for metric in REPORT_METRICS:
        values = [run["values"][metric] for run in runs]
        aggregate[metric] = {
            "values": values,
            "mean": float(np.mean(values)),
            "sample_standard_deviation": sample_standard_deviation(values),
        }
    return {"runs": runs, "aggregate": aggregate}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply the preregistered terminal three-seed decision."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=(
            PROJECT_ROOT
            / "configs"
            / "terminal_eval"
            / "terminal_eval_manifest.yaml"
        ),
    )
    parser.add_argument("--paired-latency", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    expected_seeds = [200, 201, 202]
    methods = {}
    all_runs = []
    for method_name, method in manifest["METHODS"].items():
        runs = [extract_run(method_name, method, run) for run in method["RUNS"]]
        if [run["seed"] for run in runs] != expected_seeds:
            raise RuntimeError(f"Unexpected seeds for {method_name}")
        identities = {
            json.dumps(run["training_identity"], sort_keys=True)
            for run in runs
        }
        if len(identities) != 1:
            raise RuntimeError(
                f"Frozen training configuration differs across {method_name} seeds."
            )
        methods[method_name] = summarize_method(runs)
        all_runs.extend(runs)

    dataset_hashes = {
        json.dumps(run["dataset_list_sha256"], sort_keys=True)
        for run in all_runs
    }
    pretrained_hashes = {run["pretrained_sha256"] for run in all_runs}
    if len(dataset_hashes) != 1:
        raise RuntimeError("Dataset-list hashes differ across terminal runs.")
    if len(pretrained_hashes) != 1:
        raise RuntimeError("Pretrained hashes differ across terminal runs.")

    metric_decisions = {}
    for metric in DECISION_METRICS:
        baseline = methods["fusion_baseline"]["aggregate"][metric]
        candidate = methods["abl_samf"]["aggregate"][metric]
        delta = candidate["mean"] - baseline["mean"]
        pooled = pooled_sample_standard_deviation(
            baseline["values"], candidate["values"]
        )
        if pooled == 0.0:
            ratio: float | str | None = "inf" if delta > 0.0 else None
        else:
            ratio = delta / pooled
        metric_decisions[metric] = {
            "baseline_mean": baseline["mean"],
            "baseline_sample_standard_deviation": baseline[
                "sample_standard_deviation"
            ],
            "candidate_mean": candidate["mean"],
            "candidate_sample_standard_deviation": candidate[
                "sample_standard_deviation"
            ],
            "mean_gain": delta,
            "pooled_sample_standard_deviation": pooled,
            "gain_over_pooled_sd_ratio": ratio,
            "strict_gain_gt_pooled_sd": delta > pooled,
        }

    latency = load_json(args.paired_latency.resolve())
    candidate_fps = float(latency["candidate"]["fps_from_median"])
    speed_pass = bool(latency["route_c_candidate_admitted"]) and candidate_fps >= 30.0
    miou_delta = metric_decisions["miou"]["mean_gain"]
    fire_delta = metric_decisions["iou_fire"]["mean_gain"]
    at_least_one = any(
        metric_decisions[metric]["strict_gain_gt_pooled_sd"]
        for metric in DECISION_METRICS
    )
    main_method = (
        at_least_one
        and miou_delta >= 0.0
        and fire_delta >= 0.0
        and speed_pass
    )

    report = {
        "manifest": str(manifest_path),
        "preregistration_commit": manifest["PREREGISTRATION_COMMIT"],
        "test_set_used": False,
        "checkpoint_selection": "validation_miou_strict_best_same_epoch_for_fire",
        "seeds": expected_seeds,
        "methods": methods,
        "metric_decisions": metric_decisions,
        "baseline_noise_floor": {
            "miou": methods["fusion_baseline"]["aggregate"]["miou"][
                "sample_standard_deviation"
            ],
            "iou_fire": methods["fusion_baseline"]["aggregate"]["iou_fire"][
                "sample_standard_deviation"
            ],
        },
        "paired_latency": str(args.paired_latency.resolve()),
        "candidate_rtx2060_fps": candidate_fps,
        "passes_30_fps": speed_pass,
        "overall_rule": (
            "(pass_miou OR pass_fire) AND delta_miou>=0 AND "
            "delta_fire>=0 AND paired_rtx2060_fps>=30"
        ),
        "abl_samf_is_paper_main_method": main_method,
        "paper_position": (
            "main_method_small_but_above_preregistered_seed_variation"
            if main_method
            else "positive_ablation_diagnosis_modality_asymmetry_negative_controls"
        ),
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
