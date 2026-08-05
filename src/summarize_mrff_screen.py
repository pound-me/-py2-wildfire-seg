from __future__ import annotations

import argparse
import json
from pathlib import Path


SEEDS = (200, 201, 202)
METRICS = (
    "fire_iou",
    "fire_precision",
    "fire_recall",
    "no_fire_predicted_fire_ratio",
    "empty_fire_predicted_fire_ratio",
)
PROTOCOL_KEYS = (
    "DATASET_TYPE",
    "TRAINSET",
    "VALIDSET",
    "TESTSET",
    "MODE",
    "MULTISCALE",
    "FLIP",
    "BRIGHTNESS",
    "SCALE_MIN",
    "SCALE_MAX",
    "CROP_SIZE",
    "BASE_SIZE",
    "TRAINING_OBJECTIVE",
    "BALANCE_WEIGHTS",
    "T_THRESH_BDLOSS",
    "BD_WEIGHT",
    "SB_WEIGHTS",
    "BATCHSIZE",
    "LR",
    "LR_TOTAL_EPOCHS",
    "WD",
    "MOMENTUM",
    "AMP_INIT_SCALE",
    "SELECTION_METRIC",
    "METRIC_PROTOCOL",
)


def parse_seed_paths(values: list[str], label: str) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for value in values:
        seed_text, separator, path_text = value.partition("=")
        if not separator:
            raise ValueError(f"{label} must use SEED=RUN_DIRECTORY syntax: {value}")
        seed = int(seed_text)
        if seed in result:
            raise ValueError(f"Duplicate {label} seed: {seed}")
        path = Path(path_text).resolve()
        if not path.is_dir():
            raise FileNotFoundError(path)
        result[seed] = path
    if set(result) != set(SEEDS):
        raise ValueError(f"{label} must contain exactly seeds {SEEDS}.")
    return result


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_window(run_directory: Path, first_epoch: int, last_epoch: int) -> list[dict]:
    records: dict[int, dict] = {}
    metrics_path = run_directory / "metrics.jsonl"
    if not metrics_path.is_file():
        raise FileNotFoundError(metrics_path)
    for line in metrics_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        epoch = int(record["epoch"])
        if epoch in records:
            raise RuntimeError(f"Duplicate epoch {epoch}: {metrics_path}")
        records[epoch] = record
    required = set(range(first_epoch, last_epoch + 1))
    missing = sorted(required.difference(records))
    if missing:
        raise RuntimeError(f"Missing epochs {missing}: {metrics_path}")
    return [records[epoch] for epoch in range(first_epoch, last_epoch + 1)]


def mean(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


def summarize_window(records: list[dict]) -> dict:
    summary = {
        key: mean([float(record["validation"][key]) for record in records])
        for key in METRICS
    }
    gates = [record.get("modality_gate", {}).get("validation") for record in records]
    gates = [gate for gate in gates if gate is not None]
    if gates:
        summary["gate"] = {
            "rgb_mean": mean(
                [float(gate["modalities"]["rgb"]["mean"]) for gate in gates]
            ),
            "thermal_mean": mean(
                [float(gate["modalities"]["thermal"]["mean"]) for gate in gates]
            ),
            "thermal_std": mean(
                [float(gate["modalities"]["thermal"]["std"]) for gate in gates]
            ),
            "thermal_fire_core_mean": mean(
                [
                    float(gate["thermal_weight_by_region"]["fire_core"]["mean"])
                    for gate in gates
                ]
            ),
            "thermal_no_fire_mean": mean(
                [
                    float(gate["thermal_weight_by_region"]["no_fire"]["mean"])
                    for gate in gates
                ]
            ),
            "thermal_fire_file_noncore_mean": mean(
                [
                    float(
                        gate["thermal_weight_by_region"]["fire_file_noncore"]["mean"]
                    )
                    for gate in gates
                ]
            ),
            "collapse_epochs": sum(
                bool(gate["collapse_diagnostic"]["all_rgb_like"])
                or bool(gate["collapse_diagnostic"]["all_thermal_like"])
                for gate in gates
            ),
        }
    return summary


def audit_pair(baseline_directory: Path, candidate_directory: Path, seed: int) -> dict:
    baseline_config = read_json(baseline_directory / "resolved_config.json")
    candidate_config = read_json(candidate_directory / "resolved_config.json")
    if int(baseline_config["SEED"]) != seed or int(candidate_config["SEED"]) != seed:
        raise RuntimeError(f"Resolved-config seed mismatch for seed {seed}.")
    mismatches = {
        key: {"baseline": baseline_config.get(key), "candidate": candidate_config.get(key)}
        for key in PROTOCOL_KEYS
        if baseline_config.get(key) != candidate_config.get(key)
    }
    baseline_environment = read_json(baseline_directory / "environment.json")
    candidate_environment = read_json(candidate_directory / "environment.json")
    dataset_hash_match = (
        baseline_environment.get("dataset_list_sha256")
        == candidate_environment.get("dataset_list_sha256")
    )
    pretrained_hash_match = (
        baseline_environment.get("pretrained_sha256")
        == candidate_environment.get("pretrained_sha256")
    )
    passed = not mismatches and dataset_hash_match and pretrained_hash_match
    return {
        "seed": seed,
        "passed": passed,
        "protocol_key_mismatches": mismatches,
        "dataset_split_hashes_match": dataset_hash_match,
        "pretrained_sha256_matches": pretrained_hash_match,
        "expected_architecture_specific_pretrain_difference": {
            "baseline_skip": baseline_config.get("PRETRAIN_SKIP_KEYS"),
            "candidate_skip": candidate_config.get("PRETRAIN_SKIP_KEYS"),
            "candidate_policy": candidate_config.get("MRFF_PRETRAIN_POLICY"),
        },
        "shared_code_audit_scope": (
            "MRFF is selected only by MODEL=pidnet_s_mrff; baseline model, dataset "
            "and partial-label criterion branches remain unchanged by code review."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply the preregistered paired three-seed MRFF screening rule."
    )
    parser.add_argument("--baseline-run", action="append", required=True)
    parser.add_argument("--candidate-run", action="append", required=True)
    parser.add_argument("--first-epoch", type=int, default=26)
    parser.add_argument("--last-epoch", type=int, default=30)
    parser.add_argument("--paired-fps-json", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (args.first_epoch, args.last_epoch) not in {(26, 30), (46, 50)}:
        raise ValueError("Only preregistered windows 26-30 or 46-50 are allowed.")
    baseline_runs = parse_seed_paths(args.baseline_run, "baseline-run")
    candidate_runs = parse_seed_paths(args.candidate_run, "candidate-run")

    pairs = []
    audits = []
    for seed in SEEDS:
        audits.append(audit_pair(baseline_runs[seed], candidate_runs[seed], seed))
        baseline = summarize_window(
            read_window(baseline_runs[seed], args.first_epoch, args.last_epoch)
        )
        candidate = summarize_window(
            read_window(candidate_runs[seed], args.first_epoch, args.last_epoch)
        )
        pairs.append(
            {
                "seed": seed,
                "baseline": baseline,
                "candidate": candidate,
                "fire_iou_gain": candidate["fire_iou"] - baseline["fire_iou"],
            }
        )
    if not all(audit["passed"] for audit in audits):
        raise RuntimeError(f"Fusion/MRFF protocol audit failed: {audits}")

    mean_gain = mean([pair["fire_iou_gain"] for pair in pairs])
    improved_seeds = sum(pair["fire_iou_gain"] > 0.0 for pair in pairs)
    if mean_gain >= 0.005 and improved_seeds >= 2:
        accuracy_decision = "pass"
    elif 0.0 < mean_gain < 0.005:
        accuracy_decision = "borderline_resume_all_seeds_to_50"
    else:
        accuracy_decision = "stop"

    fps_result = None
    fps_pass = None
    if args.paired_fps_json is not None:
        fps_result = read_json(args.paired_fps_json.resolve())
        fps_pass = bool(fps_result.get("route_c_candidate_admitted", False))
    final_decision = accuracy_decision
    if accuracy_decision == "pass":
        final_decision = (
            "pass"
            if fps_pass is True
            else "accuracy_pass_fps_pending_or_failed"
        )
    result = {
        "window": [args.first_epoch, args.last_epoch],
        "metric": "validation.fire_iou arithmetic mean over the fixed five epochs",
        "pairs": pairs,
        "protocol_audits": audits,
        "three_seed_mean_fire_iou_gain": mean_gain,
        "improved_seed_count": improved_seeds,
        "accuracy_decision": accuracy_decision,
        "paired_fps": fps_result,
        "fps_passes_30": fps_pass,
        "final_decision": final_decision,
        "rule": {
            "pass": "mean gain >=0.005 and at least 2/3 seeds improve and FPS >=30",
            "borderline": "0 < mean gain <0.005; exact-resume all seeds to epoch 50",
            "stop": "mean gain <=0 or only one seed improves",
        },
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
