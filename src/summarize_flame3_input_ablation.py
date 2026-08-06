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


def parse_runs(values: list[str], label: str) -> dict[int, Path]:
    runs: dict[int, Path] = {}
    for value in values:
        seed_text, separator, path_text = value.partition("=")
        if not separator:
            raise ValueError(f"{label} must use SEED=RUN_DIRECTORY: {value}")
        seed = int(seed_text)
        path = Path(path_text).resolve()
        if seed in runs:
            raise ValueError(f"Duplicate {label} seed {seed}")
        if not path.is_dir():
            raise FileNotFoundError(path)
        runs[seed] = path
    if set(runs) != set(SEEDS):
        raise ValueError(f"{label} must contain exactly seeds {SEEDS}")
    return runs


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def effective_pretrained_provenance(run: Path, environment: dict, seed: int) -> dict:
    native_hash = environment.get("pretrained_sha256")
    if native_hash:
        return {
            "sha256": str(native_hash),
            "source": "environment.json",
            "resume_provenance": None,
        }
    provenance_path = run / "resume_provenance.json"
    if not provenance_path.is_file():
        return {
            "sha256": None,
            "source": "missing",
            "resume_provenance": None,
        }
    provenance = read_json(provenance_path)
    if int(provenance.get("checkpoint_seed", -1)) != seed:
        raise RuntimeError(f"Resume provenance seed mismatch: {provenance_path}")
    if int(provenance.get("checkpoint_epoch", 0)) <= 0:
        raise RuntimeError(f"Invalid resume checkpoint epoch: {provenance_path}")
    pretrained_hash = provenance.get("pretrained_sha256")
    if not pretrained_hash:
        raise RuntimeError(f"Missing pretrained hash in {provenance_path}")
    return {
        "sha256": str(pretrained_hash),
        "source": "resume_provenance.json",
        "resume_provenance": provenance,
    }


def read_window(run: Path, first: int = 26, last: int = 30) -> list[dict]:
    records: dict[int, dict] = {}
    for line in (run / "metrics.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        records[int(record["epoch"])] = record
    missing = [epoch for epoch in range(first, last + 1) if epoch not in records]
    if missing:
        raise RuntimeError(f"Missing epochs {missing}: {run}")
    return [records[epoch] for epoch in range(first, last + 1)]


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def summarize(run: Path) -> dict[str, float]:
    records = read_window(run)
    return {
        metric: mean([float(record["validation"][metric]) for record in records])
        for metric in METRICS
    }


def audit_reference(reference: Path, candidate: Path, seed: int) -> dict:
    reference_config = read_json(reference / "resolved_config.json")
    candidate_config = read_json(candidate / "resolved_config.json")
    if int(reference_config["SEED"]) != seed or int(candidate_config["SEED"]) != seed:
        raise RuntimeError(f"Seed mismatch for {seed}")
    mismatches = {
        key: {
            "reference": reference_config.get(key),
            "candidate": candidate_config.get(key),
        }
        for key in PROTOCOL_KEYS
        if reference_config.get(key) != candidate_config.get(key)
    }
    reference_environment = read_json(reference / "environment.json")
    candidate_environment = read_json(candidate / "environment.json")
    split_match = (
        reference_environment.get("dataset_list_sha256")
        == candidate_environment.get("dataset_list_sha256")
    )
    reference_pretrained = effective_pretrained_provenance(
        reference, reference_environment, seed
    )
    candidate_pretrained = effective_pretrained_provenance(
        candidate, candidate_environment, seed
    )
    pretrained_match = (
        reference_pretrained["sha256"] is not None
        and reference_pretrained["sha256"] == candidate_pretrained["sha256"]
    )
    return {
        "seed": seed,
        "passed": not mismatches and split_match and pretrained_match,
        "protocol_key_mismatches": mismatches,
        "dataset_split_hashes_match": split_match,
        "pretrained_sha256_matches": pretrained_match,
        "pretrained_provenance": {
            "reference": reference_pretrained,
            "candidate": candidate_pretrained,
        },
        "mode_difference_expected": {
            "reference": reference_config.get("MODE"),
            "candidate": candidate_config.get("MODE"),
        },
        "stem_policy": {
            "reference_skip": reference_config.get("PRETRAIN_SKIP_KEYS"),
            "candidate_skip": candidate_config.get("PRETRAIN_SKIP_KEYS"),
            "candidate_initialization": candidate_config.get("INPUT_STEM_INITIALIZATION"),
        },
    }


def paired_comparison(fusion: dict[int, dict], other: dict[int, dict]) -> dict:
    pairs = []
    for seed in SEEDS:
        gain = fusion[seed]["fire_iou"] - other[seed]["fire_iou"]
        pairs.append(
            {
                "seed": seed,
                "fusion": fusion[seed],
                "other": other[seed],
                "fire_iou_gain": gain,
            }
        )
    mean_gain = mean([item["fire_iou_gain"] for item in pairs])
    improved = sum(item["fire_iou_gain"] > 0.0 for item in pairs)
    return {
        "pairs": pairs,
        "mean_fire_iou_gain": mean_gain,
        "improved_seed_count": improved,
        "passes_cmrc_complementarity_gate": mean_gain >= 0.005 and improved >= 2,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize frozen FLAME3 RGB/IR/Fusion three-seed input ablation."
    )
    parser.add_argument("--rgb-run", action="append", required=True)
    parser.add_argument("--ir-run", action="append", required=True)
    parser.add_argument("--fusion-run", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    runs = {
        "rgb": parse_runs(args.rgb_run, "rgb-run"),
        "ir": parse_runs(args.ir_run, "ir-run"),
        "fusion": parse_runs(args.fusion_run, "fusion-run"),
    }
    summaries = {
        mode: {seed: summarize(path) for seed, path in mode_runs.items()}
        for mode, mode_runs in runs.items()
    }
    audits = {"rgb": [], "ir": []}
    for mode in ("rgb", "ir"):
        for seed in SEEDS:
            audit = audit_reference(runs["fusion"][seed], runs[mode][seed], seed)
            audits[mode].append(audit)
            if not audit["passed"]:
                raise RuntimeError(f"Protocol audit failed: {audit}")
    fusion_vs_rgb = paired_comparison(summaries["fusion"], summaries["rgb"])
    fusion_vs_ir = paired_comparison(summaries["fusion"], summaries["ir"])
    proceed = bool(
        fusion_vs_rgb["passes_cmrc_complementarity_gate"]
        and fusion_vs_ir["passes_cmrc_complementarity_gate"]
    )
    result = {
        "window": [26, 30],
        "metric": "validation.fire_iou arithmetic mean over five fixed epochs",
        "summaries": summaries,
        "protocol_audits": audits,
        "fusion_vs_rgb": fusion_vs_rgb,
        "fusion_vs_ir": fusion_vs_ir,
        "cmrc_input_complementarity_gate": "pass" if proceed else "stop",
        "rule": (
            "Fusion must exceed both RGB and IR by mean Fire IoU >=0.005 "
            "with at least 2/3 paired seeds improving."
        ),
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
