from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]

COMMON_CRITICAL_KEYS = (
    "DEVICE",
    "NUM_WORKERS",
    "MODEL",
    "NUM_OUTPUTS",
    "PRETRAINED",
    "CHECKPOINT",
    "ROOTDATASET",
    "TRAINSET",
    "VALIDSET",
    "TESTSET",
    "NUM_CLASSES",
    "CLS_NAMES",
    "MODE",
    "MULTISCALE",
    "FLIP",
    "BRIGHTNESS",
    "SINGLE_SOURCE",
    "COMP_MASK",
    "BLEND_IMGS",
    "IGNORE_LABEL",
    "SCALE_FACTOR",
    "CROP_SIZE",
    "BASE_SIZE",
    "CLASS_WEIGHTS",
    "ALIGN_CORNERS",
    "LR",
    "BATCHSIZE",
    "OPTIM",
    "SCHED",
    "USE_OHEM",
    "OHEMTHRES",
    "OHEMKEEP",
    "WD",
    "MOMENTUM",
    "BALANCE_WEIGHTS",
    "T_THRESH_BDLOSS",
    "BD_WEIGHT",
    "SB_WEIGHTS",
    "SCALE_MIN",
    "SCALE_MAX",
    "ADOPTED_PROTOCOL_SHA256",
    "INPUT_CHANNELS",
    "PRETRAINED_STEM_POLICY",
)

CANDIDATE_CRITICAL_KEYS = (
    "TRAINING_OBJECTIVE",
    "BOUNDARY_TOLERANCE",
    "ABL_WEIGHT",
    "ABL_DETACH_NEIGHBORS",
    "ABL_MAX_BOUNDARY_RATIO",
    "ABL_LABEL_SMOOTHING",
    "ABL_LABEL_SMOOTHING_BEHAVIOR",
    "ABL_MAX_CLIP_DISTANCE",
    "ABL_THRESHOLD_SCOPE",
    "ABL_FP32_UNDER_AMP",
    "ABL_UPSTREAM_COMMIT",
    "ABL_UPSTREAM_SOURCE_SHA256",
    "ABL_LINKED_LSSCE_COMMIT",
    "ABL_LINKED_LSSCE_SHA256",
    "SAMF_SMOKE_CLASS",
    "SAMF_THERMAL_CHANNEL",
    "SAMF_BETA_INIT",
    "SAMF_GATE_SOURCE",
    "SAMF_GATE_DETACH",
    "SAMF_THERMAL_ALIGNMENT",
    "SAMF_THERMAL_PROJECTION",
    "SAMF_INSERTION",
    "SAMF_INFERENCE_INTERNAL_HEAD",
    "ABL_SAMF_ONLY_ROUTE_C_COMBINATION",
    "ABL_SAMF_FRESH_IMAGENET_INITIALIZATION",
)


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=PROJECT_ROOT, text=True
    ).strip()


def compare_frozen(source: dict, candidate: dict, keys: tuple[str, ...]) -> dict:
    return {
        key: {"source": source.get(key), "candidate": candidate.get(key)}
        for key in keys
        if source.get(key) != candidate.get(key)
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate terminal three-seed configs before any new run."
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
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    manifest = load_yaml(manifest_path)
    preregistration = Path(manifest["PREREGISTRATION_DOCUMENT"])
    if not preregistration.is_file():
        raise FileNotFoundError(preregistration)
    preregistration_commit = str(manifest["PREREGISTRATION_COMMIT"])
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", preregistration_commit, "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
    )
    preregistration_commit_files = git(
        "show", "--pretty=format:", "--name-only", preregistration_commit
    ).splitlines()
    expected_preregistration_path = str(
        preregistration.resolve().relative_to(PROJECT_ROOT)
    ).replace("\\", "/")
    if expected_preregistration_path not in preregistration_commit_files:
        raise RuntimeError("Preregistration document is not in the pinned commit.")

    expected_seeds = [200, 201, 202]
    if list(manifest["EXPECTED_SEEDS"]) != expected_seeds:
        raise RuntimeError("Terminal seeds differ from the preregistered set.")
    if int(manifest["EXPECTED_EPOCHS"]) != 100:
        raise RuntimeError("Terminal runs must use exactly 100 epochs.")
    decision = manifest["DECISION"]
    expected_decision = {
        "METRICS": ["miou", "iou_fire"],
        "SAMPLE_STANDARD_DEVIATION_DDOF": 1,
        "POOLED_STANDARD_DEVIATION": "equal_n_within_group_pooled_sample_sd",
        "STRICT_COMPARISON": "mean_gain_gt_pooled_sd",
        "OVERALL": (
            "at_least_one_metric_passes_and_neither_mean_declines_"
            "and_rtx2060_fps_ge_30"
        ),
    }
    if decision != expected_decision:
        raise RuntimeError(f"Decision formula changed: {decision}")

    records = []
    new_run_count = 0
    for method_name, method in manifest["METHODS"].items():
        source = load_yaml(Path(method["SOURCE_CONFIG"]))
        runs = method["RUNS"]
        seeds = [int(run["SEED"]) for run in runs]
        if seeds != expected_seeds:
            raise RuntimeError(f"{method_name} seeds differ: {seeds}")
        for run in runs:
            seed = int(run["SEED"])
            config_path = Path(run["CONFIG"])
            config = load_yaml(config_path)
            if config.get("SEED") != seed:
                raise RuntimeError(f"Seed mismatch in {config_path}")
            if config.get("MODEL") != method["MODEL"]:
                raise RuntimeError(f"Model mismatch in {config_path}")
            if config.get("TRAINING_OBJECTIVE") != method["TRAINING_OBJECTIVE"]:
                raise RuntimeError(f"Objective mismatch in {config_path}")
            if int(config.get("EPOCHS", 0)) != 100:
                raise RuntimeError(f"Non-100e config: {config_path}")
            if int(config.get("LR_TOTAL_EPOCHS", config["EPOCHS"])) != 100:
                raise RuntimeError(f"Non-100e LR horizon: {config_path}")
            if config.get("CHECKPOINT") is not None:
                raise RuntimeError(f"Run is not fresh initialized: {config_path}")
            if config.get("MODE") != "fusion":
                raise RuntimeError(f"Non-Fusion terminal run: {config_path}")
            if not bool(config.get("TESTSET_SEALED", run["EXISTING_FROZEN"])):
                raise RuntimeError(f"Test set not explicitly sealed: {config_path}")
            frozen_keys = COMMON_CRITICAL_KEYS
            if method_name == "abl_samf":
                frozen_keys += CANDIDATE_CRITICAL_KEYS
            mismatches = compare_frozen(source, config, frozen_keys)
            if mismatches:
                raise RuntimeError(
                    f"Frozen config changed in {config_path}: {mismatches}"
                )
            run_dir = Path(run["RUN_DIR"])
            if bool(run["EXISTING_FROZEN"]):
                metrics = run_dir / "metrics.jsonl"
                if len(metrics.read_text(encoding="utf-8").splitlines()) != 100:
                    raise RuntimeError("Existing Fusion seed 200 is not 100 epochs.")
            else:
                new_run_count += 1
                forbidden = [
                    run_dir / "metrics.jsonl",
                    run_dir / "best.pth",
                    run_dir / "last.pth",
                ]
                if any(path.exists() for path in forbidden):
                    raise RuntimeError(f"Target run would be overwritten: {run_dir}")
            records.append(
                {
                    "method": method_name,
                    "seed": seed,
                    "config": str(config_path.resolve()),
                    "run_dir": str(run_dir.resolve()),
                    "existing_frozen": bool(run["EXISTING_FROZEN"]),
                    "epochs": int(config["EPOCHS"]),
                    "fresh_initialization": config.get("CHECKPOINT") is None,
                }
            )

    if new_run_count != 5:
        raise RuntimeError(f"Expected five new runs, found {new_run_count}.")

    report = {
        "manifest": str(manifest_path),
        "preregistration_document": str(preregistration.resolve()),
        "preregistration_commit": preregistration_commit,
        "head_at_check": git("rev-parse", "HEAD"),
        "test_set_used": False,
        "expected_seeds": expected_seeds,
        "expected_epochs": 100,
        "new_run_count": new_run_count,
        "decision_formula_frozen": True,
        "runs": records,
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    print("Terminal evaluation preregistration/config check passed")


if __name__ == "__main__":
    main()
