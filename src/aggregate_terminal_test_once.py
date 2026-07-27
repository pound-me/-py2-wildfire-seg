from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean, stdev


METRICS = (
    "miou",
    "iou_background",
    "iou_smoke",
    "iou_fire",
    "precision_fire",
    "recall_fire",
)


def load_run(path: Path, method: str, expected_seed: int) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("split") != "test":
        raise RuntimeError(f"Not a test result: {path}")
    if int(payload.get("sample_count", -1)) != 200:
        raise RuntimeError(f"Unexpected test sample count: {path}")
    if int(payload.get("seed", -1)) != expected_seed:
        raise RuntimeError(f"Unexpected seed in {path}")
    metrics = payload["metrics"]
    boundary = payload["boundary_metrics"]["fire"]
    values = {name: float(metrics[name]) for name in METRICS}
    values["boundary_f1_fire"] = float(boundary["boundary_f1"])
    return {
        "method": method,
        "seed": expected_seed,
        "path": str(path.resolve()),
        "checkpoint": str(Path(payload["checkpoint"]).resolve()),
        "checkpoint_epoch": int(payload["checkpoint_epoch"]),
        "sample_count": int(payload["sample_count"]),
        "values": values,
    }


def aggregate(runs: list[dict]) -> dict:
    result: dict[str, dict] = {}
    for metric in (*METRICS, "boundary_f1_fire"):
        values = [run["values"][metric] for run in runs]
        result[metric] = {
            "values": values,
            "mean": mean(values),
            "sample_standard_deviation": stdev(values),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate the six frozen terminal test-once results."
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    specifications = {
        "fusion_baseline": {
            200: root / "experiments/route_a_pidnet_s_fusion/route_a_fusion_100e_label_fix_seed200/test_once/metrics.json",
            201: root / "experiments/terminal_eval_pidnet_s_fusion/terminal_eval_fusion_100e_seed201/test_once/metrics.json",
            202: root / "experiments/terminal_eval_pidnet_s_fusion/terminal_eval_fusion_100e_seed202/test_once/metrics.json",
        },
        "abl_samf": {
            200: root / "experiments/terminal_eval_pidnet_s_abl_samf/terminal_eval_abl_samf_100e_seed200/test_once/metrics.json",
            201: root / "experiments/terminal_eval_pidnet_s_abl_samf/terminal_eval_abl_samf_100e_seed201/test_once/metrics.json",
            202: root / "experiments/terminal_eval_pidnet_s_abl_samf/terminal_eval_abl_samf_100e_seed202/test_once/metrics.json",
        },
    }

    methods: dict[str, dict] = {}
    checkpoints: set[str] = set()
    for method, seed_paths in specifications.items():
        runs = [
            load_run(path, method, seed)
            for seed, path in sorted(seed_paths.items())
        ]
        if [run["seed"] for run in runs] != [200, 201, 202]:
            raise RuntimeError(f"Unexpected seed set for {method}")
        for run in runs:
            checkpoint = run["checkpoint"]
            if checkpoint in checkpoints:
                raise RuntimeError(f"Checkpoint evaluated more than once: {checkpoint}")
            checkpoints.add(checkpoint)
        methods[method] = {"runs": runs, "aggregate": aggregate(runs)}

    baseline = methods["fusion_baseline"]["aggregate"]
    candidate = methods["abl_samf"]["aggregate"]
    deltas = {
        metric: candidate[metric]["mean"] - baseline[metric]["mean"]
        for metric in (*METRICS, "boundary_f1_fire")
    }
    output = {
        "scope": "frozen test set evaluated once per checkpoint after validation decision",
        "test_result_may_change_method_identity": False,
        "seeds": [200, 201, 202],
        "methods": methods,
        "candidate_minus_baseline_mean": deltas,
        "all_six_unique_checkpoints": len(checkpoints) == 6,
        "all_runs_have_200_samples": all(
            run["sample_count"] == 200
            for method in methods.values()
            for run in method["runs"]
        ),
    }
    for value in deltas.values():
        if not math.isfinite(value):
            raise RuntimeError("Non-finite aggregate result.")

    destination = args.output.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
