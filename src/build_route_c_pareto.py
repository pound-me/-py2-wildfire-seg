from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


REQUIRED_FIELDS = (
    "method_id",
    "display_name",
    "stage",
    "split_id",
    "miou",
    "latency_ms_median",
    "parameters",
    "gflops",
    "status",
    "notes",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and plot the Route C validation-mIoU/latency Pareto registry."
    )
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument(
        "--split-id", default="flame2_val_label_fix_seed200"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def optional_float(raw: str, field: str, method_id: str) -> float | None:
    value = raw.strip()
    if not value:
        return None
    try:
        parsed = float(value)
    except ValueError as error:
        raise ValueError(f"Invalid {field} for {method_id}: {raw}") from error
    return parsed


def write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=REQUIRED_FIELDS)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in REQUIRED_FIELDS} for row in rows)


def main() -> None:
    args = parse_args()
    registry_path = args.registry.resolve()
    with registry_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != REQUIRED_FIELDS:
            raise ValueError(
                f"Registry fields must be exactly {REQUIRED_FIELDS}, got {reader.fieldnames}"
            )
        raw_rows = list(reader)
    ids = [row["method_id"].strip() for row in raw_rows]
    if any(not method_id for method_id in ids) or len(ids) != len(set(ids)):
        raise ValueError("method_id values must be non-empty and unique.")

    plotted = []
    pending = []
    for row in raw_rows:
        method_id = row["method_id"].strip()
        split_id = row["split_id"].strip()
        if "test" in split_id.lower():
            raise ValueError(f"Test-set row is prohibited in Pareto selection: {method_id}")
        miou = optional_float(row["miou"], "miou", method_id)
        latency = optional_float(
            row["latency_ms_median"], "latency_ms_median", method_id
        )
        parameters = optional_float(row["parameters"], "parameters", method_id)
        gflops = optional_float(row["gflops"], "gflops", method_id)
        if miou is not None and not 0.0 <= miou <= 1.0:
            raise ValueError(f"mIoU outside [0,1] for {method_id}: {miou}")
        if latency is not None and latency <= 0.0:
            raise ValueError(f"Latency must be positive for {method_id}")
        if parameters is not None and parameters <= 0.0:
            raise ValueError(f"Parameters must be positive for {method_id}")
        if gflops is not None and gflops <= 0.0:
            raise ValueError(f"GFLOPs must be positive for {method_id}")
        normalized = dict(row)
        normalized.update(
            {
                "miou": miou,
                "latency_ms_median": latency,
                "parameters": parameters,
                "gflops": gflops,
            }
        )
        if split_id == args.split_id and miou is not None and latency is not None:
            normalized["fps"] = 1000.0 / latency
            normalized["passes_30_fps"] = latency <= 1000.0 / 30.0
            plotted.append(normalized)
        else:
            reasons = []
            if split_id != args.split_id:
                reasons.append("different_or_missing_split")
            if miou is None:
                reasons.append("missing_miou")
            if latency is None:
                reasons.append("missing_latency")
            normalized["not_plotted_reasons"] = reasons
            pending.append(normalized)

    frontier = []
    best_miou = -1.0
    for row in sorted(plotted, key=lambda item: (item["latency_ms_median"], -item["miou"])):
        if row["miou"] > best_miou:
            frontier.append(row)
            best_miou = row["miou"]
    frontier_ids = {row["method_id"] for row in frontier}
    for row in plotted:
        row["pareto_frontier"] = row["method_id"] in frontier_ids

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "plotted_rows.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as stream:
        fields = [*REQUIRED_FIELDS, "fps", "passes_30_fps", "pareto_frontier"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in plotted)
    write_rows(output_dir / "pending_rows.csv", pending)

    figure, axis = plt.subplots(figsize=(9, 6), dpi=160)
    if plotted:
        for row in plotted:
            marker = "*" if row["pareto_frontier"] else "o"
            axis.scatter(
                row["latency_ms_median"],
                row["miou"],
                marker=marker,
                s=120 if marker == "*" else 55,
            )
            axis.annotate(
                row["display_name"],
                (row["latency_ms_median"], row["miou"]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
            )
        ordered_frontier = sorted(frontier, key=lambda row: row["latency_ms_median"])
        axis.plot(
            [row["latency_ms_median"] for row in ordered_frontier],
            [row["miou"] for row in ordered_frontier],
            linestyle="--",
            linewidth=1.0,
            color="tab:red",
            label="Pareto frontier",
        )
    axis.axvline(1000.0 / 30.0, color="gray", linestyle=":", label="30 FPS gate")
    axis.set_xlabel("RTX 2060 median latency (ms/image, lower is better)")
    axis.set_ylabel("Validation mIoU (higher is better)")
    axis.set_title("Route C validation mIoU–latency Pareto comparison")
    axis.grid(alpha=0.25)
    axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(output_dir / "miou_latency_pareto.png")
    plt.close(figure)

    summary = {
        "registry": str(registry_path),
        "split_id": args.split_id,
        "test_set_used": False,
        "plotted_count": len(plotted),
        "pending_count": len(pending),
        "pareto_frontier_method_ids": [row["method_id"] for row in frontier],
        "real_time_gate_ms": 1000.0 / 30.0,
        "discipline": (
            "Rows without same-split validation mIoU and measured median latency "
            "remain pending and are not assigned invented coordinates."
        ),
    }
    (output_dir / "pareto_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
