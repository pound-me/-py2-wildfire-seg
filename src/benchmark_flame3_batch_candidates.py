from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root-dataset", type=Path, required=True)
    parser.add_argument("--pretrained", type=Path, required=True)
    parser.add_argument("--train-csv", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--measure-steps", type=int, default=20)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    smoke_script = Path(__file__).resolve().with_name(
        "smoke_test_flame3_partial_training.py"
    )
    results: dict[str, dict[str, object]] = {}
    for batch_size in (4, 8):
        output = output_dir / f"batch_{batch_size}_smoke.json"
        command = [
            sys.executable,
            str(smoke_script),
            "--config",
            str(args.config.resolve()),
            "--root-dataset",
            str(args.root_dataset.resolve()),
            "--pretrained",
            str(args.pretrained.resolve()),
            "--batch-size",
            str(batch_size),
            "--warmup-steps",
            str(args.warmup_steps),
            "--measure-steps",
            str(args.measure_steps),
            "--output",
            str(output),
        ]
        if args.train_csv is not None:
            command.extend(["--train-csv", str(args.train_csv.resolve())])
        completed = subprocess.run(
            command,
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        (output_dir / f"batch_{batch_size}_console.txt").write_text(
            completed.stdout, encoding="utf-8"
        )
        if completed.returncode == 0 and output.is_file():
            result = json.loads(output.read_text(encoding="utf-8"))
            result["process_returncode"] = completed.returncode
        else:
            result = {
                "status": "failed",
                "batch_size": batch_size,
                "process_returncode": completed.returncode,
                "console_log": str(output_dir / f"batch_{batch_size}_console.txt"),
            }
        results[str(batch_size)] = result

    def eligible(batch_size: int) -> bool:
        result = results[str(batch_size)]
        return (
            result.get("status") == "passed"
            and float(result.get("peak_allocated_ratio", 1.0)) <= 0.80
            and int(result.get("measured_steps", 0)) >= args.measure_steps
        )

    if eligible(8):
        selected = 8
        reason = "batch 8 passed all steps and peak allocated memory was <=80%"
    elif eligible(4):
        selected = 4
        reason = "batch 8 was ineligible; batch 4 passed with <=80% allocated memory"
    else:
        raise RuntimeError(
            f"Neither batch candidate met the preregistered rule: {results}"
        )
    preregistration = {
        "status": "frozen_before_flame3_accuracy_training",
        "created_at": datetime.now().isoformat(),
        "selected_batch": selected,
        "selection_reason": reason,
        "candidate_results": results,
        "config": str(args.config.resolve()),
        "config_sha256": sha256_file(args.config.resolve()),
        "accuracy_or_validation_used": False,
        "test_images_or_labels_read": False,
    }
    destination = output_dir / "flame3_4090_batch_preregistered.json"
    destination.write_text(
        json.dumps(preregistration, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(preregistration, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
