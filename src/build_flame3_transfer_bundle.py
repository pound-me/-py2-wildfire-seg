from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath


BUNDLE_VERSION = "flame3_4090_bundle_v1_20260731"
EXPECTED_COUNTS = {"Fire": 622, "No Fire": 116}
MODALITIES = (
    ("RGB/Corrected FOV", ".JPG"),
    ("Thermal/Raw JPG", ".JPG"),
    ("Thermal/Celsius TIFF", ".TIFF"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a portable, checksummed FLAME3 training bundle."
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-parent", type=Path, required=True)
    parser.add_argument("--bundle-name", default=BUNDLE_VERSION)
    parser.add_argument("--compression-level", type=int, default=6)
    return parser.parse_args()


def sha256_file(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def copy_file(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def should_skip(path: Path) -> bool:
    skip_parts = {".git", "__pycache__", ".pytest_cache", ".mypy_cache"}
    return any(part in skip_parts for part in path.parts) or path.suffix == ".pyc"


def copy_tree_filtered(source: Path, destination: Path) -> int:
    if not source.is_dir():
        raise FileNotFoundError(source)
    copied = 0
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        if should_skip(relative) or not path.is_file():
            continue
        copy_file(path, destination / relative)
        copied += 1
    return copied


def git_output(project_root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=project_root,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        return f"UNAVAILABLE: {result.stderr.strip()}"
    return result.stdout.strip()


def relative_data_path(sample_class: str, sample_id: str, subdir: str, suffix: str) -> str:
    return (PurePosixPath("data/flame3") / sample_class / subdir / f"{sample_id}{suffix}").as_posix()


def relative_mask_path(sample_class: str, sample_id: str) -> str:
    class_dir = sample_class.replace(" ", "_").lower()
    return (PurePosixPath("labels/temperature_train_masks") / class_dir / f"{sample_id}.png").as_posix()


def portable_path_exists(bundle_root: Path, value: str) -> bool:
    return (bundle_root / Path(*PurePosixPath(value).parts)).is_file()


def write_portable_splits(project_root: Path, bundle_root: Path) -> dict[str, int]:
    source_root = project_root / "data" / "flame3_splits_preregistered"
    original_root = bundle_root / "splits" / "original_absolute"
    portable_root = bundle_root / "splits" / "portable"
    original_root.mkdir(parents=True, exist_ok=True)
    portable_root.mkdir(parents=True, exist_ok=True)

    split_counts: dict[str, int] = {}
    for split_name in ("train", "val", "test"):
        source_csv = source_root / f"{split_name}.csv"
        copy_file(source_csv, original_root / source_csv.name)
        with source_csv.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            raise RuntimeError(f"Empty split: {source_csv}")

        fieldnames = list(rows[0])
        required = {
            "sample_key",
            "sample_class",
            "sample_id",
            "raw_rgb_path",
            "corrected_rgb_path",
            "raw_thermal_path",
            "thermal_tiff_path",
            "temperature_mask_path",
        }
        missing = required.difference(fieldnames)
        if missing:
            raise RuntimeError(f"Missing split fields in {source_csv}: {sorted(missing)}")

        seen: set[str] = set()
        for row in rows:
            sample_class = row["sample_class"]
            sample_id = row["sample_id"]
            sample_key = row["sample_key"]
            if sample_class not in EXPECTED_COUNTS:
                raise RuntimeError(f"Unexpected sample class: {sample_class}")
            if sample_key in seen:
                raise RuntimeError(f"Duplicate sample key in {split_name}: {sample_key}")
            seen.add(sample_key)
            row["raw_rgb_path"] = ""
            row["corrected_rgb_path"] = relative_data_path(
                sample_class, sample_id, "RGB/Corrected FOV", ".JPG"
            )
            row["raw_thermal_path"] = relative_data_path(
                sample_class, sample_id, "Thermal/Raw JPG", ".JPG"
            )
            row["thermal_tiff_path"] = relative_data_path(
                sample_class, sample_id, "Thermal/Celsius TIFF", ".TIFF"
            )
            row["temperature_mask_path"] = relative_mask_path(sample_class, sample_id)
            for key in (
                "corrected_rgb_path",
                "raw_thermal_path",
                "thermal_tiff_path",
                "temperature_mask_path",
            ):
                if not portable_path_exists(bundle_root, row[key]):
                    raise FileNotFoundError(f"Portable split path missing: {row[key]}")

        destination = portable_root / source_csv.name
        with destination.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        split_counts[split_name] = len(rows)

    copy_file(source_root / "split_manifest.json", original_root / "split_manifest.json")
    return split_counts


def write_verify_script(bundle_root: Path) -> None:
    script = r'''from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path


def sha256_file(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    root = Path(__file__).resolve().parent
    manifest_path = root / "manifest_sha256.csv"
    summary_path = root / "bundle_summary.json"
    failures: list[str] = []
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for index, row in enumerate(rows, start=1):
        path = root / Path(*row["relative_path"].split("/"))
        if not path.is_file():
            failures.append(f"missing: {row['relative_path']}")
            continue
        actual_size = path.stat().st_size
        if actual_size != int(row["size_bytes"]):
            failures.append(
                f"size mismatch: {row['relative_path']} {actual_size} != {row['size_bytes']}"
            )
            continue
        actual_hash = sha256_file(path)
        if actual_hash != row["sha256"]:
            failures.append(f"hash mismatch: {row['relative_path']}")
        if index % 500 == 0:
            print(f"Verified {index}/{len(rows)} files")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if failures:
        print("Bundle verification FAILED")
        for failure in failures[:50]:
            print(f"  {failure}")
        if len(failures) > 50:
            print(f"  ... {len(failures) - 50} additional failures")
        raise SystemExit(1)
    print("FLAME3 4090 transfer bundle verification passed")
    print(f"Manifest files: {len(rows)}")
    print(f"Payload bytes: {summary['payload_bytes']}")
    print(f"Splits: {summary['split_counts']}")
    print("Test split remains sealed; verification does not run model inference.")


if __name__ == "__main__":
    main()
'''
    (bundle_root / "verify_bundle.py").write_text(script, encoding="utf-8")


def write_readme(bundle_root: Path) -> None:
    readme = r'''# FLAME3 RTX 4090 portable training bundle

This bundle contains only the material required to prepare FLAME3 training on the
temporary RTX 4090 workstation. It intentionally excludes 4000x3000 raw RGB,
legacy experiments, validation visualizations, and test predictions.

## Included

- `data/flame3/`: Corrected FOV RGB, Raw Thermal JPG, and Celsius TIFF for 738 samples.
- `labels/temperature_train_masks/`: preregistered 0/2/255 temperature masks.
- `splits/original_absolute/`: immutable source split records for audit only.
- `splits/portable/`: the same frozen split membership with paths relative to this bundle.
- `weights/PIDNet_S_ImageNet.pth.tar`: PIDNet-S ImageNet initialization.
- `project_support/`: current source/config snapshot and required RoboFireFuseNet code.
- `metadata/`: pseudolabel audit material and review conclusion.
- `manifest_sha256.csv`: size and SHA256 for every payload file.
- `verify_bundle.py`: integrity verifier.

## Important protocol boundaries

- `test.csv` is transferred but remains sealed. Do not use it for model selection,
  debugging, threshold changes, or checkpoint selection.
- The temperature mask denotes a temperature-supported active-fire region, not
  necessarily a visible RGB flame.
- Fire-image pixels outside the pseudo-fire region are not automatically safe
  Background supervision because they may contain Smoke.
- Formal training remains blocked until the partial-label/smoke supervision rule
  and the split-imbalance decision are preregistered.

## Remote placement

Extract the archive under:

`C:\Users\Admin\Desktop\7.31\`

Expected resulting root:

`C:\Users\Admin\Desktop\7.31\flame3_4090_bundle_v1_20260731\`

Then verify without touching the test set:

```powershell
python "C:\Users\Admin\Desktop\7.31\flame3_4090_bundle_v1_20260731\verify_bundle.py"
```

The portable CSV paths must be resolved relative to the extracted bundle root.
Do not run the old absolute-path CSV files on the remote workstation.
'''
    (bundle_root / "README_4090_TRANSFER.md").write_text(readme, encoding="utf-8")


def collect_manifest(bundle_root: Path) -> tuple[list[dict[str, object]], int]:
    excluded = {"manifest_sha256.csv", "bundle_summary.json"}
    rows: list[dict[str, object]] = []
    payload_bytes = 0
    for path in sorted(bundle_root.rglob("*")):
        if not path.is_file() or path.name in excluded:
            continue
        relative = path.relative_to(bundle_root).as_posix()
        size = path.stat().st_size
        rows.append(
            {
                "relative_path": relative,
                "size_bytes": size,
                "sha256": sha256_file(path),
            }
        )
        payload_bytes += size
    return rows, payload_bytes


def write_manifest(bundle_root: Path, rows: list[dict[str, object]]) -> None:
    with (bundle_root / "manifest_sha256.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["relative_path", "size_bytes", "sha256"]
        )
        writer.writeheader()
        writer.writerows(rows)


def build_archive(bundle_root: Path, archive_path: Path, compression_level: int) -> None:
    with zipfile.ZipFile(
        archive_path,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=compression_level,
        allowZip64=True,
    ) as archive:
        for index, path in enumerate(sorted(bundle_root.rglob("*")), start=1):
            if not path.is_file():
                continue
            archive.write(path, arcname=(Path(bundle_root.name) / path.relative_to(bundle_root)))
            if index % 500 == 0:
                print(f"Archived {index} entries", flush=True)


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    project_root = args.project_root.resolve()
    output_parent = args.output_parent.resolve()
    bundle_root = output_parent / args.bundle_name
    archive_path = output_parent / f"{args.bundle_name}.zip"
    if bundle_root.exists() or archive_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing bundle output: {bundle_root} or {archive_path}"
        )
    output_parent.mkdir(parents=True, exist_ok=True)
    bundle_root.mkdir(parents=True)

    started = time.time()
    copied_counts: dict[str, dict[str, int]] = {}
    print("Copying FLAME3 modalities...", flush=True)
    for sample_class, expected_count in EXPECTED_COUNTS.items():
        copied_counts[sample_class] = {}
        for subdir, suffix in MODALITIES:
            source_dir = dataset_root / sample_class / Path(subdir)
            destination_dir = bundle_root / "data" / "flame3" / sample_class / Path(subdir)
            files = sorted(source_dir.glob(f"*{suffix}"))
            if len(files) != expected_count:
                raise RuntimeError(
                    f"Unexpected file count for {sample_class}/{subdir}: "
                    f"{len(files)} != {expected_count}"
                )
            for source in files:
                copy_file(source, destination_dir / source.name)
            copied_counts[sample_class][subdir] = len(files)
            print(f"  {sample_class}/{subdir}: {len(files)}", flush=True)

    print("Copying temperature masks...", flush=True)
    mask_source_root = (
        project_root
        / "experiments"
        / "flame3_temperature_pseudolabels_preregistered"
        / "train_mask_templates"
    )
    for sample_class, expected_count in EXPECTED_COUNTS.items():
        class_dir = sample_class.replace(" ", "_").lower()
        sources = sorted((mask_source_root / class_dir).glob("*.png"))
        if len(sources) != expected_count:
            raise RuntimeError(
                f"Unexpected mask count for {class_dir}: {len(sources)} != {expected_count}"
            )
        for source in sources:
            copy_file(
                source,
                bundle_root / "labels" / "temperature_train_masks" / class_dir / source.name,
            )
        copied_counts[sample_class]["temperature_train_masks"] = len(sources)

    split_counts = write_portable_splits(project_root, bundle_root)

    print("Copying metadata, weight, and project support...", flush=True)
    pseudo_root = (
        project_root / "experiments" / "flame3_temperature_pseudolabels_preregistered"
    )
    for filename in ("pseudolabel_summary.json", "pseudolabel_statistics.csv"):
        copy_file(pseudo_root / filename, bundle_root / "metadata" / filename)
    review_conclusion = pseudo_root / "manual_review_30" / "manual_review_conclusion.md"
    copy_file(review_conclusion, bundle_root / "metadata" / review_conclusion.name)
    copy_file(
        project_root / "weights" / "PIDNet_S_ImageNet.pth.tar",
        bundle_root / "weights" / "PIDNet_S_ImageNet.pth.tar",
    )

    copy_tree_filtered(project_root / "src", bundle_root / "project_support" / "src")
    copy_tree_filtered(project_root / "configs", bundle_root / "project_support" / "configs")
    copy_tree_filtered(
        project_root / "third_party" / "RoboFireFuseNet",
        bundle_root / "project_support" / "third_party" / "RoboFireFuseNet",
    )
    for filename in ("README.md", ".gitignore"):
        source = project_root / filename
        if source.is_file():
            copy_file(source, bundle_root / "project_support" / filename)
    flame3_docs = sorted((project_root / "docs").glob("FLAME3_*.md"))
    for source in flame3_docs:
        copy_file(source, bundle_root / "project_support" / "docs" / source.name)

    repository_info = {
        "branch": git_output(project_root, "branch", "--show-current"),
        "commit": git_output(project_root, "rev-parse", "HEAD"),
        "remote": git_output(project_root, "remote", "get-url", "origin"),
        "tracked_status": git_output(
            project_root, "status", "--short", "--untracked-files=no"
        ),
        "note": "FLAME3 files may be untracked in the source repository; this bundle is the transfer snapshot.",
    }
    (bundle_root / "project_support" / "repository_info.json").write_text(
        json.dumps(repository_info, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    write_verify_script(bundle_root)
    write_readme(bundle_root)
    manifest_rows, payload_bytes = collect_manifest(bundle_root)
    write_manifest(bundle_root, manifest_rows)
    summary = {
        "bundle_name": args.bundle_name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_root_source": str(dataset_root),
        "project_root_source": str(project_root),
        "payload_files": len(manifest_rows),
        "payload_bytes": payload_bytes,
        "copied_counts": copied_counts,
        "split_counts": split_counts,
        "excluded": [
            "4000x3000 raw RGB",
            "legacy experiments and visualizations",
            "zero-shot predictions",
            "test inference outputs",
            "FreqFusion and unrelated third-party repositories",
        ],
        "test_split_sealed": True,
        "formal_training_authorized": False,
    }
    (bundle_root / "bundle_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print(f"Building ZIP archive: {archive_path}", flush=True)
    build_archive(bundle_root, archive_path, args.compression_level)
    archive_hash = sha256_file(archive_path)
    result = {
        "bundle_root": str(bundle_root),
        "archive_path": str(archive_path),
        "archive_bytes": archive_path.stat().st_size,
        "archive_sha256": archive_hash,
        "payload_files": len(manifest_rows),
        "payload_bytes": payload_bytes,
        "elapsed_seconds": round(time.time() - started, 1),
    }
    result_path = output_parent / f"{args.bundle_name}_archive.json"
    result_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise
