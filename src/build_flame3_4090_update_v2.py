from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath


DEFAULT_NAME = "flame3_4090_update_split_v2_partial_20260731"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--base-bundle-root", type=Path, required=True)
    parser.add_argument("--output-parent", type=Path, required=True)
    parser.add_argument("--update-name", default=DEFAULT_NAME)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_file(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def copy_tree(source: Path, destination: Path) -> None:
    for path in sorted(source.rglob("*")):
        if not path.is_file():
            continue
        if any(part in {"__pycache__", ".pytest_cache", ".git"} for part in path.parts):
            continue
        if path.suffix == ".pyc":
            continue
        copy_file(path, destination / path.relative_to(source))


def portable_path(sample_class: str, sample_id: str, kind: str) -> str:
    if kind == "rgb":
        path = PurePosixPath("data/flame3") / sample_class / "RGB/Corrected FOV" / f"{sample_id}.JPG"
    elif kind == "thermal":
        path = PurePosixPath("data/flame3") / sample_class / "Thermal/Raw JPG" / f"{sample_id}.JPG"
    elif kind == "tiff":
        path = PurePosixPath("data/flame3") / sample_class / "Thermal/Celsius TIFF" / f"{sample_id}.TIFF"
    elif kind == "mask":
        class_dir = sample_class.replace(" ", "_").lower()
        path = PurePosixPath("labels/temperature_train_masks") / class_dir / f"{sample_id}.png"
    else:
        raise ValueError(kind)
    return path.as_posix()


def write_splits(project_root: Path, base_bundle: Path, update_root: Path) -> dict[str, int]:
    source = project_root / "data" / "flame3_splits_v2_preregistered"
    original = update_root / "splits" / "original_absolute"
    portable = update_root / "splits" / "portable"
    original.mkdir(parents=True, exist_ok=True)
    portable.mkdir(parents=True, exist_ok=True)
    for path in source.iterdir():
        if path.is_file():
            copy_file(path, original / path.name)
    counts: dict[str, int] = {}
    all_keys: set[str] = set()
    for split in ("train", "val", "test"):
        source_csv = source / f"{split}.csv"
        with source_csv.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            key = row["sample_key"]
            if key in all_keys:
                raise RuntimeError(f"Duplicate v2 sample key: {key}")
            all_keys.add(key)
            sample_class = row["sample_class"]
            sample_id = row["sample_id"]
            row["raw_rgb_path"] = ""
            row["corrected_rgb_path"] = portable_path(sample_class, sample_id, "rgb")
            row["raw_thermal_path"] = portable_path(sample_class, sample_id, "thermal")
            row["thermal_tiff_path"] = portable_path(sample_class, sample_id, "tiff")
            row["temperature_mask_path"] = portable_path(sample_class, sample_id, "mask")
            for field in (
                "corrected_rgb_path",
                "raw_thermal_path",
                "thermal_tiff_path",
                "temperature_mask_path",
            ):
                target = base_bundle / Path(*PurePosixPath(row[field]).parts)
                if not target.is_file():
                    raise FileNotFoundError(target)
        destination = portable / source_csv.name
        with destination.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        counts[split] = len(rows)
    if counts != {"train": 493, "val": 134, "test": 107}:
        raise RuntimeError(f"Unexpected split v2 counts: {counts}")
    if len(all_keys) != 734:
        raise RuntimeError(f"Expected 734 retained samples, got {len(all_keys)}")
    return counts


def git_text(project_root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=project_root,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return result.stdout.strip()


def write_verifier(update_root: Path) -> None:
    text = r'''from __future__ import annotations
import csv, hashlib, sys
from pathlib import Path

def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()

root = Path(__file__).resolve().parent
bundle = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else root
with (root / "update_manifest_sha256.csv").open("r", encoding="utf-8-sig", newline="") as handle:
    rows = list(csv.DictReader(handle))
failures = []
for row in rows:
    target = bundle / Path(*row["relative_path"].split("/"))
    if not target.is_file():
        failures.append("missing " + row["relative_path"])
    elif target.stat().st_size != int(row["size_bytes"]):
        failures.append("size " + row["relative_path"])
    elif digest(target) != row["sha256"]:
        failures.append("hash " + row["relative_path"])
if failures:
    print("FLAME3 v2 update verification FAILED")
    print("\n".join(failures[:50]))
    raise SystemExit(1)
print(f"FLAME3 v2 update verification passed: {len(rows)} files")
print("Test images and labels were not opened.")
'''
    (update_root / "verify_update.py").write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    base_bundle = args.base_bundle_root.resolve()
    output_parent = args.output_parent.resolve()
    update_root = output_parent / args.update_name
    archive = output_parent / f"{args.update_name}.zip"
    if update_root.exists() or archive.exists():
        raise FileExistsError(f"Refusing to overwrite {update_root} or {archive}")
    update_root.mkdir(parents=True)
    split_counts = write_splits(project_root, base_bundle, update_root)
    copy_tree(project_root / "src", update_root / "project_support" / "src")
    copy_tree(
        project_root / "configs" / "flame3",
        update_root / "project_support" / "configs" / "flame3",
    )
    copy_file(
        project_root
        / "third_party"
        / "RoboFireFuseNet"
        / "datasets"
        / "__init__.py",
        update_root
        / "project_support"
        / "third_party"
        / "RoboFireFuseNet"
        / "datasets"
        / "__init__.py",
    )
    for path in sorted((project_root / "docs").glob("FLAME3_*.md")):
        copy_file(path, update_root / "project_support" / "docs" / path.name)
    scripts_root = project_root / "scripts"
    if scripts_root.is_dir():
        for name in (
            "launch_flame3_30e_4090.ps1",
            "launch_flame3_100e_resume_4090.ps1",
            "launch_flame3_input_ablation_30e_4090.ps1",
            "launch_flame3_input_ablation_pair_30e_4090.ps1",
        ):
            path = scripts_root / name
            if path.is_file():
                copy_file(path, update_root / "project_support" / "scripts" / name)
    repository = {
        "branch": git_text(project_root, "branch", "--show-current"),
        "commit": git_text(project_root, "rev-parse", "HEAD"),
        "status": git_text(project_root, "status", "--short"),
    }
    (update_root / "project_support" / "repository_info_split_v2.json").write_text(
        json.dumps(repository, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    write_verifier(update_root)
    files = [
        path
        for path in sorted(update_root.rglob("*"))
        if path.is_file() and path.name not in {"update_manifest_sha256.csv", "update_summary.json"}
    ]
    rows = [
        {
            "relative_path": path.relative_to(update_root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in files
    ]
    with (update_root / "update_manifest_sha256.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["relative_path", "size_bytes", "sha256"]
        )
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "name": args.update_name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_bundle": base_bundle.name,
        "split_counts": split_counts,
        "retained_samples": 734,
        "excluded_boundary_buffer": 4,
        "payload_files": len(rows),
        "test_inference_performed": False,
        "formal_training_blocked_until_batch_preregistration": True,
    }
    (update_root / "update_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zip_file:
        for path in sorted(update_root.rglob("*")):
            if path.is_file():
                zip_file.write(path, path.relative_to(update_root).as_posix())
    print(json.dumps({**summary, "archive": str(archive), "archive_sha256": sha256_file(archive)}, indent=2))


if __name__ == "__main__":
    main()
