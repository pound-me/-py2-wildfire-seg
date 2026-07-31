from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path

import torch


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command_output(arguments: list[str], cwd: Path | None = None) -> str:
    result = subprocess.run(
        arguments,
        cwd=cwd,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return result.stdout.strip()


def package_versions(names: list[str]) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    bundle_root = args.bundle_root.resolve()
    config_path = args.config.resolve()
    train_split = bundle_root / "splits" / "portable" / "train.csv"
    val_split = bundle_root / "splits" / "portable" / "val.csv"
    required = [config_path, train_split, val_split]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    device = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(device)
    if "4090" not in gpu_name:
        raise RuntimeError(f"Expected RTX 4090, got {gpu_name}")

    source_files = sorted((project_root / "src").rglob("*.py"))
    source_digest = hashlib.sha256()
    for path in source_files:
        source_digest.update(path.relative_to(project_root).as_posix().encode())
        source_digest.update(path.read_bytes())
    payload = {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "python_executable": sys.executable,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": gpu_name,
        "gpu_capability": list(torch.cuda.get_device_capability(device)),
        "gpu_total_memory_bytes": torch.cuda.get_device_properties(device).total_memory,
        "nvidia_smi": command_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version,pstate",
                "--format=csv,noheader",
            ]
        ),
        "packages": package_versions(
            ["numpy", "Pillow", "opencv-python", "PyYAML", "scikit-learn", "torchvision"]
        ),
        "project_root": str(project_root),
        "bundle_root": str(bundle_root),
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "train_split_sha256": sha256_file(train_split),
        "val_split_sha256": sha256_file(val_split),
        "source_sha256": source_digest.hexdigest(),
        "git_branch": command_output(["git", "branch", "--show-current"], project_root),
        "git_commit": command_output(["git", "rev-parse", "HEAD"], project_root),
        "test_images_or_labels_read": False,
    }
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.resolve().write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
