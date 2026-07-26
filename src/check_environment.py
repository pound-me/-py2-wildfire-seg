from __future__ import annotations

import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PIDNET_ROOT = PROJECT_ROOT / "third_party" / "PIDNet"

if not PIDNET_ROOT.is_dir():
    raise FileNotFoundError(f"PIDNet repository not found: {PIDNET_ROOT}")

sys.path.insert(0, str(PIDNET_ROOT))

from models.pidnet import get_pred_model  # noqa: E402


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the selected VS Code interpreter.")

    device = torch.device("cuda:0")
    model = get_pred_model("pidnet-s", num_classes=3).to(device).eval()
    sample = torch.randn(1, 3, 256, 256, device=device)

    with torch.inference_mode():
        output = model(sample)

    parameters = sum(parameter.numel() for parameter in model.parameters())
    allocated_mb = torch.cuda.memory_allocated(device) / 1024**2

    print("Environment check passed")
    print(f"Python: {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA runtime: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"PIDNet-S parameters: {parameters / 1e6:.3f} M")
    print(f"Output shape: {tuple(output.shape)}")
    print(f"Allocated GPU memory: {allocated_mb:.1f} MB")


if __name__ == "__main__":
    main()
