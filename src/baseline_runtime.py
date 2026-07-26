from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROBOFIRE_ROOT = PROJECT_ROOT / "third_party" / "RoboFireFuseNet"

for import_path in (
    ROBOFIRE_ROOT,
    ROBOFIRE_ROOT / "models",
    ROBOFIRE_ROOT / "datasets",
):
    sys.path.insert(0, str(import_path))

from datasets.wildfire import WildFire  # noqa: E402
from models.pidnet import PIDNet  # noqa: E402
from utils.total_loss import TotalLoss  # noqa: E402
from custom_models.pidnet_lscm import PIDNetLSCM, PIDNetLSCMV2  # noqa: E402
from custom_models.pidnet_lscm_v21 import PIDNetLSCMV21  # noqa: E402
from custom_models.pidnet_lscm_v3 import PIDNetLSCMV3  # noqa: E402
from custom_models.pidnet_lscm_v31 import PIDNetLSCMV31  # noqa: E402
from custom_models.pidnet_deconv import PIDNetDEConv  # noqa: E402
from custom_models.pidnet_dfm_mproto import (  # noqa: E402
    PIDNetDEConvMProto,
    PIDNetDFMMProto,
)
from custom_models.pidnet_dysample import PIDNetDySample  # noqa: E402


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def build_dataset(config: dict, split: str) -> WildFire:
    split_to_key = {
        "train": "TRAINSET",
        "val": "VALIDSET",
        "test": "TESTSET",
    }
    if split not in split_to_key:
        raise ValueError(f"Unsupported split: {split}")
    training = split == "train"
    return WildFire(
        root=config["ROOTDATASET"],
        list_path=config[split_to_key[split]],
        num_classes=config["NUM_CLASSES"],
        multi_scale=config["MULTISCALE"] if training else False,
        flip=config["FLIP"] if training else False,
        brightness=config["BRIGHTNESS"] if training else False,
        ignore_label=config["IGNORE_LABEL"],
        scale_factor=config["SCALE_FACTOR"],
        crop_size=config["CROP_SIZE"],
        base_size=config["BASE_SIZE"],
        bd_dilate_size=4,
        mode=config["MODE"],
        blend_images=config["BLEND_IMGS"] if training else False,
        comp_mask=config["COMP_MASK"] if training else False,
        single_source=config["SINGLE_SOURCE"] if training else False,
        seed=config["SEED"],
        scale_min=float(config.get("SCALE_MIN", 0.5)),
        scale_max=float(config.get("SCALE_MAX", 1.5)),
        fire_component_cache=(
            config.get("FIRE_COMPONENT_CACHE")
            if config.get("TRAINING_OBJECTIVE") == "ema_mproto"
            else None
        ),
    )


def build_model(config: dict, augment: bool = True) -> PIDNet:
    model_name = config["MODEL"]
    input_channels = {
        "rgb": 3,
        "ir": 1,
        "fusion": 4,
    }
    mode = str(config["MODE"]).lower()
    if mode not in input_channels:
        raise ValueError(f"Unsupported input mode: {config['MODE']}")
    if model_name == "pidnet_s":
        model_class = PIDNet
    elif model_name == "pidnet_s_lscm":
        model_class = PIDNetLSCM
    elif model_name == "pidnet_s_lscm_v2":
        model_class = PIDNetLSCMV2
    elif model_name == "pidnet_s_lscm_v21":
        model_class = PIDNetLSCMV21
    elif model_name == "pidnet_s_lscm_v3":
        model_class = PIDNetLSCMV3
    elif model_name == "pidnet_s_lscm_v31":
        model_class = PIDNetLSCMV31
    elif model_name == "pidnet_s_deconv":
        model_class = PIDNetDEConv
    elif model_name == "pidnet_s_dfm_mproto":
        model_class = PIDNetDFMMProto
    elif model_name == "pidnet_s_deconv_mproto":
        model_class = PIDNetDEConvMProto
    elif model_name == "pidnet_s_dysample":
        model_class = PIDNetDySample
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    model_kwargs = dict(
        m=2,
        n=3,
        num_classes=config["NUM_CLASSES"],
        planes=32,
        ppm_planes=96,
        head_planes=128,
        augment=augment,
        channels=input_channels[mode],
    )
    if model_name in {"pidnet_s_deconv", "pidnet_s_deconv_mproto"}:
        model_kwargs["deconv_variant"] = config.get("DECONV_VARIANT", "D1")
    if model_name == "pidnet_s_dysample":
        model_kwargs["dysample_variant"] = config.get(
            "DYSAMPLE_VARIANT",
            "pag4",
        )
    return model_class(**model_kwargs)


def load_pretrained_if_available(model: PIDNet, config: dict) -> int:
    configured_path = config.get("PRETRAINED")
    if not configured_path:
        print("Pretrained initialization: disabled")
        return 0

    checkpoint_path = Path(configured_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Pretrained checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    if "state_dict" in checkpoint:
        source_state = checkpoint["state_dict"]
    elif "model_state_dict" in checkpoint:
        source_state = checkpoint["model_state_dict"]
    else:
        source_state = checkpoint

    model_state = model.state_dict()
    matched_state = {
        key: value
        for key, value in source_state.items()
        if key in model_state and model_state[key].shape == value.shape
    }
    model_state.update(matched_state)
    model.load_state_dict(model_state, strict=False)
    print(
        f"Pretrained initialization: {checkpoint_path} "
        f"({len(matched_state)} tensors matched)"
    )
    return len(matched_state)


__all__ = [
    "PROJECT_ROOT",
    "TotalLoss",
    "build_dataset",
    "build_model",
    "load_config",
    "load_pretrained_if_available",
    "seed_everything",
]
