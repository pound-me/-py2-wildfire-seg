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
from flame3_dataset import Flame3CsvDataset  # noqa: E402


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


def build_dataset(config: dict, split: str):
    split_to_key = {
        "train": "TRAINSET",
        "val": "VALIDSET",
        "test": "TESTSET",
    }
    if split not in split_to_key:
        raise ValueError(f"Unsupported split: {split}")
    training = split == "train"
    if str(config.get("DATASET_TYPE", "wildfire_txt")).lower() == "flame3_csv":
        if config.get("TRAINING_OBJECTIVE") != "partial_label":
            raise ValueError(
                "FLAME3 CSV training requires TRAINING_OBJECTIVE=partial_label"
            )
        return Flame3CsvDataset(
            root=config["ROOTDATASET"],
            csv_path=config[split_to_key[split]],
            mode=config["MODE"],
            multi_scale=config["MULTISCALE"] if training else False,
            flip=config["FLIP"] if training else False,
            brightness=config["BRIGHTNESS"] if training else False,
            ignore_label=config["IGNORE_LABEL"],
            scale_factor=config["SCALE_FACTOR"],
            crop_size=config["CROP_SIZE"],
            base_size=config["BASE_SIZE"],
            bd_dilate_size=int(config.get("BD_DILATE_SIZE", 4)),
            comp_mask=config["COMP_MASK"] if training else False,
            single_source=config["SINGLE_SOURCE"] if training else False,
            scale_min=float(config.get("SCALE_MIN", 0.8)),
            scale_max=float(config.get("SCALE_MAX", 1.5)),
        )
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
        from custom_models.pidnet_lscm import PIDNetLSCM

        model_class = PIDNetLSCM
    elif model_name == "pidnet_s_lscm_v2":
        from custom_models.pidnet_lscm import PIDNetLSCMV2

        model_class = PIDNetLSCMV2
    elif model_name == "pidnet_s_lscm_v21":
        from custom_models.pidnet_lscm_v21 import PIDNetLSCMV21

        model_class = PIDNetLSCMV21
    elif model_name == "pidnet_s_lscm_v3":
        from custom_models.pidnet_lscm_v3 import PIDNetLSCMV3

        model_class = PIDNetLSCMV3
    elif model_name == "pidnet_s_lscm_v31":
        from custom_models.pidnet_lscm_v31 import PIDNetLSCMV31

        model_class = PIDNetLSCMV31
    elif model_name == "pidnet_s_deconv":
        from custom_models.pidnet_deconv import PIDNetDEConv

        model_class = PIDNetDEConv
    elif model_name == "pidnet_s_dfm_mproto":
        from custom_models.pidnet_dfm_mproto import PIDNetDFMMProto

        model_class = PIDNetDFMMProto
    elif model_name == "pidnet_s_deconv_mproto":
        from custom_models.pidnet_dfm_mproto import PIDNetDEConvMProto

        model_class = PIDNetDEConvMProto
    elif model_name == "pidnet_s_dysample":
        from custom_models.pidnet_dysample import PIDNetDySample

        model_class = PIDNetDySample
    elif model_name == "pidnet_s_freqfusion":
        from custom_models.pidnet_freqfusion import PIDNetFreqFusion

        model_class = PIDNetFreqFusion
    elif model_name == "pidnet_s_samf":
        from custom_models.pidnet_samf import PIDNetSAMF

        model_class = PIDNetSAMF
    elif model_name == "pidnet_s_tgm":
        from custom_models.pidnet_tgm import PIDNetTGM

        model_class = PIDNetTGM
    elif model_name == "pidnet_s_erctc":
        from custom_models.pidnet_erctc import PIDNetERCTC

        model_class = PIDNetERCTC
    elif model_name == "pidnet_s_mrff":
        from custom_models.pidnet_mrff import PIDNetMRFF

        model_class = PIDNetMRFF
    elif model_name == "pidnet_s_cmrc":
        from custom_models.pidnet_cmrc import PIDNetCMRC

        model_class = PIDNetCMRC
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
    if model_name == "pidnet_s_freqfusion":
        model_kwargs["freqfusion_variant"] = config.get(
            "FREQFUSION_VARIANT",
            "pag3",
        )
        model_kwargs["compressed_channels"] = int(
            config.get("FREQFUSION_COMPRESSED_CHANNELS", 16)
        )
        model_kwargs["feature_resample_group"] = int(
            config.get("FREQFUSION_FEATURE_RESAMPLE_GROUP", 4)
        )
    if model_name == "pidnet_s_samf":
        model_kwargs["smoke_class"] = int(config.get("SAMF_SMOKE_CLASS", 1))
        model_kwargs["thermal_channel"] = int(
            config.get("SAMF_THERMAL_CHANNEL", 3)
        )
    if model_name == "pidnet_s_tgm":
        model_kwargs["thermal_channel"] = int(
            config.get("TGM_THERMAL_CHANNEL", 3)
        )
    if model_name == "pidnet_s_erctc":
        model_kwargs["thermal_channel"] = int(
            config.get("ERCTC_THERMAL_CHANNEL", 3)
        )
        model_kwargs["compressed_channels"] = int(
            config.get("ERCTC_COMPRESSED_CHANNELS", 16)
        )
    if model_name == "pidnet_s_mrff":
        model_kwargs["thermal_channel"] = int(
            config.get("MRFF_THERMAL_CHANNEL", 3)
        )
        model_kwargs["gate_hidden_channels"] = int(
            config.get("MRFF_GATE_HIDDEN_CHANNELS", 16)
        )
    if model_name == "pidnet_s_cmrc":
        model_kwargs["thermal_channel"] = int(
            config.get("CMRC_THERMAL_CHANNEL", 3)
        )
        model_kwargs["hint_stem_channels"] = int(
            config.get("CMRC_HINT_STEM_CHANNELS", 8)
        )
        model_kwargs["hint_channels"] = int(
            config.get("CMRC_HINT_CHANNELS", 16)
        )
        model_kwargs["context_channels"] = int(
            config.get("CMRC_CONTEXT_CHANNELS", 16)
        )
        model_kwargs["correction_hidden_channels"] = int(
            config.get("CMRC_CORRECTION_HIDDEN_CHANNELS", 16)
        )
        model_kwargs["residual_limit"] = float(
            config.get("CMRC_RESIDUAL_LIMIT", 0.1)
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
    configured_skip_keys = config.get("PRETRAIN_SKIP_KEYS", [])
    if isinstance(configured_skip_keys, str):
        configured_skip_keys = [configured_skip_keys]
    skip_keys = {str(key) for key in configured_skip_keys}
    unknown_source_keys = sorted(skip_keys.difference(source_state))
    unknown_model_keys = sorted(skip_keys.difference(model_state))
    if unknown_source_keys or unknown_model_keys:
        raise KeyError(
            "Invalid PRETRAIN_SKIP_KEYS: "
            f"missing_from_checkpoint={unknown_source_keys}, "
            f"missing_from_model={unknown_model_keys}"
        )
    matched_state = {
        key: value
        for key, value in source_state.items()
        if key in model_state
        and model_state[key].shape == value.shape
        and key not in skip_keys
    }
    model_state.update(matched_state)
    model.load_state_dict(model_state, strict=False)
    print(
        f"Pretrained initialization: {checkpoint_path} "
        f"({len(matched_state)} tensors matched)"
    )
    if skip_keys:
        print(f"Explicit pretrained skips: {sorted(skip_keys)}")
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
