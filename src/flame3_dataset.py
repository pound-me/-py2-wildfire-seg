from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from datasets.base_dataset import BaseDataset


FIRE_CLASS = 2
IGNORE_LABEL = 255
ALLOWED_LABEL_VALUES = {0, FIRE_CLASS, IGNORE_LABEL}


def _resolve_path(root: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def build_fire_core_edge(label: np.ndarray, dilation_size: int) -> np.ndarray:
    if dilation_size <= 0:
        raise ValueError("dilation_size must be positive")
    fire_core = ((label == FIRE_CLASS).astype(np.uint8) * 255)
    edge = cv2.Canny(fire_core, 0.1, 0.2)
    kernel = np.ones((dilation_size, dilation_size), dtype=np.uint8)
    return (cv2.dilate(edge, kernel, iterations=1) > 50).astype(np.float32)


class Flame3CsvDataset(BaseDataset):
    """FLAME3 paired RGB/thermal dataset with image-level partial-label identity."""

    def __init__(
        self,
        root: str | Path,
        csv_path: str | Path,
        mode: str = "fusion",
        multi_scale: bool = True,
        flip: bool = True,
        brightness: bool = False,
        ignore_label: int = IGNORE_LABEL,
        base_size: int = 640,
        crop_size: tuple[int, int] | list[int] = (512, 640),
        scale_factor: int = 10,
        scale_min: float = 0.8,
        scale_max: float = 1.5,
        bd_dilate_size: int = 4,
        comp_mask: bool = False,
        single_source: bool = False,
    ) -> None:
        if int(ignore_label) != IGNORE_LABEL:
            raise ValueError("FLAME3 temperature masks require IGNORE_LABEL=255")
        super().__init__(
            ignore_label=ignore_label,
            base_size=base_size,
            crop_size=tuple(int(value) for value in crop_size),
            scale_factor=scale_factor,
            mean=[0, 0, 0, 0],
            std=[1, 1, 1, 1],
            scale_min=scale_min,
            scale_max=scale_max,
        )
        self.root = Path(root).resolve()
        csv_candidate = Path(csv_path)
        self.csv_path = (
            csv_candidate.resolve()
            if csv_candidate.is_absolute()
            else (self.root / csv_candidate).resolve()
        )
        if not self.csv_path.is_file():
            raise FileNotFoundError(f"FLAME3 split CSV not found: {self.csv_path}")
        self.mode = str(mode).lower()
        if self.mode not in {"rgb", "ir", "fusion"}:
            raise ValueError(f"Unsupported FLAME3 input mode: {mode}")
        self.multi_scale = bool(multi_scale)
        self.flip = bool(flip)
        self.brightness = bool(brightness)
        self.comp_mask = bool(comp_mask)
        self.single_source = bool(single_source)
        self.bd_dilate_size = int(bd_dilate_size)
        with self.csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            self.rows = list(csv.DictReader(handle))
        if not self.rows:
            raise RuntimeError(f"Empty FLAME3 split CSV: {self.csv_path}")
        required = {
            "sample_key",
            "sample_class",
            "sample_id",
            "corrected_rgb_path",
            "raw_thermal_path",
            "temperature_mask_path",
        }
        missing = required.difference(self.rows[0])
        if missing:
            raise RuntimeError(
                f"FLAME3 split CSV is missing fields: {sorted(missing)}"
            )
        keys = [row["sample_key"] for row in self.rows]
        if len(keys) != len(set(keys)):
            raise RuntimeError(f"Duplicate sample keys in {self.csv_path}")

    def __len__(self) -> int:
        return len(self.rows)

    def _load_raw(self, index: int) -> tuple[list[np.ndarray], np.ndarray, bool, str]:
        row = self.rows[index]
        sample_class = row["sample_class"]
        if sample_class not in {"Fire", "No Fire"}:
            raise RuntimeError(
                f"Unexpected sample_class={sample_class!r} for {row['sample_key']}"
            )
        is_fire_folder = sample_class == "Fire"
        rgb_path = _resolve_path(self.root, row["corrected_rgb_path"])
        thermal_path = _resolve_path(self.root, row["raw_thermal_path"])
        label_path = _resolve_path(self.root, row["temperature_mask_path"])
        for path in (rgb_path, thermal_path, label_path):
            if not path.is_file():
                raise FileNotFoundError(f"Missing FLAME3 sample file: {path}")

        rgb = np.asarray(Image.open(rgb_path).convert("RGB"), dtype=np.uint8).copy()
        thermal = np.asarray(
            Image.open(thermal_path).convert("L"), dtype=np.uint8
        ).copy()[..., None]
        label = np.asarray(Image.open(label_path), dtype=np.uint8).copy()
        if rgb.shape[:2] != thermal.shape[:2] or label.shape != thermal.shape[:2]:
            raise RuntimeError(
                f"Shape mismatch for {row['sample_key']}: "
                f"rgb={rgb.shape}, thermal={thermal.shape}, label={label.shape}"
            )
        values = set(int(value) for value in np.unique(label).tolist())
        if not values.issubset(ALLOWED_LABEL_VALUES):
            raise RuntimeError(
                f"Unexpected label values {sorted(values)} for {row['sample_key']}"
            )
        if not is_fire_folder and values != {0}:
            raise RuntimeError(
                f"No Fire sample must be hard Background only: {row['sample_key']} "
                f"has {sorted(values)}"
            )
        return [rgb, thermal], label, is_fire_folder, row["sample_key"]

    def __getitem__(self, index: int):
        images, label, is_fire_folder, sample_key = self._load_raw(index)
        generated = self.gen_sample(
            images,
            label,
            multi_scale=self.multi_scale,
            is_flip=self.flip,
            edge_pad=False,
            edge_size=self.bd_dilate_size,
            brightness=self.brightness,
            comp_mask=self.comp_mask,
            single_source=self.single_source,
        )
        images, label, _unused_edge = generated
        edge = build_fire_core_edge(label, self.bd_dilate_size)
        stacked = np.concatenate(images, axis=0)
        if self.mode == "rgb":
            stacked = stacked[:3]
        elif self.mode == "ir":
            stacked = stacked[3:4]
        return (
            stacked.copy(),
            label.copy(),
            edge.copy(),
            [sample_key],
            np.uint8(is_fire_folder),
        )


__all__ = [
    "ALLOWED_LABEL_VALUES",
    "FIRE_CLASS",
    "Flame3CsvDataset",
    "IGNORE_LABEL",
    "build_fire_core_edge",
]
