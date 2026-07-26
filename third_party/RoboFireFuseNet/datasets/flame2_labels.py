from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


FLAME2_COLOR_TO_CLASS = {
    (0, 0, 0): 0,
    (125, 125, 125): 1,
    (255, 255, 255): 2,
}


def decode_three_class_label(
    color_map: np.ndarray,
    *,
    source: str = "<array>",
) -> np.ndarray:
    """Decode one FLAME2 RGB mask and reject every unknown color."""
    color = np.asarray(color_map, dtype=np.uint8)
    if color.ndim != 3 or color.shape[2] < 3:
        raise ValueError(
            f"FLAME2 label must be an RGB image in {source}; got shape "
            f"{color.shape}."
        )

    rgb = color[:, :, :3]
    label = np.empty(rgb.shape[:2], dtype=np.uint8)
    matched = np.zeros(rgb.shape[:2], dtype=bool)
    for color_value, class_index in FLAME2_COLOR_TO_CLASS.items():
        class_mask = np.all(rgb == color_value, axis=2)
        label[class_mask] = class_index
        matched |= class_mask

    if not np.all(matched):
        unknown = np.unique(rgb[~matched].reshape(-1, 3), axis=0)
        raise ValueError(
            f"Unknown FLAME2 label colors in {source}: "
            f"{unknown.tolist()}"
        )
    return label


def load_three_class_label(path: str | Path) -> np.ndarray:
    """Load and strictly decode one FLAME2 three-class label file."""
    label_path = Path(path)
    color = np.asarray(Image.open(label_path).convert("RGB"), dtype=np.uint8)
    return decode_three_class_label(color, source=str(label_path))

