from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

from baseline_runtime import build_dataset, build_model, load_config, seed_everything


PANEL_SIZE = (320, 256)
TITLE_HEIGHT = 44


def safe_name(value: str) -> str:
    return value.replace("/", "_").replace("\\", "_")


def unwrap_sample_key(value) -> str:
    while isinstance(value, (list, tuple)) and len(value) == 1:
        value = value[0]
    return str(value)


def read_csv_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    return {str(row["sample_key"]): row for row in rows}


def resolve_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def load_thermal_jpg(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        gray = np.asarray(image.convert("L"), dtype=np.uint8)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)


def load_temperature(path: Path) -> np.ndarray:
    temperature = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if temperature is None:
        raise FileNotFoundError(path)
    if temperature.ndim == 3:
        temperature = temperature[..., 0]
    normalized = np.clip(temperature.astype(np.float32), 0.0, 500.0)
    normalized = np.round(normalized / 500.0 * 255.0).astype(np.uint8)
    colored = cv2.applyColorMap(normalized, cv2.COLORMAP_INFERNO)
    return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)


def load_label(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image, dtype=np.uint8)


def colorize_label(label: np.ndarray) -> np.ndarray:
    output = np.zeros((*label.shape, 3), dtype=np.uint8)
    output[label == 1] = (160, 160, 160)
    output[label == 2] = (255, 80, 0)
    output[label == 255] = (255, 220, 0)
    return output


def overlay_prediction(rgb: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    output = rgb.astype(np.float32).copy()
    smoke = prediction == 1
    fire = prediction == 2
    output[smoke] = 0.45 * output[smoke] + 0.55 * np.asarray(
        [165, 165, 165], dtype=np.float32
    )
    output[fire] = 0.30 * output[fire] + 0.70 * np.asarray(
        [255, 55, 0], dtype=np.float32
    )
    return np.clip(output, 0, 255).astype(np.uint8)


def change_overlay(
    rgb: np.ndarray,
    label: np.ndarray,
    fusion: np.ndarray,
    cmrc: np.ndarray,
) -> tuple[np.ndarray, dict[str, int]]:
    valid = label != 255
    target_fire = label == 2
    fusion_fire = fusion == 2
    cmrc_fire = cmrc == 2
    masks = {
        "recovered_fire": target_fire & ~fusion_fire & cmrc_fire,
        "lost_fire": target_fire & fusion_fire & ~cmrc_fire,
        "removed_pseudo_fp": valid & ~target_fire & fusion_fire & ~cmrc_fire,
        "added_pseudo_fp": valid & ~target_fire & ~fusion_fire & cmrc_fire,
    }
    colors = {
        "recovered_fire": np.asarray([0, 255, 0], dtype=np.float32),
        "lost_fire": np.asarray([30, 100, 255], dtype=np.float32),
        "removed_pseudo_fp": np.asarray([0, 255, 255], dtype=np.float32),
        "added_pseudo_fp": np.asarray([255, 0, 180], dtype=np.float32),
    }
    output = (rgb.astype(np.float32) * 0.45).copy()
    for name, mask in masks.items():
        output[mask] = 0.15 * output[mask] + 0.85 * colors[name]
    return np.clip(output, 0, 255).astype(np.uint8), {
        name: int(mask.sum()) for name, mask in masks.items()
    }


def caption_panel(image: np.ndarray, title: str) -> Image.Image:
    panel = Image.new("RGB", (PANEL_SIZE[0], PANEL_SIZE[1] + TITLE_HEIGHT), "white")
    resized = Image.fromarray(image).resize(PANEL_SIZE, Image.Resampling.BILINEAR)
    panel.paste(resized, (0, TITLE_HEIGHT))
    draw = ImageDraw.Draw(panel)
    draw.text((8, 7), title, fill="black")
    return panel


def build_visual(
    sample_key: str,
    rgb: np.ndarray,
    thermal_jpg: np.ndarray,
    temperature: np.ndarray,
    label: np.ndarray,
    fusion: np.ndarray,
    cmrc: np.ndarray,
) -> tuple[Image.Image, dict[str, int]]:
    change, counts = change_overlay(rgb, label, fusion, cmrc)
    panels = [
        caption_panel(rgb, f"RGB | {sample_key}"),
        caption_panel(thermal_jpg, "Raw thermal JPG"),
        caption_panel(temperature, "Celsius TIFF (0-500C)"),
        caption_panel(colorize_label(label), "Pseudo label: fire=orange"),
        caption_panel(overlay_prediction(rgb, fusion), "Fusion prediction"),
        caption_panel(overlay_prediction(rgb, cmrc), "CMRC prediction"),
        caption_panel(
            change,
            "Change: green recover / blue lose / cyan remove / magenta add",
        ),
    ]
    canvas = Image.new(
        "RGB",
        (sum(panel.width for panel in panels), max(panel.height for panel in panels)),
        "white",
    )
    x = 0
    for panel in panels:
        canvas.paste(panel, (x, 0))
        x += panel.width
    return canvas, counts


def save_contact_sheet(paths: list[Path], destination: Path, columns: int = 1) -> None:
    images = [Image.open(path).convert("RGB") for path in paths]
    if not images:
        return
    target_width = 1600
    resized = []
    for image in images:
        height = max(int(round(image.height * target_width / image.width)), 1)
        resized.append(image.resize((target_width, height), Image.Resampling.LANCZOS))
    rows = (len(resized) + columns - 1) // columns
    cell_width = target_width
    cell_height = max(image.height for image in resized)
    sheet = Image.new("RGB", (cell_width * columns, cell_height * rows), "white")
    for index, image in enumerate(resized):
        x = (index % columns) * cell_width
        y = (index // columns) * cell_height
        sheet.paste(image, (x, y))
    sheet.save(destination, quality=92)
    for image in images:
        image.close()


def load_checkpoint(model: torch.nn.Module, path: Path) -> None:
    checkpoint = torch.load(path, map_location="cpu")
    state = checkpoint.get("model_state_dict", checkpoint)
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.missing_keys:
        raise RuntimeError(f"Missing checkpoint keys for {path}: {incompatible.missing_keys}")


def select_top(
    records: dict[str, dict],
    metric: str,
    count: int,
) -> list[str]:
    return [
        key
        for key, record in sorted(
            records.items(),
            key=lambda item: (-int(item[1][metric]), item[0]),
        )
        if int(record[metric]) > 0
    ][:count]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate validation-only paired Fusion/CMRC prediction visuals."
    )
    parser.add_argument("--fusion-config", type=Path, required=True)
    parser.add_argument("--cmrc-config", type=Path, required=True)
    parser.add_argument("--fusion-checkpoint", type=Path, required=True)
    parser.add_argument("--cmrc-checkpoint", type=Path, required=True)
    parser.add_argument("--root-dataset", type=Path, required=True)
    parser.add_argument("--validation-csv", type=Path, required=True)
    parser.add_argument("--audit-manifest", type=Path)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--top-per-category", type=int, default=6)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for paired visual inference.")
    if args.top_per_category <= 0:
        raise ValueError("top-per-category must be positive.")

    root = args.root_dataset.resolve()
    validation_csv = args.validation_csv.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows = read_csv_rows(validation_csv)
    fusion_config = load_config(args.fusion_config.resolve())
    cmrc_config = load_config(args.cmrc_config.resolve())
    for config in (fusion_config, cmrc_config):
        config["ROOTDATASET"] = str(root)
        config["VALIDSET"] = str(validation_csv)
        config["NUM_WORKERS"] = 0
        config["BATCHSIZE"] = 1
    if fusion_config.get("MODEL") != "pidnet_s":
        raise ValueError("Fusion config must use MODEL=pidnet_s.")
    if cmrc_config.get("MODEL") != "pidnet_s_cmrc":
        raise ValueError("CMRC config must use MODEL=pidnet_s_cmrc.")
    if fusion_config.get("MODE") != "fusion" or cmrc_config.get("MODE") != "fusion":
        raise ValueError("Both models must use Fusion input mode.")

    seed_everything(int(args.seed))
    device = torch.device("cuda:0")
    fusion_model = build_model(fusion_config, augment=False)
    cmrc_model = build_model(cmrc_config, augment=False)
    load_checkpoint(fusion_model, args.fusion_checkpoint.resolve())
    load_checkpoint(cmrc_model, args.cmrc_checkpoint.resolve())
    fusion_model = fusion_model.to(device).eval()
    cmrc_model = cmrc_model.to(device).eval()
    dataset = build_dataset(fusion_config, "val")
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    records: dict[str, dict] = {}
    predictions: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    with torch.inference_mode():
        for batch in loader:
            images, labels, sample_keys = batch[0], batch[1], batch[3]
            images = images.to(device=device, dtype=torch.float32)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                fusion_logits = fusion_model(images)
                cmrc_logits = cmrc_model(images)
            target_size = labels.shape[-2:]
            fusion_logits = F.interpolate(
                fusion_logits,
                size=target_size,
                mode="bilinear",
                align_corners=True,
            )
            cmrc_logits = F.interpolate(
                cmrc_logits,
                size=target_size,
                mode="bilinear",
                align_corners=True,
            )
            label = labels[0].numpy().astype(np.uint8)
            fusion = fusion_logits[0].argmax(0).cpu().numpy().astype(np.uint8)
            cmrc = cmrc_logits[0].argmax(0).cpu().numpy().astype(np.uint8)
            sample_key = unwrap_sample_key(sample_keys)
            _change, counts = change_overlay(
                np.zeros((*label.shape, 3), dtype=np.uint8),
                label,
                fusion,
                cmrc,
            )
            records[sample_key] = counts
            predictions[sample_key] = (label, fusion, cmrc)

    selections = {
        "top_recovered_fire": select_top(
            records, "recovered_fire", args.top_per_category
        ),
        "top_lost_fire": select_top(records, "lost_fire", args.top_per_category),
        "top_removed_pseudo_fp": select_top(
            records, "removed_pseudo_fp", args.top_per_category
        ),
        "top_added_pseudo_fp": select_top(
            records, "added_pseudo_fp", args.top_per_category
        ),
    }
    if args.audit_manifest and args.audit_manifest.resolve().is_file():
        manifest = json.loads(args.audit_manifest.resolve().read_text(encoding="utf-8"))
        groups = {
            "audit_fn_001_030": set(),
            "audit_fp_residual_heat_031_045": set(),
            "audit_fp_label_gap_046_050": set(),
            "audit_tp_051_070": set(),
        }
        for item in manifest.get("items", []):
            audit_id = str(item.get("audit_id", ""))
            number = int(audit_id.split("_", 1)[0])
            sample_key = str(item.get("sample_key", ""))
            if 1 <= number <= 30:
                groups["audit_fn_001_030"].add(sample_key)
            elif 31 <= number <= 45:
                groups["audit_fp_residual_heat_031_045"].add(sample_key)
            elif 46 <= number <= 50:
                groups["audit_fp_label_gap_046_050"].add(sample_key)
            elif 51 <= number <= 70:
                groups["audit_tp_051_070"].add(sample_key)
        for name, keys in groups.items():
            selections[name] = sorted(key for key in keys if key in records)

    visual_paths: dict[str, Path] = {}
    all_keys = sorted({key for keys in selections.values() for key in keys})
    visual_dir = output / "visuals"
    visual_dir.mkdir(parents=True, exist_ok=True)
    for sample_key in all_keys:
        row = rows[sample_key]
        rgb = load_rgb(resolve_path(root, row["corrected_rgb_path"]))
        thermal_jpg = load_thermal_jpg(resolve_path(root, row["raw_thermal_path"]))
        temperature = load_temperature(resolve_path(root, row["thermal_tiff_path"]))
        label, fusion, cmrc = predictions[sample_key]
        if rgb.shape[:2] != label.shape:
            rgb = np.asarray(
                Image.fromarray(rgb).resize(
                    (label.shape[1], label.shape[0]), Image.Resampling.BILINEAR
                )
            )
        thermal_jpg = np.asarray(
            Image.fromarray(thermal_jpg).resize(
                (label.shape[1], label.shape[0]), Image.Resampling.BILINEAR
            )
        )
        temperature = np.asarray(
            Image.fromarray(temperature).resize(
                (label.shape[1], label.shape[0]), Image.Resampling.BILINEAR
            )
        )
        visual, counts = build_visual(
            sample_key,
            rgb,
            thermal_jpg,
            temperature,
            label,
            fusion,
            cmrc,
        )
        path = visual_dir / f"{safe_name(sample_key)}.jpg"
        visual.save(path, quality=92)
        visual_paths[sample_key] = path
        records[sample_key].update(counts)

    contact_sheets = {}
    for category, keys in selections.items():
        paths = [visual_paths[key] for key in keys if key in visual_paths]
        if not paths:
            continue
        destination = output / f"{category}_contact_sheet.jpg"
        save_contact_sheet(paths, destination)
        contact_sheets[category] = str(destination)

    result = {
        "protocol": "validation_only_fixed_paired_fusion_cmrc_visual_comparison",
        "seed": int(args.seed),
        "fusion_checkpoint": str(args.fusion_checkpoint.resolve()),
        "cmrc_checkpoint": str(args.cmrc_checkpoint.resolve()),
        "validation_csv": str(validation_csv),
        "selection_rules": {
            "green": "target Fire missed by Fusion and recovered by CMRC",
            "blue": "target Fire found by Fusion and lost by CMRC",
            "cyan": "non-Fire pseudo-label pixel predicted Fire only by Fusion",
            "magenta": "non-Fire pseudo-label pixel predicted Fire only by CMRC",
        },
        "selections": selections,
        "records": records,
        "contact_sheets": contact_sheets,
        "test_images_or_labels_read": False,
        "interpretation_caveat": (
            "cyan/magenta are pseudo-label consistency changes, not automatically "
            "true false-positive changes in semantically ambiguous hot or smoky regions"
        ),
    }
    (output / "comparison_manifest.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
