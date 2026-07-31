from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


CLASS_ORDER = ("Fire", "No Fire")


@dataclass
class RegistrationResult:
    sample_class: str
    sample_id: str
    rgb_path: str
    thermal_path: str
    ecc_success: bool
    ecc_score: float | None
    similarity_before: float
    similarity_after: float
    similarity_gain: float
    dx_px: float | None
    dy_px: float | None
    rotation_deg: float | None
    scale_x: float | None
    scale_y: float | None
    shear: float | None
    phase_dx_px: float | None
    phase_dy_px: float | None
    phase_response: float | None
    plausible_transform: bool
    confidence: str
    failure_reason: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit residual registration between FLAME 3 Corrected FOV RGB "
            "images and radiometric thermal TIFFs. The tool never modifies "
            "source data."
        )
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--seed", type=int, default=200)
    parser.add_argument("--max-translation", type=float, default=20.0)
    parser.add_argument("--max-rotation", type=float, default=3.0)
    parser.add_argument("--min-scale", type=float, default=0.95)
    parser.add_argument("--max-scale", type=float, default=1.05)
    parser.add_argument("--min-similarity-gain", type=float, default=0.005)
    return parser.parse_args()


def collect_pairs(data_root: Path) -> dict[str, list[tuple[str, Path, Path]]]:
    pairs: dict[str, list[tuple[str, Path, Path]]] = {}
    for class_name in CLASS_ORDER:
        rgb_dir = data_root / class_name / "RGB" / "Corrected FOV"
        thermal_dir = data_root / class_name / "Thermal" / "Celsius TIFF"
        if not rgb_dir.is_dir() or not thermal_dir.is_dir():
            raise FileNotFoundError(
                f"Missing FLAME 3 directories for {class_name}: "
                f"{rgb_dir} / {thermal_dir}"
            )
        rgb_by_stem = {path.stem: path for path in rgb_dir.glob("*.JPG")}
        thermal_by_stem = {path.stem: path for path in thermal_dir.glob("*.TIFF")}
        if set(rgb_by_stem) != set(thermal_by_stem):
            missing_rgb = sorted(set(thermal_by_stem) - set(rgb_by_stem))
            missing_thermal = sorted(set(rgb_by_stem) - set(thermal_by_stem))
            raise RuntimeError(
                f"Pair mismatch for {class_name}: missing_rgb={missing_rgb[:10]}, "
                f"missing_thermal={missing_thermal[:10]}"
            )
        pairs[class_name] = [
            (stem, rgb_by_stem[stem], thermal_by_stem[stem])
            for stem in sorted(rgb_by_stem)
        ]
    return pairs


def allocate_sample_counts(
    pairs: dict[str, list[tuple[str, Path, Path]]], total: int
) -> dict[str, int]:
    if total < len(CLASS_ORDER):
        raise ValueError(f"samples must be at least {len(CLASS_ORDER)}")
    available = {name: len(pairs[name]) for name in CLASS_ORDER}
    # Registration reliability, not dataset prevalence, is the objective here.
    # Balance Fire and No Fire so unobscured landscape geometry is adequately
    # represented instead of letting the 622/116 class ratio dominate.
    counts = {
        name: min(available[name], total // len(CLASS_ORDER))
        for name in CLASS_ORDER
    }
    while sum(counts.values()) > total:
        name = max(CLASS_ORDER, key=lambda item: counts[item])
        if counts[name] <= 1:
            break
        counts[name] -= 1
    while sum(counts.values()) < total:
        candidates = [name for name in CLASS_ORDER if counts[name] < available[name]]
        if not candidates:
            break
        name = min(candidates, key=lambda item: counts[item])
        counts[name] += 1
    return counts


def stratified_even_sample(
    pairs: dict[str, list[tuple[str, Path, Path]]], total: int
) -> list[tuple[str, str, Path, Path]]:
    counts = allocate_sample_counts(pairs, total)
    selected: list[tuple[str, str, Path, Path]] = []
    for class_name in CLASS_ORDER:
        class_pairs = pairs[class_name]
        count = min(counts[class_name], len(class_pairs))
        indices = np.linspace(0, len(class_pairs) - 1, count, dtype=int)
        for index in indices.tolist():
            stem, rgb_path, thermal_path = class_pairs[index]
            selected.append((class_name, stem, rgb_path, thermal_path))
    return selected


def robust_normalize(array: np.ndarray) -> np.ndarray:
    values = np.asarray(array, dtype=np.float32)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("Image contains no finite values")
    low, high = np.percentile(finite, [1.0, 99.0])
    if high <= low:
        low = float(finite.min())
        high = float(finite.max())
    if high <= low:
        return np.zeros(values.shape, dtype=np.float32)
    values = np.clip(values, low, high)
    values = (values - low) / (high - low)
    return values.astype(np.float32)


def gradient_map(normalized: np.ndarray) -> np.ndarray:
    source = cv2.GaussianBlur(normalized.astype(np.float32), (5, 5), 0)
    grad_x = cv2.Sobel(source, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(source, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(grad_x, grad_y)
    magnitude = robust_normalize(magnitude)
    return cv2.GaussianBlur(magnitude, (3, 3), 0)


def normalized_correlation(first: np.ndarray, second: np.ndarray) -> float:
    left = first.astype(np.float64).ravel()
    right = second.astype(np.float64).ravel()
    left -= left.mean()
    right -= right.mean()
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 1e-12:
        return 0.0
    return float(np.dot(left, right) / denominator)


def affine_components(warp: np.ndarray) -> tuple[float, float, float, float, float, float]:
    a, b, tx = [float(value) for value in warp[0]]
    c, d, ty = [float(value) for value in warp[1]]
    scale_x = math.sqrt(a * a + c * c)
    scale_y = math.sqrt(b * b + d * d)
    rotation = math.degrees(math.atan2(c, a))
    shear = (a * b + c * d) / max(scale_x * scale_y, 1e-12)
    return tx, ty, rotation, scale_x, scale_y, shear


def estimate_registration(
    class_name: str,
    stem: str,
    rgb_path: Path,
    thermal_path: Path,
    args: argparse.Namespace,
) -> tuple[RegistrationResult, np.ndarray, np.ndarray, np.ndarray | None]:
    rgb_bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    if rgb_bgr is None:
        raise RuntimeError(f"Failed to read RGB image: {rgb_path}")
    rgb_gray = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    thermal = np.asarray(Image.open(thermal_path), dtype=np.float32)
    if rgb_gray.shape != thermal.shape:
        raise RuntimeError(
            f"Shape mismatch for {class_name}/{stem}: "
            f"RGB={rgb_gray.shape}, thermal={thermal.shape}"
        )

    rgb_edge = gradient_map(robust_normalize(rgb_gray))
    thermal_edge = gradient_map(robust_normalize(thermal))
    similarity_before = normalized_correlation(thermal_edge, rgb_edge)

    phase_dx = phase_dy = phase_response = None
    try:
        phase_shift, phase_response_raw = cv2.phaseCorrelate(thermal_edge, rgb_edge)
        phase_dx, phase_dy = float(phase_shift[0]), float(phase_shift[1])
        phase_response = float(phase_response_raw)
    except cv2.error:
        pass

    warp = np.eye(2, 3, dtype=np.float32)
    aligned_edge: np.ndarray | None = None
    ecc_score: float | None = None
    failure_reason: str | None = None
    try:
        criteria = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            250,
            1e-7,
        )
        ecc_score_raw, warp = cv2.findTransformECC(
            thermal_edge,
            rgb_edge,
            warp,
            cv2.MOTION_AFFINE,
            criteria,
            None,
            5,
        )
        ecc_score = float(ecc_score_raw)
        aligned_edge = cv2.warpAffine(
            rgb_edge,
            warp,
            (thermal_edge.shape[1], thermal_edge.shape[0]),
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        similarity_after = normalized_correlation(thermal_edge, aligned_edge)
        dx, dy, rotation, scale_x, scale_y, shear = affine_components(warp)
        plausible = (
            abs(dx) <= args.max_translation
            and abs(dy) <= args.max_translation
            and abs(rotation) <= args.max_rotation
            and args.min_scale <= scale_x <= args.max_scale
            and args.min_scale <= scale_y <= args.max_scale
        )
        gain = similarity_after - similarity_before
        if plausible and gain >= args.min_similarity_gain and ecc_score >= 0.10:
            confidence = "usable"
        elif plausible:
            confidence = "low_gain"
        else:
            confidence = "implausible"
    except cv2.error as error:
        similarity_after = similarity_before
        gain = 0.0
        dx = dy = rotation = scale_x = scale_y = shear = None
        plausible = False
        confidence = "failed"
        failure_reason = str(error).splitlines()[0]

    result = RegistrationResult(
        sample_class=class_name,
        sample_id=stem,
        rgb_path=str(rgb_path),
        thermal_path=str(thermal_path),
        ecc_success=ecc_score is not None,
        ecc_score=ecc_score,
        similarity_before=similarity_before,
        similarity_after=similarity_after,
        similarity_gain=gain,
        dx_px=dx,
        dy_px=dy,
        rotation_deg=rotation,
        scale_x=scale_x,
        scale_y=scale_y,
        shear=shear,
        phase_dx_px=phase_dx,
        phase_dy_px=phase_dy,
        phase_response=phase_response,
        plausible_transform=plausible,
        confidence=confidence,
        failure_reason=failure_reason,
    )
    return result, rgb_bgr, thermal, warp if ecc_score is not None else None


def thermal_color(thermal: np.ndarray) -> np.ndarray:
    normalized = (robust_normalize(thermal) * 255.0).astype(np.uint8)
    return cv2.applyColorMap(normalized, cv2.COLORMAP_INFERNO)


def edge_overlay(rgb_bgr: np.ndarray, thermal: np.ndarray) -> np.ndarray:
    output = rgb_bgr.copy()
    thermal_edges = gradient_map(robust_normalize(thermal))
    threshold = float(np.percentile(thermal_edges, 92.0))
    mask = thermal_edges >= threshold
    output[mask] = (0, 0, 255)
    return output


def make_visualization(
    result: RegistrationResult,
    rgb_bgr: np.ndarray,
    thermal: np.ndarray,
    warp: np.ndarray | None,
) -> np.ndarray:
    height, width = thermal.shape
    thermal_bgr = thermal_color(thermal)
    before = edge_overlay(rgb_bgr, thermal)
    if warp is None or not result.plausible_transform:
        aligned_rgb = rgb_bgr.copy()
    else:
        aligned_rgb = cv2.warpAffine(
            rgb_bgr,
            warp,
            (width, height),
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
    after = edge_overlay(aligned_rgb, thermal)
    checker = np.zeros_like(rgb_bgr)
    tile = 32
    yy, xx = np.indices((height, width))
    choose_thermal = ((xx // tile) + (yy // tile)) % 2 == 0
    checker[choose_thermal] = thermal_bgr[choose_thermal]
    checker[~choose_thermal] = rgb_bgr[~choose_thermal]

    panels = [rgb_bgr, thermal_bgr, before, after, checker]
    labels = ["RGB corrected", "Thermal", "Edges before", "Edges after", "Checkerboard"]
    for panel, label in zip(panels, labels):
        cv2.rectangle(panel, (0, 0), (width, 30), (0, 0, 0), thickness=-1)
        cv2.putText(
            panel,
            label,
            (8, 21),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    canvas = np.concatenate(panels, axis=1)
    caption = (
        f"{result.sample_class}/{result.sample_id} | "
        f"dx={result.dx_px if result.dx_px is not None else float('nan'):.2f}, "
        f"dy={result.dy_px if result.dy_px is not None else float('nan'):.2f}, "
        f"rot={result.rotation_deg if result.rotation_deg is not None else float('nan'):.3f}, "
        f"gain={result.similarity_gain:.4f}, {result.confidence}"
    )
    cv2.rectangle(canvas, (0, height - 28), (canvas.shape[1], height), (0, 0, 0), -1)
    cv2.putText(
        canvas,
        caption,
        (8, height - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return canvas


def summarize(results: list[RegistrationResult], args: argparse.Namespace) -> dict[str, object]:
    usable = [item for item in results if item.confidence == "usable"]
    plausible = [item for item in results if item.plausible_transform]

    def values(field: str, source: list[RegistrationResult]) -> np.ndarray:
        return np.asarray(
            [getattr(item, field) for item in source if getattr(item, field) is not None],
            dtype=np.float64,
        )

    def stats(field: str, source: list[RegistrationResult]) -> dict[str, float | None]:
        array = values(field, source)
        if array.size == 0:
            return {"median": None, "mad": None, "minimum": None, "maximum": None}
        median = float(np.median(array))
        mad = float(np.median(np.abs(array - median)))
        return {
            "median": median,
            "mad": mad,
            "minimum": float(array.min()),
            "maximum": float(array.max()),
        }

    source = usable if usable else plausible
    dx = values("dx_px", source)
    dy = values("dy_px", source)
    rotation = values("rotation_deg", source)
    scale_x = values("scale_x", source)
    scale_y = values("scale_y", source)
    gains = values("similarity_gain", source)

    sufficient = len(source) >= max(10, int(math.ceil(0.4 * len(results))))
    stable_translation = (
        dx.size > 0
        and dy.size > 0
        and float(np.median(np.abs(dx - np.median(dx)))) <= 1.5
        and float(np.median(np.abs(dy - np.median(dy)))) <= 1.5
    )
    stable_rotation = (
        rotation.size > 0
        and float(np.median(np.abs(rotation - np.median(rotation)))) <= 0.20
    )
    stable_scale = (
        scale_x.size > 0
        and scale_y.size > 0
        and float(np.median(np.abs(scale_x - np.median(scale_x)))) <= 0.003
        and float(np.median(np.abs(scale_y - np.median(scale_y)))) <= 0.003
    )
    material_offset = (
        dx.size > 0
        and dy.size > 0
        and (
            abs(float(np.median(dx))) >= 1.5
            or abs(float(np.median(dy))) >= 1.5
            or (
                rotation.size > 0 and abs(float(np.median(rotation))) >= 0.15
            )
            or (
                scale_x.size > 0
                and abs(float(np.median(scale_x)) - 1.0) >= 0.003
            )
            or (
                scale_y.size > 0
                and abs(float(np.median(scale_y)) - 1.0) >= 0.003
            )
        )
    )
    meaningful_gain = gains.size > 0 and float(np.median(gains)) >= 0.01
    recommend = bool(
        sufficient
        and stable_translation
        and stable_rotation
        and stable_scale
        and material_offset
        and meaningful_gain
    )

    candidate = None
    if source:
        candidate = {
            "dx_px": float(np.median(dx)) if dx.size else None,
            "dy_px": float(np.median(dy)) if dy.size else None,
            "rotation_deg": float(np.median(rotation)) if rotation.size else None,
            "scale_x": float(np.median(scale_x)) if scale_x.size else None,
            "scale_y": float(np.median(scale_y)) if scale_y.size else None,
        }

    return {
        "data_root": str(args.data_root.resolve()),
        "sample_count": len(results),
        "class_counts": {
            name: sum(item.sample_class == name for item in results)
            for name in CLASS_ORDER
        },
        "confidence_counts": {
            label: sum(item.confidence == label for item in results)
            for label in ("usable", "low_gain", "implausible", "failed")
        },
        "statistics_source": "usable" if usable else "plausible",
        "statistics_source_count": len(source),
        "dx_px": stats("dx_px", source),
        "dy_px": stats("dy_px", source),
        "rotation_deg": stats("rotation_deg", source),
        "scale_x": stats("scale_x", source),
        "scale_y": stats("scale_y", source),
        "similarity_gain": stats("similarity_gain", source),
        "decision_rule": {
            "minimum_source_count": max(10, int(math.ceil(0.4 * len(results)))),
            "translation_mad_max_px": 1.5,
            "rotation_mad_max_deg": 0.20,
            "scale_mad_max": 0.003,
            "material_translation_px": 1.5,
            "material_rotation_deg": 0.15,
            "material_scale_delta": 0.003,
            "median_similarity_gain_min": 0.01,
            "visual_review_required": True,
        },
        "global_affine_recommended_by_numeric_rule": recommend,
        "global_affine_candidate": candidate,
        "final_application_status": "pending_manual_overlay_review",
        "note": (
            "A global affine must not be applied from numeric evidence alone. "
            "Manual review of all sampled overlays is required."
        ),
    }


def save_csv(results: list[RegistrationResult], path: Path) -> None:
    rows = [asdict(result) for result in results]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_contact_sheet(visual_paths: list[Path], output_path: Path) -> None:
    thumbs: list[Image.Image] = []
    for path in visual_paths:
        image = Image.open(path).convert("RGB")
        image.thumbnail((1200, 245), Image.Resampling.LANCZOS)
        thumbs.append(image.copy())
    if not thumbs:
        return
    width = max(image.width for image in thumbs)
    height = sum(image.height for image in thumbs)
    sheet = Image.new("RGB", (width, height), color=(20, 20, 20))
    y = 0
    for image in thumbs:
        sheet.paste(image, (0, y))
        y += image.height
    sheet.save(output_path, quality=92)


def save_markdown(summary: dict[str, object], output_path: Path) -> None:
    decision = summary["global_affine_recommended_by_numeric_rule"]
    candidate = summary["global_affine_candidate"] or {}
    lines = [
        "# FLAME3 配准审计（自动估计阶段）",
        "",
        "本审计只读取原始数据，不修改 RGB 或热红外文件。",
        "",
        "## 冻结判定规则",
        "",
        "- 抽样覆盖 Fire 与 No Fire，并按文件序列均匀取样。",
        "- 使用 RGB/热红外梯度图进行仿射 ECC 估计；超出预注册范围的变换判为不可信。",
        "- 只有样本数充足、偏移方向稳定、离散度低、偏移达到实质幅度且相似度有稳定提升时，数值规则才推荐全局仿射。",
        "- 即使数值规则推荐，也必须人工检查全部叠图后才能启用；禁止逐图用验证结果拟合变换。",
        "",
        "## 自动结果",
        "",
        f"- 样本数：{summary['sample_count']}，类别分布：{summary['class_counts']}。",
        f"- 置信度分布：{summary['confidence_counts']}。",
        f"- 数值规则是否推荐全局仿射：`{decision}`。",
        f"- 候选全局参数（仅供人工复核）：`{candidate}`。",
        "- 最终应用状态：`pending_manual_overlay_review`。",
        "",
        "详细数值见 `registration_metrics.csv` 和 `registration_summary.json`，",
        "人工检查图见 `registration_contact_sheet.jpg` 与 `visuals/`。",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.samples <= 0:
        raise ValueError("samples must be positive")
    args.data_root = args.data_root.resolve()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    visual_dir = args.output / "visuals"
    visual_dir.mkdir(parents=True, exist_ok=True)

    pairs = collect_pairs(args.data_root)
    selected = stratified_even_sample(pairs, args.samples)
    results: list[RegistrationResult] = []
    visual_paths: list[Path] = []
    for class_name, stem, rgb_path, thermal_path in selected:
        result, rgb_bgr, thermal, warp = estimate_registration(
            class_name, stem, rgb_path, thermal_path, args
        )
        results.append(result)
        visual = make_visualization(result, rgb_bgr, thermal, warp)
        safe_class = class_name.lower().replace(" ", "_")
        visual_path = visual_dir / f"{safe_class}_{stem}_registration.jpg"
        if not cv2.imwrite(str(visual_path), visual):
            raise RuntimeError(f"Failed to save visualization: {visual_path}")
        visual_paths.append(visual_path)

    summary = summarize(results, args)
    save_csv(results, args.output / "registration_metrics.csv")
    (args.output / "registration_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    save_contact_sheet(visual_paths, args.output / "registration_contact_sheet.jpg")
    save_markdown(summary, args.output / "FLAME3_REGISTRATION_AUDIT.md")

    print(f"Registration audit completed: {args.output}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
