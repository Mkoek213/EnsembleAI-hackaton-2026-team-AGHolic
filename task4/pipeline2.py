#!/usr/bin/env python3
"""Task 4 ECG digitization pipeline v2.

Changes vs baseline:
1) Stage 1 normalization + grid calibration,
2) rotation correction from grid lines,
3) separate X/Y minor grid spacing estimation,
4) grid phase estimation (x0/y0),
5) better signal mask inside ROI,
6) more stable trace extraction using contiguous-component centers.

Still intentionally pragmatic:
- assumes standard 3x4 + rhythm II layout,
- keeps the same final .npz format,
- no training required.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Optional

import cv2
import numpy as np
from PIL import Image


LEADS_ORDER = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]
ROW_LAYOUT = [
    ["I", "AVR", "V1", "V4"],
    ["II", "AVL", "V2", "V5"],
    ["III", "AVF", "V3", "V6"],
]


# =========================
# Data structures
# =========================

@dataclass
class Stage1Geometry:
    rgb: np.ndarray
    gray: np.ndarray
    minor_dx_px: float
    minor_dy_px: float
    grid_x0: float
    grid_y0: float
    rotation_deg: float
    confidence: float
    square_grid_assumed: bool
    debug: Dict[str, np.ndarray | float | int | str]


# =========================
# Basic I/O helpers
# =========================

def load_rgb(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def crop_non_black(rgb: np.ndarray, min_non_black: int = 8) -> np.ndarray:
    """Removes black triangle borders after rotation/photos, if present."""
    gray = rgb.mean(axis=2)
    ys, xs = np.where(gray > min_non_black)
    if len(xs) == 0:
        return rgb
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    return rgb[y0:y1, x0:x1]


def resize_long_side(rgb: np.ndarray, target_long_side: int = 2200) -> np.ndarray:
    h, w = rgb.shape[:2]
    long_side = max(h, w)
    if long_side <= target_long_side:
        return rgb
    scale = target_long_side / float(long_side)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    return cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)


def smooth_1d(x: np.ndarray, win: int) -> np.ndarray:
    win = max(3, int(win))
    if win % 2 == 0:
        win += 1
    kernel = np.ones(win, dtype=np.float32) / float(win)
    return np.convolve(x.astype(np.float32), kernel, mode="same")


def clahe_gray(gray: np.ndarray) -> np.ndarray:
    gray_u8 = gray.astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray_u8)


def rotate_array(arr: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate while preserving canvas size."""
    h, w = arr.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle_deg, 1.0)
    return cv2.warpAffine(
        arr,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def save_image(path: Path, arr: np.ndarray) -> None:
    """Saves 2D/3D arrays as PNG for debugging."""
    if arr.ndim == 2:
        if arr.dtype == np.bool_:
            img = arr.astype(np.uint8) * 255
        else:
            img = np.clip(arr, 0, 255).astype(np.uint8)
        Image.fromarray(img, mode="L").save(path)
        return
    if arr.ndim == 3 and arr.shape[2] == 3:
        img = np.clip(arr, 0, 255).astype(np.uint8)
        Image.fromarray(img, mode="RGB").save(path)
        return
    raise ValueError(f"Unsupported image array shape for save: {arr.shape}")


def dump_stage1_debug(
    rec_dir: Path,
    rgb0: np.ndarray,
    stage1: Stage1Geometry,
) -> None:
    save_image(rec_dir / "00_input_rgb.png", rgb0)
    save_image(rec_dir / "01_normalized_rgb.png", stage1.rgb)

    if isinstance(stage1.debug.get("grid_mask"), np.ndarray):
        save_image(rec_dir / "02_grid_mask.png", stage1.debug["grid_mask"])
    if isinstance(stage1.debug.get("horiz_mask"), np.ndarray):
        save_image(rec_dir / "03_horiz_mask.png", stage1.debug["horiz_mask"])
    if isinstance(stage1.debug.get("vert_mask"), np.ndarray):
        save_image(rec_dir / "04_vert_mask.png", stage1.debug["vert_mask"])

    metrics = {
        "rotation_deg": float(stage1.rotation_deg),
        "confidence": float(stage1.confidence),
        "minor_dx_px": float(stage1.minor_dx_px),
        "minor_dy_px": float(stage1.minor_dy_px),
        "grid_x0": float(stage1.grid_x0),
        "grid_y0": float(stage1.grid_y0),
        "square_grid_assumed": bool(stage1.square_grid_assumed),
    }
    for key in (
        "rotation_raw_deg",
        "rotation_candidates",
        "rotation_inliers",
        "rotation_spread_deg",
        "rotation_pre_quality",
        "rotation_post_quality",
        "rotation_post_residual_deg",
        "rotation_applied",
        "rotation_grid_raw_deg",
        "rotation_edge_raw_deg",
        "dx_hough",
        "dy_hough",
        "dx_fft",
        "dy_fft",
        "num_vert_lines",
        "num_horiz_lines",
    ):
        val = stage1.debug.get(key)
        if isinstance(val, (int, float, np.integer, np.floating)):
            metrics[key] = float(val)
    reason = stage1.debug.get("rotation_reason")
    if isinstance(reason, str):
        metrics["rotation_reason"] = reason
    source = stage1.debug.get("rotation_source")
    if isinstance(source, str):
        metrics["rotation_source"] = source
    grid_reason = stage1.debug.get("rotation_grid_reason")
    if isinstance(grid_reason, str):
        metrics["rotation_grid_reason"] = grid_reason
    edge_reason = stage1.debug.get("rotation_edge_reason")
    if isinstance(edge_reason, str):
        metrics["rotation_edge_reason"] = edge_reason

    (rec_dir / "stage1_metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )


def dump_layout_debug(
    rec_dir: Path,
    rgb: np.ndarray,
    row_centers: np.ndarray,
    half_band: int,
    x_split: np.ndarray,
) -> None:
    h, w = rgb.shape[:2]
    vis = rgb.copy()

    for yc in row_centers:
        y = int(yc)
        cv2.line(vis, (0, y), (w - 1, y), (0, 255, 0), 2)
        cv2.line(vis, (0, max(0, y - half_band)), (w - 1, max(0, y - half_band)), (255, 255, 0), 1)
        cv2.line(vis, (0, min(h - 1, y + half_band)), (w - 1, min(h - 1, y + half_band)), (255, 255, 0), 1)

    for x in x_split:
        xx = int(np.clip(x, 0, w - 1))
        cv2.line(vis, (xx, 0), (xx, h - 1), (255, 128, 0), 1)

    save_image(rec_dir / "05_layout_overlay.png", vis)


def dump_roi_debug(rec_dir: Path, tag: str, roi_gray: np.ndarray) -> None:
    roi_u8 = np.clip(roi_gray, 0, 255).astype(np.uint8)
    sig_mask = adaptive_signal_mask(roi_gray).astype(np.uint8) * 255
    save_image(rec_dir / f"{tag}_roi.png", roi_u8)
    save_image(rec_dir / f"{tag}_mask.png", sig_mask)


# =========================
# Stage 1: normalization + calibration
# =========================

def build_grid_mask(rgb: np.ndarray, gray_eq: np.ndarray) -> np.ndarray:
    """
    Multi-source grid candidate mask:
    - red/pink score,
    - kmeans on grayscale intensities,
    - high-pass fallback for grayscale / photocopies.
    """
    rgb_f = rgb.astype(np.float32)
    r = rgb_f[:, :, 0]
    g = rgb_f[:, :, 1]
    b = rgb_f[:, :, 2]

    # 1) Red/pink grid candidate
    red_score = r - 0.5 * (g + b)
    red_score = cv2.GaussianBlur(red_score, (5, 5), 0)
    red_score_u8 = cv2.normalize(red_score, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, red_mask = cv2.threshold(red_score_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 2) KMeans on grayscale, pick median-intensity cluster
    Z = gray_eq.reshape((-1, 1)).astype(np.float32)
    K = 4
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(Z, K, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    centers = centers.flatten()
    order = np.argsort(centers)
    median_cluster = order[len(order) // 2]
    km_mask = (labels.reshape(gray_eq.shape) == median_cluster).astype(np.uint8) * 255

    # 3) High-pass fallback
    blur = cv2.GaussianBlur(gray_eq, (0, 0), 7)
    high = cv2.addWeighted(gray_eq, 1.5, blur, -0.5, 0)
    _, hp_mask = cv2.threshold(high, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # If high-pass mask is mostly background, invert it
    if float((hp_mask > 0).mean()) < 0.15:
        hp_mask = cv2.bitwise_not(hp_mask)

    # Combine
    mask = cv2.bitwise_or(red_mask, km_mask)
    mask = cv2.bitwise_or(mask, hp_mask)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    return mask


def separate_grid_lines(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h, w = mask.shape[:2]
    kx = max(15, w // 80)
    ky = max(15, h // 80)

    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kx, 1))
    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, ky))

    horiz = cv2.morphologyEx(mask, cv2.MORPH_OPEN, horiz_kernel)
    vert = cv2.morphologyEx(mask, cv2.MORPH_OPEN, vert_kernel)

    horiz = cv2.morphologyEx(horiz, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    vert = cv2.morphologyEx(vert, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    return horiz, vert


def estimate_rotation_from_grid_mask(mask: np.ndarray) -> tuple[float, Dict[str, float | int | str]]:
    """
    Estimate global skew from Hough lines on grid-like mask.
    Returns (angle_deg, metadata). Angle is 0.0 when estimate is rejected.
    """
    lines = cv2.HoughLinesP(
        mask,
        rho=1,
        theta=np.pi / 180,
        threshold=120,
        minLineLength=max(mask.shape) // 6,
        maxLineGap=20,
    )
    if lines is None:
        return 0.0, {
            "rotation_raw_deg": 0.0,
            "rotation_candidates": 0,
            "rotation_inliers": 0,
            "rotation_spread_deg": -1.0,
            "rotation_reason": "no_hough_lines",
        }

    angles: List[float] = []
    for line in lines[:, 0]:
        x1, y1, x2, y2 = line
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0 and dy == 0:
            continue

        angle = np.degrees(np.arctan2(dy, dx))
        if angle < -90:
            angle += 180
        if angle > 90:
            angle -= 180

        # near horizontal
        if abs(angle) < 20:
            angles.append(angle)
        # near vertical
        elif abs(abs(angle) - 90) < 20:
            corr = angle - 90 if angle > 0 else angle + 90
            angles.append(corr)

    if not angles:
        return 0.0, {
            "rotation_raw_deg": 0.0,
            "rotation_candidates": 0,
            "rotation_inliers": 0,
            "rotation_spread_deg": -1.0,
            "rotation_reason": "no_axis_aligned_lines",
        }

    arr = np.array(angles, dtype=np.float32)
    raw = float(np.median(arr))
    abs_dev = np.abs(arr - raw)
    mad = float(np.median(abs_dev))
    thr = max(1.5, 2.5 * mad)
    inlier_mask = abs_dev <= thr
    inliers = arr[inlier_mask]
    spread = float(np.percentile(inliers, 90) - np.percentile(inliers, 10)) if inliers.size >= 2 else 0.0

    meta = {
        "rotation_raw_deg": raw,
        "rotation_candidates": int(arr.size),
        "rotation_inliers": int(inliers.size),
        "rotation_spread_deg": spread,
        "rotation_reason": "ok",
    }

    if inliers.size < 6:
        meta["rotation_reason"] = "too_few_inliers"
        return 0.0, meta

    robust_angle = float(np.median(inliers))
    if spread > 4.0:
        meta["rotation_reason"] = "high_spread"
        return 0.0, meta

    if abs(robust_angle) > 15.0:
        meta["rotation_reason"] = "angle_too_large"
        return 0.0, meta

    return robust_angle, meta


def estimate_rotation_from_edges(gray: np.ndarray) -> tuple[float, Dict[str, float | int | str]]:
    """
    Fallback skew estimation from generic edges (more robust when grid mask is weak).
    """
    gray_u8 = np.clip(gray, 0, 255).astype(np.uint8)
    edges = cv2.Canny(gray_u8, 50, 150)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=100,
        minLineLength=max(edges.shape) // 6,
        maxLineGap=20,
    )
    if lines is None:
        return 0.0, {
            "rotation_raw_deg": 0.0,
            "rotation_candidates": 0,
            "rotation_inliers": 0,
            "rotation_spread_deg": -1.0,
            "rotation_reason": "no_hough_lines",
        }

    angles: List[float] = []
    for line in lines[:, 0]:
        x1, y1, x2, y2 = line
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0 and dy == 0:
            continue

        angle = np.degrees(np.arctan2(dy, dx))
        if angle < -90:
            angle += 180
        if angle > 90:
            angle -= 180

        if abs(angle) < 20:
            angles.append(angle)
        elif abs(abs(angle) - 90) < 20:
            corr = angle - 90 if angle > 0 else angle + 90
            angles.append(corr)

    if not angles:
        return 0.0, {
            "rotation_raw_deg": 0.0,
            "rotation_candidates": 0,
            "rotation_inliers": 0,
            "rotation_spread_deg": -1.0,
            "rotation_reason": "no_axis_aligned_lines",
        }

    arr = np.array(angles, dtype=np.float32)
    raw = float(np.median(arr))
    abs_dev = np.abs(arr - raw)
    mad = float(np.median(abs_dev))
    thr = max(1.0, 2.0 * mad)
    inlier_mask = abs_dev <= thr
    inliers = arr[inlier_mask]
    spread = float(np.percentile(inliers, 90) - np.percentile(inliers, 10)) if inliers.size >= 2 else 0.0

    meta = {
        "rotation_raw_deg": raw,
        "rotation_candidates": int(arr.size),
        "rotation_inliers": int(inliers.size),
        "rotation_spread_deg": spread,
        "rotation_reason": "ok",
    }

    if inliers.size < 20:
        meta["rotation_reason"] = "too_few_inliers"
        return 0.0, meta

    robust_angle = float(np.median(inliers))
    if spread > 2.0:
        meta["rotation_reason"] = "high_spread"
        return 0.0, meta

    if abs(robust_angle) > 15.0:
        meta["rotation_reason"] = "angle_too_large"
        return 0.0, meta

    return robust_angle, meta


def grid_line_quality(mask: np.ndarray) -> Dict[str, float]:
    """Returns simple quality metrics: number of horizontal/vertical grid peaks."""
    horiz, vert = separate_grid_lines(mask)
    x_positions = extract_line_positions(vert, axis="vertical")
    y_positions = extract_line_positions(horiz, axis="horizontal")

    num_vert = int(len(x_positions))
    num_horiz = int(len(y_positions))
    quality = float(min(num_vert, 40) + min(num_horiz, 40))

    return {
        "num_vert_lines": float(num_vert),
        "num_horiz_lines": float(num_horiz),
        "quality_score": quality,
    }


def extract_line_positions(mask: np.ndarray, axis: str) -> np.ndarray:
    if axis == "vertical":
        proj = (mask > 0).sum(axis=0).astype(np.float32)
    else:
        proj = (mask > 0).sum(axis=1).astype(np.float32)

    proj = smooth_1d(proj, 9)
    if np.max(proj) <= 0:
        return np.empty(0, dtype=np.float32)

    thr = 0.35 * float(np.max(proj))

    positions = []
    in_peak = False
    start = 0
    for i, val in enumerate(proj):
        if val >= thr and not in_peak:
            in_peak = True
            start = i
        elif val < thr and in_peak:
            end = i - 1
            positions.append((start + end) / 2.0)
            in_peak = False
    if in_peak:
        positions.append((start + len(proj) - 1) / 2.0)

    return np.array(positions, dtype=np.float32)


def dominant_spacing(positions: np.ndarray) -> float | None:
    if len(positions) < 2:
        return None

    diffs = np.diff(np.sort(positions))
    diffs = diffs[(diffs > 2) & (diffs < 500)]
    if len(diffs) == 0:
        return None

    hist, bins = np.histogram(diffs, bins=min(50, max(10, len(diffs) // 2)))
    idx = np.argmax(hist)
    lo, hi = bins[idx], bins[idx + 1]
    vals = diffs[(diffs >= lo) & (diffs <= hi)]

    if len(vals) == 0:
        return float(np.median(diffs))
    return float(np.median(vals))


def dominant_period_fft(signal_1d: np.ndarray, min_period: int = 4, max_period: int = 200) -> float | None:
    s = signal_1d.astype(np.float32)
    s = s - np.mean(s)
    if np.std(s) < 1e-6:
        return None

    n = len(s)
    fft = np.fft.rfft(s)
    mag = np.abs(fft)
    freqs = np.fft.rfftfreq(n, d=1.0)

    valid = freqs > 0
    freqs = freqs[valid]
    mag = mag[valid]
    periods = 1.0 / freqs

    valid2 = (periods >= min_period) & (periods <= max_period)
    if not np.any(valid2):
        return None

    periods = periods[valid2]
    mag = mag[valid2]
    return float(periods[np.argmax(mag)])


def fuse_spacing(a: float | None, b: float | None) -> float | None:
    if a is None and b is None:
        return None
    if a is None:
        return b
    if b is None:
        return a

    rel = abs(a - b) / max(a, b, 1e-6)
    if rel < 0.15:
        return float(0.5 * (a + b))
    return float(a)


def estimate_grid_spacing(horiz_mask: np.ndarray, vert_mask: np.ndarray, gray_eq: np.ndarray):
    x_positions = extract_line_positions(vert_mask, axis="vertical")
    y_positions = extract_line_positions(horiz_mask, axis="horizontal")

    dx_hough = dominant_spacing(x_positions)
    dy_hough = dominant_spacing(y_positions)

    inv = 255 - gray_eq.astype(np.uint8)
    proj_x = inv.sum(axis=0).astype(np.float32)
    proj_y = inv.sum(axis=1).astype(np.float32)

    dx_fft = dominant_period_fft(proj_x, 4, 200)
    dy_fft = dominant_period_fft(proj_y, 4, 200)

    dx = fuse_spacing(dx_hough, dx_fft)
    dy = fuse_spacing(dy_hough, dy_fft)

    meta = {
        "dx_hough": dx_hough,
        "dy_hough": dy_hough,
        "dx_fft": dx_fft,
        "dy_fft": dy_fft,
        "num_vert_lines": len(x_positions),
        "num_horiz_lines": len(y_positions),
        "x_positions": x_positions,
        "y_positions": y_positions,
        "proj_x": proj_x,
        "proj_y": proj_y,
    }
    return dx, dy, meta


def estimate_grid_phase_x(vert_mask: np.ndarray, dx: float) -> float:
    x_positions = extract_line_positions(vert_mask, axis="vertical")
    if dx is None or len(x_positions) == 0:
        return 0.0

    mods = np.mod(x_positions, dx)
    hist, bins = np.histogram(mods, bins=32, range=(0, dx))
    idx = np.argmax(hist)
    lo, hi = bins[idx], bins[idx + 1]
    vals = mods[(mods >= lo) & (mods <= hi)]
    if len(vals) == 0:
        return float(np.median(mods))
    return float(np.median(vals))


def estimate_grid_phase_y(horiz_mask: np.ndarray, dy: float) -> float:
    y_positions = extract_line_positions(horiz_mask, axis="horizontal")
    if dy is None or len(y_positions) == 0:
        return 0.0

    mods = np.mod(y_positions, dy)
    hist, bins = np.histogram(mods, bins=32, range=(0, dy))
    idx = np.argmax(hist)
    lo, hi = bins[idx], bins[idx + 1]
    vals = mods[(mods >= lo) & (mods <= hi)]
    if len(vals) == 0:
        return float(np.median(mods))
    return float(np.median(vals))


def estimate_grid_confidence(meta: Dict[str, object], dx: float | None, dy: float | None, x0: float, y0: float) -> float:
    score = 0.0

    if dx is not None:
        score += 0.2
    if dy is not None:
        score += 0.2

    if int(meta["num_vert_lines"]) >= 5:
        score += 0.15
    if int(meta["num_horiz_lines"]) >= 5:
        score += 0.15

    if meta["dx_hough"] is not None and meta["dx_fft"] is not None:
        rel = abs(float(meta["dx_hough"]) - float(meta["dx_fft"])) / max(float(meta["dx_hough"]), float(meta["dx_fft"]), 1e-6)
        score += max(0.0, 0.15 * (1.0 - rel / 0.15))

    if meta["dy_hough"] is not None and meta["dy_fft"] is not None:
        rel = abs(float(meta["dy_hough"]) - float(meta["dy_fft"])) / max(float(meta["dy_hough"]), float(meta["dy_fft"]), 1e-6)
        score += max(0.0, 0.15 * (1.0 - rel / 0.15))

    return float(np.clip(score, 0.0, 1.0))


def normalize_and_calibrate(rgb: np.ndarray) -> Stage1Geometry:
    rgb = crop_non_black(rgb)
    rgb = resize_long_side(rgb, 2200)

    gray0 = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray_eq0 = clahe_gray(gray0)

    grid_mask0 = build_grid_mask(rgb, gray_eq0)
    grid_angle, grid_meta = estimate_rotation_from_grid_mask(grid_mask0)
    edge_angle, edge_meta = estimate_rotation_from_edges(gray0)
    pre_q = grid_line_quality(grid_mask0)

    selected_meta: Dict[str, float | int | str]
    proposed_angle = 0.0
    rotation_source = "none"
    edge_ok = str(edge_meta.get("rotation_reason", "")) == "ok"
    grid_ok = str(grid_meta.get("rotation_reason", "")) == "ok"

    if edge_ok:
        proposed_angle = float(edge_angle)
        selected_meta = edge_meta
        rotation_source = "edges"
    elif grid_ok:
        proposed_angle = float(grid_angle)
        selected_meta = grid_meta
        rotation_source = "grid"
    else:
        if int(edge_meta.get("rotation_candidates", 0)) >= int(grid_meta.get("rotation_candidates", 0)):
            selected_meta = edge_meta
            rotation_source = "edges"
        else:
            selected_meta = grid_meta
            rotation_source = "grid"

    angle = 0.0
    rotation_applied = False
    rotation_reason = str(selected_meta.get("rotation_reason", "unknown"))
    post_q_score = -1.0
    post_angle_residual = -1.0

    inliers0 = int(selected_meta.get("rotation_inliers", 0))
    spread0 = float(selected_meta.get("rotation_spread_deg", -1.0))
    num_vert0 = int(pre_q["num_vert_lines"])
    num_horiz0 = int(pre_q["num_horiz_lines"])
    min_inliers = 20 if rotation_source == "edges" else 8
    max_spread = 2.0 if rotation_source == "edges" else 3.5

    eligible = (
        rotation_reason == "ok"
        and abs(proposed_angle) > 0.2
        and inliers0 >= min_inliers
        and 0.0 <= spread0 <= max_spread
    )
    if rotation_source == "grid":
        eligible = eligible and num_vert0 >= 3 and num_horiz0 >= 3

    if eligible:
        rgb_rot = rotate_array(rgb, proposed_angle)
        rgb_rot = crop_non_black(rgb_rot)

        gray_rot = cv2.cvtColor(rgb_rot, cv2.COLOR_RGB2GRAY)
        gray_eq_rot = clahe_gray(gray_rot)
        grid_mask_rot = build_grid_mask(rgb_rot, gray_eq_rot)
        post_q = grid_line_quality(grid_mask_rot)
        post_q_score = float(post_q["quality_score"])

        if rotation_source == "edges":
            _, post_meta = estimate_rotation_from_edges(gray_rot)
        else:
            _, post_meta = estimate_rotation_from_grid_mask(grid_mask_rot)
            if str(post_meta.get("rotation_reason", "")) != "ok":
                _, post_meta = estimate_rotation_from_edges(gray_rot)
        post_angle_residual = abs(float(post_meta.get("rotation_raw_deg", 0.0)))

        quality_gain = post_q_score - float(pre_q["quality_score"])
        residual_better = post_angle_residual <= max(0.5, 0.55 * abs(proposed_angle))
        quality_ok = quality_gain >= -1.0

        if residual_better and quality_ok:
            rgb = rgb_rot
            angle = proposed_angle
            rotation_applied = True
            rotation_reason = "applied"
        else:
            rotation_reason = "rejected_no_gain"
    else:
        if rotation_reason != "ok":
            rotation_reason = f"skip_{rotation_reason}"
        elif abs(proposed_angle) <= 0.2:
            rotation_reason = "skip_small_angle"
        elif inliers0 < min_inliers:
            rotation_reason = "skip_low_support"
        elif spread0 > max_spread:
            rotation_reason = "skip_high_spread"
        elif rotation_source == "grid" and (num_vert0 < 3 or num_horiz0 < 3):
            rotation_reason = "skip_few_grid_lines"

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray_eq = clahe_gray(gray)

    grid_mask = build_grid_mask(rgb, gray_eq)
    horiz_mask, vert_mask = separate_grid_lines(grid_mask)

    dx, dy, meta = estimate_grid_spacing(horiz_mask, vert_mask, gray_eq)

    square_grid_assumed = False
    if dx is None and dy is None:
        dx = dy = float(np.clip(rgb.shape[1] / 250.0, 4.0, 30.0))
        square_grid_assumed = True
    elif dx is None:
        dx = float(dy)
        square_grid_assumed = True
    elif dy is None:
        dy = float(dx)
        square_grid_assumed = True

    x0 = estimate_grid_phase_x(vert_mask, dx)
    y0 = estimate_grid_phase_y(horiz_mask, dy)

    conf = estimate_grid_confidence(meta, dx, dy, x0, y0)

    debug = {
        "grid_mask": grid_mask,
        "horiz_mask": horiz_mask,
        "vert_mask": vert_mask,
        "rotation_deg": angle,
        "rotation_source": rotation_source,
        "rotation_raw_deg": float(selected_meta.get("rotation_raw_deg", proposed_angle)),
        "rotation_candidates": int(selected_meta.get("rotation_candidates", 0)),
        "rotation_inliers": inliers0,
        "rotation_spread_deg": spread0,
        "rotation_pre_quality": float(pre_q["quality_score"]),
        "rotation_post_quality": float(post_q_score),
        "rotation_post_residual_deg": float(post_angle_residual),
        "rotation_applied": int(rotation_applied),
        "rotation_reason": rotation_reason,
        "rotation_grid_raw_deg": float(grid_meta.get("rotation_raw_deg", 0.0)),
        "rotation_grid_reason": str(grid_meta.get("rotation_reason", "unknown")),
        "rotation_edge_raw_deg": float(edge_meta.get("rotation_raw_deg", 0.0)),
        "rotation_edge_reason": str(edge_meta.get("rotation_reason", "unknown")),
        "dx_hough": meta["dx_hough"] if meta["dx_hough"] is not None else -1.0,
        "dy_hough": meta["dy_hough"] if meta["dy_hough"] is not None else -1.0,
        "dx_fft": meta["dx_fft"] if meta["dx_fft"] is not None else -1.0,
        "dy_fft": meta["dy_fft"] if meta["dy_fft"] is not None else -1.0,
        "num_vert_lines": int(meta["num_vert_lines"]),
        "num_horiz_lines": int(meta["num_horiz_lines"]),
    }

    return Stage1Geometry(
        rgb=rgb,
        gray=gray.astype(np.float32),
        minor_dx_px=float(dx),
        minor_dy_px=float(dy),
        grid_x0=float(x0),
        grid_y0=float(y0),
        rotation_deg=float(angle),
        confidence=float(conf),
        square_grid_assumed=bool(square_grid_assumed),
        debug=debug,
    )


# =========================
# Layout / row detection
# =========================

def detect_row_centers(rgb: np.ndarray) -> np.ndarray:
    """
    Finds 4 row centers using darkest pixels (signal/text) profile.
    Works better after normalization/deskew.
    """
    gray = rgb.mean(axis=2)
    thr = np.percentile(gray, 8.0)
    dark = gray <= thr
    profile = dark.sum(axis=1).astype(np.float32)
    profile = smooth_1d(profile, max(7, rgb.shape[0] // 90))

    min_dist = max(20, rgb.shape[0] // 8)
    order = np.argsort(profile)[::-1]
    picked: List[int] = []
    for idx in order:
        if all(abs(int(idx) - p) >= min_dist for p in picked):
            picked.append(int(idx))
        if len(picked) == 4:
            break

    if len(picked) < 4:
        h = rgb.shape[0]
        return np.array([int(h * t) for t in (0.27, 0.43, 0.59, 0.76)], dtype=np.int32)
    return np.array(sorted(picked), dtype=np.int32)


# =========================
# Signal extraction
# =========================

def adaptive_signal_mask(roi_gray: np.ndarray) -> np.ndarray:
    """
    More stable than a single percentile threshold.
    Searches for a dark-foreground mask with plausible density.
    """
    roi_u8 = np.clip(roi_gray, 0, 255).astype(np.uint8)
    otsu_thr, _ = cv2.threshold(roi_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    best = None
    for factor in np.arange(1.0, 0.55, -0.05):
        thr = int(round(float(otsu_thr) * float(factor)))
        mask = roi_u8 <= thr
        density = float(mask.mean())
        if 0.002 < density < 0.20:
            best = mask
            break

    if best is None:
        best = roi_u8 <= otsu_thr

    return best.astype(bool)


def contiguous_centers(binary_col: np.ndarray) -> np.ndarray:
    ys = np.flatnonzero(binary_col)
    if ys.size == 0:
        return np.empty(0, dtype=np.float32)

    groups = []
    start = ys[0]
    prev = ys[0]
    for y in ys[1:]:
        if y == prev + 1:
            prev = y
        else:
            groups.append((start, prev))
            start = y
            prev = y
    groups.append((start, prev))
    return np.array([(a + b) / 2.0 for a, b in groups], dtype=np.float32)


def extract_trace(
    roi_gray: np.ndarray,
    target_len: int,
    px_per_millivolt: float,
) -> np.ndarray:
    """
    Column-wise path extraction with continuity preference.
    Improved version:
    - adaptive threshold mask,
    - contiguous component centers instead of all dark pixels,
    - stronger fallback continuity.
    """
    h, w = roi_gray.shape
    if h < 4 or w < 4:
        return np.zeros(target_len, dtype=np.float32)

    mask = adaptive_signal_mask(roi_gray)

    # Light cleanup
    mask_u8 = (mask.astype(np.uint8) * 255)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    mask = mask_u8 > 0

    y_vals = np.empty(w, dtype=np.float32)
    has_prev = False
    prev = 0.0

    for x in range(w):
        cands = contiguous_centers(mask[:, x])
        if cands.size == 0:
            y_vals[x] = prev if has_prev else (h / 2.0)
            continue

        if not has_prev:
            y = float(np.median(cands))
            has_prev = True
        else:
            idx = int(np.argmin(np.abs(cands - prev)))
            y = float(cands[idx])

        y_vals[x] = y
        prev = y

    y_vals = smooth_1d(y_vals, max(5, w // 160))
    baseline = float(np.median(y_vals))

    sig = (baseline - y_vals) / max(1e-6, px_per_millivolt)
    sig -= np.median(sig)

    x_old = np.linspace(0.0, 1.0, num=sig.size, dtype=np.float32)
    x_new = np.linspace(0.0, 1.0, num=target_len, dtype=np.float32)
    out = np.interp(x_new, x_old, sig).astype(np.float32)

    return np.clip(out, -5.0, 5.0)


# =========================
# Main digitization logic
# =========================

def digitize_image(path: Path, debug_dir: Optional[Path] = None) -> Dict[str, np.ndarray]:
    rgb0 = load_rgb(path)
    stage1 = normalize_and_calibrate(rgb0)
    rec_dir: Optional[Path] = None

    if debug_dir is not None:
        rec_dir = debug_dir / path.stem
        rec_dir.mkdir(parents=True, exist_ok=True)
        dump_stage1_debug(rec_dir, rgb0, stage1)

    rgb = stage1.rgb
    gray = stage1.gray
    h, w, _ = rgb.shape

    # Separate X/Y grid spacing
    minor_px_x = stage1.minor_dx_px
    minor_px_y = stage1.minor_dy_px

    # For amplitude: 10 mm = 1 mV
    px_per_millivolt = minor_px_y * 10.0

    row_centers = detect_row_centers(rgb)
    row_gap = int(max(12, np.median(np.diff(row_centers))))
    half_band = int(max(10, row_gap * 0.32))

    x_split = np.linspace(0, w, num=5, dtype=np.int32)
    if rec_dir is not None:
        dump_layout_debug(rec_dir, rgb, row_centers, half_band, x_split)

    signals: Dict[str, np.ndarray] = {}

    # 3 rows x 4 columns => 12 short leads (2.5 s = 1250 samples)
    for row_idx, leads in enumerate(ROW_LAYOUT):
        yc = int(row_centers[row_idx])
        y0 = max(0, yc - half_band)
        y1 = min(h, yc + half_band)

        for col_idx, lead_name in enumerate(leads):
            c0, c1 = int(x_split[col_idx]), int(x_split[col_idx + 1])

            seg_w = c1 - c0
            x0 = max(0, c0 + int(0.04 * seg_w))
            x1 = min(w, c1 - int(0.02 * seg_w))

            roi = gray[y0:y1, x0:x1]
            if rec_dir is not None:
                dump_roi_debug(rec_dir, f"lead_{lead_name}_r{row_idx+1}c{col_idx+1}", roi)
            signals[lead_name] = extract_trace(
                roi,
                target_len=1250,
                px_per_millivolt=px_per_millivolt,
            )

    # Rhythm strip (row 4) for lead II: 10 s = 5000 samples.
    yc = int(row_centers[3])
    y0 = max(0, yc - half_band)
    y1 = min(h, yc + half_band)
    x0 = int(0.04 * w)
    x1 = int(0.96 * w)
    roi = gray[y0:y1, x0:x1]
    if rec_dir is not None:
        dump_roi_debug(rec_dir, "lead_II_rhythm", roi)
    signals["II"] = extract_trace(
        roi,
        target_len=5000,
        px_per_millivolt=px_per_millivolt,
    )

    # Ensure all expected keys are present.
    for lead in LEADS_ORDER:
        if lead not in signals:
            signals[lead] = np.zeros(1250 if lead != "II" else 5000, dtype=np.float32)

    return signals


# =========================
# Submission build
# =========================

def build_submission(
    input_dir: Path,
    output_npz: Path,
    limit: int | None = None,
    debug_dir: Optional[Path] = None,
) -> Path:
    images = sorted(input_dir.glob("*.png"))
    if limit is not None:
        images = images[:limit]

    submission: Dict[str, np.ndarray] = {}
    for i, img_path in enumerate(images, start=1):
        rec = img_path.stem
        sigs = digitize_image(img_path, debug_dir=debug_dir)
        for lead in LEADS_ORDER:
            key = f"{rec}_{lead}"
            submission[key] = sigs[lead].astype(np.float16)
        if i % 50 == 0 or i == len(images):
            print(f"[{i}/{len(images)}] processed {img_path.name}")

    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_npz, **submission)
    return output_npz


def maybe_submit(npz_path: Path, endpoint: str = "task4") -> None:
    try:
        import requests
        from dotenv import load_dotenv
    except Exception as exc:
        raise RuntimeError(
            "Submission requires requests + python-dotenv. Install dependencies first."
        ) from exc

    root = Path(__file__).resolve().parents[1]
    load_dotenv(root / ".env")
    token = os.getenv("TEAM_TOKEN")
    server_url = os.getenv("SERVER_URL")
    if not token or not server_url:
        raise RuntimeError("Missing TEAM_TOKEN or SERVER_URL in .env")

    with npz_path.open("rb") as f:
        response = requests.post(
            f"{server_url.rstrip('/')}/{endpoint}",
            files={"npz_file": f},
            headers={"X-API-Token": token},
            timeout=600,
        )
    try:
        payload = response.json()
    except Exception:
        payload = response.text
    print("submission response:", response.status_code, payload)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Task 4 ECG digitization pipeline v2.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=root / "test",
        help="Directory with ECG PNG files.",
    )
    parser.add_argument(
        "--output-npz",
        type=Path,
        default=root / "data" / "out" / "task4_submission_v2.npz",
        help="Where to save submission .npz.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Process only first N files.")
    parser.add_argument(
        "--debug-dir",
        type=Path,
        default=None,
        help="Optional directory for saving per-image debug artifacts.",
    )
    parser.add_argument("--submit", action="store_true", help="Submit to server after generation.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    npz_path = build_submission(
        args.input_dir,
        args.output_npz,
        limit=args.limit,
        debug_dir=args.debug_dir,
    )
    print(f"saved: {npz_path}")
    if args.submit:
        maybe_submit(npz_path)


if __name__ == "__main__":
    main()
