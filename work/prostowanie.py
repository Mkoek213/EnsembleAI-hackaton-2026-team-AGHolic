#!/usr/bin/env python3
"""Task 4 ECG digitization pipeline v2 - Hackathon Optimized.

Changes:
- Replaced global Hough rotation with ROI-based document corner detection.
- Added adaptive X-axis boundaries for time trimming.
- Added median filter baseline wander removal.
- Added cubic spline interpolation to preserve R-peak amplitudes.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Optional

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import median_filter
from scipy.interpolate import interp1d

LEADS_ORDER = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]
ROW_LAYOUT = [
    ["I", "AVR", "V1", "V4"],
    ["II", "AVL", "V2", "V5"],
    ["III", "AVF", "V3", "V6"],
]

# --- PARAMETRY NOWEJ ROTACJI ---
THRESHOLD_BASE = 30
BLUR_KERNEL = (5, 5)
CANNY_LOW = 150
CANNY_HIGH = 220
HOUGH_RHO = 1
HOUGH_THETA = np.pi / 180
HOUGH_THRESHOLD = 30
HOUGH_MIN_LINE_LENGTH = 20
HOUGH_MAX_LINE_GAP = 5

# --- STABILIZACJA SKALI AMPLITUDY ---
# Korekta globalna: lekko zwiększa amplitudę mV (mniejsze px/mV).
AMP_SCALE_CALIB = 1.00
# Kalibracja pulsem bywa niestabilna, więc tylko mały udział.
PULSE_BLEND = 0.12
PULSE_RATIO_MIN = 0.45
PULSE_RATIO_MAX = 1.45

# --- STABILIZACJA PODZIAŁÓW LEADÓW ---
USE_ROW_LOCAL_REFINEMENT = False
FORCE_FIXED_WINDOWS = True
FORCE_FIXED_ROWS = False
FIXED_X_SPLIT_RATIO = (0.05, 0.285, 0.52, 0.755, 0.99)
FIXED_RHYTHM_X_BOUNDS_RATIO = (0.05, 0.99)
FIXED_ROW_CENTERS_RATIO = (0.27, 0.43, 0.59, 0.76)
FIXED_HALF_BAND_RATIO = 0.052
# Ręczna korekta seamów w pikselach (wariant, który dawał dotąd najlepszy progres).
SEAM_LEFT_SHIFT_PX = (0, 10, 50, 50, 0)
RHYTHM_LEFT_SHIFT_PX = 5
ENABLE_LEFT_BIASED_TUNING = False

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

def load_rgb(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))

def crop_non_black(rgb: np.ndarray, min_non_black: int = 8) -> np.ndarray:
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

def save_image(path: Path, arr: np.ndarray) -> None:
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


def expected_lead_len(lead: str) -> int:
    return 5000 if lead == "II" else 1250


def resample_1d(sig: np.ndarray, n: int) -> np.ndarray:
    if sig.size == n:
        return sig.astype(np.float32)
    xp = np.linspace(0.0, 1.0, num=sig.size, dtype=np.float32)
    x = np.linspace(0.0, 1.0, num=n, dtype=np.float32)
    return np.interp(x, xp, sig).astype(np.float32)


def apply_lead_remap(signals: Dict[str, np.ndarray], lead_remap: Optional[Dict[str, str]]) -> Dict[str, np.ndarray]:
    if not lead_remap:
        return signals

    remapped: Dict[str, np.ndarray] = dict(signals)
    for src, dst in lead_remap.items():
        s = str(src).upper()
        d = str(dst).upper()
        if s not in signals or d not in LEADS_ORDER:
            continue
        arr = signals[s].astype(np.float32)
        exp_n = expected_lead_len(d)
        remapped[d] = resample_1d(arr, exp_n) if arr.size != exp_n else arr
    return remapped


def load_lead_remap(path: Optional[Path]) -> Optional[Dict[str, str]]:
    if path is None:
        return None
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Lead remap file must contain JSON object, got: {type(payload)}")
    out: Dict[str, str] = {}
    for k, v in payload.items():
        ks = str(k).upper()
        vs = str(v).upper()
        if ks in LEADS_ORDER and vs in LEADS_ORDER:
            out[ks] = vs
    return out or None

# --- NOWE FUNKCJE ROTACJI ---
def rotation_to_nearest_right_angle(detected_angle: float) -> float:
    nearest_right_angle = round(detected_angle / 90.0) * 90.0
    return nearest_right_angle - detected_angle

def roi_brightness(roi_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_RGB2GRAY)
    return float(np.mean(gray))

def detect_best_line_in_roi(roi_bgr: np.ndarray, x_offset: int, y_offset: int):
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, THRESHOLD_BASE, 255, cv2.THRESH_BINARY)
    blurred = cv2.GaussianBlur(thresh, BLUR_KERNEL, 2)
    edges = cv2.Canny(blurred, CANNY_LOW, CANNY_HIGH)

    lines = cv2.HoughLinesP(
        edges,
        rho=HOUGH_RHO,
        theta=HOUGH_THETA,
        threshold=HOUGH_THRESHOLD,
        minLineLength=HOUGH_MIN_LINE_LENGTH,
        maxLineGap=HOUGH_MAX_LINE_GAP,
    )

    if lines is None:
        return None

    best = None
    best_len = 0.0
    for line in lines[:, 0]:
        x1, y1, x2, y2 = line
        length = math.hypot(x2 - x1, y2 - y1)
        if length > best_len:
            best_len = length
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
            best = (angle, length, (x1 + x_offset, y1 + y_offset, x2 + x_offset, y2 + y_offset))
    return best

# --- FUNKCJE GRID ---
def build_grid_mask(rgb: np.ndarray, gray_eq: np.ndarray) -> np.ndarray:
    rgb_f = rgb.astype(np.float32)
    r, g, b = rgb_f[:, :, 0], rgb_f[:, :, 1], rgb_f[:, :, 2]

    red_score = r - 0.5 * (g + b)
    red_score = cv2.GaussianBlur(red_score, (5, 5), 0)
    red_score_u8 = cv2.normalize(red_score, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, red_mask = cv2.threshold(red_score_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    Z = gray_eq.reshape((-1, 1)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(Z, 4, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    order = np.argsort(centers.flatten())
    km_mask = (labels.reshape(gray_eq.shape) == order[len(order) // 2]).astype(np.uint8) * 255

    blur = cv2.GaussianBlur(gray_eq, (0, 0), 7)
    high = cv2.addWeighted(gray_eq, 1.5, blur, -0.5, 0)
    _, hp_mask = cv2.threshold(high, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if float((hp_mask > 0).mean()) < 0.15:
        hp_mask = cv2.bitwise_not(hp_mask)

    mask = cv2.bitwise_or(red_mask, km_mask)
    mask = cv2.bitwise_or(mask, hp_mask)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

def separate_grid_lines(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h, w = mask.shape[:2]
    kx, ky = max(15, w // 80), max(15, h // 80)
    horiz = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (kx, 1)))
    vert = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, ky)))
    horiz = cv2.morphologyEx(horiz, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    vert = cv2.morphologyEx(vert, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    return horiz, vert

def extract_line_positions(mask: np.ndarray, axis: str) -> np.ndarray:
    proj = (mask > 0).sum(axis=0 if axis == "vertical" else 1).astype(np.float32)
    proj = smooth_1d(proj, 9)
    if np.max(proj) <= 0: return np.empty(0, dtype=np.float32)
    
    thr = 0.35 * float(np.max(proj))
    positions = []
    in_peak = False
    start = 0
    for i, val in enumerate(proj):
        if val >= thr and not in_peak:
            in_peak, start = True, i
        elif val < thr and in_peak:
            positions.append((start + i - 1) / 2.0)
            in_peak = False
    if in_peak: positions.append((start + len(proj) - 1) / 2.0)
    return np.array(positions, dtype=np.float32)

def dominant_spacing(positions: np.ndarray) -> float | None:
    if len(positions) < 2: return None
    diffs = np.diff(np.sort(positions))
    diffs = diffs[(diffs > 2) & (diffs < 500)]
    if len(diffs) == 0: return None
    hist, bins = np.histogram(diffs, bins=min(50, max(10, len(diffs) // 2)))
    idx = np.argmax(hist)
    vals = diffs[(diffs >= bins[idx]) & (diffs <= bins[idx + 1])]
    return float(np.median(vals)) if len(vals) > 0 else float(np.median(diffs))

def dominant_period_fft(signal_1d: np.ndarray) -> float | None:
    s = signal_1d.astype(np.float32) - np.mean(signal_1d)
    if np.std(s) < 1e-6: return None
    freqs = np.fft.rfftfreq(len(s), d=1.0)
    mag = np.abs(np.fft.rfft(s))[freqs > 0]
    periods = 1.0 / freqs[freqs > 0]
    valid2 = (periods >= 4) & (periods <= 200)
    if not np.any(valid2): return None
    return float(periods[valid2][np.argmax(mag[valid2])])

def estimate_grid_spacing(horiz_mask: np.ndarray, vert_mask: np.ndarray, gray_eq: np.ndarray):
    x_pos = extract_line_positions(vert_mask, "vertical")
    y_pos = extract_line_positions(horiz_mask, "horizontal")
    dx_h, dy_h = dominant_spacing(x_pos), dominant_spacing(y_pos)
    
    inv = 255 - gray_eq.astype(np.uint8)
    dx_f, dy_f = dominant_period_fft(inv.sum(axis=0)), dominant_period_fft(inv.sum(axis=1))
    
    def fuse(a, b):
        if a is None or b is None: return a or b
        return float(0.5 * (a + b)) if abs(a - b) / max(a, b, 1e-6) < 0.15 else float(a)
        
    return fuse(dx_h, dx_f), fuse(dy_h, dy_f), {
        "dx_hough": dx_h, "dy_hough": dy_h, "dx_fft": dx_f, "dy_fft": dy_f,
        "num_vert_lines": len(x_pos), "num_horiz_lines": len(y_pos)
    }

def estimate_grid_phase(mask: np.ndarray, d: float, axis: str) -> float:
    pos = extract_line_positions(mask, axis)
    if d is None or len(pos) == 0: return 0.0
    mods = np.mod(pos, d)
    hist, bins = np.histogram(mods, bins=32, range=(0, d))
    idx = np.argmax(hist)
    vals = mods[(mods >= bins[idx]) & (mods <= bins[idx + 1])]
    return float(np.median(vals)) if len(vals) > 0 else float(np.median(mods))

def sanitize_minor_spacing(dx: float, dy: float, width_px: int) -> tuple[float, float, Dict[str, float]]:
    expected = float(np.clip(width_px / 260.0, 4.0, 20.0))
    def pick(v: float) -> float:
        cands = [c for c in [v, v / 5.0, v * 5.0, v / 2.0, v * 2.0] if np.isfinite(c) and c > 0]
        if not cands: return expected
        c = min(cands, key=lambda z: abs(np.log(max(z, 1e-6) / expected)))
        return float(c) if expected * 0.45 <= c <= expected * 2.2 else expected

    dx0, dy0 = float(dx), float(dy)
    dx, dy = pick(dx0), pick(dy0)

    ratio = dy / max(dx, 1e-6)
    if ratio > 1.35:
        dy = dy / 5.0 if 3.5 < ratio < 6.5 else (dy / 2.0 if 1.7 < ratio < 2.7 else dx)
    elif ratio < 0.74:
        inv = 1.0 / max(ratio, 1e-6)
        dy = dy * 5.0 if 3.5 < inv < 6.5 else (dy * 2.0 if 1.7 < inv < 2.7 else dx)

    return float(np.clip(dx, expected * 0.45, expected * 2.2)), float(np.clip(dy, expected * 0.45, expected * 2.2)), {
        "spacing_expected": expected, "spacing_dx_raw": dx0, "spacing_dy_raw": dy0,
        "spacing_dx_sanitized": dx, "spacing_dy_sanitized": dy
    }

# --- GŁÓWNA NORMALIZACJA ---
def normalize_and_calibrate(rgb: np.ndarray) -> Stage1Geometry:
    rgb = crop_non_black(rgb)
    rgb = resize_long_side(rgb, 2200)

    # Nowy, szybki mechanizm obrotu z ROI
    h, w = rgb.shape[:2]
    roi1_h, roi1_w = max(1, int(h * 0.3)), max(1, int(w * 0.1))
    roi2_h, roi2_w = max(1, int(h * 0.1)), max(1, int(w * 0.3))

    roi1, roi2 = rgb[0:roi1_h, 0:roi1_w], rgb[0:roi2_h, 0:roi2_w]
    b1, b2 = roi_brightness(roi1), roi_brightness(roi2)

    selected_roi, fallback_roi = (roi1, roi2) if b1 <= b2 else (roi2, roi1)
    
    cand1 = detect_best_line_in_roi(selected_roi, 0, 0)
    best_candidate = cand1 if cand1 else detect_best_line_in_roi(fallback_roi, 0, 0)
    
    angle = 0.0
    if best_candidate is not None:
        cand_angle, _, _ = best_candidate
        angle = -rotation_to_nearest_right_angle(cand_angle)

    if abs(angle) > 0.1:
        M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
        rgb = cv2.warpAffine(rgb, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        # KRYTYCZNE: odcinamy czarne marginesy wygenerowane przez obrót!
        rgb = crop_non_black(rgb)

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray_eq = clahe_gray(gray)

    grid_mask = build_grid_mask(rgb, gray_eq)
    horiz_mask, vert_mask = separate_grid_lines(grid_mask)
    dx, dy, meta = estimate_grid_spacing(horiz_mask, vert_mask, gray_eq)

    square_grid_assumed = False
    if dx is None and dy is None:
        dx = dy = float(np.clip(rgb.shape[1] / 250.0, 4.0, 30.0))
        square_grid_assumed = True
    elif dx is None: dx, square_grid_assumed = float(dy), True
    elif dy is None: dy, square_grid_assumed = float(dx), True

    dx, dy, spacing_meta = sanitize_minor_spacing(dx, dy, rgb.shape[1])
    x0, y0 = estimate_grid_phase(vert_mask, dx, "vertical"), estimate_grid_phase(horiz_mask, dy, "horizontal")

    # Wypełniamy dummy meta, aby nie zepsuć Twojego dump_stage1_debug
    debug = {
        "grid_mask": grid_mask, "horiz_mask": horiz_mask, "vert_mask": vert_mask,
        "rotation_deg": angle, "rotation_source": "roi",
        "rotation_raw_deg": angle, "rotation_candidates": 1, "rotation_inliers": 1,
        "rotation_spread_deg": 0.0, "rotation_pre_quality": 0.0, "rotation_post_quality": 0.0,
        "rotation_post_residual_deg": 0.0, "rotation_applied": int(abs(angle) > 0.1),
        "rotation_reason": "ok", "rotation_grid_raw_deg": 0.0, "rotation_grid_reason": "",
        "rotation_edge_raw_deg": 0.0, "rotation_edge_reason": "",
        "dx_hough": meta["dx_hough"] or -1.0, "dy_hough": meta["dy_hough"] or -1.0,
        "dx_fft": meta["dx_fft"] or -1.0, "dy_fft": meta["dy_fft"] or -1.0,
        "num_vert_lines": int(meta["num_vert_lines"]), "num_horiz_lines": int(meta["num_horiz_lines"]),
        "spacing_expected": float(spacing_meta["spacing_expected"]),
        "spacing_dx_raw": float(spacing_meta["spacing_dx_raw"]),
        "spacing_dy_raw": float(spacing_meta["spacing_dy_raw"]),
        "spacing_dx_sanitized": float(spacing_meta["spacing_dx_sanitized"]),
        "spacing_dy_sanitized": float(spacing_meta["spacing_dy_sanitized"]),
    }

    return Stage1Geometry(rgb, gray.astype(np.float32), dx, dy, x0, y0, angle, 1.0, square_grid_assumed, debug)

# --- DETEKCJA WIERZY I CZASU ---
def detect_row_centers(
    rgb: np.ndarray,
    min_row_y_ratio: float = 0.12,
    max_row_y_ratio: float = 0.95,
) -> np.ndarray:
    gray = rgb.mean(axis=2)
    work = gray[:, int(0.08 * gray.shape[1]):int(0.98 * gray.shape[1])]
    dark = work <= np.percentile(work, 10.0)
    h = rgb.shape[0]
    profile = smooth_1d(dark.sum(axis=1).astype(np.float32), max(7, h // 90))
    y_min = int(np.clip(round(h * float(min_row_y_ratio)), 0, max(0, h - 1)))
    y_max = int(np.clip(round(h * float(max_row_y_ratio)), y_min + 1, h))
    min_dist = max(20, h // 8)
    valid_rows = np.arange(y_min, y_max, dtype=np.int32)

    picked = []
    for idx in valid_rows[np.argsort(profile[valid_rows])[::-1]]:
        if all(abs(int(idx) - p) >= min_dist for p in picked):
            picked.append(int(idx))
        if len(picked) == 4:
            break
    if len(picked) < 4:
        fallback = np.array([int(h * t) for t in (0.27, 0.43, 0.59, 0.76)], dtype=np.int32)
        return np.clip(fallback, y_min, y_max - 1)
    return np.array(sorted(picked), dtype=np.int32)

def contiguous_runs(indices: np.ndarray) -> List[Tuple[int, int]]:
    if indices.size == 0: return []
    runs, start, prev = [], int(indices[0]), int(indices[0])
    for x in indices[1:]:
        x = int(x)
        if x == prev + 1: prev = x
        else:
            runs.append((start, prev))
            start = prev = x
    runs.append((start, prev))
    return runs

def align_to_grid_phase(x: float, phase: float, step: float) -> float:
    return float(x) if step <= 1e-6 else float(phase + round((x - phase) / step) * step)

def estimate_pulse_end_x(roi_gray: np.ndarray) -> Optional[int]:
    h, w = roi_gray.shape
    if h < 8 or w < 20: return None
    mask = adaptive_signal_mask(roi_gray)
    left_w = max(20, int(0.22 * w))
    col_energy = mask[:, :left_w].sum(axis=0).astype(np.float32)
    if np.max(col_energy) <= 0: return None
    cols = np.flatnonzero(col_energy >= max(float(np.percentile(col_energy, 92)), 0.20 * h))
    runs = contiguous_runs(cols)
    if not runs: return None
    best = max(runs, key=lambda ab: (ab[1] - ab[0] + 1) + 0.15 * float(np.max(col_energy[ab[0] : ab[1] + 1])))
    return int(best[1]) if best[1] < int(0.85 * left_w) else None

def estimate_pulse_height_px(roi_gray: np.ndarray, minor_dy_px: float) -> Optional[float]:
    h, w = roi_gray.shape
    if h < 10 or w < 20: return None
    sub = adaptive_signal_mask(roi_gray)[:, :max(20, int(0.22 * w))].astype(np.uint8)
    expected = max(8.0, float(minor_dy_px) * 10.0)
    num, _, stats, _ = cv2.connectedComponentsWithStats(sub, 8)
    best_h, best_err = None, 1e18
    for i in range(1, num):
        x, y, ww, hh, area = stats[i]
        if area < max(20, int(0.004 * sub.size)) or x > int(0.75 * sub.shape[1]): continue
        if int(0.45 * expected) <= hh <= int(2.2 * expected):
            err = abs(float(hh) - expected)
            if err < best_err: best_err, best_h = err, float(hh)
    return best_h


def estimate_px_per_millivolt(base_px_per_millivolt: float, pulse_h: Optional[float]) -> float:
    base = float(max(1e-6, base_px_per_millivolt))
    est = base
    if pulse_h is not None and np.isfinite(float(pulse_h)):
        ph = float(pulse_h)
        ratio = ph / base
        if PULSE_RATIO_MIN <= ratio <= PULSE_RATIO_MAX:
            est = (1.0 - PULSE_BLEND) * base + PULSE_BLEND * ph
    est *= AMP_SCALE_CALIB
    return float(np.clip(est, 0.70 * base, 1.40 * base))

def build_time_windows(gray: np.ndarray, row_centers: np.ndarray, half_band: int, minor_dx_px: float, grid_x0: float) -> Tuple[np.ndarray, Tuple[int, int]]:
    h, w = gray.shape
    seg_w, rhythm_w = max(80.0, 62.5 * minor_dx_px), max(300.0, 250.0 * minor_dx_px)
    starts = [float(p + int(0.004 * w)) for yc in row_centers[:3] if (p := estimate_pulse_end_x(gray[max(0, int(yc) - half_band):min(h, int(yc) + half_band), :])) is not None]
    x_start = align_to_grid_phase(float(np.median(starts)) if starts else float(0.04 * w), grid_x0, minor_dx_px)
    x_start = float(np.clip(x_start, 0.0, max(0.0, w - 4.0 * seg_w - 1.0)))

    split = np.maximum.accumulate(np.clip(np.round([x_start + i * seg_w for i in range(5)]), 0, w - 1).astype(np.int32))
    for i in range(1, len(split)):
        if split[i] <= split[i - 1]: split[i] = min(w - 1, split[i - 1] + 1)

    rhythm_x0 = float(np.clip(align_to_grid_phase(x_start, grid_x0, minor_dx_px), 0.0, max(0.0, w - rhythm_w - 1.0)))
    return split, (int(round(rhythm_x0)), int(round(np.clip(rhythm_x0 + rhythm_w, rhythm_x0 + 10.0, w))))


def build_split_from_start(w: int, x_start: float, seg_w: float) -> np.ndarray:
    split = np.maximum.accumulate(
        np.clip(np.round([x_start + i * seg_w for i in range(5)]), 0, w - 1).astype(np.int32)
    )
    for i in range(1, len(split)):
        if split[i] <= split[i - 1]:
            split[i] = min(w - 1, split[i - 1] + 1)
    return split


def refine_global_split_from_rows(
    gray: np.ndarray,
    row_centers: np.ndarray,
    half_band: int,
    default_split: np.ndarray,
) -> np.ndarray:
    h, w = gray.shape
    if default_split.size != 5:
        return default_split.astype(np.int32).copy()

    row_profiles: List[np.ndarray] = []
    for yc in row_centers[:3]:
        y0 = max(0, int(yc) - half_band)
        y1 = min(h, int(yc) + half_band)
        roi = gray[y0:y1, :]
        if roi.size == 0:
            continue
        mask = adaptive_signal_mask(roi)
        prof = smooth_1d(mask.sum(axis=0).astype(np.float32), max(5, w // 140))
        vmax = float(np.max(prof))
        if vmax > 0.0:
            row_profiles.append(prof / vmax)

    if len(row_profiles) < 2:
        return default_split.astype(np.int32).copy()

    energy = np.median(np.stack(row_profiles, axis=0), axis=0).astype(np.float32)
    refined = default_split.astype(np.int32).copy()
    target = float(np.median(np.diff(default_split)))
    search = max(12, int(0.14 * target))

    for i in range(1, len(refined) - 1):
        seam = int(refined[i])
        left = int(refined[i - 1])
        right = int(refined[i + 1])
        lo = max(left + 8, seam - search)
        hi = min(right - 8, seam + search)
        if hi > lo:
            local = energy[lo:hi]
            cand = int(lo + np.argmin(local))
            if float(np.min(local)) <= 0.92 * float(np.median(local)):
                refined[i] = cand

    for i in range(1, len(refined)):
        if refined[i] <= refined[i - 1]:
            refined[i] = min(w - 1, refined[i - 1] + 1)

    widths = np.diff(refined).astype(np.float32)
    if np.any(widths < 0.70 * target) or np.any(widths > 1.35 * target):
        return default_split.astype(np.int32).copy()
    return refined


def build_fixed_layout_windows(w: int) -> Tuple[np.ndarray, Tuple[int, int]]:
    split = np.maximum.accumulate(
        np.clip(np.round([r * w for r in FIXED_X_SPLIT_RATIO]), 0, w - 1).astype(np.int32)
    )
    for i in range(1, len(split)):
        if split[i] <= split[i - 1]:
            split[i] = min(w - 1, split[i - 1] + 1)
    rx0 = int(np.clip(round(FIXED_RHYTHM_X_BOUNDS_RATIO[0] * w), 0, w - 2))
    rx1 = int(np.clip(round(FIXED_RHYTHM_X_BOUNDS_RATIO[1] * w), rx0 + 1, w))
    return split, (rx0, rx1)


def apply_systematic_split_shift(
    split: np.ndarray,
    rhythm_bounds: Tuple[int, int],
    w: int,
) -> Tuple[np.ndarray, Tuple[int, int]]:
    out = split.astype(np.int32).copy()
    if out.size == len(SEAM_LEFT_SHIFT_PX):
        for i, dx in enumerate(SEAM_LEFT_SHIFT_PX):
            out[i] = int(np.clip(out[i] - int(dx), 0, w - 1))

    for i in range(1, len(out)):
        if out[i] <= out[i - 1]:
            out[i] = min(w - 1, out[i - 1] + 1)

    rx0, rx1 = rhythm_bounds
    rdx = int(RHYTHM_LEFT_SHIFT_PX)
    rx0 = int(np.clip(rx0 - rdx, 0, w - 2))
    rx1 = int(np.clip(rx1 - rdx, rx0 + 1, w))
    return out, (rx0, rx1)


def split_valley_confidence(gray: np.ndarray, row_centers: np.ndarray, half_band: int, split: np.ndarray) -> float:
    h, w = gray.shape
    if split.size < 5:
        return 0.0
    row_profiles: List[np.ndarray] = []
    for yc in row_centers[:3]:
        y0 = max(0, int(yc) - half_band)
        y1 = min(h, int(yc) + half_band)
        roi = gray[y0:y1, :]
        if roi.size == 0:
            continue
        mask = adaptive_signal_mask(roi)
        prof = smooth_1d(mask.sum(axis=0).astype(np.float32), max(5, w // 140))
        vmax = float(np.max(prof))
        if vmax > 0.0:
            row_profiles.append(prof / vmax)

    if len(row_profiles) < 2:
        return 0.0

    energy = np.median(np.stack(row_profiles, axis=0), axis=0).astype(np.float32)
    vals: List[float] = []
    for x in split[1:-1]:
        xi = int(np.clip(x, 0, w - 1))
        lo = max(0, xi - max(8, w // 80))
        hi = min(w, xi + max(8, w // 80))
        local = energy[lo:hi]
        if local.size < 5:
            continue
        med = float(np.median(local))
        v = float(energy[xi])
        if med > 1e-6:
            vals.append(1.0 - np.clip(v / med, 0.0, 1.2))

    return float(np.mean(vals)) if vals else 0.0


def compute_row_energy_profile(gray: np.ndarray, row_centers: np.ndarray, half_band: int) -> Optional[np.ndarray]:
    h, w = gray.shape
    row_profiles: List[np.ndarray] = []
    for yc in row_centers[:3]:
        y0 = max(0, int(yc) - half_band)
        y1 = min(h, int(yc) + half_band)
        roi = gray[y0:y1, :]
        if roi.size == 0:
            continue
        mask = adaptive_signal_mask(roi)
        prof = smooth_1d(mask.sum(axis=0).astype(np.float32), max(5, w // 140))
        vmax = float(np.max(prof))
        if vmax > 0.0:
            row_profiles.append(prof / vmax)

    if len(row_profiles) < 2:
        return None
    return np.median(np.stack(row_profiles, axis=0), axis=0).astype(np.float32)


def tune_seams_left_biased(
    gray: np.ndarray,
    row_centers: np.ndarray,
    half_band: int,
    split: np.ndarray,
) -> np.ndarray:
    h, w = gray.shape
    if split.size != 5:
        return split.astype(np.int32).copy()

    energy = compute_row_energy_profile(gray, row_centers, half_band)
    if energy is None:
        return split.astype(np.int32).copy()

    out = split.astype(np.int32).copy()
    target = float(np.median(np.diff(out)))
    left_span = max(14, int(0.22 * target))
    right_span = max(8, int(0.08 * target))

    for i in range(1, len(out) - 1):
        seam = int(out[i])
        left = int(out[i - 1])
        right = int(out[i + 1])
        lo = max(left + 8, seam - left_span)
        hi = min(right - 8, seam + right_span)
        if hi <= lo:
            continue

        local = energy[lo:hi]
        if local.size < 7:
            continue
        cand = int(lo + np.argmin(local))

        # Akceptuj tylko sensowne doliny, aby nie przeskoczyć na środek morfologii QRS.
        if float(np.min(local)) <= 0.96 * float(np.median(local)):
            out[i] = cand

    for i in range(1, len(out)):
        if out[i] <= out[i - 1]:
            out[i] = min(w - 1, out[i - 1] + 1)

    widths = np.diff(out).astype(np.float32)
    if np.any(widths < 0.68 * target) or np.any(widths > 1.40 * target):
        return split.astype(np.int32).copy()
    return out


def build_fixed_row_layout(h: int) -> Tuple[np.ndarray, int]:
    row_centers = np.array([int(round(r * h)) for r in FIXED_ROW_CENTERS_RATIO], dtype=np.int32)
    row_centers = np.clip(row_centers, 0, max(0, h - 1))
    half_band = int(np.clip(round(FIXED_HALF_BAND_RATIO * h), 10, max(10, int(0.10 * h))))
    return row_centers, half_band


def refine_row_split_with_valleys(
    roi_gray: np.ndarray,
    split: np.ndarray,
    default_split: Optional[np.ndarray] = None,
) -> np.ndarray:
    h, w = roi_gray.shape
    if h < 8 or w < 40:
        return split

    mask = adaptive_signal_mask(roi_gray)
    valid_cols = int(np.count_nonzero(mask.any(axis=0)))
    if valid_cols < max(25, int(0.10 * w)):
        return split if default_split is None else default_split.astype(np.int32).copy()

    col_energy = smooth_1d(mask.sum(axis=0).astype(np.float32), max(5, w // 140))
    if float(np.max(col_energy)) <= 0.0:
        return split

    refined = split.astype(np.int32).copy()
    for i in range(1, len(refined) - 1):
        left = int(refined[i - 1])
        seam = int(refined[i])
        right = int(refined[i + 1])
        search = max(10, int(0.12 * max(1, right - left)))
        lo = max(left + 8, seam - search)
        hi = min(right - 8, seam + search)
        if hi > lo:
            refined[i] = int(lo + np.argmin(col_energy[lo:hi]))

    if default_split is not None:
        seg_w = max(20.0, float(np.median(np.diff(default_split))))
        max_shift = max(10, int(0.28 * seg_w))
        for i in range(1, len(refined) - 1):
            d = int(default_split[i])
            refined[i] = int(np.clip(refined[i], d - max_shift, d + max_shift))

    for i in range(1, len(refined)):
        if refined[i] <= refined[i - 1]:
            refined[i] = min(w - 1, refined[i - 1] + 1)
    return refined


def build_row_split(
    gray: np.ndarray,
    y0: int,
    y1: int,
    minor_dx_px: float,
    grid_x0: float,
    default_split: np.ndarray,
    default_x_start: float,
) -> np.ndarray:
    h, w = gray.shape
    seg_w = float(max(80.0, 62.5 * minor_dx_px))
    roi = gray[max(0, y0):min(h, y1), :]
    pulse_end = estimate_pulse_end_x(roi) if roi.size > 0 else None

    split = default_split.astype(np.int32).copy()
    if pulse_end is not None:
        pulse_x = float(pulse_end + int(0.012 * w))
        if abs(pulse_x - default_x_start) <= 0.10 * w:
            x_start = align_to_grid_phase(pulse_x, grid_x0, minor_dx_px)
            shift = int(np.round(x_start - float(default_x_start)))
            shift = int(np.clip(shift, -max(8, int(0.08 * seg_w)), max(8, int(0.08 * seg_w))))
            split = np.clip(split + shift, 0, w - 1).astype(np.int32)
            for i in range(1, len(split)):
                if split[i] <= split[i - 1]:
                    split[i] = min(w - 1, split[i - 1] + 1)

    refined = refine_row_split_with_valleys(roi, split, default_split=default_split)

    widths = np.diff(refined).astype(np.float32)
    target = float(np.median(np.diff(default_split)))
    if np.any(widths < 0.72 * target) or np.any(widths > 1.32 * target):
        return default_split.astype(np.int32).copy()
    if float(np.mean(np.abs(refined.astype(np.float32) - default_split.astype(np.float32)))) > max(6.0, 0.08 * target):
        return default_split.astype(np.int32).copy()

    if roi.size > 0:
        mask = adaptive_signal_mask(roi)
        col_energy = smooth_1d(mask.sum(axis=0).astype(np.float32), max(5, roi.shape[1] // 140))
        if float(np.max(col_energy)) > 0.0:
            def seam_score(sp: np.ndarray) -> float:
                return float(np.mean([col_energy[int(np.clip(x, 0, col_energy.size - 1))] for x in sp[1:-1]]))

            global_score = seam_score(default_split.astype(np.int32))
            local_score = seam_score(refined.astype(np.int32))
            if local_score > 0.75 * global_score:
                return default_split.astype(np.int32).copy()
    return refined


# --- EKSTRAKCJA SYGNAŁU ---
def adaptive_signal_mask(roi_gray: np.ndarray, roi_rgb: Optional[np.ndarray] = None) -> np.ndarray:
    roi_u8 = np.clip(roi_gray, 0, 255).astype(np.uint8)
    enhanced = cv2.normalize(cv2.morphologyEx(roi_u8, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))), None, 0, 255, cv2.NORM_MINMAX)
    mask = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, -2) > 0
    
    if not (0.002 < float(mask.mean()) < 0.25):
        otsu_thr, _ = cv2.threshold(roi_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        best = next(((roi_u8 <= int(round(otsu_thr * f))).astype(np.uint8) * 255 for f in np.arange(1.0, 0.55, -0.05) if 0.002 < float(((roi_u8 <= int(round(otsu_thr * f))) > 0).mean()) < 0.20), None)
        mask = best > 0 if best is not None else (roi_u8 <= otsu_thr)

    if roi_rgb is not None:
        rgb_f = roi_rgb.astype(np.float32)
        spread = np.max(rgb_f, axis=2) - np.min(rgb_f, axis=2)
        if float(np.mean(spread)) > 4.5:
            mask_before = mask.copy()
            black_like = (roi_u8 <= float(np.percentile(roi_u8, 45))) & (spread <= float(np.percentile(spread, 70)) + 5.0)
            if float(black_like.mean()) > 0.01:
                mask = mask & black_like
                if float(mask.mean()) < 0.0015: mask = mask_before

    mask[:, mask.mean(axis=0) > 0.55] = False
    dense_rows = mask.mean(axis=1) > 0.80
    if np.count_nonzero(dense_rows) >= 2: mask[dense_rows, :] = False
    
    mask_u8 = (mask.astype(np.uint8) * 255)
    return cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8)) > 0

def contiguous_centers(binary_col: np.ndarray) -> np.ndarray:
    ys = np.flatnonzero(binary_col)
    if ys.size == 0: return np.empty(0, dtype=np.float32)
    groups, start, prev = [], ys[0], ys[0]
    for y in ys[1:]:
        if y == prev + 1: prev = y
        else:
            groups.append((start, prev))
            start = prev = y
    groups.append((start, prev))
    return np.array([(a + b) / 2.0 for a, b in groups], dtype=np.float32)


def extract_trace(
    roi_gray: np.ndarray, 
    target_len: int, 
    px_per_millivolt: float, 
    roi_rgb: Optional[np.ndarray] = None,
    debug_canvas: Optional[np.ndarray] = None, # NOWE
    roi_offset: Tuple[int, int] = (0, 0)       # NOWE (globalne x0, y0 wycinka)
) -> np.ndarray:
    h, w = roi_gray.shape
    if h < 4 or w < 4: return np.zeros(target_len, dtype=np.float32)

    def fallback_trace_from_darkest() -> np.ndarray:
        gray_f_loc = roi_gray.astype(np.float32)
        k = int(max(2, min(7, h // 4)))
        # Najciemniejsze piksele w każdej kolumnie jako proxy toru sygnału.
        dark_idx = np.argpartition(gray_f_loc, k - 1, axis=0)[:k, :].astype(np.float32)
        y_vals_loc = np.median(dark_idx, axis=0).astype(np.float32)
        y_vals_loc = smooth_1d(y_vals_loc, max(3, w // 160))
        baseline_loc = median_filter(y_vals_loc, size=max(11, int(w / 2.5)))
        sig_loc = (baseline_loc - y_vals_loc) / max(1e-6, px_per_millivolt)
        x_old_loc = np.linspace(0.0, 1.0, num=w, dtype=np.float32)
        x_new_loc = np.linspace(0.0, 1.0, num=target_len, dtype=np.float32)
        f_loc = interp1d(x_old_loc, sig_loc, kind='linear', bounds_error=False, fill_value="extrapolate")
        return np.clip(f_loc(x_new_loc).astype(np.float32), -5.0, 5.0)

    mask = adaptive_signal_mask(roi_gray, roi_rgb=roi_rgb)
    mask_u8 = cv2.morphologyEx((mask.astype(np.uint8) * 255), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    mask = mask_u8 > 0

    col_has_signal = mask.any(axis=0)
    valid_cols = np.flatnonzero(col_has_signal)

    if valid_cols.size < max(8, int(0.02 * w)):
        return fallback_trace_from_darkest()

    gray_f = roi_gray.astype(np.float32)
    sub_mask = mask
    sub_w = w
    center = 0.5 * (h - 1)
    jump_scale = max(5.0, 0.10 * h)
    
    max_states = 20
    lam_smooth = 0.25
    cand_cols, scores, backptr = [], [], []

    for x in range(sub_w):
        # cands = contiguous_centers(sub_mask[:, x])
        # if cands.size == 0:
        #     k = min(6, h)
        #     dark_idx = np.argpartition(gray_f[:, x], k - 1)[:k]
        #     cands = np.sort(dark_idx.astype(np.float32))

        cands = contiguous_centers(sub_mask[:, x])
        if cands.size == 0:
            if x > 0:
                best_prev_idx = int(np.argmax(scores[x - 1]))
                cands = np.array([cand_cols[x - 1][best_prev_idx]], dtype=np.float32)
            else:
                cands = np.array([center], dtype=np.float32)

        idxs = np.clip(np.round(cands).astype(int), 0, h - 1)
        darkness = (255.0 - gray_f[idxs, x]) / 255.0
        center_cost = np.abs(cands - center) / max(1.0, center)
        local_score = 0.85 * darkness - 0.15 * center_cost

        if cands.size > max_states:
            keep = np.argsort(local_score)[-max_states:]
            cands = cands[keep]
            idxs = np.clip(np.round(cands).astype(int), 0, h - 1)
            darkness = (255.0 - gray_f[idxs, x]) / 255.0
            center_cost = np.abs(cands - center) / max(1.0, center)
            local_score = 0.85 * darkness - 0.15 * center_cost

        order = np.argsort(cands)
        cands = cands[order]
        emit = local_score[order].astype(np.float32)
        cand_cols.append(cands.astype(np.float32))

        if x == 0:
            scores.append((emit - 0.10 * (np.abs(cands - center) / jump_scale)).astype(np.float32))
            backptr.append(np.full(cands.size, -1, dtype=np.int32))
            continue

        prev_cands = cand_cols[x - 1]
        trans = scores[x - 1][None, :] - lam_smooth * (np.abs(cands[:, None] - prev_cands[None, :]) / jump_scale)
        best_j = np.argmax(trans, axis=1).astype(np.int32)
        scores.append((emit + trans[np.arange(cands.size), best_j]).astype(np.float32))
        backptr.append(best_j)

    y_vals = np.empty(sub_w, dtype=np.float32)
    last_idx = int(np.argmax(scores[-1]))
    y_vals[-1] = float(cand_cols[-1][last_idx])
    for x in range(sub_w - 1, 0, -1):
        last_idx = max(0, int(backptr[x][last_idx]))
        y_vals[x - 1] = float(cand_cols[x - 1][last_idx])

    y_vals = smooth_1d(y_vals, max(3, sub_w // 160))
    
    # --- NOWY KOD: RYSOWANIE ŚCIEŻKI NA ZDJĘCIU ---
    if debug_canvas is not None:
        global_x0, global_y0 = roi_offset
        # Tworzymy listę punktów (x, y) w globalnych współrzędnych obrazu
        pts = []
        for i, local_y in enumerate(y_vals):
            global_x = int(global_x0 + i)
            global_y = int(global_y0 + local_y)
            pts.append([global_x, global_y])
            
        pts = np.array(pts, np.int32).reshape((-1, 1, 2))
        # Rysujemy jaskrawą czerwoną linię o grubości 2 pikseli
        cv2.polylines(debug_canvas, [pts], isClosed=False, color=(0, 0, 255), thickness=2, lineType=cv2.LINE_AA)
    # ----------------------------------------------

    baseline_curve = median_filter(y_vals, size=max(11, int(sub_w / 2.5)))
    sig = (baseline_curve - y_vals) / max(1e-6, px_per_millivolt)

    x_old = np.linspace(0.0, 1.0, num=sub_w, dtype=np.float32)
    x_new = np.linspace(0.0, 1.0, num=target_len, dtype=np.float32)
    f_cubic = interp1d(x_old, sig, kind='cubic', bounds_error=False, fill_value="extrapolate")
    out = f_cubic(x_new).astype(np.float32)
    return np.clip(out, -5.0, 5.0)

# --- GLÓWNE SKŁADANIE ---
def digitize_image(
    path: Path,
    debug_dir: Optional[Path] = None,
    lead_remap: Optional[Dict[str, str]] = None,
) -> Dict[str, np.ndarray]:
    rgb0 = load_rgb(path)
    stage1 = normalize_and_calibrate(rgb0)
    
    # --- NOWE: Płótno do rysowania ---
    debug_canvas = stage1.rgb.copy()
    
    gray, h, w = stage1.gray, stage1.rgb.shape[0], stage1.rgb.shape[1]
    
    # gray, h, w = stage1.gray, stage1.rgb.shape[0], stage1.rgb.shape[1]
    minor_px_y = stage1.minor_dy_px
    base_px_per_millivolt = minor_px_y * 10.0

    if FORCE_FIXED_ROWS:
        row_centers, half_band = build_fixed_row_layout(h)
    else:
        row_centers = detect_row_centers(stage1.rgb)
        half_band = int(max(10, int(max(12, np.median(np.diff(row_centers)))) * 0.32))

    if FORCE_FIXED_WINDOWS:
        x_split_global, rhythm_bounds = build_fixed_layout_windows(w)
    else:
        x_split_global, rhythm_bounds = build_time_windows(gray, row_centers, half_band, stage1.minor_dx_px, stage1.grid_x0)
        x_split_global = refine_global_split_from_rows(gray, row_centers, half_band, x_split_global)
        split_conf = split_valley_confidence(gray, row_centers, half_band, x_split_global)
        if split_conf < 0.08:
            x_split_global, rhythm_bounds = build_fixed_layout_windows(w)

    if ENABLE_LEFT_BIASED_TUNING:
        x_split_global = tune_seams_left_biased(gray, row_centers, half_band, x_split_global)
    x_split_global, rhythm_bounds = apply_systematic_split_shift(x_split_global, rhythm_bounds, w)

    row_splits: List[np.ndarray] = []
    for row_idx in range(3):
        yc = int(row_centers[row_idx])
        y0, y1 = max(0, yc - half_band), min(h, yc + half_band)
        if USE_ROW_LOCAL_REFINEMENT:
            row_split = build_row_split(
                gray,
                y0,
                y1,
                stage1.minor_dx_px,
                stage1.grid_x0,
                default_split=x_split_global,
                default_x_start=float(x_split_global[0]),
            )
        else:
            row_split = x_split_global.astype(np.int32).copy()
        row_splits.append(row_split)

        if debug_dir is not None:
            for x in row_split[1:-1]:
                cv2.line(
                    debug_canvas,
                    (int(x), int(y0)),
                    (int(x), int(y1)),
                    color=(0, 255, 0),
                    thickness=1,
                    lineType=cv2.LINE_AA,
                )
    signals: Dict[str, np.ndarray] = {}

    for row_idx, leads in enumerate(ROW_LAYOUT):
        yc = int(row_centers[row_idx])
        y0, y1 = max(0, yc - half_band), min(h, yc + half_band)
        pulse_h = estimate_pulse_height_px(gray[y0:y1, :], minor_px_y)
        row_px_per_mV = estimate_px_per_millivolt(base_px_per_millivolt, pulse_h)
        row_split = row_splits[row_idx]

        for col_idx, lead_name in enumerate(leads):
            c0, c1 = int(row_split[col_idx]), int(row_split[col_idx + 1])
            x0, x1 = max(0, c0), min(w, c1)
            
            signals[lead_name] = extract_trace(
                gray[y0:y1, x0:x1], 
                1250, 
                row_px_per_mV, 
                roi_rgb=stage1.rgb[y0:y1, x0:x1],
                debug_canvas=debug_canvas, # Podajemy płótno
                roi_offset=(x0, y0)        # Podajemy przesunięcie ROI
            )

    yc = int(row_centers[3])
    y0, y1 = max(0, yc - half_band), min(h, yc + half_band)
    x0, x1 = rhythm_bounds
    pulse_h = estimate_pulse_height_px(gray[y0:y1, :], minor_px_y)
    rhythm_px_per_mV = estimate_px_per_millivolt(base_px_per_millivolt, pulse_h)
    signals["II"] = extract_trace(gray[y0:y1, x0:x1], 5000, rhythm_px_per_mV, stage1.rgb[y0:y1, x0:x1])

    if debug_dir is not None:
        save_path = debug_dir / f"{path.stem}_overlay.png"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        # Ponieważ w save_image zakładasz format RGB, a OpenCV używa BGR, konwertujemy (chyba że używasz PIL w save_image to zostaw)
        Image.fromarray(debug_canvas).save(save_path)

    signals = apply_lead_remap(signals, lead_remap)
    for lead in LEADS_ORDER:
        if lead not in signals: signals[lead] = np.zeros(1250 if lead != "II" else 5000, dtype=np.float32)
    return signals

def build_submission(
    input_dir: Path,
    output_npz: Path,
    limit: int | None = None,
    debug_dir: Optional[Path] = None,
    lead_remap: Optional[Dict[str, str]] = None,
) -> Path:
    images = sorted(input_dir.glob("*.png")) + sorted(input_dir.glob("*.jpg"))
    if limit is not None: images = images[:limit]
    submission = {}
    for i, img_path in enumerate(images, start=1):
        sigs = digitize_image(img_path, debug_dir, lead_remap=lead_remap)
        for lead in LEADS_ORDER: submission[f"{img_path.stem}_{lead}"] = sigs[lead].astype(np.float16)
        if i % 50 == 0 or i == len(images): print(f"[{i}/{len(images)}] processed {img_path.name}")
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_npz, **submission)
    return output_npz

def maybe_submit(npz_path: Path, endpoint: str = "task4") -> None:
    try:
        import requests
        from dotenv import load_dotenv
    except Exception as exc:
        raise RuntimeError("Submission requires requests + python-dotenv. Install dependencies first.") from exc
    root = Path(__file__).resolve().parents[1]
    load_dotenv(root / ".env")
    token, server_url = os.getenv("TEAM_TOKEN"), os.getenv("SERVER_URL")
    if not token or not server_url: raise RuntimeError("Missing TEAM_TOKEN or SERVER_URL in .env")
    with npz_path.open("rb") as f:
        response = requests.post(f"{server_url.rstrip('/')}/{endpoint}", files={"npz_file": f}, headers={"X-API-Token": token}, timeout=600)
    print("submission response:", response.status_code, response.json() if response.status_code == 200 else response.text)

def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Task 4 EKG digitization v2 (Hackathon Optimized).")
    parser.add_argument("--input-dir", type=Path, default=root / "test", help="Directory with ECG images.")
    parser.add_argument("--output-npz", type=Path, default=root / "data" / "out" / "task4_submission_v2.npz")
    parser.add_argument("--limit", type=int, default=None, help="Process only first N files.")
    parser.add_argument("--debug-dir", type=Path, default=None, help="Debug dir.")
    parser.add_argument("--lead-remap-json", type=Path, default=None, help="Optional JSON mapping {predicted_lead: target_lead}.")
    parser.add_argument("--submit", action="store_true", help="Submit to server after generation.")
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    lead_remap = load_lead_remap(args.lead_remap_json)
    npz_path = build_submission(
        args.input_dir,
        args.output_npz,
        limit=args.limit,
        debug_dir=args.debug_dir,
        lead_remap=lead_remap,
    )
    print(f"saved: {npz_path}")
    if args.submit: maybe_submit(npz_path)

if __name__ == "__main__":
    main()
