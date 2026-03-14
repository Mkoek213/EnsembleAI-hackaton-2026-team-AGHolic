#!/usr/bin/env python3
"""Simple baseline pipeline for Task 4 ECG digitization.

This is intentionally lightweight:
1) assume standard 3x4 + rhythm II layout,
2) detect 4 row centers from dark-pixel profile,
3) split columns evenly,
4) extract one waveform trace per lead with a continuity heuristic,
5) save submission.npz.

It is designed for quick end-to-end testing, not leaderboard-level accuracy.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from PIL import Image


LEADS_ORDER = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]
ROW_LAYOUT = [
    ["I", "AVR", "V1", "V4"],
    ["II", "AVL", "V2", "V5"],
    ["III", "AVF", "V3", "V6"],
]


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


def estimate_minor_grid_px(rgb: np.ndarray) -> float:
    """Estimates minor grid spacing in pixels via autocorrelation on red-grid profile."""
    r = rgb[:, :, 0].astype(np.float32)
    g = rgb[:, :, 1].astype(np.float32)
    b = rgb[:, :, 2].astype(np.float32)
    red_score = r - 0.5 * (g + b)

    candidates: List[float] = []
    for profile in (red_score.mean(axis=0), red_score.mean(axis=1)):
        p = profile - np.median(profile)
        p = np.abs(np.diff(p))
        if p.size < 64:
            continue
        ac = np.correlate(p, p, mode="full")[p.size - 1 :]
        low, high = 4, min(35, ac.size - 1)
        if high <= low:
            continue
        lag = int(np.argmax(ac[low : high + 1]) + low)
        candidates.append(float(lag))

    if not candidates:
        # Fallback heuristic based on common sizes in this dataset.
        return float(np.clip(rgb.shape[1] / 250.0, 4.0, 30.0))
    return float(np.clip(np.median(candidates), 4.0, 30.0))


def smooth_1d(x: np.ndarray, win: int) -> np.ndarray:
    win = max(3, int(win))
    if win % 2 == 0:
        win += 1
    kernel = np.ones(win, dtype=np.float32) / float(win)
    return np.convolve(x.astype(np.float32), kernel, mode="same")


def detect_row_centers(rgb: np.ndarray) -> np.ndarray:
    """Finds 4 row centers using darkest pixels (signal/text) profile."""
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


def extract_trace(roi_gray: np.ndarray, target_len: int, px_per_millivolt: float) -> np.ndarray:
    """Column-wise path extraction with continuity preference."""
    h, w = roi_gray.shape
    thr = np.percentile(roi_gray, 12.0)
    mask = roi_gray <= thr

    y_vals = np.empty(w, dtype=np.float32)
    has_prev = False
    prev = 0.0
    for x in range(w):
        ys = np.flatnonzero(mask[:, x])
        if ys.size == 0:
            y_vals[x] = prev if has_prev else (h / 2.0)
            continue
        if not has_prev:
            y = float(np.median(ys))
            has_prev = True
        else:
            i = int(np.argmin(np.abs(ys.astype(np.float32) - prev)))
            y = float(ys[i])
        y_vals[x] = y
        prev = y

    y_vals = smooth_1d(y_vals, max(5, w // 160))
    baseline = float(np.median(y_vals))

    # Convert pixel displacement to mV: 10 mm = 1 mV.
    sig = (baseline - y_vals) / max(1e-6, px_per_millivolt)
    sig -= np.median(sig)

    x_old = np.linspace(0.0, 1.0, num=sig.size, dtype=np.float32)
    x_new = np.linspace(0.0, 1.0, num=target_len, dtype=np.float32)
    out = np.interp(x_new, x_old, sig).astype(np.float32)
    return np.clip(out, -5.0, 5.0)


def digitize_image(path: Path) -> Dict[str, np.ndarray]:
    rgb = crop_non_black(load_rgb(path))
    h, w, _ = rgb.shape

    minor_px = estimate_minor_grid_px(rgb)
    px_per_millivolt = minor_px * 10.0
    row_centers = detect_row_centers(rgb)
    row_gap = int(max(12, np.median(np.diff(row_centers))))
    half_band = int(max(10, row_gap * 0.32))

    x_split = np.linspace(0, w, num=5, dtype=np.int32)
    gray = rgb.mean(axis=2).astype(np.float32)

    signals: Dict[str, np.ndarray] = {}

    # 3 rows x 4 columns => 12 short leads (2.5 s = 1250 samples)
    for row_idx, leads in enumerate(ROW_LAYOUT):
        yc = int(row_centers[row_idx])
        y0 = max(0, yc - half_band)
        y1 = min(h, yc + half_band)
        for col_idx, lead_name in enumerate(leads):
            c0, c1 = int(x_split[col_idx]), int(x_split[col_idx + 1])
            # Reduce contamination at split boundaries and labels.
            seg_w = c1 - c0
            x0 = max(0, c0 + int(0.04 * seg_w))
            x1 = min(w, c1 - int(0.02 * seg_w))
            roi = gray[y0:y1, x0:x1]
            signals[lead_name] = extract_trace(roi, target_len=1250, px_per_millivolt=px_per_millivolt)

    # Rhythm strip (row 4) for lead II: 10 s = 5000 samples.
    yc = int(row_centers[3])
    y0 = max(0, yc - half_band)
    y1 = min(h, yc + half_band)
    x0 = int(0.04 * w)
    x1 = int(0.96 * w)
    roi = gray[y0:y1, x0:x1]
    signals["II"] = extract_trace(roi, target_len=5000, px_per_millivolt=px_per_millivolt)

    # Ensure all expected keys are present.
    for lead in LEADS_ORDER:
        if lead not in signals:
            signals[lead] = np.zeros(1250 if lead != "II" else 5000, dtype=np.float32)

    return signals


def build_submission(input_dir: Path, output_npz: Path, limit: int | None = None) -> Path:
    images = sorted(input_dir.glob("*.png"))
    if limit is not None:
        images = images[:limit]

    submission: Dict[str, np.ndarray] = {}
    for i, img_path in enumerate(images, start=1):
        rec = img_path.stem
        sigs = digitize_image(img_path)
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
    parser = argparse.ArgumentParser(description="Simple Task 4 baseline pipeline.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=root / "test",
        help="Directory with ECG PNG files.",
    )
    parser.add_argument(
        "--output-npz",
        type=Path,
        default=root / "data" / "out" / "task4_simple_submission.npz",
        help="Where to save submission .npz.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Process only first N files.")
    parser.add_argument("--submit", action="store_true", help="Submit to server after generation.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    npz_path = build_submission(args.input_dir, args.output_npz, limit=args.limit)
    print(f"saved: {npz_path}")
    if args.submit:
        maybe_submit(npz_path)


if __name__ == "__main__":
    main()

