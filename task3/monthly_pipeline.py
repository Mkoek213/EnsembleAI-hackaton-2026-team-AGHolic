from __future__ import annotations

from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd

from daily_pipeline import find_data_dir


SENSOR_COLS = [f"t{i}" for i in range(1, 14)]
BASE_NUM_COLS = SENSOR_COLS + ["x1"]
MONTHLY_STATS_COLS = BASE_NUM_COLS + [
    "delta_t2_t1",
    "delta_t5_t3",
    "delta_t6_t4",
    "delta_t8_t1",
    "lift_load_outdoor",
    "lift_circuit_outdoor",
]
PROFILE_COLS = ["x1", "t1", "t5", "t8"]
DAYPARTS = {
    "night": set(range(0, 6)),
    "morning": set(range(6, 12)),
    "afternoon": set(range(12, 18)),
    "evening": set(range(18, 24)),
}


def _fmt_duration(seconds: float) -> str:
    seconds = int(max(seconds, 0))
    hours, rem = divmod(seconds, 3600)
    minutes, sec = divmod(rem, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{sec:02d}"
    return f"{minutes:02d}:{sec:02d}"


def aggregate_monthly_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    ts = pd.to_datetime(chunk["timedate"], utc=True, errors="coerce")
    chunk = chunk.loc[ts.notna()].copy()
    ts = ts.loc[ts.notna()].dt.tz_convert(None)

    chunk["year"] = ts.dt.year.astype("int16")
    chunk["month"] = ts.dt.month.astype("int8")
    hour = ts.dt.hour.astype("int8")
    weekday = ts.dt.weekday.astype("int8")

    chunk["active"] = (chunk["x1"] > 0).astype("float32")
    chunk["is_weekend"] = (weekday >= 5).astype("int8")
    chunk["cold_flag"] = (chunk["t1"] < 0.35).astype("int8")
    chunk["warm_flag"] = (chunk["t1"] >= 0.55).astype("int8")
    chunk["x2_filled"] = chunk["x2"].fillna(0.0).astype("float32")
    chunk["x2_count"] = chunk["x2"].notna().astype("int16")

    chunk["delta_t2_t1"] = chunk["t2"] - chunk["t1"]
    chunk["delta_t5_t3"] = chunk["t5"] - chunk["t3"]
    chunk["delta_t6_t4"] = chunk["t6"] - chunk["t4"]
    chunk["delta_t8_t1"] = chunk["t8"] - chunk["t1"]
    chunk["lift_load_outdoor"] = chunk["t5"] - chunk["t1"]
    chunk["lift_circuit_outdoor"] = chunk["t8"] - chunk["t1"]

    for col in MONTHLY_STATS_COLS:
        chunk[f"{col}_sq"] = chunk[col] * chunk[col]

    for name, hours in DAYPARTS.items():
        mask = hour.isin(list(hours)).astype("float32")
        chunk[f"{name}_count"] = mask
        for col in PROFILE_COLS:
            chunk[f"{col}_{name}_sum"] = chunk[col] * mask

    for prefix, flag_col in [("weekend", "is_weekend"), ("cold", "cold_flag"), ("warm", "warm_flag")]:
        mask = chunk[flag_col].astype("float32")
        chunk[f"{prefix}_count"] = mask
        for col in PROFILE_COLS:
            chunk[f"{col}_{prefix}_sum"] = chunk[col] * mask

    agg_spec: dict[str, tuple[str, str]] = {
        "period": ("period", "first"),
        "deviceType": ("deviceType", "max"),
        "x3": ("x3", "max"),
        "n_rows": ("timedate", "size"),
        "active_sum": ("active", "sum"),
        "x2_sum": ("x2_filled", "sum"),
        "x2_count": ("x2_count", "sum"),
        "cold_rows": ("cold_flag", "sum"),
        "warm_rows": ("warm_flag", "sum"),
        "weekend_rows": ("is_weekend", "sum"),
    }

    for col in MONTHLY_STATS_COLS:
        agg_spec[f"{col}_sum"] = (col, "sum")
        agg_spec[f"{col}_sq_sum"] = (f"{col}_sq", "sum")
        agg_spec[f"{col}_min"] = (col, "min")
        agg_spec[f"{col}_max"] = (col, "max")

    for name in DAYPARTS:
        agg_spec[f"{name}_count"] = (f"{name}_count", "sum")
        for col in PROFILE_COLS:
            agg_spec[f"{col}_{name}_sum"] = (f"{col}_{name}_sum", "sum")

    for prefix in ["weekend", "cold", "warm"]:
        agg_spec[f"{prefix}_count"] = (f"{prefix}_count", "sum")
        for col in PROFILE_COLS:
            agg_spec[f"{col}_{prefix}_sum"] = (f"{col}_{prefix}_sum", "sum")

    return chunk.groupby(["deviceId", "year", "month"], as_index=False).agg(**agg_spec)


def collapse_monthly_partials(frames: list[pd.DataFrame]) -> pd.DataFrame:
    stacked = pd.concat(frames, ignore_index=True)

    agg_spec: dict[str, tuple[str, str]] = {
        "period": ("period", "first"),
        "deviceType": ("deviceType", "max"),
        "x3": ("x3", "max"),
        "n_rows": ("n_rows", "sum"),
        "active_sum": ("active_sum", "sum"),
        "x2_sum": ("x2_sum", "sum"),
        "x2_count": ("x2_count", "sum"),
        "cold_rows": ("cold_rows", "sum"),
        "warm_rows": ("warm_rows", "sum"),
        "weekend_rows": ("weekend_rows", "sum"),
    }

    for col in MONTHLY_STATS_COLS:
        agg_spec[f"{col}_sum"] = (f"{col}_sum", "sum")
        agg_spec[f"{col}_sq_sum"] = (f"{col}_sq_sum", "sum")
        agg_spec[f"{col}_min"] = (f"{col}_min", "min")
        agg_spec[f"{col}_max"] = (f"{col}_max", "max")

    for name in DAYPARTS:
        agg_spec[f"{name}_count"] = (f"{name}_count", "sum")
        for col in PROFILE_COLS:
            agg_spec[f"{col}_{name}_sum"] = (f"{col}_{name}_sum", "sum")

    for prefix in ["weekend", "cold", "warm"]:
        agg_spec[f"{prefix}_count"] = (f"{prefix}_count", "sum")
        for col in PROFILE_COLS:
            agg_spec[f"{col}_{prefix}_sum"] = (f"{col}_{prefix}_sum", "sum")

    return stacked.groupby(["deviceId", "year", "month"], as_index=False).agg(**agg_spec)


def finalize_monthly_features(collapsed: pd.DataFrame, data_dir: Path | None = None) -> pd.DataFrame:
    monthly = collapsed.copy()

    monthly["target_x2"] = monthly["x2_sum"] / monthly["x2_count"].replace({0: np.nan})
    monthly["active_ratio"] = monthly["active_sum"] / monthly["n_rows"]
    monthly["cold_share"] = monthly["cold_rows"] / monthly["n_rows"]
    monthly["warm_share"] = monthly["warm_rows"] / monthly["n_rows"]
    monthly["weekend_share"] = monthly["weekend_rows"] / monthly["n_rows"]
    monthly["ym"] = monthly["year"].astype("int32") * 100 + monthly["month"].astype("int32")

    angle = 2.0 * np.pi * monthly["month"] / 12.0
    monthly["month_sin"] = np.sin(angle)
    monthly["month_cos"] = np.cos(angle)

    for col in MONTHLY_STATS_COLS:
        mean_col = f"{col}_mean"
        monthly[mean_col] = monthly[f"{col}_sum"] / monthly["n_rows"]
        var = (monthly[f"{col}_sq_sum"] / monthly["n_rows"]) - (monthly[mean_col] ** 2)
        monthly[f"{col}_std"] = np.sqrt(np.clip(var, 0.0, None))

    for name in DAYPARTS:
        denom = monthly[f"{name}_count"].replace({0: np.nan})
        monthly[f"{name}_share"] = monthly[f"{name}_count"] / monthly["n_rows"]
        for col in PROFILE_COLS:
            monthly[f"{col}_{name}_mean"] = monthly[f"{col}_{name}_sum"] / denom

    for prefix in ["weekend", "cold", "warm"]:
        denom = monthly[f"{prefix}_count"].replace({0: np.nan})
        for col in PROFILE_COLS:
            monthly[f"{col}_{prefix}_mean"] = monthly[f"{col}_{prefix}_sum"] / denom

    if data_dir is not None:
        devices_path = data_dir / "devices.csv"
        if devices_path.exists():
            devices = pd.read_csv(devices_path, usecols=["deviceId", "latitude", "longitude"])
            monthly = monthly.merge(devices, on="deviceId", how="left")
            monthly["geo_r"] = np.sqrt((monthly["latitude"] ** 2) + (monthly["longitude"] ** 2))
            monthly["geo_sum"] = monthly["latitude"] + monthly["longitude"]
            monthly["geo_diff"] = monthly["latitude"] - monthly["longitude"]

    monthly = monthly.sort_values(["deviceId", "ym"]).reset_index(drop=True)
    group = monthly.groupby("deviceId", sort=False)
    lag_cols = [
        "t1_mean",
        "t5_mean",
        "t8_mean",
        "x1_mean",
        "active_ratio",
        "cold_share",
        "warm_share",
        "lift_load_outdoor_mean",
        "lift_circuit_outdoor_mean",
    ]
    new_cols: dict[str, pd.Series] = {}
    for col in lag_cols:
        if col not in monthly.columns:
            continue
        lag1 = group[col].shift(1)
        lag2 = group[col].shift(2)
        new_cols[f"{col}_lag1m"] = lag1
        new_cols[f"{col}_lag2m"] = lag2
        new_cols[f"{col}_mom1m"] = monthly[col] - lag1
    if new_cols:
        monthly = pd.concat([monthly, pd.DataFrame(new_cols, index=monthly.index)], axis=1)

    return monthly


def build_monthly_features(
    csv_path: Path,
    data_dir: Path | None = None,
    chunksize: int = 300_000,
    flush_every: int = 20,
    max_chunks: int | None = None,
) -> pd.DataFrame:
    usecols = ["deviceId", "timedate", "period", "x3", "deviceType"] + BASE_NUM_COLS + ["x2"]
    dtype: dict[str, str] = {
        "deviceId": "string",
        "period": "category",
        "x3": "Int16",
        "deviceType": "Int16",
    }
    for col in BASE_NUM_COLS + ["x2"]:
        dtype[col] = "float32"

    partial_frames: list[pd.DataFrame] = []
    reader = pd.read_csv(
        csv_path,
        usecols=usecols,
        dtype=dtype,
        chunksize=chunksize,
        low_memory=False,
    )

    start = time.perf_counter()
    for idx, chunk in enumerate(reader, start=1):
        partial_frames.append(aggregate_monthly_chunk(chunk))

        if idx % 5 == 0:
            print(f"Processed chunks: {idx}")

        if len(partial_frames) >= flush_every:
            partial_frames = [collapse_monthly_partials(partial_frames)]
            print(f"Collapsed intermediate partials at chunk {idx}")

        if max_chunks is not None and idx >= max_chunks:
            print(f"Stopping early at chunk {idx} because max_chunks={max_chunks}")
            break

    if not partial_frames:
        raise RuntimeError("No data was read from CSV.")

    collapsed = (
        collapse_monthly_partials(partial_frames)
        if len(partial_frames) > 1
        else partial_frames[0]
    )
    print(f"Monthly collapse done in {_fmt_duration(time.perf_counter() - start)}")
    return finalize_monthly_features(collapsed, data_dir=data_dir)


def default_data_dir() -> Path:
    return find_data_dir()
