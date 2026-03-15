from __future__ import annotations

from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error


SENSOR_COLS = [f"t{i}" for i in range(1, 14)]
BASE_NUM_COLS = SENSOR_COLS + ["x1"]

MODEL_STRENGTHS = {"fast", "strong", "heavy"}
SEQ_SOURCE_CANDIDATES = [
    "t8_mean",
    "t5_mean",
    "t13_mean",
    "t4_mean",
    "t6_mean",
    "t12_mean",
    "t11_mean",
    "t10_mean",
    "t1_mean",
    "active_ratio",
    "x1_mean",
    "delta_t5_t3",
    "delta_t2_t1",
]


def _is_catboost_backend(model_backend: str) -> bool:
    return model_backend.lower().startswith("catboost")


def _fmt_duration(seconds: float) -> str:
    seconds = int(max(seconds, 0))
    hours, rem = divmod(seconds, 3600)
    minutes, sec = divmod(rem, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{sec:02d}"
    return f"{minutes:02d}:{sec:02d}"


def _normalize_bucket_freq(bucket_freq: str) -> str:
    """Normalize user-provided frequency across pandas versions."""
    freq = str(bucket_freq).strip()
    if not freq:
        raise ValueError("bucket_freq cannot be empty.")

    # First try as provided (works for many aliases like 'D').
    try:
        to_offset(freq)
        return freq
    except ValueError:
        pass

    # Newer/stricter pandas versions often require lower-case intraday aliases.
    freq_lower = freq.lower()
    try:
        to_offset(freq_lower)
        return freq_lower
    except ValueError as exc:
        raise ValueError(
            f"Invalid bucket frequency '{bucket_freq}'. "
            "Examples: D, 12h, 6h, 3h, 1h, 30min."
        ) from exc


def find_data_dir(default: str = "task3") -> Path:
    for candidate in [Path("."), Path(default)]:
        if (candidate / "data.csv").exists():
            return candidate
    raise FileNotFoundError("data.csv not found in current directory or task3/")


def load_time_features_csv(
    csv_path: Path,
    labelled_only: bool = False,
    chunksize: int | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    sample = pd.read_csv(csv_path, nrows=5)
    dtype: dict[str, str] = {}
    for col in sample.columns:
        if col == "date":
            continue
        if col == "deviceId":
            dtype[col] = "string"
        elif col == "period":
            dtype[col] = "category"
        elif col in {"year", "dayofyear"}:
            dtype[col] = "int16"
        elif col in {"month", "weekday", "hour", "minute", "is_weekend"}:
            dtype[col] = "int8"
        elif col in {"deviceType", "x3"}:
            dtype[col] = "Int16"
        elif col == "n_rows":
            dtype[col] = "int32"
        else:
            dtype[col] = "float32"

    read_kwargs: dict[str, Any] = {
        "parse_dates": ["date"],
        "dtype": dtype,
        "low_memory": False,
    }

    if chunksize is None and not labelled_only:
        return pd.read_csv(csv_path, **read_kwargs)

    effective_chunksize = int(chunksize or 250_000)
    frames: list[pd.DataFrame] = []
    total_rows = 0
    kept_rows = 0

    for idx, chunk in enumerate(pd.read_csv(csv_path, chunksize=effective_chunksize, **read_kwargs), start=1):
        total_rows += len(chunk)
        if labelled_only:
            chunk = chunk[chunk["target_x2"].notna()].copy()
        kept_rows += len(chunk)
        if len(chunk) > 0:
            frames.append(chunk)
        if verbose and idx % 10 == 0:
            print(
                f"[LOAD] chunks={idx} rows_seen={total_rows} rows_kept={kept_rows} "
                f"labelled_only={labelled_only}"
            )

    if not frames:
        empty = sample.iloc[0:0].copy()
        return empty

    return pd.concat(frames, ignore_index=True)


def aggregate_chunk(
    chunk: pd.DataFrame,
    base_cols: list[str],
    bucket_freq: str = "D",
) -> pd.DataFrame:
    ts = pd.to_datetime(chunk["timedate"], utc=True, errors="coerce")
    chunk = chunk.loc[ts.notna()].copy()
    chunk["date"] = ts.loc[ts.notna()].dt.tz_convert(None).dt.floor(bucket_freq)

    chunk["active"] = (chunk["x1"] > 0).astype("float32")
    chunk["x2_filled"] = chunk["x2"].fillna(0.0).astype("float32")
    chunk["x2_count"] = chunk["x2"].notna().astype("int16")

    for col in base_cols:
        chunk[f"{col}_sq"] = chunk[col] * chunk[col]

    agg_spec: dict[str, tuple[str, str]] = {
        "period": ("period", "first"),
        "deviceType": ("deviceType", "max"),
        "x3": ("x3", "max"),
        "n_rows": ("timedate", "size"),
        "active_sum": ("active", "sum"),
        "x2_sum": ("x2_filled", "sum"),
        "x2_count": ("x2_count", "sum"),
    }
    for col in base_cols:
        agg_spec[f"{col}_sum"] = (col, "sum")
        agg_spec[f"{col}_sq_sum"] = (f"{col}_sq", "sum")
        agg_spec[f"{col}_min"] = (col, "min")
        agg_spec[f"{col}_max"] = (col, "max")

    return chunk.groupby(["deviceId", "date"], as_index=False).agg(**agg_spec)


def collapse_partial_rows(frames: list[pd.DataFrame], base_cols: list[str]) -> pd.DataFrame:
    stacked = pd.concat(frames, ignore_index=True)

    agg_spec: dict[str, tuple[str, str]] = {
        "period": ("period", "first"),
        "deviceType": ("deviceType", "max"),
        "x3": ("x3", "max"),
        "n_rows": ("n_rows", "sum"),
        "active_sum": ("active_sum", "sum"),
        "x2_sum": ("x2_sum", "sum"),
        "x2_count": ("x2_count", "sum"),
    }
    for col in base_cols:
        agg_spec[f"{col}_sum"] = (f"{col}_sum", "sum")
        agg_spec[f"{col}_sq_sum"] = (f"{col}_sq_sum", "sum")
        agg_spec[f"{col}_min"] = (f"{col}_min", "min")
        agg_spec[f"{col}_max"] = (f"{col}_max", "max")

    return stacked.groupby(["deviceId", "date"], as_index=False).agg(**agg_spec)


def finalize_time_features(
    collapsed: pd.DataFrame,
    base_cols: list[str],
) -> pd.DataFrame:
    features = collapsed.copy()
    features["active_ratio"] = features["active_sum"] / features["n_rows"]
    features["target_x2"] = features["x2_sum"] / features["x2_count"].replace({0: np.nan})

    for col in base_cols:
        mean_col = f"{col}_mean"
        features[mean_col] = features[f"{col}_sum"] / features["n_rows"]
        var = (features[f"{col}_sq_sum"] / features["n_rows"]) - (features[mean_col] ** 2)
        features[f"{col}_std"] = np.sqrt(np.clip(var, 0.0, None))

    features["year"] = features["date"].dt.year.astype("int16")
    features["month"] = features["date"].dt.month.astype("int8")
    features["dayofyear"] = features["date"].dt.dayofyear.astype("int16")
    features["weekday"] = features["date"].dt.weekday.astype("int8")
    features["hour"] = features["date"].dt.hour.astype("int8")
    features["minute"] = features["date"].dt.minute.astype("int8")
    features["is_weekend"] = (features["weekday"] >= 5).astype("int8")

    doy_angle = 2.0 * np.pi * features["dayofyear"] / 366.0
    features["doy_sin"] = np.sin(doy_angle)
    features["doy_cos"] = np.cos(doy_angle)

    time_of_day = (features["hour"].astype("float32") * 60.0 + features["minute"].astype("float32")) / 1440.0
    tod_angle = 2.0 * np.pi * time_of_day
    features["tod_sin"] = np.sin(tod_angle)
    features["tod_cos"] = np.cos(tod_angle)

    features["delta_t2_t1"] = features["t2_mean"] - features["t1_mean"]
    features["delta_t5_t3"] = features["t5_mean"] - features["t3_mean"]
    features["delta_t6_t4"] = features["t6_mean"] - features["t4_mean"]
    features["delta_t7_t2"] = features["t7_mean"] - features["t2_mean"]
    features["delta_t9_t1"] = features["t9_mean"] - features["t1_mean"]

    stat_cols: list[str] = []
    for col in base_cols:
        stat_cols.extend([f"{col}_mean", f"{col}_std", f"{col}_min", f"{col}_max"])

    keep_cols = [
        "deviceId",
        "date",
        "period",
        "year",
        "month",
        "dayofyear",
        "weekday",
        "hour",
        "minute",
        "is_weekend",
        "doy_sin",
        "doy_cos",
        "tod_sin",
        "tod_cos",
        "deviceType",
        "x3",
        "n_rows",
        "active_ratio",
    ] + stat_cols + [
        "delta_t2_t1",
        "delta_t5_t3",
        "delta_t6_t4",
        "delta_t7_t2",
        "delta_t9_t1",
        "target_x2",
    ]

    return features[keep_cols].sort_values(["deviceId", "date"]).reset_index(drop=True)


def build_daily_features(
    csv_path: Path,
    chunksize: int = 300_000,
    flush_every: int = 20,
    max_chunks: int | None = None,
) -> pd.DataFrame:
    return build_time_features(
        csv_path=csv_path,
        chunksize=chunksize,
        flush_every=flush_every,
        max_chunks=max_chunks,
        bucket_freq="D",
    )


def build_time_features(
    csv_path: Path,
    chunksize: int = 300_000,
    flush_every: int = 20,
    max_chunks: int | None = None,
    bucket_freq: str = "1h",
) -> pd.DataFrame:
    normalized_bucket_freq = _normalize_bucket_freq(bucket_freq)
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

    for idx, chunk in enumerate(reader, start=1):
        partial_frames.append(
            aggregate_chunk(chunk, BASE_NUM_COLS, bucket_freq=normalized_bucket_freq)
        )

        if idx % 5 == 0:
            print(f"Processed chunks: {idx}")

        if len(partial_frames) >= flush_every:
            partial_frames = [collapse_partial_rows(partial_frames, BASE_NUM_COLS)]
            print(f"Collapsed intermediate partials at chunk {idx}")

        if max_chunks is not None and idx >= max_chunks:
            print(f"Stopping early at chunk {idx} because max_chunks={max_chunks}")
            break

    if not partial_frames:
        raise RuntimeError("No data was read from CSV.")

    collapsed = (
        collapse_partial_rows(partial_frames, BASE_NUM_COLS)
        if len(partial_frames) > 1
        else partial_frames[0]
    )

    return finalize_time_features(collapsed, BASE_NUM_COLS)


def make_model(
    model_backend: str = "hgb",
    model_strength: str = "strong",
    catboost_iterations: int | None = None,
    catboost_log_every: int = 100,
    live_log: bool = False,
) -> Any:
    backend = model_backend.lower()
    strength = model_strength.lower()
    if strength not in MODEL_STRENGTHS:
        raise ValueError(
            f"Unknown model_strength: {model_strength}. "
            "Supported: fast, strong, heavy"
        )

    if backend == "hgb":
        params_by_strength: dict[str, dict[str, Any]] = {
            "fast": {
                "learning_rate": 0.05,
                "max_depth": 6,
                "max_iter": 600,
                "min_samples_leaf": 30,
            },
            "strong": {
                "learning_rate": 0.03,
                "max_depth": 8,
                "max_iter": 1800,
                "min_samples_leaf": 20,
            },
            "heavy": {
                "learning_rate": 0.02,
                "max_depth": 10,
                "max_iter": 4000,
                "min_samples_leaf": 15,
            },
        }
        p = params_by_strength[strength]
        return HistGradientBoostingRegressor(
            loss="absolute_error",
            learning_rate=p["learning_rate"],
            max_depth=p["max_depth"],
            max_iter=p["max_iter"],
            min_samples_leaf=p["min_samples_leaf"],
            verbose=1 if live_log else 0,
            random_state=42,
        )

    if backend in {"catboost_cpu", "catboost_gpu"}:
        try:
            from catboost import CatBoostRegressor
        except ImportError as exc:
            raise ImportError(
                "catboost is required for model_backend=catboost_*. "
                "Install it with `pip install catboost`."
            ) from exc

        params_by_strength: dict[str, dict[str, Any]] = {
            "fast": {
                "learning_rate": 0.05,
                "depth": 8,
                "iterations": 3000,
                "l2_leaf_reg": 5.0,
                "random_strength": 1.0,
            },
            "strong": {
                "learning_rate": 0.03,
                "depth": 10,
                "iterations": 12000,
                "l2_leaf_reg": 8.0,
                "random_strength": 1.5,
            },
            "heavy": {
                "learning_rate": 0.02,
                "depth": 10,
                "iterations": 30000,
                "l2_leaf_reg": 10.0,
                "random_strength": 2.0,
            },
        }
        p = params_by_strength[strength]

        params: dict[str, Any] = {
            "loss_function": "MAE",
            "eval_metric": "MAE",
            "learning_rate": p["learning_rate"],
            "depth": p["depth"],
            "iterations": int(catboost_iterations) if catboost_iterations else p["iterations"],
            "l2_leaf_reg": p["l2_leaf_reg"],
            "random_strength": p["random_strength"],
            "random_seed": 42,
            "bootstrap_type": "Bernoulli",
            "subsample": 0.8,
            "verbose": max(1, int(catboost_log_every)) if live_log else False,
            "metric_period": max(1, int(catboost_log_every)),
            "allow_writing_files": False,
        }
        if backend == "catboost_gpu":
            params["task_type"] = "GPU"
            params["devices"] = "0"
        else:
            params["task_type"] = "CPU"

        return CatBoostRegressor(**params)

    raise ValueError(
        f"Unknown model backend: {model_backend}. "
        "Supported: hgb, catboost_cpu, catboost_gpu"
    )


def prepare_X(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return df[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).astype("float32")


def feature_columns(
    daily: pd.DataFrame,
    feature_mode: str = "full",
) -> list[str]:
    excluded_cols = {"deviceId", "date", "period", "target_x2", "ym"}
    cols = [c for c in daily.columns if c not in excluded_cols]

    selected_mode = str(feature_mode).strip().lower()
    if selected_mode == "full":
        return cols
    if selected_mode == "no_raw_calendar":
        raw_calendar = {"year", "month", "dayofyear", "weekday", "hour", "minute"}
        return [c for c in cols if c not in raw_calendar and not c.endswith("_month_inter")]

    raise ValueError(
        f"Unknown feature_mode: {feature_mode}. Supported: full, no_raw_calendar"
    )


def aggregate_bucket_predictions_to_monthly(
    df: pd.DataFrame,
    pred_col: str,
    target_col: str = "target_x2",
) -> pd.DataFrame:
    return (
        df.groupby(["deviceId", "year", "month"], as_index=False)
        .agg(prediction=(pred_col, "mean"), target=(target_col, "mean"))
        .sort_values(["deviceId", "year", "month"])
        .reset_index(drop=True)
    )


def enrich_daily_with_sequence_features(
    daily: pd.DataFrame,
    data_dir: Path | None = None,
    verbose: bool = False,
    include_geo: bool = True,
    include_sequence: bool = True,
) -> pd.DataFrame:
    df = daily.sort_values(["deviceId", "date"]).copy()

    if include_geo and data_dir is not None:
        devices_path = data_dir / "devices.csv"
        if devices_path.exists():
            devices = pd.read_csv(
                devices_path, usecols=["deviceId", "latitude", "longitude"]
            )
            df = df.merge(devices, on="deviceId", how="left")
            df["geo_r"] = np.sqrt((df["latitude"] ** 2) + (df["longitude"] ** 2))
            df["geo_sum"] = df["latitude"] + df["longitude"]
            df["geo_diff"] = df["latitude"] - df["longitude"]
            if verbose:
                print(
                    f"[FE] merged devices.csv ({len(devices)} rows), "
                    f"feature count now: {len(df.columns)}"
                )

    if not include_sequence:
        return df

    seq_cols = [c for c in SEQ_SOURCE_CANDIDATES if c in df.columns]
    if verbose:
        print(f"[FE] sequence source columns: {len(seq_cols)} -> {seq_cols}")

    group = df.groupby("deviceId", sort=False)
    new_cols: dict[str, pd.Series] = {}

    for col in seq_cols:
        lag1 = group[col].shift(1)
        lag2 = group[col].shift(2)
        lag3 = group[col].shift(3)
        lag7 = group[col].shift(7)
        lag14 = group[col].shift(14)

        new_cols[f"{col}_lag1"] = lag1
        new_cols[f"{col}_lag2"] = lag2
        new_cols[f"{col}_lag3"] = lag3
        new_cols[f"{col}_lag7"] = lag7
        new_cols[f"{col}_lag14"] = lag14
        new_cols[f"{col}_roll3"] = (
            lag1.groupby(df["deviceId"])
            .rolling(3, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )
        new_cols[f"{col}_roll7"] = (
            lag1.groupby(df["deviceId"])
            .rolling(7, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )
        new_cols[f"{col}_roll14"] = (
            lag1.groupby(df["deviceId"])
            .rolling(14, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )
        new_cols[f"{col}_mom1"] = df[col] - lag1
        new_cols[f"{col}_mom7"] = df[col] - lag7

    center_cols = [
        c for c in ["t8_mean", "t5_mean", "t13_mean", "active_ratio", "x1_mean"] if c in df.columns
    ]
    for col in center_cols:
        hist_mean = (
            group[col]
            .shift(1)
            .groupby(df["deviceId"], sort=False)
            .expanding()
            .mean()
            .reset_index(level=0, drop=True)
        )
        new_cols[f"{col}_hist_mean"] = hist_mean
        new_cols[f"{col}_dev_center"] = df[col] - hist_mean
        new_cols[f"{col}_month_inter"] = df[col] * df["month"]

    if new_cols:
        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    if verbose:
        print(f"[FE] final feature count after sequence enrich: {len(df.columns)}")

    return df


def top_target_correlations(
    labelled_df: pd.DataFrame,
    top_n: int = 25,
) -> pd.DataFrame:
    if "target_x2" not in labelled_df.columns:
        return pd.DataFrame(columns=["feature", "corr", "abs_corr"])

    numeric_cols = labelled_df.select_dtypes(include=[np.number]).columns.tolist()
    if "target_x2" not in numeric_cols:
        return pd.DataFrame(columns=["feature", "corr", "abs_corr"])

    numeric_df = labelled_df[numeric_cols]
    valid_std_cols = numeric_df.std(numeric_only=True)
    valid_std_cols = valid_std_cols[valid_std_cols > 0].index.tolist()
    if "target_x2" not in valid_std_cols:
        return pd.DataFrame(columns=["feature", "corr", "abs_corr"])

    corr = numeric_df[valid_std_cols].corrwith(numeric_df["target_x2"])
    corr = corr.drop(labels=["target_x2"], errors="ignore").dropna()
    corr = corr.reindex(corr.abs().sort_values(ascending=False).index)
    corr = corr.head(top_n)
    return pd.DataFrame(
        {"feature": corr.index, "corr": corr.values, "abs_corr": corr.abs().values}
    )


def compute_train_weights(
    train_df: pd.DataFrame,
    mode: str = "none",
) -> np.ndarray | None:
    selected_mode = str(mode).strip().lower()
    if selected_mode == "none":
        return None

    weights = np.ones(len(train_df), dtype=np.float32)

    if "device_month_equal" in selected_mode:
        month_counts = (
            train_df.groupby(["deviceId", "year", "month"])["deviceId"]
            .transform("size")
            .astype("float32")
        )
        weights *= 1.0 / np.clip(month_counts.to_numpy(dtype=np.float32), 1.0, None)

    if "recent" in selected_mode:
        ym = train_df["ym"].astype("int32")
        ym_min = int(ym.min())
        ym_max = int(ym.max())
        denom = max(1, ym_max - ym_min)
        recent = 0.8 + 0.4 * ((ym - ym_min) / denom)
        weights *= recent.to_numpy(dtype=np.float32)

    if "warm" in selected_mode:
        t1 = train_df["t1_mean"].astype("float32") if "t1_mean" in train_df.columns else None
        if t1 is not None:
            q10 = float(np.nanquantile(t1, 0.10))
            q90 = float(np.nanquantile(t1, 0.90))
            temp_span = max(1e-6, q90 - q10)
            t1_norm = np.clip((t1 - q10) / temp_span, 0.0, 1.0)
            warm = 0.7 + 0.6 * t1_norm
            weights *= warm.to_numpy(dtype=np.float32)

    return np.clip(weights.astype(np.float32), 1e-4, 10.0)


def evaluate_rolling_months(
    labelled_df: pd.DataFrame,
    cols: list[str],
    first_valid_ym: int = 202501,
    verbose: bool = False,
    model_backend: str = "hgb",
    model_strength: str = "strong",
    catboost_iterations: int | None = None,
    catboost_log_every: int = 100,
    sample_weight_mode: str = "none",
) -> pd.DataFrame:
    valid_months = sorted(m for m in labelled_df["ym"].unique() if m >= first_valid_ym)
    rows: list[dict[str, float | int]] = []
    total_folds = len(valid_months)
    start_all = time.perf_counter()
    if verbose:
        print(
            f"[CV] model_backend={model_backend}, model_strength={model_strength}, "
            f"folds={total_folds}, catboost_iterations={catboost_iterations or 'default'}, "
            f"sample_weight_mode={sample_weight_mode}"
        )

    for fold_idx, valid_ym in enumerate(valid_months, start=1):
        train_df = labelled_df[labelled_df["ym"] < valid_ym].copy()
        valid_df = labelled_df[labelled_df["ym"] == valid_ym].copy()

        if train_df.empty or valid_df.empty:
            if verbose:
                print(
                    f"[CV {fold_idx}/{total_folds}] valid_ym={valid_ym} skipped (empty split)"
                )
            continue

        fold_start = time.perf_counter()
        if verbose:
            print(
                f"[CV {fold_idx}/{total_folds}] valid_ym={valid_ym} "
                f"train_rows={len(train_df)} valid_rows={len(valid_df)}"
            )

        device_mean = train_df.groupby("deviceId")["target_x2"].mean()
        global_mean = train_df["target_x2"].mean()

        train_base = train_df["deviceId"].map(device_mean).fillna(global_mean)
        valid_base = valid_df["deviceId"].map(device_mean).fillna(global_mean)

        X_train = prepare_X(train_df, cols)
        X_valid = prepare_X(valid_df, cols)
        y_train_residual = train_df["target_x2"] - train_base

        model = make_model(
            model_backend=model_backend,
            model_strength=model_strength,
            catboost_iterations=catboost_iterations,
            catboost_log_every=catboost_log_every,
            live_log=verbose,
        )
        fit_kwargs: dict[str, Any] = {}
        if _is_catboost_backend(model_backend):
            y_valid_residual = valid_df["target_x2"] - valid_base
            fit_kwargs = {
                "eval_set": (X_valid, y_valid_residual),
                "use_best_model": True,
                "early_stopping_rounds": 800,
            }

        train_weights = compute_train_weights(train_df, mode=sample_weight_mode)
        if train_weights is not None:
            fit_kwargs["sample_weight"] = train_weights

        model.fit(X_train, y_train_residual, **fit_kwargs)
        best_iter = np.nan
        if verbose and _is_catboost_backend(model_backend):
            try:
                best_iter_raw = int(model.get_best_iteration())
                if best_iter_raw >= 0:
                    best_iter = best_iter_raw + 1
                print(f"[CV {fold_idx}/{total_folds}] best_iteration={best_iter}")
            except Exception:
                pass

        valid_residual = model.predict(X_valid)
        valid_pred = (valid_base + valid_residual).clip(lower=0.0)
        row_mae = mean_absolute_error(valid_df["target_x2"], valid_pred)

        valid_eval = valid_df[["deviceId", "year", "month", "target_x2"]].copy()
        valid_eval["prediction_row"] = valid_pred.astype("float32")

        monthly_pred = aggregate_bucket_predictions_to_monthly(
            valid_eval,
            pred_col="prediction_row",
            target_col="target_x2",
        )
        model_monthly_mae = mean_absolute_error(monthly_pred["target"], monthly_pred["prediction"])

        monthly_baseline = (
            valid_eval.groupby(["deviceId", "year", "month"], as_index=False)
            .agg(target=("target_x2", "mean"))
            .sort_values(["deviceId", "year", "month"])
            .reset_index(drop=True)
        )
        monthly_baseline["prediction"] = (
            monthly_baseline["deviceId"].map(device_mean).fillna(global_mean).astype("float32")
        )
        baseline_monthly_mae = mean_absolute_error(
            monthly_baseline["target"],
            monthly_baseline["prediction"],
        )

        rows.append(
            {
                "valid_ym": int(valid_ym),
                "train_rows": len(train_df),
                "valid_rows": len(valid_df),
                "monthly_rows": int(len(monthly_pred)),
                "baseline_monthly_mae": float(baseline_monthly_mae),
                "model_monthly_mae": float(model_monthly_mae),
                "model_row_mae": float(row_mae),
                "best_iteration": best_iter,
            }
        )

        if verbose:
            fold_time = time.perf_counter() - fold_start
            elapsed = time.perf_counter() - start_all
            avg_fold = elapsed / fold_idx
            eta = avg_fold * (total_folds - fold_idx)
            print(
                f"[CV {fold_idx}/{total_folds}] done "
                f"baseline_monthly_mae={baseline_monthly_mae:.6f} "
                f"model_monthly_mae={model_monthly_mae:.6f} row_mae={row_mae:.6f} "
                f"fold_time={_fmt_duration(fold_time)} elapsed={_fmt_duration(elapsed)} "
                f"eta={_fmt_duration(eta)}"
            )

    return pd.DataFrame(rows)


def train_and_predict_monthly(
    daily: pd.DataFrame,
    out_submission_path: Path,
    verbose: bool = False,
    model_backend: str = "hgb",
    model_strength: str = "strong",
    catboost_iterations: int | None = None,
    catboost_log_every: int = 100,
    sample_weight_mode: str = "none",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    stage_start = time.perf_counter()
    labelled = daily[daily["target_x2"].notna()].copy()
    forecast = daily[daily["target_x2"].isna()].copy()

    labelled["ym"] = labelled["year"].astype("int32") * 100 + labelled["month"].astype("int32")
    forecast["ym"] = forecast["year"].astype("int32") * 100 + forecast["month"].astype("int32")
    cols = feature_columns(labelled)
    if verbose:
        print(
            f"[FINAL] labelled_rows={len(labelled)} forecast_rows={len(forecast)} "
            f"features={len(cols)} model_backend={model_backend} "
            f"model_strength={model_strength} "
            f"catboost_iterations={catboost_iterations or 'default'} "
            f"sample_weight_mode={sample_weight_mode}"
        )

    baseline_start = time.perf_counter()
    device_mean_all = labelled.groupby("deviceId")["target_x2"].mean()
    global_mean_all = labelled["target_x2"].mean()
    if verbose:
        print(
            f"[FINAL] computed device baseline in {_fmt_duration(time.perf_counter() - baseline_start)}"
        )

    train_base_all = labelled["deviceId"].map(device_mean_all).fillna(global_mean_all)
    y_train_residual_all = labelled["target_x2"] - train_base_all

    X_labelled = prepare_X(labelled, cols)
    X_forecast = prepare_X(forecast, cols)

    final_model = make_model(
        model_backend=model_backend,
        model_strength=model_strength,
        catboost_iterations=catboost_iterations,
        catboost_log_every=catboost_log_every,
        live_log=verbose,
    )
    fit_start = time.perf_counter()
    if verbose:
        print("[FINAL] fitting residual model...")
    train_weights = compute_train_weights(labelled, mode=sample_weight_mode)
    fit_kwargs: dict[str, Any] = {}
    if train_weights is not None:
        fit_kwargs["sample_weight"] = train_weights
    final_model.fit(X_labelled, y_train_residual_all, **fit_kwargs)
    if verbose:
        print(f"[FINAL] fit done in {_fmt_duration(time.perf_counter() - fit_start)}")

    pred_start = time.perf_counter()
    forecast_base = forecast["deviceId"].map(device_mean_all).fillna(global_mean_all)
    forecast_residual = final_model.predict(X_forecast)
    if verbose:
        print(f"[FINAL] daily predictions done in {_fmt_duration(time.perf_counter() - pred_start)}")

    daily_predictions = forecast[["deviceId", "date", "year", "month"]].copy()
    daily_predictions["prediction_daily"] = (forecast_base + forecast_residual).clip(lower=0.0)

    monthly_submission = (
        daily_predictions.groupby(["deviceId", "year", "month"], as_index=False)[
            "prediction_daily"
        ]
        .mean()
        .rename(columns={"prediction_daily": "prediction"})
    )

    months_table = pd.DataFrame({"year": [2025] * 6, "month": [5, 6, 7, 8, 9, 10]})
    devices_table = pd.DataFrame({"deviceId": forecast["deviceId"].unique()})
    target_grid = devices_table.assign(_k=1).merge(months_table.assign(_k=1), on="_k")
    target_grid = target_grid.drop(columns="_k")

    monthly_submission = target_grid.merge(
        monthly_submission, on=["deviceId", "year", "month"], how="left"
    )
    fallback = monthly_submission["deviceId"].map(device_mean_all).fillna(global_mean_all)
    monthly_submission["prediction"] = monthly_submission["prediction"].fillna(fallback)
    monthly_submission["prediction"] = monthly_submission["prediction"].clip(lower=0.0)
    monthly_submission = monthly_submission.sort_values(
        ["deviceId", "year", "month"]
    ).reset_index(drop=True)

    out_submission_path.parent.mkdir(parents=True, exist_ok=True)
    monthly_submission.to_csv(out_submission_path, index=False)
    if verbose:
        print(
            f"[FINAL] saved submission to {out_submission_path} "
            f"in {_fmt_duration(time.perf_counter() - stage_start)}"
        )

    return monthly_submission, labelled
