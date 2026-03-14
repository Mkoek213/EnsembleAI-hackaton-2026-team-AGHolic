from __future__ import annotations

import argparse
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from daily_pipeline import find_data_dir


SENSOR_COLS = [f"t{i}" for i in range(1, 14)]
MODEL_STRENGTHS = {"fast", "strong", "heavy"}
SEQ_COLS = ["t8", "t5", "t13", "x1", "active"]
LAG_STEPS = (1, 12, 288)  # 5min, 1h, 24h for 5-minute telemetry


def _fmt_duration(seconds: float) -> str:
    seconds = int(max(seconds, 0))
    hours, rem = divmod(seconds, 3600)
    minutes, sec = divmod(rem, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{sec:02d}"
    return f"{minutes:02d}:{sec:02d}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train raw 5-minute model (no day/hour aggregation) and produce monthly submission."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Directory containing data.csv/devices.csv. Default: auto-detect '.' or 'task3'.",
    )
    parser.add_argument(
        "--submission-path",
        type=str,
        default=None,
        help="Output submission CSV path. Default: <data-dir>/out/submission_raw_seq_gpu.csv",
    )
    parser.add_argument(
        "--save-cv",
        type=str,
        default=None,
        help="Optional path to save rolling CV results.",
    )
    parser.add_argument(
        "--cv-first-valid-ym",
        type=int,
        default=202501,
        help="First validation month (YYYYMM) for rolling CV over labelled months.",
    )
    parser.add_argument(
        "--model-backend",
        type=str,
        default="catboost_gpu",
        choices=["catboost_cpu", "catboost_gpu"],
        help="CatBoost backend.",
    )
    parser.add_argument(
        "--model-strength",
        type=str,
        default="strong",
        choices=["fast", "strong", "heavy"],
        help="Preset controlling model capacity.",
    )
    parser.add_argument(
        "--catboost-iterations",
        type=int,
        default=None,
        help="Override CatBoost iterations.",
    )
    parser.add_argument(
        "--catboost-log-every",
        type=int,
        default=100,
        help="Print CatBoost metrics every N iterations.",
    )
    parser.add_argument(
        "--train-sample-frac",
        type=float,
        default=1.0,
        help="Optional training-row subsampling for each fold and final fit (0, 1].",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable detailed logs.",
    )
    return parser.parse_args()


def _make_catboost(
    backend: str,
    strength: str,
    iterations: int | None,
    log_every: int,
    seed: int,
    live_log: bool,
) -> Any:
    try:
        from catboost import CatBoostRegressor
    except ImportError as exc:
        raise ImportError(
            "catboost is required. Install with `pip install catboost`."
        ) from exc

    params_by_strength: dict[str, dict[str, Any]] = {
        "fast": {
            "learning_rate": 0.05,
            "depth": 8,
            "iterations": 5000,
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
    if strength not in MODEL_STRENGTHS:
        raise ValueError(f"Unknown model_strength: {strength}")
    p = params_by_strength[strength]

    params: dict[str, Any] = {
        "loss_function": "MAE",
        "eval_metric": "MAE",
        "learning_rate": p["learning_rate"],
        "depth": p["depth"],
        "iterations": int(iterations) if iterations else p["iterations"],
        "l2_leaf_reg": p["l2_leaf_reg"],
        "random_strength": p["random_strength"],
        "bootstrap_type": "Bernoulli",
        "subsample": 0.8,
        "random_seed": seed,
        "allow_writing_files": False,
        "verbose": max(1, int(log_every)) if live_log else False,
        "metric_period": max(1, int(log_every)),
    }
    if backend == "catboost_gpu":
        params["task_type"] = "GPU"
        params["devices"] = "0"
    else:
        params["task_type"] = "CPU"
    return CatBoostRegressor(**params)


def _prepare_X(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return df[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).astype("float32")


def _feature_columns(df: pd.DataFrame) -> list[str]:
    excluded = {"deviceId", "timedate", "date", "period", "x2", "ym"}
    return [c for c in df.columns if c not in excluded]


def _build_raw_features(data_dir: Path, verbose: bool = False) -> pd.DataFrame:
    csv_path = data_dir / "data.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing input CSV: {csv_path}")

    usecols = ["deviceId", "timedate", "period", "x3", "deviceType"] + SENSOR_COLS + ["x1", "x2"]
    dtype: dict[str, str] = {
        "deviceId": "category",
        "period": "category",
        "x3": "Int16",
        "deviceType": "Int16",
    }
    for col in SENSOR_COLS + ["x1", "x2"]:
        dtype[col] = "float32"

    load_start = time.perf_counter()
    df = pd.read_csv(csv_path, usecols=usecols, dtype=dtype, low_memory=False)
    if verbose:
        print(f"Loaded raw rows: {len(df)} in {_fmt_duration(time.perf_counter() - load_start)}")

    ts = pd.to_datetime(df["timedate"], utc=True, errors="coerce")
    df = df.loc[ts.notna()].copy()
    df["date"] = ts.loc[ts.notna()].dt.tz_convert(None)
    df["year"] = df["date"].dt.year.astype("int16")
    df["month"] = df["date"].dt.month.astype("int8")
    df["dayofyear"] = df["date"].dt.dayofyear.astype("int16")
    df["weekday"] = df["date"].dt.weekday.astype("int8")
    df["hour"] = df["date"].dt.hour.astype("int8")
    df["minute"] = df["date"].dt.minute.astype("int8")
    df["is_weekend"] = (df["weekday"] >= 5).astype("int8")

    doy_angle = 2.0 * np.pi * df["dayofyear"] / 366.0
    tod_angle = 2.0 * np.pi * (
        (df["hour"].astype("float32") * 60.0 + df["minute"].astype("float32")) / 1440.0
    )
    df["doy_sin"] = np.sin(doy_angle)
    df["doy_cos"] = np.cos(doy_angle)
    df["tod_sin"] = np.sin(tod_angle)
    df["tod_cos"] = np.cos(tod_angle)

    df["delta_t2_t1"] = df["t2"] - df["t1"]
    df["delta_t5_t3"] = df["t5"] - df["t3"]
    df["delta_t6_t4"] = df["t6"] - df["t4"]
    df["delta_t7_t2"] = df["t7"] - df["t2"]
    df["delta_t9_t1"] = df["t9"] - df["t1"]
    df["active"] = (df["x1"] > 0).astype("float32")

    devices_path = data_dir / "devices.csv"
    if devices_path.exists():
        devices = pd.read_csv(devices_path, usecols=["deviceId", "latitude", "longitude"])
        devices["deviceId"] = devices["deviceId"].astype("string")
        df["deviceId"] = df["deviceId"].astype("string")
        df = df.merge(devices, on="deviceId", how="left")
        df["geo_r"] = np.sqrt((df["latitude"] ** 2) + (df["longitude"] ** 2))
        df["geo_sum"] = df["latitude"] + df["longitude"]
        df["geo_diff"] = df["latitude"] - df["longitude"]
        df["deviceId"] = df["deviceId"].astype("category")
        if verbose:
            print(f"Merged devices metadata ({len(devices)} rows).")

    # Sequence features on raw 5-minute telemetry
    seq_start = time.perf_counter()
    df = df.sort_values(["deviceId", "date"]).reset_index(drop=True)
    group = df.groupby("deviceId", sort=False)
    new_cols: dict[str, pd.Series] = {}
    seq_used = [c for c in SEQ_COLS if c in df.columns]
    if verbose:
        print(f"[RAW FE] sequence cols: {seq_used}, lags: {list(LAG_STEPS)}")

    for col in seq_used:
        lag_refs: dict[int, pd.Series] = {}
        for lag in LAG_STEPS:
            shifted = group[col].shift(lag)
            lag_refs[lag] = shifted
            new_cols[f"{col}_lag{lag}"] = shifted
        new_cols[f"{col}_mom1"] = df[col] - lag_refs[1]
        new_cols[f"{col}_mom288"] = df[col] - lag_refs[288]

    if new_cols:
        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    if verbose:
        print(
            f"[RAW FE] features done in {_fmt_duration(time.perf_counter() - seq_start)}; "
            f"shape={df.shape}"
        )

    return df


def _sample_train(
    train_df: pd.DataFrame,
    sample_frac: float,
    seed: int,
) -> pd.DataFrame:
    if sample_frac >= 1.0:
        return train_df
    if sample_frac <= 0.0:
        raise ValueError("--train-sample-frac must be in (0, 1].")
    # Stratify by month to preserve seasonality.
    return (
        train_df.groupby("ym", group_keys=False, observed=True)
        .apply(lambda g: g.sample(frac=sample_frac, random_state=seed))
        .reset_index(drop=True)
    )


def _monthly_from_rows(df: pd.DataFrame, pred_col: str, target_col: str = "x2") -> pd.DataFrame:
    out = (
        df.groupby(["deviceId", "year", "month"], observed=True, as_index=False)
        .agg(prediction=(pred_col, "mean"), target=(target_col, "mean"))
    )
    return out


def run_cv(
    labelled: pd.DataFrame,
    cols: list[str],
    first_valid_ym: int,
    model_backend: str,
    model_strength: str,
    catboost_iterations: int | None,
    catboost_log_every: int,
    train_sample_frac: float,
    seed: int,
    verbose: bool,
) -> pd.DataFrame:
    valid_months = sorted(m for m in labelled["ym"].unique() if m >= first_valid_ym)
    rows: list[dict[str, float | int]] = []
    total = len(valid_months)
    start_all = time.perf_counter()
    if verbose:
        print(
            f"[CV RAW] backend={model_backend} strength={model_strength} "
            f"folds={total} sample_frac={train_sample_frac}"
        )

    for fold_idx, valid_ym in enumerate(valid_months, start=1):
        train_df = labelled[labelled["ym"] < valid_ym].copy()
        valid_df = labelled[labelled["ym"] == valid_ym].copy()
        if train_df.empty or valid_df.empty:
            continue

        train_df = _sample_train(train_df, train_sample_frac, seed + fold_idx)
        fold_start = time.perf_counter()

        device_mean = train_df.groupby("deviceId", observed=True)["x2"].mean()
        global_mean = train_df["x2"].mean()
        train_base = train_df["deviceId"].map(device_mean).fillna(global_mean)
        valid_base = valid_df["deviceId"].map(device_mean).fillna(global_mean)

        model = _make_catboost(
            backend=model_backend,
            strength=model_strength,
            iterations=catboost_iterations,
            log_every=catboost_log_every,
            seed=seed,
            live_log=verbose,
        )
        X_train = _prepare_X(train_df, cols)
        X_valid = _prepare_X(valid_df, cols)
        y_train_res = train_df["x2"] - train_base
        y_valid_res = valid_df["x2"] - valid_base
        model.fit(
            X_train,
            y_train_res,
            eval_set=(X_valid, y_valid_res),
            use_best_model=True,
            early_stopping_rounds=1000,
        )

        valid_pred = (valid_base + model.predict(X_valid)).clip(lower=0.0)
        valid_eval = valid_df[["deviceId", "year", "month", "x2"]].copy()
        valid_eval["prediction_row"] = valid_pred.astype("float32")

        row_mae = mean_absolute_error(valid_eval["x2"], valid_eval["prediction_row"])

        monthly_pred = _monthly_from_rows(valid_eval, pred_col="prediction_row", target_col="x2")
        model_mae_monthly = mean_absolute_error(monthly_pred["target"], monthly_pred["prediction"])

        monthly_baseline = (
            valid_eval.groupby(["deviceId", "year", "month"], observed=True, as_index=False)
            .agg(target=("x2", "mean"))
        )
        monthly_baseline["prediction"] = (
            monthly_baseline["deviceId"].map(device_mean).fillna(global_mean).astype("float32")
        )
        baseline_mae_monthly = mean_absolute_error(
            monthly_baseline["target"],
            monthly_baseline["prediction"],
        )

        rows.append(
            {
                "valid_ym": int(valid_ym),
                "train_rows": int(len(train_df)),
                "valid_rows": int(len(valid_df)),
                "baseline_monthly_mae": float(baseline_mae_monthly),
                "model_monthly_mae": float(model_mae_monthly),
                "model_row_mae": float(row_mae),
            }
        )

        if verbose:
            fold_time = time.perf_counter() - fold_start
            elapsed = time.perf_counter() - start_all
            eta = (elapsed / fold_idx) * (total - fold_idx)
            print(
                f"[CV RAW {fold_idx}/{total}] ym={valid_ym} "
                f"baseline_monthly_mae={baseline_mae_monthly:.6f} "
                f"model_monthly_mae={model_mae_monthly:.6f} row_mae={row_mae:.6f} "
                f"fold_time={_fmt_duration(fold_time)} elapsed={_fmt_duration(elapsed)} "
                f"eta={_fmt_duration(eta)}"
            )

    return pd.DataFrame(rows)


def train_final_and_submit(
    full_df: pd.DataFrame,
    cols: list[str],
    submission_path: Path,
    model_backend: str,
    model_strength: str,
    catboost_iterations: int | None,
    catboost_log_every: int,
    train_sample_frac: float,
    seed: int,
    verbose: bool,
) -> pd.DataFrame:
    labelled = full_df[full_df["x2"].notna()].copy()
    forecast = full_df[full_df["x2"].isna()].copy()

    if labelled.empty or forecast.empty:
        raise RuntimeError("Labelled or forecast split is empty.")

    labelled["ym"] = labelled["year"] * 100 + labelled["month"]
    labelled_fit = _sample_train(labelled, train_sample_frac, seed)

    device_mean = labelled_fit.groupby("deviceId", observed=True)["x2"].mean()
    global_mean = labelled_fit["x2"].mean()
    train_base = labelled_fit["deviceId"].map(device_mean).fillna(global_mean)

    X_train = _prepare_X(labelled_fit, cols)
    y_train_res = labelled_fit["x2"] - train_base
    X_forecast = _prepare_X(forecast, cols)
    forecast_base = forecast["deviceId"].map(device_mean).fillna(global_mean)

    model = _make_catboost(
        backend=model_backend,
        strength=model_strength,
        iterations=catboost_iterations,
        log_every=catboost_log_every,
        seed=seed,
        live_log=verbose,
    )
    fit_start = time.perf_counter()
    if verbose:
        print(
            f"[FINAL RAW] fit_rows={len(labelled_fit)} forecast_rows={len(forecast)} "
            f"features={len(cols)}"
        )
    model.fit(X_train, y_train_res)
    if verbose:
        print(f"[FINAL RAW] fit done in {_fmt_duration(time.perf_counter() - fit_start)}")

    row_pred = (forecast_base + model.predict(X_forecast)).clip(lower=0.0)
    pred_rows = forecast[["deviceId", "year", "month"]].copy()
    pred_rows["prediction_row"] = row_pred.astype("float32")

    submission = (
        pred_rows.groupby(["deviceId", "year", "month"], observed=True, as_index=False)["prediction_row"]
        .mean()
        .rename(columns={"prediction_row": "prediction"})
    )

    months_table = pd.DataFrame({"year": [2025] * 6, "month": [5, 6, 7, 8, 9, 10]})
    devices_table = pd.DataFrame({"deviceId": forecast["deviceId"].astype("string").unique()})
    target_grid = devices_table.assign(_k=1).merge(months_table.assign(_k=1), on="_k").drop(columns="_k")

    submission["deviceId"] = submission["deviceId"].astype("string")
    submission = target_grid.merge(submission, on=["deviceId", "year", "month"], how="left")
    submission["prediction"] = submission["prediction"].fillna(
        submission["deviceId"].map(device_mean.astype("float32")).fillna(global_mean)
    )
    submission["prediction"] = submission["prediction"].clip(lower=0.0)
    submission = submission.sort_values(["deviceId", "year", "month"]).reset_index(drop=True)

    submission_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(submission_path, index=False)
    return submission


def main() -> None:
    args = parse_args()
    verbose = not args.quiet
    started = time.perf_counter()

    data_dir = Path(args.data_dir) if args.data_dir else find_data_dir()
    submission_path = (
        Path(args.submission_path)
        if args.submission_path
        else data_dir / "out" / "submission_raw_seq_gpu.csv"
    )

    print(f"Data dir: {data_dir.resolve()}")
    print("Building raw 5-minute feature table...")
    raw_df = _build_raw_features(data_dir=data_dir, verbose=verbose)
    print(f"Raw feature shape: {raw_df.shape}")

    labelled = raw_df[raw_df["x2"].notna()].copy()
    labelled["ym"] = labelled["year"] * 100 + labelled["month"]
    cols = _feature_columns(labelled)
    print(f"Model feature count: {len(cols)}")

    print("Running rolling month validation (monthly MAE computed from row-level predictions)...")
    cv_df = run_cv(
        labelled=labelled,
        cols=cols,
        first_valid_ym=args.cv_first_valid_ym,
        model_backend=args.model_backend,
        model_strength=args.model_strength,
        catboost_iterations=args.catboost_iterations,
        catboost_log_every=args.catboost_log_every,
        train_sample_frac=args.train_sample_frac,
        seed=args.random_seed,
        verbose=verbose,
    )
    if cv_df.empty:
        print("No CV folds generated.")
    else:
        print(cv_df.to_string(index=False))
        print(f"Average baseline monthly MAE: {cv_df['baseline_monthly_mae'].mean():.6f}")
        print(f"Average model monthly MAE:    {cv_df['model_monthly_mae'].mean():.6f}")
        print(f"Average model row MAE:        {cv_df['model_row_mae'].mean():.6f}")
        if args.save_cv:
            save_path = Path(args.save_cv)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            cv_df.to_csv(save_path, index=False)
            print(f"Saved CV table: {save_path.resolve()}")

    print("Training final model and creating submission...")
    submission = train_final_and_submit(
        full_df=raw_df,
        cols=cols,
        submission_path=submission_path,
        model_backend=args.model_backend,
        model_strength=args.model_strength,
        catboost_iterations=args.catboost_iterations,
        catboost_log_every=args.catboost_log_every,
        train_sample_frac=args.train_sample_frac,
        seed=args.random_seed,
        verbose=verbose,
    )

    actual_rows = int(len(submission))
    missing = int(submission["prediction"].isna().sum())
    months = sorted(submission["month"].unique().tolist())
    forecast_devices = int(raw_df[raw_df["x2"].isna()]["deviceId"].nunique())
    expected_rows = forecast_devices * 6

    print(f"Submission path: {submission_path.resolve()}")
    print(f"Submission rows: {actual_rows}")
    print(f"Expected rows (devices * 6): {expected_rows}")
    print(f"Missing predictions: {missing}")
    print(f"Months present: {months}")

    if actual_rows != expected_rows:
        raise RuntimeError(f"Submission row mismatch: expected {expected_rows}, got {actual_rows}")
    if missing != 0:
        raise RuntimeError(f"Submission contains {missing} missing predictions.")
    if months != [5, 6, 7, 8, 9, 10]:
        raise RuntimeError(f"Unexpected months in submission: {months}")

    print("Sanity checks passed.")
    print(f"Total runtime: {time.perf_counter() - started:.1f}s")


if __name__ == "__main__":
    main()
