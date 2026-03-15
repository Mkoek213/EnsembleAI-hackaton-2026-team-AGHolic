from __future__ import annotations

import argparse
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from daily_pipeline import find_data_dir


CATEGORICAL_FEATURES = ["deviceId", "deviceType", "x3", "month"]


def _fmt_duration(seconds: float) -> str:
    seconds = int(max(seconds, 0))
    hours, rem = divmod(seconds, 3600)
    minutes, sec = divmod(rem, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{sec:02d}"
    return f"{minutes:02d}:{sec:02d}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Direct monthly model: predict device-month target_x2 from monthly feature table."
    )
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--monthly-path", type=str, default=None)
    parser.add_argument("--submission-path", type=str, default=None)
    parser.add_argument("--save-cv", type=str, default=None)
    parser.add_argument("--cv-first-valid-ym", type=int, default=202501)
    parser.add_argument("--iterations", type=int, default=4000)
    parser.add_argument("--early-stopping-rounds", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--l2-leaf-reg", type=float, default=8.0)
    parser.add_argument("--random-strength", type=float, default=1.5)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument(
        "--backend",
        type=str,
        default="catboost_gpu",
        choices=["catboost_gpu", "catboost_cpu"],
    )
    parser.add_argument("--corr-top-n", type=int, default=20)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def _make_model(args: argparse.Namespace, live_log: bool) -> Any:
    try:
        from catboost import CatBoostRegressor
    except ImportError as exc:
        raise ImportError("catboost is required for train_monthly_direct.py") from exc

    params: dict[str, Any] = {
        "loss_function": "MAE",
        "eval_metric": "MAE",
        "iterations": int(args.iterations),
        "learning_rate": float(args.learning_rate),
        "depth": int(args.depth),
        "l2_leaf_reg": float(args.l2_leaf_reg),
        "random_strength": float(args.random_strength),
        "bootstrap_type": "Bernoulli",
        "subsample": float(args.subsample),
        "random_seed": 42,
        "allow_writing_files": False,
        "verbose": max(1, int(args.log_every)) if live_log else False,
        "metric_period": max(1, int(args.log_every)),
        "task_type": "GPU" if args.backend == "catboost_gpu" else "CPU",
    }
    if args.backend == "catboost_gpu":
        params["devices"] = "0"
    return CatBoostRegressor(**params)


def _feature_columns(df: pd.DataFrame) -> list[str]:
    excluded = {
        "period",
        "target_x2",
        "ym",
        "year",
        # Direct target leakage / train-forecast mismatch.
        "x2_sum",
        "x2_count",
    }
    return [c for c in df.columns if c not in excluded]


def _prepare_xy(
    df: pd.DataFrame,
    cols: list[str],
    target_col: str = "target_x2",
) -> tuple[pd.DataFrame, pd.Series]:
    X = df[cols].copy()
    for col in X.columns:
        if col in CATEGORICAL_FEATURES:
            X[col] = X[col].astype("string").fillna("NA")
        else:
            X[col] = pd.to_numeric(X[col], errors="coerce").astype("float32")
    y = pd.to_numeric(df[target_col], errors="coerce").astype("float32")
    return X, y


def _prepare_X(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    X = df[cols].copy()
    for col in X.columns:
        if col in CATEGORICAL_FEATURES:
            X[col] = X[col].astype("string").fillna("NA")
        else:
            X[col] = pd.to_numeric(X[col], errors="coerce").astype("float32")
    return X


def _cat_feature_indices(cols: list[str]) -> list[int]:
    return [idx for idx, col in enumerate(cols) if col in CATEGORICAL_FEATURES]


def top_target_correlations_monthly(df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    numeric_df = df.select_dtypes(include=[np.number]).copy()
    numeric_df = numeric_df.drop(columns=["x2_sum", "x2_count"], errors="ignore")
    if "target_x2" not in numeric_df.columns:
        return pd.DataFrame(columns=["feature", "corr", "abs_corr"])
    valid_cols = numeric_df.std(numeric_only=True)
    valid_cols = valid_cols[valid_cols > 0].index.tolist()
    if "target_x2" not in valid_cols:
        return pd.DataFrame(columns=["feature", "corr", "abs_corr"])
    corr = numeric_df[valid_cols].corrwith(numeric_df["target_x2"])
    corr = corr.drop(labels=["target_x2"], errors="ignore").dropna()
    corr = corr.reindex(corr.abs().sort_values(ascending=False).index).head(top_n)
    return pd.DataFrame({"feature": corr.index, "corr": corr.values, "abs_corr": corr.abs().values})


def _predict_fold(
    args: argparse.Namespace,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    cols: list[str],
    verbose: bool,
) -> tuple[np.ndarray, float]:
    device_mean = train_df.groupby("deviceId")["target_x2"].mean()
    global_mean = float(train_df["target_x2"].mean())
    train_base = pd.to_numeric(train_df["deviceId"].map(device_mean), errors="coerce").fillna(global_mean)
    valid_base = pd.to_numeric(valid_df["deviceId"].map(device_mean), errors="coerce").fillna(global_mean)

    train_df = train_df.copy()
    valid_df = valid_df.copy()
    train_df["target_residual"] = train_df["target_x2"] - train_base
    valid_df["target_residual"] = valid_df["target_x2"] - valid_base

    X_train, y_train = _prepare_xy(train_df, cols, target_col="target_residual")
    X_valid, y_valid = _prepare_xy(valid_df, cols, target_col="target_residual")
    cat_features = _cat_feature_indices(cols)

    model = _make_model(args, live_log=verbose)
    model.fit(
        X_train,
        y_train,
        cat_features=cat_features,
        eval_set=(X_valid, y_valid),
        use_best_model=True,
        early_stopping_rounds=max(50, int(args.early_stopping_rounds)),
    )
    try:
        best_iter = int(model.get_best_iteration()) + 1
    except Exception:
        best_iter = int(args.iterations)
    pred_res = model.predict(X_valid).astype("float32")
    pred = np.clip(valid_base.to_numpy(dtype=np.float32) + pred_res, 0.0, None)
    return pred, float(best_iter)


def _train_final_predict(
    args: argparse.Namespace,
    labelled: pd.DataFrame,
    forecast: pd.DataFrame,
    cols: list[str],
    final_iterations: int,
    verbose: bool,
) -> pd.DataFrame:
    devices_table = pd.DataFrame({"deviceId": forecast["deviceId"].astype("string").unique()})
    months_table = pd.DataFrame({"year": [2025] * 6, "month": [5, 6, 7, 8, 9, 10]})
    target_grid = devices_table.assign(_k=1).merge(months_table.assign(_k=1), on="_k").drop(columns="_k")

    device_mean = labelled.groupby("deviceId")["target_x2"].mean()
    global_mean = float(labelled["target_x2"].mean())
    train_base = pd.to_numeric(labelled["deviceId"].map(device_mean), errors="coerce").fillna(global_mean)
    forecast_base = pd.to_numeric(forecast["deviceId"].map(device_mean), errors="coerce").fillna(global_mean)

    labelled = labelled.copy()
    labelled["target_residual"] = labelled["target_x2"] - train_base

    train_args = argparse.Namespace(**vars(args))
    train_args.iterations = int(final_iterations)

    X_train, y_train = _prepare_xy(labelled, cols, target_col="target_residual")
    X_forecast = _prepare_X(forecast, cols)
    cat_features = _cat_feature_indices(cols)

    model = _make_model(train_args, live_log=verbose)
    fit_start = time.perf_counter()
    if verbose:
        print(
            f"[FINAL MONTHLY] fit_rows={len(labelled)} forecast_rows={len(forecast)} "
            f"features={len(cols)} iterations={final_iterations}"
        )
    model.fit(X_train, y_train, cat_features=cat_features)
    if verbose:
        print(f"[FINAL MONTHLY] fit done in {_fmt_duration(time.perf_counter() - fit_start)}")

    pred_res = model.predict(X_forecast).astype("float32")
    pred = np.clip(forecast_base.to_numpy(dtype=np.float32) + pred_res, 0.0, None)
    partial_submission = forecast[["deviceId", "year", "month"]].copy()
    partial_submission["deviceId"] = partial_submission["deviceId"].astype("string")
    partial_submission["prediction"] = pred

    submission = target_grid.merge(
        partial_submission,
        on=["deviceId", "year", "month"],
        how="left",
    )
    fallback = submission["deviceId"].map(device_mean).fillna(global_mean)
    submission["prediction"] = submission["prediction"].fillna(fallback).clip(lower=0.0)
    submission = submission.sort_values(["deviceId", "year", "month"]).reset_index(drop=True)
    return submission


def main() -> None:
    args = parse_args()
    verbose = not args.quiet
    started = time.perf_counter()

    data_dir = Path(args.data_dir) if args.data_dir else find_data_dir()
    monthly_path = (
        Path(args.monthly_path) if args.monthly_path else data_dir / "out" / "monthly_features.csv"
    )
    submission_path = (
        Path(args.submission_path)
        if args.submission_path
        else data_dir / "out" / "submission_monthly_direct.csv"
    )

    if not monthly_path.exists():
        raise FileNotFoundError(f"Missing monthly feature file: {monthly_path}")

    print(f"Loading monthly features from: {monthly_path.resolve()}")
    load_start = time.perf_counter()
    monthly = pd.read_csv(monthly_path)
    print(f"Monthly shape: {monthly.shape} (loaded in {time.perf_counter() - load_start:.1f}s)")

    labelled = monthly[monthly["target_x2"].notna()].copy()
    forecast = monthly[monthly["target_x2"].isna()].copy()
    labelled["ym"] = labelled["ym"].astype("int32")
    forecast["ym"] = forecast["ym"].astype("int32")
    cols = _feature_columns(labelled)
    print(f"Model feature count: {len(cols)}")

    if args.corr_top_n > 0:
        corr_df = top_target_correlations_monthly(labelled, top_n=args.corr_top_n)
        if not corr_df.empty:
            print(f"Top {len(corr_df)} abs correlations with target_x2:")
            print(corr_df.to_string(index=False))

    valid_months = sorted([m for m in labelled["ym"].unique() if m >= args.cv_first_valid_ym])
    rows: list[dict[str, float | int]] = []
    best_iters: list[float] = []
    start_cv = time.perf_counter()
    print(f"[CV MONTHLY] folds={len(valid_months)} backend={args.backend} iterations={args.iterations}")

    for fold_idx, valid_ym in enumerate(valid_months, start=1):
        train_df = labelled[labelled["ym"] < valid_ym].copy()
        valid_df = labelled[labelled["ym"] == valid_ym].copy()
        if train_df.empty or valid_df.empty:
            continue

        fold_start = time.perf_counter()
        print(
            f"[CV MONTHLY {fold_idx}/{len(valid_months)}] valid_ym={valid_ym} "
            f"train_rows={len(train_df)} valid_rows={len(valid_df)}"
        )

        device_mean = train_df.groupby("deviceId")["target_x2"].mean()
        global_mean = float(train_df["target_x2"].mean())
        baseline = pd.to_numeric(valid_df["deviceId"].map(device_mean), errors="coerce").fillna(global_mean)
        baseline_mae = mean_absolute_error(valid_df["target_x2"], baseline)

        pred, best_iter = _predict_fold(args, train_df, valid_df, cols, verbose=verbose)
        model_mae = mean_absolute_error(valid_df["target_x2"], pred)
        best_iters.append(best_iter)

        rows.append(
            {
                "valid_ym": int(valid_ym),
                "train_rows": int(len(train_df)),
                "valid_rows": int(len(valid_df)),
                "baseline_mae": float(baseline_mae),
                "model_mae": float(model_mae),
                "best_iteration": int(best_iter),
            }
        )

        elapsed = time.perf_counter() - start_cv
        eta = (elapsed / fold_idx) * (len(valid_months) - fold_idx)
        print(
            f"[CV MONTHLY {fold_idx}/{len(valid_months)}] baseline_mae={baseline_mae:.6f} "
            f"model_mae={model_mae:.6f} best_iteration={int(best_iter)} "
            f"fold_time={_fmt_duration(time.perf_counter() - fold_start)} "
            f"eta={_fmt_duration(eta)}"
        )

    cv_df = pd.DataFrame(rows)
    if cv_df.empty:
        raise RuntimeError("No CV folds generated.")

    print(cv_df.to_string(index=False))
    print(f"Average baseline MAE: {cv_df['baseline_mae'].mean():.6f}")
    print(f"Average model MAE:    {cv_df['model_mae'].mean():.6f}")
    if args.save_cv:
        save_cv = Path(args.save_cv)
        save_cv.parent.mkdir(parents=True, exist_ok=True)
        cv_df.to_csv(save_cv, index=False)
        print(f"Saved CV results: {save_cv.resolve()}")

    median_best_iter = int(np.median(best_iters)) if best_iters else int(args.iterations)
    final_iterations = int(np.clip(round(median_best_iter * 1.1), 100, args.iterations))
    print(
        f"Final iterations from CV best_iteration median: {median_best_iter} -> {final_iterations}"
    )

    submission = _train_final_predict(
        args=args,
        labelled=labelled,
        forecast=forecast,
        cols=cols,
        final_iterations=final_iterations,
        verbose=verbose,
    )

    submission_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(submission_path, index=False)

    expected_rows = int(forecast["deviceId"].nunique() * 6)
    actual_rows = int(len(submission))
    missing = int(submission["prediction"].isna().sum())
    months = sorted(submission["month"].unique().tolist())
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
