from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np
import pandas as pd

from daily_pipeline import (
    evaluate_rolling_months,
    enrich_daily_with_sequence_features,
    feature_columns,
    find_data_dir,
    load_time_features_csv,
    top_target_correlations,
    train_and_predict_monthly,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train model on time-bucket features and create monthly submission CSV."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Directory containing task3 files. Default: auto-detect '.' or 'task3'.",
    )
    parser.add_argument(
        "--daily-path",
        type=str,
        default=None,
        help="Path to feature table CSV. Default: <data-dir>/out/daily_features.csv",
    )
    parser.add_argument(
        "--submission-path",
        type=str,
        default=None,
        help="Output path for submission CSV. Default: <data-dir>/out/submission_daily_hgb.csv",
    )
    parser.add_argument(
        "--cv-first-valid-ym",
        type=int,
        default=202501,
        help="First validation month (YYYYMM) for rolling CV on labelled data.",
    )
    parser.add_argument(
        "--save-cv",
        type=str,
        default=None,
        help="Optional path to save rolling CV table as CSV.",
    )
    parser.add_argument(
        "--model-backend",
        type=str,
        default="catboost_gpu",
        choices=["hgb", "catboost_cpu", "catboost_gpu"],
        help="Model backend for residual regression.",
    )
    parser.add_argument(
        "--model-strength",
        type=str,
        default="strong",
        choices=["fast", "strong", "heavy"],
        help="Preset controlling model size/capacity.",
    )
    parser.add_argument(
        "--catboost-iterations",
        type=int,
        default=None,
        help="Override CatBoost iterations (for long runs, e.g. 60000).",
    )
    parser.add_argument(
        "--catboost-log-every",
        type=int,
        default=100,
        help="Print CatBoost metrics every N iterations.",
    )
    parser.add_argument(
        "--corr-top-n",
        type=int,
        default=25,
        help="Print top-N feature correlations with target_x2.",
    )
    parser.add_argument(
        "--feature-mode",
        type=str,
        default="full",
        choices=["full", "no_raw_calendar"],
        help="Feature subset mode for extrapolation-sensitive runs.",
    )
    parser.add_argument(
        "--disable-sequence-features",
        action="store_true",
        help="Disable temporal lag/rolling feature enrichment.",
    )
    parser.add_argument(
        "--disable-geo-features",
        action="store_true",
        help="Disable static geolocation metadata enrichment from devices.csv.",
    )
    parser.add_argument(
        "--sample-weight-mode",
        type=str,
        default="none",
        choices=[
            "none",
            "device_month_equal",
            "device_month_equal_recent",
            "device_month_equal_warm",
            "device_month_equal_recent_warm",
        ],
        help="Optional causal sample weighting for row-level training.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable detailed progress logs.",
    )
    parser.add_argument(
        "--cv-only",
        action="store_true",
        help="Run rolling CV only and skip final fit/submission generation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    verbose = not args.quiet

    data_dir = Path(args.data_dir) if args.data_dir else find_data_dir()
    daily_path = (
        Path(args.daily_path) if args.daily_path else data_dir / "out" / "daily_features.csv"
    )
    submission_path = (
        Path(args.submission_path)
        if args.submission_path
        else data_dir / "out" / "submission_daily_hgb.csv"
    )

    if not daily_path.exists():
        raise FileNotFoundError(
            f"Missing daily features file: {daily_path}. "
            "Run build_daily_features.py first."
        )

    print(f"Loading feature data from: {daily_path.resolve()}")
    load_start = time.perf_counter()
    daily = load_time_features_csv(
        daily_path,
        labelled_only=args.cv_only,
        chunksize=250_000 if args.cv_only else None,
        verbose=verbose,
    )
    print(f"Feature table shape: {daily.shape} (loaded in {time.perf_counter() - load_start:.1f}s)")

    if args.disable_sequence_features:
        print("Sequence feature enrichment: DISABLED")
    else:
        print("Sequence feature enrichment: ENABLED")

    if args.disable_geo_features:
        print("Geolocation metadata enrichment: DISABLED")
    else:
        print("Geolocation metadata enrichment: ENABLED")

    fe_start = time.perf_counter()
    daily = enrich_daily_with_sequence_features(
        daily=daily,
        data_dir=data_dir,
        verbose=verbose,
        include_geo=not args.disable_geo_features,
        include_sequence=not args.disable_sequence_features,
    )
    print(
        f"Enrichment done. Shape now: {daily.shape} "
        f"(in {time.perf_counter() - fe_start:.1f}s)"
    )

    labelled = daily[daily["target_x2"].notna()].copy()
    labelled["ym"] = labelled["year"].astype("int32") * 100 + labelled["month"].astype("int32")
    cols = feature_columns(labelled, feature_mode=args.feature_mode)
    print(f"Model feature count: {len(cols)}")

    if args.corr_top_n > 0:
        corr_df = top_target_correlations(labelled, top_n=args.corr_top_n)
        if not corr_df.empty:
            print(f"Top {len(corr_df)} abs correlations with target_x2:")
            print(corr_df.to_string(index=False))

    print("Running rolling month validation...")
    cv_results = evaluate_rolling_months(
        labelled_df=labelled,
        cols=cols,
        first_valid_ym=args.cv_first_valid_ym,
        verbose=verbose,
        model_backend=args.model_backend,
        model_strength=args.model_strength,
        catboost_iterations=args.catboost_iterations,
        catboost_log_every=args.catboost_log_every,
        sample_weight_mode=args.sample_weight_mode,
    )
    if cv_results.empty:
        print("No CV folds generated (check cv-first-valid-ym).")
    else:
        print(cv_results.to_string(index=False))
        print(f"Average baseline monthly MAE: {cv_results['baseline_monthly_mae'].mean():.6f}")
        print(f"Average model monthly MAE:    {cv_results['model_monthly_mae'].mean():.6f}")
        print(f"Average model row MAE:        {cv_results['model_row_mae'].mean():.6f}")
        model_better = bool(
            (cv_results["model_monthly_mae"] < cv_results["baseline_monthly_mae"]).all()
        )
        print(f"Model better on all folds (monthly MAE): {model_better}")

        if args.save_cv:
            save_cv = Path(args.save_cv)
            save_cv.parent.mkdir(parents=True, exist_ok=True)
            cv_results.to_csv(save_cv, index=False)
            print(f"Saved CV results: {save_cv.resolve()}")

    final_catboost_iterations = args.catboost_iterations
    if (
        args.model_backend.startswith("catboost")
        and not cv_results.empty
        and "best_iteration" in cv_results.columns
    ):
        best_iters = cv_results["best_iteration"].dropna().astype("int32")
        if not best_iters.empty:
            median_best_iter = int(np.median(best_iters))
            final_catboost_iterations = int(max(200, round(median_best_iter * 1.1)))
            print(
                "Final CatBoost iterations from rolling CV best_iteration: "
                f"{median_best_iter} -> {final_catboost_iterations}"
            )

    if args.cv_only:
        print("CV-only mode enabled; skipping final fit and submission generation.")
        print(f"Total runtime: {time.perf_counter() - started:.1f}s")
        return

    print("Training final model and creating submission...")
    monthly_submission, labelled_all = train_and_predict_monthly(
        daily=daily,
        out_submission_path=submission_path,
        verbose=verbose,
        model_backend=args.model_backend,
        model_strength=args.model_strength,
        catboost_iterations=final_catboost_iterations,
        catboost_log_every=args.catboost_log_every,
        sample_weight_mode=args.sample_weight_mode,
    )

    forecast_device_count = daily[daily["target_x2"].isna()]["deviceId"].nunique()
    expected_rows = int(forecast_device_count * 6)
    actual_rows = int(len(monthly_submission))
    missing = int(monthly_submission["prediction"].isna().sum())
    months = sorted(monthly_submission["month"].unique().tolist())

    print(f"Submission path: {submission_path.resolve()}")
    print(f"Submission rows: {actual_rows}")
    print(f"Expected rows (devices * 6): {expected_rows}")
    print(f"Missing predictions: {missing}")
    print(f"Months present: {months}")

    if actual_rows != expected_rows:
        raise RuntimeError(
            f"Submission row mismatch: expected {expected_rows}, got {actual_rows}"
        )
    if missing != 0:
        raise RuntimeError(f"Submission contains {missing} missing predictions.")
    if months != [5, 6, 7, 8, 9, 10]:
        raise RuntimeError(f"Unexpected months in submission: {months}")

    print("Sanity checks passed.")
    print(
        f"Labelled rows used for training: {len(labelled_all)}; "
        f"forecast devices: {forecast_device_count}"
    )
    print(f"Total runtime: {time.perf_counter() - started:.1f}s")


if __name__ == "__main__":
    main()
