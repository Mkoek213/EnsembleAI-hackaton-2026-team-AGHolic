from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from daily_pipeline import (
    enrich_daily_with_sequence_features,
    feature_columns,
    find_data_dir,
    load_time_features_csv,
    make_model,
    prepare_X,
    top_target_correlations,
)


def _fmt_duration(seconds: float) -> str:
    seconds = int(max(seconds, 0))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ensemble architecture: CatBoost + HGB residual models with CV blend-weight search."
    )
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--daily-path", type=str, default=None)
    parser.add_argument("--submission-path", type=str, default=None)
    parser.add_argument("--save-cv", type=str, default=None)
    parser.add_argument("--cv-first-valid-ym", type=int, default=202501)
    parser.add_argument("--corr-top-n", type=int, default=25)
    parser.add_argument(
        "--feature-mode",
        type=str,
        default="full",
        choices=["full", "no_raw_calendar"],
    )
    parser.add_argument("--disable-sequence-features", action="store_true")
    parser.add_argument("--disable-geo-features", action="store_true")

    parser.add_argument(
        "--catboost-backend",
        type=str,
        default="catboost_gpu",
        choices=["catboost_gpu", "catboost_cpu"],
    )
    parser.add_argument(
        "--catboost-strength",
        type=str,
        default="strong",
        choices=["fast", "strong", "heavy"],
    )
    parser.add_argument("--catboost-iterations", type=int, default=12000)
    parser.add_argument("--catboost-log-every", type=int, default=100)
    parser.add_argument("--catboost-early-stop", type=int, default=800)

    parser.add_argument(
        "--hgb-strength",
        type=str,
        default="strong",
        choices=["fast", "strong", "heavy"],
    )
    parser.add_argument(
        "--blend-grid-step",
        type=float,
        default=0.05,
        help="Grid step for catboost weight in [0,1], e.g. 0.05.",
    )
    parser.add_argument(
        "--final-blend-weight",
        type=float,
        default=None,
        help="Optional fixed catboost weight. Default: weighted average from CV.",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def _train_predict_single(
    train_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    cols: list[str],
    backend: str,
    strength: str,
    catboost_iterations: int,
    catboost_log_every: int,
    catboost_early_stop: int,
    verbose: bool,
    eval_df: pd.DataFrame | None = None,
) -> tuple[np.ndarray, float]:
    X_train = prepare_X(train_df, cols)
    X_pred = prepare_X(pred_df, cols)

    device_mean = train_df.groupby("deviceId")["target_x2"].mean()
    global_mean = float(train_df["target_x2"].mean())
    train_base = train_df["deviceId"].map(device_mean).fillna(global_mean).astype("float32")
    pred_base = pred_df["deviceId"].map(device_mean).fillna(global_mean).astype("float32")
    y_train = (train_df["target_x2"].astype("float32") - train_base).astype("float32")

    model = make_model(
        model_backend=backend,
        model_strength=strength,
        catboost_iterations=catboost_iterations,
        catboost_log_every=catboost_log_every,
        live_log=verbose,
    )

    fit_kwargs: dict = {}
    if backend.startswith("catboost") and eval_df is not None and len(eval_df) > 0:
        X_eval = prepare_X(eval_df, cols)
        eval_base = eval_df["deviceId"].map(device_mean).fillna(global_mean).astype("float32")
        y_eval = (eval_df["target_x2"].astype("float32") - eval_base).astype("float32")
        fit_kwargs = {
            "eval_set": (X_eval, y_eval),
            "use_best_model": True,
            "early_stopping_rounds": max(100, int(catboost_early_stop)),
        }

    model.fit(X_train, y_train, **fit_kwargs)
    best_iter = np.nan
    if backend.startswith("catboost"):
        try:
            best_iter_raw = int(model.get_best_iteration())
            if best_iter_raw >= 0:
                best_iter = best_iter_raw + 1
        except Exception:
            pass

    pred_res = model.predict(X_pred).astype("float32")
    pred = np.clip(pred_base.to_numpy(dtype=np.float32) + pred_res, 0.0, None)
    return pred, best_iter


def _build_month_grid(device_ids: np.ndarray) -> pd.DataFrame:
    months = pd.DataFrame({"year": [2025] * 6, "month": [5, 6, 7, 8, 9, 10]})
    devices = pd.DataFrame({"deviceId": device_ids})
    return devices.assign(_k=1).merge(months.assign(_k=1), on="_k").drop(columns="_k")


def main() -> None:
    args = parse_args()
    verbose = not args.quiet
    started = time.perf_counter()

    data_dir = Path(args.data_dir) if args.data_dir else find_data_dir()
    daily_path = (
        Path(args.daily_path) if args.daily_path else data_dir / "out" / "daily_features.csv"
    )
    submission_path = (
        Path(args.submission_path)
        if args.submission_path
        else data_dir / "out" / "submission_ensemble_cb_hgb.csv"
    )

    if not daily_path.exists():
        raise FileNotFoundError(f"Missing feature file: {daily_path}")

    print(f"Loading feature data from: {daily_path.resolve()}")
    t0 = time.perf_counter()
    daily = load_time_features_csv(daily_path)
    print(f"Feature shape: {daily.shape} (loaded in {time.perf_counter() - t0:.1f}s)")

    if args.disable_sequence_features:
        print("Sequence feature enrichment: DISABLED")
    else:
        print("Sequence feature enrichment: ENABLED")

    if args.disable_geo_features:
        print("Geolocation metadata enrichment: DISABLED")
    else:
        print("Geolocation metadata enrichment: ENABLED")

    t1 = time.perf_counter()
    daily = enrich_daily_with_sequence_features(
        daily=daily,
        data_dir=data_dir,
        verbose=verbose,
        include_geo=not args.disable_geo_features,
        include_sequence=not args.disable_sequence_features,
    )
    print(f"Enrichment done in {time.perf_counter() - t1:.1f}s. Shape: {daily.shape}")

    labelled = daily[daily["target_x2"].notna()].copy()
    forecast = daily[daily["target_x2"].isna()].copy()
    labelled["ym"] = labelled["year"].astype("int32") * 100 + labelled["month"].astype("int32")
    forecast["ym"] = forecast["year"].astype("int32") * 100 + forecast["month"].astype("int32")
    cols = feature_columns(labelled, feature_mode=args.feature_mode)
    print(f"Model feature count: {len(cols)}")

    if args.corr_top_n > 0:
        corr_df = top_target_correlations(labelled, top_n=args.corr_top_n)
        if not corr_df.empty:
            print(f"Top {len(corr_df)} abs correlations with target_x2:")
            print(corr_df.to_string(index=False))

    # Rolling CV and blend-weight search
    valid_months = sorted([m for m in labelled["ym"].unique() if m >= args.cv_first_valid_ym])
    rows = []
    best_iters: list[int] = []
    tcv = time.perf_counter()
    print(
        f"[CV ENS] folds={len(valid_months)} cb={args.catboost_backend}/{args.catboost_strength} "
        f"hgb={args.hgb_strength} blend_step={args.blend_grid_step}"
    )

    for i, valid_ym in enumerate(valid_months, start=1):
        train_df = labelled[labelled["ym"] < valid_ym].copy()
        valid_df = labelled[labelled["ym"] == valid_ym].copy()
        if train_df.empty or valid_df.empty:
            continue
        fold_start = time.perf_counter()
        print(
            f"[CV ENS {i}/{len(valid_months)}] valid_ym={valid_ym} "
            f"train_rows={len(train_df)} valid_rows={len(valid_df)}"
        )

        pred_cb, best_iter = _train_predict_single(
            train_df=train_df,
            pred_df=valid_df,
            cols=cols,
            backend=args.catboost_backend,
            strength=args.catboost_strength,
            catboost_iterations=args.catboost_iterations,
            catboost_log_every=args.catboost_log_every,
            catboost_early_stop=args.catboost_early_stop,
            verbose=verbose,
            eval_df=valid_df,
        )
        pred_hgb, _ = _train_predict_single(
            train_df=train_df,
            pred_df=valid_df,
            cols=cols,
            backend="hgb",
            strength=args.hgb_strength,
            catboost_iterations=args.catboost_iterations,
            catboost_log_every=args.catboost_log_every,
            catboost_early_stop=args.catboost_early_stop,
            verbose=verbose,
            eval_df=None,
        )

        monthly_eval = valid_df[["deviceId", "year", "month", "target_x2"]].copy()
        monthly_eval["pred_cb"] = np.clip(pred_cb, 0.0, None)
        monthly_eval["pred_hgb"] = np.clip(pred_hgb, 0.0, None)
        monthly_eval = (
            monthly_eval.groupby(["deviceId", "year", "month"], as_index=False)
            .agg(
                target=("target_x2", "mean"),
                pred_cb=("pred_cb", "mean"),
                pred_hgb=("pred_hgb", "mean"),
            )
            .sort_values(["deviceId", "year", "month"])
            .reset_index(drop=True)
        )

        mae_cb = mean_absolute_error(monthly_eval["target"], monthly_eval["pred_cb"])
        mae_hgb = mean_absolute_error(monthly_eval["target"], monthly_eval["pred_hgb"])

        best_w = 1.0
        best_mae = mae_cb
        step = max(0.01, float(args.blend_grid_step))
        for w in np.arange(0.0, 1.0 + step, step):
            pred_blend = np.clip(
                w * monthly_eval["pred_cb"] + (1.0 - w) * monthly_eval["pred_hgb"],
                0.0,
                None,
            )
            mae_blend = mean_absolute_error(monthly_eval["target"], pred_blend)
            if mae_blend < best_mae:
                best_mae = mae_blend
                best_w = float(w)

        if not np.isnan(best_iter):
            best_iters.append(int(best_iter))

        rows.append(
            {
                "valid_ym": int(valid_ym),
                "train_rows": int(len(train_df)),
                "valid_rows": int(len(valid_df)),
                "monthly_rows": int(len(monthly_eval)),
                "cb_mae": float(mae_cb),
                "hgb_mae": float(mae_hgb),
                "blend_mae": float(best_mae),
                "best_cb_weight": float(best_w),
                "best_iteration": best_iter,
            }
        )
        elapsed = time.perf_counter() - tcv
        avg = elapsed / i
        eta = avg * (len(valid_months) - i)
        print(
            f"[CV ENS {i}/{len(valid_months)}] cb_mae={mae_cb:.6f} hgb_mae={mae_hgb:.6f} "
            f"blend_mae={best_mae:.6f} best_w={best_w:.2f} "
            f"fold_time={_fmt_duration(time.perf_counter()-fold_start)} eta={_fmt_duration(eta)}"
        )

    cv_df = pd.DataFrame(rows)
    if cv_df.empty:
        raise RuntimeError("No CV folds were produced.")

    print(cv_df.to_string(index=False))
    print(f"Avg cb MAE:    {cv_df['cb_mae'].mean():.6f}")
    print(f"Avg hgb MAE:   {cv_df['hgb_mae'].mean():.6f}")
    print(f"Avg blend MAE: {cv_df['blend_mae'].mean():.6f}")

    if args.final_blend_weight is None:
        w_final = float(np.average(cv_df["best_cb_weight"], weights=cv_df["monthly_rows"]))
        print(f"Final cb blend weight from CV (weighted avg): {w_final:.4f}")
    else:
        w_final = float(args.final_blend_weight)
        print(f"Final cb blend weight (manual): {w_final:.4f}")

    final_cb_iterations = int(args.catboost_iterations)
    if best_iters:
        median_best_iter = int(np.median(best_iters))
        final_cb_iterations = int(max(200, round(median_best_iter * 1.1)))
        print(
            "Final CatBoost iterations from rolling CV best_iteration: "
            f"{median_best_iter} -> {final_cb_iterations}"
        )

    if args.save_cv:
        save_cv = Path(args.save_cv)
        save_cv.parent.mkdir(parents=True, exist_ok=True)
        cv_df.to_csv(save_cv, index=False)
        print(f"Saved CV results: {save_cv.resolve()}")

    # Final train on all labelled and predict forecast
    print("Training final CB + HGB and creating submission...")
    pred_cb_final, _ = _train_predict_single(
        train_df=labelled,
        pred_df=forecast,
        cols=cols,
        backend=args.catboost_backend,
        strength=args.catboost_strength,
        catboost_iterations=final_cb_iterations,
        catboost_log_every=args.catboost_log_every,
        catboost_early_stop=args.catboost_early_stop,
        verbose=verbose,
        eval_df=None,
    )
    pred_hgb_final, _ = _train_predict_single(
        train_df=labelled,
        pred_df=forecast,
        cols=cols,
        backend="hgb",
        strength=args.hgb_strength,
        catboost_iterations=args.catboost_iterations,
        catboost_log_every=args.catboost_log_every,
        catboost_early_stop=args.catboost_early_stop,
        verbose=verbose,
        eval_df=None,
    )

    pred_final = np.clip(w_final * pred_cb_final + (1.0 - w_final) * pred_hgb_final, 0.0, None)
    pred_rows = forecast[["deviceId", "year", "month"]].copy()
    pred_rows["prediction_row"] = pred_final.astype("float32")
    monthly = (
        pred_rows.groupby(["deviceId", "year", "month"], as_index=False)["prediction_row"]
        .mean()
        .rename(columns={"prediction_row": "prediction"})
    )

    target_grid = _build_month_grid(forecast["deviceId"].astype("string").unique())
    monthly["deviceId"] = monthly["deviceId"].astype("string")
    submission = target_grid.merge(monthly, on=["deviceId", "year", "month"], how="left")

    # Fallback for any missing rows after aggregation (should be rare).
    global_mean = float(labelled["target_x2"].mean())
    device_mean = labelled.groupby("deviceId")["target_x2"].mean()
    submission["prediction"] = submission["prediction"].fillna(
        submission["deviceId"].map(device_mean).fillna(global_mean)
    )
    submission["prediction"] = submission["prediction"].clip(lower=0.0)
    submission = submission.sort_values(["deviceId", "year", "month"]).reset_index(drop=True)

    submission_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(submission_path, index=False)

    expected_rows = int(forecast["deviceId"].nunique() * 6)
    actual_rows = int(len(submission))
    missing = int(submission["prediction"].isna().sum())
    months = sorted(submission["month"].unique().tolist())
    print(f"Submission path: {submission_path.resolve()}")
    print(f"Submission rows: {actual_rows} (expected {expected_rows})")
    print(f"Missing predictions: {missing}")
    print(f"Months present: {months}")
    if actual_rows != expected_rows:
        raise RuntimeError(f"Row mismatch: expected {expected_rows}, got {actual_rows}")
    if missing != 0:
        raise RuntimeError(f"Submission contains {missing} missing predictions")
    if months != [5, 6, 7, 8, 9, 10]:
        raise RuntimeError(f"Unexpected months in submission: {months}")
    print("Sanity checks passed.")
    print(f"Total runtime: {time.perf_counter() - started:.1f}s")


if __name__ == "__main__":
    main()
