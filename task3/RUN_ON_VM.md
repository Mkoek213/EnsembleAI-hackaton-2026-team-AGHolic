# Run on VM (Athena)

## 1) Install dependencies

```bash
cd ~/task3
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -r requirements.txt
```

## 2) Build feature table (stage 1)

Daily baseline (stary wariant):

```bash
python build_daily_features.py --data-dir . --chunksize 300000 --flush-every 20
```

Sekwencyjny wariant bez dzielenia tylko na dni (rekomendowany start: `1h`):

```bash
python build_daily_features.py --data-dir . --freq 1h --chunksize 300000 --flush-every 20 --out out/time_features_1h.csv
```

Alternatywny wariant bezpośrednio pod metrykę miesięczną:

```bash
python build_monthly_features.py --data-dir . --chunksize 300000 --flush-every 20 --out out/monthly_features.csv
```

Szybki smoke test:

```bash
python build_daily_features.py --data-dir . --chunksize 300000 --flush-every 20 --max-chunks 4 --out out/daily_features_smoke.csv
```

## 3) Train + generate submission (stage 2)

Recommended search order after the metric/causality fixes:

- `train_and_submit.py` now scores rolling CV on exact device-month MAE, not bucket-level MAE.
- `--cv-only` skips the final fit and drops unlabeled forecast buckets before enrichment, so use it for screening.
- Run the heavy commands on a compute/GPU node, not on `login01`.
- `--feature-mode no_raw_calendar` is intended for extrapolation runs where raw month/day splits may overfit.
- `--sample-weight-mode device_month_equal` aligns row-level fitting more closely to the monthly submission metric.

Recommended GPU/VM ladder:

1. Build `1h` features.
2. Run April-only screens (`--cv-first-valid-ym 202504`) to prune settings cheaply.
3. Promote the best 1-2 settings to full Jan-Apr rolling CV (`--cv-first-valid-ym 202501`).
4. Only after that, run final submission fits.

April screen 1: simple hourly baseline, no temporal sequence features, geo ON:

```bash
python -u train_and_submit.py \
  --data-dir . \
  --daily-path out/time_features_1h.csv \
  --model-backend hgb \
  --model-strength fast \
  --disable-sequence-features \
  --feature-mode full \
  --sample-weight-mode none \
  --cv-first-valid-ym 202504 \
  --cv-only \
  --save-cv out/cv_screen_1h_hgb_fast_noseq_geo_full_none_apr.csv \
  --quiet | tee out/train_screen_1h_hgb_fast_noseq_geo_full_none_apr.log
```

April screen 2: same, but drop raw calendar features:

```bash
python -u train_and_submit.py \
  --data-dir . \
  --daily-path out/time_features_1h.csv \
  --model-backend hgb \
  --model-strength fast \
  --disable-sequence-features \
  --feature-mode no_raw_calendar \
  --sample-weight-mode none \
  --cv-first-valid-ym 202504 \
  --cv-only \
  --save-cv out/cv_screen_1h_hgb_fast_noseq_geo_nocal_none_apr.csv \
  --quiet | tee out/train_screen_1h_hgb_fast_noseq_geo_nocal_none_apr.log
```

April screen 3: same as screen 2, but metric-aligned weighting:

```bash
python -u train_and_submit.py \
  --data-dir . \
  --daily-path out/time_features_1h.csv \
  --model-backend hgb \
  --model-strength fast \
  --disable-sequence-features \
  --feature-mode no_raw_calendar \
  --sample-weight-mode device_month_equal \
  --cv-first-valid-ym 202504 \
  --cv-only \
  --save-cv out/cv_screen_1h_hgb_fast_noseq_geo_nocal_dme_apr.csv \
  --quiet | tee out/train_screen_1h_hgb_fast_noseq_geo_nocal_dme_apr.log
```

April screen 4: stronger CatBoost GPU check on the most extrapolation-safe setup:

```bash
python -u train_and_submit.py \
  --data-dir . \
  --daily-path out/time_features_1h.csv \
  --model-backend catboost_gpu \
  --model-strength strong \
  --catboost-iterations 6000 \
  --catboost-log-every 100 \
  --disable-sequence-features \
  --feature-mode no_raw_calendar \
  --sample-weight-mode device_month_equal \
  --cv-first-valid-ym 202504 \
  --cv-only \
  --save-cv out/cv_screen_1h_cbgpu_strong_noseq_geo_nocal_dme_apr.csv \
  --quiet | tee out/train_screen_1h_cbgpu_strong_noseq_geo_nocal_dme_apr.log
```

Full rolling CV for the best HGB variant:

```bash
python -u train_and_submit.py \
  --data-dir . \
  --daily-path out/time_features_1h.csv \
  --model-backend hgb \
  --model-strength fast \
  --disable-sequence-features \
  --feature-mode no_raw_calendar \
  --sample-weight-mode device_month_equal \
  --cv-first-valid-ym 202501 \
  --cv-only \
  --save-cv out/cv_finalist_1h_hgb_fast_noseq_geo_nocal_dme.csv \
  --quiet | tee out/train_finalist_1h_hgb_fast_noseq_geo_nocal_dme.log
```

Full rolling CV for the best CatBoost GPU variant:

```bash
python -u train_and_submit.py \
  --data-dir . \
  --daily-path out/time_features_1h.csv \
  --model-backend catboost_gpu \
  --model-strength strong \
  --catboost-iterations 12000 \
  --catboost-log-every 100 \
  --disable-sequence-features \
  --feature-mode no_raw_calendar \
  --sample-weight-mode device_month_equal \
  --cv-first-valid-ym 202501 \
  --cv-only \
  --save-cv out/cv_finalist_1h_cbgpu_strong_noseq_geo_nocal_dme.csv \
  --quiet | tee out/train_finalist_1h_cbgpu_strong_noseq_geo_nocal_dme.log
```

Challenge run: add temporal sequence features only if the no-sequence finalist is already good:

```bash
python -u train_and_submit.py \
  --data-dir . \
  --daily-path out/time_features_1h.csv \
  --model-backend catboost_gpu \
  --model-strength strong \
  --catboost-iterations 12000 \
  --catboost-log-every 100 \
  --feature-mode no_raw_calendar \
  --sample-weight-mode device_month_equal \
  --cv-first-valid-ym 202501 \
  --cv-only \
  --save-cv out/cv_challenge_1h_cbgpu_strong_seq_geo_nocal_dme.csv \
  --quiet | tee out/train_challenge_1h_cbgpu_strong_seq_geo_nocal_dme.log
```

Final submission fit for the winning hourly variant:

```bash
python -u train_and_submit.py \
  --data-dir . \
  --daily-path out/time_features_1h.csv \
  --submission-path out/submission_1h_best.csv \
  --save-cv out/cv_1h_best.csv \
  --model-backend catboost_gpu \
  --model-strength strong \
  --catboost-iterations 12000 \
  --catboost-log-every 100 \
  --disable-sequence-features \
  --feature-mode no_raw_calendar \
  --sample-weight-mode device_month_equal \
  --cv-first-valid-ym 202501 \
  --quiet | tee out/train_1h_best.log
```

Quick comparison of saved CV runs:

```bash
python - <<'PY'
from pathlib import Path
import pandas as pd

for path in sorted(Path("out").glob("cv_*.csv")):
    df = pd.read_csv(path)
    if "model_monthly_mae" in df.columns:
        baseline = df["baseline_monthly_mae"].mean()
        model = df["model_monthly_mae"].mean()
        print(f"{path.name:55s} baseline={baseline:.6f} model={model:.6f}")
PY
```

```bash
python -u train_and_submit.py --data-dir . --daily-path out/daily_features.csv --submission-path out/submission_daily_hgb.csv --save-cv out/cv_results.csv | tee out/train.log
```

GPU run (CatBoost, recommended on GPU node, daily baseline):

```bash
python -u train_and_submit.py --data-dir . --daily-path out/daily_features.csv --submission-path out/submission_daily_hgb.csv --save-cv out/cv_results.csv --model-backend catboost_gpu --model-strength strong --catboost-log-every 100 | tee out/train.log
```

GPU run (CatBoost, `1H` features, rekomendowany do lepszej generalizacji):

```bash
python -u train_and_submit.py --data-dir . --daily-path out/time_features_1h.csv --submission-path out/submission_seq_1h_gpu.csv --save-cv out/cv_results_seq_1h_gpu.csv --model-backend catboost_gpu --model-strength heavy --catboost-iterations 20000 --catboost-log-every 100 --corr-top-n 30 | tee out/train_seq_1h_gpu.log
```

Nowa architektura: ensemble CatBoost + HGB (waga blendu dobierana w rolling CV):

```bash
python -u train_and_submit_ensemble.py --data-dir . --daily-path out/time_features_1h.csv --submission-path out/submission_ensemble_1h.csv --save-cv out/cv_results_ensemble_1h.csv --catboost-backend catboost_gpu --catboost-strength strong --catboost-iterations 12000 --catboost-log-every 100 --hgb-strength strong --blend-grid-step 0.05 --corr-top-n 20 | tee out/train_ensemble_1h.log
```

Nowe podejście: model bezpośrednio miesięczny (device-month, residual over device baseline):

```bash
python -u train_monthly_direct.py --data-dir . --monthly-path out/monthly_features.csv --submission-path out/submission_monthly_direct.csv --save-cv out/cv_results_monthly_direct.csv --backend catboost_gpu --iterations 4000 --early-stopping-rounds 300 --learning-rate 0.03 --depth 8 --l2-leaf-reg 8 --random-strength 1.5 --subsample 0.8 --log-every 50 --corr-top-n 20 | tee out/train_monthly_direct.log
```

Uwaga:
`train_monthly_direct.py` musi wykluczać `x2_sum` i `x2_count` z feature setu, bo to leakage targetu dla labelled months.

Raw 5-minute pipeline (bez agregacji do dni/godzin; miesięczna średnia dopiero na końcu):

```bash
python -u train_and_submit_raw.py --data-dir . --submission-path out/submission_raw_seq_gpu.csv --save-cv out/cv_results_raw_seq_gpu.csv --model-backend catboost_gpu --model-strength strong --catboost-iterations 12000 --catboost-log-every 100 --train-sample-frac 1.0 | tee out/train_raw_seq_gpu.log
```

Wariant z adaptacją sezonową (ważenie cieplejszych i nowszych próbek, domyślnie WŁ):

```bash
python -u train_and_submit_raw.py --data-dir . --submission-path out/submission_raw_seq_gpu_weighted.csv --save-cv out/cv_results_raw_seq_gpu_weighted.csv --model-backend catboost_gpu --model-strength strong --catboost-iterations 12000 --catboost-log-every 100 --train-sample-frac 1.0 --warm-threshold 0.55 | tee out/train_raw_seq_gpu_weighted.log
```

Bardzo ciężki GPU run (tylko jeśli chcesz bardzo długi trening):

```bash
python -u train_and_submit.py --data-dir . --daily-path out/time_features_1h.csv --submission-path out/submission_seq_1h_gpu_xl.csv --save-cv out/cv_results_seq_1h_gpu_xl.csv --model-backend catboost_gpu --model-strength heavy --catboost-iterations 60000 --catboost-log-every 200 --corr-top-n 30 | tee out/train_seq_1h_gpu_xl.log
```

CPU fallback:

```bash
python -u train_and_submit.py --data-dir . --daily-path out/daily_features.csv --submission-path out/submission_daily_hgb.csv --save-cv out/cv_results.csv --model-backend hgb | tee out/train.log
```

Expected output:

- plik CV z argumentu `--save-cv`
- plik submission z argumentu `--submission-path`

Submission CSV columns:

- `deviceId`
- `year`
- `month`
- `prediction`
