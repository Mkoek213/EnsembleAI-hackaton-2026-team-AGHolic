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

Sekwencyjny wariant bez dzielenia tylko na dni (rekomendowany start: `1H`):

```bash
python build_daily_features.py --data-dir . --freq 1H --chunksize 300000 --flush-every 20 --out out/time_features_1h.csv
```

Szybki smoke test:

```bash
python build_daily_features.py --data-dir . --chunksize 300000 --flush-every 20 --max-chunks 4 --out out/daily_features_smoke.csv
```

## 3) Train + generate submission (stage 2)

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
