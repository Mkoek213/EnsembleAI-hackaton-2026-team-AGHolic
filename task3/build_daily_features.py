from __future__ import annotations

import argparse
from pathlib import Path

from daily_pipeline import build_time_features, find_data_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build time-bucket feature table from task3/data.csv (chunked pandas)."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Directory containing data.csv. Default: auto-detect '.' or 'task3'.",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=300_000,
        help="CSV chunk size for pandas reader.",
    )
    parser.add_argument(
        "--flush-every",
        type=int,
        default=20,
        help="Collapse intermediate partial frames every N chunks.",
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=None,
        help="Optional early-stop for smoke tests.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output CSV path. Default: <data-dir>/out/daily_features.csv",
    )
    parser.add_argument(
        "--freq",
        type=str,
        default="D",
        help="Time bucket frequency for aggregation (e.g. D, 6H, 3H, 1H, 30min).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data_dir = Path(args.data_dir) if args.data_dir else find_data_dir()
    csv_path = data_dir / "data.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing input file: {csv_path}")

    out_path = Path(args.out) if args.out else data_dir / "out" / "daily_features.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Input CSV: {csv_path.resolve()}")
    print(
        f"Settings: chunksize={args.chunksize}, flush_every={args.flush_every}, "
        f"max_chunks={args.max_chunks}, freq={args.freq}"
    )
    print("Building time-bucket features...")

    daily = build_time_features(
        csv_path=csv_path,
        chunksize=args.chunksize,
        flush_every=args.flush_every,
        max_chunks=args.max_chunks,
        bucket_freq=args.freq,
    )
    daily.to_csv(out_path, index=False)

    print(f"Saved daily features: {out_path.resolve()}")
    print(f"Rows: {len(daily)}, Columns: {len(daily.columns)}")

    summary = (
        daily.groupby(["period", "year", "month"], dropna=False)
        .agg(device_count=("deviceId", "nunique"), row_count=("deviceId", "size"))
        .reset_index()
        .sort_values(["year", "month"])
    )
    print("Period summary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
