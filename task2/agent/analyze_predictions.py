from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path

from .app import load_datapoints
from .evaluate import chrf, recover_reference

FILE_SEP = "<|file_sep|>"
NOISE_PATH_TOKENS = {
    "license",
    "licenses",
    "licence",
    "licences",
    "copying",
    "notice",
    "copyright",
    "readme",
    "changelog",
    "changes",
}
NOISE_LINE_MARKERS = (
    "started by",
    "maintained by",
    "currently maintained by",
    "maintained for",
    "copyright",
    "licensed under",
    "free software foundation",
    "gnu general public license",
    "documentation:",
    "github:",
    "pypi:",
    "homepage:",
    "read the docs",
    "python markdown",
    "all rights reserved",
    "license :: osi approved",
    "license agreement",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze task2 prediction JSONL structure and quality proxies")
    parser.add_argument("--stage", required=True, help="Dataset stage, for example test or public")
    parser.add_argument("--lang", default="python", choices=["python", "kotlin"], help="Language split")
    parser.add_argument(
        "--predictions-file",
        type=Path,
        required=True,
        help="Predictions JSONL to analyze",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON output path",
    )
    parser.add_argument("--limit", type=int, help="Analyze only the first N datapoints")
    parser.add_argument(
        "--ids",
        nargs="*",
        help="Optional list of datapoint ids to keep",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent.parent
    data_dir = base_dir / "data"
    dataset_path = data_dir / f"{args.lang}-{args.stage}.jsonl"
    repos_dir = data_dir / f"repositories-{args.lang}-{args.stage}"

    if not dataset_path.exists():
        raise FileNotFoundError(dataset_path)
    if not repos_dir.exists():
        raise FileNotFoundError(repos_dir)
    if not args.predictions_file.exists():
        raise FileNotFoundError(args.predictions_file)

    datapoints = load_datapoints(dataset_path)
    predictions = load_predictions(args.predictions_file)

    if args.ids:
        keep_ids = set(args.ids)
        filtered_datapoints = []
        filtered_predictions = []
        for datapoint, prediction in zip(datapoints, predictions, strict=False):
            datapoint_id = datapoint.id or datapoint.path
            if datapoint_id in keep_ids:
                filtered_datapoints.append(datapoint)
                filtered_predictions.append(prediction)
        datapoints = filtered_datapoints
        predictions = filtered_predictions

    if args.limit is not None:
        datapoints = datapoints[: args.limit]
        predictions = predictions[: args.limit]

    if len(datapoints) != len(predictions):
        raise ValueError(
            f"Prediction count mismatch: {len(predictions)} predictions vs {len(datapoints)} datapoints"
        )

    rows: list[dict[str, object]] = []
    for datapoint, prediction in zip(datapoints, predictions, strict=True):
        repo_root = repos_dir / f"{datapoint.repo.replace('/', '__')}-{datapoint.revision}"
        context = str(prediction.get("context", ""))
        target_path = datapoint.path
        blocks = parse_context_blocks(context)
        normalized_paths = [str(block["path"]) for block in blocks if block["path"]]
        exactness = analyze_block_exactness(repo_root=repo_root, blocks=blocks)
        suspicious_hits = classify_suspicious_paths(normalized_paths, target_path=target_path)
        noise_hits = detect_noise_hits(blocks, target_path=target_path)
        recovered_reference = recover_reference(datapoint, repo_root)
        target_index = next((idx for idx, path in enumerate(normalized_paths) if path == target_path), None)
        row = {
            "id": datapoint.id or datapoint.path,
            "target_path": target_path,
            "context_length": len(context),
            "block_count": len(blocks),
            "unique_paths": len(dict.fromkeys(normalized_paths)),
            "target_present": target_index is not None,
            "target_first": target_index == 0,
            "target_in_first_2": target_index is not None and target_index < 2,
            "target_last": target_index is not None and target_index == len(normalized_paths) - 1,
            "target_index": target_index,
            "empty_context": not context,
            "empty_blocks": sum(1 for block in blocks if not str(block["content"]).strip()),
            "non_exact_block_count": exactness["non_exact_block_count"],
            "non_exact_block_refs": exactness["non_exact_block_refs"],
            "missing_file_block_refs": exactness["missing_file_block_refs"],
            "noise_hits": noise_hits,
            "noise_hit_count": len(noise_hits),
            "suspicious_paths": suspicious_hits,
            "reference_recovered": recovered_reference is not None,
            "reference_length": len(recovered_reference or ""),
            "reference_in_context": bool(recovered_reference and recovered_reference in context),
            "context_reference_chrf_proxy": (
                chrf(context, recovered_reference) if recovered_reference is not None else None
            ),
            "paths": normalized_paths,
        }
        rows.append(row)

    summary = summarize(rows)
    payload = {
        "dataset_rows": len(datapoints),
        "prediction_rows": len(predictions),
        "analyzed_rows": len(rows),
        "summary": summary,
        "rows": rows,
    }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.write_text(rendered + "\n", encoding="utf-8")


def load_predictions(path: Path) -> list[dict[str, object]]:
    predictions: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            predictions.append(json.loads(line))
    return predictions


def parse_context_blocks(context: str) -> list[dict[str, str]]:
    if not context:
        return []
    if FILE_SEP not in context:
        return [{"path": "", "content": context}]

    blocks: list[dict[str, str]] = []
    for raw_block in context.split(FILE_SEP):
        if not raw_block:
            continue
        raw_block = raw_block.lstrip("\n")
        path, sep, content = raw_block.partition("\n")
        if not sep:
            content = ""
        blocks.append({"path": path.strip(), "content": content})
    return blocks


def analyze_block_exactness(
    *,
    repo_root: Path,
    blocks: list[dict[str, str]],
) -> dict[str, object]:
    non_exact_block_refs: list[str] = []
    missing_file_block_refs: list[str] = []
    file_cache: dict[str, str | None] = {}

    for index, block in enumerate(blocks, start=1):
        path = str(block["path"])
        content = str(block["content"])
        if not path:
            non_exact_block_refs.append(f"#{index}:<no-path>")
            continue

        if path not in file_cache:
            candidate = repo_root / path
            if candidate.exists() and candidate.is_file():
                file_cache[path] = candidate.read_text(encoding="utf-8", errors="replace")
            else:
                file_cache[path] = None

        file_text = file_cache[path]
        if file_text is None:
            missing_file_block_refs.append(f"{path}#{index}")
            continue
        if content and content not in file_text:
            non_exact_block_refs.append(f"{path}#{index}")

    return {
        "non_exact_block_count": len(non_exact_block_refs),
        "non_exact_block_refs": non_exact_block_refs,
        "missing_file_block_refs": missing_file_block_refs,
    }


def classify_suspicious_paths(paths: list[str], *, target_path: str) -> list[str]:
    target_is_test = is_test_path(target_path)
    hits: list[str] = []
    for path in paths:
        lowered = path.lower().replace("\\", "/")
        name = Path(lowered).name
        stem = Path(lowered).stem
        if lowered == "setup.py":
            hits.append(f"setup:{path}")
        if name in {"pyproject.toml", "setup.cfg", "tox.ini"}:
            hits.append(f"project_meta:{path}")
        if stem in NOISE_PATH_TOKENS:
            hits.append(f"noise_path:{path}")
        if not target_is_test and is_test_path(lowered):
            hits.append(f"broad_test:{path}")
    return sorted(dict.fromkeys(hits))


def detect_noise_hits(blocks: list[dict[str, str]], *, target_path: str) -> list[str]:
    hits: list[str] = []
    for block in blocks:
        path = str(block["path"])
        content = str(block["content"])
        lowered = path.lower().replace("\\", "/")
        if path != target_path:
            stem = Path(lowered).stem
            if stem in NOISE_PATH_TOKENS:
                hits.append(f"path:{path}")
        for line in content.splitlines():
            normalized = line.strip().lower()
            if not normalized:
                continue
            if "http://" in normalized or "https://" in normalized:
                hits.append(f"url:{path}")
                break
            if any(marker in normalized for marker in NOISE_LINE_MARKERS):
                hits.append(f"noise_line:{path}")
                break
    return sorted(dict.fromkeys(hits))


def is_test_path(path: str) -> bool:
    normalized = path.lower().replace("\\", "/")
    parts = [part for part in normalized.split("/") if part]
    name = Path(normalized).name
    return any(part in {"test", "tests", "testing"} for part in parts) or name.startswith("test_")


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    context_lengths = [int(row["context_length"]) for row in rows]
    block_counts = [int(row["block_count"]) for row in rows]
    proxy_scores = [
        float(row["context_reference_chrf_proxy"])
        for row in rows
        if isinstance(row["context_reference_chrf_proxy"], float)
    ]
    suspicious_counter = Counter(
        hit
        for row in rows
        for hit in row["suspicious_paths"]
        if isinstance(hit, str)
    )

    return {
        "empty_context_ids": collect_ids(rows, lambda row: bool(row["empty_context"])),
        "context_length": summarize_numbers(context_lengths),
        "block_count": summarize_numbers(block_counts),
        "target_present": count_metric(rows, "target_present"),
        "target_first": count_metric(rows, "target_first"),
        "target_in_first_2": count_metric(rows, "target_in_first_2"),
        "target_last": count_metric(rows, "target_last"),
        "non_exact_rows": collect_ids(rows, lambda row: int(row["non_exact_block_count"]) > 0),
        "non_exact_blocks_total": sum(int(row["non_exact_block_count"]) for row in rows),
        "missing_file_rows": collect_ids(rows, lambda row: bool(row["missing_file_block_refs"])),
        "noise_rows": collect_ids(rows, lambda row: int(row["noise_hit_count"]) > 0),
        "suspicious_path_rows": collect_ids(rows, lambda row: bool(row["suspicious_paths"])),
        "top_suspicious_paths": suspicious_counter.most_common(20),
        "reference_recovered": count_metric(rows, "reference_recovered"),
        "reference_in_context": count_metric(rows, "reference_in_context"),
        "avg_context_reference_chrf_proxy": (
            sum(proxy_scores) / len(proxy_scores) if proxy_scores else None
        ),
    }


def summarize_numbers(values: list[int]) -> dict[str, float | int | None]:
    if not values:
        return {"min": None, "max": None, "avg": None, "median": None}
    return {
        "min": min(values),
        "max": max(values),
        "avg": round(statistics.fmean(values), 1),
        "median": statistics.median(values),
    }


def count_metric(rows: list[dict[str, object]], key: str) -> dict[str, float | int]:
    total = len(rows)
    hits = sum(1 for row in rows if bool(row[key]))
    return {
        "count": hits,
        "rate": hits / total if total else 0.0,
    }


def collect_ids(rows: list[dict[str, object]], predicate: object) -> list[str]:
    collected: list[str] = []
    for row in rows:
        if predicate(row):
            collected.append(str(row["id"]))
    return collected


if __name__ == "__main__":
    main()
