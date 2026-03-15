from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from .app import load_datapoints
from .models import TaskDatapoint


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local proxy evaluation for task2 context predictions")
    parser.add_argument("--stage", required=True, help="Dataset stage, for example start, practice, public")
    parser.add_argument("--lang", default="python", choices=["python", "kotlin"], help="Language split")
    parser.add_argument(
        "--predictions-file",
        type=Path,
        help="Predictions JSONL to evaluate. Defaults to predictions/{lang}-{stage}-agent.jsonl",
    )
    parser.add_argument("--limit", type=int, help="Evaluate only the first N datapoints")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent.parent
    data_dir = base_dir / "data"
    predictions_dir = base_dir / "predictions"

    dataset_path = data_dir / f"{args.lang}-{args.stage}.jsonl"
    repos_dir = data_dir / f"repositories-{args.lang}-{args.stage}"
    predictions_path = args.predictions_file or predictions_dir / f"{args.lang}-{args.stage}-agent.jsonl"

    if not dataset_path.exists():
        raise FileNotFoundError(dataset_path)
    if not repos_dir.exists():
        raise FileNotFoundError(
            f"{repos_dir} does not exist. Run prepare_data.sh for this stage first."
        )
    if not predictions_path.exists():
        raise FileNotFoundError(predictions_path)

    datapoints = load_datapoints(dataset_path)
    predictions = load_predictions(predictions_path)
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
        effective_prefix, effective_suffix = resolve_prediction_boundaries(datapoint, prediction)
        recovered = recover_reference(
            datapoint,
            repo_root,
            prefix=effective_prefix,
            suffix=effective_suffix,
        )
        context = prediction.get("context", "")
        row = {
            "id": datapoint.id or datapoint.path,
            "target_path": datapoint.path,
            "context_length": len(context),
            "target_path_mentioned": datapoint.path in context,
            "reference_recovered": recovered is not None,
            "reference_length": len(recovered or ""),
            "reference_in_context": bool(recovered and recovered in context),
            "context_reference_chrf_proxy": chrf(context, recovered) if recovered else None,
        }
        rows.append(row)

    summary = summarize(rows)
    print(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2))
    print(
        "Uwaga: to jest tylko lokalna ewaluacja proxy. "
        "Oficjalnego score nie da sie odtworzyc 1:1 bez modeli i pipeline organizatora."
    )


def load_predictions(path: Path) -> list[dict[str, object]]:
    predictions: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            predictions.append(json.loads(line))
    return predictions


def resolve_prediction_boundaries(
    datapoint: TaskDatapoint,
    prediction: dict[str, object],
) -> tuple[str, str]:
    prefix = prediction.get("prefix")
    suffix = prediction.get("suffix")
    effective_prefix = prefix if isinstance(prefix, str) else datapoint.prefix
    effective_suffix = suffix if isinstance(suffix, str) else datapoint.suffix
    return effective_prefix, effective_suffix


def recover_reference(
    datapoint: TaskDatapoint,
    repo_root: Path,
    *,
    prefix: str | None = None,
    suffix: str | None = None,
) -> str | None:
    effective_prefix = datapoint.prefix if prefix is None else prefix
    effective_suffix = datapoint.suffix if suffix is None else suffix
    candidate_paths = []
    target_path = repo_root / datapoint.path
    if target_path.exists():
        candidate_paths.append(target_path)
    for relative_path in datapoint.modified:
        candidate = repo_root / relative_path
        if candidate.exists() and candidate not in candidate_paths:
            candidate_paths.append(candidate)

    for candidate in candidate_paths:
        full_text = candidate.read_text(encoding="utf-8", errors="replace")
        reference = extract_reference_from_text(full_text, effective_prefix, effective_suffix)
        if reference is not None:
            return reference
    return None


def extract_reference_from_text(full_text: str, prefix: str, suffix: str) -> str | None:
    normalized_full = normalize_newlines(full_text)
    normalized_prefix = normalize_newlines(prefix)
    normalized_suffix = normalize_newlines(suffix)

    start_index = normalized_full.find(normalized_prefix)
    if start_index != -1:
        content_start = start_index + len(normalized_prefix)
        if normalized_suffix:
            content_end = normalized_full.find(normalized_suffix, content_start)
            if content_end != -1:
                return normalized_full[content_start:content_end]
            return None
        return normalized_full[content_start:]

    if normalized_prefix.startswith(normalized_full) and not normalized_suffix:
        return None

    return None


def chrf(hypothesis: str, reference: str, max_order: int = 6, beta: float = 2.0) -> float:
    if not hypothesis and not reference:
        return 1.0
    if not hypothesis or not reference:
        return 0.0

    precision_scores: list[float] = []
    recall_scores: list[float] = []
    for order in range(1, max_order + 1):
        hyp_counts = ngram_counts(hypothesis, order)
        ref_counts = ngram_counts(reference, order)
        hyp_total = sum(hyp_counts.values())
        ref_total = sum(ref_counts.values())
        overlap = sum((hyp_counts & ref_counts).values())

        precision_scores.append(overlap / hyp_total if hyp_total else 0.0)
        recall_scores.append(overlap / ref_total if ref_total else 0.0)

    precision = sum(precision_scores) / max_order
    recall = sum(recall_scores) / max_order
    if precision == 0.0 and recall == 0.0:
        return 0.0
    beta_sq = beta * beta
    return (1 + beta_sq) * precision * recall / (recall + beta_sq * precision)


def ngram_counts(text: str, order: int) -> Counter[str]:
    if len(text) < order:
        return Counter()
    return Counter(text[index : index + order] for index in range(len(text) - order + 1))


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    total = len(rows)
    recovered = [row for row in rows if row["reference_recovered"]]
    target_hits = [row for row in rows if row["target_path_mentioned"]]
    ref_hits = [row for row in rows if row["reference_in_context"]]
    proxy_scores = [
        row["context_reference_chrf_proxy"]
        for row in rows
        if isinstance(row["context_reference_chrf_proxy"], float)
    ]
    return {
        "total": total,
        "reference_recovered": len(recovered),
        "reference_recovered_rate": len(recovered) / total if total else 0.0,
        "target_path_mentioned": len(target_hits),
        "target_path_mentioned_rate": len(target_hits) / total if total else 0.0,
        "reference_in_context": len(ref_hits),
        "reference_in_context_rate": len(ref_hits) / total if total else 0.0,
        "avg_context_reference_chrf_proxy": sum(proxy_scores) / len(proxy_scores) if proxy_scores else None,
    }


def normalize_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


if __name__ == "__main__":
    main()
