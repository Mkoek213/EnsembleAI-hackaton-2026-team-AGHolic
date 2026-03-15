from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path

from .app import load_datapoints
from .config import Settings
from .evaluate import (
    chrf,
    load_predictions,
    normalize_newlines,
    recover_reference,
    resolve_prediction_boundaries,
)
from .observability import flush_langfuse, get_langfuse_client, get_propagate_attributes
from .openai_service import OpenAIService


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Local end-to-end proxy benchmark: context -> completion -> reference comparison"
    )
    parser.add_argument("--stage", required=True, help="Dataset stage, for example start, practice, public")
    parser.add_argument("--lang", default="python", choices=["python", "kotlin"], help="Language split")
    parser.add_argument(
        "--predictions-file",
        type=Path,
        help="Predictions JSONL to evaluate. Defaults to predictions/{lang}-{stage}-agent.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Detailed JSONL results path. Defaults to agent/workspace/completion_evals/{lang}-{stage}.jsonl",
    )
    parser.add_argument("--limit", type=int, help="Evaluate only the first N datapoints")
    parser.add_argument("--model", help="Override completion model for local benchmarking")
    parser.add_argument("--max-output-tokens", type=int, help="Override completion max_output_tokens")
    parser.add_argument(
        "--only-recovered",
        action="store_true",
        help="Call the completion model only for datapoints where local reference recovery succeeded",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    settings = Settings()
    settings.ensure_directories()
    langfuse = get_langfuse_client()
    propagate_attributes = get_propagate_attributes()
    benchmark_session_id = uuid.uuid4().hex
    benchmark_tags = [
        "task2",
        "completion-benchmark",
        args.stage,
        args.lang,
        f"experiment:{settings.experiment_name}",
    ]

    base_dir = Path(__file__).resolve().parent.parent
    data_dir = base_dir / "data"
    predictions_dir = base_dir / "predictions"
    dataset_path = data_dir / f"{args.lang}-{args.stage}.jsonl"
    repos_dir = data_dir / f"repositories-{args.lang}-{args.stage}"
    predictions_path = args.predictions_file or predictions_dir / f"{args.lang}-{args.stage}-agent.jsonl"
    output_path = args.output or settings.completion_eval_dir / f"{args.lang}-{args.stage}.jsonl"

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

    openai = OpenAIService(settings)
    rows: list[dict[str, object]] = []
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as output_handle:
        for index, (datapoint, prediction) in enumerate(zip(datapoints, predictions, strict=True), start=1):
            datapoint_id = datapoint.id or datapoint.path
            row = {}
            with langfuse.start_as_current_observation(
                name="task2-local-completion-benchmark",
                as_type="evaluator",
                input={
                    "datapoint_id": datapoint_id,
                    "target_path": datapoint.path,
                    "stage": args.stage,
                    "language": args.lang,
                    "prediction_file": str(predictions_path),
                    "limit": args.limit,
                    "only_recovered": args.only_recovered,
                    "experiment_name": settings.experiment_name,
                },
                metadata={
                    "repo": datapoint.repo,
                    "revision": datapoint.revision,
                    "modified_count": len(datapoint.modified),
                    "completion_model": args.model or settings.completion_model,
                },
            ) as observation:
                with propagate_attributes(
                    session_id=benchmark_session_id,
                    trace_name="task2-local-completion-benchmark",
                    tags=benchmark_tags,
                    metadata={
                        "datapoint_id": datapoint_id,
                        "experiment_name": settings.experiment_name,
                        "target_path": datapoint.path,
                        "repo": datapoint.repo,
                        "revision": datapoint.revision,
                    },
                ):
                    repo_root = repos_dir / f"{datapoint.repo.replace('/', '__')}-{datapoint.revision}"
                    effective_prefix, effective_suffix = resolve_prediction_boundaries(datapoint, prediction)
                    recovered_reference = recover_reference(
                        datapoint,
                        repo_root,
                        prefix=effective_prefix,
                        suffix=effective_suffix,
                    )
                    should_generate = recovered_reference is not None or not args.only_recovered
                    context = prediction.get("context", "")

                    completion = ""
                    skipped_reason = None
                    if should_generate:
                        completion = openai.generate_completion(
                            context=str(context),
                            prefix=effective_prefix,
                            suffix=effective_suffix,
                            target_path=datapoint.path,
                            language=args.lang,
                            model=args.model,
                            max_output_tokens=args.max_output_tokens,
                        )
                    else:
                        skipped_reason = "reference_not_recovered"

                    row = build_row(
                        datapoint_id=datapoint_id,
                        target_path=datapoint.path,
                        context=str(context),
                        completion=completion,
                        recovered_reference=recovered_reference,
                        skipped_reason=skipped_reason,
                    )
                    row["langfuse_trace_id"] = safe_get_trace_id(langfuse)
                    row["langfuse_trace_url"] = safe_get_trace_url(langfuse, row["langfuse_trace_id"])

                    observation.update(
                        output={
                            "completion_generated": row["completion_generated"],
                            "completion_length": row["completion_length"],
                            "reference_recovered": row["reference_recovered"],
                            "completion_reference_exact_match": row["completion_reference_exact_match"],
                            "completion_reference_exact_match_stripped": row[
                                "completion_reference_exact_match_stripped"
                            ],
                            "completion_reference_chrf_proxy": row["completion_reference_chrf_proxy"],
                            "skipped_reason": row["skipped_reason"],
                            "trace_url": row["langfuse_trace_url"],
                        },
                        metadata={
                            "context_length": row["context_length"],
                            "reference_length": row["reference_length"],
                            "experiment_name": settings.experiment_name,
                        },
                        status_message=row["skipped_reason"],
                        model=args.model or settings.completion_model,
                    )
                    score_row(langfuse, row)
            rows.append(row)
            output_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(
                f"[{index}/{len(datapoints)}] "
                f"{row['id']} generated={row['completion_generated']} "
                f"scored={row['reference_recovered']} "
                f"chrf={row['completion_reference_chrf_proxy']}"
            )

    summary = summarize(rows)
    with langfuse.start_as_current_observation(
        name="task2-local-completion-benchmark-summary",
        as_type="evaluator",
        input={
            "stage": args.stage,
            "language": args.lang,
            "prediction_file": str(predictions_path),
            "output_file": str(output_path),
            "limit": args.limit,
            "only_recovered": args.only_recovered,
            "experiment_name": settings.experiment_name,
        },
        output=summary,
        metadata={"benchmark_session_id": benchmark_session_id},
    ):
        with propagate_attributes(
            session_id=benchmark_session_id,
            trace_name="task2-local-completion-benchmark-summary",
            tags=benchmark_tags,
            metadata={"experiment_name": settings.experiment_name},
        ):
            score_summary(langfuse, summary)
    flush_langfuse()
    print(json.dumps({"summary": summary}, ensure_ascii=False, indent=2))
    print(output_path)
    print(
        "Uwaga: to jest tylko lokalny benchmark proxy. "
        "Oficjalny score nadal zalezy od modeli i pipeline organizatora."
    )


def build_row(
    *,
    datapoint_id: str,
    target_path: str,
    context: str,
    completion: str,
    recovered_reference: str | None,
    skipped_reason: str | None,
) -> dict[str, object]:
    normalized_completion = normalize_newlines(completion)
    normalized_reference = normalize_newlines(recovered_reference or "")
    exact_match = normalized_completion == normalized_reference if recovered_reference is not None else None
    exact_match_stripped = (
        normalized_completion.strip() == normalized_reference.strip()
        if recovered_reference is not None
        else None
    )

    return {
        "id": datapoint_id,
        "target_path": target_path,
        "context_length": len(context),
        "completion_generated": skipped_reason is None,
        "skipped_reason": skipped_reason,
        "completion_length": len(completion),
        "completion": completion,
        "reference_recovered": recovered_reference is not None,
        "reference_length": len(recovered_reference or ""),
        "reference": recovered_reference,
        "completion_reference_exact_match": exact_match,
        "completion_reference_exact_match_stripped": exact_match_stripped,
        "completion_reference_chrf_proxy": (
            chrf(completion, recovered_reference) if recovered_reference is not None else None
        ),
    }


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    total = len(rows)
    generated = [row for row in rows if row["completion_generated"]]
    recovered = [row for row in rows if row["reference_recovered"]]
    scored = [
        row
        for row in rows
        if isinstance(row["completion_reference_chrf_proxy"], float)
    ]
    exact = [row for row in rows if row["completion_reference_exact_match"] is True]
    exact_stripped = [
        row for row in rows if row["completion_reference_exact_match_stripped"] is True
    ]
    proxy_scores = [
        row["completion_reference_chrf_proxy"]
        for row in rows
        if isinstance(row["completion_reference_chrf_proxy"], float)
    ]

    return {
        "total": total,
        "completion_generated": len(generated),
        "completion_generated_rate": len(generated) / total if total else 0.0,
        "reference_recovered": len(recovered),
        "reference_recovered_rate": len(recovered) / total if total else 0.0,
        "scored": len(scored),
        "scored_rate": len(scored) / total if total else 0.0,
        "exact_match": len(exact),
        "exact_match_rate": len(exact) / len(scored) if scored else 0.0,
        "exact_match_stripped": len(exact_stripped),
        "exact_match_stripped_rate": len(exact_stripped) / len(scored) if scored else 0.0,
        "avg_completion_reference_chrf_proxy": sum(proxy_scores) / len(proxy_scores) if proxy_scores else None,
    }


def score_row(langfuse: object, row: dict[str, object]) -> None:
    try:
        langfuse.score_current_trace(
            name="reference_recovered",
            value=float(bool(row["reference_recovered"])),
            comment="1.0 if local reference recovery succeeded",
        )
        langfuse.score_current_span(
            name="reference_recovered",
            value=float(bool(row["reference_recovered"])),
            comment="1.0 if local reference recovery succeeded",
        )
    except Exception:
        pass

    exact_match = row["completion_reference_exact_match"]
    if exact_match is not None:
        try:
            langfuse.score_current_trace(
                name="completion_reference_exact_match",
                value=float(bool(exact_match)),
            )
            langfuse.score_current_span(
                name="completion_reference_exact_match",
                value=float(bool(exact_match)),
            )
        except Exception:
            pass

    exact_match_stripped = row["completion_reference_exact_match_stripped"]
    if exact_match_stripped is not None:
        try:
            langfuse.score_current_trace(
                name="completion_reference_exact_match_stripped",
                value=float(bool(exact_match_stripped)),
            )
            langfuse.score_current_span(
                name="completion_reference_exact_match_stripped",
                value=float(bool(exact_match_stripped)),
            )
        except Exception:
            pass

    chrf_score = row["completion_reference_chrf_proxy"]
    if isinstance(chrf_score, float):
        try:
            langfuse.score_current_trace(
                name="completion_reference_chrf_proxy",
                value=chrf_score,
            )
            langfuse.score_current_span(
                name="completion_reference_chrf_proxy",
                value=chrf_score,
            )
        except Exception:
            pass


def score_summary(langfuse: object, summary: dict[str, object]) -> None:
    for name in [
        "completion_generated_rate",
        "reference_recovered_rate",
        "scored_rate",
        "exact_match_rate",
        "exact_match_stripped_rate",
        "avg_completion_reference_chrf_proxy",
    ]:
        value = summary.get(name)
        if not isinstance(value, float):
            continue
        try:
            langfuse.score_current_trace(name=name, value=value)
            langfuse.score_current_span(name=name, value=value)
        except Exception:
            continue


def safe_get_trace_id(langfuse: object) -> str | None:
    try:
        return langfuse.get_current_trace_id()
    except Exception:
        return None


def safe_get_trace_url(langfuse: object, trace_id: str | None) -> str | None:
    try:
        return langfuse.get_trace_url(trace_id=trace_id)
    except Exception:
        return None


if __name__ == "__main__":
    main()
