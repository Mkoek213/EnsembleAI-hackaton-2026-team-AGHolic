from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from functools import partial
from pathlib import Path

from .config import Settings
from .models import ContextAnswer, TaskDatapoint
from .solver import RunConfig, TaskSolver

SEVERE_UNRESOLVED_MARKERS = (
    "model_request_failed",
    "search_stall",
    "max_agent_steps",
    "input_token_budget",
    "tool_failure_budget",
    "consecutive_tool_failures",
    "response_without_finish_tool",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Task2 context agent runner")
    parser.add_argument("--stage", required=True, help="Dataset stage, for example start, practice, public")
    parser.add_argument("--lang", default="python", choices=["python", "kotlin"], help="Language split")
    parser.add_argument("--limit", type=int, help="Process only the first N datapoints after filtering")
    parser.add_argument("--offset", type=int, default=0, help="Skip the first N datapoints")
    parser.add_argument("--datapoint-id", help="Run only a single datapoint id")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of datapoints to process concurrently. Keeps output order stable.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from an existing output file by skipping already written predictions.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output predictions JSONL path. Defaults to predictions/{lang}-{stage}-agent.jsonl",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent.parent
    data_dir = base_dir / "data"
    predictions_dir = base_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = data_dir / f"{args.lang}-{args.stage}.jsonl"
    repos_dir = data_dir / f"repositories-{args.lang}-{args.stage}"
    if not dataset_path.exists():
        raise FileNotFoundError(dataset_path)
    if not repos_dir.exists():
        raise FileNotFoundError(
            f"{repos_dir} does not exist. Run prepare_data.sh for this stage first."
        )

    output_path = args.output or predictions_dir / f"{args.lang}-{args.stage}-agent.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    datapoints = load_datapoints(dataset_path)
    if args.datapoint_id:
        datapoints = [dp for dp in datapoints if dp.id == args.datapoint_id]
    if args.offset:
        datapoints = datapoints[args.offset :]
    if args.limit is not None:
        datapoints = datapoints[: args.limit]

    total_datapoints = len(datapoints)
    if total_datapoints == 0:
        raise ValueError("No datapoints selected after applying filters.")

    if args.workers < 1:
        raise ValueError("--workers must be at least 1.")

    resume_count = 0
    if args.resume:
        resume_count = count_prediction_lines(output_path)
        if resume_count > total_datapoints:
            raise ValueError(
                f"Output file has {resume_count} predictions, but only {total_datapoints} datapoints are selected."
            )
        datapoints = datapoints[resume_count:]
        if resume_count:
            print(f"Resuming from datapoint {resume_count + 1}/{total_datapoints}")

    if not datapoints:
        print(f"Nothing to do, {resume_count}/{total_datapoints} predictions already exist.")
        print(output_path)
        return

    settings = Settings()
    write_mode = "a" if args.resume and output_path.exists() else "w"

    indexed_datapoints = list(enumerate(datapoints, start=resume_count + 1))

    with output_path.open(write_mode, encoding="utf-8") as output_file:
        if args.workers == 1:
            solver = TaskSolver(settings)
            for index, datapoint in indexed_datapoints:
                answer = run_single_datapoint(
                    datapoint=datapoint,
                    repo_root=repos_dir / f"{datapoint.repo.replace('/', '__')}-{datapoint.revision}",
                    language=args.lang,
                    stage=args.stage,
                    solver=solver,
                )
                output_file.write(json.dumps({"context": answer.context}, ensure_ascii=False) + "\n")
                output_file.flush()
                print(
                    f"[{index}/{total_datapoints}] "
                    f"{datapoint.id or datapoint.path} -> {len(answer.context)} chars"
                )
        else:
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                worker = partial(
                    run_indexed_datapoint,
                    repos_dir=repos_dir,
                    language=args.lang,
                    stage=args.stage,
                    settings=settings,
                )
                for index, datapoint, answer in executor.map(
                    worker,
                    indexed_datapoints,
                ):
                    output_file.write(json.dumps({"context": answer.context}, ensure_ascii=False) + "\n")
                    output_file.flush()
                    print(
                        f"[{index}/{total_datapoints}] "
                        f"{datapoint.id or datapoint.path} -> {len(answer.context)} chars"
                    )

    print(output_path)


def load_datapoints(dataset_path: Path) -> list[TaskDatapoint]:
    datapoints: list[TaskDatapoint] = []
    with dataset_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            datapoints.append(TaskDatapoint.model_validate(json.loads(line)))
    return datapoints


def count_prediction_lines(output_path: Path) -> int:
    if not output_path.exists():
        return 0

    count = 0
    with output_path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def run_single_datapoint(
    *,
    datapoint: TaskDatapoint,
    repo_root: Path,
    language: str,
    stage: str,
    solver: TaskSolver | None = None,
    settings: Settings | None = None,
) -> ContextAnswer:
    if not repo_root.exists():
        raise FileNotFoundError(repo_root)

    selected_settings = settings or (solver.settings if solver is not None else Settings())
    solver = solver or TaskSolver(selected_settings)
    answer = solver.run(
        RunConfig(
            datapoint=datapoint,
            repo_root=repo_root,
            language=language,
            stage=stage,
        )
    )
    if not selected_settings.enable_severe_retry_lane:
        return answer
    if not _is_severe_answer(answer):
        return answer

    retry_settings = replace(
        selected_settings,
        agent_model=selected_settings.severe_retry_model,
        max_agent_steps=max(
            selected_settings.max_agent_steps,
            selected_settings.severe_retry_max_agent_steps,
        ),
        max_input_tokens_per_datapoint=max(
            selected_settings.max_input_tokens_per_datapoint,
            selected_settings.severe_retry_max_input_tokens_per_datapoint,
        ),
        enable_severe_retry_lane=False,
    )
    retry_solver = TaskSolver(retry_settings)
    safe_id = (datapoint.id or datapoint.path).replace("/", "__")
    retry_answer = retry_solver.run(
        RunConfig(
            datapoint=datapoint,
            repo_root=repo_root,
            language=language,
            stage=stage,
            artifact_dir=retry_settings.artifacts_dir / f"{stage}-{language}-{safe_id}-retry",
        )
    )
    if _is_retry_answer_better(primary=answer, retry=retry_answer):
        return retry_answer
    return answer


def run_indexed_datapoint(
    item: tuple[int, TaskDatapoint],
    *,
    repos_dir: Path,
    language: str,
    stage: str,
    settings: Settings | None = None,
) -> tuple[int, TaskDatapoint, ContextAnswer]:
    index, datapoint = item
    answer = run_single_datapoint(
        datapoint=datapoint,
        repo_root=repos_dir / f"{datapoint.repo.replace('/', '__')}-{datapoint.revision}",
        language=language,
        stage=stage,
        settings=settings,
    )
    return index, datapoint, answer


def _is_severe_answer(answer: ContextAnswer) -> bool:
    unresolved = answer.unresolved_points or []
    for issue in unresolved:
        issue_lower = issue.lower()
        if any(marker in issue_lower for marker in SEVERE_UNRESOLVED_MARKERS):
            return True
    return False


def _is_retry_answer_better(*, primary: ContextAnswer, retry: ContextAnswer) -> bool:
    def rank(answer: ContextAnswer) -> tuple[int, int, int, int]:
        unresolved = answer.unresolved_points or []
        severe_count = sum(
            1
            for issue in unresolved
            if any(marker in issue.lower() for marker in SEVERE_UNRESOLVED_MARKERS)
        )
        return (
            -severe_count,
            -len(unresolved),
            len(answer.selected_paths),
            len(answer.context),
        )

    return rank(retry) > rank(primary)


if __name__ == "__main__":
    main()
