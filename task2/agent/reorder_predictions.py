from __future__ import annotations

import argparse
import json
from pathlib import Path

from .app import load_datapoints

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
SUPPORT_METADATA_FILENAMES = {
    "setup.py",
    "setup.cfg",
    "pyproject.toml",
    "tox.ini",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reorder exact context blocks inside an existing predictions JSONL")
    parser.add_argument("--stage", required=True, help="Dataset stage, for example test or public")
    parser.add_argument("--lang", default="python", choices=["python", "kotlin"], help="Language split")
    parser.add_argument("--input", type=Path, required=True, help="Input predictions JSONL")
    parser.add_argument("--output", type=Path, required=True, help="Output predictions JSONL")
    parser.add_argument("--limit", type=int, help="Process only the first N rows")
    parser.add_argument("--max-blocks", type=int, help="Optional maximum number of blocks to keep per row")
    parser.add_argument("--char-budget", type=int, help="Optional character budget after reordering")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent.parent
    dataset_path = base_dir / "data" / f"{args.lang}-{args.stage}.jsonl"
    if not dataset_path.exists():
        raise FileNotFoundError(dataset_path)
    if not args.input.exists():
        raise FileNotFoundError(args.input)

    datapoints = load_datapoints(dataset_path)
    predictions = load_predictions(args.input)
    if args.limit is not None:
        datapoints = datapoints[: args.limit]
        predictions = predictions[: args.limit]
    if len(datapoints) != len(predictions):
        raise ValueError(
            f"Prediction count mismatch: {len(predictions)} predictions vs {len(datapoints)} datapoints"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for datapoint, prediction in zip(datapoints, predictions, strict=True):
            blocks = parse_blocks(str(prediction.get("context", "")))
            ordered = reorder_blocks(
                blocks=blocks,
                target_path=datapoint.path,
                modified_paths=list(dict.fromkeys(datapoint.modified)),
                max_blocks=args.max_blocks,
                char_budget=args.char_budget,
            )
            row = dict(prediction)
            row["context"] = render_blocks(ordered)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(args.output)


def load_predictions(path: Path) -> list[dict[str, object]]:
    predictions: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            predictions.append(json.loads(line))
    return predictions


def parse_blocks(context: str) -> list[dict[str, str]]:
    if not context:
        return []
    if FILE_SEP not in context:
        return [{"path": "", "content": context, "raw": context}]

    blocks: list[dict[str, str]] = []
    for raw in context.split(FILE_SEP):
        if not raw:
            continue
        parsed = raw.lstrip("\n")
        path, sep, content = parsed.partition("\n")
        if not sep:
            content = ""
        blocks.append({"path": path.strip(), "content": content, "raw": raw})
    return blocks


def render_blocks(blocks: list[dict[str, str]]) -> str:
    if not blocks:
        return ""
    return "".join(f"{FILE_SEP}{block['raw']}" for block in blocks)


def reorder_blocks(
    *,
    blocks: list[dict[str, str]],
    target_path: str,
    modified_paths: list[str],
    max_blocks: int | None,
    char_budget: int | None,
) -> list[dict[str, str]]:
    unique_blocks = drop_redundant_blocks(blocks)
    target_parent = str(Path(target_path).parent)
    modified_set = set(modified_paths)

    ordered = sorted(
        unique_blocks,
        key=lambda block: (
            -block_score(
                path=block["path"],
                content=block["content"],
                target_path=target_path,
                target_parent=target_parent,
                modified_paths=modified_set,
            ),
            blocks.index(block),
        ),
    )

    if max_blocks is not None and max_blocks > 0:
        ordered = ordered[:max_blocks]

    if char_budget is not None and char_budget > 0:
        fitted: list[dict[str, str]] = []
        used = 0
        for block in ordered:
            rendered = f"{FILE_SEP}{block['path']}\n{block['content']}"
            addition = len(rendered) + (1 if fitted else 0)
            if fitted and used + addition > char_budget:
                continue
            if not fitted and len(rendered) > char_budget:
                fitted.append(block)
                break
            fitted.append(block)
            used += addition
        ordered = fitted

    return ordered


def drop_redundant_blocks(blocks: list[dict[str, str]]) -> list[dict[str, str]]:
    kept: list[dict[str, str]] = []
    for block in blocks:
        path = block["path"]
        content = block["content"]
        replaced = False
        for index, existing in enumerate(kept):
            if existing["path"] != path:
                continue
            if content and content in existing["content"]:
                replaced = True
                break
            if existing["content"] and existing["content"] in content:
                kept[index] = block
                replaced = True
                break
        if not replaced:
            kept.append(block)
    return kept


def block_score(
    *,
    path: str,
    content: str,
    target_path: str,
    target_parent: str,
    modified_paths: set[str],
) -> float:
    score = 0.0
    if path == target_path:
        score += 1000.0
    if path in modified_paths:
        score += 350.0
    if target_parent and target_parent != "." and path.startswith(f"{target_parent}/"):
        score += 220.0
    line_count = content.count("\n") + 1 if content else 0
    score += min(line_count, 220) * 0.4
    if is_support_metadata_path(path, target_path):
        score -= 320.0
    if is_broad_test_support_path(path, target_path):
        score -= 220.0
    return score


def is_support_metadata_path(path: str, target_path: str) -> bool:
    if path == target_path:
        return False
    normalized = path.lower().replace("\\", "/")
    name = Path(normalized).name
    stem = Path(normalized).stem
    parts = [part for part in normalized.split("/") if part]
    if name in SUPPORT_METADATA_FILENAMES:
        return True
    if stem in NOISE_PATH_TOKENS:
        return True
    return any(part in NOISE_PATH_TOKENS for part in parts[:-1])


def is_broad_test_support_path(path: str, target_path: str) -> bool:
    if path == target_path:
        return False
    if is_test_path(target_path):
        return False
    if not is_test_path(path):
        return False
    return shared_parent_depth(path, target_path) < 2


def is_test_path(path: str) -> bool:
    normalized = path.lower().replace("\\", "/")
    parts = [part for part in normalized.split("/") if part]
    name = Path(normalized).name
    return any(part in {"test", "tests", "testing"} for part in parts) or name.startswith("test_")


def shared_parent_depth(left_path: str, right_path: str) -> int:
    left_parts = Path(left_path).parent.parts
    right_parts = Path(right_path).parent.parts
    shared = 0
    for left, right in zip(left_parts, right_parts):
        if left != right:
            break
        shared += 1
    return shared


if __name__ == "__main__":
    main()
