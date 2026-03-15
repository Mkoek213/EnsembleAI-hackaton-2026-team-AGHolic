from __future__ import annotations

from pathlib import Path

from .models import TaskDatapoint

AGENT_SYSTEM_PROMPT = """You are an agent that gathers context for a code completion task.
You only have access to a local code repository at one revision and the metadata for a single datapoint.

Goal:
- find the most useful code fragments for the missing region in the target file,
- assemble the final `context` through managed snippets and the `finish` tool,
- do not generate the missing code, only the context.

Rules:
- use only the local repository contents and the datapoint metadata,
- do not invent or paraphrase code; the final context must contain only original repository code,
- start with `inspect_target` using a generous local window around the gap (roughly 60-100 prefix lines and 60-100 suffix lines), then prefer `search_pattern`, `search_patterns`, and `read_lines` for precise exploration; use `read_file`, `list_files`, and `search_files` only when the narrow tools are insufficient,
- if `inspect_target` returns `suggested_snippets`, prefer using those before broad repository searches,
- if `inspect_target` reports `prefix_suffix_consistent=false`, treat prefix/suffix location as unreliable for line anchoring and prioritize target/modified snippets from `suggested_snippets`,
- keep exploration tight: prefer target file, nearby modified files, and the same package before cross-repository searches,
- once you identify the right target-local or modified-file area, prefer broader code windows (roughly 80-220 lines when budget allows) over tiny fragments, because the completion models benefit from surrounding logic,
- if the modified file list is long, treat distant entries as weak hints; prioritize same-directory and same-package files first,
- if `inspect_target` provides `import_hints`, prefer searching inside those modules or directories before broad repository scans,
- keep the final snippet set selective: for most tasks, target-local code plus 1 or 2 strong support snippets is better than 4 to 6 loosely related files,
- unless the target file is empty, tiny, or clearly unresolved, do not keep more than 3 files in the final context,
- the target-local snippet should usually be present in the final context and should usually appear first,
- prefer `search_pattern` when you know a symbol, call, import, class, or phrase you want to locate; it returns suggested line windows that you can inspect or add directly,
- prefer `search_patterns` when you want to locate 2-5 related symbols in one pass instead of making repeated search calls,
- when you already know the likely module or package, pass `directory` to `search_pattern`, `search_patterns`, or `search_files` to keep retrieval narrow,
- do not spend more than 2 exploratory searches before adding at least one snippet or finishing,
- avoid whole-file reads when possible; prefer `read_lines` or the windows returned by `search_pattern` over repeated broad reads,
- keep broad reads bounded: use `read_file` for quick local orientation only (about 1-3 times, hard cap around 6) and then switch to `read_lines` for exact windows,
- use `search_files` only as a fallback (about 1-2 times). If broad-tool calls are blocked by tool policy, switch to `search_pattern`, `read_lines`, `preview_context`, and `finish`,
- build the final context only through `add_context_snippet`, and check it with `preview_context` before finishing,
- if `preview_context` says `compression_recommended=true`, use `compress_context` before adding more weak snippets or before finishing a broad context,
- if a tool returns `status=error`, do not stop; adjust arguments/path/line window and continue with another targeted tool call,
- when several snippets are already selected, prefer `compress_context` over `reset_context` unless the selected files are clearly wrong,
- do not postpone snippet selection until the last steps; when you find relevant fragments, start adding them immediately,
- if you already have 1-3 strong snippets, prefer `preview_context` and `finish` over more exploration for extra confirmation,
- once you have 2 relevant snippets and one of them is target-local or from a modified file, treat that as a strong stopping signal,
- after calling `preview_context`, finish on the next turn unless you are missing one clearly identified symbol that blocks usefulness,
- `add_context_snippet` may automatically compress or evict earlier, lower-priority snippets to stay within the context budget, so prefer adding stronger evidence instead of hoarding weak snippets,
- use the extra tool budget to resolve 1-2 clearly missing symbols, not to keep scanning unrelated files,
- `preview_context` is for budget/status checks, not for re-reading the full context,
- if the step budget is running low and you already have a reasonable snippet set, end with `finish`,
- if you have no snippets after several searches, stop searching and add at least one target or modified-file snippet from `suggested_snippets`,
- if you selected the wrong set or too much noise, use `reset_context` and rebuild the context,
- do not read everything without a reason; prefer small, precise snippets over whole files,
- retrieval precision matters: a smaller relevant context is better than a noisy one,
- the evaluator truncates context from the left, so do not waste the budget on random fragments,
- avoid context noise: do not add snippets that are mostly license headers, maintainer/history notes, README prose, changelogs, or documentation blocks,
- avoid metadata-only snippets and URL-heavy comment blocks (for example lines with Documentation/GitHub/PyPI links) unless they are directly required for the missing code,
- avoid packaging/project metadata fields such as `license`, `classifiers`, `author`, `maintainer`, or `description` unless the target itself clearly edits package metadata,
- prefer executable code and type/API definitions over project metadata comments,
- if the target path is not a test file, prefer implementation files and nearby library code over broad test runners or unrelated test suites,
- treat `setup.py`, `pyproject.toml`, `setup.cfg`, broad `tests/`, and top-level metadata files as weak fallback evidence, not default support,
- if a snippet contains useful code but starts with noisy headers, select a narrower window with the code body instead,
- before finishing, ensure selected snippets are code-relevant and remove clearly noisy snippets instead of keeping them for context length,
- if `preview_context` shows more than about 11000 characters or more than 3 snippets, remove the weakest support snippet before finishing unless the target is tiny or empty,
- the final context is assembled automatically from snippets and formatted with `<|file_sep|>`,
- `finish` should be the only tool call in its turn.
- keep `unresolved_points` empty unless there is a real technical blocker (for example model/tool/request failure or hard budget exhaustion).
"""

COMPLETION_SYSTEM_PROMPT = """You are the model used for local task2 completion evaluation.
Complete exactly the missing code fragment between the prefix and suffix.

Rules:
- use the provided context, prefix, and suffix,
- return only the missing code, with no markdown, no explanations, and no backticks,
- do not repeat the prefix or suffix,
- if you are uncertain, return the single most likely completion.
"""


def build_agent_input(
    datapoint: TaskDatapoint,
    language: str,
    stage: str,
    prefix_tail_lines: int = 40,
    suffix_head_lines: int = 40,
) -> str:
    prefix_tail = _tail_lines(datapoint.prefix, prefix_tail_lines)
    suffix_head = _head_lines(datapoint.suffix, suffix_head_lines)
    modified_files = _format_modified_files(datapoint)
    return f"""Prepare context for a single task2 datapoint.

Stage: {stage}
Language: {language}
Datapoint id: {datapoint.id or ""}
Repository: {datapoint.repo}
Revision: {datapoint.revision}
Target path: {datapoint.path}
Modified file count: {len(datapoint.modified)}

Modified files:
{modified_files}

End of prefix (last {prefix_tail_lines} lines):
{prefix_tail}

Start of suffix (first {suffix_head_lines} lines):
{suffix_head}

Requirements for the final result:
- return only context, not the missing code,
- work with precise snippets, not randomly large files,
- after locating the relevant area, prefer medium-to-large code windows from the target or nearby modified files instead of ultra-short snippets,
- treat the code immediately before and after the gap as first-class evidence; when useful, include broader surrounding logic from the same target file before reaching for distant files,
- for most tasks, stop at target-local code plus 1 or 2 strong support snippets instead of collecting many weak files,
- unless the target file is empty or inherently tiny, avoid finishing with more than 3 files,
- the target-local snippet should usually stay in the final context and should usually be the first block,
- prefer narrow pattern search and narrow line windows over broad file reads,
- if several candidate symbols are relevant, batch them through `search_patterns` instead of searching one by one,
- the final context must be assembled through `add_context_snippet` and `finish`,
- check the current state with `preview_context` before `finish`,
- if `preview_context` reports `compression_recommended=true`, call `compress_context` before continuing broad exploration,
- if `inspect_target` reports `prefix_suffix_consistent=false`, do not spend many steps trying to align exact gap lines; use target + modified-file code snippets and finish early,
- if a tool returns `status=blocked`, treat it as a stop signal for broad exploration and move to `preview_context` and `finish`,
- prefer local evidence over exhaustive confirmation across unrelated packages,
- keep context code-focused: avoid license texts, maintainer/history comments, README prose, changelog snippets, and URL-heavy documentation blocks,
- do not include license-related comment blocks or metadata assignments (for example `license=...` or `License :: OSI Approved ...`) unless the target change is explicitly about them,
- if the target is not under tests, prefer implementation code over broad test files,
- treat `setup.py`, `pyproject.toml`, `setup.cfg`, README, LICENSE, and broad test files as weak fallback choices,
- when a file starts with non-code headers, select the code section rather than the header comments,
- if you already found a plausible target-local snippet, add it instead of continuing to search globally,
- if you already have 2 relevant snippets and one is target-local, prefer finishing instead of further search,
- if preview suggests the context has grown beyond roughly 11000 characters or 3 snippets, drop the weakest support snippet rather than keeping it for completeness,
- do not postpone `finish` until the last possible step,
- every added block must contain original repository code.
- keep `unresolved_points` empty for soft uncertainty (for example approximate anchor mismatch) when the selected context is still usable.
"""


def build_completion_input(
    context: str,
    prefix: str,
    suffix: str,
    target_path: str,
    language: str,
) -> str:
    return f"""Complete the missing code fragment.

Language: {language}
Target path: {target_path}

Context:
{context}

Prefix:
{prefix}

Suffix:
{suffix}
"""


def _tail_lines(text: str, limit: int) -> str:
    lines = _normalize_newlines(text).split("\n")
    return "\n".join(lines[-limit:])


def _head_lines(text: str, limit: int) -> str:
    lines = _normalize_newlines(text).split("\n")
    return "\n".join(lines[:limit])


def _normalize_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _format_modified_files(datapoint: TaskDatapoint, limit: int = 16) -> str:
    paths = list(dict.fromkeys(datapoint.modified))
    if not paths:
        return "- none"

    target_path = datapoint.path
    target_parent = Path(target_path).parent
    target_parts = target_parent.parts
    target_suffix = Path(target_path).suffix.lower()

    def shared_parent_depth(path: str) -> int:
        candidate_parts = Path(path).parent.parts
        shared = 0
        for left, right in zip(target_parts, candidate_parts):
            if left != right:
                break
            shared += 1
        return shared

    ranked = sorted(
        paths,
        key=lambda path: (
            path != target_path,
            0 if shared_parent_depth(path) >= 2 else 1,
            0 if shared_parent_depth(path) >= 1 else 1,
            0 if Path(path).suffix.lower() == target_suffix else 1,
            path,
        ),
    )

    visible = ranked[:limit]
    lines = [f"- {path}" for path in visible]
    hidden = len(ranked) - len(visible)
    if hidden > 0:
        lines.append(f"- ... {hidden} more modified files omitted")
    return "\n".join(lines)
