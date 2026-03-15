from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from .context_manager import ContextManager, ContextSnippet
from .models import ContextAnswer
from .observability import get_langfuse_client

if TYPE_CHECKING:
    from .documents import CodeRepository
    from .solver import RunConfig
    from .config import Settings

ABSOLUTE_READ_FILE_CALL_LIMIT = 10
READ_FILE_CALL_LIMIT_WITH_SNIPPETS = 6
ABSOLUTE_SEARCH_FILES_CALL_LIMIT = 4
SEARCH_FILES_CALL_LIMIT_WITH_SNIPPETS = 2
ABSOLUTE_SEARCH_PATTERN_CALL_LIMIT = 10
SEARCH_PATTERN_CALL_LIMIT_WITHOUT_SNIPPETS = 6
ABSOLUTE_SEARCH_PATTERNS_CALL_LIMIT = 6
SEARCH_PATTERNS_CALL_LIMIT_WITHOUT_SNIPPETS = 4
STALL_SEARCH_CALL_THRESHOLD = 6
STALL_BLOCKED_CALL_THRESHOLD = 2
STALL_MIN_STEP = 6
SEVERE_UNRESOLVED_MARKERS = (
    "model_request_failed",
    "search_stall",
    "max_agent_steps",
    "input_token_budget",
    "tool_failure_budget",
    "consecutive_tool_failures",
    "response_without_finish_tool",
)
BENIGN_UNRESOLVED_MARKERS = (
    "prefix/suffix",
    "prefix_suffix",
    "anchor",
    "exact missing",
    "gap location",
    "uncertain",
    "not present in repository",
    "inconsistent",
)
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
FORBIDDEN_LINE_MARKERS = (
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
FORBIDDEN_METADATA_ASSIGNMENT_RE = re.compile(
    r"""
    ^\s*
    (?:['"])?(
        license|
        classifier[s]?|
        author(?:_email)?|
        maintainer(?:_email)?|
        description|
        long_description|
        project_urls?|
        url
    )(?:['"])?
    \s*[:=]
    """,
    re.IGNORECASE | re.VERBOSE,
)
FORBIDDEN_LICENSE_LITERAL_RE = re.compile(
    r"""
    \b(
        license|
        licensed\ under|
        copyright|
        all\ rights\ reserved|
        free\ software\ foundation
    )\b
    """,
    re.IGNORECASE | re.VERBOSE,
)
CODE_LINE_PATTERN = re.compile(
    r"\\b(def|class|return|if|elif|else|for|while|with|try|except|raise|import|from|assert|yield|lambda|pass|break|continue)\\b"
)


class ToolArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")


class InspectTargetArgs(ToolArgs):
    prefix_tail_lines: int = Field(
        default=int(os.getenv("TASK2_INSPECT_TARGET_PREFIX_TAIL_LINES", "80")),
        description="How many last lines of the prefix to return. Prefer 40-100 for stronger local anchoring.",
    )
    suffix_head_lines: int = Field(
        default=int(os.getenv("TASK2_INSPECT_TARGET_SUFFIX_HEAD_LINES", "80")),
        description="How many first lines of the suffix to return. Prefer 40-100 for stronger local anchoring.",
    )


class ListFilesArgs(ToolArgs):
    directory: str = Field(
        default=".",
        description="Directory relative to repository root.",
    )
    recursive: bool = Field(
        default=False,
        description="Whether to recurse into subdirectories.",
    )
    extension: str | None = Field(
        default=None,
        description="Optional extension filter like '.py' or '.kt'.",
    )
    limit: int = Field(
        default=40,
        description="Maximum number of entries to return. Prefer 20-40 unless you have a specific reason.",
    )


class ReadFileArgs(ToolArgs):
    path: str = Field(
        description="Repository-relative file path to read.",
    )
    start_line: int = Field(
        default=1,
        description="1-based first line number to include.",
    )
    end_line: int = Field(
        default=80,
        description="1-based last line number to include. Keep read windows under about 80 lines when possible.",
    )


class ReadLinesArgs(ToolArgs):
    path: str = Field(
        description="Repository-relative file path to read.",
    )
    start_line: int = Field(
        description="1-based first line number to include.",
    )
    end_line: int = Field(
        description="1-based last line number to include. This tool is for narrow windows only.",
    )


class SearchFilesArgs(ToolArgs):
    query: str = Field(
        description="Case-insensitive phrase or symbol to search for.",
    )
    directory: str | None = Field(
        default=None,
        description="Optional repository-relative directory to limit the search to.",
        validation_alias=AliasChoices("directory", "path"),
    )
    extension: str | None = Field(
        default=None,
        description="Optional extension filter like '.py' or '.kt'.",
    )
    limit: int = Field(
        default=8,
        description="Maximum number of matches to return. Prefer 3-8 targeted matches.",
    )


class SearchPatternArgs(ToolArgs):
    pattern: str = Field(
        description="Literal text or regex pattern to find in repository files.",
    )
    directory: str | None = Field(
        default=None,
        description="Optional repository-relative directory to limit the search to.",
        validation_alias=AliasChoices("directory", "path"),
    )
    extension: str | None = Field(
        default=None,
        description="Optional extension filter like '.py' or '.kt'.",
    )
    limit: int = Field(
        default=6,
        description="Maximum number of pattern matches to return. Prefer 2-6.",
    )
    context_lines: int = Field(
        default=6,
        description="How many surrounding lines to suggest around the match. Prefer 3-8.",
    )
    regex: bool = Field(
        default=False,
        description="Whether `pattern` should be treated as a regular expression.",
    )
    case_sensitive: bool = Field(
        default=False,
        description="Whether matching should be case-sensitive.",
    )


class SearchPatternsArgs(ToolArgs):
    patterns: list[str] = Field(
        description="Several literal texts or regex patterns to find in repository files.",
    )
    directory: str | None = Field(
        default=None,
        description="Optional repository-relative directory to limit the search to.",
        validation_alias=AliasChoices("directory", "path"),
    )
    extension: str | None = Field(
        default=None,
        description="Optional extension filter like '.py' or '.kt'.",
    )
    per_pattern_limit: int = Field(
        default=3,
        description="Maximum number of matches to return for each pattern. Prefer 1-3.",
    )
    total_limit: int = Field(
        default=10,
        description="Maximum number of matches to return across all patterns.",
    )
    context_lines: int = Field(
        default=6,
        description="How many surrounding lines to suggest around each match. Prefer 3-8.",
    )
    regex: bool = Field(
        default=False,
        description="Whether each pattern should be treated as a regular expression.",
    )
    case_sensitive: bool = Field(
        default=False,
        description="Whether matching should be case-sensitive.",
    )


class AddContextSnippetArgs(ToolArgs):
    path: str = Field(
        description="Repository-relative file path to add to the final context.",
    )
    start_line: int = Field(
        description="1-based first line number of the snippet.",
    )
    end_line: int = Field(
        description="1-based last line number of the snippet.",
    )


class PreviewContextArgs(ToolArgs):
    pass


class CompressContextArgs(ToolArgs):
    target_lines: int = Field(
        default=32,
        description="Target line count for broad snippets after compression. Prefer 24-48.",
    )
    keep_recent_snippets: int = Field(
        default=1,
        description="Keep this many most recently added snippets unchanged when possible.",
    )
    drop_if_needed: bool = Field(
        default=False,
        description="Whether to drop weaker snippets if compression alone still cannot fit the budget.",
    )
    only_if_needed: bool = Field(
        default=True,
        description="Skip compression if the current context is already compact enough.",
    )


class ResetContextArgs(ToolArgs):
    pass


class FinishArgs(ToolArgs):
    evidence: list[str] = Field(
        default_factory=list,
        description="Short notes justifying the chosen context.",
    )
    unresolved_points: list[str] = Field(
        default_factory=list,
        description="Remaining uncertainties, if any.",
    )


@dataclass(frozen=True)
class ToolExecutionResult:
    output: str = ""
    final_answer: ContextAnswer | None = None


class AgentToolRuntime:
    def __init__(
        self,
        settings: Settings,
        run_config: RunConfig,
        repository: CodeRepository,
    ) -> None:
        self.settings = settings
        self.run_config = run_config
        self.repository = repository
        self.model_logs: list[dict[str, Any]] = []
        self.tool_logs: list[dict[str, Any]] = []
        self.langfuse = get_langfuse_client()
        self.context_manager = ContextManager(
            char_budget=settings.context_char_budget,
            max_snippets=settings.context_max_snippets,
        )
        self._tool_call_counts: dict[str, int] = {}

    def tool_definitions(self) -> list[dict[str, Any]]:
        return [
            self._tool_definition(
                "inspect_target",
                "Return metadata about the completion target, modified files, and trimmed prefix/suffix.",
                InspectTargetArgs,
            ),
            self._tool_definition(
                "search_pattern",
                "Find literal or regex pattern matches and return compact line windows around each match.",
                SearchPatternArgs,
            ),
            self._tool_definition(
                "search_patterns",
                "Find several literal or regex patterns in one pass and return compact line windows around the matches.",
                SearchPatternsArgs,
            ),
            self._tool_definition(
                "read_lines",
                "Read a narrow, exact line window from a repository file.",
                ReadLinesArgs,
            ),
            self._tool_definition(
                "list_files",
                "List files or directories inside the repository.",
                ListFilesArgs,
            ),
            self._tool_definition(
                "read_file",
                "Read exact line ranges from a repository file. Use this sparingly for broader local inspection.",
                ReadFileArgs,
            ),
            self._tool_definition(
                "search_files",
                "Search repository text files for a phrase or symbol. Use this as a broader fallback search.",
                SearchFilesArgs,
            ),
            self._tool_definition(
                "add_context_snippet",
                "Add an exact snippet from a repository file into the managed final context. If the context budget is tight, earlier low-priority snippets may be compressed or removed automatically to preserve the strongest evidence.",
                AddContextSnippetArgs,
            ),
            self._tool_definition(
                "preview_context",
                "Preview the currently selected context, snippet list, and remaining budget.",
                PreviewContextArgs,
            ),
            self._tool_definition(
                "compress_context",
                "Compress the currently selected snippets into narrower code windows around their anchors. Use this when preview says compression is recommended or before adding more snippets near the budget.",
                CompressContextArgs,
            ),
            self._tool_definition(
                "reset_context",
                "Clear all currently selected context snippets and start over.",
                ResetContextArgs,
            ),
            self._tool_definition(
                "finish",
                "Finish the run using the currently selected context snippets.",
                FinishArgs,
            ),
        ]

    def execute(self, call: Any) -> ToolExecutionResult:
        name = call.name

        if name == "inspect_target":
            args = InspectTargetArgs.model_validate_json(call.arguments or "{}")
            arguments = args.model_dump(mode="json")
            return self._execute_observed_tool(
                call,
                name,
                arguments,
                lambda: self._log_and_wrap(call, name, arguments, self._handle_inspect_target(args)),
            )

        if name == "list_files":
            args = ListFilesArgs.model_validate_json(call.arguments or "{}")
            arguments = args.model_dump(mode="json")
            return self._execute_observed_tool(
                call,
                name,
                arguments,
                lambda: self._log_and_wrap(call, name, arguments, self._handle_list_files(args)),
            )

        if name == "read_file":
            args = ReadFileArgs.model_validate_json(call.arguments)
            arguments = args.model_dump(mode="json")
            return self._execute_observed_tool(
                call,
                name,
                arguments,
                lambda: self._log_and_wrap(call, name, arguments, self._handle_read_file(args)),
            )

        if name == "read_lines":
            args = ReadLinesArgs.model_validate_json(call.arguments)
            arguments = args.model_dump(mode="json")
            return self._execute_observed_tool(
                call,
                name,
                arguments,
                lambda: self._log_and_wrap(call, name, arguments, self._handle_read_lines(args)),
            )

        if name == "search_files":
            args = SearchFilesArgs.model_validate_json(call.arguments)
            arguments = args.model_dump(mode="json")
            return self._execute_observed_tool(
                call,
                name,
                arguments,
                lambda: self._log_and_wrap(call, name, arguments, self._handle_search_files(args)),
            )

        if name == "search_pattern":
            args = SearchPatternArgs.model_validate_json(call.arguments)
            arguments = args.model_dump(mode="json")
            return self._execute_observed_tool(
                call,
                name,
                arguments,
                lambda: self._log_and_wrap(call, name, arguments, self._handle_search_pattern(args)),
            )

        if name == "search_patterns":
            args = SearchPatternsArgs.model_validate_json(call.arguments)
            arguments = args.model_dump(mode="json")
            return self._execute_observed_tool(
                call,
                name,
                arguments,
                lambda: self._log_and_wrap(call, name, arguments, self._handle_search_patterns(args)),
            )

        if name == "add_context_snippet":
            args = AddContextSnippetArgs.model_validate_json(call.arguments)
            arguments = args.model_dump(mode="json")
            return self._execute_observed_tool(
                call,
                name,
                arguments,
                lambda: self._log_and_wrap(call, name, arguments, self._handle_add_context_snippet(args)),
            )

        if name == "preview_context":
            args = PreviewContextArgs.model_validate_json(call.arguments or "{}")
            arguments = args.model_dump(mode="json")
            return self._execute_observed_tool(
                call,
                name,
                arguments,
                lambda: self._log_and_wrap(call, name, arguments, self._handle_preview_context(args)),
            )

        if name == "compress_context":
            args = CompressContextArgs.model_validate_json(call.arguments or "{}")
            arguments = args.model_dump(mode="json")
            return self._execute_observed_tool(
                call,
                name,
                arguments,
                lambda: self._log_and_wrap(call, name, arguments, self._handle_compress_context(args)),
            )

        if name == "reset_context":
            args = ResetContextArgs.model_validate_json(call.arguments or "{}")
            arguments = args.model_dump(mode="json")
            return self._execute_observed_tool(
                call,
                name,
                arguments,
                lambda: self._log_and_wrap(call, name, arguments, self._handle_reset_context(args)),
            )

        if name == "finish":
            args = FinishArgs.model_validate_json(call.arguments)
            arguments = args.model_dump(mode="json")
            return self._execute_observed_tool(
                call,
                name,
                arguments,
                lambda: self._handle_finish(call, arguments, args),
            )

        raise ValueError(f"Unsupported tool: {name}")

    def _handle_inspect_target(self, args: InspectTargetArgs) -> dict[str, Any]:
        return self.repository.inspect_target(
            prefix_tail_lines=args.prefix_tail_lines,
            suffix_head_lines=args.suffix_head_lines,
        )

    def _handle_list_files(self, args: ListFilesArgs) -> dict[str, Any]:
        return self.repository.list_files(
            directory=args.directory,
            recursive=args.recursive,
            extension=args.extension,
            limit=args.limit,
        )

    def _handle_read_file(self, args: ReadFileArgs) -> dict[str, Any]:
        read_calls = self._tool_call_count("read_file")
        snippet_count = int(self.context_manager.preview().get("snippet_count", 0))
        if read_calls > ABSOLUTE_READ_FILE_CALL_LIMIT:
            return {
                "status": "blocked",
                "reason": "read_file_call_limit_reached",
                "limit": ABSOLUTE_READ_FILE_CALL_LIMIT,
                "read_file_calls": read_calls,
                "snippet_count": snippet_count,
                "suggestion": "use_read_lines_or_finish",
            }
        if snippet_count >= 2 and read_calls > READ_FILE_CALL_LIMIT_WITH_SNIPPETS:
            return {
                "status": "blocked",
                "reason": "read_file_call_limit_after_snippets",
                "limit": READ_FILE_CALL_LIMIT_WITH_SNIPPETS,
                "read_file_calls": read_calls,
                "snippet_count": snippet_count,
                "suggestion": "preview_context_then_finish",
            }
        return self.repository.read_file(
            path=args.path,
            start_line=args.start_line,
            end_line=args.end_line,
        )

    def _handle_read_lines(self, args: ReadLinesArgs) -> dict[str, Any]:
        return self.repository.read_lines(
            path=args.path,
            start_line=args.start_line,
            end_line=args.end_line,
        )

    def _handle_search_files(self, args: SearchFilesArgs) -> dict[str, Any]:
        search_calls = self._tool_call_count("search_files")
        snippet_count = int(self.context_manager.preview().get("snippet_count", 0))
        if search_calls > ABSOLUTE_SEARCH_FILES_CALL_LIMIT:
            return {
                "status": "blocked",
                "reason": "search_files_call_limit_reached",
                "limit": ABSOLUTE_SEARCH_FILES_CALL_LIMIT,
                "search_files_calls": search_calls,
                "snippet_count": snippet_count,
                "suggestion": "use_search_pattern_or_finish",
            }
        if snippet_count >= 2 and search_calls > SEARCH_FILES_CALL_LIMIT_WITH_SNIPPETS:
            return {
                "status": "blocked",
                "reason": "search_files_call_limit_after_snippets",
                "limit": SEARCH_FILES_CALL_LIMIT_WITH_SNIPPETS,
                "search_files_calls": search_calls,
                "snippet_count": snippet_count,
                "suggestion": "preview_context_then_finish",
            }
        return self.repository.search_files(
            query=args.query,
            directory=args.directory,
            extension=args.extension,
            limit=args.limit,
        )

    def _handle_search_pattern(self, args: SearchPatternArgs) -> dict[str, Any]:
        search_calls = self._tool_call_count("search_pattern")
        snippet_count = int(self.context_manager.preview().get("snippet_count", 0))
        if search_calls > ABSOLUTE_SEARCH_PATTERN_CALL_LIMIT:
            return {
                "status": "blocked",
                "reason": "search_pattern_call_limit_reached",
                "limit": ABSOLUTE_SEARCH_PATTERN_CALL_LIMIT,
                "search_pattern_calls": search_calls,
                "snippet_count": snippet_count,
                "suggestion": "add_context_from_suggestions_then_finish",
                "suggested_snippets": self.repository.suggest_context_snippets(limit=3),
            }
        if snippet_count == 0 and search_calls > SEARCH_PATTERN_CALL_LIMIT_WITHOUT_SNIPPETS:
            return {
                "status": "blocked",
                "reason": "search_pattern_stall_without_snippets",
                "limit": SEARCH_PATTERN_CALL_LIMIT_WITHOUT_SNIPPETS,
                "search_pattern_calls": search_calls,
                "snippet_count": snippet_count,
                "suggestion": "add_target_or_modified_snippet_before_more_search",
                "suggested_snippets": self.repository.suggest_context_snippets(limit=3),
            }
        return self.repository.search_pattern(
            pattern=args.pattern,
            directory=args.directory,
            extension=args.extension,
            limit=args.limit,
            context_lines=args.context_lines,
            regex=args.regex,
            case_sensitive=args.case_sensitive,
        )

    def _handle_search_patterns(self, args: SearchPatternsArgs) -> dict[str, Any]:
        search_calls = self._tool_call_count("search_patterns")
        snippet_count = int(self.context_manager.preview().get("snippet_count", 0))
        if search_calls > ABSOLUTE_SEARCH_PATTERNS_CALL_LIMIT:
            return {
                "status": "blocked",
                "reason": "search_patterns_call_limit_reached",
                "limit": ABSOLUTE_SEARCH_PATTERNS_CALL_LIMIT,
                "search_patterns_calls": search_calls,
                "snippet_count": snippet_count,
                "suggestion": "add_context_from_suggestions_then_finish",
                "suggested_snippets": self.repository.suggest_context_snippets(limit=3),
            }
        if snippet_count == 0 and search_calls > SEARCH_PATTERNS_CALL_LIMIT_WITHOUT_SNIPPETS:
            return {
                "status": "blocked",
                "reason": "search_patterns_stall_without_snippets",
                "limit": SEARCH_PATTERNS_CALL_LIMIT_WITHOUT_SNIPPETS,
                "search_patterns_calls": search_calls,
                "snippet_count": snippet_count,
                "suggestion": "add_target_or_modified_snippet_before_more_search",
                "suggested_snippets": self.repository.suggest_context_snippets(limit=3),
            }
        return self.repository.search_patterns(
            patterns=args.patterns,
            directory=args.directory,
            extension=args.extension,
            per_pattern_limit=args.per_pattern_limit,
            total_limit=args.total_limit,
            context_lines=args.context_lines,
            regex=args.regex,
            case_sensitive=args.case_sensitive,
        )

    def _handle_add_context_snippet(self, args: AddContextSnippetArgs) -> dict[str, Any]:
        snippet = self.repository.read_snippet(
            path=args.path,
            start_line=args.start_line,
            end_line=args.end_line,
        )
        candidate = self._to_context_snippet(snippet)
        cleaned = self._sanitize_context_snippet(candidate)
        if cleaned is None:
            preview = self.context_manager.preview()
            preview.update(
                {
                    "status": "rejected",
                    "reason": "noisy_or_non_code_snippet",
                    "candidate_ref": candidate.ref,
                }
            )
            return preview
        return self.context_manager.add_snippet(cleaned)

    def _handle_preview_context(self, args: PreviewContextArgs) -> dict[str, Any]:
        _ = args
        return self.context_manager.preview()

    def _handle_compress_context(self, args: CompressContextArgs) -> dict[str, Any]:
        return self.context_manager.compress(
            target_lines=args.target_lines,
            keep_recent_snippets=args.keep_recent_snippets,
            drop_if_needed=args.drop_if_needed,
            only_if_needed=args.only_if_needed,
        )

    def _handle_reset_context(self, args: ResetContextArgs) -> dict[str, Any]:
        _ = args
        return self.context_manager.reset()

    def _handle_finish(
        self,
        call: Any,
        arguments: dict[str, Any],
        args: FinishArgs,
    ) -> ToolExecutionResult:
        cleanup_summary = self._stabilize_context_after_quality_floor()
        packing_summary = self._run_final_context_pack()
        preview = self.context_manager.preview()
        preview["noise_cleanup"] = cleanup_summary
        preview["final_packing"] = packing_summary
        unresolved_points = self._normalize_unresolved_points(args.unresolved_points)
        final_answer = ContextAnswer(
            context=self.context_manager.build_context(),
            selected_paths=self.context_manager.selected_paths(),
            selected_spans=self.context_manager.selected_spans(),
            evidence=args.evidence,
            unresolved_points=unresolved_points,
        )
        self.tool_logs.append(
            {
                "call_id": call.call_id,
                "tool_name": "finish",
                "arguments": arguments,
                "context_preview": preview,
                "final_answer": final_answer.model_dump(mode="json"),
            }
        )
        return ToolExecutionResult(final_answer=final_answer)

    def build_timeout_answer(
        self,
        unresolved_point: str = "max_agent_steps_reached_heuristic_fallback",
        evidence_note: str | None = None,
        auto_finished_unresolved_point: str = "max_agent_steps_reached_auto_finished",
        existing_context_unresolved_point: str | None = "max_agent_steps_reached_auto_finished",
    ) -> ContextAnswer:
        existing_context = self.context_manager.build_context()
        if existing_context:
            self._stabilize_context_after_quality_floor()
            self._run_final_context_pack()
            self.context_manager.compress(target_lines=96, keep_recent_snippets=1, only_if_needed=True)
            resolved_unresolved_point = existing_context_unresolved_point
            if resolved_unresolved_point == "max_agent_steps_reached_auto_finished":
                resolved_unresolved_point = auto_finished_unresolved_point
            unresolved_points = []
            if not self._context_is_reasonable():
                unresolved_points = (
                    [resolved_unresolved_point]
                    if resolved_unresolved_point
                    else []
                )
            return ContextAnswer(
                context=self.context_manager.build_context(),
                selected_paths=self.context_manager.selected_paths(),
                selected_spans=self.context_manager.selected_spans(),
                unresolved_points=unresolved_points,
            )

        self._seed_context_from_suggestions()
        self._stabilize_context_after_quality_floor()
        self._run_final_context_pack()
        self.context_manager.compress(target_lines=96, keep_recent_snippets=1, only_if_needed=True)
        heuristic_context = self.context_manager.build_context()
        if heuristic_context:
            unresolved_points = [unresolved_point]
            if self._context_is_reasonable():
                unresolved_points = []
            return ContextAnswer(
                context=heuristic_context,
                selected_paths=self.context_manager.selected_paths(),
                selected_spans=self.context_manager.selected_spans(),
                evidence=[
                    note
                    for note in [
                        "Fallback context assembled from repository-local suggested snippets.",
                        evidence_note,
                    ]
                    if note
                ],
                unresolved_points=unresolved_points,
            )

        return ContextAnswer(
            context="",
            unresolved_points=["max_agent_steps_reached"],
        )

    def _log_and_wrap(
        self,
        call: Any,
        tool_name: str,
        arguments: dict[str, Any],
        result: dict[str, Any],
    ) -> ToolExecutionResult:
        self.tool_logs.append(
            {
                "call_id": call.call_id,
                "tool_name": tool_name,
                "arguments": arguments,
                "result": result,
            }
        )
        compact_result = self._compact_result_for_model(tool_name, result)
        return ToolExecutionResult(output=json.dumps(compact_result, ensure_ascii=False))

    def _execute_observed_tool(
        self,
        call: Any,
        tool_name: str,
        arguments: dict[str, Any],
        handler: Any,
    ) -> ToolExecutionResult:
        self._tool_call_counts[tool_name] = self._tool_call_counts.get(tool_name, 0) + 1
        with self.langfuse.start_as_current_observation(
            name=tool_name,
            as_type="tool",
            input=arguments,
            metadata={
                "call_id": getattr(call, "call_id", None),
                "datapoint_id": self.run_config.datapoint.id or self.run_config.datapoint.path,
                "target_path": self.run_config.datapoint.path,
                "repo": self.run_config.datapoint.repo,
                "revision": self.run_config.datapoint.revision,
            },
        ) as observation:
            try:
                execution_result = handler()
            except Exception as exc:
                observation.update(
                    level="ERROR",
                    status_message=str(exc),
                    output={"error": str(exc)},
                )
                raise

            observation.update(output=self._observation_output(tool_name, execution_result))
            return execution_result

    def _tool_call_count(self, tool_name: str) -> int:
        return int(self._tool_call_counts.get(tool_name, 0))

    def _observation_output(
        self,
        tool_name: str,
        execution_result: ToolExecutionResult,
    ) -> dict[str, Any]:
        if execution_result.final_answer is not None:
            final_answer = execution_result.final_answer.model_dump(mode="json")
            return {
                "tool_name": tool_name,
                "kind": "final_answer",
                "context_length": len(execution_result.final_answer.context),
                "selected_paths": execution_result.final_answer.selected_paths,
                "selected_spans": execution_result.final_answer.selected_spans,
                "final_answer": final_answer,
            }

        payload: dict[str, Any] = {
            "tool_name": tool_name,
            "kind": "tool_result",
        }
        try:
            payload["result"] = json.loads(execution_result.output)
        except Exception:
            payload["result"] = execution_result.output
        return payload

    def _compact_result_for_model(
        self,
        tool_name: str,
        result: dict[str, Any],
    ) -> dict[str, Any]:
        if tool_name == "inspect_target":
            return {
                "target_path": result.get("target_path"),
                "target_exists_in_repo": result.get("target_exists_in_repo"),
                "target_total_lines": result.get("target_total_lines"),
                "modified_files": result.get("modified_files"),
                "modified_files_total": result.get("modified_files_total"),
                "modified_files_truncated": result.get("modified_files_truncated"),
                "import_hints": result.get("import_hints", []),
                "prefix_total_lines": result.get("prefix_total_lines"),
                "suffix_total_lines": result.get("suffix_total_lines"),
                "prefix_suffix_consistent": result.get("prefix_suffix_consistent"),
                "prefix_suffix_status": result.get("prefix_suffix_status"),
                "estimated_gap_start_line": result.get("estimated_gap_start_line"),
                "estimated_gap_end_line": result.get("estimated_gap_end_line"),
                "prefix_tail": result.get("prefix_tail"),
                "suffix_head": result.get("suffix_head"),
                "same_directory_entries": self._compact_entries(result.get("same_directory_entries", [])),
                "suggested_snippets": result.get("suggested_snippets", []),
            }

        if tool_name == "list_files":
            return {
                "directory": result.get("directory"),
                "recursive": result.get("recursive"),
                "extension": result.get("extension"),
                "entries": self._compact_entries(result.get("entries", [])),
            }

        if tool_name in {"read_file", "read_lines"}:
            if result.get("status") == "blocked":
                return {
                    "status": result.get("status"),
                    "reason": result.get("reason"),
                    "limit": result.get("limit"),
                    "snippet_count": result.get("snippet_count"),
                    "suggestion": result.get("suggestion"),
                }
            return {
                "path": result.get("path"),
                "start_line": result.get("start_line"),
                "end_line": result.get("end_line"),
                "content": result.get("content"),
            }

        if tool_name == "search_files":
            if result.get("status") == "blocked":
                return {
                    "status": result.get("status"),
                    "reason": result.get("reason"),
                    "limit": result.get("limit"),
                    "snippet_count": result.get("snippet_count"),
                    "suggestion": result.get("suggestion"),
                }
            return {
                "query": result.get("query"),
                "directory": result.get("directory"),
                "matches": self._compact_matches(result.get("matches", [])),
            }

        if tool_name == "search_pattern":
            if result.get("status") == "blocked":
                return {
                    "status": result.get("status"),
                    "reason": result.get("reason"),
                    "limit": result.get("limit"),
                    "snippet_count": result.get("snippet_count"),
                    "suggestion": result.get("suggestion"),
                    "suggested_snippets": result.get("suggested_snippets", []),
                }
            return {
                "pattern": result.get("pattern"),
                "directory": result.get("directory"),
                "regex": result.get("regex"),
                "case_sensitive": result.get("case_sensitive"),
                "context_lines": result.get("context_lines"),
                "matches": self._compact_pattern_matches(result.get("matches", [])),
            }

        if tool_name == "search_patterns":
            if result.get("status") == "blocked":
                return {
                    "status": result.get("status"),
                    "reason": result.get("reason"),
                    "limit": result.get("limit"),
                    "snippet_count": result.get("snippet_count"),
                    "suggestion": result.get("suggestion"),
                    "suggested_snippets": result.get("suggested_snippets", []),
                }
            return {
                "patterns": result.get("patterns", []),
                "directory": result.get("directory"),
                "regex": result.get("regex"),
                "case_sensitive": result.get("case_sensitive"),
                "context_lines": result.get("context_lines"),
                "matches": self._compact_pattern_matches(result.get("matches", [])),
            }

        if tool_name in {"add_context_snippet", "preview_context", "compress_context", "reset_context"}:
            return {
                "status": result.get("status"),
                "reason": result.get("reason"),
                "candidate_ref": result.get("candidate_ref"),
                "snippet_count": result.get("snippet_count"),
                "max_snippets": result.get("max_snippets"),
                "used_chars": result.get("used_chars"),
                "char_budget": result.get("char_budget"),
                "remaining_chars": result.get("remaining_chars"),
                "fits_budget": result.get("fits_budget"),
                "compression_recommended": result.get("compression_recommended"),
                "compression_reason": result.get("compression_reason"),
                "selected_paths": result.get("selected_paths"),
                "selected_spans": result.get("selected_spans"),
                "chars_before": result.get("chars_before"),
                "chars_after": result.get("chars_after"),
                "compressed_refs": result.get("compressed_refs", []),
                "dropped_refs": result.get("dropped_refs", []),
            }

        return result

    def _compact_entries(self, entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
        compact_entries: list[dict[str, Any]] = []
        for entry in entries:
            compact_entries.append(
                {
                    "path": entry.get("path"),
                    "type": entry.get("type"),
                    "priority": entry.get("priority"),
                }
            )
        return compact_entries

    def _compact_matches(self, matches: list[dict[str, Any]]) -> list[dict[str, Any]]:
        compact_matches: list[dict[str, Any]] = []
        for match in matches:
            compact_matches.append(
                {
                    "path": match.get("path"),
                    "line": match.get("line"),
                    "snippet": str(match.get("snippet", ""))[:160],
                    "priority": match.get("priority"),
                }
            )
        return compact_matches

    def _compact_pattern_matches(self, matches: list[dict[str, Any]]) -> list[dict[str, Any]]:
        compact_matches: list[dict[str, Any]] = []
        for match in matches:
            compact_matches.append(
                {
                    "path": match.get("path"),
                    "match_line": match.get("match_line"),
                    "start_line": match.get("start_line"),
                    "end_line": match.get("end_line"),
                    "preview": str(match.get("preview", ""))[:160],
                    "patterns": match.get("patterns", []),
                    "priority": match.get("priority"),
                }
            )
        return compact_matches

    def _tool_definition(
        self,
        name: str,
        description: str,
        model: type[BaseModel],
    ) -> dict[str, Any]:
        return {
            "type": "function",
            "name": name,
            "description": description,
            "parameters": self._strict_tool_schema(model),
            "strict": False,
        }

    def _strict_tool_schema(self, model: type[BaseModel]) -> dict[str, Any]:
        schema = model.model_json_schema()
        schema["additionalProperties"] = False
        return schema

    def _seed_context_from_suggestions(self) -> None:
        for candidate in self.repository.suggest_context_snippets(
            limit=min(8, self.settings.context_max_snippets)
        ):
            preview = self._add_snippet_safely(
                str(candidate["path"]),
                int(candidate["start_line"]),
                int(candidate["end_line"]),
            )
            if not preview:
                continue
            if preview.get("status") == "rejected" and preview.get("reason") == "char_budget_exceeded":
                break

    def _apply_quality_floor(self) -> None:
        self._ensure_target_anchor_snippet()
        self._ensure_minimum_context()
        cleanup_summary = self._run_final_context_cleanup()
        if cleanup_summary.get("changed", False):
            self._ensure_target_anchor_snippet()
            self._ensure_minimum_context()

    def _ensure_target_anchor_snippet(self) -> None:
        target_path = self.repository.target_path
        anchor_line = self._target_anchor_line()
        if self._path_anchor_is_selected(target_path, anchor_line):
            return

        start_line = max(1, anchor_line - 64)
        end_line = max(start_line, anchor_line + 220)
        self._add_snippet_safely(target_path, start_line, end_line)

    def _ensure_minimum_context(self) -> None:
        if self.settings.min_context_chars <= 0:
            return
        if len(self.context_manager.build_context()) >= self.settings.min_context_chars:
            return

        selected_spans = set(self.context_manager.selected_spans())
        for candidate in self.repository.suggest_context_snippets(
            limit=min(8, self.settings.context_max_snippets)
        ):
            candidate_ref = f"{candidate['path']}:{candidate['start_line']}-{candidate['end_line']}"
            if candidate_ref in selected_spans:
                continue
            self._add_snippet_safely(
                str(candidate["path"]),
                int(candidate["start_line"]),
                int(candidate["end_line"]),
            )
            if len(self.context_manager.build_context()) >= self.settings.min_context_chars:
                break
            if len(self.context_manager.snippets()) >= self.settings.context_max_snippets:
                break

        if len(self.context_manager.build_context()) < self.settings.min_context_chars:
            self._expand_context_to_minimum()

    def should_auto_finalize(self, *, step_number: int, max_steps: int) -> tuple[bool, str]:
        snippet_count = int(self.context_manager.preview().get("snippet_count", 0))
        context_len = len(self.context_manager.build_context())
        reached_minimum = context_len >= max(600, self.settings.min_context_chars)
        if snippet_count > 0 and step_number >= max_steps - 1:
            if step_number < max_steps and not reached_minimum:
                return False, ""
            return True, "step_budget_near_limit_with_context"

        search_calls = self._tool_call_count("search_pattern") + self._tool_call_count("search_patterns")
        blocked_calls = 0
        for log in self.tool_logs:
            result = log.get("result")
            if isinstance(result, dict) and result.get("status") == "blocked":
                blocked_calls += 1

        if snippet_count == 0 and step_number >= STALL_MIN_STEP and search_calls >= STALL_SEARCH_CALL_THRESHOLD:
            return True, "search_stall_without_snippets"
        if snippet_count == 0 and blocked_calls >= STALL_BLOCKED_CALL_THRESHOLD:
            return True, "blocked_tool_stall_without_snippets"

        return False, ""

    def _add_snippet_safely(
        self,
        path: str,
        start_line: int,
        end_line: int,
    ) -> dict[str, object] | None:
        try:
            snippet = self.repository.read_snippet(
                path=path,
                start_line=start_line,
                end_line=end_line,
            )
        except Exception:
            return None

        candidate = self._to_context_snippet(snippet)
        cleaned = self._sanitize_context_snippet(candidate)
        if cleaned is None:
            return None
        return self.context_manager.add_snippet(cleaned)

    def _expand_context_to_minimum(self) -> None:
        if self.settings.min_context_chars <= 0:
            return

        expansion_steps = (160, 260, 360, 480)
        while len(self.context_manager.build_context()) < self.settings.min_context_chars:
            snippets = self.context_manager.snippets()
            if not snippets:
                return
            if len(snippets) >= self.settings.context_max_snippets:
                return

            changed = False
            ordered = sorted(
                snippets,
                key=lambda snippet: (
                    snippet.path != self.repository.target_path,
                    snippet.path not in self.repository.modified_paths,
                    snippet.priority,
                    snippet.selection_index,
                ),
            )

            for expansion in expansion_steps:
                if len(self.context_manager.build_context()) >= self.settings.min_context_chars:
                    return
                if len(self.context_manager.snippets()) >= self.settings.context_max_snippets:
                    return

                for snippet in ordered:
                    if len(self.context_manager.build_context()) >= self.settings.min_context_chars:
                        return
                    if len(self.context_manager.snippets()) >= self.settings.context_max_snippets:
                        return

                    anchor_line = snippet.anchor_line or ((snippet.start_line + snippet.end_line) // 2)
                    before = expansion if snippet.path == self.repository.target_path else max(80, expansion // 2)
                    after = expansion * 2 if snippet.path == self.repository.target_path else expansion
                    start_line = max(1, anchor_line - before)
                    end_line = max(start_line, anchor_line + after)
                    preview = self._add_snippet_safely(snippet.path, start_line, end_line)
                    if preview and preview.get("status") in {"added", "added_compressed"}:
                        changed = True

                if changed:
                    break

            if not changed:
                return

    def _stabilize_context_after_quality_floor(self) -> dict[str, object]:
        self._apply_quality_floor()
        cleanup_summary = self._run_final_context_cleanup()
        if cleanup_summary.get("changed", False):
            self._apply_quality_floor()
            followup_cleanup = self._run_final_context_cleanup()
            cleanup_summary = self._merge_cleanup_summaries(cleanup_summary, followup_cleanup)
        return cleanup_summary

    def _merge_cleanup_summaries(
        self,
        first: dict[str, object],
        second: dict[str, object],
    ) -> dict[str, object]:
        return {
            "changed": bool(first.get("changed", False) or second.get("changed", False)),
            "kept_refs": second.get("kept_refs", first.get("kept_refs", [])),
            "dropped_refs": list(dict.fromkeys([*first.get("dropped_refs", []), *second.get("dropped_refs", [])])),
            "redundant_refs": list(
                dict.fromkeys([*first.get("redundant_refs", []), *second.get("redundant_refs", [])])
            ),
            "sanitized_refs": list(
                dict.fromkeys([*first.get("sanitized_refs", []), *second.get("sanitized_refs", [])])
            ),
            "merged_refs": list(dict.fromkeys([*first.get("merged_refs", []), *second.get("merged_refs", [])])),
        }

    def _to_context_snippet(self, snippet: dict[str, object]) -> ContextSnippet:
        path = str(snippet["path"])
        start_line = int(snippet["start_line"])
        end_line = int(snippet["end_line"])
        return ContextSnippet(
            path=path,
            start_line=start_line,
            end_line=end_line,
            content=str(snippet["content"]),
            priority=self.repository.path_priority(path),
            anchor_line=(start_line + end_line) // 2,
        )

    def _run_final_context_cleanup(self) -> dict[str, object]:
        snippets = self.context_manager.snippets()
        if not snippets:
            return {
                "changed": False,
                "kept_refs": [],
                "dropped_refs": [],
                "redundant_refs": [],
                "sanitized_refs": [],
                "merged_refs": [],
            }

        cleaned_snippets: list[ContextSnippet] = []
        seen: set[tuple[str, str]] = set()
        dropped_refs: list[str] = []
        redundant_refs: list[str] = []
        sanitized_refs: list[str] = []

        for snippet in snippets:
            cleaned = self._sanitize_context_snippet(snippet)
            if cleaned is None:
                dropped_refs.append(snippet.ref)
                continue
            key = (cleaned.path, cleaned.content)
            if key in seen:
                dropped_refs.append(snippet.ref)
                continue
            seen.add(key)
            if cleaned.ref != snippet.ref or cleaned.content != snippet.content:
                sanitized_refs.append(f"{snippet.ref}->{cleaned.ref}")
            cleaned_snippets.append(cleaned)

        deduped_snippets, redundant_refs = self._drop_redundant_snippets(cleaned_snippets)
        merged_snippets, merged_refs = self._merge_overlapping_snippets(deduped_snippets)
        changed = bool(
            dropped_refs
            or redundant_refs
            or sanitized_refs
            or merged_refs
            or len(merged_snippets) != len(snippets)
        )
        if changed:
            self.context_manager.replace_snippets(merged_snippets)

        return {
            "changed": changed,
            "kept_refs": [snippet.ref for snippet in merged_snippets],
            "dropped_refs": dropped_refs,
            "redundant_refs": redundant_refs,
            "sanitized_refs": sanitized_refs,
            "merged_refs": merged_refs,
        }

    def _drop_redundant_snippets(
        self,
        snippets: list[ContextSnippet],
    ) -> tuple[list[ContextSnippet], list[str]]:
        if len(snippets) < 2:
            return snippets, []

        redundant_refs: list[str] = []
        kept: list[ContextSnippet] = []

        for candidate in sorted(
            snippets,
            key=lambda snippet: (
                snippet.path,
                snippet.start_line,
                -(snippet.end_line - snippet.start_line),
                snippet.selection_index,
            ),
        ):
            replaced = False
            for index, existing in enumerate(kept):
                if existing.path != candidate.path:
                    continue
                if (
                    candidate.content
                    and candidate.content in existing.content
                ):
                    redundant_refs.append(candidate.ref)
                    replaced = True
                    break
                if (
                    existing.content
                    and existing.content in candidate.content
                ):
                    redundant_refs.append(existing.ref)
                    kept[index] = candidate
                    replaced = True
                    break
            if not replaced:
                kept.append(candidate)

        kept.sort(key=lambda snippet: snippet.selection_index)
        return kept, redundant_refs

    def _run_final_context_pack(self) -> dict[str, object]:
        snippets = self.context_manager.snippets()
        if not snippets:
            return {
                "changed": False,
                "reordered": False,
                "compressed_refs": [],
                "dropped_refs": [],
                "soft_budget": 0,
                "target_is_small": False,
                "chars_before": 0,
                "chars_after": 0,
            }

        chars_before = len(self.context_manager.build_context())
        ordered = sorted(snippets, key=self._final_output_order_key)
        reordered = [snippet.ref for snippet in ordered] != [snippet.ref for snippet in snippets]
        working = ordered
        target_is_small = self._target_is_small(working)
        target_selected = any(snippet.path == self.repository.target_path for snippet in working)
        soft_budget = self._final_soft_char_budget(target_is_small)
        max_support_snippets = self._final_max_support_snippets(
            target_is_small=target_is_small,
            target_selected=target_selected,
        )
        working, dropped_refs = self._prune_final_support_snippets(
            working,
            max_support_snippets=max_support_snippets,
        )
        working, compressed_refs = self._shrink_support_snippets_to_budget(
            working,
            soft_budget=soft_budget,
            target_is_small=target_is_small,
        )
        working, budget_drop_refs = self._drop_support_snippets_to_budget(
            working,
            soft_budget=soft_budget,
        )
        dropped_refs.extend(budget_drop_refs)

        if reordered or compressed_refs or dropped_refs:
            self.context_manager.replace_snippets(working)

        return {
            "changed": bool(reordered or compressed_refs or dropped_refs),
            "reordered": reordered,
            "compressed_refs": compressed_refs,
            "dropped_refs": dropped_refs,
            "soft_budget": soft_budget,
            "target_is_small": target_is_small,
            "target_selected": target_selected,
            "max_support_snippets": max_support_snippets,
            "chars_before": chars_before,
            "chars_after": len(self.context_manager.build_context()),
        }

    def _target_is_small(self, snippets: list[ContextSnippet]) -> bool:
        try:
            target_file = self.repository.get_file(self.repository.target_path)
        except Exception:
            target_file = None

        if target_file is None:
            return True

        try:
            target_text = target_file.read_text()
        except Exception:
            target_text = ""

        normalized_text = self._normalize_text(target_text).rstrip("\n")
        if not normalized_text.strip():
            return True

        target_lines = normalized_text.count("\n") + 1
        if target_lines <= self.settings.final_small_target_max_lines:
            return True
        if len(normalized_text) <= self.settings.final_small_target_max_chars:
            return True

        selected_target = [
            snippet
            for snippet in snippets
            if snippet.path == self.repository.target_path
        ]
        if not selected_target:
            return True

        longest_target = max(selected_target, key=lambda snippet: (snippet.line_count, len(snippet.content)))
        return (
            longest_target.line_count <= max(24, self.settings.final_small_target_max_lines // 2)
            and len(longest_target.content) <= max(800, self.settings.final_small_target_max_chars // 2)
        )

    def _final_soft_char_budget(self, target_is_small: bool) -> int:
        if target_is_small:
            return min(
                self.settings.context_char_budget,
                self.settings.final_context_small_target_soft_char_budget,
            )
        return min(
            self.settings.context_char_budget,
            self.settings.final_context_soft_char_budget,
        )

    def _final_max_support_snippets(
        self,
        *,
        target_is_small: bool,
        target_selected: bool,
    ) -> int:
        if target_is_small and not target_selected:
            return max(0, self.settings.final_context_small_target_max_support_snippets)
        return max(0, self.settings.final_context_max_support_snippets)

    def _prune_final_support_snippets(
        self,
        snippets: list[ContextSnippet],
        *,
        max_support_snippets: int,
    ) -> tuple[list[ContextSnippet], list[str]]:
        support_snippets = [snippet for snippet in snippets if snippet.path != self.repository.target_path]
        if len(support_snippets) <= max_support_snippets:
            return snippets, []

        ranked_support = sorted(
            support_snippets,
            key=lambda snippet: (self._final_strength(snippet), snippet.selection_index),
            reverse=True,
        )
        keep_support_ids = {
            snippet.selection_index
            for snippet in ranked_support[:max_support_snippets]
        }

        kept: list[ContextSnippet] = []
        dropped_refs: list[str] = []
        for snippet in snippets:
            if snippet.path == self.repository.target_path or snippet.selection_index in keep_support_ids:
                kept.append(snippet)
                continue
            dropped_refs.append(snippet.ref)
        return kept, dropped_refs

    def _shrink_support_snippets_to_budget(
        self,
        snippets: list[ContextSnippet],
        *,
        soft_budget: int,
        target_is_small: bool,
    ) -> tuple[list[ContextSnippet], list[str]]:
        if self._render_snippet_list(snippets) <= soft_budget:
            return snippets, []

        line_targets = (72, 56, 40, 32) if target_is_small else (56, 40, 32, 24)
        compressed_refs: list[str] = []
        working = list(snippets)

        support_order = [
            index
            for index, snippet in sorted(
                enumerate(working),
                key=lambda item: (self._final_strength(item[1]), item[1].selection_index),
            )
            if snippet.path != self.repository.target_path
        ]

        for target_lines in line_targets:
            changed_in_pass = False
            for index in support_order:
                if self._render_snippet_list(working) <= soft_budget:
                    return working, compressed_refs
                candidate = working[index]
                compressed = candidate.compressed(target_lines)
                if compressed.ref == candidate.ref:
                    continue
                working[index] = compressed
                compressed_refs.append(f"{candidate.ref}->{compressed.ref}")
                changed_in_pass = True
            if not changed_in_pass:
                continue

        return working, compressed_refs

    def _drop_support_snippets_to_budget(
        self,
        snippets: list[ContextSnippet],
        *,
        soft_budget: int,
    ) -> tuple[list[ContextSnippet], list[str]]:
        if self._render_snippet_list(snippets) <= soft_budget:
            return snippets, []

        working = list(snippets)
        dropped_refs: list[str] = []
        while self._render_snippet_list(working) > soft_budget:
            drop_index = self._final_drop_candidate_index(working)
            if drop_index is None:
                break
            dropped_refs.append(working.pop(drop_index).ref)
        return working, dropped_refs

    def _final_drop_candidate_index(self, snippets: list[ContextSnippet]) -> int | None:
        weakest_index: int | None = None
        weakest_key: tuple[float, int] | None = None

        for index, snippet in enumerate(snippets):
            if snippet.path == self.repository.target_path:
                continue
            candidate_key = (self._final_strength(snippet), -snippet.selection_index)
            if weakest_key is None or candidate_key < weakest_key:
                weakest_key = candidate_key
                weakest_index = index

        return weakest_index

    def _render_snippet_list(self, snippets: list[ContextSnippet]) -> int:
        if not snippets:
            return 0
        return len("\n".join(snippet.render() for snippet in snippets))

    def _merge_overlapping_snippets(
        self,
        snippets: list[ContextSnippet],
    ) -> tuple[list[ContextSnippet], list[str]]:
        if not snippets:
            return [], []

        by_path: dict[str, list[ContextSnippet]] = {}
        for snippet in snippets:
            by_path.setdefault(snippet.path, []).append(snippet)

        merged_all: list[ContextSnippet] = []
        merged_refs: list[str] = []

        for path, items in by_path.items():
            items_sorted = sorted(
                items,
                key=lambda snippet: (snippet.start_line, snippet.end_line, snippet.selection_index),
            )
            cluster: list[ContextSnippet] = []
            cluster_start = 0
            cluster_end = 0

            def flush_cluster() -> None:
                nonlocal cluster, cluster_start, cluster_end
                if not cluster:
                    return
                if len(cluster) == 1:
                    merged_all.append(cluster[0])
                    cluster = []
                    return

                first = min(cluster, key=lambda snippet: snippet.selection_index)
                anchor_values = [snippet.anchor_line for snippet in cluster if snippet.anchor_line is not None]
                anchor_line = int(sum(anchor_values) / len(anchor_values)) if anchor_values else first.anchor_line
                try:
                    snippet_payload = self.repository.read_snippet(
                        path=path,
                        start_line=cluster_start,
                        end_line=cluster_end,
                    )
                except Exception:
                    merged_all.extend(cluster)
                    cluster = []
                    return

                merged_candidate = ContextSnippet(
                    path=path,
                    start_line=int(snippet_payload["start_line"]),
                    end_line=int(snippet_payload["end_line"]),
                    content=str(snippet_payload["content"]),
                    priority=self.repository.path_priority(path),
                    anchor_line=anchor_line,
                    selection_index=first.selection_index,
                )
                merged_clean = self._sanitize_context_snippet(merged_candidate)
                if merged_clean is None:
                    merged_all.extend(cluster)
                    cluster = []
                    return

                source_refs = "+".join(snippet.ref for snippet in cluster)
                merged_refs.append(f"{source_refs}->{merged_clean.ref}")
                merged_all.append(merged_clean)
                cluster = []

            for snippet in items_sorted:
                if not cluster:
                    cluster = [snippet]
                    cluster_start = snippet.start_line
                    cluster_end = snippet.end_line
                    continue

                if snippet.start_line <= cluster_end:
                    cluster.append(snippet)
                    cluster_end = max(cluster_end, snippet.end_line)
                    cluster_start = min(cluster_start, snippet.start_line)
                    continue

                flush_cluster()
                cluster = [snippet]
                cluster_start = snippet.start_line
                cluster_end = snippet.end_line

            flush_cluster()

        merged_all.sort(key=lambda snippet: snippet.selection_index)
        return merged_all, merged_refs

    def _final_output_order_key(self, snippet: ContextSnippet) -> tuple[float, int]:
        # Stronger evidence must appear earlier in the final context.
        return (-self._final_strength(snippet), snippet.selection_index)

    def _final_strength(self, snippet: ContextSnippet) -> float:
        score = 0.0
        path = snippet.path
        target_path = self.repository.target_path
        if path == target_path:
            score += 1000.0
        if path in self.repository.modified_paths:
            score += 350.0
        if self.repository.target_parent and path.startswith(f"{self.repository.target_parent}/"):
            score += 220.0
        score += max(0, 10 - self.repository.path_priority(path)) * 20.0
        score += min(snippet.line_count, 220) * 0.4
        if self.repository.is_support_metadata_path(path):
            score -= 320.0
        if self.repository.is_broad_test_support_path(path):
            score -= 220.0
        return score

    def _context_is_reasonable(self) -> bool:
        context = self.context_manager.build_context()
        if not context:
            return False
        selected_paths = self.context_manager.selected_paths()
        if self.repository.target_path not in selected_paths:
            return False
        required_chars = max(600, self.settings.min_context_chars)
        return len(context) >= required_chars

    def _normalize_unresolved_points(self, unresolved_points: list[str]) -> list[str]:
        if not unresolved_points:
            return []

        normalized: list[str] = []
        seen: set[str] = set()
        for point in unresolved_points:
            text = str(point).strip()
            if not text:
                continue
            key = text.lower()
            if key in seen:
                continue
            seen.add(key)
            normalized.append(text)

        if not normalized:
            return []

        severe = [
            point
            for point in normalized
            if any(marker in point.lower() for marker in SEVERE_UNRESOLVED_MARKERS)
        ]
        if severe:
            return severe

        if self._context_is_reasonable():
            filtered = [
                point
                for point in normalized
                if not any(marker in point.lower() for marker in BENIGN_UNRESOLVED_MARKERS)
            ]
            return filtered

        return normalized

    def _sanitize_context_snippet(self, snippet: ContextSnippet) -> ContextSnippet | None:
        target_path = self.repository.target_path
        if self._path_is_noise(snippet.path, target_path):
            return None

        sanitized_content = self._sanitize_snippet_content(
            snippet.path,
            snippet.content,
            target_snippet=snippet.path == target_path,
        )
        if sanitized_content is None:
            return None

        location = self._locate_content_window(
            snippet.path,
            sanitized_content,
            preferred_start_line=snippet.start_line,
        )
        if location is None:
            return None

        start_line, end_line = location
        anchor_line = snippet.anchor_line
        if anchor_line is None:
            anchor_line = (start_line + end_line) // 2
        anchor_line = min(max(anchor_line, start_line), end_line)

        return replace(
            snippet,
            start_line=start_line,
            end_line=end_line,
            content=sanitized_content,
            anchor_line=anchor_line,
            priority=self.repository.path_priority(snippet.path),
        )

    def _sanitize_snippet_content(
        self,
        path: str,
        content: str,
        *,
        target_snippet: bool,
    ) -> str | None:
        lines = self._normalize_text(content).split("\n")
        if not lines:
            return None

        minimum_lines = 1 if target_snippet else 6
        minimum_code_lines = 1 if target_snippet else 3
        best_block: list[str] | None = None
        best_score = -10_000.0
        block_start = 0

        while block_start < len(lines):
            while block_start < len(lines) and (
                self._line_is_forbidden(lines[block_start]) or not lines[block_start].strip()
            ):
                block_start += 1
            if block_start >= len(lines):
                break

            block_end = block_start
            while block_end < len(lines) and not self._line_is_forbidden(lines[block_end]):
                block_end += 1

            raw_block = lines[block_start:block_end]
            while raw_block and not raw_block[0].strip():
                raw_block = raw_block[1:]
            while raw_block and not raw_block[-1].strip():
                raw_block = raw_block[:-1]

            if len(raw_block) >= minimum_lines:
                code_lines = sum(1 for line in raw_block if self._line_is_code_like(line))
                comment_lines = sum(1 for line in raw_block if self._line_is_comment(line))
                if code_lines >= minimum_code_lines:
                    score = code_lines * 2.2 - comment_lines * 0.2 + min(len(raw_block), 80) * 0.01
                    if target_snippet:
                        score += 0.2
                    if score > best_score:
                        best_score = score
                        best_block = raw_block

            block_start = block_end + 1

        if not best_block:
            return None
        if any(self._line_is_forbidden(line) for line in best_block):
            return None

        sanitized_content = "\n".join(best_block).rstrip("\n")
        if not sanitized_content.strip():
            return None
        return sanitized_content

    def _locate_content_window(
        self,
        path: str,
        content: str,
        *,
        preferred_start_line: int,
    ) -> tuple[int, int] | None:
        try:
            file_text = self.repository.get_file(path).read_text()
        except Exception:
            return None

        normalized_file_text = self._normalize_text(file_text)
        normalized_content = self._normalize_text(content).rstrip("\n")
        if not normalized_content:
            return None

        positions: list[tuple[int, int]] = []
        start_index = 0
        while True:
            match_index = normalized_file_text.find(normalized_content, start_index)
            if match_index < 0:
                break
            start_line = normalized_file_text[:match_index].count("\n") + 1
            end_line = start_line + normalized_content.count("\n")
            positions.append((start_line, end_line))
            start_index = match_index + 1

        if not positions:
            return None

        return min(positions, key=lambda item: abs(item[0] - preferred_start_line))

    def _path_is_noise(self, path: str, target_path: str) -> bool:
        if path == target_path:
            return False
        normalized_path = path.lower().replace("\\", "/")
        parts = [part for part in normalized_path.split("/") if part]
        if not parts:
            return False
        stem = parts[-1].split(".", 1)[0]
        if stem in NOISE_PATH_TOKENS:
            return True
        if any(stem.startswith(f"{token}-") or stem.startswith(f"{token}_") for token in NOISE_PATH_TOKENS):
            return True
        if any(part in NOISE_PATH_TOKENS for part in parts[:-1]):
            return True
        return False


    def _line_is_forbidden(self, line: str) -> bool:
        normalized_line = line.strip().lower()
        if not normalized_line:
            return False
        if "http://" in normalized_line or "https://" in normalized_line:
            return True
        if FORBIDDEN_METADATA_ASSIGNMENT_RE.search(line):
            return True
        if "::" in line and FORBIDDEN_LICENSE_LITERAL_RE.search(line):
            return True
        if self._line_is_comment(line) and FORBIDDEN_LICENSE_LITERAL_RE.search(line):
            return True
        return any(marker in normalized_line for marker in FORBIDDEN_LINE_MARKERS)

    def _line_is_comment(self, line: str) -> bool:
        stripped = line.lstrip()
        return stripped.startswith("#") or stripped.startswith("//") or stripped.startswith("/*") or stripped.startswith("*")

    def _line_is_code_like(self, line: str) -> bool:
        if not line.strip():
            return False
        if self._line_is_forbidden(line):
            return False
        if self._line_is_comment(line):
            return False
        if CODE_LINE_PATTERN.search(line):
            return True
        if "=" in line and not line.lstrip().startswith("#"):
            return True
        if any(char in line for char in "()[]{}"):
            return True
        return line.rstrip().endswith(":")

    def _normalize_text(self, text: str) -> str:
        return text.replace("\r\n", "\n").replace("\r", "\n")

    def _target_anchor_line(self) -> int:
        prefix = self.repository.target_prefix.replace("\r\n", "\n").replace("\r", "\n")
        anchor_line = max(1, len(prefix.split("\n")) + 1)
        try:
            target_file = self.repository.get_file(self.repository.target_path)
            total_lines = len(target_file.read_text().replace("\r\n", "\n").replace("\r", "\n").split("\n"))
            if total_lines > 0:
                return min(anchor_line, total_lines)
        except Exception:
            return anchor_line
        return anchor_line

    def _path_anchor_is_selected(self, path: str, anchor_line: int) -> bool:
        for span in self.context_manager.selected_spans():
            if ":" not in span or "-" not in span:
                continue
            span_path, lines = span.rsplit(":", 1)
            if span_path != path:
                continue
            start_text, end_text = lines.split("-", 1)
            try:
                start_line = int(start_text)
                end_line = int(end_text)
            except ValueError:
                continue
            if start_line <= anchor_line <= end_line:
                return True
        return False
