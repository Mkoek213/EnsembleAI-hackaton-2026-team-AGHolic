from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
from typing import Callable

from .models import FileKind

TEXT_EXTENSIONS = {
    ".py",
    ".kt",
    ".java",
    ".kts",
    ".md",
    ".txt",
    ".rst",
    ".json",
    ".toml",
    ".yaml",
    ".yml",
    ".ini",
    ".cfg",
    ".gradle",
    ".properties",
    ".xml",
    ".sh",
    ".bat",
}
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
MAX_SEARCH_FILE_BYTES = 512_000
MAX_READ_FILE_LINES = 80
MAX_PRECISE_READ_LINES = int(os.getenv("TASK2_PRECISE_READ_MAX_LINES", "96"))
MAX_SNIPPET_FILE_LINES = int(os.getenv("TASK2_SNIPPET_MAX_LINES", "240"))
MAX_LIST_FILE_ENTRIES = 60
MAX_SEARCH_MATCHES = 8
MAX_PATTERN_MATCHES = 6
MAX_MULTI_PATTERN_MATCHES = 10
MAX_MULTI_PATTERN_QUERIES = 6
MAX_PATTERN_CONTEXT_LINES = 8
MAX_MATCH_PREVIEW_CHARS = 160
MAX_INSPECT_MODIFIED_FILES = 16
MAX_SUGGESTED_SNIPPETS = int(os.getenv("TASK2_MAX_SUGGESTED_SNIPPETS", "6"))
TARGET_WINDOW_PRE_LINES = int(os.getenv("TASK2_TARGET_WINDOW_PRE_LINES", "48"))
TARGET_WINDOW_POST_LINES = int(os.getenv("TASK2_TARGET_WINDOW_POST_LINES", "180"))
TARGET_WINDOW_FALLBACK_PRE_LINES = int(os.getenv("TASK2_TARGET_WINDOW_FALLBACK_PRE_LINES", "64"))
TARGET_WINDOW_FALLBACK_POST_LINES = int(os.getenv("TASK2_TARGET_WINDOW_FALLBACK_POST_LINES", "220"))
TARGET_WINDOW_MAX_LINES = int(os.getenv("TASK2_TARGET_WINDOW_MAX_LINES", "320"))
ALIGNMENT_LINE_TOLERANCE = 6


@dataclass(frozen=True, slots=True)
class LocalCodeFile:
    path: Path
    relative_path: str
    kind: FileKind
    size_bytes: int

    def read_text(self) -> str:
        return self.path.read_text(encoding="utf-8", errors="replace")


class CodeRepository:
    def __init__(
        self,
        root_dir: Path,
        target_path: str,
        modified_paths: list[str],
        language: str,
    ) -> None:
        self.root_dir = root_dir.resolve()
        self.target_path = target_path
        self.modified_paths = modified_paths
        self.language = language
        self.target_parent = str(Path(target_path).parent)
        self.target_prefix = ""
        self.target_suffix = ""
        self._root_safe_files_cache: list[Path] | None = None
        self._text_cache: dict[str, str] = {}

    def inspect_target(
        self,
        prefix_tail_lines: int = 40,
        suffix_head_lines: int = 40,
    ) -> dict[str, object]:
        prefix_lines = self._split_lines(self._normalize_newlines(self.target_prefix))
        suffix_lines = self._split_lines(self._normalize_newlines(self.target_suffix))
        target_exists = (self.root_dir / self.target_path).exists()
        alignment = self._target_alignment_info()
        same_directory = self.list_files(
            directory=self.target_parent if self.target_parent != "." else ".",
            recursive=False,
            extension=self._default_extension(),
            limit=24,
        )
        prioritized_modified = self._prioritize_paths(self.modified_paths)
        return {
            "target_path": self.target_path,
            "target_exists_in_repo": target_exists,
            "target_total_lines": alignment["target_total_lines"],
            "modified_files": prioritized_modified[:MAX_INSPECT_MODIFIED_FILES],
            "modified_files_total": len(self.modified_paths),
            "modified_files_truncated": len(prioritized_modified) > MAX_INSPECT_MODIFIED_FILES,
            "import_hints": self._import_hint_modules()[:8],
            "prefix_total_lines": len(prefix_lines),
            "suffix_total_lines": len(suffix_lines),
            "prefix_suffix_consistent": alignment["consistent"],
            "prefix_suffix_status": alignment["status"],
            "estimated_gap_start_line": alignment["estimated_gap_start_line"],
            "estimated_gap_end_line": alignment["estimated_gap_end_line"],
            "prefix_tail": "\n".join(prefix_lines[-prefix_tail_lines:]),
            "suffix_head": "\n".join(suffix_lines[:suffix_head_lines]),
            "same_directory_entries": same_directory["entries"],
            "suggested_snippets": self.suggest_context_snippets(),
        }

    def list_files(
        self,
        directory: str = ".",
        recursive: bool = False,
        extension: str | None = None,
        limit: int = 100,
    ) -> dict[str, object]:
        resolved_directory = self._resolve_directory(directory)
        limit = max(1, min(limit, MAX_LIST_FILE_ENTRIES))
        entries: list[dict[str, object]] = []

        if recursive:
            for path in self._iter_safe_files(resolved_directory):
                local_file = self._to_local_file(path)
                if extension and path.suffix.lower() != extension.lower():
                    continue
                entries.append(self._entry_payload(local_file, entry_type="file"))
        else:
            for path in sorted(resolved_directory.iterdir()):
                if path.is_symlink():
                    continue
                if path.is_dir():
                    entries.append(
                        {
                            "path": str(path.relative_to(self.root_dir)),
                            "type": "directory",
                            "priority": self._priority_for_path(str(path.relative_to(self.root_dir))),
                        }
                    )
                    continue
                local_file = self._to_local_file(path)
                if extension and path.suffix.lower() != extension.lower():
                    continue
                entries.append(self._entry_payload(local_file, entry_type="file"))

        entries.sort(key=lambda item: (item["priority"], item["path"]))
        return {
            "directory": str(resolved_directory.relative_to(self.root_dir)),
            "recursive": recursive,
            "extension": extension or "",
            "entries": entries[:limit],
        }

    def read_file(
        self,
        path: str,
        start_line: int = 1,
        end_line: int = 80,
    ) -> dict[str, object]:
        return self._read_line_window(
            path=path,
            start_line=start_line,
            end_line=end_line,
            max_lines=MAX_READ_FILE_LINES,
        )

    def read_lines(
        self,
        path: str,
        start_line: int,
        end_line: int,
    ) -> dict[str, object]:
        return self._read_line_window(
            path=path,
            start_line=start_line,
            end_line=end_line,
            max_lines=MAX_PRECISE_READ_LINES,
        )

    def _read_line_window(
        self,
        path: str,
        start_line: int,
        end_line: int,
        *,
        max_lines: int,
    ) -> dict[str, object]:
        document = self.get_file(path)
        if document.kind is not FileKind.TEXT:
            raise ValueError(f"Unsupported file type: {path}")

        lines = self._split_lines(self._read_text(document))
        if lines:
            first_line = min(max(1, start_line), len(lines))
        else:
            first_line = 1
        last_line = max(first_line, end_line)
        last_line = min(last_line, first_line + max_lines - 1)
        selected = lines[first_line - 1 : last_line]
        numbered = [
            f"{line_no}: {line}"
            for line_no, line in enumerate(selected, start=first_line)
        ]
        return {
            "path": document.relative_path,
            "start_line": first_line,
            "end_line": min(last_line, len(lines)),
            "total_lines": len(lines),
            "content": "\n".join(numbered),
        }

    def read_snippet(
        self,
        path: str,
        start_line: int = 1,
        end_line: int = 200,
    ) -> dict[str, object]:
        document = self.get_file(path)
        if document.kind is not FileKind.TEXT:
            raise ValueError(f"Unsupported file type: {path}")

        lines = self._split_lines(self._read_text(document))
        first_line = max(1, start_line)
        last_line = max(first_line, end_line)
        last_line = min(last_line, first_line + MAX_SNIPPET_FILE_LINES - 1)
        selected = lines[first_line - 1 : last_line]
        return {
            "path": document.relative_path,
            "start_line": first_line,
            "end_line": min(last_line, len(lines)),
            "total_lines": len(lines),
            "content": "\n".join(selected),
        }

    def search_files(
        self,
        query: str,
        directory: str | None = None,
        limit: int = 8,
        extension: str | None = None,
    ) -> dict[str, object]:
        limit = max(1, min(limit, MAX_SEARCH_MATCHES))
        normalized_query = query.lower()
        for candidates in self._search_candidate_groups(extension, directory=directory):
            matches = self._search_in_paths(candidates, normalized_query, limit)
            if matches:
                matches.sort(key=lambda item: (item["priority"], item["path"], item["line"]))
                return {"query": query, "directory": directory or "", "matches": matches[:limit]}

        return {"query": query, "directory": directory or "", "matches": []}

    def search_pattern(
        self,
        pattern: str,
        directory: str | None = None,
        limit: int = 6,
        extension: str | None = None,
        context_lines: int = 6,
        regex: bool = False,
        case_sensitive: bool = False,
    ) -> dict[str, object]:
        bounded_limit = max(1, min(limit, MAX_PATTERN_MATCHES))
        bounded_context_lines = max(0, min(context_lines, MAX_PATTERN_CONTEXT_LINES))
        matcher = self._compile_pattern_matcher(
            pattern=pattern,
            regex=regex,
            case_sensitive=case_sensitive,
        )

        for candidates in self._search_candidate_groups(extension, directory=directory):
            matches = self._search_pattern_in_paths(
                candidates=candidates,
                matcher=matcher,
                limit=bounded_limit,
                context_lines=bounded_context_lines,
            )
            if matches:
                matches.sort(key=lambda item: (item["priority"], item["path"], item["match_line"]))
                return {
                    "pattern": pattern,
                    "directory": directory or "",
                    "regex": regex,
                    "case_sensitive": case_sensitive,
                    "context_lines": bounded_context_lines,
                    "matches": matches[:bounded_limit],
                }

        return {
            "pattern": pattern,
            "directory": directory or "",
            "regex": regex,
            "case_sensitive": case_sensitive,
            "context_lines": bounded_context_lines,
            "matches": [],
        }

    def search_patterns(
        self,
        patterns: list[str],
        directory: str | None = None,
        per_pattern_limit: int = 3,
        total_limit: int = 10,
        extension: str | None = None,
        context_lines: int = 6,
        regex: bool = False,
        case_sensitive: bool = False,
    ) -> dict[str, object]:
        normalized_patterns: list[str] = []
        seen_patterns: set[str] = set()
        for pattern in patterns:
            candidate = pattern.strip()
            if not candidate or candidate in seen_patterns:
                continue
            normalized_patterns.append(candidate)
            seen_patterns.add(candidate)
            if len(normalized_patterns) >= MAX_MULTI_PATTERN_QUERIES:
                break

        bounded_context_lines = max(0, min(context_lines, MAX_PATTERN_CONTEXT_LINES))
        bounded_per_pattern_limit = max(1, min(per_pattern_limit, MAX_PATTERN_MATCHES))
        bounded_total_limit = max(1, min(total_limit, MAX_MULTI_PATTERN_MATCHES))
        compiled = [
            (
                pattern,
                self._compile_pattern_matcher(
                    pattern=pattern,
                    regex=regex,
                    case_sensitive=case_sensitive,
                ),
            )
            for pattern in normalized_patterns
        ]

        if not compiled:
            return {
                "patterns": [],
                "directory": directory or "",
                "regex": regex,
                "case_sensitive": case_sensitive,
                "context_lines": bounded_context_lines,
                "matches": [],
            }

        for candidates in self._search_candidate_groups(extension, directory=directory):
            matches = self._search_multiple_patterns_in_paths(
                candidates=candidates,
                compiled_patterns=compiled,
                per_pattern_limit=bounded_per_pattern_limit,
                total_limit=bounded_total_limit,
                context_lines=bounded_context_lines,
            )
            if matches:
                matches.sort(key=lambda item: (item["priority"], item["path"], item["match_line"]))
                return {
                    "patterns": normalized_patterns,
                    "directory": directory or "",
                    "regex": regex,
                    "case_sensitive": case_sensitive,
                    "context_lines": bounded_context_lines,
                    "matches": matches[:bounded_total_limit],
                }

        return {
            "patterns": normalized_patterns,
            "directory": directory or "",
            "regex": regex,
            "case_sensitive": case_sensitive,
            "context_lines": bounded_context_lines,
            "matches": [],
        }

    def get_file(self, path: str) -> LocalCodeFile:
        resolved = self._resolve_file(path)
        local_file = self._to_local_file(resolved)
        if local_file.kind is FileKind.OTHER:
            raise ValueError(f"Unsupported file type: {path}")
        return local_file

    def path_priority(self, path: str) -> int:
        return self._priority_for_path(path)

    def is_support_metadata_path(self, path: str) -> bool:
        return path != self.target_path and self._is_metadata_path(path)

    def is_broad_test_support_path(self, path: str) -> bool:
        return path != self.target_path and self._is_broad_test_path(path)

    def _resolve_directory(self, directory: str) -> Path:
        candidate = self.root_dir / directory
        resolved = candidate.resolve()
        if not resolved.exists() or not resolved.is_dir():
            raise FileNotFoundError(directory)
        self._ensure_within_root(resolved)
        return resolved

    def _resolve_file(self, path: str) -> Path:
        candidate = self.root_dir / path
        resolved = candidate.resolve()
        if not resolved.exists() or not resolved.is_file():
            raise FileNotFoundError(path)
        self._ensure_within_root(resolved)
        return resolved

    def _ensure_within_root(self, path: Path) -> None:
        if not path.is_relative_to(self.root_dir.resolve()):
            raise ValueError(f"Path outside repository root: {path}")

    def _iter_safe_files(self, directory: Path) -> list[Path]:
        if directory == self.root_dir and self._root_safe_files_cache is not None:
            return self._root_safe_files_cache

        safe_files: list[Path] = []
        for candidate in sorted(directory.rglob("*")):
            if candidate.is_symlink() or not candidate.is_file():
                continue
            resolved = candidate.resolve()
            try:
                self._ensure_within_root(resolved)
            except ValueError:
                continue
            safe_files.append(candidate)

        if directory == self.root_dir:
            self._root_safe_files_cache = safe_files
        return safe_files

    def _read_text(self, document: LocalCodeFile) -> str:
        if document.size_bytes <= MAX_SEARCH_FILE_BYTES:
            cached = self._text_cache.get(document.relative_path)
            if cached is not None:
                return cached
            text = document.read_text()
            self._text_cache[document.relative_path] = text
            return text
        return document.read_text()

    def _search_candidate_groups(self, extension: str | None, directory: str | None = None) -> list[list[Path]]:
        search_root = self.root_dir if not directory else self._resolve_directory(directory)
        all_files = self._iter_safe_files(search_root)
        selected: list[list[Path]] = []
        seen: set[str] = set()

        def add_group(paths: list[Path]) -> None:
            group: list[Path] = []
            for path in paths:
                rel = str(path.relative_to(self.root_dir))
                if rel in seen:
                    continue
                if extension and path.suffix.lower() != extension.lower():
                    continue
                seen.add(rel)
                group.append(path)
            if group:
                selected.append(group)

        add_group(self._target_file_paths(search_root))
        add_group(self._import_hint_files(search_root))
        add_group(self._same_directory_paths(search_root))
        add_group(self._import_hint_directory_paths(search_root))
        add_group(self._prioritized_modified_search_paths(search_root))

        package_prefix = self.target_path.split("/", 1)[0]
        add_group(
            [
                path
                for path in all_files
                if str(path.relative_to(self.root_dir)).startswith(f"{package_prefix}/")
            ]
        )
        add_group(all_files)
        return selected

    def _target_file_paths(self, search_root: Path) -> list[Path]:
        try:
            target = self._resolve_file(self.target_path)
        except FileNotFoundError:
            return []
        return [target] if target.is_relative_to(search_root) else []

    def _same_directory_paths(self, search_root: Path) -> list[Path]:
        if self.target_parent == ".":
            same_dir = self.root_dir
        else:
            try:
                same_dir = self._resolve_directory(self.target_parent)
            except FileNotFoundError:
                return []
        if not same_dir.is_relative_to(search_root) and search_root != same_dir:
            return []
        paths: list[Path] = []
        for path in sorted(same_dir.iterdir()):
            if path.is_symlink() or not path.is_file():
                continue
            if path.is_relative_to(search_root):
                paths.append(path)
        return paths

    def _prioritized_modified_search_paths(self, search_root: Path, limit: int = 12) -> list[Path]:
        paths: list[Path] = []
        for relative_path in self._prioritize_paths(self.modified_paths)[:limit]:
            try:
                resolved = self._resolve_file(relative_path)
            except FileNotFoundError:
                continue
            if resolved.is_relative_to(search_root):
                paths.append(resolved)
        return paths

    def _import_hint_files(self, search_root: Path) -> list[Path]:
        paths: list[Path] = []
        for relative_path in self._import_hint_path_strings():
            candidate = self.root_dir / relative_path
            if not candidate.exists() or not candidate.is_file():
                continue
            resolved = candidate.resolve()
            self._ensure_within_root(resolved)
            if resolved.is_relative_to(search_root):
                paths.append(resolved)
        return paths

    def _import_hint_directory_paths(self, search_root: Path, per_directory_limit: int = 40) -> list[Path]:
        paths: list[Path] = []
        seen_dirs: set[Path] = set()
        for relative_dir in self._import_hint_directories():
            candidate = self.root_dir / relative_dir
            if not candidate.exists() or not candidate.is_dir():
                continue
            resolved_dir = candidate.resolve()
            self._ensure_within_root(resolved_dir)
            if not resolved_dir.is_relative_to(search_root) and search_root != resolved_dir:
                continue
            if resolved_dir in seen_dirs:
                continue
            seen_dirs.add(resolved_dir)
            count = 0
            for path in sorted(resolved_dir.rglob("*")):
                if path.is_symlink() or not path.is_file():
                    continue
                if not path.is_relative_to(search_root) and search_root != resolved_dir:
                    continue
                paths.append(path)
                count += 1
                if count >= per_directory_limit:
                    break
        return paths

    def _search_in_paths(
        self,
        candidates: list[Path],
        normalized_query: str,
        limit: int,
    ) -> list[dict[str, object]]:
        matches: list[dict[str, object]] = []

        for path in candidates:
            if path.stat().st_size > MAX_SEARCH_FILE_BYTES:
                continue

            local_file = self._to_local_file(path)
            if local_file.kind is not FileKind.TEXT:
                continue

            lines = self._split_lines(self._read_text(local_file))
            for line_number, line in enumerate(lines, start=1):
                if normalized_query not in line.lower():
                    continue
                matches.append(
                    {
                        "path": local_file.relative_path,
                        "line": line_number,
                        "snippet": line.strip(),
                        "priority": self._priority_for_path(local_file.relative_path),
                    }
                )
                if len(matches) >= limit:
                    return matches

        return matches

    def _search_pattern_in_paths(
        self,
        candidates: list[Path],
        matcher: Callable[[str], bool],
        limit: int,
        context_lines: int,
    ) -> list[dict[str, object]]:
        matches: list[dict[str, object]] = []

        for path in candidates:
            if path.stat().st_size > MAX_SEARCH_FILE_BYTES:
                continue

            local_file = self._to_local_file(path)
            if local_file.kind is not FileKind.TEXT:
                continue

            lines = self._split_lines(self._read_text(local_file))
            for line_number, line in enumerate(lines, start=1):
                if not matcher(line):
                    continue

                start_line = max(1, line_number - context_lines)
                end_line = min(len(lines), line_number + context_lines)
                matches.append(
                    {
                        "path": local_file.relative_path,
                        "match_line": line_number,
                        "start_line": start_line,
                        "end_line": end_line,
                        "preview": line.strip()[:MAX_MATCH_PREVIEW_CHARS],
                        "priority": self._priority_for_path(local_file.relative_path),
                    }
                )
                if len(matches) >= limit:
                    return matches

        return matches

    def _search_multiple_patterns_in_paths(
        self,
        candidates: list[Path],
        compiled_patterns: list[tuple[str, Callable[[str], bool]]],
        per_pattern_limit: int,
        total_limit: int,
        context_lines: int,
    ) -> list[dict[str, object]]:
        matches: list[dict[str, object]] = []
        counts: dict[str, int] = {pattern: 0 for pattern, _ in compiled_patterns}
        by_key: dict[tuple[str, int, int], dict[str, object]] = {}

        for path in candidates:
            if path.stat().st_size > MAX_SEARCH_FILE_BYTES:
                continue

            local_file = self._to_local_file(path)
            if local_file.kind is not FileKind.TEXT:
                continue

            lines = self._split_lines(self._read_text(local_file))
            for line_number, line in enumerate(lines, start=1):
                matched_patterns = [
                    pattern
                    for pattern, matcher in compiled_patterns
                    if counts[pattern] < per_pattern_limit and matcher(line)
                ]
                if not matched_patterns:
                    continue

                start_line = max(1, line_number - context_lines)
                end_line = min(len(lines), line_number + context_lines)
                key = (local_file.relative_path, start_line, end_line)
                existing = by_key.get(key)
                if existing is None:
                    existing = {
                        "path": local_file.relative_path,
                        "match_line": line_number,
                        "start_line": start_line,
                        "end_line": end_line,
                        "preview": line.strip()[:MAX_MATCH_PREVIEW_CHARS],
                        "patterns": [],
                        "priority": self._priority_for_path(local_file.relative_path),
                    }
                    by_key[key] = existing
                    matches.append(existing)

                for pattern in matched_patterns:
                    if pattern in existing["patterns"]:
                        continue
                    existing["patterns"].append(pattern)
                    counts[pattern] += 1

                if len(matches) >= total_limit:
                    return matches[:total_limit]

        return matches[:total_limit]

    def _compile_pattern_matcher(
        self,
        *,
        pattern: str,
        regex: bool,
        case_sensitive: bool,
    ) -> Callable[[str], bool]:
        if regex:
            flags = 0 if case_sensitive else re.IGNORECASE
            compiled = re.compile(pattern, flags=flags)
            return lambda line: bool(compiled.search(line))

        if case_sensitive:
            return lambda line: pattern in line

        lowered_pattern = pattern.lower()
        return lambda line: lowered_pattern in line.lower()

    def suggest_context_snippets(self, limit: int = 4) -> list[dict[str, object]]:
        limit = max(1, min(limit, MAX_SUGGESTED_SNIPPETS))
        suggestions: list[dict[str, object]] = []
        seen: set[tuple[str, int, int]] = set()
        alignment = self._target_alignment_info()
        prefix_suffix_mismatch = not bool(alignment.get("consistent", True))

        def add_suggestion(candidate: dict[str, object] | None) -> None:
            if not candidate:
                return
            key = (
                str(candidate["path"]),
                int(candidate["start_line"]),
                int(candidate["end_line"]),
            )
            if key in seen:
                return
            seen.add(key)
            suggestions.append(candidate)

        add_suggestion(self._target_window_snippet())
        if prefix_suffix_mismatch:
            add_suggestion(self._target_tail_snippet())

        prioritized_modified = [
            path
            for path in self._prioritize_paths(self.modified_paths)
            if path != self.target_path
        ]
        for weak_paths_last in (False, True):
            for path in prioritized_modified:
                if self._support_path_is_weak_fallback(path) != weak_paths_last:
                    continue
                add_suggestion(
                    self._file_head_snippet(
                        path,
                        max_lines=self._suggested_head_lines(path, reason="modified_file"),
                        reason="modified_file",
                    )
                )
                if len(suggestions) >= limit:
                    return suggestions[:limit]

        if not prefix_suffix_mismatch:
            import_hint_paths = self._import_hint_path_strings()
        else:
            # Prefix/suffix can come from a stale file revision; trust imports less in that case.
            import_hint_paths = self._import_hint_path_strings()[:4]

        for weak_paths_last in (False, True):
            for path in import_hint_paths:
                if path == self.target_path:
                    continue
                if self._support_path_is_weak_fallback(path) != weak_paths_last:
                    continue
                add_suggestion(self._file_head_snippet(path, max_lines=48, reason="import_hint"))
                if len(suggestions) >= limit:
                    return suggestions[:limit]

        if self.target_parent != ".":
            try:
                same_dir = self._resolve_directory(self.target_parent)
            except FileNotFoundError:
                same_dir = None
            if same_dir is not None:
                siblings = sorted(same_dir.iterdir())
                for weak_paths_last in (False, True):
                    for sibling in siblings:
                        if sibling.is_symlink() or not sibling.is_file():
                            continue
                        rel = str(sibling.relative_to(self.root_dir))
                        if rel == self.target_path or rel in self.modified_paths:
                            continue
                        if sibling.suffix.lower() != self._default_extension().lower():
                            continue
                        if self._support_path_is_weak_fallback(rel) != weak_paths_last:
                            continue
                        add_suggestion(
                            self._file_head_snippet(
                                rel,
                                max_lines=self._suggested_head_lines(rel, reason="same_directory"),
                                reason="same_directory",
                            )
                        )
                        if len(suggestions) >= limit:
                            break
                    if len(suggestions) >= limit:
                        break

        return suggestions[:limit]

    def _target_window_snippet(self) -> dict[str, object] | None:
        try:
            document = self.get_file(self.target_path)
        except FileNotFoundError:
            return None

        lines = self._split_lines(self._read_text(document))
        if not lines or not any(line.strip() for line in lines):
            return None

        normalized_prefix = self._normalize_newlines(self.target_prefix)
        normalized_suffix = self._normalize_newlines(self.target_suffix)
        prefix_lines = self._split_lines(normalized_prefix)
        suffix_lines = self._split_lines(normalized_suffix)
        alignment = self._target_alignment_from_counts(
            total_lines=len(lines),
            prefix_count=len(prefix_lines),
            suffix_count=len(suffix_lines),
        )
        prefix_probe = self._probe_lines(normalized_prefix, tail=True)
        suffix_probe = self._probe_lines(normalized_suffix, tail=False)

        expected_prefix_index = max(0, min(len(lines) - 1, len(prefix_lines) - 1))
        expected_suffix_index = max(0, min(len(lines) - 1, len(lines) - len(suffix_lines)))
        prefix_match = self._find_line_sequence(lines, prefix_probe, preferred_index=expected_prefix_index)
        suffix_match = self._find_line_sequence(lines, suffix_probe, preferred_index=expected_suffix_index)
        if prefix_match is None and prefix_probe:
            prefix_match = self._find_line_sequence_fuzzy(
                lines,
                prefix_probe,
                preferred_index=expected_prefix_index,
            )
        if suffix_match is None and suffix_probe:
            suffix_match = self._find_line_sequence_fuzzy(
                lines,
                suffix_probe,
                preferred_index=expected_suffix_index,
            )

        if not alignment["consistent"]:
            anchor_line = self._nearest_code_anchor(lines, int(alignment["anchor_line"]))
            start_line = max(1, anchor_line - TARGET_WINDOW_FALLBACK_PRE_LINES)
            end_line = min(len(lines), anchor_line + TARGET_WINDOW_FALLBACK_POST_LINES)
        elif prefix_match and suffix_match and prefix_match[1] < suffix_match[0]:
            start_line = max(1, prefix_match[0] + 1 - TARGET_WINDOW_PRE_LINES)
            end_line = min(len(lines), suffix_match[1] + 1 + TARGET_WINDOW_PRE_LINES)
        elif prefix_match:
            start_line = max(1, prefix_match[0] + 1 - TARGET_WINDOW_PRE_LINES)
            end_line = min(len(lines), prefix_match[1] + 1 + TARGET_WINDOW_POST_LINES)
        elif suffix_match:
            start_line = max(1, suffix_match[0] + 1 - TARGET_WINDOW_POST_LINES)
            end_line = min(len(lines), suffix_match[1] + 1 + TARGET_WINDOW_PRE_LINES)
        else:
            # Anchor fallback: estimate the missing region location from prefix/suffix lengths.
            prefix_anchor = max(1, min(len(lines), len(prefix_lines) + 1))
            suffix_anchor = max(1, min(len(lines), len(lines) - len(suffix_lines) + 1))
            if suffix_anchor >= prefix_anchor:
                start_line = max(1, prefix_anchor - TARGET_WINDOW_FALLBACK_PRE_LINES)
                end_line = min(len(lines), suffix_anchor + TARGET_WINDOW_PRE_LINES)
            else:
                focus_line = prefix_anchor
                start_line = max(1, focus_line - TARGET_WINDOW_FALLBACK_PRE_LINES)
                end_line = min(len(lines), focus_line + TARGET_WINDOW_FALLBACK_POST_LINES)

        if end_line - start_line > TARGET_WINDOW_MAX_LINES:
            end_line = start_line + TARGET_WINDOW_MAX_LINES

        return {
            "path": self.target_path,
            "start_line": start_line,
            "end_line": end_line,
            "reason": "target_window" if alignment["consistent"] else "target_window_prefix_suffix_mismatch",
        }

    def _target_tail_snippet(self, max_lines: int = 96) -> dict[str, object] | None:
        try:
            document = self.get_file(self.target_path)
        except FileNotFoundError:
            return None

        lines = self._split_lines(self._read_text(document))
        if not lines or not any(line.strip() for line in lines):
            return None
        if len(lines) <= max_lines:
            return None

        start_line = max(1, len(lines) - max_lines + 1)
        return {
            "path": document.relative_path,
            "start_line": start_line,
            "end_line": len(lines),
            "reason": "target_tail",
        }

    def _file_head_snippet(self, path: str, max_lines: int, reason: str) -> dict[str, object] | None:
        try:
            document = self.get_file(path)
        except FileNotFoundError:
            return None

        lines = self._split_lines(self._read_text(document))
        if not lines or not any(line.strip() for line in lines):
            return None

        return {
            "path": document.relative_path,
            "start_line": 1,
            "end_line": min(len(lines), max_lines),
            "reason": reason,
        }

    def _prioritize_paths(self, paths: list[str]) -> list[str]:
        unique_paths = list(dict.fromkeys(paths))
        return sorted(
            unique_paths,
            key=lambda path: (
                path != self.target_path,
                self._priority_for_path(path),
                -self._shared_parent_depth(path),
                0 if Path(path).suffix.lower() == self._default_extension().lower() else 1,
                path,
            ),
        )

    def _shared_parent_depth(self, path: str) -> int:
        target_parts = Path(self.target_path).parent.parts
        candidate_parts = Path(path).parent.parts
        shared = 0
        for left, right in zip(target_parts, candidate_parts):
            if left != right:
                break
            shared += 1
        return shared

    def _suggested_head_lines(self, path: str, *, reason: str) -> int:
        shared_depth = self._shared_parent_depth(path)
        if path == self.target_path:
            return 160
        if reason == "modified_file":
            if shared_depth >= 2:
                return 96
            if shared_depth >= 1:
                return 88
            return 72
        if reason == "same_directory":
            return 72
        return 64

    def _support_path_is_weak_fallback(self, path: str) -> bool:
        return self.is_support_metadata_path(path) or self.is_broad_test_support_path(path)

    def _target_alignment_info(self) -> dict[str, object]:
        prefix_count = len(self._split_lines(self._normalize_newlines(self.target_prefix)))
        suffix_count = len(self._split_lines(self._normalize_newlines(self.target_suffix)))
        try:
            target_file = self.get_file(self.target_path)
            total_lines = len(self._split_lines(self._read_text(target_file)))
        except Exception:
            total_lines = 0
        return self._target_alignment_from_counts(
            total_lines=total_lines,
            prefix_count=prefix_count,
            suffix_count=suffix_count,
        )

    def _target_alignment_from_counts(
        self,
        *,
        total_lines: int,
        prefix_count: int,
        suffix_count: int,
    ) -> dict[str, object]:
        if total_lines <= 0:
            return {
                "consistent": True,
                "status": "target_missing_or_empty",
                "estimated_gap_start_line": max(1, prefix_count + 1),
                "estimated_gap_end_line": max(0, total_lines - suffix_count),
                "anchor_line": max(1, prefix_count + 1),
                "target_total_lines": total_lines,
            }

        estimated_gap_start_line = max(1, prefix_count + 1)
        estimated_gap_end_line = total_lines - suffix_count
        prefix_in_range = prefix_count <= total_lines + ALIGNMENT_LINE_TOLERANCE
        suffix_in_range = suffix_count <= total_lines + ALIGNMENT_LINE_TOLERANCE
        gap_non_inverted = estimated_gap_end_line >= estimated_gap_start_line - 1
        consistent = prefix_in_range and suffix_in_range and gap_non_inverted

        status_parts: list[str] = []
        if not prefix_in_range:
            status_parts.append("prefix_out_of_range")
        if not suffix_in_range:
            status_parts.append("suffix_out_of_range")
        if not gap_non_inverted:
            status_parts.append("estimated_gap_inverted")
        status = "aligned" if not status_parts else "|".join(status_parts)

        anchor_line = min(
            total_lines,
            max(
                1,
                min(max(1, estimated_gap_start_line), max(1, estimated_gap_end_line)),
            ),
        )
        if not consistent:
            anchor_line = min(total_lines, max(1, min(prefix_count + 1, total_lines)))

        return {
            "consistent": consistent,
            "status": status,
            "estimated_gap_start_line": estimated_gap_start_line,
            "estimated_gap_end_line": estimated_gap_end_line,
            "anchor_line": anchor_line,
            "target_total_lines": total_lines,
        }

    def _nearest_code_anchor(self, lines: list[str], preferred_line: int) -> int:
        if not lines:
            return 1
        preferred = max(1, min(len(lines), preferred_line))
        anchor_candidates = [
            index + 1
            for index, line in enumerate(lines)
            if re.match(r"\s*(def|class)\s+[A-Za-z_]\w*", line)
        ]
        if not anchor_candidates:
            return preferred
        return min(anchor_candidates, key=lambda line_no: abs(line_no - preferred))

    def _probe_lines(self, text: str, *, tail: bool, limit: int = 12) -> list[str]:
        lines = [line for line in self._split_lines(self._normalize_newlines(text)) if line.strip()]
        if tail:
            return lines[-limit:]
        return lines[:limit]

    def _find_line_sequence(
        self,
        haystack: list[str],
        needle: list[str],
        preferred_index: int | None = None,
    ) -> tuple[int, int] | None:
        if not needle:
            return None

        max_window = min(len(needle), 12)
        min_window = min(3, max_window)
        for window in range(max_window, min_window - 1, -1):
            probe = needle[-window:] if len(needle) > window else needle
            candidate_matches: list[tuple[int, int]] = []
            for index in range(0, len(haystack) - len(probe) + 1):
                if haystack[index : index + len(probe)] == probe:
                    if preferred_index is None:
                        return index, index + len(probe) - 1
                    candidate_matches.append((index, index + len(probe) - 1))
            if candidate_matches:
                return min(
                    candidate_matches,
                    key=lambda item: abs(item[0] - preferred_index),
                )
        return None

    def _find_line_sequence_fuzzy(
        self,
        haystack: list[str],
        needle: list[str],
        preferred_index: int | None = None,
    ) -> tuple[int, int] | None:
        normalized_haystack = [self._normalize_line_for_match(line) for line in haystack]
        normalized_needle = [self._normalize_line_for_match(line) for line in needle]
        normalized_needle = [line for line in normalized_needle if line]
        if not normalized_needle:
            return None

        return self._find_line_sequence(
            normalized_haystack,
            normalized_needle,
            preferred_index=preferred_index,
        )

    def _normalize_line_for_match(self, line: str) -> str:
        compact = " ".join(line.strip().split())
        return compact

    def _to_local_file(self, path: Path) -> LocalCodeFile:
        return LocalCodeFile(
            path=path,
            relative_path=str(path.relative_to(self.root_dir)),
            kind=self._detect_kind(path),
            size_bytes=path.stat().st_size,
        )

    def _detect_kind(self, path: Path) -> FileKind:
        if path.suffix.lower() in TEXT_EXTENSIONS:
            return FileKind.TEXT
        return FileKind.OTHER

    def _entry_payload(self, document: LocalCodeFile, entry_type: str) -> dict[str, object]:
        return {
            "path": document.relative_path,
            "type": entry_type,
            "kind": document.kind,
            "size_bytes": document.size_bytes,
            "priority": self._priority_for_path(document.relative_path),
        }

    def _priority_for_path(self, path: str) -> int:
        if path == self.target_path:
            return 0
        penalty = self._support_path_penalty(path)
        if path in self._import_hint_path_strings():
            return 1 + penalty
        if path in self.modified_paths:
            base_priority = 2 if self.target_parent and path.startswith(f"{self.target_parent}/") else 3
            return base_priority + penalty
        if self.target_parent and path.startswith(f"{self.target_parent}/"):
            return 2 + penalty
        for relative_dir in self._import_hint_directories():
            if path.startswith(f"{relative_dir}/"):
                return 2 + penalty
        return 4 + penalty

    def _support_path_penalty(self, path: str) -> int:
        penalty = 0
        if self._is_metadata_path(path):
            penalty += 4
        if self._is_broad_test_path(path):
            penalty += 3
        return penalty

    def _is_metadata_path(self, path: str) -> bool:
        normalized = path.lower().replace("\\", "/")
        name = Path(normalized).name
        stem = Path(normalized).stem
        parts = [part for part in normalized.split("/") if part]
        if name in SUPPORT_METADATA_FILENAMES:
            return True
        if stem in NOISE_PATH_TOKENS:
            return True
        if any(part in NOISE_PATH_TOKENS for part in parts[:-1]):
            return True
        return False

    def _is_broad_test_path(self, path: str) -> bool:
        if self._is_test_path(self.target_path):
            return False
        if not self._is_test_path(path):
            return False
        return self._shared_parent_depth(path) < 2

    def _is_test_path(self, path: str) -> bool:
        normalized = path.lower().replace("\\", "/")
        parts = [part for part in normalized.split("/") if part]
        name = Path(normalized).name
        return any(part in {"test", "tests", "testing"} for part in parts) or name.startswith("test_")

    def _default_extension(self) -> str:
        return ".py" if self.language == "python" else ".kt"

    def _normalize_newlines(self, text: str) -> str:
        return text.replace("\r\n", "\n").replace("\r", "\n")

    def _split_lines(self, text: str) -> list[str]:
        return self._normalize_newlines(text).split("\n")

    def _import_hint_path_strings(self) -> list[str]:
        paths: list[str] = []
        seen: set[str] = set()
        for module in self._import_hint_modules():
            base = module.replace(".", "/")
            for candidate in (f"{base}.py", f"{base}/__init__.py"):
                if candidate in seen:
                    continue
                candidate_path = self.root_dir / candidate
                if candidate_path.exists():
                    seen.add(candidate)
                    paths.append(candidate)
        return paths

    def _import_hint_directories(self) -> list[str]:
        directories: list[str] = []
        seen: set[str] = set()
        for module in self._import_hint_modules():
            base = module.replace(".", "/")
            candidate = self.root_dir / base
            if candidate.exists() and candidate.is_dir() and base not in seen:
                seen.add(base)
                directories.append(base)
        return directories

    def _import_hint_modules(self) -> list[str]:
        text = f"{self.target_prefix}\n{self.target_suffix}"
        modules: list[str] = []
        seen: set[str] = set()
        for line in self._split_lines(self._normalize_newlines(text)):
            stripped = line.strip()
            import_match = re.match(r"import\s+([A-Za-z_][\w.]*)", stripped)
            if import_match:
                module = import_match.group(1)
                if module not in seen:
                    seen.add(module)
                    modules.append(module)
            from_match = re.match(r"from\s+([A-Za-z_][\w.]*)\s+import\s+", stripped)
            if from_match:
                module = from_match.group(1)
                if module not in seen:
                    seen.add(module)
                    modules.append(module)
        return modules
