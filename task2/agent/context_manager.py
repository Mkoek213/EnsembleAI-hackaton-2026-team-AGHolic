from __future__ import annotations

from dataclasses import dataclass, replace

COMPRESSION_LINE_TARGETS = (96, 64, 48, 32, 24, 16, 12)
MIN_COMPRESSED_SNIPPET_LINES = 12
COMPRESSION_RECOMMENDED_BUDGET_RATIO = 0.75
COMPRESSION_RECOMMENDED_REMAINING_CHARS = 4000


@dataclass(frozen=True, slots=True)
class ContextSnippet:
    path: str
    start_line: int
    end_line: int
    content: str
    priority: int = 0
    anchor_line: int | None = None
    selection_index: int = -1

    @property
    def ref(self) -> str:
        return f"{self.path}:{self.start_line}-{self.end_line}"

    @property
    def line_count(self) -> int:
        if not self.content:
            return 0
        return len(self.content.split("\n"))

    def render(self) -> str:
        return f"<|file_sep|>{self.path}\n{self.content}"

    def compressed(self, target_lines: int) -> "ContextSnippet":
        lines = self.content.split("\n")
        if len(lines) <= target_lines:
            return self

        desired_lines = max(MIN_COMPRESSED_SNIPPET_LINES, min(target_lines, len(lines)))
        if desired_lines >= len(lines):
            return self

        absolute_anchor = self.anchor_line
        if absolute_anchor is None:
            absolute_anchor = self.start_line + max(0, len(lines) // 2)

        anchor_index = min(len(lines) - 1, max(0, absolute_anchor - self.start_line))
        start_index = max(0, anchor_index - desired_lines // 2)
        end_index = start_index + desired_lines
        if end_index > len(lines):
            end_index = len(lines)
            start_index = max(0, end_index - desired_lines)

        new_lines = lines[start_index:end_index]
        new_start_line = self.start_line + start_index
        new_end_line = new_start_line + len(new_lines) - 1
        new_anchor_line = min(max(absolute_anchor, new_start_line), new_end_line)
        return replace(
            self,
            start_line=new_start_line,
            end_line=new_end_line,
            content="\n".join(new_lines),
            anchor_line=new_anchor_line,
        )


class ContextManager:
    def __init__(self, *, char_budget: int, max_snippets: int) -> None:
        self.char_budget = char_budget
        self.max_snippets = max_snippets
        self._snippets: list[ContextSnippet] = []
        self._next_selection_index = 0

    def add_snippet(self, snippet: ContextSnippet) -> dict[str, object]:
        snippet = self._normalize_snippet(snippet)
        if any(existing.ref == snippet.ref for existing in self._snippets):
            preview = self.preview()
            preview.update(
                {
                    "status": "ignored",
                    "reason": "duplicate_snippet",
                    "candidate_ref": snippet.ref,
                }
            )
            return preview

        projected_snippets = [*self._snippets, snippet]
        fitted_snippets, compression = self._fit_snippets(projected_snippets)
        if not any(item.selection_index == snippet.selection_index for item in fitted_snippets):
            preview = self.preview()
            preview.update(
                {
                    "status": "rejected",
                    "reason": compression.get("rejection_reason", "char_budget_exceeded"),
                    "candidate_ref": snippet.ref,
                    "projected_chars": len(self._render(projected_snippets)),
                }
            )
            return preview

        self._snippets = fitted_snippets
        preview = self.preview()
        preview.update(
            {
                "status": "added" if not compression["compressed_refs"] and not compression["dropped_refs"] else "added_compressed",
                "candidate_ref": snippet.ref,
                "compressed_refs": compression["compressed_refs"],
                "dropped_refs": compression["dropped_refs"],
            }
        )
        return preview

    def reset(self) -> dict[str, object]:
        self._snippets.clear()
        preview = self.preview()
        preview.update({"status": "reset"})
        return preview

    def preview(
        self,
        *,
        include_context_preview: bool = False,
        context_preview_chars: int = 1600,
    ) -> dict[str, object]:
        context = self._render(self._snippets)
        used_chars = len(context)
        selected_paths = list(dict.fromkeys(snippet.path for snippet in self._snippets))
        selected_spans = [snippet.ref for snippet in self._snippets]
        preview: dict[str, object] = {
            "snippet_count": len(self._snippets),
            "max_snippets": self.max_snippets,
            "used_chars": used_chars,
            "char_budget": self.char_budget,
            "remaining_chars": max(0, self.char_budget - used_chars),
            "fits_budget": used_chars <= self.char_budget,
            "selected_paths": selected_paths,
            "selected_spans": selected_spans,
            "compression_recommended": self._compression_recommended(self._snippets),
            "compression_reason": self._compression_reason(self._snippets, used_chars),
            "compression_line_targets": list(COMPRESSION_LINE_TARGETS),
        }
        if include_context_preview:
            preview["context_preview"] = context[:context_preview_chars]
            preview["context_preview_truncated"] = len(context) > context_preview_chars
        return preview

    def build_context(self) -> str:
        return self._render(self._snippets)

    def snippets(self) -> list[ContextSnippet]:
        return list(self._snippets)

    def replace_snippets(self, snippets: list[ContextSnippet]) -> dict[str, object]:
        normalized: list[ContextSnippet] = []
        max_selection_index = self._next_selection_index - 1
        for snippet in snippets:
            normalized_snippet = self._normalize_snippet(snippet)
            normalized.append(normalized_snippet)
            max_selection_index = max(max_selection_index, normalized_snippet.selection_index)

        self._snippets = normalized
        self._next_selection_index = max(self._next_selection_index, max_selection_index + 1)
        preview = self.preview()
        preview.update({"status": "replaced"})
        return preview

    def selected_paths(self) -> list[str]:
        return list(dict.fromkeys(snippet.path for snippet in self._snippets))

    def selected_spans(self) -> list[str]:
        return [snippet.ref for snippet in self._snippets]

    def compress(
        self,
        *,
        target_lines: int = 32,
        keep_recent_snippets: int = 1,
        drop_if_needed: bool = False,
        only_if_needed: bool = False,
    ) -> dict[str, object]:
        if not self._snippets:
            preview = self.preview()
            preview.update({"status": "ignored", "reason": "no_snippets_selected"})
            return preview

        current_chars = len(self._render(self._snippets))
        if only_if_needed and not self._compression_recommended(self._snippets):
            preview = self.preview()
            preview.update({"status": "ignored", "reason": "compression_not_needed"})
            return preview

        working = list(self._snippets)
        compressed_refs: list[str] = []
        dropped_refs: list[str] = []
        keep_recent = max(0, keep_recent_snippets)
        protected_ids = {
            snippet.selection_index
            for snippet in sorted(
                working,
                key=lambda snippet: snippet.selection_index,
                reverse=True,
            )[:keep_recent]
        }

        order = [
            index
            for index in self._compression_order(working)
            if working[index].selection_index not in protected_ids
        ]
        if not order:
            order = self._compression_order(working)

        bounded_target_lines = max(MIN_COMPRESSED_SNIPPET_LINES, target_lines)
        changed = False
        for index in order:
            candidate = working[index]
            compressed = candidate.compressed(bounded_target_lines)
            if compressed.ref == candidate.ref:
                continue
            working[index] = compressed
            compressed_refs.append(f"{candidate.ref}->{compressed.ref}")
            changed = True

        if drop_if_needed:
            while not self._fits_budget(working) and len(working) > 1:
                drop_index = self._drop_candidate_index(working)
                dropped_refs.append(working.pop(drop_index).ref)
                changed = True

        self._snippets = working
        preview = self.preview()
        preview.update(
            {
                "status": "compressed" if changed else "unchanged",
                "reason": None if changed else "already_compact",
                "compressed_refs": compressed_refs,
                "dropped_refs": dropped_refs,
                "chars_before": current_chars,
                "chars_after": len(self._render(self._snippets)),
            }
        )
        return preview

    def _render(self, snippets: list[ContextSnippet]) -> str:
        if not snippets:
            return ""
        return "\n".join(snippet.render() for snippet in snippets)

    def _normalize_snippet(self, snippet: ContextSnippet) -> ContextSnippet:
        if snippet.selection_index >= 0:
            return snippet
        normalized = replace(snippet, selection_index=self._next_selection_index)
        self._next_selection_index += 1
        return normalized

    def _fit_snippets(
        self,
        snippets: list[ContextSnippet],
    ) -> tuple[list[ContextSnippet], dict[str, list[str] | str]]:
        working = list(snippets)
        compressed_refs: list[str] = []
        dropped_refs: list[str] = []

        while len(working) > self.max_snippets and len(working) > 1:
            drop_index = self._drop_candidate_index(working)
            dropped_refs.append(working.pop(drop_index).ref)

        if self._fits_budget(working):
            return working, {"compressed_refs": compressed_refs, "dropped_refs": dropped_refs}

        for target_lines in COMPRESSION_LINE_TARGETS:
            changed = False
            for index in self._compression_order(working):
                candidate = working[index]
                compressed = candidate.compressed(target_lines)
                if compressed.ref == candidate.ref:
                    continue
                compressed_refs.append(f"{candidate.ref}->{compressed.ref}")
                working[index] = compressed
                changed = True
                if self._fits_budget(working):
                    return working, {"compressed_refs": compressed_refs, "dropped_refs": dropped_refs}
            if not changed and self._fits_budget(working):
                return working, {"compressed_refs": compressed_refs, "dropped_refs": dropped_refs}

        while not self._fits_budget(working) and len(working) > 1:
            drop_index = self._drop_candidate_index(working)
            dropped_refs.append(working.pop(drop_index).ref)

        if self._fits_budget(working):
            return working, {"compressed_refs": compressed_refs, "dropped_refs": dropped_refs}

        return working, {
            "compressed_refs": compressed_refs,
            "dropped_refs": dropped_refs,
            "rejection_reason": "char_budget_exceeded_after_compression",
        }

    def _fits_budget(self, snippets: list[ContextSnippet]) -> bool:
        return len(self._render(snippets)) <= self.char_budget

    def _compression_recommended(self, snippets: list[ContextSnippet]) -> bool:
        used_chars = len(self._render(snippets))
        return self._compression_reason(snippets, used_chars) is not None

    def _compression_reason(self, snippets: list[ContextSnippet], used_chars: int) -> str | None:
        if len(snippets) <= 1:
            return None
        if used_chars >= int(self.char_budget * COMPRESSION_RECOMMENDED_BUDGET_RATIO):
            return "near_char_budget"
        if max((snippet.line_count for snippet in snippets), default=0) > 80 and len(snippets) >= 2:
            return "wide_snippets"
        if len(snippets) >= min(self.max_snippets, 4):
            return "many_snippets"
        if max(0, self.char_budget - used_chars) <= COMPRESSION_RECOMMENDED_REMAINING_CHARS:
            return "low_remaining_chars"
        return None

    def _compression_order(self, snippets: list[ContextSnippet]) -> list[int]:
        newest_index = max(range(len(snippets)), key=lambda idx: snippets[idx].selection_index)
        return sorted(
            range(len(snippets)),
            key=lambda idx: (
                idx == newest_index,
                -snippets[idx].priority,
                snippets[idx].selection_index,
                -snippets[idx].line_count,
            ),
        )

    def _drop_candidate_index(self, snippets: list[ContextSnippet]) -> int:
        return self._compression_order(snippets)[0]
