from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class FileKind(StrEnum):
    TEXT = "text"
    OTHER = "other"


class StrictSchemaModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TaskDatapoint(StrictSchemaModel):
    id: str | None = None
    repo: str
    revision: str
    path: str
    modified: list[str] = Field(default_factory=list)
    prefix: str
    suffix: str
    archive: str | None = None


class ContextAnswer(StrictSchemaModel):
    context: str
    selected_paths: list[str] = Field(default_factory=list)
    selected_spans: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    unresolved_points: list[str] = Field(default_factory=list)


class RetrievedFile(StrictSchemaModel):
    path: str
    kind: FileKind
    size_bytes: int
