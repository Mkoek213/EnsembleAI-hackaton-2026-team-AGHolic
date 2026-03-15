from __future__ import annotations

import atexit
import os
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

try:
    from langfuse import get_client, propagate_attributes
except ImportError:
    get_client = None
    propagate_attributes = None

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
REPOSITORY_ROOT = PROJECT_ROOT.parent.parent

load_dotenv(REPOSITORY_ROOT / ".env")
load_dotenv(PROJECT_ROOT / ".env")


class _DummyObservation:
    def __enter__(self) -> "_DummyObservation":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        return False

    def update(self, *_: Any, **__: Any) -> None:
        return


class _DummyClient:
    def start_as_current_observation(self, *_: Any, **__: Any) -> _DummyObservation:
        return _DummyObservation()

    def flush(self) -> None:
        return

    def shutdown(self) -> None:
        return

    def get_current_trace_id(self) -> None:
        return None

    def get_trace_url(self, *_: Any, **__: Any) -> None:
        return None

    def set_current_trace_io(self, *_: Any, **__: Any) -> None:
        return

    def score_current_trace(self, *_: Any, **__: Any) -> None:
        return

    def score_current_span(self, *_: Any, **__: Any) -> None:
        return

    def update_current_span(self, *_: Any, **__: Any) -> None:
        return


@contextmanager
def _dummy_propagate_attributes(**_: Any) -> Any:
    yield


def langfuse_is_enabled() -> bool:
    return (
        os.getenv("TASK2_DISABLE_LANGFUSE") != "1"
        and bool(os.getenv("LANGFUSE_PUBLIC_KEY"))
        and bool(os.getenv("LANGFUSE_SECRET_KEY"))
        and get_client is not None
    )


@lru_cache(maxsize=1)
def get_langfuse_client() -> Any:
    if not langfuse_is_enabled():
        return _DummyClient()
    try:
        return get_client()
    except Exception:
        return _DummyClient()


def get_propagate_attributes() -> Any:
    return propagate_attributes or _dummy_propagate_attributes


def flush_langfuse() -> None:
    try:
        get_langfuse_client().flush()
    except Exception:
        return


def shutdown_langfuse() -> None:
    try:
        get_langfuse_client().shutdown()
    except Exception:
        return


atexit.register(shutdown_langfuse)
