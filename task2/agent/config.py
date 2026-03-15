from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
REPOSITORY_ROOT = PROJECT_ROOT.parent.parent

load_dotenv(REPOSITORY_ROOT / ".env")
load_dotenv(PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    base_dir: Path = BASE_DIR
    workspace_dir: Path = BASE_DIR / "workspace"
    artifacts_dir: Path = BASE_DIR / "workspace" / "artifacts"
    completion_eval_dir: Path = BASE_DIR / "workspace" / "completion_evals"
    experiment_name: str = os.getenv("TASK2_EXPERIMENT_NAME", "default")
    agent_model: str = os.getenv("TASK2_AGENT_MODEL", "gpt-5-mini")
    completion_model: str = os.getenv("TASK2_COMPLETION_MODEL", "gpt-5-mini")
    completion_max_output_tokens: int = int(os.getenv("TASK2_COMPLETION_MAX_OUTPUT_TOKENS", "1200"))
    agent_prompt_prefix_tail_lines: int = int(os.getenv("TASK2_AGENT_PROMPT_PREFIX_TAIL_LINES", "80"))
    agent_prompt_suffix_head_lines: int = int(os.getenv("TASK2_AGENT_PROMPT_SUFFIX_HEAD_LINES", "80"))
    context_char_budget: int = int(os.getenv("TASK2_CONTEXT_CHAR_BUDGET", "12000"))
    context_max_snippets: int = int(os.getenv("TASK2_CONTEXT_MAX_SNIPPETS", "5"))
    min_context_chars: int = int(os.getenv("TASK2_MIN_CONTEXT_CHARS", "1600"))
    final_context_soft_char_budget: int = int(
        os.getenv("TASK2_FINAL_CONTEXT_SOFT_CHAR_BUDGET", "10800")
    )
    final_context_small_target_soft_char_budget: int = int(
        os.getenv("TASK2_FINAL_CONTEXT_SMALL_TARGET_SOFT_CHAR_BUDGET", "11800")
    )
    final_context_max_support_snippets: int = int(
        os.getenv("TASK2_FINAL_CONTEXT_MAX_SUPPORT_SNIPPETS", "2")
    )
    final_context_small_target_max_support_snippets: int = int(
        os.getenv("TASK2_FINAL_CONTEXT_SMALL_TARGET_MAX_SUPPORT_SNIPPETS", "3")
    )
    final_small_target_max_lines: int = int(
        os.getenv("TASK2_FINAL_SMALL_TARGET_MAX_LINES", "64")
    )
    final_small_target_max_chars: int = int(
        os.getenv("TASK2_FINAL_SMALL_TARGET_MAX_CHARS", "2200")
    )
    max_agent_steps: int = int(os.getenv("TASK2_MAX_AGENT_STEPS", "16"))
    max_tool_failures_per_datapoint: int = int(
        os.getenv("TASK2_MAX_TOOL_FAILURES_PER_DATAPOINT", "24")
    )
    max_consecutive_tool_failure_steps: int = int(
        os.getenv("TASK2_MAX_CONSECUTIVE_TOOL_FAILURE_STEPS", "6")
    )
    enable_last_chance_finalize: bool = os.getenv(
        "TASK2_ENABLE_LAST_CHANCE_FINALIZE", "1"
    ) not in {"0", "false", "False"}
    last_chance_finalize_steps: int = int(
        os.getenv("TASK2_LAST_CHANCE_FINALIZE_STEPS", "2")
    )
    enable_severe_retry_lane: bool = os.getenv(
        "TASK2_ENABLE_SEVERE_RETRY_LANE", "0"
    ) not in {"0", "false", "False"}
    severe_retry_model: str = os.getenv("TASK2_SEVERE_RETRY_MODEL", "gpt-5")
    severe_retry_max_agent_steps: int = int(
        os.getenv("TASK2_SEVERE_RETRY_MAX_AGENT_STEPS", "28")
    )
    severe_retry_max_input_tokens_per_datapoint: int = int(
        os.getenv("TASK2_SEVERE_RETRY_MAX_INPUT_TOKENS_PER_DATAPOINT", "260000")
    )
    max_input_tokens_per_datapoint: int = int(
        os.getenv("TASK2_MAX_INPUT_TOKENS_PER_DATAPOINT", "160000")
    )
    openai_timeout_seconds: float = float(os.getenv("TASK2_OPENAI_TIMEOUT_SECONDS", "120"))
    openai_max_retries: int = int(os.getenv("TASK2_OPENAI_MAX_RETRIES", "8"))
    openai_sdk_max_retries: int = int(os.getenv("TASK2_OPENAI_SDK_MAX_RETRIES", "2"))
    openai_retry_base_seconds: float = float(os.getenv("TASK2_OPENAI_RETRY_BASE_SECONDS", "0.5"))
    openai_retry_max_seconds: float = float(os.getenv("TASK2_OPENAI_RETRY_MAX_SECONDS", "20"))
    openai_max_concurrent_requests: int = int(os.getenv("TASK2_OPENAI_MAX_CONCURRENT_REQUESTS", "4"))
    openai_prompt_cache_retention: str = os.getenv("TASK2_OPENAI_PROMPT_CACHE_RETENTION", "24h")

    @property
    def openai_api_key(self) -> str:
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_TOKEN")
        if not api_key:
            raise ValueError("OPENAI_API_KEY or OPENAI_API_TOKEN is not set.")
        return api_key

    def ensure_directories(self) -> None:
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.completion_eval_dir.mkdir(parents=True, exist_ok=True)
