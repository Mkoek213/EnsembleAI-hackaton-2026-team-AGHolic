from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from pathlib import Path

from .agent_tools import AgentToolRuntime
from .config import Settings
from .documents import CodeRepository
from .models import ContextAnswer, TaskDatapoint
from .observability import flush_langfuse, get_langfuse_client, get_propagate_attributes
from .openai_service import OpenAIService
from .prompts import build_agent_input


@dataclass(frozen=True)
class RunConfig:
    datapoint: TaskDatapoint
    repo_root: Path
    language: str
    stage: str
    artifact_dir: Path | None = None


class TaskSolver:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.openai = OpenAIService(settings)

    def run(self, run_config: RunConfig) -> ContextAnswer:
        self.settings.ensure_directories()
        langfuse = get_langfuse_client()
        propagate_attributes = get_propagate_attributes()
        run_id = uuid.uuid4().hex

        repository = CodeRepository(
            root_dir=run_config.repo_root,
            target_path=run_config.datapoint.path,
            modified_paths=run_config.datapoint.modified,
            language=run_config.language,
        )
        repository.target_prefix = run_config.datapoint.prefix
        repository.target_suffix = run_config.datapoint.suffix

        runtime = AgentToolRuntime(
            settings=self.settings,
            run_config=run_config,
            repository=repository,
        )
        tools = runtime.tool_definitions()
        task_input = build_agent_input(
            datapoint=run_config.datapoint,
            language=run_config.language,
            stage=run_config.stage,
            prefix_tail_lines=self.settings.agent_prompt_prefix_tail_lines,
            suffix_head_lines=self.settings.agent_prompt_suffix_head_lines,
        )
        langfuse_input = self._langfuse_run_input(run_config)
        langfuse_metadata = {
            "datapoint_id": run_config.datapoint.id or "",
            "experiment_name": self.settings.experiment_name,
            "target_path": run_config.datapoint.path,
            "repo": run_config.datapoint.repo,
            "revision": run_config.datapoint.revision,
            "agent_model": self.settings.agent_model,
            "repo_root": str(run_config.repo_root),
        }

        with langfuse.start_as_current_observation(
            name="task2-context-agent-run",
            as_type="agent",
            input=langfuse_input,
            metadata=langfuse_metadata,
            model=self.settings.agent_model,
        ) as root_span:
            with propagate_attributes(
                session_id=run_id,
                trace_name="task2-context-agent-run",
                tags=[
                    "task2",
                    "context-agent",
                    run_config.stage,
                    run_config.language,
                    f"experiment:{self.settings.experiment_name}",
                ],
                metadata=langfuse_metadata,
            ):
                try:
                    response = self.openai.start_agent_run(task_input=task_input, tools=tools)
                    runtime.model_logs.append(
                        {
                            "phase": "initial",
                            **self.openai.describe_response(response),
                        }
                    )

                    total_tool_failures = 0
                    consecutive_tool_failure_steps = 0
                    for step_number in range(1, self.settings.max_agent_steps + 1):
                        function_calls = self.openai.extract_function_calls(response)
                        if not function_calls:
                            rescue_answer = self._attempt_last_chance_finish(
                                runtime=runtime,
                                response=response,
                                reason="response_without_finish_tool",
                            )
                            if rescue_answer is not None:
                                trace_info = self._trace_info(langfuse)
                                self._write_artifacts(runtime, run_config, rescue_answer, trace_info)
                                self._update_root_span(root_span, rescue_answer)
                                flush_langfuse()
                                return rescue_answer
                            trace_info = self._trace_info(langfuse)
                            answer = runtime.build_timeout_answer(
                                unresolved_point="response_without_finish_tool_heuristic_fallback",
                                auto_finished_unresolved_point="response_without_finish_tool_auto_finished",
                                evidence_note="Model returned plain text without using finish tool.",
                            )
                            self._write_artifacts(runtime, run_config, answer, trace_info)
                            self._update_root_span(root_span, answer)
                            flush_langfuse()
                            return answer

                        tool_outputs: list[dict[str, str]] = []
                        successful_tool_calls = 0
                        failed_tool_calls = 0
                        for call in function_calls:
                            try:
                                result = runtime.execute(call)
                            except Exception as exc:
                                failed_tool_calls += 1
                                total_tool_failures += 1
                                error_payload = self._record_tool_error(
                                    runtime=runtime,
                                    call=call,
                                    exc=exc,
                                )
                                tool_outputs.append(
                                    {
                                        "type": "function_call_output",
                                        "call_id": call.call_id,
                                        "output": json.dumps(error_payload, ensure_ascii=False),
                                    }
                                )
                                continue

                            successful_tool_calls += 1
                            if result.final_answer is not None:
                                trace_info = self._trace_info(langfuse)
                                self._write_artifacts(runtime, run_config, result.final_answer, trace_info)
                                self._update_root_span(root_span, result.final_answer)
                                flush_langfuse()
                                return result.final_answer

                            tool_outputs.append(
                                {
                                    "type": "function_call_output",
                                    "call_id": call.call_id,
                                    "output": result.output,
                                }
                            )

                        if successful_tool_calls == 0 and failed_tool_calls > 0:
                            consecutive_tool_failure_steps += 1
                        else:
                            consecutive_tool_failure_steps = 0

                        if (
                            self.settings.max_tool_failures_per_datapoint > 0
                            and total_tool_failures >= self.settings.max_tool_failures_per_datapoint
                        ):
                            rescue_answer = self._attempt_last_chance_finish(
                                runtime=runtime,
                                response=response,
                                reason="tool_failure_budget_reached",
                                pending_tool_outputs=tool_outputs,
                            )
                            if rescue_answer is not None:
                                trace_info = self._trace_info(langfuse)
                                self._write_artifacts(runtime, run_config, rescue_answer, trace_info)
                                self._update_root_span(root_span, rescue_answer)
                                flush_langfuse()
                                return rescue_answer
                            trace_info = self._trace_info(langfuse)
                            answer = runtime.build_timeout_answer(
                                unresolved_point="tool_failure_budget_reached_heuristic_fallback",
                                auto_finished_unresolved_point="tool_failure_budget_reached_auto_finished",
                                existing_context_unresolved_point=None,
                                evidence_note=(
                                    f"Tool failure budget reached: "
                                    f"{self.settings.max_tool_failures_per_datapoint}"
                                ),
                            )
                            root_span.update(
                                level="WARNING" if answer.context else "ERROR",
                                status_message="tool_failure_budget_reached",
                                output=self._langfuse_run_output(answer),
                            )
                            self._write_artifacts(runtime, run_config, answer, trace_info)
                            flush_langfuse()
                            return answer

                        if (
                            self.settings.max_consecutive_tool_failure_steps > 0
                            and consecutive_tool_failure_steps
                            >= self.settings.max_consecutive_tool_failure_steps
                        ):
                            rescue_answer = self._attempt_last_chance_finish(
                                runtime=runtime,
                                response=response,
                                reason="consecutive_tool_failures_reached",
                                pending_tool_outputs=tool_outputs,
                            )
                            if rescue_answer is not None:
                                trace_info = self._trace_info(langfuse)
                                self._write_artifacts(runtime, run_config, rescue_answer, trace_info)
                                self._update_root_span(root_span, rescue_answer)
                                flush_langfuse()
                                return rescue_answer
                            trace_info = self._trace_info(langfuse)
                            answer = runtime.build_timeout_answer(
                                unresolved_point="consecutive_tool_failures_heuristic_fallback",
                                auto_finished_unresolved_point="consecutive_tool_failures_auto_finished",
                                existing_context_unresolved_point=None,
                                evidence_note=(
                                    f"Consecutive tool-failure steps reached: "
                                    f"{self.settings.max_consecutive_tool_failure_steps}"
                                ),
                            )
                            root_span.update(
                                level="WARNING" if answer.context else "ERROR",
                                status_message="consecutive_tool_failures_reached",
                                output=self._langfuse_run_output(answer),
                            )
                            self._write_artifacts(runtime, run_config, answer, trace_info)
                            flush_langfuse()
                            return answer

                        should_finalize, finalize_reason = runtime.should_auto_finalize(
                            step_number=step_number,
                            max_steps=self.settings.max_agent_steps,
                        )
                        if should_finalize:
                            rescue_answer = self._attempt_last_chance_finish(
                                runtime=runtime,
                                response=response,
                                reason=finalize_reason,
                                pending_tool_outputs=tool_outputs,
                            )
                            if rescue_answer is not None:
                                trace_info = self._trace_info(langfuse)
                                self._write_artifacts(runtime, run_config, rescue_answer, trace_info)
                                self._update_root_span(root_span, rescue_answer)
                                flush_langfuse()
                                return rescue_answer
                            trace_info = self._trace_info(langfuse)
                            answer = runtime.build_timeout_answer(
                                unresolved_point=f"{finalize_reason}_heuristic_fallback",
                                auto_finished_unresolved_point=f"{finalize_reason}_auto_finished",
                                existing_context_unresolved_point=None,
                                evidence_note=f"Auto-finished early: {finalize_reason}.",
                            )
                            span_update_payload = {
                                "status_message": f"auto_finished:{finalize_reason}",
                                "output": self._langfuse_run_output(answer),
                            }
                            if answer.unresolved_points:
                                span_update_payload["level"] = "WARNING"
                            root_span.update(**span_update_payload)
                            self._write_artifacts(runtime, run_config, answer, trace_info)
                            flush_langfuse()
                            return answer

                        response = self.openai.continue_agent_run(
                            previous_response_id=response.id,
                            tool_outputs=tool_outputs,
                            tools=tools,
                        )
                        runtime.model_logs.append(
                            {
                                "phase": "continue",
                                "step_number": step_number,
                                **self.openai.describe_response(response),
                            }
                        )
                        if (
                            self.settings.max_input_tokens_per_datapoint > 0
                            and self._model_input_tokens(runtime.model_logs)
                            >= self.settings.max_input_tokens_per_datapoint
                        ):
                            rescue_answer = self._attempt_last_chance_finish(
                                runtime=runtime,
                                response=response,
                                reason="input_token_budget_reached",
                            )
                            if rescue_answer is not None:
                                trace_info = self._trace_info(langfuse)
                                self._write_artifacts(runtime, run_config, rescue_answer, trace_info)
                                self._update_root_span(root_span, rescue_answer)
                                flush_langfuse()
                                return rescue_answer
                            trace_info = self._trace_info(langfuse)
                            answer = runtime.build_timeout_answer(
                                unresolved_point="input_token_budget_reached_heuristic_fallback",
                                auto_finished_unresolved_point="input_token_budget_reached_auto_finished",
                                evidence_note=(
                                    f"Input token budget reached: "
                                    f"{self.settings.max_input_tokens_per_datapoint}"
                                ),
                            )
                            root_span.update(
                                level="WARNING",
                                status_message="input_token_budget_reached",
                                output=self._langfuse_run_output(answer),
                            )
                            self._write_artifacts(runtime, run_config, answer, trace_info)
                            flush_langfuse()
                            return answer

                    rescue_answer = self._attempt_last_chance_finish(
                        runtime=runtime,
                        response=response,
                        reason="max_agent_steps_reached",
                    )
                    if rescue_answer is not None:
                        trace_info = self._trace_info(langfuse)
                        self._write_artifacts(runtime, run_config, rescue_answer, trace_info)
                        self._update_root_span(root_span, rescue_answer)
                        flush_langfuse()
                        return rescue_answer

                    trace_info = self._trace_info(langfuse)
                    answer = runtime.build_timeout_answer()
                    status_message = (
                        "max_agent_steps_reached"
                        if not answer.context
                        else ",".join(answer.unresolved_points or ["max_agent_steps_reached"])
                    )
                    root_span.update(
                        level="WARNING" if answer.context else "ERROR",
                        status_message=status_message,
                        output=self._langfuse_run_output(answer),
                    )
                    self._write_artifacts(runtime, run_config, answer, trace_info)
                    flush_langfuse()
                    return answer
                except Exception as exc:
                    trace_info = self._trace_info(langfuse)
                    answer = runtime.build_timeout_answer(
                        unresolved_point="model_request_failed_heuristic_fallback",
                        evidence_note=f"OpenAI request failed after retries: {type(exc).__name__}",
                    )
                    root_span.update(
                        level="WARNING" if answer.context else "ERROR",
                        status_message=f"model_request_failed:{type(exc).__name__}",
                        output=self._langfuse_run_output(answer),
                    )
                    self._write_artifacts(runtime, run_config, answer, trace_info)
                    flush_langfuse()
                    return answer

    def _record_tool_error(
        self,
        *,
        runtime: AgentToolRuntime,
        call: object,
        exc: Exception,
    ) -> dict[str, str]:
        error_payload = {
            "status": "error",
            "tool_name": str(getattr(call, "name", "") or ""),
            "error_type": type(exc).__name__,
            "message": str(exc)[:280],
            "suggestion": (
                "adjust_arguments_or_choose_other_snippet;"
                "finish_if_context_is_sufficient"
            ),
        }
        runtime.tool_logs.append(
            {
                "call_id": getattr(call, "call_id", None),
                "tool_name": getattr(call, "name", ""),
                "arguments": getattr(call, "arguments", None),
                "result": error_payload,
            }
        )
        return error_payload

    def _attempt_last_chance_finish(
        self,
        *,
        runtime: AgentToolRuntime,
        response: object,
        reason: str,
        pending_tool_outputs: list[dict[str, str]] | None = None,
    ) -> ContextAnswer | None:
        if not self.settings.enable_last_chance_finalize:
            return None

        full_tools = runtime.tool_definitions()
        finalize_tools = [
            tool
            for tool in full_tools
            if tool.get("name") in {"preview_context", "compress_context", "finish"}
        ]
        if not finalize_tools:
            return None

        current_response = response
        if pending_tool_outputs:
            try:
                current_response = self.openai.continue_agent_run(
                    previous_response_id=getattr(current_response, "id"),
                    tool_outputs=pending_tool_outputs,
                    tools=full_tools,
                )
                runtime.model_logs.append(
                    {
                        "phase": "finalize_flush",
                        "reason": reason,
                        **self.openai.describe_response(current_response),
                    }
                )
            except Exception:
                return None

        # If there are unresolved tool calls on the current response, flush them first.
        for resolve_step in range(1, 4):
            unresolved_calls = self.openai.extract_function_calls(current_response)
            if not unresolved_calls:
                break

            resolve_outputs: list[dict[str, str]] = []
            for call in unresolved_calls:
                try:
                    result = runtime.execute(call)
                except Exception as exc:
                    error_payload = self._record_tool_error(
                        runtime=runtime,
                        call=call,
                        exc=exc,
                    )
                    resolve_outputs.append(
                        {
                            "type": "function_call_output",
                            "call_id": call.call_id,
                            "output": json.dumps(error_payload, ensure_ascii=False),
                        }
                    )
                    continue

                if result.final_answer is not None:
                    return result.final_answer
                resolve_outputs.append(
                    {
                        "type": "function_call_output",
                        "call_id": call.call_id,
                        "output": result.output,
                    }
                )

            try:
                current_response = self.openai.continue_agent_run(
                    previous_response_id=getattr(current_response, "id"),
                    tool_outputs=resolve_outputs,
                    tools=full_tools,
                )
                runtime.model_logs.append(
                    {
                        "phase": "finalize_resolve_pending_calls",
                        "reason": reason,
                        "resolve_step": resolve_step,
                        **self.openai.describe_response(current_response),
                    }
                )
            except Exception:
                return None

        # If pending function calls still remain, do not issue a nudge request.
        if self.openai.extract_function_calls(current_response):
            return None

        finalization_prompt = (
            "Finalization mode: use only the currently selected snippets. "
            "Do not call search/read/list/add/reset tools. "
            "Optionally call preview_context or compress_context once, then call finish now."
        )
        try:
            current_response = self.openai.continue_agent_run_with_input(
                previous_response_id=getattr(current_response, "id"),
                input_payload=finalization_prompt,
                tools=finalize_tools,
            )
            runtime.model_logs.append(
                {
                    "phase": "finalize_nudge",
                    "reason": reason,
                    **self.openai.describe_response(current_response),
                }
            )
        except Exception:
            return None

        max_finalize_steps = max(1, self.settings.last_chance_finalize_steps)
        for finalize_step in range(1, max_finalize_steps + 1):
            function_calls = self.openai.extract_function_calls(current_response)
            if not function_calls:
                return None

            tool_outputs: list[dict[str, str]] = []
            for call in function_calls:
                try:
                    result = runtime.execute(call)
                except Exception as exc:
                    error_payload = self._record_tool_error(
                        runtime=runtime,
                        call=call,
                        exc=exc,
                    )
                    tool_outputs.append(
                        {
                            "type": "function_call_output",
                            "call_id": call.call_id,
                            "output": json.dumps(error_payload, ensure_ascii=False),
                        }
                    )
                    continue

                if result.final_answer is not None:
                    return result.final_answer
                tool_outputs.append(
                    {
                        "type": "function_call_output",
                        "call_id": call.call_id,
                        "output": result.output,
                    }
                )

            if finalize_step >= max_finalize_steps:
                break
            try:
                current_response = self.openai.continue_agent_run(
                    previous_response_id=getattr(current_response, "id"),
                    tool_outputs=tool_outputs,
                    tools=finalize_tools,
                )
                runtime.model_logs.append(
                    {
                        "phase": "finalize_continue",
                        "reason": reason,
                        "finalize_step": finalize_step,
                        **self.openai.describe_response(current_response),
                    }
                )
            except Exception:
                return None

        return None

    def _write_artifacts(
        self,
        runtime: AgentToolRuntime,
        run_config: RunConfig,
        answer: ContextAnswer,
        trace_info: dict[str, str | None] | None = None,
    ) -> None:
        artifact_dir = run_config.artifact_dir or self._default_artifact_dir(run_config)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        self._write_json(artifact_dir / "datapoint.json", run_config.datapoint.model_dump(mode="json"))
        self._write_json(artifact_dir / "tool_logs.json", runtime.tool_logs)
        self._write_json(artifact_dir / "model_logs.json", runtime.model_logs)
        self._write_json(artifact_dir / "answer.json", answer.model_dump(mode="json"))
        if trace_info is not None:
            self._write_json(artifact_dir / "trace.json", trace_info)

    def _default_artifact_dir(self, run_config: RunConfig) -> Path:
        datapoint_id = run_config.datapoint.id or f"{run_config.datapoint.repo}-{run_config.datapoint.revision}"
        safe_id = datapoint_id.replace("/", "__")
        return self.settings.artifacts_dir / f"{run_config.stage}-{run_config.language}-{safe_id}"

    def _write_json(self, target: Path, payload: object) -> None:
        target.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _trace_info(self, langfuse: object) -> dict[str, str | None]:
        trace_id = None
        trace_url = None
        try:
            trace_id = langfuse.get_current_trace_id()
        except Exception:
            trace_id = None
        try:
            trace_url = langfuse.get_trace_url(trace_id=trace_id)
        except Exception:
            trace_url = None
        return {"trace_id": trace_id, "trace_url": trace_url}

    def _update_root_span(
        self,
        root_span: object,
        answer: ContextAnswer,
    ) -> None:
        output = self._langfuse_run_output(answer)
        try:
            root_span.update(output=output)
        except Exception:
            try:
                root_span.update(output=output)
            except Exception:
                pass

    def _langfuse_run_input(self, run_config: RunConfig) -> dict[str, object]:
        datapoint = run_config.datapoint
        return {
            "stage": run_config.stage,
            "language": run_config.language,
            "datapoint_id": datapoint.id or "",
            "repository": datapoint.repo,
            "revision": datapoint.revision,
            "target_path": datapoint.path,
            "modified_files": datapoint.modified,
            "prefix_lines": len(datapoint.prefix.splitlines()),
            "suffix_lines": len(datapoint.suffix.splitlines()),
        }

    def _langfuse_run_output(self, answer: ContextAnswer) -> dict[str, object]:
        return {
            "context_length": len(answer.context),
            "selected_paths": answer.selected_paths,
            "selected_spans": answer.selected_spans,
            "evidence": answer.evidence,
            "unresolved_points": answer.unresolved_points,
        }

    def _model_input_tokens(self, model_logs: list[dict[str, object]]) -> int:
        total = 0
        for entry in model_logs:
            usage = entry.get("usage")
            if not isinstance(usage, dict):
                continue
            total += int(usage.get("input_tokens", 0) or 0)
        return total
