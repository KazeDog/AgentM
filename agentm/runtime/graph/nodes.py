from __future__ import annotations

import time
from typing import Any, Dict

from agentm.runtime.artifacts import report_payload
from agentm.runtime.compat import to_jsonable
from agentm.runtime.guards import (
    GuardPolicy,
    GuardViolation,
    repair_evidence_record_integrity,
    repair_prediction_record_integrity,
)
from agentm.runtime.tools import AgentAction, MultiAgentToolRegistry, ToolResult


PREDICTION_TOOLS = {
    "predict_single",
    "predict_ddi",
    "run_zero_shot_fallback",
    "classify_reaction_type",
    "predict_buchwald_hartwig_yield",
}


class MultiAgentGraphNodes:
    def __init__(
        self,
        *,
        registry: MultiAgentToolRegistry,
        guard: GuardPolicy,
        orchestrator: Any,
        output_dir: str,
        experiment_hook: Any | None = None,
    ) -> None:
        self.registry = registry
        self.guard = guard
        self.orchestrator = orchestrator
        self.output_dir = output_dir
        self.experiment_hook = experiment_hook

    def initialize_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        metadata = context.setdefault("metadata", {})
        metadata["loop_status"] = "ready"
        metadata["agent_allowlists"] = self.registry.agent_allowlists()
        metadata["tool_schemas"] = self.registry.schemas()
        if hasattr(self.orchestrator, "build_plan"):
            plan = self.orchestrator.build_plan(context)
            context["orchestrator_plan"] = plan
            context["task_graph"] = plan.get("task_graph", context.get("task_graph", {}))
        self._notify_hook("after_initialize", context=context)
        context.setdefault("task_graph", {}).setdefault("status", "ready")
        context.setdefault("multi_agent_trace", []).append(
            self._event(
                "OrchestratorAgent",
                "initialize_context",
                "build_plan",
                "completed",
                {"n_agents": len(self.registry.agents()), "planner_type": context.get("metadata", {}).get("planner_type")},
            )
        )
        return context

    def orchestrator_loop(self, context: Dict[str, Any]) -> Dict[str, Any]:
        metadata = context.setdefault("metadata", {})
        metadata["loop_status"] = "running"
        max_rounds = int(metadata.get("max_rounds", 12))
        for _ in range(max_rounds + 1):
            # An invalid controlled fault is an experiment setup failure, not
            # a scientific/tool failure the orchestrator should keep solving.
            # Abort before another (possibly paid) planning call.
            if metadata.get('controlled_injection_error'):
                raise RuntimeError('controlled_injection_invalid: ' + str(metadata['controlled_injection_error']))
            # A successful model-tool trace is the only permitted repair source.
            # This catches schema loss or an unsupported certainty wrapper before
            # confidence assessment, reasoning, and reporting consume it.
            repair_prediction_record_integrity(context)
            # Evidence is repairable only when the active citation/chunk differs
            # from the immutable content captured by a successful retrieval.
            repair_evidence_record_integrity(context)
            action = self.orchestrator.next_action(context)
            if action.tool == "finish":
                metadata["loop_status"] = "finished"
                context.setdefault("agent_messages", []).append(self._message("OrchestratorAgent", action, "finish"))
                context.setdefault("multi_agent_trace", []).append(
                    self._event("OrchestratorAgent", "finish", "finish", "completed", {"reason": action.rationale})
                )
                return context
            started = time.time()
            call = {
                "round": len(context.get("tool_calls", []) or []) + 1,
                "agent": action.agent,
                "tool": action.tool,
                "args": action.args,
                "rationale": action.rationale,
                "status": "started",
            }
            try:
                self.guard.validate_action(action.to_dict(), context)
                result = self._execute_with_hook(action, context)
                call.update(result.to_dict())
                self._record_terminal_prediction_failure(action, result, context)
                self._mark_stale_after(action, context)
                self._notify_hook("after_tool", action=action, result=result, context=context)
            except GuardViolation as exc:
                revised = self._revise_guard_blocked_action(context, action, exc)
                if revised is None:
                    call.update({"status": "guard_blocked", "error": str(exc)})
                    context.setdefault("warnings", []).append(f"guard_blocked {action.agent}.{action.tool}: {exc}")
                    metadata["loop_status"] = "guard_blocked"
                    context.setdefault("tool_calls", []).append(to_jsonable(call))
                    context.setdefault("multi_agent_trace", []).append(
                        self._event(action.agent, action.tool, action.tool, "guard_blocked", {"error": str(exc)})
                    )
                    return context
                call["revised_from"] = {"agent": action.agent, "tool": action.tool, "args": action.args, "error": str(exc)}
                action = revised
                call.update({"agent": action.agent, "tool": action.tool, "args": action.args, "rationale": action.rationale, "status": "started"})
                try:
                    self.guard.validate_action(action.to_dict(), context)
                    result = self._execute_with_hook(action, context)
                    call.update(result.to_dict())
                    self._record_terminal_prediction_failure(action, result, context)
                    call["guard_revision_used"] = True
                    self._mark_stale_after(action, context)
                    self._notify_hook("after_tool", action=action, result=result, context=context)
                except GuardViolation as revised_exc:
                    call.update({"status": "guard_revision_blocked", "error": str(revised_exc), "guard_revision_used": True})
                    context.setdefault("warnings", []).append(
                        f"guard_revision_blocked {action.agent}.{action.tool}: {revised_exc}"
                    )
                    context.setdefault("multi_agent_trace", []).append(
                        self._event(
                            action.agent,
                            action.tool,
                            action.tool,
                            "guard_revision_blocked",
                            {"error": str(revised_exc), "revised_from": call.get("revised_from")},
                        )
                    )
            except Exception as exc:
                call.update({"status": "failed", "error": str(exc)})
                context.setdefault("warnings", []).append(f"tool_failed {action.agent}.{action.tool}: {exc}")
            finally:
                call["duration_sec"] = round(time.time() - started, 6)
            call["fallback_used"] = call.get("status") == "fallback_completed"
            context.setdefault("tool_calls", []).append(to_jsonable(call))
            metadata.setdefault("completed_actions", []).append(f"{action.agent}.{action.tool}:{call.get('status')}")
            metadata["round"] = len(context.get("tool_calls", []) or [])
            context.setdefault("agent_messages", []).append(self._message(action.agent, action, call.get("status", "unknown")))
            context.setdefault("multi_agent_trace", []).append(
                self._event(
                    action.agent,
                    action.tool,
                    action.tool,
                    call.get("status", "unknown"),
                    {
                        "inputs_summary": self._inputs_summary(action),
                        "outputs_summary": self._outputs_summary(call),
                        "duration_sec": call["duration_sec"],
                        "fallback_used": call["fallback_used"],
                    },
                )
            )
            if self._is_terminal_artifact_action(action, call):
                metadata["loop_status"] = "finished"
                self._refresh_final_report_payload(context)
                context.setdefault("agent_messages", []).append(self._message("OrchestratorAgent", AgentAction("OrchestratorAgent", "finish"), "finish"))
                context.setdefault("multi_agent_trace", []).append(
                    self._event("OrchestratorAgent", "finish", "finish", "completed", {"reason": "Artifacts were written successfully."})
                )
                return context
        context.setdefault("warnings", []).append("max_rounds_exceeded")
        metadata["loop_status"] = "max_rounds_exceeded"
        return context

    def guarded_finalize(self, context: Dict[str, Any]) -> Dict[str, Any]:
        metadata = context.setdefault("metadata", {})
        if metadata.get("loop_status") in {"ready", "running", "finished"}:
            metadata["loop_status"] = "finalized"
        self._refresh_final_report_payload(context)
        self._refresh_final_report_status(context)
        if not any(call.get("tool") == "write_artifacts" and call.get("status") == "completed" for call in context.get("tool_calls", []) or []):
            try:
                result = self.registry.execute(AgentAction("ReporterAgent", "write_artifacts", rationale="Guarded finalization writes artifacts."), context)
                context.setdefault("tool_calls", []).append(
                    {
                        "round": len(context.get("tool_calls", []) or []) + 1,
                        "agent": "ReporterAgent",
                        "tool": "write_artifacts",
                        "status": result.status,
                        "observation": to_jsonable(result.observation),
                        "error": result.error,
                        "rationale": "Guarded finalization writes artifacts.",
                        "fallback_used": False,
                    }
                )
                self._refresh_final_report_payload(context)
            except Exception as exc:
                context.setdefault("warnings", []).append(f"finalize_write_artifacts_failed: {exc}")
        metadata["loop_status"] = metadata.get("loop_status") or "finalized"
        self._refresh_final_report_payload(context)
        self._refresh_final_report_status(context)
        if any(call.get("tool") == "write_artifacts" and call.get("status") == "completed" for call in context.get("tool_calls", []) or []):
            try:
                self.registry.execute(AgentAction("ReporterAgent", "write_artifacts", rationale="Guarded finalization refreshes finalized artifacts."), context)
            except Exception as exc:
                context.setdefault("warnings", []).append(f"finalize_refresh_artifacts_failed: {exc}")
        context.setdefault("multi_agent_trace", []).append(
            self._event("System", "guarded_finalize", "guarded_finalize", "completed", {"artifact_count": len(context.get("artifacts", {}))})
        )
        self._notify_hook("after_finalize", context=context)
        return context

    def _execute_with_hook(self, action: AgentAction, context: Dict[str, Any]) -> ToolResult:
        """Execute a registered tool unless a reliability hook intercepts it.

        Hooks are optional and receive the same mutable context as graph nodes.
        A hook may return a ToolResult only for a declared controlled-failure
        experiment. Returning any other object is an explicit error.
        """

        if self.experiment_hook is not None and hasattr(self.experiment_hook, "before_tool"):
            intercepted = self.experiment_hook.before_tool(action=action, context=context)
            if intercepted is not None:
                if not isinstance(intercepted, ToolResult):
                    raise TypeError("experiment_hook.before_tool must return ToolResult or None")
                return intercepted
        return self.registry.execute(action, context)

    def _notify_hook(self, event: str, **payload: Any) -> None:
        if self.experiment_hook is None:
            return
        handler = getattr(self.experiment_hook, event, None)
        if handler is not None:
            handler(**payload)

    def _record_terminal_prediction_failure(
        self,
        action: AgentAction,
        result: ToolResult,
        context: Dict[str, Any],
    ) -> None:
        if action.tool not in PREDICTION_TOOLS or result.status != "failed":
            return
        observation = result.observation if isinstance(result.observation, dict) else {}
        failure_kind = str(observation.get("failure_kind") or "")
        terminal = bool(observation.get("recoverable") is False) or failure_kind == "prediction_tool_or_service_failure"
        if not terminal:
            return
        metadata = context.setdefault("metadata", {})
        metadata["safe_abstention_required"] = True
        metadata["prediction_failure"] = {
            "tool": action.tool,
            "agent": action.agent,
            "failure_kind": failure_kind or "unrecoverable_prediction_failure",
            "error": result.error,
            "controlled_failure": bool(observation.get("controlled_failure")),
            "recoverable": False,
        }
        warning = f"prediction_tool_or_service_failure: {action.tool} unavailable; task failed safely without a prediction"
        if warning not in context.setdefault("warnings", []):
            context["warnings"].append(warning)

    def _event(self, agent: str, action: str, tool: str, status: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "agent": agent,
            "action": action,
            "tool": tool,
            "status": status,
            "payload": to_jsonable(payload),
        }

    def _message(self, agent: str, action: AgentAction, status: str) -> Dict[str, Any]:
        return {
            "agent": agent,
            "role": "assistant" if agent == "OrchestratorAgent" else "tool",
            "action": action.to_dict(),
            "status": status,
        }

    def _inputs_summary(self, action: AgentAction) -> Dict[str, Any]:
        return {"args_keys": sorted(action.args), "rationale": action.rationale}

    def _outputs_summary(self, call: Dict[str, Any]) -> Dict[str, Any]:
        observation = call.get("observation") or {}
        return {
            "status": call.get("status"),
            "error": call.get("error"),
            "observation_keys": sorted(observation) if isinstance(observation, dict) else [],
        }

    def _is_terminal_artifact_action(self, action: AgentAction, call: Dict[str, Any]) -> bool:
        return action.agent == "ReporterAgent" and action.tool == "write_artifacts" and call.get("status") == "completed"

    def _refresh_final_report_status(self, context: Dict[str, Any]) -> None:
        report = context.get("final_report")
        if not isinstance(report, dict):
            return
        multi_agent = report.setdefault("multi_agent", {})
        if isinstance(multi_agent, dict):
            multi_agent["loop_status"] = context.get("metadata", {}).get("loop_status")
            multi_agent["rounds"] = len(context.get("tool_calls", []) or [])

    def _refresh_final_report_payload(self, context: Dict[str, Any]) -> None:
        context["final_report"] = report_payload(context)

    def _revise_guard_blocked_action(self, context: Dict[str, Any], action: AgentAction, exc: GuardViolation) -> AgentAction | None:
        if not hasattr(self.orchestrator, "revise_after_guard"):
            return None
        revised = self.orchestrator.revise_after_guard(context, action, exc)
        if revised is None:
            return None
        context.setdefault("warnings", []).append(f"guard_revised {action.agent}.{action.tool}: {exc}")
        context.setdefault("multi_agent_trace", []).append(
            self._event(
                action.agent,
                action.tool,
                action.tool,
                "guard_revised",
                {"error": str(exc), "revised_action": revised.to_dict()},
            )
        )
        return revised

    def _mark_stale_after(self, action: AgentAction, context: Dict[str, Any]) -> None:
        metadata = context.setdefault("metadata", {})
        if action.agent == "RetrieveAgent":
            if metadata.get("reflection_retrieval_requested"):
                metadata["reflection_retrieval_done"] = True
            if context.get("reasoning", {}).get("structured_explanation"):
                metadata["reasoning_stale"] = True
            if context.get("final_report"):
                metadata["report_stale"] = True
            if context.get("verifier_report", {}).get("citations"):
                metadata["citation_verification_stale"] = True
            if context.get("verifier_report", {}).get("schema"):
                metadata["schema_verification_stale"] = True
        if action.agent == "ReasoningAgent":
            if context.get("final_report"):
                metadata["report_stale"] = True
            if context.get("verifier_report", {}).get("schema"):
                metadata["schema_verification_stale"] = True
        if action.agent == "ReporterAgent" and action.tool in {"build_runtime_report", "build_compatible_report"}:
            if context.get("verifier_report", {}).get("schema"):
                metadata["schema_verification_stale"] = True
