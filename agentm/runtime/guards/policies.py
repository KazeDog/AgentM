from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping


@dataclass
class GuardViolation(Exception):
    message: str
    agent: str = ""
    tool: str = ""

    def __str__(self) -> str:
        return self.message


class GuardPolicy:
    def __init__(self, *, agent_allowlists: Mapping[str, Iterable[str]], max_rounds: int = 12):
        self.agent_allowlists = {agent: set(tools) for agent, tools in agent_allowlists.items()}
        self.max_rounds = int(max_rounds)

    def validate_action(self, action: Mapping[str, Any], context: Mapping[str, Any]) -> None:
        agent = str(action.get("agent") or "")
        tool = str(action.get("tool") or "")
        if not agent or not tool:
            raise GuardViolation("multi_agent_action_missing_agent_or_tool", agent=agent, tool=tool)
        if agent not in self.agent_allowlists:
            raise GuardViolation(f"agent_not_allowed: {agent}", agent=agent, tool=tool)
        if tool not in self.agent_allowlists[agent]:
            raise GuardViolation(f"tool_not_allowed_for_agent: {agent}.{tool}", agent=agent, tool=tool)
        if len(context.get("tool_calls", []) or []) >= self.max_rounds:
            raise GuardViolation("max_rounds_exceeded", agent=agent, tool=tool)
        if not context.get("metadata", {}).get("enable_fallbacks", False) and tool in {"run_zero_shot_fallback"}:
            raise GuardViolation("fallback_tool_disabled", agent=agent, tool=tool)
        self._validate_preconditions(agent, tool, context)
        self._validate_no_fabricated_prediction(agent, tool, action)
        self._validate_no_fabricated_citation(agent, tool, action)

    def _validate_preconditions(self, agent: str, tool: str, context: Mapping[str, Any]) -> None:
        molecules = context.get("molecules", []) or []
        features = context.get("features", []) or []
        predictions = context.get("predictions", []) or []
        general_profile = context.get("task_type") == "general_drug_profile"
        intent_gate_task = context.get("task_type") in {"out_of_domain", "missing_entity"}
        reaction_task = str(context.get("task_type") or "").startswith("reaction_")
        safe_abstention = bool(context.get("metadata", {}).get("safe_abstention_required"))
        prediction_tools = {
            "predict_single",
            "predict_ddi",
            "run_zero_shot_fallback",
            "classify_reaction_type",
            "predict_buchwald_hartwig_yield",
        }
        if safe_abstention and tool in prediction_tools:
            raise GuardViolation("terminal_prediction_failure_requires_safe_abstention", agent=agent, tool=tool)
        if reaction_task:
            if tool in {
                "retrieve_mit_template_evidence",
                "retrieve_orderly_condition_evidence",
                "classify_reaction_type",
                "predict_buchwald_hartwig_yield",
                "build_reaction_profile",
            } and not context.get("reaction"):
                raise GuardViolation(f"{tool}_requires_parsed_reaction", agent=agent, tool=tool)
            if tool in {"build_structured_explanation", "build_runtime_report"} and not context.get("reaction_profile"):
                raise GuardViolation(f"{tool}_requires_reaction_profile", agent=agent, tool=tool)
            if tool in {"verify_report_schema", "reflect_on_report"} and not context.get("final_report"):
                raise GuardViolation("verify_report_schema_requires_final_report", agent=agent, tool=tool)
            if tool == "apply_reflection_feedback" and not context.get("reflection"):
                raise GuardViolation("apply_reflection_feedback_requires_reflection", agent=agent, tool=tool)
            return
        if intent_gate_task:
            allowed = {
                "build_structured_explanation",
                "build_runtime_report",
                "build_compatible_report",
                "verify_report_schema",
                "write_artifacts",
            }
            if tool not in allowed:
                raise GuardViolation("intent_gate_prediction_not_requested", agent=agent, tool=tool)
            if tool == "verify_report_schema" and not context.get("final_report"):
                raise GuardViolation("verify_report_schema_requires_final_report", agent=agent, tool=tool)
            return
        if general_profile and tool in self._general_profile_endpoint_tools():
            raise GuardViolation("general_profile_endpoint_prediction_not_requested", agent=agent, tool=tool)
        if tool in {"build_features", "predict_single", "predict_ddi", "retrieve_external_evidence"} and not molecules:
            raise GuardViolation(f"{tool}_requires_resolved_molecule", agent=agent, tool=tool)
        if general_profile and molecules and tool in self._general_profile_downstream_tools() and not self._tool_attempted(context, "retrieve_external_evidence"):
            raise GuardViolation("general_profile_requires_retrieve_external_evidence", agent=agent, tool=tool)
        if tool == "predict_single" and not features:
            raise GuardViolation("predict_single_requires_features", agent=agent, tool=tool)
        if tool == "predict_ddi" and not context.get("ddi_pair"):
            raise GuardViolation("predict_ddi_requires_ddi_pair", agent=agent, tool=tool)
        if tool == "assess_confidence" and not predictions and not general_profile:
            raise GuardViolation(f"{tool}_requires_prediction", agent=agent, tool=tool)
        if tool in {"retrieve_local_rag", "retrieve_elasticsearch_hybrid"} and not predictions and not general_profile and not safe_abstention:
            raise GuardViolation(f"{tool}_requires_prediction", agent=agent, tool=tool)
        if tool == "verify_prediction_grounding" and safe_abstention:
            raise GuardViolation("safe_abstention_has_no_prediction_to_verify", agent=agent, tool=tool)
        if tool in {"build_structured_explanation", "summarize_evidence_support", "verify_prediction_grounding", "build_runtime_report"} and not predictions and not general_profile and not safe_abstention:
            raise GuardViolation(f"{tool}_requires_prediction", agent=agent, tool=tool)
        if tool in {"verify_report_schema", "reflect_on_report"} and not context.get("final_report"):
            raise GuardViolation("verify_report_schema_requires_final_report", agent=agent, tool=tool)
        if tool == "apply_reflection_feedback" and not context.get("reflection"):
            raise GuardViolation("apply_reflection_feedback_requires_reflection", agent=agent, tool=tool)

    def _general_profile_downstream_tools(self) -> set[str]:
        return {
            "build_features",
            "retrieve_local_rag",
            "retrieve_elasticsearch_hybrid",
            "expand_evidence_query",
            "build_structured_explanation",
            "summarize_evidence_support",
            "build_runtime_report",
            "build_compatible_report",
            "verify_citation_grounding",
            "verify_report_schema",
            "reflect_on_report",
            "apply_reflection_feedback",
            "write_artifacts",
        }

    def _general_profile_endpoint_tools(self) -> set[str]:
        return {
            "discover_rules",
            "predict_single",
            "predict_ddi",
            "assess_confidence",
            "run_zero_shot_fallback",
            "verify_prediction_grounding",
        }

    def _tool_attempted(self, context: Mapping[str, Any], tool: str) -> bool:
        return any(call.get("tool") == tool for call in context.get("tool_calls", []) or [])

    def _validate_no_fabricated_prediction(self, agent: str, tool: str, action: Mapping[str, Any]) -> None:
        args = action.get("args") or {}
        if not isinstance(args, Mapping):
            return
        prediction_tools = {"predict_single", "predict_ddi", "run_zero_shot_fallback"}
        reaction_prediction_tools = {"classify_reaction_type", "predict_buchwald_hartwig_yield"}
        if not ((agent == "GaugeAgent" and tool in prediction_tools) or (agent == "ReactionAgent" and tool in reaction_prediction_tools)):
            forbidden = {"label", "score", "probability", "prediction", "predictions"}
            if forbidden.intersection(args):
                raise GuardViolation("prediction_fields_not_allowed_outside_gauge_prediction_tools", agent=agent, tool=tool)

    def _validate_no_fabricated_citation(self, agent: str, tool: str, action: Mapping[str, Any]) -> None:
        args = action.get("args") or {}
        if not isinstance(args, Mapping):
            return
        retrieval_tools = {
            "retrieve_local_rag",
            "retrieve_elasticsearch_hybrid",
            "retrieve_external_evidence",
            "retrieve_mit_template_evidence",
            "retrieve_orderly_condition_evidence",
        }
        if not (agent == "RetrieveAgent" and tool in retrieval_tools):
            forbidden = {"citation_id", "chunk_id", "source_path", "citation_map"}
            if forbidden.intersection(args):
                raise GuardViolation("citation_fields_not_allowed_outside_retrieval_tools", agent=agent, tool=tool)


def guard_warning(exc: Exception) -> Dict[str, Any]:
    return {"status": "blocked", "guard": "GuardPolicy", "error": str(exc)}
