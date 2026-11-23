from __future__ import annotations

from copy import deepcopy
import random
from typing import Any, Mapping

from agentm.runtime.tools import AgentAction, ToolResult
from agentm.runtime.compat import to_jsonable
from experiments.agentm_reliability.prediction_mutations import mutate_prediction, InjectionNotApplicable

try:
    from .common import canonical_json, stable_hash, utc_now
    from .gold_rules import (
        citation_links_from_evidence,
        evidence_id,
        reconcile_runtime_gold_after_injection,
        resolve_runtime_gold,
    )
except ImportError:  # direct script execution
    from common import canonical_json, stable_hash, utc_now
    from gold_rules import (
        citation_links_from_evidence,
        evidence_id,
        reconcile_runtime_gold_after_injection,
        resolve_runtime_gold,
    )


PREDICTION_TOOLS = {
    "predict_single",
    "predict_ddi",
    "classify_reaction_type",
    "predict_buchwald_hartwig_yield",
    "run_zero_shot_fallback",
}
REPORT_TOOLS = {"build_runtime_report", "build_compatible_report", "build_reaction_report"}


def _prediction_field(context: Mapping[str, Any]) -> str:
    if str(context.get("task_type") or "").startswith("reaction_") and context.get("reaction_predictions") is not None:
        return "reaction_predictions"
    return "predictions"


def _target_snapshot(context: Mapping[str, Any], field: str) -> Any:
    return to_jsonable(deepcopy(context.get(field)))


def _record(*, fault: str, field: str, before: Any, after: Any, details: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "injection_type": fault,
        "target_field": field,
        "applied_at": utc_now(),
        "before": before,
        "after": after,
        "before_hash": stable_hash(before),
        "after_hash": stable_hash(after),
        "changed": canonical_json(before) != canonical_json(after),
        "details": dict(details),
    }


def apply_injection(context: dict[str, Any], fault: str, *, seed: int = 2026, truth_reference: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Apply exactly one declared mutation and return a fully traceable record."""

    rng = random.Random(seed)
    if fault == "citation_mismatch":
        field = "citation_map"
        before = _target_snapshot(context, field)
        evidence = [item for item in context.get("evidence", []) or [] if isinstance(item, Mapping)]
        correct = citation_links_from_evidence(evidence)
        keys = list(correct) or ["claim_1"]
        values = list(correct.values()) or ["missing_evidence"]
        wrong: list[dict[str, Any]] = []
        for index, claim_id in enumerate(keys):
            source_id = values[(index + 1) % len(values)] if len(values) > 1 else f"mismatched::{values[index % len(values)]}"
            wrong.append(
                {
                    "claim_id": claim_id,
                    "chunk_id": source_id,
                    "citation_exists": False,
                    "injection_marker": "citation_mismatch",
                }
            )
        context[field] = wrong
        after = _target_snapshot(context, field)
        return _record(fault=fault, field=field, before=before, after=after, details={"gold_citation_map": correct})

    if fault in {"entity_or_endpoint_mismatch", "irrelevant_or_weak_evidence", "conflicting_evidence", "no_direct_evidence_or_ambiguous_entity"}:
        field = "evidence"
        before = _target_snapshot(context, field)
        evidence = deepcopy(list(context.get(field, []) or []))
        if fault == "entity_or_endpoint_mismatch":
            if not evidence:
                raise ValueError("entity_or_endpoint_mismatch requires at least one evidence record")
            index = rng.randrange(len(evidence))
            item = dict(evidence[index]) if isinstance(evidence[index], Mapping) else {"claim": str(evidence[index])}
            decoy = "unrelated comparator entity and non-target endpoint"
            item["claim"] = decoy
            item["snippet"] = decoy
            item["endpoint"] = "non_target_endpoint"
            item["compound_specific"] = False
            item["endpoint_specific"] = False
            nested = dict(item.get("value") or {}) if isinstance(item.get("value"), Mapping) else {}
            nested.update({"snippet": decoy, "endpoint": "non_target_endpoint", "compound_specific": False, "endpoint_specific": False})
            item["value"] = nested
            evidence[index] = item
        elif fault == "irrelevant_or_weak_evidence":
            if not evidence:
                raise ValueError("irrelevant_or_weak_evidence requires at least one evidence record")
            index = rng.randrange(len(evidence))
            item = dict(evidence[index]) if isinstance(evidence[index], Mapping) else {"claim": str(evidence[index])}
            background = "General background information; it does not directly test the requested entity, endpoint, drug pair, or reaction."
            item.update(
                {
                    "claim": background,
                    "snippet": background,
                    "evidence_scope": "background_only",
                    "compound_specific": False,
                    "endpoint_specific": False,
                    "reaction_specific": False,
                    "evidence_direction": "neutral",
                }
            )
            nested = dict(item.get("value") or {}) if isinstance(item.get("value"), Mapping) else {}
            nested.update({"snippet": background, "compound_specific": False, "endpoint_specific": False})
            item["value"] = nested
            evidence[index] = item
        elif fault == "conflicting_evidence":
            if not evidence:
                raise ValueError("conflicting_evidence requires at least one evidence record")
            original = dict(evidence[0]) if isinstance(evidence[0], Mapping) else {"claim": str(evidence[0])}
            original.setdefault("citation_id", evidence_id(original, 0))
            original["evidence_direction"] = "support"
            conflict = deepcopy(original)
            conflict_id = f"injected_conflict_{stable_hash({'seed': seed, 'base': original}, length=8)}"
            conflict.update(
                {
                    "citation_id": conflict_id,
                    "chunk_id": conflict_id,
                    "source": f"controlled_conflict_peer::{original.get('source', 'unknown')}",
                    "source_path": f"controlled://{conflict_id}",
                    "claim": "Comparable-scope evidence reports the opposite direction for the same target and endpoint.",
                    "snippet": "Comparable-scope evidence reports the opposite direction for the same target and endpoint.",
                    "evidence_direction": "contradiction",
                    "controlled_injection": True,
                }
            )
            evidence[0] = original
            evidence.append(conflict)
        else:
            evidence = []
        context[field] = evidence
        after = _target_snapshot(context, field)
        details: dict[str, Any] = {}
        if fault == "conflicting_evidence" and after:
            details["conflicting_evidence_ids"] = [evidence_id(item, index) for index, item in enumerate(after[-2:])]
        return _record(fault=fault, field=field, before=before, after=after, details=details)

    if fault in {"missing_field_or_overclaim", "plausible_but_wrong_prediction"}:
        field = _prediction_field(context)
        before = _target_snapshot(context, field)
        predictions = deepcopy(list(context.get(field, []) or []))
        if not predictions:
            raise ValueError(f"{fault} requires at least one prediction record")
        original_target = (context.get(field) or [])[-1]
        item, details = mutate_prediction(predictions[-1], fault, seed=seed,
                                          truth_reference=truth_reference, context=context)
        predictions[-1] = item
        context[field] = predictions
        # Reaction tools keep different lists pointing at the same prediction
        # object. Synchronize only that identity, not unrelated predictions.
        if field == "reaction_predictions":
            context['predictions'] = [item if p is original_target else p for p in context.get('predictions', [])]
        after = _target_snapshot(context, field)
        return _record(fault=fault, field=field, before=before, after=after, details=details)

    if fault == "prediction_tool_or_service_failure":
        raise ValueError("prediction_tool_or_service_failure must be injected through ControlledInjectionHook.before_tool")
    raise ValueError(f"Unknown injection type: {fault}")


class ControlledInjectionHook:
    """Optional graph hook for one controlled error per reliability case."""

    def __init__(self, case: Mapping[str, Any]) -> None:
        self.case = deepcopy(dict(case))
        self.fault = str(case.get("injection_type") or "")
        self.seed = int(case.get("mutation_seed", 2026))
        self.applied = False
        self.pre_injection_context: dict[str, Any] | None = None
        self.post_injection_context: dict[str, Any] | None = None
        self.runtime_gold: dict[str, Any] | None = None
        self.mutation_record: dict[str, Any] | None = None
        self.pre_reflection_report: dict[str, Any] | None = None
        self.final_report: dict[str, Any] | None = None
        self.events: list[dict[str, Any]] = []

    def after_initialize(self, *, context: dict[str, Any]) -> None:
        if self.fault or self.case.get("transient_fault"):
            context.setdefault("metadata", {})["controlled_experiment"] = {
                "stress_case_id": self.case.get("stress_case_id"),
                "injection_type": self.fault or self.case.get("transient_fault", {}).get("kind"),
                "mutation_seed": self.seed,
            }

    def before_tool(self, *, action: AgentAction, context: dict[str, Any]) -> ToolResult | None:
        if action.tool == "reflect_on_report" and self.pre_reflection_report is None:
            self.pre_reflection_report = deepcopy(context.get("final_report") or {})
        if self.fault == "prediction_tool_or_service_failure" and action.tool in PREDICTION_TOOLS:
            result = ToolResult(
                tool=action.tool,
                status="failed",
                observation={
                    "controlled_failure": True,
                    "failure_kind": "prediction_tool_or_service_failure",
                    "recoverable": False,
                    "persistent": True,
                },
                error="controlled_injection: prediction tool unavailable",
            )
            if not self.applied:
                self.pre_injection_context = to_jsonable(deepcopy(context))
                self.runtime_gold = resolve_runtime_gold(self.case, self.pre_injection_context)
                self.mutation_record = {
                    "injection_type": self.fault,
                    "target_field": "tool_result",
                    "applied_at": utc_now(),
                    "before": None,
                    "after": result.to_dict(),
                    "before_hash": stable_hash(None),
                    "after_hash": stable_hash(result.to_dict()),
                    "changed": True,
                    "details": {"intercepted_agent": action.agent, "intercepted_tool": action.tool, "persistent": True, "intercept_count": 0},
                }
                self.applied = True
                self.post_injection_context = to_jsonable(deepcopy(context))
            self.mutation_record["details"]["intercept_count"] += 1
            self.events.append({"event": "before_tool_intercept", "tool": action.tool, "timestamp": utc_now()})
            return result
        if self.applied:
            return None
        transient = self.case.get("transient_fault")
        if isinstance(transient, Mapping) and action.tool == transient.get("tool"):
            self.pre_injection_context = to_jsonable(deepcopy(context))
            result = ToolResult(
                tool=action.tool,
                status="failed",
                observation={"controlled_failure": True, "failure_kind": str(transient.get("kind") or "transient_tool_failure")},
                error="controlled_injection: one-time recoverable tool failure",
            )
            self.mutation_record = {
                "injection_type": str(transient.get("kind") or "transient_tool_failure"),
                "target_field": "tool_result",
                "applied_at": utc_now(),
                "before": None,
                "after": result.to_dict(),
                "before_hash": stable_hash(None),
                "after_hash": stable_hash(result.to_dict()),
                "changed": True,
                "details": {"intercepted_agent": action.agent, "intercepted_tool": action.tool, "persistent": False},
            }
            self.applied = True
            self.post_injection_context = to_jsonable(deepcopy(context))
            return result
        return None

    def after_tool(self, *, action: AgentAction, result: ToolResult, context: dict[str, Any]) -> None:
        if self.fault == "no_direct_evidence_or_ambiguous_entity" and self.applied:
            self._enforce_persistent_no_direct_evidence(action=action, context=context)
            return
        if self.applied or self.fault == "prediction_tool_or_service_failure" or (self.mutation_record or {}).get('error'):
            return
        if not self._triggered(action, context):
            return
        self.pre_injection_context = to_jsonable(deepcopy(context))
        self.runtime_gold = resolve_runtime_gold(self.case, self.pre_injection_context)
        try:
            self.mutation_record = apply_injection(context, self.fault, seed=self.seed,
                                                  truth_reference=self.case.get('gold_truth_reference'))
        except Exception as exc:
            self.mutation_record = {
                "injection_type": self.fault,
                "target_field": str(self.case.get("injection_stage") or "unknown"),
                "applied_at": utc_now(),
                "changed": False,
                "error": str(exc),
                "details": {"trigger_agent": action.agent, "trigger_tool": action.tool},
            }
            prefix='controlled_injection_not_applicable' if isinstance(exc,InjectionNotApplicable) else 'controlled_injection_failed'
            context.setdefault("warnings", []).append(f"{prefix}:{self.fault}:{exc}")
            context.setdefault('metadata', {})['controlled_injection_error'] = str(exc)
            if isinstance(exc, InjectionNotApplicable):
                field = _prediction_field(context)
                self.mutation_record.update(status='not_applicable', target_field=field,
                    before=deepcopy(self.pre_injection_context.get(field)), after=_target_snapshot(context,field),
                    not_applicable_reason=str(exc), eligible_for_injection_metrics=False,
                    scientific_success=None, replacement_allowed=False)
                context['metadata']['controlled_injection_applicability']='not_applicable'
        else:
            self.mutation_record.setdefault("details", {}).update({"trigger_agent": action.agent, "trigger_tool": action.tool})
            if self.fault in {"conflicting_evidence", "no_direct_evidence_or_ambiguous_entity"} and self.runtime_gold is not None:
                self.runtime_gold = reconcile_runtime_gold_after_injection(
                    self.runtime_gold,
                    context,
                    self.mutation_record,
                )
            if self.fault == "no_direct_evidence_or_ambiguous_entity":
                self._enable_no_direct_evidence_policy(context=context)
            context.setdefault("warnings", []).append(f"controlled_injection_applied:{self.fault}")
        self.applied = bool(self.mutation_record.get('changed')) and not self.mutation_record.get('error')
        self.post_injection_context = to_jsonable(deepcopy(context))
        self.events.append({"event": "after_tool_injection", "tool": action.tool, "timestamp": utc_now()})

    def _enable_no_direct_evidence_policy(self, *, context: dict[str, Any]) -> None:
        """Make the declared evidence absence a persistent fail-closed state."""

        metadata = context.setdefault("metadata", {})
        metadata["safe_abstention_required"] = True
        metadata["retrieval_outcome"] = "no_direct_evidence"
        metadata["direct_evidence_available"] = False
        metadata["controlled_evidence_absence"] = True
        metadata["prediction_failure"] = {
            "tool": "evidence_retrieval",
            "agent": "RetrieveAgent",
            "failure_kind": "no_direct_evidence_or_ambiguous_entity",
            "error": "controlled evidence absence or ambiguous entity",
            "controlled_failure": True,
            "recoverable": False,
        }
        details = self.mutation_record.setdefault("details", {}) if self.mutation_record is not None else {}
        details.update({"persistent": True, "enforcement_count": 1})
        self._clear_derived_evidence_state(context)

    def _enforce_persistent_no_direct_evidence(
        self,
        *,
        action: AgentAction,
        context: dict[str, Any],
    ) -> None:
        """Prevent later retrieval tools from silently undoing the same fault."""

        evidence_before = len(context.get("evidence", []) or [])
        artifact_before = sum(
            len((context.get("artifacts", {}) or {}).get(key, []) or [])
            for key in ("reaction_template_evidence", "reaction_condition_evidence", "retrieved_context", "rag_context")
        )
        prediction_before = len(context.get("reaction_predictions", []) or context.get("predictions", []) or [])
        self._clear_derived_evidence_state(context)
        metadata = context.setdefault("metadata", {})
        metadata["safe_abstention_required"] = True
        metadata["retrieval_outcome"] = "no_direct_evidence"
        metadata["direct_evidence_available"] = False
        metadata["controlled_evidence_absence"] = True
        if self.mutation_record is not None:
            details = self.mutation_record.setdefault("details", {})
            details["enforcement_count"] = int(details.get("enforcement_count", 1) or 1) + 1
            details["suppressed_evidence_count"] = int(details.get("suppressed_evidence_count", 0) or 0) + evidence_before
            details["suppressed_artifact_evidence_count"] = int(
                details.get("suppressed_artifact_evidence_count", 0) or 0
            ) + artifact_before
            details["suppressed_prediction_count"] = int(details.get("suppressed_prediction_count", 0) or 0) + prediction_before
        if evidence_before or artifact_before or prediction_before:
            self.events.append(
                {
                    "event": "persistent_evidence_absence_enforced",
                    "tool": action.tool,
                    "suppressed_evidence_count": evidence_before,
                    "suppressed_artifact_evidence_count": artifact_before,
                    "suppressed_prediction_count": prediction_before,
                    "timestamp": utc_now(),
                }
            )

    def _clear_derived_evidence_state(self, context: dict[str, Any]) -> None:
        context["evidence"] = []
        context["citation_map"] = []
        context["predictions"] = []
        context["reaction_predictions"] = []
        artifacts = context.setdefault("artifacts", {})
        for key in (
            "reaction_template_evidence",
            "reaction_condition_evidence",
            "retrieved_context",
            "rag_context",
        ):
            artifacts[key] = []

    def after_finalize(self, *, context: dict[str, Any]) -> None:
        self.final_report = deepcopy(context.get("final_report") or {})
        if not self.applied and self.fault:
            context.setdefault("warnings", []).append(f"controlled_injection_not_applied:{self.fault}")

    def _triggered(self, action: AgentAction, context: Mapping[str, Any]) -> bool:
        if self.fault in {"missing_field_or_overclaim", "plausible_but_wrong_prediction"}:
            return action.tool in PREDICTION_TOOLS and bool(context.get(_prediction_field(context)))
        if self.fault == "citation_mismatch":
            return action.tool in REPORT_TOOLS and bool(context.get("evidence"))
        if self.fault in {
            "entity_or_endpoint_mismatch",
            "irrelevant_or_weak_evidence",
            "conflicting_evidence",
            "no_direct_evidence_or_ambiguous_entity",
        }:
            return bool(context.get("evidence")) and (
                action.agent in {"RetrieveAgent", "ReactionAgent"}
                or action.tool.startswith("retrieve_")
                or "evidence" in action.tool
                or "template" in action.tool
            )
        return False

    def report_changed_by_reflection(self) -> bool | None:
        if self.pre_reflection_report is None:
            return None
        return stable_hash(self.pre_reflection_report) != stable_hash(self.final_report or {})

    def export(self) -> dict[str, Any]:
        return to_jsonable({
            "case": self.case,
            "applied": self.applied,
            "runtime_gold": self.runtime_gold,
            "mutation_record": self.mutation_record,
            "pre_injection_context": self.pre_injection_context,
            "post_injection_context": self.post_injection_context,
            "pre_reflection_report": self.pre_reflection_report,
            "final_report": self.final_report,
            "report_changed_by_reflection": self.report_changed_by_reflection(),
            "events": self.events,
        })
