#!/usr/bin/env python
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping

try:
    from .canonicalize_runs import canonicalize
    from .common import (
        RELIABILITY_ROOT,
        canonical_json,
        ensure_output_layout,
        nested_text,
        read_json,
        resolve_path,
        stable_hash,
        write_csv,
        write_json,
    )
    from .gold_rules import citation_links_from_evidence, evidence_id
    from .evidence_provenance import (
        audit_evidence_provenance,
        retained_evidence_records,
        retrieved_evidence_records,
    )
except ImportError:
    from canonicalize_runs import canonicalize
    from common import (
        RELIABILITY_ROOT,
        canonical_json,
        ensure_output_layout,
        nested_text,
        read_json,
        resolve_path,
        stable_hash,
        write_csv,
        write_json,
    )
    from gold_rules import citation_links_from_evidence, evidence_id
    from evidence_provenance import (
        audit_evidence_provenance,
        retained_evidence_records,
        retrieved_evidence_records,
    )


UNCERTAINTY_TERMS = (
    "uncertain",
    "uncertainty",
    "insufficient",
    "limited evidence",
    "no direct evidence",
    "ambiguous",
    "cannot determine",
    "could not",
    "failed",
    "unavailable",
    "conflict",
    "contradict",
    "inconsistent",
    "cautious",
    "not experimentally confirmed",
    "不确定",
    "证据不足",
    "缺少直接证据",
    "无法",
    "失败",
    "冲突",
    "谨慎",
)
FORCED_TERMS = (
    "experimentally confirmed",
    "definitively proves",
    "proven fact",
    "known fact",
    "certainly",
    "已被实验确认",
    "确定证明",
    "已证实事实",
)

DETECTION_TERMS = {
    "citation_mismatch": ("citation mismatch", "unsupported citation", "citation is not", "unsupported_claim", "引用错配", "引用不支持"),
    "entity_or_endpoint_mismatch": (
        "endpoint_specific_evidence_missing",
        "compound_specific_evidence_missing",
        "non_endpoint_specific",
        "non_compound_specific",
        "entity mismatch",
        "endpoint mismatch",
        "对象不一致",
        "终点不一致",
    ),
    "irrelevant_or_weak_evidence": ("low_relevance", "background only", "weak evidence", "not direct", "相关性低", "仅为背景"),
    "missing_field_or_overclaim": ("missing field", "model type missing", "score missing", "overclaim", "experimentally confirmed", "字段缺失", "过度结论"),
    "conflicting_evidence": ("conflict", "contradict", "opposite direction", "inconsistent evidence", "冲突", "相反证据"),
    "plausible_but_wrong_prediction": ("prediction conflicts", "model prediction", "not experimentally confirmed", "预测与证据冲突", "仅为预测"),
    "prediction_tool_or_service_failure": (
        "prediction_tool_or_service_failure",
        "prediction tool unavailable",
        "predict_single unavailable",
        "predict_ddi unavailable",
        "tool_failed",
        "service failure",
        "timeout",
        "预测工具",
        "服务失败",
    ),
    "no_direct_evidence_or_ambiguous_entity": ("no direct evidence", "ambiguous entity", "insufficient evidence", "缺少直接证据", "对象存在歧义"),
}


def _clean_system_text(value: Any) -> str:
    markers = ("controlled_injection", "injection_marker", "stress_case_id")

    def sanitize(item: Any) -> Any:
        if isinstance(item, str):
            return "" if any(marker in item.lower() for marker in markers) else item
        if isinstance(item, Mapping):
            return {
                key: sanitize(nested)
                for key, nested in item.items()
                if not any(marker in str(key).lower() for marker in markers)
            }
        if isinstance(item, list):
            return [sanitize(nested) for nested in item]
        if isinstance(item, tuple):
            return tuple(sanitize(nested) for nested in item)
        return item

    return nested_text(sanitize(value)).lower()


def _narrative(context: Mapping[str, Any], report: Mapping[str, Any]) -> str:
    # Evidence and citation records are excluded: their injected text must not
    # count as the system detecting its own error.
    payload = {
        "explanation": report.get("explanation"),
        "reasoning": report.get("reasoning") or context.get("reasoning"),
        "warnings": [w for w in context.get("warnings", []) or [] if "controlled_injection" not in str(w)],
        "reflection": context.get("reflection"),
        "objective_messages": (context.get("objective_status", {}) or {}).get("missing_objectives"),
        "tool_errors": [
            {"tool": call.get("tool"), "status": call.get("status"), "error": call.get("error")}
            for call in context.get("tool_calls", []) or []
            if call.get("status") not in {"completed", "fallback_completed"} or call.get("error")
        ],
    }
    return _clean_system_text(payload)


def _contains(text: str, terms: Iterable[str]) -> bool:
    return any(term.lower() in text for term in terms)


def _final_citation_map(context: Mapping[str, Any], report: Mapping[str, Any]) -> dict[str, str]:
    entries = context.get("citation_map", []) or report.get("citation_map", []) or []
    mapping: dict[str, str] = {}
    for index, item in enumerate(entries):
        if not isinstance(item, Mapping):
            continue
        claim = str(item.get("claim_id") or f"claim_{index + 1}")
        target = item.get("chunk_id") or item.get("source_path") or item.get("citation_id")
        if target is not None:
            mapping[claim] = str(target)
    return mapping


def _verified_empty_evidence_state(context: Mapping[str, Any], report: Mapping[str, Any]) -> bool:
    """Recognize a traceable empty result without treating it as missing data."""

    metadata = context.get("metadata", {}) or {}
    status = report.get("evidence_status") if isinstance(report.get("evidence_status"), Mapping) else {}
    outcome = status.get("retrieval_outcome") or metadata.get("retrieval_outcome")
    direct = status.get("direct_evidence_available")
    if direct is None:
        direct = metadata.get("direct_evidence_available")
    trace = report.get("retrieval_trace") or context.get("retrieval_trace") or []
    backend_valid = bool(
        status.get("real_rag_execution_valid")
        or metadata.get("real_rag_execution_valid")
        or any(
            isinstance(item, Mapping)
            and item.get("backend") in {"elasticsearch_hybrid_bm25_dense", "local_reaction_data"}
            for item in trace
        )
    )
    explicit_fields = "evidence" in report and "citation_map" in report
    narrative = _narrative(context, report)
    explicit_language = _contains(
        narrative,
        ("no direct evidence", "direct evidence was unavailable", "缺少直接证据", "无直接证据"),
    )
    return bool(
        outcome == "no_direct_evidence"
        and direct is False
        and backend_valid
        and explicit_fields
        and not (report.get("evidence") or context.get("evidence"))
        and not (report.get("citation_map") or context.get("citation_map"))
        and explicit_language
    )


def _citation_confusion(
    gold: Mapping[str, Any],
    final: Mapping[str, Any],
    *,
    trace_grounded: Mapping[str, Any] | None = None,
) -> tuple[int, int, int]:
    """Score citations against mandatory gold plus provenance-grounded additions.

    Runtime gold is frozen when the controlled mutation is injected.  A valid
    recovery may retrieve additional evidence afterward; those links must not
    become false positives.  The retrieval trace is immutable under the
    evidence-record injection, so it is a suitable independent allow-list.
    It also lets old runs be read with the current chunk/source-path link
    semantics without rewriting their raw artifacts.
    """

    trace_grounded = trace_grounded or {}
    normalized_gold = {
        str(key): str(trace_grounded.get(str(key), value))
        for key, value in gold.items()
    }
    gold_pairs = set(normalized_gold.items())
    allowed_pairs = gold_pairs | {(str(key), str(value)) for key, value in trace_grounded.items()}
    final_pairs = {(str(key), str(value)) for key, value in final.items()}
    return len(allowed_pairs & final_pairs), len(final_pairs - allowed_pairs), len(gold_pairs - final_pairs)


def _support_claim_counts(context: Mapping[str, Any], report: Mapping[str, Any]) -> tuple[int, int]:
    evidence = retained_evidence_records(context) or retrieved_evidence_records(context)
    valid_ids: set[str] = set()
    for index, item in enumerate(evidence):
        valid_ids.add(evidence_id(item, index))
        for key in ("citation_id", "chunk_id", "source_path"):
            if item.get(key):
                valid_ids.add(str(item[key]))
        nested = item.get("value") if isinstance(item.get("value"), Mapping) else {}
        for key in ("chunk_id", "source_path"):
            if nested.get(key):
                valid_ids.add(str(nested[key]))

    claim_links: dict[str, str | None] = {}

    def add_claim(claim_key: Any, citation_target: Any) -> None:
        key = str(claim_key)
        target = str(citation_target) if citation_target else None
        if key not in claim_links or (claim_links[key] is None and target is not None):
            claim_links[key] = target

    for claim_id, citation_target in _final_citation_map(context, report).items():
        add_claim(claim_id, citation_target)

    reasoning = report.get("reasoning") or context.get("reasoning") or {}
    if isinstance(reasoning, Mapping):
        for index, citation_id in enumerate(reasoning.get("citation_ids", []) or []):
            if citation_id:
                add_claim(f"reasoning_citation_{index + 1}", citation_id)
        for index, claim in enumerate(reasoning.get("claims", []) or []):
            if isinstance(claim, Mapping):
                add_claim(
                    claim.get("claim_id") or claim.get("claim") or f"reasoning_claim_{index + 1}",
                    claim.get("citation_id") or claim.get("chunk_id") or claim.get("source_path"),
                )
    explanation = report.get("explanation") or {}
    if isinstance(explanation, Mapping):
        for index, citation_id in enumerate(explanation.get("citation_ids", []) or []):
            if citation_id:
                add_claim(f"explanation_citation_{index + 1}", citation_id)
        for index, claim in enumerate(explanation.get("claims", []) or []):
            if isinstance(claim, Mapping):
                add_claim(
                    claim.get("claim_id") or claim.get("claim") or f"explanation_claim_{index + 1}",
                    claim.get("citation_id") or claim.get("chunk_id") or claim.get("source_path"),
                )
    unsupported = sum(target is None or target not in valid_ids for target in claim_links.values())
    return unsupported, len(claim_links)


def _unsupported_claim_count(context: Mapping[str, Any], report: Mapping[str, Any]) -> int:
    return _support_claim_counts(context, report)[0]


def _citation_link_metrics(
    context: Mapping[str, Any],
    report: Mapping[str, Any],
) -> tuple[float | None, float | None, str]:
    evidence = retained_evidence_records(context) or retrieved_evidence_records(context)
    valid_ids: set[str] = set()
    for index, item in enumerate(evidence):
        valid_ids.add(evidence_id(item, index))
        for key in ("citation_id", "chunk_id", "source_path"):
            if item.get(key):
                valid_ids.add(str(item[key]))
    final_map = _final_citation_map(context, report)
    targets = list(final_map.values())
    if not targets:
        return None, None, "not_estimable_no_citation_links"
    supported = [target for target in targets if target in valid_ids]
    precision = len(supported) / len(targets)
    cited_unique = len(set(supported))
    recall = cited_unique / len(evidence) if evidence else None
    return precision, recall, "structural_evidence_linkage"


def _compact(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def _expected_scientific_objects(case: Mapping[str, Any]) -> list[str]:
    gold = case.get("gold_plan") if isinstance(case.get("gold_plan"), Mapping) else {}
    explicit = [str(item) for item in gold.get("expected_final_scientific_objects", []) or [] if item]
    if explicit:
        return explicit
    base = case.get("base_case") if isinstance(case.get("base_case"), Mapping) else case
    request = str(base.get("reaction_smiles") or base.get("request") or "").strip()
    task_type = str(base.get("task_type") or "")
    if not request:
        return []
    if task_type == "ddi_prediction":
        return [part.strip() for part in re.split(r"[,;+]", request) if part.strip()]
    if task_type.startswith("reaction_"):
        match = re.search(r"([A-Za-z0-9@+\-\[\]().=#:/\\]+>>[A-Za-z0-9@+\-\[\]().=#:/\\]+)", request)
        return [match.group(1)] if match else []
    if " " not in request or any(token in request for token in ("=", "#", "[", "]", "(")):
        return [request]
    return []


def _scientific_object_alignment(
    case: Mapping[str, Any],
    context: Mapping[str, Any],
    report: Mapping[str, Any],
) -> bool | None:
    expected = _expected_scientific_objects(case)
    if not expected:
        return None
    active = {
        "molecules": report.get("molecules") or context.get("molecules"),
        "ddi_pair": context.get("ddi_pair"),
        "reaction": report.get("reaction") or context.get("reaction"),
        "reaction_profile": report.get("reaction_profile") or context.get("reaction_profile"),
        "predictions": report.get("reaction_predictions") or report.get("predictions"),
    }
    active_compact = _compact(nested_text(active))
    return all(_compact(item) in active_compact for item in expected if _compact(item))


def _report_schema_audit(
    context: Mapping[str, Any],
    report: Mapping[str, Any],
) -> tuple[float, list[str]]:
    metadata = context.get("metadata", {}) or {}
    safe = bool(metadata.get("safe_abstention_required"))
    task_type = str(report.get("task_type") or context.get("task_type") or "")
    checks: dict[str, bool] = {
        "request": bool(report.get("request")),
        "task_type": bool(task_type),
        "explanation": bool(report.get("explanation")),
        "tool_calls": bool(report.get("tool_calls")),
        "retrieval_trace": bool(report.get("retrieval_trace")),
        "objective_status_field": "objective_status" in report,
    }
    if safe:
        predictions = report.get("reaction_predictions") or report.get("predictions") or []
        safety = report.get("safety") if isinstance(report.get("safety"), Mapping) else {}
        checks.update(
            {
                "no_fabricated_prediction": not bool(predictions),
                "safe_abstention_status": bool(safety.get("safe_abstention_required")),
                "failure_warning": bool(report.get("warnings")),
            }
        )
    elif task_type in {"reaction_profile", "reaction_condition_retrieval"}:
        checks.update(
            {
                "reaction_profile_record": bool(report.get("reaction_profile") or context.get("reaction_profile")),
                "evidence_record": bool(report.get("evidence")) or _verified_empty_evidence_state(context, report),
                "citation_map_field": "citation_map" in report,
            }
        )
    elif task_type not in {"general_drug_profile", "out_of_domain", "missing_entity"}:
        predictions = report.get("reaction_predictions") or report.get("predictions") or []
        checks.update(
            {
                "prediction_record": bool(predictions),
                "evidence_record": bool(report.get("evidence")) or _verified_empty_evidence_state(context, report),
                "citation_map_field": "citation_map" in report,
            }
        )
    missing = [name for name, passed in checks.items() if not passed]
    return sum(checks.values()) / len(checks), missing


def evaluate_report_quality(
    *,
    case: Mapping[str, Any],
    context: Mapping[str, Any],
    report: Mapping[str, Any],
    group_evaluation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Audit objectively checkable report properties without Verifier pass fields."""

    group_evaluation = group_evaluation or {}
    narrative = _narrative(context, report)
    provenance = audit_evidence_provenance(context)
    schema_score, schema_missing = _report_schema_audit(context, report)
    broken_links, explicit_claim_links = _support_claim_counts(context, report)
    citation_precision, citation_recall, citation_basis = _citation_link_metrics(context, report)
    if group_evaluation.get("citation_precision_applicable"):
        citation_precision = group_evaluation.get("citation_precision")
        citation_basis = "stress_case_gold_citation_map"
    if group_evaluation.get("citation_recall_applicable"):
        citation_recall = group_evaluation.get("citation_recall")
        citation_basis = "stress_case_gold_citation_map"
    semantic_unsupported_count: int | None = None
    semantic_claim_count: int | None = None
    semantic_unsupported_rate: float | None = None
    unsupported_basis = "not_estimable_without_claim_support_gold"
    if group_evaluation.get("citation_precision_applicable"):
        semantic_unsupported_count = int(group_evaluation.get("citation_fp") or 0)
        semantic_claim_count = int(group_evaluation.get("citation_tp") or 0) + semantic_unsupported_count
        semantic_unsupported_rate = (
            semantic_unsupported_count / semantic_claim_count if semantic_claim_count else None
        )
        unsupported_basis = "stress_case_gold_citation_map"

    alignment = group_evaluation.get("entity_alignment_correct")
    if alignment is None:
        alignment = _scientific_object_alignment(case, context, report)
    directions = {
        str(item.get("evidence_direction") or "neutral")
        for item in (context.get("evidence", []) or [])
        if isinstance(item, Mapping)
    }
    conflict_present = {"support", "contradiction"}.issubset(directions)
    conflict_detected = group_evaluation.get("conflict_detected")
    if conflict_detected is None:
        conflict_detected = conflict_present and _contains(narrative, DETECTION_TERMS["conflicting_evidence"])
    conflict_preserved = group_evaluation.get("conflicting_sources_preserved")
    if conflict_preserved is None:
        conflict_preserved = conflict_present

    uncertainty_required = bool(
        case.get("gold_required_uncertainty")
        or context.get("metadata", {}).get("safe_abstention_required")
        or conflict_present
        or not provenance["direct_evidence_coverage"]
    )
    uncertainty_present = _contains(narrative, UNCERTAINTY_TERMS)
    forced = group_evaluation.get("forced_conclusion")
    if forced is None:
        forced = bool(uncertainty_required and _contains(narrative, FORCED_TERMS) and not uncertainty_present)
    prediction_fact = group_evaluation.get("prediction_fact_separation_correct")
    if prediction_fact is None:
        prediction_fact = _prediction_fact_separated(context, narrative)

    return {
        **provenance,
        "qc_only_rate": 1.0 if provenance["qc_only"] else 0.0,
        "scientific_object_alignment": alignment,
        "citation_precision": citation_precision,
        "citation_recall": citation_recall,
        "citation_metric_basis": citation_basis,
        "broken_or_missing_citation_link_count": broken_links,
        "explicit_claim_link_count": explicit_claim_links,
        "structural_unsupported_link_rate": broken_links / explicit_claim_links if explicit_claim_links else None,
        "unsupported_claim_count_independent": semantic_unsupported_count,
        "claim_count_requiring_support": semantic_claim_count,
        "unsupported_claim_rate": semantic_unsupported_rate,
        "unsupported_claim_metric_basis": unsupported_basis,
        "prediction_fact_separation_accuracy": bool(prediction_fact),
        "uncertainty_expression_accuracy": bool(uncertainty_present if uncertainty_required else not forced),
        "conflict_detection_rate": bool(conflict_detected),
        "conflict_preservation_rate": bool(conflict_preserved),
        "forced_conclusion_rate": bool(forced),
        "report_schema_completeness": schema_score,
        "report_schema_missing_json": canonical_json(schema_missing),
        "objective_coverage": group_evaluation.get("objective_coverage"),
        "state_consistency": group_evaluation.get("state_consistency"),
        "stale_result_removal_accuracy": group_evaluation.get("stale_result_removal_accuracy"),
        "citation_update_accuracy": group_evaluation.get("citation_update_accuracy"),
        "automatic_audit_independent_of_verifier": True,
        "scientific_correctness_evaluable": False,
        "subjective_dimensions_not_evaluated_json": canonical_json(
            ["language_style", "readability", "subjective_persuasiveness", "expert_clinical_utility"]
        ),
    }


def _prediction_complete(context: Mapping[str, Any]) -> bool:
    predictions = context.get("reaction_predictions", []) or context.get("predictions", []) or []
    if not predictions:
        return False
    item = predictions[-1]
    if not isinstance(item, Mapping):
        return False
    has_value = any(item.get(key) is not None for key in ("label", "value", "score", "probability"))
    return bool(item.get("model_type") and has_value)


def _prediction_fact_separated(context: Mapping[str, Any], narrative: str) -> bool:
    predictions = context.get("reaction_predictions", []) or context.get("predictions", []) or []
    if not predictions:
        return not _contains(narrative, FORCED_TERMS)
    prediction_marked = _contains(narrative, ("predict", "model output", "estimated", "预测", "模型输出", "估计"))
    return prediction_marked and not _contains(narrative, FORCED_TERMS)


def _injected_text_absent(fault: str, context: Mapping[str, Any]) -> bool:
    evidence_text = _clean_system_text(context.get("evidence", []))
    if fault == "entity_or_endpoint_mismatch":
        return "unrelated comparator entity" not in evidence_text and "non_target_endpoint" not in evidence_text
    if fault == "irrelevant_or_weak_evidence":
        return "general background information" not in evidence_text or "background_only" not in evidence_text
    if fault == "citation_mismatch":
        return "mismatched::" not in _clean_system_text(context.get("citation_map", []))
    return True


def evaluate_stress_case(
    *,
    case: Mapping[str, Any],
    context: Mapping[str, Any],
    report: Mapping[str, Any],
    hook: Mapping[str, Any],
) -> dict[str, Any]:
    """Score one stress run without using AgentM verifier pass/fail fields."""

    fault = str(case["injection_type"])
    gold = hook.get("runtime_gold") if isinstance(hook.get("runtime_gold"), Mapping) else case
    narrative = _narrative(context, report)
    detected = _contains(narrative, DETECTION_TERMS[fault])
    category_correct = detected
    final_evidence = [item for item in context.get("evidence", []) or [] if isinstance(item, Mapping)]
    final_citations = _final_citation_map(context, report)
    gold_citations = gold.get("gold_citation_map", {}) if isinstance(gold, Mapping) else {}
    trace_citations = citation_links_from_evidence(retrieved_evidence_records(context))
    citation_tp, citation_fp, citation_fn = _citation_confusion(
        gold_citations or {},
        final_citations,
        trace_grounded=trace_citations,
    )
    uncertainty_required = bool(gold.get("gold_required_uncertainty")) if isinstance(gold, Mapping) else False
    uncertainty_present = _contains(narrative, UNCERTAINTY_TERMS)
    forced_conclusion = uncertainty_required and _contains(narrative, FORCED_TERMS) and not uncertainty_present
    fact_separation = _prediction_fact_separated(context, narrative)

    directions = {str(item.get("evidence_direction") or "neutral") for item in final_evidence}
    conflict_detected = fault == "conflicting_evidence" and _contains(narrative, DETECTION_TERMS["conflicting_evidence"])
    conflict_preserved = fault == "conflicting_evidence" and {"support", "contradiction"}.issubset(directions) and len(
        {evidence_id(item, index) for index, item in enumerate(final_evidence)}
    ) >= 2

    repaired = False
    if case.get("recoverability") == "recoverable":
        if fault in {"citation_mismatch", "entity_or_endpoint_mismatch", "irrelevant_or_weak_evidence"}:
            repaired = _injected_text_absent(fault, context) and bool(final_evidence)
        elif fault == "missing_field_or_overclaim":
            repaired = _prediction_complete(context) and not _contains(narrative, FORCED_TERMS)
        elif fault == "conflicting_evidence":
            repaired = conflict_preserved and uncertainty_present and not forced_conclusion

    no_fabricated_prediction = not bool(context.get("reaction_predictions", []) or context.get("predictions", []))
    no_fabricated_evidence = not final_evidence or all(
        not str(item.get("source") or "").startswith("fabricated") for item in final_evidence
    )
    if fault == "prediction_tool_or_service_failure":
        safe_abstention = no_fabricated_prediction and no_fabricated_evidence and uncertainty_present and not forced_conclusion
    elif fault == "no_direct_evidence_or_ambiguous_entity":
        safe_abstention = no_fabricated_evidence and uncertainty_present and fact_separation and not forced_conclusion
    elif fault == "plausible_but_wrong_prediction":
        safe_abstention = fact_separation and (uncertainty_present or not _contains(narrative, FORCED_TERMS))
    else:
        safe_abstention = False

    reflection_used = any(call.get("agent") == "ReflectionAgent" for call in context.get("tool_calls", []) or [])
    pre_reflection_report = hook.get("pre_reflection_report") if isinstance(hook.get("pre_reflection_report"), Mapping) else {}
    pre_narrative = _narrative(context, pre_reflection_report) if pre_reflection_report else ""
    pre_forced = uncertainty_required and _contains(pre_narrative, FORCED_TERMS) and not _contains(pre_narrative, UNCERTAINTY_TERMS)
    broken_links_final, explicit_claim_links = _support_claim_counts(context, report)
    broken_links_pre = _unsupported_claim_count(context, pre_reflection_report) if pre_reflection_report else broken_links_final
    post_injection = hook.get("post_injection_context") if isinstance(hook.get("post_injection_context"), Mapping) else {}
    post_evidence_count = len(post_injection.get("evidence", []) or []) if post_injection else len(final_evidence)
    new_critical_error = bool(
        (forced_conclusion and not pre_forced)
        or broken_links_final > broken_links_pre
        or (post_evidence_count > 0 and len(final_evidence) == 0 and fault != "no_direct_evidence_or_ambiguous_entity")
    )
    false_repair = bool(reflection_used and new_critical_error)

    if gold.get("gold_final_status") == "completed":
        final_status_correct = repaired and bool(report)
    elif gold.get("gold_final_status") == "partial":
        final_status_correct = bool(report) and (uncertainty_present or conflict_preserved or fact_separation)
    else:
        final_status_correct = safe_abstention

    entity_alignment = _injected_text_absent("entity_or_endpoint_mismatch", context)
    notes: list[str] = []
    if not hook.get("applied"):
        notes.append("controlled injection was not applied; run is invalid for the primary stress endpoint")
    if not detected:
        notes.append("no system-generated, category-specific detection signal was found")
    if case.get("recoverability") == "recoverable" and not repaired:
        notes.append("the injected target did not meet the independent repair rule")
    if fault == "plausible_but_wrong_prediction" and not gold.get("gold_truth_reference"):
        notes.append("independent truth reference missing; label recovery is not scored, only prediction–fact separation")

    repair_applicable = case.get("recoverability") == "recoverable"
    abstention_applicable = case.get("recoverability") == "unrecoverable"
    conflict_applicable = fault == "conflicting_evidence"
    false_repair_applicable = bool(reflection_used)
    forced_conclusion_applicable = bool(uncertainty_required)
    citation_precision = citation_tp / (citation_tp + citation_fp) if citation_tp + citation_fp else None
    citation_recall = citation_tp / (citation_tp + citation_fn) if citation_tp + citation_fn else None
    structurally_unsupported_rate = (
        broken_links_final / explicit_claim_links if explicit_claim_links else None
    )
    semantic_claim_count = citation_tp + citation_fp
    unsupported_claim_rate = citation_fp / semantic_claim_count if semantic_claim_count else None

    return {
        "fault_type": fault,
        "recoverability": case.get("recoverability"),
        "stress_task_group": case.get("task_type"),
        "gold_expected_action": gold.get("gold_expected_action"),
        "detected": bool(detected),
        "detection_category_correct": bool(category_correct),
        "repaired": bool(repaired),
        "repair_applicable": bool(repair_applicable),
        "safe_abstention": bool(safe_abstention),
        "safe_abstention_applicable": bool(abstention_applicable),
        "false_repair": bool(false_repair),
        "false_repair_applicable": bool(false_repair_applicable),
        "final_status_correct": bool(final_status_correct),
        "citation_tp": citation_tp,
        "citation_fp": citation_fp,
        "citation_fn": citation_fn,
        "citation_precision": citation_precision,
        "citation_recall": citation_recall,
        "citation_precision_applicable": citation_tp + citation_fp > 0,
        "citation_recall_applicable": citation_tp + citation_fn > 0,
        "broken_or_missing_citation_link_count": broken_links_final,
        "explicit_claim_link_count": explicit_claim_links,
        "structural_unsupported_link_rate": structurally_unsupported_rate,
        "unsupported_claim_count_independent": citation_fp if semantic_claim_count else None,
        "claim_count_requiring_support": semantic_claim_count if semantic_claim_count else None,
        "unsupported_claim_rate": unsupported_claim_rate,
        "unsupported_claim_rate_applicable": semantic_claim_count > 0,
        "unsupported_claim_metric_basis": (
            "stress_case_gold_citation_map" if semantic_claim_count else "not_estimable_without_claim_support_gold"
        ),
        "entity_alignment_correct": bool(entity_alignment),
        "conflict_detected": bool(conflict_detected),
        "conflicting_sources_preserved": bool(conflict_preserved),
        "conflict_applicable": bool(conflict_applicable),
        "forced_conclusion": bool(forced_conclusion),
        "forced_conclusion_applicable": bool(forced_conclusion_applicable),
        "uncertainty_expression_correct": bool(uncertainty_present if uncertainty_required else not forced_conclusion),
        "uncertainty_expression_applicable": bool(uncertainty_required),
        "prediction_fact_separation_correct": bool(fact_separation),
        "evaluation_valid": bool(hook.get("applied")),
        "notes": notes,
    }


def _tool_sequence(context: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [dict(call) for call in context.get("tool_calls", []) or [] if isinstance(call, Mapping)]


def _objective_completed(objective: str, context: Mapping[str, Any], report: Mapping[str, Any]) -> bool:
    objective_l = objective.lower()
    tools = _tool_sequence(context)
    completed_tools = {str(call.get("tool")) for call in tools if call.get("status") in {"completed", "fallback_completed"}}
    predictions = context.get("reaction_predictions", []) or context.get("predictions", []) or []
    text = _narrative(context, report)
    if objective_l.startswith("predict:"):
        endpoint = objective_l.split(":")[-1]
        return any(endpoint in _clean_system_text(item) for item in predictions)
    if objective_l == "reaction_classification":
        return "classify_reaction_type" in completed_tools
    if objective_l == "yield_prediction":
        return "predict_buchwald_hartwig_yield" in completed_tools or _contains(text, ("not applicable", "unsupported"))
    if objective_l == "condition_evidence":
        return "retrieve_orderly_condition_evidence" in completed_tools
    if objective_l == "template_evidence":
        return "retrieve_mit_template_evidence" in completed_tools
    if "compare_evidence" in objective_l:
        return _contains(text, ("compare", "stronger", "weaker", "evidence strength", "比较", "证据强"))
    if "separate" in objective_l:
        return _contains(text, ("separate", "pair-level", "single-drug", "区分", "药物对"))
    return any(token in text for token in re.findall(r"[a-z0-9]+", objective_l) if len(token) > 4)


def evaluate_open_task(*, case: Mapping[str, Any], context: Mapping[str, Any], report: Mapping[str, Any]) -> dict[str, Any]:
    gold = case["gold_plan"]
    tools = _tool_sequence(context)
    completed_tools = [str(call.get("tool")) for call in tools if call.get("status") in {"completed", "fallback_completed"}]
    required_tools = list(gold.get("required_tools", []))
    objectives = list(gold.get("required_objectives", []))
    objective_hits = [_objective_completed(item, context, report) for item in objectives]
    tool_hits = [tool in completed_tools for tool in required_tools]
    order_hits: list[bool] = []
    for before, after in gold.get("required_dependencies", []) or []:
        before_indices = [index for index, tool in enumerate(completed_tools) if tool == before]
        after_indices = [index for index, tool in enumerate(completed_tools) if tool == after]
        order_hits.append(bool(before_indices and after_indices and min(before_indices) < max(after_indices)))
    invalid = [
        call
        for call in tools
        if call.get("status") in {"guard_blocked", "guard_revision_blocked"}
        or "unknown_tool" in str(call.get("error") or "")
        or "tool_agent_mismatch" in str(call.get("error") or "")
    ]
    transient = case.get("transient_fault")
    recovery_success = None
    if isinstance(transient, Mapping):
        target = str(transient.get("tool"))
        attempts = [call for call in tools if call.get("tool") == target]
        recovery_success = bool(
            attempts
            and attempts[0].get("status") == "failed"
            and any(call.get("status") in {"completed", "fallback_completed"} for call in attempts[1:])
        )
    intervention = sum(
        1
        for warning in context.get("warnings", []) or []
        if _contains(str(warning).lower(), ("user input required", "missing parameter", "manual intervention"))
    )
    return {
        "open_task_type": case.get("open_task_type"),
        "objective_coverage": sum(objective_hits) / len(objective_hits) if objective_hits else 0.0,
        "objectives_completed": sum(objective_hits),
        "objectives_required": len(objective_hits),
        "tool_selection_accuracy": sum(tool_hits) / len(tool_hits) if tool_hits else 0.0,
        "dependency_order_accuracy": sum(order_hits) / len(order_hits) if order_hits else 1.0,
        "invalid_tool_call_rate": len(invalid) / len(tools) if tools else 0.0,
        "recovery_success": recovery_success,
        "human_intervention_count": intervention,
        "unsupported_claim_count_independent": _unsupported_claim_count(context, report),
        "evaluation_valid": True,
        "objective_details": dict(zip(objectives, objective_hits)),
        "required_tool_details": dict(zip(required_tools, tool_hits)),
    }


def evaluate_multi_turn(*, case: Mapping[str, Any], turn_contexts: list[Mapping[str, Any]], final_report: Mapping[str, Any]) -> dict[str, Any]:
    gold = case["gold_plan"]
    final_context = turn_contexts[-1] if turn_contexts else {}
    active_report = {
        key: final_report.get(key)
        for key in (
            "request",
            "task_type",
            "dataset_name",
            "molecules",
            "reaction",
            "predictions",
            "reaction_predictions",
            "evidence",
            "evidence_status",
            "explanation",
            "citation_map",
        )
    }
    active_payload = {
        key: final_context.get(key)
        for key in ("request", "task_type", "dataset_name", "molecules", "ddi_pair", "reaction", "predictions", "reaction_predictions", "evidence", "citation_map", "reasoning", "final_report")
    }
    # Audit history (previous_run_context, traces, planner proposals, and retry
    # records) is intentionally retained outside the active scientific state.
    # It must not be counted as a stale prediction, evidence item, or citation.
    active_payload["final_report"] = active_report
    active_text = _clean_system_text(active_payload)
    final_objects = [str(item).lower() for item in gold.get("expected_final_scientific_objects", []) or [] if item]
    object_hits = [obj in active_text for obj in final_objects]
    stale = [str(item).lower() for item in gold.get("stale_records_to_remove", []) or [] if item]
    stale_removed = [item not in active_text for item in stale]
    contamination = any(not value for value in stale_removed)
    all_tools = [call for context in turn_contexts for call in _tool_sequence(context)]
    completed_tools = [str(call.get("tool")) for call in all_tools if call.get("status") in {"completed", "fallback_completed"}]
    required_tools = list(gold.get("required_tools", []))
    tool_hits = [tool in completed_tools for tool in required_tools]
    replanned = len(turn_contexts) == len(case.get("turns", [])) and all(bool(context.get("orchestrator_plan")) for context in turn_contexts)
    final_citations = _final_citation_map(final_context, final_report)
    empty_citation_refresh = _verified_empty_evidence_state(final_context, final_report)
    citation_update = not contamination and (bool(final_citations) or empty_citation_refresh)
    state_consistent = bool(turn_contexts) and all(object_hits or [True]) and all(stale_removed or [True])
    return {
        "multi_turn_type": case.get("multi_turn_type"),
        "objective_coverage": sum(tool_hits) / len(tool_hits) if tool_hits else 0.0,
        "tool_selection_accuracy": sum(tool_hits) / len(tool_hits) if tool_hits else 0.0,
        "state_consistency": bool(state_consistent),
        "stale_result_removal_accuracy": sum(stale_removed) / len(stale_removed) if stale_removed else 1.0,
        "replanning_success": bool(replanned),
        "object_update_accuracy": sum(object_hits) / len(object_hits) if object_hits else 1.0,
        "citation_update_accuracy": bool(citation_update),
        "citation_update_basis": "verified_empty_evidence_state" if empty_citation_refresh and not final_citations else "active_citation_map",
        "cross_turn_contamination": bool(contamination),
        "unsupported_claim_count_independent": _unsupported_claim_count(final_context, final_report),
        "evaluation_valid": len(turn_contexts) == len(case.get("turns", [])),
        "stale_item_details": dict(zip(stale, stale_removed)),
        "final_object_details": dict(zip(final_objects, object_hits)),
    }


def _load_run_payload(raw_dir: Path) -> tuple[dict, dict, dict, dict, dict]:
    record = read_json(raw_dir / "run_record.json", default={}) or {}
    job = read_json(raw_dir / "job.json", default={}) or {}
    context_path = Path(record.get("global_context_path") or raw_dir / "runtime" / "global_context.json")
    report_path = Path(record.get("final_report_path") or raw_dir / "runtime" / "multi_agent_report.json")
    context = read_json(context_path, default={}) or {}
    report = read_json(report_path, default={}) or {}
    hook = read_json(raw_dir / "hook_record.json", default={}) or {}
    return record, job, context, report, hook


def _multi_turn_usage(raw_dir: Path) -> dict[str, int]:
    summaries = [read_json(path, default={}) or {} for path in sorted(raw_dir.glob("turns/turn_*/api_usage/api_usage_summary.json"))]
    return {
        "llm_call_count": sum(int(item.get("n_calls", 0) or 0) for item in summaries),
        "llm_failed_call_count": sum(int(item.get("n_failed_calls", 0) or 0) for item in summaries),
        "prompt_tokens": sum(int(item.get("prompt_tokens", 0) or 0) for item in summaries),
        "completion_tokens": sum(int(item.get("completion_tokens", 0) or 0) for item in summaries),
        "total_tokens": sum(int(item.get("total_tokens", 0) or 0) for item in summaries),
    }


def run_all(output_root: str | Path) -> dict[str, int]:
    layout = ensure_output_layout(output_root)
    evaluation_dir = layout["metrics"] / "evaluations"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    rows_by_group: dict[str, list[dict[str, Any]]] = {"clean": [], "stress": [], "open_task": [], "multi_turn": []}
    report_quality_rows: list[dict[str, Any]] = []
    evidence_provenance_rows: list[dict[str, Any]] = []
    canonical, _ = canonicalize(output_root, require_full=False)
    for selected in canonical:
        raw_dir = selected["raw_dir"]
        stored_record, job, context, report, hook = _load_run_payload(raw_dir)
        record = {**stored_record, **selected["record"]}
        group = str(record.get("dataset_group") or "")
        case = job.get("case", {}) if isinstance(job, Mapping) else {}
        evaluation: dict[str, Any] = {}
        if group == "stress" and case:
            evaluation = evaluate_stress_case(case=case, context=context, report=report, hook=hook)
        elif group == "open_task" and case:
            evaluation = evaluate_open_task(case=case, context=context, report=report)
        elif group == "multi_turn" and case:
            turn_paths = read_json(raw_dir / "turn_context_paths.json", default=[]) or []
            turn_contexts = [read_json(path, default={}) or {} for path in turn_paths]
            evaluation = evaluate_multi_turn(case=case, turn_contexts=turn_contexts, final_report=report)
            record.update(_multi_turn_usage(raw_dir))
        if group == "clean":
            evaluation = {
                "analysis_role": "primary" if int(record.get("repeat_index", 0) or 0) == 0 else "stability_repeat",
                "evaluation_valid": bool(record.get("completed")),
            }
        report_quality = evaluate_report_quality(
            case=case,
            context=context,
            report=report,
            group_evaluation=evaluation,
        )
        scalar_evaluation = {key: value for key, value in evaluation.items() if not isinstance(value, (dict, list))}
        combined = {**record, **scalar_evaluation, **report_quality}
        if group in rows_by_group:
            rows_by_group[group].append(combined)
        identity = {
            key: record.get(key)
            for key in (
                "run_id",
                "case_id",
                "dataset_group",
                "task_type",
                "orchestrator",
                "reflection_enabled",
                "seed",
                "repeat_index",
                "completed",
                "real_rag_valid",
            )
        }
        report_quality_rows.append({**identity, **report_quality})
        evidence_provenance_rows.append(
            {
                **identity,
                **{
                    key: value
                    for key, value in report_quality.items()
                    if key.startswith("evidence_")
                    or key
                    in {
                        "retained_evidence_count",
                        "retrieved_evidence_count",
                        "audited_evidence_count",
                        "external_evidence_count",
                        "direct_evidence_count",
                        "qc_grounding_count",
                        "local_reaction_evidence_count",
                        "external_evidence_coverage",
                        "direct_evidence_coverage",
                        "qc_only",
                        "qc_only_rate",
                    }
                },
            }
        )
        write_json(
            evaluation_dir / f"{record.get('run_id', raw_dir.name)}.json",
            {
                "run_id": record.get("run_id"),
                "case_id": record.get("case_id"),
                "dataset_group": group,
                "evaluator": "agentm_reliability_offline",
                "independence_note": "No AgentM verifier pass/status field is used as scientific ground truth or to determine stress/report-quality outcomes.",
                "evaluation": evaluation,
                "report_quality_audit": report_quality,
                "evaluation_hash": stable_hash({"evaluation": evaluation, "report_quality": report_quality}),
            },
        )
    filenames = {
        "clean": "clean_case_level.csv",
        "stress": "stress_case_level.csv",
        "open_task": "open_task_case_level.csv",
        "multi_turn": "multi_turn_case_level.csv",
    }
    for group, rows in rows_by_group.items():
        write_csv(layout["metrics"] / filenames[group], rows)
    write_csv(layout["metrics"] / "report_quality_case_level.csv", report_quality_rows)
    write_csv(layout["metrics"] / "evidence_provenance_case_level.csv", evidence_provenance_rows)
    return {
        **{group: len(rows) for group, rows in rows_by_group.items()},
        "report_quality": len(report_quality_rows),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Independently score AgentM reliability raw runs.")
    parser.add_argument("output_root", nargs="?", default="outputs/agentm_reliability")
    args = parser.parse_args()
    counts = run_all(resolve_path(args.output_root))
    print(json.dumps(counts, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
