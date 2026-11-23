from __future__ import annotations

from typing import Any, Dict, Mapping


ZERO_SHOT_MODEL_TYPE = "zero_shot_rule_semantic_fusion"


def evaluate_task_objectives(context: Mapping[str, Any]) -> Dict[str, Any]:
    metadata = context.get("metadata", {}) or {}
    verifier = context.get("verifier_report", {}) or {}
    summary = verifier.get("summary", {}) or {}
    prediction_report = verifier.get("prediction", {}) or {}
    prediction_semantics_report = verifier.get("prediction_semantics", {}) or {}
    citation_report = verifier.get("citations", {}) or {}
    schema_report = verifier.get("schema", {}) or {}
    predictions = context.get("predictions", []) or []
    prediction = predictions[0] if predictions else None
    model_type = _prediction_value(prediction, "model_type")
    min_evidence = int(metadata.get("min_evidence", 1) or 0)
    evidence_quality_required = min_evidence > 0 or bool(metadata.get("require_real_rag"))
    evidence_count = len(_unique_evidence(context.get("evidence", []) or []))
    unsupported_claims = int(citation_report.get("n_unsupported", 0) or 0)
    reflection = context.get("reflection", {}) or {}
    reflection_required = bool(metadata.get("enable_reflection", True)) and int(metadata.get("max_reflection_rounds", 1) or 0) > 0
    reflection_applied = bool(
        not reflection_required
        or reflection.get("application_status")
        or metadata.get("reflection_feedback_applied")
        or int(metadata.get("reflection_rounds", 0) or 0) > 0
    )
    if context.get("task_type") in {"out_of_domain", "missing_entity"}:
        checks = {
            "intent_gate_recorded": bool(metadata.get("intent_gate")),
            "endpoint_prediction_not_requested": not bool(predictions),
            "molecule_resolution_not_required": True,
            "schema_complete": bool(schema_report.get("schema_complete") or summary.get("schema_complete")),
        }
        missing = [name for name, ok in checks.items() if not ok]
        return {
            "status": "pass" if not missing else "warning",
            "checks": checks,
            "missing_objectives": missing,
            "min_evidence": min_evidence,
            "evidence_count": evidence_count,
            "unsupported_claims": unsupported_claims,
            "prediction_model_type": None,
            "strict_mode": not bool(metadata.get("enable_fallbacks", False)),
            "profile_type": context.get("task_type"),
        }
    if str(context.get("task_type") or "").startswith("reaction_"):
        task_type = str(context.get("task_type") or "")
        reaction_predictions = context.get("reaction_predictions") or predictions
        prediction_required = task_type in {"reaction_classification", "reaction_yield_prediction"}
        checks = {
            "reaction_parsed": bool(context.get("reaction", {}).get("is_valid_reaction")),
            "reaction_prediction_grounded": (not prediction_required)
            or bool(prediction_report.get("prediction_grounded") or summary.get("prediction_grounded")),
            "reaction_profile_built": bool(context.get("reaction_profile")),
            "evidence_count_minimum": evidence_count >= min_evidence,
            "citation_grounded": bool(citation_report.get("citation_grounded", evidence_count == 0)),
            "unsupported_claims_zero": unsupported_claims == 0,
            "schema_complete": bool(schema_report.get("schema_complete") or summary.get("schema_complete")),
            "reflection_applied": reflection_applied,
        }
        missing = [name for name, ok in checks.items() if not ok]
        return {
            "status": "pass" if not missing else "warning",
            "checks": checks,
            "missing_objectives": missing,
            "min_evidence": min_evidence,
            "evidence_count": evidence_count,
            "unsupported_claims": unsupported_claims,
            "prediction_model_type": _prediction_value(reaction_predictions[-1] if reaction_predictions else None, "model_type"),
            "strict_mode": not bool(metadata.get("enable_fallbacks", False)),
            "profile_type": task_type,
        }
    if context.get("task_type") == "general_drug_profile":
        checks = {
            "molecule_resolved": bool(context.get("molecules")),
            "endpoint_prediction_not_requested": not bool(predictions),
            "evidence_count_minimum": evidence_count >= min_evidence,
            "citation_grounded": bool(citation_report.get("citation_grounded", evidence_count == 0)),
            "evidence_relevance_grounded": (not evidence_quality_required)
            or bool(citation_report.get("evidence_relevance_grounded", evidence_count == 0)),
            "compound_specific_grounded": (not evidence_quality_required)
            or bool(citation_report.get("compound_specific_grounded", evidence_count == 0)),
            "endpoint_specific_not_required": True,
            "unsupported_claims_zero": unsupported_claims == 0,
            "schema_complete": bool(schema_report.get("schema_complete") or summary.get("schema_complete")),
            "reflection_applied": reflection_applied,
        }
        if metadata.get("require_real_rag"):
            checks["real_rag_required"] = _real_rag_execution_valid(metadata)
            checks["direct_evidence_available"] = bool(metadata.get("direct_evidence_available"))
        if metadata.get("require_llm_reasoning"):
            checks["llm_reasoning_required"] = metadata.get("reasoning_mode") == "llm_expert"
        missing = [name for name, ok in checks.items() if not ok]
        status = "pass" if not missing else "warning"
        if metadata.get("require_real_rag") and "real_rag_required" in missing:
            status = "fail"
        if metadata.get("require_llm_reasoning") and "llm_reasoning_required" in missing:
            status = "fail"
        return {
            "status": status,
            "checks": checks,
            "missing_objectives": missing,
            "min_evidence": min_evidence,
            "evidence_count": evidence_count,
            "unsupported_claims": unsupported_claims,
            "prediction_model_type": None,
            "strict_mode": not bool(metadata.get("enable_fallbacks", False)),
            "profile_type": "general_drug_profile",
        }
    checks = {
        "prediction_grounded": bool(prediction_report.get("prediction_grounded") or summary.get("prediction_grounded")),
        "prediction_semantics_preserved": bool(
            prediction_semantics_report.get("multiclass_ddi_semantics_preserved", True)
        ),
        "real_model_prediction": bool(prediction and model_type and str(model_type) != ZERO_SHOT_MODEL_TYPE),
        "confidence_assessed": bool(context.get("confidence")),
        "evidence_count_minimum": evidence_count >= min_evidence,
        "citation_grounded": bool(citation_report.get("citation_grounded", evidence_count == 0)),
        "evidence_relevance_grounded": (not evidence_quality_required)
        or bool(citation_report.get("evidence_relevance_grounded", evidence_count == 0)),
        "compound_specific_grounded": (not evidence_quality_required)
        or bool(citation_report.get("compound_specific_grounded", evidence_count == 0)),
        "endpoint_specific_grounded": (not evidence_quality_required)
        or bool(citation_report.get("endpoint_specific_grounded", True)),
        "unsupported_claims_zero": unsupported_claims == 0,
        "schema_complete": bool(schema_report.get("schema_complete") or summary.get("schema_complete")),
        "reflection_applied": reflection_applied,
    }
    if metadata.get("require_real_model"):
        checks["real_model_required"] = checks["real_model_prediction"]
    if metadata.get("require_real_rag"):
        checks["real_rag_required"] = _real_rag_execution_valid(metadata)
        checks["direct_evidence_available"] = bool(metadata.get("direct_evidence_available"))
    if metadata.get("require_llm_reasoning"):
        checks["llm_reasoning_required"] = metadata.get("reasoning_mode") == "llm_expert"
    missing = [name for name, ok in checks.items() if not ok]
    status = "pass" if not missing else "warning"
    if metadata.get("require_real_model") and "real_model_required" in missing:
        status = "fail"
    if metadata.get("require_real_rag") and "real_rag_required" in missing:
        status = "fail"
    if metadata.get("require_llm_reasoning") and "llm_reasoning_required" in missing:
        status = "fail"
    if "prediction_semantics_preserved" in missing:
        status = "fail"
    return {
        "status": status,
        "checks": checks,
        "missing_objectives": missing,
        "min_evidence": min_evidence,
        "evidence_count": evidence_count,
        "unsupported_claims": unsupported_claims,
        "prediction_model_type": model_type,
        "strict_mode": not bool(metadata.get("enable_fallbacks", False)),
    }


def _prediction_value(prediction: Any, key: str) -> Any:
    if prediction is None:
        return None
    if isinstance(prediction, Mapping):
        return prediction.get(key)
    return getattr(prediction, key, None)


def _real_rag_execution_valid(metadata: Mapping[str, Any]) -> bool:
    """Separate verified retrieval execution from scientific hit coverage."""

    explicit = metadata.get("real_rag_execution_valid")
    if explicit is not None:
        return bool(explicit)
    return bool(
        metadata.get("retrieval_backend_used") == "elasticsearch_hybrid_bm25_dense"
        and metadata.get("retrieval_outcome") in {"evidence_found", "no_direct_evidence"}
    )


def _unique_evidence(items: Any) -> list[Any]:
    unique: list[Any] = []
    seen: set[tuple[str, str, str]] = set()
    for item in items or []:
        if not isinstance(item, Mapping):
            unique.append(item)
            continue
        key = _evidence_key(item)
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


def _evidence_key(item: Mapping[str, Any]) -> tuple[str, str, str]:
    source = str(item.get("source") or item.get("source_group") or item.get("provider") or "")
    locator = str(
        item.get("citation_id")
        or item.get("pmid")
        or item.get("doi")
        or item.get("chunk_id")
        or item.get("source_path")
        or item.get("title")
        or ""
    )
    claim = str(item.get("claim") or item.get("snippet") or item.get("mechanism") or "")[:500]
    return (source, locator, claim)
