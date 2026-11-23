from __future__ import annotations

import re
from typing import Any, Dict, List, Mapping, MutableMapping

from agentm.runtime.prediction_semantics import audit_multiclass_ddi_narrative
from agentm.runtime.retrieval.evidence_schema import compact_term, endpoint_terms, infer_endpoint_from_context


PREDICTION_TOOL_NAMES = {
    "predict_single",
    "predict_ddi",
    "run_zero_shot_fallback",
    "classify_reaction_type",
    "predict_buchwald_hartwig_yield",
}
PREDICTION_IDENTITY_FIELDS = ("task_type", "dataset_name")
PREDICTION_VALUE_FIELDS = ("label", "value", "score", "probability")
REPAIRABLE_MISSING_FIELDS = {"model_type", "score", "probability", "confidence", "uncertainty"}
OVERCLAIM_STATUSES = {
    "experimentally_confirmed",
    "experimentally_confirmed_result",
    "confirmed",
    "proven",
    "ground_truth",
}
EVIDENCE_ID_FIELDS = ("citation_id", "chunk_id")
EVIDENCE_CONTENT_FIELDS = (
    "claim",
    "snippet",
    "endpoint",
    "compound_specific",
    "endpoint_specific",
    "reaction_specific",
    "evidence_scope",
    "source",
    "source_path",
    "reaction_smiles",
    "reactant_smiles",
    "product_smiles",
    "template_r",
    "template_p",
    "conditions",
    "score",
)


def _prediction_mapping(value: Any) -> Dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "to_dict"):
        payload = value.to_dict()
        return dict(payload) if isinstance(payload, Mapping) else {}
    return {}


def _grounded_prediction_outputs(context: Mapping[str, Any]) -> List[Dict[str, Any]]:
    outputs: List[Dict[str, Any]] = []
    for call in context.get("tool_calls", []) or []:
        if (
            not isinstance(call, Mapping)
            or call.get("tool") not in PREDICTION_TOOL_NAMES
            or call.get("status") not in {"completed", "fallback_completed"}
        ):
            continue
        observation = call.get("observation") if isinstance(call.get("observation"), Mapping) else {}
        prediction = _prediction_mapping(observation.get("prediction"))
        if prediction:
            outputs.append(prediction)
    return outputs


def _matching_grounded_prediction(candidate: Mapping[str, Any], grounded: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Return a trace-grounded prediction only when all active values agree.

    Missing schema fields and an unsupported claim-status wrapper are repairable;
    a changed label, score, probability, task, or dataset is deliberately not.
    """

    comparable = set(PREDICTION_IDENTITY_FIELDS) | set(PREDICTION_VALUE_FIELDS) | {"model_type", "details"}
    best: Dict[str, Any] = {}
    best_matches = -1
    for trusted in grounded:
        shared = [key for key in comparable if key in candidate and key in trusted]
        if any(candidate.get(key) != trusted.get(key) for key in shared):
            continue
        identity_matches = sum(
            candidate.get(key) == trusted.get(key)
            for key in PREDICTION_IDENTITY_FIELDS
            if candidate.get(key) is not None and trusted.get(key) is not None
        )
        value_matches = sum(
            candidate.get(key) == trusted.get(key)
            for key in PREDICTION_VALUE_FIELDS
            if candidate.get(key) is not None and trusted.get(key) is not None
        )
        if identity_matches < 1 or value_matches < 1:
            continue
        matches = identity_matches + value_matches + sum(
            candidate.get(key) == trusted.get(key)
            for key in ("model_type", "details")
            if key in candidate and key in trusted
        )
        if matches > best_matches:
            best = trusted
            best_matches = matches
    return dict(best)


def repair_prediction_record_integrity(context: MutableMapping[str, Any]) -> Dict[str, Any]:
    """Repair schema loss/overclaim only from an already grounded tool output.

    This guard is intentionally conservative. It restores a record only when
    every scientific value still present in active state agrees with a
    successful registered prediction-tool trace. Conflicting values are
    detected but never silently overwritten.
    """

    field = "reaction_predictions" if str(context.get("task_type") or "").startswith("reaction_") else "predictions"
    records = list(context.get(field, []) or [])
    if not records:
        return {"detected": False, "repaired": False, "field": field, "records": []}
    grounded = _grounded_prediction_outputs(context)
    if not grounded:
        return {"detected": False, "repaired": False, "field": field, "records": []}

    events: List[Dict[str, Any]] = []
    changed = False
    for index, raw in enumerate(records):
        item = _prediction_mapping(raw)
        if not item:
            continue
        claim_status = str(item.get("claim_status") or "").strip().lower()
        overclaim = claim_status in OVERCLAIM_STATUSES
        nested = _prediction_mapping(item.get("value"))
        wrapper = bool(nested) and not any(item.get(key) is not None for key in ("label", "score", "probability", "model_type"))
        candidate = nested if wrapper else {key: value for key, value in item.items() if key != "claim_status"}
        trusted = _matching_grounded_prediction(candidate, grounded)
        missing = sorted(
            key
            for key in REPAIRABLE_MISSING_FIELDS
            if trusted.get(key) is not None and candidate.get(key) is None
        ) if trusted else []
        detected = bool(overclaim or wrapper or missing)
        if not detected:
            continue
        repaired = bool(trusted and (overclaim or wrapper or missing))
        events.append(
            {
                "index": index,
                "overclaim": overclaim,
                "wrapper": wrapper,
                "missing_fields": missing,
                "repaired": repaired,
                "repair_source": "successful_registered_prediction_tool_trace" if repaired else None,
            }
        )
        if repaired:
            records[index] = trusted
            changed = True

    if not events:
        return {"detected": False, "repaired": False, "field": field, "records": []}

    if changed:
        context[field] = records
        if field == "reaction_predictions" and context.get("predictions"):
            context["predictions"] = list(records)
    warnings = context.setdefault("warnings", [])
    detection_warning = "prediction record missing field or overclaim detected"
    if detection_warning not in warnings:
        warnings.append(detection_warning)
    if changed:
        repair_warning = "prediction record repaired from grounded model-tool output"
        if repair_warning not in warnings:
            warnings.append(repair_warning)
    audit = {
        "detected": True,
        "repaired": changed and all(event["repaired"] for event in events),
        "field": field,
        "records": events,
    }
    context.setdefault("metadata", {})["prediction_record_integrity"] = audit
    return audit


def _evidence_identity(item: Mapping[str, Any]) -> tuple[str, str] | None:
    for key in EVIDENCE_ID_FIELDS:
        if item.get(key):
            return key, str(item[key])
    return None


def _grounded_evidence_outputs(context: Mapping[str, Any]) -> Dict[tuple[str, str], Dict[str, Any]]:
    grounded: Dict[tuple[str, str], Dict[str, Any]] = {}

    def add(values: Any) -> None:
        for raw in values or []:
            if not isinstance(raw, Mapping):
                continue
            item = dict(raw)
            key = _evidence_identity(item)
            if key is not None and key not in grounded:
                grounded[key] = item

    for call in context.get("tool_calls", []) or []:
        if (
            not isinstance(call, Mapping)
            or call.get("status") not in {"completed", "fallback_completed"}
            or call.get("agent") not in {"RetrieveAgent", "ReactionAgent"}
        ):
            continue
        observation = call.get("observation") if isinstance(call.get("observation"), Mapping) else {}
        add(observation.get("evidence_items"))
        add(observation.get("evidence"))
    for trace in context.get("retrieval_trace", []) or []:
        if isinstance(trace, Mapping):
            add(trace.get("evidence_items"))
    return grounded


def _evidence_anomaly_category(active: Mapping[str, Any], trusted: Mapping[str, Any]) -> str | None:
    changed = any(
        key in active and key in trusted and active.get(key) != trusted.get(key)
        for key in EVIDENCE_CONTENT_FIELDS
    )
    nested = active.get("value") if isinstance(active.get("value"), Mapping) else {}
    mismatch_marker = bool(
        active.get("endpoint") == "non_target_endpoint"
        or nested.get("endpoint") == "non_target_endpoint"
        or active.get("compound_specific") is False
        or nested.get("compound_specific") is False
        or active.get("endpoint_specific") is False
        or nested.get("endpoint_specific") is False
    )
    weak_marker = bool(
        active.get("evidence_scope") == "background_only"
        or active.get("reaction_specific") is False
        or "general background information" in str(active.get("claim") or "").lower()
    )
    # The controlled weak-evidence mutation deliberately marks the record as
    # non-specific as part of the weakened record.  Classify its explicit
    # background-only markers before the generic entity/endpoint flags so the
    # report names the injected fault rather than a broader downstream symptom.
    if weak_marker:
        return "irrelevant_or_weak_evidence"
    if mismatch_marker:
        return "entity_or_endpoint_mismatch"
    return "grounded_evidence_content_mismatch" if changed else None


def repair_evidence_record_integrity(context: MutableMapping[str, Any]) -> Dict[str, Any]:
    """Restore mutated evidence only from its successful retrieval trace.

    New evidence without a matching trace, such as a declared conflicting peer,
    is retained. Only an existing citation/chunk whose scientific content
    diverges from the captured retrieval output is restored.
    """

    records = list(context.get("evidence", []) or [])
    grounded = _grounded_evidence_outputs(context)
    if not records or not grounded:
        return {"detected": False, "repaired": False, "field": "evidence", "records": []}
    events: List[Dict[str, Any]] = []
    changed = False
    for index, raw in enumerate(records):
        if not isinstance(raw, Mapping):
            continue
        active = dict(raw)
        key = _evidence_identity(active)
        trusted = grounded.get(key) if key is not None else None
        if trusted is None:
            continue
        category = _evidence_anomaly_category(active, trusted)
        if category is None:
            continue
        repaired = dict(trusted)
        for scoped_key in (
            "open_task_subtask_id",
            "open_task_dataset_name",
            "open_task_evidence_scope",
            "evidence_direction",
        ):
            if scoped_key in active and scoped_key not in repaired:
                repaired[scoped_key] = active[scoped_key]
        records[index] = repaired
        changed = True
        events.append(
            {
                "index": index,
                "identity": list(key) if key is not None else None,
                "category": category,
                "repaired": True,
                "repair_source": "successful_retrieval_trace",
            }
        )
    if not events:
        return {"detected": False, "repaired": False, "field": "evidence", "records": []}
    if changed:
        context["evidence"] = records
    warnings = context.setdefault("warnings", [])
    for category in sorted({event["category"] for event in events}):
        if category == "entity_or_endpoint_mismatch":
            warning = "evidence record entity mismatch or endpoint mismatch detected and repaired from retrieval trace"
        elif category == "irrelevant_or_weak_evidence":
            warning = (
                "irrelevant or weak evidence detected and repaired from retrieval trace; "
                "the resulting conclusion should be interpreted cautiously"
            )
        else:
            warning = "grounded evidence content mismatch detected and repaired from retrieval trace"
        if warning not in warnings:
            warnings.append(warning)
    audit = {
        "detected": True,
        "repaired": changed and all(event["repaired"] for event in events),
        "field": "evidence",
        "records": events,
    }
    context.setdefault("metadata", {})["evidence_record_integrity"] = audit
    return audit


def verify_prediction_grounding(context: Mapping[str, Any]) -> Dict[str, Any]:
    if str(context.get("task_type") or "").startswith("reaction_"):
        predictions = context.get("reaction_predictions") or context.get("predictions", []) or []
        tool_calls = context.get("tool_calls", []) or []
        grounded = bool(predictions) and any(
            call.get("agent") == "ReactionAgent"
            and call.get("tool") in {"classify_reaction_type", "predict_buchwald_hartwig_yield"}
            and call.get("status") in {"completed", "fallback_completed"}
            for call in tool_calls
        )
        if context.get("task_type") in {"reaction_profile", "reaction_condition_retrieval"} and not predictions:
            grounded = True
        return {
            "prediction_grounded": grounded,
            "n_predictions": len(predictions),
            "prediction_tool_calls": [
                call
                for call in tool_calls
                if call.get("agent") == "ReactionAgent"
                and call.get("tool") in {"classify_reaction_type", "predict_buchwald_hartwig_yield"}
            ],
            "warnings": [] if grounded else ["reaction_prediction_missing_or_not_reaction_agent_grounded"],
        }
    if context.get("task_type") == "general_drug_profile":
        return {
            "prediction_grounded": True,
            "prediction_not_requested": True,
            "n_predictions": len(context.get("predictions", []) or []),
            "prediction_tool_calls": [],
            "warnings": [],
        }
    predictions = context.get("predictions", []) or []
    tool_calls = context.get("tool_calls", []) or []
    grounded = bool(predictions) and any(
        call.get("agent") == "GaugeAgent"
        and call.get("tool") in {"predict_single", "predict_ddi", "run_zero_shot_fallback"}
        and call.get("status") in {"completed", "fallback_completed"}
        for call in tool_calls
    )
    return {
        "prediction_grounded": grounded,
        "n_predictions": len(predictions),
        "prediction_tool_calls": [
            call for call in tool_calls if call.get("agent") == "GaugeAgent" and call.get("tool") in {"predict_single", "predict_ddi", "run_zero_shot_fallback"}
        ],
        "warnings": [] if grounded else ["prediction_missing_or_not_gauge_grounded"],
    }


def verify_prediction_semantics(context: Mapping[str, Any]) -> Dict[str, Any]:
    """Independently check that multiclass DDI labels remain multiclass in prose."""

    return audit_multiclass_ddi_narrative(context)


def verify_citations(context: Mapping[str, Any]) -> Dict[str, Any]:
    if str(context.get("task_type") or "").startswith("reaction_"):
        return _verify_reaction_citations(context)
    evidence = context.get("evidence", []) or []
    artifacts = context.get("artifacts", {}) or {}
    retrieved = artifacts.get("retrieved_context", []) or artifacts.get("rag_context", []) or []
    retrieved_ids = {str(item.get("chunk_id")) for item in retrieved if isinstance(item, Mapping) and item.get("chunk_id") is not None}
    query_terms = _query_terms(context)
    molecule_terms = _molecule_terms(context)
    endpoint = "" if context.get("task_type") == "general_drug_profile" else infer_endpoint_from_context(context)
    default_endpoint_terms = set(endpoint_terms(endpoint))
    citation_map: List[Dict[str, Any]] = []
    unsupported = 0
    low_relevance = 0
    non_compound_specific = 0
    non_endpoint_specific = 0
    for idx, item in enumerate(evidence):
        if not isinstance(item, Mapping):
            item = {"claim": str(item)}
        nested_value = item.get("value") if isinstance(item.get("value"), Mapping) else {}
        value = {**item, **nested_value}
        scoped_endpoint = str(item.get("open_task_dataset_name") or "")
        endpoint_term_set = set(endpoint_terms(scoped_endpoint)) if scoped_endpoint else default_endpoint_terms
        chunk_id = item.get("chunk_id") or value.get("chunk_id")
        source_path = item.get("source_path") or value.get("source_path")
        source = item.get("source") or value.get("source") or "unknown"
        claim = item.get("claim") or item.get("mechanism") or item.get("interaction_type") or item.get("snippet") or value.get("snippet") or str(item)[:240]
        relevance = _evidence_relevance(
            claim,
            value,
            query_terms=query_terms,
            molecule_terms=molecule_terms,
            endpoint_terms=endpoint_term_set,
            source=str(source),
        )
        citation_exists = (
            bool(chunk_id and str(chunk_id) in retrieved_ids)
            or bool(source_path)
            or source in {"feature_similarity", "local_ddi_hint", "pubchem", "dailymed", "chembl", "pubmed", "elasticsearch", "database_profile", "resolver_metadata"}
            or source in {"mit_reaction_template", "orderly_condition", "reaction_model", "reaction_evidence"}
        )
        if not citation_exists:
            unsupported += 1
        if not relevance["relevance_ok"]:
            low_relevance += 1
        if not relevance["compound_specific"]:
            non_compound_specific += 1
        if endpoint_term_set and not relevance["endpoint_specific"]:
            non_endpoint_specific += 1
        citation_map.append(
            {
                "claim_id": item.get("citation_id") or f"claim_{idx + 1}",
                "citation_id": item.get("citation_id"),
                "claim": claim,
                "source": source,
                "chunk_id": chunk_id,
                "source_path": source_path,
                "citation_exists": citation_exists,
                "relevance_ok": relevance["relevance_ok"],
                "relevance_score": relevance["relevance_score"],
                "matched_terms": relevance["matched_terms"],
                "compound_specific": relevance["compound_specific"],
                "endpoint_specific": relevance["endpoint_specific"],
                "endpoint_matches": relevance["endpoint_matches"],
                "evidence_direction": item.get("evidence_direction", "neutral"),
                "open_task_subtask_id": item.get("open_task_subtask_id"),
                "open_task_evidence_scope": item.get("open_task_evidence_scope"),
            }
        )
    warnings = []
    if unsupported:
        warnings.append(f"unsupported_claims:{unsupported}")
    if low_relevance:
        warnings.append(f"low_relevance_evidence:{low_relevance}")
    if evidence and molecule_terms and non_compound_specific == len(evidence):
        warnings.append("compound_specific_evidence_missing")
    elif non_compound_specific:
        warnings.append(f"non_compound_specific_evidence:{non_compound_specific}")
    if evidence and endpoint_term_set and non_endpoint_specific == len(evidence):
        warnings.append("endpoint_specific_evidence_missing")
    elif non_endpoint_specific:
        warnings.append(f"non_endpoint_specific_evidence:{non_endpoint_specific}")
    return {
        "citation_grounded": unsupported == 0,
        "evidence_relevance_grounded": low_relevance == 0,
        "compound_specific_grounded": not (evidence and molecule_terms and non_compound_specific > 0),
        "endpoint_specific_grounded": not (evidence and endpoint_term_set and non_endpoint_specific > 0),
        "n_evidence_items": len(evidence),
        "n_claims": len(citation_map),
        "n_unsupported": unsupported,
        "n_low_relevance": low_relevance,
        "n_non_compound_specific": non_compound_specific,
        "n_non_endpoint_specific": non_endpoint_specific,
        "citation_map": citation_map,
        "warnings": warnings,
    }


def _verify_reaction_citations(context: Mapping[str, Any]) -> Dict[str, Any]:
    evidence = context.get("evidence", []) or []
    citation_map: List[Dict[str, Any]] = []
    unsupported = 0
    low_relevance = 0
    non_compound_specific = 0
    non_endpoint_specific = 0
    for idx, item in enumerate(evidence):
        if not isinstance(item, Mapping):
            item = {"claim": str(item)}
        source = str(item.get("source") or "unknown")
        citation_id = item.get("citation_id") or f"reaction_claim_{idx + 1}"
        claim = item.get("claim") or item.get("snippet") or str(item)[:240]
        citation_exists = bool(citation_id) and source in {"mit_reaction_template", "orderly_condition", "reaction_model", "reaction_evidence"}
        citation_exists = citation_exists or bool(item.get("source_path"))
        reaction_specific = item.get("reaction_specific") is not False
        compound_specific = item.get("compound_specific") is not False and reaction_specific
        endpoint_specific = item.get("endpoint_specific") is not False and reaction_specific
        relevance_ok = reaction_specific and compound_specific and endpoint_specific
        if not citation_exists:
            unsupported += 1
        if not relevance_ok:
            low_relevance += 1
        if not compound_specific:
            non_compound_specific += 1
        if not endpoint_specific:
            non_endpoint_specific += 1
        citation_map.append(
            {
                "claim_id": citation_id,
                "citation_id": citation_id,
                "claim": claim,
                "source": source,
                "chunk_id": item.get("chunk_id"),
                "source_path": item.get("source_path"),
                "citation_exists": citation_exists,
                "relevance_ok": relevance_ok,
                "relevance_score": item.get("score", 1.0),
                "matched_terms": ["reaction"],
                "compound_specific": compound_specific,
                "endpoint_specific": endpoint_specific,
                "endpoint_matches": ["reaction"],
                "reaction_specific": reaction_specific,
                "evidence_direction": item.get("evidence_direction", "neutral"),
            }
        )
    warnings = [f"unsupported_claims:{unsupported}"] if unsupported else []
    if low_relevance:
        warnings.append(f"low_relevance_evidence:{low_relevance}")
    if non_compound_specific:
        warnings.append(f"non_compound_specific_evidence:{non_compound_specific}")
    if non_endpoint_specific:
        warnings.append(f"non_endpoint_specific_evidence:{non_endpoint_specific}")
    return {
        "citation_grounded": unsupported == 0,
        "evidence_relevance_grounded": low_relevance == 0,
        "compound_specific_grounded": non_compound_specific == 0,
        "endpoint_specific_grounded": non_endpoint_specific == 0,
        "reaction_specific_grounded": low_relevance == 0,
        "n_evidence_items": len(evidence),
        "n_claims": len(citation_map),
        "n_unsupported": unsupported,
        "n_low_relevance": low_relevance,
        "n_non_compound_specific": non_compound_specific,
        "n_non_endpoint_specific": non_endpoint_specific,
        "citation_map": citation_map,
        "warnings": warnings,
    }


def _query_terms(context: Mapping[str, Any]) -> set[str]:
    values: List[str] = [
        str(context.get("request") or ""),
        str(context.get("task_type") or ""),
        str(context.get("dataset_name") or ""),
    ]
    for item in context.get("retrieval_trace", []) or []:
        if isinstance(item, Mapping):
            values.append(str(item.get("query") or ""))
    artifacts = context.get("artifacts", {}) or {}
    for item in artifacts.get("retrieved_context", []) or artifacts.get("rag_context", []) or []:
        if isinstance(item, Mapping):
            values.append(str(item.get("query") or ""))
    terms = _terms(" ".join(values))
    aliases = set()
    if {"bbbp", "bbb"} & terms or "blood" in terms or "brain" in terms:
        aliases.update({"bbbp", "bbb", "blood", "brain", "barrier", "permeability", "permeable", "penetration", "cns"})
    if "dili" in terms:
        aliases.update({"dili", "liver", "hepatic", "hepatotoxicity", "injury"})
    if "herg" in terms:
        aliases.update({"herg", "qt", "cardiac", "arrhythmia"})
    return terms | aliases


def _molecule_terms(context: Mapping[str, Any]) -> set[str]:
    values: List[str] = []
    for molecule in context.get("molecules", []) or []:
        if isinstance(molecule, Mapping):
            values.extend(str(molecule.get(key) or "") for key in ["name", "original_input", "canonical_smiles"])
            values.extend(str(item) for item in molecule.get("synonyms", []) or [])
        else:
            values.extend(str(getattr(molecule, key, "") or "") for key in ["name", "original_input", "canonical_smiles"])
            values.extend(str(item) for item in getattr(molecule, "synonyms", []) or [])
    request_terms = _terms(str(context.get("request") or ""))
    common_drugs = {"aspirin", "warfarin", "caffeine", "atorvastatin", "simvastatin", "ketoconazole"}
    values.extend(sorted(request_terms & common_drugs))
    return {term for term in _terms(" ".join(values)) if (len(term) >= 5 or term in common_drugs) and not term.startswith("c1")}


def _evidence_relevance(
    claim: Any,
    value: Mapping[str, Any],
    *,
    query_terms: set[str],
    molecule_terms: set[str],
    endpoint_terms: set[str],
    source: str,
) -> Dict[str, Any]:
    # Relevance must be judged from evidence text, not from retrieval query metadata.
    # Otherwise an unrelated chunk retrieved by an "aspirin BBBP" query can look
    # compound-specific simply because the query was stored beside the chunk.
    text = " ".join(
        str(part or "")
        for part in [
            claim,
            value.get("snippet") if isinstance(value, Mapping) else "",
            value.get("title") if isinstance(value, Mapping) else "",
        ]
    )
    text_terms = _terms(text)
    matched = sorted(term for term in query_terms if term in text_terms)
    compound_matches = sorted(term for term in molecule_terms if term in text_terms)
    metadata = value.get("metadata") if isinstance(value.get("metadata"), Mapping) else {}
    metadata_endpoint_specific = bool(value.get("endpoint_specific")) or bool(metadata.get("endpoint_specific"))
    metadata_endpoint = compact_term(value.get("endpoint") or metadata.get("endpoint") or "") if metadata_endpoint_specific else ""
    metadata_endpoint_terms = (
        set(str(item).lower() for item in (value.get("endpoint_terms") or metadata.get("endpoint_terms") or []) if item)
        if metadata_endpoint_specific
        else set()
    )
    endpoint_matches = sorted(term for term in endpoint_terms if term in text_terms or term in metadata_endpoint_terms or compact_term(term) == metadata_endpoint)
    structured_source = source in {"pubchem", "dailymed", "chembl", "database_profile", "resolver_metadata", "feature_similarity", "local_ddi_hint"}
    metadata_compound_specific = bool(value.get("compound_specific")) or bool(metadata.get("compound_specific"))
    endpoint_specific = bool(endpoint_matches) or metadata_endpoint_specific
    compound_specific = structured_source or metadata_compound_specific or bool(compound_matches)
    score = len(matched) + (2 if compound_specific else 0) + (2 if endpoint_specific else 0)
    return {
        "relevance_ok": structured_source or score >= 2,
        "relevance_score": score,
        "matched_terms": matched[:12],
        "compound_specific": compound_specific,
        "endpoint_specific": endpoint_specific,
        "endpoint_matches": endpoint_matches[:12],
    }


def _terms(text: str) -> set[str]:
    stop = {
        "the",
        "and",
        "for",
        "with",
        "that",
        "this",
        "from",
        "task",
        "prediction",
        "evidence",
        "citation",
        "acid",
        "sodium",
        "tablet",
        "capsule",
        "是否",
        "通过",
        "任务",
        "解释",
        "依据",
    }
    return {term for term in re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{2,}", str(text).lower()) if term not in stop}


def verify_report_schema(context: Mapping[str, Any]) -> Dict[str, Any]:
    report = context.get("final_report") or {}
    required = [
        "request",
        "task_type",
        "molecules",
        "predictions",
        "evidence",
        "explanation",
        "artifacts",
        "warnings",
        "trace",
        "reasoning",
    ]
    missing = [field for field in required if field not in report]
    return {
        "schema_complete": not missing,
        "missing_fields": missing,
        "warnings": [] if not missing else [f"report_missing_fields:{','.join(missing)}"],
    }


def merge_verifier_reports(*reports: Mapping[str, Any]) -> Dict[str, Any]:
    warnings: List[str] = []
    merged: Dict[str, Any] = {}
    for report in reports:
        merged.update(report)
        warnings.extend(report.get("warnings", []) or [])
    merged["warnings"] = warnings
    merged["status"] = "pass" if not warnings else "warning"
    return merged
