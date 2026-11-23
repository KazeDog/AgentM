from __future__ import annotations

from typing import Any, Dict, List, Mapping, Tuple

import numpy as np


SOURCE_RELIABILITY = {
    "local_ddi_hint": 0.95,
    "dailymed": 0.90,
    "pubmed": 0.85,
    "pubchem": 0.78,
    "llm_rule_feature": 0.65,
    "rdkit_descriptor": 0.58,
    "feature_similarity": 0.45,
    "agent_confidence_review": 0.35,
    "external_evidence_error": 0.0,
}


def score_evidence_for_prediction(
    evidence: List[Mapping[str, Any]],
    prediction: Mapping[str, Any],
    *,
    task_type: str,
    dataset_name: str,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    annotated: List[Dict[str, Any]] = []
    support = 0.0
    contradiction = 0.0
    informative = 0
    reliabilities: List[float] = []
    sources = set()

    for item in evidence:
        scored = _score_item(dict(item), prediction, task_type=task_type, dataset_name=dataset_name)
        annotated.append(scored)
        sources.add(str(scored.get("source")))
        direction = scored["evidence_direction"]
        weighted_strength = float(scored["evidence_weight"]) * float(scored["source_reliability"])
        if direction == "support":
            support += weighted_strength
            informative += 1
            reliabilities.append(float(scored["source_reliability"]))
        elif direction == "contradiction":
            contradiction += weighted_strength
            informative += 1
            reliabilities.append(float(scored["source_reliability"]))

    denom = support + contradiction
    agreement = support / denom if denom > 0 else 0.0
    contradiction_score = contradiction / denom if denom > 0 else 0.0
    reliability_score = float(np.mean(reliabilities)) if reliabilities else 0.0
    coverage_score = float(min(1.0, 0.25 * informative + 0.10 * max(0, len(sources) - 1)))
    grounded = float(np.clip(0.55 * agreement + 0.25 * reliability_score + 0.20 * coverage_score, 0.0, 1.0))

    return (
        {
            "evidence_grounded_score": grounded,
            "support_score": float(agreement),
            "contradiction_score": float(contradiction_score),
            "source_reliability_score": reliability_score,
            "coverage_score": coverage_score,
            "coverage_level": _coverage_level(coverage_score),
            "n_evidence_items": len(evidence),
            "n_informative_items": informative,
            "sources": sorted(sources),
            "recommendation": _recommendation(grounded, contradiction_score, informative),
            "evidence_quality_summary": _quality_summary(annotated),
        },
        annotated,
    )


def _score_item(item: Dict[str, Any], prediction: Mapping[str, Any], *, task_type: str, dataset_name: str) -> Dict[str, Any]:
    quality = item.get("evidence_quality") if isinstance(item.get("evidence_quality"), Mapping) else None
    if quality:
        item["source_reliability"] = float(quality.get("source_reliability_calibrated", item.get("source_reliability", 0.40)) or 0.0)
        item["evidence_direction"] = str(quality.get("direction_rule", item.get("evidence_direction", "neutral")) or "neutral")
        item["evidence_weight"] = float(item.get("evidence_weight", 0.0) or 0.0)
        if item["evidence_direction"] != "neutral" and item["evidence_weight"] <= 0:
            relevance = float(quality.get("relevance_score", 0.0) or 0.0)
            reliability = float(quality.get("source_reliability_calibrated", 0.0) or 0.0)
            item["evidence_weight"] = float(min(0.75, max(0.15, 0.35 * relevance + 0.30 * reliability)))
        item["evidence_scoring_rationale"] = str(quality.get("quality_rationale") or item.get("evidence_scoring_rationale") or "")
        return item

    source = str(item.get("source") or "unknown")
    reliability = SOURCE_RELIABILITY.get(source, 0.40)
    pred_positive = _prediction_is_positive(prediction)
    direction = "neutral"
    strength = 0.0
    rationale = "Evidence was retained as context but was not directional for this task."

    claim = " ".join(
        str(item.get(key) or "")
        for key in ["claim", "mechanism", "interaction_type", "section", "title"]
    ).lower()

    if source == "external_evidence_error":
        direction, strength, rationale = "neutral", 0.0, "External retrieval failed; this lowers coverage but is not evidence for or against the prediction."
    elif source == "local_ddi_hint":
        direction = "support" if pred_positive else "contradiction"
        strength = _severity_strength(item.get("severity"))
        rationale = "Curated local DDI hint is directional for interaction risk."
    elif source in {"dailymed", "pubmed", "pubchem"} and task_type == "ddi_prediction":
        has_signal = _external_item_has_signal(item)
        if has_signal:
            direction = "support" if pred_positive else "contradiction"
            strength = 0.65 if source == "dailymed" else 0.55 if source == "pubmed" else 0.45
            rationale = f"{source} returned pair-level evidence or annotation; it supports an interaction-risk prediction more than a no-interaction prediction."
    elif source == "feature_similarity" and task_type == "ddi_prediction":
        similarity = _extract_similarity(item.get("value"))
        if similarity >= 0.45:
            direction = "support" if pred_positive else "contradiction"
            strength = min(0.45, similarity)
            rationale = "High fingerprint similarity is weak directional evidence for shared interaction mechanisms."
        elif similarity <= 0.15:
            direction = "support" if not pred_positive else "contradiction"
            strength = 0.20
            rationale = "Low fingerprint similarity weakly supports a no-interaction prediction."
    elif source == "rdkit_descriptor":
        risk_like = any(token in claim for token in ["high", "risk", "poor", "reduce", "liability"])
        if risk_like:
            risk_prediction = pred_positive if task_type == "admet_prediction" else not pred_positive if dataset_name == "BBBP" else pred_positive
            direction = "support" if risk_prediction else "contradiction"
            strength = 0.35
            rationale = "Descriptor evidence indicates a risk or unfavorable property cue."
    elif source == "llm_rule_feature":
        risk_like = any(token in claim for token in ["risk", "alert", "tox", "high", "above", "inhibit"])
        if risk_like:
            risk_prediction = pred_positive if task_type in {"admet_prediction", "ddi_prediction"} else not pred_positive if dataset_name == "BBBP" else pred_positive
            direction = "support" if risk_prediction else "contradiction"
            strength = 0.35
            rationale = "Matched LLM4SD-style rule is interpreted as a directional task cue."
        else:
            direction = "support"
            strength = 0.20
            rationale = "Matched LLM4SD-style rule is a weak positive task cue."

    item["source_reliability"] = reliability
    item["evidence_direction"] = direction
    item["evidence_weight"] = float(strength)
    item["evidence_scoring_rationale"] = rationale
    return item


def _prediction_is_positive(prediction: Mapping[str, Any]) -> bool:
    label = prediction.get("label")
    if label is not None:
        try:
            return int(label) == 1
        except Exception:
            pass
    probability = prediction.get("probability")
    score = probability if probability is not None else prediction.get("score")
    try:
        return float(score) >= 0.5
    except Exception:
        return False


def _severity_strength(severity: Any) -> float:
    text = str(severity or "").lower()
    if text == "high":
        return 0.95
    if text == "moderate":
        return 0.70
    if text:
        return 0.45
    return 0.50


def _external_item_has_signal(item: Mapping[str, Any]) -> bool:
    value = item.get("value")
    if isinstance(value, Mapping):
        if value.get("pmids") or value.get("cid") or value.get("sections") or value.get("interaction"):
            return True
    return bool(value or item.get("claim") or item.get("mechanism"))


def _extract_similarity(value: Any) -> float:
    if isinstance(value, Mapping):
        for key in ["morgan_tanimoto", "maccs_tanimoto", "similarity"]:
            if key in value:
                try:
                    return float(value[key])
                except Exception:
                    continue
    try:
        return float(value)
    except Exception:
        return 0.0


def _coverage_level(score: float) -> str:
    if score >= 0.75:
        return "strong"
    if score >= 0.45:
        return "moderate"
    if score > 0:
        return "weak"
    return "none"


def _recommendation(grounded_score: float, contradiction_score: float, informative: int) -> str:
    if informative == 0:
        return "insufficient_evidence"
    if contradiction_score >= 0.55:
        return "evidence_conflicts_with_prediction"
    if grounded_score >= 0.70:
        return "prediction_evidence_supported"
    if grounded_score >= 0.45:
        return "prediction_weakly_supported"
    return "prediction_has_low_evidence_support"


def _quality_summary(annotated: List[Mapping[str, Any]]) -> Dict[str, Any]:
    rows = [item.get("evidence_quality") for item in annotated if isinstance(item.get("evidence_quality"), Mapping)]
    if not rows:
        return {}
    relevance = [float(row.get("relevance_score", 0.0) or 0.0) for row in rows]
    reliability = [float(row.get("source_reliability_calibrated", 0.0) or 0.0) for row in rows]
    rerank = [float(row.get("rerank_score", 0.0) or 0.0) for row in rows]
    quality = [float(row.get("quality_score", 0.0) or 0.0) for row in rows]
    evidence_types: Dict[str, int] = {}
    support_levels: Dict[str, int] = {}
    for row in rows:
        evidence_type = str(row.get("evidence_type") or "unknown")
        support_level = str(row.get("support_level") or "unknown")
        evidence_types[evidence_type] = evidence_types.get(evidence_type, 0) + 1
        support_levels[support_level] = support_levels.get(support_level, 0) + 1
    return {
        "n_quality_annotated": len(rows),
        "average_relevance_score": float(np.mean(relevance)) if relevance else 0.0,
        "average_source_reliability": float(np.mean(reliability)) if reliability else 0.0,
        "average_rerank_score": float(np.mean(rerank)) if rerank else 0.0,
        "average_quality_score": float(np.mean(quality)) if quality else 0.0,
        "evidence_type_counts": evidence_types,
        "support_level_counts": support_levels,
    }
