from __future__ import annotations

import json
from typing import Any, Dict, List, Mapping, Optional

from agentm.llm.provider import build_llm_client


QUALITY_SCORE_KEYS = (
    "report_quality_score",
    "supported_claim_ratio",
    "evidence_citation_rate",
    "explanation_specificity",
    "uncertainty_awareness",
    "structure_explanation_score",
    "hallucination_risk",
)


def evaluate_agent_report_quality(
    case: Mapping[str, Any],
    report: Mapping[str, Any],
    *,
    judge_mode: str = "heuristic",
    judge_profile: Optional[str] = None,
) -> Dict[str, Any]:
    heuristic = heuristic_agent_report_quality(case, report)
    if judge_mode not in {"llm", "auto"}:
        return heuristic

    llm_result = _try_llm_judge(case, report, judge_profile=judge_profile, baseline_payload=None)
    if llm_result is None:
        if judge_mode == "llm":
            heuristic["judge_warning"] = "LLM judge unavailable; used heuristic judge."
        return heuristic
    return _merge_judge_scores(heuristic, llm_result)


def evaluate_vanilla_report_quality(
    case: Mapping[str, Any],
    payload: Mapping[str, Any],
    *,
    judge_mode: str = "heuristic",
    judge_profile: Optional[str] = None,
) -> Dict[str, Any]:
    heuristic = heuristic_vanilla_quality(case, payload)
    if judge_mode not in {"llm", "auto"}:
        return heuristic

    llm_result = _try_llm_judge(case, {}, judge_profile=judge_profile, baseline_payload=payload)
    if llm_result is None:
        if judge_mode == "llm":
            heuristic["judge_warning"] = "LLM judge unavailable; used heuristic judge."
        return heuristic
    return _merge_judge_scores(heuristic, llm_result)


def heuristic_agent_report_quality(case: Mapping[str, Any], report: Mapping[str, Any]) -> Dict[str, Any]:
    evidence = list(report.get("evidence", []) or [])
    artifacts = report.get("artifacts", {}) if isinstance(report.get("artifacts"), Mapping) else {}
    prediction = (report.get("predictions") or [{}])[0]
    details = prediction.get("details", {}) if isinstance(prediction.get("details"), Mapping) else {}
    evidence_contributions = details.get("evidence_contributions", []) if isinstance(details.get("evidence_contributions"), list) else []
    warnings = list(report.get("warnings", []) or [])
    text = _report_text(report)

    supported_claim_ratio = _supported_claim_ratio(evidence, evidence_contributions)
    evidence_citation_rate = _evidence_citation_rate(evidence, text)
    explanation_specificity = _explanation_specificity(text, report)
    uncertainty_awareness = _uncertainty_awareness(text, warnings, details)
    structure_explanation_score = _structure_explanation_score(text, artifacts)
    hallucination_risk = _agent_hallucination_risk(evidence, artifacts, text)
    report_quality_score = _quality_score(
        supported_claim_ratio=supported_claim_ratio,
        evidence_citation_rate=evidence_citation_rate,
        explanation_specificity=explanation_specificity,
        uncertainty_awareness=uncertainty_awareness,
        structure_explanation_score=structure_explanation_score,
        hallucination_risk=hallucination_risk,
    )
    return {
        "judge_provider": "heuristic",
        "report_quality_score": report_quality_score,
        "supported_claim_ratio": supported_claim_ratio,
        "evidence_citation_rate": evidence_citation_rate,
        "explanation_specificity": explanation_specificity,
        "uncertainty_awareness": uncertainty_awareness,
        "structure_explanation_score": structure_explanation_score,
        "hallucination_risk": hallucination_risk,
        "judge_notes": "Heuristic judge based on evidence sources, traceable artifacts, uncertainty wording, and structure-specific explanation cues.",
    }


def heuristic_vanilla_quality(case: Mapping[str, Any], payload: Mapping[str, Any]) -> Dict[str, Any]:
    evidence_claims = payload.get("evidence_claims") if isinstance(payload.get("evidence_claims"), list) else []
    warnings = payload.get("warnings") if isinstance(payload.get("warnings"), list) else []
    answer = " ".join(str(payload.get(key) or "") for key in ["answer", "summary", "rationale"])
    text = f"{answer}\n" + "\n".join(str(item) for item in evidence_claims + warnings)
    supported_claim_ratio = 0.0 if evidence_claims else 0.35
    evidence_citation_rate = 0.15 if evidence_claims else 0.0
    explanation_specificity = _text_specificity(text)
    uncertainty_awareness = _text_uncertainty(text)
    structure_explanation_score = _text_structure_score(text)
    hallucination_risk = 0.75 if evidence_claims else 0.55
    if "do not know" in text.lower() or "cannot verify" in text.lower():
        hallucination_risk = min(hallucination_risk, 0.35)
    report_quality_score = _quality_score(
        supported_claim_ratio=supported_claim_ratio,
        evidence_citation_rate=evidence_citation_rate,
        explanation_specificity=explanation_specificity,
        uncertainty_awareness=uncertainty_awareness,
        structure_explanation_score=structure_explanation_score,
        hallucination_risk=hallucination_risk,
    )
    return {
        "judge_provider": "heuristic",
        "report_quality_score": report_quality_score,
        "supported_claim_ratio": supported_claim_ratio,
        "evidence_citation_rate": evidence_citation_rate,
        "explanation_specificity": explanation_specificity,
        "uncertainty_awareness": uncertainty_awareness,
        "structure_explanation_score": structure_explanation_score,
        "hallucination_risk": hallucination_risk,
        "judge_notes": "Heuristic vanilla-LLM judge penalizes unsupported evidence claims because no retrieval trace is available.",
    }


def _try_llm_judge(
    case: Mapping[str, Any],
    report: Mapping[str, Any],
    *,
    judge_profile: Optional[str],
    baseline_payload: Optional[Mapping[str, Any]],
) -> Optional[Dict[str, Any]]:
    client = build_llm_client(profile_name=judge_profile)
    if not client.available:
        return None
    try:
        prompt = _judge_prompt(case, report, baseline_payload)
        response = client.chat_json(
            [
                {
                    "role": "system",
                    "content": (
                        "You are a strict evaluator for drug-discovery agent reports. "
                        "Return JSON only. Penalize unsupported claims and reward traceable evidence."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
        )
        payload = _parse_json_object(response)
        scores = {key: _clip01(payload.get(key)) for key in QUALITY_SCORE_KEYS}
        scores["judge_provider"] = f"llm:{client.config.profile}:{client.config.model}"
        scores["judge_notes"] = str(payload.get("judge_notes") or payload.get("notes") or "")
        return scores
    except Exception:
        return None


def _judge_prompt(case: Mapping[str, Any], report: Mapping[str, Any], baseline_payload: Optional[Mapping[str, Any]]) -> str:
    if baseline_payload is not None:
        candidate = {"type": "vanilla_llm_payload", "payload": baseline_payload}
    else:
        evidence = list(report.get("evidence", []) or [])[:12]
        artifacts = report.get("artifacts", {}) if isinstance(report.get("artifacts"), Mapping) else {}
        candidate = {
            "type": "agent_report",
            "task_type": report.get("task_type"),
            "prediction": (report.get("predictions") or [{}])[0],
            "evidence": evidence,
            "explanation": report.get("explanation"),
            "narrative_report": report.get("narrative_report", {}),
            "reasoning_summary": (report.get("reasoning") or {}).get("summary") if isinstance(report.get("reasoning"), Mapping) else None,
            "warnings": report.get("warnings", []),
            "artifact_keys": sorted(artifacts.keys()),
        }
    return (
        "Evaluate the candidate report for the natural-language molecular task.\n"
        f"Case:\n{json.dumps(dict(case), ensure_ascii=False, indent=2, default=str)}\n\n"
        f"Candidate:\n{json.dumps(candidate, ensure_ascii=False, indent=2, default=str)}\n\n"
        "Return JSON with numeric scores in [0,1]: report_quality_score, "
        "supported_claim_ratio, evidence_citation_rate, explanation_specificity, "
        "uncertainty_awareness, structure_explanation_score, hallucination_risk, "
        "and a short judge_notes field."
    )


def _report_text(report: Mapping[str, Any]) -> str:
    chunks = [
        str(report.get("explanation") or ""),
        str((report.get("reasoning") or {}).get("summary") if isinstance(report.get("reasoning"), Mapping) else ""),
    ]
    narrative = report.get("narrative_report", {}) if isinstance(report.get("narrative_report"), Mapping) else {}
    chunks.extend(
        str(narrative.get(key) or "")
        for key in ["headline", "plain_language_summary", "evidence_summary", "uncertainty", "recommended_use"]
    )
    chunks.extend(str(item) for item in narrative.get("key_reasons", []) if item)
    structured = narrative.get("structured_sections", {}) if isinstance(narrative.get("structured_sections"), Mapping) else {}
    for value in structured.values():
        if isinstance(value, list):
            chunks.extend(str(item) for item in value if item)
        elif value:
            chunks.append(str(value))
    for item in narrative.get("evidence_citations", []) if isinstance(narrative.get("evidence_citations"), list) else []:
        chunks.append(str(item))
    for item in report.get("evidence", []) or []:
        chunks.append(str(item.get("claim") or item.get("mechanism") or item.get("interaction_type") or ""))
    prediction = (report.get("predictions") or [{}])[0]
    details = prediction.get("details", {}) if isinstance(prediction.get("details"), Mapping) else {}
    for item in details.get("evidence_contributions", []) if isinstance(details.get("evidence_contributions"), list) else []:
        if isinstance(item, Mapping):
            chunks.append(str(item.get("rationale") or ""))
    return "\n".join(chunks)


def _supported_claim_ratio(evidence: List[Mapping[str, Any]], contributions: List[Any]) -> float:
    if not evidence and not contributions:
        return 0.0
    supported_evidence = [
        item
        for item in evidence
        if item.get("source")
        and item.get("source") != "external_evidence_error"
        and str(item.get("evidence_direction", "neutral")) != "neutral"
    ]
    supported_contributions = [item for item in contributions if isinstance(item, Mapping) and item.get("rationale")]
    total = max(1, len(evidence) + len(contributions))
    return _clip01((len(supported_evidence) + len(supported_contributions)) / total)


def _evidence_citation_rate(evidence: List[Mapping[str, Any]], text: str) -> float:
    if not evidence:
        return 0.0
    sources = {str(item.get("source") or "") for item in evidence if item.get("source")}
    source_score = min(1.0, len(sources) / 4.0)
    text_lower = text.lower()
    cue_score = 0.25 if any(source in text_lower for source in sources if source) else 0.0
    return _clip01(0.75 * source_score + cue_score)


def _explanation_specificity(text: str, report: Mapping[str, Any]) -> float:
    score = _text_specificity(text)
    if report.get("molecules"):
        score += 0.15
    if report.get("predictions"):
        score += 0.10
    return _clip01(score)


def _uncertainty_awareness(text: str, warnings: List[Any], details: Mapping[str, Any]) -> float:
    score = _text_uncertainty(text)
    warning_text = " ".join(str(item).lower() for item in warnings)
    if warnings:
        score += 0.35
    if "zero-shot" in warning_text or "non-calibrated" in warning_text or "not calibrated" in warning_text:
        score += 0.30
    if details.get("calibrated_probability") is None and details.get("warning"):
        score += 0.20
    return _clip01(score)


def _structure_explanation_score(text: str, artifacts: Mapping[str, Any]) -> float:
    score = _text_structure_score(text)
    if artifacts.get("visual_semantics") or artifacts.get("drug_a_smiles_features") or artifacts.get("smiles_features"):
        score += 0.20
    if artifacts.get("rule_features"):
        score += 0.15
    if artifacts.get("image_path") or artifacts.get("drug_a_image_path"):
        score += 0.10
    return _clip01(score)


def _agent_hallucination_risk(evidence: List[Mapping[str, Any]], artifacts: Mapping[str, Any], text: str) -> float:
    risk = 0.25
    if not evidence:
        risk += 0.25
    if not artifacts.get("agent_trace_path"):
        risk += 0.20
    if "pubmed" in text.lower() and not any(item.get("source") == "pubmed" for item in evidence):
        risk += 0.30
    if "dailymed" in text.lower() and not any(item.get("source") == "dailymed" for item in evidence):
        risk += 0.30
    return _clip01(risk)


def _text_specificity(text: str) -> float:
    lower = text.lower()
    cues = [
        "smiles",
        "tpsa",
        "logp",
        "molecular weight",
        "h-bond",
        "hydrogen",
        "cyp",
        "herg",
        "ddI".lower(),
        "fingerprint",
        "rule",
        "descriptor",
        "functional group",
        "aromatic",
        "血脑屏障",
        "肝损伤",
        "相互作用",
        "结构",
    ]
    return _clip01(sum(1 for cue in cues if cue in lower) / 8.0)


def _text_structure_score(text: str) -> float:
    lower = text.lower()
    cues = [
        "functional group",
        "aromatic",
        "ring",
        "hbond",
        "hydrogen bond",
        "tpsa",
        "logp",
        "molwt",
        "descriptor",
        "substructure",
        "carboxy",
        "amine",
        "ester",
        "结构",
        "芳香",
        "氢键",
    ]
    return _clip01(sum(1 for cue in cues if cue in lower) / 6.0)


def _text_uncertainty(text: str) -> float:
    lower = text.lower()
    cues = [
        "uncertain",
        "uncertainty",
        "low confidence",
        "non-calibrated",
        "not calibrated",
        "not a calibrated probability",
        "fallback",
        "limited",
        "insufficient",
        "cautious",
        "不确定",
        "置信",
        "谨慎",
        "证据不足",
    ]
    return _clip01(sum(1 for cue in cues if cue in lower) / 4.0)


def _quality_score(
    *,
    supported_claim_ratio: float,
    evidence_citation_rate: float,
    explanation_specificity: float,
    uncertainty_awareness: float,
    structure_explanation_score: float,
    hallucination_risk: float,
) -> float:
    return _clip01(
        0.25 * supported_claim_ratio
        + 0.20 * evidence_citation_rate
        + 0.20 * explanation_specificity
        + 0.15 * uncertainty_awareness
        + 0.15 * structure_explanation_score
        + 0.05 * (1.0 - hallucination_risk)
    )


def _merge_judge_scores(heuristic: Mapping[str, Any], llm_result: Mapping[str, Any]) -> Dict[str, Any]:
    merged = dict(heuristic)
    for key in QUALITY_SCORE_KEYS:
        if key in llm_result:
            merged[key] = _clip01(llm_result[key])
    merged["judge_provider"] = llm_result.get("judge_provider", "llm")
    merged["judge_notes"] = llm_result.get("judge_notes") or heuristic.get("judge_notes")
    merged["heuristic_report_quality_score"] = heuristic.get("report_quality_score")
    return merged


def _parse_json_object(response: str) -> Dict[str, Any]:
    text = (response or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        text = text[start : end + 1]
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("LLM judge response must be a JSON object.")
    return payload


def _clip01(value: Any) -> float:
    try:
        number = float(value)
    except Exception:
        return 0.0
    if number < 0.0:
        return 0.0
    if number > 1.0:
        return 1.0
    return number
