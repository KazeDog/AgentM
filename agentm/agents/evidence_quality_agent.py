from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
import re
from typing import Any, Dict, List, Mapping, Optional, Tuple


SOURCE_BASE_RELIABILITY = {
    "local_ddi_hint": 0.95,
    "dailymed": 0.86,
    "pubmed": 0.78,
    "pubchem": 0.58,
    "rag_literature": 0.52,
    "llm_rule_feature": 0.65,
    "rdkit_descriptor": 0.58,
    "feature_similarity": 0.45,
    "agent_confidence_review": 0.35,
    "external_evidence_error": 0.0,
}

ALWAYS_KEEP_SOURCES = {
    "agent_confidence_review",
    "feature_similarity",
    "llm_rule_feature",
    "local_ddi_hint",
    "rdkit_descriptor",
}

SUPPORT_TERMS = {
    "risk",
    "increase",
    "increased",
    "warning",
    "contraindicated",
    "contraindication",
    "monitor",
    "toxicity",
    "adverse",
    "bleeding",
    "qt",
    "inhibit",
    "inhibition",
    "poor",
    "reduce",
    "reduced",
    "alert",
    "permeability",
    "penetration",
    "supports",
}

HIGH_VALUE_SOURCE_TYPES = {
    "direct_pair_evidence",
    "mechanism_evidence",
    "single_drug_label_evidence",
}

CONTRADICTION_TERMS = {
    "no interaction",
    "no significant",
    "not significant",
    "no effect",
    "did not increase",
    "not associated",
    "safe",
    "well tolerated",
    "without increasing",
}

DDI_MECHANISM_TERMS = {
    "cyp",
    "cytochrome",
    "transporter",
    "p-glycoprotein",
    "pgp",
    "bleeding",
    "qt",
    "anticoagulant",
    "antiplatelet",
    "inr",
    "warfarin",
    "exposure",
    "auc",
    "pharmacokinetic",
    "pharmacodynamic",
}

ADMET_ENDPOINT_TERMS = {
    "bbbp": {"bbb", "blood brain barrier", "brain", "cns", "permeability", "penetration"},
    "bbb_martins": {"bbb", "blood brain barrier", "brain", "cns", "permeability", "penetration"},
    "dili": {"dili", "liver injury", "hepatotoxicity", "hepatic", "liver"},
    "herg": {"herg", "qt", "kcnh2", "cardiotoxicity"},
    "ames": {"ames", "mutagenicity", "mutagenic"},
    "clintox": {"clinical toxicity", "toxicity", "adverse"},
}


@dataclass
class EvidenceQualityResult:
    annotated_evidence: List[Dict[str, Any]]
    summary: Dict[str, Any]
    dropped_evidence: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class EvidenceQualityAgent:
    """Rule-based evidence quality layer: relevance, reranking, direction, and reliability calibration."""

    def __init__(self, *, max_evidence: int = 32, min_keep_relevance: float = 0.05):
        self.max_evidence = int(max_evidence)
        self.min_keep_relevance = float(min_keep_relevance)

    def assess(
        self,
        evidence: List[Mapping[str, Any]],
        prediction: Mapping[str, Any],
        *,
        task_type: str,
        dataset_name: str,
        request: str = "",
        molecules: Optional[List[Mapping[str, Any]]] = None,
    ) -> EvidenceQualityResult:
        context = _QualityContext(
            prediction=prediction,
            task_type=task_type,
            dataset_name=dataset_name,
            request=request,
            molecules=molecules or [],
        )
        scored = []
        dropped = []
        for idx, item in enumerate(evidence, start=1):
            annotated = self._annotate_item(dict(item), context=context, original_rank=idx)
            source = str(annotated.get("source") or "unknown")
            if (
                annotated["evidence_quality"]["relevance_score"] < self.min_keep_relevance
                and source not in ALWAYS_KEEP_SOURCES
                and source != "external_evidence_error"
            ):
                dropped.append(annotated)
            else:
                scored.append(annotated)
        scored.sort(
            key=lambda item: (
                item["evidence_quality"]["rerank_score"],
                item["evidence_quality"]["source_reliability_calibrated"],
                item["evidence_quality"]["relevance_score"],
            ),
            reverse=True,
        )
        for rank, item in enumerate(scored, start=1):
            item["evidence_quality"]["quality_rank"] = rank
        kept = scored[: self.max_evidence]
        dropped.extend(scored[self.max_evidence :])
        return EvidenceQualityResult(
            annotated_evidence=kept,
            dropped_evidence=dropped,
            summary=self._summary(kept, dropped),
        )

    def _annotate_item(self, item: Dict[str, Any], *, context: "_QualityContext", original_rank: int) -> Dict[str, Any]:
        text = _evidence_text(item)
        lower = text.lower()
        source = str(item.get("source") or "unknown")
        value = item.get("value") if isinstance(item.get("value"), Mapping) else {}
        source_path = str(value.get("source_path") or item.get("source_path") or "")
        section = _section_from_text_or_path(lower, source_path)
        matched_entities = _matched_entities(lower, context)
        matched_endpoint = _matched_endpoint_terms(lower, context.dataset_name, context.task_type)
        matched_mechanisms = sorted(term for term in DDI_MECHANISM_TERMS if term in lower)
        contradiction_hits = sorted(term for term in CONTRADICTION_TERMS if term in lower)
        support_hits = sorted(term for term in SUPPORT_TERMS if term in lower)
        evidence_type = _evidence_type(
            source=source,
            text_lower=lower,
            source_path=source_path,
            matched_entities=matched_entities,
            matched_endpoint=matched_endpoint,
            matched_mechanisms=matched_mechanisms,
            context=context,
        )
        relevance = self._relevance_score(
            source=source,
            evidence_type=evidence_type,
            matched_entities=matched_entities,
            matched_endpoint=matched_endpoint,
            matched_mechanisms=matched_mechanisms,
            support_hits=support_hits,
            contradiction_hits=contradiction_hits,
            context=context,
            value=value,
            source_path=source_path,
        )
        reliability = self._reliability(source=source, section=section, value=value, source_path=source_path, text_lower=lower)
        direction, weight, rationale = self._direction(
            source=source,
            lower=lower,
            support_hits=support_hits,
            contradiction_hits=contradiction_hits,
            matched_entities=matched_entities,
            matched_endpoint=matched_endpoint,
            matched_mechanisms=matched_mechanisms,
            relevance=relevance,
            reliability=reliability,
            context=context,
            item=item,
        )
        pair_directness = _pair_directness_score(evidence_type, matched_entities, context)
        mechanism_match = _clip01(0.35 * bool(matched_mechanisms) + 0.35 * bool(matched_endpoint) + 0.30 * bool(support_hits or contradiction_hits))
        endpoint_relevance = _clip01(0.70 * bool(matched_endpoint) + 0.30 * (context.task_type == "ddi_prediction" and bool(matched_mechanisms)))
        citation_completeness = _citation_completeness(source, item, value=value, source_path=source_path, text_lower=lower)
        recency_score = _recency_score(value, lower)
        quality_score = _clip01(
            0.26 * relevance
            + 0.22 * reliability
            + 0.18 * pair_directness
            + 0.16 * mechanism_match
            + 0.10 * citation_completeness
            + 0.08 * recency_score
        )
        rerank_score = round(0.42 * quality_score + 0.30 * relevance + 0.18 * reliability + 0.10 * weight, 6)
        quality = {
            "original_rank": original_rank,
            "quality_rank": None,
            "evidence_type": evidence_type,
            "quality_score": round(quality_score, 6),
            "relevance_score": round(relevance, 6),
            "rerank_score": rerank_score,
            "entity_match": round(_clip01(len(matched_entities) / 2.0), 6) if context.task_type == "ddi_prediction" else round(float(bool(matched_entities)), 6),
            "pair_directness": round(pair_directness, 6),
            "mechanism_match": round(mechanism_match, 6),
            "endpoint_relevance": round(endpoint_relevance, 6),
            "recency_score": round(recency_score, 6),
            "citation_completeness": round(citation_completeness, 6),
            "matched_entities": matched_entities,
            "matched_endpoint_terms": matched_endpoint,
            "matched_mechanism_terms": matched_mechanisms[:12],
            "support_terms": support_hits[:12],
            "contradiction_terms": contradiction_hits[:8],
            "source_section": section,
            "source_reliability_base": SOURCE_BASE_RELIABILITY.get(source, 0.40),
            "source_reliability_calibrated": round(reliability, 6),
            "direction_rule": direction,
            "support_level": _support_level(quality_score, direction),
            "quality_rationale": rationale,
        }
        item["evidence_quality"] = quality
        item["evidence_type"] = evidence_type
        item["evidence_quality_score"] = quality["quality_score"]
        item["support_level"] = quality["support_level"]
        item["source_reliability"] = reliability
        item["evidence_direction"] = direction
        item["evidence_weight"] = weight
        item["evidence_scoring_rationale"] = rationale
        return item

    def _relevance_score(
        self,
        *,
        source: str,
        evidence_type: str,
        matched_entities: List[str],
        matched_endpoint: List[str],
        matched_mechanisms: List[str],
        support_hits: List[str],
        contradiction_hits: List[str],
        context: "_QualityContext",
        value: Mapping[str, Any],
        source_path: str,
    ) -> float:
        score = 0.0
        if matched_entities:
            if context.task_type == "ddi_prediction":
                score += 0.20 * min(2, len(matched_entities))
                if len(matched_entities) >= 2:
                    score += 0.20
            else:
                score += 0.30
        if matched_endpoint:
            score += 0.25
        if matched_mechanisms and context.task_type == "ddi_prediction":
            score += 0.25
        if support_hits or contradiction_hits:
            score += 0.15
        if evidence_type in HIGH_VALUE_SOURCE_TYPES:
            score += 0.12
        has_content_match = bool(matched_entities or matched_endpoint or matched_mechanisms)
        if has_content_match and (value.get("pmids") or value.get("pmid") or "pubmed" in source_path.lower()):
            score += 0.05
        if has_content_match and (value.get("cid") or source == "pubchem"):
            score += 0.03
        if context.task_type == "ddi_prediction" and source in {"pubmed", "pubchem", "rag_literature"} and not matched_entities:
            score = min(score, 0.20)
        if source == "external_evidence_error":
            return 0.0
        return _clip01(score)

    def _reliability(self, *, source: str, section: str, value: Mapping[str, Any], source_path: str, text_lower: str) -> float:
        reliability = SOURCE_BASE_RELIABILITY.get(source, 0.40)
        if source == "dailymed":
            if section == "drug_interactions":
                reliability = 0.95
            elif section in {"warnings", "contraindications"}:
                reliability = 0.90
            elif section == "clinical_pharmacology":
                reliability = 0.84
            elif value.get("labels") and not value.get("sections"):
                reliability = 0.72
        elif source == "pubmed":
            if "randomized" in text_lower or "clinical trial" in text_lower:
                reliability = 0.88
            elif "review" in text_lower or "meta-analysis" in text_lower:
                reliability = 0.84
            elif value.get("pmids") or "pubmed_pair_literature" in source_path:
                reliability = 0.80
        elif source == "pubchem":
            if value.get("cid") or "cid" in text_lower:
                reliability = 0.68
            else:
                reliability = 0.50
        elif source == "rag_literature":
            if "dailymed" in source_path.lower():
                reliability = 0.86
            elif "pubmed" in source_path.lower():
                reliability = 0.78
            elif "pubchem" in source_path.lower():
                reliability = 0.58
        return _clip01(reliability)

    def _direction(
        self,
        *,
        source: str,
        lower: str,
        support_hits: List[str],
        contradiction_hits: List[str],
        matched_entities: List[str],
        matched_endpoint: List[str],
        matched_mechanisms: List[str],
        relevance: float,
        reliability: float,
        context: "_QualityContext",
        item: Mapping[str, Any],
    ) -> Tuple[str, float, str]:
        pred_positive = _prediction_is_positive(context.prediction)
        if source == "external_evidence_error":
            return "neutral", 0.0, "External retrieval failed; retained as workflow context only."
        if relevance < 0.12 and source not in {"local_ddi_hint", "llm_rule_feature", "feature_similarity", "rdkit_descriptor"}:
            return "neutral", 0.0, "Evidence had low entity/endpoint/mechanism relevance after quality screening."
        if contradiction_hits:
            direction = "contradiction" if pred_positive else "support"
            return direction, _clip01(0.35 + 0.35 * relevance), "Evidence contains no-effect/no-significant-interaction language."
        if source == "local_ddi_hint":
            direction = "support" if pred_positive else "contradiction"
            return direction, _severity_strength(item.get("severity")), "Curated local DDI hint matched this pair."
        if source == "feature_similarity" and context.task_type == "ddi_prediction":
            similarity = _extract_similarity(item.get("value"))
            if similarity >= 0.45:
                return ("support" if pred_positive else "contradiction"), min(0.45, similarity), "High pair similarity is weak directional DDI evidence."
            if similarity <= 0.15:
                return ("support" if not pred_positive else "contradiction"), 0.20, "Low pair similarity weakly supports a no-interaction prediction."
        if source == "rdkit_descriptor":
            risk_like = bool(support_hits)
            if risk_like:
                risk_prediction = pred_positive if context.task_type == "admet_prediction" else not pred_positive if context.dataset_name.lower() == "bbbp" else pred_positive
                return ("support" if risk_prediction else "contradiction"), 0.35, "Descriptor evidence indicates a risk or unfavorable property cue."
        if source == "llm_rule_feature":
            risk_like = any(token in lower for token in ["risk", "alert", "tox", "high", "above", "inhibit"])
            if risk_like:
                risk_prediction = pred_positive if context.task_type in {"admet_prediction", "ddi_prediction"} else not pred_positive if context.dataset_name.lower() == "bbbp" else pred_positive
                return ("support" if risk_prediction else "contradiction"), 0.35, "Matched LLM4SD-style rule is interpreted as a directional task cue."
            return "support", 0.20, "Matched LLM4SD-style rule is a weak positive task cue."
        if source in {"dailymed", "pubmed", "pubchem", "rag_literature"}:
            if context.task_type == "ddi_prediction" and source in {"pubmed", "pubchem", "rag_literature"} and not matched_entities:
                return "neutral", 0.0, "External DDI evidence lacks drug-name/entity matches in the evidence body; retained as context only."
            if support_hits or matched_endpoint or matched_mechanisms:
                direction = "support" if pred_positive else "contradiction"
                base = 0.65 if source == "dailymed" else 0.55 if source == "pubmed" else 0.45 if source == "rag_literature" else 0.25
                strength = _clip01(base * (0.55 + 0.45 * relevance) * (0.65 + 0.35 * reliability))
                return direction, strength, "Evidence has relevant entity/endpoint/mechanism terms aligned with the predicted risk/property direction."
        return "neutral", 0.0, "Evidence was retained as context but was not directional after quality screening."

    def _summary(self, kept: List[Dict[str, Any]], dropped: List[Dict[str, Any]]) -> Dict[str, Any]:
        directions = Counter(str(item.get("evidence_direction", "neutral")) for item in kept)
        sources = Counter(str(item.get("source", "unknown")) for item in kept)
        evidence_types = Counter(str((item.get("evidence_quality") or {}).get("evidence_type", item.get("evidence_type", "unknown"))) for item in kept)
        avg_relevance = _mean([item["evidence_quality"]["relevance_score"] for item in kept])
        avg_reliability = _mean([item["evidence_quality"]["source_reliability_calibrated"] for item in kept])
        avg_rerank = _mean([item["evidence_quality"]["rerank_score"] for item in kept])
        avg_quality = _mean([item["evidence_quality"]["quality_score"] for item in kept])
        conflicts = directions.get("support", 0) > 0 and directions.get("contradiction", 0) > 0
        direct_pair_count = evidence_types.get("direct_pair_evidence", 0)
        mechanism_count = evidence_types.get("mechanism_evidence", 0)
        return {
            "n_input_evidence": len(kept) + len(dropped),
            "n_kept_evidence": len(kept),
            "n_dropped_evidence": len(dropped),
            "direction_counts": dict(directions),
            "source_counts": dict(sources),
            "evidence_type_counts": dict(evidence_types),
            "average_relevance_score": avg_relevance,
            "average_source_reliability": avg_reliability,
            "average_rerank_score": avg_rerank,
            "average_quality_score": avg_quality,
            "direct_pair_evidence_count": direct_pair_count,
            "mechanism_evidence_count": mechanism_count,
            "direct_pair_evidence_rate": direct_pair_count / len(kept) if kept else 0.0,
            "mechanism_evidence_rate": mechanism_count / len(kept) if kept else 0.0,
            "has_conflicting_evidence": conflicts,
            "top_evidence": [
                {
                    "source": item.get("source"),
                    "evidence_type": (item.get("evidence_quality") or {}).get("evidence_type"),
                    "claim": str(item.get("claim") or item.get("mechanism") or "")[:240],
                    "direction": item.get("evidence_direction"),
                    "quality_score": item["evidence_quality"]["quality_score"],
                    "relevance_score": item["evidence_quality"]["relevance_score"],
                    "rerank_score": item["evidence_quality"]["rerank_score"],
                    "quality_rationale": item["evidence_quality"]["quality_rationale"],
                }
                for item in kept[:8]
            ],
        }


class _QualityContext:
    def __init__(
        self,
        *,
        prediction: Mapping[str, Any],
        task_type: str,
        dataset_name: str,
        request: str,
        molecules: List[Mapping[str, Any]],
    ):
        self.prediction = prediction
        self.task_type = task_type
        self.dataset_name = dataset_name or ""
        self.request = request or ""
        self.molecules = molecules
        self.entity_terms = _entity_terms(request, molecules)


def _evidence_text(item: Mapping[str, Any]) -> str:
    parts = []
    for key in ["claim", "mechanism", "interaction_type", "section", "title"]:
        if item.get(key):
            parts.append(str(item.get(key)))
    value = item.get("value")
    if isinstance(value, Mapping):
        for key in ["snippet", "labels", "sections", "summary", "abstract"]:
            if value.get(key):
                parts.append(str(value.get(key)))
    elif value:
        parts.append(str(value))
    return _strip_retrieval_metadata(re.sub(r"\s+", " ", " ".join(parts)).strip())


def _strip_retrieval_metadata(text: str) -> str:
    cleaned = text
    cleaned = re.sub(r"(?i)#\s*pubmed pair literature for .*?(?=(?:##\s*pmid|\s[-•]\s*url\s*:|$))", " ", cleaned)
    cleaned = re.sub(r"(?i)#\s*dailymed .*? for .*?(?=(?:##\s*setid|\s[-•]\s*url\s*:|$))", " ", cleaned)
    cleaned = re.sub(
        r"(?i)(?:^|\s)[-•]?\s*query\s*:\s*.*?(?=(?:\s[-•]\s*(?:url|journal|year|retrieved_at|license_note|source|task|pmid)\s*:)|$)",
        " ",
        cleaned,
    )
    cleaned = re.sub(r"(?i)(?:^|\s)[-•]?\s*url\s*:\s*\S+", " ", cleaned)
    cleaned = re.sub(r"(?i)(?:^|\s)[-•]?\s*retrieved_at\s*:\s*\S+", " ", cleaned)
    cleaned = re.sub(r"https?://\S+", " ", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def _entity_terms(request: str, molecules: List[Mapping[str, Any]]) -> List[str]:
    terms = []
    for molecule in molecules:
        for key in ["name", "original_input", "canonical_smiles"]:
            text = str(molecule.get(key) or "").strip()
            if text and len(text) <= 120:
                terms.append(text.lower())
    for token in re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", request or ""):
        if token.lower() not in {"the", "and", "with", "ddi", "risk"}:
            terms.append(token.lower())
    deduped = []
    seen = set()
    for term in terms:
        clean = term.strip().lower()
        if clean and clean not in seen:
            deduped.append(clean)
            seen.add(clean)
    return deduped[:12]


def _matched_entities(lower: str, context: _QualityContext) -> List[str]:
    hits = []
    for term in context.entity_terms:
        if len(term) < 3:
            continue
        if term in lower:
            hits.append(term)
    return hits[:8]


def _matched_endpoint_terms(lower: str, dataset_name: str, task_type: str) -> List[str]:
    dataset = (dataset_name or "").lower()
    terms = set()
    for key, values in ADMET_ENDPOINT_TERMS.items():
        if dataset == key or dataset.replace("-", "_") == key:
            terms.update(values)
    if task_type == "ddi_prediction":
        terms.update({"interaction", "interactions", "drug interaction", "adverse", "toxicity", "risk"})
    return sorted(term for term in terms if term in lower)


def _evidence_type(
    *,
    source: str,
    text_lower: str,
    source_path: str,
    matched_entities: List[str],
    matched_endpoint: List[str],
    matched_mechanisms: List[str],
    context: _QualityContext,
) -> str:
    path_lower = source_path.lower()
    if source == "external_evidence_error":
        return "retrieval_failure"
    if context.task_type == "ddi_prediction":
        if len(matched_entities) >= 2 and (
            source in {"pubmed", "dailymed", "rag_literature", "local_ddi_hint"}
            or "pubmed_pair_literature" in path_lower
            or "dailymed_interactions" in path_lower
        ):
            return "direct_pair_evidence"
        if matched_mechanisms or "clinical pharmacology" in text_lower or "drug interactions" in text_lower:
            return "mechanism_evidence"
        if source == "dailymed":
            return "single_drug_label_evidence"
        if source == "pubchem":
            return "compound_property_evidence"
        return "background_literature" if source in {"pubmed", "rag_literature"} else "model_or_rule_evidence"
    if source == "dailymed":
        return "single_drug_label_evidence"
    if source == "pubchem":
        return "compound_property_evidence"
    if matched_endpoint or matched_mechanisms:
        return "mechanism_evidence"
    if source in {"pubmed", "rag_literature"}:
        return "background_literature"
    return "model_or_rule_evidence"


def _pair_directness_score(evidence_type: str, matched_entities: List[str], context: _QualityContext) -> float:
    if context.task_type != "ddi_prediction":
        return 0.0
    if evidence_type == "direct_pair_evidence":
        return 1.0
    if len(matched_entities) >= 2:
        return 0.8
    if len(matched_entities) == 1:
        return 0.35
    return 0.0


def _citation_completeness(source: str, item: Mapping[str, Any], *, value: Mapping[str, Any], source_path: str, text_lower: str) -> float:
    if source == "external_evidence_error":
        return 0.0
    score = 0.0
    if source_path:
        score += 0.35
    url = item.get("url") or value.get("url")
    if url:
        score += 0.25
    if source == "pubmed" and (item.get("pmid") or value.get("pmid") or value.get("pmids") or "pmid" in text_lower):
        score += 0.40
    elif source == "pubchem" and (item.get("cid") or value.get("cid") or "cid" in text_lower):
        score += 0.40
    elif source == "dailymed" and (item.get("setid") or value.get("setid") or "setid" in text_lower):
        score += 0.40
    elif source in {"rag_literature", "local_ddi_hint", "llm_rule_feature", "rdkit_descriptor", "feature_similarity"}:
        score += 0.25 if source_path else 0.10
    return _clip01(score)


def _recency_score(value: Mapping[str, Any], text_lower: str) -> float:
    candidates = []
    for key in ["year", "published_date", "retrieved_at"]:
        if value.get(key):
            candidates.append(str(value.get(key)))
    candidates.append(text_lower)
    years = []
    for text in candidates:
        for match in re.findall(r"\b(19[7-9]\d|20[0-3]\d)\b", str(text)):
            try:
                years.append(int(match))
            except Exception:
                continue
    if not years:
        return 0.45
    newest = max(years)
    if newest >= 2020:
        return 1.0
    if newest >= 2015:
        return 0.85
    if newest >= 2010:
        return 0.70
    if newest >= 2000:
        return 0.50
    return 0.30


def _support_level(score: float, direction: str) -> str:
    if direction == "neutral":
        return "context_only"
    if score >= 0.75:
        return "strong"
    if score >= 0.50:
        return "moderate"
    if score >= 0.25:
        return "weak"
    return "minimal"


def _section_from_text_or_path(lower: str, source_path: str) -> str:
    joined = f"{lower} {source_path.lower()}"
    if "drug interactions" in joined or "dailymed_interactions" in joined:
        return "drug_interactions"
    if "contraindications" in joined:
        return "contraindications"
    if "warnings" in joined or "precautions" in joined:
        return "warnings"
    if "clinical pharmacology" in joined:
        return "clinical_pharmacology"
    if "adverse reactions" in joined:
        return "adverse_reactions"
    if "pubmed" in joined:
        return "literature"
    if "pubchem" in joined:
        return "compound_identity"
    return "unspecified"


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


def _mean(values: List[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
