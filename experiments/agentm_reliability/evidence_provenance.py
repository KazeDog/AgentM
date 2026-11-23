from __future__ import annotations

from collections import Counter
import json
import re
from typing import Any, Iterable, Mapping

from experiments.agentm_reliability.evidence_text import source_body_text, source_title_text


PROVENANCE_SCHEMA = "agentm_evidence_provenance"

EXTERNAL_LAYERS = {
    "external_literature",
    "authoritative_drug_label",
    "external_reference_database",
    "external_other",
}
QC_SOURCES = {"paper_qc_curated", "benchmark_qc_grounding", "paper_qc"}


def _as_mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "direct"}


def normalize_evidence_record(item: Mapping[str, Any]) -> dict[str, Any]:
    """Flatten the runtime evidence wrapper without discarding provenance."""

    top = dict(item)
    nested = _as_mapping(top.get("value"))
    record = {**nested, **top}
    nested_metadata = _as_mapping(nested.get("metadata"))
    top_metadata = _as_mapping(top.get("metadata"))
    metadata = {**nested_metadata, **top_metadata}
    record["metadata"] = metadata
    for key in (
        "task_type",
        "dataset_name",
        "evidence_type",
        "endpoint",
        "evidence_scope",
        "compound_specific",
        "endpoint_specific",
        "pair_specific",
        "reaction_specific",
    ):
        if record.get(key) is None and metadata.get(key) is not None:
            record[key] = metadata[key]
    record["source"] = str(top.get("source") or nested.get("source") or "unknown")
    record["source_path"] = str(top.get("source_path") or nested.get("source_path") or "")
    record["chunk_id"] = str(top.get("chunk_id") or nested.get("chunk_id") or "")
    return record


def classify_evidence_layer(item: Mapping[str, Any]) -> str:
    record = normalize_evidence_record(item)
    source = str(record.get("source") or "").strip().lower()
    path = str(record.get("source_path") or "").strip().lower()
    doc_id = str(record.get("doc_id") or "").strip().lower()
    evidence_type = str(record.get("evidence_type") or "").strip().lower()
    marker = " ".join((source, path, doc_id, evidence_type))

    if _bool(record.get("controlled_injection")) or source.startswith("controlled_") or path.startswith("controlled://"):
        return "controlled_injection"
    if source in QC_SOURCES or "paper_qc" in marker or evidence_type == "compound_endpoint_grounding":
        return "benchmark_qc_grounding"
    if any(token in marker for token in ("dailymed", "accessdata.fda.gov", "drug label", "drug_label", "ema.europa.eu")):
        return "authoritative_drug_label"
    if record.get("pmid") or any(token in marker for token in ("pubmed", "europepmc", "pmc.ncbi.nlm.nih.gov", "doi.org", "crossref")):
        return "external_literature"
    if any(token in marker for token in ("pubchem", "chembl", "drugbank", "uniprot")):
        return "external_reference_database"
    if any(token in marker for token in ("mit_reaction", "orderly", "ord_", "local_reaction_data", "reaction_template")):
        return "local_reaction_dataset"
    if path.startswith(("http://", "https://")):
        return "external_other"
    if source in {"rag_literature", "local_rag", "admet_curated", "hybrid_retriever"} or path:
        return "local_secondary_corpus"
    return "unknown"


def _text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple, set)):
        return " ".join(_text(item) for item in value)
    return ""


def _source_semantic_content(record: Mapping[str, Any]) -> str:
    """Exclude retrieval/routing metadata from scientific directness checks."""

    return " ".join(
        part for part in (source_title_text(record), source_body_text(record)) if part
    ).lower()


def _terms(record: Mapping[str, Any], keys: Iterable[str]) -> list[str]:
    values: list[str] = []
    for key in keys:
        raw = record.get(key)
        if isinstance(raw, str):
            values.extend(part.strip().lower() for part in re.split(r"[,;|]", raw) if part.strip())
        elif isinstance(raw, (list, tuple, set)):
            values.extend(str(part).strip().lower() for part in raw if str(part).strip())
    seen: set[str] = set()
    return [value for value in values if not (value in seen or seen.add(value))]


def _content_mentions(content: str, terms: Iterable[str], *, minimum: int = 1) -> bool:
    normalized = [term for term in terms if len(term) >= 2]
    if not normalized:
        return False
    def present(term: str) -> bool:
        if len(term) <= 4 and re.fullmatch(r"[a-z0-9]+", term):
            return bool(re.search(rf"\b{re.escape(term)}\b", content))
        return term in content

    return sum(present(term) for term in normalized) >= minimum


def evidence_is_direct(item: Mapping[str, Any]) -> bool:
    """Conservatively identify evidence that directly addresses the target.

    Entity/endpoint flags alone are insufficient because the QC cards carry
    those flags solely to test retrieval alignment.  Directness additionally
    requires source content that mentions the target scope.
    """

    record = normalize_evidence_record(item)
    layer = classify_evidence_layer(record)
    if layer in {"benchmark_qc_grounding", "controlled_injection", "unknown"}:
        return False
    direct_declaration: bool | None = None
    for key in ("direct_evidence", "is_direct_evidence", "direct_support"):
        if record.get(key) is not None:
            direct_declaration = _bool(record.get(key))
            break
    if direct_declaration is False:
        return False

    scope = str(record.get("evidence_scope") or "").lower()
    content = _source_semantic_content(record)
    if layer == "local_reaction_dataset":
        return bool(
            direct_declaration is True
            or _bool(record.get("reaction_specific"))
            or scope in {"reaction", "reaction_template", "reaction_condition", "reaction_specific"}
        )

    if layer not in EXTERNAL_LAYERS:
        return False
    entity_terms = _terms(
        record,
        ("compound_names", "drug_names", "entity_terms", "compound_synonyms"),
    )
    endpoint_terms = _terms(record, ("endpoint_terms", "endpoint"))
    endpoint_mentioned = _content_mentions(content, endpoint_terms)
    if scope == "ddi_pair" or _bool(record.get("pair_specific")):
        entity_mentioned = _content_mentions(content, entity_terms, minimum=min(2, len(entity_terms)))
        interaction_mentioned = any(term in content for term in ("interaction", "drug-drug", "ddi", "coadministr"))
        return bool(entity_mentioned and interaction_mentioned)
    entity_mentioned = _content_mentions(content, entity_terms)
    specific = direct_declaration is True or (
        _bool(record.get("compound_specific")) and _bool(record.get("endpoint_specific"))
    )
    return bool(specific and entity_mentioned and endpoint_mentioned)


def evidence_is_relevant(item: Mapping[str, Any]) -> bool:
    record = normalize_evidence_record(item)
    layer = classify_evidence_layer(record)
    scope = str(record.get("evidence_scope") or "").lower()
    if layer == "controlled_injection" or scope == "background_only":
        return False
    if layer == "benchmark_qc_grounding":
        return bool(
            scope in {"compound_endpoint", "ddi_pair"}
            and (_bool(record.get("endpoint_specific")) or scope == "ddi_pair")
        )
    if evidence_is_direct(record):
        return True
    if layer == "local_reaction_dataset":
        return bool(scope.startswith("reaction") or _bool(record.get("reaction_specific")))
    return False


def _record_key(record: Mapping[str, Any]) -> str:
    normalized = normalize_evidence_record(record)
    for key in ("chunk_id", "citation_id", "source_path", "doc_id"):
        value = str(normalized.get(key) or "")
        if value:
            return f"{normalized.get('source')}::{key}::{value}"
    return json.dumps(normalized, ensure_ascii=False, sort_keys=True, default=str)


def _deduplicate(records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in records:
        normalized = normalize_evidence_record(item)
        key = _record_key(normalized)
        if key in seen:
            continue
        seen.add(key)
        output.append(normalized)
    return output


def retained_evidence_records(context: Mapping[str, Any]) -> list[dict[str, Any]]:
    return _deduplicate(
        item for item in context.get("evidence", []) or [] if isinstance(item, Mapping)
    )


def retrieved_evidence_records(context: Mapping[str, Any]) -> list[dict[str, Any]]:
    records: list[Mapping[str, Any]] = []
    for trace in context.get("retrieval_trace", []) or []:
        if not isinstance(trace, Mapping):
            continue
        for key in ("retrieved_context", "evidence_items"):
            records.extend(item for item in trace.get(key, []) or [] if isinstance(item, Mapping))
    return _deduplicate(records)


def audit_evidence_provenance(context: Mapping[str, Any]) -> dict[str, Any]:
    retained = retained_evidence_records(context)
    retrieved = retrieved_evidence_records(context)
    audited = retained if retained else retrieved
    layers = [classify_evidence_layer(item) for item in audited]
    sources = [str(normalize_evidence_record(item).get("source") or "unknown") for item in audited]
    direct = [evidence_is_direct(item) for item in audited]
    relevant = [evidence_is_relevant(item) for item in audited]
    layer_counts = Counter(layers)
    source_counts = Counter(sources)
    total = len(audited)
    external_count = sum(layer in EXTERNAL_LAYERS for layer in layers)
    qc_count = layer_counts.get("benchmark_qc_grounding", 0)
    return {
        "evidence_provenance_schema": PROVENANCE_SCHEMA,
        "evidence_audit_basis": "retained_evidence" if retained else "retrieval_trace_only",
        "retained_evidence_count": len(retained),
        "retrieved_evidence_count": len(retrieved),
        "audited_evidence_count": total,
        "external_evidence_count": external_count,
        "direct_evidence_count": sum(direct),
        "qc_grounding_count": qc_count,
        "local_reaction_evidence_count": layer_counts.get("local_reaction_dataset", 0),
        "external_evidence_coverage": external_count > 0,
        "direct_evidence_coverage": any(direct),
        "qc_only": bool(total > 0 and qc_count == total),
        "evidence_relevance": (sum(relevant) / total) if total else None,
        "evidence_source_diversity": len(source_counts),
        "evidence_layer_counts_json": json.dumps(dict(sorted(layer_counts.items())), ensure_ascii=False, sort_keys=True),
        "evidence_source_counts_json": json.dumps(dict(sorted(source_counts.items())), ensure_ascii=False, sort_keys=True),
    }
