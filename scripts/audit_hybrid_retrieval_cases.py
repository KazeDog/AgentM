#!/usr/bin/env python
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import sys
from typing import Any, Dict, Iterable, List, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agentm.runtime.retrieval import HybridRetriever
from agentm.runtime.retrieval.evidence_schema import endpoint_terms, iter_structured_evidence_documents, unique_strings
from scripts.audit_evidence_relevance_cases import (
    EvidenceAuditCase,
    audit_case,
    doc_matches_endpoint,
    doc_matches_entities,
    doc_text_matches_endpoint,
    doc_text_matches_entities,
    load_documents,
    parse_case,
    render_markdown as render_relevance_markdown,
)


DEFAULT_CASES = [
    "aspirin|2244|CHEMBL25:dili",
    "caffeine|2519|CHEMBL113:herg",
    "aspirin|2244+warfarin|54678486:ddi",
    "simvastatin|54454+ketoconazole|456201:ddi",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit AgentM hybrid retrieval outputs for endpoint/entity filtering, "
            "metadata rerank, and top-k evidence grounding."
        )
    )
    parser.add_argument("--backend", default="local", choices=["local", "es"], help="local replays filters/rerank over JSONL/corpus; es calls HybridRetriever.")
    parser.add_argument("--jsonl", action="append", default=[], help="Structured evidence JSONL for local replay.")
    parser.add_argument("--corpus-dir", action="append", default=[], help="Raw/backfilled evidence corpus for local replay.")
    parser.add_argument("--case", action="append", default=[], help="Case syntax: aspirin|2244:dili or aspirin+warfarin:ddi.")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--min-docs", type=int, default=1)
    parser.add_argument("--es-url", default="")
    parser.add_argument("--index-name", default="agentm_evidence")
    parser.add_argument("--require-dense", action="store_true")
    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-md", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cases = [parse_case(text) for text in (args.case or DEFAULT_CASES)]
    if args.backend == "local":
        docs = load_documents(jsonl_paths=[Path(path) for path in args.jsonl], corpus_dirs=[Path(path) for path in args.corpus_dir])
        rows = [audit_local_retrieval_case(docs, case, top_k=args.top_k, min_docs=args.min_docs) for case in cases]
        backend_info = {"backend": "local_jsonl_bm25_metadata_rerank", "n_documents": len(docs)}
    else:
        rows = [
            audit_es_retrieval_case(
                case,
                top_k=args.top_k,
                min_docs=args.min_docs,
                es_url=args.es_url or None,
                index_name=args.index_name,
                require_dense=args.require_dense,
            )
            for case in cases
        ]
        backend_info = {"backend": "elasticsearch_hybrid", "index_name": args.index_name, "require_dense": bool(args.require_dense)}
    summary = {
        "status": "pass" if all(row["status"] == "pass" for row in rows) else "needs_attention",
        "backend": backend_info,
        "cases": rows,
    }
    payload = json.dumps(summary, indent=2, ensure_ascii=False)
    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(payload + "\n", encoding="utf-8")
    if args.output_md:
        Path(args.output_md).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_md).write_text(render_markdown(summary), encoding="utf-8")
    print(payload)
    return 0 if summary["status"] == "pass" else 2


def audit_local_retrieval_case(
    docs: Sequence[Mapping[str, Any]],
    case: EvidenceAuditCase,
    *,
    top_k: int,
    min_docs: int,
) -> Dict[str, Any]:
    retriever = HybridRetriever(use_dense=False)
    query = query_for_case(case)
    metadata = metadata_for_case(case)
    hits: List[Dict[str, Any]] = []
    filtered_out = 0
    for doc in docs:
        if not local_doc_passes_filters(doc, case):
            filtered_out += 1
            continue
        lexical = lexical_score(doc, query)
        bonus = retriever._metadata_rerank_bonus(doc, metadata)
        hits.append(
            {
                **dict(doc),
                "retrieval_score": lexical + bonus,
                "lexical_score": lexical,
                "metadata_rerank_bonus": bonus,
            }
        )
    hits.sort(key=lambda item: (float(item.get("retrieval_score") or 0.0), float(item.get("metadata_rerank_bonus") or 0.0)), reverse=True)
    top_docs = hits[: max(0, int(top_k))]
    relevance = audit_case(top_docs, case, min_docs=min_docs, strict_returned=True)
    return {
        **relevance,
        "query": query,
        "retrieval_backend": "local_jsonl_bm25_metadata_rerank",
        "n_candidate_documents": len(docs),
        "n_filter_passed": len(hits),
        "n_filtered_out": filtered_out,
        "top_k": int(top_k),
        "top_hits": [retrieval_hit_summary(doc) for doc in top_docs],
    }


def audit_es_retrieval_case(
    case: EvidenceAuditCase,
    *,
    top_k: int,
    min_docs: int,
    es_url: str | None,
    index_name: str,
    require_dense: bool,
) -> Dict[str, Any]:
    retriever = HybridRetriever(es_url=es_url, index_name=index_name, top_k=top_k)
    result = retriever.retrieve(query_for_case(case), metadata=metadata_for_case(case), backend="es", top_k=top_k)
    docs = [context_to_doc(item) for item in result.retrieved_context]
    relevance = audit_case(docs, case, min_docs=min_docs, strict_returned=True)
    dense_status = str(result.metadata.get("dense_status") or "")
    dense_ok = not require_dense or dense_status == "completed"
    status = relevance["status"] if dense_ok else "dense_retrieval_not_completed"
    return {
        **relevance,
        "status": status,
        "query": result.query,
        "retrieval_backend": result.backend,
        "warnings": result.warnings,
        "retrieval_metadata": result.metadata,
        "top_k": int(top_k),
        "top_hits": [retrieval_hit_summary(doc) for doc in docs],
    }


def local_doc_passes_filters(doc: Mapping[str, Any], case: EvidenceAuditCase) -> bool:
    return (
        doc_matches_endpoint(doc, case.endpoint)
        and bool(doc.get("endpoint_specific"))
        and doc_text_matches_endpoint(doc, case.endpoint)
        and doc_matches_entities(doc, case.entity_groups)
        and doc_text_matches_entities(doc, case.entity_groups)
        and bool(doc.get("compound_specific"))
    )


def query_for_case(case: EvidenceAuditCase) -> str:
    primary_terms = [group[0] for group in case.entity_groups if group]
    endpoint_aliases = endpoint_terms(case.endpoint)[:3]
    return " ".join(unique_strings([*primary_terms, *endpoint_aliases], lower=True))


def metadata_for_case(case: EvidenceAuditCase) -> Dict[str, Any]:
    flat_terms = [term for group in case.entity_groups for term in group]
    return {
        "task_type": "ddi_prediction" if case.endpoint == "ddi" else "admet_prediction",
        "endpoint": case.endpoint,
        "endpoint_terms": endpoint_terms(case.endpoint),
        "entity_terms": flat_terms,
        "drug_names": [group[0] for group in case.entity_groups if group],
        "entity_groups": case.entity_groups,
    }


def lexical_score(doc: Mapping[str, Any], query: str) -> float:
    haystack = " ".join(str(doc.get(key) or "") for key in ["title", "text", "claim", "snippet"]).lower()
    if not haystack:
        return 0.0
    score = 0.0
    for term in query_terms(query):
        if not term:
            continue
        count = haystack.count(term)
        if count:
            score += 1.0 + min(count, 5) * 0.2
    return score


def query_terms(query: str) -> List[str]:
    return [term for term in re.split(r"[^a-z0-9]+", query.lower()) if len(term) >= 3]


def context_to_doc(context: Mapping[str, Any]) -> Dict[str, Any]:
    doc = dict(context)
    metadata = context.get("metadata") if isinstance(context.get("metadata"), Mapping) else {}
    for key, value in metadata.items():
        doc.setdefault(key, value)
    doc.setdefault("text", context.get("snippet") or "")
    return doc


def retrieval_hit_summary(doc: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "chunk_id": doc.get("chunk_id"),
        "source": doc.get("source"),
        "source_path": doc.get("source_path") or doc.get("url"),
        "pmid": doc.get("pmid"),
        "endpoint": doc.get("endpoint"),
        "evidence_scope": doc.get("evidence_scope"),
        "title": doc.get("title"),
        "score": doc.get("retrieval_score") or doc.get("rerank_score") or doc.get("hybrid_score") or doc.get("score"),
        "metadata_rerank_bonus": doc.get("metadata_rerank_bonus"),
    }


def render_markdown(summary: Mapping[str, Any]) -> str:
    relevance_view = {
        "status": summary.get("status"),
        "n_documents": summary.get("backend", {}).get("n_documents", ""),
        "cases": summary.get("cases", []),
    }
    lines = ["# AgentM Hybrid Retrieval Audit", "", f"- Backend: `{summary.get('backend', {}).get('backend')}`", ""]
    lines.append(render_relevance_markdown(relevance_view))
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
