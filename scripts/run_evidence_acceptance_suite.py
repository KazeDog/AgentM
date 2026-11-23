#!/usr/bin/env python
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_evidence_relevance_cases import parse_case
from scripts.audit_hybrid_retrieval_cases import (
    audit_es_retrieval_case,
    audit_local_retrieval_case,
    load_documents,
)
from agentm.runtime.retrieval.evidence_schema import endpoint_from_dataset


DEFAULT_ACCEPTANCE_CASES = [
    "2,6-Dimethylaniline|6896|CHEMBL2228801:ames",
    "9-Cyclopentyl-6-((2-pyridinylmethyl)thio)-9H-purin-2-amine|246464|CHEMBL1411098:cyp1a2",
    "caffeine:bbbp",
    "prochlorperazine:bioavailability",
    "Pentobarbital|db00312+Alprazolam|db00404:ddi",
    "cyclosporin A|db00091+Loxoprofen|db09212:ddi",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a strict AgentM evidence retrieval acceptance suite over "
            "representative ADMET/DDI cases."
        )
    )
    parser.add_argument("--backend", choices=["es", "local"], default="es")
    parser.add_argument("--es-url", default=os.getenv("AGENTM_ES_URL", "http://127.0.0.1:9200"))
    parser.add_argument(
        "--index-name",
        default=os.getenv("AGENTM_ES_INDEX", "agentm_evidence_paper_qc_dense"),
    )
    parser.add_argument("--jsonl", action="append", default=[], help="Structured evidence JSONL for local acceptance mode.")
    parser.add_argument("--corpus-dir", action="append", default=[], help="Raw evidence corpus dirs for local acceptance mode.")
    parser.add_argument("--case", action="append", default=[], help="Case syntax: drug|cid:endpoint or drugA|idA+drugB|idB:ddi.")
    parser.add_argument("--case-file", action="append", default=[], help="Benchmark JSONL file converted to acceptance cases.")
    parser.add_argument("--entity-registry-json", default="", help="Optional compound registry used to expand case entity aliases.")
    parser.add_argument("--endpoint-ontology-json", default="", help="Optional endpoint ontology recorded in output metadata.")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--min-docs", type=int, default=1)
    parser.add_argument("--require-dense", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--strict-scope", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-citation", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-json", default="outputs/evidence_acceptance/acceptance_summary.json")
    parser.add_argument("--output-md", default="outputs/evidence_acceptance/acceptance_summary.md")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    case_texts = load_case_texts(args)
    cases = [parse_case(text) for text in case_texts]
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
                es_url=args.es_url,
                index_name=args.index_name,
                require_dense=args.require_dense,
            )
            for case in cases
        ]
        backend_info = {
            "backend": "elasticsearch_hybrid",
            "es_url": args.es_url,
            "index_name": args.index_name,
            "require_dense": bool(args.require_dense),
        }
    summary = build_acceptance_summary(rows, backend_info=backend_info, args=args)
    write_outputs(summary, json_path=Path(args.output_json), md_path=Path(args.output_md))
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if summary["status"] == "pass" else 2


def build_acceptance_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    backend_info: Mapping[str, Any],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    case_reports = [
        evaluate_case(
            row,
            require_dense=bool(args.require_dense),
            strict_scope=bool(args.strict_scope),
            require_citation=bool(args.require_citation),
        )
        for row in rows
    ]
    status_counts = Counter(report["status"] for report in case_reports)
    source_counts: Counter[str] = Counter()
    scope_counts: Counter[str] = Counter()
    dense_completed = 0
    for row in rows:
        source_counts.update(row.get("source_counts") or {})
        scope_counts.update(row.get("evidence_scope_counts") or {})
        if row.get("retrieval_metadata", {}).get("dense_status") == "completed":
            dense_completed += 1
    status = "pass" if all(report["status"] == "pass" for report in case_reports) else "needs_attention"
    return {
        "created_at": utc_now(),
        "status": status,
        "backend": dict(backend_info),
        "requirements": {
            "min_docs": int(args.min_docs),
            "top_k": int(args.top_k),
            "require_dense": bool(args.require_dense),
            "strict_scope": bool(args.strict_scope),
            "require_citation": bool(args.require_citation),
            "returned_not_grounded_allowed": 0,
            "case_files": list(getattr(args, "case_file", []) or []),
            "entity_registry_json": getattr(args, "entity_registry_json", "") or None,
            "endpoint_ontology_json": getattr(args, "endpoint_ontology_json", "") or None,
        },
        "metrics": {
            "n_cases": len(case_reports),
            "case_status_counts": dict(status_counts),
            "dense_completed_cases": dense_completed,
            "source_counts": dict(source_counts),
            "evidence_scope_counts": dict(scope_counts),
        },
        "cases": case_reports,
    }


def evaluate_case(row: Mapping[str, Any], *, require_dense: bool, strict_scope: bool, require_citation: bool) -> Dict[str, Any]:
    label = str(row.get("label") or "")
    endpoint = str(row.get("endpoint") or "")
    expected_scope = "ddi_pair" if endpoint == "ddi" else "compound_endpoint"
    top_hits = list(row.get("top_hits") or [])
    scopes = [str(hit.get("evidence_scope") or "") for hit in top_hits]
    failures: List[str] = []
    if row.get("status") != "pass":
        failures.append(f"retrieval_status:{row.get('status')}")
    if int(row.get("n_returned_not_grounded") or 0) != 0:
        failures.append(f"returned_not_grounded:{row.get('n_returned_not_grounded')}")
    if int(row.get("n_compound_endpoint_grounded") or 0) < 1:
        failures.append("no_compound_endpoint_or_pair_grounded_hit")
    if require_dense:
        metadata = row.get("retrieval_metadata") if isinstance(row.get("retrieval_metadata"), Mapping) else {}
        if metadata.get("dense_status") != "completed":
            failures.append(f"dense_status:{metadata.get('dense_status')}")
        if metadata.get("dense_status") != "completed" and "dense" not in str(row.get("retrieval_backend") or ""):
            failures.append(f"retrieval_backend_not_dense:{row.get('retrieval_backend')}")
    if strict_scope and any(scope and scope != expected_scope for scope in scopes):
        failures.append(f"unexpected_scope:{sorted(set(scopes))}")
    citation_failures = citation_failures_for_hits(top_hits)
    if require_citation and citation_failures:
        failures.extend(citation_failures)
    return {
        "label": label,
        "endpoint": endpoint,
        "status": "pass" if not failures else "needs_attention",
        "failures": failures,
        "expected_scope": expected_scope,
        "n_entity_related": row.get("n_entity_related"),
        "n_endpoint_related": row.get("n_endpoint_related"),
        "n_grounded": row.get("n_compound_endpoint_grounded"),
        "n_returned_not_grounded": row.get("n_returned_not_grounded"),
        "retrieval_backend": row.get("retrieval_backend"),
        "warnings": row.get("warnings") or [],
        "dense_status": row.get("retrieval_metadata", {}).get("dense_status")
        if isinstance(row.get("retrieval_metadata"), Mapping)
        else None,
        "source_counts": row.get("source_counts") or {},
        "evidence_scope_counts": row.get("evidence_scope_counts") or {},
        "citation_grounded": not citation_failures,
        "citation_failures": citation_failures,
        "top_hits": top_hits,
    }


def citation_failures_for_hits(top_hits: Sequence[Mapping[str, Any]]) -> List[str]:
    failures: List[str] = []
    if not top_hits:
        return ["missing_citation_hits"]
    for idx, hit in enumerate(top_hits, start=1):
        chunk_id = hit.get("chunk_id")
        source = hit.get("source")
        source_path = hit.get("source_path")
        pmid = hit.get("pmid")
        if not chunk_id:
            failures.append(f"hit_{idx}_missing_chunk_id")
        if not source:
            failures.append(f"hit_{idx}_missing_source")
        if not source_path and not pmid:
            failures.append(f"hit_{idx}_missing_source_path_or_pmid")
    return failures


def write_outputs(summary: Mapping[str, Any], *, json_path: Path, md_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(summary), encoding="utf-8")


def load_case_texts(args: argparse.Namespace) -> List[str]:
    texts: List[str] = []
    registry = read_json(Path(args.entity_registry_json)) if getattr(args, "entity_registry_json", "") else {}
    for path_text in args.case_file or []:
        texts.extend(case_text_from_benchmark_record(record, registry=registry) for record in read_jsonl(Path(path_text)))
    texts.extend(args.case or [])
    return [text for text in texts if text] or list(DEFAULT_ACCEPTANCE_CASES)


def read_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if isinstance(payload, Mapping):
            rows.append(dict(payload))
    return rows


def case_text_from_benchmark_record(record: Mapping[str, Any], *, registry: Mapping[str, Any] | None = None) -> str:
    endpoint = str(record.get("expected_endpoint") or endpoint_from_dataset(record.get("dataset_name"), task_type=record.get("task_type")) or "")
    endpoint = endpoint.strip().lower()
    task_type = str(record.get("task_type") or "")
    if endpoint == "ddi" or task_type == "ddi_prediction":
        left, right = ddi_drugs_from_request(str(record.get("request") or record.get("name") or ""))
        left_group = entity_group_text(left, registry=registry)
        right_group = entity_group_text(right, registry=registry)
        return f"{left_group}+{right_group}:ddi" if left_group and right_group else ""
    drug = drug_name_from_benchmark_record(record, endpoint=endpoint)
    group = entity_group_text(drug, registry=registry)
    return f"{group}:{endpoint}" if group and endpoint else ""


def entity_group_text(name: str, *, registry: Mapping[str, Any] | None = None) -> str:
    normalized = str(name or "").strip().lower()
    if not normalized:
        return ""
    entry = (registry or {}).get(normalized) if isinstance(registry, Mapping) else None
    values: List[Any] = [normalized]
    if isinstance(entry, Mapping):
        values.extend(entry.get("synonyms") or [])
        values.extend(entry.get("canonical_smiles") or [])
        values.extend(entry.get("pubchem_cids") or [])
        values.extend(entry.get("chembl_ids") or [])
    return "|".join(unique_for_case(values))


def unique_for_case(values: Sequence[Any]) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
    return out[:20]


def ddi_drugs_from_request(request: str) -> tuple[str, str]:
    parts = [part.strip() for part in request.replace(";", ",").replace("+", ",").split(",") if part.strip()]
    if len(parts) >= 2:
        return parts[0], parts[1]
    return "", ""


def drug_name_from_benchmark_record(record: Mapping[str, Any], *, endpoint: str) -> str:
    name = str(record.get("name") or "").strip()
    dataset = str(record.get("dataset_name") or "").strip()
    candidates = [
        endpoint,
        endpoint_from_dataset(dataset),
        dataset.lower(),
        dataset.lower().replace("_veith", ""),
        dataset.lower().replace("_martins", ""),
        dataset.lower().replace("_wang", ""),
        dataset.lower().replace("_hou", ""),
        dataset.lower().replace("_ma", ""),
        dataset.lower().replace("_broccatelli", ""),
        dataset.lower().replace("_astrazeneca", ""),
    ]
    drug = name
    for suffix in sorted({normalize_suffix(item) for item in candidates if item}, key=len, reverse=True):
        marker = f"_{suffix}"
        if drug.lower().endswith(marker):
            drug = drug[: -len(marker)]
            break
    return drug.replace("_", " ").strip()


def normalize_suffix(value: str) -> str:
    return str(value or "").strip().lower().replace("-", "_")


def render_markdown(summary: Mapping[str, Any]) -> str:
    metrics = summary.get("metrics", {})
    backend = summary.get("backend", {})
    lines = [
        "# AgentM Evidence Acceptance Suite",
        "",
        f"- Status: `{summary.get('status')}`",
        f"- Backend: `{backend.get('backend')}`",
        f"- Index: `{backend.get('index_name', '')}`",
        f"- Cases: `{metrics.get('n_cases')}`",
        f"- Dense completed cases: `{metrics.get('dense_completed_cases')}`",
        "",
        "## Case Results",
        "",
    ]
    for case in summary.get("cases", []):
        lines.append(
            f"- `{case.get('label')}`: `{case.get('status')}` "
            f"scope=`{case.get('expected_scope')}` grounded=`{case.get('n_grounded')}` "
            f"citation=`{case.get('citation_grounded')}`"
        )
        failures = case.get("failures") or []
        if failures:
            lines.append(f"  - failures: `{', '.join(str(item) for item in failures)}`")
    lines.extend(
        [
            "",
            "## Aggregate Counts",
            "",
            f"- Source counts: `{metrics.get('source_counts')}`",
            f"- Evidence scope counts: `{metrics.get('evidence_scope_counts')}`",
            "",
        ]
    )
    return "\n".join(lines)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
