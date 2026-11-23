#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


DEFAULT_CASES = [
    "piroxicam:dili",
    "aspirin:dili",
    "caffeine:herg",
    "caffeine:bbbp",
    "aspirin:ddi",
    "warfarin:ddi",
    "simvastatin:ddi",
    "ketoconazole:ddi",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit AgentM structured evidence coverage by drug/entity and endpoint.")
    parser.add_argument("--jsonl", required=True, help="JSONL produced by scripts/index_evidence_es.py --output-jsonl.")
    parser.add_argument("--case", action="append", default=[], help="Coverage case in the form drug:endpoint, e.g. aspirin:dili.")
    parser.add_argument("--output-json", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    docs = list(_read_jsonl(Path(args.jsonl)))
    cases = [_parse_case(item) for item in (args.case or DEFAULT_CASES)]
    rows = [_audit_case(docs, drug, endpoint) for drug, endpoint in cases]
    summary = {
        "jsonl": str(Path(args.jsonl)),
        "n_documents": len(docs),
        "source_counts": dict(Counter(str(doc.get("source") or "unknown") for doc in docs)),
        "endpoint_counts": dict(Counter(str(doc.get("endpoint") or "unknown") for doc in docs)),
        "scope_counts": dict(Counter(str(doc.get("evidence_scope") or "unknown") for doc in docs)),
        "cases": rows,
    }
    payload = json.dumps(summary, indent=2, ensure_ascii=False)
    if args.output_json:
        Path(args.output_json).write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0


def _read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            yield payload


def _parse_case(text: str) -> Tuple[str, str]:
    if ":" not in text:
        raise SystemExit(f"Invalid --case {text!r}; expected drug:endpoint")
    drug, endpoint = text.split(":", 1)
    return drug.strip().lower(), endpoint.strip().lower()


def _audit_case(docs: List[Dict[str, Any]], drug: str, endpoint: str) -> Dict[str, Any]:
    endpoint_docs = [doc for doc in docs if str(doc.get("endpoint") or "").lower() == endpoint]
    entity_docs = [doc for doc in endpoint_docs if drug in _entity_terms(doc)]
    entity_endpoint_docs = [
        doc
        for doc in entity_docs
        if bool(doc.get("compound_specific")) and bool(doc.get("endpoint_specific"))
    ]
    entity_background_docs = [doc for doc in entity_docs if doc not in entity_endpoint_docs]
    endpoint_background = [
        doc
        for doc in endpoint_docs
        if bool(doc.get("endpoint_specific")) and not bool(doc.get("compound_specific"))
    ]
    source_counts = Counter(str(doc.get("source") or "unknown") for doc in entity_docs)
    scope_counts = Counter(str(doc.get("evidence_scope") or "unknown") for doc in entity_docs)
    status = "covered" if entity_endpoint_docs else ("entity_background_only" if entity_docs else ("endpoint_background_only" if endpoint_background else "missing"))
    return {
        "drug": drug,
        "endpoint": endpoint,
        "status": status,
        "entity_documents": len(entity_docs),
        "entity_endpoint_documents": len(entity_endpoint_docs),
        "entity_background_documents": len(entity_background_docs),
        "endpoint_background_documents": len(endpoint_background),
        "source_counts": dict(source_counts),
        "scope_counts": dict(scope_counts),
        "examples": [
            {
                "chunk_id": doc.get("chunk_id"),
                "source": doc.get("source"),
                "scope": doc.get("evidence_scope"),
                "title": doc.get("title"),
            }
            for doc in [*entity_endpoint_docs, *entity_background_docs][:3]
        ],
    }


def _entity_terms(doc: Dict[str, Any]) -> set[str]:
    values: List[Any] = []
    for key in [
        "entity_terms",
        "drug_names",
        "compound_names",
        "compound_synonyms",
        "compound_cids",
        "chembl_ids",
        "canonical_smiles",
        "pair_drug_names",
        "pair_drug_ids",
    ]:
        raw = doc.get(key)
        if isinstance(raw, str):
            values.append(raw)
        else:
            values.extend(raw or [])
    return {str(value).strip().lower() for value in values if str(value).strip()}


if __name__ == "__main__":
    raise SystemExit(main())
