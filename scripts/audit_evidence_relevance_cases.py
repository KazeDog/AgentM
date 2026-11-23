#!/usr/bin/env python
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agentm.runtime.retrieval.evidence_schema import compact_term, endpoint_terms, iter_structured_evidence_documents, unique_strings


DEFAULT_CASES = [
    "aspirin|2244|CHEMBL25:dili",
    "caffeine|2519|CHEMBL113:herg",
    "aspirin|2244+warfarin|54678486:ddi",
    "simvastatin|54454+ketoconazole|456201:ddi",
]


@dataclass(frozen=True)
class EvidenceAuditCase:
    label: str
    endpoint: str
    entity_groups: List[List[str]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit representative AgentM evidence cases for compound/entity alignment, "
            "endpoint relevance, and citation-groundable compound+endpoint evidence."
        )
    )
    parser.add_argument("--jsonl", action="append", default=[], help="Structured evidence JSONL, e.g. index_evidence_es.py --output-jsonl.")
    parser.add_argument("--corpus-dir", action="append", default=[], help="Raw/backfilled evidence corpus directory to normalize before auditing.")
    parser.add_argument("--case", action="append", default=[], help="Case syntax: aspirin|2244:dili or aspirin+warfarin:ddi.")
    parser.add_argument("--min-docs", type=int, default=1, help="Minimum compound+endpoint grounded docs required per case.")
    parser.add_argument(
        "--require-text-match",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require title/text/snippet to mention the audited entity group(s) and endpoint terms, not only metadata.",
    )
    parser.add_argument(
        "--strict-returned",
        action="store_true",
        help="Treat loaded docs as a retrieved result set and fail if any returned doc is not compound+endpoint grounded for the case.",
    )
    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-md", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    docs = load_documents(jsonl_paths=[Path(path) for path in args.jsonl], corpus_dirs=[Path(path) for path in args.corpus_dir])
    cases = [parse_case(text) for text in (args.case or DEFAULT_CASES)]
    rows = [
        audit_case(docs, case, min_docs=args.min_docs, strict_returned=args.strict_returned, require_text_match=args.require_text_match)
        for case in cases
    ]
    summary = {
        "status": "pass" if all(row["status"] == "pass" for row in rows) else "needs_attention",
        "n_documents": len(docs),
        "source_counts": dict(Counter(str(doc.get("source") or "unknown") for doc in docs)),
        "endpoint_counts": dict(Counter(str(doc.get("endpoint") or "unknown") for doc in docs)),
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


def load_documents(*, jsonl_paths: Sequence[Path], corpus_dirs: Sequence[Path]) -> List[Dict[str, Any]]:
    docs: List[Dict[str, Any]] = []
    for path in jsonl_paths:
        docs.extend(read_jsonl(path))
    if corpus_dirs:
        docs.extend(iter_structured_evidence_documents(corpus_dirs))
    return docs


def audit_case(
    docs: Sequence[Mapping[str, Any]],
    case: EvidenceAuditCase,
    *,
    min_docs: int = 1,
    strict_returned: bool = False,
    require_text_match: bool = True,
) -> Dict[str, Any]:
    endpoint_matched: List[Mapping[str, Any]] = []
    entity_matched: List[Mapping[str, Any]] = []
    grounded: List[Mapping[str, Any]] = []
    returned_not_grounded: List[Mapping[str, Any]] = []
    for doc in docs:
        endpoint_match = doc_matches_endpoint(doc, case.endpoint) and (not require_text_match or doc_text_matches_endpoint(doc, case.endpoint))
        entity_match = doc_matches_entities(doc, case.entity_groups) and (not require_text_match or doc_text_matches_entities(doc, case.entity_groups))
        compound_specific = bool(doc.get("compound_specific")) or entity_match
        endpoint_specific = bool(doc.get("endpoint_specific")) and endpoint_match
        if endpoint_match and endpoint_specific:
            endpoint_matched.append(doc)
        if entity_match and compound_specific:
            entity_matched.append(doc)
        is_grounded = entity_match and endpoint_match and compound_specific and endpoint_specific
        if is_grounded:
            grounded.append(doc)
        elif strict_returned:
            returned_not_grounded.append(doc)
    status = case_status(
        n_grounded=len(grounded),
        n_entity=len(entity_matched),
        n_endpoint=len(endpoint_matched),
        min_docs=min_docs,
        n_returned_not_grounded=len(returned_not_grounded),
        strict_returned=strict_returned,
    )
    return {
        "label": case.label,
        "endpoint": case.endpoint,
        "entity_groups": case.entity_groups,
        "status": status,
        "min_docs": int(min_docs),
        "require_text_match": bool(require_text_match),
        "n_entity_related": len(entity_matched),
        "n_endpoint_related": len(endpoint_matched),
        "n_compound_endpoint_grounded": len(grounded),
        "n_returned_not_grounded": len(returned_not_grounded),
        "source_counts": dict(Counter(str(doc.get("source") or "unknown") for doc in grounded)),
        "evidence_scope_counts": dict(Counter(str(doc.get("evidence_scope") or "unknown") for doc in grounded)),
        "examples": [summarize_doc(doc) for doc in grounded[:5]],
        "non_grounded_examples": [summarize_doc(doc) for doc in returned_not_grounded[:5]],
    }


def case_status(
    *,
    n_grounded: int,
    n_entity: int,
    n_endpoint: int,
    min_docs: int,
    n_returned_not_grounded: int,
    strict_returned: bool,
) -> str:
    if n_grounded >= int(min_docs) and (not strict_returned or n_returned_not_grounded == 0):
        return "pass"
    if strict_returned and n_returned_not_grounded:
        return "returned_evidence_not_grounded"
    if n_entity and n_endpoint:
        return "entity_and_endpoint_separate_only"
    if n_entity:
        return "entity_background_only"
    if n_endpoint:
        return "endpoint_background_only"
    return "missing"


def parse_case(text: str) -> EvidenceAuditCase:
    if ":" not in text:
        raise SystemExit(f"Invalid --case {text!r}; expected entity:endpoint")
    entity_text, endpoint = text.split(":", 1)
    endpoint = compact_term(endpoint)
    groups: List[List[str]] = []
    for group_text in split_entity_groups(entity_text, endpoint=endpoint):
        terms = unique_strings([part.strip() for part in group_text.split("|")], lower=True)
        if terms:
            groups.append(terms)
    if not groups or not endpoint.strip():
        raise SystemExit(f"Invalid --case {text!r}; expected entity:endpoint")
    return EvidenceAuditCase(label=text, endpoint=endpoint, entity_groups=groups)


def split_entity_groups(entity_text: str, *, endpoint: str) -> List[str]:
    if compact_term(endpoint) != "ddi":
        return [entity_text]
    groups: List[str] = []
    current: List[str] = []
    bracket_depth = 0
    for char in entity_text:
        if char == "[":
            bracket_depth += 1
        elif char == "]" and bracket_depth:
            bracket_depth -= 1
        if char == "+" and bracket_depth == 0:
            groups.append("".join(current))
            current = []
            continue
        current.append(char)
    groups.append("".join(current))
    return groups


def doc_matches_endpoint(doc: Mapping[str, Any], endpoint: str) -> bool:
    expected = {compact_term(endpoint), *[compact_term(term) for term in endpoint_terms(endpoint)]}
    observed = {compact_term(doc.get("endpoint"))}
    observed.update(compact_term(term) for term in _list_values(doc.get("endpoint_terms")))
    observed.update(compact_term(term) for term in _list_values(doc.get("endpoint_aliases")))
    return bool(expected.intersection(observed))


def doc_matches_entities(doc: Mapping[str, Any], entity_groups: Sequence[Sequence[str]]) -> bool:
    terms = doc_entity_terms(doc)
    if not terms:
        return False
    for group in entity_groups:
        alternatives = normalized_match_terms(group)
        if not terms.intersection(alternatives):
            return False
    return True


def doc_text_matches_endpoint(doc: Mapping[str, Any], endpoint: str) -> bool:
    text_terms = normalized_match_terms([doc_text(doc)])
    if not text_terms:
        return False
    expected = normalized_match_terms(endpoint_terms(endpoint))
    expected.add(str(endpoint or "").lower())
    for term in expected:
        if len(term) < 3:
            continue
        if term in text_terms or compact_term(term) in text_terms:
            return True
        if term in doc_text(doc).lower():
            return True
    return False


def doc_text_matches_entities(doc: Mapping[str, Any], entity_groups: Sequence[Sequence[str]]) -> bool:
    text = doc_text(doc)
    if not text:
        return False
    text_terms = normalized_match_terms([text])
    for group in entity_groups:
        alternatives = normalized_match_terms(group)
        if not any(_term_matches_text(term, text=text, text_terms=text_terms) for term in alternatives):
            return False
    return True


def doc_entity_terms(doc: Mapping[str, Any]) -> set[str]:
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
        values.extend(_list_values(doc.get(key)))
    return normalized_match_terms(values)


def normalized_match_terms(values: Iterable[Any]) -> set[str]:
    out: set[str] = set()
    for value in values:
        text = str(value or "").strip().lower()
        if not text:
            continue
        out.add(text)
        compact = compact_term(text)
        if compact:
            out.add(compact)
    return out


def doc_text(doc: Mapping[str, Any]) -> str:
    return " ".join(str(doc.get(key) or "") for key in ["title", "text", "snippet", "abstract", "claim"])


def _term_matches_text(term: str, *, text: str, text_terms: set[str]) -> bool:
    raw = str(term or "").strip().lower()
    compact = compact_term(raw)
    if len(raw) < 3 and len(compact) < 3:
        return False
    if raw in text_terms or compact in text_terms:
        return True
    lowered = text.lower()
    if len(raw) >= 4 and raw in lowered:
        return True
    if len(compact) >= 6 and compact in compact_term(lowered):
        return True
    return False


def summarize_doc(doc: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "chunk_id": doc.get("chunk_id"),
        "source": doc.get("source"),
        "endpoint": doc.get("endpoint"),
        "evidence_scope": doc.get("evidence_scope"),
        "title": doc.get("title"),
        "pmid": doc.get("pmid"),
        "source_path": doc.get("source_path") or doc.get("url"),
    }


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    docs: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            docs.append(payload)
    return docs


def render_markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# AgentM Evidence Relevance Audit",
        "",
        f"- Status: `{summary.get('status')}`",
        f"- Documents: `{summary.get('n_documents')}`",
        "",
        "## Cases",
        "",
    ]
    for row in summary.get("cases", []):
        lines.extend(
            [
                f"### {row.get('label')}",
                "",
                f"- Status: `{row.get('status')}`",
                f"- Compound+endpoint grounded docs: `{row.get('n_compound_endpoint_grounded')}`",
                f"- Entity-related docs: `{row.get('n_entity_related')}`",
                f"- Endpoint-related docs: `{row.get('n_endpoint_related')}`",
                "",
            ]
        )
    return "\n".join(lines)


def _list_values(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Sequence):
        return list(value)
    return [value]


if __name__ == "__main__":
    raise SystemExit(main())
