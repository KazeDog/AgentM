#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agentm.agents.ddi_rag_evidence_agent import DDIRAGEvidenceAgent
from agentm.defaults import DEFAULT_DDI_RAG_CORPUS_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build pair-level DDI RAG corpus from PubChem, DailyMed, and PubMed.")
    parser.add_argument("--pair", action="append", default=None, help="Drug pair as 'drug_a,drug_b'. Can be repeated.")
    parser.add_argument("--pairs-csv", default=None, help="CSV with drug_a/drug_b or drug_a_name/drug_b_name columns.")
    parser.add_argument("--corpus-dir", default=DEFAULT_DDI_RAG_CORPUS_DIR)
    parser.add_argument("--timeout", type=float, default=12.0)
    parser.add_argument("--request-delay", type=float, default=0.1)
    parser.add_argument("--max-pubmed", type=int, default=8)
    parser.add_argument("--max-labels-per-drug", type=int, default=2)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pairs = _pairs_from_args(args)
    if not pairs:
        print(json.dumps({"status": "failed", "error": "No DDI pairs were provided."}, indent=2, ensure_ascii=False))
        return 1
    agent = DDIRAGEvidenceAgent(
        corpus_dir=args.corpus_dir,
        output_dir=Path(args.corpus_dir) / "_build_logs",
        build_on_demand=True,
        timeout=args.timeout,
        request_delay=args.request_delay,
        max_pubmed=args.max_pubmed,
        max_labels_per_drug=args.max_labels_per_drug,
        rag_backend="tfidf",
    )
    summaries = []
    for drug_a, drug_b in pairs:
        summaries.append(agent.build_pair_corpus(drug_a=drug_a, drug_b=drug_b))
    payload = {
        "status": "completed",
        "corpus_dir": args.corpus_dir,
        "n_pairs": len(summaries),
        "pairs": summaries,
        "manifest_path": str(Path(args.corpus_dir) / "manifest.json"),
        "failures_path": str(Path(args.corpus_dir) / "failures.jsonl"),
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def _pairs_from_args(args: argparse.Namespace) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for item in args.pair or []:
        left, sep, right = str(item).partition(",")
        if sep and left.strip() and right.strip():
            pairs.append((left.strip(), right.strip()))
    if args.pairs_csv:
        with Path(args.pairs_csv).open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                drug_a = _first(row, ["drug_a", "drug_a_name", "Drug1", "drug1", "a"])
                drug_b = _first(row, ["drug_b", "drug_b_name", "Drug2", "drug2", "b"])
                if drug_a and drug_b:
                    pairs.append((drug_a, drug_b))
    seen = set()
    deduped = []
    for pair in pairs:
        key = tuple(sorted([pair[0].lower(), pair[1].lower()]))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(pair)
    return deduped


def _first(row: dict, columns: list[str]) -> str | None:
    for column in columns:
        value = row.get(column)
        if value and str(value).strip():
            return str(value).strip()
    return None


if __name__ == "__main__":
    raise SystemExit(main())
