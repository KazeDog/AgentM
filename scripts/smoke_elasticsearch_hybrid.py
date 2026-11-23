#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agentm.runtime.retrieval import HybridRetriever
from agentm.runtime.retrieval.evidence_schema import (
    endpoint_from_dataset,
    endpoint_terms,
    unique_strings,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test AgentM Elasticsearch hybrid retrieval.")
    parser.add_argument("--es-url", default=os.getenv("AGENTM_ES_URL"))
    parser.add_argument("--index-name", default="agentm_evidence")
    parser.add_argument("--query", default="DILI aspirin drug induced liver injury evidence")
    parser.add_argument("--task-type", default="admet_prediction")
    parser.add_argument("--dataset-name", default="DILI")
    parser.add_argument("--endpoint", default=None)
    parser.add_argument("--drug-name", action="append", default=[])
    parser.add_argument("--cid", action="append", default=[])
    parser.add_argument("--chembl-id", action="append", default=[])
    parser.add_argument("--smiles", action="append", default=[])
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    endpoint = args.endpoint or endpoint_from_dataset(args.dataset_name)
    entity_terms = unique_strings(
        [*args.drug_name, *args.cid, *args.chembl_id, *args.smiles],
        lower=True,
    )
    metadata = {
        "task_type": args.task_type,
        "dataset_name": args.dataset_name,
        "endpoint": endpoint,
        "endpoint_terms": endpoint_terms(endpoint),
        "drug_names": unique_strings(args.drug_name, lower=True),
        "compound_names": unique_strings(args.drug_name, lower=True),
        "compound_cids": unique_strings(args.cid, lower=False),
        "chembl_ids": unique_strings(args.chembl_id, lower=True),
        "canonical_smiles": unique_strings(args.smiles, lower=False),
        "entity_terms": entity_terms,
    }
    retriever = HybridRetriever(es_url=args.es_url, index_name=args.index_name, top_k=args.top_k)
    result = retriever.retrieve(
        args.query,
        metadata=metadata,
        backend="es",
        top_k=args.top_k,
    )
    payload = result.to_dict()
    compact = {
        "status": "completed" if result.retrieved_context else "empty",
        "backend": result.backend,
        "provider": result.provider,
        "n_context": len(result.retrieved_context),
        "n_evidence": len(result.evidence_items),
        "warnings": result.warnings,
        "metadata": result.metadata,
        "query_metadata": metadata,
        "top_context": result.retrieved_context[:2],
    }
    print(json.dumps(compact if result.retrieved_context else payload, indent=2, ensure_ascii=False))
    return 0 if result.retrieved_context else 1


if __name__ == "__main__":
    raise SystemExit(main())
