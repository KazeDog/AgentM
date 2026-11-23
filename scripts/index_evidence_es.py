#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agentm.runtime.compat import RESOURCE_ROOT
from agentm.runtime.resources import default_resource_path
from agentm.runtime.retrieval.evidence_schema import iter_structured_evidence_documents

from agentm.defaults import DEFAULT_RAG_EMBEDDING_MODEL_PATH


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Index AgentM local evidence corpus into Elasticsearch when available.")
    parser.add_argument("--es-url", default=os.getenv("AGENTM_ES_URL"))
    parser.add_argument("--index-name", default="agentm_evidence")
    parser.add_argument("--corpus-dir", action="append", default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output-jsonl", default="", help="Write structured evidence documents to JSONL before optional ES indexing.")
    parser.add_argument("--recreate-index", action="store_true", help="Delete and recreate the target index before indexing.")
    parser.add_argument("--with-embeddings", action="store_true", help="Store dense embeddings for ES hybrid BM25+dense retrieval.")
    parser.add_argument("--embedding-model-name", default=str(default_resource_path(DEFAULT_RAG_EMBEDDING_MODEL_PATH)))
    parser.add_argument("--embedding-backend", default="auto", choices=["auto", "sentence_transformer", "transformers"])
    parser.add_argument("--embedding-device", default="auto")
    parser.add_argument("--embedding-batch-size", type=int, default=16)
    parser.add_argument("--embedding-local-files-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--bulk-error-json", default="", help="Optional path for Elasticsearch bulk indexing error details.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    default_corpus_dirs = [
        RESOURCE_ROOT / "data" / "rag_corpus" / "tdc_real",
        RESOURCE_ROOT / "data" / "rag_corpus" / "ddi_real",
        ROOT / "data" / "evidence_backfill",
        ROOT / "data" / "evidence_backfill_full",
    ]
    corpus_dirs = [Path(path) for path in (args.corpus_dir or default_corpus_dirs)]
    docs = list(iter_structured_evidence_documents(corpus_dirs, limit=args.limit))
    if not docs:
        print(json.dumps({"status": "skipped", "reason": "no_documents_found", "corpus_dirs": [str(path) for path in corpus_dirs]}, indent=2))
        return 0
    if args.output_jsonl:
        output_path = Path(args.output_jsonl)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("\n".join(json.dumps(doc, ensure_ascii=False) for doc in docs) + "\n", encoding="utf-8")
    if not args.es_url:
        print(
            json.dumps(
                {
                    "status": "structured_index_written" if args.output_jsonl else "skipped",
                    "reason": "AGENTM_ES_URL_not_configured",
                    "n_documents": len(docs),
                    "output_jsonl": args.output_jsonl or None,
                },
                indent=2,
            )
        )
        return 0
    try:
        from elasticsearch import Elasticsearch, helpers
    except Exception as exc:
        print(json.dumps({"status": "skipped", "reason": f"elasticsearch_package_unavailable:{exc}", "n_documents": len(docs)}, indent=2))
        return 0
    embedding_info: Dict[str, Any] = {"enabled": False}
    if args.with_embeddings:
        try:
            embedding_info = _attach_embeddings(
                docs,
                model_name=args.embedding_model_name,
                backend=args.embedding_backend,
                device=args.embedding_device,
                batch_size=args.embedding_batch_size,
                local_files_only=args.embedding_local_files_only,
            )
        except Exception as exc:
            print(json.dumps({"status": "failed", "reason": f"dense_embedding_failed:{exc}"}, indent=2))
            return 1
    client = Elasticsearch(args.es_url)
    if args.recreate_index and client.indices.exists(index=args.index_name):
        client.indices.delete(index=args.index_name)
    if not client.indices.exists(index=args.index_name):
        client.indices.create(index=args.index_name, mappings=_mapping(embedding_dim=embedding_info.get("embedding_dim")))
    elif args.with_embeddings and not _index_has_dense_vector(client, args.index_name):
        print(
            json.dumps(
                {
                    "status": "failed",
                    "reason": "target_index_missing_dense_vector_mapping",
                    "hint": "rerun with --recreate-index --with-embeddings",
                    "index_name": args.index_name,
                },
                indent=2,
            )
        )
        return 1
    actions = [
        {
            "_index": args.index_name,
            "_id": doc["chunk_id"],
            "_source": doc,
        }
        for doc in docs
    ]
    ok, errors = helpers.bulk(client, actions, stats_only=False, raise_on_error=False)
    if errors:
        error_path = Path(args.bulk_error_json or (Path(args.output_jsonl).parent / "bulk_errors.json" if args.output_jsonl else ROOT / "outputs" / "evidence_bulk_errors.json"))
        error_path.parent.mkdir(parents=True, exist_ok=True)
        error_path.write_text(json.dumps(errors, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "completed",
                "index_name": args.index_name,
                "indexed": ok,
                "errors": len(errors),
                "n_documents": len(docs),
                "output_jsonl": args.output_jsonl or None,
                "bulk_error_json": str(error_path) if errors else None,
                "embedding": embedding_info,
            },
            indent=2,
        )
    )
    return 0 if not errors else 1


def _mapping(*, embedding_dim: Optional[int] = None) -> Dict[str, Any]:
    def keyword(ignore_above: int = 512) -> Dict[str, Any]:
        return {"type": "keyword", "ignore_above": ignore_above}

    properties: Dict[str, Any] = {
        "evidence_id": keyword(1024),
        "chunk_id": keyword(1024),
        "doc_id": keyword(1024),
        "source_path": keyword(2048),
        "url": keyword(2048),
        "source": keyword(),
        "task_type": keyword(),
        "dataset_name": keyword(),
        "endpoint": keyword(),
        "endpoint_aliases": keyword(),
        "endpoint_terms": keyword(),
        "evidence_type": keyword(),
        "evidence_scope": keyword(),
        "compound_specific": {"type": "boolean"},
        "endpoint_specific": {"type": "boolean"},
        "pmid": keyword(),
        "year": keyword(),
        "drug_names": keyword(),
        "compound_names": keyword(),
        "compound_synonyms": keyword(1024),
        "compound_cids": keyword(),
        "chembl_ids": keyword(),
        "canonical_smiles": keyword(2048),
        "entity_terms": keyword(),
        "text_entity_terms": keyword(1024),
        "ddi_pair_key": keyword(1024),
        "pair_drug_ids": keyword(),
        "pair_drug_names": keyword(),
        "pair_drug_smiles": keyword(2048),
        "targets": keyword(1024),
        "title": {"type": "text", "fields": {"keyword": {"type": "keyword", "ignore_above": 512}}},
        "text": {"type": "text"},
    }
    if embedding_dim:
        properties["embedding"] = {"type": "dense_vector", "dims": int(embedding_dim), "index": True, "similarity": "cosine"}
        properties["embedding_model"] = keyword(2048)
        properties["embedding_backend"] = keyword()
    return {"properties": properties}


def _index_has_dense_vector(client: Any, index_name: str) -> bool:
    mapping = client.indices.get_mapping(index=index_name)
    payload = mapping.get(index_name, mapping)
    properties = payload.get("mappings", {}).get("properties", {})
    return properties.get("embedding", {}).get("type") == "dense_vector"


def _attach_embeddings(
    docs: List[Dict[str, Any]],
    *,
    model_name: str,
    backend: str,
    device: str,
    batch_size: int,
    local_files_only: bool,
) -> Dict[str, Any]:
    from agentm.tools.rag import DenseTextEmbedder

    model_path = str(default_resource_path(model_name))
    embedder = DenseTextEmbedder(
        model_path,
        backend=backend,
        device=device,
        local_files_only=local_files_only,
        batch_size=batch_size,
    )
    texts = [_embedding_text(doc) for doc in docs]
    embeddings = embedder.encode(texts)
    if embeddings.ndim != 2 or embeddings.shape[0] != len(docs):
        raise ValueError("Dense embedder returned an invalid embedding matrix.")
    for doc, vector in zip(docs, embeddings):
        doc["embedding"] = [float(value) for value in vector.tolist()]
        doc["embedding_model"] = embedder.model_name_or_path
        doc["embedding_backend"] = embedder.backend
    return {
        "enabled": True,
        "embedding_model": embedder.model_name_or_path,
        "embedding_backend": embedder.backend,
        "embedding_dim": int(embeddings.shape[1]),
        "n_embedded": len(docs),
    }


def _embedding_text(doc: Dict[str, Any]) -> str:
    title = str(doc.get("title") or "")
    text = str(doc.get("text") or "")
    metadata = " ".join(
        str(doc.get(key) or "")
        for key in [
            "task_type",
            "dataset_name",
            "endpoint",
            "endpoint_terms",
            "drug_names",
            "compound_names",
            "compound_synonyms",
            "compound_cids",
            "chembl_ids",
            "pair_drug_names",
            "pair_drug_smiles",
            "targets",
        ]
    )
    return f"{title}\n{metadata}\n{text}".strip()


if __name__ == "__main__":
    raise SystemExit(main())
