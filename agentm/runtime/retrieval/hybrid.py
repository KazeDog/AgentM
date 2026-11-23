from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional

from agentm.runtime.compat import ensure_import_paths
from agentm.runtime.resources import default_resource_path
from agentm.runtime.retrieval.evidence_schema import compact_term, endpoint_terms, unique_strings

ensure_import_paths()

from agentm.defaults import DEFAULT_RAG_EMBEDDING_MODEL_PATH


DEFAULT_ES_URL = "http://127.0.0.1:9200"
DEFAULT_ES_INDEX = "agentm_evidence_paper_qc_dense"


@dataclass
class HybridRetrievalResult:
    provider: str
    backend: str
    query: str
    retrieved_context: List[Dict[str, Any]]
    evidence_items: List[Dict[str, Any]]
    warnings: List[str]
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "provider": self.provider,
            "backend": self.backend,
            "query": self.query,
            "retrieved_context": self.retrieved_context,
            "evidence_items": self.evidence_items,
            "warnings": self.warnings,
            "metadata": self.metadata,
        }


class HybridRetriever:
    """Optional Elasticsearch hybrid retriever with explicit local fallback signal."""

    def __init__(
        self,
        *,
        es_url: Optional[str] = None,
        index_name: Optional[str] = None,
        top_k: int = 5,
        use_dense: Optional[bool] = None,
        embedding_model_name: Optional[str] = None,
        embedding_backend: str = "auto",
        embedding_device: str = "auto",
        embedding_local_files_only: bool = True,
        embedding_batch_size: int = 16,
    ):
        self.es_url = es_url or os.getenv("AGENTM_ES_URL") or DEFAULT_ES_URL
        self.index_name = index_name or os.getenv("AGENTM_ES_INDEX") or DEFAULT_ES_INDEX
        self.top_k = int(top_k)
        dense_env = os.getenv("AGENTM_ES_USE_DENSE", "1").strip().lower()
        self.use_dense = (dense_env not in {"0", "false", "no", "off"}) if use_dense is None else bool(use_dense)
        self.embedding_model_name = embedding_model_name or os.getenv("AGENTM_ES_EMBEDDING_MODEL") or str(default_resource_path(DEFAULT_RAG_EMBEDDING_MODEL_PATH))
        self.embedding_backend = os.getenv("AGENTM_ES_EMBEDDING_BACKEND", embedding_backend)
        self.embedding_device = os.getenv("AGENTM_ES_EMBEDDING_DEVICE", embedding_device)
        self.embedding_local_files_only = bool(embedding_local_files_only)
        self.embedding_batch_size = int(os.getenv("AGENTM_ES_EMBEDDING_BATCH_SIZE", str(embedding_batch_size)))
        self.request_timeout_seconds = float(os.getenv("AGENTM_ES_REQUEST_TIMEOUT", "60"))
        self.max_retries = int(os.getenv("AGENTM_ES_MAX_RETRIES", "1"))
        self._embedder = None
        self._dense_mapping_checked: Optional[bool] = None

    def retrieve(
        self,
        query: str,
        *,
        metadata: Optional[Mapping[str, Any]] = None,
        backend: str = "auto",
        top_k: Optional[int] = None,
    ) -> HybridRetrievalResult:
        requested = backend or "auto"
        if requested in {"auto", "es"} and self.es_url:
            try:
                return self._retrieve_es(query, metadata=metadata or {}, top_k=top_k or self.top_k)
            except Exception as exc:
                warning = f"elasticsearch_retrieval_failed: {exc}"
                if requested == "es":
                    return self._empty(query, backend="elasticsearch", warnings=[warning])
                return self._empty(query, backend="local_fallback_required", warnings=[f"{warning}; used_local_fallback"])
        if requested == "es":
            return self._empty(query, backend="elasticsearch", warnings=["elasticsearch_url_not_configured"])
        return self._empty(query, backend="local_fallback_required", warnings=["local_fallback_required"])

    def _retrieve_es(self, query: str, *, metadata: Mapping[str, Any], top_k: int) -> HybridRetrievalResult:
        from elasticsearch import Elasticsearch

        client = Elasticsearch(
            self.es_url,
            request_timeout=self.request_timeout_seconds,
            max_retries=self.max_retries,
            retry_on_timeout=True,
        )
        filters, lexical_boosts = self._filters_and_boosts(metadata)
        warnings: List[str] = []
        candidate_k = max(top_k * 10, 50)
        bm25_hits = self._search_bm25(client, query, filters=filters, lexical_boosts=lexical_boosts, top_k=candidate_k)
        dense_hits: List[Dict[str, Any]] = []
        dense_enabled = False
        dense_status = "disabled"
        if self.use_dense:
            dense_enabled = self._index_has_dense_vector(client)
            if dense_enabled:
                try:
                    dense_hits = self._search_dense(client, query, filters=filters, top_k=candidate_k)
                    dense_status = "completed"
                except Exception as exc:
                    dense_status = "failed"
                    warnings.append(f"dense_knn_failed: {exc}")
            else:
                dense_status = "missing_dense_vector_mapping"
                warnings.append("dense_knn_unavailable: index_missing_dense_vector_mapping")
        merged_hits = self._merge_ranked_hits(bm25_hits, dense_hits, metadata=metadata, top_k=top_k)
        contexts: List[Dict[str, Any]] = []
        evidence: List[Dict[str, Any]] = []
        seen = set()
        for rank, hit in enumerate(merged_hits, start=1):
            source = hit.get("_source", {}) or {}
            chunk_id = source.get("chunk_id") or hit.get("_id")
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            snippet = " ".join(hit.get("highlight", {}).get("text", [])) or str(source.get("text", ""))[:800]
            chunk = {
                "rank": rank,
                "score": hit.get("_score"),
                "chunk_id": chunk_id,
                "doc_id": source.get("doc_id"),
                "source": source.get("source", "elasticsearch"),
                "source_path": source.get("url") or source.get("source_path"),
                "title": source.get("title"),
                "pmid": source.get("pmid"),
                "snippet": snippet,
                "text": source.get("text"),
                "source_body": source.get("source_body"),
                "metadata": {
                    key: source.get(key)
                    for key in ["task_type", "dataset_name", "drug_names", "targets", "year", "evidence_type"]
                    if key in source
                },
                "compound_names": source.get("compound_names", []),
                "compound_synonyms": source.get("compound_synonyms", []),
                "compound_cids": source.get("compound_cids", []),
                "chembl_ids": source.get("chembl_ids", []),
                "canonical_smiles": source.get("canonical_smiles", []),
                "endpoint": source.get("endpoint"),
                "endpoint_terms": source.get("endpoint_terms", []),
                "evidence_scope": source.get("evidence_scope"),
                "compound_specific": source.get("compound_specific"),
                "endpoint_specific": source.get("endpoint_specific"),
                "entity_terms": source.get("entity_terms", []),
                "text_entity_terms": source.get("text_entity_terms", []),
                "metadata_rerank_bonus": hit.get("_agentm_metadata_rerank_bonus"),
                "rerank_score": hit.get("_agentm_rerank_score"),
                "hybrid_score": hit.get("_agentm_hybrid_score"),
                "component_scores": hit.get("_agentm_component_scores", {}),
                "component_ranks": hit.get("_agentm_component_ranks", {}),
            }
            contexts.append(chunk)
            evidence.append(
                {
                    "source": chunk["source"],
                    "claim": snippet,
                    "value": chunk,
                    "chunk_id": chunk_id,
                    "source_path": chunk["source_path"],
                    "citation_id": f"cite_{len(evidence) + 1}",
                    "evidence_direction": "neutral",
                }
            )
        # Record the retrieval path that executed, not whether that path
        # happened to return a hit for this scientific object. An empty,
        # successfully executed dense query is a legitimate no-direct-evidence
        # outcome and must not be mislabeled as BM25-only.
        backend = (
            "elasticsearch_hybrid_bm25_dense"
            if dense_enabled and dense_status == "completed"
            else "elasticsearch_bm25"
        )
        return HybridRetrievalResult(
            provider="elasticsearch_hybrid",
            backend=backend,
            query=query,
            retrieved_context=contexts,
            evidence_items=evidence,
            warnings=warnings,
            metadata={
                "index_name": self.index_name,
                "top_k": top_k,
                "filters": filters,
                "lexical_boosts": lexical_boosts,
                "candidate_k": candidate_k,
                "bm25_hits": len(bm25_hits),
                "dense_hits": len(dense_hits),
                "dense_enabled": dense_enabled,
                "dense_status": dense_status,
                "embedding_model": self.embedding_model_name if dense_enabled else None,
                "request_timeout_seconds": self.request_timeout_seconds,
                "max_retries": self.max_retries,
            },
        )

    def _empty(self, query: str, *, backend: str, warnings: List[str]) -> HybridRetrievalResult:
        return HybridRetrievalResult(
            provider="hybrid_retriever",
            backend=backend,
            query=query,
            retrieved_context=[],
            evidence_items=[],
            warnings=warnings,
            metadata={"index_name": self.index_name, "es_url_configured": bool(self.es_url)},
        )

    def _filters_and_boosts(self, metadata: Mapping[str, Any]) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        filters: List[Dict[str, Any]] = []
        boosts: List[Dict[str, Any]] = []
        for field in ["task_type", "source", "evidence_type", "pmid", "year"]:
            if metadata.get(field):
                filters.append({"term": {field: metadata[field]}})
        if metadata.get("profile_type") == "general_drug_profile":
            filters.append(
                {
                    "bool": {
                        "must_not": [
                            {"term": {"endpoint": "ddi"}},
                            {"terms": {"evidence_scope": ["ddi_pair", "ddi_pair_background"]}},
                        ]
                    }
                }
            )
        if metadata.get("endpoint"):
            filters.append({"term": {"endpoint": compact_term(metadata["endpoint"])}})
            filters.append({"term": {"endpoint_specific": True}})
        for field in ["drug_names", "targets"]:
            values = metadata.get(field)
            if not values:
                continue
            normalized = values if isinstance(values, list) else [values]
            boosts.append({"terms": {field: normalized, "boost": 1.4}})
        entity_groups = _normalized_entity_groups(metadata)
        entity_values = _normalized_entity_values(metadata)
        if entity_groups:
            filters.append(
                {
                    "bool": {
                        "must": [
                            {
                                "bool": {
                                    "should": _entity_match_filters(group),
                                    "minimum_should_match": 1,
                                }
                            }
                            for group in entity_groups
                        ]
                    }
                }
            )
        elif entity_values:
            filters.append(
                {
                    "bool": {
                        "should": _entity_match_filters(entity_values),
                        "minimum_should_match": 1,
                    }
                }
            )
        if entity_values:
            boosts.extend(
                [
                    {"terms": {"entity_terms": entity_values, "boost": 3.0}},
                    {"terms": {"compound_names": entity_values, "boost": 2.5}},
                    {"terms": {"compound_synonyms": entity_values, "boost": 2.0}},
                    {"terms": {"pair_drug_names": entity_values, "boost": 2.5}},
                ]
            )
        endpoint_values = unique_strings([metadata.get("endpoint"), *(metadata.get("endpoint_terms") or endpoint_terms(metadata.get("endpoint")))], lower=True)
        if endpoint_values:
            boosts.append({"terms": {"endpoint_terms": endpoint_values, "boost": 2.0}})
        return filters, boosts

    def _search_bm25(
        self,
        client: Any,
        query: str,
        *,
        filters: List[Dict[str, Any]],
        lexical_boosts: List[Dict[str, Any]],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        body = {
            "size": top_k,
            "query": {
                "bool": {
                    "should": [
                        {"match": {"text": {"query": query, "boost": 2.0}}},
                        {"match": {"title": {"query": query, "boost": 1.2}}},
                        *lexical_boosts,
                    ],
                    "filter": filters,
                    "minimum_should_match": 1,
                }
            },
            "highlight": {"fields": {"text": {"fragment_size": 240, "number_of_fragments": 1}}},
        }
        response = client.search(index=self.index_name, **body)
        hits = response.get("hits", {}).get("hits", []) or []
        for rank, hit in enumerate(hits, start=1):
            hit["_agentm_channel"] = "bm25"
            hit["_agentm_bm25_rank"] = rank
            hit["_agentm_bm25_score"] = hit.get("_score")
        return hits

    def _search_dense(self, client: Any, query: str, *, filters: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        query_vector = self._embed_query(query)
        dense_filter = [*filters, {"exists": {"field": "embedding"}}]
        body = {
            "size": top_k,
            "query": {
                "script_score": {
                    "query": {"bool": {"filter": dense_filter}},
                    "script": {
                        "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                        "params": {"query_vector": query_vector},
                    },
                }
            },
        }
        response = client.search(index=self.index_name, **body)
        hits = response.get("hits", {}).get("hits", []) or []
        for rank, hit in enumerate(hits, start=1):
            hit["_agentm_channel"] = "dense"
            hit["_agentm_dense_rank"] = rank
            hit["_agentm_dense_score"] = hit.get("_score")
        return hits

    def _merge_ranked_hits(
        self,
        bm25_hits: List[Dict[str, Any]],
        dense_hits: List[Dict[str, Any]],
        *,
        metadata: Optional[Mapping[str, Any]] = None,
        top_k: int,
    ) -> List[Dict[str, Any]]:
        metadata = metadata or {}
        merged: Dict[str, Dict[str, Any]] = {}
        rank_constant = 60.0
        for channel, hits in [("bm25", bm25_hits), ("dense", dense_hits)]:
            for rank, hit in enumerate(hits, start=1):
                source = hit.get("_source", {}) or {}
                chunk_id = str(source.get("chunk_id") or hit.get("_id"))
                if chunk_id not in merged:
                    merged[chunk_id] = dict(hit)
                    merged[chunk_id]["_agentm_component_scores"] = {}
                    merged[chunk_id]["_agentm_component_ranks"] = {}
                    merged[chunk_id]["_agentm_hybrid_score"] = 0.0
                merged_hit = merged[chunk_id]
                merged_hit["_agentm_component_scores"][channel] = hit.get("_score")
                merged_hit["_agentm_component_ranks"][channel] = rank
                merged_hit["_agentm_hybrid_score"] += 1.0 / (rank_constant + rank)
                rerank_bonus = self._metadata_rerank_bonus(source, metadata)
                merged_hit["_agentm_metadata_rerank_bonus"] = rerank_bonus
                merged_hit["_agentm_rerank_score"] = merged_hit["_agentm_hybrid_score"] + rerank_bonus
                merged_hit["_score"] = merged_hit["_agentm_rerank_score"]
                merged_hit["_agentm_channel"] = "hybrid" if len(merged_hit["_agentm_component_scores"]) > 1 else channel
        ranked = sorted(
            merged.values(),
            key=lambda hit: (
                float(hit.get("_agentm_rerank_score", hit.get("_agentm_hybrid_score", 0.0))),
                float(hit.get("_agentm_hybrid_score", 0.0)),
                float(hit.get("_score", 0.0) or 0.0),
            ),
            reverse=True,
        )
        return ranked[:top_k]

    def _metadata_rerank_bonus(self, source: Mapping[str, Any], metadata: Mapping[str, Any]) -> float:
        source_entities = set(_lower_list(source.get("entity_terms")))
        source_text_entities = set(_lower_list(source.get("text_entity_terms")))
        source_entities.update(_lower_list(source.get("drug_names")))
        source_entities.update(_lower_list(source.get("compound_names")))
        source_entities.update(_lower_list(source.get("compound_synonyms")))
        source_entities.update(_lower_list(source.get("compound_cids")))
        source_entities.update(_lower_list(source.get("chembl_ids")))
        source_entities.update(_lower_list(source.get("canonical_smiles")))
        source_entities.update(_lower_list(source.get("pair_drug_names")))
        source_entities.update(_lower_list(source.get("pair_drug_ids")))
        entity_groups = _normalized_entity_groups(metadata)
        query_entities = set(_normalized_entity_values(metadata))
        endpoint = compact_term(metadata.get("endpoint"))
        source_endpoint = compact_term(source.get("endpoint"))
        source_endpoint_terms = set(_lower_list(source.get("endpoint_terms")))
        bonus = 0.0
        if entity_groups:
            match_space = source_text_entities or source_entities
            matched_groups = sum(1 for group in entity_groups if match_space.intersection(group))
            if matched_groups == len(entity_groups):
                bonus += 7.0
            elif matched_groups:
                bonus += 1.5 * matched_groups
                bonus -= 3.0 * (len(entity_groups) - matched_groups)
            else:
                bonus -= 4.0
        elif query_entities and source_entities.intersection(query_entities):
            bonus += 5.0
        elif query_entities and source.get("compound_specific"):
            bonus += 1.0
        if endpoint and source.get("endpoint_specific") and (endpoint == source_endpoint or endpoint in source_endpoint_terms):
            bonus += 3.0
        if metadata.get("profile_type") == "general_drug_profile":
            if source_endpoint == "ddi" or source.get("evidence_scope") in {"ddi_pair", "ddi_pair_background"}:
                bonus -= 8.0
            if source.get("evidence_scope") in {"background", "compound_background"}:
                bonus += 2.0
            if not source.get("endpoint_specific"):
                bonus += 1.0
        if source.get("evidence_scope") in {"compound_endpoint", "ddi_pair"}:
            bonus += 2.0
        if source.get("source") in {"pubchem", "dailymed", "chembl"}:
            bonus += 0.5
        if query_entities and not entity_groups and not source_entities.intersection(query_entities):
            bonus -= 4.0
        return bonus

    def _index_has_dense_vector(self, client: Any) -> bool:
        if self._dense_mapping_checked is not None:
            return self._dense_mapping_checked
        mapping = client.indices.get_mapping(index=self.index_name)
        payload = mapping.get(self.index_name)
        # Elasticsearch resolves an alias to one or more concrete index names
        # and keys the mapping response by the concrete name.  A single-index
        # versioned alias must therefore be unwrapped before inspecting fields.
        if not isinstance(payload, Mapping) and len(mapping) == 1:
            candidate = next(iter(mapping.values()))
            payload = candidate if isinstance(candidate, Mapping) else {}
        if not isinstance(payload, Mapping):
            payload = {}
        properties = payload.get("mappings", {}).get("properties", {})
        self._dense_mapping_checked = properties.get("embedding", {}).get("type") == "dense_vector"
        return self._dense_mapping_checked

    def _embed_query(self, query: str) -> List[float]:
        if self._embedder is None:
            from agentm.tools.rag import DenseTextEmbedder

            self._embedder = DenseTextEmbedder(
                str(default_resource_path(self.embedding_model_name)),
                backend=self.embedding_backend,
                device=self.embedding_device,
                local_files_only=self.embedding_local_files_only,
                batch_size=self.embedding_batch_size,
            )
        embedding = self._embedder.encode([query])
        if embedding.ndim != 2 or embedding.shape[0] != 1:
            raise ValueError("Dense embedder returned an invalid query vector.")
        return [float(value) for value in embedding[0].tolist()]


def _lower_list(values: Any) -> List[str]:
    if values is None:
        return []
    if isinstance(values, str):
        values = [values]
    return unique_strings(values or [], lower=True)


def _normalized_entity_values(metadata: Mapping[str, Any]) -> List[str]:
    values: List[Any] = []
    for group in metadata.get("entity_groups") or []:
        if isinstance(group, str):
            values.append(group)
        else:
            values.extend(group or [])
    for key in ["drug_names", "compound_names", "compound_synonyms", "entity_terms"]:
        raw = metadata.get(key)
        if isinstance(raw, str):
            values.append(raw)
        else:
            values.extend(raw or [])
    return unique_strings(values, lower=True)


def _normalized_entity_groups(metadata: Mapping[str, Any]) -> List[List[str]]:
    groups: List[List[str]] = []
    for raw_group in metadata.get("entity_groups") or []:
        values = [raw_group] if isinstance(raw_group, str) else list(raw_group or [])
        group = unique_strings(values, lower=True)
        if group:
            groups.append(group)
    return groups


def _entity_match_filters(entity_values: List[str]) -> List[Dict[str, Any]]:
    return [
        {"terms": {"text_entity_terms": entity_values}},
    ]
