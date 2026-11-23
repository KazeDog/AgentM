from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional

from agentm.defaults import DEFAULT_RAG_BACKEND, DEFAULT_RAG_CORPUS_DIR, DEFAULT_RAG_EMBEDDING_MODEL_PATH, DEFAULT_RAG_TOP_K
from agentm.llm.provider import LLMClient, build_llm_client
from agentm.tools.rag import DenseFaissRAGIndex, LiteratureLoader, TextChunker, TfidfRAGIndex
from agentm.utils.io import ensure_dir, write_json, write_text


@dataclass
class RAGEvidenceResult:
    query: str
    evidence_items: List[Dict[str, Any]] = field(default_factory=list)
    retrieved_context: List[Dict[str, Any]] = field(default_factory=list)
    provider: str = "rag_unavailable"
    backend: str = DEFAULT_RAG_BACKEND
    n_documents: int = 0
    n_chunks: int = 0
    answer: Optional[str] = None
    artifacts: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class RAGEvidenceAgent:
    """Retrieve local literature snippets and convert them into agent evidence items."""

    def __init__(
        self,
        output_dir: str | Path = "outputs/rag",
        *,
        corpus_dirs: Optional[Iterable[str | Path]] = None,
        top_k: int = DEFAULT_RAG_TOP_K,
        rag_backend: str = DEFAULT_RAG_BACKEND,
        embedding_model_name: str = DEFAULT_RAG_EMBEDDING_MODEL_PATH,
        embedding_backend: str = "auto",
        embedding_device: str = "auto",
        embedding_local_files_only: bool = True,
        embedding_batch_size: int = 16,
        chunk_size: int = 1000,
        chunk_overlap: int = 100,
        use_llm: bool = False,
        llm_profile: Optional[str] = None,
        llm_client: Optional[LLMClient] = None,
    ):
        self.output_dir = ensure_dir(output_dir)
        self.corpus_dirs = [Path(path) for path in (corpus_dirs or [DEFAULT_RAG_CORPUS_DIR])]
        self.top_k = int(top_k)
        self.rag_backend = rag_backend
        self.embedding_model_name = embedding_model_name
        self.embedding_backend = embedding_backend
        self.embedding_device = embedding_device
        self.embedding_local_files_only = bool(embedding_local_files_only)
        self.embedding_batch_size = int(embedding_batch_size)
        self.loader = LiteratureLoader()
        self.chunker = TextChunker(chunk_size=chunk_size, overlap=chunk_overlap)
        self.use_llm = bool(use_llm)
        self.llm_client = llm_client if llm_client is not None else (build_llm_client(llm_profile) if use_llm else None)

    def retrieve(
        self,
        *,
        query: str,
        context_metadata: Optional[Mapping[str, Any]] = None,
        corpus_dirs: Optional[Iterable[str | Path]] = None,
        top_k: Optional[int] = None,
    ) -> Dict[str, Any]:
        query = re.sub(r"\s+", " ", query or "").strip()
        warnings: List[str] = []
        context_metadata = context_metadata or {}
        docs = self._load_documents(corpus_dirs or self.corpus_dirs, warnings)
        task_hint = _corpus_task_hint(context_metadata)
        if task_hint and docs:
            task_docs = [
                doc
                for doc in docs
                if f"/tasks/{task_hint}/" in str(doc.source_path or "").replace("\\", "/")
            ]
            if task_docs:
                docs = task_docs
            else:
                warnings.append(f"rag_task_filter_no_matching_docs:{task_hint}")
        if not query:
            warnings.append("rag_query_empty")
        if not docs:
            result = RAGEvidenceResult(query=query, warnings=warnings or ["rag_corpus_unavailable"])
            return self._write_result(result)

        chunks = self.chunker.chunk_documents(docs)
        if not chunks:
            result = RAGEvidenceResult(
                query=query,
                n_documents=len(docs),
                warnings=warnings + ["rag_corpus_has_no_chunks"],
            )
            return self._write_result(result)

        retrieved: List[Dict[str, Any]] = []
        backend_used = self.rag_backend
        provider = f"{backend_used}_retrieval"
        try:
            index = self._build_index(chunks, backend=self.rag_backend)
            retrieved = index.search(query, top_k=top_k or self.top_k)
            index.save_metadata(self.output_dir)
        except Exception as exc:
            if self.rag_backend != "tfidf":
                warnings.append(f"rag_dense_backend_failed_used_tfidf: {exc}")
                backend_used = "tfidf"
                provider = "tfidf_retrieval"
                try:
                    index = self._build_index(chunks, backend="tfidf")
                    retrieved = index.search(query, top_k=top_k or self.top_k)
                    index.save_metadata(self.output_dir)
                except Exception as fallback_exc:
                    warnings.append(f"rag_tfidf_backend_failed: {fallback_exc}")
            else:
                warnings.append(f"rag_tfidf_backend_failed: {exc}")

        compact_context = self._compact_context(retrieved)
        evidence_items = self._evidence_items(query, compact_context, context_metadata, backend_used)
        answer = self._optional_llm_answer(query, retrieved, warnings)
        if answer:
            provider = f"llm_rag:{self.llm_client.config.profile}" if self.llm_client else provider
        if not retrieved:
            warnings.append("rag_no_retrieved_context")
        result = RAGEvidenceResult(
            query=query,
            evidence_items=evidence_items,
            retrieved_context=compact_context,
            provider=provider,
            backend=backend_used,
            n_documents=len(docs),
            n_chunks=len(chunks),
            answer=answer,
            warnings=warnings,
        )
        return self._write_result(result)

    def _load_documents(self, paths: Iterable[str | Path], warnings: List[str]) -> List[Any]:
        docs = []
        seen = set()
        for path_like in paths:
            path = Path(path_like)
            if not path.exists():
                if str(path).endswith("data/rag_corpus/tdc_real") or path.name == "tdc_real":
                    warnings.append("real_rag_corpus_missing")
                warnings.append(f"rag_corpus_path_missing:{path}")
                continue
            if path.is_dir():
                nested_paths = sorted(
                    item
                    for suffix in ("*.txt", "*.md", "*.pdf")
                    for item in path.rglob(suffix)
                    if item.is_file()
                )
                loaded = self.loader.load_paths(nested_paths)
            else:
                loaded = self.loader.load_paths([path])
            for doc in loaded:
                key = (doc.doc_id, doc.source_path)
                if key in seen:
                    continue
                seen.add(key)
                docs.append(doc)
        return docs

    def _build_index(self, chunks: List[Any], *, backend: str):
        if backend == "tfidf":
            return TfidfRAGIndex(chunks=chunks)
        if backend == "dense_faiss":
            return DenseFaissRAGIndex(
                chunks=chunks,
                embedding_model_name=self.embedding_model_name,
                embedding_backend=self.embedding_backend,
                embedding_device=self.embedding_device,
                local_files_only=self.embedding_local_files_only,
                batch_size=self.embedding_batch_size,
            )
        raise ValueError(f"Unsupported rag_backend: {backend}")

    def _compact_context(self, retrieved: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        compact = []
        for item in retrieved:
            compact.append(
                {
                    "rank": item.get("rank"),
                    "score": item.get("score"),
                    "chunk_id": item.get("chunk_id"),
                    "doc_id": item.get("doc_id"),
                    "source_path": item.get("source_path"),
                    "snippet": str(item.get("text") or "")[:800],
                    "metadata": item.get("metadata") or {},
                }
            )
        return compact

    def _evidence_items(
        self,
        query: str,
        context: List[Dict[str, Any]],
        metadata: Mapping[str, Any],
        backend_used: str,
    ) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        for item in context:
            snippet = re.sub(r"\s+", " ", str(item.get("snippet") or "")).strip()
            pointers = citation_pointers_from_text(snippet, source_path=item.get("source_path"))
            items.append(
                {
                    "source": "rag_literature",
                    "claim": f"RAG retrieved context: {snippet[:240]}",
                    "value": {
                        "query": query,
                        "rank": item.get("rank"),
                        "score": item.get("score"),
                        "chunk_id": item.get("chunk_id"),
                        "doc_id": item.get("doc_id"),
                        "source_path": item.get("source_path"),
                        "snippet": snippet,
                        "task_type": metadata.get("task_type"),
                        "dataset_name": metadata.get("dataset_name"),
                        **pointers,
                    },
                    "evidence_direction": "neutral",
                    "evidence_weight": 0.0,
                    "source_reliability": 0.55,
                    "evidence_scoring_rationale": f"Retrieved by local {backend_used} RAG; direction was left neutral unless another tool scores it.",
                }
            )
        return items

    def _optional_llm_answer(self, query: str, retrieved: List[Dict[str, Any]], warnings: List[str]) -> Optional[str]:
        if not (self.use_llm and self.llm_client and self.llm_client.available and retrieved):
            return None
        context = "\n\n".join(
            f"[{idx}] chunk_id={item.get('chunk_id')} source={item.get('source_path')}\n{str(item.get('text') or '')[:1200]}"
            for idx, item in enumerate(retrieved, start=1)
        )
        prompt = {
            "instruction": "Answer using only the retrieved context. If context is insufficient, say so.",
            "question": query,
            "retrieved_context": context,
        }
        try:
            response = self.llm_client.chat_json(
                [
                    {
                        "role": "system",
                        "content": "You are a careful biomedical RAG assistant. Return JSON with field answer.",
                    },
                    {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
                ],
                temperature=0.0,
            )
            data = json.loads(response)
            if isinstance(data, dict) and data.get("answer"):
                return str(data["answer"])
            return response
        except Exception as exc:
            warnings.append(f"llm_rag_answer_failed:{exc}")
            return None

    def _write_result(self, result: RAGEvidenceResult) -> Dict[str, Any]:
        json_path = self.output_dir / "rag_evidence.json"
        md_path = self.output_dir / "rag_evidence.md"
        result.artifacts.update({"rag_evidence_path": str(json_path), "rag_evidence_markdown_path": str(md_path)})
        write_json(json_path, result.to_dict())
        lines = [
            "# RAG Evidence",
            "",
            f"**Query:** {result.query or 'n/a'}",
            f"**Provider:** {result.provider}",
            f"**Backend:** {result.backend}",
            f"**Documents:** {result.n_documents}",
            f"**Chunks:** {result.n_chunks}",
            "",
            "## Retrieved Context",
        ]
        if result.retrieved_context:
            lines.append("| Rank | Score | Source | Chunk | Snippet |")
            lines.append("|---:|---:|---|---|---|")
            for item in result.retrieved_context:
                lines.append(
                    "| {rank} | {score:.4f} | `{source}` | `{chunk}` | {snippet} |".format(
                        rank=item.get("rank") or 0,
                        score=float(item.get("score") or 0.0),
                        source=_pipe(item.get("source_path")),
                        chunk=_pipe(item.get("chunk_id")),
                        snippet=_pipe(str(item.get("snippet") or "")[:240]),
                    )
                )
        else:
            lines.append("- No retrieved context.")
        if result.answer:
            lines.extend(["", "## Optional LLM Answer", result.answer])
        if result.warnings:
            lines.extend(["", "## Warnings"])
            lines.extend(f"- {warning}" for warning in result.warnings)
        write_text(md_path, "\n".join(lines).rstrip() + "\n")
        return result.to_dict()


def _pipe(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ").strip()


def citation_pointers_from_text(text: str, *, source_path: Any = None) -> Dict[str, Any]:
    combined = f"{text or ''} {source_path or ''}"
    pointers: Dict[str, Any] = {}
    pmid_match = re.search(r"\bPMID[:\s]+(\d{4,12})\b", combined, flags=re.IGNORECASE)
    cid_match = re.search(r"\bCID[:\s]+(\d{1,12})\b|/compound/(\d{1,12})\b", combined, flags=re.IGNORECASE)
    setid_match = re.search(r"\bsetid[:=\s]+([A-Za-z0-9-]{6,80})", combined, flags=re.IGNORECASE)
    url_match = re.search(r"https?://[^\s)`\]}]+", combined)
    if pmid_match:
        pointers["pmid"] = pmid_match.group(1)
        pointers.setdefault("url", f"https://pubmed.ncbi.nlm.nih.gov/{pointers['pmid']}/")
    if cid_match:
        pointers["cid"] = cid_match.group(1) or cid_match.group(2)
        pointers.setdefault("url", f"https://pubchem.ncbi.nlm.nih.gov/compound/{pointers['cid']}")
    if setid_match:
        pointers["setid"] = setid_match.group(1)
        pointers.setdefault("url", f"https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid={pointers['setid']}")
    if url_match:
        pointers["url"] = url_match.group(0).rstrip(".,;")
    return pointers


def _corpus_task_hint(metadata: Mapping[str, Any]) -> Optional[str]:
    dataset = str(metadata.get("dataset_name") or "").strip()
    if not dataset:
        return None
    aliases = {
        "bbbp": "BBB_Martins",
        "bbb": "BBB_Martins",
        "bbb_martins": "BBB_Martins",
        "hia": "HIA_Hou",
        "hia_hou": "HIA_Hou",
        "pgp": "Pgp_Broccatelli",
        "pgp_broccatelli": "Pgp_Broccatelli",
        "bioavailability": "Bioavailability_Ma",
        "bioavailability_ma": "Bioavailability_Ma",
        "cyp1a2": "CYP1A2_Veith",
        "cyp1a2_veith": "CYP1A2_Veith",
        "cyp2c9": "CYP2C9_Veith",
        "cyp2c9_veith": "CYP2C9_Veith",
        "cyp2c19": "CYP2C19_Veith",
        "cyp2c19_veith": "CYP2C19_Veith",
        "cyp2d6": "CYP2D6_Veith",
        "cyp2d6_veith": "CYP2D6_Veith",
        "cyp3a4": "CYP3A4_Veith",
        "cyp3a4_veith": "CYP3A4_Veith",
        "caco2": "Caco2_Wang",
        "caco2_wang": "Caco2_Wang",
        "lipophilicity": "Lipophilicity_AstraZeneca",
        "lipophilicity_astrazeneca": "Lipophilicity_AstraZeneca",
        "solubility": "Solubility_AqSolDB",
        "solubility_aqsolDB".lower(): "Solubility_AqSolDB",
        "hydrationfreeenergy_freesolv": "HydrationFreeEnergy_FreeSolv",
        "freesolv": "HydrationFreeEnergy_FreeSolv",
        "half_life_obach": "Half_Life_Obach",
        "half_life": "Half_Life_Obach",
        "clearance_hepatocyte_az": "Clearance_Hepatocyte_AZ",
        "hepatocyte_clearance": "Clearance_Hepatocyte_AZ",
        "clearance_microsome_az": "Clearance_Microsome_AZ",
        "microsome_clearance": "Clearance_Microsome_AZ",
        "herg": "hERG",
        "ames": "AMES",
        "dili": "DILI",
        "clintox": "ClinTox",
        "ld50": "LD50_Zhu",
        "ld50_zhu": "LD50_Zhu",
    }
    return aliases.get(dataset.lower(), dataset if dataset in set(aliases.values()) else None)
