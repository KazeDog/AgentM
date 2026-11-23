from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from agentm.defaults import DEFAULT_RAG_BACKEND, DEFAULT_RAG_EMBEDDING_MODEL_PATH
from agentm.llm.provider import LLMClient, build_llm_client
from agentm.tools.rag import DenseFaissRAGIndex, LiteratureLoader, TextChunker, TfidfRAGIndex
from agentm.utils.io import ensure_dir, write_json, write_text


@dataclass
class RAGAnswer:
    question: str
    answer: str
    retrieved_context: List[Dict[str, Any]] = field(default_factory=list)
    provider: str = "extractive_fallback"
    artifacts: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class RAGQuestionAnsweringAgent:
    """Literature-grounded scientific QA over extracted papers/PDFs."""

    def __init__(
        self,
        output_dir: str | Path = "outputs/rag_qa",
        *,
        llm_profile: Optional[str] = None,
        use_llm: bool = True,
        llm_client: Optional[LLMClient] = None,
        rag_backend: str = DEFAULT_RAG_BACKEND,
        embedding_model_name: str = DEFAULT_RAG_EMBEDDING_MODEL_PATH,
        embedding_backend: str = "auto",
        embedding_device: str = "auto",
        embedding_local_files_only: bool = True,
        embedding_batch_size: int = 16,
        chunk_size: int = 1000,
        chunk_overlap: int = 100,
    ):
        self.output_dir = ensure_dir(output_dir)
        self.loader = LiteratureLoader()
        self.chunker = TextChunker(chunk_size=chunk_size, overlap=chunk_overlap)
        self.use_llm = bool(use_llm)
        self.llm_client = llm_client if llm_client is not None else (build_llm_client(llm_profile) if use_llm else None)
        self.rag_backend = rag_backend
        self.embedding_model_name = embedding_model_name
        self.embedding_backend = embedding_backend
        self.embedding_device = embedding_device
        self.embedding_local_files_only = bool(embedding_local_files_only)
        self.embedding_batch_size = int(embedding_batch_size)

    def build_index(self, literature_dir: str | Path, *, extra_paths: Optional[List[str | Path]] = None):
        docs = self.loader.load_directory(literature_dir)
        if extra_paths:
            docs.extend(self.loader.load_paths(extra_paths))
        chunks = self.chunker.chunk_documents(docs)
        if self.rag_backend == "tfidf":
            index = TfidfRAGIndex(chunks=chunks)
        elif self.rag_backend == "dense_faiss":
            index = DenseFaissRAGIndex(
                chunks=chunks,
                embedding_model_name=self.embedding_model_name,
                embedding_backend=self.embedding_backend,
                embedding_device=self.embedding_device,
                local_files_only=self.embedding_local_files_only,
                batch_size=self.embedding_batch_size,
            )
        else:
            raise ValueError(f"Unsupported rag_backend: {self.rag_backend}")
        index.save_metadata(self.output_dir)
        return index

    def answer(self, question: str, index, *, top_k: int = 5) -> RAGAnswer:
        retrieved = index.search(question, top_k=top_k)
        warnings: List[str] = []
        if not retrieved:
            answer = "I could not find relevant retrieved context for this question."
            rag_answer = RAGAnswer(question=question, answer=answer, retrieved_context=[], warnings=["no_retrieved_context"])
            self._write_answer(rag_answer)
            return rag_answer

        if self.use_llm and self.llm_client and self.llm_client.available:
            try:
                answer = self._llm_answer(question, retrieved)
                rag_answer = RAGAnswer(
                    question=question,
                    answer=answer,
                    retrieved_context=self._compact_context(retrieved),
                    provider=f"llm_rag:{self.llm_client.config.profile}",
                )
                self._write_answer(rag_answer)
                return rag_answer
            except Exception as exc:
                warnings.append(f"LLM RAG answer failed; used extractive fallback: {exc}")

        answer = self._extractive_answer(question, retrieved)
        rag_answer = RAGAnswer(
            question=question,
            answer=answer,
            retrieved_context=self._compact_context(retrieved),
            provider="extractive_fallback",
            warnings=warnings,
        )
        self._write_answer(rag_answer)
        return rag_answer

    def _llm_answer(self, question: str, retrieved: List[Dict[str, Any]]) -> str:
        assert self.llm_client is not None
        context = self._context_text(retrieved)
        prompt = f"""
Answer the scientific question using only the retrieved context.
If the context is insufficient, say what is missing.
Cite evidence using bracketed chunk ids such as [1] or [2].

Question:
{question}

Retrieved context:
{context}
""".strip()
        response = self.llm_client.chat_json(
            [
                {
                    "role": "system",
                    "content": (
                        "You are a careful biomedical RAG assistant. "
                        "Return JSON with fields answer and cited_chunk_ids."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
        )
        try:
            payload = json.loads(response)
            if isinstance(payload, dict) and payload.get("answer"):
                return str(payload["answer"])
        except Exception:
            pass
        return response

    def _extractive_answer(self, question: str, retrieved: List[Dict[str, Any]]) -> str:
        lines = [
            "Retrieved evidence suggests the following context is relevant to the question.",
            f"Question: {question}",
        ]
        for idx, item in enumerate(retrieved[:3], start=1):
            snippet = item["text"][:600].strip()
            lines.append(f"[{idx}] {snippet}")
        return "\n".join(lines)

    def _context_text(self, retrieved: List[Dict[str, Any]]) -> str:
        parts = []
        for idx, item in enumerate(retrieved, start=1):
            source = item.get("source_path") or item.get("doc_id")
            parts.append(f"[{idx}] chunk_id={item.get('chunk_id')} source={source}\n{item.get('text')}")
        return "\n\n".join(parts)

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
                    "snippet": str(item.get("text") or "")[:500],
                }
            )
        return compact

    def _write_answer(self, answer: RAGAnswer) -> None:
        answer_path = self.output_dir / "rag_answer.json"
        md_path = self.output_dir / "rag_answer.md"
        answer.artifacts.update({"answer_path": str(answer_path), "markdown_path": str(md_path)})
        write_json(answer_path, answer.to_dict())
        markdown = [
            "# RAG QA Answer",
            "",
            f"**Question:** {answer.question}",
            f"**Provider:** {answer.provider}",
            "",
            "## Answer",
            answer.answer,
            "",
            "## Retrieved Context",
        ]
        for item in answer.retrieved_context:
            markdown.append(f"- [{item.get('rank')}] `{item.get('chunk_id')}` score={item.get('score'):.4f} source=`{item.get('source_path')}`")
        if answer.warnings:
            markdown.extend(["", "## Warnings"])
            markdown.extend(f"- {warning}" for warning in answer.warnings)
        write_text(md_path, "\n".join(markdown).rstrip() + "\n")
