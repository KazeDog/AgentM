from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, Optional, Protocol

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from agentm.defaults import DEFAULT_RAG_EMBEDDING_MODEL_PATH
from agentm.utils.io import ensure_dir, write_json


@dataclass
class TextDocument:
    doc_id: str
    text: str
    source_path: Optional[str] = None
    metadata: Dict[str, Any] | None = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TextChunk:
    chunk_id: str
    doc_id: str
    text: str
    source_path: Optional[str] = None
    metadata: Dict[str, Any] | None = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class LiteratureLoader:
    def load_paths(self, paths: Iterable[str | Path]) -> List[TextDocument]:
        docs: List[TextDocument] = []
        for path_like in paths:
            path = Path(path_like)
            if not path.exists() or path.is_dir():
                continue
            text = self._read_path(path)
            if not text.strip():
                continue
            docs.append(
                TextDocument(
                    doc_id=self._doc_id(path),
                    text=text,
                    source_path=str(path),
                    metadata={"suffix": path.suffix.lower()},
                )
            )
        return docs

    def load_directory(self, directory: str | Path) -> List[TextDocument]:
        root = Path(directory)
        if not root.exists():
            return []
        paths = sorted([*root.glob("*.txt"), *root.glob("*.md"), *root.glob("*.pdf")])
        return self.load_paths(paths)

    def _read_path(self, path: Path) -> str:
        if path.suffix.lower() == ".pdf":
            return self._read_pdf(path)
        return path.read_text(encoding="utf-8", errors="ignore")

    def _read_pdf(self, path: Path) -> str:
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise RuntimeError("pypdf is required for PDF RAG extraction.") from exc
        reader = PdfReader(str(path))
        pages = []
        for idx, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                pages.append(f"[page {idx + 1}]\n{text}")
        return "\n\n".join(pages)

    def _doc_id(self, path: Path) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", path.stem)[:100]


class TextChunker:
    def __init__(self, *, chunk_size: int = 1000, overlap: int = 100):
        self.chunk_size = int(chunk_size)
        self.overlap = int(overlap)

    def chunk_documents(self, documents: Iterable[TextDocument]) -> List[TextChunk]:
        chunks: List[TextChunk] = []
        for doc in documents:
            text = self._normalize(doc.text)
            if not text:
                continue
            step = max(1, self.chunk_size - self.overlap)
            start = 0
            part = 0
            while start < len(text):
                end = min(len(text), start + self.chunk_size)
                chunk_text = text[start:end].strip()
                if chunk_text:
                    chunks.append(
                        TextChunk(
                            chunk_id=f"{doc.doc_id}::chunk_{part}",
                            doc_id=doc.doc_id,
                            text=chunk_text,
                            source_path=doc.source_path,
                            metadata={**(doc.metadata or {}), "start": start, "end": end},
                        )
                    )
                part += 1
                if end >= len(text):
                    break
                start += step
        return chunks

    def _normalize(self, text: str) -> str:
        return re.sub(r"\s+", " ", text or "").strip()


class TfidfRAGIndex:
    """Small local vector index for literature-grounded QA."""

    def __init__(self, *, chunks: Optional[List[TextChunk]] = None):
        self.chunks = chunks or []
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.matrix = None
        if self.chunks:
            self.fit(self.chunks)

    def fit(self, chunks: List[TextChunk]) -> "TfidfRAGIndex":
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=50000)
        self.matrix = self.vectorizer.fit_transform([chunk.text for chunk in chunks])
        return self

    def search(self, query: str, *, top_k: int = 5) -> List[Dict[str, Any]]:
        if not self.chunks or self.vectorizer is None or self.matrix is None:
            return []
        q = self.vectorizer.transform([query])
        scores = cosine_similarity(q, self.matrix)[0]
        if not np.any(scores):
            return []
        order = np.argsort(scores)[::-1][:top_k]
        results = []
        for rank, idx in enumerate(order, start=1):
            score = float(scores[idx])
            if score <= 0:
                continue
            chunk = self.chunks[int(idx)]
            results.append(
                {
                    "rank": rank,
                    "score": score,
                    "chunk_id": chunk.chunk_id,
                    "doc_id": chunk.doc_id,
                    "text": chunk.text,
                    "source_path": chunk.source_path,
                    "metadata": chunk.metadata or {},
                }
            )
        return results

    def save_metadata(self, output_dir: str | Path) -> Dict[str, Any]:
        target = ensure_dir(output_dir)
        chunks_path = write_json(target / "rag_chunks.json", {"chunks": [chunk.to_dict() for chunk in self.chunks]})
        index_info_path = write_json(
            target / "rag_index_info.json",
            {
                "index_type": "tfidf_cosine",
                "n_chunks": len(self.chunks),
                "vectorizer": "sklearn.TfidfVectorizer",
            },
        )
        return {"chunks_path": str(chunks_path), "index_info_path": str(index_info_path)}


class TextEmbedder(Protocol):
    model_name_or_path: str
    backend: str

    def encode(self, texts: List[str]) -> np.ndarray:
        ...


class DenseTextEmbedder:
    """Dense text embeddings with SentenceTransformer or SciBERT mean pooling."""

    def __init__(
        self,
        model_name_or_path: str = DEFAULT_RAG_EMBEDDING_MODEL_PATH,
        *,
        backend: str = "auto",
        device: str = "auto",
        local_files_only: bool = True,
        batch_size: int = 16,
        max_length: int = 512,
    ):
        self.model_name_or_path = self._resolve_model_name_or_path(model_name_or_path)
        self.backend = backend
        self.device_name = device
        self.local_files_only = bool(local_files_only)
        self.batch_size = int(batch_size)
        self.max_length = int(max_length)
        self._st_model = None
        self._tokenizer = None
        self._model = None
        self._device = None
        self._load()

    def encode(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        if self.backend == "sentence_transformer":
            embeddings = self._st_model.encode(
                texts,
                batch_size=self.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            return np.asarray(embeddings, dtype=np.float32)
        return self._encode_transformers(texts)

    def _load(self) -> None:
        requested = self.backend
        if requested in {"auto", "sentence_transformer"}:
            try:
                from sentence_transformers import SentenceTransformer

                kwargs = {}
                if self.device_name and self.device_name != "auto":
                    kwargs["device"] = self.device_name
                self._st_model = SentenceTransformer(self.model_name_or_path, local_files_only=self.local_files_only, **kwargs)
                self.backend = "sentence_transformer"
                return
            except Exception:
                if requested == "sentence_transformer":
                    raise

        self._load_transformers()
        self.backend = "transformers_mean_pool"

    def _load_transformers(self) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        if self.device_name == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = self.device_name
        self._device = torch.device(device)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, local_files_only=self.local_files_only)
        self._model = AutoModel.from_pretrained(self.model_name_or_path, local_files_only=self.local_files_only)
        self._model.to(self._device)
        self._model.eval()

    def _encode_transformers(self, texts: List[str]) -> np.ndarray:
        import torch

        vectors: List[np.ndarray] = []
        assert self._tokenizer is not None and self._model is not None and self._device is not None
        with torch.no_grad():
            for start in range(0, len(texts), self.batch_size):
                batch = texts[start : start + self.batch_size]
                encoded = self._tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                encoded = {key: value.to(self._device) for key, value in encoded.items()}
                output = self._model(**encoded)
                token_embeddings = output.last_hidden_state
                attention_mask = encoded["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()
                summed = torch.sum(token_embeddings * attention_mask, dim=1)
                counts = torch.clamp(attention_mask.sum(dim=1), min=1e-9)
                pooled = summed / counts
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
                vectors.append(pooled.detach().cpu().numpy())
        return np.concatenate(vectors, axis=0).astype(np.float32)

    def _resolve_model_name_or_path(self, model_name_or_path: str) -> str:
        path = Path(model_name_or_path)
        if path.exists():
            return str(path.resolve())
        project_root = Path(__file__).resolve().parents[2]
        candidate = project_root / model_name_or_path
        if candidate.exists():
            return str(candidate.resolve())
        return model_name_or_path


class DenseFaissRAGIndex:
    """Dense semantic retrieval backed by FAISS inner-product search."""

    def __init__(
        self,
        *,
        chunks: Optional[List[TextChunk]] = None,
        embedder: Optional[TextEmbedder] = None,
        embedding_model_name: str = DEFAULT_RAG_EMBEDDING_MODEL_PATH,
        embedding_backend: str = "auto",
        embedding_device: str = "auto",
        local_files_only: bool = True,
        batch_size: int = 16,
    ):
        self.chunks = chunks or []
        self.embedder = embedder or DenseTextEmbedder(
            embedding_model_name,
            backend=embedding_backend,
            device=embedding_device,
            local_files_only=local_files_only,
            batch_size=batch_size,
        )
        self.index = None
        self.embeddings: Optional[np.ndarray] = None
        if self.chunks:
            self.fit(self.chunks)

    def fit(self, chunks: List[TextChunk]) -> "DenseFaissRAGIndex":
        import faiss

        self.chunks = chunks
        embeddings = self.embedder.encode([chunk.text for chunk in chunks]).astype(np.float32)
        if embeddings.ndim != 2 or embeddings.shape[0] != len(chunks):
            raise ValueError("Dense embedder returned an invalid embedding matrix.")
        self.embeddings = embeddings
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)
        return self

    def search(self, query: str, *, top_k: int = 5) -> List[Dict[str, Any]]:
        if not self.chunks or self.index is None:
            return []
        query_embedding = self.embedder.encode([query]).astype(np.float32)
        scores, indices = self.index.search(query_embedding, min(top_k, len(self.chunks)))
        results = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
            if idx < 0:
                continue
            chunk = self.chunks[int(idx)]
            results.append(
                {
                    "rank": rank,
                    "score": float(score),
                    "chunk_id": chunk.chunk_id,
                    "doc_id": chunk.doc_id,
                    "text": chunk.text,
                    "source_path": chunk.source_path,
                    "metadata": chunk.metadata or {},
                }
            )
        return results

    def save_metadata(self, output_dir: str | Path) -> Dict[str, Any]:
        target = ensure_dir(output_dir)
        chunks_path = write_json(target / "rag_chunks.json", {"chunks": [chunk.to_dict() for chunk in self.chunks]})
        index_path = None
        if self.index is not None:
            try:
                import faiss

                index_path_obj = target / "rag_faiss.index"
                faiss.write_index(self.index, str(index_path_obj))
                index_path = str(index_path_obj)
            except Exception:
                index_path = None
        index_info_path = write_json(
            target / "rag_index_info.json",
            {
                "index_type": "dense_faiss",
                "n_chunks": len(self.chunks),
                "embedding_model": self.embedder.model_name_or_path,
                "embedding_backend": self.embedder.backend,
                "embedding_dim": int(self.embeddings.shape[1]) if self.embeddings is not None and self.embeddings.ndim == 2 else None,
                "faiss_index_path": index_path,
            },
        )
        return {"chunks_path": str(chunks_path), "index_info_path": str(index_info_path), "faiss_index_path": index_path}
