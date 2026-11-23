from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from agentm.defaults import DEFAULT_CHEMBERTA2_MODEL_PATH

DEFAULT_CHEMBERTA2_MODEL = DEFAULT_CHEMBERTA2_MODEL_PATH
_SMILES_ENCODER_CACHE: Dict[tuple, SmilesEncoder] = {}


class SmilesEncoder:
    encoder_name = "smiles_encoder_interface"

    def __init__(
        self,
        model_name: Optional[str] = None,
        embedding_dim: int = 128,
        pooling: str = "cls",
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.model_name = model_name
        self.embedding_dim = int(embedding_dim)
        self.pooling = pooling
        self.metadata = metadata or {}

    def encode(self, smiles: str) -> List[float]:
        raise NotImplementedError


class MockSmilesEncoder(SmilesEncoder):
    """Deterministic offline fallback for SMILES representation tests and demos."""

    encoder_name = "mock_smiles_hash_encoder"

    def encode(self, smiles: str) -> List[float]:
        payload = f"{self.model_name or 'mock'}::{self.pooling}::{smiles}".encode("utf-8")
        seed_bytes = hashlib.sha256(payload).digest()[:8]
        seed = int.from_bytes(seed_bytes, byteorder="big", signed=False)
        rng = np.random.default_rng(seed)
        vec = rng.normal(loc=0.0, scale=1.0, size=self.embedding_dim)
        norm = float(np.linalg.norm(vec)) or 1.0
        return (vec / norm).astype(float).tolist()


class ChemBERTa2Encoder(SmilesEncoder):
    """Hugging Face ChemBERTa-2 encoder for canonical SMILES strings."""

    encoder_name = "deepchem_chemberta2"

    def __init__(
        self,
        model_name: str = DEFAULT_CHEMBERTA2_MODEL,
        *,
        device: str = "auto",
        pooling: str = "cls",
        max_length: int = 256,
        local_files_only: bool = True,
    ):
        self.max_length = int(max_length)
        self.local_files_only = bool(local_files_only)
        self.device_name = device
        resolved_model_name = _resolve_model_name_or_path(model_name)
        self.device, self.tokenizer, self.model = self._load_model(resolved_model_name, device)
        hidden = int(getattr(self.model.config, "hidden_size", 0) or 0)
        super().__init__(
            model_name=resolved_model_name,
            embedding_dim=hidden,
            pooling=pooling,
            metadata={
                "local_files_only": self.local_files_only,
                "max_length": self.max_length,
                "device": str(self.device),
            },
        )
        self.model.eval()

    def encode(self, smiles: str) -> List[float]:
        import torch

        encoded = self.tokenizer(
            smiles,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=self.max_length,
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        with torch.no_grad():
            output = self.model(**encoded)
        token_embeddings = output.last_hidden_state
        if self.pooling == "mean":
            mask = encoded["attention_mask"].unsqueeze(-1).float()
            embedding = (token_embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
            embedding = embedding.squeeze(0)
        else:
            embedding = token_embeddings[:, 0, :].squeeze(0)
        values = embedding.detach().cpu().float().numpy()
        norm = float(np.linalg.norm(values)) or 1.0
        return (values / norm).astype(float).tolist()

    def _load_model(self, model_name: str, device_name: str):
        _ensure_hf_cache_not_old_project()
        import torch
        from transformers import AutoModel, AutoTokenizer

        if device_name == "auto":
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(device_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=self.local_files_only)
        try:
            model = AutoModel.from_pretrained(model_name, local_files_only=self.local_files_only, add_pooling_layer=False)
        except TypeError:
            model = AutoModel.from_pretrained(model_name, local_files_only=self.local_files_only)
        model.to(device)
        return device, tokenizer, model


def _ensure_hf_cache_not_old_project() -> None:
    cache_values = [os.environ.get("HF_HOME"), os.environ.get("TRANSFORMERS_CACHE")]
    if not any(value and "AgentD-release" in value for value in cache_values):
        return
    project_root = Path(__file__).resolve().parents[2]
    cache_root = project_root / "checkpoints" / "huggingface"
    transformers_cache = cache_root / "transformers"
    transformers_cache.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache_root)
    os.environ["TRANSFORMERS_CACHE"] = str(transformers_cache)


def _resolve_model_name_or_path(model_name: str) -> str:
    path = Path(model_name).expanduser()
    if path.exists():
        return str(path.resolve())
    project_root = Path(__file__).resolve().parents[2]
    candidate = project_root / path
    if candidate.exists():
        return str(candidate.resolve())
    return model_name


def load_smiles_encoder(
    *,
    model_name: Optional[str] = None,
    use_smiles_encoder: bool = False,
    device: str = "auto",
    pooling: str = "cls",
    local_files_only: bool = True,
    embedding_dim: int = 128,
) -> Optional[SmilesEncoder]:
    if not use_smiles_encoder:
        return None
    cache_key = (
        str(model_name or "mock"),
        bool(use_smiles_encoder),
        str(device),
        str(pooling),
        bool(local_files_only),
        int(embedding_dim),
    )
    if cache_key in _SMILES_ENCODER_CACHE:
        return _SMILES_ENCODER_CACHE[cache_key]
    if not model_name or str(model_name).lower() in {"mock", "none", "disabled"}:
        encoder = MockSmilesEncoder(model_name=model_name or "mock", embedding_dim=embedding_dim, pooling=pooling)
        _SMILES_ENCODER_CACHE[cache_key] = encoder
        return encoder
    try:
        encoder = ChemBERTa2Encoder(
            model_name=model_name,
            device=device,
            pooling=pooling,
            local_files_only=local_files_only,
        )
        _SMILES_ENCODER_CACHE[cache_key] = encoder
        return encoder
    except Exception as exc:
        encoder = MockSmilesEncoder(
            model_name=model_name,
            embedding_dim=embedding_dim,
            pooling=pooling,
            metadata={
                "fallback_reason": str(exc),
                "requested_encoder": "ChemBERTa2Encoder",
                "local_files_only": local_files_only,
            },
        )
        _SMILES_ENCODER_CACHE[cache_key] = encoder
        return encoder
