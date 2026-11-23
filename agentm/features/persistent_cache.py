from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

from agentm.features.builder import FeatureBuilder
from agentm.schema import MoleculeRecord, Rule, SingleMoleculeFeatureBundle
from agentm.utils.io import ensure_dir, write_json


class PersistentSingleFeatureCache:
    """Disk cache for single-molecule feature bundles used by DDI jobs."""

    schema_name = "agentm_single_molecule_features"

    def __init__(
        self,
        cache_dir: str | Path,
        *,
        read: bool = True,
        write: bool = True,
    ):
        self.cache_dir = ensure_dir(cache_dir)
        self.read = bool(read)
        self.write = bool(write)
        self.stats: Dict[str, Any] = {
            "enabled": True,
            "cache_dir": str(self.cache_dir),
            "read": self.read,
            "write": self.write,
            "hits": 0,
            "misses": 0,
            "writes": 0,
            "read_errors": 0,
            "write_errors": 0,
        }

    def load(
        self,
        *,
        record: MoleculeRecord,
        builder: FeatureBuilder,
        rules: Iterable[Rule],
        task_type: str,
        dataset_name: str,
    ) -> Optional[SingleMoleculeFeatureBundle]:
        key, metadata = self.cache_key(
            record=record,
            builder=builder,
            rules=rules,
            task_type=task_type,
            dataset_name=dataset_name,
        )
        path = self._path_for_key(key)
        if not self.read:
            return None
        if not path.exists():
            self.stats["misses"] += 1
            return None
        try:
            with path.open("rb") as handle:
                payload = pickle.load(handle)
            if payload.get("cache_key") != key:
                self.stats["misses"] += 1
                return None
            bundle = payload.get("bundle")
            if not isinstance(bundle, SingleMoleculeFeatureBundle):
                self.stats["misses"] += 1
                return None
            if not bundle.feature_vector or not bundle.feature_schema.feature_names:
                self.stats["misses"] += 1
                return None
            self.stats["hits"] += 1
            return bundle
        except Exception as exc:
            self.stats["read_errors"] += 1
            self._write_error("read", key, metadata, exc)
            return None

    def store(
        self,
        *,
        record: MoleculeRecord,
        builder: FeatureBuilder,
        rules: Iterable[Rule],
        task_type: str,
        dataset_name: str,
        bundle: SingleMoleculeFeatureBundle,
    ) -> Optional[Path]:
        if not self.write:
            return None
        key, metadata = self.cache_key(
            record=record,
            builder=builder,
            rules=rules,
            task_type=task_type,
            dataset_name=dataset_name,
        )
        path = self._path_for_key(key)
        try:
            payload = {
                "cache_key": key,
                "metadata": metadata,
                "bundle": bundle,
            }
            tmp_path = path.with_suffix(".tmp")
            with tmp_path.open("wb") as handle:
                pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
            tmp_path.replace(path)
            self.stats["writes"] += 1
            if self.stats["writes"] % 100 == 0:
                self.write_manifest()
            return path
        except Exception as exc:
            self.stats["write_errors"] += 1
            self._write_error("write", key, metadata, exc)
            return None

    def cache_key(
        self,
        *,
        record: MoleculeRecord,
        builder: FeatureBuilder,
        rules: Iterable[Rule],
        task_type: str,
        dataset_name: str,
    ) -> Tuple[str, Dict[str, Any]]:
        canonical_smiles = record.canonical_smiles or record.smiles or record.original_input
        metadata = {
            "cache_schema": self.schema_name,
            "canonical_smiles": canonical_smiles,
            "task_type": task_type,
            "dataset_name": dataset_name,
            "feature_set": builder.feature_set,
            "imagemol_checkpoint_path": builder.imagemol_checkpoint_path,
            "use_smiles_encoder": builder.use_smiles_encoder,
            "smiles_encoder_model_name": builder.smiles_encoder_model_name,
            "smiles_pooling": builder.smiles_pooling,
            "visual_provider": getattr(builder.visual_extractor, "provider", None),
            "use_mllm_visual_semantic": bool(getattr(builder.visual_extractor, "use_mllm", False)),
            "rules_signature": rules_signature(rules),
        }
        key = _sha256_json(metadata)
        metadata["cache_key"] = key
        return key, metadata

    def write_manifest(self) -> Path:
        return write_json(self.cache_dir / "manifest.json", self.stats)

    def _path_for_key(self, key: str) -> Path:
        return self.cache_dir / f"{key}.pkl"

    def _write_error(self, operation: str, key: str, metadata: Dict[str, Any], exc: Exception) -> None:
        error_path = self.cache_dir / "errors" / f"{operation}_{key}.json"
        write_json(error_path, {"operation": operation, "cache_key": key, "metadata": metadata, "error": str(exc)})


def rules_signature(rules: Iterable[Rule]) -> str:
    payload = []
    for rule in rules:
        payload.append(
            {
                "rule_id": rule.rule_id,
                "task_type": rule.task_type,
                "source": rule.source,
                "text": rule.text,
                "feature_name": rule.feature_name,
                "code": rule.code,
                "metadata": rule.metadata,
                "is_executable": rule.is_executable,
                "error": rule.error,
            }
        )
    payload.sort(key=lambda item: (str(item["feature_name"]), str(item["source"]), str(item["text"])))
    return _sha256_json(payload)


def _sha256_json(value: Any) -> str:
    text = json.dumps(value, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
