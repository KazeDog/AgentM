from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from agentm.schema import VisualSemanticBundle
from agentm.utils.io import ensure_dir, read_json, write_json


CACHE_SCHEMA = "agentm_visual_semantics"


def visual_semantic_bundle_from_dict(data: Dict[str, Any]) -> VisualSemanticBundle:
    return VisualSemanticBundle(
        functional_groups=[str(item) for item in data.get("functional_groups") or []],
        ring_systems=[str(item) for item in data.get("ring_systems") or []],
        hbond_features=dict(data.get("hbond_features") or {}),
        hydrophobic_regions=[str(item) for item in data.get("hydrophobic_regions") or []],
        ionizable_centers=[str(item) for item in data.get("ionizable_centers") or []],
        admet_alerts=[str(item) for item in data.get("admet_alerts") or []],
        task_relevant_visual_cues=[str(item) for item in data.get("task_relevant_visual_cues") or []],
        summary=str(data.get("summary") or ""),
        embedding=[float(value) for value in data.get("embedding") or []],
        provider=str(data.get("provider") or "fallback_rdkit"),
        metadata=dict(data.get("metadata") or {}),
    )


class VisualSemanticCache:
    """Small file-backed cache for expensive MLLM visual-semantic annotations."""

    def __init__(self, cache_dir: str | Path, *, profile: Optional[str] = None):
        self.cache_dir = ensure_dir(cache_dir)
        self.profile = profile or "unknown_mllm_profile"

    def key(self, *, smiles: str, task_type: str, dataset_name: Optional[str]) -> str:
        payload = "|".join(
            [
                CACHE_SCHEMA,
                self.profile,
                str(task_type or ""),
                str(dataset_name or ""),
                str(smiles or ""),
            ]
        )
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()

    def path_for(self, *, smiles: str, task_type: str, dataset_name: Optional[str]) -> Path:
        dataset = _safe_name(dataset_name or "unknown_dataset")
        return self.cache_dir / dataset / f"{self.key(smiles=smiles, task_type=task_type, dataset_name=dataset_name)}.json"

    def load(self, *, smiles: str, task_type: str, dataset_name: Optional[str]) -> Optional[VisualSemanticBundle]:
        path = self.path_for(smiles=smiles, task_type=task_type, dataset_name=dataset_name)
        if not path.exists():
            return None
        payload = read_json(path)
        bundle_payload = payload.get("bundle") if isinstance(payload.get("bundle"), dict) else payload
        bundle = visual_semantic_bundle_from_dict(bundle_payload)
        metadata = dict(bundle.metadata)
        metadata.update(
            {
                "cache_hit": True,
                "cache_path": str(path),
                "cache_key": payload.get("cache_key") or path.stem,
                "cache_profile": payload.get("mllm_profile") or self.profile,
            }
        )
        bundle.metadata = metadata
        return bundle

    def store(
        self,
        bundle: VisualSemanticBundle,
        *,
        smiles: str,
        task_type: str,
        dataset_name: Optional[str],
        image_path: Optional[str | Path] = None,
    ) -> Path:
        path = self.path_for(smiles=smiles, task_type=task_type, dataset_name=dataset_name)
        key = path.stem
        payload = {
            "cache_schema": CACHE_SCHEMA,
            "cache_key": key,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "smiles": smiles,
            "task_type": task_type,
            "dataset_name": dataset_name,
            "mllm_profile": self.profile,
            "provider": bundle.provider,
            "image_path": str(image_path) if image_path else None,
            "bundle": bundle.to_dict(),
        }
        write_json(path, payload)
        return path

    def exists(self, *, smiles: str, task_type: str, dataset_name: Optional[str]) -> bool:
        return self.path_for(smiles=smiles, task_type=task_type, dataset_name=dataset_name).exists()


def _safe_name(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)
    return safe[:100] or "unknown"
