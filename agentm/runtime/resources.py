from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from agentm.runtime.compat import RESOURCE_ROOT, ensure_import_paths, to_jsonable

ensure_import_paths()

from agentm.defaults import DEFAULT_DDI_RAG_CORPUS_DIR, DEFAULT_RAG_CORPUS_DIR, DEFAULT_RAG_EMBEDDING_MODEL_PATH
from agentm.models.predictors import SklearnPredictor


DATASET_ALIASES = {
    "bbbp": "BBB_Martins",
    "bbb": "BBB_Martins",
    "drugbank": "DDI",
    "ddi": "DDI",
    "cyp1a2 inhibition": "CYP1A2_Veith",
    "cyp2c9 inhibition": "CYP2C9_Veith",
    "cyp2c19 inhibition": "CYP2C19_Veith",
    "cyp2d6 inhibition": "CYP2D6_Veith",
    "cyp3a4 inhibition": "CYP3A4_Veith",
    "esol": "Solubility_AqSolDB",
    "freesolv": "HydrationFreeEnergy_FreeSolv",
    "lipophilicity": "Lipophilicity_AstraZeneca",
}


def resolve_model_resource(
    *,
    task_type: str,
    dataset_name: str,
    feature_set: str,
    explicit_model_path: Optional[str] = None,
    auto_model: bool = False,
    model_registry_path: Optional[str | Path] = None,
    resource_root: Path = RESOURCE_ROOT,
) -> Dict[str, Any]:
    """Resolve a real core model for strict AgentM ADMET/DDI execution."""

    task = _normalize_task_type(task_type)
    dataset = normalize_dataset_name(dataset_name, task_type=task)
    candidates = _candidate_models(
        task_type=task,
        dataset_name=dataset,
        feature_set=feature_set,
        explicit_model_path=explicit_model_path,
        auto_model=auto_model,
        model_registry_path=model_registry_path,
        resource_root=resource_root,
    )
    inspected = [
        _inspect_candidate(path, task_type=task, dataset_name=dataset, feature_set=feature_set, priority=idx + 1)
        for idx, path in enumerate(candidates)
    ]
    selected = _select_candidate(inspected)
    for item in inspected:
        item["selected"] = bool(selected and Path(item["path"]) == Path(selected["path"]))
    missing = [item["path"] for item in inspected if not item.get("exists")]
    return {
        "kind": "model",
        "task_type": task,
        "dataset_name": dataset,
        "feature_set": feature_set,
        "auto_model": bool(auto_model),
        "explicit_model_path": str(explicit_model_path) if explicit_model_path else None,
        "model_registry_path": str(model_registry_path) if model_registry_path else None,
        "selected_path": selected.get("path") if selected else None,
        "selected_reason": selected.get("compatibility_status") if selected else "no_compatible_model",
        "schema_compatible": bool(selected and selected.get("schema_compatible")),
        "candidate_paths": [item["path"] for item in inspected],
        "missing_paths": missing,
        "candidates": inspected,
    }


def resolve_rag_resources(
    *,
    task_type: str,
    rag_corpus_dirs: Optional[Iterable[str]] = None,
    ddi_rag_corpus_dir: Optional[str] = None,
    rag_embedding_model_name: Optional[str] = None,
    resource_root: Path = RESOURCE_ROOT,
) -> Dict[str, Any]:
    """Resolve real sibling core RAG resources with absolute paths."""

    admet_dirs = [_absolute_resource_path(item, resource_root=resource_root) for item in (rag_corpus_dirs or [DEFAULT_RAG_CORPUS_DIR])]
    ddi_dir = _absolute_resource_path(ddi_rag_corpus_dir or DEFAULT_DDI_RAG_CORPUS_DIR, resource_root=resource_root)
    embedding = _absolute_resource_path(rag_embedding_model_name or DEFAULT_RAG_EMBEDDING_MODEL_PATH, resource_root=resource_root)
    selected_dirs = [ddi_dir] if _normalize_task_type(task_type) == "ddi_prediction" else admet_dirs
    return {
        "kind": "rag",
        "task_type": _normalize_task_type(task_type),
        "selected_corpus_dirs": [str(path) for path in selected_dirs],
        "admet_corpus_dirs": [_path_status(path) for path in admet_dirs],
        "ddi_corpus_dir": _path_status(ddi_dir),
        "embedding_model_path": _path_status(embedding),
        "all_required_present": all(path.exists() for path in selected_dirs) and embedding.exists(),
    }


def default_resource_path(path: str | Path, *, resource_root: Path = RESOURCE_ROOT) -> str:
    return str(_absolute_resource_path(path, resource_root=resource_root))


def normalize_dataset_name(dataset_name: str, *, task_type: str = "") -> str:
    if _normalize_task_type(task_type) == "ddi_prediction":
        return "DDI" if not dataset_name else DATASET_ALIASES.get(str(dataset_name).strip().lower(), str(dataset_name).strip())
    value = str(dataset_name or "").strip()
    if not value:
        return value
    return DATASET_ALIASES.get(value.lower(), value)


def _candidate_models(
    *,
    task_type: str,
    dataset_name: str,
    feature_set: str,
    explicit_model_path: Optional[str],
    auto_model: bool,
    model_registry_path: Optional[str | Path],
    resource_root: Path,
) -> List[Path]:
    candidates: List[Path] = []
    if explicit_model_path:
        candidates.append(_absolute_resource_path(explicit_model_path, resource_root=resource_root))
    registry = _load_model_registry(model_registry_path, task_type=task_type, dataset_name=dataset_name, feature_set=feature_set)
    candidates.extend(_absolute_resource_path(path, resource_root=resource_root) for path in registry)
    if auto_model:
        if task_type == "ddi_prediction":
            candidates.extend(
                [
                    resource_root / "outputs" / "drugbank_ddi_full_low_token" / f"random_{feature_set}" / "model.pkl",
                    resource_root / "outputs" / "drugbank_ddi_full_low_token" / "random_all_features" / "model.pkl",
                    resource_root / "outputs" / "ddi" / "model.pkl",
                ]
            )
        else:
            candidates.extend(
                [
                    resource_root / "outputs" / "tdc_ablation" / dataset_name / feature_set / "model.pkl",
                    resource_root / "outputs" / "tdc" / dataset_name / "model.pkl",
                    resource_root / "outputs" / "tdc_representative_calibration" / dataset_name / "model.pkl",
                    resource_root / "outputs" / "property_bbbp_real_fast" / "model.pkl",
                ]
            )
    return _dedupe_paths(candidates)


def _inspect_candidate(path: Path, *, task_type: str, dataset_name: str, feature_set: str, priority: int) -> Dict[str, Any]:
    item: Dict[str, Any] = {
        "priority": priority,
        "path": str(path),
        "exists": path.exists(),
        "schema_compatible": False,
        "compatibility_status": "missing",
        "selected": False,
    }
    if not path.exists():
        return item
    try:
        model = SklearnPredictor.load(path)
        metadata = dict(model.metadata or {})
        model_feature_set = metadata.get("feature_set")
        model_task_type = metadata.get("task_type")
        model_dataset = metadata.get("dataset_name")
        feature_schema = model.feature_schema or {}
        feature_names = feature_schema.get("feature_names") if isinstance(feature_schema, dict) else None
        item.update(
            {
                "model_type": model.model_type,
                "mode": model.mode,
                "metadata": to_jsonable(metadata),
                "metadata_task_type": model_task_type,
                "metadata_dataset_name": model_dataset,
                "metadata_feature_set": model_feature_set,
                "feature_dim": len(feature_names) if isinstance(feature_names, list) else None,
            }
        )
        mismatches: List[str] = []
        if model_task_type and not _compatible_task_type(str(model_task_type), task_type, str(model_dataset or ""), dataset_name):
            mismatches.append("task_type")
        if model_dataset and _normalize_dataset_key(str(model_dataset)) != _normalize_dataset_key(dataset_name):
            if not (task_type == "ddi_prediction" and _normalize_dataset_key(str(model_dataset)) in {"ddi", "drugbank"}):
                mismatches.append("dataset_name")
        if model_feature_set and str(model_feature_set) != str(feature_set):
            mismatches.append("feature_set")
        if mismatches:
            item["compatibility_status"] = "_mismatch:".join(["schema"] + mismatches)
            return item
        if not metadata:
            item["compatibility_status"] = "legacy_unknown"
            item["schema_compatible"] = True
            return item
        item["compatibility_status"] = "compatible"
        item["schema_compatible"] = True
        return item
    except Exception as exc:
        item["compatibility_status"] = "load_error"
        item["load_error"] = str(exc)
        return item


def _select_candidate(candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    for status in ["compatible", "legacy_unknown"]:
        for item in candidates:
            if item.get("exists") and item.get("schema_compatible") and item.get("compatibility_status") == status:
                return item
    return {}


def _load_model_registry(
    path: Optional[str | Path],
    *,
    task_type: str,
    dataset_name: str,
    feature_set: str,
) -> List[str]:
    if not path:
        return []
    target = Path(path)
    if not target.exists():
        return []
    payload = json.loads(target.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [str(item) for item in payload]
    if not isinstance(payload, dict):
        return []
    keys = [
        f"{task_type}:{dataset_name}:{feature_set}",
        f"{task_type}:{dataset_name}",
        task_type,
        "default",
    ]
    paths: List[str] = []
    for key in keys:
        value = payload.get(key)
        if isinstance(value, str):
            paths.append(value)
        elif isinstance(value, list):
            paths.extend(str(item) for item in value)
    return paths


def _path_status(path: Path) -> Dict[str, Any]:
    return {"path": str(path), "exists": path.exists(), "is_dir": path.is_dir(), "is_file": path.is_file()}


def _absolute_resource_path(path: str | Path, *, resource_root: Path) -> Path:
    target = Path(path).expanduser()
    if target.is_absolute():
        return target
    return resource_root / target


def _dedupe_paths(paths: Iterable[Path]) -> List[Path]:
    seen = set()
    result: List[Path] = []
    for path in paths:
        key = str(path)
        if key not in seen:
            seen.add(key)
            result.append(path)
    return result


def _normalize_task_type(task_type: str) -> str:
    if str(task_type or "").strip() == "ddi_prediction":
        return "ddi_prediction"
    if str(task_type or "").strip() in {"property_prediction", "admet_prediction"}:
        return "admet_prediction"
    return str(task_type or "admet_prediction").strip() or "admet_prediction"


def _normalize_dataset_key(dataset_name: str) -> str:
    return str(normalize_dataset_name(dataset_name)).strip().lower()


def _compatible_task_type(trained_task: str, requested_task: str, trained_dataset: str, requested_dataset: str) -> bool:
    if trained_task == requested_task:
        return True
    single = {"property_prediction", "admet_prediction"}
    if trained_task in single and requested_task in single:
        return _normalize_dataset_key(trained_dataset) == _normalize_dataset_key(requested_dataset)
    return False
