#!/usr/bin/env python
"""Freeze and materialize the prospective 64 x 4 stress rerun.

This script performs no LLM calls.  It clones the currently accepted evidence
index into a write-blocked experiment index, hashes every runtime/model/data
resource loaded by the stress panel, and writes 256 immutable job files plus
fail-stop lane lists.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import platform
import sys
from typing import Any, Iterable

import requests

try:
    from .common import (
        PROJECT_ROOT,
        canonical_json,
        file_sha256,
        load_config,
        source_tree_hash,
        stable_hash,
        utc_now,
        write_json,
    )
    from .run_matrix import build_jobs
    from .runtime_guards import effective_api_budget
except ImportError:
    from common import (
        PROJECT_ROOT,
        canonical_json,
        file_sha256,
        load_config,
        source_tree_hash,
        stable_hash,
        utc_now,
        write_json,
    )
    from run_matrix import build_jobs
    from runtime_guards import effective_api_budget


PROTOCOL = "agentm_prospective_single_build_stress_64x4"
CONFIG = PROJECT_ROOT / "experiments" / "agentm_reliability" / "configs" / "stress_single_build.yaml"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "agentm_reliability"
ARTIFACT_ROOT = OUTPUT_ROOT / "single_build_stress"
JOB_ROOT = OUTPUT_ROOT / "_single_build_stress_jobs"
FREEZE_PATH = ARTIFACT_ROOT / "freeze_manifest.json"
SOURCE_ES_INDEX = "agentm_evidence_paper_qc_dense"
TARGET_ES_INDEX = "agentm_evidence_stress_single_build_20260813"
ES_URL = "http://127.0.0.1:9200"
EXPECTED_CASES = 64
EXPECTED_STATES = 256


def _request(method: str, path: str, payload: Any | None = None, timeout: int = 300) -> Any:
    response = requests.request(
        method,
        ES_URL.rstrip("/") + "/" + path.lstrip("/"),
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json() if response.content else {}


def _index_exists(name: str) -> bool:
    response = requests.head(ES_URL.rstrip("/") + "/" + name, timeout=30)
    return response.status_code == 200


def _index_identity(name: str) -> dict[str, Any]:
    stats = _request("GET", f"{name}/_stats/docs,store")
    index_stats = stats["indices"][name]
    settings = _request("GET", f"{name}/_settings")[name]["settings"]["index"]
    mapping = _request("GET", f"{name}/_mapping")[name]["mappings"]
    return {
        "name": name,
        "uuid": index_stats["uuid"],
        "document_count": int(index_stats["primaries"]["docs"]["count"]),
        "deleted_document_count": int(index_stats["primaries"]["docs"]["deleted"]),
        "primary_store_size_bytes": int(index_stats["primaries"]["store"]["size_in_bytes"]),
        "write_block": str(settings.get("blocks", {}).get("write", "false")).lower() == "true",
        "mapping_sha256": hashlib.sha256(canonical_json(mapping).encode("utf-8")).hexdigest(),
        "dense_vector_mapped": "dense_vector" in canonical_json(mapping),
    }


def _index_content_hash(name: str) -> str:
    response = _request(
        "POST",
        f"{name}/_search?scroll=2m",
        {"size": 500, "sort": ["_doc"], "_source": True, "query": {"match_all": {}}},
    )
    scroll_id = response.get("_scroll_id")
    rows: list[tuple[str, Any]] = []
    while True:
        hits = (response.get("hits") or {}).get("hits") or []
        if not hits:
            break
        rows.extend((str(hit.get("_id")), hit.get("_source")) for hit in hits)
        response = _request("POST", "_search/scroll", {"scroll": "2m", "scroll_id": scroll_id})
        scroll_id = response.get("_scroll_id") or scroll_id
    if scroll_id:
        try:
            _request("DELETE", "_search/scroll", {"scroll_id": [scroll_id]})
        except Exception:
            pass
    digest = hashlib.sha256()
    for doc_id, source in sorted(rows, key=lambda item: item[0]):
        digest.update(doc_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(canonical_json(source).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _clone_and_lock_index() -> dict[str, Any]:
    source = _index_identity(SOURCE_ES_INDEX)
    if not _index_exists(TARGET_ES_INDEX):
        mapping = _request("GET", f"{SOURCE_ES_INDEX}/_mapping")[SOURCE_ES_INDEX]["mappings"]
        _request(
            "PUT",
            TARGET_ES_INDEX,
            {
                "settings": {"number_of_shards": 1, "number_of_replicas": 0},
                "mappings": mapping,
            },
        )
        _request(
            "POST",
            "_reindex?wait_for_completion=true&refresh=true",
            {"source": {"index": SOURCE_ES_INDEX}, "dest": {"index": TARGET_ES_INDEX, "op_type": "create"}},
            timeout=600,
        )
        _request("PUT", f"{TARGET_ES_INDEX}/_settings", {"index.blocks.write": True})
    target = _index_identity(TARGET_ES_INDEX)
    if target["document_count"] != source["document_count"]:
        raise RuntimeError(f"Evidence-index count mismatch: source={source} target={target}")
    if target["mapping_sha256"] != source["mapping_sha256"]:
        raise RuntimeError("Evidence-index mapping mismatch after clone")
    if not target["write_block"] or not target["dense_vector_mapped"]:
        raise RuntimeError("Frozen evidence index is not write-blocked dense-hybrid")
    target["content_sha256"] = _index_content_hash(TARGET_ES_INDEX)
    source["content_sha256"] = _index_content_hash(SOURCE_ES_INDEX)
    if target["content_sha256"] != source["content_sha256"]:
        raise RuntimeError("Evidence-index content mismatch after clone")
    return {"source": source, "frozen": target}


def _iter_files(path: Path) -> Iterable[Path]:
    if path.is_file():
        yield path
        return
    for item in sorted(path.rglob("*")):
        if item.is_file() and "__pycache__" not in item.parts and not item.name.startswith("."):
            yield item


def _hash_resource(path: Path, *, role: str) -> dict[str, Any]:
    path = path.resolve()
    if not path.exists():
        raise RuntimeError(f"Missing frozen resource: {path}")
    files = list(_iter_files(path))
    digest = hashlib.sha256()
    total = 0
    records = []
    for file_path in files:
        sha = file_sha256(file_path)
        size = file_path.stat().st_size
        rel = file_path.relative_to(path).as_posix() if path.is_dir() else file_path.name
        digest.update(rel.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha.encode("ascii"))
        digest.update(b"\0")
        total += size
        records.append({"relative_path": rel, "size_bytes": size, "sha256": sha})
    return {
        "role": role,
        "path": str(path),
        "kind": "directory" if path.is_dir() else "file",
        "file_count": len(files),
        "size_bytes": total,
        "content_sha256": digest.hexdigest(),
        "files": records,
    }


def _model_resources(config: dict[str, Any], cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    from agentm.runtime.compat import RESOURCE_ROOT, ensure_import_paths
    from agentm.runtime.resources import resolve_model_resource

    ensure_import_paths()
    runtime = config["runtime"]
    selected: dict[str, set[str]] = {}
    for case in cases:
        base = case.get("base_case") or case
        task = str(base.get("task_type") or "")
        if task.startswith("reaction"):
            continue
        key = (task, str(base.get("dataset_name") or ""), str(base.get("feature_set") or runtime["feature_set"]), str(base.get("model_path") or ""))
        resolution = resolve_model_resource(
            task_type=key[0],
            dataset_name=key[1],
            feature_set=key[2],
            explicit_model_path=key[3] or None,
            auto_model=bool(runtime.get("auto_model")),
            resource_root=RESOURCE_ROOT,
        )
        path = str(resolution.get("selected_path") or "")
        if not path or not resolution.get("schema_compatible"):
            raise RuntimeError(f"No compatible frozen model for {key}: {resolution}")
        selected.setdefault(path, set()).add(f"{key[0]}::{key[1]}")

    resources = [
        _hash_resource(Path(path), role="prediction_model:" + ",".join(sorted(roles)))
        for path, roles in sorted(selected.items())
    ]
    resource_root = PROJECT_ROOT
    resources.extend(
        [
            _hash_resource(resource_root / "checkpoints" / "imagemol" / "ImageMol.pth.tar", role="imagemol_checkpoint"),
            _hash_resource(resource_root / "checkpoints" / "chemberta2" / "DeepChem-ChemBERTa-77M-MTR", role="chemberta2_checkpoint"),
            _hash_resource(resource_root / "checkpoints" / "rag" / "scibert_scivocab_uncased", role="dense_retrieval_embedding_checkpoint"),
            _hash_resource(resource_root / "outputs" / "reaction_models_full", role="reaction_prediction_models"),
            _hash_resource(Path("data/reaction_sources/MIT/train.csv"), role="reaction_template_evidence_source"),
            _hash_resource(
                Path("data/reaction_sources/ORDerly/orderly_condition_with_rare_train_dedup_uspto_condition_valtest.parquet"),
                role="reaction_condition_evidence_source",
            ),
        ]
    )
    return resources


def _write_list(name: str, rows: list[dict[str, Any]]) -> str:
    path = ARTIFACT_ROOT / f"{name}.txt"
    path.write_text("\n".join(row["job_path"] for row in rows) + ("\n" if rows else ""), encoding="utf-8")
    return str(path)


def _case_pairs(rows: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    by_case: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_case.setdefault(row["case_id"], []).append(row)
    pairs = []
    for case_id, items in sorted(by_case.items()):
        if len(items) != 2 or {bool(item["reflection_enabled"]) for item in items} != {False, True}:
            raise RuntimeError(f"Expected an on/off pair for {case_id}")
        reverse = int(stable_hash({"case_id": case_id}, length=2), 16) % 2 == 1
        pairs.append(sorted(items, key=lambda item: bool(item["reflection_enabled"]), reverse=reverse))
    return pairs


def _split_pair_lanes(pairs: list[list[dict[str, Any]]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    lanes = [[], []]
    for index, pair in enumerate(pairs):
        lanes[index % 2].extend(pair)
    return lanes[0], lanes[1]


def main() -> int:
    argparse.ArgumentParser(description=__doc__).parse_args()
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    JOB_ROOT.mkdir(parents=True, exist_ok=True)
    evidence = _clone_and_lock_index()
    config = load_config(CONFIG)
    if config["runtime"].get("es_index_name") != TARGET_ES_INDEX:
        raise RuntimeError("Locked config does not name the frozen evidence index")
    jobs = build_jobs(config, selected=set(), max_cases=None, include_repeats=False)
    if len(jobs) != EXPECTED_STATES or len({job["case_id"] for job in jobs}) != EXPECTED_CASES:
        raise RuntimeError(f"Unexpected factorial matrix: {len(jobs)} jobs")
    source_hash = source_tree_hash()
    if {job["source_tree_hash"] for job in jobs} != {source_hash}:
        raise RuntimeError("Job source identity differs from current source snapshot")
    cases = [job["case"] for job in jobs[::4]]
    resources = _model_resources(config, cases)

    from agentm.runtime.compat import ensure_import_paths
    ensure_import_paths()
    from agentm.llm.provider import build_llm_client
    llm = build_llm_client(str(config["runtime"]["llm_profile"]))
    if not llm.available:
        raise RuntimeError("The frozen LLM profile is unavailable")

    queue: list[dict[str, Any]] = []
    for job in jobs:
        job = dict(job)
        job["single_build_protocol"] = PROTOCOL
        job["frozen_resource_manifest"] = str(FREEZE_PATH)
        job["frozen_evidence_index_uuid"] = evidence["frozen"]["uuid"]
        target = JOB_ROOT / f"{job['run_id']}.json"
        raw = OUTPUT_ROOT / "raw_runs" / job["run_id"]
        staging = OUTPUT_ROOT / "_staging" / f"{job['run_id']}.pending"
        if raw.exists() or staging.exists():
            raise RuntimeError(f"Prospective target is not pristine: {job['run_id']}")
        if target.exists():
            existing = json.loads(target.read_text(encoding="utf-8"))
            if existing != job:
                raise RuntimeError(f"Immutable job collision: {target}")
        else:
            write_json(target, job, overwrite=False)
        max_calls, max_tokens = effective_api_budget(job)
        queue.append(
            {
                "case_id": job["case_id"],
                "task_type": job["case"].get("task_type"),
                "fault_type": job["case"].get("injection_type"),
                "orchestrator": job["orchestrator"],
                "reflection_enabled": bool(job["reflection_enabled"]),
                "run_id": job["run_id"],
                "source_tree_hash": job["source_tree_hash"],
                "config_hash": job["config_hash"],
                "job_path": str(target),
                "job_sha256": file_sha256(target),
                "max_llm_calls": int(max_calls),
                "max_total_tokens": int(max_tokens),
            }
        )

    heuristic = [row for row in queue if row["orchestrator"] == "heuristic"]
    llm_rows = [row for row in queue if row["orchestrator"] == "llm"]
    heuristic_lanes = _split_pair_lanes(_case_pairs(heuristic))
    llm_pairs = _case_pairs(llm_rows)
    canary_pair = llm_pairs[0]
    remaining_pairs = llm_pairs[1:]
    wave_pair_counts = [16, 16, 16, 15]
    waves = []
    cursor = 0
    for count in wave_pair_counts:
        waves.append(remaining_pairs[cursor : cursor + count])
        cursor += count
    if cursor != len(remaining_pairs):
        raise RuntimeError("LLM wave split does not cover all remaining pairs")

    lists: dict[str, str] = {
        "heuristic_gpu1": _write_list("heuristic_gpu1", heuristic_lanes[0]),
        "heuristic_gpu4": _write_list("heuristic_gpu4", heuristic_lanes[1]),
        "llm_canary_gpu1": _write_list("llm_canary_gpu1", canary_pair),
        "llm_canary_gpu4": _write_list("llm_canary_gpu4", []),
    }
    for wave_index, wave_pairs in enumerate(waves, start=1):
        lane0, lane1 = _split_pair_lanes(wave_pairs)
        lists[f"llm_wave{wave_index}_gpu1"] = _write_list(f"llm_wave{wave_index}_gpu1", lane0)
        lists[f"llm_wave{wave_index}_gpu4"] = _write_list(f"llm_wave{wave_index}_gpu4", lane1)

    resource_manifest_hash = stable_hash(
        [{key: value for key, value in row.items() if key != "files"} for row in resources]
    )
    payload = {
        "schema": PROTOCOL,
        "generated_at": utc_now(),
        "status": "frozen_preflight_pending_no_llm_calls",
        "code_identity": {
            "kind": "content_addressed_source_snapshot_no_git_metadata",
            "source_tree_hash": source_hash,
            "git_commit": None,
        },
        "config": {
            "path": str(CONFIG),
            "file_sha256": file_sha256(CONFIG),
            "effective_hash": jobs[0]["config_hash"],
            "temperature": config["experiment"]["temperature"],
        },
        "case_manifest": {
            "path": jobs[0]["source_manifest"],
            "sha256": file_sha256(jobs[0]["source_manifest"]),
            "case_count": EXPECTED_CASES,
        },
        "evidence_index": evidence,
        "model_resources": resources,
        "model_resource_manifest_sha256": resource_manifest_hash,
        "external_llm": {
            "profile": llm.config.profile,
            "provider": llm.config.provider,
            "model": llm.config.model,
            "base_url": llm.config.base_url,
            "temperature": config["experiment"]["temperature"],
            "available": llm.available,
            "weight_snapshot_guarantee": "provider_model_name_fixed_but_remote_weights_not_locally_hashable",
        },
        "environment": {"python": sys.version, "platform": platform.platform()},
        "design": {
            "independent_cases": EXPECTED_CASES,
            "settings_per_case": 4,
            "total_runs": EXPECTED_STATES,
            "settings": [
                "heuristic_reflection_off",
                "heuristic_reflection_on",
                "llm_reflection_off",
                "llm_reflection_on",
            ],
            "within_case_reflection_order": "deterministically counterbalanced",
            "seed": config["experiment"]["seed"],
        },
        "queue": queue,
        "job_lists": lists,
        "budgets": {
            "expected_from_historical_mean": {"llm_calls": 1187, "llm_tokens": 4312961},
            "absolute_hard_cap": {
                "llm_calls": sum(row["max_llm_calls"] for row in llm_rows),
                "llm_tokens": sum(row["max_total_tokens"] for row in llm_rows),
            },
            "heuristic_expected": {"llm_calls": 0, "llm_tokens": 0},
        },
        "pristine_target_count": EXPECTED_STATES,
        "execution_authorized": False,
        "usage_at_freeze": {"llm_calls": 0, "llm_tokens": 0},
    }
    write_json(FREEZE_PATH, payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "source_tree_hash": source_hash,
                "config_hash": jobs[0]["config_hash"],
                "frozen_index": evidence["frozen"],
                "jobs": len(queue),
                "heuristic_jobs": len(heuristic),
                "llm_jobs": len(llm_rows),
                "hard_cap": payload["budgets"]["absolute_hard_cap"],
                "freeze_manifest": str(FREEZE_PATH),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
