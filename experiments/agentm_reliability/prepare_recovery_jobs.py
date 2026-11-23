#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from .audit_runtime_dependencies import audit_dependencies
    from .canonicalize_runs import EXPECTED_FULL_COUNTS, canonicalize
    from .common import (
        ensure_output_layout,
        load_config,
        read_json,
        resolve_path,
        run_id_for,
        source_tree_hash,
        utc_now,
        write_json,
    )
    from .multi_turn_runner import build_jobs as build_multi_turn_jobs
    from .run_matrix import build_jobs
except ImportError:
    from audit_runtime_dependencies import audit_dependencies
    from canonicalize_runs import EXPECTED_FULL_COUNTS, canonicalize
    from common import (
        ensure_output_layout,
        load_config,
        read_json,
        resolve_path,
        run_id_for,
        source_tree_hash,
        utc_now,
        write_json,
    )
    from multi_turn_runner import build_jobs as build_multi_turn_jobs
    from run_matrix import build_jobs


TIMEOUT_SECONDS = 900
NON_STRESS_GROUPS = ("clean", "open_task", "multi_turn")
EXPECTED_CONTROLLED_STRESS = 256
CANARY_CELLS = (
    {"dataset_group": "clean", "reflection_enabled": False, "task_type": "admet_prediction"},
    {"dataset_group": "clean", "reflection_enabled": True, "task_type": "ddi_prediction"},
    {"dataset_group": "stress", "reflection_enabled": False, "task_type": "reaction_prediction"},
    {"dataset_group": "stress", "reflection_enabled": True, "task_type": "reaction_evidence"},
    {"dataset_group": "open_task", "reflection_enabled": False, "case_id_prefix": "open_multi_endpoint_admet_"},
    {"dataset_group": "open_task", "reflection_enabled": False, "case_id_prefix": "open_ddi_plus_individual_safety_"},
    {"dataset_group": "open_task", "reflection_enabled": False, "case_id_prefix": "open_reaction_multi_tool_"},
    {"dataset_group": "multi_turn", "reflection_enabled": False, "case_id_prefix": "multi_turn_withdraw_and_replace_"},
)


def _estimated_seconds(job: dict[str, Any], original_latency: float | None = None) -> float:
    if original_latency is not None:
        return max(1.0, float(original_latency))
    estimate = 25.0 if job.get("orchestrator") == "heuristic" else 120.0
    if job.get("reflection_enabled"):
        estimate *= 1.2
    return estimate


def _condition_identity(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("dataset_group"),
        row.get("case_id"),
        row.get("orchestrator"),
        bool(row.get("reflection_enabled")),
        int(row.get("repeat_index", 0) or 0),
    )


def _select_canary_entries(
    queued: list[dict[str, Any]],
    *,
    orchestrator: str,
) -> list[dict[str, Any]]:
    """Select deterministic jobs covering orchestration, state, and science paths."""

    selected: list[dict[str, Any]] = []
    for cell in CANARY_CELLS:
        candidates = []
        for entry in queued:
            job = entry["job"]
            case = job.get("case", {}) or {}
            if job.get("dataset_group") != cell["dataset_group"]:
                continue
            if job.get("orchestrator") != orchestrator:
                continue
            if bool(job.get("reflection_enabled")) != bool(cell["reflection_enabled"]):
                continue
            if cell.get("task_type") and str(case.get("task_type") or "") != cell["task_type"]:
                continue
            if cell.get("case_id_prefix") and not str(job.get("case_id") or "").startswith(
                str(cell["case_id_prefix"])
            ):
                continue
            candidates.append(entry)
        if not candidates:
            raise RuntimeError(
                f"Missing queued canary cell for orchestrator={orchestrator}: {cell}"
            )
        candidates.sort(
            key=lambda entry: (
                int(entry["job"].get("repeat_index", 0) or 0) != 0,
                str(entry["job"].get("case_id") or ""),
                str(entry["job"].get("run_id") or ""),
            )
        )
        selected.append(candidates[0])
    return selected


def _build_dependency_retries(
    output_root: Path,
    valid_identities: set[tuple[Any, ...]],
    invalid_by_identity: dict[tuple[Any, ...], dict[str, Any]],
    current_source_hash: str,
    retained_valid_jobs: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    configs = {
        "clean": "experiments/agentm_reliability/configs/clean_matrix.yaml",
        "open_task": "experiments/agentm_reliability/configs/open_tasks.yaml",
        "multi_turn": "experiments/agentm_reliability/configs/multi_turn_tasks.yaml",
    }
    candidates: list[dict[str, Any]] = []
    for group, config_path in configs.items():
        config = load_config(config_path)
        config.setdefault("experiment", {})["timeout_seconds"] = TIMEOUT_SECONDS
        config["experiment"]["max_retries"] = 0
        config.setdefault("_meta", {})["retry_policy"] = "canonical_real_rag_recovery_append_only"
        if group == "multi_turn":
            jobs = build_multi_turn_jobs(config, set(), None, shard_index=0, num_shards=1)
        else:
            jobs = build_jobs(
                config,
                selected=set(),
                max_cases=None,
                include_repeats=group == "clean",
                shard_index=0,
                num_shards=1,
            )
        for job in jobs:
            identity = _condition_identity(job)
            if identity in valid_identities:
                if retained_valid_jobs is not None:
                    job.update(
                        {
                            "source_tree_hash": current_source_hash,
                            "recovery_protocol": "canonical_valid_canary_only",
                        }
                    )
                    job["run_id"] = run_id_for(job, job["config_hash"])
                    retained_valid_jobs.append(job)
                continue
            invalid = invalid_by_identity.get(identity, {})
            previous_run_id = invalid.get("selected_run_id")
            previous_job = (
                read_json(output_root / "raw_runs" / str(previous_run_id) / "job.json", default={}) or {}
                if previous_run_id
                else {}
            )
            retry_attempt = int(previous_job.get("retry_attempt", 0) or 0) + 1
            job.update(
                {
                    "source_tree_hash": current_source_hash,
                    "retry_of_run_id": previous_run_id,
                    "retry_attempt": retry_attempt,
                    "retry_reason": "no canonical real-RAG-valid attempt",
                    "recovery_protocol": "canonical_real_rag_recovery_append_only",
                }
            )
            job["run_id"] = run_id_for(job, job["config_hash"])
            candidates.append(job)
    return candidates


def prepare(output_root: str | Path, stress_config_path: str | Path) -> dict[str, Any]:
    layout = ensure_output_layout(output_root)
    recovery_root = layout["root"] / "recovery_tools"
    job_root = layout["root"] / "_recovery_jobs"
    recovery_root.mkdir(parents=True, exist_ok=True)
    job_root.mkdir(parents=True, exist_ok=True)
    current_source_hash = source_tree_hash()

    audit_path = layout["metrics"] / "runtime_dependency_audit.json"
    audit = read_json(audit_path, default=None)
    if audit is None:
        audit = audit_dependencies(output_root)
    if int(audit.get("audited_condition_count", 0)) != 916:
        raise RuntimeError(f"Unexpected dependency-audit scope in {audit_path}")
    canonical_selected, canonical_manifest = canonicalize(output_root, require_full=False)
    valid_identities = {
        _condition_identity(item["record"])
        for item in canonical_selected
        if bool(item["record"].get("completed"))
    }
    invalid_by_identity = {
        _condition_identity(entry.get("identity", {})): entry
        for entry in canonical_manifest.get("real_rag_invalid_entries", [])
    }
    for item in canonical_selected:
        record = item["record"]
        if not bool(record.get("completed")):
            invalid_by_identity[_condition_identity(record)] = {
                "selected_run_id": record.get("run_id"),
                "identity": {key: record.get(key) for key in ("dataset_group", "case_id", "orchestrator", "reflection_enabled", "repeat_index")},
                "reasons": ["run_not_completed"],
            }
    retained_valid_nonstress_jobs: list[dict[str, Any]] = []
    dependency_jobs = _build_dependency_retries(
        Path(output_root),
        valid_identities,
        invalid_by_identity,
        current_source_hash,
        retained_valid_jobs=retained_valid_nonstress_jobs,
    )
    dependency_counts: dict[str, int] = {}
    valid_counts: dict[str, int] = {}
    for group in NON_STRESS_GROUPS:
        dependency_counts[group] = sum(job.get("dataset_group") == group for job in dependency_jobs)
        valid_counts[group] = sum(identity[0] == group for identity in valid_identities)
        expected = int(EXPECTED_FULL_COUNTS[group])
        if valid_counts[group] + dependency_counts[group] != expected:
            raise RuntimeError(
                f"Canonical recovery scope does not close {group}: valid={valid_counts[group]}, "
                f"queued={dependency_counts[group]}, expected={expected}"
            )

    queued: list[dict[str, Any]] = []
    for job in dependency_jobs:
        queued.append(
            {
                "job": job,
                "queue_kind": "dependency_retry",
                "estimated_seconds": _estimated_seconds(job),
            }
        )

    stress_config = load_config(stress_config_path)
    stress_config.setdefault("experiment", {})["timeout_seconds"] = TIMEOUT_SECONDS
    stress_config["experiment"]["max_retries"] = 0
    stress_config.setdefault("_meta", {})["recovery_policy"] = "primary_stress_controlled"
    all_stress_jobs = build_jobs(
        stress_config,
        selected=set(),
        max_cases=None,
        include_repeats=False,
        shard_index=0,
        num_shards=1,
    )
    if len(all_stress_jobs) != EXPECTED_CONTROLLED_STRESS:
        raise RuntimeError(f"Expected {EXPECTED_CONTROLLED_STRESS} controlled stress jobs, observed {len(all_stress_jobs)}")
    stress_jobs: list[dict[str, Any]] = []
    for job in all_stress_jobs:
        identity = _condition_identity(job)
        if identity in valid_identities:
            job["recovery_protocol"] = "primary_stress_controlled_canary_only"
            continue
        invalid = invalid_by_identity.get(identity, {})
        previous_run_id = invalid.get("selected_run_id")
        previous_job = (
            read_json(Path(output_root) / "raw_runs" / str(previous_run_id) / "job.json", default={}) or {}
            if previous_run_id
            else {}
        )
        job.update(
            {
                "retry_of_run_id": previous_run_id,
                "retry_attempt": int(previous_job.get("retry_attempt", 0) or 0) + 1,
                "retry_reason": "no completed canonical task-appropriate retrieval-valid attempt",
                "recovery_protocol": "primary_stress_controlled_recovery",
            }
        )
        job["run_id"] = run_id_for(job, job["config_hash"])
        stress_jobs.append(job)
        queued.append(
            {
                "job": job,
                "queue_kind": "primary_stress_controlled_retry",
                "estimated_seconds": _estimated_seconds(job),
            }
        )

    dependency_counts["stress"] = len(stress_jobs)
    valid_counts["stress"] = sum(identity[0] == "stress" for identity in valid_identities)
    if valid_counts["stress"] + dependency_counts["stress"] != EXPECTED_CONTROLLED_STRESS:
        raise RuntimeError(
            "Canonical recovery scope does not close stress: "
            f"valid={valid_counts['stress']}, queued={dependency_counts['stress']}, "
            f"expected={EXPECTED_CONTROLLED_STRESS}"
        )

    queued_run_ids = {str(entry["job"]["run_id"]) for entry in queued}
    retained_valid_stress_jobs = [
        job
        for job in all_stress_jobs
        if str(job["run_id"]) not in queued_run_ids
    ]
    canary_only_entries = [
        {
            "job": job,
            "queue_kind": "canary_only_existing_valid_condition",
            "estimated_seconds": _estimated_seconds(job),
        }
        for job in [*retained_valid_nonstress_jobs, *retained_valid_stress_jobs]
    ]
    all_job_entries = [*queued, *canary_only_entries]
    for entry in all_job_entries:
        job = entry["job"]
        path = job_root / f"{job['run_id']}.json"
        existing = read_json(path, default=None)
        if existing is not None and _stable_job_payload(existing) != _stable_job_payload(job):
            raise RuntimeError(f"Immutable recovery-job collision: {path}")
        if existing is None:
            write_json(path, job, overwrite=False)
        entry["job_path"] = str(path)

    shards: list[list[dict[str, Any]]] = [[], []]
    shard_estimates = [0.0, 0.0]
    for entry in sorted(queued, key=lambda row: (-float(row["estimated_seconds"]), row["job"]["run_id"])):
        shard_index = min(range(2), key=lambda index: (shard_estimates[index], index))
        shards[shard_index].append(entry)
        shard_estimates[shard_index] += float(entry["estimated_seconds"])

    for index, shard in enumerate(shards):
        list_path = recovery_root / f"recovery_jobs_shard_{index}.txt"
        list_path.write_text("\n".join(entry["job_path"] for entry in shard) + "\n", encoding="utf-8")

    canary_phases: dict[str, dict[str, Any]] = {}
    for orchestrator in ("heuristic", "llm"):
        phase_entries = _select_canary_entries(all_job_entries, orchestrator=orchestrator)
        phase_shards = [phase_entries[::2], phase_entries[1::2]]
        phase_name = f"{orchestrator}_matrix_canary"
        list_paths: list[str] = []
        for index, shard in enumerate(phase_shards):
            list_path = recovery_root / f"recovery_jobs_{phase_name}_shard_{index}.txt"
            list_path.write_text(
                "\n".join(entry["job_path"] for entry in shard) + "\n",
                encoding="utf-8",
            )
            list_paths.append(str(list_path))
        canary_phases[phase_name] = {
            "orchestrator": orchestrator,
            "expected_llm_usage": "zero" if orchestrator == "heuristic" else "budgeted",
            "job_count": len(phase_entries),
            "shard_sizes": [len(shard) for shard in phase_shards],
            "job_lists": list_paths,
            "entries": [
                {
                    "run_id": entry["job"]["run_id"],
                    "job_path": entry["job_path"],
                    "dataset_group": entry["job"]["dataset_group"],
                    "case_id": entry["job"]["case_id"],
                    "reflection_enabled": entry["job"]["reflection_enabled"],
                }
                for entry in phase_entries
            ],
        }

    manifest = {
        "prepared_at": utc_now(),
        "policy": (
            "append-only replacement of every missing, incomplete, or task-appropriate retrieval-invalid "
            "condition; retain completed valid primary controlled stress conditions"
        ),
        "source_tree_hash": current_source_hash,
        "timeout_seconds": TIMEOUT_SECONDS,
        "dependency_audit_affected_count": audit["affected_condition_count"],
        "canonical_real_rag_valid_counts": valid_counts,
        "canonical_real_rag_invalid_condition_count": canonical_manifest.get("real_rag_invalid_condition_count"),
        "canonical_missing_real_rag_valid_counts": canonical_manifest.get("missing_real_rag_valid_counts"),
        "dependency_retry_counts": dependency_counts,
        "stress_controlled_count": len(stress_jobs),
        "stress_controlled_expected_condition_count": EXPECTED_CONTROLLED_STRESS,
        "canary_only_job_count": len(canary_only_entries),
        "total_jobs": len(queued),
        "shard_sizes": [len(shard) for shard in shards],
        "shard_estimated_seconds": shard_estimates,
        "canary_policy": (
            "two phases covering clean ADMET/DDI, stressed reaction prediction/evidence, "
            "all three open-task families, multi-turn replacement, and both applicable "
            "reflection levels: eight strict zero-LLM heuristic jobs, then eight "
            "token-budgeted LLM jobs"
        ),
        "canary_phases": canary_phases,
        "entries": [
            {
                "run_id": entry["job"]["run_id"],
                "job_path": entry["job_path"],
                "queue_kind": entry["queue_kind"],
                "dataset_group": entry["job"]["dataset_group"],
                "case_id": entry["job"]["case_id"],
                "orchestrator": entry["job"]["orchestrator"],
                "reflection_enabled": entry["job"]["reflection_enabled"],
                "repeat_index": entry["job"].get("repeat_index", 0),
                "estimated_seconds": entry["estimated_seconds"],
            }
            for entry in queued
        ],
    }
    write_json(recovery_root / "recovery_manifest.json", manifest)
    return manifest


def _stable_job_payload(job: dict[str, Any]) -> dict[str, Any]:
    """Ignore config-load timestamps when checking an immutable job collision."""

    payload = json.loads(json.dumps(job, ensure_ascii=False, default=str))
    config_meta = (payload.get("config", {}) or {}).get("_meta", {}) or {}
    config_meta.pop("loaded_at", None)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare append-only recovery jobs after a real-RAG outage.")
    parser.add_argument("output_root", nargs="?", default="outputs/agentm_reliability")
    parser.add_argument(
        "--stress-config",
        default="experiments/agentm_reliability/configs/stress_matrix_controlled.yaml",
    )
    args = parser.parse_args()
    manifest = prepare(resolve_path(args.output_root), resolve_path(args.stress_config))
    print(
        json.dumps(
            {
                key: manifest[key]
                for key in ["total_jobs", "dependency_retry_counts", "stress_controlled_count", "shard_sizes", "source_tree_hash"]
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
