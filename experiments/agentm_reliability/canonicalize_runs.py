#!/usr/bin/env python
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping

try:
    from .common import ensure_output_layout, read_json, resolve_path, stable_hash, utc_now, write_csv, write_json
    from .runtime_guards import (
        REACTION_RAG_BACKEND,
        REACTION_RAG_PROVIDER,
        is_reaction_case,
    )
    from .validation import is_verified_elasticsearch_hybrid_attempt
except ImportError:
    from common import ensure_output_layout, read_json, resolve_path, stable_hash, utc_now, write_csv, write_json
    from runtime_guards import REACTION_RAG_BACKEND, REACTION_RAG_PROVIDER, is_reaction_case
    from validation import is_verified_elasticsearch_hybrid_attempt


IDENTITY_FIELDS = (
    "dataset_group",
    "case_id",
    "orchestrator",
    "reflection_enabled",
    "seed",
    "repeat_index",
)

EXPECTED_FULL_COUNTS = {
    "clean": 560,
    "stress": 256,
    "open_task": 60,
    "multi_turn": 40,
}

PRIMARY_STRESS_PROTOCOL = "compatible_successful_targets"
REAL_RAG_BACKEND = "elasticsearch_hybrid_bm25_dense"
REAL_RAG_PROVIDER = "elasticsearch_hybrid"
REAL_RAG_REQUIRED_GROUPS = frozenset(EXPECTED_FULL_COUNTS)
DEPENDENCY_ERROR_PATTERNS = (
    "connection refused",
    "all connection attempts failed",
    "failed to establish a new connection",
    "connection timed out",
    "read timed out",
)


def run_identity(record: Mapping[str, Any]) -> tuple[Any, ...]:
    if "reflection_depth" in record:
        raise ValueError("depth-aware experiment requires paired_depth_rows; legacy boolean aggregation is forbidden")
    return tuple(record.get(field) for field in IDENTITY_FIELDS)


def _timeout_cutoff_seconds(record: Mapping[str, Any]) -> float:
    if record.get("failure_kind") != "timeout":
        return 0.0
    match = re.search(r"(?:exceeded|timeout_sec_exceeded):(\d+(?:\.\d+)?)", str(record.get("error") or ""))
    if match:
        return float(match.group(1))
    value = record.get("timeout_seconds")
    return float(value) if value is not None else 0.0


def _attempt(raw_dir: Path) -> dict[str, Any]:
    record = read_json(raw_dir / "run_record.json", default={}) or {}
    job = read_json(raw_dir / "job.json", default={}) or {}
    retry_attempt = int(job.get("retry_attempt", 0) or 0)
    latency = record.get("latency_seconds")
    known_latency = float(latency) if latency is not None else _timeout_cutoff_seconds(record)
    return {
        "raw_dir": raw_dir,
        "record": record,
        "job": job,
        "retry_attempt": retry_attempt,
        "retry_of_run_id": job.get("retry_of_run_id"),
        "known_latency_seconds": known_latency,
        "timeout_cutoff_seconds": _timeout_cutoff_seconds(record),
    }


def _contains_dependency_error(payload: Any) -> bool:
    serialized = json.dumps(payload, ensure_ascii=False).lower()
    return any(pattern in serialized for pattern in DEPENDENCY_ERROR_PATTERNS)


def _trace_entries(payload: Any) -> list[Mapping[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, Mapping)]
    if not isinstance(payload, Mapping):
        return []
    for key in ("entries", "retrievals", "events"):
        value = payload.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, Mapping)]
    return [payload]


def _retrieval_trace_paths(raw_dir: Path, dataset_group: str) -> list[Path]:
    if dataset_group == "multi_turn":
        return sorted(raw_dir.glob("turns/turn_*/retrieval_trace.json"))
    path = raw_dir / "runtime" / "retrieval_trace.json"
    return [path] if path.exists() else []


def _trace_is_valid(path: Path, *, reaction_mode: bool = False) -> tuple[bool, list[str]]:
    payload = read_json(path, default=None)
    if payload is None:
        return False, ["retrieval_trace_unreadable"]
    if _contains_dependency_error(payload):
        return False, ["retrieval_trace_dependency_error"]
    entries = _trace_entries(payload)
    if reaction_mode:
        qualifying = [
            entry
            for entry in entries
            if str(entry.get("provider") or "") == REACTION_RAG_PROVIDER
            and str(entry.get("backend") or "") == REACTION_RAG_BACKEND
            and bool(entry.get("evidence_items"))
            and not bool(entry.get("warnings"))
        ]
        if not qualifying:
            return False, ["retrieval_trace_missing_successful_reaction_data_result"]
    else:
        qualifying = [
            entry
            for entry in entries
            if is_verified_elasticsearch_hybrid_attempt(entry)
        ]
        if not qualifying:
            return False, ["retrieval_trace_missing_verified_elasticsearch_hybrid_attempt"]
    return True, []


def _attempt_is_reaction(attempt: Mapping[str, Any]) -> bool:
    job = attempt.get("job", {}) or {}
    case = job.get("case", {}) or {}
    if is_reaction_case(case):
        return True
    return str((attempt.get("record", {}) or {}).get("task_type") or "").lower().startswith(
        "reaction"
    )


def _reaction_resources_valid(raw_dir: Path) -> tuple[bool, list[str]]:
    runtime_dir = raw_dir / "runtime"
    resolution = read_json(runtime_dir / "reaction_resource_resolution.json", default={}) or {}
    if not resolution:
        resources = read_json(runtime_dir / "resource_resolution.json", default={}) or {}
        resolution = (resources.get("reaction", {}) or {}) if isinstance(resources, Mapping) else {}
    if not resolution:
        context = read_json(runtime_dir / "global_context.json", default={}) or {}
        resolution = (context.get("resource_resolution", {}) or {}).get(
            "reaction",
            {},
        ) or context.get("reaction_resource_resolution", {}) or {}
    if not bool(resolution.get("all_required_present")):
        return False, ["reaction_resources_incomplete"]
    return True, []


def _real_rag_validity(
    attempt: Mapping[str, Any],
    *,
    affected_run_ids: frozenset[str],
) -> dict[str, Any]:
    record = attempt["record"]
    raw_dir = attempt["raw_dir"]
    dataset_group = str(record.get("dataset_group") or "")
    reaction_mode = _attempt_is_reaction(attempt)
    if dataset_group not in REAL_RAG_REQUIRED_GROUPS:
        return {"valid": True, "reasons": [], "trace_count": 0}

    reasons: list[str] = []
    run_id = str(record.get("run_id") or raw_dir.name)
    if run_id in affected_run_ids:
        reasons.append("dependency_audit_flagged")
    guard_protocol = str(attempt.get("job", {}).get("runtime_guard_protocol") or "")
    if guard_protocol == "pre_llm_real_rag_fail_closed":
        if not bool(record.get("runtime_preflight_passed")):
            reasons.append("runtime_preflight_not_passed")
        if bool(record.get("llm_budget_exceeded")):
            reasons.append("llm_budget_exceeded")
        if str(record.get("orchestrator") or "") == "heuristic" and int(
            record.get("llm_call_count", 0) or 0
        ) != 0:
            reasons.append("heuristic_orchestrator_used_llm")
        runtime = (attempt.get("job", {}).get("config", {}) or {}).get("runtime", {}) or {}
        max_calls = runtime.get("max_llm_calls_per_run")
        max_tokens = runtime.get("max_total_tokens_per_run")
        if max_calls is not None and int(record.get("llm_call_count", 0) or 0) > int(max_calls):
            reasons.append("llm_call_budget_exceeded")
        if max_tokens is not None and int(record.get("total_tokens", 0) or 0) > int(max_tokens):
            reasons.append("llm_token_budget_exceeded")
    backend = str(record.get("runtime_retrieval_backend") or "")
    if reaction_mode:
        resources_valid, resource_reasons = _reaction_resources_valid(raw_dir)
        if not resources_valid:
            reasons.extend(resource_reasons)
        if backend and backend != REACTION_RAG_BACKEND:
            reasons.append(f"runtime_retrieval_backend={backend}")
    elif backend != REAL_RAG_BACKEND:
        reasons.append(f"runtime_retrieval_backend={backend or 'missing'}")
    retrieval_calls = int(record.get("retrieval_call_count", 0) or 0)
    if retrieval_calls < 1:
        reasons.append("retrieval_call_count<1")

    trace_paths = _retrieval_trace_paths(raw_dir, dataset_group)
    if dataset_group == "multi_turn":
        expected_turn_count = int(
            record.get("expected_turn_count", 0) or record.get("turn_count", 0) or 0
        )
        if expected_turn_count < 1:
            reasons.append("expected_turn_count<1")
        else:
            if len(trace_paths) != expected_turn_count:
                reasons.append(
                    f"retrieval_trace_count={len(trace_paths)}!=expected_turn_count={expected_turn_count}"
                )
            if retrieval_calls < expected_turn_count:
                reasons.append(
                    f"retrieval_call_count={retrieval_calls}<expected_turn_count={expected_turn_count}"
                )
    if not trace_paths:
        reasons.append("retrieval_trace_missing")
    else:
        for trace_path in trace_paths:
            valid, trace_reasons = _trace_is_valid(
                trace_path,
                reaction_mode=reaction_mode,
            )
            if not valid:
                trace_label = trace_path.parent.name if dataset_group == "multi_turn" else "runtime"
                reasons.extend(f"{trace_label}:{reason}" for reason in trace_reasons)
    return {
        "valid": not reasons,
        "reasons": sorted(set(reasons)),
        "trace_count": len(trace_paths),
    }


def select_canonical_attempts(
    raw_dirs: Iterable[Path],
    *,
    affected_run_ids: Iterable[str] = (),
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    affected_ids = frozenset(str(value) for value in affected_run_ids)
    raw_paths = list(raw_dirs)
    with ThreadPoolExecutor(max_workers=8) as executor:
        loaded = list(executor.map(_attempt, raw_paths))
        validity = list(
            executor.map(
                lambda item: _real_rag_validity(item, affected_run_ids=affected_ids),
                loaded,
            )
        )
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for attempt, real_rag in zip(loaded, validity):
        record = attempt["record"]
        if not record:
            continue
        attempt["real_rag"] = real_rag
        grouped[run_identity(record)].append(attempt)

    selected: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    for identity, attempts in sorted(grouped.items(), key=lambda item: tuple(str(value) for value in item[0])):
        # Recovery jobs are append-only and may explicitly supersede a prior
        # scientifically invalid result.  Honour that lineage before comparing
        # legacy retry counters: counters from separately prepared recovery
        # waves are not guaranteed to be globally monotonic.
        successfully_superseded_run_ids: set[str] = set()
        for item in attempts:
            if not bool(item["real_rag"]["valid"]) or not bool(item["record"].get("completed")):
                continue
            for field in ("supersedes_run_id", "retry_of_run_id", "parent_run_id"):
                predecessor = item["job"].get(field)
                if predecessor:
                    successfully_superseded_run_ids.add(str(predecessor))
        for item in attempts:
            item["explicitly_superseded"] = (
                str(item["record"].get("run_id") or "") in successfully_superseded_run_ids
            )
        attempts.sort(
            key=lambda item: (
                not bool(item["explicitly_superseded"]),
                bool(item["real_rag"]["valid"]),
                bool(item["record"].get("completed")),
                item["retry_attempt"],
                str(item["record"].get("finished_at") or ""),
                str(item["record"].get("run_id") or ""),
            ),
            reverse=True,
        )
        chosen = attempts[0]
        initial = min(
            attempts,
            key=lambda item: (
                item["retry_attempt"],
                str(item["record"].get("started_at") or ""),
                str(item["record"].get("run_id") or ""),
            ),
        )
        record = dict(chosen["record"])
        record.update(
            {
                "canonical_selected": True,
                "canonical_selected_via_supersession": bool(chosen["job"].get("supersedes_run_id")),
                "canonical_supersedes_run_id": chosen["job"].get("supersedes_run_id"),
                "canonical_attempt_count": len(attempts),
                "canonical_retry_count": max(0, len(attempts) - 1),
                "canonical_selected_from_retry": chosen["retry_attempt"] > 0,
                "canonical_retry_success": bool(chosen["record"].get("completed")) and not bool(initial["record"].get("completed")),
                "first_attempt_completed": bool(initial["record"].get("completed")),
                "first_attempt_run_id": initial["record"].get("run_id"),
                "first_attempt_failure_kind": initial["record"].get("failure_kind"),
                "selected_attempt_run_id": chosen["record"].get("run_id"),
                "operational_latency_seconds": sum(item["known_latency_seconds"] for item in attempts),
                "timeout_overhead_seconds": sum(item["timeout_cutoff_seconds"] for item in attempts),
                "token_usage_complete_across_attempts": all(
                    item["record"].get("failure_kind") != "timeout" for item in attempts
                ),
                "real_rag_valid": bool(chosen["real_rag"]["valid"]),
                "real_rag_invalid_reasons": list(chosen["real_rag"]["reasons"]),
                "real_rag_trace_count": int(chosen["real_rag"]["trace_count"]),
                "dependency_audit_flagged": str(chosen["record"].get("run_id")) in affected_ids,
            }
        )
        selected.append({"raw_dir": chosen["raw_dir"], "record": record, "job": chosen["job"]})
        audit_rows.append(
            {
                **dict(zip(IDENTITY_FIELDS, identity)),
                "selected_run_id": chosen["record"].get("run_id"),
                "selected_supersedes_run_id": chosen["job"].get("supersedes_run_id"),
                "selected_via_supersession": bool(chosen["job"].get("supersedes_run_id")),
                "first_attempt_run_id": initial["record"].get("run_id"),
                "first_attempt_completed": bool(initial["record"].get("completed")),
                "eventual_completed": bool(chosen["record"].get("completed")),
                "attempt_count": len(attempts),
                "retry_count": max(0, len(attempts) - 1),
                "selected_from_retry": chosen["retry_attempt"] > 0,
                "retry_success": bool(record["canonical_retry_success"]),
                "operational_latency_seconds": record["operational_latency_seconds"],
                "timeout_overhead_seconds": record["timeout_overhead_seconds"],
                "real_rag_valid": record["real_rag_valid"],
                "real_rag_invalid_reasons": ";".join(record["real_rag_invalid_reasons"]),
                "real_rag_invalid_attempt_count": sum(
                    not bool(item["real_rag"]["valid"]) for item in attempts
                ),
                "superseded_run_ids": ";".join(
                    str(item["record"].get("run_id")) for item in attempts if item is not chosen
                ),
            }
        )
    return selected, audit_rows


def canonicalize(
    output_root: str | Path,
    *,
    require_full: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    layout = ensure_output_layout(output_root)
    raw_dirs = sorted(path.parent for path in layout["raw_runs"].glob("*/run_record.json"))
    dependency_audit_path = layout["metrics"] / "runtime_dependency_audit.json"
    dependency_audit = read_json(dependency_audit_path, default={}) or {}
    affected_run_ids = {
        str(row.get("run_id"))
        for row in dependency_audit.get("affected_conditions", [])
        if row.get("run_id")
    }
    audited_selected, audit_rows = select_canonical_attempts(
        raw_dirs,
        affected_run_ids=affected_run_ids,
    )
    selected: list[dict[str, Any]] = []
    protocol_excluded: list[dict[str, Any]] = []
    real_rag_invalid: list[dict[str, Any]] = []
    audit_by_run_id = {str(row.get("selected_run_id")): row for row in audit_rows}
    for item in audited_selected:
        record = item["record"]
        case = item.get("job", {}).get("case", {})
        stress_protocol = case.get("stress_protocol") if isinstance(case, Mapping) else None
        included = record.get("dataset_group") != "stress" or stress_protocol == PRIMARY_STRESS_PROTOCOL
        audit = audit_by_run_id.get(str(record.get("run_id")))
        if audit is not None:
            audit["stress_protocol"] = stress_protocol
            audit["primary_protocol_included"] = included
            audit["protocol_exclusion_reason"] = (
                "superseded stress-core pilot protocol" if not included else ""
            )
        if not included:
            protocol_excluded.append(item)
        elif bool(record.get("real_rag_valid")):
            selected.append(item)
        else:
            real_rag_invalid.append(item)
    counts: dict[str, int] = defaultdict(int)
    completed_counts: dict[str, int] = defaultdict(int)
    first_attempt_counts: dict[str, int] = defaultdict(int)
    for item in selected:
        record = item["record"]
        group = str(record.get("dataset_group"))
        counts[group] += 1
        completed_counts[group] += int(bool(record.get("completed")))
        first_attempt_counts[group] += int(bool(record.get("first_attempt_completed")))

    if require_full:
        if dict(counts) != EXPECTED_FULL_COUNTS:
            missing = {
                group: max(0, expected - int(counts.get(group, 0)))
                for group, expected in EXPECTED_FULL_COUNTS.items()
            }
            raise RuntimeError(
                "Canonical real-RAG-valid full-matrix counts differ from protocol: "
                f"observed={dict(counts)}, missing={missing}, "
                f"real_rag_invalid={len(real_rag_invalid)}"
            )
        incomplete = [item["record"].get("run_id") for item in selected if not item["record"].get("completed")]
        if incomplete:
            raise RuntimeError(f"Canonical matrix still contains {len(incomplete)} incomplete conditions")

    entries = []
    for item in selected:
        record = item["record"]
        entries.append(
            {
                "identity": {field: record.get(field) for field in IDENTITY_FIELDS},
                "selected_run_id": record.get("run_id"),
                "selected_raw_dir": str(item["raw_dir"]),
                "selected_from_retry": record.get("canonical_selected_from_retry"),
                "first_attempt_completed": record.get("first_attempt_completed"),
                "attempt_count": record.get("canonical_attempt_count"),
                "operational_latency_seconds": record.get("operational_latency_seconds"),
            }
        )
    protocol_excluded_entries = [
        {
            "identity": {field: item["record"].get(field) for field in IDENTITY_FIELDS},
            "selected_run_id": item["record"].get("run_id"),
            "stress_protocol": item.get("job", {}).get("case", {}).get("stress_protocol"),
            "reason": "superseded stress-core pilot protocol",
        }
        for item in protocol_excluded
    ]
    real_rag_invalid_entries = [
        {
            "identity": {field: item["record"].get(field) for field in IDENTITY_FIELDS},
            "selected_run_id": item["record"].get("run_id"),
            "runtime_retrieval_backend": item["record"].get("runtime_retrieval_backend"),
            "retrieval_call_count": item["record"].get("retrieval_call_count"),
            "dependency_audit_flagged": item["record"].get("dependency_audit_flagged"),
            "reasons": item["record"].get("real_rag_invalid_reasons", []),
        }
        for item in real_rag_invalid
    ]
    missing_valid_counts = {
        group: max(0, expected - int(counts.get(group, 0)))
        for group, expected in EXPECTED_FULL_COUNTS.items()
    }
    manifest = {
        "generated_at": utc_now(),
        "selection_rule": (
            "one condition per case/setting/seed/repeat; prefer an attempt with verified task-appropriate "
            "retrieval (Elasticsearch dense hybrid for molecular tasks; MIT/ORDerly reaction data for "
            "reaction tasks), then a completed attempt, then the latest retry; retain but "
            "exclude dependency-invalid attempts and the superseded stress-core pilot protocol; never delete raw attempts"
        ),
        "real_rag_policy": {
            "molecular_required_backend": REAL_RAG_BACKEND,
            "molecular_required_provider": REAL_RAG_PROVIDER,
            "reaction_required_backend": REACTION_RAG_BACKEND,
            "reaction_required_provider": REACTION_RAG_PROVIDER,
            "reaction_requires_complete_resources": True,
            "minimum_retrieval_calls": 1,
            "require_verified_task_appropriate_retrieval_attempt": True,
            "allow_empty_task_specific_result_after_nonempty_preflight": True,
            "direct_evidence_is_reported_separately": True,
            "guarded_recovery_protocol": "pre_llm_real_rag_fail_closed",
            "guarded_recovery_requires_preflight": True,
            "guarded_recovery_requires_budget_compliance": True,
            "reject_dependency_warning_patterns": list(DEPENDENCY_ERROR_PATTERNS),
            "multi_turn_policy": (
                "retrieval-trace count must equal expected_turn_count and every turn trace must contain "
                "a valid real-RAG result"
            ),
        },
        "dependency_audit": {
            "path": str(dependency_audit_path),
            "available": dependency_audit_path.exists(),
            "affected_run_id_count": len(affected_run_ids),
            "reported_affected_condition_count": dependency_audit.get("affected_condition_count"),
        },
        "identity_fields": list(IDENTITY_FIELDS),
        "primary_stress_protocol": PRIMARY_STRESS_PROTOCOL,
        "raw_attempt_count": len(raw_dirs),
        "audited_condition_count": len(audited_selected),
        "canonical_condition_count": len(selected),
        "protocol_excluded_condition_count": len(protocol_excluded),
        "real_rag_invalid_condition_count": len(real_rag_invalid),
        "counts": dict(counts),
        "missing_real_rag_valid_counts": missing_valid_counts,
        "eventual_completed_counts": dict(completed_counts),
        "first_attempt_completed_counts": dict(first_attempt_counts),
        "retry_selected_count": sum(int(bool(item["record"].get("canonical_selected_from_retry"))) for item in selected),
        "manifest_hash": stable_hash(entries),
        "entries": entries,
        "protocol_excluded_entries": protocol_excluded_entries,
        "real_rag_invalid_entries": real_rag_invalid_entries,
    }
    write_json(layout["metrics"] / "canonical_run_manifest.json", manifest)
    write_csv(layout["metrics"] / "canonical_run_selection.csv", audit_rows)
    return selected, manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Select one append-only canonical attempt for each AgentM reliability condition.")
    parser.add_argument("output_root", nargs="?", default="outputs/agentm_reliability")
    parser.add_argument("--require-full", action="store_true")
    args = parser.parse_args()
    _, manifest = canonicalize(resolve_path(args.output_root), require_full=args.require_full)
    print(
        json.dumps(
            {
                key: manifest[key]
                for key in [
                    "raw_attempt_count",
                    "canonical_condition_count",
                    "real_rag_invalid_condition_count",
                    "counts",
                    "missing_real_rag_valid_counts",
                    "retry_selected_count",
                    "manifest_hash",
                ]
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
