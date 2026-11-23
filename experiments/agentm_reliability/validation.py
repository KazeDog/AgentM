from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

try:
    from .common import stable_hash
    from .runtime_guards import (
        REACTION_RAG_BACKEND,
        REACTION_RAG_PROVIDER,
        is_reaction_case,
    )
except ImportError:
    from common import stable_hash
    from runtime_guards import REACTION_RAG_BACKEND, REACTION_RAG_PROVIDER, is_reaction_case


REQUIRED_RESULT_FIELDS = (
    "run_id",
    "case_id",
    "dataset_group",
    "task_type",
    "orchestrator",
    "reflection_enabled",
    "seed",
    "started_at",
    "finished_at",
    "latency_seconds",
    "completed",
    "objective_passed",
    "verifier_passed",
    "reflection_step_count",
    "llm_call_count",
    "tool_call_count",
    "retrieval_call_count",
    "unsupported_claim_count",
    "external_service_error_count",
    "global_context_path",
    "final_report_path",
    "trace_path",
    "error_path",
    "code_commit",
    "source_tree_hash",
    "config_hash",
)

RUNTIME_GUARD_PROTOCOL = "pre_llm_real_rag_fail_closed"
REAL_RAG_PROVIDER = "elasticsearch_hybrid"
REAL_RAG_BACKEND = "elasticsearch_hybrid_bm25_dense"
DEPENDENCY_ERROR_PATTERNS = (
    "connection refused",
    "all connection attempts failed",
    "failed to establish a new connection",
    "connection timed out",
    "read timed out",
)


def is_verified_elasticsearch_hybrid_attempt(entry: Mapping[str, Any]) -> bool:
    """Whether the real dense-hybrid path executed without dependency fallback.

    A target-specific query may validly return no evidence. In that case the
    trace must prove that dense retrieval was enabled and completed; BM25-only,
    local, warning-bearing, and dependency-failed attempts remain invalid.
    """

    if str(entry.get("provider") or "") != REAL_RAG_PROVIDER:
        return False
    if str(entry.get("backend") or "") != REAL_RAG_BACKEND:
        return False
    if bool(entry.get("warnings")):
        return False
    if bool(entry.get("retrieved_context")):
        return True
    metadata = entry.get("metadata", {}) or {}
    return bool(metadata.get("dense_enabled")) and str(
        metadata.get("dense_status") or ""
    ) == "completed"


def validate_result_record(record: Mapping[str, Any]) -> None:
    missing = [field for field in REQUIRED_RESULT_FIELDS if field not in record]
    if missing:
        raise ValueError(f"run result missing required field(s): {', '.join(missing)}")
    if not record.get("run_id") or not record.get("case_id"):
        raise ValueError("run_id and case_id must be non-empty")
    if record.get("orchestrator") not in {"heuristic", "llm"}:
        raise ValueError("orchestrator must be heuristic or llm")


def run_invariant_violations(job: Mapping[str, Any], context: Mapping[str, Any]) -> list[str]:
    violations: list[str] = []
    tool_calls = context.get("tool_calls", []) or []
    if not bool(job.get("reflection_enabled")) and any(call.get("agent") == "ReflectionAgent" for call in tool_calls):
        violations.append("reflection_false_called_ReflectionAgent")
    if job.get("dataset_group") == "stress":
        case = job.get("case", {}) or {}
        if int(case.get("primary_fault_count", 0)) != 1:
            violations.append("stress_case_primary_fault_count_not_one")
    if str(job.get("runtime_guard_protocol") or "") == RUNTIME_GUARD_PROTOCOL:
        case = job.get("case", {}) or {}
        runtime_case = case.get("base_case") if isinstance(case.get("base_case"), Mapping) else case
        reaction_mode = is_reaction_case(runtime_case) or is_reaction_case(context)
        metadata = context.get("metadata", {}) or {}
        trace = context.get("retrieval_trace", []) or []
        if isinstance(trace, Mapping):
            trace = [trace]
        entries = [item for item in trace if isinstance(item, Mapping)]
        serialized = json.dumps(entries, ensure_ascii=False).lower()
        if any(pattern in serialized for pattern in DEPENDENCY_ERROR_PATTERNS):
            violations.append("retrieval_trace_dependency_error")
        if reaction_mode:
            reaction_resolution = (context.get("resource_resolution", {}) or {}).get(
                "reaction",
                {},
            ) or context.get("reaction_resource_resolution", {}) or {}
            if not bool(reaction_resolution.get("all_required_present")):
                violations.append("reaction_resources_incomplete")
            qualifying = [
                item
                for item in entries
                if str(item.get("provider") or "") == REACTION_RAG_PROVIDER
                and str(item.get("backend") or "") == REACTION_RAG_BACKEND
                and bool(item.get("evidence_items"))
                and not bool(item.get("warnings"))
            ]
            if not qualifying:
                violations.append("retrieval_trace_missing_successful_reaction_data_result")
        else:
            backend = str(runtime_retrieval_backend(context) or "")
            if backend != REAL_RAG_BACKEND:
                violations.append(f"runtime_retrieval_backend={backend or 'missing'}")
            qualifying = [
                item
                for item in entries
                if is_verified_elasticsearch_hybrid_attempt(item)
            ]
            if not qualifying:
                violations.append("retrieval_trace_missing_verified_elasticsearch_hybrid_attempt")
    return violations


def runtime_retrieval_backend(context: Mapping[str, Any]) -> str | None:
    metadata_backend = str((context.get("metadata", {}) or {}).get("retrieval_backend_used") or "")
    if metadata_backend:
        return metadata_backend
    trace = context.get("retrieval_trace", []) or []
    if isinstance(trace, Mapping):
        trace = [trace]
    for item in trace:
        if not isinstance(item, Mapping):
            continue
        if is_verified_elasticsearch_hybrid_attempt(item):
            return REAL_RAG_BACKEND
        if (
            str(item.get("provider") or "") == REACTION_RAG_PROVIDER
            and str(item.get("backend") or "") == REACTION_RAG_BACKEND
            and bool(item.get("evidence_items"))
        ):
            return REACTION_RAG_BACKEND
    return None


def scientific_resource_signature(config: Mapping[str, Any], case: Mapping[str, Any]) -> str:
    runtime = config.get("runtime", {}) or {}
    paths = config.get("paths", {}) or {}
    base = case.get("base_case") if isinstance(case.get("base_case"), Mapping) else case
    payload = {
        "feature_set": base.get("feature_set") or runtime.get("feature_set"),
        "model_path": base.get("model_path"),
        "auto_model": runtime.get("auto_model"),
        "require_real_model": runtime.get("require_real_model"),
        "require_real_rag": runtime.get("require_real_rag"),
        "retrieval_backend": runtime.get("retrieval_backend"),
        "es_url": runtime.get("es_url"),
        "es_index_name": runtime.get("es_index_name"),
        "rag_embedding_model_name": runtime.get("rag_embedding_model_name"),
        "llm_profile": runtime.get("llm_profile"),
        "min_evidence": runtime.get("min_evidence"),
        "reaction_model_dir": paths.get("reaction_model_dir"),
        "reaction_index_dir": paths.get("reaction_index_dir"),
    }
    return stable_hash(payload)
