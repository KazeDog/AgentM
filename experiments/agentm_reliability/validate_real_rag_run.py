#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from .canonicalize_runs import _attempt, _real_rag_validity
    from .common import read_json
except ImportError:
    from canonicalize_runs import _attempt, _real_rag_validity
    from common import read_json


def validate_raw_run(raw_dir: str | Path, *, dependency_audit_path: str | Path | None = None) -> dict[str, Any]:
    raw = Path(raw_dir)
    attempt = _attempt(raw)
    record = attempt["record"]
    job = attempt["job"]
    affected_run_ids: set[str] = set()
    if dependency_audit_path:
        audit = read_json(dependency_audit_path, default={}) or {}
        affected_run_ids = {
            str(row.get("run_id"))
            for row in audit.get("affected_conditions", [])
            if row.get("run_id")
        }
    real_rag = _real_rag_validity(attempt, affected_run_ids=frozenset(affected_run_ids))
    reasons = list(real_rag["reasons"])
    if not record:
        reasons.append("run_record_missing")
    if not bool(record.get("completed")):
        reasons.append("run_not_completed")
    runtime = (job.get("config", {}) or {}).get("runtime", {}) or {}
    if bool(runtime.get("require_real_rag", True)) and not bool(record.get("runtime_preflight_passed")):
        reasons.append("runtime_preflight_not_passed")
    if bool(record.get("llm_budget_exceeded")):
        reasons.append("llm_budget_exceeded")
    if str(record.get("orchestrator") or "") == "heuristic" and int(
        record.get("llm_call_count", 0) or 0
    ) != 0:
        reasons.append("heuristic_orchestrator_used_llm")
    max_calls = runtime.get("max_llm_calls_per_run")
    max_tokens = runtime.get("max_total_tokens_per_run")
    if max_calls is not None and int(record.get("llm_call_count", 0) or 0) > int(max_calls):
        reasons.append("llm_call_budget_exceeded")
    if max_tokens is not None and int(record.get("total_tokens", 0) or 0) > int(max_tokens):
        reasons.append("llm_token_budget_exceeded")
    return {
        "run_id": record.get("run_id") or raw.name,
        "raw_dir": str(raw),
        "valid": not reasons,
        "reasons": sorted(set(reasons)),
        "completed": bool(record.get("completed")),
        "runtime_preflight_passed": bool(record.get("runtime_preflight_passed")),
        "runtime_retrieval_backend": record.get("runtime_retrieval_backend"),
        "retrieval_call_count": int(record.get("retrieval_call_count", 0) or 0),
        "real_rag_trace_count": int(real_rag.get("trace_count", 0) or 0),
        "llm_call_count": int(record.get("llm_call_count", 0) or 0),
        "total_tokens": int(record.get("total_tokens", 0) or 0),
        "max_llm_calls_per_run": max_calls,
        "max_total_tokens_per_run": max_tokens,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Fail-closed validation for one completed recovery raw run.")
    parser.add_argument("raw_dir")
    parser.add_argument("--dependency-audit", default=None)
    args = parser.parse_args()
    result = validate_raw_run(args.raw_dir, dependency_audit_path=args.dependency_audit)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
