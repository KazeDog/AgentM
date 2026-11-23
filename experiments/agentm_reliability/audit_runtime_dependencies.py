#!/usr/bin/env python
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import csv
import json
from pathlib import Path
from typing import Any, Mapping

try:
    from .common import ensure_output_layout, resolve_path, utc_now, write_json
except ImportError:
    from common import ensure_output_layout, resolve_path, utc_now, write_json


DEPENDENCY_ERROR_PATTERNS = (
    "connection refused",
    "all connection attempts failed",
    "failed to establish a new connection",
    "connection timed out",
    "read timed out",
)

CASE_LEVEL_FILES = (
    "clean_case_level.csv",
    "stress_case_level.csv",
    "open_task_case_level.csv",
    "multi_turn_case_level.csv",
)


def dependency_error_in_context(path: str | Path | None) -> bool:
    if not path:
        return False
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(payload, Mapping):
        return False
    retrieval_trace = payload.get("retrieval_trace", []) or []
    serialized = json.dumps(retrieval_trace, ensure_ascii=False).lower()
    return any(pattern in serialized for pattern in DEPENDENCY_ERROR_PATTERNS)


def _case_level_rows(metrics_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name in CASE_LEVEL_FILES:
        path = metrics_dir / name
        if not path.exists():
            raise FileNotFoundError(f"Run the offline evaluator before dependency audit: {path}")
        with path.open(encoding="utf-8", newline="") as handle:
            rows.extend(dict(row) for row in csv.DictReader(handle))
    return rows


def audit_dependencies(output_root: str | Path) -> dict[str, Any]:
    layout = ensure_output_layout(output_root)
    selected = _case_level_rows(layout["metrics"])
    with ThreadPoolExecutor(max_workers=4) as executor:
        flags = list(
            executor.map(
                dependency_error_in_context,
                (item.get("global_context_path") for item in selected),
            )
        )

    affected: list[dict[str, Any]] = []
    for record, failed in zip(selected, flags):
        if not failed:
            continue
        affected.append(
            {
                "run_id": record.get("run_id"),
                "dataset_group": record.get("dataset_group"),
                "case_id": record.get("case_id"),
                "task_type": record.get("task_type"),
                "orchestrator": record.get("orchestrator"),
                "reflection_enabled": str(record.get("reflection_enabled")).lower() == "true",
                "repeat_index": record.get("repeat_index"),
                "started_at": record.get("started_at"),
                "finished_at": record.get("finished_at"),
                "global_context_path": record.get("global_context_path"),
                "stress_protocol": (
                    "compatible_successful_targets"
                    if str(record.get("case_id") or "").startswith("stress_")
                    else None
                ),
            }
        )

    by_group = Counter(str(row["dataset_group"]) for row in affected)
    by_task = Counter(f"{row['dataset_group']}::{row['task_type']}" for row in affected)
    by_setting = Counter(
        f"{row['dataset_group']}::{row['orchestrator']}::reflection-{str(bool(row['reflection_enabled'])).lower()}"
        for row in affected
    )
    report = {
        "generated_at": utc_now(),
        "audit_scope": "offline-evaluator canonical case-level rows",
        "dependency": "local Elasticsearch real-RAG service",
        "error_patterns": list(DEPENDENCY_ERROR_PATTERNS),
        "audited_condition_count": len(selected),
        "affected_condition_count": len(affected),
        "affected_by_group": dict(sorted(by_group.items())),
        "affected_by_task": dict(sorted(by_task.items())),
        "affected_by_setting": dict(sorted(by_setting.items())),
        "affected_conditions": affected,
    }
    write_json(layout["metrics"] / "runtime_dependency_audit.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit canonical runs for an unavailable real-RAG dependency.")
    parser.add_argument("output_root", nargs="?", default="outputs/agentm_reliability")
    args = parser.parse_args()
    report = audit_dependencies(resolve_path(args.output_root))
    print(
        json.dumps(
            {
                key: report[key]
                for key in ["audited_condition_count", "affected_condition_count", "affected_by_group"]
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
