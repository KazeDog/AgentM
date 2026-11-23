#!/usr/bin/env python
"""Independent zero-LLM acceptance for the prospective single-build panel."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import requests

try:
    from .common import PROJECT_ROOT, file_sha256, source_tree_hash, stable_hash, utc_now, write_json
except ImportError:
    from common import PROJECT_ROOT, file_sha256, source_tree_hash, stable_hash, utc_now, write_json


OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "agentm_reliability"
ROOT = OUTPUT_ROOT / "single_build_stress"
FREEZE = ROOT / "freeze_manifest.json"
ACCEPTANCE = ROOT / "preflight_acceptance.json"


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _index_state(name: str) -> dict[str, Any]:
    base = "http://127.0.0.1:9200"
    stats = requests.get(f"{base}/{name}/_stats/docs,store", timeout=60)
    stats.raise_for_status()
    item = stats.json()["indices"][name]
    settings_response = requests.get(f"{base}/{name}/_settings", timeout=60)
    settings_response.raise_for_status()
    settings = settings_response.json()[name]["settings"]["index"]
    mapping_response = requests.get(f"{base}/{name}/_mapping", timeout=60)
    mapping_response.raise_for_status()
    mapping = mapping_response.json()[name]["mappings"]
    return {
        "uuid": item["uuid"],
        "document_count": int(item["primaries"]["docs"]["count"]),
        "write_block": str(settings.get("blocks", {}).get("write", "false")).lower() == "true",
        "dense_vector_mapped": "dense_vector" in json.dumps(mapping, sort_keys=True),
    }


def main() -> int:
    argparse.ArgumentParser(description=__doc__).parse_args()
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    checks: dict[str, bool] = {}
    checks["source_snapshot_unchanged"] = source_tree_hash() == freeze["code_identity"]["source_tree_hash"]
    checks["config_file_unchanged"] = file_sha256(freeze["config"]["path"]) == freeze["config"]["file_sha256"]
    checks["case_manifest_unchanged"] = file_sha256(freeze["case_manifest"]["path"]) == freeze["case_manifest"]["sha256"]
    queue = freeze["queue"]
    checks["exactly_256_unique_jobs"] = len(queue) == len({row["run_id"] for row in queue}) == 256
    identities = Counter((row["case_id"], row["orchestrator"], bool(row["reflection_enabled"])) for row in queue)
    checks["complete_64_by_4_factorial"] = (
        len({row["case_id"] for row in queue}) == 64
        and len(identities) == 256
        and set(identities.values()) == {1}
    )
    source_hashes = {row["source_tree_hash"] for row in queue}
    config_hashes = {row["config_hash"] for row in queue}
    checks["single_source_and_config"] = source_hashes == {freeze["code_identity"]["source_tree_hash"]} and config_hashes == {freeze["config"]["effective_hash"]}
    checks["job_files_immutable"] = all(file_sha256(row["job_path"]) == row["job_sha256"] for row in queue)
    checks["pristine_output_targets"] = all(
        not (OUTPUT_ROOT / "raw_runs" / row["run_id"]).exists()
        and not (OUTPUT_ROOT / "_staging" / f"{row['run_id']}.pending").exists()
        for row in queue
    )
    listed = []
    for path in freeze["job_lists"].values():
        listed.extend(line for line in Path(path).read_text(encoding="utf-8").splitlines() if line)
    checks["job_lists_complete_and_disjoint"] = len(listed) == len(set(listed)) == 256 and set(listed) == {row["job_path"] for row in queue}
    by_orchestrator = Counter(row["orchestrator"] for row in queue)
    checks["factor_counts"] = by_orchestrator == {"heuristic": 128, "llm": 128}
    checks["budgets_fail_closed"] = all(
        (row["max_llm_calls"], row["max_total_tokens"]) == ((0, 0) if row["orchestrator"] == "heuristic" else (32, 120000))
        for row in queue
    )
    frozen = freeze["evidence_index"]["frozen"]
    current_index = _index_state(frozen["name"])
    checks["frozen_index_identity"] = (
        current_index["uuid"] == frozen["uuid"]
        and current_index["document_count"] == frozen["document_count"]
        and current_index["write_block"]
        and current_index["dense_vector_mapped"]
    )
    checks["resource_files_unchanged"] = all(
        Path(item["path"]).exists()
        and sum(row["size_bytes"] for row in item["files"]) == item["size_bytes"]
        and all(
            (Path(item["path"]) / row["relative_path"] if Path(item["path"]).is_dir() else Path(item["path"])).stat().st_size == row["size_bytes"]
            for row in item["files"]
        )
        for item in freeze["model_resources"]
    )
    checks["zero_llm_usage"] = freeze["usage_at_freeze"] == {"llm_calls": 0, "llm_tokens": 0}

    test = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", "experiments/agentm_reliability/tests", "-p", "test_*.py"],
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=300,
        check=False,
    )
    checks["source_tests_passed"] = test.returncode == 0
    for name, passed in checks.items():
        _check(passed, f"preflight check failed: {name}")
    payload = {
        "schema": "agentm_prospective_single_build_stress_preflight",
        "generated_at": utc_now(),
        "passed": True,
        "status": "accepted_zero_llm_preflight_execution_budget_authorization_pending",
        "checks": checks,
        "counts": {"jobs": 256, "cases": 64, "heuristic": 128, "llm": 128},
        "source_tree_hash": freeze["code_identity"]["source_tree_hash"],
        "config_hash": freeze["config"]["effective_hash"],
        "frozen_index": current_index,
        "source_tests": {"returncode": test.returncode, "transcript_tail": test.stdout[-3000:]},
        "usage": {"llm_calls": 0, "llm_tokens": 0},
        "hard_cap": freeze["budgets"]["absolute_hard_cap"],
        "next_gate": "explicit authorization for the 128 paid LLM states under the recorded aggregate hard cap",
    }
    write_json(ACCEPTANCE, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
