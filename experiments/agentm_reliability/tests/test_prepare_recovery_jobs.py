from __future__ import annotations

from collections import Counter
from pathlib import Path
import tempfile
import unittest

from experiments.agentm_reliability.prepare_recovery_jobs import (
    CANARY_CELLS,
    _build_dependency_retries,
    _select_canary_entries,
    _stable_job_payload,
)


class PrepareRecoveryJobsTests(unittest.TestCase):
    def test_canary_covers_every_allowed_group_reflection_cell(self) -> None:
        queued = []
        for cell_index, cell in enumerate(CANARY_CELLS):
            for orchestrator in ("heuristic", "llm"):
                for repeat_index in (1, 0):
                    case_id = f"{cell.get('case_id_prefix') or 'case-'}{cell_index}-{repeat_index}"
                    queued.append(
                        {
                            "job": {
                                "run_id": f"cell-{cell_index}-{orchestrator}-{repeat_index}",
                                "case_id": case_id,
                                "dataset_group": cell["dataset_group"],
                                "orchestrator": orchestrator,
                                "reflection_enabled": cell["reflection_enabled"],
                                "repeat_index": repeat_index,
                                "case": {"task_type": cell.get("task_type")},
                            },
                            "job_path": f"/cell-{cell_index}-{orchestrator}-{repeat_index}.json",
                        }
                    )
        selected = _select_canary_entries(queued, orchestrator="llm")
        self.assertEqual(len(selected), len(CANARY_CELLS))
        self.assertTrue(all(entry["job"]["orchestrator"] == "llm" for entry in selected))
        self.assertTrue(all(entry["job"]["repeat_index"] == 0 for entry in selected))
        self.assertEqual(len({entry["job"]["run_id"] for entry in selected}), len(CANARY_CELLS))

    def test_empty_valid_set_queues_the_complete_non_stress_matrix(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            jobs = _build_dependency_retries(
                Path(tmp),
                valid_identities=set(),
                invalid_by_identity={},
                current_source_hash="runtime-hash",
            )
        counts = Counter(str(job["dataset_group"]) for job in jobs)
        self.assertEqual(counts, {"clean": 560, "open_task": 60, "multi_turn": 40})
        self.assertTrue(all(job["recovery_protocol"] == "canonical_real_rag_recovery_append_only" for job in jobs))
        self.assertTrue(all(job["retry_reason"] == "no canonical real-RAG-valid attempt" for job in jobs))
        self.assertTrue(all(job["source_tree_hash"] == "runtime-hash" for job in jobs))

    def test_valid_condition_is_retained_for_current_source_canary(self) -> None:
        retained = []
        with tempfile.TemporaryDirectory() as tmp:
            jobs = _build_dependency_retries(
                Path(tmp),
                valid_identities={
                    ("clean", "aspirin_dili", "heuristic", False, 0)
                },
                invalid_by_identity={},
                current_source_hash="runtime-hash",
                retained_valid_jobs=retained,
            )
        self.assertEqual(len(jobs), 659)
        self.assertEqual(len(retained), 1)
        self.assertEqual(retained[0]["recovery_protocol"], "canonical_valid_canary_only")
        self.assertEqual(retained[0]["source_tree_hash"], "runtime-hash")

    def test_job_collision_comparison_ignores_only_load_timestamp(self) -> None:
        first = {"run_id": "run", "config": {"_meta": {"loaded_at": "first", "config_hash": "h"}}}
        second = {"run_id": "run", "config": {"_meta": {"loaded_at": "second", "config_hash": "h"}}}
        self.assertEqual(_stable_job_payload(first), _stable_job_payload(second))


if __name__ == "__main__":
    unittest.main()
