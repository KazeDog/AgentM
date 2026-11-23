from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from experiments.agentm_reliability.canonicalize_runs import canonicalize, select_canonical_attempts
from experiments.agentm_reliability.common import write_json


class CanonicalizeRunsTests(unittest.TestCase):
    def _attempt(
        self,
        root: Path,
        run_id: str,
        *,
        completed: bool,
        retry_attempt: int = 0,
        real_rag: bool = True,
        dependency_warning: bool = False,
        supersedes_run_id: str | None = None,
        retry_of_run_id: str | None = None,
    ) -> Path:
        raw = root / run_id
        raw.mkdir()
        record = {
            "run_id": run_id,
            "case_id": "case-1",
            "dataset_group": "clean",
            "orchestrator": "llm",
            "reflection_enabled": False,
            "seed": 2026,
            "repeat_index": 0,
            "completed": completed,
            "failure_kind": None if completed else "timeout",
            "error": None if completed else "case_timeout_sec_exceeded:300",
            "latency_seconds": 120.0 if completed else None,
            "started_at": "2026-01-01T00:00:00Z",
            "finished_at": "2026-01-01T00:02:00Z",
            "runtime_retrieval_backend": (
                "elasticsearch_hybrid_bm25_dense" if real_rag else "dense_faiss"
            ),
            "retrieval_call_count": 1,
        }
        job = {
            "retry_attempt": retry_attempt,
            "retry_of_run_id": retry_of_run_id if retry_of_run_id is not None else ("initial" if retry_attempt else None),
            "supersedes_run_id": supersedes_run_id,
        }
        write_json(raw / "run_record.json", record, overwrite=False)
        write_json(raw / "job.json", job, overwrite=False)
        write_json(
            raw / "runtime" / "retrieval_trace.json",
            [
                {
                    "provider": "elasticsearch_hybrid" if real_rag else "hybrid_retriever",
                    "backend": (
                        "elasticsearch_hybrid_bm25_dense" if real_rag else "elasticsearch"
                    ),
                    "retrieved_context": [{"chunk_id": "chunk-1"}] if real_rag else [],
                    "warnings": ["Connection refused"] if dependency_warning else [],
                }
            ],
            overwrite=False,
        )
        return raw

    def test_completed_retry_is_selected_without_deleting_timeout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            initial = self._attempt(root, "initial", completed=False)
            retry = self._attempt(root, "retry", completed=True, retry_attempt=1)
            selected, audit = select_canonical_attempts([initial, retry])
            self.assertEqual(len(selected), 1)
            self.assertEqual(selected[0]["record"]["run_id"], "retry")
            self.assertFalse(selected[0]["record"]["first_attempt_completed"])
            self.assertEqual(selected[0]["record"]["operational_latency_seconds"], 420.0)
            self.assertTrue((initial / "run_record.json").exists())
            self.assertEqual(audit[0]["superseded_run_ids"], "initial")

    def test_real_rag_valid_retry_beats_completed_dependency_invalid_attempt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._attempt(root, "initial", completed=True, real_rag=False)
            self._attempt(root, "recovery", completed=True, retry_attempt=2, real_rag=True)
            selected, audit = select_canonical_attempts([root / "initial", root / "recovery"])
            self.assertEqual(selected[0]["record"]["run_id"], "recovery")
            self.assertTrue(selected[0]["record"]["real_rag_valid"])
            self.assertEqual(audit[0]["real_rag_invalid_attempt_count"], 1)

    def test_valid_explicit_supersession_beats_nonmonotonic_legacy_retry_counter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old = self._attempt(root, "old", completed=True, retry_attempt=5)
            new = self._attempt(
                root,
                "new",
                completed=True,
                retry_attempt=1,
                supersedes_run_id="old",
            )

            selected, audit = select_canonical_attempts([old, new])

            self.assertEqual(selected[0]["record"]["run_id"], "new")
            self.assertTrue(selected[0]["record"]["canonical_selected_via_supersession"])
            self.assertEqual(selected[0]["record"]["canonical_supersedes_run_id"], "old")
            self.assertEqual(audit[0]["selected_supersedes_run_id"], "old")

    def test_valid_retry_lineage_removes_older_branch_with_higher_counter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old = self._attempt(root, "old", completed=True, retry_attempt=2)
            middle = self._attempt(
                root,
                "middle",
                completed=True,
                retry_attempt=3,
                retry_of_run_id="old",
            )
            final = self._attempt(
                root,
                "final",
                completed=True,
                retry_attempt=1,
                retry_of_run_id="middle",
                supersedes_run_id="middle",
            )

            selected, _ = select_canonical_attempts([old, middle, final])

            self.assertEqual(selected[0]["record"]["run_id"], "final")

    def test_dependency_warning_invalidates_apparently_successful_trace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            attempt = self._attempt(
                root,
                "warning",
                completed=True,
                real_rag=True,
                dependency_warning=True,
            )
            selected, _ = select_canonical_attempts([attempt])
            record = selected[0]["record"]
            self.assertFalse(record["real_rag_valid"])
            self.assertIn(
                "runtime:retrieval_trace_dependency_error",
                record["real_rag_invalid_reasons"],
            )

    def test_reaction_attempt_uses_reaction_trace_and_resource_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "reaction"
            raw.mkdir()
            write_json(
                raw / "run_record.json",
                {
                    "run_id": "reaction",
                    "case_id": "reaction-case",
                    "dataset_group": "stress",
                    "task_type": "reaction_yield_prediction",
                    "orchestrator": "heuristic",
                    "reflection_enabled": False,
                    "seed": 2026,
                    "repeat_index": 0,
                    "completed": True,
                    "runtime_retrieval_backend": None,
                    "retrieval_call_count": 2,
                },
                overwrite=False,
            )
            write_json(
                raw / "job.json",
                {
                    "case": {
                        "task_type": "reaction_prediction",
                        "base_case": {"task_type": "reaction_yield_prediction"},
                    }
                },
                overwrite=False,
            )
            write_json(
                raw / "runtime" / "reaction_resource_resolution.json",
                {"all_required_present": True},
                overwrite=False,
            )
            write_json(
                raw / "runtime" / "retrieval_trace.json",
                [
                    {
                        "provider": "RetrieveAgent",
                        "backend": "local_reaction_data",
                        "evidence_items": [{"citation_id": "mit-1"}],
                        "warnings": [],
                    }
                ],
                overwrite=False,
            )
            selected, _ = select_canonical_attempts([raw])
            self.assertTrue(
                selected[0]["record"]["real_rag_valid"],
                selected[0]["record"]["real_rag_invalid_reasons"],
            )

    def test_guarded_attempt_without_preflight_is_not_real_rag_valid(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            attempt = self._attempt(root, "guarded", completed=True, real_rag=True)
            write_json(
                attempt / "job.json",
                {
                    "runtime_guard_protocol": "pre_llm_real_rag_fail_closed",
                    "config": {
                        "runtime": {
                            "max_llm_calls_per_run": 5,
                            "max_total_tokens_per_run": 10000,
                        }
                    },
                },
                overwrite=True,
            )
            selected, _ = select_canonical_attempts([attempt])
            record = selected[0]["record"]
            self.assertFalse(record["real_rag_valid"])
            self.assertIn("runtime_preflight_not_passed", record["real_rag_invalid_reasons"])

    def test_audit_flagged_run_is_excluded_from_canonical_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp) / "output"
            raw_root = output_root / "raw_runs"
            raw_root.mkdir(parents=True)
            self._attempt(raw_root, "flagged", completed=True, real_rag=True)
            write_json(
                output_root / "metrics" / "runtime_dependency_audit.json",
                {
                    "affected_condition_count": 1,
                    "affected_conditions": [{"run_id": "flagged"}],
                },
                overwrite=False,
            )
            selected, manifest = canonicalize(output_root)
            self.assertEqual(selected, [])
            self.assertEqual(manifest["canonical_condition_count"], 0)
            self.assertEqual(manifest["real_rag_invalid_condition_count"], 1)
            self.assertEqual(manifest["missing_real_rag_valid_counts"]["clean"], 560)

    def test_multi_turn_requires_valid_real_rag_for_every_turn(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "multi"
            raw.mkdir()
            write_json(
                raw / "run_record.json",
                {
                    "run_id": "multi",
                    "case_id": "case-multi",
                    "dataset_group": "multi_turn",
                    "orchestrator": "llm",
                    "reflection_enabled": False,
                    "seed": 2026,
                    "repeat_index": 0,
                    "completed": True,
                    "runtime_retrieval_backend": "elasticsearch_hybrid_bm25_dense",
                    "retrieval_call_count": 2,
                    "expected_turn_count": 2,
                },
                overwrite=False,
            )
            write_json(raw / "job.json", {}, overwrite=False)
            for turn, retrieved in (("turn_01", True), ("turn_02", False)):
                write_json(
                    raw / "turns" / turn / "retrieval_trace.json",
                    [
                        {
                            "provider": "elasticsearch_hybrid",
                            "backend": "elasticsearch_hybrid_bm25_dense",
                            "retrieved_context": [{"chunk_id": "chunk"}] if retrieved else [],
                            "warnings": [],
                        }
                    ],
                    overwrite=False,
                )
            selected, _ = select_canonical_attempts([raw])
            record = selected[0]["record"]
            self.assertFalse(record["real_rag_valid"])
            self.assertIn(
                "turn_02:retrieval_trace_missing_verified_elasticsearch_hybrid_attempt",
                record["real_rag_invalid_reasons"],
            )

    def test_verified_empty_dense_attempt_is_canonical_real_rag_valid(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            attempt = self._attempt(root, "empty-dense", completed=True, real_rag=True)
            write_json(
                attempt / "runtime" / "retrieval_trace.json",
                [
                    {
                        "provider": "elasticsearch_hybrid",
                        "backend": "elasticsearch_hybrid_bm25_dense",
                        "retrieved_context": [],
                        "evidence_items": [],
                        "warnings": [],
                        "metadata": {"dense_enabled": True, "dense_status": "completed"},
                    }
                ],
                overwrite=True,
            )
            selected, _ = select_canonical_attempts([attempt])
            self.assertTrue(
                selected[0]["record"]["real_rag_valid"],
                selected[0]["record"]["real_rag_invalid_reasons"],
            )

    def test_multi_turn_rejects_a_missing_turn_trace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "multi"
            raw.mkdir()
            write_json(
                raw / "run_record.json",
                {
                    "run_id": "multi",
                    "case_id": "case-multi",
                    "dataset_group": "multi_turn",
                    "orchestrator": "llm",
                    "reflection_enabled": False,
                    "seed": 2026,
                    "repeat_index": 0,
                    "completed": True,
                    "runtime_retrieval_backend": "elasticsearch_hybrid_bm25_dense",
                    "retrieval_call_count": 2,
                    "expected_turn_count": 2,
                },
                overwrite=False,
            )
            write_json(raw / "job.json", {}, overwrite=False)
            write_json(
                raw / "turns" / "turn_01" / "retrieval_trace.json",
                [
                    {
                        "provider": "elasticsearch_hybrid",
                        "backend": "elasticsearch_hybrid_bm25_dense",
                        "retrieved_context": [{"chunk_id": "chunk"}],
                        "warnings": [],
                    }
                ],
                overwrite=False,
            )
            selected, _ = select_canonical_attempts([raw])
            record = selected[0]["record"]
            self.assertFalse(record["real_rag_valid"])
            self.assertIn(
                "retrieval_trace_count=1!=expected_turn_count=2",
                record["real_rag_invalid_reasons"],
            )

    def test_require_full_rejects_incomplete_real_rag_matrix(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp) / "output"
            raw_root = output_root / "raw_runs"
            raw_root.mkdir(parents=True)
            self._attempt(raw_root, "only-one", completed=True, real_rag=True)
            with self.assertRaisesRegex(RuntimeError, "real-RAG-valid full-matrix"):
                canonicalize(output_root, require_full=True)


if __name__ == "__main__":
    unittest.main()
