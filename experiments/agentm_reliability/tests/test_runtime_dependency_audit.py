from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from experiments.agentm_reliability.audit_runtime_dependencies import dependency_error_in_context


class RuntimeDependencyAuditTests(unittest.TestCase):
    def test_connection_refusal_is_detected_case_insensitively(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "context.json"
            path.write_text(
                '{"retrieval_trace":[{"warnings":["Connection Refused at 127.0.0.1:9200"]}]}',
                encoding="utf-8",
            )
            self.assertTrue(dependency_error_in_context(path))

    def test_recovered_llm_timeout_is_not_a_retrieval_dependency_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "context.json"
            path.write_text(
                '{"retrieval_trace":[{"provider":"elasticsearch_hybrid","warnings":[]}],'
                '"llm_retry_trace":[{"status":"failed","error":"Read timed out"},'
                '{"status":"completed","retry_used":true}]}',
                encoding="utf-8",
            )
            self.assertFalse(dependency_error_in_context(path))

    def test_healthy_context_is_not_flagged(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "context.json"
            path.write_text('{"evidence_count":3}', encoding="utf-8")
            self.assertFalse(dependency_error_in_context(path))


if __name__ == "__main__":
    unittest.main()
