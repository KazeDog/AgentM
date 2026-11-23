from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import run_benchmark_suite as benchmark


class FakeController:
    init_count = 0
    calls = []

    def __init__(self, **kwargs):
        type(self).init_count += 1
        self.kwargs = kwargs

    def run(self, request, task_type=None, dataset_name=None):
        type(self).calls.append({"request": request, "task_type": task_type, "dataset_name": dataset_name})
        if "FAIL" in str(request):
            raise RuntimeError("synthetic failure")
        output_dir = Path(self.kwargs["output_dir"])
        report_path = output_dir / "multi_agent_report.json"
        report_path.write_text("{}", encoding="utf-8")
        return {
            "task_type": task_type,
            "predictions": [{"dataset_name": dataset_name, "model_type": "random_forest", "score": 0.73}],
            "evidence": [{"source": "elasticsearch_hybrid", "citation_id": "cite_1"}],
            "warnings": [],
            "tool_calls": [
                {"agent": "GaugeAgent", "tool": "predict_single"},
                {"agent": "ReflectionAgent", "tool": "reflect_on_report"},
            ],
            "plan_revision_trace": [{"stage": "action_guard"}],
            "multi_agent": {"planner_type": "llm"},
            "objective_status": {"status": "pass"},
            "verifier_report": {
                "summary": {"status": "pass"},
                "citations": {
                    "citation_grounded": True,
                    "evidence_relevance_grounded": True,
                    "compound_specific_grounded": True,
                    "endpoint_specific_grounded": True,
                },
            },
            "artifacts": {"multi_agent_report_path": str(report_path)},
        }


class BenchmarkSuiteTests(unittest.TestCase):
    def setUp(self):
        FakeController.init_count = 0
        FakeController.calls = []

    def test_benchmark_runner_writes_json_csv_md(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            case_file = self._case_file(root, [{"name": "case_a"}, {"name": "case_b"}])
            output_root = root / "bench"
            with patch.object(benchmark, "AgentMMultiAgentController", FakeController):
                code = benchmark.main(
                    [
                        "--case-file",
                        str(case_file),
                        "--output-root",
                        str(output_root),
                        "--max-cases",
                        "2",
                        "--retrieval-backend",
                        "es",
                        "--es-url",
                        "http://127.0.0.1:9200",
                        "--no-use-llm-orchestrator",
                    ]
                )
            self.assertEqual(code, 0)
            summary = json.loads((output_root / "benchmark_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["metrics"]["n_cases"], 2)
            self.assertEqual(summary["metrics"]["n_completed"], 2)
            self.assertEqual(summary["metrics"]["real_model_success_rate"], 1.0)
            self.assertEqual(summary["settings"]["use_llm_orchestrator"], False)
            with (output_root / "case_metrics.csv").open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["real_model_success"], "True")
            self.assertIn("AgentM Benchmark Summary", (output_root / "benchmark_summary.md").read_text(encoding="utf-8"))

    def test_resume_skips_completed_case(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            case_file = self._case_file(root, [{"name": "case_a"}])
            output_root = root / "bench"
            with patch.object(benchmark, "AgentMMultiAgentController", FakeController):
                self.assertEqual(benchmark.main(["--case-file", str(case_file), "--output-root", str(output_root)]), 0)
            FakeController.init_count = 0
            FakeController.calls = []
            with patch.object(benchmark, "AgentMMultiAgentController", FakeController):
                self.assertEqual(benchmark.main(["--case-file", str(case_file), "--output-root", str(output_root), "--resume"]), 0)
            self.assertEqual(FakeController.init_count, 0)
            summary = json.loads((output_root / "benchmark_summary.json").read_text(encoding="utf-8"))
            self.assertTrue(summary["cases"][0]["resume_skipped"])

    def test_failure_report_contains_traceback_and_case_filter_works(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            case_file = self._case_file(root, [{"name": "case_ok"}, {"name": "case_fail", "request": "FAIL"}])
            output_root = root / "bench"
            with patch.object(benchmark, "AgentMMultiAgentController", FakeController):
                code = benchmark.main(["--case-file", str(case_file), "--output-root", str(output_root), "--case", "case_fail"])
            self.assertEqual(code, 2)
            failure_report = json.loads((output_root / "failure_report.json").read_text(encoding="utf-8"))
            self.assertEqual(len(failure_report["failures"]), 1)
            self.assertEqual(failure_report["failures"][0]["name"], "case_fail")
            self.assertIn("synthetic failure", failure_report["failures"][0]["error"])
            self.assertIn("Traceback", failure_report["failures"][0]["traceback"])

    def test_unknown_case_filter_fails_fast(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            case_file = self._case_file(root, [{"name": "case_a"}])
            with self.assertRaises(SystemExit):
                benchmark.main(["--case-file", str(case_file), "--output-root", str(root / "bench"), "--case", "missing"])

    def test_gold_labeled_case_reports_scientific_accuracy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            case_file = self._case_file(
                root,
                [
                    {
                        "name": "gold_case",
                        "gold_label": 1,
                        "gold_split": "test",
                        "gold_source": "synthetic:test",
                        "scientific_mode": "classification",
                        "model_path": "outputs/example/model.pkl",
                    }
                ],
            )
            output_root = root / "bench"
            original_run = FakeController.run

            def labeled_run(self, request, task_type=None, dataset_name=None):
                report = original_run(
                    self,
                    request,
                    task_type=task_type,
                    dataset_name=dataset_name,
                )
                report["predictions"][0]["label"] = 1
                return report

            with patch.object(FakeController, "run", labeled_run), patch.object(
                benchmark,
                "AgentMMultiAgentController",
                FakeController,
            ):
                self.assertEqual(
                    benchmark.main(
                        [
                            "--case-file",
                            str(case_file),
                            "--output-root",
                            str(output_root),
                        ]
                    ),
                    0,
                )
            summary = json.loads(
                (output_root / "benchmark_summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                summary["scientific_performance"]["coverage"]["n_evaluable"],
                1,
            )
            self.assertEqual(
                summary["scientific_performance"]["classification"]["accuracy"],
                1.0,
            )
            self.assertEqual(
                FakeController.calls[0]["dataset_name"],
                "DILI",
            )

    def _case_file(self, root: Path, overrides):
        path = root / "cases.jsonl"
        rows = []
        for item in overrides:
            case = {
                "name": item.get("name", "case"),
                "request": item.get("request", "CCO"),
                "task_type": item.get("task_type", "admet_prediction"),
                "dataset_name": item.get("dataset_name", "DILI"),
                "category": item.get("category", "admet"),
                "expected_endpoint": item.get("expected_endpoint", "dili"),
                "notes": item.get("notes", "test case"),
            }
            case.update(
                {
                    key: value
                    for key, value in item.items()
                    if key
                    not in {
                        "name",
                        "request",
                        "task_type",
                        "dataset_name",
                        "category",
                        "expected_endpoint",
                        "notes",
                    }
                }
            )
            rows.append(json.dumps(case))
        path.write_text("\n".join(rows) + "\n", encoding="utf-8")
        return path


if __name__ == "__main__":
    unittest.main()
