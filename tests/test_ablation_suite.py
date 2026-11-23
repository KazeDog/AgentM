from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import run_agentm_ablation as ablation


class FakeAblationController:
    init_kwargs = []

    def __init__(self, **kwargs):
        type(self).init_kwargs.append(kwargs)
        self.kwargs = kwargs

    def run(self, request, task_type=None, dataset_name=None):
        output_dir = Path(self.kwargs["output_dir"])
        report_path = output_dir / "multi_agent_report.json"
        report_path.write_text("{}", encoding="utf-8")
        use_llm = bool(self.kwargs.get("use_llm_orchestrator"))
        enable_reflection = bool(self.kwargs.get("enable_reflection"))
        weak_without_reflection = str(request) == "WEAK" and not enable_reflection
        tool_calls = [
            {"agent": "RetrieveAgent", "tool": "retrieve_local_rag", "duration_sec": 0.2},
            {"agent": "GaugeAgent", "tool": "predict_single", "duration_sec": 0.1},
        ]
        if enable_reflection:
            tool_calls.append({"agent": "ReflectionAgent", "tool": "reflect_on_report", "duration_sec": 0.05})
            tool_calls.append({"agent": "ReflectionAgent", "tool": "apply_reflection_feedback", "duration_sec": 0.01})
        evidence = [] if weak_without_reflection else [{"source": "rag_literature", "citation_id": "cite_1"}]
        warnings = ["objective_missing:evidence_count_minimum", "citation_not_grounded"] if weak_without_reflection else []
        objective_status = "fail" if weak_without_reflection else "pass"
        verifier_status = "fail" if weak_without_reflection else "pass"
        citation_flags = not weak_without_reflection
        reflection = None
        if enable_reflection:
            reflection = {
                "input_report_summary": {
                    "verifier_status": "fail" if str(request) == "WEAK" else "warning",
                    "warnings": (
                        ["objective_missing:evidence_count_minimum", "citation_not_grounded"]
                        if str(request) == "WEAK"
                        else ["objective_missing:reflection_applied"]
                    ),
                },
                "requires_reasoning_refresh": True,
                "requires_retrieval_refinement": str(request) == "WEAK",
                "application_status": "reflection_applied",
            }
        report = {
            "task_type": task_type,
            "predictions": [{"dataset_name": dataset_name, "model_type": "random_forest", "score": 0.73}],
            "evidence": evidence,
            "warnings": warnings,
            "tool_calls": tool_calls,
            "plan_revision_trace": [{"stage": "action_guard"}] if use_llm else [],
            "llm_plan_proposals": [{"action": "route_task"}] if use_llm else [],
            "retrieval_trace": [{"backend": "dense_faiss"}],
            "multi_agent": {"planner_type": "llm" if use_llm else "heuristic"},
            "objective_status": {
                "status": objective_status,
                "missing_objectives": ["evidence_count_minimum"] if weak_without_reflection else [],
            },
            "verifier_report": {
                "summary": {"status": verifier_status},
                "citations": {
                    "citation_grounded": citation_flags,
                    "evidence_relevance_grounded": citation_flags,
                    "compound_specific_grounded": citation_flags,
                    "endpoint_specific_grounded": citation_flags,
                },
            },
            "artifacts": {"multi_agent_report_path": str(report_path)},
        }
        if reflection:
            report["reflection"] = reflection
        return report


class AblationSuiteTests(unittest.TestCase):
    def setUp(self):
        FakeAblationController.init_kwargs = []

    def test_ablation_runner_writes_multi_variant_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            case_file = self._case_file(root, [{"name": "case_a"}, {"name": "case_b", "task_type": "ddi_prediction", "dataset_name": "DDI"}])
            output_root = root / "ablation"
            with patch.object(ablation, "AgentMMultiAgentController", FakeAblationController):
                code = ablation.main(
                    [
                        "--case-file",
                        str(case_file),
                        "--output-root",
                        str(output_root),
                        "--variant",
                        "llm_planner_reflection_on_retrieval_on",
                        "--variant",
                        "heuristic_planner_reflection_off_retrieval_on",
                        "--all-cases",
                        "--no-resume",
                    ]
                )
            self.assertEqual(code, 0)
            summary = json.loads((output_root / "ablation_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["metrics"]["n_runs"], 4)
            self.assertEqual(summary["by_variant"]["llm_planner_reflection_on_retrieval_on"]["llm_plan_valid_rate"], 1.0)
            self.assertEqual(summary["by_variant"]["heuristic_planner_reflection_off_retrieval_on"]["mean_reflections"], 0.0)
            self.assertEqual(summary["by_variant"]["llm_planner_reflection_on_retrieval_on"]["first_pass_reflection_repair_rate"], 1.0)
            self.assertEqual(summary["reflection_repair_summary"]["n_first_pass_repaired"], 2)
            with (output_root / "ablation_metrics.csv").open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 4)
            self.assertIn("llm_plan_valid", rows[0])
            self.assertIn("reflection_repaired_first_pass", rows[0])
            self.assertIn("AgentM LLM Orchestration Ablation", (output_root / "ablation_summary.md").read_text(encoding="utf-8"))
            self.assertIn("Reflection Repair Summary", (output_root / "reflection_repair_summary.md").read_text(encoding="utf-8"))

    def test_reflection_repair_summary_pairs_on_off_variants(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            case_file = self._case_file(root, [{"name": "case_weak", "request": "WEAK"}])
            output_root = root / "ablation"
            with patch.object(ablation, "AgentMMultiAgentController", FakeAblationController):
                code = ablation.main(
                    [
                        "--case-file",
                        str(case_file),
                        "--output-root",
                        str(output_root),
                        "--variant",
                        "heuristic_planner_reflection_on_retrieval_on",
                        "--variant",
                        "heuristic_planner_reflection_off_retrieval_on",
                        "--all-cases",
                        "--no-resume",
                    ]
                )
            self.assertEqual(code, 0)
            summary = json.loads((output_root / "ablation_summary.json").read_text(encoding="utf-8"))
            repair = summary["reflection_repair_summary"]
            self.assertEqual(repair["n_paired_cases"], 1)
            self.assertEqual(repair["n_paired_cases_with_any_repair"], 1)
            self.assertEqual(repair["paired_taxonomy"]["objective_status"], 1)
            self.assertEqual(repair["paired_taxonomy"]["verifier_status"], 1)
            self.assertEqual(repair["repaired_objectives"]["evidence_count_minimum"], 1)
            paired_case = repair["paired_cases"][0]
            self.assertTrue(paired_case["first_pass_repaired"])
            self.assertIn("first_pass_weak_report", paired_case["improved_fields"])

    def test_default_ablation_uses_representative_cases(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            case_file = self._case_file(
                root,
                [
                    {"name": "aspirin_dili"},
                    {"name": "aspirin_bbbp"},
                    {"name": "caffeine_herg"},
                    {"name": "simvastatin_ketoconazole_ddi", "task_type": "ddi_prediction", "dataset_name": "DDI"},
                    {"name": "clopidogrel_omeprazole_ddi", "task_type": "ddi_prediction", "dataset_name": "DDI"},
                    {"name": "extra_case"},
                ],
            )
            output_root = root / "ablation"
            with patch.object(ablation, "AgentMMultiAgentController", FakeAblationController):
                code = ablation.main(["--case-file", str(case_file), "--output-root", str(output_root), "--variant", "heuristic_planner_reflection_on_retrieval_on"])
            self.assertEqual(code, 0)
            summary = json.loads((output_root / "ablation_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(
                summary["cases"],
                [
                    "aspirin_dili",
                    "aspirin_bbbp",
                    "caffeine_herg",
                    "simvastatin_ketoconazole_ddi",
                    "clopidogrel_omeprazole_ddi",
                ],
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
            rows.append(json.dumps(case))
        path.write_text("\n".join(rows) + "\n", encoding="utf-8")
        return path


if __name__ == "__main__":
    unittest.main()
