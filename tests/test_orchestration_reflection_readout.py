from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts import summarize_orchestration_reflection as readout


class OrchestrationReflectionReadoutTests(unittest.TestCase):
    def test_readout_summarizes_llm_and_reflection_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            heuristic_dir = root / "heuristic"
            llm_dir = root / "llm"
            heuristic_summary = self._benchmark_summary(
                heuristic_dir,
                use_llm=False,
                objective_statuses=["pass", "fail"],
                verifier_statuses=["pass", "fail"],
                citations=[True, True],
                evidence_counts=[1, 0],
            )
            llm_summary = self._benchmark_summary(
                llm_dir,
                use_llm=True,
                objective_statuses=["pass", "pass"],
                verifier_statuses=["pass", "pass"],
                citations=[True, False],
                evidence_counts=[2, 1],
            )
            reflection_summary = {
                "settings": {"mode": "full", "retrieval_backend": "es"},
                "by_variant": {
                    "heuristic_planner_reflection_on_retrieval_on": {
                        "n_runs": 2,
                        "objective_pass_rate": 1.0,
                        "verifier_pass_rate": 1.0,
                        "real_model_success_rate": 1.0,
                        "citation_grounding_rate": 1.0,
                        "mean_reflections": 2.0,
                        "mean_evidence_count": 1.5,
                        "mean_report_quality_score": 10.2,
                        "mean_latency_sec": 2.0,
                    },
                    "heuristic_planner_reflection_off_retrieval_on": {
                        "n_runs": 2,
                        "objective_pass_rate": 0.5,
                        "verifier_pass_rate": 0.5,
                        "real_model_success_rate": 1.0,
                        "citation_grounding_rate": 1.0,
                        "mean_reflections": 0.0,
                        "mean_evidence_count": 1.0,
                        "mean_report_quality_score": 9.0,
                        "mean_latency_sec": 1.0,
                    },
                },
                "reflection_repair_summary": {
                    "n_paired_cases": 2,
                    "n_first_pass_repaired": 1,
                    "first_pass_repair_rate": 0.5,
                    "n_paired_cases_with_any_repair": 2,
                    "paired_repair_rate": 1.0,
                    "paired_taxonomy": {"quality_score": 2},
                    "repair_categories": {"warning_reduction": 2},
                    "repaired_objectives": {"reflection_applied": 2},
                    "paired_cases": [
                        {"case_name": "case_a", "any_repair": True},
                        {"case_name": "case_b", "any_repair": True},
                    ],
                },
            }

            heuristic_path = root / "heuristic_summary.json"
            llm_path = root / "llm_summary.json"
            reflection_path = root / "reflection_summary.json"
            heuristic_path.write_text(json.dumps(heuristic_summary), encoding="utf-8")
            llm_path.write_text(json.dumps(llm_summary), encoding="utf-8")
            reflection_path.write_text(json.dumps(reflection_summary), encoding="utf-8")
            output_root = root / "out"
            paper_tex = root / "paper" / "tables.tex"
            paper_md = root / "paper" / "readout.md"

            code = readout.main(
                [
                    "--heuristic-summary",
                    str(heuristic_path),
                    "--llm-summary",
                    str(llm_path),
                    "--reflection-summary",
                    str(reflection_path),
                    "--output-root",
                    str(output_root),
                    "--paper-tex",
                    str(paper_tex),
                    "--paper-md",
                    str(paper_md),
                ]
            )

            self.assertEqual(code, 0)
            payload = json.loads((output_root / "orchestration_reflection_readout.json").read_text(encoding="utf-8"))
            llm_eval = payload["llm_orchestration"]["llm_orchestrated_reflection_strict"]
            comparison = payload["llm_orchestration"]["comparison"]
            self.assertEqual(llm_eval["llm_plan_valid"]["count"], 2)
            self.assertEqual(llm_eval["llm_action_proposals_total"], 4)
            self.assertEqual(comparison["deltas"]["objective_pass"]["delta_count"], 1)
            self.assertEqual(comparison["deltas"]["evidence_positive"]["delta_count"], 1)
            self.assertEqual(payload["reflection"]["paired_summary"]["n_first_pass_repaired"], 1)
            self.assertTrue((output_root / "orchestration_reflection_metrics.csv").exists())
            with (output_root / "orchestration_reflection_metrics.csv").open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 3)
            self.assertIn("LLM orchestration", (output_root / "orchestration_reflection_readout.md").read_text(encoding="utf-8"))
            self.assertIn("\\label{tab:orchestration-reflection-systematic}", paper_tex.read_text(encoding="utf-8"))
            self.assertIn("Paired Repair", paper_md.read_text(encoding="utf-8"))

    def _benchmark_summary(
        self,
        root: Path,
        *,
        use_llm: bool,
        objective_statuses,
        verifier_statuses,
        citations,
        evidence_counts,
    ):
        rows = []
        for idx, objective in enumerate(objective_statuses):
            case_dir = root / f"case_{idx}"
            case_dir.mkdir(parents=True)
            report = {
                "warnings": [],
                "llm_plan_proposals": [{"tool": "route_task"}, {"tool": "build_features"}] if use_llm else [],
            }
            (case_dir / "multi_agent_report.json").write_text(json.dumps(report), encoding="utf-8")
            rows.append(
                {
                    "name": f"case_{idx}",
                    "run_status": "completed",
                    "objective_status": objective,
                    "verifier_status": verifier_statuses[idx],
                    "real_model_success": True,
                    "citation_grounded": citations[idx],
                    "n_evidence": evidence_counts[idx],
                    "n_reflections": 2 if use_llm else 0,
                    "n_guard_revisions": 1 if use_llm else 0,
                    "planner_type": "llm" if use_llm else "heuristic",
                    "latency_sec": 10.0 if use_llm else 1.0,
                    "case_dir": str(case_dir),
                    "report_path": str(case_dir / "multi_agent_report.json"),
                }
            )
        return {
            "case_file": "cases.jsonl",
            "settings": {"use_llm_orchestrator": use_llm},
            "cases": rows,
        }


if __name__ == "__main__":
    unittest.main()
