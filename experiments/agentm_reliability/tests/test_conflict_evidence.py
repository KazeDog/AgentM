from __future__ import annotations

import unittest
import tempfile

from agentm.runtime.tools import AgentAction, ToolResult
from agentm.runtime.tools.scientific import ScientificToolRuntime
from experiments.agentm_reliability.injections import ControlledInjectionHook, apply_injection
from experiments.agentm_reliability.offline_evaluator import evaluate_stress_case


class ConflictEvidenceTests(unittest.TestCase):
    def test_conflict_injection_preserves_both_sources_and_gold(self) -> None:
        context = {
            "task_type": "admet_prediction",
            "evidence": [{"citation_id": "original", "chunk_id": "chunk", "source": "source_a", "claim": "supports", "evidence_direction": "support"}],
            "predictions": [{"label": 1, "model_type": "rf", "score": 0.7}],
            "warnings": [],
        }
        record = apply_injection(context, "conflicting_evidence", seed=2026)
        self.assertEqual(len(context["evidence"]), 2)
        self.assertEqual({item["evidence_direction"] for item in context["evidence"]}, {"support", "contradiction"})
        self.assertEqual(len(record["details"]["conflicting_evidence_ids"]), 2)

        case = {
            "injection_type": "conflicting_evidence",
            "recoverability": "recoverable",
            "task_type": "admet",
            "gold_required_uncertainty": True,
            "gold_final_status": "partial",
            "gold_expected_action": "preserve_conflict",
            "gold_citation_map": {},
        }
        context["reasoning"] = {"summary": "The evidence conflicts and remains uncertain; both sources are retained."}
        context["tool_calls"] = []
        result = evaluate_stress_case(case=case, context=context, report={"reasoning": context["reasoning"]}, hook={"applied": True, "runtime_gold": case})
        self.assertTrue(result["conflict_detected"])
        self.assertTrue(result["conflicting_sources_preserved"])

    def test_hook_reconciles_gold_to_actual_injected_citation_links(self) -> None:
        context = {
            "task_type": "admet_prediction",
            "evidence": [
                {
                    "citation_id": "original",
                    "chunk_id": "chunk_actual",
                    "source": "source_a",
                    "source_path": "source://actual",
                    "claim": "supports",
                    "evidence_direction": "support",
                }
            ],
            "predictions": [{"label": 1, "model_type": "rf", "score": 0.7}],
            "warnings": [],
        }
        case = {
            "injection_type": "conflicting_evidence",
            "mutation_seed": 2026,
            "stress_case_id": "stress_conflict",
            "base_case_id": "base",
            "task_type": "admet",
            "gold_citation_map": {},
        }
        hook = ControlledInjectionHook(case)
        hook.after_tool(
            action=AgentAction("RetrieveAgent", "retrieve_elasticsearch_hybrid"),
            result=ToolResult("retrieve_elasticsearch_hybrid", "completed"),
            context=context,
        )

        injected_id = context["evidence"][-1]["citation_id"]
        self.assertEqual(
            hook.runtime_gold["gold_citation_map"],
            {"original": "chunk_actual", injected_id: injected_id},
        )
        self.assertEqual(hook.runtime_gold["gold_conflicting_evidence_ids"], ["original", injected_id])

    def test_deterministic_report_explicitly_detects_conflict_even_for_qc_only_base(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = ScientificToolRuntime(output_dir=tmp, feature_set="rdkit_only")
            context = {
                "request": "ibuprofen ClinTox",
                "task_type": "admet_prediction",
                "dataset_name": "ClinTox",
                "molecules": [{"name": "ibuprofen"}],
                "predictions": [
                    {
                        "task_type": "admet_prediction",
                        "dataset_name": "ClinTox",
                        "label": 0,
                        "score": 0.1,
                        "probability": 0.1,
                        "model_type": "random_forest",
                        "details": {},
                    }
                ],
                "features": [],
                "evidence": [
                    {"citation_id": "a", "source": "qc", "claim": "support", "evidence_direction": "support"},
                    {"citation_id": "b", "source": "peer", "claim": "opposite", "evidence_direction": "contradiction"},
                ],
                "metadata": {"retrieval_outcome": "no_direct_evidence", "use_llm_reasoning": False},
                "reasoning": {},
            }
            result = runtime.build_structured_explanation({}, context)

        self.assertEqual(result.status, "completed")
        explanation = context["reasoning"]["structured_explanation"].lower()
        self.assertIn("evidence conflict", explanation)
        self.assertIn("uncertain", explanation)
        self.assertIn("opposite directions", explanation)


if __name__ == "__main__":
    unittest.main()
