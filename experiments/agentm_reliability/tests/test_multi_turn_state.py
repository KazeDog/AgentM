from __future__ import annotations

import unittest

from experiments.agentm_reliability.offline_evaluator import evaluate_multi_turn


class MultiTurnStateTests(unittest.TestCase):
    def _case(self) -> dict:
        return {
            "multi_turn_type": "withdraw_and_replace",
            "turns": [{"turn": 1}, {"turn": 2}],
            "gold_plan": {
                "required_tools": ["resolve_molecule", "predict_single"],
                "expected_final_scientific_objects": ["NEW_OBJECT"],
                "stale_records_to_remove": ["OLD_OBJECT"],
            },
        }

    def test_withdrawn_target_is_detected_as_stale(self) -> None:
        contexts = [
            {"orchestrator_plan": {"id": 1}, "tool_calls": []},
            {
                "orchestrator_plan": {"id": 2},
                "request": "NEW_OBJECT",
                "molecules": [{"original_input": "NEW_OBJECT"}],
                "predictions": [{"model_type": "rf", "label": 1}],
                "evidence": [{"claim": "OLD_OBJECT evidence remains", "citation_id": "c1", "source_path": "s"}],
                "citation_map": [{"claim_id": "c1", "source_path": "s"}],
                "tool_calls": [
                    {"tool": "resolve_molecule", "status": "completed"},
                    {"tool": "predict_single", "status": "completed"},
                ],
            },
        ]
        result = evaluate_multi_turn(case=self._case(), turn_contexts=contexts, final_report={})
        self.assertTrue(result["cross_turn_contamination"])
        self.assertEqual(result["stale_result_removal_accuracy"], 0.0)

    def test_clean_replacement_removes_stale_target(self) -> None:
        contexts = [
            {"orchestrator_plan": {"id": 1}, "tool_calls": []},
            {
                "orchestrator_plan": {"id": 2},
                "request": "NEW_OBJECT",
                "molecules": [{"original_input": "NEW_OBJECT"}],
                "predictions": [{"model_type": "rf", "label": 1}],
                "evidence": [{"claim": "NEW_OBJECT direct evidence", "citation_id": "c1", "source_path": "s"}],
                "citation_map": [{"claim_id": "c1", "source_path": "s"}],
                "tool_calls": [
                    {"tool": "resolve_molecule", "status": "completed"},
                    {"tool": "predict_single", "status": "completed"},
                ],
            },
        ]
        result = evaluate_multi_turn(case=self._case(), turn_contexts=contexts, final_report={})
        self.assertFalse(result["cross_turn_contamination"])
        self.assertEqual(result["stale_result_removal_accuracy"], 1.0)

    def test_previous_turn_audit_history_is_not_active_state_contamination(self) -> None:
        contexts = [
            {"orchestrator_plan": {"id": 1}, "tool_calls": []},
            {
                "orchestrator_plan": {"id": 2},
                "request": "NEW_OBJECT",
                "molecules": [{"original_input": "NEW_OBJECT"}],
                "predictions": [{"model_type": "rf", "label": 1}],
                "evidence": [{"claim": "NEW_OBJECT direct evidence", "citation_id": "c1", "source_path": "s"}],
                "citation_map": [{"claim_id": "c1", "source_path": "s"}],
                "tool_calls": [
                    {"tool": "resolve_molecule", "status": "completed"},
                    {"tool": "predict_single", "status": "completed"},
                ],
            },
        ]
        report = {
            "request": "NEW_OBJECT",
            "molecules": [{"original_input": "NEW_OBJECT"}],
            "predictions": [{"model_type": "rf", "label": 1}],
            "evidence": [{"claim": "NEW_OBJECT direct evidence"}],
            "citation_map": [{"claim_id": "c1", "source_path": "s"}],
            "previous_run_context": {"request": "OLD_OBJECT", "predictions": [{"label": 0}]},
            "artifacts": {"trace": "OLD_OBJECT was replaced"},
        }

        result = evaluate_multi_turn(case=self._case(), turn_contexts=contexts, final_report=report)

        self.assertFalse(result["cross_turn_contamination"])
        self.assertEqual(result["stale_result_removal_accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
