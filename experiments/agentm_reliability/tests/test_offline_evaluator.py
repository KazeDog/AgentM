from __future__ import annotations

from copy import deepcopy
import unittest

from experiments.agentm_reliability.offline_evaluator import (
    evaluate_multi_turn,
    evaluate_report_quality,
    evaluate_stress_case,
)


def stress_case(fault: str, recoverability: str, expected_action: str) -> dict:
    return {
        "injection_type": fault,
        "recoverability": recoverability,
        "task_type": "admet",
        "gold_required_uncertainty": True,
        "gold_final_status": "failed_safely" if expected_action == "abstain" else "partial",
        "gold_expected_action": expected_action,
        "gold_citation_map": {},
        "gold_truth_reference": {"kind": "held_out_dataset_label", "value": 1},
    }


class OfflineEvaluatorTests(unittest.TestCase):
    def test_stress_citations_allow_trace_grounded_post_injection_expansion(self) -> None:
        case = {
            "injection_type": "entity_or_endpoint_mismatch",
            "recoverability": "recoverable",
            "task_type": "reaction_evidence",
            "gold_required_uncertainty": False,
            "gold_final_status": "completed",
            "gold_expected_action": "retrieve",
            # This is the legacy citation-id target representation.  The
            # immutable trace supplies the current source-path representation.
            "gold_citation_map": {"condition-1": "condition-1"},
        }
        trace_evidence = [
            {
                "citation_id": "condition-1",
                "source": "orderly_condition",
                "source_path": "source://conditions",
                "claim": "Condition analogue one.",
                "reaction_specific": True,
            },
            {
                "citation_id": "template-1",
                "source": "mit_reaction",
                "source_path": "source://templates",
                "claim": "Template analogue one.",
                "reaction_specific": True,
            },
        ]
        context = {
            "task_type": "reaction_condition_retrieval",
            "evidence": deepcopy(trace_evidence),
            "citation_map": [
                {"claim_id": "condition-1", "source_path": "source://conditions"},
                {"claim_id": "template-1", "source_path": "source://templates"},
            ],
            "retrieval_trace": [{"backend": "local_reaction_data", "evidence_items": trace_evidence}],
            "warnings": ["evidence record entity mismatch or endpoint mismatch detected and repaired from retrieval trace"],
            "tool_calls": [],
        }

        result = evaluate_stress_case(
            case=case,
            context=context,
            report={"explanation": "The endpoint mismatch was repaired from retrieved evidence."},
            hook={"applied": True, "runtime_gold": case},
        )

        self.assertEqual(result["citation_tp"], 2)
        self.assertEqual(result["citation_fp"], 0)
        self.assertEqual(result["citation_fn"], 0)
        self.assertEqual(result["citation_precision"], 1.0)
        self.assertEqual(result["citation_recall"], 1.0)

    def test_stress_citations_reject_untraced_addition(self) -> None:
        case = {
            "injection_type": "citation_mismatch",
            "recoverability": "recoverable",
            "task_type": "admet",
            "gold_required_uncertainty": False,
            "gold_final_status": "completed",
            "gold_expected_action": "repair",
            "gold_citation_map": {"claim-1": "source://one"},
        }
        context = {
            "evidence": [{"citation_id": "claim-1", "source_path": "source://one"}],
            "citation_map": [
                {"claim_id": "claim-1", "source_path": "source://one"},
                {"claim_id": "fabricated", "source_path": "source://fabricated"},
            ],
            "retrieval_trace": [
                {
                    "backend": "elasticsearch_hybrid_bm25_dense",
                    "evidence_items": [{"citation_id": "claim-1", "source_path": "source://one"}],
                }
            ],
            "warnings": ["citation mismatch detected"],
            "tool_calls": [],
        }

        result = evaluate_stress_case(
            case=case,
            context=context,
            report={"explanation": "A citation mismatch was detected and repaired."},
            hook={"applied": True, "runtime_gold": case},
        )

        self.assertEqual(result["citation_tp"], 1)
        self.assertEqual(result["citation_fp"], 1)
        self.assertEqual(result["citation_fn"], 0)

    def test_repaired_weak_evidence_warning_supplies_category_and_uncertainty(self) -> None:
        case = stress_case("irrelevant_or_weak_evidence", "recoverable", "weaken_claim")
        case.update(
            {
                "gold_required_uncertainty": True,
                "gold_final_status": "partial",
                "gold_citation_map": {"template-1": "template-1"},
            }
        )
        evidence = {
            "citation_id": "template-1",
            "source": "mit_reaction_template",
            "claim": "Reaction-specific template evidence.",
            "reaction_specific": True,
        }
        context = {
            "task_type": "reaction_yield_prediction",
            "predictions": [{"label": "low_yield", "score": 14.2, "model_type": "reaction_model"}],
            "reaction_predictions": [{"label": "low_yield", "score": 14.2, "model_type": "reaction_model"}],
            "evidence": [evidence],
            "citation_map": [{"claim_id": "template-1", "citation_id": "template-1"}],
            "retrieval_trace": [{"backend": "local_reaction_data", "evidence_items": [evidence]}],
            "warnings": [
                "irrelevant or weak evidence detected and repaired from retrieval trace; "
                "the resulting conclusion should be interpreted cautiously"
            ],
            "tool_calls": [],
        }

        result = evaluate_stress_case(
            case=case,
            context=context,
            report={"explanation": "The model prediction is retained as a prediction."},
            hook={"applied": True, "runtime_gold": case},
        )

        self.assertTrue(result["detected"])
        self.assertTrue(result["detection_category_correct"])
        self.assertTrue(result["repaired"])
        self.assertTrue(result["uncertainty_expression_correct"])
        self.assertTrue(result["final_status_correct"])

    def test_reaction_condition_retrieval_uses_reaction_profile_schema(self) -> None:
        context = {
            "task_type": "reaction_condition_retrieval",
            "reaction_profile": {"conditions": [{"catalyst": "Pd"}]},
            "evidence": [{"citation_id": "condition-1", "source_path": "source://condition-1"}],
            "citation_map": [{"claim_id": "condition", "citation_id": "condition-1"}],
            "retrieval_trace": [{"backend": "local_reaction_data"}],
            "tool_calls": [{"tool": "retrieve_reaction_conditions", "status": "completed"}],
        }
        report = {
            "request": "retrieve condition context",
            "task_type": "reaction_condition_retrieval",
            "reaction_profile": context["reaction_profile"],
            "evidence": context["evidence"],
            "citation_map": context["citation_map"],
            "explanation": "Retrieved reaction-condition evidence.",
            "retrieval_trace": context["retrieval_trace"],
            "tool_calls": context["tool_calls"],
            "objective_status": {"status": "completed"},
        }

        result = evaluate_report_quality(case={}, context=context, report=report)

        self.assertEqual(result["report_schema_completeness"], 1.0)
        self.assertEqual(result["report_schema_missing_json"], "[]")

    def test_verified_empty_real_rag_result_is_a_complete_evidence_record(self) -> None:
        context = {
            "task_type": "admet_prediction",
            "predictions": [{"label": 1, "model_type": "random_forest"}],
            "evidence": [],
            "citation_map": [],
            "warnings": ["no_direct_evidence_for_target"],
            "metadata": {
                "retrieval_outcome": "no_direct_evidence",
                "direct_evidence_available": False,
                "real_rag_execution_valid": True,
            },
            "retrieval_trace": [{"backend": "elasticsearch_hybrid_bm25_dense"}],
            "tool_calls": [{"tool": "predict_single", "status": "completed"}],
        }
        report = {
            "request": "molecule",
            "task_type": "admet_prediction",
            "predictions": context["predictions"],
            "evidence": [],
            "citation_map": [],
            "explanation": "This is a model prediction; no direct evidence was available.",
            "retrieval_trace": context["retrieval_trace"],
            "tool_calls": context["tool_calls"],
            "objective_status": {"status": "warning"},
        }

        result = evaluate_report_quality(case={}, context=context, report=report)

        self.assertEqual(result["report_schema_completeness"], 1.0)
        self.assertEqual(result["report_schema_missing_json"], "[]")

    def test_multi_turn_empty_citation_map_counts_as_verified_refresh(self) -> None:
        first = {
            "request": "aspirin",
            "task_type": "admet_prediction",
            "predictions": [{"label": 0, "model_type": "random_forest"}],
            "evidence": [{"citation_id": "aspirin-cite", "source_path": "source://aspirin"}],
            "citation_map": [{"claim_id": "aspirin", "citation_id": "aspirin-cite"}],
            "orchestrator_plan": {"status": "planned"},
            "tool_calls": [{"tool": "predict_single", "status": "completed"}],
        }
        final = {
            "request": "omeprazole",
            "task_type": "admet_prediction",
            "molecules": [{"name": "Omeprazole"}],
            "predictions": [{"label": 1, "model_type": "random_forest"}],
            "evidence": [],
            "citation_map": [],
            "warnings": ["no_direct_evidence_for_target"],
            "metadata": {
                "retrieval_outcome": "no_direct_evidence",
                "direct_evidence_available": False,
                "real_rag_execution_valid": True,
            },
            "retrieval_trace": [{"backend": "elasticsearch_hybrid_bm25_dense"}],
            "orchestrator_plan": {"status": "planned"},
            "tool_calls": [{"tool": "predict_single", "status": "completed"}],
        }
        report = {
            "request": "omeprazole",
            "task_type": "admet_prediction",
            "molecules": [{"name": "Omeprazole"}],
            "predictions": final["predictions"],
            "evidence": [],
            "citation_map": [],
            "explanation": "Omeprazole is a model prediction; no direct evidence was available.",
            "retrieval_trace": final["retrieval_trace"],
        }
        case = {
            "turns": [{"request": "aspirin"}, {"request": "omeprazole"}],
            "gold_plan": {
                "expected_final_scientific_objects": ["Omeprazole"],
                "stale_records_to_remove": ["aspirin"],
                "required_tools": ["predict_single"],
            },
        }

        result = evaluate_multi_turn(case=case, turn_contexts=[first, final], final_report=report)

        self.assertTrue(result["citation_update_accuracy"])
        self.assertEqual(result["citation_update_basis"], "verified_empty_evidence_state")

    def test_service_failure_is_safe_abstention_not_repair(self) -> None:
        case = stress_case("prediction_tool_or_service_failure", "unrecoverable", "abstain")
        context = {
            "predictions": [],
            "evidence": [],
            "warnings": ["prediction tool unavailable; task failed safely"],
            "tool_calls": [{"agent": "GaugeAgent", "tool": "predict_single", "status": "failed", "error": "prediction tool unavailable"}],
        }
        result = evaluate_stress_case(case=case, context=context, report={"explanation": "Prediction failed; no result is available."}, hook={"applied": True, "runtime_gold": case})
        self.assertTrue(result["safe_abstention"])
        self.assertFalse(result["repaired"])
        self.assertTrue(result["safe_abstention_applicable"])
        self.assertFalse(result["repair_applicable"])

    def test_controlled_error_marker_does_not_erase_safe_abstention_narrative(self) -> None:
        case = stress_case("prediction_tool_or_service_failure", "unrecoverable", "abstain")
        context = {
            "predictions": [],
            "evidence": [],
            "warnings": ["prediction_tool_or_service_failure: predict_single unavailable"],
            "tool_calls": [
                {
                    "agent": "GaugeAgent",
                    "tool": "predict_single",
                    "status": "failed",
                    "error": "controlled_injection: prediction tool unavailable",
                }
            ],
        }
        report = {
            "explanation": (
                "The prediction service was unavailable, so no prediction was produced. "
                "This is an explicit safe abstention."
            )
        }
        result = evaluate_stress_case(
            case=case,
            context=context,
            report=report,
            hook={"applied": True, "runtime_gold": case},
        )
        self.assertTrue(result["detected"])
        self.assertTrue(result["uncertainty_expression_correct"])
        self.assertTrue(result["safe_abstention"])
        self.assertTrue(result["final_status_correct"])

    def test_ddi_service_failure_warning_is_detected_without_using_injection_error(self) -> None:
        case = stress_case("prediction_tool_or_service_failure", "unrecoverable", "abstain")
        context = {
            "predictions": [],
            "evidence": [],
            "warnings": [
                "prediction_tool_or_service_failure: predict_ddi unavailable; "
                "task failed safely without a prediction"
            ],
            "tool_calls": [
                {
                    "agent": "GaugeAgent",
                    "tool": "predict_ddi",
                    "status": "failed",
                    "error": "controlled_injection: prediction tool unavailable",
                }
            ],
        }
        report = {
            "explanation": (
                "The dependent service was unavailable, so no prediction was produced. "
                "This is an explicit safe abstention."
            )
        }

        result = evaluate_stress_case(
            case=case,
            context=context,
            report=report,
            hook={"applied": True, "runtime_gold": case},
        )

        self.assertTrue(result["detected"])
        self.assertTrue(result["detection_category_correct"])
        self.assertTrue(result["safe_abstention"])
        self.assertTrue(result["final_status_correct"])

    def test_safe_abstention_rejects_fabricated_prediction(self) -> None:
        case = stress_case("prediction_tool_or_service_failure", "unrecoverable", "abstain")
        context = {
            "predictions": [{"label": 1, "model_type": "fabricated"}],
            "evidence": [],
            "warnings": ["prediction tool unavailable"],
            "tool_calls": [{"status": "failed", "error": "unavailable"}],
        }
        result = evaluate_stress_case(case=case, context=context, report={"explanation": "The prediction is certain."}, hook={"applied": True, "runtime_gold": case})
        self.assertFalse(result["safe_abstention"])

    def test_plausible_wrong_prediction_is_not_validated_by_schema_or_verifier_pass(self) -> None:
        case = stress_case("plausible_but_wrong_prediction", "unrecoverable", "mark_prediction")
        context = {
            "predictions": [{"label": 0, "score": 0.8, "model_type": "random_forest"}],
            "evidence": [],
            "warnings": [],
            "tool_calls": [],
            "verifier_report": {"summary": {"status": "pass"}},
        }
        report = {"explanation": "This result is experimentally confirmed and is a proven fact."}
        result = evaluate_stress_case(case=case, context=context, report=report, hook={"applied": True, "runtime_gold": case})
        self.assertFalse(result["prediction_fact_separation_correct"])
        self.assertFalse(result["safe_abstention"])

    def test_verifier_pass_field_does_not_change_independent_outcome(self) -> None:
        case = stress_case("no_direct_evidence_or_ambiguous_entity", "unrecoverable", "abstain")
        base = {"predictions": [], "evidence": [], "warnings": ["no direct evidence; ambiguous entity"], "tool_calls": []}
        report = {"explanation": "No direct evidence is available; the entity is ambiguous and no conclusion can be made."}
        a = deepcopy(base)
        b = deepcopy(base)
        a["verifier_report"] = {"summary": {"status": "pass"}}
        b["verifier_report"] = {"summary": {"status": "fail"}}
        result_a = evaluate_stress_case(case=case, context=a, report=report, hook={"applied": True, "runtime_gold": case})
        result_b = evaluate_stress_case(case=case, context=b, report=report, hook={"applied": True, "runtime_gold": case})
        self.assertEqual(result_a, result_b)


if __name__ == "__main__":
    unittest.main()
