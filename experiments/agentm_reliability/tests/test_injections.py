from __future__ import annotations

from copy import deepcopy
import unittest

from agentm.runtime.tools import AgentAction, ToolResult
from experiments.agentm_reliability.injections import ControlledInjectionHook, apply_injection


def base_context() -> dict:
    return {
        "task_type": "admet_prediction",
        "predictions": [{"label": 1, "score": 0.8, "probability": 0.8, "model_type": "random_forest"}],
        "evidence": [
            {
                "citation_id": "cite_1",
                "chunk_id": "chunk_1",
                "source": "curated",
                "source_path": "source://one",
                "claim": "Target-specific endpoint evidence.",
                "evidence_direction": "support",
            }
        ],
        "citation_map": [{"claim_id": "cite_1", "chunk_id": "chunk_1"}],
        "final_report": {"request": "molecule", "predictions": [], "evidence": []},
        "warnings": [],
    }


class InjectionIsolationTests(unittest.TestCase):
    def test_each_injector_changes_only_declared_top_level_field(self) -> None:
        expected = {
            "citation_mismatch": "citation_map",
            "entity_or_endpoint_mismatch": "evidence",
            "irrelevant_or_weak_evidence": "evidence",
            "missing_field_or_overclaim": "predictions",
            "conflicting_evidence": "evidence",
            "plausible_but_wrong_prediction": "predictions",
            "no_direct_evidence_or_ambiguous_entity": "evidence",
        }
        for fault, target in expected.items():
            with self.subTest(fault=fault):
                context = base_context()
                before = deepcopy(context)
                record = apply_injection(context, fault, seed=2026, truth_reference={'value':1})
                changed = {key for key in set(before) | set(context) if before.get(key) != context.get(key)}
                self.assertEqual(changed, {target})
                self.assertEqual(record["target_field"], target)
                self.assertNotEqual(record["before_hash"], record["after_hash"])

    def test_persistent_prediction_failure_never_fabricates_result(self) -> None:
        case = {
            "injection_type": "prediction_tool_or_service_failure",
            "mutation_seed": 2026,
            "stress_case_id": "stress_failure",
            "base_case_id": "base",
            "task_type": "admet",
            "gold_citation_map": {},
        }
        hook = ControlledInjectionHook(case)
        context = {"task_type": "admet_prediction", "predictions": [], "evidence": [], "metadata": {}}
        action = AgentAction("GaugeAgent", "predict_single")
        first = hook.before_tool(action=action, context=context)
        second = hook.before_tool(action=action, context=context)
        self.assertEqual(first.status, "failed")
        self.assertEqual(second.status, "failed")
        self.assertEqual(context["predictions"], [])
        self.assertEqual(hook.mutation_record["details"]["intercept_count"], 2)

    def test_no_direct_evidence_fault_remains_active_after_later_retrieval(self) -> None:
        case = {
            "injection_type": "no_direct_evidence_or_ambiguous_entity",
            "mutation_seed": 2026,
            "stress_case_id": "stress_no_evidence",
            "base_case_id": "base",
            "task_type": "reaction_prediction",
            "gold_citation_map": {},
        }
        hook = ControlledInjectionHook(case)
        context = base_context()
        context.update(
            {
                "task_type": "reaction_classification",
                "reaction_predictions": [],
                "artifacts": {"reaction_template_evidence": deepcopy(context["evidence"])},
                "metadata": {},
            }
        )
        hook.after_tool(
            action=AgentAction("RetrieveAgent", "retrieve_mit_template_evidence"),
            result=ToolResult("retrieve_mit_template_evidence", "completed"),
            context=context,
        )
        self.assertTrue(hook.applied)
        self.assertTrue(context["metadata"]["safe_abstention_required"])
        self.assertEqual(context["evidence"], [])

        context["evidence"] = deepcopy(base_context()["evidence"])
        context["reaction_predictions"] = [{"label": "class", "model_type": "test"}]
        context["predictions"] = list(context["reaction_predictions"])
        context["artifacts"]["reaction_condition_evidence"] = deepcopy(context["evidence"])
        hook.after_tool(
            action=AgentAction("RetrieveAgent", "retrieve_orderly_condition_evidence"),
            result=ToolResult("retrieve_orderly_condition_evidence", "completed"),
            context=context,
        )

        self.assertEqual(context["evidence"], [])
        self.assertEqual(context["predictions"], [])
        self.assertEqual(context["reaction_predictions"], [])
        self.assertEqual(context["artifacts"]["reaction_condition_evidence"], [])
        self.assertGreaterEqual(hook.mutation_record["details"]["enforcement_count"], 2)
        self.assertEqual(hook.runtime_gold["gold_citation_map"], {})


if __name__ == "__main__":
    unittest.main()
