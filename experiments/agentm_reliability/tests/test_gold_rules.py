from __future__ import annotations

from collections import Counter
import unittest

from experiments.agentm_reliability.build_stress_manifest import STRESS_PROTOCOL, build
from experiments.agentm_reliability.common import load_config
from experiments.agentm_reliability.gold_rules import validate_stress_case
from experiments.agentm_reliability.offline_evaluator import _citation_confusion


class GoldRuleTests(unittest.TestCase):
    def test_citation_true_false_positive_and_false_negative(self) -> None:
        tp, fp, fn = _citation_confusion({"c1": "e1", "c2": "e2"}, {"c1": "e1", "c2": "wrong", "c3": "e3"})
        self.assertEqual((tp, fp, fn), (1, 2, 1))

    def test_missing_required_field_is_explicit(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing required field"):
            validate_stress_case({"stress_case_id": "bad"})

    def test_manifest_is_deterministic_balanced_and_single_fault(self) -> None:
        config = load_config("experiments/agentm_reliability/configs/base.yaml")
        first = build(config)
        second = build(config)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 64)
        self.assertEqual(set(Counter(row["injection_type"] for row in first).values()), {8})
        self.assertEqual(set(Counter(row["task_type"] for row in first).values()), {16})
        self.assertTrue(all(row["primary_fault_count"] == 1 for row in first))
        self.assertTrue(all(row["stress_protocol"] == STRESS_PROTOCOL for row in first))
        evidence_faults = {
            row["injection_type"] for row in first if row["task_type"] == "reaction_evidence"
        }
        self.assertNotIn("missing_field_or_overclaim", evidence_faults)
        self.assertNotIn("plausible_but_wrong_prediction", evidence_faults)
        self.assertNotIn("prediction_tool_or_service_failure", evidence_faults)
        for row in first:
            if row["task_type"] not in {"admet", "ddi"}:
                continue
            if row["injection_type"] in {
                "missing_field_or_overclaim",
                "plausible_but_wrong_prediction",
                "prediction_tool_or_service_failure",
            }:
                self.assertEqual(row["base_case_pool"], "held_out_prediction_gold")
            else:
                self.assertEqual(row["base_case_pool"], "named_direct_evidence")
            if row["injection_type"] == "plausible_but_wrong_prediction":
                self.assertIsNotNone(row["gold_truth_reference"])


if __name__ == "__main__":
    unittest.main()
