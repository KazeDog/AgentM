from __future__ import annotations

import unittest
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

from agentm.runtime.scientific_metrics import (
    classification_metrics,
    evaluate_scientific_case,
    regression_metrics,
    summarize_scientific_rows,
)
from agentm.runtime.tools.scientific import ScientificToolRuntime
from scripts.audit_balanced100_scientific_performance import classify_membership


class ScientificMetricTests(unittest.TestCase):
    def test_evaluate_scientific_case_enforces_test_boundary(self):
        held_out = evaluate_scientific_case(
            gold_value=1,
            prediction_value=1.0,
            mode="classification",
            gold_split="test",
        )
        self.assertTrue(held_out["scientific_evaluable"])
        self.assertTrue(held_out["scientific_correct"])

        training = evaluate_scientific_case(
            gold_value=1,
            prediction_value=1,
            mode="classification",
            gold_split="train",
        )
        self.assertFalse(training["scientific_evaluable"])
        self.assertEqual(training["scientific_status"], "excluded_non_test")

    def test_multiclass_metrics_match_balanced100_ddi_example(self):
        metrics = classification_metrics(
            [57, 73, 47, 47, 73],
            [57, 47, 47, 49, 73],
        )
        self.assertEqual(metrics["n"], 5)
        self.assertEqual(metrics["n_correct"], 3)
        self.assertEqual(metrics["accuracy"], 0.6)
        self.assertEqual(metrics["balanced_accuracy"], 0.666667)
        self.assertEqual(metrics["macro_f1"], 0.541667)
        self.assertLess(metrics["accuracy_wilson_95ci"][0], metrics["accuracy"])
        self.assertGreater(metrics["accuracy_wilson_95ci"][1], metrics["accuracy"])

    def test_regression_metrics(self):
        metrics = regression_metrics([1.0, 2.0, 3.0], [1.5, 2.0, 2.5])
        self.assertEqual(metrics["n"], 3)
        self.assertEqual(metrics["mae"], 0.333333)
        self.assertEqual(metrics["rmse"], 0.408248)
        self.assertEqual(metrics["r2"], 0.75)

    def test_summary_excludes_non_test_rows(self):
        rows = [
            {
                "category": "admet",
                "dataset_name": "DILI",
                **evaluate_scientific_case(
                    gold_value=1,
                    prediction_value=1,
                    mode="classification",
                    gold_split="test",
                ),
            },
            {
                "category": "admet",
                "dataset_name": "DILI",
                **evaluate_scientific_case(
                    gold_value=0,
                    prediction_value=0,
                    mode="classification",
                    gold_split="train",
                ),
            },
        ]
        summary = summarize_scientific_rows(rows)
        self.assertEqual(summary["coverage"]["n_cases"], 2)
        self.assertEqual(summary["coverage"]["n_evaluable"], 1)
        self.assertEqual(summary["classification"]["n"], 1)

    def test_summary_does_not_pool_heterogeneous_label_spaces(self):
        rows = [
            {
                "category": "admet",
                "dataset_name": "DILI",
                **evaluate_scientific_case(
                    gold_value=1,
                    prediction_value=1,
                    mode="classification",
                    gold_split="test",
                ),
            },
            {
                "category": "ddi",
                "dataset_name": "DDI",
                **evaluate_scientific_case(
                    gold_value=70,
                    prediction_value=70,
                    mode="classification",
                    gold_split="test",
                ),
            },
        ]
        summary = summarize_scientific_rows(rows)
        self.assertEqual(
            summary["classification"]["status"],
            "heterogeneous_datasets_not_pooled",
        )
        self.assertEqual(summary["classification"]["exact_match_accuracy"], 1.0)
        self.assertNotIn("macro_f1", summary["classification"])


class SplitMembershipTests(unittest.TestCase):
    def test_consistent_test_match_is_gold(self):
        result = classify_membership(
            [
                {"split": "test", "label": 1, "row_id": 3},
                {"split": "test", "label": 1.0, "row_id": 4},
            ]
        )
        self.assertEqual(result["membership_status"], "held_out_test")
        self.assertEqual(result["gold_split"], "test")
        self.assertEqual(result["gold_value"], 1)

    def test_conflicting_duplicate_labels_are_not_evaluable(self):
        result = classify_membership(
            [
                {"split": "train", "label": 0, "row_id": 1},
                {"split": "train", "label": 1, "row_id": 2},
            ]
        )
        self.assertEqual(result["membership_status"], "ambiguous_label")
        self.assertIsNone(result["gold_value"])

    def test_multiple_split_membership_is_rejected(self):
        result = classify_membership(
            [
                {"split": "train", "label": 1, "row_id": 1},
                {"split": "test", "label": 1, "row_id": 2},
            ]
        )
        self.assertEqual(result["membership_status"], "multiple_splits")
        self.assertIsNone(result["gold_split"])


class CachedRuleAlignmentTests(unittest.TestCase):
    def test_model_adjacent_rules_used_file_is_loaded(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            model_path = model_dir / "model.pkl"
            model_path.write_bytes(b"placeholder")
            rules_path = model_dir / "rules_used.json"
            rules_path.write_text(
                json.dumps(
                    {
                        "rules": [
                            {
                                "rule_id": "ddi_rule_1",
                                "task_type": "ddi_prediction",
                                "source": "training",
                                "text": "Synthetic training rule.",
                                "feature_name": "ddi_rule_1",
                                "code": "1.0",
                                "is_executable": True,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            runtime = ScientificToolRuntime.__new__(ScientificToolRuntime)
            context = {
                "resource_resolution": {
                    "model": {
                        "selected_path": str(model_path),
                        "candidates": [
                            {
                                "selected": True,
                                "path": str(model_path),
                                "metadata": {},
                            }
                        ],
                    }
                },
                "artifacts": {},
            }
            with patch.object(runtime, "_update_resource_resolution", return_value=None):
                rules = runtime._load_cached_training_rules(context)
            self.assertIsNotNone(rules)
            self.assertEqual([rule.feature_name for rule in rules], ["ddi_rule_1"])
            self.assertEqual(context["artifacts"]["rules_source_path"], str(rules_path))


if __name__ == "__main__":
    unittest.main()
