from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from agentm.runtime.guards.verifier import verify_prediction_semantics
from agentm.runtime.prediction_semantics import (
    annotate_multiclass_ddi_prediction,
    prediction_meaning,
)
from agentm.runtime.tools.scientific import ScientificToolRuntime


def _multiclass_prediction(label: int = 73) -> dict:
    return {
        "task_type": "ddi_prediction",
        "dataset_name": "DrugBank",
        "label": label,
        "score": 0.97,
        "probability": 0.97,
        "model_type": "sgd_classifier",
        "details": {
            "label_mode": "multiclass",
            "classes": [1, 20, 73, 86],
            "positive_label": 1,
            "positive_label_meaning": "DDI risk/interaction",
            "negative_label": 0,
            "negative_label_meaning": "no DDI risk/interaction",
            "open_task_molecules": ["simvastatin", "ketoconazole"],
        },
    }


class PredictionSemanticsTests(unittest.TestCase):
    def test_annotation_replaces_legacy_binary_metadata_with_interaction_type(self):
        prediction = _multiclass_prediction()
        with tempfile.TemporaryDirectory() as tmp:
            mapping = Path(tmp) / "drugbank.tab"
            mapping.write_text(
                'ID1\tID2\tY\tMap\tX1\tX2\n'
                '"a"\t"b"\t73\t"The serum concentration of #Drug2 can be increased when it is combined with #Drug1."\t"C"\t"CC"\n',
                encoding="utf-8",
            )
            annotate_multiclass_ddi_prediction(
                prediction,
                drug_names=["simvastatin", "ketoconazole"],
                interaction_map_path=mapping,
            )

        details = prediction["details"]
        self.assertFalse(details["binary_interpretation_applicable"])
        self.assertNotIn("positive_label", details)
        self.assertNotIn("negative_label", details)
        self.assertEqual(details["interaction_type_class"], 73)
        self.assertIn("ketoconazole", details["interaction_type_description"])
        self.assertIn("simvastatin", details["interaction_type_description"])
        self.assertIn("legacy_binary_metadata_ignored", details)

    def test_simple_report_preserves_multiclass_ddi_semantics(self):
        runtime = ScientificToolRuntime.__new__(ScientificToolRuntime)
        prediction = _multiclass_prediction()
        prediction["details"].update(
            {
                "interaction_type_class": 73,
                "interaction_type_description": (
                    "The serum concentration of ketoconazole can be increased when it is combined with simvastatin."
                ),
            }
        )
        context = {
            "request": "Assess the interaction between simvastatin and ketoconazole.",
            "task_type": "ddi_prediction",
            "dataset_name": "DrugBank",
            "molecules": [{"name": "simvastatin"}, {"name": "ketoconazole"}],
            "predictions": [prediction],
            "confidence": {"low_confidence": False},
            "evidence": [],
            "artifacts": {},
            "metadata": {},
        }

        explanation = runtime._deterministic_explanation(context)

        self.assertIn("interaction-type class 73", explanation)
        self.assertIn("serum concentration of ketoconazole", explanation)
        self.assertIn("not a binary interaction/no-interaction outcome", explanation)
        self.assertNotIn("does not show a clear drug-drug interaction signal", explanation)
        self.assertNotIn("no DDI risk", explanation)

    def test_composite_report_preserves_multiclass_ddi_semantics(self):
        runtime = ScientificToolRuntime.__new__(ScientificToolRuntime)
        prediction = _multiclass_prediction()
        prediction["details"].update(
            {
                "open_task_subtask_id": "pair::simvastatin::ketoconazole::DDI",
                "interaction_type_class": 73,
                "interaction_type_description": (
                    "The serum concentration of ketoconazole can be increased when it is combined with simvastatin."
                ),
            }
        )
        contract = {
            "open_task_type": "ddi_plus_individual_safety",
            "subtasks": [
                {
                    "subtask_id": "pair::simvastatin::ketoconazole::DDI",
                    "kind": "ddi",
                    "dataset_name": "DDI",
                    "molecules": ["simvastatin", "ketoconazole"],
                }
            ],
        }
        context = {"predictions": [prediction], "evidence": [], "metadata": {"open_task_subtask_results": {}}}

        explanation = runtime._composite_open_task_explanation(context, contract)

        self.assertIn("interaction-type class 73", explanation)
        self.assertIn("not a binary no-interaction result", explanation)
        self.assertNotIn("no clear DDI risk signal", explanation)

    def test_verifier_fails_closed_on_binary_collapse_and_accepts_fixed_text(self):
        prediction = _multiclass_prediction()
        prediction["details"].update(
            {
                "interaction_type_class": 73,
                "interaction_type_description": (
                    "The serum concentration of ketoconazole can be increased when it is combined with simvastatin."
                ),
            }
        )
        context = {
            "predictions": [prediction],
            "final_report": {"explanation": "simvastatin + ketoconazole: no clear DDI risk signal."},
        }
        failed = verify_prediction_semantics(context)
        self.assertFalse(failed["multiclass_ddi_semantics_preserved"])
        self.assertIn("multiclass_result_reduced_to_binary_absence", failed["cases"][0]["issues"])

        context["final_report"]["explanation"] = (
            "The model assigns simvastatin + ketoconazole to DrugBank interaction-type class 73 "
            "(class definition: The serum concentration of ketoconazole can be increased when it is combined with simvastatin.). "
            "This is not a binary interaction/no-interaction result."
        )
        passed = verify_prediction_semantics(context)
        self.assertTrue(passed["multiclass_ddi_semantics_preserved"])

    def test_prediction_meaning_never_uses_binary_absence_for_multiclass(self):
        meaning = prediction_meaning(_multiclass_prediction())
        self.assertIn("interaction-type class 73", meaning)
        self.assertNotIn("no DDI", meaning)

    def test_multiclass_ddi_reasoning_is_deterministic_even_when_llm_is_enabled(self):
        class ExplodingClient:
            available = True

            def chat_json(self, *args, **kwargs):
                raise AssertionError("LLM must not be called for multiclass label semantics")

        runtime = ScientificToolRuntime.__new__(ScientificToolRuntime)
        runtime.use_llm_reasoning = True
        runtime.require_llm_reasoning = False
        runtime.reasoning_llm_client = ExplodingClient()
        prediction = _multiclass_prediction()
        prediction["details"].update(
            {
                "interaction_type_class": 73,
                "interaction_type_description": (
                    "The serum concentration of ketoconazole can be increased when it is combined with simvastatin."
                ),
            }
        )
        context = {
            "request": "Assess simvastatin and ketoconazole.",
            "task_type": "ddi_prediction",
            "dataset_name": "DrugBank",
            "molecules": [{"name": "simvastatin"}, {"name": "ketoconazole"}],
            "predictions": [prediction],
            "confidence": {"low_confidence": False},
            "evidence": [],
            "artifacts": {},
            "metadata": {"use_llm_reasoning": True},
        }

        result = runtime.build_structured_explanation({}, context)

        self.assertEqual(result.status, "completed")
        self.assertEqual(context["metadata"]["reasoning_mode"], "multiclass_semantics_template")
        self.assertIn("interaction-type class 73", context["reasoning"]["structured_explanation"])

    def test_report_verification_fails_objective_on_semantic_collapse(self):
        runtime = ScientificToolRuntime.__new__(ScientificToolRuntime)
        prediction = _multiclass_prediction()
        prediction["details"].update(
            {
                "interaction_type_class": 73,
                "interaction_type_description": (
                    "The serum concentration of ketoconazole can be increased when it is combined with simvastatin."
                ),
            }
        )
        report = {
            field: None
            for field in [
                "request",
                "task_type",
                "molecules",
                "predictions",
                "evidence",
                "explanation",
                "artifacts",
                "warnings",
                "trace",
                "reasoning",
            ]
        }
        report["explanation"] = "simvastatin + ketoconazole: no clear DDI risk signal."
        context = {
            "task_type": "ddi_prediction",
            "predictions": [prediction],
            "confidence": {"low_confidence": False},
            "evidence": [],
            "metadata": {"min_evidence": 0, "enable_reflection": False},
            "reflection": {},
            "final_report": report,
            "verifier_report": {
                "prediction": {"prediction_grounded": True},
                "citations": {"citation_grounded": True, "n_unsupported": 0},
            },
            "warnings": [],
        }

        runtime.verify_report_schema({}, context)

        self.assertEqual(context["objective_status"]["status"], "fail")
        self.assertIn("prediction_semantics_preserved", context["objective_status"]["missing_objectives"])
        self.assertEqual(context["verifier_report"]["summary"]["status"], "fail")


if __name__ == "__main__":
    unittest.main()
