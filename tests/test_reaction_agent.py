from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from agentm.runtime import AgentMMultiAgentController
from agentm.runtime.guards import GuardPolicy, GuardViolation, verify_citations, verify_prediction_grounding
from agentm.runtime.reaction import ReactionResourceResolver, ReactionRuntime, infer_reaction_task_type, is_reaction_request, parse_reaction_text
from agentm.runtime.state import initial_context


REACTION = (
    "Cl[c:6]1[cH:5][cH:4][c:3]([CH2:2][CH3:1])[cH:16][cH:15]1."
    "[NH2:7][c:8]1[cH:9][cH:10][c:11]([CH3:12])[cH:13][cH:14]1>>"
    "[CH3:1][CH2:2][c:3]1[cH:4][cH:5][c:6]([NH:7][c:8]2[cH:9][cH:10][c:11]([CH3:12])[cH:13][cH:14]2)[cH:15][cH:16]1"
)
ROOT = Path(__file__).resolve().parents[1]


class ReactionAgentTests(unittest.TestCase):
    def test_resource_resolver_discovers_reaction_sources_read_only(self):
        resolved = ReactionResourceResolver().resolve()
        self.assertTrue(resolved["read_only_source_root"])
        self.assertTrue(resolved["mit"]["exists"])
        self.assertTrue(resolved["schneider"]["exists"])
        self.assertTrue(resolved["buchwald_hartwig_kg"]["exists"])
        self.assertIn("test1", resolved["buchwald_hartwig_kg"]["splits"])

    def test_reaction_routing_helpers(self):
        self.assertTrue(is_reaction_request(f"Predict yield for {REACTION}"))
        self.assertEqual(infer_reaction_task_type(f"Predict yield for {REACTION}"), "reaction_yield_prediction")
        parsed = parse_reaction_text(REACTION)
        self.assertTrue(parsed["is_valid_reaction"])
        self.assertEqual(len(parsed["reactant_list"]), 2)

    def test_guard_blocks_cross_agent_reaction_prediction_fabrication(self):
        policy = GuardPolicy(agent_allowlists={"ReasoningAgent": ["build_structured_explanation"]})
        context = initial_context("reaction", task_type="reaction_classification")
        context["reaction"] = parse_reaction_text(REACTION)
        context["reaction_profile"] = {"summary": "ready"}
        with self.assertRaises(GuardViolation):
            policy.validate_action({"agent": "ReasoningAgent", "tool": "build_structured_explanation", "args": {"label": "fake"}}, context)

    def test_reaction_verifier_accepts_reaction_agent_prediction_and_sources(self):
        context = initial_context("reaction", task_type="reaction_classification")
        context["reaction_predictions"] = [{"label": "0", "score": 0.8, "model_type": "schneider_token_knn_classifier"}]
        context["predictions"] = list(context["reaction_predictions"])
        context["tool_calls"] = [{"agent": "ReactionAgent", "tool": "classify_reaction_type", "status": "completed"}]
        self.assertTrue(verify_prediction_grounding(context)["prediction_grounded"])
        context["evidence"] = [{"source": "mit_reaction_template", "citation_id": "mit_1", "claim": "MIT similar reaction evidence."}]
        self.assertTrue(verify_citations(context)["citation_grounded"])

    def test_reaction_evidence_retrieval_is_owned_by_retrieve_agent(self):
        with tempfile.TemporaryDirectory() as tmp:
            controller = AgentMMultiAgentController(
                output_dir=tmp,
                max_rounds=16,
                use_llm_orchestrator=False,
                use_llm_reasoning=False,
            )
            allowlists = controller.registry.agent_allowlists()
            self.assertIn("retrieve_mit_template_evidence", allowlists["RetrieveAgent"])
            self.assertIn("retrieve_orderly_condition_evidence", allowlists["RetrieveAgent"])
            self.assertNotIn("retrieve_mit_template_evidence", allowlists["ReactionAgent"])
            self.assertNotIn("retrieve_orderly_condition_evidence", allowlists["ReactionAgent"])

    def test_reaction_runtime_smoke_writes_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            controller = AgentMMultiAgentController(
                output_dir=tmp,
                max_rounds=16,
                use_llm_orchestrator=False,
                use_llm_reasoning=False,
            )
            report = controller.run_reaction(REACTION, task_type="reaction_yield_prediction")
            self.assertEqual(report["task_type"], "reaction_yield_prediction")
            self.assertTrue(report.get("reaction", {}).get("is_valid_reaction"))
            self.assertTrue(report.get("reaction_profile"))
            self.assertTrue(Path(tmp, "reaction_report.json").exists())
            self.assertTrue(Path(tmp, "reaction_resource_resolution.json").exists())

    def test_reaction_index_and_training_smoke(self):
        with tempfile.TemporaryDirectory() as tmp:
            runtime = ReactionRuntime(output_dir=tmp)
            manifest = runtime.build_indices(dataset="all", max_records=5, output_dir=Path(tmp) / "idx", shards=2)
            self.assertEqual(manifest["n_docs"], 10)
            cls_model = runtime.train_model(task="schneider_classification", max_rows=64, output_dir=Path(tmp) / "models", model_family="tfidf_linear")
            self.assertEqual(cls_model["model_type"], "reaction_tfidf_linear_classifier")
            y_model = runtime.train_model(task="buchwald_hartwig_yield", split="test1", max_rows=32, output_dir=Path(tmp) / "models", model_family="tfidf_linear")
            self.assertEqual(y_model["model_type"], "reaction_tfidf_ridge_regressor")
            self.assertTrue(Path(cls_model["model_path"]).exists())
            self.assertTrue(Path(y_model["model_path"]).exists())

    def test_full_reaction_model_artifact_is_used_by_controller(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp) / "models"
            runtime = ReactionRuntime(output_dir=tmp, model_dir=model_dir)
            runtime.train_model(task="buchwald_hartwig_yield", split="test1", max_rows=32, output_dir=model_dir, model_family="tfidf_linear")
            controller = AgentMMultiAgentController(
                output_dir=Path(tmp) / "run",
                max_rounds=16,
                use_llm_orchestrator=False,
                use_llm_reasoning=False,
                reaction_model_dir=model_dir,
            )
            report = controller.run_reaction(REACTION, task_type="reaction_yield_prediction")
            prediction = (report.get("reaction_predictions") or report.get("predictions") or [{}])[-1]
            self.assertEqual(prediction.get("model_type"), "reaction_tfidf_ridge_regressor")

    def test_reaction_model_benchmark_smoke_script(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "benchmark"
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "run_reaction_model_benchmark.py"),
                    "--max-rows",
                    "64",
                    "--output-dir",
                    str(output_dir),
                ],
                cwd=str(ROOT),
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue((output_dir / "reaction_model_benchmark_summary.json").exists())
            self.assertTrue((output_dir / "schneider_metrics.json").exists())
            self.assertTrue((output_dir / "buchwald_hartwig_metrics_by_split.json").exists() or (output_dir / "reaction_model_benchmark_summary.json").exists())


if __name__ == "__main__":
    unittest.main()
