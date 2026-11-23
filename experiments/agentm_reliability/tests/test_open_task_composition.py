from __future__ import annotations

import tempfile
import unittest

from agentm.runtime.agents.orchestrator import DeterministicMultiAgentPlanner, LLMOrchestratorPlanner
from agentm.runtime.state import initial_context
from agentm.runtime.tools.registry import MultiAgentToolRegistry
from agentm.runtime.tools.scientific import ScientificToolRuntime
from experiments.agentm_reliability.offline_evaluator import evaluate_open_task
from experiments.agentm_reliability.run_matrix import _open_task_contract


def ddi_case() -> dict:
    return {
        "case_id": "open_ddi_test",
        "dataset_group": "open_task",
        "open_task_type": "ddi_plus_individual_safety",
        "request": (
            "Evaluate the DDI risk for simvastatin and ketoconazole; also assess hERG risk for "
            "simvastatin and DILI risk for ketoconazole. Keep pair-level and single-drug evidence separate."
        ),
        "task_type": "ddi_prediction",
        "dataset_name": "DDI",
        "base_case": {
            "request": "simvastatin, ketoconazole",
            "task_type": "ddi_prediction",
            "dataset_name": "DDI",
        },
        "gold_plan": {
            "required_objectives": [
                "predict:ddi",
                "predict:simvastatin:herg",
                "predict:ketoconazole:dili",
                "separate_pair_and_single_evidence",
            ],
            "required_tools": [
                "resolve_molecule",
                "predict_ddi",
                "predict_single",
                "retrieve_elasticsearch_hybrid",
                "verify_citation_grounding",
            ],
            "required_dependencies": [
                ["resolve_molecule", "predict_ddi"],
                ["predict_ddi", "verify_prediction_grounding"],
            ],
        },
    }


class OpenTaskCompositionTests(unittest.TestCase):
    def test_contract_contains_request_subtasks_but_no_gold(self) -> None:
        contract = _open_task_contract(ddi_case())
        self.assertEqual(contract["open_task_type"], "ddi_plus_individual_safety")
        self.assertEqual([item["kind"] for item in contract["subtasks"]], ["single", "single", "ddi"])
        self.assertEqual([item["dataset_name"] for item in contract["subtasks"]], ["hERG", "DILI", "DDI"])
        self.assertNotIn("gold_plan", contract)
        self.assertNotIn("required_objectives", str(contract))

    def test_heuristic_contract_completes_three_scoped_predictions_within_budget(self) -> None:
        contract = _open_task_contract(ddi_case())
        context = initial_context(
            ddi_case()["request"],
            task_type="ddi_prediction",
            dataset_name="DDI",
            max_rounds=32,
            enable_reflection=False,
        )
        context["metadata"]["task_contract"] = contract
        planner = DeterministicMultiAgentPlanner()
        actions = []
        for _ in range(25):
            action = planner.next_action(context)
            if action.tool == "finish":
                break
            actions.append(action)
            if action.tool in {"predict_single", "predict_ddi"}:
                context.setdefault("predictions", []).append(
                    {
                        "task_type": action.args["task_type"],
                        "dataset_name": action.args["dataset_name"],
                        "label": 1,
                        "model_type": "test",
                        "details": {"open_task_subtask_id": action.args["subtask_id"]},
                    }
                )
            if action.tool == "build_runtime_report":
                context["final_report"] = {"request": context["request"]}
            context.setdefault("tool_calls", []).append(
                {
                    "agent": action.agent,
                    "tool": action.tool,
                    "args": dict(action.args),
                    "status": "completed",
                }
            )

        self.assertLessEqual(len(actions), 22)
        scoped_predictions = [action for action in actions if action.tool in {"predict_single", "predict_ddi"}]
        self.assertEqual([action.tool for action in scoped_predictions], ["predict_single", "predict_single", "predict_ddi"])
        self.assertTrue(all(action.args.get("subtask_id") for action in scoped_predictions))
        self.assertEqual(actions[-1].tool, "write_artifacts")

    def test_llm_path_does_not_finalize_after_first_scoped_empty_retrieval(self) -> None:
        class NeverCallClient:
            available = True

            def __init__(self) -> None:
                self.calls = 0

            def chat_json(self, messages, temperature=0.0):
                self.calls += 1
                raise AssertionError("request-derived composite contract should select the next scoped action")

        contract = _open_task_contract(ddi_case())
        context = initial_context(
            ddi_case()["request"],
            task_type="ddi_prediction",
            dataset_name="DDI",
            max_rounds=32,
            enable_reflection=False,
            require_llm_orchestrator=True,
        )
        context["metadata"]["task_contract"] = contract
        context["metadata"]["retrieval_outcome"] = "no_direct_evidence"
        context["predictions"] = [
            {
                "task_type": "admet_prediction",
                "dataset_name": "hERG",
                "label": 0,
                "model_type": "test",
                "details": {"open_task_subtask_id": "single::simvastatin::hERG"},
            }
        ]
        first_subtask = "single::simvastatin::hERG"
        context["tool_calls"] = [
            {"agent": "MoleculeAgent", "tool": "route_task", "status": "completed", "args": {}},
            *[
                {
                    "agent": "TestAgent",
                    "tool": tool,
                    "status": "completed",
                    "args": {"subtask_id": first_subtask},
                }
                for tool in (
                    "resolve_molecule",
                    "build_features",
                    "predict_single",
                    "assess_confidence",
                    "retrieve_elasticsearch_hybrid",
                )
            ],
        ]
        client = NeverCallClient()
        planner = LLMOrchestratorPlanner(
            registry=MultiAgentToolRegistry(),
            llm_client=client,
            temperature=0.0,
        )

        action = planner.next_action(context)

        self.assertEqual((action.agent, action.tool), ("MoleculeAgent", "resolve_molecule"))
        self.assertEqual(action.args.get("subtask_id"), "single::ketoconazole::DILI")
        self.assertEqual(client.calls, 0)
        self.assertEqual(context["metadata"]["last_action_source"], "open_task_contract")
        self.assertEqual(context["llm_plan_proposals"][-1]["status"], "open_task_contract_override")
        self.assertFalse(context["llm_plan_proposals"][-1]["fallback_used"])

    def test_boundary_policy_remains_inactive_until_composite_science_is_complete(self) -> None:
        contract = _open_task_contract(ddi_case())
        context = initial_context(
            ddi_case()["request"],
            task_type="ddi_prediction",
            dataset_name="DDI",
        )
        context["metadata"]["task_contract"] = contract
        context["metadata"]["retrieval_outcome"] = "no_direct_evidence"
        context["predictions"] = [{"label": 0, "model_type": "test"}]
        context["tool_calls"] = [
            {
                "tool": "retrieve_elasticsearch_hybrid",
                "status": "completed",
                "args": {"subtask_id": "single::simvastatin::hERG"},
            }
        ]

        action = DeterministicMultiAgentPlanner().no_direct_evidence_action(context)

        self.assertIsNone(action)

    def test_runtime_keeps_predictions_and_evidence_scoped_then_builds_composite_narrative(self) -> None:
        contract = _open_task_contract(ddi_case())
        with tempfile.TemporaryDirectory() as tmp:
            runtime = ScientificToolRuntime(output_dir=tmp, feature_set="rdkit_only")
            context = initial_context(ddi_case()["request"], task_type="ddi_prediction", dataset_name="DDI")
            context["metadata"]["task_contract"] = contract
            for index, subtask in enumerate(contract["subtasks"]):
                args = dict(subtask)
                runtime._activate_open_subtask(args, context, expected_kind=subtask["kind"])
                prediction = {
                    "task_type": subtask["task_type"],
                    "dataset_name": subtask["dataset_name"],
                    "label": 1 if index != 1 else 0,
                    "score": 0.8 if index != 1 else 0.2,
                    "probability": 0.8 if index != 1 else 0.2,
                    "model_type": "test_model",
                    "details": {"positive_label": 1, "negative_label": 0},
                }
                runtime._append_open_task_prediction(context, prediction)
                runtime._append_evidence(
                    context,
                    [
                        {
                            "citation_id": "cite_1",
                            "chunk_id": f"chunk_{index}",
                            "source": "test",
                            "source_path": f"source://{index}",
                            "claim": f"{subtask['dataset_name']} evidence for {' '.join(subtask['molecules'])}",
                            "compound_specific": True,
                            "endpoint_specific": True,
                        }
                    ],
                )
            result = runtime.build_structured_explanation({}, context)

        self.assertEqual(result.status, "completed")
        self.assertEqual(len(context["predictions"]), 3)
        self.assertEqual(len({item["citation_id"] for item in context["evidence"]}), 3)
        self.assertTrue(all(item.get("open_task_subtask_id") for item in context["evidence"]))
        narrative = context["reasoning"]["structured_explanation"]
        self.assertIn("Pair-level evidence and single-drug evidence are kept separate", narrative)
        self.assertIn("All requested prediction subtasks were completed", narrative)

        calls = []
        for subtask in contract["subtasks"]:
            resolve = "resolve_ddi_pair" if subtask["kind"] == "ddi" else "resolve_molecule"
            predict = "predict_ddi" if subtask["kind"] == "ddi" else "predict_single"
            for tool in [resolve, "build_features", predict, "retrieve_elasticsearch_hybrid"]:
                calls.append({"tool": tool, "status": "completed", "args": {"subtask_id": subtask["subtask_id"]}})
        calls.extend(
            [
                {"tool": "verify_prediction_grounding", "status": "completed"},
                {"tool": "verify_citation_grounding", "status": "completed"},
            ]
        )
        context["tool_calls"] = calls
        evaluation = evaluate_open_task(case=ddi_case(), context=context, report={"explanation": narrative})
        self.assertEqual(evaluation["objective_coverage"], 1.0)
        self.assertEqual(evaluation["tool_selection_accuracy"], 1.0)
        self.assertEqual(evaluation["dependency_order_accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
