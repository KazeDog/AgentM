from __future__ import annotations

import unittest

from agentm.runtime.controller import AgentMMultiAgentController


class _Runtime:
    def __init__(self) -> None:
        self.resource_updates = 0

    def _update_resource_resolution(self, context):
        self.resource_updates += 1
        reaction = {
            "all_required_present": True,
            "data_root": "/read-only/reaction-data",
            "read_only_source_root": True,
        }
        context["reaction_resource_resolution"] = reaction
        context.setdefault("resource_resolution", {})["reaction"] = reaction
        context.setdefault("artifacts", {})["resource_resolution"] = context["resource_resolution"]


class _Graph:
    def __init__(self) -> None:
        self.context = None

    def invoke(self, context):
        self.context = context
        return {"final_report": context}


def _controller() -> AgentMMultiAgentController:
    controller = object.__new__(AgentMMultiAgentController)
    controller.mode = "full"
    controller.retrieval_backend = "es"
    controller.max_rounds = 20
    controller.llm_profile = "test"
    controller.use_llm_orchestrator = True
    controller.enable_fallbacks = False
    controller.enable_reflection = False
    controller.max_reflection_rounds = 0
    controller.auto_model = False
    controller.model_registry_path = None
    controller.require_real_model = True
    controller.require_real_rag = True
    controller.require_llm_orchestrator = True
    controller.llm_orchestrator_max_retries = 0
    controller.llm_orchestrator_retry_backoff_seconds = 0.0
    controller.use_llm_reasoning = False
    controller.require_llm_reasoning = False
    controller.use_rag_evidence = True
    controller.min_evidence = 1
    controller.use_external_evidence = False
    controller.previous_run_path = None
    controller.runtime = _Runtime()
    controller.graph = _Graph()
    return controller


class ControllerResourceInitializationTests(unittest.TestCase):
    def test_explicit_reaction_task_persists_resources_before_graph_planning(self) -> None:
        controller = _controller()
        report = controller.run(
            "CCBr.CN>>CCN",
            task_type="reaction_profile",
            dataset_name="reaction",
            task_contract={"schema": "test_contract", "open_task_type": "reaction_multi_tool"},
        )
        self.assertEqual(controller.runtime.resource_updates, 1)
        self.assertTrue(report["reaction_resource_resolution"]["all_required_present"])
        self.assertTrue(report["resource_resolution"]["reaction"]["all_required_present"])
        self.assertTrue(report["artifacts"]["resource_resolution"]["reaction"]["all_required_present"])

    def test_nonreaction_task_does_not_eagerly_resolve_scientific_resources(self) -> None:
        controller = _controller()
        report = controller.run(
            "aspirin DILI",
            task_type="admet_prediction",
            dataset_name="DILI",
        )
        self.assertEqual(controller.runtime.resource_updates, 0)
        self.assertEqual(report["reaction_resource_resolution"], {})
        self.assertEqual(report["resource_resolution"], {})


if __name__ == "__main__":
    unittest.main()
