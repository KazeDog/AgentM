"""Unavailable LLMs must not be revived by deterministic protocol gates."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from agentm.runtime import AgentMMultiAgentController
from agentm.runtime.agents.orchestrator import LLMOrchestratorPlanner, OrchestratorAgent
from agentm.runtime.state import initial_context

from .test_orchestrator_factorial import _registry, _WriteArtifactsClient


class UnavailableClient:
    available = False

    def __init__(self):
        self.calls = 0

    def chat_json(self, *args, **kwargs):
        self.calls += 1
        raise AssertionError("Unavailable client must never be called")


class RaisingClient(UnavailableClient):
    available = True

    def chat_json(self, *args, **kwargs):
        self.calls += 1
        raise RuntimeError("synthetic API unavailable")


class LLMTerminalStopTests(unittest.TestCase):
    def context(self, *, required=True, fallbacks=False, reflection=True):
        return initial_context(
            "aspirin DILI", task_type="admet_prediction", dataset_name="DILI",
            require_llm_orchestrator=required, enable_fallbacks=fallbacks,
            enable_reflection=reflection, max_reflection_rounds=3,
        )

    def assert_stopped(self, action, context):
        self.assertEqual((action.agent, action.tool), ("OrchestratorAgent", "finish"))
        self.assertEqual(context["metadata"]["last_action_source"], "llm_failed_no_fallback")
        self.assertEqual(context["metadata"]["planner_type"], "llm_failed")
        self.assertFalse(context["llm_plan_proposals"][-1]["fallback_used"])
        self.assertFalse(any(row.get("status") == "factorial_protocol_gate_override"
                             for row in context["llm_plan_proposals"]))

    def test_unavailable_required_llm_stops_for_both_reflection_and_fallback_flags(self):
        for reflection in (False, True):
            for fallbacks in (False, True):
                with self.subTest(reflection=reflection, fallbacks=fallbacks):
                    client = UnavailableClient()
                    orchestrator = OrchestratorAgent(registry=_registry(), llm_client=client)
                    context = self.context(reflection=reflection, fallbacks=fallbacks)
                    orchestrator.build_plan(context)
                    self.assert_stopped(orchestrator.next_action(context), context)
                    self.assertEqual(client.calls, 0)

    def test_pending_reflection_does_not_override_unavailable_llm(self):
        client = UnavailableClient()
        orchestrator = OrchestratorAgent(registry=_registry(), llm_client=client)
        context = self.context()
        context["metadata"]["reflection_rounds"] = 1
        context["metadata"]["reflection_feedback_applied_round"] = 0
        context["reflection"] = {"round": 1, "recommendation": "finalize"}
        with patch.object(orchestrator, "_pending_reflection_protocol_action",
                          side_effect=AssertionError("Must stop before reflection gate")):
            self.assert_stopped(orchestrator.next_action(context), context)
        self.assertEqual(context["metadata"]["reflection_feedback_applied_round"], 0)
        self.assertEqual(client.calls, 0)

    def test_planner_stops_before_safety_composition_or_evidence_policy(self):
        planner = LLMOrchestratorPlanner(registry=_registry(), llm_client=UnavailableClient())
        context = self.context()
        with patch.object(planner.fallback, "safe_abstention_action", side_effect=AssertionError("policy executed")), \
             patch.object(planner.fallback, "open_task_action", side_effect=AssertionError("contract executed")), \
             patch.object(planner.fallback, "no_direct_evidence_action", side_effect=AssertionError("evidence policy executed")):
            self.assert_stopped(planner.next_action(context), context)

    def test_runtime_llm_exception_stops_without_reflection_resurrection(self):
        client = RaisingClient()
        orchestrator = OrchestratorAgent(registry=_registry(), llm_client=client)
        context = self.context()
        self.assert_stopped(orchestrator.next_action(context), context)
        self.assertEqual(client.calls, 1)
        self.assert_stopped(orchestrator.next_action(context), context)
        self.assertEqual(client.calls, 1)

    def test_initial_plan_exception_stops_before_any_protocol_step(self):
        client = RaisingClient()
        orchestrator = OrchestratorAgent(registry=_registry(), llm_client=client)
        context = self.context()
        self.assertEqual(orchestrator.build_plan(context)["status"], "failed")
        self.assert_stopped(orchestrator.next_action(context), context)
        self.assertEqual(client.calls, 1)

    def test_optional_llm_with_fallback_disabled_also_stops(self):
        context = self.context(required=False, fallbacks=False)
        orchestrator = OrchestratorAgent(registry=_registry(), llm_client=UnavailableClient())
        self.assert_stopped(orchestrator.next_action(context), context)

    def test_explicitly_allowed_fallback_remains_available(self):
        context = self.context(required=False, fallbacks=True)
        orchestrator = OrchestratorAgent(registry=_registry(), llm_client=UnavailableClient())
        action = orchestrator.next_action(context)
        self.assertEqual((action.agent, action.tool), ("MoleculeAgent", "route_task"))
        self.assertEqual(context["metadata"]["last_action_source"], "heuristic_fallback")

    def test_healthy_llm_still_cannot_skip_reflection_prerequisites(self):
        client = _WriteArtifactsClient()
        context = self.context()
        orchestrator = OrchestratorAgent(registry=_registry(), llm_client=client)
        action = orchestrator.next_action(context)
        self.assertNotEqual(action.tool, "finish")
        self.assertEqual(context["metadata"]["last_action_source"], "factorial_protocol_gate")
        self.assertEqual(client.calls, 1)

    def test_controller_only_writes_failure_artifacts_at_large_round_limit(self):
        for reflection in (False, True):
            with self.subTest(reflection=reflection), tempfile.TemporaryDirectory() as tmp:
                client = UnavailableClient()
                controller = AgentMMultiAgentController(
                    output_dir=tmp, feature_set="rdkit_only", max_rounds=20,
                    orchestrator_llm_client=client, require_llm_orchestrator=True,
                    enable_fallbacks=True, enable_reflection=reflection,
                    max_reflection_rounds=3, use_llm_reasoning=False,
                    use_rag_evidence=False, use_external_evidence=False,
                )
                report = controller.run("aspirin DILI", task_type="admet_prediction", dataset_name="DILI")
                self.assertFalse(report["predictions"])
                self.assertEqual(report["multi_agent"]["planner_type"], "llm_failed")
                self.assertTrue(report["tool_calls"])
                self.assertTrue(all(call["tool"] == "write_artifacts" for call in report["tool_calls"]))
                self.assertEqual(client.calls, 0)
                artifact = Path(tmp) / "multi_agent_report.json"
                self.assertTrue(artifact.exists())
                saved = json.loads(artifact.read_text())
                self.assertFalse(saved["predictions"])
                self.assertEqual(saved["multi_agent"]["planner_type"], "llm_failed")
                self.assertTrue(any("llm_task_graph_planning_failed_required" in warning
                                    for warning in saved["warnings"]))


if __name__ == "__main__":
    unittest.main()
