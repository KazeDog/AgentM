from __future__ import annotations

import json
import unittest

from agentm.runtime.agents.orchestrator import LLMOrchestratorPlanner, OrchestratorAgent
from agentm.runtime.state import initial_context
from agentm.runtime.tools.registry import MultiAgentToolRegistry, ToolResult, ToolSpec
from agentm.runtime.tools.scientific import ScientificToolRuntime


class _ReflectionOnlyClient:
    available = True

    def chat_json(self, messages, temperature=0.0):
        return json.dumps(
            {
                "agent": "ReflectionAgent",
                "tool": "reflect_on_report",
                "args": {},
                "rationale": "deliberate condition violation",
            }
        )


class _WriteArtifactsClient:
    available = True

    def __init__(self) -> None:
        self.calls = 0

    def chat_json(self, messages, temperature=0.0):
        self.calls += 1
        return json.dumps(
            {
                "agent": "ReporterAgent",
                "tool": "write_artifacts",
                "args": {},
                "rationale": "finish immediately",
            }
        )


def _registry() -> MultiAgentToolRegistry:
    registry = MultiAgentToolRegistry()
    for name, agent in [
        ("write_artifacts", "ReporterAgent"),
        ("verify_report_schema", "VerifierAgent"),
        ("reflect_on_report", "ReflectionAgent"),
        ("apply_reflection_feedback", "ReflectionAgent"),
    ]:
        registry.register(ToolSpec(name, agent, name, {}, lambda args, context, tool=name: ToolResult(tool, "completed")))
    return registry


class OrchestratorFactorialConditionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.context = initial_context(
            "aspirin DILI",
            task_type="admet_prediction",
            dataset_name="DILI",
            enable_reflection=False,
            max_reflection_rounds=1,
        )
        self.planner = LLMOrchestratorPlanner(registry=_registry(), llm_client=_ReflectionOnlyClient(), temperature=0.0)

    def test_reflection_tools_are_hidden_and_protocol_omits_reflection(self) -> None:
        messages = self.planner._messages(self.context)
        payload = json.loads(messages[1]["content"])
        self.assertFalse(any(schema["agent"] == "ReflectionAgent" for schema in payload["tool_schemas"]))
        self.assertFalse(any("reflect" in step.lower() for step in payload["context"]["required_protocol"]))

    def test_plan_normalization_removes_reflection_nodes_and_edges(self) -> None:
        payload = {
            "task_graph": {
                "nodes": [
                    {"id": "verify", "agent": "VerifierAgent", "goal": "verify"},
                    {"id": "reflect", "agent": "ReflectionAgent", "goal": "reflect"},
                    {"id": "write", "agent": "ReporterAgent", "goal": "write"},
                ],
                "edges": [["verify", "reflect"], ["reflect", "write"]],
            },
            "stop_conditions": ["verifier_pass", "reflection_applied"],
        }
        plan = self.planner._normalize_plan(payload, self.context)
        self.assertNotIn("ReflectionAgent", {node["agent"] for node in plan["task_graph"]["nodes"]})
        self.assertEqual(plan["task_graph"]["edges"], [])
        self.assertNotIn("reflection_applied", plan["stop_conditions"])

    def test_runtime_condition_gate_never_executes_reflection(self) -> None:
        self.context["final_report"] = {"status": "ready"}
        self.context["tool_calls"] = [{"agent": "VerifierAgent", "tool": "verify_report_schema", "status": "completed"}]
        action = self.planner.next_action(self.context)
        self.assertEqual((action.agent, action.tool), ("ReporterAgent", "write_artifacts"))
        self.assertEqual(self.context["metadata"]["last_action_source"], "factorial_condition_gate")
        self.assertFalse(self.context["llm_plan_proposals"][-1]["fallback_used"])

    def test_routed_graph_omits_reflection_when_disabled(self) -> None:
        runtime = object.__new__(ScientificToolRuntime)
        graph = runtime._task_graph_for(self.context)
        self.assertNotIn("ReflectionAgent", {node["agent"] for node in graph["nodes"]})

    def _verified_reflection_context(self):
        context = initial_context(
            "aspirin DILI",
            task_type="admet_prediction",
            dataset_name="DILI",
            enable_reflection=True,
            max_reflection_rounds=1,
        )
        context["molecules"] = [{"name": "aspirin", "is_valid": True}]
        context["features"] = [{"feature_set": "test"}]
        context["predictions"] = [{"label": 1, "model_type": "test"}]
        context["confidence"] = {"low_confidence": False}
        context["evidence"] = [{"citation_id": "C1", "chunk_id": "K1"}]
        context["final_report"] = {"task_type": "admet_prediction", "predictions": context["predictions"]}
        context["verifier_report"] = {
            "prediction": {"prediction_grounded": True},
            "citations": {"citation_grounded": True},
            "schema": {"schema_complete": True},
            "summary": {"status": "pass"},
        }
        context["tool_calls"] = [
            {"agent": "MoleculeAgent", "tool": "route_task", "status": "completed"},
            {"agent": "MoleculeAgent", "tool": "resolve_molecule", "status": "completed"},
            {"agent": "GaugeAgent", "tool": "discover_rules", "status": "completed"},
            {"agent": "GaugeAgent", "tool": "build_features", "status": "completed"},
            {"agent": "GaugeAgent", "tool": "predict_single", "status": "completed"},
            {"agent": "GaugeAgent", "tool": "assess_confidence", "status": "completed"},
            {"agent": "RetrieveAgent", "tool": "retrieve_elasticsearch_hybrid", "status": "completed"},
            {"agent": "ReasoningAgent", "tool": "build_structured_explanation", "status": "completed"},
            {"agent": "ReporterAgent", "tool": "build_runtime_report", "status": "completed"},
            {"agent": "VerifierAgent", "tool": "verify_prediction_grounding", "status": "completed"},
            {"agent": "VerifierAgent", "tool": "verify_citation_grounding", "status": "completed"},
            {"agent": "VerifierAgent", "tool": "verify_report_schema", "status": "completed"},
        ]
        return context

    def test_reflection_on_blocks_early_artifact_finalization(self) -> None:
        client = _WriteArtifactsClient()
        orchestrator = OrchestratorAgent(
            registry=_registry(),
            llm_client=client,
            use_llm=True,
            temperature=0.0,
        )
        context = self._verified_reflection_context()

        action = orchestrator.next_action(context)

        self.assertEqual((action.agent, action.tool), ("ReflectionAgent", "reflect_on_report"))
        self.assertEqual(client.calls, 1)
        self.assertEqual(context["metadata"]["last_action_source"], "factorial_protocol_gate")
        self.assertEqual(context["llm_plan_proposals"][-1]["status"], "factorial_protocol_gate_override")
        self.assertFalse(context["llm_plan_proposals"][-1]["fallback_used"])

    def test_started_reflection_forces_feedback_application_without_llm_call(self) -> None:
        client = _WriteArtifactsClient()
        orchestrator = OrchestratorAgent(
            registry=_registry(),
            llm_client=client,
            use_llm=True,
            temperature=0.0,
        )
        context = self._verified_reflection_context()
        context["reflection"] = {"round": 1, "recommendation": "finalize"}
        context["metadata"]["reflection_rounds"] = 1
        context["tool_calls"].append(
            {"agent": "ReflectionAgent", "tool": "reflect_on_report", "status": "completed"}
        )

        action = orchestrator.next_action(context)

        self.assertEqual((action.agent, action.tool), ("ReflectionAgent", "apply_reflection_feedback"))
        self.assertEqual(client.calls, 0)
        self.assertEqual(context["metadata"]["last_action_source"], "factorial_protocol_gate")
        self.assertFalse(context["llm_plan_proposals"][-1]["fallback_used"])


if __name__ == "__main__":
    unittest.main()
