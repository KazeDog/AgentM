from __future__ import annotations

import json
import tempfile
import unittest

from agentm.runtime.agents.orchestrator import DeterministicMultiAgentPlanner, LLMOrchestratorPlanner
from agentm.runtime.graph.nodes import MultiAgentGraphNodes
from agentm.runtime.guards.policies import GuardPolicy, GuardViolation
from agentm.runtime.state import initial_context
from agentm.runtime.tools.registry import AgentAction, MultiAgentToolRegistry, ToolResult, ToolSpec


class _SequenceClient:
    available = True

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def chat_json(self, messages, temperature=0.0):
        self.calls += 1
        value = self.responses.pop(0)
        if isinstance(value, Exception):
            raise value
        return json.dumps(value)


class _NeverCallClient:
    available = True

    def __init__(self):
        self.calls = 0

    def chat_json(self, messages, temperature=0.0):
        self.calls += 1
        raise AssertionError("safety policy must not call the LLM")


def _registry() -> MultiAgentToolRegistry:
    registry = MultiAgentToolRegistry()
    for tool, agent in (
        ("route_task", "MoleculeAgent"),
        ("predict_single", "GaugeAgent"),
        ("retrieve_elasticsearch_hybrid", "RetrieveAgent"),
        ("build_structured_explanation", "ReasoningAgent"),
        ("verify_citation_grounding", "VerifierAgent"),
        ("verify_report_schema", "VerifierAgent"),
        ("reflect_on_report", "ReflectionAgent"),
        ("apply_reflection_feedback", "ReflectionAgent"),
        ("build_runtime_report", "ReporterAgent"),
        ("write_artifacts", "ReporterAgent"),
    ):
        registry.register(
            ToolSpec(
                tool,
                agent,
                tool,
                {},
                lambda args, context, name=tool: ToolResult(name, "completed"),
            )
        )
    return registry


class LLMRetryAndSafeAbstentionTests(unittest.TestCase):
    def test_empty_action_payload_retries_before_accepting_valid_action(self) -> None:
        client = _SequenceClient(
            [
                {},
                {
                    "agent": "ReporterAgent",
                    "tool": "write_artifacts",
                    "args": {},
                    "rationale": "continue",
                },
            ]
        )
        context = initial_context(
            "aspirin DILI",
            task_type="admet_prediction",
            dataset_name="DILI",
            enable_fallbacks=False,
            require_llm_orchestrator=True,
        )
        planner = LLMOrchestratorPlanner(
            registry=_registry(),
            llm_client=client,
            max_llm_retries=2,
            retry_backoff_seconds=0,
        )

        action = planner.next_action(context)

        self.assertEqual((action.agent, action.tool), ("ReporterAgent", "write_artifacts"))
        self.assertEqual(client.calls, 2)
        self.assertEqual([row["status"] for row in context["llm_retry_trace"]], ["failed", "completed"])
        self.assertIn("requires non-empty string agent and tool", context["llm_retry_trace"][0]["error"])
        self.assertEqual(context["metadata"]["last_action_source"], "llm")
        self.assertFalse(context["llm_plan_proposals"][-1]["fallback_used"])

    def test_nested_empty_action_payload_retries(self) -> None:
        client = _SequenceClient(
            [
                {"action": {}},
                {"action": {"agent": "ReporterAgent", "tool": "write_artifacts", "args": {}}},
            ]
        )
        context = initial_context(
            "aspirin DILI",
            task_type="admet_prediction",
            dataset_name="DILI",
            enable_fallbacks=False,
            require_llm_orchestrator=True,
        )
        planner = LLMOrchestratorPlanner(
            registry=_registry(),
            llm_client=client,
            max_llm_retries=2,
            retry_backoff_seconds=0,
        )

        action = planner.next_action(context)

        self.assertEqual((action.agent, action.tool), ("ReporterAgent", "write_artifacts"))
        self.assertEqual(client.calls, 2)
        self.assertEqual([row["status"] for row in context["llm_retry_trace"]], ["failed", "completed"])

    def test_invalid_action_retry_exhaustion_fails_closed_without_fallback(self) -> None:
        client = _SequenceClient([{}, {"agent": "ReporterAgent"}, {"action": {"tool": "write_artifacts"}}])
        context = initial_context(
            "aspirin DILI",
            task_type="admet_prediction",
            dataset_name="DILI",
            enable_fallbacks=False,
            require_llm_orchestrator=True,
        )
        planner = LLMOrchestratorPlanner(
            registry=_registry(),
            llm_client=client,
            max_llm_retries=2,
            retry_backoff_seconds=0,
        )

        action = planner.next_action(context)

        self.assertEqual((action.agent, action.tool), ("OrchestratorAgent", "finish"))
        self.assertEqual(client.calls, 3)
        self.assertTrue(all(row["status"] == "failed" for row in context["llm_retry_trace"]))
        self.assertEqual(context["llm_plan_proposals"][-1]["status"], "failed_fallback_disabled")

    def test_transient_timeout_retries_same_llm_without_heuristic_fallback(self) -> None:
        client = _SequenceClient(
            [
                TimeoutError("read timed out"),
                {
                    "agent": "ReporterAgent",
                    "tool": "write_artifacts",
                    "args": {},
                    "rationale": "continue",
                },
            ]
        )
        context = initial_context(
            "aspirin DILI",
            task_type="admet_prediction",
            dataset_name="DILI",
            enable_fallbacks=False,
            require_llm_orchestrator=True,
        )
        planner = LLMOrchestratorPlanner(
            registry=_registry(),
            llm_client=client,
            max_llm_retries=2,
            retry_backoff_seconds=0,
        )
        action = planner.next_action(context)
        self.assertEqual((action.agent, action.tool), ("ReporterAgent", "write_artifacts"))
        self.assertEqual(client.calls, 2)
        self.assertEqual([row["status"] for row in context["llm_retry_trace"]], ["failed", "completed"])
        self.assertEqual(context["metadata"]["last_action_source"], "llm")
        self.assertFalse(context["llm_plan_proposals"][-1]["fallback_used"])

    def test_retry_exhaustion_is_visible_and_does_not_enable_fallback(self) -> None:
        client = _SequenceClient([TimeoutError("t1"), TimeoutError("t2"), TimeoutError("t3")])
        context = initial_context(
            "aspirin DILI",
            task_type="admet_prediction",
            dataset_name="DILI",
            enable_fallbacks=False,
            require_llm_orchestrator=True,
        )
        planner = LLMOrchestratorPlanner(
            registry=_registry(),
            llm_client=client,
            max_llm_retries=2,
            retry_backoff_seconds=0,
        )
        action = planner.next_action(context)
        self.assertEqual(action.tool, "finish")
        self.assertEqual(client.calls, 3)
        self.assertEqual(len(context["llm_retry_trace"]), 3)
        self.assertTrue(all(row["status"] == "failed" for row in context["llm_retry_trace"]))
        self.assertEqual(context["llm_plan_proposals"][-1]["status"], "failed_fallback_disabled")

    def test_terminal_prediction_failure_uses_safety_policy_without_llm(self) -> None:
        client = _NeverCallClient()
        context = initial_context(
            "aspirin DILI",
            task_type="admet_prediction",
            dataset_name="DILI",
            require_real_rag=True,
            enable_fallbacks=False,
            require_llm_orchestrator=True,
        )
        context["metadata"]["safe_abstention_required"] = True
        context["tool_calls"] = [
            {"agent": "MoleculeAgent", "tool": "route_task", "status": "completed"},
            {"agent": "GaugeAgent", "tool": "predict_single", "status": "failed"},
        ]
        planner = LLMOrchestratorPlanner(registry=_registry(), llm_client=client)
        action = planner.next_action(context)
        self.assertEqual((action.agent, action.tool), ("RetrieveAgent", "retrieve_elasticsearch_hybrid"))
        self.assertEqual(client.calls, 0)
        self.assertEqual(context["metadata"]["last_action_source"], "safety_policy")
        self.assertEqual(context["llm_plan_proposals"][-1]["status"], "safety_policy_override")

    def test_safe_abstention_preserves_reflection_on_condition(self) -> None:
        client = _NeverCallClient()
        context = initial_context(
            "novel reaction",
            task_type="reaction_classification",
            dataset_name="schneider",
            enable_fallbacks=False,
            require_llm_orchestrator=True,
            enable_reflection=True,
            max_reflection_rounds=1,
        )
        context["metadata"].update(
            {
                "safe_abstention_required": True,
                "retrieval_outcome": "no_direct_evidence",
                "direct_evidence_available": False,
            }
        )
        context["final_report"] = {"task_type": "reaction_classification", "predictions": []}
        context["verifier_report"] = {
            "summary": {"status": "warning"},
            "citations": {"citation_grounded": True},
            "schema": {"schema_complete": True},
        }
        context["tool_calls"] = [
            {"agent": "MoleculeAgent", "tool": "route_task", "status": "completed"},
            {"agent": "ReactionAgent", "tool": "parse_reaction", "status": "completed"},
            {"agent": "RetrieveAgent", "tool": "retrieve_mit_template_evidence", "status": "completed"},
            {"agent": "RetrieveAgent", "tool": "retrieve_orderly_condition_evidence", "status": "completed"},
            {"agent": "ReactionAgent", "tool": "build_reaction_profile", "status": "completed"},
            {"agent": "ReasoningAgent", "tool": "build_structured_explanation", "status": "completed"},
            {"agent": "VerifierAgent", "tool": "verify_citation_grounding", "status": "completed"},
            {"agent": "VerifierAgent", "tool": "verify_report_schema", "status": "completed"},
        ]
        planner = LLMOrchestratorPlanner(registry=_registry(), llm_client=client)

        action = planner.next_action(context)

        self.assertEqual((action.agent, action.tool), ("ReflectionAgent", "reflect_on_report"))
        self.assertEqual(client.calls, 0)
        self.assertEqual(context["metadata"]["last_action_source"], "safety_policy")

    def test_safe_abstention_applies_reflection_before_artifacts(self) -> None:
        client = _NeverCallClient()
        context = initial_context(
            "novel reaction",
            task_type="reaction_classification",
            dataset_name="schneider",
            enable_fallbacks=False,
            require_llm_orchestrator=True,
            enable_reflection=True,
            max_reflection_rounds=1,
        )
        context["metadata"].update(
            {
                "safe_abstention_required": True,
                "retrieval_outcome": "no_direct_evidence",
                "direct_evidence_available": False,
                "reflection_rounds": 1,
            }
        )
        context["final_report"] = {"task_type": "reaction_classification", "predictions": []}
        context["reflection"] = {"round": 1, "recommendation": "finalize"}
        context["verifier_report"] = {"summary": {"status": "warning"}}
        context["tool_calls"] = [
            {"agent": "MoleculeAgent", "tool": "route_task", "status": "completed"},
            {"agent": "ReactionAgent", "tool": "parse_reaction", "status": "completed"},
            {"agent": "RetrieveAgent", "tool": "retrieve_mit_template_evidence", "status": "completed"},
            {"agent": "RetrieveAgent", "tool": "retrieve_orderly_condition_evidence", "status": "completed"},
            {"agent": "ReactionAgent", "tool": "build_reaction_profile", "status": "completed"},
            {"agent": "ReasoningAgent", "tool": "build_structured_explanation", "status": "completed"},
            {"agent": "VerifierAgent", "tool": "verify_citation_grounding", "status": "completed"},
            {"agent": "VerifierAgent", "tool": "verify_report_schema", "status": "completed"},
            {"agent": "ReflectionAgent", "tool": "reflect_on_report", "status": "completed"},
        ]
        planner = LLMOrchestratorPlanner(registry=_registry(), llm_client=client)

        action = planner.next_action(context)

        self.assertEqual((action.agent, action.tool), ("ReflectionAgent", "apply_reflection_feedback"))
        self.assertEqual(client.calls, 0)
        self.assertEqual(context["metadata"]["last_action_source"], "safety_policy")

    def test_verified_no_direct_evidence_uses_boundary_policy_without_llm(self) -> None:
        client = _NeverCallClient()
        context = initial_context(
            "novel molecule DILI",
            task_type="admet_prediction",
            dataset_name="DILI",
            require_real_rag=True,
            enable_fallbacks=False,
            require_llm_orchestrator=True,
        )
        context["metadata"]["retrieval_outcome"] = "no_direct_evidence"
        context["predictions"] = [{"label": 1, "model_type": "random_forest"}]
        context["tool_calls"] = [
            {"agent": "MoleculeAgent", "tool": "route_task", "status": "completed"},
            {"agent": "RetrieveAgent", "tool": "retrieve_elasticsearch_hybrid", "status": "completed"},
        ]
        planner = LLMOrchestratorPlanner(registry=_registry(), llm_client=client)

        action = planner.next_action(context)

        self.assertEqual((action.agent, action.tool), ("ReasoningAgent", "build_structured_explanation"))
        self.assertEqual(client.calls, 0)
        self.assertEqual(context["metadata"]["last_action_source"], "evidence_boundary_policy")
        self.assertEqual(context["llm_plan_proposals"][-1]["status"], "evidence_boundary_policy_override")

    def test_verified_no_direct_evidence_preserves_reflection_on_condition(self) -> None:
        client = _NeverCallClient()
        context = initial_context(
            "novel molecule DILI",
            task_type="admet_prediction",
            dataset_name="DILI",
            require_real_rag=True,
            enable_fallbacks=False,
            require_llm_orchestrator=True,
            enable_reflection=True,
            max_reflection_rounds=1,
        )
        context["metadata"]["retrieval_outcome"] = "no_direct_evidence"
        context["predictions"] = [{"label": 1, "model_type": "random_forest"}]
        context["final_report"] = {"task_type": "admet_prediction", "predictions": context["predictions"]}
        context["verifier_report"] = {
            "summary": {"status": "warning"},
            "citations": {"citation_grounded": True},
            "schema": {"schema_complete": True},
        }
        context["tool_calls"] = [
            {"agent": "ReasoningAgent", "tool": "build_structured_explanation", "status": "completed"},
            {"agent": "VerifierAgent", "tool": "verify_prediction_grounding", "status": "completed"},
            {"agent": "VerifierAgent", "tool": "verify_citation_grounding", "status": "completed"},
            {"agent": "VerifierAgent", "tool": "verify_report_schema", "status": "completed"},
        ]
        planner = LLMOrchestratorPlanner(registry=_registry(), llm_client=client)

        action = planner.next_action(context)

        self.assertEqual((action.agent, action.tool), ("ReflectionAgent", "reflect_on_report"))
        self.assertEqual(client.calls, 0)
        self.assertEqual(context["metadata"]["last_action_source"], "evidence_boundary_policy")

    def test_guard_allows_context_retrieval_but_blocks_new_prediction_after_terminal_failure(self) -> None:
        context = initial_context("aspirin DILI", task_type="admet_prediction", dataset_name="DILI")
        context["metadata"]["safe_abstention_required"] = True
        context["molecules"] = [{"name": "aspirin", "is_valid": True}]
        context["features"] = [{"feature_set": "test"}]
        guard = GuardPolicy(agent_allowlists=_registry().agent_allowlists(), max_rounds=20)
        guard.validate_action(
            AgentAction("RetrieveAgent", "retrieve_elasticsearch_hybrid").to_dict(),
            context,
        )
        guard.validate_action(
            AgentAction("ReasoningAgent", "build_structured_explanation").to_dict(),
            context,
        )
        with self.assertRaisesRegex(GuardViolation, "terminal_prediction_failure"):
            guard.validate_action(AgentAction("GaugeAgent", "predict_single").to_dict(), context)

    def test_graph_marks_controlled_unrecoverable_prediction_failure(self) -> None:
        registry = _registry()
        with tempfile.TemporaryDirectory() as tmp:
            nodes = MultiAgentGraphNodes(
                registry=registry,
                guard=GuardPolicy(agent_allowlists=registry.agent_allowlists(), max_rounds=20),
                orchestrator=DeterministicMultiAgentPlanner(),
                output_dir=tmp,
            )
            context = initial_context("aspirin DILI", task_type="admet_prediction", dataset_name="DILI")
            nodes._record_terminal_prediction_failure(
                AgentAction("GaugeAgent", "predict_single"),
                ToolResult(
                    "predict_single",
                    "failed",
                    {
                        "controlled_failure": True,
                        "failure_kind": "prediction_tool_or_service_failure",
                        "recoverable": False,
                    },
                    "prediction service unavailable",
                ),
                context,
            )
        self.assertTrue(context["metadata"]["safe_abstention_required"])
        self.assertEqual(context["metadata"]["prediction_failure"]["tool"], "predict_single")
        self.assertTrue(any("failed safely" in warning for warning in context["warnings"]))


if __name__ == "__main__":
    unittest.main()
