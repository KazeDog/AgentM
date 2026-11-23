from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional

from agentm.runtime.compat import ensure_import_paths
from agentm.runtime.tools.registry import AgentAction, MultiAgentToolRegistry

ensure_import_paths()


INTENT_GATE_TASK_TYPES = {"out_of_domain", "missing_entity"}


class DeterministicMultiAgentPlanner:
    """Deterministic fallback planner for the guarded multi-agent runtime."""

    def build_plan(self, context: Dict[str, Any], registry: MultiAgentToolRegistry) -> Dict[str, Any]:
        context.setdefault("metadata", {})["planner_type"] = "heuristic_fallback"
        context["metadata"]["orchestrator_mode"] = "heuristic"
        task_type = context.get("task_type") or "unknown_until_route_task"
        contract = context.get("metadata", {}).get("task_contract")
        if isinstance(contract, dict):
            nodes = self._open_task_nodes(contract)
            return {
                "status": "planned",
                "planner_type": "heuristic_fallback",
                "objective": context.get("request"),
                "task_type_hint": task_type,
                "agents": registry.agents(),
                "task_graph": {
                    "status": "planned",
                    "nodes": nodes,
                    "edges": [[nodes[i]["id"], nodes[i + 1]["id"]] for i in range(len(nodes) - 1)],
                },
                "stop_conditions": ["all_requested_subtasks_completed", "scoped_citations_checked", "schema_checked", "budget_not_exceeded"],
            }
        if task_type in INTENT_GATE_TASK_TYPES:
            nodes = self._intent_gate_nodes(task_type)
            return {
                "status": "planned",
                "planner_type": "heuristic_fallback",
                "objective": context.get("request"),
                "task_type_hint": task_type,
                "agents": registry.agents(),
                "task_graph": {"status": "planned", "nodes": nodes, "edges": [[nodes[i]["id"], nodes[i + 1]["id"]] for i in range(len(nodes) - 1)]},
                "stop_conditions": ["scientific_intent_checked", "prediction_not_requested", "schema_checked", "budget_not_exceeded"],
            }
        if task_type == "general_drug_profile":
            nodes = self._general_profile_nodes()
            return {
                "status": "planned",
                "planner_type": "heuristic_fallback",
                "objective": context.get("request"),
                "task_type_hint": task_type,
                "agents": registry.agents(),
                "task_graph": {"status": "planned", "nodes": nodes, "edges": [[nodes[i]["id"], nodes[i + 1]["id"]] for i in range(len(nodes) - 1)]},
                "stop_conditions": ["molecule_resolved", "citation_checked", "schema_checked", "reflection_applied", "budget_not_exceeded"],
            }
        if str(task_type or "").startswith("reaction_"):
            nodes = self._reaction_nodes(task_type)
            return {
                "status": "planned",
                "planner_type": "heuristic_fallback",
                "objective": context.get("request"),
                "task_type_hint": task_type,
                "agents": registry.agents(),
                "task_graph": {"status": "planned", "nodes": nodes, "edges": [[nodes[i]["id"], nodes[i + 1]["id"]] for i in range(len(nodes) - 1)]},
                "stop_conditions": ["reaction_parsed", "reaction_claims_grounded", "schema_checked", "reflection_applied", "budget_not_exceeded"],
            }
        is_ddi = task_type == "ddi_prediction"
        resolve = "resolve_ddi_pair" if is_ddi else "resolve_molecule"
        predict = "predict_ddi" if is_ddi else "predict_single"
        nodes = [
            {"id": "route_task", "agent": "MoleculeAgent", "goal": "Route request and infer task/dataset."},
            {"id": resolve, "agent": "MoleculeAgent", "goal": "Resolve molecule inputs."},
            {"id": "discover_rules", "agent": "GaugeAgent", "goal": "Load task rules."},
            {"id": "build_features", "agent": "GaugeAgent", "goal": "Build core feature bundles."},
            {"id": predict, "agent": "GaugeAgent", "goal": "Produce tool-grounded prediction."},
            {"id": "retrieve_evidence", "agent": "RetrieveAgent", "goal": "Retrieve evidence and citations."},
            {"id": "reason", "agent": "ReasoningAgent", "goal": "Build grounded explanation."},
            {"id": "verify", "agent": "VerifierAgent", "goal": "Verify prediction, citations, and schema."},
            {"id": "reflect", "agent": "ReflectionAgent", "goal": "Use verifier output as the next input."},
            {"id": "report", "agent": "ReporterAgent", "goal": "Write report artifacts."},
        ]
        plan = {
            "status": "planned",
            "planner_type": "heuristic_fallback",
            "objective": context.get("request"),
            "task_type_hint": task_type,
            "agents": registry.agents(),
            "task_graph": {"status": "planned", "nodes": nodes, "edges": [[nodes[i]["id"], nodes[i + 1]["id"]] for i in range(len(nodes) - 1)]},
            "stop_conditions": ["prediction_grounded", "citation_checked", "schema_checked", "reflection_applied", "budget_not_exceeded"],
        }
        return plan

    def next_action(self, context: Dict[str, Any]) -> AgentAction:
        context.setdefault("metadata", {})["last_action_source"] = "heuristic"
        fallbacks_enabled = bool(context.get("metadata", {}).get("enable_fallbacks", False))
        mode = context.get("metadata", {}).get("mode", "full")
        retrieval_backend = context.get("metadata", {}).get("retrieval_backend_requested", "auto")
        if not self._attempted(context, "route_task"):
            return AgentAction("MoleculeAgent", "route_task", rationale="Route task before selecting specialized agents.")
        safe_action = self.safe_abstention_action(context)
        if safe_action is not None:
            context.setdefault("metadata", {})["last_action_source"] = "safety_policy"
            return safe_action
        open_task_action = self.open_task_action(context)
        if open_task_action is not None:
            context.setdefault("metadata", {})["last_action_source"] = "open_task_contract"
            return open_task_action
        evidence_boundary_action = self.no_direct_evidence_action(context)
        if evidence_boundary_action is not None:
            context.setdefault("metadata", {})["last_action_source"] = "evidence_boundary_policy"
            return evidence_boundary_action
        if context.get("task_type") in INTENT_GATE_TASK_TYPES:
            return self._next_intent_gate_action(context)
        if str(context.get("task_type") or "").startswith("reaction_"):
            return self._next_reaction_action(context)
        if context.get("task_type") == "ddi_prediction":
            if not self._attempted(context, "resolve_ddi_pair"):
                return AgentAction("MoleculeAgent", "resolve_ddi_pair", rationale="Resolve two molecules for DDI prediction.")
        elif not self._attempted(context, "resolve_molecule"):
            return AgentAction("MoleculeAgent", "resolve_molecule", rationale="Resolve molecule before feature construction.")
        if context.get("task_type") == "general_drug_profile":
            if self._has_invalid_molecule(context):
                if not self._done(context, "build_structured_explanation") or self._needs_reasoning_refresh(context):
                    return AgentAction("ReasoningAgent", "build_structured_explanation", rationale="Explain why the general drug profile cannot be generated.")
                if not context.get("final_report") or self._needs_report_refresh(context):
                    return AgentAction("ReporterAgent", "build_runtime_report", rationale="Build runtime unresolved-molecule report.")
                if not self._done(context, "verify_report_schema") or self._needs_schema_reverify(context):
                    return AgentAction("VerifierAgent", "verify_report_schema", rationale="Verify report schema.")
                return AgentAction("OrchestratorAgent", "finish", rationale="General profile stopped because molecule resolution failed.")
            if not self._done(context, "retrieve_external_evidence"):
                return AgentAction("RetrieveAgent", "retrieve_external_evidence", rationale="Retrieve database drug profile before optional endpoint analysis.")
            if not self._attempted(context, "build_features"):
                return AgentAction("GaugeAgent", "build_features", rationale="Build structural descriptors for a general drug profile.")
            if context.get("metadata", {}).get("use_external_evidence") and not self._done(context, "retrieve_external_evidence"):
                return AgentAction("RetrieveAgent", "retrieve_external_evidence", rationale="Collect structured general evidence when requested.")
            use_rag = bool(context.get("metadata", {}).get("use_rag_evidence", True))
            if use_rag:
                if mode == "deep" and int(context.get("metadata", {}).get("max_rounds", 12)) > 12 and not self._done(context, "expand_evidence_query"):
                    return AgentAction("RetrieveAgent", "expand_evidence_query", rationale="Deep mode expands the general evidence query.")
                if mode == "fast":
                    if not self._attempted(context, "retrieve_local_rag"):
                        return AgentAction("RetrieveAgent", "retrieve_local_rag", rationale="Fast mode retrieves general local evidence.")
                elif not self._attempted(context, "retrieve_elasticsearch_hybrid"):
                    return AgentAction(
                        "RetrieveAgent",
                        "retrieve_elasticsearch_hybrid",
                        args={"backend": retrieval_backend},
                        rationale="Full/deep mode retrieves general evidence with ES auto hybrid.",
                    )
            if self._needs_evidence_repair(context):
                return AgentAction("RetrieveAgent", "expand_evidence_query", rationale="Verifier found unsupported evidence; expand query once.")
            if self._needs_retrieval_after_repair(context) or self._needs_reflection_retrieval(context):
                return AgentAction(
                    "RetrieveAgent",
                    "retrieve_elasticsearch_hybrid" if mode != "fast" else "retrieve_local_rag",
                    args={"query": context.get("reasoning", {}).get("expanded_evidence_query"), "backend": retrieval_backend},
                    rationale="Run bounded repair retrieval for the general profile.",
                )
            if not self._done(context, "build_structured_explanation") or self._needs_reasoning_refresh(context):
                return AgentAction("ReasoningAgent", "build_structured_explanation", rationale="Build a general drug-profile explanation.")
            if not context.get("final_report") or self._needs_report_refresh(context):
                return AgentAction("ReporterAgent", "build_runtime_report", rationale="Build runtime general profile report before schema verification.")
            if not self._done(context, "verify_citation_grounding") or self._needs_citation_reverify(context):
                return AgentAction("VerifierAgent", "verify_citation_grounding", rationale="Verify citation grounding.")
            if not self._done(context, "verify_report_schema") or self._needs_schema_reverify(context):
                return AgentAction("VerifierAgent", "verify_report_schema", rationale="Verify report schema.")
            if self._should_reflect(context):
                return AgentAction("ReflectionAgent", "reflect_on_report", rationale="Use first-pass report and verifier output as the next input.")
            if self._should_apply_reflection(context):
                return AgentAction("ReflectionAgent", "apply_reflection_feedback", rationale="Apply reflection feedback before finalization.")
            return AgentAction("OrchestratorAgent", "finish", rationale="General profile objective satisfied.")
        if not self._attempted(context, "discover_rules"):
            return AgentAction("GaugeAgent", "discover_rules", rationale="Load rule context for feature construction.")
        if not self._attempted(context, "build_features"):
            return AgentAction("GaugeAgent", "build_features", rationale="Build model-ready feature bundles.")
        prediction_tool = "predict_ddi" if context.get("task_type") == "ddi_prediction" else "predict_single"
        if not context.get("predictions") and not self._attempted(context, prediction_tool):
            return AgentAction("GaugeAgent", prediction_tool, rationale="Prediction must be produced by GaugeAgent.")
        if not context.get("predictions") and fallbacks_enabled and not self._attempted(context, "run_zero_shot_fallback"):
            return AgentAction("GaugeAgent", "run_zero_shot_fallback", rationale="Use zero-shot fallback after missing prediction.")
        if not context.get("predictions"):
            return AgentAction("OrchestratorAgent", "finish", rationale="Prediction unavailable and fallbacks are disabled.")
        if not self._done(context, "assess_confidence"):
            return AgentAction("GaugeAgent", "assess_confidence", rationale="Assess uncertainty before evidence retrieval.")
        if context.get("metadata", {}).get("use_external_evidence") and not self._done(context, "retrieve_external_evidence"):
            return AgentAction("RetrieveAgent", "retrieve_external_evidence", rationale="Collect structured evidence before RAG.")
        if mode == "deep" and int(context.get("metadata", {}).get("max_rounds", 12)) > 12 and not self._done(context, "expand_evidence_query"):
            return AgentAction("RetrieveAgent", "expand_evidence_query", rationale="Deep mode expands evidence query.")
        if mode == "fast":
            if not self._attempted(context, "retrieve_local_rag"):
                return AgentAction("RetrieveAgent", "retrieve_local_rag", rationale="Fast mode uses local retrieval.")
        elif not self._attempted(context, "retrieve_elasticsearch_hybrid"):
            return AgentAction(
                "RetrieveAgent",
                "retrieve_elasticsearch_hybrid",
                args={"backend": retrieval_backend},
                rationale="Full/deep mode uses ES auto hybrid with local fallback.",
            )
        if self._needs_evidence_repair(context):
            return AgentAction("RetrieveAgent", "expand_evidence_query", rationale="Verifier found unsupported evidence; expand query once.")
        if self._needs_retrieval_after_repair(context):
            return AgentAction(
                "RetrieveAgent",
                "retrieve_elasticsearch_hybrid" if mode != "fast" else "retrieve_local_rag",
                args={"query": context.get("reasoning", {}).get("expanded_evidence_query"), "backend": retrieval_backend},
                rationale="Run repair retrieval after expanded query.",
            )
        if self._needs_reflection_retrieval(context):
            return AgentAction(
                "RetrieveAgent",
                "retrieve_elasticsearch_hybrid" if mode != "fast" else "retrieve_local_rag",
                args={"query": context.get("reasoning", {}).get("expanded_evidence_query"), "backend": retrieval_backend},
                rationale="Use reflection output as the next evidence retrieval input.",
            )
        if not self._done(context, "build_structured_explanation") or self._needs_reasoning_refresh(context):
            return AgentAction("ReasoningAgent", "build_structured_explanation", rationale="Build or refresh grounded explanation.")
        if not context.get("final_report") or self._needs_report_refresh(context):
            return AgentAction("ReporterAgent", "build_runtime_report", rationale="Build runtime report before schema verification.")
        if not self._done(context, "verify_prediction_grounding"):
            return AgentAction("VerifierAgent", "verify_prediction_grounding", rationale="Verify GaugeAgent prediction grounding.")
        if not self._done(context, "verify_citation_grounding") or self._needs_citation_reverify(context):
            return AgentAction("VerifierAgent", "verify_citation_grounding", rationale="Verify citation grounding.")
        if not self._done(context, "verify_report_schema") or self._needs_schema_reverify(context):
            return AgentAction("VerifierAgent", "verify_report_schema", rationale="Verify core-compatible report schema.")
        if self._should_reflect(context):
            return AgentAction("ReflectionAgent", "reflect_on_report", rationale="Use first-pass report and verifier output as the next input.")
        if self._should_apply_reflection(context):
            return AgentAction("ReflectionAgent", "apply_reflection_feedback", rationale="Apply reflection feedback before finalization.")
        return AgentAction("OrchestratorAgent", "finish", rationale="Guarded multi-agent objective satisfied.")

    def safe_abstention_action(self, context: Dict[str, Any]) -> AgentAction | None:
        """Advance an unrecoverable prediction failure without fabricating output.

        This is a fail-closed safety policy, not a scientific heuristic fallback:
        it never supplies a label, score, or substitute prediction.  It records
        available evidence, writes an explicit failure report, and terminates.
        """

        metadata = context.get("metadata", {}) or {}
        if not bool(metadata.get("safe_abstention_required")):
            return None
        task_type = str(context.get("task_type") or "")
        if task_type.startswith("reaction_"):
            if not self._attempted(context, "parse_reaction"):
                return AgentAction("ReactionAgent", "parse_reaction", rationale="Parse the reaction before a safe failure report.")
            if not self._attempted(context, "retrieve_mit_template_evidence"):
                return AgentAction("RetrieveAgent", "retrieve_mit_template_evidence", rationale="Preserve real reaction-template evidence before abstaining.")
            if not self._attempted(context, "retrieve_orderly_condition_evidence"):
                return AgentAction("RetrieveAgent", "retrieve_orderly_condition_evidence", rationale="Preserve real reaction-condition evidence before abstaining.")
            if not self._done(context, "build_reaction_profile"):
                return AgentAction("ReactionAgent", "build_reaction_profile", rationale="Build a non-predictive reaction profile after the prediction tool failed.")
        else:
            mode = str(metadata.get("mode") or "full")
            retrieval_backend = metadata.get("retrieval_backend_requested", "auto")
            retrieval_tool = "retrieve_local_rag" if mode == "fast" and not metadata.get("require_real_rag") else "retrieve_elasticsearch_hybrid"
            if not self._attempted(context, retrieval_tool):
                return AgentAction(
                    "RetrieveAgent",
                    retrieval_tool,
                    args={"backend": retrieval_backend} if retrieval_tool == "retrieve_elasticsearch_hybrid" else {},
                    rationale="Retrieve traceable evidence for context; evidence must not substitute for the unavailable prediction.",
                )
        if not self._done(context, "build_structured_explanation") or self._needs_reasoning_refresh(context):
            return AgentAction("ReasoningAgent", "build_structured_explanation", rationale="Explain the prediction failure and safe abstention boundary.")
        if not context.get("final_report") or self._needs_report_refresh(context):
            return AgentAction("ReporterAgent", "build_runtime_report", rationale="Build an explicit no-prediction safety report.")
        if not self._done(context, "verify_citation_grounding") or self._needs_citation_reverify(context):
            return AgentAction("VerifierAgent", "verify_citation_grounding", rationale="Verify any contextual evidence citations without validating a prediction.")
        if not self._done(context, "verify_report_schema") or self._needs_schema_reverify(context):
            return AgentAction("VerifierAgent", "verify_report_schema", rationale="Verify the safe-abstention report schema.")
        # A terminal scientific boundary must not erase the reflection-on
        # factorial condition.  Reflection may review the abstention report,
        # but the safety guard still blocks a fabricated replacement
        # prediction and the no-direct-evidence policy blocks retrieval loops.
        if self._should_reflect(context):
            return AgentAction(
                "ReflectionAgent",
                "reflect_on_report",
                rationale="Review the safe-abstention report without promoting unavailable evidence or predictions.",
            )
        if self._should_apply_reflection(context):
            return AgentAction(
                "ReflectionAgent",
                "apply_reflection_feedback",
                rationale="Apply bounded reflection feedback while preserving the safe-abstention boundary.",
            )
        if not self._done(context, "write_artifacts"):
            return AgentAction("ReporterAgent", "write_artifacts", rationale="Persist the explicit safe-abstention record.")
        return AgentAction("OrchestratorAgent", "finish", rationale="Prediction failed irrecoverably; the task ended with an explicit safe abstention.")

    def no_direct_evidence_action(self, context: Dict[str, Any]) -> AgentAction | None:
        """Finish a verified empty retrieval without repeated LLM-driven queries.

        The prediction remains available as a model output, but the report must
        disclose that the target-specific dense-hybrid query found no direct
        evidence.  This policy controls execution only; it never upgrades the
        scientific objective or verifier result to success.
        """

        metadata = context.get("metadata", {}) or {}
        if metadata.get("retrieval_outcome") != "no_direct_evidence":
            return None
        contract = metadata.get("task_contract")
        if isinstance(contract, dict):
            if not self._open_task_science_complete(context, contract):
                # An empty retrieval is scoped to the active subtask.  It must
                # not terminate a composite request while other requested
                # endpoints or pair-level predictions are still outstanding.
                return None
        if not context.get("predictions"):
            return None
        if not self._done(context, "build_structured_explanation") or self._needs_reasoning_refresh(context):
            return AgentAction(
                "ReasoningAgent",
                "build_structured_explanation",
                rationale="State the verified direct-evidence gap and keep the model prediction separate from experimental fact.",
            )
        if not context.get("final_report") or self._needs_report_refresh(context):
            return AgentAction(
                "ReporterAgent",
                "build_runtime_report",
                rationale="Build an evidence-bounded prediction report.",
            )
        if not self._done(context, "verify_prediction_grounding"):
            return AgentAction(
                "VerifierAgent",
                "verify_prediction_grounding",
                rationale="Verify that the reported prediction came from the registered model tool.",
            )
        if not self._done(context, "verify_citation_grounding") or self._needs_citation_reverify(context):
            return AgentAction(
                "VerifierAgent",
                "verify_citation_grounding",
                rationale="Record that no direct citation supports the target-specific prediction.",
            )
        if not self._done(context, "verify_report_schema") or self._needs_schema_reverify(context):
            return AgentAction(
                "VerifierAgent",
                "verify_report_schema",
                rationale="Verify the evidence-bounded report schema without promoting it to scientific success.",
            )
        # A verified lack of direct evidence is a scientific boundary, not a
        # reason to silently disable the reflection-on factorial condition.
        # Reflection may review and preserve the boundary, but this policy
        # still forbids another unbounded retrieval loop.
        if self._should_reflect(context):
            return AgentAction(
                "ReflectionAgent",
                "reflect_on_report",
                rationale="Review the no-direct-evidence boundary without upgrading it to scientific support.",
            )
        if self._should_apply_reflection(context):
            return AgentAction(
                "ReflectionAgent",
                "apply_reflection_feedback",
                rationale="Apply bounded reflection feedback while preserving the no-direct-evidence status.",
            )
        if not self._done(context, "write_artifacts"):
            return AgentAction(
                "ReporterAgent",
                "write_artifacts",
                rationale="Persist the explicit no-direct-evidence outcome.",
            )
        return AgentAction(
            "OrchestratorAgent",
            "finish",
            rationale="The real retrieval path completed with no direct evidence; the evidence-bounded report is complete.",
        )

    def open_task_action(self, context: Dict[str, Any]) -> AgentAction | None:
        """Execute a compact, scoped protocol for request-derived compositions."""

        contract = (context.get("metadata", {}) or {}).get("task_contract")
        if not isinstance(contract, dict):
            return None
        if contract.get("open_task_type") == "reaction_multi_tool":
            return self._open_reaction_action(context, contract)
        subtasks = [item for item in contract.get("subtasks", []) or [] if isinstance(item, dict)]
        if not subtasks:
            return None
        retrieval_backend = context.get("metadata", {}).get("retrieval_backend_requested", "auto")
        for subtask in subtasks:
            subtask_id = str(subtask.get("subtask_id") or "")
            args = {
                "subtask_id": subtask_id,
                "task_type": subtask.get("task_type"),
                "dataset_name": subtask.get("dataset_name"),
                "molecules": list(subtask.get("molecules", []) or []),
            }
            kind = str(subtask.get("kind") or "single")
            resolve_tool = "resolve_ddi_pair" if kind == "ddi" else "resolve_molecule"
            predict_tool = "predict_ddi" if kind == "ddi" else "predict_single"
            if not self._open_phase_done(context, resolve_tool, subtask_id):
                return AgentAction("MoleculeAgent", resolve_tool, args=args, rationale=f"Resolve scientific objects for scoped subtask {subtask_id}.")
            if not self._open_phase_done(context, "build_features", subtask_id):
                return AgentAction("GaugeAgent", "build_features", args=args, rationale=f"Build model-compatible features for scoped subtask {subtask_id}.")
            if not self._open_phase_done(context, predict_tool, subtask_id):
                return AgentAction("GaugeAgent", predict_tool, args=args, rationale=f"Run the registered model for scoped subtask {subtask_id}.")
            if not self._open_phase_done(context, "assess_confidence", subtask_id):
                return AgentAction("GaugeAgent", "assess_confidence", args=args, rationale=f"Assess model uncertainty for scoped subtask {subtask_id}.")
            if not self._open_phase_done(context, "retrieve_elasticsearch_hybrid", subtask_id):
                return AgentAction(
                    "RetrieveAgent",
                    "retrieve_elasticsearch_hybrid",
                    args={**args, "backend": retrieval_backend},
                    rationale=f"Retrieve separately scoped evidence for subtask {subtask_id}.",
                )
        self._restore_open_task_root(context, contract)
        return self._open_task_finalize_action(context)

    def _open_reaction_action(self, context: Dict[str, Any], contract: Dict[str, Any]) -> AgentAction:
        operations = set(str(item) for item in contract.get("operations", []) or [])
        if not self._done(context, "parse_reaction"):
            return AgentAction("ReactionAgent", "parse_reaction", rationale="Parse the open reaction task before scoped tools.")
        if "template_evidence" in operations and not self._done(context, "retrieve_mit_template_evidence"):
            return AgentAction("RetrieveAgent", "retrieve_mit_template_evidence", rationale="Retrieve template evidence; retry if the controlled first call failed.")
        if "condition_evidence" in operations and not self._done(context, "retrieve_orderly_condition_evidence"):
            return AgentAction("RetrieveAgent", "retrieve_orderly_condition_evidence", rationale="Retrieve separately traceable condition evidence.")
        if "reaction_classification" in operations and not self._done(context, "classify_reaction_type"):
            return AgentAction("ReactionAgent", "classify_reaction_type", rationale="Run the registered reaction classification tool.")
        if "yield_prediction" in operations and not self._done(context, "predict_buchwald_hartwig_yield"):
            return AgentAction("ReactionAgent", "predict_buchwald_hartwig_yield", rationale="Run yield prediction when supported by the reaction model.")
        if not self._done(context, "build_reaction_profile"):
            return AgentAction("ReactionAgent", "build_reaction_profile", rationale="Aggregate classification, yield, template, and condition outputs.")
        context.setdefault("metadata", {})["open_task_execution_complete"] = True
        return self._open_task_finalize_action(context)

    def _open_task_finalize_action(self, context: Dict[str, Any]) -> AgentAction:
        if not self._done(context, "build_structured_explanation"):
            return AgentAction("ReasoningAgent", "build_structured_explanation", rationale="Explain all completed composition subtasks without merging their evidence scopes.")
        if not context.get("final_report"):
            return AgentAction("ReporterAgent", "build_runtime_report", rationale="Build the composite scientific report.")
        if (context.get("reaction_predictions") or context.get("predictions")) and not self._done(context, "verify_prediction_grounding"):
            return AgentAction("VerifierAgent", "verify_prediction_grounding", rationale="Verify every composed prediction remains tool-grounded.")
        if not self._done(context, "verify_citation_grounding"):
            return AgentAction("VerifierAgent", "verify_citation_grounding", rationale="Verify scoped pair, single-drug, or reaction citations.")
        if not self._done(context, "verify_report_schema"):
            return AgentAction("VerifierAgent", "verify_report_schema", rationale="Verify composite report schema.")
        if not self._done(context, "write_artifacts"):
            return AgentAction("ReporterAgent", "write_artifacts", rationale="Persist the completed composite task and scoped traces.")
        return AgentAction("OrchestratorAgent", "finish", rationale="All request-derived composition subtasks are complete.")

    def _open_phase_done(self, context: Dict[str, Any], tool: str, subtask_id: str) -> bool:
        return any(
            call.get("tool") == tool
            and call.get("status") in {"completed", "fallback_completed"}
            and str((call.get("args") or {}).get("subtask_id") or "") == subtask_id
            for call in context.get("tool_calls", []) or []
        )

    def _open_task_science_complete(self, context: Dict[str, Any], contract: Dict[str, Any]) -> bool:
        if contract.get("open_task_type") == "reaction_multi_tool":
            operations = set(str(item) for item in contract.get("operations", []) or [])
            required = ["parse_reaction"]
            if "template_evidence" in operations:
                required.append("retrieve_mit_template_evidence")
            if "condition_evidence" in operations:
                required.append("retrieve_orderly_condition_evidence")
            if "reaction_classification" in operations:
                required.append("classify_reaction_type")
            if "yield_prediction" in operations:
                required.append("predict_buchwald_hartwig_yield")
            required.append("build_reaction_profile")
            return all(self._done(context, tool) for tool in required)

        subtasks = [item for item in contract.get("subtasks", []) or [] if isinstance(item, dict)]
        if not subtasks:
            return False
        for subtask in subtasks:
            subtask_id = str(subtask.get("subtask_id") or "")
            kind = str(subtask.get("kind") or "single")
            required = [
                "resolve_ddi_pair" if kind == "ddi" else "resolve_molecule",
                "build_features",
                "predict_ddi" if kind == "ddi" else "predict_single",
                "assess_confidence",
                "retrieve_elasticsearch_hybrid",
            ]
            if not all(self._open_phase_done(context, tool, subtask_id) for tool in required):
                return False
        return True

    def _restore_open_task_root(self, context: Dict[str, Any], contract: Dict[str, Any]) -> None:
        metadata = context.setdefault("metadata", {})
        metadata["active_open_subtask_id"] = ""
        metadata["open_task_execution_complete"] = True
        context["task_type"] = str(contract.get("root_task_type") or metadata.get("root_task_type") or context.get("task_type") or "")
        context["dataset_name"] = str(contract.get("root_dataset_name") or metadata.get("root_dataset_name") or context.get("dataset_name") or "")

    def _done(self, context: Dict[str, Any], tool: str) -> bool:
        return any(call.get("tool") == tool and call.get("status") in {"completed", "fallback_completed"} for call in context.get("tool_calls", []) or [])

    def _attempted(self, context: Dict[str, Any], tool: str) -> bool:
        return any(call.get("tool") == tool for call in context.get("tool_calls", []) or [])

    def _count(self, context: Dict[str, Any], tool: str) -> int:
        return sum(1 for call in context.get("tool_calls", []) or [] if call.get("tool") == tool)

    def _needs_evidence_repair(self, context: Dict[str, Any]) -> bool:
        citations = context.get("verifier_report", {}).get("citations", {})
        if not citations or citations.get("citation_grounded", True):
            return False
        return int(context.get("metadata", {}).get("evidence_expansion_rounds", 0)) < 1 and context.get("metadata", {}).get("mode") == "deep"

    def _needs_retrieval_after_repair(self, context: Dict[str, Any]) -> bool:
        expansion_rounds = int(context.get("metadata", {}).get("evidence_expansion_rounds", 0))
        return expansion_rounds > 0 and self._count(context, "retrieve_elasticsearch_hybrid") + self._count(context, "retrieve_local_rag") <= 1

    def _needs_reflection_retrieval(self, context: Dict[str, Any]) -> bool:
        metadata = context.get("metadata", {})
        return bool(metadata.get("reflection_retrieval_requested")) and not bool(metadata.get("reflection_retrieval_done"))

    def _should_reflect(self, context: Dict[str, Any]) -> bool:
        metadata = context.get("metadata", {})
        if not metadata.get("enable_reflection", True):
            return False
        if not context.get("final_report") or not context.get("verifier_report", {}).get("summary"):
            return False
        # A round is reflect -> apply -> refresh/reverify, not just a counter
        # increment.  Never overwrite feedback that has not yet been applied.
        if self._should_apply_reflection(context) or self._needs_reflection_retrieval(context):
            return False
        if any(metadata.get(key) for key in (
            "reasoning_stale", "report_stale", "citation_verification_stale", "schema_verification_stale"
        )):
            return False
        return int(metadata.get("reflection_rounds", 0)) < int(metadata.get("max_reflection_rounds", 1))

    def _should_apply_reflection(self, context: Dict[str, Any]) -> bool:
        reflection = context.get("reflection", {}) or {}
        if not reflection:
            return False
        applied_round = int(context.get("metadata", {}).get("reflection_feedback_applied_round", 0))
        return int(reflection.get("round", 0)) > applied_round

    def _needs_reasoning_refresh(self, context: Dict[str, Any]) -> bool:
        return bool(context.get("metadata", {}).pop("reasoning_stale", False))

    def _needs_report_refresh(self, context: Dict[str, Any]) -> bool:
        return bool(context.get("metadata", {}).pop("report_stale", False))

    def _needs_citation_reverify(self, context: Dict[str, Any]) -> bool:
        return bool(context.get("metadata", {}).pop("citation_verification_stale", False))

    def _needs_schema_reverify(self, context: Dict[str, Any]) -> bool:
        return bool(context.get("metadata", {}).pop("schema_verification_stale", False))

    def _has_invalid_molecule(self, context: Dict[str, Any]) -> bool:
        molecules = context.get("molecules", []) or []
        if not molecules:
            return False
        for molecule in molecules:
            if isinstance(molecule, dict):
                valid = molecule.get("is_valid", True)
            else:
                valid = getattr(molecule, "is_valid", True)
            if not valid:
                return True
        return False

    def _general_profile_nodes(self) -> List[Dict[str, str]]:
        return [
            {"id": "route_task", "agent": "MoleculeAgent", "goal": "Route request as a general drug profile."},
            {"id": "resolve_molecule", "agent": "MoleculeAgent", "goal": "Resolve the requested molecule."},
            {"id": "retrieve_database_profile", "agent": "RetrieveAgent", "goal": "Collect database profile facts before endpoint analysis."},
            {"id": "build_features", "agent": "GaugeAgent", "goal": "Build structural descriptors without endpoint prediction."},
            {"id": "retrieve_optional_rag", "agent": "RetrieveAgent", "goal": "Optionally collect broader evidence and citations."},
            {"id": "reason", "agent": "ReasoningAgent", "goal": "Build non-predictive drug-profile explanation."},
            {"id": "verify_citations", "agent": "VerifierAgent", "goal": "Verify citation grounding."},
            {"id": "verify_schema", "agent": "VerifierAgent", "goal": "Verify report schema."},
            {"id": "reflect", "agent": "ReflectionAgent", "goal": "Use verifier output as the next input."},
            {"id": "report", "agent": "ReporterAgent", "goal": "Write report artifacts."},
        ]

    def _open_task_nodes(self, contract: Dict[str, Any]) -> List[Dict[str, str]]:
        nodes: List[Dict[str, str]] = [
            {"id": "route_task", "agent": "MoleculeAgent", "goal": "Route the composite scientific request."}
        ]
        if contract.get("open_task_type") == "reaction_multi_tool":
            nodes.extend(
                [
                    {"id": "parse_reaction", "agent": "ReactionAgent", "goal": "Parse the reaction."},
                    {"id": "retrieve_mit_template_evidence", "agent": "RetrieveAgent", "goal": "Retrieve template evidence with bounded retry."},
                    {"id": "retrieve_orderly_condition_evidence", "agent": "RetrieveAgent", "goal": "Retrieve condition evidence."},
                    {"id": "classify_reaction_type", "agent": "ReactionAgent", "goal": "Classify the reaction."},
                    {"id": "predict_buchwald_hartwig_yield", "agent": "ReactionAgent", "goal": "Predict yield when supported."},
                    {"id": "build_reaction_profile", "agent": "ReactionAgent", "goal": "Aggregate reaction outputs."},
                ]
            )
        else:
            for subtask in contract.get("subtasks", []) or []:
                if not isinstance(subtask, dict):
                    continue
                subtask_id = str(subtask.get("subtask_id") or "subtask")
                kind = str(subtask.get("kind") or "single")
                nodes.extend(
                    [
                        {"id": f"{subtask_id}:resolve", "agent": "MoleculeAgent", "goal": f"Resolve objects for {subtask_id}."},
                        {"id": f"{subtask_id}:features", "agent": "GaugeAgent", "goal": f"Build endpoint-specific features for {subtask_id}."},
                        {"id": f"{subtask_id}:predict", "agent": "GaugeAgent", "goal": f"Run {'DDI' if kind == 'ddi' else 'single-molecule'} prediction for {subtask_id}."},
                        {"id": f"{subtask_id}:confidence", "agent": "GaugeAgent", "goal": f"Assess prediction confidence for {subtask_id}."},
                        {"id": f"{subtask_id}:retrieve", "agent": "RetrieveAgent", "goal": f"Retrieve evidence scoped only to {subtask_id}."},
                    ]
                )
        nodes.extend(
            [
                {"id": "build_structured_explanation", "agent": "ReasoningAgent", "goal": "Compare and explain scoped outputs."},
                {"id": "build_runtime_report", "agent": "ReporterAgent", "goal": "Build the composite report."},
                {"id": "verify_prediction_grounding", "agent": "VerifierAgent", "goal": "Verify prediction provenance."},
                {"id": "verify_citation_grounding", "agent": "VerifierAgent", "goal": "Verify scoped citations."},
                {"id": "verify_report_schema", "agent": "VerifierAgent", "goal": "Verify report schema."},
                {"id": "write_artifacts", "agent": "ReporterAgent", "goal": "Persist composite artifacts."},
            ]
        )
        return nodes

    def _intent_gate_nodes(self, task_type: str) -> List[Dict[str, str]]:
        goal = "Explain that a concrete scientific entity is required." if task_type == "missing_entity" else "Explain that no scientific routing intent was detected."
        return [
            {"id": "route_task", "agent": "MoleculeAgent", "goal": "Run the scientific intent gate before ADMET, DDI, or reaction routing."},
            {"id": "build_runtime_report", "agent": "ReporterAgent", "goal": goal},
            {"id": "verify_report_schema", "agent": "VerifierAgent", "goal": "Verify the non-predictive report schema."},
            {"id": "write_artifacts", "agent": "ReporterAgent", "goal": "Write report artifacts."},
        ]

    def _next_intent_gate_action(self, context: Dict[str, Any]) -> AgentAction:
        if not context.get("final_report") or self._needs_report_refresh(context):
            return AgentAction("ReporterAgent", "build_runtime_report", rationale="Build an intent-gate report without molecule resolution or prediction.")
        if not self._done(context, "verify_report_schema") or self._needs_schema_reverify(context):
            return AgentAction("VerifierAgent", "verify_report_schema", rationale="Verify the intent-gate report schema.")
        if not self._done(context, "write_artifacts"):
            return AgentAction("ReporterAgent", "write_artifacts", rationale="Write the intent-gate report artifacts.")
        return AgentAction("OrchestratorAgent", "finish", rationale="Input was stopped by the scientific intent gate.")

    def _reaction_nodes(self, task_type: str) -> List[Dict[str, str]]:
        nodes = [
            {"id": "route_task", "agent": "MoleculeAgent", "goal": "Route request as a reaction task."},
            {"id": "parse_reaction", "agent": "ReactionAgent", "goal": "Parse reaction SMILES into reactants and products."},
            {"id": "retrieve_mit_template_evidence", "agent": "RetrieveAgent", "goal": "Retrieve product/template evidence from MIT."},
            {"id": "retrieve_orderly_condition_evidence", "agent": "RetrieveAgent", "goal": "Retrieve similar reaction-condition evidence from ORDerly."},
        ]
        if task_type == "reaction_classification":
            nodes.append({"id": "classify_reaction_type", "agent": "ReactionAgent", "goal": "Classify reaction type with the Schneider model."})
        if task_type == "reaction_yield_prediction":
            nodes.append({"id": "predict_buchwald_hartwig_yield", "agent": "ReactionAgent", "goal": "Predict Buchwald-Hartwig yield with the reaction model."})
        nodes.extend(
            [
                {"id": "build_reaction_profile", "agent": "ReactionAgent", "goal": "Build a grounded reaction profile."},
                {"id": "build_structured_explanation", "agent": "ReasoningAgent", "goal": "Convert the profile into a readable explanation."},
                {"id": "build_runtime_report", "agent": "ReporterAgent", "goal": "Build the runtime report payload."},
                {"id": "verify_prediction_grounding", "agent": "VerifierAgent", "goal": "Verify reaction predictions came from ReactionAgent tools."},
                {"id": "verify_citation_grounding", "agent": "VerifierAgent", "goal": "Verify reaction evidence citations."},
                {"id": "verify_report_schema", "agent": "VerifierAgent", "goal": "Verify report schema."},
                {"id": "reflect", "agent": "ReflectionAgent", "goal": "Use verifier output as the next input."},
                {"id": "write_artifacts", "agent": "ReporterAgent", "goal": "Write reaction artifacts."},
            ]
        )
        return nodes

    def _next_reaction_action(self, context: Dict[str, Any]) -> AgentAction:
        task_type = str(context.get("task_type") or "")
        if not self._attempted(context, "parse_reaction"):
            return AgentAction("ReactionAgent", "parse_reaction", rationale="Parse reaction before evidence retrieval or prediction.")
        if not self._attempted(context, "retrieve_mit_template_evidence"):
            return AgentAction("RetrieveAgent", "retrieve_mit_template_evidence", rationale="Retrieve template/product grounding from MIT.")
        if not self._attempted(context, "retrieve_orderly_condition_evidence"):
            return AgentAction("RetrieveAgent", "retrieve_orderly_condition_evidence", rationale="Retrieve condition grounding from ORDerly.")
        if task_type == "reaction_classification" and not self._attempted(context, "classify_reaction_type"):
            return AgentAction("ReactionAgent", "classify_reaction_type", rationale="Classify reaction type with a ReactionAgent model.")
        if task_type == "reaction_yield_prediction" and not self._attempted(context, "predict_buchwald_hartwig_yield"):
            return AgentAction("ReactionAgent", "predict_buchwald_hartwig_yield", rationale="Predict yield with the Buchwald-Hartwig KG ReactionAgent model.")
        if not self._done(context, "build_reaction_profile"):
            return AgentAction("ReactionAgent", "build_reaction_profile", rationale="Build a structured reaction profile from grounded outputs.")
        if not self._done(context, "build_structured_explanation") or self._needs_reasoning_refresh(context):
            return AgentAction("ReasoningAgent", "build_structured_explanation", rationale="Build readable reaction explanation from the profile.")
        if not context.get("final_report") or self._needs_report_refresh(context):
            return AgentAction("ReporterAgent", "build_runtime_report", rationale="Build runtime reaction report before verification.")
        if task_type in {"reaction_classification", "reaction_yield_prediction"} and not self._done(context, "verify_prediction_grounding"):
            return AgentAction("VerifierAgent", "verify_prediction_grounding", rationale="Verify ReactionAgent prediction grounding.")
        if not self._done(context, "verify_citation_grounding") or self._needs_citation_reverify(context):
            return AgentAction("VerifierAgent", "verify_citation_grounding", rationale="Verify reaction citation grounding.")
        if not self._done(context, "verify_report_schema") or self._needs_schema_reverify(context):
            return AgentAction("VerifierAgent", "verify_report_schema", rationale="Verify reaction report schema.")
        if self._should_reflect(context):
            return AgentAction("ReflectionAgent", "reflect_on_report", rationale="Use verifier output as the next input.")
        if self._should_apply_reflection(context):
            return AgentAction("ReflectionAgent", "apply_reflection_feedback", rationale="Apply reflection feedback before finalization.")
        if not self._done(context, "write_artifacts"):
            return AgentAction("ReporterAgent", "write_artifacts", rationale="Write reaction artifacts.")
        return AgentAction("OrchestratorAgent", "finish", rationale="Reaction objective satisfied.")


class LLMOrchestratorPlanner:
    """Optional LLM action planner, guarded by fallback and policy checks."""

    def __init__(
        self,
        *,
        registry: MultiAgentToolRegistry,
        llm_profile: Optional[str] = None,
        temperature: float = 0.1,
        fallback: Optional[DeterministicMultiAgentPlanner] = None,
        llm_client: Any | None = None,
        max_llm_retries: int = 0,
        retry_backoff_seconds: float = 0.0,
    ) -> None:
        self.registry = registry
        self.temperature = float(temperature)
        self.fallback = fallback or DeterministicMultiAgentPlanner()
        self.max_llm_retries = min(5, max(0, int(max_llm_retries)))
        self.retry_backoff_seconds = max(0.0, float(retry_backoff_seconds))
        self.disabled_reason = ""
        if llm_client is not None:
            self.client = llm_client
        else:
            try:
                from agentm.llm.provider import build_llm_client

                self.client = build_llm_client(profile_name=llm_profile)
            except Exception:
                self.client = None

    def build_plan(self, context: Dict[str, Any]) -> Dict[str, Any]:
        if self.client is None or self.disabled_reason or not getattr(self.client, "available", False):
            if not self._fallback_allowed(context):
                reason = self.disabled_reason or "llm_unavailable"
                plan = self._disabled_plan(context, reason)
                suffix = "required" if context.get("metadata", {}).get("require_llm_orchestrator") else "fallback_disabled"
                context.setdefault("warnings", []).append(f"llm_task_graph_planning_failed_{suffix}: {reason}")
                context.setdefault("llm_plan_proposals", []).append(
                    {"kind": "task_graph", "status": "failed_fallback_disabled", "fallback_used": False, "reason": reason, "proposal": plan}
                )
                context.setdefault("plan_revision_trace", []).append({"stage": "initial_plan", "status": "failed", "reason": reason, "fallback_disabled": True})
                return plan
            plan = self.fallback.build_plan(context, self.registry)
            proposal = {
                "kind": "task_graph",
                "status": "llm_unavailable_used_fallback",
                "fallback_used": True,
                "reason": self.disabled_reason or "llm_unavailable",
                "proposal": plan,
            }
            context.setdefault("llm_plan_proposals", []).append(proposal)
            context.setdefault("plan_revision_trace", []).append({"stage": "initial_plan", "status": "fallback", "reason": "llm_unavailable"})
            return plan
        try:
            payload = self._chat_json_payload(context, self._plan_messages(context), stage="task_graph")
            plan = self._normalize_plan(payload, context)
            context.setdefault("llm_plan_proposals", []).append(
                {"kind": "task_graph", "status": "accepted", "fallback_used": False, "raw": payload, "proposal": plan}
            )
            context.setdefault("plan_revision_trace", []).append({"stage": "initial_plan", "status": "accepted", "planner": "llm"})
            context.setdefault("metadata", {})["planner_type"] = "llm"
            context["metadata"]["orchestrator_mode"] = "llm_orchestrated"
            return plan
        except Exception as exc:
            self.disabled_reason = str(exc)
            self.client = None
            if not self._fallback_allowed(context):
                plan = self._disabled_plan(context, str(exc))
                suffix = "required" if context.get("metadata", {}).get("require_llm_orchestrator") else "fallback_disabled"
                context.setdefault("warnings", []).append(f"llm_task_graph_planning_failed_{suffix}: {exc}")
                context.setdefault("llm_plan_proposals", []).append(
                    {"kind": "task_graph", "status": "failed_fallback_disabled", "fallback_used": False, "error": str(exc), "proposal": plan}
                )
                context.setdefault("plan_revision_trace", []).append({"stage": "initial_plan", "status": "failed", "error": str(exc), "fallback_disabled": True})
                return plan
            plan = self.fallback.build_plan(context, self.registry)
            context.setdefault("warnings", []).append(f"llm_task_graph_planning_fallback: {exc}")
            context.setdefault("llm_plan_proposals", []).append(
                {"kind": "task_graph", "status": "failed_used_fallback", "fallback_used": True, "error": str(exc), "proposal": plan}
            )
            context.setdefault("plan_revision_trace", []).append({"stage": "initial_plan", "status": "fallback", "error": str(exc)})
            return plan

    def unavailable_stop_action(self, context: Dict[str, Any]) -> AgentAction | None:
        """Fail closed before deterministic policies when LLM fallback is forbidden."""
        unavailable = self.client is None or self.disabled_reason or not getattr(self.client, "available", False)
        if unavailable and not self._fallback_allowed(context):
            return self._stop_without_fallback(context, self.disabled_reason or "llm_unavailable")
        return None

    def _stop_without_fallback(self, context: Dict[str, Any], reason: str) -> AgentAction:
        metadata = context.setdefault("metadata", {})
        metadata["last_action_source"] = "llm_failed_no_fallback"
        metadata["planner_type"] = "llm_failed"
        metadata["orchestrator_mode"] = "llm_orchestration_failed_no_fallback"
        context.setdefault("llm_plan_proposals", []).append(
            {"kind": "action", "status": "failed_fallback_disabled", "fallback_used": False, "reason": reason}
        )
        return AgentAction("OrchestratorAgent", "finish", rationale=f"LLM orchestration unavailable and fallbacks disabled: {reason}")

    def next_action(self, context: Dict[str, Any]) -> AgentAction:
        terminal_failure = self.unavailable_stop_action(context)
        if terminal_failure is not None:
            return terminal_failure
        safe_action = self.fallback.safe_abstention_action(context)
        if safe_action is not None:
            context.setdefault("metadata", {})["last_action_source"] = "safety_policy"
            context.setdefault("llm_plan_proposals", []).append(
                {
                    "kind": "action",
                    "status": "safety_policy_override",
                    "fallback_used": False,
                    "reason": "unrecoverable_prediction_failure",
                    "action": safe_action.to_dict(),
                }
            )
            return safe_action
        open_task_action = self.fallback.open_task_action(context)
        if open_task_action is not None:
            context.setdefault("metadata", {})["last_action_source"] = "open_task_contract"
            context.setdefault("llm_plan_proposals", []).append(
                {
                    "kind": "action",
                    "status": "open_task_contract_override",
                    "fallback_used": False,
                    "reason": "request_derived_composite_protocol",
                    "action": open_task_action.to_dict(),
                }
            )
            return open_task_action
        evidence_boundary_action = self.fallback.no_direct_evidence_action(context)
        if evidence_boundary_action is not None:
            context.setdefault("metadata", {})["last_action_source"] = "evidence_boundary_policy"
            context.setdefault("llm_plan_proposals", []).append(
                {
                    "kind": "action",
                    "status": "evidence_boundary_policy_override",
                    "fallback_used": False,
                    "reason": "verified_dense_hybrid_no_direct_evidence",
                    "action": evidence_boundary_action.to_dict(),
                }
            )
            return evidence_boundary_action
        if self.client is None or self.disabled_reason or not getattr(self.client, "available", False):
            fallback_action = self.fallback.next_action(context)
            context.setdefault("metadata", {})["last_action_source"] = "heuristic_fallback"
            context.setdefault("llm_plan_proposals", []).append(
                {
                    "kind": "action",
                    "status": "llm_unavailable_used_fallback",
                    "fallback_used": True,
                    "reason": self.disabled_reason or "llm_unavailable",
                    "fallback_action": fallback_action.to_dict(),
                }
            )
            return fallback_action
        try:
            payload = self._chat_json_payload(context, self._messages(context), stage="next_action")
            if "action" in payload and isinstance(payload["action"], dict):
                payload = payload["action"]
            action = AgentAction.from_mapping(payload)
            terminal = self._terminal_action(action)
            if terminal is not None:
                context.setdefault("metadata", {})["last_action_source"] = "llm"
                context.setdefault("llm_plan_proposals", []).append(
                    {
                        "kind": "action",
                        "status": "accepted_terminal",
                        "fallback_used": False,
                        "raw": payload,
                        "action": terminal.to_dict(),
                    }
                )
                return terminal
            if action.agent == "ReflectionAgent" and not self._reflection_enabled(context):
                replacement = self._reflection_disabled_action(context, action)
                context.setdefault("warnings", []).append(
                    f"llm_orchestrator_reflection_rejected_by_condition: {action.tool}"
                )
                context.setdefault("metadata", {})["last_action_source"] = "factorial_condition_gate"
                context.setdefault("llm_plan_proposals", []).append(
                    {
                        "kind": "action",
                        "status": "rejected_by_reflection_disabled_condition",
                        "fallback_used": False,
                        "raw": payload,
                        "rejected_action": action.to_dict(),
                        "action": replacement.to_dict(),
                    }
                )
                return replacement
            if action.agent in self.registry.agents() and action.tool in self.registry.allowlist_for_agent(action.agent):
                context.setdefault("metadata", {})["last_action_source"] = "llm"
                context.setdefault("llm_plan_proposals", []).append(
                    {"kind": "action", "status": "accepted_pending_guard", "fallback_used": False, "raw": payload, "action": action.to_dict()}
                )
                return action
            context.setdefault("warnings", []).append(f"llm_orchestrator_action_rejected: {action.to_dict()}")
            context.setdefault("llm_plan_proposals", []).append(
                {"kind": "action", "status": "rejected_used_fallback", "fallback_used": True, "raw": payload, "action": action.to_dict()}
            )
            if not context.get("metadata", {}).get("enable_fallbacks", False):
                context["llm_plan_proposals"][-1]["status"] = "rejected_fallback_disabled"
                context["llm_plan_proposals"][-1]["fallback_used"] = False
                return AgentAction("OrchestratorAgent", "finish", rationale="LLM action was rejected and fallbacks are disabled.")
            if context.get("metadata", {}).get("require_llm_orchestrator"):
                context["llm_plan_proposals"][-1]["status"] = "rejected_llm_required"
                context["llm_plan_proposals"][-1]["fallback_used"] = False
                return AgentAction("OrchestratorAgent", "finish", rationale="LLM action was rejected and heuristic fallback is disallowed by require_llm_orchestrator.")
        except Exception as exc:
            self.disabled_reason = str(exc)
            self.client = None
            context.setdefault("warnings", []).append(f"llm_orchestrator_fallback: {exc}")
            if not self._fallback_allowed(context):
                return self._stop_without_fallback(context, str(exc))
            context.setdefault("llm_plan_proposals", []).append(
                {"kind": "action", "status": "failed_used_fallback", "fallback_used": True, "error": str(exc)}
            )
        fallback_action = self.fallback.next_action(context)
        context.setdefault("metadata", {})["last_action_source"] = "heuristic_fallback"
        context["llm_plan_proposals"][-1]["fallback_action"] = fallback_action.to_dict()
        return fallback_action

    def revise_after_guard(self, context: Dict[str, Any], blocked_action: AgentAction, error: Exception) -> AgentAction | None:
        if self.client is not None and not self.disabled_reason and getattr(self.client, "available", False):
            try:
                payload = self._chat_json_payload(
                    context,
                    self._revision_messages(context, blocked_action, error),
                    stage="guard_revision",
                )
                if "action" in payload and isinstance(payload["action"], dict):
                    payload = payload["action"]
                revised = AgentAction.from_mapping(payload)
                if revised.agent == "ReflectionAgent" and not self._reflection_enabled(context):
                    context.setdefault("llm_plan_proposals", []).append(
                        {
                            "kind": "action_revision",
                            "status": "rejected_by_reflection_disabled_condition",
                            "fallback_used": False,
                            "blocked_action": blocked_action.to_dict(),
                            "guard_error": str(error),
                            "raw": payload,
                        }
                    )
                    return self._reflection_disabled_action(context, revised)
                if revised.agent in self.registry.agents() and revised.tool in self.registry.allowlist_for_agent(revised.agent):
                    context.setdefault("metadata", {})["last_action_source"] = "llm_revision_after_guard"
                    context.setdefault("llm_plan_proposals", []).append(
                        {
                            "kind": "action_revision",
                            "status": "accepted_pending_guard",
                            "fallback_used": False,
                            "blocked_action": blocked_action.to_dict(),
                            "guard_error": str(error),
                            "raw": payload,
                            "action": revised.to_dict(),
                        }
                    )
                    context.setdefault("plan_revision_trace", []).append(
                        {
                            "stage": "action_guard",
                            "status": "llm_revised_after_guard",
                            "blocked_action": blocked_action.to_dict(),
                            "guard_error": str(error),
                            "revised_action": revised.to_dict(),
                        }
                    )
                    return revised
                context.setdefault("llm_plan_proposals", []).append(
                    {
                        "kind": "action_revision",
                        "status": "rejected_unknown_tool",
                        "fallback_used": False,
                        "blocked_action": blocked_action.to_dict(),
                        "guard_error": str(error),
                        "raw": payload,
                    }
                )
            except Exception as exc:
                self.disabled_reason = str(exc)
                self.client = None
                context.setdefault("llm_plan_proposals", []).append(
                    {
                        "kind": "action_revision",
                        "status": "failed",
                        "fallback_used": False,
                        "blocked_action": blocked_action.to_dict(),
                        "guard_error": str(error),
                        "error": str(exc),
                    }
                )
        if not self._fallback_allowed(context):
            context.setdefault("plan_revision_trace", []).append(
                {
                    "stage": "action_guard",
                    "status": "blocked_no_revision_fallback_disabled",
                    "blocked_action": blocked_action.to_dict(),
                    "guard_error": str(error),
                }
            )
            return None
        revised = self.fallback.next_action(context)
        context.setdefault("metadata", {})["last_action_source"] = "heuristic_fallback_after_guard"
        context.setdefault("plan_revision_trace", []).append(
            {
                "stage": "action_guard",
                "status": "revised_to_fallback",
                "blocked_action": blocked_action.to_dict(),
                "guard_error": str(error),
                "revised_action": revised.to_dict(),
            }
        )
        return revised

    def _chat_json_payload(
        self,
        context: Dict[str, Any],
        messages: List[Dict[str, str]],
        *,
        stage: str,
    ) -> Dict[str, Any]:
        """Call the same LLM planner with a bounded, fully audited retry loop."""

        last_error: Exception | None = None
        attempts = self.max_llm_retries + 1
        for attempt_index in range(attempts):
            attempt_number = attempt_index + 1
            try:
                response = self.client.chat_json(messages, temperature=self.temperature)
                payload = json.loads(response) if isinstance(response, str) else response
                if not isinstance(payload, dict):
                    raise ValueError("LLM orchestrator response must be a JSON object")
                self._validate_stage_payload(payload, stage=stage)
                context.setdefault("llm_retry_trace", []).append(
                    {
                        "stage": stage,
                        "attempt": attempt_number,
                        "max_attempts": attempts,
                        "status": "completed",
                        "retry_used": attempt_number > 1,
                    }
                )
                return payload
            except Exception as exc:
                last_error = exc
                retry_scheduled = attempt_number < attempts
                delay = self.retry_backoff_seconds * (2**attempt_index) if retry_scheduled else 0.0
                context.setdefault("llm_retry_trace", []).append(
                    {
                        "stage": stage,
                        "attempt": attempt_number,
                        "max_attempts": attempts,
                        "status": "failed",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "retry_scheduled": retry_scheduled,
                        "backoff_seconds": delay,
                    }
                )
                if retry_scheduled and delay > 0:
                    time.sleep(delay)
        assert last_error is not None
        raise last_error

    @staticmethod
    def _validate_stage_payload(payload: Dict[str, Any], *, stage: str) -> None:
        """Reject malformed action payloads inside the bounded retry loop."""

        if stage not in {"next_action", "guard_revision"}:
            return
        candidate: Any = payload.get("action") if "action" in payload else payload
        if not isinstance(candidate, dict):
            raise ValueError(f"LLM orchestrator {stage} response must contain an action object")
        agent = candidate.get("agent")
        tool = candidate.get("tool")
        if not isinstance(agent, str) or not agent.strip() or not isinstance(tool, str) or not tool.strip():
            raise ValueError(
                f"LLM orchestrator {stage} response requires non-empty string agent and tool fields"
            )
        args = candidate.get("args", {})
        if args is not None and not isinstance(args, dict):
            raise ValueError(f"LLM orchestrator {stage} response args must be a JSON object")

    def _messages(self, context: Dict[str, Any]) -> List[Dict[str, str]]:
        reflection_enabled = self._reflection_enabled(context)
        task_contract = context.get("metadata", {}).get("task_contract")
        summary = {
            "request": context.get("request"),
            "task_type": context.get("task_type"),
            "dataset_name": context.get("dataset_name"),
            "mode": context.get("metadata", {}).get("mode"),
            "n_molecules": len(context.get("molecules", []) or []),
            "n_rules": len(context.get("rules", []) or []),
            "n_features": len(context.get("features", []) or []),
            "has_ddi_pair": bool(context.get("ddi_pair")),
            "n_predictions": len(context.get("predictions", []) or []),
            "n_evidence": len(context.get("evidence", []) or []),
            "confidence_assessed": bool(context.get("confidence")),
            "has_final_report": bool(context.get("final_report")),
            "reflection_enabled": reflection_enabled,
            "completed_actions": context.get("metadata", {}).get("completed_actions", []),
            "verifier_report": context.get("verifier_report", {}),
            "required_protocol": self._required_protocol(context),
            "task_contract": task_contract,
            "open_task_progress": self._open_task_progress(context),
        }
        return [
            {
                "role": "system",
                "content": (
                    "You are AgentM OrchestratorAgent. Choose exactly one next specialized-agent tool call. "
                    "Return JSON only: {\"agent\":\"...\",\"tool\":\"...\",\"args\":{},\"rationale\":\"...\"}. "
                    "Never invent prediction labels, scores, probabilities, citation ids, or source paths. "
                    "ADMET/DDI predictions must come from GaugeAgent tools; reaction type/yield predictions must come from ReactionAgent tools. "
                    "Citations must come from RetrieveAgent retrieval tools. "
                    "Respect tool prerequisites exactly: route first; resolve molecules before rules/features or parse reaction before reaction evidence/prediction; build_features before ADMET/DDI prediction; "
                    "assess confidence after ADMET/DDI prediction; retrieve evidence after prediction for endpoint tasks; "
                    "for task_type general_drug_profile, call RetrieveAgent.retrieve_external_evidence immediately after molecule resolution "
                    "to collect PubChem/DailyMed/ChEMBL context before build_features, RAG retrieval, reasoning, or report building; "
                    "for task_type general_drug_profile, never choose discover_rules, predict_single, predict_ddi, assess_confidence, "
                    "run_zero_shot_fallback, or verify_prediction_grounding because no endpoint prediction was requested; "
                    "for task_type out_of_domain or missing_entity, do not resolve molecules, retrieve evidence, or run prediction tools; "
                    "build an intent-gate report, verify schema, then write artifacts; "
                    "build report before schema verification. "
                    + (
                        "This is a composite open task. Complete every request-derived subtask in task_contract before reporting. "
                        "Use the exact subtask_id, task_type, dataset_name, and molecules in every scoped resolve/build_features/predict/retrieve call. "
                        "The compact composite protocol intentionally omits only discover_rules because build_features loads the registered model schema; assess confidence separately for every requested prediction. "
                        "Never finish after only one endpoint, and never merge pair-level evidence with single-drug evidence. "
                        if isinstance(task_contract, dict)
                        else ""
                    )
                    + (
                        "Reflection is enabled: use ReflectionAgent only after report and verifier checks."
                        if reflection_enabled
                        else "Reflection is disabled for this factorial condition: never select ReflectionAgent, reflect_on_report, or apply_reflection_feedback; write artifacts immediately after verification."
                    )
                ),
            },
            {
                "role": "user",
                "content": json.dumps({"context": summary, "tool_schemas": self._tool_schemas(context)}, ensure_ascii=False),
            },
        ]

    def _revision_messages(self, context: Dict[str, Any], blocked_action: AgentAction, error: Exception) -> List[Dict[str, str]]:
        summary = {
            "request": context.get("request"),
            "task_type": context.get("task_type"),
            "dataset_name": context.get("dataset_name"),
            "n_molecules": len(context.get("molecules", []) or []),
            "n_rules": len(context.get("rules", []) or []),
            "n_features": len(context.get("features", []) or []),
            "n_predictions": len(context.get("predictions", []) or []),
            "n_evidence": len(context.get("evidence", []) or []),
            "completed_actions": context.get("metadata", {}).get("completed_actions", []),
            "required_protocol": self._required_protocol(context),
            "blocked_action": blocked_action.to_dict(),
            "guard_error": str(error),
            "reflection_enabled": self._reflection_enabled(context),
            "task_contract": context.get("metadata", {}).get("task_contract"),
            "open_task_progress": self._open_task_progress(context),
        }
        return [
            {
                "role": "system",
                "content": (
                    "You are AgentM OrchestratorAgent repairing a guard-blocked action. "
                    "Return exactly one corrected tool call as JSON: {\"agent\":\"...\",\"tool\":\"...\",\"args\":{},\"rationale\":\"...\"}. "
                    "Do not repeat the blocked action. Choose the earliest unmet prerequisite from required_protocol."
                ),
            },
            {"role": "user", "content": json.dumps({"context": summary, "tool_schemas": self._tool_schemas(context)}, ensure_ascii=False)},
        ]

    def _plan_messages(self, context: Dict[str, Any]) -> List[Dict[str, str]]:
        reflection_enabled = self._reflection_enabled(context)
        summary = {
            "request": context.get("request"),
            "task_type_hint": context.get("task_type"),
            "dataset_name_hint": context.get("dataset_name"),
            "mode": context.get("metadata", {}).get("mode"),
            "max_rounds": context.get("metadata", {}).get("max_rounds"),
            "previous_run_loaded": bool(context.get("previous_run_context")),
            "reflection_enabled": reflection_enabled,
            "task_contract": context.get("metadata", {}).get("task_contract"),
        }
        return [
            {
                "role": "system",
                "content": (
                    "You are AgentM OrchestratorAgent. Build a FROGENT-style multi-agent task graph. "
                    "Return JSON only with fields: objective, task_graph {nodes:[{id,agent,goal}], edges:[[from,to]]}, "
                    "execution_strategy, stop_conditions. Use only allowed agents/tools from tool_schemas. "
                    "Do not include prediction labels, scores, probabilities, citation ids, or fabricated sources. "
                    "For general_drug_profile requests, include RetrieveAgent.retrieve_external_evidence immediately after molecule resolution "
                    "so public database context is gathered before structural descriptors, RAG, reasoning, and reporting. "
                    "For general_drug_profile, do not plan predict_single, predict_ddi, assess_confidence, or verify_prediction_grounding; "
                    "only plan endpoint prediction tools when the user explicitly asks for a concrete ADMET/DDI endpoint. "
                    + (
                        "For a composite task_contract, include every listed subtask or reaction operation, preserve its subtask_id and evidence scope, and do not stop after the root task alone. "
                        if isinstance(context.get("metadata", {}).get("task_contract"), dict)
                        else ""
                    )
                    + (
                        "Reflection is enabled, so a bounded ReflectionAgent stage may follow verification."
                        if reflection_enabled
                        else "Reflection is disabled for this factorial condition; omit ReflectionAgent and all reflection steps from the graph."
                    )
                ),
            },
            {
                "role": "user",
                "content": json.dumps({"context": summary, "tool_schemas": self._tool_schemas(context)}, ensure_ascii=False),
            },
        ]

    def _normalize_plan(self, payload: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        task_graph = payload.get("task_graph") if isinstance(payload.get("task_graph"), dict) else {}
        nodes = task_graph.get("nodes") if isinstance(task_graph.get("nodes"), list) else []
        edges = task_graph.get("edges") if isinstance(task_graph.get("edges"), list) else []
        allowed_agents = set(self.registry.agents())
        if not self._reflection_enabled(context):
            allowed_agents.discard("ReflectionAgent")
        normalized_nodes = []
        for idx, node in enumerate(nodes):
            if not isinstance(node, dict):
                continue
            agent = str(node.get("agent") or "")
            if agent and agent not in allowed_agents:
                continue
            normalized = self._normalize_plan_node(node, idx, context)
            if normalized is not None:
                normalized_nodes.append(normalized)
        if context.get("task_type") == "general_drug_profile":
            normalized_nodes = self._deduplicate_general_profile_nodes(normalized_nodes)
            edges = [[normalized_nodes[i]["id"], normalized_nodes[i + 1]["id"]] for i in range(len(normalized_nodes) - 1)]
        else:
            valid_ids = {node["id"] for node in normalized_nodes}
            edges = [
                edge
                for edge in edges
                if isinstance(edge, list) and len(edge) == 2 and edge[0] in valid_ids and edge[1] in valid_ids
            ]
        if not normalized_nodes:
            if not self._fallback_allowed(context):
                return self._disabled_plan(context, "llm_plan_invalid")
            fallback = self.fallback.build_plan(context, self.registry)
            fallback["planner_type"] = "llm_invalid_fallback"
            return fallback
        return {
            "status": "planned",
            "planner_type": "llm",
            "objective": str(payload.get("objective") or context.get("request") or ""),
            "task_type_hint": context.get("task_type") or "unknown_until_route_task",
            "agents": sorted(allowed_agents),
            "task_graph": {"status": "planned", "nodes": normalized_nodes, "edges": edges},
            "execution_strategy": payload.get("execution_strategy") or "LLM-generated task graph with guarded tool execution.",
            "stop_conditions": self._condition_stop_conditions(payload.get("stop_conditions"), context),
        }

    def _normalize_plan_node(self, node: Dict[str, Any], idx: int, context: Dict[str, Any]) -> Dict[str, str] | None:
        agent = str(node.get("agent") or "") or "OrchestratorAgent"
        node_id = str(node.get("id") or f"task_{idx + 1}")
        goal = str(node.get("goal") or node.get("description") or "")
        if context.get("task_type") != "general_drug_profile":
            return {"id": node_id, "agent": agent, "goal": goal}
        text = f"{node_id} {goal}".lower()
        if agent == "GaugeAgent" and self._general_profile_prediction_text(text):
            return {
                "id": "build_features",
                "agent": "GaugeAgent",
                "goal": "Build structural descriptors for the general drug profile; do not run endpoint prediction unless a concrete endpoint is requested.",
            }
        if agent == "VerifierAgent" and ("prediction" in text or "gaugeagent" in text or "tool trace" in text):
            return None
        return {"id": node_id, "agent": agent, "goal": goal}

    def _general_profile_prediction_text(self, text: str) -> bool:
        markers = ["predict", "prediction", "model", "score", "probability", "pharmacological", "safety attribute"]
        return any(marker in text for marker in markers)

    def _deduplicate_general_profile_nodes(self, nodes: List[Dict[str, str]]) -> List[Dict[str, str]]:
        deduped: List[Dict[str, str]] = []
        seen_keys: set[tuple[str, str]] = set()
        for node in nodes:
            key = (node.get("agent", ""), node.get("id", ""))
            if key in seen_keys:
                continue
            seen_keys.add(key)
            deduped.append(node)
        return deduped

    def _fallback_allowed(self, context: Dict[str, Any]) -> bool:
        metadata = context.get("metadata", {}) or {}
        return bool(metadata.get("enable_fallbacks", False)) and not bool(metadata.get("require_llm_orchestrator", False))

    def _reflection_enabled(self, context: Dict[str, Any]) -> bool:
        metadata = context.get("metadata", {}) or {}
        return bool(metadata.get("enable_reflection", True)) and int(metadata.get("max_reflection_rounds", 1) or 0) > 0

    def _tool_schemas(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        schemas = self.registry.schemas()
        if self._reflection_enabled(context):
            return schemas
        return [schema for schema in schemas if schema.get("agent") != "ReflectionAgent"]

    def _condition_stop_conditions(self, value: Any, context: Dict[str, Any]) -> List[str]:
        if isinstance(value, list):
            conditions = [str(item) for item in value]
        elif value:
            conditions = [str(value)]
        else:
            conditions = ["verifier_pass", "reflection_applied", "budget_not_exceeded"]
        if not self._reflection_enabled(context):
            conditions = [condition for condition in conditions if "reflect" not in condition.lower()]
        return conditions or ["verifier_pass", "budget_not_exceeded"]

    def _reflection_disabled_action(self, context: Dict[str, Any], rejected: AgentAction) -> AgentAction:
        """Enforce the prespecified reflection-off condition without a heuristic fallback.

        A reflection proposal is only expected after the report and verifier
        stages. At that point the condition gate advances directly to artifact
        writing. An early reflection proposal terminates visibly instead of
        contaminating the LLM-orchestration arm with a heuristic recovery.
        """

        calls = context.get("tool_calls", []) or []
        schema_checked = any(
            call.get("tool") == "verify_report_schema" and call.get("status") in {"completed", "fallback_completed"}
            for call in calls
        )
        if context.get("final_report") and schema_checked:
            return AgentAction(
                "ReporterAgent",
                "write_artifacts",
                rationale="Reflection is disabled by the factorial condition; proceed from verified report to artifact writing.",
            )
        return AgentAction(
            "OrchestratorAgent",
            "finish",
            rationale=f"LLM proposed forbidden reflection action before verification completed: {rejected.tool}",
        )

    def _terminal_action(self, action: AgentAction) -> AgentAction | None:
        if action.agent != "OrchestratorAgent":
            return None
        tool = str(action.tool or "").strip().lower().replace("-", "_").replace(" ", "_")
        terminal_tools = {
            "finish",
            "done",
            "terminate",
            "stop",
            "complete",
            "completed",
            "complete_task",
            "task_complete",
            "final",
            "finalize",
            "end",
            "end_task",
        }
        if tool not in terminal_tools:
            return None
        return AgentAction("OrchestratorAgent", "finish", args={}, rationale=action.rationale or "LLM orchestrator marked the task complete.")

    def _required_protocol(self, context: Dict[str, Any]) -> List[str]:
        contract = context.get("metadata", {}).get("task_contract")
        if isinstance(contract, dict):
            if contract.get("open_task_type") == "reaction_multi_tool":
                return [
                    "route_task",
                    "parse_reaction",
                    "retrieve_mit_template_evidence (retry after a failed transient first call)",
                    "retrieve_orderly_condition_evidence",
                    "classify_reaction_type",
                    "predict_buchwald_hartwig_yield",
                    "build_reaction_profile",
                    "build_structured_explanation",
                    "build_runtime_report",
                    "verify_prediction_grounding",
                    "verify_citation_grounding",
                    "verify_report_schema",
                    "write_artifacts",
                ]
            protocol = ["route_task"]
            for subtask in contract.get("subtasks", []) or []:
                if not isinstance(subtask, dict):
                    continue
                args = {
                    "subtask_id": subtask.get("subtask_id"),
                    "task_type": subtask.get("task_type"),
                    "dataset_name": subtask.get("dataset_name"),
                    "molecules": subtask.get("molecules"),
                }
                resolve = "resolve_ddi_pair" if subtask.get("kind") == "ddi" else "resolve_molecule"
                predict = "predict_ddi" if subtask.get("kind") == "ddi" else "predict_single"
                arg_text = json.dumps(args, ensure_ascii=False, separators=(",", ":"))
                protocol.extend(
                    [
                        f"{resolve} args={arg_text}",
                        f"build_features args={arg_text}",
                        f"{predict} args={arg_text}",
                        f"assess_confidence args={arg_text}",
                        f"retrieve_elasticsearch_hybrid args={arg_text}",
                    ]
                )
            protocol.extend(
                [
                    "build_structured_explanation after every scoped prediction and retrieval is complete",
                    "build_runtime_report",
                    "verify_prediction_grounding",
                    "verify_citation_grounding",
                    "verify_report_schema",
                    "write_artifacts",
                ]
            )
            return protocol
        if str(context.get("task_type") or "").startswith("reaction_"):
            protocol = [
                "route_task",
                "parse_reaction",
                "retrieve_mit_template_evidence",
                "retrieve_orderly_condition_evidence",
            ]
            if context.get("task_type") == "reaction_classification":
                protocol.append("classify_reaction_type")
            if context.get("task_type") == "reaction_yield_prediction":
                protocol.append("predict_buchwald_hartwig_yield")
            protocol.extend(
                [
                    "build_reaction_profile",
                    "build_structured_explanation",
                    "build_runtime_report",
                    "verify_prediction_grounding when a reaction prediction is requested",
                    "verify_citation_grounding",
                    "verify_report_schema",
                    "reflect_on_report",
                    "apply_reflection_feedback",
                    "write_artifacts",
                ]
            )
            return self._condition_protocol(protocol, context)
        if context.get("task_type") == "general_drug_profile":
            return self._condition_protocol([
                "route_task",
                "resolve_molecule",
                "retrieve_external_evidence",
                "build_features",
                "retrieve_elasticsearch_hybrid or retrieve_local_rag (optional broader evidence)",
                "build_structured_explanation",
                "build_runtime_report",
                "verify_citation_grounding",
                "verify_report_schema",
                "reflect_on_report",
                "apply_reflection_feedback",
                "write_artifacts",
            ], context)
        if context.get("task_type") in INTENT_GATE_TASK_TYPES:
            return ["route_task", "build_runtime_report", "verify_report_schema", "write_artifacts"]
        if context.get("task_type") == "ddi_prediction":
            return self._condition_protocol([
                "route_task",
                "resolve_ddi_pair",
                "discover_rules",
                "build_features",
                "predict_ddi",
                "assess_confidence",
                "retrieve_elasticsearch_hybrid or retrieve_local_rag",
                "build_structured_explanation",
                "build_runtime_report",
                "verify_prediction_grounding",
                "verify_citation_grounding",
                "verify_report_schema",
                "reflect_on_report",
                "apply_reflection_feedback",
                "write_artifacts",
            ], context)
        return self._condition_protocol([
            "route_task",
            "resolve_molecule",
            "discover_rules",
            "build_features",
            "predict_single",
            "assess_confidence",
            "retrieve_elasticsearch_hybrid or retrieve_local_rag",
            "build_structured_explanation",
            "build_runtime_report",
            "verify_prediction_grounding",
            "verify_citation_grounding",
            "verify_report_schema",
            "reflect_on_report",
            "apply_reflection_feedback",
            "write_artifacts",
        ], context)

    def _open_task_progress(self, context: Dict[str, Any]) -> Dict[str, Any]:
        contract = context.get("metadata", {}).get("task_contract")
        if not isinstance(contract, dict):
            return {}
        completed = {
            (str(call.get("tool") or ""), str((call.get("args") or {}).get("subtask_id") or ""))
            for call in context.get("tool_calls", []) or []
            if call.get("status") in {"completed", "fallback_completed"}
        }
        if contract.get("open_task_type") == "reaction_multi_tool":
            required = [
                "parse_reaction",
                "retrieve_mit_template_evidence",
                "retrieve_orderly_condition_evidence",
                "classify_reaction_type",
                "predict_buchwald_hartwig_yield",
            ]
            done_tools = {tool for tool, _ in completed}
            return {
                "operations": {tool: tool in done_tools for tool in required},
                "all_scientific_subtasks_complete": all(tool in done_tools for tool in required),
            }
        progress: Dict[str, Any] = {}
        all_complete = True
        for subtask in contract.get("subtasks", []) or []:
            if not isinstance(subtask, dict):
                continue
            subtask_id = str(subtask.get("subtask_id") or "")
            required = [
                "resolve_ddi_pair" if subtask.get("kind") == "ddi" else "resolve_molecule",
                "build_features",
                "predict_ddi" if subtask.get("kind") == "ddi" else "predict_single",
                "assess_confidence",
                "retrieve_elasticsearch_hybrid",
            ]
            status = {tool: (tool, subtask_id) in completed for tool in required}
            status["complete"] = all(status.values())
            all_complete = all_complete and bool(status["complete"])
            progress[subtask_id] = status
        return {"subtasks": progress, "all_scientific_subtasks_complete": all_complete}

    def _condition_protocol(self, protocol: List[str], context: Dict[str, Any]) -> List[str]:
        if self._reflection_enabled(context):
            return protocol
        return [step for step in protocol if "reflect" not in step.lower()]

    def _disabled_plan(self, context: Dict[str, Any], reason: str) -> Dict[str, Any]:
        context.setdefault("metadata", {})["planner_type"] = "llm_failed"
        context["metadata"]["orchestrator_mode"] = "llm_orchestration_failed_no_fallback"
        return {
            "status": "failed",
            "planner_type": "llm_failed",
            "objective": context.get("request"),
            "task_type_hint": context.get("task_type") or "unknown_until_route_task",
            "task_graph": {"status": "failed", "nodes": [], "edges": [], "reason": reason},
            "execution_strategy": "No fallback execution was allowed.",
            "stop_conditions": ["fallback_disabled"],
        }


class OrchestratorAgent:
    def __init__(
        self,
        *,
        registry: MultiAgentToolRegistry,
        llm_profile: Optional[str] = None,
        temperature: float = 0.1,
        use_llm: bool = True,
        llm_client: Any | None = None,
        max_llm_retries: int = 0,
        retry_backoff_seconds: float = 0.0,
    ) -> None:
        self.registry = registry
        self.use_llm = bool(use_llm)
        self.deterministic = DeterministicMultiAgentPlanner()
        self.planner = (
            LLMOrchestratorPlanner(
                registry=registry,
                llm_profile=llm_profile,
                temperature=temperature,
                fallback=self.deterministic,
                llm_client=llm_client,
                max_llm_retries=max_llm_retries,
                retry_backoff_seconds=retry_backoff_seconds,
            )
            if use_llm
            else self.deterministic
        )

    def build_plan(self, context: Dict[str, Any]) -> Dict[str, Any]:
        if isinstance(self.planner, DeterministicMultiAgentPlanner):
            plan = self.planner.build_plan(context, self.registry)
            if not self.use_llm:
                plan["planner_type"] = "heuristic"
                context.setdefault("metadata", {})["planner_type"] = "heuristic"
                context["metadata"]["orchestrator_mode"] = "heuristic"
        elif hasattr(self.planner, "build_plan"):
            plan = self.planner.build_plan(context)
        else:
            plan = self.deterministic.build_plan(context, self.registry)
        context["orchestrator_plan"] = plan
        context["task_graph"] = plan.get("task_graph", {})
        return plan

    def next_action(self, context: Dict[str, Any]) -> AgentAction:
        # A failed mandatory/non-fallback LLM is a terminal runtime failure,
        # not an early successful completion.  It outranks reflection and
        # all deterministic protocol prerequisites, including an already
        # started reflection round.  Guarded finalization still writes the
        # failure diagnostics without performing further scientific work.
        if isinstance(self.planner, LLMOrchestratorPlanner):
            terminal_failure = self.planner.unavailable_stop_action(context)
            if terminal_failure is not None:
                return terminal_failure
        # Reflection is a prespecified factorial condition, not an optional
        # suggestion that the LLM planner may terminate around.  Once a
        # reflection round has started, finish the deterministic protocol
        # (including any still-missing verifier steps and feedback
        # application) before asking the LLM for another action.  This avoids
        # spending an extra planning call and prevents a one-step reflection
        # trace from being finalized as a completed reflection-on run.
        pending = self._pending_reflection_protocol_action(context)
        if pending is not None:
            return self._record_factorial_protocol_override(
                context,
                proposed=None,
                required=pending,
                reason="reflection_round_started_but_protocol_incomplete",
            )

        action = self.planner.next_action(context)
        # Planning can fail during this call, after the entry check.  Do not
        # turn that explicit failure stop into another reflection/tool step.
        if (
            action.agent == "OrchestratorAgent"
            and action.tool == "finish"
            and context.get("metadata", {}).get("last_action_source") == "llm_failed_no_fallback"
        ):
            return action
        reflection_enabled = self._reflection_condition_enabled(context)
        verifier_complete = bool((context.get("verifier_report", {}) or {}).get("summary"))

        # Reflection must consume completed verifier output.  If an LLM asks
        # to reflect early, advance through the deterministic protocol until
        # the verifier summary exists.
        if (
            reflection_enabled
            and action.agent == "ReflectionAgent"
            and action.tool == "reflect_on_report"
            and (not verifier_complete or not self.deterministic._should_reflect(context))
        ):
            required = self.deterministic.next_action(context)
            if (required.agent, required.tool) != (action.agent, action.tool):
                return self._record_factorial_protocol_override(
                    context,
                    proposed=action,
                    required=required,
                    reason="reflection_requires_completed_verifier_output",
                )

        # A reflection-on condition may not finish or persist artifacts until
        # the configured reflection round and its feedback-application step
        # are both complete.  Use the deterministic planner only as a
        # protocol gate for the earliest unmet prerequisite; this is recorded
        # explicitly and is not a heuristic fallback.
        if (
            reflection_enabled
            and self._is_finalization_action(action)
            and not self._reflection_protocol_complete(context)
        ):
            required = self.deterministic.next_action(context)
            if not self._is_finalization_action(required):
                return self._record_factorial_protocol_override(
                    context,
                    proposed=action,
                    required=required,
                    reason="finalization_blocked_until_reflection_protocol_complete",
                )
        return action

    def _reflection_condition_enabled(self, context: Dict[str, Any]) -> bool:
        metadata = context.get("metadata", {}) or {}
        return bool(metadata.get("enable_reflection", True)) and int(
            metadata.get("max_reflection_rounds", 1) or 0
        ) > 0

    def _reflection_protocol_complete(self, context: Dict[str, Any]) -> bool:
        if not self._reflection_condition_enabled(context):
            return True
        metadata = context.get("metadata", {}) or {}
        required_rounds = int(metadata.get("max_reflection_rounds", 1) or 0)
        completed_rounds = int(metadata.get("reflection_rounds", 0) or 0)
        applied_round = int(metadata.get("reflection_feedback_applied_round", 0) or 0)
        refresh_pending = self.deterministic._needs_reflection_retrieval(context) or any(
            metadata.get(key) for key in (
                "reasoning_stale", "report_stale", "citation_verification_stale", "schema_verification_stale"
            )
        )
        return completed_rounds >= required_rounds and applied_round >= completed_rounds and not refresh_pending

    def _pending_reflection_protocol_action(
        self, context: Dict[str, Any]
    ) -> AgentAction | None:
        if not self._reflection_condition_enabled(context):
            return None
        metadata = context.get("metadata", {}) or {}
        reflection_round = int(metadata.get("reflection_rounds", 0) or 0)
        applied_round = int(metadata.get("reflection_feedback_applied_round", 0) or 0)
        refresh_pending = self.deterministic._needs_reflection_retrieval(context) or any(
            metadata.get(key) for key in (
                "reasoning_stale", "report_stale", "citation_verification_stale", "schema_verification_stale"
            )
        )
        if reflection_round <= applied_round and not (reflection_round > 0 and refresh_pending):
            return None
        required = self.deterministic.next_action(context)
        return None if self._is_finalization_action(required) else required

    def _is_finalization_action(self, action: AgentAction) -> bool:
        return action.tool == "finish" or (
            action.agent == "ReporterAgent" and action.tool == "write_artifacts"
        )

    def _record_factorial_protocol_override(
        self,
        context: Dict[str, Any],
        *,
        proposed: AgentAction | None,
        required: AgentAction,
        reason: str,
    ) -> AgentAction:
        context.setdefault("metadata", {})["last_action_source"] = (
            "factorial_protocol_gate"
        )
        context.setdefault("llm_plan_proposals", []).append(
            {
                "kind": "action",
                "status": "factorial_protocol_gate_override",
                "fallback_used": False,
                "reason": reason,
                "proposed_action": proposed.to_dict() if proposed is not None else None,
                "action": required.to_dict(),
            }
        )
        context.setdefault("plan_revision_trace", []).append(
            {
                "stage": "factorial_protocol_gate",
                "status": "required_action_selected",
                "reason": reason,
                "proposed_action": proposed.to_dict() if proposed is not None else None,
                "required_action": required.to_dict(),
            }
        )
        return required

    def revise_after_guard(self, context: Dict[str, Any], blocked_action: AgentAction, error: Exception) -> AgentAction | None:
        if hasattr(self.planner, "revise_after_guard") and context.get("metadata", {}).get("last_action_source") == "llm":
            return self.planner.revise_after_guard(context, blocked_action, error)
        return None
