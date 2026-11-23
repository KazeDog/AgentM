from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict


class GlobalContext(TypedDict, total=False):
    request: str
    task_type: str
    dataset_name: str
    task_graph: Dict[str, Any]
    orchestrator_plan: Dict[str, Any]
    molecule_mentions: List[str]
    molecules: List[Any]
    rules: List[Any]
    features: List[Any]
    ddi_pair: Any
    reaction: Dict[str, Any]
    reaction_profile: Dict[str, Any]
    reaction_predictions: List[Dict[str, Any]]
    reaction_resource_resolution: Dict[str, Any]
    predictions: List[Any]
    confidence: Dict[str, Any]
    evidence: List[Dict[str, Any]]
    citation_map: List[Dict[str, Any]]
    reasoning: Dict[str, Any]
    reflection: Dict[str, Any]
    previous_run_context: Dict[str, Any]
    final_report: Dict[str, Any]
    compatible_report: Dict[str, Any]
    artifacts: Dict[str, Any]
    warnings: List[str]
    fallbacks: List[Dict[str, Any]]
    agent_messages: List[Dict[str, Any]]
    tool_calls: List[Dict[str, Any]]
    llm_plan_proposals: List[Dict[str, Any]]
    llm_retry_trace: List[Dict[str, Any]]
    plan_revision_trace: List[Dict[str, Any]]
    multi_agent_trace: List[Dict[str, Any]]
    retrieval_trace: List[Dict[str, Any]]
    verifier_report: Dict[str, Any]
    resource_resolution: Dict[str, Any]
    objective_status: Dict[str, Any]
    metadata: Dict[str, Any]


def initial_context(
    request: str,
    *,
    task_type: Optional[str] = None,
    dataset_name: Optional[str] = None,
    mode: str = "full",
    retrieval_backend: str = "auto",
    max_rounds: int = 16,
    llm_profile: Optional[str] = "qwen_flash",
    use_llm_orchestrator: bool = True,
    enable_fallbacks: bool = False,
    enable_reflection: bool = True,
    max_reflection_rounds: int = 1,
    auto_model: bool = False,
    model_registry_path: Optional[str] = None,
    require_real_model: bool = False,
    require_real_rag: bool = False,
    require_llm_orchestrator: bool = False,
    llm_orchestrator_max_retries: int = 0,
    llm_orchestrator_retry_backoff_seconds: float = 0.0,
    use_llm_reasoning: bool = True,
    require_llm_reasoning: bool = False,
    use_rag_evidence: bool = True,
    min_evidence: int = 1,
) -> GlobalContext:
    return GlobalContext(
        request=request,
        task_type=task_type or "",
        dataset_name=dataset_name or "",
        task_graph={"nodes": [], "edges": [], "status": "uninitialized"},
        orchestrator_plan={},
        molecule_mentions=[],
        molecules=[],
        rules=[],
        features=[],
        reaction={},
        reaction_profile={},
        reaction_predictions=[],
        reaction_resource_resolution={},
        predictions=[],
        confidence={},
        evidence=[],
        citation_map=[],
        reasoning={},
        reflection={},
        previous_run_context={},
        final_report={},
        compatible_report={},
        artifacts={},
        warnings=[],
        fallbacks=[],
        agent_messages=[],
        tool_calls=[],
        llm_plan_proposals=[],
        llm_retry_trace=[],
        plan_revision_trace=[],
        multi_agent_trace=[],
        retrieval_trace=[],
        verifier_report={},
        resource_resolution={},
        objective_status={},
        metadata={
            "runtime": "agentm_guarded_multi_agent",
            "mode": mode,
            "retrieval_backend_requested": retrieval_backend,
            "max_rounds": int(max_rounds),
            "llm_profile": llm_profile,
            "use_llm_orchestrator": bool(use_llm_orchestrator),
            "enable_fallbacks": bool(enable_fallbacks),
            "enable_reflection": bool(enable_reflection),
            "max_reflection_rounds": int(max_reflection_rounds),
            "auto_model": bool(auto_model),
            "model_registry_path": model_registry_path,
            "require_real_model": bool(require_real_model),
            "require_real_rag": bool(require_real_rag),
            "require_llm_orchestrator": bool(require_llm_orchestrator),
            "llm_orchestrator_max_retries": min(5, max(0, int(llm_orchestrator_max_retries))),
            "llm_orchestrator_retry_backoff_seconds": max(0.0, float(llm_orchestrator_retry_backoff_seconds)),
            "use_llm_reasoning": bool(use_llm_reasoning),
            "require_llm_reasoning": bool(require_llm_reasoning),
            "use_rag_evidence": bool(use_rag_evidence),
            "min_evidence": int(min_evidence),
            "loop_status": "initialized",
            "round": 0,
            "completed_actions": [],
            "evidence_expansion_rounds": 0,
            "reflection_rounds": 0,
        },
    )


def is_ddi(context: GlobalContext) -> bool:
    return context.get("task_type") == "ddi_prediction"


def prediction_present(context: GlobalContext) -> bool:
    return bool(context.get("predictions"))
