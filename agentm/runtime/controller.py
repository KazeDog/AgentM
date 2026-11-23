from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from agentm.runtime.agents import OrchestratorAgent
from agentm.runtime.compat import ensure_import_paths, to_jsonable
from agentm.runtime.graph.nodes import MultiAgentGraphNodes
from agentm.runtime.graph.workflow import build_workflow
from agentm.runtime.guards import GuardPolicy
from agentm.runtime.reaction import is_reaction_task
from agentm.runtime.resources import default_resource_path
from agentm.runtime.state import initial_context
from agentm.runtime.task_contracts import endpoint_addition_contract
from agentm.runtime.tools import ScientificToolRuntime

ensure_import_paths()

from agentm.defaults import (
    DEFAULT_DDI_RAG_CORPUS_DIR,
    DEFAULT_FEATURE_SET,
    DEFAULT_LLM_PROFILE,
    DEFAULT_RAG_BACKEND,
    DEFAULT_RAG_CORPUS_DIR,
    DEFAULT_RAG_EMBEDDING_MODEL_PATH,
    DEFAULT_RAG_TOP_K,
)
from agentm.utils.io import ensure_dir


class AgentMMultiAgentController:
    """FROGENT-style guarded multi-agent runtime for AgentM."""

    def __init__(
        self,
        output_dir: str | Path = "outputs/multi_agent",
        *,
        feature_set: str = DEFAULT_FEATURE_SET,
        model_path: Optional[str] = None,
        auto_model: bool = False,
        model_registry_path: Optional[str | Path] = None,
        require_real_model: bool = False,
        require_real_rag: bool = False,
        require_llm_orchestrator: bool = False,
        min_evidence: int = 1,
        mode: str = "full",
        retrieval_backend: str = "auto",
        es_url: Optional[str] = None,
        es_index_name: Optional[str] = None,
        max_rounds: int = 16,
        temperature: float = 0.1,
        llm_profile: Optional[str] = DEFAULT_LLM_PROFILE,
        use_llm_orchestrator: bool = True,
        orchestrator_llm_client: Any | None = None,
        llm_orchestrator_max_retries: int = 0,
        llm_orchestrator_retry_backoff_seconds: float = 0.0,
        use_llm_reasoning: bool = True,
        require_llm_reasoning: bool = False,
        reasoning_llm_client: Any | None = None,
        enable_fallbacks: bool = False,
        enable_reflection: bool = True,
        max_reflection_rounds: int = 1,
        previous_run_path: Optional[str | Path] = None,
        rag_corpus_dirs: Optional[List[str]] = None,
        rag_top_k: int = DEFAULT_RAG_TOP_K,
        rag_backend: str = DEFAULT_RAG_BACKEND,
        rag_embedding_model_name: str = DEFAULT_RAG_EMBEDDING_MODEL_PATH,
        ddi_rag_corpus_dir: str = DEFAULT_DDI_RAG_CORPUS_DIR,
        use_external_evidence: bool = False,
        use_rag_evidence: Optional[bool] = None,
        use_smiles_encoder: bool = False,
        use_mllm_visual_semantic: bool = False,
        reaction_data_root: Optional[str | Path] = None,
        reaction_model_dir: Optional[str | Path] = None,
        reaction_index_dir: Optional[str | Path] = None,
        experiment_hook: Any | None = None,
    ) -> None:
        self.output_dir = ensure_dir(output_dir)
        self.feature_set = feature_set
        inferred_use_smiles_encoder = bool(use_smiles_encoder) or feature_set in {"all_features", "smiles_only"}
        self.model_path = model_path
        self.auto_model = bool(auto_model)
        self.model_registry_path = Path(model_registry_path) if model_registry_path else None
        self.require_real_model = bool(require_real_model)
        self.require_real_rag = bool(require_real_rag)
        self.require_llm_orchestrator = bool(require_llm_orchestrator)
        self.min_evidence = int(min_evidence)
        self.mode = mode
        self.retrieval_backend = retrieval_backend
        self.es_url = es_url
        self.max_rounds = int(max_rounds)
        self.temperature = float(temperature)
        self.llm_profile = llm_profile
        self.use_llm_orchestrator = bool(use_llm_orchestrator)
        self.llm_orchestrator_max_retries = min(5, max(0, int(llm_orchestrator_max_retries)))
        self.llm_orchestrator_retry_backoff_seconds = max(0.0, float(llm_orchestrator_retry_backoff_seconds))
        self.use_llm_reasoning = bool(use_llm_reasoning)
        self.require_llm_reasoning = bool(require_llm_reasoning)
        self.enable_fallbacks = bool(enable_fallbacks)
        self.enable_reflection = bool(enable_reflection)
        self.max_reflection_rounds = int(max_reflection_rounds)
        self.previous_run_path = Path(previous_run_path) if previous_run_path else None
        # Optional, no-op-by-default hook used by controlled reliability tests.
        # It is deliberately kept outside the scientific runtime so ordinary
        # AgentM executions retain exactly the same tool registry and behavior.
        self.experiment_hook = experiment_hook
        self.use_external_evidence = bool(use_external_evidence)
        self.use_rag_evidence = bool(mode != "fast") if use_rag_evidence is None else bool(use_rag_evidence)
        resolved_rag_corpus_dirs = [default_resource_path(path) for path in (rag_corpus_dirs or [DEFAULT_RAG_CORPUS_DIR])]
        resolved_ddi_rag_corpus_dir = default_resource_path(ddi_rag_corpus_dir)
        resolved_rag_embedding_model_name = default_resource_path(rag_embedding_model_name)
        self.runtime = ScientificToolRuntime(
            output_dir=self.output_dir,
            feature_set=feature_set,
            model_path=model_path,
            auto_model=auto_model,
            model_registry_path=str(self.model_registry_path) if self.model_registry_path else None,
            require_real_model=require_real_model,
            require_real_rag=require_real_rag,
            min_evidence=min_evidence,
            use_external_evidence=use_external_evidence,
            rag_corpus_dirs=resolved_rag_corpus_dirs,
            rag_top_k=rag_top_k,
            rag_backend=rag_backend,
            rag_embedding_model_name=resolved_rag_embedding_model_name,
            ddi_rag_corpus_dir=resolved_ddi_rag_corpus_dir,
            use_smiles_encoder=inferred_use_smiles_encoder,
            use_mllm_visual_semantic=use_mllm_visual_semantic,
            retrieval_backend=retrieval_backend,
            es_url=es_url,
            es_index_name=es_index_name,
            llm_profile=llm_profile,
            use_llm_reasoning=use_llm_reasoning,
            require_llm_reasoning=require_llm_reasoning,
            reasoning_llm_client=reasoning_llm_client,
            reaction_data_root=reaction_data_root,
            reaction_model_dir=reaction_model_dir,
            reaction_index_dir=reaction_index_dir,
        )
        self.registry = self.runtime.registry()
        self.guard = GuardPolicy(agent_allowlists=self.registry.agent_allowlists(), max_rounds=max_rounds)
        self.orchestrator = OrchestratorAgent(
            registry=self.registry,
            llm_profile=llm_profile,
            temperature=temperature,
            use_llm=use_llm_orchestrator,
            llm_client=orchestrator_llm_client,
            max_llm_retries=self.llm_orchestrator_max_retries,
            retry_backoff_seconds=self.llm_orchestrator_retry_backoff_seconds,
        )
        self.nodes = MultiAgentGraphNodes(
            registry=self.registry,
            guard=self.guard,
            orchestrator=self.orchestrator,
            output_dir=str(self.output_dir),
            experiment_hook=self.experiment_hook,
        )
        self.graph = build_workflow(self.nodes)

    def run(
        self,
        request: str,
        task_type: Optional[str] = None,
        dataset_name: Optional[str] = None,
        *,
        mode: Optional[str] = None,
        retrieval_backend: Optional[str] = None,
        max_rounds: Optional[int] = None,
        previous_run_path: Optional[str | Path] = None,
        previous_run_context: Optional[Dict[str, Any]] = None,
        task_contract: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        context = initial_context(
            request,
            task_type=task_type,
            dataset_name=dataset_name,
            mode=mode or self.mode,
            retrieval_backend=retrieval_backend or self.retrieval_backend,
            max_rounds=max_rounds or self.max_rounds,
            llm_profile=self.llm_profile,
            use_llm_orchestrator=self.use_llm_orchestrator,
            enable_fallbacks=self.enable_fallbacks,
            enable_reflection=self.enable_reflection,
            max_reflection_rounds=self.max_reflection_rounds,
            auto_model=self.auto_model,
            model_registry_path=str(self.model_registry_path) if self.model_registry_path else None,
            require_real_model=self.require_real_model,
            require_real_rag=self.require_real_rag,
            require_llm_orchestrator=self.require_llm_orchestrator,
            llm_orchestrator_max_retries=self.llm_orchestrator_max_retries,
            llm_orchestrator_retry_backoff_seconds=self.llm_orchestrator_retry_backoff_seconds,
            use_llm_reasoning=self.use_llm_reasoning,
            require_llm_reasoning=self.require_llm_reasoning,
            use_rag_evidence=self.use_rag_evidence,
            min_evidence=self.min_evidence,
        )
        context.setdefault("metadata", {})["use_external_evidence"] = self.use_external_evidence
        context.setdefault("metadata", {})["use_rag_evidence"] = self.use_rag_evidence
        if is_reaction_task(context.get("task_type")):
            # LLM composition contracts may begin directly with reaction tools
            # and therefore legitimately omit route_task.  Persist the local
            # reaction-resource resolution before planning so the terminal
            # fail-closed invariant observes the same resources that the
            # pre-LLM runtime preflight already verified.
            self.runtime._update_resource_resolution(context)
        loaded_previous = previous_run_context or self._load_previous_run(previous_run_path or self.previous_run_path)
        derived_contract = endpoint_addition_contract(
            request=request, task_type=task_type, dataset_name=dataset_name,
            previous=loaded_previous,
        )
        if derived_contract:
            if task_contract and task_contract != derived_contract:
                raise ValueError('explicit_contract_conflicts_with_endpoint_addition_request')
            task_contract = derived_contract
        if task_contract:
            # A composition contract describes only user-requested scientific
            # subtasks.  Benchmark gold, expected labels, and evaluator fields
            # must never be passed through this interface.
            context["metadata"]["task_contract"] = to_jsonable(task_contract)
            context["metadata"]["root_task_type"] = task_type or ""
            context["metadata"]["root_dataset_name"] = dataset_name or ""
        if loaded_previous:
            context["previous_run_context"] = loaded_previous
            context["metadata"]["previous_run_loaded"] = True
            context["metadata"]["previous_run_request"] = loaded_previous.get("request")
        result = self.graph.invoke(context)
        return to_jsonable(result.get("final_report") or {})

    def run_single(self, request: str, *, dataset_name: Optional[str] = None, **kwargs: Any) -> Dict[str, Any]:
        return self.run(request, task_type="admet_prediction", dataset_name=dataset_name, **kwargs)

    def run_ddi(self, request: str, *, dataset_name: Optional[str] = "DDI", **kwargs: Any) -> Dict[str, Any]:
        return self.run(request, task_type="ddi_prediction", dataset_name=dataset_name, **kwargs)

    def run_reaction(self, request: str, *, task_type: str = "reaction_profile", dataset_name: Optional[str] = None, **kwargs: Any) -> Dict[str, Any]:
        return self.run(request, task_type=task_type, dataset_name=dataset_name or "reaction", **kwargs)

    def _load_previous_run(self, path: Optional[str | Path]) -> Dict[str, Any]:
        if not path:
            return {}
        target = Path(path)
        try:
            return json.loads(target.read_text(encoding="utf-8"))
        except Exception as exc:
            return {"load_error": str(exc), "source_path": str(target)}
