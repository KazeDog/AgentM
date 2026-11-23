from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from agentm.agents.ddi_reasoning_agent import DDIReasoningAgent
from agentm.agents.ddi_rag_evidence_agent import DDIRAGEvidenceAgent
from agentm.agents.evidence_agent import EvidenceAgent
from agentm.agents.evidence_quality_agent import EvidenceQualityAgent
from agentm.agents.input_agent import ParsedInput, InputAgent
from agentm.agents.llm_decision_agent import LLMDecisionAgent
from agentm.agents.molecule_processing_agent import MoleculeProcessingAgent
from agentm.agents.narrative_report_agent import NarrativeReportAgent
from agentm.agents.prediction_agent import PredictionAgent
from agentm.agents.rag_evidence_agent import RAGEvidenceAgent
from agentm.agents.report_agent import ReportAgent
from agentm.agents.rule_discovery_agent import RuleDiscoveryAgent
from agentm.agents.zero_shot_prediction_agent import ZERO_SHOT_MODEL_TYPE
from agentm.defaults import (
    DEFAULT_CHEMBERTA2_MODEL_PATH,
    DEFAULT_DDI_RAG_CORPUS_DIR,
    DEFAULT_FEATURE_SET,
    DEFAULT_IMAGEMOL_CHECKPOINT_PATH,
    DEFAULT_LLM_PROFILE,
    DEFAULT_MLLM_PROFILE,
    DEFAULT_RAG_BACKEND,
    DEFAULT_RAG_CORPUS_DIR,
    DEFAULT_RAG_EMBEDDING_MODEL_PATH,
    DEFAULT_RAG_TOP_K,
    DEFAULT_USE_LLM_DECISION_LAYER,
    DEFAULT_USE_LLM_PLANNER,
    DEFAULT_USE_MLLM_VISUAL_SEMANTIC,
    DEFAULT_USE_SMILES_ENCODER,
)
from agentm.evaluation.reproducibility import write_reproducibility_log
from agentm.evaluation.evidence_scoring import score_evidence_for_prediction
from agentm.evaluation.robustness import assess_failure_modes
from agentm.features.builder import FeatureBuilder
from agentm.features.visual_semantic import VisualSemanticExtractor
from agentm.llm.provider import LLMClient, build_llm_client
from agentm.reports.case_study import write_case_study_report
from agentm.schema import (
    AgentReport,
    DDIPairFeatureBundle,
    MoleculeRecord,
    PredictionResult,
    Rule,
    SingleMoleculeFeatureBundle,
)
from agentm.tools.registry import ToolRegistry
from agentm.utils.api_usage import api_usage_context, usage_artifact_paths
from agentm.utils.io import ensure_dir, write_json, write_text


@dataclass
class AgentPlanStep:
    step_id: str
    tool_name: str
    description: str
    params: Dict[str, Any] = field(default_factory=dict)
    depends_on: List[str] = field(default_factory=list)
    fallback_tool: Optional[str] = None
    condition: Optional[str] = None
    status: str = "pending"
    message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AgentPlan:
    goal: str
    task_type: str = "auto"
    dataset_name: str = "auto"
    steps: List[AgentPlanStep] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "goal": self.goal,
            "task_type": self.task_type,
            "dataset_name": self.dataset_name,
            "metadata": self.metadata,
            "steps": [step.to_dict() for step in self.steps],
        }


@dataclass
class AgentObservation:
    step_id: str
    tool_name: str
    status: str
    duration_sec: float
    output_summary: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    fallback_tool: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AgentState:
    request: str
    output_dir: Path
    parsed: Optional[ParsedInput] = None
    plan: Optional[AgentPlan] = None
    trace: List[AgentObservation] = field(default_factory=list)
    molecules: List[MoleculeRecord] = field(default_factory=list)
    rules: List[Rule] = field(default_factory=list)
    single_features: List[SingleMoleculeFeatureBundle] = field(default_factory=list)
    ddi_pair: Optional[DDIPairFeatureBundle] = None
    predictions: List[PredictionResult] = field(default_factory=list)
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    artifacts: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    reasoning: Dict[str, Any] = field(
        default_factory=lambda: {
            "decisions": [],
            "fallbacks": [],
            "confidence_signals": [],
            "summary": "",
        }
    )
    report: Optional[AgentReport] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class DecisionPolicy:
    """Creates task-specific plans from parsed user intent."""

    def plan_after_parse(self, parsed: ParsedInput) -> List[AgentPlanStep]:
        if parsed.task_type == "ddi_prediction":
            return self._ddi_steps()
        return self._single_molecule_steps()

    def _single_molecule_steps(self) -> List[AgentPlanStep]:
        return [
            AgentPlanStep(
                step_id="resolve_molecule",
                tool_name="resolve_molecule",
                description="Resolve the molecule mention to a canonical SMILES.",
                params={"index": 0, "role": "molecule"},
                depends_on=["parse_input"],
            ),
            AgentPlanStep(
                step_id="discover_rules",
                tool_name="discover_rules",
                description="Select trained rules or discover LLM4SD-style computable rules.",
                depends_on=["resolve_molecule"],
                fallback_tool="discover_rules_fallback",
            ),
            AgentPlanStep(
                step_id="maybe_enable_mllm_visual_semantic",
                tool_name="maybe_enable_mllm_visual_semantic",
                description="Decide whether this request should use the MLLM visual-semantic branch.",
                depends_on=["discover_rules"],
            ),
            AgentPlanStep(
                step_id="build_single_features",
                tool_name="build_single_features",
                description="Build RDKit, fingerprint, rule, ImageMol, and visual-semantic features.",
                params={"index": 0, "role": "molecule"},
                depends_on=["maybe_enable_mllm_visual_semantic"],
                fallback_tool="build_single_features_fallback",
            ),
            AgentPlanStep(
                step_id="select_prediction_strategy",
                tool_name="select_prediction_strategy",
                description="Choose trained-model prediction or zero-shot prediction before scoring.",
                depends_on=["build_single_features"],
            ),
            AgentPlanStep(
                step_id="predict_single",
                tool_name="predict_single",
                description="Run the selected predictor, falling back to ZeroShotPredictionAgent if needed.",
                depends_on=["select_prediction_strategy"],
                fallback_tool="predict_single_zero_shot",
            ),
            AgentPlanStep(
                step_id="assess_prediction_confidence",
                tool_name="assess_prediction_confidence",
                description="Assess threshold margin and mark low-confidence predictions.",
                depends_on=["predict_single"],
            ),
            AgentPlanStep(
                step_id="retrieve_evidence",
                tool_name="retrieve_evidence",
                description="Retrieve local descriptor, rule, and task evidence.",
                depends_on=["assess_prediction_confidence"],
                fallback_tool="retrieve_empty_evidence",
            ),
            AgentPlanStep(
                step_id="retrieve_additional_evidence",
                tool_name="retrieve_additional_evidence",
                description="Add extra local evidence notes when the prediction is low-confidence.",
                depends_on=["retrieve_evidence"],
                condition="low_confidence_prediction",
            ),
            AgentPlanStep(
                step_id="retrieve_rag_evidence",
                tool_name="retrieve_rag_evidence",
                description="Retrieve local RAG literature snippets as evidence context.",
                depends_on=["retrieve_additional_evidence"],
            ),
            AgentPlanStep(
                step_id="reason_over_prediction",
                tool_name="reason_over_prediction",
                description="Summarize decisions, fallbacks, confidence signals, and explanation cues.",
                depends_on=["retrieve_rag_evidence"],
            ),
            AgentPlanStep(
                step_id="build_report",
                tool_name="build_report",
                description="Assemble the user-facing report and attach plan and trace artifacts.",
                depends_on=["reason_over_prediction"],
            ),
        ]

    def _ddi_steps(self) -> List[AgentPlanStep]:
        return [
            AgentPlanStep(
                step_id="resolve_drug_a",
                tool_name="resolve_molecule",
                description="Resolve drug A to a canonical SMILES.",
                params={"index": 0, "role": "drug_a"},
                depends_on=["parse_input"],
            ),
            AgentPlanStep(
                step_id="resolve_drug_b",
                tool_name="resolve_molecule",
                description="Resolve drug B to a canonical SMILES.",
                params={"index": 1, "role": "drug_b"},
                depends_on=["resolve_drug_a"],
            ),
            AgentPlanStep(
                step_id="discover_rules",
                tool_name="discover_rules",
                description="Select trained DDI rules or discover computable DDI proxy rules.",
                depends_on=["resolve_drug_b"],
                fallback_tool="discover_rules_fallback",
            ),
            AgentPlanStep(
                step_id="maybe_enable_mllm_visual_semantic",
                tool_name="maybe_enable_mllm_visual_semantic",
                description="Decide whether DDI reasoning should use the MLLM visual-semantic branch.",
                depends_on=["discover_rules"],
            ),
            AgentPlanStep(
                step_id="build_drug_a_features",
                tool_name="build_single_features",
                description="Build single-molecule features for drug A.",
                params={"index": 0, "role": "drug_a"},
                depends_on=["maybe_enable_mllm_visual_semantic"],
                fallback_tool="build_single_features_fallback",
            ),
            AgentPlanStep(
                step_id="build_drug_b_features",
                tool_name="build_single_features",
                description="Build single-molecule features for drug B.",
                params={"index": 1, "role": "drug_b"},
                depends_on=["build_drug_a_features"],
                fallback_tool="build_single_features_fallback",
            ),
            AgentPlanStep(
                step_id="build_ddi_pair",
                tool_name="build_ddi_pair",
                description="Fuse drug A and drug B into DDI pair features.",
                depends_on=["build_drug_b_features"],
                fallback_tool="build_ddi_pair_fallback",
            ),
            AgentPlanStep(
                step_id="select_prediction_strategy",
                tool_name="select_prediction_strategy",
                description="Choose trained-model DDI prediction or zero-shot prediction before scoring.",
                depends_on=["build_ddi_pair"],
            ),
            AgentPlanStep(
                step_id="predict_ddi",
                tool_name="predict_ddi",
                description="Predict DDI risk, falling back to ZeroShotPredictionAgent if needed.",
                depends_on=["select_prediction_strategy"],
                fallback_tool="predict_ddi_zero_shot",
            ),
            AgentPlanStep(
                step_id="assess_prediction_confidence",
                tool_name="assess_prediction_confidence",
                description="Assess threshold margin and mark low-confidence DDI predictions.",
                depends_on=["predict_ddi"],
            ),
            AgentPlanStep(
                step_id="retrieve_evidence",
                tool_name="retrieve_evidence",
                description="Retrieve local DDI and molecular evidence.",
                depends_on=["assess_prediction_confidence"],
                fallback_tool="retrieve_empty_evidence",
            ),
            AgentPlanStep(
                step_id="retrieve_additional_evidence",
                tool_name="retrieve_additional_evidence",
                description="Add extra local DDI evidence notes when the prediction is low-confidence.",
                depends_on=["retrieve_evidence"],
                condition="low_confidence_prediction",
            ),
            AgentPlanStep(
                step_id="retrieve_rag_evidence",
                tool_name="retrieve_rag_evidence",
                description="Retrieve local RAG literature snippets as DDI evidence context.",
                depends_on=["retrieve_additional_evidence"],
            ),
            AgentPlanStep(
                step_id="reason_over_prediction",
                tool_name="reason_over_prediction",
                description="Summarize DDI mechanism cues, confidence signals, and fallbacks.",
                depends_on=["retrieve_rag_evidence"],
            ),
            AgentPlanStep(
                step_id="build_report",
                tool_name="build_report",
                description="Assemble the user-facing DDI report and attach plan and trace artifacts.",
                depends_on=["reason_over_prediction"],
            ),
        ]


class EnhancedPlanner:
    """Hybrid planner with a safe rule fallback and optional LLM JSON planning."""

    REQUIRED_STEP_IDS = {
        "property_prediction": {
            "resolve_molecule",
            "discover_rules",
            "maybe_enable_mllm_visual_semantic",
            "build_single_features",
            "select_prediction_strategy",
            "predict_single",
            "assess_prediction_confidence",
            "retrieve_evidence",
            "retrieve_additional_evidence",
            "retrieve_rag_evidence",
            "reason_over_prediction",
            "build_report",
        },
        "admet_prediction": {
            "resolve_molecule",
            "discover_rules",
            "maybe_enable_mllm_visual_semantic",
            "build_single_features",
            "select_prediction_strategy",
            "predict_single",
            "assess_prediction_confidence",
            "retrieve_evidence",
            "retrieve_additional_evidence",
            "retrieve_rag_evidence",
            "reason_over_prediction",
            "build_report",
        },
        "ddi_prediction": {
            "resolve_drug_a",
            "resolve_drug_b",
            "discover_rules",
            "maybe_enable_mllm_visual_semantic",
            "build_drug_a_features",
            "build_drug_b_features",
            "build_ddi_pair",
            "select_prediction_strategy",
            "predict_ddi",
            "assess_prediction_confidence",
            "retrieve_evidence",
            "retrieve_additional_evidence",
            "retrieve_rag_evidence",
            "reason_over_prediction",
            "build_report",
        },
    }
    ALLOWED_CONDITIONS = {None, "low_confidence_prediction"}

    def __init__(
        self,
        base_policy: DecisionPolicy,
        *,
        use_llm_planner: bool = DEFAULT_USE_LLM_PLANNER,
        llm_client: Optional[LLMClient] = None,
    ):
        self.base_policy = base_policy
        self.use_llm_planner = use_llm_planner
        self.llm_client = llm_client

    def plan_after_parse(
        self,
        parsed: ParsedInput,
        *,
        context: Dict[str, Any],
        allowed_tools: Set[str],
    ) -> Tuple[List[AgentPlanStep], Dict[str, Any]]:
        base_steps = self.base_policy.plan_after_parse(parsed)
        metadata: Dict[str, Any] = {
            "planner_mode": "rule_policy",
            "llm_planner_requested": self.use_llm_planner,
            "llm_planner_available": bool(self.llm_client and self.llm_client.available),
            "validation_status": "not_needed",
            "recommendations": self._local_recommendations(parsed, context),
        }
        if not self.use_llm_planner:
            return base_steps, metadata
        if self.llm_client is None or not self.llm_client.available:
            metadata.update(
                {
                    "planner_mode": "rule_policy_after_llm_unavailable",
                    "validation_status": "llm_unavailable",
                }
            )
            return base_steps, metadata

        try:
            proposed, llm_payload = self._llm_plan(parsed, context, base_steps, allowed_tools)
            self._validate_steps(proposed, parsed.task_type, allowed_tools)
            metadata.update(
                {
                    "planner_mode": "llm_json_plan",
                    "validation_status": "valid",
                    "llm_profile": self.llm_client.config.profile,
                    "llm_model": self.llm_client.config.model,
                    "llm_strategy": llm_payload.get("strategy"),
                    "recommendations": {
                        **metadata["recommendations"],
                        **self._recommendations_from_payload(llm_payload),
                    },
                }
            )
            return proposed, metadata
        except Exception as exc:
            metadata.update(
                {
                    "planner_mode": "rule_policy_after_llm_plan_error",
                    "validation_status": "failed",
                    "llm_error": str(exc),
                }
            )
            return base_steps, metadata

    def _llm_plan(
        self,
        parsed: ParsedInput,
        context: Dict[str, Any],
        base_steps: List[AgentPlanStep],
        allowed_tools: Set[str],
    ) -> Tuple[List[AgentPlanStep], Dict[str, Any]]:
        assert self.llm_client is not None
        prompt = self._planner_prompt(parsed, context, base_steps, allowed_tools)
        response = self.llm_client.chat_json(
            [
                {
                    "role": "system",
                    "content": (
                        "You are a cautious scientific workflow planner. "
                        "Return JSON only and use only the provided tools."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
        )
        payload = self._parse_json_object(response)
        steps = [self._step_from_payload(item) for item in payload.get("steps", [])]
        if not steps:
            raise ValueError("LLM planner returned no steps.")
        return steps, payload

    def _planner_prompt(
        self,
        parsed: ParsedInput,
        context: Dict[str, Any],
        base_steps: List[AgentPlanStep],
        allowed_tools: Set[str],
    ) -> str:
        base_plan = [step.to_dict() for step in base_steps]
        compact_context = {
            "request": context.get("request"),
            "task_type": parsed.task_type,
            "dataset_name": parsed.dataset_name,
            "molecule_mentions": parsed.molecules,
            "feature_set": context.get("feature_set"),
            "model_available": context.get("model_available"),
            "imagemol_encoder": context.get("imagemol_encoder"),
            "mllm_currently_enabled": context.get("mllm_currently_enabled"),
            "allow_planner_mllm": context.get("allow_planner_mllm"),
            "confidence_margin_threshold": context.get("confidence_margin_threshold"),
            "use_rag_evidence": context.get("use_rag_evidence"),
            "rag_backend": context.get("rag_backend"),
            "rag_top_k": context.get("rag_top_k"),
        }
        return f"""
Create a safe post-parse execution plan for a small-molecule multimodal prediction agent.

Context:
{json.dumps(compact_context, indent=2, sort_keys=True)}

Allowed tool names:
{json.dumps(sorted(allowed_tools), indent=2)}

Required base plan:
{json.dumps(base_plan, indent=2)}

Rules:
- Return JSON only.
- Keep every required step_id from the base plan.
- Do not include parse_input; it has already run.
- Use only allowed tool names and fallback tools.
- Use only these conditions: null, "low_confidence_prediction".
- Preserve dependency order unless there is a clear reason to tighten it.
- You may update descriptions and add a short strategy/recommendations object.

Return schema:
{{
  "strategy": "short planner rationale",
  "recommendations": {{"use_mllm_visual_semantic": false}},
  "steps": [
    {{
      "step_id": "...",
      "tool_name": "...",
      "description": "...",
      "params": {{}},
      "depends_on": ["..."],
      "fallback_tool": null,
      "condition": null
    }}
  ]
}}
""".strip()

    def _parse_json_object(self, response: str) -> Dict[str, Any]:
        text = response.strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:].strip()
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            text = text[start : end + 1]
        data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError("Planner response must be a JSON object.")
        return data

    def _step_from_payload(self, item: Dict[str, Any]) -> AgentPlanStep:
        if not isinstance(item, dict):
            raise ValueError("Each planner step must be an object.")
        params = item.get("params")
        depends_on = item.get("depends_on")
        return AgentPlanStep(
            step_id=str(item.get("step_id") or "").strip(),
            tool_name=str(item.get("tool_name") or "").strip(),
            description=str(item.get("description") or "").strip(),
            params=params if isinstance(params, dict) else {},
            depends_on=[str(value) for value in depends_on] if isinstance(depends_on, list) else [],
            fallback_tool=str(item["fallback_tool"]).strip() if item.get("fallback_tool") else None,
            condition=str(item["condition"]).strip() if item.get("condition") else None,
        )

    def _validate_steps(self, steps: List[AgentPlanStep], task_type: str, allowed_tools: Set[str]) -> None:
        if len(steps) > 20:
            raise ValueError("Planner returned too many steps.")
        seen: Set[str] = {"parse_input"}
        step_ids = [step.step_id for step in steps]
        if len(step_ids) != len(set(step_ids)):
            raise ValueError("Planner returned duplicate step_id values.")
        required = self.REQUIRED_STEP_IDS.get(task_type, self.REQUIRED_STEP_IDS["property_prediction"])
        missing = sorted(required - set(step_ids))
        if missing:
            raise ValueError(f"Planner omitted required steps: {missing}")
        for step in steps:
            if not step.step_id:
                raise ValueError("Planner returned an empty step_id.")
            if step.tool_name not in allowed_tools:
                raise ValueError(f"Planner used unknown tool: {step.tool_name}")
            if step.fallback_tool and step.fallback_tool not in allowed_tools:
                raise ValueError(f"Planner used unknown fallback tool: {step.fallback_tool}")
            if step.condition not in self.ALLOWED_CONDITIONS:
                raise ValueError(f"Planner used unsupported condition: {step.condition}")
            unknown_dependencies = [dep for dep in step.depends_on if dep not in seen]
            if unknown_dependencies:
                raise ValueError(f"Planner step {step.step_id} has invalid dependencies: {unknown_dependencies}")
            seen.add(step.step_id)
        if steps[-1].tool_name != "build_report":
            raise ValueError("The final planner step must build the report.")

    def _local_recommendations(self, parsed: ParsedInput, context: Dict[str, Any]) -> Dict[str, Any]:
        request = str(context.get("request") or "").lower()
        asks_visual = any(token in request for token in ["mllm", "multimodal", "visual", "image", "图像", "视觉", "结构", "解释"])
        return {
            "use_mllm_visual_semantic": bool(context.get("allow_planner_mllm") and asks_visual),
            "extra_evidence_on_low_confidence": True,
        }

    def _recommendations_from_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        recommendations = payload.get("recommendations")
        if not isinstance(recommendations, dict):
            return {}
        allowed = {"use_mllm_visual_semantic", "extra_evidence_on_low_confidence"}
        return {str(key): value for key, value in recommendations.items() if key in allowed}


class AgentController:
    """Agentic orchestrator around the molecular feature and prediction stack."""

    def __init__(
        self,
        output_dir: str | Path = "outputs/demo",
        feature_set: str = DEFAULT_FEATURE_SET,
        model_path: Optional[str] = None,
        imagemol_checkpoint_path: Optional[str] = DEFAULT_IMAGEMOL_CHECKPOINT_PATH,
        use_smiles_encoder: bool = DEFAULT_USE_SMILES_ENCODER,
        smiles_encoder_model_name: Optional[str] = DEFAULT_CHEMBERTA2_MODEL_PATH,
        smiles_encoder_device: str = "auto",
        smiles_pooling: str = "cls",
        smiles_local_files_only: bool = True,
        llm_profile: Optional[str] = DEFAULT_LLM_PROFILE,
        use_mllm_visual_semantic: bool = DEFAULT_USE_MLLM_VISUAL_SEMANTIC,
        mllm_profile: Optional[str] = DEFAULT_MLLM_PROFILE,
        visual_semantic_cache_dir: Optional[str] = None,
        visual_semantic_cache_read: bool = True,
        visual_semantic_cache_write: bool = True,
        visual_semantic_cache_only: bool = False,
        use_llm_planner: bool = DEFAULT_USE_LLM_PLANNER,
        planner_profile: Optional[str] = None,
        allow_planner_mllm: bool = False,
        use_llm_decision_layer: bool = DEFAULT_USE_LLM_DECISION_LAYER,
        decision_profile: Optional[str] = None,
        enable_llm_routing: bool = True,
        enable_plan_critic: bool = True,
        use_external_evidence: bool = False,
        confidence_margin_threshold: float = 0.12,
        use_narrative_llm: bool = False,
        narrative_llm_profile: Optional[str] = None,
        narrative_language: str = "auto",
        use_rag_evidence: bool = True,
        rag_corpus_dirs: Optional[List[str]] = None,
        rag_top_k: int = DEFAULT_RAG_TOP_K,
        rag_backend: str = DEFAULT_RAG_BACKEND,
        rag_embedding_model_name: str = DEFAULT_RAG_EMBEDDING_MODEL_PATH,
        rag_embedding_backend: str = "auto",
        rag_embedding_device: str = "auto",
        rag_embedding_local_files_only: bool = True,
        rag_embedding_batch_size: int = 16,
        use_llm_rag: bool = False,
        ddi_rag_corpus_dir: str = DEFAULT_DDI_RAG_CORPUS_DIR,
        ddi_rag_build_on_demand: bool = True,
        ddi_rag_timeout: float = 12.0,
        ddi_rag_max_pubmed: int = 8,
        ddi_rag_max_labels_per_drug: int = 2,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ):
        self.output_dir = ensure_dir(output_dir)
        self.feature_set = feature_set
        self.use_smiles_encoder = bool(use_smiles_encoder)
        self.smiles_encoder_model_name = smiles_encoder_model_name
        self.smiles_encoder_device = smiles_encoder_device
        self.smiles_pooling = smiles_pooling
        self.smiles_local_files_only = smiles_local_files_only
        self.llm_profile = llm_profile
        self.mllm_profile = mllm_profile
        self.use_mllm_visual_semantic = use_mllm_visual_semantic
        self.visual_semantic_cache_dir = visual_semantic_cache_dir
        self.visual_semantic_cache_read = visual_semantic_cache_read
        self.visual_semantic_cache_write = visual_semantic_cache_write
        self.visual_semantic_cache_only = visual_semantic_cache_only
        self.allow_planner_mllm = allow_planner_mllm
        self.use_llm_decision_layer = bool(use_llm_decision_layer)
        self.decision_profile = decision_profile or llm_profile
        self.enable_llm_routing = bool(enable_llm_routing)
        self.enable_plan_critic = bool(enable_plan_critic)
        self.use_external_evidence = bool(use_external_evidence)
        self.confidence_margin_threshold = float(confidence_margin_threshold)
        self.use_narrative_llm = bool(use_narrative_llm)
        self.narrative_llm_profile = narrative_llm_profile or llm_profile
        self.narrative_language = narrative_language
        self.use_rag_evidence = bool(use_rag_evidence)
        self.rag_corpus_dirs = list(rag_corpus_dirs or [DEFAULT_RAG_CORPUS_DIR])
        self.rag_top_k = int(rag_top_k)
        self.rag_backend = rag_backend
        self.rag_embedding_model_name = rag_embedding_model_name
        self.rag_embedding_backend = rag_embedding_backend
        self.rag_embedding_device = rag_embedding_device
        self.rag_embedding_local_files_only = bool(rag_embedding_local_files_only)
        self.rag_embedding_batch_size = int(rag_embedding_batch_size)
        self.use_llm_rag = bool(use_llm_rag)
        self.ddi_rag_corpus_dir = ddi_rag_corpus_dir
        self.ddi_rag_build_on_demand = bool(ddi_rag_build_on_demand)
        self.ddi_rag_timeout = float(ddi_rag_timeout)
        self.ddi_rag_max_pubmed = int(ddi_rag_max_pubmed)
        self.ddi_rag_max_labels_per_drug = int(ddi_rag_max_labels_per_drug)
        self.progress_callback = progress_callback
        self.runtime_config = {
            "output_dir": str(output_dir),
            "feature_set": feature_set,
            "model_path": model_path,
            "imagemol_checkpoint_path": imagemol_checkpoint_path,
            "use_smiles_encoder": use_smiles_encoder,
            "smiles_encoder_model_name": smiles_encoder_model_name,
            "smiles_encoder_device": smiles_encoder_device,
            "smiles_pooling": smiles_pooling,
            "smiles_local_files_only": smiles_local_files_only,
            "llm_profile": llm_profile,
            "use_mllm_visual_semantic": use_mllm_visual_semantic,
            "mllm_profile": mllm_profile,
            "visual_semantic_cache_dir": visual_semantic_cache_dir,
            "visual_semantic_cache_read": visual_semantic_cache_read,
            "visual_semantic_cache_write": visual_semantic_cache_write,
            "visual_semantic_cache_only": visual_semantic_cache_only,
            "use_llm_planner": use_llm_planner,
            "planner_profile": planner_profile,
            "allow_planner_mllm": allow_planner_mllm,
            "use_llm_decision_layer": self.use_llm_decision_layer,
            "decision_profile": self.decision_profile,
            "enable_llm_routing": self.enable_llm_routing,
            "enable_plan_critic": self.enable_plan_critic,
            "use_external_evidence": use_external_evidence,
            "confidence_margin_threshold": confidence_margin_threshold,
            "use_narrative_llm": use_narrative_llm,
            "narrative_llm_profile": self.narrative_llm_profile,
            "narrative_language": narrative_language,
            "use_rag_evidence": self.use_rag_evidence,
            "rag_corpus_dirs": self.rag_corpus_dirs,
            "rag_top_k": self.rag_top_k,
            "rag_backend": self.rag_backend,
            "rag_embedding_model_name": self.rag_embedding_model_name,
            "rag_embedding_backend": self.rag_embedding_backend,
            "rag_embedding_device": self.rag_embedding_device,
            "rag_embedding_local_files_only": self.rag_embedding_local_files_only,
            "rag_embedding_batch_size": self.rag_embedding_batch_size,
            "use_llm_rag": self.use_llm_rag,
            "ddi_rag_corpus_dir": self.ddi_rag_corpus_dir,
            "ddi_rag_build_on_demand": self.ddi_rag_build_on_demand,
            "ddi_rag_timeout": self.ddi_rag_timeout,
            "ddi_rag_max_pubmed": self.ddi_rag_max_pubmed,
            "ddi_rag_max_labels_per_drug": self.ddi_rag_max_labels_per_drug,
        }
        self.input_agent = InputAgent()
        self.processor = MoleculeProcessingAgent()
        self.prediction_agent = PredictionAgent(model_path=model_path)
        trained_metadata = self.prediction_agent.model.metadata if self.prediction_agent.model is not None else {}
        if self.prediction_agent.model is not None:
            feature_set = trained_metadata.get("feature_set", feature_set)
            imagemol_checkpoint_path = imagemol_checkpoint_path or trained_metadata.get("imagemol_checkpoint_path")
            trained_schema = self.prediction_agent.model.feature_schema or {}
            trained_blocks = trained_schema.get("blocks", {}) if isinstance(trained_schema, dict) else {}
            if "use_smiles_encoder" in trained_metadata:
                use_smiles_encoder = bool(trained_metadata.get("use_smiles_encoder"))
            elif trained_schema and "smiles" not in trained_blocks:
                use_smiles_encoder = False
            smiles_encoder_model_name = smiles_encoder_model_name or trained_metadata.get("smiles_encoder_model_name")
            smiles_encoder_device = smiles_encoder_device or trained_metadata.get("smiles_encoder_device", "auto")
            smiles_pooling = smiles_pooling or trained_metadata.get("smiles_pooling", "cls")
            smiles_local_files_only = bool(trained_metadata.get("smiles_local_files_only", smiles_local_files_only))
            use_mllm_visual_semantic = use_mllm_visual_semantic or bool(trained_metadata.get("use_mllm_visual_semantic", False))
            mllm_profile = mllm_profile or trained_metadata.get("mllm_profile")
            visual_semantic_cache_dir = visual_semantic_cache_dir or trained_metadata.get("visual_semantic_cache_dir")
            visual_semantic_cache_read = bool(trained_metadata.get("visual_semantic_cache_read", visual_semantic_cache_read))
            visual_semantic_cache_write = bool(trained_metadata.get("visual_semantic_cache_write", visual_semantic_cache_write))
            visual_semantic_cache_only = bool(trained_metadata.get("visual_semantic_cache_only", visual_semantic_cache_only))
            self.feature_set = feature_set
            self.use_smiles_encoder = use_smiles_encoder
            self.smiles_encoder_model_name = smiles_encoder_model_name
            self.smiles_encoder_device = smiles_encoder_device
            self.smiles_pooling = smiles_pooling
            self.smiles_local_files_only = smiles_local_files_only
            self.mllm_profile = mllm_profile
            self.use_mllm_visual_semantic = use_mllm_visual_semantic
            self.visual_semantic_cache_dir = visual_semantic_cache_dir
            self.visual_semantic_cache_read = visual_semantic_cache_read
            self.visual_semantic_cache_write = visual_semantic_cache_write
            self.visual_semantic_cache_only = visual_semantic_cache_only

        self.runtime_config.update(
            {
                "feature_set": self.feature_set,
                "imagemol_checkpoint_path": imagemol_checkpoint_path,
                "use_smiles_encoder": self.use_smiles_encoder,
                "smiles_encoder_model_name": self.smiles_encoder_model_name,
                "smiles_encoder_device": self.smiles_encoder_device,
                "smiles_pooling": self.smiles_pooling,
                "smiles_local_files_only": self.smiles_local_files_only,
                "use_mllm_visual_semantic": self.use_mllm_visual_semantic,
                "mllm_profile": self.mllm_profile,
                "visual_semantic_cache_dir": self.visual_semantic_cache_dir,
                "visual_semantic_cache_read": self.visual_semantic_cache_read,
                "visual_semantic_cache_write": self.visual_semantic_cache_write,
                "visual_semantic_cache_only": self.visual_semantic_cache_only,
            }
        )

        self.rule_agent = RuleDiscoveryAgent(output_dir=self.output_dir / "rules", llm_profile=llm_profile)
        self.feature_builder = FeatureBuilder(
            output_dir=self.output_dir / "features",
            feature_set=feature_set,
            imagemol_checkpoint_path=imagemol_checkpoint_path,
            use_smiles_encoder=use_smiles_encoder,
            smiles_encoder_model_name=smiles_encoder_model_name,
            smiles_encoder_device=smiles_encoder_device,
            smiles_pooling=smiles_pooling,
            smiles_local_files_only=smiles_local_files_only,
            use_mllm_visual_semantic=use_mllm_visual_semantic,
            mllm_profile=mllm_profile,
            visual_semantic_cache_dir=visual_semantic_cache_dir,
            visual_semantic_cache_read=visual_semantic_cache_read,
            visual_semantic_cache_write=visual_semantic_cache_write,
            visual_semantic_cache_only=visual_semantic_cache_only,
            generate_grad_cam=True,
        )
        self.ddi_agent = DDIReasoningAgent(self.feature_builder)
        self.evidence_agent = EvidenceAgent(use_external=self.use_external_evidence)
        self.evidence_quality_agent = EvidenceQualityAgent()
        self.rag_evidence_agent = RAGEvidenceAgent(
            output_dir=self.output_dir / "rag",
            corpus_dirs=self.rag_corpus_dirs,
            top_k=self.rag_top_k,
            rag_backend=self.rag_backend,
            embedding_model_name=self.rag_embedding_model_name,
            embedding_backend=self.rag_embedding_backend,
            embedding_device=self.rag_embedding_device,
            embedding_local_files_only=self.rag_embedding_local_files_only,
            embedding_batch_size=self.rag_embedding_batch_size,
            use_llm=self.use_llm_rag,
            llm_profile=llm_profile,
        )
        self.ddi_rag_evidence_agent = DDIRAGEvidenceAgent(
            output_dir=self.output_dir / "ddi_rag",
            corpus_dir=self.ddi_rag_corpus_dir,
            build_on_demand=self.ddi_rag_build_on_demand,
            timeout=self.ddi_rag_timeout,
            max_pubmed=self.ddi_rag_max_pubmed,
            max_labels_per_drug=self.ddi_rag_max_labels_per_drug,
            top_k=self.rag_top_k,
            rag_backend=self.rag_backend,
            embedding_model_name=self.rag_embedding_model_name,
            embedding_backend=self.rag_embedding_backend,
            embedding_device=self.rag_embedding_device,
            embedding_local_files_only=self.rag_embedding_local_files_only,
            embedding_batch_size=self.rag_embedding_batch_size,
            use_llm=self.use_llm_rag,
            llm_profile=llm_profile,
        )
        self.report_agent = ReportAgent()
        self.narrative_agent = NarrativeReportAgent(
            output_dir=self.output_dir / "narrative",
            llm_profile=self.narrative_llm_profile,
            use_llm=self.use_narrative_llm,
            language=self.narrative_language,
        )
        self.policy = DecisionPolicy()
        self.registry = ToolRegistry()
        self._fallback_feature_builder: Optional[FeatureBuilder] = None
        self._register_tools()
        self.decision_agent = LLMDecisionAgent(
            use_llm=self.use_llm_decision_layer,
            profile_name=self.decision_profile,
            enable_routing=self.enable_llm_routing,
            enable_plan_critic=self.enable_plan_critic,
        )
        planner_client = build_llm_client(profile_name=planner_profile or llm_profile) if use_llm_planner else None
        self.planner = EnhancedPlanner(
            self.policy,
            use_llm_planner=use_llm_planner,
            llm_client=planner_client,
        )

    def run_single(
        self,
        request: str,
        task_type: Optional[str] = None,
        dataset_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        return self.run(request=request, task_type=task_type, dataset_name=dataset_name)

    def run_ddi(self, request: str, dataset_name: Optional[str] = None) -> Dict[str, Any]:
        return self.run(request=request, task_type="ddi_prediction", dataset_name=dataset_name or "DDI")

    def run(
        self,
        request: str,
        task_type: Optional[str] = None,
        dataset_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        with api_usage_context(self.output_dir):
            state = AgentState(request=request, output_dir=self.output_dir)
            state.plan = AgentPlan(
                goal=request,
                task_type=task_type or "auto",
                dataset_name=dataset_name or "auto",
                metadata={"tool_registry": self.registry.describe()},
            )
            parse_step = AgentPlanStep(
                step_id="parse_input",
                tool_name="parse_input",
                description="Parse task intent, target dataset, and molecule mentions.",
                params={"task_type": task_type, "dataset_name": dataset_name},
            )
            state.plan.steps.append(parse_step)
            self._execute_step(state, parse_step)

            assert state.parsed is not None
            decision_metadata = self.decision_agent.decide_after_parse(
                state.parsed,
                context=self._planner_context(state),
                explicit_task_type=task_type,
                explicit_dataset_name=dataset_name,
            )
            state.plan.metadata["llm_decision"] = decision_metadata
            self._record_decision_warning(state, decision_metadata)
            self._apply_decision_route(
                state,
                decision_metadata,
                explicit_task_type=task_type,
                explicit_dataset_name=dataset_name,
            )
            assert state.parsed is not None
            state.plan.task_type = state.parsed.task_type
            state.plan.dataset_name = state.parsed.dataset_name
            remaining_steps, planner_metadata = self.planner.plan_after_parse(
                state.parsed,
                context=self._planner_context(state),
                allowed_tools=self.registry.names(),
            )
            self._merge_decision_recommendations(planner_metadata, decision_metadata)
            critic_metadata = self.decision_agent.critique_plan(
                state.parsed,
                context={
                    **self._planner_context(state),
                    "llm_decision": decision_metadata,
                    "planner_metadata": planner_metadata,
                },
                steps=remaining_steps,
                allowed_tools=self.registry.names(),
            )
            self._record_planner_warning(state, planner_metadata)
            self._record_plan_critic_warning(state, critic_metadata)
            state.plan.metadata["planner"] = planner_metadata
            state.plan.metadata["plan_critic"] = critic_metadata
            state.reasoning["decisions"].append(
                {
                    "step": "planner",
                    "decision": planner_metadata.get("planner_mode"),
                    "validation_status": planner_metadata.get("validation_status"),
                    "recommendations": planner_metadata.get("recommendations", {}),
                }
            )
            state.reasoning["decisions"].append(
                {
                    "step": "plan_critic",
                    "decision": critic_metadata.get("critic_mode"),
                    "status": critic_metadata.get("status"),
                    "validation_status": critic_metadata.get("validation_status"),
                    "recommended_changes": critic_metadata.get("recommended_changes", []),
                }
            )
            state.plan.steps.extend(remaining_steps)
            self._emit_progress(
                {
                    "event_type": "plan_ready",
                    "tool_name": "planner",
                    "description": "Planner prepared the remaining agent workflow.",
                    "status": "completed",
                    "message": f"{len(state.plan.steps)} steps planned",
                    "output_summary": {
                        "total_steps": len(state.plan.steps),
                        "steps": [
                            {
                                "step_id": step.step_id,
                                "tool_name": step.tool_name,
                                "description": step.description,
                                "condition": step.condition,
                            }
                            for step in state.plan.steps
                        ],
                    },
                }
            )
            for step in remaining_steps:
                self._execute_step(state, step)

            self._finalize(state)
            assert state.report is not None
            return state.report.to_dict()

    def _register_tools(self) -> None:
        self.registry.register("parse_input", self._tool_parse_input, "Infer task, dataset, and molecule mentions from a user request.")
        self.registry.register("resolve_molecule", self._tool_resolve_molecule, "Resolve a SMILES or drug name to a valid MoleculeRecord.")
        self.registry.register("discover_rules", self._tool_discover_rules, "Use trained rules or discover/compile LLM4SD-style rules.")
        self.registry.register("discover_rules_fallback", self._tool_discover_rules_fallback, "Use local rule-library defaults when rule discovery fails.")
        self.registry.register("maybe_enable_mllm_visual_semantic", self._tool_maybe_enable_mllm_visual_semantic, "Optionally enable MLLM visual semantics when the planner recommends it.")
        self.registry.register("build_single_features", self._tool_build_single_features, "Build the configured single-molecule feature bundle.")
        self.registry.register("build_single_features_fallback", self._tool_build_single_features_fallback, "Build a smaller RDKit plus rules feature bundle.")
        self.registry.register("build_ddi_pair", self._tool_build_ddi_pair, "Fuse two single-molecule feature bundles into a DDI pair bundle.")
        self.registry.register("build_ddi_pair_fallback", self._tool_build_ddi_pair_fallback, "Rebuild both molecules with fallback features and fuse them.")
        self.registry.register("select_prediction_strategy", self._tool_select_prediction_strategy, "Choose a trained model or zero-shot prediction strategy.")
        self.registry.register("predict_single", self._tool_predict_single, "Predict a single-molecule property or ADMET endpoint.")
        self.registry.register("predict_single_zero_shot", self._tool_predict_single_heuristic, "Use ZeroShotPredictionAgent for single-molecule prediction.")
        self.registry.register("predict_single_heuristic", self._tool_predict_single_heuristic, "Compatibility tool: use ZeroShotPredictionAgent for single-molecule prediction.")
        self.registry.register("predict_ddi", self._tool_predict_ddi, "Predict DDI risk.")
        self.registry.register("predict_ddi_zero_shot", self._tool_predict_ddi_heuristic, "Use ZeroShotPredictionAgent for DDI prediction.")
        self.registry.register("predict_ddi_heuristic", self._tool_predict_ddi_heuristic, "Compatibility tool: use ZeroShotPredictionAgent for DDI prediction.")
        self.registry.register("assess_prediction_confidence", self._tool_assess_prediction_confidence, "Assess score/probability margin and mark low-confidence predictions.")
        self.registry.register("retrieve_evidence", self._tool_retrieve_evidence, "Retrieve local evidence for the prediction.")
        self.registry.register("retrieve_additional_evidence", self._tool_retrieve_additional_evidence, "Append extra local evidence for low-confidence predictions.")
        self.registry.register("retrieve_rag_evidence", self._tool_retrieve_rag_evidence, "Retrieve local RAG snippets and attach them as evidence context.")
        self.registry.register("retrieve_empty_evidence", self._tool_retrieve_empty_evidence, "Continue with empty evidence.")
        self.registry.register("reason_over_prediction", self._tool_reason_over_prediction, "Produce structured reasoning over decisions, fallbacks, and signals.")
        self.registry.register("build_report", self._tool_build_report, "Build the final report object.")

    def _execute_step(self, state: AgentState, step: AgentPlanStep) -> None:
        start = perf_counter()
        if step.condition and not self._condition_met(state, step.condition):
            step.status = "skipped"
            step.message = f"condition_not_met:{step.condition}"
            state.trace.append(
                AgentObservation(
                    step_id=step.step_id,
                    tool_name=step.tool_name,
                    status=step.status,
                    duration_sec=0.0,
                    output_summary={"condition": step.condition},
                )
            )
            self._emit_progress(
                {
                    "event_type": "step_skipped",
                    "step_id": step.step_id,
                    "tool_name": step.tool_name,
                    "description": step.description,
                    "status": step.status,
                    "duration_sec": 0.0,
                    "message": step.message,
                }
            )
            return
        step.status = "running"
        self._emit_progress(
            {
                "event_type": "step_started",
                "step_id": step.step_id,
                "tool_name": step.tool_name,
                "description": step.description,
                "status": step.status,
                "message": "running",
            }
        )
        output: Dict[str, Any] = {}
        error: Optional[str] = None
        fallback_used: Optional[str] = None
        try:
            output = self.registry.invoke(step.tool_name, state=state, **step.params)
            step.status = "completed"
            step.message = "completed"
        except Exception as exc:
            error = str(exc)
            if not step.fallback_tool:
                step.status = "failed"
                step.message = error
                duration = round(perf_counter() - start, 6)
                state.trace.append(
                    AgentObservation(
                        step_id=step.step_id,
                        tool_name=step.tool_name,
                        status=step.status,
                        duration_sec=duration,
                        output_summary=output,
                        error=error,
                    )
                )
                self._emit_progress(
                    {
                        "event_type": "step_failed",
                        "step_id": step.step_id,
                        "tool_name": step.tool_name,
                        "description": step.description,
                        "status": step.status,
                        "duration_sec": duration,
                        "message": error,
                        "error": error,
                        "output_summary": output,
                    }
                )
                raise
            fallback_used = step.fallback_tool
            self._emit_progress(
                {
                    "event_type": "step_fallback",
                    "step_id": step.step_id,
                    "tool_name": step.tool_name,
                    "description": step.description,
                    "status": "fallback_running",
                    "message": f"fallback_used:{fallback_used}",
                    "error": error,
                    "fallback_tool": fallback_used,
                }
            )
            state.warnings.append(f"{step.tool_name} failed; used fallback {fallback_used}: {error}")
            state.reasoning["fallbacks"].append(
                {
                    "step_id": step.step_id,
                    "failed_tool": step.tool_name,
                    "fallback_tool": fallback_used,
                    "error": error,
                }
            )
            output = self.registry.invoke(fallback_used, state=state, **step.params)
            step.status = "fallback_completed"
            step.message = f"fallback_used:{fallback_used}"
        duration = round(perf_counter() - start, 6)
        state.trace.append(
            AgentObservation(
                step_id=step.step_id,
                tool_name=step.tool_name,
                status=step.status,
                duration_sec=duration,
                output_summary=output,
                error=error,
                fallback_tool=fallback_used,
            )
        )
        self._emit_progress(
            {
                "event_type": "step_completed" if fallback_used is None else "step_fallback_completed",
                "step_id": step.step_id,
                "tool_name": step.tool_name,
                "description": step.description,
                "status": step.status,
                "duration_sec": duration,
                "message": step.message,
                "error": error,
                "fallback_tool": fallback_used,
                "output_summary": output,
            }
        )

    def _emit_progress(self, event: Dict[str, Any]) -> None:
        if self.progress_callback is None:
            return
        try:
            self.progress_callback(event)
        except Exception:
            pass

    def _tool_parse_input(self, *, state: AgentState, task_type: Optional[str] = None, dataset_name: Optional[str] = None) -> Dict[str, Any]:
        parsed = self.input_agent.parse(state.request, task_type=task_type, dataset_name=dataset_name)
        state.parsed = parsed
        state.reasoning["decisions"].append(
            {
                "step": "parse_input",
                "decision": "task_and_dataset_selected",
                "task_type": parsed.task_type,
                "dataset_name": parsed.dataset_name,
                "molecule_mentions": parsed.molecules,
            }
        )
        return {
            "task_type": parsed.task_type,
            "dataset_name": parsed.dataset_name,
            "molecule_mentions": parsed.molecules,
        }

    def _tool_resolve_molecule(self, *, state: AgentState, index: int, role: str) -> Dict[str, Any]:
        if state.parsed is None:
            raise ValueError("Input must be parsed before molecule resolution.")
        if index >= len(state.parsed.molecules):
            raise ValueError(f"Missing molecule mention at index {index}.")
        mention = state.parsed.molecules[index]
        record = self.processor.process_input(mention, self.input_agent)
        self._set_at(state.molecules, index, record)
        if record.errors:
            state.warnings.extend(record.errors)
        resolution_warnings = record.metadata.get("resolution_warnings", []) if isinstance(record.metadata, dict) else []
        if resolution_warnings:
            state.warnings.extend(str(item) for item in resolution_warnings)
        if not record.is_valid:
            raise ValueError(f"Invalid molecule input for {role}: {record.errors}")
        return {
            "role": role,
            "original_input": record.original_input,
            "canonical_smiles": record.canonical_smiles,
            "name": record.name,
            "source": record.source,
            "resolution_trace": record.metadata.get("resolution_trace", []) if isinstance(record.metadata, dict) else [],
        }

    def _tool_discover_rules(self, *, state: AgentState) -> Dict[str, Any]:
        if state.parsed is None:
            raise ValueError("Input must be parsed before rule discovery.")
        if self.prediction_agent.training_rules:
            state.rules = list(self.prediction_agent.training_rules)
            source = "trained_model_rules"
        else:
            state.rules = self.rule_agent.discover(state.parsed.task_type, state.parsed.dataset_name)
            source = "rule_discovery_agent"
        state.reasoning["decisions"].append(
            {
                "step": "discover_rules",
                "decision": source,
                "n_rules": len(state.rules),
            }
        )
        return self._rule_summary(source, state.rules)

    def _tool_discover_rules_fallback(self, *, state: AgentState) -> Dict[str, Any]:
        if state.parsed is None:
            raise ValueError("Input must be parsed before fallback rule discovery.")
        state.rules = self.rule_agent.library.literature_rules(state.parsed.dataset_name, state.parsed.task_type)
        return self._rule_summary("local_rule_library_fallback", state.rules)

    def _tool_maybe_enable_mllm_visual_semantic(self, *, state: AgentState) -> Dict[str, Any]:
        visual_extractor = self.feature_builder.visual_extractor
        already_enabled = bool(getattr(visual_extractor, "use_mllm", False))
        planner_metadata = state.plan.metadata.get("planner", {}) if state.plan else {}
        recommendations = planner_metadata.get("recommendations", {}) if isinstance(planner_metadata, dict) else {}
        planner_recommended = bool(recommendations.get("use_mllm_visual_semantic"))
        should_enable = bool(already_enabled or (self.allow_planner_mllm and planner_recommended))

        if should_enable and not already_enabled:
            self.feature_builder.visual_extractor = VisualSemanticExtractor(
                use_mllm=True,
                mllm_profile=self.mllm_profile,
                cache_dir=self.visual_semantic_cache_dir,
                cache_read=self.visual_semantic_cache_read,
                cache_write=self.visual_semantic_cache_write,
                cache_only=self.visual_semantic_cache_only,
            )
            state.reasoning["decisions"].append(
                {
                    "step": "maybe_enable_mllm_visual_semantic",
                    "decision": "enabled_by_planner",
                    "mllm_profile": self.mllm_profile,
                }
            )
            return {
                "mllm_visual_semantic": "enabled_by_planner",
                "mllm_profile": self.mllm_profile,
            }

        decision = "already_enabled" if already_enabled else "kept_current_configuration"
        state.reasoning["decisions"].append(
            {
                "step": "maybe_enable_mllm_visual_semantic",
                "decision": decision,
                "allow_planner_mllm": self.allow_planner_mllm,
                "planner_recommended": planner_recommended,
            }
        )
        return {
            "mllm_visual_semantic": decision,
            "allow_planner_mllm": self.allow_planner_mllm,
            "planner_recommended": planner_recommended,
        }

    def _tool_build_single_features(self, *, state: AgentState, index: int, role: str) -> Dict[str, Any]:
        record = self._molecule_at(state, index)
        bundle = self.feature_builder.build_single(
            record,
            rules=state.rules,
            task_type=self._task_type(state),
            dataset_name=self._dataset_name(state),
        )
        self._set_at(state.single_features, index, bundle)
        self._attach_single_artifact_paths(state, role, bundle)
        return self._single_feature_summary(role, bundle)

    def _tool_build_single_features_fallback(self, *, state: AgentState, index: int, role: str) -> Dict[str, Any]:
        record = self._molecule_at(state, index)
        builder = self._get_fallback_feature_builder()
        bundle = builder.build_single(
            record,
            rules=state.rules,
            task_type=self._task_type(state),
            dataset_name=self._dataset_name(state),
        )
        self._set_at(state.single_features, index, bundle)
        self._attach_single_artifact_paths(state, role, bundle)
        state.metadata["feature_fallback_used"] = True
        return self._single_feature_summary(role, bundle, fallback=True)

    def _tool_build_ddi_pair(self, *, state: AgentState) -> Dict[str, Any]:
        pair = self.ddi_agent.build_pair(self._feature_at(state, 0), self._feature_at(state, 1))
        state.ddi_pair = pair
        return self._ddi_pair_summary(pair)

    def _tool_build_ddi_pair_fallback(self, *, state: AgentState) -> Dict[str, Any]:
        builder = self._get_fallback_feature_builder()
        features: List[SingleMoleculeFeatureBundle] = []
        for index, role in [(0, "drug_a"), (1, "drug_b")]:
            bundle = builder.build_single(
                self._molecule_at(state, index),
                rules=state.rules,
                task_type="ddi_prediction",
                dataset_name=self._dataset_name(state),
            )
            self._set_at(state.single_features, index, bundle)
            self._attach_single_artifact_paths(state, role, bundle)
            features.append(bundle)
        state.ddi_pair = builder.build_ddi_pair(features[0], features[1])
        state.metadata["ddi_pair_fallback_used"] = True
        return self._ddi_pair_summary(state.ddi_pair, fallback=True)

    def _tool_select_prediction_strategy(self, *, state: AgentState) -> Dict[str, Any]:
        if self.prediction_agent.model is not None:
            strategy = "trained_model"
            model_type = self.prediction_agent.model.model_type
        else:
            strategy = "zero_shot_expected"
            model_type = ZERO_SHOT_MODEL_TYPE
            warning = "No trained task-specific model was loaded; using ZeroShotPredictionAgent with a non-calibrated score."
            if warning not in state.warnings:
                state.warnings.append(warning)
        state.metadata["prediction_strategy"] = strategy
        state.reasoning["decisions"].append(
            {
                "step": "select_prediction_strategy",
                "decision": strategy,
                "model_type": model_type,
            }
        )
        return {
            "prediction_strategy": strategy,
            "model_type": model_type,
        }

    def _tool_predict_single(self, *, state: AgentState) -> Dict[str, Any]:
        prediction = self.prediction_agent.predict_single(self._feature_at(state, 0), self._task_type(state), self._dataset_name(state))
        state.predictions = [prediction]
        return self._prediction_summary(prediction)

    def _tool_predict_single_heuristic(self, *, state: AgentState) -> Dict[str, Any]:
        state.metadata["prediction_strategy"] = "zero_shot_after_prediction_failure"
        prediction = self.prediction_agent.zero_shot.predict_single(self._feature_at(state, 0), self._task_type(state), self._dataset_name(state))
        state.predictions = [prediction]
        return self._prediction_summary(prediction, fallback=True)

    def _tool_predict_ddi(self, *, state: AgentState) -> Dict[str, Any]:
        if state.ddi_pair is None:
            raise ValueError("DDI pair features are missing.")
        prediction = self.prediction_agent.predict_ddi(state.ddi_pair, dataset_name=self._dataset_name(state))
        state.predictions = [prediction]
        return self._prediction_summary(prediction)

    def _tool_predict_ddi_heuristic(self, *, state: AgentState) -> Dict[str, Any]:
        if state.ddi_pair is None:
            raise ValueError("DDI pair features are missing.")
        state.metadata["prediction_strategy"] = "zero_shot_after_prediction_failure"
        prediction = self.prediction_agent.zero_shot.predict_ddi(state.ddi_pair, dataset_name=self._dataset_name(state))
        state.predictions = [prediction]
        return self._prediction_summary(prediction, fallback=True)

    def _tool_assess_prediction_confidence(self, *, state: AgentState) -> Dict[str, Any]:
        if not state.predictions:
            raise ValueError("Prediction must be produced before confidence assessment.")
        prediction = state.predictions[0]
        probability = prediction.probability
        margin = None
        reasons: List[str] = []
        low_confidence = False
        if probability is not None:
            margin = abs(float(probability) - 0.5)
            if margin < self.confidence_margin_threshold:
                low_confidence = True
                reasons.append("near_decision_threshold")
        else:
            if prediction.model_type == ZERO_SHOT_MODEL_TYPE:
                confidence_level = str(prediction.details.get("confidence_level", "low"))
                low_confidence = True
                reasons.append("zero_shot_non_calibrated_score")
                if confidence_level == "low":
                    reasons.append("zero_shot_low_confidence_evidence")
                else:
                    reasons.append(f"zero_shot_requires_caution_despite_{confidence_level}_evidence")
            else:
                low_confidence = True
                reasons.append("no_probability_available")
        if not state.evidence and state.metadata.get("prediction_strategy") in {"zero_shot_expected", "zero_shot_after_prediction_failure"}:
            reasons.append("no_evidence_retrieved_yet")

        state.metadata["low_confidence_prediction"] = low_confidence
        state.metadata["prediction_confidence"] = {
            "probability": probability,
            "zero_shot_score": prediction.details.get("zero_shot_score"),
            "zero_shot_confidence_level": prediction.details.get("confidence_level"),
            "margin_from_threshold": margin,
            "threshold": self.confidence_margin_threshold,
            "low_confidence": low_confidence,
            "reasons": reasons,
        }
        state.reasoning["decisions"].append(
            {
                "step": "assess_prediction_confidence",
                "decision": "low_confidence" if low_confidence else "sufficient_margin",
                "probability": probability,
                "zero_shot_score": prediction.details.get("zero_shot_score"),
                "zero_shot_confidence_level": prediction.details.get("confidence_level"),
                "margin_from_threshold": margin,
                "reasons": reasons,
            }
        )
        return state.metadata["prediction_confidence"]

    def _tool_retrieve_evidence(self, *, state: AgentState) -> Dict[str, Any]:
        if self._task_type(state) == "ddi_prediction":
            if state.ddi_pair is None:
                raise ValueError("DDI pair features are missing.")
            state.evidence = self.evidence_agent.retrieve_ddi(state.ddi_pair)
        else:
            state.evidence = self.evidence_agent.retrieve_single(self._feature_at(state, 0), dataset_name=self._dataset_name(state))
        return {
            "n_evidence_items": len(state.evidence),
            "sources": sorted({str(item.get("source")) for item in state.evidence}),
        }

    def _tool_retrieve_additional_evidence(self, *, state: AgentState) -> Dict[str, Any]:
        confidence = state.metadata.get("prediction_confidence", {})
        reasons = confidence.get("reasons", [])
        claim = "Prediction is near the decision threshold, so the agent marked the report for cautious interpretation."
        if "zero_shot_non_calibrated_score" in reasons:
            claim = "Prediction uses ZeroShotPredictionAgent, so the agent marked the report for cautious interpretation even when evidence is directional."
        item: Dict[str, Any] = {
            "source": "agent_confidence_review",
            "claim": claim,
            "value": confidence,
        }
        if self._task_type(state) == "ddi_prediction" and state.ddi_pair is not None:
            item["pair_signal"] = {
                "fingerprint_similarity": state.ddi_pair.fingerprint_similarity,
                "admet_risk_features": state.ddi_pair.admet_risk_features,
                "cyp_related_features": state.ddi_pair.cyp_related_features,
            }
        elif state.single_features:
            bundle = self._feature_at(state, 0)
            item["molecule_signal"] = {
                "descriptor_snapshot": {
                    key: bundle.rdkit_descriptors.get(key)
                    for key in ["MolWt", "MolLogP", "TPSA", "NumHDonors", "NumHAcceptors", "QED"]
                },
                "matched_rules": {key: value for key, value in bundle.llm_rule_features.values.items() if value},
            }
        state.evidence.append(item)
        state.reasoning["decisions"].append(
            {
                "step": "retrieve_additional_evidence",
                "decision": "appended_low_confidence_review",
                "n_evidence_items": len(state.evidence),
            }
        )
        return {
            "n_evidence_items": len(state.evidence),
            "added_source": item["source"],
        }

    def _tool_retrieve_rag_evidence(self, *, state: AgentState) -> Dict[str, Any]:
        if not self.use_rag_evidence:
            state.metadata["rag_evidence"] = {"enabled": False, "reason": "disabled_by_config"}
            return {"rag_evidence": "disabled", "n_rag_evidence_items": 0}

        query = self._rag_query(state)
        context_metadata = {
            "task_type": self._task_type(state),
            "dataset_name": self._dataset_name(state),
            "request": state.request,
        }
        if self._task_type(state) == "ddi_prediction" and state.ddi_pair is not None:
            result = self.ddi_rag_evidence_agent.retrieve_for_pair(
                state.ddi_pair,
                query=query,
                context_metadata=context_metadata,
                top_k=self.rag_top_k,
            )
        else:
            result = self.rag_evidence_agent.retrieve(
                query=query,
                context_metadata=context_metadata,
                corpus_dirs=self.rag_corpus_dirs,
                top_k=self.rag_top_k,
            )
        evidence_items = list(result.get("evidence_items", []) or [])
        state.evidence.extend(evidence_items)
        state.artifacts["rag_evidence"] = result
        state.artifacts["rag_context"] = result.get("retrieved_context", [])
        state.artifacts["retrieved_context"] = result.get("retrieved_context", [])
        for key in ["rag_evidence_path", "rag_evidence_markdown_path"]:
            value = (result.get("artifacts") or {}).get(key)
            if value:
                state.artifacts[key] = value
        warnings = list(result.get("warnings", []) or [])
        for warning in warnings:
            message = f"RAG evidence warning: {warning}"
            if message not in state.warnings:
                state.warnings.append(message)
        state.metadata["rag_evidence"] = {
            "enabled": True,
            "query": query,
            "provider": result.get("provider"),
            "backend": result.get("backend"),
            "n_documents": result.get("n_documents"),
            "n_chunks": result.get("n_chunks"),
            "n_retrieved_context": len(result.get("retrieved_context", []) or []),
            "n_evidence_items": len(evidence_items),
            "warnings": warnings,
        }
        state.reasoning["decisions"].append(
            {
                "step": "retrieve_rag_evidence",
                "decision": "retrieved_rag_context" if evidence_items else "no_rag_context",
                "query": query,
                "provider": result.get("provider"),
                "backend": result.get("backend"),
                "n_evidence_items": len(evidence_items),
                "warnings": warnings,
            }
        )
        return {
            "query": query,
            "provider": result.get("provider"),
            "backend": result.get("backend"),
            "n_rag_evidence_items": len(evidence_items),
            "n_retrieved_context": len(result.get("retrieved_context", []) or []),
            "warnings": warnings,
        }

    def _tool_retrieve_empty_evidence(self, *, state: AgentState) -> Dict[str, Any]:
        state.evidence = []
        return {"n_evidence_items": 0, "sources": [], "fallback": True}

    def _tool_reason_over_prediction(self, *, state: AgentState) -> Dict[str, Any]:
        if not state.predictions:
            raise ValueError("Prediction must be produced before reasoning.")
        prediction = state.predictions[0]
        quality_result = self.evidence_quality_agent.assess(
            state.evidence,
            prediction.to_dict(),
            task_type=self._task_type(state),
            dataset_name=self._dataset_name(state),
            request=state.request,
            molecules=[molecule.to_dict() for molecule in state.molecules],
        )
        state.evidence = quality_result.annotated_evidence
        state.artifacts["evidence_quality"] = quality_result.summary
        if quality_result.dropped_evidence:
            state.artifacts["dropped_evidence"] = quality_result.dropped_evidence[:20]
        evidence_scoring, annotated_evidence = score_evidence_for_prediction(
            state.evidence,
            prediction.to_dict(),
            task_type=self._task_type(state),
            dataset_name=self._dataset_name(state),
        )
        state.evidence = annotated_evidence
        prediction.details["evidence_grounded_scoring"] = evidence_scoring
        state.artifacts["evidence_grounded_scoring"] = evidence_scoring
        signals = self._confidence_signals(state, prediction)
        state.reasoning["confidence_signals"] = signals
        state.reasoning["summary"] = self._reasoning_summary(state, prediction, signals)
        state.reasoning["decisions"].append(
            {
                "step": "reason_over_prediction",
                "decision": "summarized_prediction_context",
                "model_type": prediction.model_type,
                "n_evidence_items": len(state.evidence),
                "evidence_grounded_score": evidence_scoring.get("evidence_grounded_score"),
                "evidence_grounded_recommendation": evidence_scoring.get("recommendation"),
                "evidence_quality": quality_result.summary,
            }
        )
        return {
            "summary": state.reasoning["summary"],
            "n_confidence_signals": len(signals),
            "evidence_grounded_scoring": evidence_scoring,
            "evidence_quality": quality_result.summary,
        }

    def _tool_build_report(self, *, state: AgentState) -> Dict[str, Any]:
        artifacts = self._report_artifacts(state)
        state.artifacts.update(artifacts)
        provisional_report = self.report_agent.build(
            request=state.request,
            task_type=self._task_type(state),
            molecules=state.molecules,
            predictions=state.predictions,
            evidence=state.evidence,
            artifacts=state.artifacts,
            warnings=state.warnings,
            plan=state.plan.to_dict() if state.plan else {},
            trace=[obs.to_dict() for obs in state.trace],
            reasoning=state.reasoning,
        )
        state.artifacts["failure_mode_assessment"] = assess_failure_modes(provisional_report.to_dict())
        narrative_report = self.narrative_agent.build(
            request=state.request,
            task_type=self._task_type(state),
            molecules=state.molecules,
            predictions=state.predictions,
            evidence=state.evidence,
            artifacts=state.artifacts,
            reasoning=state.reasoning,
            trace=[obs.to_dict() for obs in state.trace],
            warnings=state.warnings,
        )
        state.artifacts["narrative_report"] = narrative_report
        narrative_artifacts = narrative_report.get("artifacts", {}) if isinstance(narrative_report, dict) else {}
        for key in ["narrative_report_path", "narrative_markdown_path"]:
            if narrative_artifacts.get(key):
                state.artifacts[key] = narrative_artifacts[key]
        state.report = self.report_agent.build(
            request=state.request,
            task_type=self._task_type(state),
            molecules=state.molecules,
            predictions=state.predictions,
            evidence=state.evidence,
            artifacts=state.artifacts,
            warnings=state.warnings,
            plan=state.plan.to_dict() if state.plan else {},
            trace=[obs.to_dict() for obs in state.trace],
            reasoning=state.reasoning,
            narrative_report=narrative_report,
        )
        return {
            "report_ready": True,
            "n_warnings": len(state.warnings),
            "narrative_provider": narrative_report.get("provider") if isinstance(narrative_report, dict) else None,
        }

    def _finalize(self, state: AgentState) -> None:
        plan_path = state.output_dir / "agent_plan.json"
        trace_path = state.output_dir / "agent_trace.json"
        state_path = state.output_dir / "agent_state_summary.json"
        markdown_path = state.output_dir / "report.md"
        case_study_path = state.output_dir / "case_study_report.md"
        reproducibility_path = state.output_dir / "reproducibility.json"
        state.artifacts.update(
            {
                "agent_plan_path": str(plan_path),
                "agent_trace_path": str(trace_path),
                "agent_state_summary_path": str(state_path),
                "markdown_report_path": str(markdown_path),
                "case_study_report_path": str(case_study_path),
                "reproducibility_log_path": str(reproducibility_path),
                "narrative_report_path": state.artifacts.get("narrative_report_path"),
                "narrative_markdown_path": state.artifacts.get("narrative_markdown_path"),
                "rag_evidence_path": state.artifacts.get("rag_evidence_path"),
                "rag_evidence_markdown_path": state.artifacts.get("rag_evidence_markdown_path"),
                **usage_artifact_paths(state.output_dir),
            }
        )
        write_json(plan_path, state.plan.to_dict() if state.plan else {})
        write_json(trace_path, {"observations": [obs.to_dict() for obs in state.trace]})
        write_json(state_path, self._state_summary(state))
        if state.report is None:
            raise ValueError("No report was built.")
        state.report.plan = state.plan.to_dict() if state.plan else {}
        state.report.trace = [obs.to_dict() for obs in state.trace]
        state.report.reasoning = state.reasoning
        state.report.artifacts.update(state.artifacts)
        write_json(self._report_path(state), state.report.to_dict())
        write_text(markdown_path, self.report_agent.to_markdown(state.report))
        write_case_study_report(state.report.to_dict(), case_study_path)
        write_reproducibility_log(
            state.output_dir,
            config={
                **self.runtime_config,
                "task_type": self._task_type(state),
                "dataset_name": self._dataset_name(state),
                "request": state.request,
            },
            artifacts={
                "model_path": self.runtime_config.get("model_path"),
                "imagemol_checkpoint_path": self.runtime_config.get("imagemol_checkpoint_path"),
                "smiles_encoder_model_name": self.runtime_config.get("smiles_encoder_model_name"),
                "report_path": self._report_path(state),
                "markdown_report_path": markdown_path,
                "narrative_report_path": state.artifacts.get("narrative_report_path"),
                "narrative_markdown_path": state.artifacts.get("narrative_markdown_path"),
                "rag_evidence_path": state.artifacts.get("rag_evidence_path"),
                "rag_evidence_markdown_path": state.artifacts.get("rag_evidence_markdown_path"),
                "api_usage_log_path": state.artifacts.get("api_usage_log_path"),
                "api_usage_summary_path": state.artifacts.get("api_usage_summary_path"),
                "agent_plan_path": plan_path,
                "agent_trace_path": trace_path,
            },
            extra={
                "n_warnings": len(state.warnings),
                "n_evidence_items": len(state.evidence),
                "prediction_strategy": state.metadata.get("prediction_strategy"),
                "feature_fallback_used": state.metadata.get("feature_fallback_used", False),
                "ddi_pair_fallback_used": state.metadata.get("ddi_pair_fallback_used", False),
                "rag_evidence": state.metadata.get("rag_evidence", {}),
            },
        )

    def _record_planner_warning(self, state: AgentState, planner_metadata: Dict[str, Any]) -> None:
        mode = planner_metadata.get("planner_mode")
        warning: Optional[str] = None
        if mode == "rule_policy_after_llm_unavailable":
            warning = "LLM planner is unavailable; the workflow used the default deterministic planner."
        elif mode == "rule_policy_after_llm_plan_error":
            reason = planner_metadata.get("llm_error") or "unknown planner error"
            warning = f"LLM planner failed or returned an unsafe plan; the workflow used the default deterministic planner. Reason: {reason}"
        if not warning:
            return

        planner_metadata["planner_warning"] = warning
        state.warnings.append(warning)
        state.reasoning["fallbacks"].append(
            {
                "step_id": "planner",
                "failed_tool": "llm_planner",
                "fallback_tool": "rule_policy",
                "error": warning,
                "planner_mode": mode,
                "validation_status": planner_metadata.get("validation_status"),
            }
        )

    def _record_decision_warning(self, state: AgentState, decision_metadata: Dict[str, Any]) -> None:
        warning = decision_metadata.get("llm_warning")
        if warning:
            state.warnings.append(str(warning))
            state.reasoning["fallbacks"].append(
                {
                    "step_id": "llm_decision",
                    "failed_tool": "llm_decision_layer",
                    "fallback_tool": "local_decision_policy",
                    "error": str(warning),
                    "decision_mode": decision_metadata.get("decision_mode"),
                    "validation_status": decision_metadata.get("validation_status"),
                }
            )
        for item in decision_metadata.get("warnings", []) or []:
            message = f"LLM decision warning: {item}"
            if message not in state.warnings:
                state.warnings.append(message)

    def _record_plan_critic_warning(self, state: AgentState, critic_metadata: Dict[str, Any]) -> None:
        warning = critic_metadata.get("critic_warning")
        if warning:
            state.warnings.append(str(warning))
            state.reasoning["fallbacks"].append(
                {
                    "step_id": "plan_critic",
                    "failed_tool": "llm_plan_critic",
                    "fallback_tool": "local_plan_critic",
                    "error": str(warning),
                    "critic_mode": critic_metadata.get("critic_mode"),
                    "validation_status": critic_metadata.get("validation_status"),
                }
            )
        status = critic_metadata.get("status")
        if status in {"warn", "fail"}:
            message = f"Plan critic status={status}: {critic_metadata.get('rationale') or 'review plan metadata'}"
            if message not in state.warnings:
                state.warnings.append(message)

    def _apply_decision_route(
        self,
        state: AgentState,
        decision_metadata: Dict[str, Any],
        *,
        explicit_task_type: Optional[str],
        explicit_dataset_name: Optional[str],
    ) -> None:
        if state.parsed is None:
            return
        route = decision_metadata.get("route", {})
        if not isinstance(route, dict):
            return
        confidence = float(route.get("route_confidence") or 0.0)
        if confidence < 0.70:
            state.reasoning["decisions"].append(
                {
                    "step": "llm_decision",
                    "decision": decision_metadata.get("decision_mode"),
                    "route_applied": False,
                    "reason": "route_confidence_below_threshold",
                    "route_confidence": confidence,
                }
            )
            return

        parsed = state.parsed
        task_type = parsed.task_type
        dataset_name = parsed.dataset_name
        molecules = list(parsed.molecules)
        applied: Dict[str, Any] = {}

        route_task = str(route.get("task_type") or "").strip()
        if route_task and explicit_task_type is None and route_task != task_type:
            task_type = route_task
            applied["task_type"] = route_task

        route_dataset = str(route.get("dataset_name") or "").strip()
        if route_dataset and explicit_dataset_name is None and route_dataset != dataset_name:
            dataset_name = route_dataset
            applied["dataset_name"] = route_dataset

        route_molecules = route.get("molecules")
        if isinstance(route_molecules, list):
            cleaned_molecules = [str(item).strip() for item in route_molecules if str(item).strip()]
            if task_type == "ddi_prediction":
                cleaned_molecules = cleaned_molecules[:2]
            else:
                cleaned_molecules = cleaned_molecules[:1]
            if cleaned_molecules and self._route_molecules_are_safe(state.request, cleaned_molecules, molecules):
                if cleaned_molecules != molecules:
                    molecules = cleaned_molecules
                    applied["molecules"] = cleaned_molecules

        if applied:
            state.parsed = ParsedInput(
                task_type=task_type,
                dataset_name=dataset_name,
                molecules=molecules,
                raw_request=parsed.raw_request,
            )

        state.reasoning["decisions"].append(
            {
                "step": "llm_decision",
                "decision": decision_metadata.get("decision_mode"),
                "route_applied": bool(applied),
                "applied": applied,
                "route_confidence": confidence,
                "recommendations": decision_metadata.get("recommendations", {}),
                "rationale": route.get("rationale"),
                "needs_clarification": route.get("needs_clarification"),
                "clarification_question": route.get("clarification_question"),
            }
        )

    def _route_molecules_are_safe(self, request: str, proposed: List[str], current: List[str]) -> bool:
        request_norm = "".join(str(request or "").lower().split())
        current_norm = {"".join(item.lower().split()) for item in current}
        if not current:
            return True
        for item in proposed:
            item_norm = "".join(item.lower().split())
            if item_norm in current_norm:
                continue
            if item_norm and item_norm in request_norm:
                continue
            return False
        return True

    def _merge_decision_recommendations(
        self,
        planner_metadata: Dict[str, Any],
        decision_metadata: Dict[str, Any],
    ) -> None:
        planner_recommendations = planner_metadata.get("recommendations", {})
        if not isinstance(planner_recommendations, dict):
            planner_recommendations = {}
        decision_recommendations = decision_metadata.get("recommendations", {})
        if not isinstance(decision_recommendations, dict):
            decision_recommendations = {}
        planner_metadata["recommendations"] = {
            **planner_recommendations,
            **decision_recommendations,
        }

    def _report_path(self, state: AgentState) -> Path:
        if self._task_type(state) == "ddi_prediction":
            return state.output_dir / "ddi_report.json"
        return state.output_dir / "single_molecule_report.json"

    def _planner_context(self, state: AgentState) -> Dict[str, Any]:
        visual_extractor = self.feature_builder.visual_extractor
        return {
            "request": state.request,
            "feature_set": self.feature_builder.feature_set,
            "model_available": self.prediction_agent.model is not None,
            "trained_rule_count": len(self.prediction_agent.training_rules),
            "imagemol_encoder": self.feature_builder.imagemol_encoder.encoder_name if self.feature_builder.imagemol_encoder else "disabled",
            "smiles_encoder": self.feature_builder.smiles_encoder.encoder_name if self.feature_builder.smiles_encoder else None,
            "use_smiles_encoder": self.feature_builder.use_smiles_encoder,
            "mllm_currently_enabled": bool(getattr(visual_extractor, "use_mllm", False)),
            "mllm_profile": self.mllm_profile,
            "allow_planner_mllm": self.allow_planner_mllm,
            "confidence_margin_threshold": self.confidence_margin_threshold,
            "use_rag_evidence": self.use_rag_evidence,
            "use_external_evidence": self.use_external_evidence,
            "rag_backend": self.rag_backend,
            "rag_top_k": self.rag_top_k,
            "rag_corpus_dirs": self.rag_corpus_dirs,
        }

    def _condition_met(self, state: AgentState, condition: str) -> bool:
        if condition == "low_confidence_prediction":
            return bool(state.metadata.get("low_confidence_prediction"))
        raise ValueError(f"Unsupported plan condition: {condition}")

    def _get_fallback_feature_builder(self) -> FeatureBuilder:
        if self._fallback_feature_builder is None:
            self._fallback_feature_builder = FeatureBuilder(
                output_dir=self.output_dir / "fallback_features",
                feature_set="rdkit_plus_llm_rules",
                imagemol_checkpoint_path=None,
                use_mllm_visual_semantic=False,
                generate_grad_cam=False,
            )
        return self._fallback_feature_builder

    def _task_type(self, state: AgentState) -> str:
        return state.parsed.task_type if state.parsed else "property_prediction"

    def _dataset_name(self, state: AgentState) -> str:
        return state.parsed.dataset_name if state.parsed else "BBBP"

    def _set_at(self, values: List[Any], index: int, value: Any) -> None:
        while len(values) <= index:
            values.append(None)
        values[index] = value

    def _molecule_at(self, state: AgentState, index: int) -> MoleculeRecord:
        if index >= len(state.molecules) or state.molecules[index] is None:
            raise ValueError(f"Molecule at index {index} is missing.")
        return state.molecules[index]

    def _feature_at(self, state: AgentState, index: int) -> SingleMoleculeFeatureBundle:
        if index >= len(state.single_features) or state.single_features[index] is None:
            raise ValueError(f"Feature bundle at index {index} is missing.")
        return state.single_features[index]

    def _rule_summary(self, source: str, rules: List[Rule]) -> Dict[str, Any]:
        source_counts = Counter(rule.source for rule in rules)
        return {
            "source": source,
            "n_rules": len(rules),
            "n_executable_rules": sum(1 for rule in rules if rule.is_executable),
            "source_counts": dict(source_counts),
        }

    def _single_feature_summary(
        self,
        role: str,
        bundle: SingleMoleculeFeatureBundle,
        fallback: bool = False,
    ) -> Dict[str, Any]:
        return {
            "role": role,
            "canonical_smiles": bundle.molecule.canonical_smiles,
            "feature_set": bundle.feature_schema.feature_set,
            "feature_dim": len(bundle.feature_vector),
            "blocks": bundle.feature_schema.blocks,
            "image_encoder": bundle.image.encoder_name if bundle.image else None,
            "smiles_encoder": bundle.smiles.encoder_name if bundle.smiles else None,
            "visual_provider": bundle.visual_semantics.provider if bundle.visual_semantics else None,
            "fallback": fallback,
        }

    def _ddi_pair_summary(self, pair: DDIPairFeatureBundle, fallback: bool = False) -> Dict[str, Any]:
        return {
            "feature_set": pair.feature_schema.feature_set,
            "feature_dim": len(pair.feature_vector),
            "fingerprint_similarity": pair.fingerprint_similarity,
            "fallback": fallback,
        }

    def _prediction_summary(self, prediction: PredictionResult, fallback: bool = False) -> Dict[str, Any]:
        return {
            "task_type": prediction.task_type,
            "dataset_name": prediction.dataset_name,
            "label": prediction.label,
            "score": prediction.score,
            "probability": prediction.probability,
            "model_type": prediction.model_type,
            "zero_shot_score": prediction.details.get("zero_shot_score") if prediction.details else None,
            "confidence_level": prediction.details.get("confidence_level") if prediction.details else None,
            "fallback": fallback,
        }

    def _rag_query(self, state: AgentState) -> str:
        prediction = state.predictions[0] if state.predictions else None
        parts = [
            state.request,
            self._task_type(state),
            self._dataset_name(state),
            self._prediction_meaning(prediction) if prediction else "",
        ]
        if self._task_type(state) == "ddi_prediction":
            for idx, role in enumerate(["drug A", "drug B"]):
                if idx < len(state.molecules) and state.molecules[idx] is not None:
                    molecule = state.molecules[idx]
                    parts.append(f"{role}: {molecule.name or molecule.original_input} {molecule.canonical_smiles}")
            parts.extend(["drug drug interaction", "CYP transporter toxicity adverse event clinical evidence"])
            if state.ddi_pair is not None:
                parts.append(f"fingerprint similarity {state.ddi_pair.fingerprint_similarity}")
                parts.append(f"CYP signals {state.ddi_pair.cyp_related_features}")
                parts.append(f"toxicity overlap {state.ddi_pair.toxicity_overlap_features}")
        else:
            if state.molecules:
                molecule = state.molecules[0]
                parts.append(f"molecule: {molecule.name or molecule.original_input} {molecule.canonical_smiles}")
            parts.append(self._endpoint_rag_terms(self._dataset_name(state), self._task_type(state)))
            if state.single_features:
                bundle = state.single_features[0]
                if bundle is not None:
                    descriptor_text = " ".join(
                        f"{key}={bundle.rdkit_descriptors.get(key)}"
                        for key in ["MolWt", "MolLogP", "TPSA", "NumHDonors", "NumHAcceptors", "QED"]
                        if key in bundle.rdkit_descriptors
                    )
                    parts.append(descriptor_text)
                    if bundle.visual_semantics is not None:
                        parts.append(bundle.visual_semantics.summary)
                    matched_rules = [key for key, value in bundle.llm_rule_features.values.items() if value]
                    if matched_rules:
                        parts.append("matched rules " + " ".join(matched_rules[:8]))
        query = " ".join(str(part) for part in parts if part)
        return " ".join(query.split())[:1200]

    def _endpoint_rag_terms(self, dataset_name: str, task_type: str) -> str:
        dataset = (dataset_name or "").lower()
        terms_by_dataset = {
            "bbbp": "blood brain barrier BBB CNS permeability penetration P-glycoprotein transporter polarity TPSA",
            "bbb_martins": "blood brain barrier BBB CNS permeability penetration P-glycoprotein transporter polarity TPSA",
            "dili": "drug induced liver injury DILI hepatotoxicity hepatic toxicity liver enzyme warning",
            "herg": "hERG KCNH2 QT prolongation cardiotoxicity arrhythmia ion channel",
            "ames": "Ames mutagenicity mutagenic genotoxicity bacterial reverse mutation",
            "clintox": "clinical toxicity adverse event warning contraindication safety",
            "tox21": "toxicity nuclear receptor stress response toxicology assay",
            "sider": "side effect adverse reaction pharmacovigilance",
            "cyp1a2": "CYP1A2 inhibition cytochrome metabolism drug interaction",
            "cyp2c9": "CYP2C9 inhibition cytochrome metabolism drug interaction",
            "cyp2c19": "CYP2C19 inhibition cytochrome metabolism drug interaction",
            "cyp2d6": "CYP2D6 inhibition cytochrome metabolism drug interaction",
            "cyp3a4": "CYP3A4 inhibition cytochrome metabolism drug interaction",
            "hia_hou": "human intestinal absorption HIA permeability oral absorption transporter",
            "pgp_broccatelli": "P-glycoprotein P-gp efflux transporter substrate inhibitor",
            "bioavailability_ma": "oral bioavailability pharmacokinetics absorption first-pass metabolism",
            "caco2_wang": "Caco-2 permeability intestinal absorption efflux transporter",
            "lipophilicity_astrazeneca": "lipophilicity logD logP membrane permeability",
            "solubility_aqsoldb": "aqueous solubility ESOL dissolution polarity hydrogen bonding",
            "hydrationfreeenergy_freesolv": "hydration free energy solvation aqueous thermodynamics",
            "half_life_obach": "half life clearance metabolism pharmacokinetics",
            "clearance_hepatocyte_az": "hepatocyte clearance intrinsic clearance hepatic metabolism",
            "clearance_microsome_az": "microsome clearance intrinsic clearance hepatic metabolism",
            "ld50_zhu": "LD50 acute toxicity mortality dose toxicology",
        }
        terms = []
        for key, value in terms_by_dataset.items():
            if key in dataset:
                terms.append(value)
        if task_type == "admet_prediction":
            terms.append("ADMET pharmacokinetics toxicity metabolism transporter safety")
        elif task_type == "property_prediction":
            terms.append("molecular property structure activity relationship descriptor literature")
        return " ".join(terms)

    def _prediction_meaning(self, prediction: Optional[PredictionResult]) -> str:
        if prediction is None:
            return ""
        details = prediction.details or {}
        if prediction.label == details.get("positive_label"):
            return str(details.get("positive_label_meaning") or prediction.label)
        if prediction.label == details.get("negative_label"):
            return str(details.get("negative_label_meaning") or prediction.label)
        if prediction.task_type == "ddi_prediction":
            return "DDI risk interaction" if prediction.label == 1 else "no DDI risk interaction"
        return str(prediction.label)

    def _attach_single_artifact_paths(self, state: AgentState, role: str, bundle: SingleMoleculeFeatureBundle) -> None:
        prefix = "" if role == "molecule" else f"{role}_"
        if bundle.image is not None:
            state.artifacts[f"{prefix}image_path"] = bundle.image.image_path
            state.artifacts[f"{prefix}grad_cam_path"] = bundle.image.grad_cam_path
        if bundle.smiles is not None:
            state.artifacts[f"{prefix}smiles_features"] = bundle.smiles.to_dict()
        state.artifacts[f"{prefix}feature_schema"] = bundle.feature_schema.to_dict()
        if bundle.visual_semantics is not None:
            state.artifacts[f"{prefix}visual_semantics"] = bundle.visual_semantics.to_dict()
        state.artifacts[f"{prefix}rule_features"] = bundle.llm_rule_features.to_dict()

    def _report_artifacts(self, state: AgentState) -> Dict[str, Any]:
        if self._task_type(state) != "ddi_prediction":
            bundle = self._feature_at(state, 0)
            return {
                "image_path": bundle.image.image_path if bundle.image else None,
                "grad_cam_path": bundle.image.grad_cam_path if bundle.image else None,
                "smiles_features": bundle.smiles.to_dict() if bundle.smiles else None,
                "feature_schema": bundle.feature_schema.to_dict(),
                "visual_semantics": bundle.visual_semantics.to_dict() if bundle.visual_semantics else None,
                "rule_features": bundle.llm_rule_features.to_dict(),
            }

        if state.ddi_pair is None:
            raise ValueError("DDI pair features are missing.")
        pair = state.ddi_pair
        return {
            "drug_a_image_path": pair.drug_a_features.image.image_path if pair.drug_a_features.image else None,
            "drug_b_image_path": pair.drug_b_features.image.image_path if pair.drug_b_features.image else None,
            "drug_a_grad_cam_path": pair.drug_a_features.image.grad_cam_path if pair.drug_a_features.image else None,
            "drug_b_grad_cam_path": pair.drug_b_features.image.grad_cam_path if pair.drug_b_features.image else None,
            "drug_a_smiles_features": pair.drug_a_features.smiles.to_dict() if pair.drug_a_features.smiles else None,
            "drug_b_smiles_features": pair.drug_b_features.smiles.to_dict() if pair.drug_b_features.smiles else None,
            "feature_schema": pair.feature_schema.to_dict(),
            "fingerprint_similarity": pair.fingerprint_similarity,
            "admet_risk_features": pair.admet_risk_features,
            "cyp_related_features": pair.cyp_related_features,
            "toxicity_overlap_features": pair.toxicity_overlap_features,
            "llm_ddi_mechanism_features": pair.llm_ddi_mechanism_features,
        }

    def _confidence_signals(self, state: AgentState, prediction: PredictionResult) -> List[Dict[str, Any]]:
        signals: List[Dict[str, Any]] = [
            {"name": "model_type", "value": prediction.model_type},
            {"name": "evidence_count", "value": len(state.evidence)},
        ]
        if prediction.probability is not None:
            margin = abs(float(prediction.probability) - 0.5)
            signals.append({"name": "probability_margin_from_threshold", "value": round(margin, 6)})
        if prediction.model_type == ZERO_SHOT_MODEL_TYPE:
            signals.append({"name": "zero_shot_score", "value": prediction.details.get("zero_shot_score")})
            signals.append({"name": "zero_shot_confidence_level", "value": prediction.details.get("confidence_level")})
            signals.append({"name": "zero_shot_contribution_count", "value": len(prediction.details.get("evidence_contributions", []))})
        if "prediction_confidence" in state.metadata:
            signals.append({"name": "low_confidence_prediction", "value": state.metadata["prediction_confidence"].get("low_confidence")})
        if state.rules:
            signals.append({"name": "rule_count", "value": len(state.rules)})
        if state.single_features:
            first = state.single_features[0]
            if first is not None:
                if first.image is not None:
                    signals.append({"name": "image_encoder", "value": first.image.encoder_name})
                if first.smiles is not None:
                    signals.append({"name": "smiles_encoder", "value": first.smiles.encoder_name})
                if first.visual_semantics is not None:
                    signals.append({"name": "visual_semantic_provider", "value": first.visual_semantics.provider})
        if state.ddi_pair is not None:
            signals.append({"name": "morgan_tanimoto", "value": state.ddi_pair.fingerprint_similarity.get("morgan_tanimoto")})
        return signals

    def _reasoning_summary(self, state: AgentState, prediction: PredictionResult, signals: List[Dict[str, Any]]) -> str:
        task = self._task_type(state)
        if prediction.model_type == ZERO_SHOT_MODEL_TYPE:
            confidence = prediction.details.get("confidence_level", "unknown")
            model_note = f"model={prediction.model_type} (non-calibrated zero-shot score, confidence={confidence})"
        else:
            model_note = f"model={prediction.model_type}"
        evidence_note = f"evidence_items={len(state.evidence)}"
        fallback_note = f"fallbacks={len(state.reasoning.get('fallbacks', []))}"
        if task == "ddi_prediction":
            return f"DDI decision used {model_note}, {evidence_note}, {fallback_note}, and pair-level similarity/ADMET mechanism cues."
        dataset = self._dataset_name(state)
        return f"{dataset} decision used {model_note}, {evidence_note}, {fallback_note}, molecular rules, image/visual features, and RDKit descriptors."

    def _state_summary(self, state: AgentState) -> Dict[str, Any]:
        return {
            "request": state.request,
            "task_type": self._task_type(state),
            "dataset_name": self._dataset_name(state),
            "n_molecules": len([mol for mol in state.molecules if mol is not None]),
            "n_rules": len(state.rules),
            "n_predictions": len(state.predictions),
            "n_evidence_items": len(state.evidence),
            "n_warnings": len(state.warnings),
            "planner": state.plan.metadata.get("planner", {}) if state.plan else {},
            "prediction_confidence": state.metadata.get("prediction_confidence"),
            "step_statuses": {step.step_id: step.status for step in state.plan.steps} if state.plan else {},
            "artifacts": state.artifacts,
        }
