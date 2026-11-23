from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Protocol, Sequence, Set

from agentm.llm.provider import LLMClient, build_llm_client


class ParsedInputLike(Protocol):
    task_type: str
    dataset_name: str
    molecules: List[str]
    raw_request: str


class AgentPlanStepLike(Protocol):
    step_id: str
    tool_name: str
    description: str
    params: Dict[str, Any]
    depends_on: List[str]
    fallback_tool: Optional[str]
    condition: Optional[str]


TASK_TYPES = {"property_prediction", "admet_prediction", "ddi_prediction"}
DATASET_NAMES = {
    "BBBP",
    "BACE",
    "HIV",
    "ESOL",
    "FreeSolv",
    "Lipophilicity",
    "ClinTox",
    "Tox21",
    "SIDER",
    "CYP1A2 inhibition",
    "CYP2C9 inhibition",
    "CYP2C19 inhibition",
    "CYP2D6 inhibition",
    "CYP3A4 inhibition",
    "hERG",
    "Ames",
    "DILI",
    "DDI",
}


class LLMDecisionAgent:
    """LLM-assisted routing and plan critique for the custom agent controller.

    This layer is deliberately advisory. It can propose task routing, molecule
    mentions, evidence policy, and MLLM usage, but the controller still validates
    the plan and falls back to deterministic behavior when the LLM is unavailable
    or returns invalid JSON.
    """

    def __init__(
        self,
        *,
        use_llm: bool = False,
        profile_name: Optional[str] = None,
        llm_client: Optional[LLMClient] = None,
        enable_routing: bool = True,
        enable_plan_critic: bool = True,
    ) -> None:
        env_enabled = os.getenv("AGENTM_ENABLE_LLM_DECISION", "").strip().lower() in {"1", "true", "yes", "on"}
        self.use_llm = bool(use_llm or env_enabled)
        self.llm_client = llm_client or (build_llm_client(profile_name=profile_name) if self.use_llm else None)
        self.enable_routing = bool(enable_routing)
        self.enable_plan_critic = bool(enable_plan_critic)

    def decide_after_parse(
        self,
        parsed: ParsedInputLike,
        *,
        context: Dict[str, Any],
        explicit_task_type: Optional[str],
        explicit_dataset_name: Optional[str],
    ) -> Dict[str, Any]:
        local = self._local_decision(parsed, context)
        metadata: Dict[str, Any] = {
            "decision_mode": "local_policy",
            "llm_decision_requested": self.use_llm,
            "llm_decision_available": bool(self.llm_client and self.llm_client.available),
            "validation_status": "not_needed",
            "explicit_task_type": explicit_task_type,
            "explicit_dataset_name": explicit_dataset_name,
            **local,
        }
        if not self.enable_routing:
            metadata["decision_mode"] = "disabled"
            return metadata
        if not self.use_llm:
            return metadata
        if self.llm_client is None or not self.llm_client.available:
            metadata.update(
                {
                    "decision_mode": "local_policy_after_llm_unavailable",
                    "validation_status": "llm_unavailable",
                    "llm_warning": "LLM decision layer is unavailable; local routing policy was used.",
                }
            )
            return metadata

        try:
            payload = self._llm_route(parsed, context)
            validated = self._validate_route_payload(payload, parsed)
            metadata.update(
                {
                    "decision_mode": "llm_route_policy",
                    "validation_status": "valid",
                    "llm_profile": self.llm_client.config.profile,
                    "llm_model": self.llm_client.config.model,
                    **validated,
                }
            )
            return metadata
        except Exception as exc:
            metadata.update(
                {
                    "decision_mode": "local_policy_after_llm_decision_error",
                    "validation_status": "failed",
                    "llm_error": str(exc),
                    "llm_warning": f"LLM decision layer failed; local routing policy was used. Reason: {exc}",
                }
            )
            return metadata

    def critique_plan(
        self,
        parsed: ParsedInputLike,
        *,
        context: Dict[str, Any],
        steps: Sequence[AgentPlanStepLike],
        allowed_tools: Set[str],
    ) -> Dict[str, Any]:
        local = self._local_plan_critique(parsed, steps, allowed_tools)
        metadata: Dict[str, Any] = {
            "critic_mode": "local_policy",
            "llm_critic_requested": self.use_llm,
            "llm_critic_available": bool(self.llm_client and self.llm_client.available),
            "validation_status": "not_needed",
            **local,
        }
        if not self.enable_plan_critic:
            metadata["critic_mode"] = "disabled"
            return metadata
        if not self.use_llm:
            return metadata
        if self.llm_client is None or not self.llm_client.available:
            metadata.update(
                {
                    "critic_mode": "local_policy_after_llm_unavailable",
                    "validation_status": "llm_unavailable",
                    "critic_warning": "LLM plan critic is unavailable; local plan checks were used.",
                }
            )
            return metadata

        try:
            payload = self._llm_plan_critique(parsed, context, steps, allowed_tools)
            validated = self._validate_critique_payload(payload)
            metadata.update(
                {
                    "critic_mode": "llm_plan_critic",
                    "validation_status": "valid",
                    "llm_profile": self.llm_client.config.profile,
                    "llm_model": self.llm_client.config.model,
                    **validated,
                }
            )
            return metadata
        except Exception as exc:
            metadata.update(
                {
                    "critic_mode": "local_policy_after_llm_critic_error",
                    "validation_status": "failed",
                    "llm_error": str(exc),
                    "critic_warning": f"LLM plan critic failed; local plan checks were used. Reason: {exc}",
                }
            )
            return metadata

    def _llm_route(self, parsed: ParsedInputLike, context: Dict[str, Any]) -> Dict[str, Any]:
        assert self.llm_client is not None
        prompt = f"""
Route this molecular prediction request and recommend execution policy.

Current deterministic parse:
{json.dumps(_parsed_dict(parsed), indent=2, ensure_ascii=False)}

Runtime context:
{json.dumps(_compact_context(context), indent=2, ensure_ascii=False, sort_keys=True)}

Allowed task_type values: {sorted(TASK_TYPES)}
Allowed dataset_name values: {sorted(DATASET_NAMES)}

Return JSON only:
{{
  "task_type": "property_prediction|admet_prediction|ddi_prediction",
  "dataset_name": "one allowed dataset",
  "molecules": ["mention 1", "mention 2 if DDI"],
  "route_confidence": 0.0,
  "needs_clarification": false,
  "clarification_question": "",
  "use_mllm_visual_semantic": false,
  "use_rag_evidence": true,
  "use_external_evidence": false,
  "prefer_zero_shot": false,
  "reasoning_depth": "brief|standard|case_study",
  "rationale": "short reason",
  "warnings": []
}}

Rules:
- Do not invent SMILES strings.
- Molecules should be names or SMILES explicitly present in the request when possible.
- Prefer DDI when the request asks whether two drugs can be used together.
- Prefer ADMET for toxicity, DILI, hERG, Ames, CYP, side-effect, metabolism, or safety questions.
- Prefer property_prediction for BBBP, solubility, lipophilicity, BACE, HIV, ESOL, FreeSolv.
""".strip()
        response = self.llm_client.chat_json(
            [
                {"role": "system", "content": "You are a cautious molecular-agent routing policy. Return JSON only."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
        )
        return _parse_json_object(response)

    def _llm_plan_critique(
        self,
        parsed: ParsedInputLike,
        context: Dict[str, Any],
        steps: Sequence[AgentPlanStepLike],
        allowed_tools: Set[str],
    ) -> Dict[str, Any]:
        assert self.llm_client is not None
        plan = [_step_dict(step) for step in steps]
        prompt = f"""
Critique this molecular-agent execution plan. Do not rewrite code. Identify missing safety, evidence, trace, or fallback issues.

Parsed request:
{json.dumps(_parsed_dict(parsed), indent=2, ensure_ascii=False)}

Plan:
{json.dumps(plan, indent=2, ensure_ascii=False)}

Allowed tools:
{json.dumps(sorted(allowed_tools), indent=2)}

Context:
{json.dumps(_compact_context(context), indent=2, ensure_ascii=False, sort_keys=True)}

Return JSON only:
{{
  "status": "pass|warn|fail",
  "rationale": "short critique",
  "missing_steps": [],
  "unsafe_steps": [],
  "recommended_changes": [],
  "requires_fallback": false
}}
""".strip()
        response = self.llm_client.chat_json(
            [
                {"role": "system", "content": "You are a strict scientific workflow plan critic. Return JSON only."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
        )
        return _parse_json_object(response)

    def _local_decision(self, parsed: ParsedInputLike, context: Dict[str, Any]) -> Dict[str, Any]:
        request = str(context.get("request") or parsed.raw_request or "").lower()
        asks_visual = any(token in request for token in ["mllm", "multimodal", "visual", "image", "图像", "视觉", "结构", "解释"])
        asks_evidence = any(token in request for token in ["evidence", "literature", "pubmed", "证据", "文献", "依据", "为什么"])
        asks_case = any(token in request for token in ["case", "case study", "详细", "完整", "报告"])
        use_external = any(token in request for token in ["pubchem", "dailymed", "pubmed", "外部", "数据库"])
        return {
            "route": {
                "task_type": parsed.task_type,
                "dataset_name": parsed.dataset_name,
                "molecules": list(parsed.molecules),
                "route_confidence": 0.65,
                "needs_clarification": not bool(parsed.molecules),
                "clarification_question": "Please provide a drug name, SMILES string, or drug pair." if not parsed.molecules else "",
                "rationale": "Local keyword and molecule-pattern routing.",
            },
            "recommendations": {
                "use_mllm_visual_semantic": bool(context.get("allow_planner_mllm") and asks_visual),
                "use_rag_evidence": bool(context.get("use_rag_evidence", True) or asks_evidence),
                "use_external_evidence": bool(context.get("use_external_evidence") or use_external),
                "prefer_zero_shot": not bool(context.get("model_available")),
                "reasoning_depth": "case_study" if asks_case else "standard",
                "extra_evidence_on_low_confidence": True,
            },
            "warnings": [],
        }

    def _validate_route_payload(self, payload: Dict[str, Any], parsed: ParsedInputLike) -> Dict[str, Any]:
        task_type = str(payload.get("task_type") or parsed.task_type).strip()
        dataset_name = str(payload.get("dataset_name") or parsed.dataset_name).strip()
        if task_type not in TASK_TYPES:
            raise ValueError(f"Unsupported LLM route task_type: {task_type}")
        if dataset_name not in DATASET_NAMES:
            raise ValueError(f"Unsupported LLM route dataset_name: {dataset_name}")
        molecules = payload.get("molecules")
        if not isinstance(molecules, list):
            molecules = list(parsed.molecules)
        cleaned_molecules = [str(item).strip() for item in molecules if str(item).strip()]
        if task_type == "ddi_prediction":
            cleaned_molecules = cleaned_molecules[:2]
        elif cleaned_molecules:
            cleaned_molecules = cleaned_molecules[:1]
        confidence = _float_in_range(payload.get("route_confidence"), default=0.5)
        recommendations = {
            "use_mllm_visual_semantic": bool(payload.get("use_mllm_visual_semantic", False)),
            "use_rag_evidence": bool(payload.get("use_rag_evidence", True)),
            "use_external_evidence": bool(payload.get("use_external_evidence", False)),
            "prefer_zero_shot": bool(payload.get("prefer_zero_shot", False)),
            "reasoning_depth": _choice(payload.get("reasoning_depth"), {"brief", "standard", "case_study"}, "standard"),
            "extra_evidence_on_low_confidence": True,
        }
        return {
            "route": {
                "task_type": task_type,
                "dataset_name": dataset_name,
                "molecules": cleaned_molecules,
                "route_confidence": confidence,
                "needs_clarification": bool(payload.get("needs_clarification", False)),
                "clarification_question": str(payload.get("clarification_question") or "").strip(),
                "rationale": str(payload.get("rationale") or "").strip(),
            },
            "recommendations": recommendations,
            "warnings": [str(item) for item in payload.get("warnings", []) if str(item).strip()]
            if isinstance(payload.get("warnings"), list)
            else [],
        }

    def _local_plan_critique(
        self,
        parsed: ParsedInputLike,
        steps: Sequence[AgentPlanStepLike],
        allowed_tools: Set[str],
    ) -> Dict[str, Any]:
        step_ids = [str(getattr(step, "step_id", "")) for step in steps]
        tool_names = [str(getattr(step, "tool_name", "")) for step in steps]
        unsafe_steps = [tool for tool in tool_names if tool not in allowed_tools]
        missing_steps: List[str] = []
        if "build_report" not in tool_names:
            missing_steps.append("build_report")
        if "retrieve_rag_evidence" not in tool_names:
            missing_steps.append("retrieve_rag_evidence")
        if parsed.task_type == "ddi_prediction" and "build_ddi_pair" not in tool_names:
            missing_steps.append("build_ddi_pair")
        status = "pass" if not missing_steps and not unsafe_steps else "warn"
        return {
            "status": status,
            "rationale": "Local plan audit checked tool allow-list and required evidence/report steps.",
            "missing_steps": missing_steps,
            "unsafe_steps": unsafe_steps,
            "recommended_changes": [],
            "requires_fallback": False,
            "n_steps_checked": len(step_ids),
        }

    def _validate_critique_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        status = _choice(payload.get("status"), {"pass", "warn", "fail"}, "warn")
        return {
            "status": status,
            "rationale": str(payload.get("rationale") or "").strip(),
            "missing_steps": _string_list(payload.get("missing_steps")),
            "unsafe_steps": _string_list(payload.get("unsafe_steps")),
            "recommended_changes": _string_list(payload.get("recommended_changes")),
            "requires_fallback": bool(payload.get("requires_fallback", False)),
        }


def _parsed_dict(parsed: ParsedInputLike) -> Dict[str, Any]:
    return {
        "task_type": parsed.task_type,
        "dataset_name": parsed.dataset_name,
        "molecules": list(parsed.molecules),
        "raw_request": parsed.raw_request,
    }


def _step_dict(step: AgentPlanStepLike) -> Dict[str, Any]:
    to_dict = getattr(step, "to_dict", None)
    if callable(to_dict):
        return to_dict()
    return {
        "step_id": getattr(step, "step_id", ""),
        "tool_name": getattr(step, "tool_name", ""),
        "description": getattr(step, "description", ""),
        "params": getattr(step, "params", {}),
        "depends_on": getattr(step, "depends_on", []),
        "fallback_tool": getattr(step, "fallback_tool", None),
        "condition": getattr(step, "condition", None),
    }


def _compact_context(context: Dict[str, Any]) -> Dict[str, Any]:
    keep = {
        "request",
        "feature_set",
        "model_available",
        "trained_rule_count",
        "imagemol_encoder",
        "smiles_encoder",
        "use_smiles_encoder",
        "mllm_currently_enabled",
        "mllm_profile",
        "allow_planner_mllm",
        "confidence_margin_threshold",
        "use_rag_evidence",
        "use_external_evidence",
        "rag_backend",
        "rag_top_k",
    }
    return {key: context.get(key) for key in keep if key in context}


def _parse_json_object(response: str) -> Dict[str, Any]:
    text = (response or "").strip()
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
        raise ValueError("LLM decision response must be a JSON object.")
    return data


def _float_in_range(value: Any, *, default: float) -> float:
    try:
        parsed = float(value)
    except Exception:
        return default
    return max(0.0, min(1.0, parsed))


def _choice(value: Any, allowed: Set[str], default: str) -> str:
    text = str(value or "").strip()
    return text if text in allowed else default


def _string_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item).strip()]
