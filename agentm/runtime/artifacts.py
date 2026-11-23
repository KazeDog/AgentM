from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from agentm.runtime.compat import to_jsonable, write_json, write_text
from agentm.runtime.prediction_semantics import is_multiclass_ddi_prediction, prediction_meaning


def _previous_run_report_metadata(previous: Any) -> Dict[str, Any]:
    """Keep audit continuity without embedding stale scientific objects.

    The full previous context remains available in ``global_context.json``.
    Copying it into the active report duplicated predictions, evidence, and
    citations from an earlier turn and made them look like current state.
    """

    if not isinstance(previous, dict) or not previous:
        return {}
    objectives = previous.get("objective_status", {}) or {}
    verifier = previous.get("verifier_report", {}) or {}
    return {
        "loaded": True,
        "task_type": previous.get("task_type"),
        "dataset_name": previous.get("dataset_name"),
        "prediction_count": len(previous.get("predictions", []) or previous.get("reaction_predictions", []) or []),
        "evidence_count": len(previous.get("evidence", []) or []),
        "objective_status": objectives.get("status") if isinstance(objectives, dict) else None,
        "verifier_status": ((verifier.get("summary", {}) or {}).get("status") if isinstance(verifier, dict) else None),
    }


def report_payload(context: Dict[str, Any]) -> Dict[str, Any]:
    evidence = _unique_evidence(context.get("evidence", []))
    warnings = _unique_strings(context.get("warnings", []))
    metadata = context.get("metadata", {}) or {}
    evidence_status = {
        "retrieval_outcome": metadata.get("retrieval_outcome"),
        "direct_evidence_available": metadata.get("direct_evidence_available"),
        "retained_evidence_count": len(evidence),
        "citation_count": len(context.get("citation_map", []) or []),
        "retrieval_backend": metadata.get("retrieval_backend_used"),
        "real_rag_execution_valid": metadata.get("real_rag_execution_valid"),
        "explicit_absence": bool(
            metadata.get("retrieval_outcome") == "no_direct_evidence"
            and metadata.get("direct_evidence_available") is False
            and not evidence
        ),
    }
    artifacts = dict(context.get("artifacts", {}) or {})
    artifacts.update(
        {
            "task_graph": context.get("task_graph", {}),
            "orchestrator_plan": context.get("orchestrator_plan", {}),
            "llm_plan_proposals": context.get("llm_plan_proposals", []),
            "llm_retry_trace": context.get("llm_retry_trace", []),
            "plan_revision_trace": context.get("plan_revision_trace", []),
            "citation_map": context.get("citation_map", []),
            "verifier_report": context.get("verifier_report", {}),
            "retrieval_trace": context.get("retrieval_trace", []),
            "reflection": context.get("reflection", {}),
            "resource_resolution": context.get("resource_resolution", {}),
            "objective_status": context.get("objective_status", {}),
            "previous_run_loaded": bool(context.get("previous_run_context")),
        }
    )
    return to_jsonable(
        {
            "request": context.get("request"),
            "task_type": context.get("task_type"),
            "dataset_name": context.get("dataset_name"),
            "molecules": context.get("molecules", []),
            "reaction": context.get("reaction", {}),
            "reaction_profile": context.get("reaction_profile", {}),
            "reaction_predictions": context.get("reaction_predictions", []),
            "predictions": context.get("predictions", []),
            "evidence": evidence,
            "evidence_status": evidence_status,
            "explanation": _explanation(context),
            "artifacts": artifacts,
            "warnings": warnings,
            "trace": context.get("multi_agent_trace", []),
            "reasoning": context.get("reasoning", {}),
            "reflection": context.get("reflection", {}),
            "previous_run_context": _previous_run_report_metadata(context.get("previous_run_context", {})),
            "agent_messages": context.get("agent_messages", []),
            "tool_calls": context.get("tool_calls", []),
            "llm_plan_proposals": context.get("llm_plan_proposals", []),
            "llm_retry_trace": context.get("llm_retry_trace", []),
            "plan_revision_trace": context.get("plan_revision_trace", []),
            "retrieval_trace": context.get("retrieval_trace", []),
            "citation_map": context.get("citation_map", []),
            "verifier_report": context.get("verifier_report", {}),
            "resource_resolution": context.get("resource_resolution", {}),
            "objective_status": context.get("objective_status", {}),
            "safety": {
                "safe_abstention_required": bool(context.get("metadata", {}).get("safe_abstention_required")),
                "safe_abstention_completed": bool(context.get("metadata", {}).get("safe_abstention_completed")),
                "prediction_failure": context.get("metadata", {}).get("prediction_failure", {}),
            },
            "multi_agent": {
                "runtime": "agentm_guarded_multi_agent",
                "mode": context.get("metadata", {}).get("mode"),
                "orchestrator": "FROGENT-style guarded OrchestratorAgent",
                "orchestrator_mode": context.get("metadata", {}).get("orchestrator_mode"),
                "planner_type": context.get("metadata", {}).get("planner_type"),
                "fallbacks_enabled": context.get("metadata", {}).get("enable_fallbacks"),
                "agents": [
                    "MoleculeAgent",
                    "RetrieveAgent",
                    "GaugeAgent",
                    "ReactionAgent",
                    "ReasoningAgent",
                    "VerifierAgent",
                    "ReflectionAgent",
                    "ReporterAgent",
                ],
                "rounds": len(context.get("tool_calls", []) or []),
                "loop_status": context.get("metadata", {}).get("loop_status"),
                "llm_orchestrator_enabled": context.get("metadata", {}).get("use_llm_orchestrator"),
                "llm_reasoning_enabled": context.get("metadata", {}).get("use_llm_reasoning"),
                "reasoning_mode": context.get("metadata", {}).get("reasoning_mode"),
                "llm_orchestrator_max_retries": context.get("metadata", {}).get("llm_orchestrator_max_retries"),
            },
        }
    )


def write_artifacts(context: Dict[str, Any], output_dir: str | Path) -> Dict[str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    task_type = context.get("task_type")
    if str(task_type or "").startswith("reaction_"):
        compatibility_name = "reaction_report.json"
    else:
        compatibility_name = "ddi_report.json" if task_type == "ddi_prediction" else "single_molecule_report.json"
    report = context.get("final_report") or report_payload(context)
    paths = {
        "multi_agent_report_path": str(write_json(out / "multi_agent_report.json", report)),
        "compat_report_path": str(write_json(out / compatibility_name, context.get("compatible_report") or report)),
        "reaction_report_path": str(write_json(out / "reaction_report.json", report)) if str(task_type or "").startswith("reaction_") else "",
        "reaction_profile_path": str(write_json(out / "reaction_profile.json", context.get("reaction_profile", {}))) if str(task_type or "").startswith("reaction_") else "",
        "reaction_prediction_path": str(write_json(out / "reaction_prediction.json", context.get("reaction_predictions", []))) if str(task_type or "").startswith("reaction_") else "",
        "reaction_condition_evidence_path": str(write_json(out / "reaction_condition_evidence.json", context.get("artifacts", {}).get("reaction_condition_evidence", []))) if str(task_type or "").startswith("reaction_") else "",
        "reaction_template_evidence_path": str(write_json(out / "reaction_template_evidence.json", context.get("artifacts", {}).get("reaction_template_evidence", []))) if str(task_type or "").startswith("reaction_") else "",
        "reaction_resource_resolution_path": str(write_json(out / "reaction_resource_resolution.json", context.get("reaction_resource_resolution", {}))) if str(task_type or "").startswith("reaction_") else "",
        "global_context_path": str(write_json(out / "global_context.json", context)),
        "orchestrator_plan_path": str(write_json(out / "orchestrator_plan.json", context.get("orchestrator_plan", {}))),
        "task_graph_path": str(write_json(out / "task_graph.json", context.get("task_graph", {}))),
        "llm_plan_proposals_path": str(write_json(out / "llm_plan_proposals.json", context.get("llm_plan_proposals", []))),
        "llm_retry_trace_path": str(write_json(out / "llm_retry_trace.json", context.get("llm_retry_trace", []))),
        "plan_revision_trace_path": str(write_json(out / "plan_revision_trace.json", context.get("plan_revision_trace", []))),
        "multi_agent_trace_path": str(write_json(out / "multi_agent_trace.json", context.get("multi_agent_trace", []))),
        "agent_messages_path": str(write_json(out / "agent_messages.json", context.get("agent_messages", []))),
        "tool_calls_path": str(write_json(out / "tool_calls.json", context.get("tool_calls", []))),
        "retrieval_trace_path": str(write_json(out / "retrieval_trace.json", context.get("retrieval_trace", []))),
        "citation_map_path": str(write_json(out / "citation_map.json", context.get("citation_map", []))),
        "verifier_report_path": str(write_json(out / "verifier_report.json", context.get("verifier_report", {}))),
        "resource_resolution_path": str(write_json(out / "resource_resolution.json", context.get("resource_resolution", {}))),
        "objective_status_path": str(write_json(out / "objective_status.json", context.get("objective_status", {}))),
        "reflection_path": str(write_json(out / "reflection.json", context.get("reflection", {}))),
        "reproducibility_log_path": str(write_json(out / "reproducibility.json", reproducibility_payload(context))),
        "markdown_report_path": str(write_text(out / "report.md", markdown_report(report))),
    }
    return paths


def reproducibility_payload(context: Dict[str, Any]) -> Dict[str, Any]:
    metadata = context.get("metadata", {}) or {}
    return {
        "runtime": "agentm_guarded_multi_agent",
        "mode": metadata.get("mode"),
        "retrieval_backend_requested": metadata.get("retrieval_backend_requested"),
        "retrieval_backend_used": metadata.get("retrieval_backend_used"),
        "max_rounds": metadata.get("max_rounds"),
        "llm_profile": metadata.get("llm_profile"),
        "use_llm_orchestrator": metadata.get("use_llm_orchestrator"),
        "completed_actions": metadata.get("completed_actions", []),
        "planner_type": metadata.get("planner_type"),
        "orchestrator_mode": metadata.get("orchestrator_mode"),
        "enable_fallbacks": metadata.get("enable_fallbacks"),
        "auto_model": metadata.get("auto_model"),
        "require_real_model": metadata.get("require_real_model"),
        "require_real_rag": metadata.get("require_real_rag"),
        "require_llm_orchestrator": metadata.get("require_llm_orchestrator"),
        "llm_orchestrator_max_retries": metadata.get("llm_orchestrator_max_retries"),
        "llm_orchestrator_retry_backoff_seconds": metadata.get("llm_orchestrator_retry_backoff_seconds"),
        "use_llm_reasoning": metadata.get("use_llm_reasoning"),
        "require_llm_reasoning": metadata.get("require_llm_reasoning"),
        "reasoning_mode": metadata.get("reasoning_mode"),
        "safe_abstention_required": metadata.get("safe_abstention_required"),
        "safe_abstention_completed": metadata.get("safe_abstention_completed"),
        "prediction_failure": metadata.get("prediction_failure", {}),
        "min_evidence": metadata.get("min_evidence"),
        "resource_resolution": context.get("resource_resolution", {}),
        "objective_status": context.get("objective_status", {}),
        "warnings": _unique_strings(context.get("warnings", [])),
        "fallbacks": context.get("fallbacks", []),
        "reflection": context.get("reflection", {}),
        "previous_run_loaded": bool(context.get("previous_run_context")),
    }


def markdown_report(report: Dict[str, Any]) -> str:
    if str(report.get("task_type") or "").startswith("reaction_"):
        return _reaction_markdown_report(report)
    prediction = (report.get("predictions") or [{}])[0]
    multi_agent = report.get("multi_agent") or {}
    prediction_label = prediction.get("label")
    prediction_score = prediction.get("score")
    prediction_model = prediction.get("model_type")
    if report.get("task_type") in {"general_drug_profile", "out_of_domain", "missing_entity"} and not report.get("predictions"):
        prediction_label = "not requested"
        prediction_score = "not requested"
        prediction_model = "not requested"
    lines = [
        "# AgentM Multi-Agent Report",
        "",
        f"**Request:** {report.get('request', '')}",
        f"**Task:** {report.get('task_type', '')}",
        f"**Dataset:** {prediction.get('dataset_name') or report.get('dataset_name', '')}",
        f"**Prediction:** {prediction_label}",
        f"**Score:** {prediction_score}",
        f"**Model:** {prediction_model}",
        f"**Rounds:** {multi_agent.get('rounds')}",
        "",
        "## Explanation",
        str(report.get("explanation") or ""),
        "",
        "## Evidence",
    ]
    evidence = report.get("evidence") or []
    if evidence:
        for item in evidence[:12]:
            lines.append(f"- `{item.get('source', 'unknown')}`: {item.get('claim') or item.get('snippet') or str(item)[:240]}")
    else:
        lines.append("- No evidence items were retrieved.")
    warnings = report.get("warnings") or []
    if warnings:
        lines.extend(["", "## Warnings"])
        lines.extend(f"- {warning}" for warning in warnings)
    return "\n".join(lines).rstrip() + "\n"


def _reaction_markdown_report(report: Dict[str, Any]) -> str:
    prediction = (report.get("reaction_predictions") or report.get("predictions") or [{}])[-1]
    reaction = report.get("reaction") or {}
    lines = [
        "# AgentM Reaction Report",
        "",
        f"**Request:** {report.get('request', '')}",
        f"**Task:** {report.get('task_type', '')}",
        f"**Reaction:** {reaction.get('reaction_smiles', '')}",
        f"**Prediction:** {prediction.get('label', 'not requested')}",
        f"**Model:** {prediction.get('model_type', 'not requested')}",
        "",
        "## Explanation",
        str(report.get("explanation") or ""),
        "",
        "## Reaction Evidence",
    ]
    evidence = report.get("evidence") or []
    if evidence:
        for item in evidence[:12]:
            lines.append(f"- `{item.get('source', 'unknown')}`: {item.get('claim') or item.get('snippet') or str(item)[:240]}")
    else:
        lines.append("- No reaction evidence items were retrieved.")
    warnings = report.get("warnings") or []
    if warnings:
        lines.extend(["", "## Warnings"])
        lines.extend(f"- {warning}" for warning in warnings)
    return "\n".join(lines).rstrip() + "\n"


def _explanation(context: Dict[str, Any]) -> str:
    reasoning = context.get("reasoning", {}) or {}
    if reasoning.get("structured_explanation"):
        return reasoning["structured_explanation"]
    prediction = (context.get("predictions") or [None])[0]
    if prediction is None:
        return _no_prediction_explanation(context)
    label = getattr(prediction, "label", None) if not isinstance(prediction, dict) else prediction.get("label")
    model_type = getattr(prediction, "model_type", None) if not isinstance(prediction, dict) else prediction.get("model_type")
    details = getattr(prediction, "details", {}) if not isinstance(prediction, dict) else prediction.get("details", {})
    details = details if isinstance(details, dict) else {}
    if is_multiclass_ddi_prediction(prediction):
        names = [
            str(item.get("name") or item.get("original_input") or "")
            for item in context.get("molecules", []) or []
            if isinstance(item, dict)
        ]
        label_text = prediction_meaning(prediction, drug_names=names)
    elif label == details.get("positive_label"):
        label_text = details.get("positive_label_meaning") or "positive"
    elif label == details.get("negative_label"):
        label_text = details.get("negative_label_meaning") or "negative"
    else:
        label_text = f"label {label}"
    evidence_count = len(_unique_evidence(context.get("evidence", []) or []))
    base = (
        f"AgentM produced a GaugeAgent-grounded prediction interpreted as {label_text} "
        f"using `{model_type}` and considered {evidence_count} unique evidence items. "
        "Prediction grounding, citation grounding, and report schema were checked by VerifierAgent."
    )
    if str(model_type) == "zero_shot_rule_semantic_fusion":
        base += " No task-specific supervised model was used for this run; treat the result as a zero-shot screening signal, not a calibrated model probability."
    return base


def _no_prediction_explanation(context: Dict[str, Any]) -> str:
    request = str(context.get("request") or "")
    chinese = _looks_chinese(request)
    task_type = context.get("task_type")
    gate = context.get("metadata", {}).get("intent_gate", {}) or {}
    if task_type == "missing_entity":
        if chinese:
            return (
                "本次没有进入 ADMET、DDI 或化学反应预测流程，因为请求像是药物/分子相关问题，但没有给出可解析的具体药物名、化合物名或 SMILES。"
                "请补充明确实体，例如“阿司匹林 BBBP”“阿司匹林和华法林相互作用”或一个标准 SMILES。"
            )
        return (
            "The workflow did not enter ADMET, DDI, or reaction prediction because the request appears scientific but does not provide a resolvable drug name, compound name, or SMILES. "
            "Please provide a concrete entity, for example `aspirin BBBP`, an interaction pair, or a standard SMILES."
        )
    if task_type == "out_of_domain":
        suggestion = str(gate.get("suggestion") or "").strip()
        if chinese:
            return (
                "本次没有进入 ADMET、DDI 或化学反应预测流程，因为输入没有明确的药物、分子、SMILES、ADMET/DDI 端点或反应任务。"
                "请改写为具体科学任务，例如“阿司匹林 BBBP”“阿司匹林和华法林相互作用”或“判断这个反应类型”。"
            )
        return (
            "The workflow did not enter ADMET, DDI, or reaction prediction because the input does not specify a drug, molecule, SMILES, ADMET/DDI endpoint, or reaction task. "
            + (suggestion or "Please rewrite it as a concrete scientific task, such as `aspirin BBBP`, a drug-interaction pair, or a reaction classification request.")
        )
    molecules = [item for item in to_jsonable(context.get("molecules", []) or []) if isinstance(item, dict)]
    invalid = [item for item in molecules if not item.get("is_valid")]
    warnings = [str(item) for item in context.get("warnings", []) or []]
    if invalid:
        names = ", ".join(str(item.get("original_input") or item.get("name") or "unknown") for item in invalid)
        general_profile = context.get("task_type") == "general_drug_profile"
        if chinese:
            if general_profile:
                return (
                    f"本次没有生成可靠药物概览，因为系统未能把输入中的药物名解析为有效分子结构：{names}。"
                    "没有有效 SMILES 时，AgentM 不能构建分子特征，也不会让 LLM 直接编造药物性质或预测结论。"
                    "请检查药物拼写，或改用英文通用名/标准 SMILES；如果需要解析不在本地词典中的药物，请在 Custom 模式开启 network lookup。"
                )
            return (
                f"本次没有生成可靠预测，因为系统未能把输入中的药物名解析为有效分子结构：{names}。"
                "没有有效 SMILES 时，AgentM 不能构建分子特征，也不会让 LLM 直接编造预测结果。"
                "请检查药物拼写，或改用英文通用名/标准 SMILES；如果需要解析不在本地词典中的药物，请在 Custom 模式开启 network lookup。"
            )
        if general_profile:
            return (
                f"No reliable drug overview was generated because the molecule name could not be resolved to a valid structure: {names}. "
                "Without a valid SMILES, AgentM cannot build molecular features, and the guard prevents the LLM from inventing drug properties or prediction conclusions. "
                "Please check the spelling, provide a standard English drug name or SMILES, or enable network lookup in Custom mode."
            )
        return (
            f"No reliable prediction was generated because the molecule name could not be resolved to a valid structure: {names}. "
            "Without a valid SMILES, AgentM cannot build molecular features, and the guard prevents the LLM from inventing a prediction. "
            "Please check the spelling, provide a standard English drug name or SMILES, or enable network lookup in Custom mode."
        )
    if warnings:
        reason = "; ".join(warnings[:3])
        if chinese:
            return f"本次没有生成可靠预测，因为预测流程在进入模型前被中止：{reason}。系统已保留 trace，可在 Trace/Artifacts 中查看具体失败步骤。"
        return f"No reliable prediction was generated because the workflow stopped before model prediction: {reason}. See Trace/Artifacts for the failed step."
    if chinese:
        return "本次没有生成可靠预测。系统没有检测到可追溯的预测工具输出，因此不会把 LLM 文本当作预测结果。"
    return "No reliable prediction was generated. No traceable prediction-tool output was found, so LLM text was not treated as a prediction."


def _looks_chinese(text: Any) -> bool:
    return any("\u4e00" <= char <= "\u9fff" for char in str(text or ""))


def _unique_strings(items: Any) -> list[str]:
    unique: list[str] = []
    seen: set[str] = set()
    for item in items or []:
        text = str(item)
        if text in seen:
            continue
        seen.add(text)
        unique.append(text)
    return unique


def _unique_evidence(items: Any) -> list[Any]:
    unique: list[Any] = []
    seen: set[tuple[str, str, str]] = set()
    for item in items or []:
        if not isinstance(item, dict):
            unique.append(item)
            continue
        key = _evidence_key(item)
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


def _evidence_key(item: Dict[str, Any]) -> tuple[str, str, str]:
    source = str(item.get("source") or item.get("source_group") or item.get("provider") or "")
    locator = str(
        item.get("citation_id")
        or item.get("pmid")
        or item.get("doi")
        or item.get("chunk_id")
        or item.get("source_path")
        or item.get("title")
        or ""
    )
    claim = str(item.get("claim") or item.get("snippet") or item.get("mechanism") or "")[:500]
    return (source, locator, claim)
