from __future__ import annotations

import traceback
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from agentm.utils.io import ensure_dir, write_json


def assess_failure_modes(report: Mapping[str, Any]) -> Dict[str, Any]:
    warnings = list(report.get("warnings", []) or [])
    trace = list(report.get("trace", []) or [])
    reasoning = report.get("reasoning", {}) if isinstance(report.get("reasoning"), Mapping) else {}
    artifacts = report.get("artifacts", {}) if isinstance(report.get("artifacts"), Mapping) else {}
    predictions = list(report.get("predictions", []) or [])
    prediction = predictions[0] if predictions else {}
    confidence = _prediction_confidence_from_reasoning(reasoning)

    failed_steps = [item for item in trace if item.get("status") == "failed"]
    fallback_steps = [item for item in trace if item.get("fallback_tool") or str(item.get("status", "")).startswith("fallback")]
    external_errors = [item for item in report.get("evidence", []) or [] if item.get("source") == "external_evidence_error"]
    issues: List[str] = []

    if prediction.get("model_type") == "zero_shot_rule_semantic_fusion":
        issues.append("zero_shot_prediction_used")
    if confidence.get("low_confidence"):
        issues.append("low_confidence_prediction")
    if warnings:
        issues.append("warnings_present")
    if fallback_steps:
        issues.append("tool_fallbacks_used")
    if failed_steps:
        issues.append("failed_steps_present")
    if external_errors:
        issues.append("external_evidence_errors")
    if artifacts.get("feature_fallback_used") or artifacts.get("ddi_pair_fallback_used"):
        issues.append("feature_fallback_used")

    image_paths = [value for key, value in artifacts.items() if key.endswith("image_path")]
    missing_images = [path for path in image_paths if path and not Path(str(path)).exists()]
    if missing_images:
        issues.append("missing_image_artifacts")

    if failed_steps:
        status = "failed"
    elif issues:
        status = "degraded"
    else:
        status = "ok"

    return {
        "robustness_status": status,
        "n_warnings": len(warnings),
        "n_fallbacks": len(fallback_steps),
        "n_failed_steps": len(failed_steps),
        "n_external_evidence_errors": len(external_errors),
        "low_confidence_prediction": confidence.get("low_confidence"),
        "issues": sorted(set(issues)),
        "failed_steps": [_trace_step(item) for item in failed_steps],
        "fallback_steps": [_trace_step(item) for item in fallback_steps],
        "warnings": warnings[:20],
    }


def run_failure_mode_probes(
    output_dir: str | Path,
    *,
    agent_kwargs: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    from agentm.agents.controller import AgentController

    root = ensure_dir(output_dir)
    common = {
        "feature_set": "rdkit_plus_llm_rules",
        "imagemol_checkpoint_path": None,
        "use_smiles_encoder": False,
        "use_mllm_visual_semantic": False,
        "use_rag_evidence": False,
    }
    common.update(dict(agent_kwargs or {}))
    probes = [
        {
            "name": "zero_shot_no_model",
            "request": "aspirin",
            "task_type": "property_prediction",
            "dataset_name": "BBBP",
            "kwargs": {},
            "expected": "completed_with_zero_shot_warning",
        },
        {
            "name": "invalid_molecule_input",
            "request": "C1CC",
            "task_type": "property_prediction",
            "dataset_name": "BBBP",
            "kwargs": {},
            "expected": "failed_gracefully",
        },
        {
            "name": "missing_imagemol_checkpoint_fallback",
            "request": "caffeine",
            "task_type": "property_prediction",
            "dataset_name": "BBBP",
            "kwargs": {"imagemol_checkpoint_path": "checkpoints/imagemol/does_not_exist.pth"},
            "expected": "completed_with_mock_imagemol",
        },
        {
            "name": "ddi_known_pair_without_model",
            "request": "warfarin and aspirin",
            "task_type": "ddi_prediction",
            "dataset_name": "DDI",
            "kwargs": {},
            "expected": "completed_zero_shot_ddi",
        },
    ]
    results: List[Dict[str, Any]] = []
    for probe in probes:
        case_dir = ensure_dir(root / probe["name"])
        kwargs = dict(common)
        kwargs.update(probe.get("kwargs", {}))
        item = {
            "name": probe["name"],
            "request": probe["request"],
            "expected": probe["expected"],
            "status": "pending",
            "output_dir": str(case_dir),
        }
        try:
            agent = AgentController(output_dir=case_dir, **kwargs)
            agent.rule_agent.use_llm = False
            report = agent.run(probe["request"], task_type=probe["task_type"], dataset_name=probe["dataset_name"])
            assessment = report.get("artifacts", {}).get("failure_mode_assessment") or assess_failure_modes(report)
            item.update(
                {
                    "status": "completed",
                    "robustness_status": assessment.get("robustness_status"),
                    "issues": assessment.get("issues", []),
                    "warnings": report.get("warnings", []),
                    "report_path": report.get("artifacts", {}).get("markdown_report_path"),
                }
            )
        except Exception as exc:
            item.update(
                {
                    "status": "failed_gracefully",
                    "error": str(exc),
                    "traceback": traceback.format_exc(limit=8),
                }
            )
        results.append(item)

    summary = {
        "n_probes": len(results),
        "n_completed": sum(1 for item in results if item["status"] == "completed"),
        "n_failed_gracefully": sum(1 for item in results if item["status"] == "failed_gracefully"),
        "probes": results,
    }
    write_json(root / "failure_mode_result.json", summary)
    return summary


def _prediction_confidence_from_reasoning(reasoning: Mapping[str, Any]) -> Dict[str, Any]:
    for item in reasoning.get("decisions", []) or []:
        if item.get("step") == "assess_prediction_confidence":
            return {
                "low_confidence": item.get("decision") == "low_confidence",
                "reasons": item.get("reasons", []),
                "margin_from_threshold": item.get("margin_from_threshold"),
            }
    return {}


def _trace_step(item: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "step_id": item.get("step_id"),
        "tool_name": item.get("tool_name"),
        "status": item.get("status"),
        "fallback_tool": item.get("fallback_tool"),
        "error": item.get("error"),
    }
