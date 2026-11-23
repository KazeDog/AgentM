#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agentm.runtime import AgentMMultiAgentController


ZERO_SHOT_MODEL_TYPE = "zero_shot_rule_semantic_fusion"
DEFAULT_CASE_FILE = "configs/benchmark_cases.jsonl"
DEFAULT_ABLATION_CASES = [
    "aspirin_dili",
    "aspirin_bbbp",
    "caffeine_herg",
    "simvastatin_ketoconazole_ddi",
    "clopidogrel_omeprazole_ddi",
]

VARIANT_CATALOG: Dict[str, Dict[str, Any]] = {
    "llm_planner_reflection_on_retrieval_on": {
        "description": "LLM orchestrator enabled, reflection enabled, strict real-RAG objective enabled.",
        "use_llm_orchestrator": True,
        "enable_reflection": True,
        "require_real_rag": True,
        "min_evidence": 1,
        "enable_fallbacks": False,
    },
    "heuristic_planner_reflection_on_retrieval_on": {
        "description": "Heuristic orchestrator, reflection enabled, strict real-RAG objective enabled.",
        "use_llm_orchestrator": False,
        "enable_reflection": True,
        "require_real_rag": True,
        "min_evidence": 1,
        "enable_fallbacks": False,
    },
    "heuristic_planner_reflection_off_retrieval_on": {
        "description": "Heuristic orchestrator, reflection disabled, strict real-RAG objective enabled.",
        "use_llm_orchestrator": False,
        "enable_reflection": False,
        "require_real_rag": True,
        "min_evidence": 1,
        "enable_fallbacks": False,
    },
    "heuristic_planner_reflection_on_retrieval_required_off": {
        "description": "Heuristic orchestrator, reflection enabled, evidence hard requirement disabled.",
        "use_llm_orchestrator": False,
        "enable_reflection": True,
        "require_real_rag": False,
        "min_evidence": 0,
        "enable_fallbacks": False,
    },
}

CSV_FIELDS = [
    "variant",
    "case_name",
    "category",
    "task_type",
    "dataset_name",
    "expected_endpoint",
    "run_status",
    "objective_status",
    "verifier_status",
    "planner_type",
    "llm_plan_valid",
    "llm_plan_fallback",
    "llm_action_proposals",
    "model_type",
    "real_model_success",
    "n_evidence",
    "citation_grounded",
    "evidence_relevance_grounded",
    "compound_specific_grounded",
    "endpoint_specific_grounded",
    "n_guard_revisions",
    "n_reflections",
    "n_warnings",
    "n_objective_warnings",
    "report_quality_score",
    "reflection_input_verifier_status",
    "reflection_input_warning_count",
    "reflection_repaired_warning_count",
    "reflection_repaired_first_pass",
    "reflection_repaired_objectives",
    "reflection_repair_categories",
    "reflection_application_status",
    "n_retrieval_tool_calls",
    "n_retrieval_trace_items",
    "retrieval_backend_used",
    "mean_tool_duration_sec",
    "latency_sec",
    "report_path",
    "case_dir",
    "error",
]


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AgentM LLM-orchestration ablations over ADMET/DDI cases.")
    parser.add_argument("--case-file", default=DEFAULT_CASE_FILE)
    parser.add_argument("--output-root", default="outputs/ablation")
    parser.add_argument("--feature-set", default="all_features")
    parser.add_argument("--mode", default="full", choices=["fast", "full", "deep"])
    parser.add_argument("--retrieval-backend", default="local", choices=["auto", "es", "local"])
    parser.add_argument("--es-url", default=os.getenv("AGENTM_ES_URL"))
    parser.add_argument("--max-rounds", type=int, default=20)
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--case", action="append", default=[], help="Run only selected case name(s).")
    parser.add_argument("--all-cases", action="store_true", help="Run all cases from --case-file when --case is not set.")
    parser.add_argument("--variant", action="append", choices=sorted(VARIANT_CATALOG), default=[], help="Run only selected ablation variant(s).")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-real-model", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-real-rag", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-llm-orchestrator", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use-llm-reasoning", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--enable-network-lookup", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--pubchem-timeout", type=float, default=None)
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    if args.enable_network_lookup is not None:
        os.environ["AGENTM_ENABLE_NETWORK_LOOKUP"] = "1" if args.enable_network_lookup else "0"
    if args.pubchem_timeout is not None:
        os.environ["AGENTM_PUBCHEM_TIMEOUT"] = str(args.pubchem_timeout)

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    selected_case_names = list(args.case or [])
    if not selected_case_names and not args.all_cases and not args.max_cases:
        selected_case_names = list(DEFAULT_ABLATION_CASES)
    cases = select_cases(load_cases(Path(args.case_file)), names=selected_case_names, max_cases=args.max_cases)
    variants = select_variants(args.variant)

    rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    for variant_name, variant in variants.items():
        for case in cases:
            row = run_one_variant_case(variant_name, variant, case, args=args, output_root=output_root)
            rows.append(row)
            if row.get("run_status") == "failed_exception" or row.get("objective_status") == "fail":
                failures.append(row)
            print(json.dumps(row, ensure_ascii=False))

    summary = build_summary(rows, args=args, output_root=output_root, cases=cases, variants=variants)
    write_outputs(output_root, summary=summary, rows=rows, failures=failures)
    print(json.dumps({"ablation_summary_path": str(output_root / "ablation_summary.json"), "n_runs": len(rows)}, ensure_ascii=False))
    return 0 if not any(row.get("run_status") == "failed_exception" for row in rows) else 2


def load_cases(case_file: Path) -> List[Dict[str, Any]]:
    cases: List[Dict[str, Any]] = []
    with case_file.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                case = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Invalid JSONL at {case_file}:{line_no}: {exc}") from exc
            missing = [key for key in ["name", "request", "task_type", "dataset_name", "category", "expected_endpoint", "notes"] if key not in case]
            if missing:
                raise SystemExit(f"Case {case_file}:{line_no} missing required fields: {missing}")
            cases.append(case)
    names = [str(case["name"]) for case in cases]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise SystemExit(f"Duplicate benchmark case names: {duplicates}")
    return cases


def select_cases(cases: Iterable[Mapping[str, Any]], *, names: List[str], max_cases: int) -> List[Dict[str, Any]]:
    wanted = list(dict.fromkeys(str(name) for name in names if str(name)))
    wanted_set = set(wanted)
    selected = [dict(case) for case in cases if not wanted_set or str(case.get("name")) in wanted_set]
    if wanted:
        found = {str(case.get("name")) for case in selected}
        missing = [name for name in wanted if name not in found]
        if missing:
            raise SystemExit(f"Unknown --case value(s): {missing}")
        order = {name: idx for idx, name in enumerate(wanted)}
        selected.sort(key=lambda item: order.get(str(item.get("name")), 9999))
    if max_cases and max_cases > 0:
        selected = selected[:max_cases]
    return selected


def select_variants(names: List[str]) -> Dict[str, Dict[str, Any]]:
    if not names:
        return dict(VARIANT_CATALOG)
    return {name: dict(VARIANT_CATALOG[name]) for name in names}


def run_one_variant_case(
    variant_name: str,
    variant: Mapping[str, Any],
    case: Mapping[str, Any],
    *,
    args: argparse.Namespace,
    output_root: Path,
) -> Dict[str, Any]:
    case_name = str(case["name"])
    case_dir = output_root / variant_name / case_name
    metrics_path = case_dir / "ablation_metrics.json"
    if args.resume and metrics_path.exists():
        try:
            cached = json.loads(metrics_path.read_text(encoding="utf-8"))
        except Exception:
            cached = {}
        if cached.get("run_status") == "completed":
            cached["resume_skipped"] = True
            return normalize_row(cached)

    case_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    try:
        controller = AgentMMultiAgentController(
            output_dir=case_dir,
            feature_set=args.feature_set,
            auto_model=True,
            require_real_model=args.require_real_model,
            require_real_rag=bool(args.require_real_rag and variant.get("require_real_rag", True)),
            require_llm_orchestrator=bool(args.require_llm_orchestrator and variant.get("use_llm_orchestrator")),
            min_evidence=int(variant.get("min_evidence", 1)),
            mode=args.mode,
            retrieval_backend=args.retrieval_backend,
            es_url=args.es_url,
            max_rounds=args.max_rounds,
            use_llm_orchestrator=bool(variant.get("use_llm_orchestrator", False)),
            use_llm_reasoning=bool(args.use_llm_reasoning),
            enable_reflection=bool(variant.get("enable_reflection", True)),
            enable_fallbacks=bool(variant.get("enable_fallbacks", False)),
        )
        report = controller.run(str(case["request"]), task_type=str(case["task_type"]), dataset_name=str(case["dataset_name"]))
        row = metrics_from_report(variant_name, variant, case, report, case_dir=case_dir, latency_sec=time.time() - started)
    except Exception as exc:
        row = {
            **case_identity(variant_name, variant, case, case_dir=case_dir),
            "run_status": "failed_exception",
            "objective_status": None,
            "verifier_status": None,
            "planner_type": None,
            "llm_plan_valid": False,
            "llm_plan_fallback": None,
            "llm_action_proposals": 0,
            "model_type": None,
            "real_model_success": False,
            "n_evidence": 0,
            "citation_grounded": None,
            "evidence_relevance_grounded": None,
            "compound_specific_grounded": None,
            "endpoint_specific_grounded": None,
            "n_guard_revisions": 0,
            "n_reflections": 0,
            "n_warnings": 0,
            "n_objective_warnings": 0,
            "report_quality_score": 0.0,
            "reflection_input_verifier_status": None,
            "reflection_input_warning_count": 0,
            "reflection_repaired_warning_count": 0,
            "reflection_repaired_first_pass": False,
            "reflection_repaired_objectives": [],
            "reflection_repair_categories": [],
            "reflection_application_status": None,
            "n_retrieval_tool_calls": 0,
            "n_retrieval_trace_items": 0,
            "retrieval_backend_used": None,
            "mean_tool_duration_sec": None,
            "latency_sec": round(time.time() - started, 3),
            "report_path": None,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
    metrics_path.write_text(json.dumps(row, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return normalize_row(row)


def metrics_from_report(
    variant_name: str,
    variant: Mapping[str, Any],
    case: Mapping[str, Any],
    report: Mapping[str, Any],
    *,
    case_dir: Path,
    latency_sec: float,
) -> Dict[str, Any]:
    prediction = first_mapping(report.get("predictions"))
    verifier = report.get("verifier_report") if isinstance(report.get("verifier_report"), Mapping) else {}
    citations = verifier.get("citations") if isinstance(verifier.get("citations"), Mapping) else {}
    summary = verifier.get("summary") if isinstance(verifier.get("summary"), Mapping) else {}
    objectives = report.get("objective_status") if isinstance(report.get("objective_status"), Mapping) else {}
    warnings = [str(item) for item in report.get("warnings", []) or []]
    artifacts = report.get("artifacts") if isinstance(report.get("artifacts"), Mapping) else {}
    tool_calls = [item for item in report.get("tool_calls", []) or [] if isinstance(item, Mapping)]
    plan_revision_trace = [item for item in report.get("plan_revision_trace", []) or [] if isinstance(item, Mapping)]
    llm_proposals = [item for item in report.get("llm_plan_proposals", []) or [] if isinstance(item, Mapping)]
    multi_agent = report.get("multi_agent") if isinstance(report.get("multi_agent"), Mapping) else {}
    retrieval_tool_calls = [
        item
        for item in tool_calls
        if item.get("agent") == "RetrieveAgent" or str(item.get("tool") or "").startswith("retrieve_") or item.get("tool") == "expand_evidence_query"
    ]
    tool_durations = [float(item.get("duration_sec") or 0.0) for item in tool_calls if item.get("duration_sec") is not None]
    model_type = prediction.get("model_type")
    planner_type = multi_agent.get("planner_type")
    llm_fallback = any(
        "llm_task_graph_planning_fallback" in warning
        or "llm_task_graph_planning_failed" in warning
        or ("llm_orchestrator" in warning and "fallback" in warning)
        for warning in warnings
    )
    llm_plan_valid = bool(variant.get("use_llm_orchestrator") and planner_type == "llm" and llm_proposals and not llm_fallback)
    objective_status = objectives.get("status")
    verifier_status = summary.get("status")
    model_type = prediction.get("model_type")
    real_model_success = bool(model_type and model_type != ZERO_SHOT_MODEL_TYPE)
    n_evidence = len(report.get("evidence", []) or [])
    n_objective_warnings = len([warning for warning in warnings if str(warning).startswith("objective_missing:")])
    reflection_analysis = analyze_reflection_effect(report, objectives=objectives, verifier_summary=summary, final_warnings=warnings)
    quality_score = report_quality_score(
        objective_status=objective_status,
        verifier_status=verifier_status,
        real_model_success=real_model_success,
        n_evidence=n_evidence,
        citation_grounded=citations.get("citation_grounded"),
        evidence_relevance_grounded=citations.get("evidence_relevance_grounded"),
        compound_specific_grounded=citations.get("compound_specific_grounded"),
        endpoint_specific_grounded=citations.get("endpoint_specific_grounded"),
        n_warnings=len(warnings),
        n_objective_warnings=n_objective_warnings,
    )
    return {
        **case_identity(variant_name, variant, case, case_dir=case_dir),
        "run_status": "completed",
        "objective_status": objective_status,
        "verifier_status": verifier_status,
        "planner_type": planner_type,
        "llm_plan_valid": llm_plan_valid,
        "llm_plan_fallback": llm_fallback,
        "llm_action_proposals": len(llm_proposals),
        "model_type": model_type,
        "real_model_success": real_model_success,
        "n_evidence": n_evidence,
        "citation_grounded": citations.get("citation_grounded"),
        "evidence_relevance_grounded": citations.get("evidence_relevance_grounded"),
        "compound_specific_grounded": citations.get("compound_specific_grounded"),
        "endpoint_specific_grounded": citations.get("endpoint_specific_grounded"),
        "n_guard_revisions": len([item for item in plan_revision_trace if item.get("stage") == "action_guard"]),
        "n_reflections": len([item for item in tool_calls if item.get("agent") == "ReflectionAgent"]),
        "n_warnings": len(warnings),
        "n_objective_warnings": n_objective_warnings,
        "report_quality_score": quality_score,
        "reflection_input_verifier_status": reflection_analysis.get("input_verifier_status"),
        "reflection_input_warning_count": reflection_analysis.get("input_warning_count"),
        "reflection_repaired_warning_count": reflection_analysis.get("repaired_warning_count"),
        "reflection_repaired_first_pass": reflection_analysis.get("repaired_first_pass"),
        "reflection_repaired_objectives": reflection_analysis.get("repaired_objectives"),
        "reflection_repair_categories": reflection_analysis.get("repair_categories"),
        "reflection_application_status": reflection_analysis.get("application_status"),
        "n_retrieval_tool_calls": len(retrieval_tool_calls),
        "n_retrieval_trace_items": len(report.get("retrieval_trace", []) or []),
        "retrieval_backend_used": (report.get("metadata") or {}).get("retrieval_backend_used") if isinstance(report.get("metadata"), Mapping) else _retrieval_backend_from_artifacts(report),
        "mean_tool_duration_sec": mean(tool_durations),
        "latency_sec": round(float(latency_sec), 3),
        "report_path": artifacts.get("multi_agent_report_path"),
        "error": "",
        "warnings": warnings,
    }


def _retrieval_backend_from_artifacts(report: Mapping[str, Any]) -> str:
    retrieval_trace = report.get("retrieval_trace", []) or []
    for item in reversed(retrieval_trace):
        if isinstance(item, Mapping) and item.get("backend"):
            return str(item.get("backend"))
    return ""


def analyze_reflection_effect(
    report: Mapping[str, Any],
    *,
    objectives: Mapping[str, Any],
    verifier_summary: Mapping[str, Any],
    final_warnings: List[str],
) -> Dict[str, Any]:
    reflection = report.get("reflection") if isinstance(report.get("reflection"), Mapping) else {}
    if not reflection:
        return {
            "reflection_present": False,
            "input_verifier_status": None,
            "input_warning_count": 0,
            "repaired_warning_count": 0,
            "repaired_first_pass": False,
            "repaired_objectives": [],
            "repair_categories": [],
            "application_status": None,
        }
    input_summary = reflection.get("input_report_summary") if isinstance(reflection.get("input_report_summary"), Mapping) else {}
    input_warnings = [str(warning) for warning in input_summary.get("warnings", []) or []]
    input_objectives = sorted(
        {
            warning.split("objective_missing:", 1)[1]
            for warning in input_warnings
            if warning.startswith("objective_missing:") and warning.split("objective_missing:", 1)[1]
        }
    )
    final_missing = set(str(item) for item in objectives.get("missing_objectives", []) or [])
    final_warning_set = set(str(warning) for warning in final_warnings)
    repaired_objectives = [name for name in input_objectives if name not in final_missing]
    repaired_warnings = [warning for warning in input_warnings if warning not in final_warning_set]
    input_status = input_summary.get("verifier_status")
    final_status = verifier_summary.get("status")
    repaired_first_pass = bool(input_status in {"warning", "fail"} and final_status == "pass")
    categories: List[str] = []
    if repaired_first_pass:
        categories.append("first_pass_verifier_status")
    if repaired_objectives:
        categories.append("objective_gap")
    if repaired_warnings:
        categories.append("warning_reduction")
    if reflection.get("requires_reasoning_refresh"):
        categories.append("reasoning_refresh")
    if reflection.get("requires_retrieval_refinement"):
        categories.append("retrieval_refinement_requested")
    return {
        "reflection_present": True,
        "input_verifier_status": input_status,
        "input_warning_count": len(input_warnings),
        "repaired_warning_count": len(repaired_warnings),
        "repaired_first_pass": repaired_first_pass,
        "repaired_objectives": repaired_objectives,
        "repair_categories": sorted(set(categories)),
        "application_status": reflection.get("application_status"),
    }


def report_quality_score(
    *,
    objective_status: Any,
    verifier_status: Any,
    real_model_success: bool,
    n_evidence: int,
    citation_grounded: Any,
    evidence_relevance_grounded: Any,
    compound_specific_grounded: Any,
    endpoint_specific_grounded: Any,
    n_warnings: int,
    n_objective_warnings: int,
) -> float:
    score = 0.0
    score += {"pass": 4.0, "warning": 2.0, "fail": 0.0}.get(str(objective_status or ""), 0.0)
    score += {"pass": 4.0, "warning": 2.0, "fail": 0.0}.get(str(verifier_status or ""), 0.0)
    score += 2.0 if real_model_success else 0.0
    for flag in [citation_grounded, evidence_relevance_grounded, compound_specific_grounded, endpoint_specific_grounded]:
        score += 1.0 if flag is True else 0.0
    score += min(max(int(n_evidence or 0), 0), 3) * 0.5
    score -= min(max(int(n_objective_warnings or 0), 0), 8) * 0.25
    score -= min(max(int(n_warnings or 0), 0), 12) * 0.05
    return round(score, 3)


def case_identity(variant_name: str, variant: Mapping[str, Any], case: Mapping[str, Any], *, case_dir: Path) -> Dict[str, Any]:
    return {
        "variant": variant_name,
        "variant_description": str(variant.get("description") or ""),
        "case_name": str(case.get("name") or ""),
        "request": str(case.get("request") or ""),
        "category": str(case.get("category") or ""),
        "task_type": str(case.get("task_type") or ""),
        "dataset_name": str(case.get("dataset_name") or ""),
        "expected_endpoint": str(case.get("expected_endpoint") or ""),
        "notes": str(case.get("notes") or ""),
        "case_dir": str(case_dir),
    }


def first_mapping(value: Any) -> Dict[str, Any]:
    if isinstance(value, list) and value and isinstance(value[0], Mapping):
        return dict(value[0])
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def normalize_row(row: Mapping[str, Any]) -> Dict[str, Any]:
    normalized = dict(row)
    for field in CSV_FIELDS:
        normalized.setdefault(field, "")
    return normalized


def build_summary(
    rows: List[Mapping[str, Any]],
    *,
    args: argparse.Namespace,
    output_root: Path,
    cases: List[Mapping[str, Any]],
    variants: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    completed = [row for row in rows if row.get("run_status") == "completed"]
    by_variant: Dict[str, Dict[str, Any]] = {}
    for variant_name in variants:
        subset = [row for row in completed if row.get("variant") == variant_name]
        by_variant[variant_name] = summarize_rows(subset)
    reflection_repair_summary = build_reflection_repair_summary(rows, variants)
    reflection_runs = [row for row in completed if row.get("reflection_application_status")]
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "output_root": str(output_root),
        "case_file": str(args.case_file),
        "cases": [str(case.get("name")) for case in cases],
        "variants": {name: dict(config) for name, config in variants.items()},
        "settings": {
            "feature_set": args.feature_set,
            "mode": args.mode,
            "retrieval_backend": args.retrieval_backend,
            "es_url": args.es_url,
            "max_rounds": args.max_rounds,
            "require_real_model": bool(args.require_real_model),
            "require_real_rag": bool(args.require_real_rag),
            "require_llm_orchestrator": bool(args.require_llm_orchestrator),
            "use_llm_reasoning": bool(args.use_llm_reasoning),
        },
        "metrics": {
            "n_runs": len(rows),
            "n_completed": len(completed),
            "n_failed_exception": len([row for row in rows if row.get("run_status") == "failed_exception"]),
            "run_success_rate": ratio(len(completed), len(rows)),
            "objective_pass_rate": ratio(len([row for row in completed if row.get("objective_status") == "pass"]), len(completed)),
            "verifier_pass_rate": ratio(len([row for row in completed if row.get("verifier_status") == "pass"]), len(completed)),
            "llm_plan_valid_rate": ratio(len([row for row in completed if row.get("llm_plan_valid") is True]), len(completed)),
            "mean_guard_revisions": mean([float(row.get("n_guard_revisions") or 0) for row in completed]),
            "mean_reflections": mean([float(row.get("n_reflections") or 0) for row in completed]),
            "reflection_first_pass_repair_rate": ratio(
                len([row for row in reflection_runs if row.get("reflection_repaired_first_pass") is True]),
                len(reflection_runs),
            ),
            "mean_retrieval_tool_calls": mean([float(row.get("n_retrieval_tool_calls") or 0) for row in completed]),
            "mean_latency_sec": mean([float(row.get("latency_sec") or 0) for row in completed]),
        },
        "by_variant": by_variant,
        "reflection_repair_summary": reflection_repair_summary,
        "runs": [normalize_row(row) for row in rows],
    }


def summarize_rows(rows: List[Mapping[str, Any]]) -> Dict[str, Any]:
    reflection_rows = [row for row in rows if row.get("reflection_application_status")]
    return {
        "n_runs": len(rows),
        "objective_pass_rate": ratio(len([row for row in rows if row.get("objective_status") == "pass"]), len(rows)),
        "verifier_pass_rate": ratio(len([row for row in rows if row.get("verifier_status") == "pass"]), len(rows)),
        "real_model_success_rate": ratio(len([row for row in rows if row.get("real_model_success")]), len(rows)),
        "citation_grounding_rate": ratio(len([row for row in rows if row.get("citation_grounded") is True]), len(rows)),
        "llm_plan_valid_rate": ratio(len([row for row in rows if row.get("llm_plan_valid") is True]), len(rows)),
        "mean_guard_revisions": mean([float(row.get("n_guard_revisions") or 0) for row in rows]),
        "mean_reflections": mean([float(row.get("n_reflections") or 0) for row in rows]),
        "first_pass_reflection_repair_rate": ratio(
            len([row for row in reflection_rows if row.get("reflection_repaired_first_pass") is True]),
            len(reflection_rows),
        ),
        "mean_retrieval_tool_calls": mean([float(row.get("n_retrieval_tool_calls") or 0) for row in rows]),
        "mean_evidence_count": mean([float(row.get("n_evidence") or 0) for row in rows]),
        "mean_report_quality_score": mean([float(row.get("report_quality_score") or 0) for row in rows]),
        "mean_latency_sec": mean([float(row.get("latency_sec") or 0) for row in rows]),
    }


def build_reflection_repair_summary(rows: List[Mapping[str, Any]], variants: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    completed = [normalize_row(row) for row in rows if row.get("run_status") == "completed"]
    reflection_rows = [row for row in completed if row.get("reflection_application_status")]
    paired_variant_names = find_reflection_variant_pairs(variants)
    rows_by_variant_case = {(row.get("variant"), row.get("case_name")): row for row in completed}
    paired_cases: List[Dict[str, Any]] = []
    taxonomy_counts: Dict[str, int] = {}
    for on_variant, off_variant in paired_variant_names:
        case_names = sorted(
            {
                str(row.get("case_name"))
                for row in completed
                if row.get("variant") in {on_variant, off_variant} and row.get("case_name")
            }
        )
        for case_name in case_names:
            on_row = rows_by_variant_case.get((on_variant, case_name))
            off_row = rows_by_variant_case.get((off_variant, case_name))
            if not on_row or not off_row:
                continue
            comparison = compare_reflection_pair(on_row, off_row)
            paired_cases.append(comparison)
            for field in comparison.get("improved_fields", []) or []:
                taxonomy_counts[field] = taxonomy_counts.get(field, 0) + 1
    first_pass_repairs = [row for row in reflection_rows if row.get("reflection_repaired_first_pass") is True]
    repaired_objectives: Dict[str, int] = {}
    repair_categories: Dict[str, int] = {}
    for row in reflection_rows:
        for objective in ensure_list(row.get("reflection_repaired_objectives")):
            repaired_objectives[str(objective)] = repaired_objectives.get(str(objective), 0) + 1
        for category in ensure_list(row.get("reflection_repair_categories")):
            repair_categories[str(category)] = repair_categories.get(str(category), 0) + 1
    return {
        "n_reflection_runs": len(reflection_rows),
        "n_first_pass_repaired": len(first_pass_repairs),
        "first_pass_repair_rate": ratio(len(first_pass_repairs), len(reflection_rows)),
        "repaired_objectives": dict(sorted(repaired_objectives.items())),
        "repair_categories": dict(sorted(repair_categories.items())),
        "n_paired_cases": len(paired_cases),
        "n_paired_cases_with_any_repair": len([case for case in paired_cases if case.get("any_repair")]),
        "paired_repair_rate": ratio(len([case for case in paired_cases if case.get("any_repair")]), len(paired_cases)),
        "paired_taxonomy": dict(sorted(taxonomy_counts.items())),
        "paired_variant_pairs": [{"reflection_on": on, "reflection_off": off} for on, off in paired_variant_names],
        "paired_cases": paired_cases,
    }


def find_reflection_variant_pairs(variants: Mapping[str, Mapping[str, Any]]) -> List[tuple[str, str]]:
    on_variants = [(name, cfg) for name, cfg in variants.items() if cfg.get("enable_reflection")]
    off_variants = [(name, cfg) for name, cfg in variants.items() if not cfg.get("enable_reflection")]
    pairs: List[tuple[str, str]] = []
    for on_name, on_cfg in on_variants:
        for off_name, off_cfg in off_variants:
            if variant_configs_match_except_reflection(on_cfg, off_cfg):
                pairs.append((on_name, off_name))
    return pairs


def variant_configs_match_except_reflection(on_cfg: Mapping[str, Any], off_cfg: Mapping[str, Any]) -> bool:
    keys = ["use_llm_orchestrator", "require_real_rag", "min_evidence", "enable_fallbacks"]
    return all(on_cfg.get(key) == off_cfg.get(key) for key in keys)


def compare_reflection_pair(on_row: Mapping[str, Any], off_row: Mapping[str, Any]) -> Dict[str, Any]:
    improved_fields: List[str] = []
    if off_row.get("objective_status") != "pass" and on_row.get("objective_status") == "pass":
        improved_fields.append("objective_status")
    if off_row.get("verifier_status") != "pass" and on_row.get("verifier_status") == "pass":
        improved_fields.append("verifier_status")
    if not bool(off_row.get("real_model_success")) and bool(on_row.get("real_model_success")):
        improved_fields.append("real_model_success")
    for field in ["citation_grounded", "evidence_relevance_grounded", "compound_specific_grounded", "endpoint_specific_grounded"]:
        if off_row.get(field) is not True and on_row.get(field) is True:
            improved_fields.append(field)
    evidence_delta = int(on_row.get("n_evidence") or 0) - int(off_row.get("n_evidence") or 0)
    if evidence_delta > 0:
        improved_fields.append("evidence_count")
    warning_delta = int(off_row.get("n_warnings") or 0) - int(on_row.get("n_warnings") or 0)
    if warning_delta > 0:
        improved_fields.append("warning_count")
    objective_warning_delta = int(off_row.get("n_objective_warnings") or 0) - int(on_row.get("n_objective_warnings") or 0)
    if objective_warning_delta > 0:
        improved_fields.append("objective_warning_count")
    quality_delta = round(float(on_row.get("report_quality_score") or 0) - float(off_row.get("report_quality_score") or 0), 3)
    if quality_delta > 0:
        improved_fields.append("quality_score")
    if on_row.get("reflection_repaired_first_pass") is True:
        improved_fields.append("first_pass_weak_report")
    improved_fields = sorted(set(improved_fields))
    return {
        "case_name": on_row.get("case_name"),
        "reflection_on_variant": on_row.get("variant"),
        "reflection_off_variant": off_row.get("variant"),
        "off_objective_status": off_row.get("objective_status"),
        "on_objective_status": on_row.get("objective_status"),
        "off_verifier_status": off_row.get("verifier_status"),
        "on_verifier_status": on_row.get("verifier_status"),
        "off_quality_score": off_row.get("report_quality_score"),
        "on_quality_score": on_row.get("report_quality_score"),
        "quality_score_delta": quality_delta,
        "evidence_count_delta": evidence_delta,
        "warning_count_delta": warning_delta,
        "objective_warning_count_delta": objective_warning_delta,
        "first_pass_repaired": bool(on_row.get("reflection_repaired_first_pass")),
        "repaired_objectives": ensure_list(on_row.get("reflection_repaired_objectives")),
        "improved_fields": improved_fields,
        "any_repair": bool(improved_fields),
    }


def ensure_list(value: Any) -> List[Any]:
    if isinstance(value, list):
        return value
    if value in (None, ""):
        return []
    return [value]


def ratio(numerator: int, denominator: int) -> Optional[float]:
    if denominator <= 0:
        return None
    return round(float(numerator) / float(denominator), 4)


def mean(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return round(sum(values) / len(values), 3)


def write_outputs(output_root: Path, *, summary: Mapping[str, Any], rows: List[Mapping[str, Any]], failures: List[Mapping[str, Any]]) -> None:
    (output_root / "ablation_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_root / "ablation_failures.json").write_text(json.dumps({"failures": failures}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    reflection_summary = summary.get("reflection_repair_summary", {})
    (output_root / "reflection_repair_summary.json").write_text(json.dumps(reflection_summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_csv(output_root / "ablation_metrics.csv", rows)
    (output_root / "ablation_summary.md").write_text(render_markdown(summary), encoding="utf-8")
    (output_root / "reflection_repair_summary.md").write_text(render_reflection_markdown(reflection_summary), encoding="utf-8")


def write_csv(path: Path, rows: List[Mapping[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(normalize_row(row))


def render_markdown(summary: Mapping[str, Any]) -> str:
    metrics = summary.get("metrics", {})
    settings = summary.get("settings", {})
    lines = [
        "# AgentM LLM Orchestration Ablation",
        "",
        f"- Created: `{summary.get('created_at')}`",
        f"- Runs: `{metrics.get('n_runs')}`",
        f"- Completed: `{metrics.get('n_completed')}`",
        f"- Failed exceptions: `{metrics.get('n_failed_exception')}`",
        f"- Objective pass rate: `{metrics.get('objective_pass_rate')}`",
        f"- Verifier pass rate: `{metrics.get('verifier_pass_rate')}`",
        f"- LLM plan valid rate: `{metrics.get('llm_plan_valid_rate')}`",
        f"- Mean guard revisions: `{metrics.get('mean_guard_revisions')}`",
        f"- Mean reflections: `{metrics.get('mean_reflections')}`",
        f"- First-pass reflection repair rate: `{metrics.get('reflection_first_pass_repair_rate')}`",
        f"- Mean retrieval tool calls: `{metrics.get('mean_retrieval_tool_calls')}`",
        f"- Mean latency sec: `{metrics.get('mean_latency_sec')}`",
        "",
        "## Settings",
        "",
        f"- Mode: `{settings.get('mode')}`",
        f"- Retrieval backend: `{settings.get('retrieval_backend')}`",
        f"- Feature set: `{settings.get('feature_set')}`",
        f"- Require real model: `{settings.get('require_real_model')}`",
        f"- Require real RAG: `{settings.get('require_real_rag')}`",
        f"- Require LLM orchestrator: `{settings.get('require_llm_orchestrator')}`",
        f"- Use LLM reasoning: `{settings.get('use_llm_reasoning')}`",
        "",
        "## Variant Summary",
        "",
        "| Variant | Runs | Objective pass | Verifier pass | LLM valid | Guard revisions | Reflections | First-pass repair | Retrieval calls | Evidence | Quality | Latency |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in (summary.get("by_variant") or {}).items():
        lines.append(
            "| {name} | {n} | {objective} | {verifier} | {llm} | {guard} | {reflection} | {first_pass} | {retrieval} | {evidence} | {quality} | {latency} |".format(
                name=name,
                n=row.get("n_runs"),
                objective=row.get("objective_pass_rate"),
                verifier=row.get("verifier_pass_rate"),
                llm=row.get("llm_plan_valid_rate"),
                guard=row.get("mean_guard_revisions"),
                reflection=row.get("mean_reflections"),
                first_pass=row.get("first_pass_reflection_repair_rate"),
                retrieval=row.get("mean_retrieval_tool_calls"),
                evidence=row.get("mean_evidence_count"),
                quality=row.get("mean_report_quality_score"),
                latency=row.get("mean_latency_sec"),
            )
        )
    reflection_summary = summary.get("reflection_repair_summary") or {}
    lines.extend(
        [
            "",
            "## Reflection Repair Summary",
            "",
            f"- Reflection runs: `{reflection_summary.get('n_reflection_runs')}`",
            f"- First-pass weak reports repaired: `{reflection_summary.get('n_first_pass_repaired')}`",
            f"- First-pass repair rate: `{reflection_summary.get('first_pass_repair_rate')}`",
            f"- Paired reflection on/off cases: `{reflection_summary.get('n_paired_cases')}`",
            f"- Paired cases with any repair or quality improvement: `{reflection_summary.get('n_paired_cases_with_any_repair')}`",
            f"- Paired repair rate: `{reflection_summary.get('paired_repair_rate')}`",
            f"- Repaired objectives: `{json.dumps(reflection_summary.get('repaired_objectives', {}), ensure_ascii=False)}`",
            f"- Repair categories: `{json.dumps(reflection_summary.get('repair_categories', {}), ensure_ascii=False)}`",
            "",
            "| Case | Off status | On status | First-pass repaired | Quality delta | Improved fields |",
            "|---|---|---|---:|---:|---|",
        ]
    )
    for case in reflection_summary.get("paired_cases", []) or []:
        lines.append(
            "| {case} | {off_status}/{off_verifier} | {on_status}/{on_verifier} | {first_pass} | {quality_delta} | {fields} |".format(
                case=case.get("case_name", ""),
                off_status=case.get("off_objective_status", ""),
                off_verifier=case.get("off_verifier_status", ""),
                on_status=case.get("on_objective_status", ""),
                on_verifier=case.get("on_verifier_status", ""),
                first_pass=case.get("first_pass_repaired", ""),
                quality_delta=case.get("quality_score_delta", ""),
                fields=", ".join(case.get("improved_fields", []) or []),
            )
        )
    lines.extend(
        [
            "",
            "## Runs",
            "",
            "| Variant | Case | Status | Verifier | Planner | LLM valid | Guard | Reflection | Retrieval | Evidence | Report |",
            "|---|---|---|---|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in summary.get("runs", []) or []:
        report = row.get("report_path") or ""
        report_text = f"`{report}`" if report else ""
        lines.append(
            "| {variant} | {case} | {status} | {verifier} | {planner} | {llm} | {guard} | {reflection} | {retrieval} | {evidence} | {report} |".format(
                variant=row.get("variant", ""),
                case=row.get("case_name", ""),
                status=row.get("objective_status") or row.get("run_status"),
                verifier=row.get("verifier_status", ""),
                planner=row.get("planner_type", ""),
                llm=row.get("llm_plan_valid", ""),
                guard=row.get("n_guard_revisions", ""),
                reflection=row.get("n_reflections", ""),
                retrieval=row.get("n_retrieval_tool_calls", ""),
                evidence=row.get("n_evidence", ""),
                report=report_text,
            )
        )
    lines.append("")
    return "\n".join(lines)


def render_reflection_markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# AgentM Reflection Repair Summary",
        "",
        f"- Reflection runs: `{summary.get('n_reflection_runs')}`",
        f"- First-pass weak reports repaired: `{summary.get('n_first_pass_repaired')}`",
        f"- First-pass repair rate: `{summary.get('first_pass_repair_rate')}`",
        f"- Paired reflection on/off cases: `{summary.get('n_paired_cases')}`",
        f"- Paired cases with any repair or quality improvement: `{summary.get('n_paired_cases_with_any_repair')}`",
        f"- Paired repair rate: `{summary.get('paired_repair_rate')}`",
        "",
        "## Repaired Objectives",
        "",
    ]
    repaired = summary.get("repaired_objectives") or {}
    if repaired:
        for name, count in repaired.items():
            lines.append(f"- `{name}`: `{count}`")
    else:
        lines.append("- None recorded.")
    lines.extend(["", "## Repair Categories", ""])
    categories = summary.get("repair_categories") or {}
    if categories:
        for name, count in categories.items():
            lines.append(f"- `{name}`: `{count}`")
    else:
        lines.append("- None recorded.")
    lines.extend(
        [
            "",
            "## Paired Reflection On/Off Cases",
            "",
            "| Case | Off objective/verifier | On objective/verifier | First-pass repaired | Quality delta | Improved fields |",
            "|---|---|---|---:|---:|---|",
        ]
    )
    for case in summary.get("paired_cases", []) or []:
        lines.append(
            "| {case_name} | {off_obj}/{off_verifier} | {on_obj}/{on_verifier} | {first_pass} | {delta} | {fields} |".format(
                case_name=case.get("case_name", ""),
                off_obj=case.get("off_objective_status", ""),
                off_verifier=case.get("off_verifier_status", ""),
                on_obj=case.get("on_objective_status", ""),
                on_verifier=case.get("on_verifier_status", ""),
                first_pass=case.get("first_pass_repaired", ""),
                delta=case.get("quality_score_delta", ""),
                fields=", ".join(case.get("improved_fields", []) or []),
            )
        )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
