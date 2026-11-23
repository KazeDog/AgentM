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
from agentm.runtime.scientific_metrics import evaluate_scientific_case, summarize_scientific_rows


ZERO_SHOT_MODEL_TYPE = "zero_shot_rule_semantic_fusion"
DEFAULT_CASE_FILE = "configs/benchmark_cases.jsonl"

CSV_FIELDS = [
    "name",
    "category",
    "task_type",
    "dataset_name",
    "expected_endpoint",
    "run_status",
    "objective_status",
    "verifier_status",
    "model_type",
    "requested_model_path",
    "requested_feature_set",
    "real_model_success",
    "scientific_mode",
    "gold_value",
    "gold_split",
    "gold_source",
    "prediction_value",
    "scientific_status",
    "scientific_evaluable",
    "scientific_correct",
    "absolute_error",
    "n_evidence",
    "citation_grounded",
    "evidence_relevance_grounded",
    "compound_specific_grounded",
    "endpoint_specific_grounded",
    "n_objective_warnings",
    "n_guard_revisions",
    "n_reflections",
    "planner_type",
    "latency_sec",
    "report_path",
    "case_dir",
    "error",
]


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an AgentM paper benchmark suite over ADMET/DDI cases.")
    parser.add_argument("--case-file", default=DEFAULT_CASE_FILE)
    parser.add_argument("--output-root", default="outputs/benchmark_suite")
    parser.add_argument("--feature-set", default="rdkit_only")
    parser.add_argument("--mode", default="full", choices=["fast", "full", "deep"])
    parser.add_argument("--retrieval-backend", default="auto", choices=["auto", "es", "local"])
    parser.add_argument("--es-url", default=os.getenv("AGENTM_ES_URL"))
    parser.add_argument("--max-rounds", type=int, default=20)
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--case", action="append", default=[], help="Run only selected case name(s).")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-real-model", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-real-rag", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-llm-orchestrator", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-llm-reasoning", action=argparse.BooleanOptionalAction, default=True)
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
    cases = select_cases(load_cases(Path(args.case_file)), names=set(args.case or []), max_cases=args.max_cases)
    rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []

    for case in cases:
        row = run_one_case(case, args=args, output_root=output_root)
        rows.append(row)
        if row.get("run_status") == "failed_exception" or row.get("objective_status") == "fail":
            failures.append(row)
        print(json.dumps(row, ensure_ascii=False))

    summary = build_summary(rows, args=args, output_root=output_root)
    write_outputs(output_root, summary=summary, rows=rows, failures=failures)
    print(json.dumps({"benchmark_summary_path": str(output_root / "benchmark_summary.json"), "n_cases": len(rows)}, ensure_ascii=False))
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


def select_cases(cases: Iterable[Mapping[str, Any]], *, names: set[str], max_cases: int) -> List[Dict[str, Any]]:
    selected = [dict(case) for case in cases if not names or str(case.get("name")) in names]
    if names:
        found = {str(case.get("name")) for case in selected}
        missing = sorted(names - found)
        if missing:
            raise SystemExit(f"Unknown --case value(s): {missing}")
    if max_cases and max_cases > 0:
        selected = selected[:max_cases]
    return selected


def run_one_case(case: Mapping[str, Any], *, args: argparse.Namespace, output_root: Path) -> Dict[str, Any]:
    case_name = str(case["name"])
    case_dir = output_root / case_name
    metrics_path = case_dir / "case_metrics.json"
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
            model_path=case.get("model_path"),
            feature_set=str(case.get("feature_set") or args.feature_set),
            auto_model=True,
            require_real_model=args.require_real_model,
            require_real_rag=args.require_real_rag,
            min_evidence=1,
            mode=args.mode,
            retrieval_backend=args.retrieval_backend,
            es_url=args.es_url,
            max_rounds=args.max_rounds,
            use_llm_orchestrator=args.use_llm_orchestrator,
            use_llm_reasoning=args.use_llm_reasoning,
            enable_fallbacks=False,
        )
        report = controller.run(str(case["request"]), task_type=str(case["task_type"]), dataset_name=str(case["dataset_name"]))
        row = metrics_from_report(case, report, case_dir=case_dir, latency_sec=time.time() - started)
    except Exception as exc:
        row = {
            **case_identity(case, case_dir=case_dir),
            "run_status": "failed_exception",
            "objective_status": None,
            "verifier_status": None,
            "model_type": None,
            "real_model_success": False,
            "n_evidence": 0,
            "citation_grounded": None,
            "evidence_relevance_grounded": None,
            "compound_specific_grounded": None,
            "endpoint_specific_grounded": None,
            "n_objective_warnings": 0,
            "n_guard_revisions": 0,
            "n_reflections": 0,
            "planner_type": None,
            "latency_sec": round(time.time() - started, 3),
            "report_path": None,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
    metrics_path.write_text(json.dumps(row, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return normalize_row(row)


def metrics_from_report(case: Mapping[str, Any], report: Mapping[str, Any], *, case_dir: Path, latency_sec: float) -> Dict[str, Any]:
    prediction = first_mapping(report.get("predictions"))
    scientific = scientific_metrics_from_case(case, prediction)
    verifier = report.get("verifier_report") if isinstance(report.get("verifier_report"), Mapping) else {}
    citations = verifier.get("citations") if isinstance(verifier.get("citations"), Mapping) else {}
    summary = verifier.get("summary") if isinstance(verifier.get("summary"), Mapping) else {}
    objectives = report.get("objective_status") if isinstance(report.get("objective_status"), Mapping) else {}
    warnings = [str(item) for item in report.get("warnings", []) or []]
    artifacts = report.get("artifacts") if isinstance(report.get("artifacts"), Mapping) else {}
    tool_calls = [item for item in report.get("tool_calls", []) or [] if isinstance(item, Mapping)]
    plan_revision_trace = [item for item in report.get("plan_revision_trace", []) or [] if isinstance(item, Mapping)]
    model_type = prediction.get("model_type")
    return {
        **case_identity(case, case_dir=case_dir),
        "run_status": "completed",
        "objective_status": objectives.get("status"),
        "verifier_status": summary.get("status"),
        "model_type": model_type,
        "real_model_success": bool(model_type and model_type != ZERO_SHOT_MODEL_TYPE),
        **scientific,
        "n_evidence": len(report.get("evidence", []) or []),
        "citation_grounded": citations.get("citation_grounded"),
        "evidence_relevance_grounded": citations.get("evidence_relevance_grounded"),
        "compound_specific_grounded": citations.get("compound_specific_grounded"),
        "endpoint_specific_grounded": citations.get("endpoint_specific_grounded"),
        "n_objective_warnings": len([warning for warning in warnings if warning.startswith("objective_missing:")]),
        "n_guard_revisions": len([item for item in plan_revision_trace if item.get("stage") == "action_guard"]),
        "n_reflections": len([item for item in tool_calls if item.get("agent") == "ReflectionAgent"]),
        "planner_type": (report.get("multi_agent") or {}).get("planner_type") if isinstance(report.get("multi_agent"), Mapping) else None,
        "latency_sec": round(float(latency_sec), 3),
        "report_path": artifacts.get("multi_agent_report_path"),
        "error": "",
        "warnings": warnings,
    }


def case_identity(case: Mapping[str, Any], *, case_dir: Path) -> Dict[str, Any]:
    gold_value = case.get("gold_value") if "gold_value" in case else case.get("gold_label")
    return {
        "name": str(case.get("name") or ""),
        "request": str(case.get("request") or ""),
        "category": str(case.get("category") or ""),
        "task_type": str(case.get("task_type") or ""),
        "dataset_name": str(case.get("dataset_name") or ""),
        "expected_endpoint": str(case.get("expected_endpoint") or ""),
        "notes": str(case.get("notes") or ""),
        "requested_model_path": str(case.get("model_path") or ""),
        "requested_feature_set": str(case.get("feature_set") or ""),
        "scientific_mode": str(case.get("scientific_mode") or ""),
        "gold_value": gold_value,
        "gold_split": case.get("gold_split"),
        "gold_source": case.get("gold_source"),
        "case_dir": str(case_dir),
    }


def scientific_metrics_from_case(
    case: Mapping[str, Any],
    prediction: Mapping[str, Any],
) -> Dict[str, Any]:
    has_gold = "gold_value" in case or "gold_label" in case
    mode = str(case.get("scientific_mode") or "")
    if not mode and "gold_label" in case:
        mode = "classification"
    prediction_value = (
        prediction.get("score")
        if mode == "regression"
        else prediction.get("label")
    )
    if prediction_value is None and mode == "regression":
        prediction_value = prediction.get("label")
    if not has_gold:
        return {
            "scientific_mode": mode,
            "gold_value": None,
            "gold_split": case.get("gold_split"),
            "gold_source": case.get("gold_source"),
            "prediction_value": prediction_value,
            "scientific_status": "not_labeled",
            "scientific_evaluable": False,
            "scientific_correct": None,
            "absolute_error": None,
        }
    gold_value = case.get("gold_value") if "gold_value" in case else case.get("gold_label")
    evaluated = evaluate_scientific_case(
        gold_value=gold_value,
        prediction_value=prediction_value,
        mode=mode,
        gold_split=str(case.get("gold_split") or ""),
    )
    evaluated["gold_source"] = case.get("gold_source")
    return evaluated


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


def build_summary(rows: List[Mapping[str, Any]], *, args: argparse.Namespace, output_root: Path) -> Dict[str, Any]:
    completed = [row for row in rows if row.get("run_status") == "completed"]
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "output_root": str(output_root),
        "case_file": str(args.case_file),
        "settings": {
            "feature_set": args.feature_set,
            "mode": args.mode,
            "retrieval_backend": args.retrieval_backend,
            "es_url": args.es_url,
            "max_rounds": args.max_rounds,
            "require_real_model": bool(args.require_real_model),
            "require_real_rag": bool(args.require_real_rag),
            "use_llm_orchestrator": bool(args.use_llm_orchestrator),
            "use_llm_reasoning": bool(args.use_llm_reasoning),
        },
        "metrics": {
            "n_cases": len(rows),
            "n_completed": len(completed),
            "n_failed_exception": len([row for row in rows if row.get("run_status") == "failed_exception"]),
            "task_success_rate": ratio(len(completed), len(rows)),
            "real_model_success_rate": ratio(len([row for row in completed if row.get("real_model_success")]), len(completed)),
            "citation_grounding_rate": ratio(len([row for row in completed if row.get("citation_grounded") is True]), len(completed)),
            "verifier_pass_rate": ratio(len([row for row in completed if row.get("verifier_status") == "pass"]), len(completed)),
            "mean_latency_sec": mean([float(row.get("latency_sec") or 0) for row in completed]),
        },
        "scientific_performance": summarize_scientific_rows(rows),
        "cases": [normalize_row(row) for row in rows],
    }


def ratio(numerator: int, denominator: int) -> Optional[float]:
    if denominator <= 0:
        return None
    return round(float(numerator) / float(denominator), 4)


def mean(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return round(sum(values) / len(values), 3)


def write_outputs(output_root: Path, *, summary: Mapping[str, Any], rows: List[Mapping[str, Any]], failures: List[Mapping[str, Any]]) -> None:
    (output_root / "benchmark_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_root / "failure_report.json").write_text(json.dumps({"failures": failures}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_csv(output_root / "case_metrics.csv", rows)
    (output_root / "benchmark_summary.md").write_text(render_markdown(summary), encoding="utf-8")


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
        "# AgentM Benchmark Summary",
        "",
        f"- Created: `{summary.get('created_at')}`",
        f"- Cases: `{metrics.get('n_cases')}`",
        f"- Completed: `{metrics.get('n_completed')}`",
        f"- Failed exceptions: `{metrics.get('n_failed_exception')}`",
        f"- Task success rate: `{metrics.get('task_success_rate')}`",
        f"- Real-model success rate: `{metrics.get('real_model_success_rate')}`",
        f"- Citation grounding rate: `{metrics.get('citation_grounding_rate')}`",
        f"- Verifier pass rate: `{metrics.get('verifier_pass_rate')}`",
        f"- Mean latency sec: `{metrics.get('mean_latency_sec')}`",
        "",
        "## Scientific Performance",
        "",
        (
            f"- Held-out evaluable cases: "
            f"`{(summary.get('scientific_performance', {}).get('coverage') or {}).get('n_evaluable', 0)}`"
        ),
        (
            f"- Classification: "
            f"`{summary.get('scientific_performance', {}).get('classification')}`"
        ),
        (
            f"- Regression: "
            f"`{summary.get('scientific_performance', {}).get('regression')}`"
        ),
        "- Training/validation labels are excluded from these scientific metrics.",
        "",
        "## Settings",
        "",
        f"- Mode: `{settings.get('mode')}`",
        f"- Retrieval backend: `{settings.get('retrieval_backend')}`",
        f"- Feature set: `{settings.get('feature_set')}`",
        f"- Require real model: `{settings.get('require_real_model')}`",
        f"- Require real RAG: `{settings.get('require_real_rag')}`",
        f"- Use LLM orchestrator: `{settings.get('use_llm_orchestrator')}`",
        f"- Use LLM reasoning: `{settings.get('use_llm_reasoning')}`",
        "",
        "## Cases",
        "",
        "| Case | Task | Dataset | Status | Verifier | Model | Evidence | Citation | Report |",
        "|---|---|---|---|---|---|---:|---|---|",
    ]
    for row in summary.get("cases", []) or []:
        report = row.get("report_path") or ""
        report_text = f"`{report}`" if report else ""
        lines.append(
            "| {name} | {task} | {dataset} | {status} | {verifier} | {model} | {evidence} | {citation} | {report} |".format(
                name=row.get("name", ""),
                task=row.get("task_type", ""),
                dataset=row.get("dataset_name", ""),
                status=row.get("objective_status") or row.get("run_status"),
                verifier=row.get("verifier_status", ""),
                model=row.get("model_type", ""),
                evidence=row.get("n_evidence", 0),
                citation=row.get("citation_grounded", ""),
                report=report_text,
            )
        )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
