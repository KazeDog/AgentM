#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agentm.runtime import AgentMMultiAgentController
from agentm.runtime.reaction import _read_csv_rows, reaction_smiles_from_row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AgentM reaction benchmark cases.")
    parser.add_argument("--case-file", default="configs/reaction_example_cases.jsonl")
    parser.add_argument("--output-root", default="outputs/reaction_benchmark")
    parser.add_argument("--mode", default="full", choices=["fast", "full", "deep"])
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--case", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--use-full-reaction-models", action="store_true")
    parser.add_argument("--reaction-model-dir", default=None, help="Override reaction model artifact directory.")
    parser.add_argument("--parallel-workers", type=int, default=1)
    parser.add_argument("--reaction-data-root", default="data/reaction_sources")
    return parser.parse_args()


def load_cases(path: Path) -> list[dict]:
    cases = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def main() -> int:
    args = parse_args()
    output_root = ROOT / args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    cases = load_cases(ROOT / args.case_file)
    if args.case:
        cases = [case for case in cases if case.get("name") == args.case]
    if args.max_cases:
        cases = cases[: args.max_cases]
    jobs = [(idx, materialize_case(case, args.reaction_data_root)) for idx, case in enumerate(cases)]
    rows = []
    failures = []
    if args.parallel_workers and args.parallel_workers > 1:
        with ThreadPoolExecutor(max_workers=args.parallel_workers) as pool:
            futures = [pool.submit(run_case, idx, case, args, output_root) for idx, case in jobs]
            for future in as_completed(futures):
                row, failure = future.result()
                rows.append(row)
                if failure:
                    failures.append(failure)
    else:
        for idx, case in jobs:
            row, failure = run_case(idx, case, args, output_root)
            rows.append(row)
            if failure:
                failures.append(failure)
    rows.sort(key=lambda item: int(item.get("case_index", 0)))
    write_outputs(output_root, rows, failures)
    print(json.dumps({"n_cases": len(cases), "n_failures": len(failures), "summary_path": str(output_root / "benchmark_summary.json")}, indent=2))
    return 0 if not failures else 1


def materialize_case(case: dict, reaction_data_root: str) -> dict:
    if case.get("request"):
        return dict(case)
    source = str(case.get("source_dataset") or "")
    part = str(case.get("part") or "test")
    row_index = int(case.get("row_index") or 0)
    data_root = Path(reaction_data_root)
    if source == "schneider":
        rows = _read_csv_rows(data_root / "schneider" / "processed_reaction_class" / f"{part}_split.csv", max_rows=row_index + 1)
        row = rows[row_index]
        reaction = reaction_smiles_from_row(row)
        copy = dict(case)
        copy["request"] = f"Classify reaction type for {reaction}"
        copy.setdefault("task_type", "reaction_classification")
        copy.setdefault("dataset_name", "schneider")
        copy["source_label"] = row.get("labels") or row.get("reaction_type")
        return copy
    if source == "buchwald_hartwig_kg":
        split = str(case.get("split") or "test1")
        rows = _read_csv_rows(data_root / "processed_yield" / "buchwald_hartwig_kg" / split / f"{part}_split.csv", max_rows=row_index + 1)
        row = rows[row_index]
        reaction = reaction_smiles_from_row(row)
        copy = dict(case)
        if copy.get("task_type") == "reaction_condition_retrieval":
            copy["request"] = f"Retrieve synthesis condition context for {reaction}"
        elif copy.get("task_type") == "reaction_profile":
            copy["request"] = f"Give a reaction profile for {reaction}"
        else:
            copy["request"] = f"Predict yield for {reaction}"
            copy.setdefault("task_type", "reaction_yield_prediction")
        if copy.get("task_type") == "reaction_yield_prediction":
            copy["dataset_name"] = f"buchwald_hartwig_kg:{split}"
        else:
            copy.setdefault("dataset_name", "buchwald_hartwig_kg")
        copy["source_yield"] = row.get("yield")
        return copy
    if source == "mit":
        rows = _read_csv_rows(data_root / "MIT" / f"{part}.csv", max_rows=row_index + 1)
        row = rows[row_index]
        reaction = reaction_smiles_from_row(row)
        copy = dict(case)
        copy["request"] = f"Give a reaction profile for {reaction}"
        copy.setdefault("task_type", "reaction_profile")
        copy.setdefault("dataset_name", "mit")
        return copy
    raise ValueError(f"case {case.get('name')} has no request and unsupported source_dataset={source!r}")


def run_case(idx: int, case: dict, args: argparse.Namespace, output_root: Path) -> tuple[dict, dict | None]:
    case_dir = output_root / str(case["name"])
    report_path = case_dir / "multi_agent_report.json"
    if args.resume and report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        row = metric_row(case, report, skipped=True, latency=0.0)
        row["case_index"] = idx
        return row, None
    started = time.time()
    try:
        controller = AgentMMultiAgentController(
            output_dir=case_dir,
            mode=args.mode,
            max_rounds=18,
            use_llm_orchestrator=False,
            use_llm_reasoning=False,
            reaction_data_root=args.reaction_data_root,
            reaction_model_dir=_reaction_model_dir(args),
            reaction_index_dir=ROOT / "data" / "reaction_indices",
        )
        report = controller.run_reaction(case["request"], task_type=case["task_type"], dataset_name=case.get("dataset_name"))
        row = metric_row(case, report, skipped=False, latency=time.time() - started)
        row["case_index"] = idx
        return row, None
    except Exception as exc:
        failure = {"case": case, "error": str(exc), "traceback": traceback.format_exc()}
        return (
            {
                "case_index": idx,
                "name": case.get("name"),
                "task_type": case.get("task_type"),
                "status": "failed_exception",
                "latency_sec": round(time.time() - started, 3),
                "error": str(exc),
            },
            failure,
        )


def _reaction_model_dir(args: argparse.Namespace) -> Path | None:
    if args.reaction_model_dir:
        path = Path(args.reaction_model_dir)
        return path if path.is_absolute() else ROOT / path
    if args.use_full_reaction_models:
        return ROOT / "outputs" / "reaction_models_full"
    return None


def metric_row(case: dict, report: dict, *, skipped: bool, latency: float) -> dict:
    verifier = report.get("verifier_report", {}).get("summary", {})
    prediction = (report.get("reaction_predictions") or report.get("predictions") or [{}])[-1]
    tool_calls = report.get("tool_calls") or []
    trace = report.get("multi_agent_trace") or report.get("trace") or []
    unsupported_claims = report.get("verifier_report", {}).get("unsupported_claims") or []
    return {
        "name": case.get("name"),
        "task_type": report.get("task_type") or case.get("task_type"),
        "category": case.get("category"),
        "status": report.get("objective_status", {}).get("status") or verifier.get("status") or "completed",
        "verifier_status": verifier.get("status"),
        "prediction_model_type": prediction.get("model_type"),
        "prediction_label": prediction.get("label"),
        "evidence_count": len(report.get("evidence") or []),
        "citation_grounded": report.get("verifier_report", {}).get("citations", {}).get("citation_grounded"),
        "unsupported_claim_count": len(unsupported_claims),
        "reflection_count": sum(1 for item in trace + tool_calls if str(item.get("agent") or "") == "ReflectionAgent"),
        "n_tool_calls": len(tool_calls),
        "skipped_resume": skipped,
        "latency_sec": round(latency, 3),
        "report_path": report.get("artifacts", {}).get("multi_agent_report_path"),
        "reaction_report_path": report.get("artifacts", {}).get("reaction_report_path"),
        "source_label": case.get("source_label"),
        "source_yield": case.get("source_yield"),
    }


def write_outputs(output_root: Path, rows: list[dict], failures: list[dict]) -> None:
    completed = [row for row in rows if row.get("status") in {"pass", "warning", "completed"}]
    grounded = [row for row in rows if row.get("citation_grounded") is not None]
    summary = {
        "n_cases": len(rows),
        "n_completed_or_warning": len(completed),
        "n_failed_exception": sum(1 for row in rows if row.get("status") == "failed_exception"),
        "n_failures": len(failures),
        "success_rate": len(completed) / len(rows) if rows else 0.0,
        "prediction_availability_rate": sum(1 for row in rows if row.get("prediction_model_type")) / len(rows) if rows else 0.0,
        "citation_grounding_rate": sum(1 for row in grounded if row.get("citation_grounded")) / len(grounded) if grounded else None,
        "mean_evidence_count": sum(float(row.get("evidence_count") or 0.0) for row in rows) / len(rows) if rows else 0.0,
        "mean_latency_sec": sum(float(row.get("latency_sec") or 0.0) for row in rows) / len(rows) if rows else 0.0,
        "total_reflection_count": sum(int(row.get("reflection_count") or 0) for row in rows),
        "total_unsupported_claim_count": sum(int(row.get("unsupported_claim_count") or 0) for row in rows),
        "rows": rows,
    }
    (output_root / "benchmark_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_root / "failure_report.json").write_text(json.dumps(failures, indent=2, ensure_ascii=False), encoding="utf-8")
    with (output_root / "case_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = sorted({key for row in rows for key in row})
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# AgentM Reaction Benchmark Summary", ""]
    for row in rows:
        lines.append(f"- `{row.get('name')}`: {row.get('status')} | evidence={row.get('evidence_count')} | model={row.get('prediction_model_type')}")
    (output_root / "benchmark_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
