#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence


DEFAULT_OUTPUT_ROOT = Path("outputs/balanced100_result_summary_20260701")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize runtime balanced100 ADMET/DDI benchmark and evidence-audit results.")
    parser.add_argument("--heuristic-summary", required=True)
    parser.add_argument("--llm-summary", required=True)
    parser.add_argument("--llm-retry-summary", default="")
    parser.add_argument("--acceptance-json", required=True)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    heuristic = read_json(Path(args.heuristic_summary))
    llm_initial = read_json(Path(args.llm_summary))
    llm_retry = read_json(Path(args.llm_retry_summary)) if args.llm_retry_summary else {}
    acceptance = read_json(Path(args.acceptance_json))

    heuristic_rows = rows_from_summary(heuristic)
    llm_initial_rows = rows_from_summary(llm_initial)
    llm_postretry_rows = merge_retry_rows(llm_initial_rows, rows_from_summary(llm_retry))

    runs = {
        "heuristic_strict": {
            "summary_path": args.heuristic_summary,
            "rows": heuristic_rows,
            "settings": heuristic.get("settings") or {},
        },
        "llm_orchestrated_initial": {
            "summary_path": args.llm_summary,
            "rows": llm_initial_rows,
            "settings": llm_initial.get("settings") or {},
        },
        "llm_orchestrated_postretry": {
            "summary_path": args.llm_summary,
            "retry_summary_path": args.llm_retry_summary or None,
            "rows": llm_postretry_rows,
            "settings": llm_initial.get("settings") or {},
        },
    }

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "case_file": heuristic.get("case_file") or llm_initial.get("case_file"),
        "runs": {
            name: {
                key: value
                for key, value in {
                    "summary_path": payload.get("summary_path"),
                    "retry_summary_path": payload.get("retry_summary_path"),
                    "settings": payload.get("settings"),
                    "combined": summarize_rows(payload["rows"]),
                    "admet": summarize_rows([row for row in payload["rows"] if row.get("category") == "admet"]),
                    "ddi": summarize_rows([row for row in payload["rows"] if row.get("category") == "ddi"]),
                    "failure_modes": failure_modes(payload["rows"]),
                }.items()
                if value is not None
            }
            for name, payload in runs.items()
        },
        "acceptance": summarize_acceptance(acceptance, args.acceptance_json),
    }

    write_json(output_root / "balanced100_summary.json", summary)
    (output_root / "balanced100_summary.md").write_text(render_markdown(summary), encoding="utf-8")
    write_csv(output_root / "heuristic_strict_case_metrics.csv", heuristic_rows)
    write_csv(output_root / "llm_orchestrated_initial_case_metrics.csv", llm_initial_rows)
    write_csv(output_root / "llm_orchestrated_postretry_case_metrics.csv", llm_postretry_rows)
    print(json.dumps({"summary_json": str(output_root / "balanced100_summary.json"), "summary_md": str(output_root / "balanced100_summary.md")}, indent=2))
    return 0


def read_json(path: Path) -> Dict[str, Any]:
    if not path:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def rows_from_summary(summary: Mapping[str, Any]) -> List[Dict[str, Any]]:
    rows = summary.get("cases") or []
    return [dict(row) for row in rows if isinstance(row, Mapping)]


def merge_retry_rows(rows: Sequence[Mapping[str, Any]], retry_rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    replacement = {str(row.get("name") or ""): dict(row) for row in retry_rows if row.get("name")}
    merged: List[Dict[str, Any]] = []
    for row in rows:
        name = str(row.get("name") or "")
        merged.append(dict(replacement.get(name, row)))
    return merged


def summarize_rows(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    completed = [row for row in rows if row.get("run_status") == "completed"]
    objective_pass = [row for row in completed if row.get("objective_status") == "pass"]
    verifier_pass = [row for row in completed if row.get("verifier_status") == "pass"]
    latencies = [float(row.get("latency_sec") or 0.0) for row in completed if row.get("latency_sec") not in {"", None}]
    return {
        "n_cases": len(rows),
        "n_completed": len(completed),
        "n_failed_exception": len([row for row in rows if row.get("run_status") == "failed_exception"]),
        "objective_pass": len(objective_pass),
        "objective_pass_rate": ratio(len(objective_pass), len(rows)),
        "verifier_pass": len(verifier_pass),
        "verifier_pass_rate": ratio(len(verifier_pass), len(completed)),
        "real_model_success": len([row for row in completed if truthy(row.get("real_model_success"))]),
        "real_model_success_rate": ratio(len([row for row in completed if truthy(row.get("real_model_success"))]), len(completed)),
        "citation_grounded": len([row for row in completed if row.get("citation_grounded") is True]),
        "citation_grounding_rate": ratio(len([row for row in completed if row.get("citation_grounded") is True]), len(completed)),
        "compound_specific_grounded": len([row for row in completed if row.get("compound_specific_grounded") is True]),
        "endpoint_specific_grounded": len([row for row in completed if row.get("endpoint_specific_grounded") is True]),
        "mean_latency_sec": mean(latencies),
        "planner_type_counts": dict(Counter(str(row.get("planner_type") or "missing") for row in completed)),
        "total_reflections": sum(int(row.get("n_reflections") or 0) for row in completed),
        "mean_reflections": mean([float(row.get("n_reflections") or 0) for row in completed]),
    }


def failure_modes(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    failures = [row for row in rows if row.get("run_status") == "failed_exception" or row.get("objective_status") != "pass"]
    categories = Counter()
    details = []
    for row in failures:
        warnings = [str(item) for item in row.get("warnings", []) or []]
        if row.get("run_status") == "failed_exception":
            categories["failed_exception"] += 1
        if row.get("objective_status") == "timeout":
            categories["case_timeout"] += 1
        if any("llm_orchestrator_fallback" in warning and "Read timed out" in warning for warning in warnings):
            categories["llm_provider_read_timeout"] += 1
        if any("drug" in warning.lower() and "resolution" in warning.lower() for warning in warnings):
            categories["drug_resolution_failure"] += 1
        if any("feature" in warning.lower() and "failed" in warning.lower() for warning in warnings):
            categories["feature_build_failure"] += 1
        if any("evidence_count_minimum" in warning for warning in warnings) or int(row.get("n_evidence") or 0) == 0:
            categories["no_runtime_evidence"] += 1
        if row.get("verifier_status") == "fail":
            categories["verifier_failure"] += 1
        details.append(
            {
                "name": row.get("name"),
                "category": row.get("category"),
                "objective_status": row.get("objective_status"),
                "verifier_status": row.get("verifier_status"),
                "n_evidence": row.get("n_evidence"),
                "warnings": warnings,
                "error": row.get("error"),
            }
        )
    return {"n_failures": len(failures), "category_counts": dict(categories), "details": details}


def summarize_acceptance(acceptance: Mapping[str, Any], path: str) -> Dict[str, Any]:
    metrics = acceptance.get("metrics") if isinstance(acceptance.get("metrics"), Mapping) else {}
    case_reports = [case for case in acceptance.get("cases", []) or [] if isinstance(case, Mapping)]
    endpoint_counts = Counter("ddi" if case.get("endpoint") == "ddi" else "admet" for case in case_reports)
    status_by_category: Dict[str, Counter[str]] = {"admet": Counter(), "ddi": Counter()}
    for case in case_reports:
        category = "ddi" if case.get("endpoint") == "ddi" else "admet"
        status_by_category[category].update([str(case.get("status") or "missing")])
    return {
        "summary_path": path,
        "status": acceptance.get("status"),
        "n_cases": metrics.get("n_cases"),
        "case_status_counts": metrics.get("case_status_counts") or {},
        "dense_completed_cases": metrics.get("dense_completed_cases"),
        "source_counts": metrics.get("source_counts") or {},
        "evidence_scope_counts": metrics.get("evidence_scope_counts") or {},
        "category_counts": dict(endpoint_counts),
        "status_by_category": {key: dict(value) for key, value in status_by_category.items()},
    }


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def render_markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# runtime Balanced100 Result Summary",
        "",
        f"- Created: `{summary.get('created_at')}`",
        f"- Case file: `{summary.get('case_file')}`",
        "",
        "## Evidence Acceptance",
        "",
    ]
    acceptance = summary.get("acceptance", {})
    lines.extend(
        [
            f"- Status: `{acceptance.get('status')}`",
            f"- Cases: `{acceptance.get('n_cases')}`",
            f"- Dense completed: `{acceptance.get('dense_completed_cases')}`",
            f"- Case status counts: `{acceptance.get('case_status_counts')}`",
            f"- Status by category: `{acceptance.get('status_by_category')}`",
            f"- Evidence scopes: `{acceptance.get('evidence_scope_counts')}`",
            "",
            "## Benchmark Runs",
            "",
            "| Run | Split | Cases | Objective pass | Verifier pass | Real model | Citation grounded | Mean latency sec | Failures |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for run_name, payload in (summary.get("runs") or {}).items():
        for split in ["combined", "admet", "ddi"]:
            stats = payload.get(split, {})
            failures = payload.get("failure_modes", {}).get("n_failures") if split == "combined" else ""
            lines.append(
                "| {run} | {split} | {cases} | {obj}/{cases} ({obj_rate}) | {ver}/{completed} ({ver_rate}) | {model}/{completed} ({model_rate}) | {cit}/{completed} ({cit_rate}) | {latency} | {failures} |".format(
                    run=run_name,
                    split=split,
                    cases=stats.get("n_cases"),
                    completed=stats.get("n_completed"),
                    obj=stats.get("objective_pass"),
                    obj_rate=stats.get("objective_pass_rate"),
                    ver=stats.get("verifier_pass"),
                    ver_rate=stats.get("verifier_pass_rate"),
                    model=stats.get("real_model_success"),
                    model_rate=stats.get("real_model_success_rate"),
                    cit=stats.get("citation_grounded"),
                    cit_rate=stats.get("citation_grounding_rate"),
                    latency=stats.get("mean_latency_sec"),
                    failures=failures,
                )
            )
    lines.append("")
    lines.append("## Failure Modes")
    lines.append("")
    for run_name, payload in (summary.get("runs") or {}).items():
        failure = payload.get("failure_modes", {})
        lines.append(f"- `{run_name}`: `{failure.get('category_counts')}`")
        for detail in failure.get("details", []) or []:
            lines.append(f"  - `{detail.get('name')}`: objective=`{detail.get('objective_status')}`, verifier=`{detail.get('verifier_status')}`, warnings=`{detail.get('warnings')}`")
    lines.append("")
    return "\n".join(lines)


def ratio(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return round(float(numerator) / float(denominator), 4)


def mean(values: Iterable[float]) -> float | None:
    vals = [float(value) for value in values]
    if not vals:
        return None
    return round(sum(vals) / len(vals), 3)


def truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "pass"}


if __name__ == "__main__":
    raise SystemExit(main())
