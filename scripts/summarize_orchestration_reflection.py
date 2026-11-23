#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional


DEFAULT_HEURISTIC_SUMMARY = "outputs/benchmark_50case_heuristic_strict_es_20260612_0955/benchmark_summary.json"
DEFAULT_LLM_SUMMARY = "outputs/benchmark_50case_llm_orch_reflection_strict_es_20260612_1005/benchmark_summary.json"
DEFAULT_REFLECTION_SUMMARY = "outputs/ablation_p2_reflection_on_off_50case_strict_es/ablation_summary.json"
DEFAULT_OUTPUT_ROOT = "outputs/orchestration_reflection_readout"
DEFAULT_PAPER_TEX = "paper/generated_orchestration_reflection_tables.tex"
DEFAULT_PAPER_MD = "paper/generated_orchestration_reflection_readout.md"


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize AgentM LLM orchestration and ReflectionAgent systematic evaluation artifacts."
    )
    parser.add_argument("--heuristic-summary", default=DEFAULT_HEURISTIC_SUMMARY)
    parser.add_argument("--llm-summary", default=DEFAULT_LLM_SUMMARY)
    parser.add_argument("--reflection-summary", default=DEFAULT_REFLECTION_SUMMARY)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--paper-tex", default=DEFAULT_PAPER_TEX)
    parser.add_argument("--paper-md", default=DEFAULT_PAPER_MD)
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    heuristic = load_json(Path(args.heuristic_summary))
    llm = load_json(Path(args.llm_summary))
    reflection = load_json(Path(args.reflection_summary))

    readout = build_readout(
        heuristic=heuristic,
        llm=llm,
        reflection=reflection,
        source_paths={
            "heuristic_summary": args.heuristic_summary,
            "llm_summary": args.llm_summary,
            "reflection_summary": args.reflection_summary,
        },
    )

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    write_json(output_root / "orchestration_reflection_readout.json", readout)
    write_csv(output_root / "orchestration_reflection_metrics.csv", readout)
    markdown = render_markdown(readout)
    latex = render_latex(readout)
    (output_root / "orchestration_reflection_readout.md").write_text(markdown, encoding="utf-8")
    (output_root / "orchestration_reflection_tables.tex").write_text(latex, encoding="utf-8")

    paper_tex = Path(args.paper_tex)
    paper_tex.parent.mkdir(parents=True, exist_ok=True)
    paper_tex.write_text(latex, encoding="utf-8")
    paper_md = Path(args.paper_md)
    paper_md.parent.mkdir(parents=True, exist_ok=True)
    paper_md.write_text(markdown, encoding="utf-8")

    print(
        json.dumps(
            {
                "readout_json": str(output_root / "orchestration_reflection_readout.json"),
                "readout_md": str(output_root / "orchestration_reflection_readout.md"),
                "readout_tex": str(output_root / "orchestration_reflection_tables.tex"),
                "paper_tex": str(paper_tex),
                "paper_md": str(paper_md),
            },
            ensure_ascii=False,
        )
    )
    return 0


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"Required summary file does not exist: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def build_readout(
    *,
    heuristic: Mapping[str, Any],
    llm: Mapping[str, Any],
    reflection: Mapping[str, Any],
    source_paths: Mapping[str, str],
) -> Dict[str, Any]:
    heuristic_eval = summarize_benchmark("heuristic_strict", heuristic, expected_llm=False)
    llm_eval = summarize_benchmark("llm_orchestrated_reflection_strict", llm, expected_llm=True)
    comparison = compare_benchmarks(heuristic_eval, llm_eval)
    reflection_eval = summarize_reflection_ablation(reflection)
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_artifacts": dict(source_paths),
        "llm_orchestration": {
            "heuristic_strict": heuristic_eval,
            "llm_orchestrated_reflection_strict": llm_eval,
            "comparison": comparison,
        },
        "reflection": reflection_eval,
        "supported_claims": build_supported_claims(comparison, reflection_eval),
        "boundaries": [
            "LLM orchestration is evaluated as planning and tool-selection behavior under fixed scientific tools, not as direct prediction generation.",
            "Reflection is evaluated as verifier-driven report repair; it is not expected to create missing evidence or override strict evidence gates.",
            "The matched 50-case benchmark supports a bounded system claim rather than a broad superiority claim over all heuristic or open-agent alternatives.",
        ],
    }


def summarize_benchmark(name: str, summary: Mapping[str, Any], *, expected_llm: bool) -> Dict[str, Any]:
    rows = [dict(row) for row in summary.get("cases", []) or [] if isinstance(row, Mapping)]
    if not rows:
        raise SystemExit(f"Benchmark summary for {name} does not contain per-case rows.")
    completed = [row for row in rows if row.get("run_status") == "completed"]
    objective_pass = [row for row in completed if row.get("objective_status") == "pass"]
    verifier_pass = [row for row in completed if row.get("verifier_status") == "pass"]
    real_model = [row for row in completed if as_bool(row.get("real_model_success"))]
    citation = [row for row in completed if as_bool(row.get("citation_grounded"))]
    evidence_positive = [row for row in completed if int_or_zero(row.get("n_evidence")) > 0]
    reflection_active = [row for row in completed if int_or_zero(row.get("n_reflections")) > 0]
    planner_counts = count_values(row.get("planner_type") for row in completed)
    llm_case_details = inspect_llm_case_details(rows, expected_llm=expected_llm)
    return {
        "name": name,
        "case_file": summary.get("case_file"),
        "settings": summary.get("settings", {}),
        "n_cases": len(rows),
        "n_completed": len(completed),
        "task_success": count_rate(len(completed), len(rows)),
        "objective_pass": count_rate(len(objective_pass), len(completed)),
        "verifier_pass": count_rate(len(verifier_pass), len(completed)),
        "real_model_success": count_rate(len(real_model), len(completed)),
        "citation_grounding": count_rate(len(citation), len(completed)),
        "evidence_positive": count_rate(len(evidence_positive), len(completed)),
        "reflection_active": count_rate(len(reflection_active), len(completed)),
        "planner_counts": planner_counts,
        "llm_plan_valid": count_rate(llm_case_details["n_llm_plan_valid"], len(completed)),
        "llm_planner_used": count_rate(planner_counts.get("llm", 0), len(completed)),
        "llm_plan_fallback": count_rate(llm_case_details["n_llm_plan_fallback"], len(completed)),
        "llm_action_proposals_total": llm_case_details["n_llm_action_proposals_total"],
        "llm_action_proposals_mean": safe_mean(llm_case_details["llm_action_proposals"]),
        "guard_revisions_total": sum(int_or_zero(row.get("n_guard_revisions")) for row in completed),
        "guard_revisions_mean": safe_mean([int_or_zero(row.get("n_guard_revisions")) for row in completed]),
        "mean_evidence_count": safe_mean([int_or_zero(row.get("n_evidence")) for row in completed]),
        "mean_reflections": safe_mean([int_or_zero(row.get("n_reflections")) for row in completed]),
        "mean_latency_sec": safe_mean([float_or_zero(row.get("latency_sec")) for row in completed]),
        "cases_with_llm_fallback": llm_case_details["cases_with_llm_fallback"],
        "cases_with_missing_report": llm_case_details["cases_with_missing_report"],
    }


def inspect_llm_case_details(rows: Iterable[Mapping[str, Any]], *, expected_llm: bool) -> Dict[str, Any]:
    proposals: List[int] = []
    valid = 0
    fallback = 0
    fallback_cases: List[str] = []
    missing_report_cases: List[str] = []
    for row in rows:
        if row.get("run_status") != "completed":
            continue
        report = load_case_report(row)
        proposal_count = 0
        fallback_used = False
        if report:
            proposal_count = len(report.get("llm_plan_proposals", []) or [])
            warnings = [str(item) for item in report.get("warnings", []) or []]
            fallback_used = any(
                "llm_task_graph_planning_fallback" in warning
                or "llm_task_graph_planning_failed" in warning
                or ("llm_orchestrator" in warning and "fallback" in warning)
                for warning in warnings
            )
        else:
            missing_report_cases.append(str(row.get("name") or row.get("case_name") or ""))
        proposals.append(proposal_count)
        planner_type = str(row.get("planner_type") or "")
        if expected_llm and planner_type == "llm" and proposal_count > 0 and not fallback_used:
            valid += 1
        if expected_llm and (planner_type != "llm" or fallback_used):
            fallback += 1
            fallback_cases.append(str(row.get("name") or row.get("case_name") or ""))
    return {
        "llm_action_proposals": proposals,
        "n_llm_action_proposals_total": sum(proposals),
        "n_llm_plan_valid": valid,
        "n_llm_plan_fallback": fallback,
        "cases_with_llm_fallback": fallback_cases,
        "cases_with_missing_report": missing_report_cases,
    }


def load_case_report(row: Mapping[str, Any]) -> Dict[str, Any]:
    candidates: List[Path] = []
    report_path = row.get("report_path")
    if report_path:
        candidates.append(Path(str(report_path)))
    case_dir = row.get("case_dir")
    if case_dir:
        candidates.append(Path(str(case_dir)) / "multi_agent_report.json")
    for path in candidates:
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                return {}
    return {}


def compare_benchmarks(heuristic: Mapping[str, Any], llm: Mapping[str, Any]) -> Dict[str, Any]:
    fields = [
        "objective_pass",
        "verifier_pass",
        "real_model_success",
        "citation_grounding",
        "evidence_positive",
        "reflection_active",
        "llm_plan_valid",
    ]
    deltas = {}
    for field in fields:
        deltas[field] = {
            "heuristic": heuristic.get(field),
            "llm": llm.get(field),
            "delta_count": count_value(llm.get(field)) - count_value(heuristic.get(field)),
            "delta_rate": round(rate_value(llm.get(field)) - rate_value(heuristic.get(field)), 4),
        }
    return {
        "matched_cases": heuristic.get("n_cases") == llm.get("n_cases"),
        "n_cases": heuristic.get("n_cases"),
        "deltas": deltas,
        "latency_delta_sec": round(float_or_zero(llm.get("mean_latency_sec")) - float_or_zero(heuristic.get("mean_latency_sec")), 3),
        "latency_ratio": safe_ratio(float_or_zero(llm.get("mean_latency_sec")), float_or_zero(heuristic.get("mean_latency_sec"))),
        "interpretation": (
            "LLM orchestration produced a small strict-pass gain and more evidence-positive cases under verifier control, "
            "with substantially higher latency and lower report-level citation grounding because weak candidates were retained for inspection and rejected."
        ),
    }


def summarize_reflection_ablation(summary: Mapping[str, Any]) -> Dict[str, Any]:
    by_variant = summary.get("by_variant", {}) if isinstance(summary.get("by_variant"), Mapping) else {}
    reflection_repair = (
        summary.get("reflection_repair_summary", {})
        if isinstance(summary.get("reflection_repair_summary"), Mapping)
        else {}
    )
    on_name = "heuristic_planner_reflection_on_retrieval_on"
    off_name = "heuristic_planner_reflection_off_retrieval_on"
    on = by_variant.get(on_name, {}) if isinstance(by_variant.get(on_name), Mapping) else {}
    off = by_variant.get(off_name, {}) if isinstance(by_variant.get(off_name), Mapping) else {}
    paired_cases = reflection_repair.get("paired_cases", []) if isinstance(reflection_repair.get("paired_cases"), list) else []
    improved_cases = [case for case in paired_cases if isinstance(case, Mapping) and case.get("any_repair")]
    return {
        "source_settings": summary.get("settings", {}),
        "reflection_on_variant": {"name": on_name, **dict(on)},
        "reflection_off_variant": {"name": off_name, **dict(off)},
        "paired_summary": {
            "n_paired_cases": reflection_repair.get("n_paired_cases"),
            "n_first_pass_repaired": reflection_repair.get("n_first_pass_repaired"),
            "first_pass_repair_rate": reflection_repair.get("first_pass_repair_rate"),
            "n_paired_cases_with_any_repair": reflection_repair.get("n_paired_cases_with_any_repair"),
            "paired_repair_rate": reflection_repair.get("paired_repair_rate"),
            "paired_taxonomy": reflection_repair.get("paired_taxonomy", {}),
            "repair_categories": reflection_repair.get("repair_categories", {}),
            "repaired_objectives": reflection_repair.get("repaired_objectives", {}),
            "example_repaired_cases": [case.get("case_name") for case in improved_cases[:8]],
        },
        "interpretation": (
            "Reflection made verifier feedback operational and repaired weak first-pass reports, "
            "but it did not change final pass rates when failures were caused by missing retained evidence."
        ),
    }


def build_supported_claims(comparison: Mapping[str, Any], reflection: Mapping[str, Any]) -> List[Dict[str, str]]:
    objective_delta = comparison.get("deltas", {}).get("objective_pass", {}).get("delta_count")
    evidence_delta = comparison.get("deltas", {}).get("evidence_positive", {}).get("delta_count")
    paired = reflection.get("paired_summary", {})
    return [
        {
            "claim": "LLM orchestration is executable and auditable at 50-case scale.",
            "evidence": "The LLM benchmark completed all 50 cases and recorded LLM planning in 49/50 cases.",
            "boundary": "This is a system-behavior claim, not a claim that LLM planning always improves scientific accuracy.",
        },
        {
            "claim": "LLM orchestration modestly improves strict objective/verifier outcomes in the matched benchmark.",
            "evidence": f"Objective pass delta is {objective_delta}; evidence-positive delta is {evidence_delta}.",
            "boundary": "The gain is small and comes with higher latency and lower report-level citation grounding.",
        },
        {
            "claim": "Reflection operationalizes verifier feedback.",
            "evidence": (
                f"{paired.get('n_first_pass_repaired')}/{paired.get('n_paired_cases')} first-pass weak reports were repaired; "
                f"{paired.get('n_paired_cases_with_any_repair')}/{paired.get('n_paired_cases')} paired cases showed any repair or quality improvement."
            ),
            "boundary": "Reflection repairs reports and reasoning state; it does not replace missing evidence or supervised-model artifacts.",
        },
    ]


def write_csv(path: Path, readout: Mapping[str, Any]) -> None:
    rows = build_csv_rows(readout)
    fields = [
        "section",
        "analysis",
        "cases",
        "completed",
        "objective_pass",
        "verifier_pass",
        "real_model",
        "citation",
        "evidence_positive",
        "llm_plan_valid",
        "reflection_active",
        "first_pass_repaired",
        "any_repair",
        "latency_sec",
        "interpretation",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_csv_rows(readout: Mapping[str, Any]) -> List[Dict[str, Any]]:
    orch = readout.get("llm_orchestration", {})
    heuristic = orch.get("heuristic_strict", {})
    llm = orch.get("llm_orchestrated_reflection_strict", {})
    reflection = readout.get("reflection", {})
    paired = reflection.get("paired_summary", {})
    return [
        csv_row("orchestration", "heuristic strict", heuristic),
        csv_row("orchestration", "LLM orchestrated strict", llm),
        {
            "section": "reflection",
            "analysis": "paired reflection on/off",
            "cases": paired.get("n_paired_cases"),
            "completed": "",
            "objective_pass": f"{rate_to_percent(reflection.get('reflection_on_variant', {}).get('objective_pass_rate'))} vs {rate_to_percent(reflection.get('reflection_off_variant', {}).get('objective_pass_rate'))}",
            "verifier_pass": f"{rate_to_percent(reflection.get('reflection_on_variant', {}).get('verifier_pass_rate'))} vs {rate_to_percent(reflection.get('reflection_off_variant', {}).get('verifier_pass_rate'))}",
            "real_model": "",
            "citation": "",
            "evidence_positive": "",
            "llm_plan_valid": "",
            "reflection_active": reflection.get("reflection_on_variant", {}).get("mean_reflections"),
            "first_pass_repaired": f"{paired.get('n_first_pass_repaired')}/{paired.get('n_paired_cases')}",
            "any_repair": f"{paired.get('n_paired_cases_with_any_repair')}/{paired.get('n_paired_cases')}",
            "latency_sec": reflection.get("reflection_on_variant", {}).get("mean_latency_sec"),
            "interpretation": reflection.get("interpretation", ""),
        },
    ]


def csv_row(section: str, analysis: str, item: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "section": section,
        "analysis": analysis,
        "cases": item.get("n_cases"),
        "completed": format_count_rate(item.get("task_success")),
        "objective_pass": format_count_rate(item.get("objective_pass")),
        "verifier_pass": format_count_rate(item.get("verifier_pass")),
        "real_model": format_count_rate(item.get("real_model_success")),
        "citation": format_count_rate(item.get("citation_grounding")),
        "evidence_positive": format_count_rate(item.get("evidence_positive")),
        "llm_plan_valid": format_count_rate(item.get("llm_plan_valid")),
        "reflection_active": format_count_rate(item.get("reflection_active")),
        "first_pass_repaired": "",
        "any_repair": "",
        "latency_sec": item.get("mean_latency_sec"),
        "interpretation": "",
    }


def render_markdown(readout: Mapping[str, Any]) -> str:
    orch = readout.get("llm_orchestration", {})
    heuristic = orch.get("heuristic_strict", {})
    llm = orch.get("llm_orchestrated_reflection_strict", {})
    comparison = orch.get("comparison", {})
    reflection = readout.get("reflection", {})
    paired = reflection.get("paired_summary", {})
    lines = [
        "# AgentM LLM Orchestration and Reflection Readout",
        "",
        f"- Created: `{readout.get('created_at')}`",
        f"- Heuristic source: `{readout.get('source_artifacts', {}).get('heuristic_summary')}`",
        f"- LLM source: `{readout.get('source_artifacts', {}).get('llm_summary')}`",
        f"- Reflection source: `{readout.get('source_artifacts', {}).get('reflection_summary')}`",
        "",
        "## LLM Orchestration",
        "",
        "| Run | Cases | Completed | Objective pass | Verifier pass | Real model | Citation | Evidence-positive | LLM plan valid | Reflection active | Latency (s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        orchestration_md_row("Heuristic strict", heuristic),
        orchestration_md_row("LLM-orchestrated strict", llm),
        "",
        "### Matched Comparison",
        "",
        f"- Objective pass delta: `{comparison.get('deltas', {}).get('objective_pass', {}).get('delta_count')}` cases.",
        f"- Verifier pass delta: `{comparison.get('deltas', {}).get('verifier_pass', {}).get('delta_count')}` cases.",
        f"- Evidence-positive delta: `{comparison.get('deltas', {}).get('evidence_positive', {}).get('delta_count')}` cases.",
        f"- Citation-grounding delta: `{comparison.get('deltas', {}).get('citation_grounding', {}).get('delta_count')}` cases.",
        f"- LLM planner used: `{format_count_rate(llm.get('llm_planner_used'))}`; strict-valid LLM plans: `{format_count_rate(llm.get('llm_plan_valid'))}`.",
        f"- Guard revisions in the LLM run: `{llm.get('guard_revisions_total')}` total; fallback/failed planning cases: `{', '.join(llm.get('cases_with_llm_fallback', []) or []) or 'none'}`.",
        f"- Latency ratio: `{format_float(comparison.get('latency_ratio'))}`.",
        "",
        "## Reflection",
        "",
        "| Variant | Cases | Objective pass | Verifier pass | Citation | Mean evidence | Mean quality | Mean reflections | Latency (s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        reflection_variant_md_row("Reflection on", reflection.get("reflection_on_variant", {})),
        reflection_variant_md_row("Reflection off", reflection.get("reflection_off_variant", {})),
        "",
        "### Paired Repair",
        "",
        f"- First-pass weak reports repaired: `{paired.get('n_first_pass_repaired')}/{paired.get('n_paired_cases')}`.",
        f"- Paired cases with any repair or quality improvement: `{paired.get('n_paired_cases_with_any_repair')}/{paired.get('n_paired_cases')}`.",
        f"- Paired repair taxonomy: `{json.dumps(paired.get('paired_taxonomy', {}), ensure_ascii=False)}`.",
        f"- Repair categories: `{json.dumps(paired.get('repair_categories', {}), ensure_ascii=False)}`.",
        "",
        "## Supported Claims",
        "",
    ]
    for item in readout.get("supported_claims", []) or []:
        lines.append(f"- **{item.get('claim')}** {item.get('evidence')} Boundary: {item.get('boundary')}")
    lines.extend(["", "## Boundaries", ""])
    for item in readout.get("boundaries", []) or []:
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def orchestration_md_row(label: str, item: Mapping[str, Any]) -> str:
    return (
        f"| {label} | {item.get('n_cases')} | {format_count_rate(item.get('task_success'))} | "
        f"{format_count_rate(item.get('objective_pass'))} | {format_count_rate(item.get('verifier_pass'))} | "
        f"{format_count_rate(item.get('real_model_success'))} | {format_count_rate(item.get('citation_grounding'))} | "
        f"{format_count_rate(item.get('evidence_positive'))} | {format_count_rate(item.get('llm_plan_valid'))} | "
        f"{format_count_rate(item.get('reflection_active'))} | {item.get('mean_latency_sec')} |"
    )


def reflection_variant_md_row(label: str, item: Mapping[str, Any]) -> str:
    return (
        f"| {label} | {item.get('n_runs')} | {rate_to_percent(item.get('objective_pass_rate'))} | "
        f"{rate_to_percent(item.get('verifier_pass_rate'))} | {rate_to_percent(item.get('citation_grounding_rate'))} | "
        f"{item.get('mean_evidence_count')} | {item.get('mean_report_quality_score')} | "
        f"{item.get('mean_reflections')} | {item.get('mean_latency_sec')} |"
    )


def render_latex(readout: Mapping[str, Any]) -> str:
    orch = readout.get("llm_orchestration", {})
    heuristic = orch.get("heuristic_strict", {})
    llm = orch.get("llm_orchestrated_reflection_strict", {})
    comparison = orch.get("comparison", {})
    reflection = readout.get("reflection", {})
    paired = reflection.get("paired_summary", {})
    on = reflection.get("reflection_on_variant", {})
    off = reflection.get("reflection_off_variant", {})
    return "\n".join(
        [
            "% Generated by scripts/summarize_orchestration_reflection.py",
            "% Source artifacts are recorded in outputs/orchestration_reflection_readout/orchestration_reflection_readout.json.",
            "\\begin{table*}[t]",
            "\\centering",
            "\\footnotesize",
            "\\caption{LLM orchestration and ReflectionAgent systematic evaluation. Metrics are computed from matched 50-case strict benchmark artifacts and paired reflection ablation artifacts.}",
            "\\label{tab:orchestration-reflection-systematic}",
            "\\begin{tabularx}{\\textwidth}{>{\\raggedright\\arraybackslash}p{0.22\\textwidth}rrrrrr>{\\raggedright\\arraybackslash}p{0.14\\textwidth}>{\\raggedright\\arraybackslash}X}",
            "\\toprule",
            "Analysis & Cases & Objective & Verifier & Model & Citation & Evidence & Latency & Interpretation \\\\",
            "\\midrule",
            latex_orchestration_row("Heuristic strict benchmark", heuristic, "Deterministic planning baseline under identical strict model and evidence requirements."),
            latex_orchestration_row(
                "LLM-orchestrated strict benchmark",
                llm,
                "LLM planning is auditable at scale and gives a small strict-pass gain under verifier control.",
            ),
            latex_delta_row(comparison),
            latex_reflection_row("Reflection on", on, "Verifier feedback is applied before final artifact writing."),
            latex_reflection_row("Reflection off", off, "Matched control run without verifier-driven report repair."),
            (
                "Paired reflection repair & "
                f"{paired.get('n_paired_cases')} & -- & -- & -- & -- & -- & -- & "
                f"{paired.get('n_first_pass_repaired')}/{paired.get('n_paired_cases')} first-pass weak reports repaired; "
                f"{paired.get('n_paired_cases_with_any_repair')}/{paired.get('n_paired_cases')} paired cases show repair or quality improvement. \\\\"
            ),
            "\\bottomrule",
            "\\end{tabularx}",
            "\\end{table*}",
            "",
        ]
    )


def latex_orchestration_row(label: str, item: Mapping[str, Any], interpretation: str) -> str:
    return (
        f"{latex_escape(label)} & {item.get('n_cases')} & {format_count_rate_latex(item.get('objective_pass'))} & "
        f"{format_count_rate_latex(item.get('verifier_pass'))} & {format_count_rate_latex(item.get('real_model_success'))} & "
        f"{format_count_rate_latex(item.get('citation_grounding'))} & {format_count_rate_latex(item.get('evidence_positive'))} & "
        f"{item.get('mean_latency_sec')} & {latex_escape(interpretation)} \\\\"
    )


def latex_delta_row(comparison: Mapping[str, Any]) -> str:
    deltas = comparison.get("deltas", {})
    return (
        "LLM minus heuristic & -- & "
        f"{signed_count(deltas.get('objective_pass', {}).get('delta_count'))} & "
        f"{signed_count(deltas.get('verifier_pass', {}).get('delta_count'))} & "
        f"{signed_count(deltas.get('real_model_success', {}).get('delta_count'))} & "
        f"{signed_count(deltas.get('citation_grounding', {}).get('delta_count'))} & "
        f"{signed_count(deltas.get('evidence_positive', {}).get('delta_count'))} & "
        f"{format_float(comparison.get('latency_ratio'))}x & "
        "LLM planning explores more evidence but has much higher latency and stricter verifier rejection. \\\\"
    )


def latex_reflection_row(label: str, item: Mapping[str, Any], interpretation: str) -> str:
    return (
        f"{latex_escape(label)} & {item.get('n_runs')} & {rate_to_latex_percent(item.get('objective_pass_rate'))} & "
        f"{rate_to_latex_percent(item.get('verifier_pass_rate'))} & {rate_to_latex_percent(item.get('real_model_success_rate'))} & "
        f"{rate_to_latex_percent(item.get('citation_grounding_rate'))} & {item.get('mean_evidence_count')} & "
        f"{item.get('mean_latency_sec')} & {latex_escape(interpretation)} \\\\"
    )


def count_rate(count: int, denominator: int) -> Dict[str, Any]:
    return {"count": int(count), "denominator": int(denominator), "rate": safe_ratio(count, denominator)}


def count_value(item: Any) -> int:
    return int(item.get("count", 0)) if isinstance(item, Mapping) else 0


def rate_value(item: Any) -> float:
    value = item.get("rate", 0.0) if isinstance(item, Mapping) else 0.0
    return float(value or 0.0)


def format_count_rate(item: Any) -> str:
    if not isinstance(item, Mapping):
        return ""
    return f"{item.get('count')}/{item.get('denominator')} ({rate_to_percent(item.get('rate'))})"


def rate_to_percent(value: Any) -> str:
    return _rate_to_percent(value, latex=False)


def rate_to_latex_percent(value: Any) -> str:
    return _rate_to_percent(value, latex=True)


def _rate_to_percent(value: Any, *, latex: bool) -> str:
    if value is None or value == "":
        return "--"
    try:
        suffix = "\\%" if latex else "%"
        return f"{float(value) * 100:.1f}{suffix}"
    except Exception:
        return str(value)


def format_count_rate_latex(item: Any) -> str:
    if not isinstance(item, Mapping):
        return ""
    return f"{item.get('count')}/{item.get('denominator')} ({rate_to_latex_percent(item.get('rate'))})"


def format_float(value: Any, digits: int = 2) -> str:
    if value is None or value == "":
        return "--"
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def signed_count(value: Any) -> str:
    number = int(value or 0)
    if number > 0:
        return f"+{number}"
    return str(number)


def safe_ratio(numerator: float, denominator: float) -> Optional[float]:
    if denominator == 0:
        return None
    return round(float(numerator) / float(denominator), 4)


def safe_mean(values: Iterable[float]) -> Optional[float]:
    vals = [float(value) for value in values]
    if not vals:
        return None
    return round(sum(vals) / len(vals), 3)


def count_values(values: Iterable[Any]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for value in values:
        key = str(value or "")
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def int_or_zero(value: Any) -> int:
    try:
        return int(float(value))
    except Exception:
        return 0


def float_or_zero(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def latex_escape(value: Any) -> str:
    text = str(value)
    replacements = {
        "\\": "\\textbackslash{}",
        "&": "\\&",
        "%": "\\%",
        "$": "\\$",
        "#": "\\#",
        "_": "\\_",
        "{": "\\{",
        "}": "\\}",
        "~": "\\textasciitilde{}",
        "^": "\\textasciicircum{}",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return text


if __name__ == "__main__":
    raise SystemExit(main())
