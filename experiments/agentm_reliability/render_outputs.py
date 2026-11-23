#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import warnings
from typing import Any, Callable, Iterable, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# This directory intentionally contains the required experiment entry point
# `statistics.py`. When this file is executed directly, that filename would
# otherwise shadow Python's standard-library `statistics` module imported by
# seaborn. Preload the standard module with the script directory temporarily
# removed, then restore the path for our direct-script imports below.
_SCRIPT_DIR = str(Path(__file__).resolve().parent)
_removed_script_paths = [item for item in sys.path if Path(item or ".").resolve() == Path(_SCRIPT_DIR)]
sys.path[:] = [item for item in sys.path if Path(item or ".").resolve() != Path(_SCRIPT_DIR)]
import statistics as _stdlib_statistics  # noqa: F401
sys.path[:0] = _removed_script_paths

import seaborn as sns

try:
    from .common import ensure_output_layout, read_json, resolve_path, stable_hash, utc_now, write_csv
except ImportError:
    from common import ensure_output_layout, read_json, resolve_path, stable_hash, utc_now, write_csv


matplotlib.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "xtick.color": "black",
        "ytick.color": "black",
        "axes.labelcolor": "black",
    }
)


COLORS = {
    "heuristic_off": "#7884B4",
    "heuristic_on": "#484878",
    "llm_off": "#E4CCD8",
    "llm_on": "#B85B79",
    "accent": "#2F8F83",
    "warning": "#C47A44",
    "neutral": "#A8A8A8",
    "dark": "#30303A",
}

SETTING_ORDER = [
    ("heuristic", False, "Heuristic\nReflection off", COLORS["heuristic_off"]),
    ("heuristic", True, "Heuristic\nReflection on", COLORS["heuristic_on"]),
    ("llm", False, "LLM\nReflection off", COLORS["llm_off"]),
    ("llm", True, "LLM\nReflection on", COLORS["llm_on"]),
]

# Metric-specific denominators are part of the prespecified statistical plan.
# They must also be enforced in every descriptive table and figure so that a
# non-applicable outcome is never rendered as a negative outcome.
METRIC_APPLICABILITY = {
    "repaired": "repair_applicable",
    "safe_abstention": "safe_abstention_applicable",
    "false_repair": "false_repair_applicable",
    "conflict_detected": "conflict_applicable",
    "conflicting_sources_preserved": "conflict_applicable",
    "forced_conclusion": "forced_conclusion_applicable",
    "uncertainty_expression_correct": "uncertainty_expression_applicable",
    "citation_precision": "citation_precision_applicable",
    "citation_recall": "citation_recall_applicable",
    "unsupported_claim_rate": "unsupported_claim_rate_applicable",
}


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def _bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.astype(float)
    return series.map({True: 1.0, False: 0.0, "True": 1.0, "False": 0.0, "true": 1.0, "false": 0.0, 1: 1.0, 0: 0.0})


def _setting_mask(frame: pd.DataFrame, orchestrator: str, reflection: bool) -> pd.Series:
    if frame.empty or "orchestrator" not in frame or "reflection_enabled" not in frame:
        return pd.Series(False, index=frame.index)
    reflection_values = _bool_series(frame["reflection_enabled"])
    return (frame["orchestrator"] == orchestrator) & (reflection_values == float(reflection))


def _metric_scope(frame: pd.DataFrame, metric: str) -> pd.DataFrame:
    applicability = METRIC_APPLICABILITY.get(metric)
    if not applicability:
        return frame
    if applicability not in frame:
        return frame.iloc[0:0]
    return frame[_bool_series(frame[applicability]) == 1]


def _primary_clean(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    if "analysis_role" in frame:
        return frame[frame["analysis_role"] == "primary"].copy()
    if "repeat_index" in frame:
        repeats = pd.to_numeric(frame["repeat_index"], errors="coerce").fillna(0)
        return frame[repeats == 0].copy()
    return frame.copy()


def _bootstrap_ci(values: np.ndarray, *, seed: int = 2026, resamples: int = 10000) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return math.nan, math.nan
    # Deterministic bootstrap resampling of observed case-level values; this
    # does not generate simulated scientific observations.
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(resamples, len(values)))].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _save(fig: plt.Figure, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="This figure includes Axes that are not compatible with tight_layout")
        fig.tight_layout(pad=1.0)
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def _placeholder(base: Path, message: str) -> None:
    fig, ax = plt.subplots(figsize=(3.5, 2.1))
    ax.axis("off")
    ax.text(0.5, 0.56, "No qualifying completed runs", ha="center", va="center", weight="bold", fontsize=8)
    ax.text(0.5, 0.38, message, ha="center", va="center", fontsize=6.5, color=COLORS["dark"], wrap=True)
    _save(fig, base)


def _rate_points(ax: plt.Axes, frame: pd.DataFrame, metrics: list[tuple[str, str]], *, filters: Callable[[pd.DataFrame], pd.DataFrame] | None = None) -> list[dict[str, Any]]:
    data = filters(frame.copy()) if filters else frame.copy()
    source_rows = []
    offsets = np.linspace(-0.16, 0.16, len(metrics)) if len(metrics) > 1 else [0.0]
    markers = ["o", "s", "D", "^"]
    active_settings = []
    for setting in SETTING_ORDER:
        orchestrator, reflection, _, _ = setting
        subset = data[_setting_mask(data, orchestrator, reflection)]
        if any(
            metric in subset and _bool_series(_metric_scope(subset, metric)[metric]).notna().any()
            for metric, _ in metrics
        ):
            active_settings.append(setting)
    for setting_index, (orchestrator, reflection, label, color) in enumerate(active_settings):
        subset = data[_setting_mask(data, orchestrator, reflection)]
        for metric_index, (metric, display) in enumerate(metrics):
            if metric not in subset:
                continue
            n_before = len(subset)
            metric_subset = _metric_scope(subset, metric)
            n_applicable = len(metric_subset)
            values = _bool_series(metric_subset[metric]).dropna().to_numpy(dtype=float)
            n_after = len(values)
            if len(values) == 0:
                continue
            mean = float(values.mean())
            low, high = _bootstrap_ci(values, seed=2026 + setting_index * 31 + metric_index)
            x = setting_index + offsets[metric_index]
            ax.errorbar(
                x,
                mean,
                yerr=[[mean - low], [high - mean]],
                fmt=markers[metric_index % len(markers)],
                ms=4.2,
                lw=1.0,
                capsize=2.2,
                color=color,
                markeredgecolor="white",
                markeredgewidth=0.5,
                label=display if setting_index == 0 else None,
                zorder=3,
            )
            source_rows.append(
                {
                    "orchestrator": orchestrator,
                    "reflection_enabled": reflection,
                    "metric": metric,
                    "n_input_rows_before_metric_filter": n_before,
                    "n_applicable_rows": n_applicable,
                    "n_excluded_not_applicable": n_before - n_applicable,
                    "n_cases": n_after,
                    "n_excluded_missing_metric": n_applicable - n_after,
                    "mean": mean,
                    "ci95_low": low,
                    "ci95_high": high,
                }
            )
    ax.set_xticks(range(len(active_settings)))
    ax.set_xticklabels([item[2] for item in active_settings], color="black")
    ax.set_ylim(-0.03, 1.05)
    ax.set_ylabel("Case-level rate")
    ax.grid(axis="y", color="#E8E8EC", linewidth=0.6, zorder=0)
    if len(metrics) > 1:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=len(metrics), handletextpad=0.4, columnspacing=1.1)
    return source_rows


def _plot_rate(frame: pd.DataFrame, base: Path, metrics: list[tuple[str, str]], *, filters=None, note: str) -> None:
    if frame.empty or not any(metric in frame for metric, _ in metrics):
        _placeholder(base, note)
        return
    fig, ax = plt.subplots(figsize=(4.4, 2.7))
    rows = _rate_points(ax, frame, metrics, filters=filters)
    if not rows:
        plt.close(fig)
        _placeholder(base, note)
        return
    write_csv(base.parent / f"source_data_{base.name}.csv", rows)
    _save(fig, base)


def _plot_latency(frame: pd.DataFrame, base: Path) -> None:
    if frame.empty or "latency_seconds" not in frame:
        _placeholder(base, "Latency and LLM-call data will be drawn only from completed case-level records.")
        return
    fig, axes = plt.subplots(1, 2, figsize=(6.2, 2.55), gridspec_kw={"wspace": 0.34})
    source_rows = []
    compact_label = {
        ("heuristic", False): "H − R",
        ("heuristic", True): "H + R",
        ("llm", False): "LLM − R",
        ("llm", True): "LLM + R",
    }
    for axis, metric, ylabel in [
        (axes[0], "latency_seconds", "Latency (s)"),
        (axes[1], "llm_call_count", "LLM calls per case"),
    ]:
        plot_rows = []
        labels, palette = [], []
        for orchestrator, reflection, label, color in SETTING_ORDER:
            subset = frame[_setting_mask(frame, orchestrator, reflection)]
            n_before = len(subset)
            values = pd.to_numeric(subset.get(metric, pd.Series(dtype=float)), errors="coerce").dropna()
            n_after = len(values)
            if len(values):
                display = compact_label[(orchestrator, reflection)]
                for value in values:
                    plot_rows.append({"setting": display, "value": value})
                labels.append(display)
                palette.append(color)
            source_rows.extend(
                {
                    "metric": metric,
                    "orchestrator": orchestrator,
                    "reflection_enabled": reflection,
                    "n_input_rows_before_metric_filter": n_before,
                    "n_rows_after_metric_filter": n_after,
                    "n_excluded_missing_metric": n_before - n_after,
                    "value": float(value),
                }
                for value in values
            )
        plot_frame = pd.DataFrame(plot_rows)
        if plot_frame.empty:
            axis.axis("off")
            continue
        sns.boxplot(
            data=plot_frame,
            x="setting",
            y="value",
            hue="setting",
            order=labels,
            palette=dict(zip(labels, palette)),
            width=0.52,
            linewidth=0.8,
            fliersize=0,
            whis=(5, 95),
            legend=False,
            ax=axis,
        )
        sns.stripplot(data=plot_frame, x="setting", y="value", order=labels, color=COLORS["dark"], size=2.0, jitter=0.16, alpha=0.45, ax=axis)
        axis.set_xlabel("")
        axis.set_ylabel(ylabel)
        axis.set_xticks(range(len(labels)))
        axis.set_xticklabels(labels, rotation=0, rotation_mode="anchor", color="black")
        axis.grid(axis="y", color="#ECECF0", linewidth=0.5)
    write_csv(base.parent / f"source_data_{base.name}.csv", source_rows)
    _save(fig, base)


def _plot_open(frame: pd.DataFrame, base: Path) -> None:
    if frame.empty or "objective_coverage" not in frame:
        _placeholder(base, "Open-task objective coverage requires real heuristic and LLM runs.")
        return
    subset = frame[_bool_series(frame["reflection_enabled"]) == 0]
    subset = subset[subset["orchestrator"].isin(["heuristic", "llm"])].copy()
    n_before = len(subset)
    subset["Objective coverage"] = pd.to_numeric(subset["objective_coverage"], errors="coerce")
    subset["Orchestrator"] = subset["orchestrator"].map({"heuristic": "Heuristic", "llm": "LLM"})
    # Missing/non-numeric objective coverage is not an independent observation;
    # source data preserve the complete case-level metrics and this panel uses
    # only rows for which its displayed metric is defined.
    subset = subset.dropna(subset=["Objective coverage", "Orchestrator"])
    n_after = len(subset)
    if subset.empty:
        _placeholder(base, "No valid open-task coverage values were available.")
        return
    fig, ax = plt.subplots(figsize=(3.4, 2.65))
    palette = {"Heuristic": COLORS["heuristic_off"], "LLM": COLORS["llm_off"]}
    sns.boxplot(data=subset, x="Orchestrator", y="Objective coverage", hue="Orchestrator", palette=palette, width=0.46, linewidth=0.9, fliersize=0, whis=(5, 95), legend=False, ax=ax)
    sns.stripplot(data=subset, x="Orchestrator", y="Objective coverage", color=COLORS["dark"], size=2.5, alpha=0.55, jitter=0.14, ax=ax)
    ax.set_ylim(-0.03, 1.05)
    ax.set_xlabel("")
    ax.set_ylabel("Objective coverage")
    ax.grid(axis="y", color="#ECECF0", linewidth=0.5)
    source_columns = [
        column
        for column in (
            "case_id",
            "open_task_type",
            "orchestrator",
            "reflection_enabled",
            "Objective coverage",
            "evaluation_valid",
        )
        if column in subset.columns
    ]
    source_rows = subset[source_columns].to_dict("records")
    for row in source_rows:
        row.update(
            {
                "n_input_rows_before_metric_filter": n_before,
                "n_rows_after_metric_filter": n_after,
                "n_excluded_missing_metric": n_before - n_after,
            }
        )
    write_csv(base.parent / f"source_data_{base.name}.csv", source_rows)
    _save(fig, base)


def _plot_failure_heatmap(frame: pd.DataFrame, base: Path) -> None:
    metrics = ["detected", "repaired", "safe_abstention", "false_repair", "forced_conclusion"]
    if frame.empty or "fault_type" not in frame:
        _placeholder(base, "Failure-mode rates require independently evaluated stress runs.")
        return
    valid = frame[_bool_series(frame.get("evaluation_valid", pd.Series(index=frame.index, dtype=float))) == 1].copy()
    rows = []
    for fault, group in valid.groupby("fault_type"):
        row = {"fault_type": fault}
        for metric in metrics:
            n_before = len(group)
            metric_group = _metric_scope(group, metric)
            values = _bool_series(metric_group[metric]).dropna() if metric in metric_group else pd.Series(dtype=float)
            row[metric] = float(values.mean()) if len(values) else math.nan
            row[f"{metric}__n"] = len(values)
            row[f"{metric}__n_excluded_not_applicable"] = n_before - len(metric_group)
            row[f"{metric}__n_excluded_missing"] = len(metric_group) - len(values)
        rows.append(row)
    source_frame = pd.DataFrame(rows).set_index("fault_type") if rows else pd.DataFrame()
    matrix_frame = source_frame[metrics] if not source_frame.empty else pd.DataFrame()
    if matrix_frame.empty:
        _placeholder(base, "No valid stress evaluations were available.")
        return
    fig_height = max(1.9, 0.36 * len(matrix_frame) + 0.9)
    fig, ax = plt.subplots(figsize=(6.3, fig_height))
    sns.heatmap(matrix_frame, vmin=0, vmax=1, cmap="crest", annot=True, fmt=".2f", linewidths=0.4, linecolor="white", cbar_kws={"label": "Case-level rate"}, ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticklabels(
        [label.replace("_", " ") for label in matrix_frame.columns],
        rotation=24,
        rotation_mode="anchor",
        ha="right",
        color="black",
    )
    ax.set_yticklabels(
        [label.replace("_", " ") for label in matrix_frame.index],
        rotation=0,
        rotation_mode="anchor",
        color="black",
    )
    write_csv(base.parent / f"source_data_{base.name}.csv", source_frame.reset_index().to_dict("records"))
    _save(fig, base)


def _md_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "No qualifying completed runs were available; no numeric values were generated.\n"
    columns = list(frame.columns)
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for _, row in frame.iterrows():
        values = []
        for value in row:
            if pd.isna(value):
                values.append("NA")
            elif isinstance(value, float):
                values.append(f"{value:.3f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def _setting_summary(frame: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    rows = []
    if frame.empty:
        return pd.DataFrame()
    for orchestrator, reflection, label, _ in SETTING_ORDER:
        group = frame[_setting_mask(frame, orchestrator, reflection)]
        if group.empty:
            continue
        row: dict[str, Any] = {"Setting": label.replace("\n", " + "), "n runs": len(group), "n cases": group.get("case_id", pd.Series(dtype=str)).nunique()}
        for metric in metrics:
            if metric not in group:
                continue
            metric_group = _metric_scope(group, metric)
            if metric in {"latency_seconds", "llm_call_count", "tool_call_count", "objective_coverage", "stale_result_removal_accuracy"}:
                values = pd.to_numeric(metric_group[metric], errors="coerce").dropna()
            else:
                values = _bool_series(metric_group[metric]).dropna()
            row[f"{metric} n"] = len(values)
            row[metric] = float(values.mean()) if len(values) else math.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _report_quality_summary(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    metrics = [
        "scientific_object_alignment",
        "evidence_relevance",
        "citation_precision",
        "citation_recall",
        "unsupported_claim_rate",
        "prediction_fact_separation_accuracy",
        "uncertainty_expression_accuracy",
        "report_schema_completeness",
        "external_evidence_coverage",
        "direct_evidence_coverage",
        "qc_only_rate",
    ]
    binary_metrics = {
        "scientific_object_alignment",
        "prediction_fact_separation_accuracy",
        "uncertainty_expression_accuracy",
        "external_evidence_coverage",
        "direct_evidence_coverage",
    }
    rows: list[dict[str, Any]] = []
    group_keys = [key for key in ("dataset_group", "orchestrator", "reflection_enabled") if key in frame]
    grouped = frame.groupby(group_keys, dropna=False) if group_keys else [((), frame)]
    for keys, group in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_keys, keys))
        row["n runs"] = len(group)
        row["n cases"] = group["case_id"].nunique() if "case_id" in group else len(group)
        for metric in metrics:
            if metric not in group:
                continue
            values = (
                _bool_series(group[metric]).dropna()
                if metric in binary_metrics
                else pd.to_numeric(group[metric], errors="coerce").dropna()
            )
            row[f"{metric} n"] = len(values)
            row[metric] = float(values.mean()) if len(values) else math.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _write_tables(
    layout: Mapping[str, Path],
    frames: Mapping[str, pd.DataFrame],
    report_quality: pd.DataFrame,
    *,
    protocol_note: str = "",
) -> None:
    specs = {
        "table_clean_regression.md": (
            "Clean molecular-safety execution regression",
            _setting_summary(frames["clean"], ["completed", "objective_passed", "verifier_passed", "latency_seconds", "llm_call_count", "tool_call_count"]),
            "Rates are case-level proportions; latency and call counts are case-level means. Internal verifier pass is reported only as a process outcome, not scientific truth.",
        ),
        "table_stress_main.md": (
            "Independent controlled-stress evaluation",
            _setting_summary(frames["stress"], ["detected", "repaired", "safe_abstention", "false_repair", "forced_conclusion", "latency_seconds"]),
            "Detection and outcome fields are produced by the independent offline evaluator. Metric-specific denominators are shown explicitly: repair uses recoverable/applicable faults, safe abstention uses unrecoverable/applicable faults, and false repair uses reflection-executed applicable cases.",
        ),
        "table_open_task_orchestration.md": (
            "Open-task orchestration",
            _setting_summary(frames["open_task"], ["objective_coverage", "tool_selection_accuracy", "dependency_order_accuracy", "recovery_success", "latency_seconds", "llm_call_count"]),
            "Each case is the independent unit. Core comparison uses reflection off.",
        ),
        "table_multi_turn_state.md": (
            "Multi-turn state updating",
            _setting_summary(frames["multi_turn"], ["state_consistency", "stale_result_removal_accuracy", "replanning_success", "object_update_accuracy", "cross_turn_contamination", "latency_seconds"]),
            "State metrics inspect active final-context fields and exclude the archived previous-run context itself.",
        ),
    }
    stress = frames["stress"]
    if not stress.empty and "fault_type" in stress:
        rows = []
        for fault, group in stress.groupby("fault_type"):
            row = {"Fault": fault, "n runs": len(group)}
            for metric in ["detected", "repaired", "safe_abstention", "false_repair", "forced_conclusion", "uncertainty_expression_correct"]:
                metric_group = _metric_scope(group, metric)
                values = _bool_series(metric_group[metric]).dropna() if metric in metric_group else pd.Series(dtype=float)
                row[f"{metric} n"] = len(values)
                row[metric] = float(values.mean()) if len(values) else math.nan
            rows.append(row)
        by_fault = pd.DataFrame(rows)
    else:
        by_fault = pd.DataFrame()
    specs["table_stress_by_fault.md"] = (
        "Stress outcomes by injected fault",
        by_fault,
        "Rates are descriptive over metric-applicable real runs for each fault; the displayed n is the exact denominator. See statistical_tests.json for paired inference.",
    )
    specs["table_report_quality_audit.md"] = (
        "Gold-standard offline scientific report-quality audit",
        _report_quality_summary(report_quality),
        "Values are case-level means from deterministic rules and frozen gold records. External evidence excludes benchmark QC grounding; direct evidence additionally requires target-specific source content. Non-stress citation precision/recall audit structural citation linkage, not semantic scientific support; unsupported-claim rate is reported only where a frozen claim--citation gold map makes it deterministically estimable. Internal Verifier pass is not used as scientific truth.",
    )
    for filename, (title, table, note) in specs.items():
        banner = f"> **{protocol_note}**\n\n" if protocol_note else ""
        text = f"# {title}\n\n{banner}{_md_table(table)}\nNote: {note}\n"
        (layout["tables"] / filename).write_text(text, encoding="utf-8")


def _case_text(title: str, row: Mapping[str, Any] | None, layout: Mapping[str, Path], reason: str) -> str:
    if row is None:
        return f"# {title}\n\nNo qualifying real run was observed. {reason}\n"
    run_id = str(row.get("run_id"))
    evaluation = read_json(layout["metrics"] / "evaluations" / f"{run_id}.json", default={}) or {}
    report = read_json(layout["raw_runs"] / run_id / "runtime" / "multi_agent_report.json", default={}) or {}
    excerpt = json.dumps(
        {key: report.get(key) for key in ["request", "task_type", "predictions", "evidence", "explanation", "warnings"] if key in report},
        ensure_ascii=False,
        indent=2,
    )[:4000]
    return (
        f"# {title}\n\n"
        f"Observed in case `{row.get('case_id')}` under `{row.get('orchestrator')}`, reflection={row.get('reflection_enabled')}.\n\n"
        f"Run ID: `{run_id}`\n\n"
        f"Independent evaluation:\n\n```json\n{json.dumps(evaluation.get('evaluation', {}), ensure_ascii=False, indent=2)}\n```\n\n"
        f"Report excerpt:\n\n```json\n{excerpt}\n```\n"
    )


def _first(frame: pd.DataFrame, predicate: pd.Series) -> Mapping[str, Any] | None:
    subset = frame[predicate]
    return subset.iloc[0].to_dict() if not subset.empty else None


def _write_cases(
    layout: Mapping[str, Path],
    frames: Mapping[str, pd.DataFrame],
    *,
    protocol_note: str = "",
) -> None:
    stress, open_frame, multi = frames["stress"], frames["open_task"], frames["multi_turn"]
    false_series = pd.Series(False, index=stress.index)
    selections = {
        "case_successful_repair.md": ("Successful repair", _first(stress, _bool_series(stress["repaired"]) == 1) if "repaired" in stress else None, "The framework does not synthesize a success example."),
        "case_safe_abstention.md": ("Safe abstention", _first(stress, _bool_series(stress["safe_abstention"]) == 1) if "safe_abstention" in stress else None, "No safe-abstention event met the independent rule."),
        "case_reflection_no_effect.md": ("Reflection with no observed effect", _first(stress, (_bool_series(stress["reflection_enabled"]) == 1) & (_bool_series(stress.get("report_changed_by_reflection", false_series)) == 0)) if not stress.empty else None, "No qualifying reflection run was available."),
        "case_false_repair.md": ("False repair", _first(stress, _bool_series(stress["false_repair"]) == 1) if "false_repair" in stress else None, "No false repair was observed; this absence is retained rather than replaced by a hypothetical case."),
        "case_conflicting_evidence.md": ("Conflicting evidence", _first(stress, stress.get("fault_type", pd.Series(index=stress.index, dtype=str)) == "conflicting_evidence") if not stress.empty else None, "No evaluated conflicting-evidence run was available."),
        "case_plausible_wrong_prediction.md": ("Plausible but wrong prediction", _first(stress, stress.get("fault_type", pd.Series(index=stress.index, dtype=str)) == "plausible_but_wrong_prediction") if not stress.empty else None, "No evaluated plausible-wrong-prediction run was available."),
        "case_llm_open_task_advantage.md": ("LLM open-task advantage", None, "No paired, real case-level LLM advantage was identified automatically."),
        "case_heuristic_efficiency_advantage.md": ("Heuristic efficiency advantage", None, "No paired, real case-level heuristic efficiency advantage was identified automatically."),
        "case_multi_turn_state_update.md": ("Multi-turn state update", _first(multi, _bool_series(multi["state_consistency"]) == 1) if "state_consistency" in multi else None, "No state-consistent multi-turn run was observed."),
    }
    if not open_frame.empty and {"case_id", "orchestrator", "objective_coverage", "latency_seconds"}.issubset(open_frame.columns):
        pivot = open_frame.pivot_table(index="case_id", columns="orchestrator", values=["objective_coverage", "latency_seconds"], aggfunc="first")
        for case_id in pivot.index:
            try:
                if pivot.loc[case_id, ("objective_coverage", "llm")] > pivot.loc[case_id, ("objective_coverage", "heuristic")]:
                    row = open_frame[(open_frame["case_id"] == case_id) & (open_frame["orchestrator"] == "llm")].iloc[0].to_dict()
                    selections["case_llm_open_task_advantage.md"] = ("LLM open-task advantage", row, "")
                    break
            except Exception:
                continue
        for case_id in pivot.index:
            try:
                if pivot.loc[case_id, ("latency_seconds", "heuristic")] < pivot.loc[case_id, ("latency_seconds", "llm")]:
                    row = open_frame[(open_frame["case_id"] == case_id) & (open_frame["orchestrator"] == "heuristic")].iloc[0].to_dict()
                    selections["case_heuristic_efficiency_advantage.md"] = ("Heuristic efficiency advantage", row, "")
                    break
            except Exception:
                continue
    for filename, (title, row, reason) in selections.items():
        text = _case_text(title, row, layout, reason)
        if protocol_note:
            text = text.replace("\n\n", f"\n\n> **{protocol_note}**\n\n", 1)
        (layout["cases"] / filename).write_text(text, encoding="utf-8")


def render(output_root: str | Path) -> dict[str, Any]:
    layout = ensure_output_layout(output_root)
    filenames = {
        "clean": "clean_case_level.csv",
        "stress": "stress_case_level.csv",
        "open_task": "open_task_case_level.csv",
        "multi_turn": "multi_turn_case_level.csv",
    }
    all_frames = {group: _read_csv(layout["metrics"] / filename) for group, filename in filenames.items()}
    frames = dict(all_frames)
    frames["clean"] = _primary_clean(all_frames["clean"])
    report_quality = _read_csv(layout["metrics"] / "report_quality_case_level.csv")
    if not report_quality.empty and {"dataset_group", "repeat_index"}.issubset(report_quality.columns):
        repeats = pd.to_numeric(report_quality["repeat_index"], errors="coerce").fillna(0)
        report_quality = report_quality[(report_quality["dataset_group"] != "clean") | (repeats == 0)].copy()
    stats = read_json(layout["metrics"] / "statistical_tests.json", default={}) or {}
    canonical_manifest = stats.get("canonical_run_manifest") or read_json(
        layout["metrics"] / "canonical_run_manifest.json", default={}
    ) or {}
    missing_counts = canonical_manifest.get("missing_real_rag_valid_counts") or {}
    protocol_complete = bool(missing_counts) and not any(int(value or 0) for value in missing_counts.values())
    protocol_note = ""
    if not protocol_complete:
        missing_text = ", ".join(
            f"{group}={int(value or 0)}" for group, value in sorted(missing_counts.items())
        ) or "full real-RAG matrix not certified"
        protocol_note = (
            "PROVISIONAL PARTIAL-DATA OUTPUT: real-RAG-valid matrix is incomplete "
            f"({missing_text}); do not use these numeric values in a manuscript"
        )
    _write_tables(layout, frames, report_quality, protocol_note=protocol_note)

    _plot_rate(frames["stress"], layout["figures"] / "fig_detection_repair", [("detected", "Detection"), ("repaired", "Repair")], note="Detection and repair require valid controlled-stress runs.")
    _plot_rate(
        frames["stress"],
        layout["figures"] / "fig_safe_abstention",
        [("safe_abstention", "Safe abstention")],
        filters=lambda frame: frame[frame.get("recoverability", pd.Series(index=frame.index, dtype=str)) == "unrecoverable"],
        note="Safe abstention is evaluated only for independently scored unrecoverable faults.",
    )
    _plot_rate(
        frames["stress"],
        layout["figures"] / "fig_false_repair",
        [("false_repair", "False repair")],
        filters=lambda frame: frame[
            (_bool_series(frame.get("reflection_enabled", pd.Series(index=frame.index, dtype=float))) == 1)
            & (_bool_series(frame.get("false_repair_applicable", pd.Series(index=frame.index, dtype=float))) == 1)
        ],
        note="False-repair rates are evaluated only when reflection executed and the independent rule marks the outcome applicable.",
    )
    combined_cost = pd.concat([frames["clean"], frames["stress"], frames["open_task"], frames["multi_turn"]], ignore_index=True)
    _plot_latency(combined_cost, layout["figures"] / "fig_latency_cost")
    _plot_open(frames["open_task"], layout["figures"] / "fig_open_task_coverage")
    _plot_rate(frames["multi_turn"], layout["figures"] / "fig_multi_turn_state_consistency", [("state_consistency", "State consistency")], note="State consistency requires complete two-turn runs.")
    _plot_rate(
        frames["stress"],
        layout["figures"] / "fig_conflicting_evidence",
        [("conflict_detected", "Detected"), ("conflicting_sources_preserved", "Both sources retained"), ("forced_conclusion", "Forced conclusion")],
        filters=lambda frame: frame[frame.get("fault_type", pd.Series(index=frame.index, dtype=str)) == "conflicting_evidence"],
        note="Conflict handling is shown only for controlled conflicting-evidence cases.",
    )
    _plot_failure_heatmap(frames["stress"], layout["figures"] / "fig_failure_modes")

    figure_contract = f"""# Figure contract

{f'> **{protocol_note}**' if protocol_note else 'Protocol status: complete real-RAG-valid matrix.'}

Core conclusion: controlled faults and open/multi-turn tasks reveal whether reflection and LLM orchestration change independent reliability outcomes, and at what case-level cost.

Figure archetype: quantitative grid (exported as focused single-panel figures).

Backend: Python/matplotlib/seaborn only.

Final size: 85–160 mm manuscript width depending on panel density; all rendered glyphs target at least 5 pt.

Evidence hierarchy: stress detection/repair is primary; safe abstention, conflict preservation and false repair are validation; latency/calls and open/multi-turn coverage are cost and boundary evidence.

Statistics: points are case-level setting means with 95% case-bootstrap intervals; boxes show median and IQR, whiskers the 5th–95th percentiles, and overlaid points individual cases. Paired tests and Holm-adjusted p values are provided in statistical_tests.json, not encoded as significance stars.

Reviewer risk: invalid injections, missing pairs, ceiling effects and internal-verifier/process metrics must not be conflated with independent scientific correctness.
"""
    (layout["figures"] / "figure_contract.md").write_text(figure_contract, encoding="utf-8")
    legends = f"""# Figure legends

{f'> **{protocol_note}**' if protocol_note else 'Protocol status: complete real-RAG-valid matrix.'}

All points represent independent benchmark cases. Metric-specific applicability rules define denominators (for example, repair uses recoverable faults and safe abstention uses unrecoverable faults); exact n values are included in the source data and tables. Rate symbols show means and 95% percentile confidence intervals from 10,000 case-level bootstrap resamples. Box plots show the median and interquartile range; whiskers span the 5th–95th percentiles, and overlaid dots are individual cases. Comparisons are paired by case. Exact McNemar or Wilcoxon results, effect sizes, pair counts and Holm-adjusted p values are provided in the statistical output. Source data are provided beside each figure.
"""
    (layout["figures"] / "figure_legends.md").write_text(legends, encoding="utf-8")
    _write_cases(layout, frames, protocol_note=protocol_note)
    return {
        "generated_at": utc_now(),
        "run_counts": {group: len(frame) for group, frame in frames.items()},
        "all_case_level_rows": {group: len(frame) for group, frame in all_frames.items()},
        "protocol_complete": protocol_complete,
        "missing_real_rag_valid_counts": missing_counts,
        "figures": 8,
        "tables": 6,
        "case_files": 9,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Render reliability tables, figures and case studies from real metrics.")
    parser.add_argument("output_root", nargs="?", default="outputs/agentm_reliability")
    args = parser.parse_args()
    result = render(resolve_path(args.output_root))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
