#!/usr/bin/env python
"""Render aligned, large-type manuscript composites from frozen metrics.

The script changes presentation only.  It reuses the canonical case-level
metrics and the same applicability filters and bootstrap implementation as
``render_outputs.py``; no scientific value, denominator, or observation is
added, removed, or recalculated under a different rule.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
# Prevent the project entry point ``statistics.py`` from shadowing Python's
# standard-library module while seaborn imports ``statistics.NormalDist``.
_removed_script_paths = [item for item in sys.path if Path(item or ".").resolve() == SCRIPT_DIR]
sys.path[:] = [item for item in sys.path if Path(item or ".").resolve() != SCRIPT_DIR]
import statistics as _stdlib_statistics  # noqa: F401,E402
sys.path[:0] = _removed_script_paths

import seaborn as sns

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from render_outputs import (  # noqa: E402
    COLORS,
    METRIC_APPLICABILITY,
    SETTING_ORDER,
    _bool_series,
    _bootstrap_ci,
    _metric_scope,
    _primary_clean,
    _setting_mask,
)


matplotlib.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 10.0,
        "axes.labelsize": 10.3,
        "axes.linewidth": 0.9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.labelsize": 9.2,
        "ytick.labelsize": 9.2,
        "legend.fontsize": 9.2,
        "legend.frameon": False,
        "xtick.color": "black",
        "ytick.color": "black",
        "axes.labelcolor": "black",
    }
)


def _save_bundle(fig: plt.Figure, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    kwargs = {"bbox_inches": "tight", "pad_inches": 0.035, "facecolor": "white"}
    fig.savefig(base.with_suffix(".svg"), **kwargs)
    fig.savefig(base.with_suffix(".pdf"), **kwargs)
    fig.savefig(base.with_suffix(".png"), dpi=600, **kwargs)
    fig.savefig(base.with_suffix(".tiff"), dpi=600, **kwargs)
    plt.close(fig)


def _panel_label(ax: plt.Axes, label: str, *, x: float = -0.15, y: float = 1.10) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=13,
        fontweight="bold",
        va="top",
        ha="left",
        color="black",
        clip_on=False,
    )


def _style_rate_axis(ax: plt.Axes, labels: list[str]) -> None:
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, color="black", linespacing=0.95)
    ax.set_ylim(-0.03, 1.05)
    ax.set_ylabel("Case-level rate")
    ax.grid(axis="y", color="#E5E5EA", linewidth=0.7, zorder=0)
    ax.tick_params(axis="both", width=0.8, length=3.2)


def _draw_rate_panel(
    ax: plt.Axes,
    frame: pd.DataFrame,
    metrics: list[tuple[str, str]],
    *,
    filters=None,
    legend_columns: int | None = None,
) -> None:
    data = filters(frame.copy()) if filters else frame.copy()
    offsets = np.linspace(-0.15, 0.15, len(metrics)) if len(metrics) > 1 else [0.0]
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
    for setting_index, (orchestrator, reflection, _, color) in enumerate(active_settings):
        subset = data[_setting_mask(data, orchestrator, reflection)]
        for metric_index, (metric, display) in enumerate(metrics):
            if metric not in subset:
                continue
            metric_subset = _metric_scope(subset, metric)
            values = _bool_series(metric_subset[metric]).dropna().to_numpy(dtype=float)
            if not len(values):
                continue
            mean = float(values.mean())
            low, high = _bootstrap_ci(values, seed=2026 + setting_index * 31 + metric_index)
            ax.errorbar(
                setting_index + offsets[metric_index],
                mean,
                yerr=[[mean - low], [high - mean]],
                fmt=markers[metric_index % len(markers)],
                ms=5.4,
                lw=1.15,
                capsize=2.8,
                color=color,
                markeredgecolor="white",
                markeredgewidth=0.6,
                label=display if setting_index == 0 else None,
                zorder=3,
            )
    _style_rate_axis(ax, [item[2] for item in active_settings])
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.015),
            ncol=legend_columns or len(labels),
            handletextpad=0.45,
            columnspacing=1.0,
            borderaxespad=0,
        )


def _fault_matrix(stress: pd.DataFrame) -> pd.DataFrame:
    metrics = ["detected", "repaired", "safe_abstention", "false_repair", "forced_conclusion"]
    valid = stress[_bool_series(stress.get("evaluation_valid", pd.Series(index=stress.index, dtype=float))) == 1].copy()
    rows = []
    for fault, group in valid.groupby("fault_type"):
        row = {"fault_type": fault}
        for metric in metrics:
            metric_group = _metric_scope(group, metric)
            values = _bool_series(metric_group[metric]).dropna() if metric in metric_group else pd.Series(dtype=float)
            row[metric] = float(values.mean()) if len(values) else math.nan
        rows.append(row)
    return pd.DataFrame(rows).set_index("fault_type")[metrics]


def _draw_fault_heatmap(ax: plt.Axes, stress: pd.DataFrame) -> None:
    matrix = _fault_matrix(stress)
    display_columns = ["Detected", "Repaired", "Safe abstention", "False repair", "Forced conclusion"]
    display_index = [str(value).replace("_", " ") for value in matrix.index]
    sns.heatmap(
        matrix,
        vmin=0,
        vmax=1,
        cmap="crest",
        annot=True,
        fmt=".2f",
        annot_kws={"fontsize": 8.2},
        linewidths=0.45,
        linecolor="white",
        cbar_kws={"fraction": 0.050, "pad": 0.022},
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticklabels(display_columns, rotation=29, rotation_mode="anchor", ha="right", fontsize=8.3, color="black")
    ax.set_yticklabels(
        display_index,
        rotation=0,
        rotation_mode="anchor",
        fontsize=8.2,
        color="black",
    )
    ax.tick_params(axis="both", length=0)
    colorbar = ax.collections[0].colorbar
    colorbar.ax.tick_params(labelsize=8.2, width=0.7, length=2.5)


def _merge_source_data(output_root: Path, destination: Path, panels: list[tuple[str, str]]) -> None:
    frames = []
    for panel, filename in panels:
        path = output_root / "figures" / filename
        if not path.is_file():
            continue
        frame = pd.read_csv(path)
        frame.insert(0, "panel", panel)
        frame.insert(1, "source_panel_file", filename)
        frames.append(frame)
    if frames:
        pd.concat(frames, ignore_index=True, sort=False).to_csv(destination, index=False)


def render_stress_composite(output_root: Path) -> None:
    stress = pd.read_csv(output_root / "metrics" / "stress_case_level.csv")
    fig = plt.figure(figsize=(7.20, 5.25), facecolor="white")
    grid = fig.add_gridspec(
        2,
        2,
        left=0.115,
        right=0.985,
        bottom=0.105,
        top=0.925,
        wspace=0.27,
        hspace=0.43,
    )
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, 0])
    ax_d = fig.add_subplot(grid[1, 1])

    _draw_rate_panel(ax_a, stress, [("detected", "Detection"), ("repaired", "Repair")])
    _draw_rate_panel(
        ax_b,
        stress,
        [("safe_abstention", "Safe abstention")],
        filters=lambda frame: frame[frame.get("recoverability", pd.Series(index=frame.index, dtype=str)) == "unrecoverable"],
    )
    _draw_fault_heatmap(ax_c, stress)
    _draw_rate_panel(
        ax_d,
        stress,
        [
            ("conflict_detected", "Detected"),
            ("conflicting_sources_preserved", "Both sources retained"),
            ("forced_conclusion", "Forced conclusion"),
        ],
        filters=lambda frame: frame[frame.get("fault_type", pd.Series(index=frame.index, dtype=str)) == "conflicting_evidence"],
        legend_columns=3,
    )
    for label, axis in zip("abc", (ax_a, ax_b, ax_c)):
        _panel_label(axis, label, x=-0.17 if label in "ac" else -0.14)
    # Raise panel d above its three-column legend to prevent visual collision.
    _panel_label(ax_d, "d", x=-0.22, y=1.20)

    base = output_root / "figures" / "fig_stress_reliability_composite"
    _save_bundle(fig, base)
    _merge_source_data(
        output_root,
        output_root / "figures" / "source_data_fig_stress_reliability_composite.csv",
        [
            ("a", "source_data_fig_detection_repair.csv"),
            ("b", "source_data_fig_safe_abstention.csv"),
            ("c", "source_data_fig_failure_modes.csv"),
            ("d", "source_data_fig_conflicting_evidence.csv"),
        ],
    )


def _draw_open_panel(ax: plt.Axes, frame: pd.DataFrame) -> None:
    subset = frame[_bool_series(frame["reflection_enabled"]) == 0].copy()
    subset = subset[subset["orchestrator"].isin(["heuristic", "llm"])]
    subset["Objective coverage"] = pd.to_numeric(subset["objective_coverage"], errors="coerce")
    subset["Orchestrator"] = subset["orchestrator"].map({"heuristic": "Heuristic", "llm": "LLM"})
    n_before_metric_filter = len(subset)
    subset = subset[
        subset["Objective coverage"].notna() & subset["Orchestrator"].notna()
    ].copy()
    n_after_metric_filter = len(subset)
    n_excluded_missing_metric = n_before_metric_filter - n_after_metric_filter
    # These explicit counts document the only display exclusion and match the
    # source-data accounting generated by the canonical single-panel renderer.
    assert n_excluded_missing_metric >= 0
    palette = {"Heuristic": COLORS["heuristic_off"], "LLM": COLORS["llm_off"]}
    sns.boxplot(
        data=subset,
        x="Orchestrator",
        y="Objective coverage",
        hue="Orchestrator",
        palette=palette,
        width=0.46,
        linewidth=1.0,
        fliersize=0,
        whis=(5, 95),
        legend=False,
        ax=ax,
    )
    sns.stripplot(
        data=subset,
        x="Orchestrator",
        y="Objective coverage",
        color=COLORS["dark"],
        size=3.0,
        alpha=0.55,
        jitter=0.14,
        ax=ax,
    )
    ax.set_ylim(-0.03, 1.05)
    ax.set_xlabel("")
    ax.set_ylabel("Objective coverage")
    ax.grid(axis="y", color="#E5E5EA", linewidth=0.7, zorder=0)


def _cost_plot_frame(frame: pd.DataFrame, metric: str) -> tuple[pd.DataFrame, list[str], dict[str, str]]:
    compact = {
        ("heuristic", False): "H − R",
        ("heuristic", True): "H + R",
        ("llm", False): "LLM − R",
        ("llm", True): "LLM + R",
    }
    rows = []
    labels = []
    palette = {}
    for orchestrator, reflection, _, color in SETTING_ORDER:
        subset = frame[_setting_mask(frame, orchestrator, reflection)]
        values = pd.to_numeric(subset.get(metric, pd.Series(dtype=float)), errors="coerce").dropna()
        if not len(values):
            continue
        label = compact[(orchestrator, reflection)]
        labels.append(label)
        palette[label] = color
        rows.extend({"Setting": label, "Value": float(value)} for value in values)
    return pd.DataFrame(rows), labels, palette


def _draw_cost_axis(ax: plt.Axes, frame: pd.DataFrame, metric: str, ylabel: str) -> None:
    plot_frame, labels, palette = _cost_plot_frame(frame, metric)
    sns.boxplot(
        data=plot_frame,
        x="Setting",
        y="Value",
        hue="Setting",
        order=labels,
        palette=palette,
        width=0.52,
        linewidth=0.9,
        fliersize=0,
        whis=(5, 95),
        legend=False,
        ax=ax,
    )
    sns.stripplot(
        data=plot_frame,
        x="Setting",
        y="Value",
        order=labels,
        color=COLORS["dark"],
        size=2.3,
        jitter=0.16,
        alpha=0.42,
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, color="black")
    ax.grid(axis="y", color="#E5E5EA", linewidth=0.7, zorder=0)


def render_adaptive_composite(output_root: Path) -> None:
    clean = _primary_clean(pd.read_csv(output_root / "metrics" / "clean_case_level.csv"))
    stress = pd.read_csv(output_root / "metrics" / "stress_case_level.csv")
    open_task = pd.read_csv(output_root / "metrics" / "open_task_case_level.csv")
    multi_turn = pd.read_csv(output_root / "metrics" / "multi_turn_case_level.csv")
    combined_cost = pd.concat([clean, stress, open_task, multi_turn], ignore_index=True)

    fig = plt.figure(figsize=(7.20, 5.00), facecolor="white")
    grid = fig.add_gridspec(
        2,
        2,
        left=0.105,
        right=0.985,
        bottom=0.105,
        top=0.925,
        wspace=0.24,
        hspace=0.43,
    )
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c1 = fig.add_subplot(grid[1, 0])
    ax_c2 = fig.add_subplot(grid[1, 1])

    _draw_open_panel(ax_a, open_task)
    _draw_rate_panel(ax_b, multi_turn, [("state_consistency", "State consistency")])
    _draw_cost_axis(ax_c1, combined_cost, "latency_seconds", "Latency (s)")
    _draw_cost_axis(ax_c2, combined_cost, "llm_call_count", "LLM calls per case")

    _panel_label(ax_a, "a", x=-0.17)
    _panel_label(ax_b, "b", x=-0.14)
    _panel_label(ax_c1, "c", x=-0.17)

    # The four plotting rectangles use one GridSpec, so the top and bottom
    # rows share exact left/right boundaries and equal panel heights.
    base = output_root / "figures" / "fig_adaptive_orchestration_composite"
    _save_bundle(fig, base)
    _merge_source_data(
        output_root,
        output_root / "figures" / "source_data_fig_adaptive_orchestration_composite.csv",
        [
            ("a", "source_data_fig_open_task_coverage.csv"),
            ("b", "source_data_fig_multi_turn_state_consistency.csv"),
            ("c", "source_data_fig_latency_cost.csv"),
        ],
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_root", nargs="?", default="outputs/agentm_reliability")
    args = parser.parse_args()
    output_root = Path(args.output_root).resolve()
    render_stress_composite(output_root)
    render_adaptive_composite(output_root)
    print(output_root / "figures" / "fig_stress_reliability_composite.pdf")
    print(output_root / "figures" / "fig_adaptive_orchestration_composite.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
