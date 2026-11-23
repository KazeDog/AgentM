#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
from scipy.stats import binomtest, rankdata, wilcoxon

try:
    from .common import ensure_output_layout, read_json, resolve_path, utc_now, write_json
except ImportError:
    from common import ensure_output_layout, read_json, resolve_path, utc_now, write_json


GROUP_METRICS = {
    "clean": {
        "binary": [
            "first_attempt_completed",
            "completed",
            "objective_passed",
            "model_record_created",
            "citation_map_created",
            "verifier_passed",
            "scientific_object_alignment",
            "external_evidence_coverage",
            "direct_evidence_coverage",
            "qc_only",
            "prediction_fact_separation_accuracy",
            "uncertainty_expression_accuracy",
        ],
        "continuous": [
            "unsupported_claim_count",
            "latency_seconds",
            "llm_call_count",
            "tool_call_count",
            "retrieval_call_count",
            "reflection_step_count",
            "external_service_error_count",
            "operational_latency_seconds",
            "evidence_relevance",
            "report_schema_completeness",
        ],
    },
    "stress": {
        "binary": [
            "detected",
            "detection_category_correct",
            "repaired",
            "safe_abstention",
            "false_repair",
            "final_status_correct",
            "entity_alignment_correct",
            "conflict_detected",
            "conflicting_sources_preserved",
            "forced_conclusion",
            "uncertainty_expression_correct",
            "prediction_fact_separation_correct",
            "first_attempt_completed",
            "external_evidence_coverage",
            "direct_evidence_coverage",
            "qc_only",
        ],
        "continuous": [
            "unsupported_claim_count_independent",
            "latency_seconds",
            "llm_call_count",
            "tool_call_count",
            "retrieval_call_count",
            "reflection_step_count",
            "citation_precision",
            "citation_recall",
            "unsupported_claim_rate",
            "operational_latency_seconds",
            "evidence_relevance",
            "report_schema_completeness",
        ],
    },
    "open_task": {
        "binary": [
            "recovery_success",
            "scientific_object_alignment",
            "external_evidence_coverage",
            "direct_evidence_coverage",
            "qc_only",
            "prediction_fact_separation_accuracy",
            "uncertainty_expression_accuracy",
        ],
        "continuous": [
            "objective_coverage",
            "tool_selection_accuracy",
            "dependency_order_accuracy",
            "invalid_tool_call_rate",
            "human_intervention_count",
            "unsupported_claim_count_independent",
            "latency_seconds",
            "llm_call_count",
            "tool_call_count",
            "operational_latency_seconds",
            "evidence_relevance",
            "report_schema_completeness",
        ],
    },
    "multi_turn": {
        "binary": [
            "state_consistency",
            "replanning_success",
            "citation_update_accuracy",
            "cross_turn_contamination",
            "scientific_object_alignment",
            "external_evidence_coverage",
            "direct_evidence_coverage",
            "qc_only",
            "prediction_fact_separation_accuracy",
            "uncertainty_expression_accuracy",
        ],
        "continuous": [
            "objective_coverage",
            "tool_selection_accuracy",
            "stale_result_removal_accuracy",
            "object_update_accuracy",
            "unsupported_claim_count_independent",
            "latency_seconds",
            "llm_call_count",
            "tool_call_count",
            "operational_latency_seconds",
            "evidence_relevance",
            "report_schema_completeness",
        ],
    },
}


APPLICABILITY_COLUMNS = {
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


def _read_metrics(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def _as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    mapping = {"true": True, "false": False, "1": True, "0": False, "yes": True, "no": False}
    return series.map(lambda value: mapping.get(str(value).strip().lower(), value if isinstance(value, bool) else np.nan))


def _analysis_frame(group: str, frame: pd.DataFrame, metric: str | None = None) -> pd.DataFrame:
    subset = frame.copy()
    if group == "clean" and "repeat_index" in subset:
        subset = subset[pd.to_numeric(subset["repeat_index"], errors="coerce").fillna(0) == 0]
    if group in {"stress", "open_task", "multi_turn"} and "evaluation_valid" in subset:
        valid = _as_bool(subset["evaluation_valid"])
        subset = subset[valid == True]  # noqa: E712
    if metric and metric in APPLICABILITY_COLUMNS:
        column = APPLICABILITY_COLUMNS[metric]
        if column not in subset:
            return subset.iloc[0:0]
        applicable = _as_bool(subset[column])
        subset = subset[applicable == True]  # noqa: E712
    return subset


def _paired_frame(
    frame: pd.DataFrame,
    *,
    metric: str,
    factor: str,
    baseline: Any,
    comparator: Any,
    fixed: Mapping[str, Any],
) -> pd.DataFrame:
    subset = frame.copy()
    for key, value in fixed.items():
        if key not in subset:
            return pd.DataFrame()
        subset = subset[subset[key] == value]
    if metric not in subset or factor not in subset:
        return pd.DataFrame()
    keys = [key for key in ["case_id", "seed", "repeat_index"] if key in subset.columns]
    if not keys:
        return pd.DataFrame()
    values = subset[subset[factor].isin([baseline, comparator])][keys + [factor, metric]].copy()
    values = values.drop_duplicates(keys + [factor], keep="last")
    paired = values.pivot(index=keys, columns=factor, values=metric).reset_index()
    if baseline not in paired or comparator not in paired:
        return pd.DataFrame()
    return paired.dropna(subset=[baseline, comparator])


def _bootstrap_difference(x: np.ndarray, y: np.ndarray, *, resamples: int, seed: int) -> list[float | None]:
    if len(x) == 0:
        return [None, None]
    rng = np.random.default_rng(seed)
    differences = y - x
    indices = rng.integers(0, len(differences), size=(resamples, len(differences)))
    means = differences[indices].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def _relative_difference(baseline: float, comparator: float) -> float | None:
    if not math.isfinite(baseline) or baseline == 0:
        return None
    return (comparator - baseline) / abs(baseline)


def paired_binary(
    x: Iterable[Any], y: Iterable[Any], *, resamples: int, seed: int
) -> dict[str, Any]:
    x_arr = np.asarray(list(x), dtype=bool).astype(float)
    y_arr = np.asarray(list(y), dtype=bool).astype(float)
    if len(x_arr) == 0:
        raise ValueError("paired_binary requires at least one pair")
    b = int(np.sum((x_arr == 1) & (y_arr == 0)))
    c = int(np.sum((x_arr == 0) & (y_arr == 1)))
    discordant = b + c
    p_value = float(binomtest(min(b, c), discordant, p=0.5, alternative="two-sided").pvalue) if discordant else 1.0
    odds_ratio = (c + 0.5) / (b + 0.5)
    baseline_mean, comparator_mean = float(x_arr.mean()), float(y_arr.mean())
    return {
        "test": "exact_McNemar_binomial",
        "n_pairs": len(x_arr),
        "baseline_mean": baseline_mean,
        "comparator_mean": comparator_mean,
        "absolute_difference": comparator_mean - baseline_mean,
        "relative_difference": _relative_difference(baseline_mean, comparator_mean),
        "difference_ci95_bootstrap": _bootstrap_difference(x_arr, y_arr, resamples=resamples, seed=seed),
        "discordant_baseline_only": b,
        "discordant_comparator_only": c,
        "matched_odds_ratio_haldane": odds_ratio,
        "p_value": p_value,
    }


def _rank_biserial(differences: np.ndarray) -> float:
    nonzero = differences[differences != 0]
    if len(nonzero) == 0:
        return 0.0
    ranks = rankdata(np.abs(nonzero), method="average")
    positive = float(ranks[nonzero > 0].sum())
    negative = float(ranks[nonzero < 0].sum())
    denominator = positive + negative
    return (positive - negative) / denominator if denominator else 0.0


def paired_continuous(
    x: Iterable[Any], y: Iterable[Any], *, resamples: int, seed: int
) -> dict[str, Any]:
    x_arr = np.asarray(list(x), dtype=float)
    y_arr = np.asarray(list(y), dtype=float)
    if len(x_arr) == 0:
        raise ValueError("paired_continuous requires at least one pair")
    differences = y_arr - x_arr
    if np.allclose(differences, 0, equal_nan=False):
        statistic, p_value = 0.0, 1.0
    else:
        result = wilcoxon(y_arr, x_arr, alternative="two-sided", zero_method="wilcox", method="auto")
        statistic, p_value = float(result.statistic), float(result.pvalue)
    baseline_mean, comparator_mean = float(np.mean(x_arr)), float(np.mean(y_arr))
    return {
        "test": "Wilcoxon_paired_signed_rank",
        "n_pairs": len(x_arr),
        "baseline_mean": baseline_mean,
        "baseline_median": float(np.median(x_arr)),
        "comparator_mean": comparator_mean,
        "comparator_median": float(np.median(y_arr)),
        "absolute_difference": comparator_mean - baseline_mean,
        "median_paired_difference": float(np.median(differences)),
        "relative_difference": _relative_difference(baseline_mean, comparator_mean),
        "difference_ci95_bootstrap": _bootstrap_difference(x_arr, y_arr, resamples=resamples, seed=seed),
        "rank_biserial_correlation": _rank_biserial(differences),
        "statistic": statistic,
        "p_value": p_value,
    }


def _holm_adjust(p_values: list[float]) -> list[float]:
    count = len(p_values)
    if count == 0:
        return []
    order = np.argsort(p_values)
    adjusted = np.empty(count, dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        candidate = (count - rank) * float(p_values[index])
        running = max(running, candidate)
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def _coerce_pair_values(paired: pd.DataFrame, baseline: Any, comparator: Any, kind: str) -> tuple[np.ndarray, np.ndarray]:
    if kind == "binary":
        x = _as_bool(paired[baseline])
        y = _as_bool(paired[comparator])
        valid = x.notna() & y.notna()
        return x[valid].astype(bool).to_numpy(), y[valid].astype(bool).to_numpy()
    x = pd.to_numeric(paired[baseline], errors="coerce")
    y = pd.to_numeric(paired[comparator], errors="coerce")
    valid = x.notna() & y.notna() & np.isfinite(x) & np.isfinite(y)
    return x[valid].to_numpy(dtype=float), y[valid].to_numpy(dtype=float)


def _comparisons_for_group(group: str, frame: pd.DataFrame) -> list[dict[str, Any]]:
    comparisons: list[dict[str, Any]] = []
    if "reflection_enabled" in frame:
        for orchestrator in sorted(frame.get("orchestrator", pd.Series(dtype=str)).dropna().unique()):
            comparisons.append(
                {
                    "name": f"reflection_on_vs_off__{orchestrator}",
                    "factor": "reflection_enabled",
                    "baseline": False,
                    "comparator": True,
                    "fixed": {"orchestrator": orchestrator},
                }
            )
    if "orchestrator" in frame:
        for reflection in sorted(frame.get("reflection_enabled", pd.Series(dtype=bool)).dropna().unique()):
            comparisons.append(
                {
                    "name": f"llm_vs_heuristic__reflection_{str(reflection).lower()}",
                    "factor": "orchestrator",
                    "baseline": "heuristic",
                    "comparator": "llm",
                    "fixed": {"reflection_enabled": reflection},
                }
            )
    return comparisons


def analyse_group(group: str, frame: pd.DataFrame, *, resamples: int, seed: int) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    results: list[dict[str, Any]] = []
    spec = GROUP_METRICS[group]
    for comparison in _comparisons_for_group(group, frame):
        family_indices: list[int] = []
        for kind in ("binary", "continuous"):
            for metric in spec[kind]:
                metric_frame = _analysis_frame(group, frame, metric)
                paired = _paired_frame(
                    metric_frame,
                    metric=metric,
                    factor=comparison["factor"],
                    baseline=comparison["baseline"],
                    comparator=comparison["comparator"],
                    fixed=comparison["fixed"],
                )
                if paired.empty:
                    continue
                x, y = _coerce_pair_values(paired, comparison["baseline"], comparison["comparator"], kind)
                if len(x) == 0:
                    continue
                stats = paired_binary(x, y, resamples=resamples, seed=seed + len(results)) if kind == "binary" else paired_continuous(
                    x, y, resamples=resamples, seed=seed + len(results)
                )
                result = {
                    "dataset_group": group,
                    "comparison": comparison["name"],
                    "baseline": comparison["baseline"],
                    "comparator": comparison["comparator"],
                    "fixed_factors": comparison["fixed"],
                    "metric": metric,
                    "metric_type": kind,
                    "independent_unit": "case",
                    **stats,
                }
                family_indices.append(len(results))
                results.append(result)
        adjusted = _holm_adjust([float(results[index]["p_value"]) for index in family_indices])
        for index, p_adjusted in zip(family_indices, adjusted):
            results[index]["p_value_holm"] = p_adjusted
            results[index]["multiplicity_family"] = f"{group}:{comparison['name']}"
    return results


def _bootstrap_mean_ci(values: np.ndarray, *, resamples: int, seed: int) -> list[float] | None:
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return None
    rng = np.random.default_rng(seed)
    samples = values[rng.integers(0, len(values), size=(resamples, len(values)))].mean(axis=1)
    return [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]


def descriptive_summary(group: str, frame: pd.DataFrame, *, resamples: int, seed: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    spec = GROUP_METRICS[group]
    settings = (
        frame[["orchestrator", "reflection_enabled"]].drop_duplicates().to_dict("records")
        if {"orchestrator", "reflection_enabled"}.issubset(frame.columns)
        else []
    )
    for setting_index, setting in enumerate(settings):
        for kind in ("binary", "continuous"):
            for metric_index, metric in enumerate(spec[kind]):
                subset = _analysis_frame(group, frame, metric)
                reflection = _as_bool(subset["reflection_enabled"])
                wanted_reflection = bool(setting["reflection_enabled"])
                subset = subset[
                    (subset["orchestrator"] == setting["orchestrator"])
                    & (reflection == wanted_reflection)
                ]
                if metric not in subset:
                    continue
                if kind == "binary":
                    values = _as_bool(subset[metric]).dropna().astype(float).to_numpy()
                else:
                    values = pd.to_numeric(subset[metric], errors="coerce").dropna().to_numpy(dtype=float)
                    values = values[np.isfinite(values)]
                if len(values) == 0:
                    continue
                rows.append(
                    {
                        "dataset_group": group,
                        "orchestrator": setting["orchestrator"],
                        "reflection_enabled": wanted_reflection,
                        "metric": metric,
                        "metric_type": kind,
                        "n_cases": int(len(values)),
                        "numerator": int(values.sum()) if kind == "binary" else None,
                        "mean": float(values.mean()),
                        "median": float(np.median(values)),
                        "sd": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                        "ci95_bootstrap": _bootstrap_mean_ci(
                            values,
                            resamples=resamples,
                            seed=seed + setting_index * 1000 + metric_index,
                        ),
                    }
                )
    return rows


def _clean_stability(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame.empty or not {"case_id", "orchestrator", "reflection_enabled", "repeat_index"}.issubset(frame.columns):
        return []
    repeated = frame.groupby(["case_id", "orchestrator", "reflection_enabled"], dropna=False).filter(lambda group: len(group) >= 3)
    rows: list[dict[str, Any]] = []
    for (orchestrator, reflection), setting in repeated.groupby(["orchestrator", "reflection_enabled"], dropna=False):
        groups = list(setting.groupby("case_id"))
        exact_hash = []
        objective_stable = []
        for _, case_group in groups:
            hashes = case_group.get("final_report_hash", pd.Series(dtype=str)).dropna().astype(str)
            if len(hashes):
                exact_hash.append(hashes.nunique() == 1)
            objectives = _as_bool(case_group.get("objective_passed", pd.Series(dtype=object))).dropna()
            if len(objectives):
                objective_stable.append(objectives.nunique() == 1)
        rows.append(
            {
                "orchestrator": orchestrator,
                "reflection_enabled": bool(reflection),
                "n_cases": len(groups),
                "n_runs": len(setting),
                "exact_report_hash_consistency": float(np.mean(exact_hash)) if exact_hash else None,
                "objective_outcome_consistency": float(np.mean(objective_stable)) if objective_stable else None,
                "first_attempt_completion_rate": float(_as_bool(setting["first_attempt_completed"]).dropna().mean()),
                "latency_mean_seconds": float(pd.to_numeric(setting["latency_seconds"], errors="coerce").mean()),
                "latency_sd_seconds": float(pd.to_numeric(setting["latency_seconds"], errors="coerce").std()),
            }
        )
    return rows


def _citation_aggregate(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame.empty or not {"citation_tp", "citation_fp", "citation_fn"}.issubset(frame.columns):
        return []
    valid = _analysis_frame("stress", frame)
    rows = []
    for (orchestrator, reflection), group in valid.groupby(["orchestrator", "reflection_enabled"], dropna=False):
        tp = int(pd.to_numeric(group["citation_tp"], errors="coerce").fillna(0).sum())
        fp = int(pd.to_numeric(group["citation_fp"], errors="coerce").fillna(0).sum())
        fn = int(pd.to_numeric(group["citation_fn"], errors="coerce").fillna(0).sum())
        rows.append(
            {
                "orchestrator": orchestrator,
                "reflection_enabled": bool(reflection),
                "citation_tp": tp,
                "citation_fp": fp,
                "citation_fn": fn,
                "micro_precision": tp / (tp + fp) if tp + fp else None,
                "micro_recall": tp / (tp + fn) if tp + fn else None,
            }
        )
    return rows






def run_statistics(output_root: str | Path, *, resamples: int = 10000, seed: int = 2026) -> dict[str, Any]:
    if resamples < 10000:
        raise ValueError("The prespecified analysis requires at least 10,000 bootstrap resamples")
    layout = ensure_output_layout(output_root)
    filenames = {
        "clean": "clean_case_level.csv",
        "stress": "stress_case_level.csv",
        "open_task": "open_task_case_level.csv",
        "multi_turn": "multi_turn_case_level.csv",
    }
    all_tests: list[dict[str, Any]] = []
    sample_counts: dict[str, Any] = {}
    descriptive: dict[str, list[dict[str, Any]]] = {}
    frames: dict[str, pd.DataFrame] = {}
    for group, filename in filenames.items():
        frame = _read_metrics(layout["metrics"] / filename)
        frames[group] = frame
        analysis = _analysis_frame(group, frame)
        sample_counts[group] = {
            "all_canonical_rows": len(frame),
            "primary_valid_rows": len(analysis),
            "unique_cases": int(analysis["case_id"].nunique()) if "case_id" in analysis else 0,
        }
        descriptive[group] = descriptive_summary(group, frame, resamples=resamples, seed=seed)
        all_tests.extend(analyse_group(group, frame, resamples=resamples, seed=seed))
    report_quality = _read_metrics(layout["metrics"] / "report_quality_case_level.csv")
    payload = {
        "generated_at": utc_now(),
        "analysis_plan": {
            "independent_unit": "case (paired across settings)",
            "binary_test": "exact McNemar test implemented as an exact two-sided binomial test on discordant pairs",
            "continuous_test": "two-sided paired Wilcoxon signed-rank test",
            "confidence_intervals": f"case-level paired bootstrap percentile intervals with {resamples} resamples",
            "multiplicity": "Holm correction within each dataset-group × prespecified comparison family",
            "effect_sizes": "matched odds ratio for binary outcomes and rank-biserial correlation for continuous outcomes",
            "missing_pairs": "complete pairs only; pair counts are reported for every test",
            "clean_repeat_policy": "repeat_index=0 is the primary case-level analysis; the prespecified three-run subset is summarized separately and is not treated as independent n",
            "stress_denominators": "valid controlled injections only; repair uses recoverable faults, safe abstention uses unrecoverable faults, conflict metrics use conflicting-evidence cases, and false repair uses runs in which reflection actually executed",
            "retry_policy": "one canonical completed attempt per prespecified condition; first-attempt completion and retry overhead are retained as separate process outcomes",
            "claim_boundary": "Statistical significance is not interpreted as practical importance or causal proof.",
            "report_quality_scope": "Only objectively auditable dimensions are analysed; internal Verifier pass is not a scientific gold standard.",
        },
        "sample_counts": sample_counts,
        "descriptive_summaries": descriptive,
        "clean_stability_summary": _clean_stability(frames["clean"]),
        "stress_citation_aggregate": _citation_aggregate(frames["stress"]),
        "canonical_run_manifest": read_json(layout["metrics"] / "canonical_run_manifest.json", default={}) or {},
        "tests": all_tests,
        "report_quality_audit_rows": len(report_quality),
        "subjective_dimensions_not_evaluated": [
            "language style",
            "readability",
            "subjective persuasiveness",
            "expert clinical utility",
        ],
    }
    write_json(layout["metrics"] / "statistical_tests.json", payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Run prespecified paired reliability statistics from raw-derived metrics.")
    parser.add_argument("output_root", nargs="?", default="outputs/agentm_reliability")
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    payload = run_statistics(resolve_path(args.output_root), resamples=args.bootstrap_resamples, seed=args.seed)
    print(json.dumps({"n_tests": len(payload["tests"]), "sample_counts": payload["sample_counts"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
