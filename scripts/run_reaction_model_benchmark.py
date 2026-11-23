#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agentm.runtime.reaction import (  # noqa: E402
    ReactionRuntime,
    _classify_with_model,
    _read_csv_rows,
    _regress_with_model,
    classification_metrics,
    parse_reaction_text,
    reaction_smiles_from_row,
    regression_metrics,
)


SCHNEIDER_SPLIT_ROOT = Path("schneider/processed_reaction_class")
BH_SPLIT_ROOT = Path("processed_yield/buchwald_hartwig_kg")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run model-level full benchmarks for AgentM reaction predictive core.")
    parser.add_argument("--full", action="store_true", help="Use all available rows.")
    parser.add_argument("--max-rows", default="smoke", help="Integer row cap per split, or 'smoke'. Ignored with --full.")
    parser.add_argument("--model-family", default="tfidf_linear", choices=["tfidf_linear", "token_knn"])
    parser.add_argument("--reaction-data-root", default="data/reaction_sources")
    parser.add_argument("--output-dir", default="outputs/reaction_models_full")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir = output_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    data_root = Path(args.reaction_data_root)
    max_rows = None if args.full else _max_rows(args.max_rows)
    runtime = ReactionRuntime(output_dir=output_dir / "_runtime", data_root=data_root, model_dir=output_dir)
    started = time.time()
    failures: List[Dict[str, Any]] = []

    try:
        schneider = run_schneider(runtime, data_root, output_dir, predictions_dir, max_rows=max_rows, model_family=args.model_family)
    except Exception as exc:
        schneider = {"status": "failed_exception", "error": str(exc)}
        failures.append({"task": "schneider_classification", "error": str(exc), "traceback": traceback.format_exc()})

    bh_metrics: Dict[str, Any] = {}
    for split in ["test1", "test2", "test3", "test4"]:
        try:
            bh_metrics[split] = run_bh_split(runtime, data_root, output_dir, predictions_dir, split=split, max_rows=max_rows, model_family=args.model_family)
        except Exception as exc:
            bh_metrics[split] = {"status": "failed_exception", "error": str(exc)}
            failures.append({"task": "buchwald_hartwig_yield", "split": split, "error": str(exc), "traceback": traceback.format_exc()})

    summary = {
        "status": "pass" if not failures else "warning",
        "model_family": args.model_family,
        "max_rows": max_rows,
        "full": bool(args.full),
        "elapsed_sec": round(time.time() - started, 3),
        "schneider": schneider,
        "buchwald_hartwig": bh_metrics,
        "n_failures": len(failures),
    }
    _write_json(output_dir / "buchwald_hartwig_metrics_by_split.json", bh_metrics)
    _write_json(output_dir / "reaction_model_benchmark_summary.json", summary)
    _write_json(output_dir / "failure_report.json", failures)
    write_summary_md(output_dir / "reaction_model_benchmark_summary.md", summary)
    print(json.dumps({"summary_path": str(output_dir / "reaction_model_benchmark_summary.json"), "n_failures": len(failures)}, indent=2))
    return 0 if not failures else 1


def run_schneider(
    runtime: ReactionRuntime,
    data_root: Path,
    output_dir: Path,
    predictions_dir: Path,
    *,
    max_rows: int | None,
    model_family: str,
) -> Dict[str, Any]:
    model_meta = runtime.train_model(
        task="schneider_classification",
        max_rows=max_rows,
        output_dir=output_dir,
        model_family=model_family,
    )
    model = runtime._load_or_build_schneider_model(max_rows=max_rows or 8000)
    metrics: Dict[str, Any] = {"model": _compact_model_meta(model_meta), "splits": {}}
    confusion_rows: List[Dict[str, Any]] = []
    for part in ["val", "test"]:
        rows = _read_csv_rows(data_root / SCHNEIDER_SPLIT_ROOT / f"{part}_split.csv", max_rows=max_rows)
        y_true: List[str] = []
        y_pred: List[str] = []
        pred_rows: List[Dict[str, Any]] = []
        for idx, row in enumerate(rows):
            true = str(row.get("labels") or row.get("reaction_type") or "")
            if not true:
                continue
            pred = _classify_with_model(parse_reaction_text(reaction_smiles_from_row(row)), model)
            predicted = str(pred.get("label") or "")
            y_true.append(true)
            y_pred.append(predicted)
            pred_rows.append(
                {
                    "row_index": idx,
                    "reaction_smiles": reaction_smiles_from_row(row),
                    "true_label": true,
                    "predicted_label": predicted,
                    "confidence": pred.get("confidence"),
                }
            )
        metrics["splits"][part] = classification_metrics(y_true, y_pred)
        confusion_rows.extend(_confusion_rows(part, y_true, y_pred))
        _write_csv(predictions_dir / f"schneider_{part}_predictions.csv", pred_rows)
    _write_csv(output_dir / "confusion_matrix.csv", confusion_rows)
    _write_json(output_dir / "schneider_metrics.json", metrics)
    return metrics


def run_bh_split(
    runtime: ReactionRuntime,
    data_root: Path,
    output_dir: Path,
    predictions_dir: Path,
    *,
    split: str,
    max_rows: int | None,
    model_family: str,
) -> Dict[str, Any]:
    model_meta = runtime.train_model(
        task="buchwald_hartwig_yield",
        split=split,
        max_rows=max_rows,
        output_dir=output_dir,
        model_family=model_family,
    )
    model = runtime._load_or_build_bh_model(split=split, max_rows=max_rows or 4000)
    metrics: Dict[str, Any] = {"model": _compact_model_meta(model_meta), "splits": {}}
    for part in ["val", "test"]:
        rows = _read_csv_rows(data_root / BH_SPLIT_ROOT / split / f"{part}_split.csv", max_rows=max_rows)
        y_true: List[float] = []
        y_pred: List[float] = []
        pred_rows: List[Dict[str, Any]] = []
        for idx, row in enumerate(rows):
            try:
                true = float(row.get("yield"))
            except (TypeError, ValueError):
                continue
            pred = _regress_with_model(parse_reaction_text(reaction_smiles_from_row(row)), model)
            predicted = float(pred.get("predicted_yield") or 0.0)
            y_true.append(true)
            y_pred.append(predicted)
            pred_rows.append(
                {
                    "row_index": idx,
                    "reaction_smiles": reaction_smiles_from_row(row),
                    "true_yield": true,
                    "predicted_yield": predicted,
                    "confidence": pred.get("confidence"),
                }
            )
        metrics["splits"][part] = regression_metrics(y_true, y_pred)
        _write_csv(predictions_dir / f"buchwald_hartwig_{split}_{part}_predictions.csv", pred_rows)
    return metrics


def write_summary_md(path: Path, summary: Dict[str, Any]) -> None:
    lines = ["# AgentM Reaction Predictive Core Benchmark", ""]
    lines.append(f"- Status: `{summary.get('status')}`")
    lines.append(f"- Model family: `{summary.get('model_family')}`")
    lines.append(f"- Full rows: `{summary.get('full')}`")
    schneider_test = summary.get("schneider", {}).get("splits", {}).get("test", {})
    if schneider_test:
        lines.append(f"- Schneider test accuracy: `{_fmt(schneider_test.get('accuracy'))}`; macro-F1: `{_fmt(schneider_test.get('macro_f1'))}`")
    for split, payload in (summary.get("buchwald_hartwig") or {}).items():
        test = payload.get("splits", {}).get("test", {}) if isinstance(payload, dict) else {}
        if test:
            lines.append(f"- Buchwald-Hartwig `{split}` test MAE: `{_fmt(test.get('mae'))}`; RMSE: `{_fmt(test.get('rmse'))}`; R2: `{_fmt(test.get('r2'))}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _max_rows(value: str | None) -> int | None:
    if value is None:
        return None
    if str(value).lower() == "smoke":
        return 4096
    return int(value)


def _compact_model_meta(model: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in model.items() if key not in {"examples", "artifact"}}


def _confusion_rows(part: str, y_true: List[str], y_pred: List[str]) -> List[Dict[str, Any]]:
    labels = sorted(set(y_true) | set(y_pred))
    rows = []
    for true in labels:
        for pred in labels:
            rows.append({"split": part, "true_label": true, "predicted_label": pred, "count": sum(1 for t, p in zip(y_true, y_pred) if t == true and p == pred)})
    return rows


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        if not fieldnames:
            handle.write("")
            return
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value: Any) -> str:
    if value is None:
        return "NA"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
