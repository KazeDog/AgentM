#!/usr/bin/env python
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import pandas as pd
from rdkit import rdBase


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = ROOT
DEFAULT_RESOURCE_ROOT = WORKSPACE_ROOT
for path in [ROOT, DEFAULT_RESOURCE_ROOT]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from agentm.data.splits import random_scaffold_split, random_split, scaffold_split
from agentm.models.tasks import task_mode
from agentm.tools.smiles import standardize_smiles
from agentm.runtime.scientific_metrics import (
    evaluate_scientific_case,
    label_key,
    summarize_scientific_rows,
)


DEFAULT_CASE_FILE = ROOT / "configs" / "molecular_safety_cases.jsonl"
DEFAULT_BENCHMARK_ROOT = (
    ROOT / "outputs" / "benchmark_balanced100_heuristic_strict_es_20260701"
)
DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "balanced100_scientific_audit_20260724"
SPLIT_ORDER = {"train": 0, "valid": 1, "test": 2}


class Canonicalizer:
    def __init__(self) -> None:
        self.cache: Dict[str, Optional[str]] = {}

    def __call__(self, value: Any) -> Optional[str]:
        text = str(value or "").strip()
        if not text:
            return None
        if text not in self.cache:
            canonical, error = standardize_smiles(text)
            self.cache[text] = None if error else canonical
        return self.cache[text]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit Balanced100 against the exact supervised-model splits and report "
            "held-out scientific performance separately from workflow pass rate."
        )
    )
    parser.add_argument("--case-file", default=str(DEFAULT_CASE_FILE))
    parser.add_argument("--benchmark-root", default=str(DEFAULT_BENCHMARK_ROOT))
    parser.add_argument("--resource-root", default=str(DEFAULT_RESOURCE_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    case_file = _resolve(args.case_file, ROOT)
    benchmark_root = _resolve(args.benchmark_root, ROOT)
    resource_root = _resolve(args.resource_root, ROOT)
    output_root = _resolve(args.output_root, ROOT)
    output_root.mkdir(parents=True, exist_ok=True)

    cases = read_jsonl(case_file)
    canonicalize = Canonicalizer()
    admet_cache: Dict[str, Dict[str, Any]] = {}
    ddi_cache: Dict[str, Dict[str, Any]] = {}
    audited_rows: List[Dict[str, Any]] = []

    for case in cases:
        report_path = benchmark_root / str(case.get("name") or "") / "multi_agent_report.json"
        report = read_json(report_path) if report_path.exists() else {}
        audited_rows.append(
            audit_case(
                case,
                report=report,
                report_path=report_path,
                resource_root=resource_root,
                canonicalize=canonicalize,
                admet_cache=admet_cache,
                ddi_cache=ddi_cache,
            )
        )

    summary = build_summary(
        audited_rows,
        case_file=case_file,
        benchmark_root=benchmark_root,
        resource_root=resource_root,
        admet_cache=admet_cache,
        ddi_cache=ddi_cache,
    )
    write_json(output_root / "scientific_audit.json", summary)
    write_csv(output_root / "case_scientific_audit.csv", audited_rows)
    write_enriched_casefile(
        output_root / "balanced100_cases_with_scientific_audit.jsonl",
        cases,
        audited_rows,
    )
    (output_root / "scientific_audit.md").write_text(
        render_markdown(summary, audited_rows),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "summary_json": str(output_root / "scientific_audit.json"),
                "summary_markdown": str(output_root / "scientific_audit.md"),
                "case_csv": str(output_root / "case_scientific_audit.csv"),
                "n_cases": len(audited_rows),
                "n_held_out_evaluable": summary["scientific_performance"]["coverage"][
                    "n_evaluable"
                ],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


def audit_case(
    case: Mapping[str, Any],
    *,
    report: Mapping[str, Any],
    report_path: Path,
    resource_root: Path,
    canonicalize: Canonicalizer,
    admet_cache: MutableMapping[str, Dict[str, Any]],
    ddi_cache: MutableMapping[str, Dict[str, Any]],
) -> Dict[str, Any]:
    task_type = str(case.get("task_type") or "")
    dataset_name = str(case.get("dataset_name") or "")
    mode = task_mode(task_type, dataset_name)
    model_path = selected_model_path(report)
    if model_path is None:
        if task_type == "ddi_prediction":
            model_path = (
                resource_root
                / "outputs"
                / "drugbank_ddi_full_low_token"
                / "random_all_features"
                / "model.pkl"
            )
        else:
            model_path = resource_root / "outputs" / "tdc" / dataset_name / "model.pkl"
    elif not model_path.is_absolute():
        model_path = resource_root / model_path

    prediction = first_mapping(report.get("predictions"))
    prediction_value = prediction.get("label")
    if mode == "regression":
        prediction_value = (
            prediction.get("score")
            if prediction.get("score") is not None
            else prediction.get("label")
        )

    row: Dict[str, Any] = {
        "name": str(case.get("name") or ""),
        "category": str(case.get("category") or ""),
        "task_type": task_type,
        "dataset_name": dataset_name,
        "scientific_mode": mode,
        "request": case.get("request"),
        "report_path": str(report_path),
        "report_found": bool(report),
        "model_path": str(model_path),
        "model_type": prediction.get("model_type"),
        "prediction_value": _json_scalar(prediction_value),
        "gold_value": None,
        "gold_split": None,
        "gold_source": None,
        "match_count": 0,
        "match_count_by_split": {},
        "matched_labels": [],
        "membership_status": "pending",
        "exact_training_overlap": False,
        "scientific_evaluable": False,
        "scientific_status": "pending",
        "scientific_correct": None,
        "absolute_error": None,
        "training_drug_overlap_count": None,
        "generalization_regime": None,
    }

    if task_type == "ddi_prediction":
        source = load_ddi_source(
            model_path.parent,
            canonicalize=canonicalize,
            cache=ddi_cache,
        )
        query = ddi_query_from_report(report, canonicalize=canonicalize)
        row["query_canonical"] = list(query) if query else []
        hits = source["index"].get(query, []) if query else []
        reverse_query = (query[1], query[0]) if query and query[0] != query[1] else None
        reverse_hits = source["index"].get(reverse_query, []) if reverse_query else []
        row["reverse_direction_match_count"] = len(reverse_hits)
        row["reverse_direction_splits"] = sorted(
            {str(hit.get("split")) for hit in reverse_hits},
            key=lambda value: SPLIT_ORDER.get(value, 99),
        )
        row["reverse_direction_labels"] = sorted(
            {
                label_key(hit.get("label"))
                for hit in reverse_hits
                if label_key(hit.get("label")) is not None
            },
            key=str,
        )
        if query:
            train_overlap = sum(1 for molecule in query if molecule in source["train_drugs"])
            row["training_drug_overlap_count"] = train_overlap
            row["generalization_regime"] = ddi_generalization_regime(train_overlap)
        row["gold_source"] = source["source_id"]
        row["split_strategy"] = source.get("split_strategy")
        row["data_hash_verified"] = source.get("data_hash_verified")
    else:
        source = load_admet_source(
            dataset_name,
            model_path.parent,
            resource_root=resource_root,
            canonicalize=canonicalize,
            cache=admet_cache,
        )
        query = single_query_from_report(case, report, canonicalize=canonicalize)
        row["query_canonical"] = query
        hits = source["index"].get(query, []) if query else []
        row["gold_source"] = source["source_id"]
        row["split_strategy"] = source.get("split_strategy")
        row["data_hash_verified"] = source.get("data_hash_verified")

    membership = classify_membership(hits)
    if (
        task_type == "ddi_prediction"
        and membership.get("membership_status") == "not_found"
        and int(row.get("reverse_direction_match_count") or 0) > 0
    ):
        membership["membership_status"] = "reverse_direction_only"
    row.update(membership)
    row["exact_training_overlap"] = "train" in membership.get("membership_splits", [])
    scientific = evaluate_scientific_case(
        gold_value=row.get("gold_value"),
        prediction_value=row.get("prediction_value"),
        mode=mode,
        gold_split=str(row.get("gold_split") or ""),
    )
    row.update(scientific)
    descriptive = evaluate_scientific_case(
        gold_value=row.get("gold_value"),
        prediction_value=row.get("prediction_value"),
        mode=mode,
        gold_split=str(row.get("gold_split") or ""),
        allow_non_test=True,
    )
    row["descriptive_evaluable"] = descriptive["scientific_evaluable"]
    row["descriptive_correct"] = descriptive["scientific_correct"]
    row["descriptive_absolute_error"] = descriptive["absolute_error"]
    return row


def load_admet_source(
    dataset_name: str,
    model_dir: Path,
    *,
    resource_root: Path,
    canonicalize: Canonicalizer,
    cache: MutableMapping[str, Dict[str, Any]],
) -> Dict[str, Any]:
    cache_key = str(model_dir.resolve())
    if cache_key in cache:
        return cache[cache_key]

    reproducibility_path = model_dir / "reproducibility.json"
    reproducibility = read_json(reproducibility_path)
    config = reproducibility.get("config") if isinstance(reproducibility.get("config"), Mapping) else {}
    split_strategy = str(config.get("split_type") or "scaffold")
    seed = int(config.get("random_seed") or 13)
    smiles_column = str(config.get("smiles_column") or "smiles")
    label_column = str(config.get("label_column") or "label")
    exact_split_paths = {
        split: model_dir / f"{split}_split.csv" for split in ["train", "valid", "test"]
    }
    frames: Dict[str, pd.DataFrame]
    source_kind: str

    if all(path.exists() for path in exact_split_paths.values()):
        frames = {split: pd.read_csv(path) for split, path in exact_split_paths.items()}
        source_kind = "saved_model_split"
        data_path = _configured_data_path(config, resource_root, dataset_name)
    else:
        data_path = _configured_data_path(config, resource_root, dataset_name)
        dataframe = pd.read_csv(data_path)
        with rdBase.BlockLogs():
            split_frames = split_single_dataframe(
                dataframe,
                split_strategy=split_strategy,
                smiles_column=smiles_column,
                seed=seed,
            )
        frames = dict(zip(["train", "valid", "test"], split_frames))
        source_kind = "deterministic_split_reconstruction"

    index: Dict[str, List[Dict[str, Any]]] = {}
    for split, frame in frames.items():
        if smiles_column not in frame.columns or label_column not in frame.columns:
            raise ValueError(
                f"{dataset_name} split {split} lacks {smiles_column!r}/{label_column!r}"
            )
        for row_index, record in frame.iterrows():
            canonical = canonicalize(record[smiles_column])
            if not canonical:
                continue
            row_id = record.get("_agentm_row_id", record.get("row_id", row_index))
            index.setdefault(canonical, []).append(
                {
                    "split": split,
                    "label": _json_scalar(record[label_column]),
                    "row_id": _json_scalar(row_id),
                }
            )

    expected_hash = _nested_get(
        reproducibility,
        ["artifacts", "data_path", "sha256"],
    )
    current_hash = sha256_file(data_path) if data_path.exists() else None
    source = {
        "dataset_name": dataset_name,
        "source_kind": source_kind,
        "source_id": f"{source_kind}:{model_dir}",
        "model_dir": str(model_dir),
        "reproducibility_path": str(reproducibility_path),
        "data_path": str(data_path),
        "data_hash_expected": expected_hash,
        "data_hash_current": current_hash,
        "data_hash_verified": bool(expected_hash and current_hash == expected_hash),
        "split_strategy": split_strategy,
        "random_seed": seed,
        "split_counts": {split: int(len(frame)) for split, frame in frames.items()},
        "index": index,
    }
    cache[cache_key] = source
    return source


def load_ddi_source(
    model_dir: Path,
    *,
    canonicalize: Canonicalizer,
    cache: MutableMapping[str, Dict[str, Any]],
) -> Dict[str, Any]:
    cache_key = str(model_dir.resolve())
    if cache_key in cache:
        return cache[cache_key]

    paths = {split: model_dir / f"{split}_split.csv" for split in ["train", "valid", "test"]}
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"DDI model split files are required; missing: {missing}")
    frames = {
        split: pd.read_csv(
            path,
            usecols=lambda column: column
            in {
                "_agentm_row_id",
                "row_id",
                "drug_a_smiles",
                "drug_b_smiles",
                "label",
            },
        )
        for split, path in paths.items()
    }
    unique_smiles: set[str] = set()
    for frame in frames.values():
        unique_smiles.update(frame["drug_a_smiles"].dropna().astype(str))
        unique_smiles.update(frame["drug_b_smiles"].dropna().astype(str))
    canonical_lookup = {smiles: canonicalize(smiles) for smiles in unique_smiles}

    index: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    train_drugs: set[str] = set()
    for split, frame in frames.items():
        row_id_column = (
            frame["_agentm_row_id"]
            if "_agentm_row_id" in frame.columns
            else frame.get("row_id", pd.Series(frame.index, index=frame.index))
        )
        for row_id, raw_a, raw_b, label in zip(
            row_id_column,
            frame["drug_a_smiles"],
            frame["drug_b_smiles"],
            frame["label"],
        ):
            molecule_a = canonical_lookup.get(str(raw_a))
            molecule_b = canonical_lookup.get(str(raw_b))
            if not molecule_a or not molecule_b:
                continue
            pair = (molecule_a, molecule_b)
            index.setdefault(pair, []).append(
                {
                    "split": split,
                    "label": _json_scalar(label),
                    "row_id": _json_scalar(row_id),
                }
            )
            if split == "train":
                train_drugs.update((molecule_a, molecule_b))

    reproducibility = read_json(model_dir / "reproducibility.json")
    config = reproducibility.get("config") if isinstance(reproducibility.get("config"), Mapping) else {}
    source = {
        "dataset_name": str(config.get("dataset_name") or "DrugBank"),
        "source_kind": "saved_model_split",
        "source_id": f"saved_model_split:{model_dir}",
        "model_dir": str(model_dir),
        "split_strategy": str(config.get("split_type") or "unknown"),
        "random_seed": int(config.get("random_seed") or 13),
        "split_counts": {split: int(len(frame)) for split, frame in frames.items()},
        "data_hash_verified": _reproducibility_data_hash_verified(
            reproducibility,
            project_root=DEFAULT_RESOURCE_ROOT,
        ),
        "index": index,
        "train_drugs": train_drugs,
    }
    cache[cache_key] = source
    return source


def classify_membership(hits: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    materialized = [dict(hit) for hit in hits]
    split_counts = Counter(str(hit.get("split") or "unknown") for hit in materialized)
    splits = sorted(split_counts, key=lambda value: SPLIT_ORDER.get(value, 99))
    labels_by_key: Dict[Any, Any] = {}
    for hit in materialized:
        key = label_key(hit.get("label"))
        if key is not None:
            labels_by_key[key] = hit.get("label")
    labels = [_json_scalar(labels_by_key[key]) for key in sorted(labels_by_key, key=str)]

    result: Dict[str, Any] = {
        "match_count": len(materialized),
        "match_count_by_split": dict(split_counts),
        "membership_splits": splits,
        "matched_labels": labels,
        "matched_rows": sorted(
            materialized,
            key=lambda hit: (
                SPLIT_ORDER.get(str(hit.get("split")), 99),
                str(hit.get("row_id")),
            ),
        ),
        "gold_value": None,
        "gold_split": None,
    }
    if not materialized:
        result["membership_status"] = "not_found"
    elif len(splits) > 1:
        result["membership_status"] = "multiple_splits"
    elif len(labels) != 1:
        result["membership_status"] = "ambiguous_label"
    else:
        split = splits[0]
        result["gold_value"] = labels[0]
        result["gold_split"] = split
        result["membership_status"] = {
            "train": "training_overlap",
            "valid": "validation_only",
            "test": "held_out_test",
        }.get(split, "unknown_split")
    return result


def split_single_dataframe(
    dataframe: pd.DataFrame,
    *,
    split_strategy: str,
    smiles_column: str,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if split_strategy in {"scaffold", "deterministic_scaffold"}:
        return scaffold_split(dataframe, smiles_column=smiles_column, seed=seed)
    if split_strategy in {"random_scaffold", "random_scaffold_split"}:
        return random_scaffold_split(dataframe, smiles_column=smiles_column, seed=seed)
    if split_strategy == "random":
        return random_split(dataframe, seed=seed)
    raise ValueError(f"Unsupported ADMET split strategy: {split_strategy}")


def build_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    case_file: Path,
    benchmark_root: Path,
    resource_root: Path,
    admet_cache: Mapping[str, Mapping[str, Any]],
    ddi_cache: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    scientific = summarize_scientific_rows(rows)
    descriptive_rows: List[Dict[str, Any]] = []
    for row in rows:
        clone = dict(row)
        clone["scientific_evaluable"] = bool(row.get("descriptive_evaluable"))
        clone["scientific_status"] = (
            "evaluated" if clone["scientific_evaluable"] else row.get("scientific_status")
        )
        clone["scientific_correct"] = row.get("descriptive_correct")
        clone["absolute_error"] = row.get("descriptive_absolute_error")
        descriptive_rows.append(clone)
    descriptive = summarize_scientific_rows(descriptive_rows)

    category_status: Dict[str, Counter[str]] = {}
    for row in rows:
        category_status.setdefault(str(row.get("category") or "missing"), Counter()).update(
            [str(row.get("membership_status") or "missing")]
        )
    ddi_test = [
        row
        for row in rows
        if row.get("category") == "ddi" and row.get("membership_status") == "held_out_test"
    ]
    ddi_regime_counts = Counter(
        str(row.get("generalization_regime") or "missing") for row in ddi_test
    )
    ddi_scientific = (scientific.get("by_category") or {}).get("ddi", {}).get(
        "classification",
        {"n": 0, "status": "not_evaluable"},
    )
    admet_scientific = (scientific.get("by_category") or {}).get("admet")
    if not admet_scientific:
        admet_scientific = {
            "n": 0,
            "classification": {"n": 0, "status": "not_evaluable"},
            "regression": {"n": 0, "status": "not_evaluable"},
        }

    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "case_file": str(case_file),
        "benchmark_root": str(benchmark_root),
        "resource_root": str(resource_root),
        "method": {
            "admet": (
                "Use saved model split CSVs when present; otherwise deterministically "
                "reconstruct the split from each model's reproducibility config, data file, "
                "split strategy, and seed. Match canonical isomeric SMILES."
            ),
            "ddi": (
                "Match ordered canonical-SMILES pairs against the saved train/valid/test "
                "CSVs of the exact DrugBank model selected in each report. Reverse-direction "
                "records are diagnostic only because the 86-class labels and concatenated "
                "pair features are directional."
            ),
            "claim_boundary": (
                "Only consistent gold labels found exclusively in a held-out test split "
                "are included in scientific performance. Training/validation matches are "
                "reported descriptively and excluded from generalization metrics."
            ),
        },
        "coverage": {
            "n_cases": len(rows),
            "n_reports_found": sum(bool(row.get("report_found")) for row in rows),
            "n_predictions_found": sum(row.get("prediction_value") is not None for row in rows),
            "membership_status_counts": dict(
                Counter(str(row.get("membership_status") or "missing") for row in rows)
            ),
            "membership_status_by_category": {
                category: dict(counts) for category, counts in category_status.items()
            },
            "n_exact_training_overlap": sum(
                bool(row.get("exact_training_overlap")) for row in rows
            ),
            "ddi_held_out_generalization_regime_counts": dict(ddi_regime_counts),
        },
        "scientific_performance": scientific,
        "matched_case_descriptive_performance": {
            "warning": (
                "Includes training and validation matches. These values are diagnostic only "
                "and must not be described as held-out accuracy."
            ),
            **descriptive,
        },
        "claim_assessment": {
            "workflow_pass_rate": {
                "supported_claim": (
                    "The separate benchmark run completed its configured workflow/objective "
                    "checks for 100/100 cases."
                ),
                "unsupported_extension": (
                    "It does not establish 100% ADMET/DDI predictive accuracy."
                ),
            },
            "admet_prediction_performance": {
                "status": "not_estimable_from_balanced100",
                "n_held_out_evaluable": admet_scientific.get("n", 0),
                "reason": (
                    "No ADMET case in molecular safety has a consistent label in the exact "
                    "held-out test split of its selected model."
                ),
            },
            "ddi_prediction_performance": {
                "status": (
                    "limited_estimate"
                    if int(ddi_scientific.get("n") or 0) > 0
                    else "not_estimable"
                ),
                "metrics": ddi_scientific,
                "caveat": (
                    "The selected DDI model uses a random pair split. Held-out pairs can "
                    "contain drugs already seen during training; this is not cold-drug "
                    "generalization."
                ),
            },
        },
        "data_sources": {
            "admet": [_source_manifest(source) for source in admet_cache.values()],
            "ddi": [_source_manifest(source) for source in ddi_cache.values()],
        },
        "cases": [dict(row) for row in rows],
    }


def render_markdown(summary: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> str:
    coverage = summary.get("coverage") or {}
    scientific = summary.get("scientific_performance") or {}
    by_category = scientific.get("by_category") or {}
    ddi_metrics = (by_category.get("ddi") or {}).get("classification") or {}
    admet_metrics = by_category.get("admet") or {}
    ddi_regimes = coverage.get("ddi_held_out_generalization_regime_counts") or {}
    status_by_category = coverage.get("membership_status_by_category") or {}

    lines = [
        "# Balanced100 Scientific Performance Audit",
        "",
        "## Conclusion",
        "",
        (
            "- The prior `100/100` result is a workflow/objective pass rate, not a "
            "predictive-accuracy result."
        ),
        (
            f"- Held-out scientific coverage is "
            f"`{(scientific.get('coverage') or {}).get('n_evaluable', 0)}/{coverage.get('n_cases', 0)}` cases."
        ),
        (
            f"- ADMET is not estimable from this case set: "
            f"`{admet_metrics.get('n', 0)}/50` cases are held-out and evaluable."
        ),
    ]
    if ddi_metrics.get("status") == "evaluated":
        ci = ddi_metrics.get("accuracy_wilson_95ci") or [None, None]
        lines.append(
            f"- DDI held-out pair accuracy is `{ddi_metrics.get('n_correct')}/{ddi_metrics.get('n')}` "
            f"(`{ddi_metrics.get('accuracy')}`; Wilson 95% CI `{ci[0]}–{ci[1]}`)."
        )
    else:
        lines.append("- DDI held-out performance is not estimable.")
    lines.extend(
        [
            (
                f"- DDI held-out regimes are `{ddi_regimes}`. "
                "Random-pair test cases with both drugs seen in training are not cold-drug tests."
            ),
            "",
            "## Split Membership",
            "",
            "| category | training overlap | validation only | held-out test | reverse only | not found | ambiguous label |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for category in ["admet", "ddi"]:
        counts = status_by_category.get(category) or {}
        lines.append(
            f"| {category.upper()} | {counts.get('training_overlap', 0)} | "
            f"{counts.get('validation_only', 0)} | {counts.get('held_out_test', 0)} | "
            f"{counts.get('reverse_direction_only', 0)} | {counts.get('not_found', 0)} | "
            f"{counts.get('ambiguous_label', 0)} |"
        )

    lines.extend(
        [
            "",
            "## Held-out Scientific Metrics",
            "",
            "| category | mode | n | primary metrics |",
            "| --- | --- | ---: | --- |",
        ]
    )
    for category in ["admet", "ddi"]:
        payload = by_category.get(category) or {}
        classification = payload.get("classification") or {"n": 0}
        regression = payload.get("regression") or {"n": 0}
        lines.append(
            f"| {category.upper()} | classification | {classification.get('n', 0)} | "
            f"accuracy={classification.get('accuracy')}, balanced_accuracy={classification.get('balanced_accuracy')}, "
            f"macro_F1={classification.get('macro_f1')} |"
        )
        lines.append(
            f"| {category.upper()} | regression | {regression.get('n', 0)} | "
            f"MAE={regression.get('mae')}, RMSE={regression.get('rmse')}, R2={regression.get('r2')} |"
        )

    held_out_rows = [row for row in rows if row.get("membership_status") == "held_out_test"]
    lines.extend(
        [
            "",
            "## Held-out Cases",
            "",
            "| case | category | gold | prediction | correct/error | regime |",
            "| --- | --- | ---: | ---: | --- | --- |",
        ]
    )
    for row in held_out_rows:
        outcome = (
            f"correct={row.get('scientific_correct')}"
            if row.get("scientific_mode") == "classification"
            else f"absolute_error={row.get('absolute_error')}"
        )
        lines.append(
            f"| `{row.get('name')}` | {row.get('category')} | {row.get('gold_value')} | "
            f"{row.get('prediction_value')} | {outcome} | {row.get('generalization_regime') or ''} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation Rules",
            "",
            "- Only rows found exclusively in the selected model's test split contribute to held-out metrics.",
            "- Exact training matches and validation matches are excluded from generalization claims.",
            "- Reverse-direction-only DDI records are not treated as gold for the submitted A→B query.",
            "- `not_found` means the case has no dataset gold label; it is useful for workflow robustness, not accuracy.",
            "- The DDI model audited here uses a random pair split. A cold-drug benchmark is still required.",
            "",
            "Full case-level evidence is in `case_scientific_audit.csv`; machine-readable details are in `scientific_audit.json`.",
            "",
        ]
    )
    return "\n".join(lines)


def selected_model_path(report: Mapping[str, Any]) -> Optional[Path]:
    resolution = report.get("resource_resolution")
    if not isinstance(resolution, Mapping):
        return None
    model = resolution.get("model")
    if not isinstance(model, Mapping) or not model.get("selected_path"):
        return None
    return Path(str(model["selected_path"]))


def single_query_from_report(
    case: Mapping[str, Any],
    report: Mapping[str, Any],
    *,
    canonicalize: Canonicalizer,
) -> Optional[str]:
    molecules = report.get("molecules")
    if isinstance(molecules, list) and molecules and isinstance(molecules[0], Mapping):
        value = molecules[0].get("canonical_smiles") or molecules[0].get("smiles")
        canonical = canonicalize(value)
        if canonical:
            return canonical
    return canonicalize(case.get("request"))


def ddi_query_from_report(
    report: Mapping[str, Any],
    *,
    canonicalize: Canonicalizer,
) -> Optional[Tuple[str, str]]:
    molecules = report.get("molecules")
    if not isinstance(molecules, list) or len(molecules) < 2:
        return None
    canonical: List[str] = []
    for molecule in molecules[:2]:
        if not isinstance(molecule, Mapping):
            return None
        value = molecule.get("canonical_smiles") or molecule.get("smiles")
        standardized = canonicalize(value)
        if not standardized:
            return None
        canonical.append(standardized)
    return canonical[0], canonical[1]


def ddi_generalization_regime(training_drug_overlap_count: int) -> str:
    return {
        2: "pair_held_out_both_drugs_seen",
        1: "single_cold_drug",
        0: "double_cold_drug",
    }.get(int(training_drug_overlap_count), "unknown")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"Expected JSON object at {path}:{line_no}")
        rows.append(payload)
    return rows


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def first_mapping(value: Any) -> Dict[str, Any]:
    if isinstance(value, list) and value and isinstance(value[0], Mapping):
        return dict(value[0])
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (
                        json.dumps(value, ensure_ascii=False, sort_keys=True)
                        if isinstance(value, (list, dict))
                        else value
                    )
                    for key, value in row.items()
                }
            )


def write_enriched_casefile(
    path: Path,
    cases: Sequence[Mapping[str, Any]],
    audit_rows: Sequence[Mapping[str, Any]],
) -> None:
    by_name = {str(row.get("name") or ""): row for row in audit_rows}
    output: List[str] = []
    for case in cases:
        audit = by_name.get(str(case.get("name") or ""), {})
        enriched = dict(case)
        enriched["scientific_audit"] = {
            key: audit.get(key)
            for key in [
                "membership_status",
                "gold_value",
                "gold_split",
                "gold_source",
                "match_count",
                "matched_labels",
                "prediction_value",
                "scientific_mode",
                "scientific_evaluable",
                "scientific_correct",
                "absolute_error",
                "exact_training_overlap",
                "training_drug_overlap_count",
                "generalization_regime",
                "model_path",
            ]
        }
        output.append(json.dumps(enriched, ensure_ascii=False))
    path.write_text("\n".join(output) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _configured_data_path(
    config: Mapping[str, Any],
    resource_root: Path,
    dataset_name: str,
) -> Path:
    raw_path = config.get("data_path")
    path = Path(str(raw_path)) if raw_path else Path("data") / "tdc" / dataset_name / f"{dataset_name}.csv"
    return path if path.is_absolute() else resource_root / path


def _reproducibility_data_hash_verified(
    reproducibility: Mapping[str, Any],
    *,
    project_root: Path,
) -> Optional[bool]:
    artifact = _nested_get(reproducibility, ["artifacts", "data_path"])
    if not isinstance(artifact, Mapping):
        return None
    expected = artifact.get("sha256")
    resolved = artifact.get("resolved_path")
    raw = artifact.get("path")
    path = Path(str(resolved or raw or ""))
    if raw and not path.is_absolute():
        path = project_root / path
    if not expected or not path.exists():
        return None
    return sha256_file(path) == expected


def _source_manifest(source: Mapping[str, Any]) -> Dict[str, Any]:
    excluded = {"index", "train_drugs"}
    return {key: value for key, value in source.items() if key not in excluded}


def _nested_get(payload: Mapping[str, Any], keys: Sequence[str]) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _resolve(value: str | Path, base: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else base / path


def _json_scalar(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "item"):
        value = value.item()
    if pd.isna(value):
        return None
    return value


if __name__ == "__main__":
    raise SystemExit(main())
