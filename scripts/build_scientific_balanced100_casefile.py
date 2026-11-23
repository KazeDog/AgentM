#!/usr/bin/env python
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import sys
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = ROOT
RESOURCE_ROOT = WORKSPACE_ROOT
for path in [ROOT, RESOURCE_ROOT]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from agentm.models.tasks import task_mode
from agentm.runtime.scientific_metrics import classification_metrics
from scripts.audit_balanced100_scientific_performance import (
    Canonicalizer,
    classify_membership,
    load_admet_source,
    load_ddi_source,
)


DEFAULT_OUTPUT = ROOT / "configs" / "molecular_safety_scientific_cases.jsonl"
DEFAULT_MANIFEST = ROOT / "configs" / "molecular_safety_scientific_cases_manifest.json"
DEFAULT_SEED = 20260724
ADMET_DATASETS = [
    "AMES",
    "BBB_Martins",
    "DILI",
    "hERG",
    "CYP2C19_Veith",
    "Caco2_Wang",
    "Lipophilicity_AstraZeneca",
    "Solubility_AqSolDB",
    "LD50_Zhu",
    "Clearance_Microsome_AZ",
]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a deterministic 100-case benchmark containing only gold-labeled "
            "held-out ADMET examples and double-cold DrugBank DDI examples."
        )
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--resource-root", default=str(RESOURCE_ROOT))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--admet-per-dataset", type=int, default=5)
    parser.add_argument("--ddi-cases", type=int, default=50)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    output_path = _resolve(args.output, ROOT)
    manifest_path = _resolve(args.manifest, ROOT)
    resource_root = _resolve(args.resource_root, ROOT)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    canonicalize = Canonicalizer()
    admet_cache: MutableMapping[str, Dict[str, Any]] = {}
    ddi_cache: MutableMapping[str, Dict[str, Any]] = {}
    cases: List[Dict[str, Any]] = []
    selection: Dict[str, Any] = {"admet": {}, "ddi": {}}

    for dataset_name in ADMET_DATASETS:
        model_dir = resource_root / "outputs" / "tdc" / dataset_name
        source = load_admet_source(
            dataset_name,
            model_dir,
            resource_root=resource_root,
            canonicalize=canonicalize,
            cache=admet_cache,
        )
        candidates = admet_candidates(
            dataset_name,
            source,
            model_path=Path("outputs") / "tdc" / dataset_name / "model.pkl",
        )
        chosen = deterministic_sample(
            candidates,
            args.admet_per_dataset,
            seed=stable_seed(args.seed, f"admet:{dataset_name}"),
        )
        if len(chosen) < args.admet_per_dataset:
            raise ValueError(
                f"{dataset_name} has only {len(chosen)} valid held-out candidates; "
                f"need {args.admet_per_dataset}"
            )
        cases.extend(chosen)
        selection["admet"][dataset_name] = {
            "candidate_count": len(candidates),
            "selected_count": len(chosen),
            "mode": task_mode("admet_prediction", dataset_name),
            "split_strategy": source.get("split_strategy"),
            "data_hash_verified": source.get("data_hash_verified"),
            "source_id": source.get("source_id"),
        }

    ddi_model_dir = (
        resource_root
        / "outputs"
        / "drugbank_ddi_full_low_token"
        / "cold_drug_all_features"
    )
    ddi_source = load_ddi_source(
        ddi_model_dir,
        canonicalize=canonicalize,
        cache=ddi_cache,
    )
    ddi_candidates_all = ddi_double_cold_candidates(
        ddi_source,
        model_path=(
            Path("outputs")
            / "drugbank_ddi_full_low_token"
            / "cold_drug_all_features"
            / "model.pkl"
        ),
    )
    ddi_chosen = deterministic_sample(
        ddi_candidates_all,
        args.ddi_cases,
        seed=stable_seed(args.seed, "ddi:double_cold"),
    )
    if len(ddi_chosen) < args.ddi_cases:
        raise ValueError(
            f"Only {len(ddi_chosen)} double-cold DDI candidates; need {args.ddi_cases}"
        )
    cases.extend(ddi_chosen)
    selection["ddi"] = {
        "candidate_count": len(ddi_candidates_all),
        "selected_count": len(ddi_chosen),
        "split_strategy": ddi_source.get("split_strategy"),
        "training_drug_overlap_required": 0,
        "source_id": ddi_source.get("source_id"),
        "data_hash_verified": ddi_source.get("data_hash_verified"),
    }

    validate_cases(
        cases,
        expected_admet=len(ADMET_DATASETS) * args.admet_per_dataset,
        expected_ddi=args.ddi_cases,
    )
    output_path.write_text(
        "\n".join(json.dumps(case, ensure_ascii=False) for case in cases) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "case_file": str(output_path),
        "seed": args.seed,
        "selection_protocol": {
            "label_independent_sampling": True,
            "admet": (
                "Equal dataset quota; uniform seeded sampling from canonical molecules "
                "found exclusively in each selected model's scaffold test split with one "
                "consistent label."
            ),
            "ddi": (
                "Uniform seeded sampling from ordered pairs in the cold-drug model test "
                "split for which neither molecule occurs in the cold-model training split."
            ),
            "important": (
                "The benchmark is macro-balanced across selected ADMET datasets and is not "
                "weighted by natural dataset prevalence."
            ),
        },
        "counts": {
            "n_cases": len(cases),
            "category_counts": dict(Counter(case["category"] for case in cases)),
            "mode_counts": dict(Counter(case["scientific_mode"] for case in cases)),
            "dataset_counts": dict(Counter(case["dataset_name"] for case in cases)),
            "gold_split_counts": dict(Counter(case["gold_split"] for case in cases)),
        },
        "selection": selection,
        "selected_label_counts": {
            dataset: dict(
                Counter(
                    str(case["gold_value"])
                    for case in cases
                    if case["dataset_name"] == dataset
                )
            )
            for dataset in sorted({case["dataset_name"] for case in cases})
        },
        "reference_model_performance": {
            "scope": (
                "Saved supervised-model evaluation artifacts; ADMET values cover each "
                "model's full scaffold test split, while DDI also includes the exact "
                "double-cold subset and selected 50-case subset."
            ),
            "admet_full_test_by_dataset": admet_reference_performance(resource_root),
            "ddi": ddi_reference_performance(
                ddi_model_dir,
                double_cold_candidates=ddi_candidates_all,
                selected_cases=ddi_chosen,
            ),
        },
        "recommended_run": (
            "PYTHONPATH=.:../agentM python scripts/run_benchmark_suite.py "
            f"--case-file {output_path} --output-root outputs/balanced100_scientific "
            "--feature-set all_features --no-use-llm-orchestrator --no-use-llm-reasoning "
            "--no-require-real-rag --retrieval-backend local"
        ),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "case_file": str(output_path),
                "manifest": str(manifest_path),
                **manifest["counts"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


def ddi_reference_performance(
    model_dir: Path,
    *,
    double_cold_candidates: Sequence[Mapping[str, Any]],
    selected_cases: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    training_result = json.loads(
        (model_dir / "training_result.json").read_text(encoding="utf-8")
    )
    predictions = pd.read_csv(model_dir / "test_predictions.csv")
    double_cold_ids = {
        int(case["gold_row_id"]) for case in double_cold_candidates
    }
    selected_ids = {int(case["gold_row_id"]) for case in selected_cases}
    double_cold = predictions[
        predictions["row_id"].astype(int).isin(double_cold_ids)
    ]
    selected = predictions[predictions["row_id"].astype(int).isin(selected_ids)]
    if len(double_cold) != len(double_cold_ids):
        raise ValueError(
            f"Missing double-cold predictions: {len(double_cold)}/{len(double_cold_ids)}"
        )
    if len(selected) != len(selected_ids):
        raise ValueError(
            f"Missing selected DDI predictions: {len(selected)}/{len(selected_ids)}"
        )
    return {
        "scope": "saved supervised-model prediction artifacts; not a new end-to-end agent run",
        "model_dir": str(model_dir),
        "full_cold_test": {
            "n": int(training_result.get("n_test") or 0),
            **_compact_existing_metrics(training_result.get("metrics") or {}),
        },
        "double_cold_test": {
            **_compact_classification_metrics(
                classification_metrics(
                    double_cold["label_true"].tolist(),
                    double_cold["label_pred"].tolist(),
                )
            ),
            "top3_accuracy": round(_top3_accuracy(double_cold), 6),
        },
        "selected_50_double_cold": {
            **_compact_classification_metrics(
                classification_metrics(
                    selected["label_true"].tolist(),
                    selected["label_pred"].tolist(),
                )
            ),
            "top3_accuracy": round(_top3_accuracy(selected), 6),
        },
    }


def admet_reference_performance(resource_root: Path) -> Dict[str, Any]:
    output: Dict[str, Any] = {}
    for dataset_name in ADMET_DATASETS:
        result_path = (
            resource_root
            / "outputs"
            / "tdc"
            / dataset_name
            / "training_result.json"
        )
        result = json.loads(result_path.read_text(encoding="utf-8"))
        mode = task_mode("admet_prediction", dataset_name)
        metrics = result.get("metrics") or {}
        if mode == "classification":
            selected_metrics = {
                key: metrics.get(key)
                for key in [
                    "accuracy",
                    "f1",
                    "precision",
                    "recall",
                    "roc_auc",
                    "aupr",
                ]
            }
        else:
            selected_metrics = {
                key: metrics.get(key) for key in ["mae", "rmse", "r2"]
            }
        output[dataset_name] = {
            "mode": mode,
            "n_test": int(result.get("n_test") or 0),
            "split_strategy": result.get("split_type"),
            **selected_metrics,
        }
    return output


def admet_candidates(
    dataset_name: str,
    source: Mapping[str, Any],
    *,
    model_path: Path,
) -> List[Dict[str, Any]]:
    mode = task_mode("admet_prediction", dataset_name)
    candidates: List[Dict[str, Any]] = []
    for canonical_smiles, hits in (source.get("index") or {}).items():
        membership = classify_membership(hits)
        if membership.get("membership_status") != "held_out_test":
            continue
        matched_rows = membership.get("matched_rows") or []
        row_id = matched_rows[0].get("row_id") if matched_rows else len(candidates)
        candidates.append(
            {
                "name": f"scientific_admet_{slug(dataset_name)}_{slug(row_id)}",
                "request": canonical_smiles,
                "task_type": "admet_prediction",
                "dataset_name": dataset_name,
                "category": "admet",
                "expected_endpoint": slug(dataset_name),
                "notes": (
                    "Molecular safety scientific cases; deterministic label-independent sample "
                    "from the exact held-out model split."
                ),
                "scientific_mode": mode,
                "gold_value": membership["gold_value"],
                "gold_split": "test",
                "gold_source": source.get("source_id"),
                "gold_row_id": row_id,
                "split_strategy": source.get("split_strategy"),
                "selection_protocol": "uniform_seeded_within_dataset_test_candidates",
                "model_path": str(model_path),
                "feature_set": "all_features",
            }
        )
    return sorted(candidates, key=lambda row: (str(row["gold_row_id"]), row["request"]))


def ddi_double_cold_candidates(
    source: Mapping[str, Any],
    *,
    model_path: Path,
) -> List[Dict[str, Any]]:
    train_drugs = set(source.get("train_drugs") or set())
    candidates: List[Dict[str, Any]] = []
    for pair, hits in (source.get("index") or {}).items():
        membership = classify_membership(hits)
        if membership.get("membership_status") != "held_out_test":
            continue
        if any(molecule in train_drugs for molecule in pair):
            continue
        matched_rows = membership.get("matched_rows") or []
        row_id = matched_rows[0].get("row_id") if matched_rows else len(candidates)
        candidates.append(
            {
                "name": f"scientific_ddi_double_cold_{slug(row_id)}",
                "request": f"{pair[0]}, {pair[1]}",
                "task_type": "ddi_prediction",
                "dataset_name": "DDI",
                "category": "ddi",
                "expected_endpoint": "ddi_multiclass_interaction_type",
                "notes": (
                    "Molecular safety scientific cases; ordered DrugBank pair from the cold-drug "
                    "test split; neither molecule occurs in the cold-model training split."
                ),
                "scientific_mode": "classification",
                "gold_value": membership["gold_value"],
                "gold_split": "test",
                "gold_source": source.get("source_id"),
                "gold_row_id": row_id,
                "split_strategy": source.get("split_strategy"),
                "selection_protocol": "uniform_seeded_double_cold_test_candidates",
                "training_drug_overlap_count": 0,
                "generalization_regime": "double_cold_drug",
                "model_path": str(model_path),
                "feature_set": "all_features",
            }
        )
    return sorted(candidates, key=lambda row: (str(row["gold_row_id"]), row["request"]))


def deterministic_sample(
    rows: Sequence[Mapping[str, Any]],
    count: int,
    *,
    seed: int,
) -> List[Dict[str, Any]]:
    materialized = [dict(row) for row in rows]
    random.Random(seed).shuffle(materialized)
    return materialized[:count]


def _compact_classification_metrics(metrics: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        key: metrics.get(key)
        for key in [
            "n",
            "n_correct",
            "accuracy",
            "accuracy_wilson_95ci",
            "balanced_accuracy",
            "macro_precision",
            "macro_recall",
            "macro_f1",
            "weighted_f1",
            "n_labels_true",
            "n_labels_predicted",
        ]
    }


def _compact_existing_metrics(metrics: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "accuracy": metrics.get("accuracy"),
        "balanced_accuracy": metrics.get("balanced_accuracy"),
        "macro_f1": metrics.get("f1_macro"),
        "weighted_f1": metrics.get("f1_weighted"),
        "macro_precision": metrics.get("precision_macro"),
        "macro_recall": metrics.get("recall_macro"),
        "top3_accuracy": metrics.get("top3_accuracy"),
    }


def _top3_accuracy(frame: pd.DataFrame) -> float:
    if frame.empty:
        return 0.0
    correct = 0
    for gold, raw_top3 in zip(frame["label_true"], frame["top3_labels"]):
        top3 = [part.strip() for part in str(raw_top3).split(";") if part.strip()]
        if str(int(gold) if float(gold).is_integer() else gold) in top3:
            correct += 1
    return correct / len(frame)


def validate_cases(
    cases: Sequence[Mapping[str, Any]],
    *,
    expected_admet: int,
    expected_ddi: int,
) -> None:
    category_counts = Counter(str(case.get("category")) for case in cases)
    expected = {"admet": expected_admet, "ddi": expected_ddi}
    if category_counts != expected:
        raise ValueError(f"Unexpected category counts: {dict(category_counts)} != {expected}")
    names = [str(case.get("name") or "") for case in cases]
    duplicates = [name for name, count in Counter(names).items() if count > 1]
    if duplicates:
        raise ValueError(f"Duplicate case names: {duplicates[:10]}")
    invalid = [
        case.get("name")
        for case in cases
        if case.get("gold_split") != "test"
        or case.get("gold_value") is None
        or case.get("scientific_mode") not in {"classification", "regression"}
    ]
    if invalid:
        raise ValueError(f"Invalid scientific cases: {invalid[:10]}")
    ddi_invalid = [
        case.get("name")
        for case in cases
        if case.get("category") == "ddi"
        and (
            case.get("training_drug_overlap_count") != 0
            or case.get("split_strategy") != "cold_drug"
        )
    ]
    if ddi_invalid:
        raise ValueError(f"DDI cases are not double-cold: {ddi_invalid[:10]}")


def stable_seed(base_seed: int, namespace: str) -> int:
    digest = hashlib.sha256(f"{base_seed}:{namespace}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def slug(value: Any) -> str:
    text = str(value or "").lower()
    return "".join(character if character.isalnum() else "_" for character in text).strip("_")


def _resolve(value: str | Path, base: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else base / path


if __name__ == "__main__":
    raise SystemExit(main())
