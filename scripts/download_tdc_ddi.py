#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agentm.defaults import DEFAULT_CHEMBERTA2_MODEL_PATH, DEFAULT_IMAGEMOL_CHECKPOINT_PATH
from agentm.utils.io import ensure_dir, write_json


TDC_DDI_DATASETS = {
    "DrugBank": {
        "tdc_name": "drugbank",
        "description": "TDC DrugBank drug-drug interaction type prediction dataset",
    },
    "TWOSIDES": {
        "tdc_name": "twosides",
        "description": "TDC TWOSIDES polypharmacy side-effect interaction prediction dataset",
    },
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download and normalize TDC DDI datasets for agentM.",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=["DrugBank", "TWOSIDES"],
        help="Dataset names: DrugBank, TWOSIDES, all, or comma-separated names.",
    )
    parser.add_argument("--list-datasets", action="store_true", help="List supported TDC DDI datasets and exit.")
    parser.add_argument("--output-dir", default="data/tdc", help="Directory for normalized DDI CSV files.")
    parser.add_argument("--cache-dir", default="data/tdc/raw", help="TDC raw-data cache directory.")
    parser.add_argument("--config-dir", default="configs/tdc", help="Directory for generated DDI YAML configs.")
    parser.add_argument("--split-type", choices=["random", "cold_drug"], default="cold_drug")
    parser.add_argument("--feature-set", default="all_features")
    parser.add_argument("--model-type", default="random_forest")
    parser.add_argument("--random-seed", type=int, default=13)
    parser.add_argument("--imagemol-checkpoint-path", default=DEFAULT_IMAGEMOL_CHECKPOINT_PATH)
    parser.add_argument("--smiles-encoder-model-name", default=DEFAULT_CHEMBERTA2_MODEL_PATH)
    parser.add_argument("--no-write-configs", action="store_true")
    parser.add_argument(
        "--standardize-smiles",
        action="store_true",
        help="Canonicalize SMILES with RDKit. This can be slow for TWOSIDES.",
    )
    args = parser.parse_args()

    if args.list_datasets:
        print(json.dumps(TDC_DDI_DATASETS, indent=2, ensure_ascii=False))
        return 0

    dataset_names = expand_dataset_names(args.datasets)
    output_dir = _resolve(args.output_dir)
    cache_dir = _resolve(args.cache_dir)
    config_dir = _resolve(args.config_dir)
    ensure_dir(output_dir)
    ensure_dir(cache_dir)
    if not args.no_write_configs:
        ensure_dir(config_dir)

    results = []
    for dataset_name in dataset_names:
        results.append(
            download_one_dataset(
                dataset_name=dataset_name,
                output_dir=output_dir,
                cache_dir=cache_dir,
                config_dir=config_dir,
                args=args,
            )
        )

    summary = {
        "n_datasets": len(results),
        "datasets": results,
        "output_dir": str(output_dir),
        "cache_dir": str(cache_dir),
        "config_dir": None if args.no_write_configs else str(config_dir),
        "note": (
            "TDC DDI datasets are multi-class interaction-type datasets. "
            "agentM saves the original TDC label in label/interaction_type; "
            "binary DDI-risk supervised training requires a separate negative-sampling or multiclass-training decision."
        ),
    }
    summary_path = output_dir / "tdc_ddi_download_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"summary_path": str(summary_path), **summary}, indent=2, ensure_ascii=False))
    return 0


def download_one_dataset(
    *,
    dataset_name: str,
    output_dir: Path,
    cache_dir: Path,
    config_dir: Path,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    spec = TDC_DDI_DATASETS[dataset_name]
    DDI = load_tdc_ddi_class()
    loader = DDI(name=spec["tdc_name"], path=str(cache_dir))
    raw_df = loader.get_data(format="df")

    task_dir = ensure_dir(output_dir / dataset_name)
    raw_path = task_dir / f"{dataset_name}_raw.csv"
    raw_df.to_csv(raw_path, index=False)

    clean_df, invalid_df, column_map = normalize_tdc_ddi_dataframe(
        raw_df,
        dataset_name=dataset_name,
        standardize_smiles=bool(args.standardize_smiles),
    )
    data_path = task_dir / f"{dataset_name}.csv"
    invalid_path = task_dir / f"{dataset_name}_invalid_rows.csv"
    clean_df.to_csv(data_path, index=False)
    invalid_df.to_csv(invalid_path, index=False)

    config_path = None
    if not args.no_write_configs:
        config_path = write_ddi_training_config(
            config_dir / f"ddi_{_safe_name(dataset_name)}.yaml",
            dataset_name=dataset_name,
            data_path=_relative_to_root(data_path),
            split_type=args.split_type,
            output_dir=f"outputs/tdc_ddi/{dataset_name}",
            feature_set=args.feature_set,
            model_type=args.model_type,
            random_seed=args.random_seed,
            imagemol_checkpoint_path=args.imagemol_checkpoint_path,
            smiles_encoder_model_name=args.smiles_encoder_model_name,
        )

    metadata = {
        "dataset": dataset_name,
        "tdc_name": spec["tdc_name"],
        "description": spec["description"],
        "task_type": "ddi_prediction",
        "tdc_task": "multi_pred/DDI",
        "label_mode": "multiclass_interaction_type",
        "standardize_smiles": bool(args.standardize_smiles),
        "raw_columns": list(raw_df.columns),
        "column_map": column_map,
        "counts": {
            "raw_rows": int(len(raw_df)),
            "rows": int(len(clean_df)),
            "invalid_rows": int(len(invalid_df)),
            "unique_drug_a": int(clean_df["drug_a_smiles"].nunique()) if not clean_df.empty else 0,
            "unique_drug_b": int(clean_df["drug_b_smiles"].nunique()) if not clean_df.empty else 0,
            "unique_drugs": int(pd.unique(pd.concat([clean_df["drug_a_smiles"], clean_df["drug_b_smiles"]], ignore_index=True)).size)
            if not clean_df.empty
            else 0,
            "unique_pairs": int(clean_df[["drug_a_smiles", "drug_b_smiles"]].drop_duplicates().shape[0]) if not clean_df.empty else 0,
            "unique_labels": int(clean_df["label"].nunique()) if not clean_df.empty else 0,
        },
        "label_counts_top20": _value_counts(clean_df["label"], topn=20) if not clean_df.empty else {},
        "files": {
            "raw": _relative_to_root(raw_path),
            "normalized": _relative_to_root(data_path),
            "invalid_rows": _relative_to_root(invalid_path),
            **({"config": _relative_to_root(config_path)} if config_path else {}),
        },
        "warning": (
            "The normalized label column preserves TDC multiclass interaction labels. "
            "Do not treat these configs as final binary DDI-risk benchmarks until agentM multiclass DDI metrics "
            "or explicit negative sampling are enabled."
        ),
    }
    metadata_path = write_json(task_dir / "metadata.json", metadata)
    return {
        "dataset": dataset_name,
        "tdc_name": spec["tdc_name"],
        "counts": metadata["counts"],
        "files": {**metadata["files"], "metadata": _relative_to_root(metadata_path)},
    }


def normalize_tdc_ddi_dataframe(
    df: pd.DataFrame,
    *,
    dataset_name: str,
    standardize_smiles: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Optional[str]]]:
    drug_a_col = _first_present(df, ["Drug1", "drug1", "drug_a_smiles", "Drug1_SMILES", "Drug_A"])
    drug_b_col = _first_present(df, ["Drug2", "drug2", "drug_b_smiles", "Drug2_SMILES", "Drug_B"])
    drug_a_id_col = _first_present(df, ["Drug1_ID", "drug1_id", "drug_a_id", "Drug1ID", "Drug_A_ID"], required=False)
    drug_b_id_col = _first_present(df, ["Drug2_ID", "drug2_id", "drug_b_id", "Drug2ID", "Drug_B_ID"], required=False)
    label_col = _first_present(df, ["Y", "y", "Label", "label", "interaction_type", "Side_Effect"])

    work = pd.DataFrame(
        {
            "drug_a_smiles": df[drug_a_col].astype(str).str.strip(),
            "drug_b_smiles": df[drug_b_col].astype(str).str.strip(),
            "label": df[label_col],
            "label_original": df[label_col],
            "interaction_type": df[label_col].astype(str),
            "tdc_dataset": dataset_name,
        }
    )
    work["drug_a_id"] = df[drug_a_id_col].astype(str).str.strip() if drug_a_id_col else ""
    work["drug_b_id"] = df[drug_b_id_col].astype(str).str.strip() if drug_b_id_col else ""
    work["drug_a_name"] = work["drug_a_id"]
    work["drug_b_name"] = work["drug_b_id"]
    work["mechanism_text"] = (
        "TDC "
        + dataset_name
        + " interaction label: "
        + work["interaction_type"].astype(str)
    )

    missing_mask = (
        work["drug_a_smiles"].eq("")
        | work["drug_b_smiles"].eq("")
        | work["drug_a_smiles"].str.lower().eq("nan")
        | work["drug_b_smiles"].str.lower().eq("nan")
        | work["label"].isna()
    )
    invalid = work.loc[missing_mask].copy()
    if not invalid.empty:
        invalid["error"] = "missing_drug_or_label"
    clean = work.loc[~missing_mask].copy()

    if standardize_smiles:
        clean, std_invalid = _standardize_pair_smiles(clean)
        if not std_invalid.empty:
            invalid = pd.concat([invalid, std_invalid], ignore_index=True)

    try:
        clean["label"] = pd.to_numeric(clean["label"])
    except Exception:
        pass

    ordered_columns = [
        "drug_a_smiles",
        "drug_b_smiles",
        "drug_a_name",
        "drug_b_name",
        "label",
        "interaction_type",
        "mechanism_text",
        "drug_a_id",
        "drug_b_id",
        "label_original",
        "tdc_dataset",
    ]
    return clean[ordered_columns].reset_index(drop=True), invalid.reset_index(drop=True), {
        "drug_a_smiles": drug_a_col,
        "drug_b_smiles": drug_b_col,
        "drug_a_id": drug_a_id_col,
        "drug_b_id": drug_b_id_col,
        "label": label_col,
    }


def load_tdc_ddi_class():
    existing_package = sys.modules.get("tdc.multi_pred")
    if existing_package is not None:
        existing_ddi = getattr(existing_package, "DDI", None)
        if existing_ddi is not None:
            return existing_ddi
        try:
            from tdc.multi_pred.ddi import DDI

            setattr(existing_package, "DDI", DDI)
            return DDI
        except Exception:
            pass

    try:
        from tdc.multi_pred import DDI

        return DDI
    except ModuleNotFoundError as exc:
        if exc.name != "tiledbsoma":
            raise
    except ImportError:
        raise

    # Recent TDC releases import optional single-cell dependencies from
    # tdc.multi_pred.__init__ before exposing DDI. Bypass that package
    # initializer so DDI can still be used in a lightweight TDC environment.
    import tdc

    multi_pred_dir = Path(tdc.__file__).resolve().parent / "multi_pred"
    package = types.ModuleType("tdc.multi_pred")
    package.__path__ = [str(multi_pred_dir)]
    package.__package__ = "tdc.multi_pred"
    sys.modules["tdc.multi_pred"] = package
    setattr(tdc, "multi_pred", package)
    from tdc.multi_pred.ddi import DDI

    setattr(package, "DDI", DDI)
    return DDI


def write_ddi_training_config(
    path: str | Path,
    *,
    dataset_name: str,
    data_path: str | Path,
    split_type: str,
    output_dir: str | Path,
    feature_set: str,
    model_type: str,
    random_seed: int,
    imagemol_checkpoint_path: str,
    smiles_encoder_model_name: str,
) -> Path:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to write DDI YAML configs.") from exc

    config = {
        "task_type": "ddi_prediction",
        "dataset_name": dataset_name,
        "data_path": str(data_path),
        "drug_a_smiles_column": "drug_a_smiles",
        "drug_b_smiles_column": "drug_b_smiles",
        "drug_a_name_column": "drug_a_name",
        "drug_b_name_column": "drug_b_name",
        "label_column": "label",
        "split_type": split_type,
        "feature_set": feature_set,
        "model_type": model_type,
        "use_llm_rules": True,
        "use_imagemol": True,
        "use_smiles_encoder": True,
        "use_visual_semantic": True,
        "use_evidence_retrieval": True,
        "imagemol_checkpoint_path": imagemol_checkpoint_path,
        "smiles_encoder_model_name": smiles_encoder_model_name,
        "smiles_encoder_device": "auto",
        "smiles_pooling": "cls",
        "smiles_local_files_only": True,
        "llm_profile": "qwen_flash",
        "use_mllm_visual_semantic": False,
        "mllm_profile": "kimi_k2_6_bailian",
        "output_dir": str(output_dir),
        "random_seed": int(random_seed),
        "tdc_task": "multi_pred/DDI",
        "tdc_label_mode": "multiclass_interaction_type",
        "positive_label": 1,
        "positive_label_meaning": "DDI risk/interaction",
        "negative_label": 0,
        "negative_label_meaning": "no DDI risk/interaction",
        "decision_threshold": 0.5,
        "warning": (
            "This config points to TDC multiclass DDI labels. agentM's current DDI metrics are binary; "
            "enable multiclass DDI metrics or build a binary negative-sampled file before using this as a benchmark."
        ),
    }
    target = Path(path)
    ensure_dir(target.parent)
    target.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return target


def expand_dataset_names(values: Iterable[str] | None) -> List[str]:
    if not values:
        return list(TDC_DDI_DATASETS)
    names: List[str] = []
    aliases = {key.lower(): key for key in TDC_DDI_DATASETS}
    for value in values:
        for item in str(value).split(","):
            name = item.strip()
            if not name:
                continue
            if name.lower() == "all":
                names.extend(TDC_DDI_DATASETS)
                continue
            canonical = aliases.get(name.lower())
            if canonical is None:
                known = ", ".join(TDC_DDI_DATASETS)
                raise ValueError(f"Unknown TDC DDI dataset '{name}'. Known datasets: {known}")
            names.append(canonical)
    deduped: List[str] = []
    seen = set()
    for name in names:
        if name not in seen:
            deduped.append(name)
            seen.add(name)
    return deduped


def _standardize_pair_smiles(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    from agentm.tools.smiles import standardize_smiles

    clean_rows = []
    invalid_rows = []
    for row_idx, row in df.iterrows():
        a, err_a = standardize_smiles(str(row["drug_a_smiles"]))
        b, err_b = standardize_smiles(str(row["drug_b_smiles"]))
        if err_a or err_b or not a or not b:
            item = row.to_dict()
            item["row_index"] = row_idx
            item["error"] = f"invalid_smiles_a={err_a or ''}; invalid_smiles_b={err_b or ''}"
            invalid_rows.append(item)
            continue
        item = row.to_dict()
        item["drug_a_smiles"] = a
        item["drug_b_smiles"] = b
        clean_rows.append(item)
    return pd.DataFrame(clean_rows), pd.DataFrame(invalid_rows)


def _first_present(df: pd.DataFrame, names: List[str], *, required: bool = True) -> Optional[str]:
    for name in names:
        if name in df.columns:
            return name
    if required:
        raise ValueError(f"Could not find any of columns {names}; available columns: {list(df.columns)}")
    return None


def _resolve(path: str) -> Path:
    target = Path(path).expanduser()
    if target.is_absolute():
        return target
    return ROOT / target


def _relative_to_root(path: str | Path | None) -> str:
    if path is None:
        return ""
    target = Path(path)
    try:
        return str(target.resolve().relative_to(ROOT))
    except Exception:
        return str(target)


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)


def _value_counts(series: pd.Series, *, topn: int) -> Dict[str, int]:
    counts = series.value_counts(dropna=False).head(topn)
    return {str(index): int(value) for index, value in counts.items()}


if __name__ == "__main__":
    raise SystemExit(main())
