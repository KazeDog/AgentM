#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Mapping

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agentm.data.tdc_datasets import (
    TDC_TASKS,
    expand_task_names,
    normalize_tdc_dataframe,
    task_spec,
    write_dataset_metadata,
    write_tdc_training_config,
)
from agentm.defaults import DEFAULT_CHEMBERTA2_MODEL_PATH, DEFAULT_IMAGEMOL_CHECKPOINT_PATH
from agentm.utils.io import ensure_dir


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download and normalize TDC ADME/Tox datasets for agentM ADMET benchmarks.",
    )
    parser.add_argument(
        "--tasks",
        nargs="*",
        default=["recommended"],
        help="TDC task names, comma-separated names, 'recommended', or 'all'.",
    )
    parser.add_argument("--list-tasks", action="store_true", help="List supported task names and exit.")
    parser.add_argument("--output-dir", default="data/tdc", help="Directory for normalized CSV files.")
    parser.add_argument("--cache-dir", default="data/tdc/raw", help="TDC raw-data cache directory.")
    parser.add_argument("--config-dir", default="configs/tdc", help="Directory for generated training YAML configs.")
    parser.add_argument("--split-type", choices=["none", "random", "scaffold"], default="scaffold")
    parser.add_argument("--frac-train", type=float, default=0.7)
    parser.add_argument("--frac-valid", type=float, default=0.1)
    parser.add_argument("--frac-test", type=float, default=0.2)
    parser.add_argument("--random-seed", type=int, default=13)
    parser.add_argument("--feature-set", default="all_features")
    parser.add_argument("--model-type", default="random_forest")
    parser.add_argument("--imagemol-checkpoint-path", default=DEFAULT_IMAGEMOL_CHECKPOINT_PATH)
    parser.add_argument("--smiles-encoder-model-name", default=DEFAULT_CHEMBERTA2_MODEL_PATH)
    parser.add_argument("--no-use-smiles-encoder", action="store_true")
    parser.add_argument("--use-mllm-visual-semantic", action="store_true")
    parser.add_argument("--llm-profile", default="qwen_flash")
    parser.add_argument("--mllm-profile", default="kimi_k2_6_bailian")
    parser.add_argument("--no-standardize-smiles", action="store_true")
    parser.add_argument("--no-write-configs", action="store_true")
    args = parser.parse_args()

    if args.list_tasks:
        print(json.dumps({"tasks": {name: spec.to_dict() for name, spec in sorted(TDC_TASKS.items())}}, indent=2))
        return 0

    task_names = expand_task_names(args.tasks)
    output_dir = _resolve(args.output_dir)
    cache_dir = _resolve(args.cache_dir)
    config_dir = _resolve(args.config_dir)
    ensure_dir(output_dir)
    ensure_dir(cache_dir)
    if not args.no_write_configs:
        ensure_dir(config_dir)

    results = []
    for name in task_names:
        spec = task_spec(name)
        result = _download_one_task(
            spec_name=name,
            output_dir=output_dir,
            cache_dir=cache_dir,
            config_dir=config_dir,
            args=args,
        )
        results.append(result)

    summary = {
        "n_tasks": len(results),
        "tasks": results,
        "output_dir": str(output_dir),
        "config_dir": None if args.no_write_configs else str(config_dir),
    }
    summary_path = output_dir / "tdc_download_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"summary_path": str(summary_path), **summary}, indent=2, ensure_ascii=False))
    return 0


def _download_one_task(*, spec_name: str, output_dir: Path, cache_dir: Path, config_dir: Path, args: argparse.Namespace) -> Dict[str, Any]:
    spec = task_spec(spec_name)
    loader = _load_tdc_dataset(spec.name, spec.collection, cache_dir)
    task_dir = ensure_dir(output_dir / spec.name)
    split_files: Dict[str, str] = {}
    split_counts: Dict[str, int] = {}
    invalid_counts: Dict[str, int] = {}

    if args.split_type == "none":
        raw_df = _get_full_dataframe(loader)
        clean, invalid = normalize_tdc_dataframe(raw_df, spec, standardize=not args.no_standardize_smiles)
        data_path = task_dir / f"{spec.name}.csv"
        invalid_path = task_dir / f"{spec.name}_invalid_rows.csv"
        clean.to_csv(data_path, index=False)
        invalid.to_csv(invalid_path, index=False)
        split_files["full"] = _relative_to_root(data_path)
        split_counts["full"] = int(len(clean))
        invalid_counts["full"] = int(len(invalid))
    else:
        splits = _get_split_dataframe(
            loader,
            method=args.split_type,
            seed=args.random_seed,
            frac=[args.frac_train, args.frac_valid, args.frac_test],
        )
        clean_parts = []
        invalid_parts = []
        for split_name, raw_df in splits.items():
            clean, invalid = normalize_tdc_dataframe(raw_df, spec, standardize=not args.no_standardize_smiles)
            split_path = task_dir / f"{split_name}.csv"
            invalid_path = task_dir / f"{split_name}_invalid_rows.csv"
            clean.to_csv(split_path, index=False)
            invalid.to_csv(invalid_path, index=False)
            split_files[split_name] = _relative_to_root(split_path)
            split_counts[split_name] = int(len(clean))
            invalid_counts[split_name] = int(len(invalid))
            clean_parts.append(clean)
            if not invalid.empty:
                invalid_parts.append(invalid.assign(split=split_name))
        full = pd.concat(clean_parts, ignore_index=True) if clean_parts else pd.DataFrame(columns=["smiles", "label"])
        full_path = task_dir / f"{spec.name}.csv"
        full.to_csv(full_path, index=False)
        split_files["full"] = _relative_to_root(full_path)
        split_counts["full"] = int(len(full))
        if invalid_parts:
            pd.concat(invalid_parts, ignore_index=True).to_csv(task_dir / f"{spec.name}_invalid_rows.csv", index=False)

    config_path = None
    if not args.no_write_configs:
        config_path = write_tdc_training_config(
            config_dir / f"admet_{_safe_name(spec.name)}.yaml",
            spec=spec,
            data_path=split_files["full"],
            split_type="scaffold" if args.split_type == "none" else args.split_type,
            output_dir=f"outputs/tdc/{spec.name}",
            feature_set=args.feature_set,
            model_type=args.model_type,
            random_seed=args.random_seed,
            imagemol_checkpoint_path=args.imagemol_checkpoint_path,
            smiles_encoder_model_name=args.smiles_encoder_model_name,
            use_smiles_encoder=not args.no_use_smiles_encoder,
            use_mllm_visual_semantic=args.use_mllm_visual_semantic,
            llm_profile=args.llm_profile,
            mllm_profile=args.mllm_profile,
        )

    metadata_path = write_dataset_metadata(
        task_dir / "metadata.json",
        spec=spec,
        counts={**split_counts, **{f"invalid_{key}": value for key, value in invalid_counts.items()}},
        files={**split_files, **({"config": _relative_to_root(config_path)} if config_path else {})},
    )
    return {
        "task": spec.name,
        "collection": spec.collection,
        "mode": spec.mode,
        "counts": split_counts,
        "invalid_counts": invalid_counts,
        "files": {**split_files, "metadata": _relative_to_root(metadata_path), **({"config": _relative_to_root(config_path)} if config_path else {})},
    }


def _load_tdc_dataset(name: str, collection: str, cache_dir: Path):
    try:
        from tdc.single_pred import ADME, Tox
    except ImportError as exc:
        raise SystemExit(
            "The TDC Python package is not installed. Install it in a dedicated tdc environment with:\n"
            "  conda create -n tdc python=3.10\n"
            "  conda run -n tdc pip install 'setuptools<81' numpy==1.26.4 pandas requests tqdm fuzzywuzzy scikit-learn rdkit==2023.9.6 PyYAML openpyxl huggingface_hub==0.36.2\n"
            "  conda run -n tdc pip install --no-deps pytdc\n"
            "Then run this script with:\n"
            "  conda run -n tdc python scripts/download_tdc_admet.py --tasks recommended\n"
        ) from exc
    dataset_cls = ADME if collection == "ADME" else Tox
    try:
        return dataset_cls(name=name, path=str(cache_dir))
    except TypeError:
        return dataset_cls(name=name)


def _get_full_dataframe(loader) -> pd.DataFrame:
    if hasattr(loader, "get_data"):
        return loader.get_data()
    split = loader.get_split(method="random", seed=13)
    return pd.concat([split[key] for key in ["train", "valid", "test"] if key in split], ignore_index=True)


def _get_split_dataframe(loader, *, method: str, seed: int, frac: list[float]) -> Mapping[str, pd.DataFrame]:
    try:
        split = loader.get_split(method=method, seed=seed, frac=frac)
    except TypeError:
        split = loader.get_split(method=method, seed=seed)
    return {key: split[key] for key in ["train", "valid", "test"] if key in split}


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


if __name__ == "__main__":
    raise SystemExit(main())
