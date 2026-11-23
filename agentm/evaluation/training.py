from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from agentm.agents.input_agent import InputAgent
from agentm.agents.molecule_processing_agent import MoleculeProcessingAgent
from agentm.agents.rule_discovery_agent import RuleDiscoveryAgent
from agentm.data.splits import cold_drug_split, random_scaffold_split, random_split, scaffold_split
from agentm.features.builder import FeatureBuilder
from agentm.features.persistent_cache import PersistentSingleFeatureCache, rules_signature
from agentm.models.predictors import SklearnPredictor
from agentm.models.tasks import task_mode
from agentm.rules.library import RuleLibrary
from agentm.schema import FeatureSchema, MoleculeRecord, Rule, SingleMoleculeFeatureBundle
from agentm.utils.io import ensure_dir, write_json
from agentm.evaluation.metrics import classification_metrics, regression_metrics
from agentm.evaluation.reproducibility import write_reproducibility_log


def train_tabular(config: Dict[str, Any]) -> Dict[str, Any]:
    task_type = config["task_type"]
    dataset_name = config.get("dataset_name", "dataset")
    data_path = Path(config["data_path"])
    output_dir = ensure_dir(config.get("output_dir", f"outputs/{dataset_name}"))
    seed = int(config.get("random_seed", 13))
    feature_set = config.get("feature_set", "all_features")
    split_type = config.get("split_type", "scaffold")
    model_type = config.get("model_type", "random_forest")

    df = pd.read_csv(data_path)
    if "_agentm_row_id" not in df.columns:
        df.insert(0, "_agentm_row_id", range(len(df)))
    if task_type == "ddi_prediction":
        return _train_ddi(df, config, output_dir, seed, feature_set, split_type, model_type)
    return _train_single(df, config, output_dir, seed, feature_set, split_type, model_type)


def _train_single(
    df: pd.DataFrame,
    config: Dict[str, Any],
    output_dir: Path,
    seed: int,
    feature_set: str,
    split_type: str,
    model_type: str,
) -> Dict[str, Any]:
    task_type = config["task_type"]
    dataset_name = config.get("dataset_name", "dataset")
    smiles_col = config.get("smiles_column", "smiles")
    label_col = config.get("label_column", "label")
    if smiles_col not in df and "SMILES" in df:
        smiles_col = "SMILES"
    if label_col not in df:
        label_candidates = [c for c in df.columns if c not in {smiles_col, "name", "mol_id"}]
        if not label_candidates:
            raise ValueError("Could not infer label column.")
        label_col = label_candidates[0]

    train_df, valid_df, test_df = _split_single_dataframe(df, split_type, smiles_col, seed)
    train_df.to_csv(output_dir / "train_split.csv", index=False)
    valid_df.to_csv(output_dir / "valid_split.csv", index=False)
    test_df.to_csv(output_dir / "test_split.csv", index=False)
    rules, rule_metadata = _discover_training_rules(
        config=config,
        output_dir=output_dir,
        task_type=task_type,
        dataset_name=dataset_name,
        train_df=train_df,
        smiles_col=smiles_col,
        label_col=label_col,
    )
    builder = FeatureBuilder(
        output_dir=output_dir / "features",
        feature_set=feature_set,
        imagemol_checkpoint_path=config.get("imagemol_checkpoint_path"),
        use_smiles_encoder=_as_bool(config.get("use_smiles_encoder"), default=False),
        smiles_encoder_model_name=config.get("smiles_encoder_model_name"),
        smiles_encoder_device=config.get("smiles_encoder_device", "auto"),
        smiles_pooling=config.get("smiles_pooling", "cls"),
        smiles_local_files_only=_as_bool(config.get("smiles_local_files_only"), default=True),
        use_mllm_visual_semantic=_as_bool(config.get("use_mllm_visual_semantic"), default=False),
        mllm_profile=config.get("mllm_profile") or config.get("mlmm_profile"),
        visual_semantic_cache_dir=config.get("visual_semantic_cache_dir"),
        visual_semantic_cache_read=_as_bool(config.get("visual_semantic_cache_read"), default=True),
        visual_semantic_cache_write=_as_bool(config.get("visual_semantic_cache_write"), default=True),
        visual_semantic_cache_only=_as_bool(config.get("visual_semantic_cache_only"), default=False),
    )
    processor = MoleculeProcessingAgent()
    persistent_feature_cache = _build_persistent_feature_cache(config, cache_enabled=True)
    feature_cache: Dict[str, SingleMoleculeFeatureBundle] = {}
    molecule_cache: Dict[str, MoleculeRecord] = {}
    cache_stats: Dict[str, Any] = {
        "enabled": persistent_feature_cache is not None,
        "feature_hits": 0,
        "feature_misses": 0,
        "molecule_hits": 0,
        "molecule_misses": 0,
    }

    x_train, y_train, train_schema, train_rows = _single_xy_with_rows(
        train_df,
        smiles_col,
        label_col,
        processor,
        builder,
        rules,
        task_type,
        dataset_name,
        feature_cache=feature_cache,
        molecule_cache=molecule_cache,
        persistent_feature_cache=persistent_feature_cache,
        cache_stats=cache_stats,
    )
    x_valid, y_valid, _, valid_rows = _single_xy_with_rows(
        valid_df,
        smiles_col,
        label_col,
        processor,
        builder,
        rules,
        task_type,
        dataset_name,
        feature_cache=feature_cache,
        molecule_cache=molecule_cache,
        persistent_feature_cache=persistent_feature_cache,
        cache_stats=cache_stats,
    )
    x_test, y_test, _, test_rows = _single_xy_with_rows(
        test_df,
        smiles_col,
        label_col,
        processor,
        builder,
        rules,
        task_type,
        dataset_name,
        feature_cache=feature_cache,
        molecule_cache=molecule_cache,
        persistent_feature_cache=persistent_feature_cache,
        cache_stats=cache_stats,
    )
    if persistent_feature_cache is not None:
        persistent_feature_cache.write_manifest()
        cache_stats["persistent_feature_cache"] = dict(persistent_feature_cache.stats)
    mode = str(config.get("tdc_task_mode") or task_mode(task_type, dataset_name)).strip().lower()
    if mode not in {"classification", "regression"}:
        raise ValueError(f"Unsupported task mode '{mode}' for dataset '{dataset_name}'.")
    predictor = SklearnPredictor.train(
        x_train,
        y_train,
        mode=mode,
        model_type=model_type,
        random_seed=seed,
        model_params=config.get("model_params"),
    )
    model_path = predictor.save(
        output_dir / "model.pkl",
        feature_schema=train_schema.to_dict() if train_schema else None,
        rules=[rule.to_dict() for rule in rules],
        metadata={
            "task_type": task_type,
            "dataset_name": dataset_name,
            "split_type": split_type,
            "feature_set": feature_set,
            "model_params": config.get("model_params"),
            "imagemol_checkpoint_path": config.get("imagemol_checkpoint_path"),
            "use_smiles_encoder": _as_bool(config.get("use_smiles_encoder"), default=False),
            "smiles_encoder_model_name": config.get("smiles_encoder_model_name"),
            "smiles_encoder_device": config.get("smiles_encoder_device", "auto"),
            "smiles_pooling": config.get("smiles_pooling", "cls"),
            "smiles_local_files_only": _as_bool(config.get("smiles_local_files_only"), default=True),
            "use_mllm_visual_semantic": _as_bool(config.get("use_mllm_visual_semantic"), default=False),
            "mllm_profile": config.get("mllm_profile") or config.get("mlmm_profile"),
            "visual_semantic_cache_dir": config.get("visual_semantic_cache_dir"),
            "visual_semantic_cache_read": _as_bool(config.get("visual_semantic_cache_read"), default=True),
            "visual_semantic_cache_write": _as_bool(config.get("visual_semantic_cache_write"), default=True),
            "visual_semantic_cache_only": _as_bool(config.get("visual_semantic_cache_only"), default=False),
            "positive_label": config.get("positive_label", 1),
            "positive_label_meaning": config.get("positive_label_meaning"),
            "negative_label": config.get("negative_label", 0),
            "negative_label_meaning": config.get("negative_label_meaning"),
            "decision_threshold": float(config.get("decision_threshold", 0.5)),
            "rule_metadata": rule_metadata,
        },
    )
    pred = predictor.model.predict(x_test)
    if mode == "classification":
        scores_matrix = predictor.predict_scores(x_test)
        if scores_matrix is not None and scores_matrix.ndim == 2 and scores_matrix.shape[1] == 2:
            scores = scores_matrix[:, 1]
        else:
            scores = scores_matrix if scores_matrix is not None else pred
        metrics = classification_metrics(y_test, scores, pred)
    else:
        metrics = regression_metrics(y_test, pred)
    prediction_files = _write_prediction_files(
        output_dir=output_dir,
        predictor=predictor,
        mode=mode,
        splits={
            "train": (x_train, y_train, train_rows),
            "valid": (x_valid, y_valid, valid_rows),
            "test": (x_test, y_test, test_rows),
        },
        label_col=label_col,
        smiles_col=smiles_col,
    )

    result = {
        "task_type": task_type,
        "dataset_name": dataset_name,
        "model_path": str(model_path),
        "split_type": split_type,
        "metrics": metrics,
        "n_train": int(len(train_df)),
        "n_valid": int(len(valid_df)),
        "n_test": int(len(test_df)),
        "n_train_valid_resolved": int(len(x_train)),
        "n_valid_valid_resolved": int(len(x_valid)),
        "n_test_valid_resolved": int(len(x_test)),
        "feature_matrix_dtype": str(x_train.dtype),
        "feature_schema": train_schema.to_dict() if train_schema else None,
        "split_files": {
            "train": str(output_dir / "train_split.csv"),
            "valid": str(output_dir / "valid_split.csv"),
            "test": str(output_dir / "test_split.csv"),
        },
        "prediction_files": prediction_files,
        "feature_cache_stats": {
            **cache_stats,
            "unique_features": int(len(feature_cache)),
            "unique_molecules": int(len(molecule_cache)),
        },
        "rule_metadata": rule_metadata,
        "label_metadata": {
            "positive_label": config.get("positive_label", 1),
            "positive_label_meaning": config.get("positive_label_meaning"),
            "negative_label": config.get("negative_label", 0),
            "negative_label_meaning": config.get("negative_label_meaning"),
            "decision_threshold": float(config.get("decision_threshold", 0.5)),
        },
        "rules": [r.to_dict() for r in rules],
    }
    reproducibility_path = write_reproducibility_log(
        output_dir,
        config=config,
        artifacts={
            "data_path": config.get("data_path"),
            "model_path": model_path,
            "imagemol_checkpoint_path": config.get("imagemol_checkpoint_path"),
            "smiles_encoder_model_name": config.get("smiles_encoder_model_name"),
            "visual_semantic_cache_dir": config.get("visual_semantic_cache_dir"),
            "train_split": output_dir / "train_split.csv",
            "valid_split": output_dir / "valid_split.csv",
            "test_split": output_dir / "test_split.csv",
            "predictions": output_dir / "predictions.csv",
        },
        extra={
            "metrics": metrics,
            "split_counts": {"train": len(train_df), "valid": len(valid_df), "test": len(test_df)},
            "feature_dim": len(train_schema.feature_names) if train_schema else None,
            "feature_cache_stats": result["feature_cache_stats"],
        },
    )
    result["reproducibility_log_path"] = str(reproducibility_path)
    write_json(output_dir / "rules_used.json", {"metadata": rule_metadata, "rules": [r.to_dict() for r in rules]})
    write_json(output_dir / "training_result.json", result)
    return result


def _single_xy(df, smiles_col, label_col, processor, builder, rules, task_type, dataset_name=None) -> Tuple[np.ndarray, np.ndarray, Optional[FeatureSchema]]:
    x, y, schema, _ = _single_xy_with_rows(df, smiles_col, label_col, processor, builder, rules, task_type, dataset_name)
    return x, y, schema


def _single_xy_with_rows(
    df,
    smiles_col,
    label_col,
    processor,
    builder,
    rules,
    task_type,
    dataset_name=None,
    *,
    feature_cache: Optional[Dict[str, SingleMoleculeFeatureBundle]] = None,
    molecule_cache: Optional[Dict[str, MoleculeRecord]] = None,
    persistent_feature_cache: Optional[PersistentSingleFeatureCache] = None,
    cache_stats: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[FeatureSchema], List[Dict[str, Any]]]:
    x: List[List[float]] = []
    y: List[Any] = []
    rows: List[Dict[str, Any]] = []
    schema: Optional[FeatureSchema] = None
    for row_position, (_, row) in enumerate(df.iterrows(), start=1):
        raw_smiles = str(row[smiles_col])
        if raw_smiles in (molecule_cache or {}):
            record = molecule_cache[raw_smiles]
            if cache_stats is not None:
                cache_stats["molecule_hits"] = int(cache_stats.get("molecule_hits", 0)) + 1
        else:
            record = processor.process_smiles(raw_smiles)
            if molecule_cache is not None:
                molecule_cache[raw_smiles] = record
            if cache_stats is not None:
                cache_stats["molecule_misses"] = int(cache_stats.get("molecule_misses", 0)) + 1
        if not record.is_valid:
            continue
        feature_key = record.canonical_smiles or record.smiles or raw_smiles
        bundle = feature_cache.get(feature_key) if feature_cache is not None else None
        if bundle is not None:
            if cache_stats is not None:
                cache_stats["feature_hits"] = int(cache_stats.get("feature_hits", 0)) + 1
        else:
            if persistent_feature_cache is not None:
                bundle = persistent_feature_cache.load(
                    record=record,
                    builder=builder,
                    rules=rules,
                    task_type=task_type,
                    dataset_name=dataset_name or "dataset",
                )
            if bundle is None:
                bundle = builder.build_single(record, rules=rules, task_type=task_type, dataset_name=dataset_name)
                if persistent_feature_cache is not None:
                    persistent_feature_cache.store(
                        record=record,
                        builder=builder,
                        rules=rules,
                        task_type=task_type,
                        dataset_name=dataset_name or "dataset",
                        bundle=bundle,
                    )
                if cache_stats is not None:
                    cache_stats["feature_misses"] = int(cache_stats.get("feature_misses", 0)) + 1
            elif cache_stats is not None:
                cache_stats["feature_hits"] = int(cache_stats.get("feature_hits", 0)) + 1
            if feature_cache is not None:
                feature_cache[feature_key] = bundle
        schema = schema or bundle.feature_schema
        x.append(bundle.feature_vector)
        y.append(row[label_col])
        rows.append(_prediction_row_metadata(row, smiles_col=smiles_col, label_col=label_col))
    return np.asarray(x, dtype=float), np.asarray(y), schema, rows


def _train_ddi(
    df: pd.DataFrame,
    config: Dict[str, Any],
    output_dir: Path,
    seed: int,
    feature_set: str,
    split_type: str,
    model_type: str,
) -> Dict[str, Any]:
    label_col = config.get("label_column", "label")
    drug_a_smiles_col, drug_b_smiles_col, drug_a_name_col, drug_b_name_col = _ddi_column_config(config)
    _validate_ddi_columns(df, label_col, drug_a_smiles_col, drug_b_smiles_col, drug_a_name_col, drug_b_name_col)
    df_before_sampling = df
    df, sample_metadata = _sample_training_dataframe(df, config=config, label_col=label_col, seed=seed)
    sample_metadata.update(
        {
            "unique_drugs_before_sampling": _ddi_unique_drug_count(
                df_before_sampling,
                drug_a_smiles_col=drug_a_smiles_col,
                drug_b_smiles_col=drug_b_smiles_col,
                drug_a_name_col=drug_a_name_col,
                drug_b_name_col=drug_b_name_col,
            ),
            "unique_drugs_after_sampling": _ddi_unique_drug_count(
                df,
                drug_a_smiles_col=drug_a_smiles_col,
                drug_b_smiles_col=drug_b_smiles_col,
                drug_a_name_col=drug_a_name_col,
                drug_b_name_col=drug_b_name_col,
            ),
            "label_counts_before_sampling": _label_counts(df_before_sampling, label_col),
            "label_counts_after_sampling": _label_counts(df, label_col),
        }
    )
    sample_metadata["unique_drugs"] = sample_metadata["unique_drugs_after_sampling"]
    if sample_metadata["enabled"]:
        df.to_csv(output_dir / "sampled_dataset.csv", index=False)
    train_df, valid_df, test_df = _split_ddi_dataframe(df, split_type, seed, config)
    print(
        f"[ddi-train] split={split_type} feature_set={feature_set} "
        f"rows train={len(train_df)} valid={len(valid_df)} test={len(test_df)}",
        flush=True,
    )
    train_df.to_csv(output_dir / "train_split.csv", index=False)
    valid_df.to_csv(output_dir / "valid_split.csv", index=False)
    test_df.to_csv(output_dir / "test_split.csv", index=False)
    rules, rule_metadata = _discover_ddi_rules(config, output_dir)
    builder = FeatureBuilder(
        output_dir=output_dir / "features",
        feature_set=feature_set,
        imagemol_checkpoint_path=config.get("imagemol_checkpoint_path"),
        use_smiles_encoder=_as_bool(config.get("use_smiles_encoder"), default=False),
        smiles_encoder_model_name=config.get("smiles_encoder_model_name"),
        smiles_encoder_device=config.get("smiles_encoder_device", "auto"),
        smiles_pooling=config.get("smiles_pooling", "cls"),
        smiles_local_files_only=_as_bool(config.get("smiles_local_files_only"), default=True),
        use_mllm_visual_semantic=_as_bool(config.get("use_mllm_visual_semantic"), default=False),
        mllm_profile=config.get("mllm_profile") or config.get("mlmm_profile"),
        visual_semantic_cache_dir=config.get("visual_semantic_cache_dir"),
        visual_semantic_cache_read=_as_bool(config.get("visual_semantic_cache_read"), default=True),
        visual_semantic_cache_write=_as_bool(config.get("visual_semantic_cache_write"), default=True),
        visual_semantic_cache_only=_as_bool(config.get("visual_semantic_cache_only"), default=False),
    )
    input_agent = InputAgent()
    processor = MoleculeProcessingAgent()
    cache_enabled = _as_bool(config.get("cache_unique_drug_features"), default=True)
    feature_cache: Optional[Dict[str, SingleMoleculeFeatureBundle]] = {} if cache_enabled else None
    molecule_cache: Optional[Dict[str, MoleculeRecord]] = {} if cache_enabled else None
    persistent_feature_cache = _build_persistent_feature_cache(config, cache_enabled=cache_enabled)
    cache_stats: Dict[str, Any] = {
        "enabled": cache_enabled,
        "feature_hits": 0,
        "feature_misses": 0,
        "molecule_hits": 0,
        "molecule_misses": 0,
    }

    dataset_name = config.get("dataset_name", "DDI")
    print(f"[ddi-train] building train matrix rows={len(train_df)}", flush=True)
    x_train, y_train, train_schema, train_stats, train_rows = _ddi_xy_with_rows(
        train_df,
        label_col,
        input_agent,
        processor,
        builder,
        rules,
        dataset_name,
        drug_a_smiles_col=drug_a_smiles_col,
        drug_b_smiles_col=drug_b_smiles_col,
        drug_a_name_col=drug_a_name_col,
        drug_b_name_col=drug_b_name_col,
        feature_cache=feature_cache,
        molecule_cache=molecule_cache,
        persistent_feature_cache=persistent_feature_cache,
        cache_stats=cache_stats,
    )
    print(f"[ddi-train] train matrix shape={x_train.shape} resolved={train_stats['resolved_rows']}", flush=True)
    print(f"[ddi-train] building valid matrix rows={len(valid_df)}", flush=True)
    x_valid, y_valid, _, valid_stats, valid_rows = _ddi_xy_with_rows(
        valid_df,
        label_col,
        input_agent,
        processor,
        builder,
        rules,
        dataset_name,
        drug_a_smiles_col=drug_a_smiles_col,
        drug_b_smiles_col=drug_b_smiles_col,
        drug_a_name_col=drug_a_name_col,
        drug_b_name_col=drug_b_name_col,
        feature_cache=feature_cache,
        molecule_cache=molecule_cache,
        persistent_feature_cache=persistent_feature_cache,
        cache_stats=cache_stats,
    )
    print(f"[ddi-train] valid matrix shape={x_valid.shape} resolved={valid_stats['resolved_rows']}", flush=True)
    print(f"[ddi-train] building test matrix rows={len(test_df)}", flush=True)
    x_test, y_test, _, test_stats, test_rows = _ddi_xy_with_rows(
        test_df,
        label_col,
        input_agent,
        processor,
        builder,
        rules,
        dataset_name,
        drug_a_smiles_col=drug_a_smiles_col,
        drug_b_smiles_col=drug_b_smiles_col,
        drug_a_name_col=drug_a_name_col,
        drug_b_name_col=drug_b_name_col,
        feature_cache=feature_cache,
        molecule_cache=molecule_cache,
        persistent_feature_cache=persistent_feature_cache,
        cache_stats=cache_stats,
    )
    print(f"[ddi-train] test matrix shape={x_test.shape} resolved={test_stats['resolved_rows']}", flush=True)
    if len(x_train) == 0:
        raise ValueError(f"No valid DDI training rows after molecule resolution. Skipped rows: {train_stats}")
    if len(x_test) == 0:
        raise ValueError(f"No valid DDI test rows after molecule resolution. Skipped rows: {test_stats}")
    if persistent_feature_cache is not None:
        persistent_feature_cache.write_manifest()
        cache_stats["persistent_feature_cache"] = dict(persistent_feature_cache.stats)
    print(f"[ddi-train] fitting model_type={model_type} model_params={config.get('model_params')}", flush=True)
    predictor = SklearnPredictor.train(
        x_train,
        y_train,
        mode="classification",
        model_type=model_type,
        random_seed=seed,
        model_params=config.get("model_params"),
    )
    print("[ddi-train] model fit complete", flush=True)
    model_path = predictor.save(
        output_dir / "model.pkl",
        feature_schema=train_schema.to_dict() if train_schema else None,
        rules=[rule.to_dict() for rule in rules],
        metadata={
            "task_type": "ddi_prediction",
            "dataset_name": config.get("dataset_name", "DDI"),
            "split_type": split_type,
            "feature_set": feature_set,
            "model_params": config.get("model_params"),
            "imagemol_checkpoint_path": config.get("imagemol_checkpoint_path"),
            "use_smiles_encoder": _as_bool(config.get("use_smiles_encoder"), default=False),
            "smiles_encoder_model_name": config.get("smiles_encoder_model_name"),
            "smiles_encoder_device": config.get("smiles_encoder_device", "auto"),
            "smiles_pooling": config.get("smiles_pooling", "cls"),
            "smiles_local_files_only": _as_bool(config.get("smiles_local_files_only"), default=True),
            "use_mllm_visual_semantic": _as_bool(config.get("use_mllm_visual_semantic"), default=False),
            "mllm_profile": config.get("mllm_profile") or config.get("mlmm_profile"),
            "visual_semantic_cache_dir": config.get("visual_semantic_cache_dir"),
            "visual_semantic_cache_read": _as_bool(config.get("visual_semantic_cache_read"), default=True),
            "visual_semantic_cache_write": _as_bool(config.get("visual_semantic_cache_write"), default=True),
            "visual_semantic_cache_only": _as_bool(config.get("visual_semantic_cache_only"), default=False),
            "positive_label": config.get("positive_label", 1),
            "positive_label_meaning": config.get("positive_label_meaning", "DDI risk/interaction"),
            "negative_label": config.get("negative_label", 0),
            "negative_label_meaning": config.get("negative_label_meaning", "no DDI risk/interaction"),
            "decision_threshold": float(config.get("decision_threshold", 0.5)),
            "rule_metadata": rule_metadata,
            "drug_a_smiles_column": drug_a_smiles_col,
            "drug_b_smiles_column": drug_b_smiles_col,
            "drug_a_name_column": drug_a_name_col,
            "drug_b_name_column": drug_b_name_col,
            "train_resolution_stats": train_stats,
            "valid_resolution_stats": valid_stats,
            "test_resolution_stats": test_stats,
            "sample_metadata": sample_metadata,
            "feature_cache_stats": {
                **cache_stats,
                "unique_features": len(feature_cache or {}),
                "unique_molecules": len(molecule_cache or {}),
            },
        },
    )
    pred = predictor.model.predict(x_test)
    scores = predictor.predict_scores(x_test)
    if scores is None:
        scores = pred
    metrics = classification_metrics(y_test, scores, pred, classes=getattr(predictor.model, "classes_", None))
    prediction_files = _write_prediction_files(
        output_dir=output_dir,
        predictor=predictor,
        mode="classification",
        splits={
            "train": (x_train, y_train, train_rows),
            "valid": (x_valid, y_valid, valid_rows),
            "test": (x_test, y_test, test_rows),
        },
        label_col=label_col,
        smiles_col=drug_a_smiles_col,
    )
    result = {
        "task_type": "ddi_prediction",
        "dataset_name": config.get("dataset_name", "DDI"),
        "model_path": str(model_path),
        "split_type": split_type,
        "metrics": metrics,
        "n_train": int(len(train_df)),
        "n_valid": int(len(valid_df)),
        "n_test": int(len(test_df)),
        "n_train_valid_resolved": int(len(x_train)),
        "n_valid_valid_resolved": int(len(x_valid)),
        "n_test_valid_resolved": int(len(x_test)),
        "feature_schema": train_schema.to_dict() if train_schema else None,
        "rule_metadata": rule_metadata,
        "sample_metadata": sample_metadata,
        "resolution_stats": {
            "train": train_stats,
            "valid": valid_stats,
            "test": test_stats,
        },
        "feature_cache_stats": {
            **cache_stats,
            "unique_features": int(len(feature_cache or {})),
            "unique_molecules": int(len(molecule_cache or {})),
        },
        "split_files": {
            "train": str(output_dir / "train_split.csv"),
            "valid": str(output_dir / "valid_split.csv"),
            "test": str(output_dir / "test_split.csv"),
        },
        "prediction_files": prediction_files,
        "rules": [r.to_dict() for r in rules],
    }
    reproducibility_path = write_reproducibility_log(
        output_dir,
        config=config,
        artifacts={
            "data_path": config.get("data_path"),
            "sampled_dataset": output_dir / "sampled_dataset.csv" if sample_metadata["enabled"] else None,
            "model_path": model_path,
            "imagemol_checkpoint_path": config.get("imagemol_checkpoint_path"),
            "smiles_encoder_model_name": config.get("smiles_encoder_model_name"),
            "visual_semantic_cache_dir": config.get("visual_semantic_cache_dir"),
            "train_split": output_dir / "train_split.csv",
            "valid_split": output_dir / "valid_split.csv",
            "test_split": output_dir / "test_split.csv",
        },
        extra={
            "metrics": metrics,
            "split_counts": {"train": len(train_df), "valid": len(valid_df), "test": len(test_df)},
            "feature_dim": len(train_schema.feature_names) if train_schema else None,
            "feature_matrix_dtype": str(x_train.dtype),
            "resolution_stats": result["resolution_stats"],
            "sample_metadata": sample_metadata,
            "feature_cache_stats": result["feature_cache_stats"],
            "token_api_usage_note": "DDI supervised training does not call LLM/MLLM per sample; LLM rules are task-level only when enabled.",
        },
    )
    result["reproducibility_log_path"] = str(reproducibility_path)
    write_json(output_dir / "rules_used.json", {"metadata": rule_metadata, "rules": [r.to_dict() for r in rules]})
    write_json(output_dir / "training_result.json", result)
    return result


def _split_single_dataframe(
    df: pd.DataFrame,
    split_type: str,
    smiles_col: str,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if split_type in {"scaffold", "deterministic_scaffold"}:
        return scaffold_split(df, smiles_column=smiles_col, seed=seed)
    if split_type in {"random_scaffold", "random_scaffold_split"}:
        return random_scaffold_split(df, smiles_column=smiles_col, seed=seed)
    if split_type == "random":
        return random_split(df, seed=seed)
    raise ValueError(f"Unsupported single-molecule split_type: {split_type}")


def _split_ddi_dataframe(
    df: pd.DataFrame,
    split_type: str,
    seed: int,
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    config = config or {}
    if split_type == "cold_drug":
        drug_a_col, drug_b_col = _ddi_split_columns(df, config)
        return cold_drug_split(df, drug_a_col=drug_a_col, drug_b_col=drug_b_col, seed=seed)
    if split_type == "random":
        return random_split(df, seed=seed)
    raise ValueError(f"Unsupported DDI split_type: {split_type}")


def _build_persistent_feature_cache(
    config: Dict[str, Any],
    *,
    cache_enabled: bool,
) -> Optional[PersistentSingleFeatureCache]:
    cache_dir = (
        config.get("single_drug_feature_cache_dir")
        or config.get("persistent_single_feature_cache_dir")
        or config.get("persistent_feature_cache_dir")
    )
    if not cache_enabled or not cache_dir:
        return None
    return PersistentSingleFeatureCache(
        cache_dir,
        read=_as_bool(config.get("single_drug_feature_cache_read"), default=True),
        write=_as_bool(config.get("single_drug_feature_cache_write"), default=True),
    )


def _sample_training_dataframe(
    df: pd.DataFrame,
    *,
    config: Dict[str, Any],
    label_col: str,
    seed: int,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    fraction = config.get("sample_fraction")
    n_rows = config.get("sample_n")
    if fraction is None and n_rows is None:
        return df, {
            "enabled": False,
            "strategy": "none",
            "sample_fraction": None,
            "sample_n": None,
            "rows_before": int(len(df)),
            "rows_after": int(len(df)),
            "label_column": label_col,
            "label_counts": _label_counts(df, label_col),
        }

    strategy = str(config.get("sample_strategy", "stratified_by_label")).strip().lower()
    rows_before = int(len(df))
    if n_rows is not None:
        n_target = min(int(n_rows), rows_before)
        fraction_value = n_target / max(rows_before, 1)
    else:
        fraction_value = float(fraction)
        if not 0 < fraction_value <= 1:
            raise ValueError("sample_fraction must be in (0, 1].")
        n_target = max(1, int(round(rows_before * fraction_value)))

    if strategy == "stratified_by_label" and label_col in df:
        parts = []
        for _, group in df.groupby(label_col, sort=False):
            if len(group) == 0:
                continue
            parts.append(group.sample(n=max(1, int(round(len(group) * fraction_value))), random_state=seed))
        sampled = pd.concat(parts, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True) if parts else df.iloc[[]].copy()
        if len(sampled) > n_target:
            sampled = sampled.sample(n=n_target, random_state=seed).reset_index(drop=True)
    else:
        sampled = df.sample(n=n_target, random_state=seed).reset_index(drop=True)

    metadata = {
        "enabled": True,
        "strategy": strategy,
        "sample_fraction": fraction_value,
        "sample_n": int(n_target),
        "rows_before": rows_before,
        "rows_after": int(len(sampled)),
        "label_column": label_col,
        "label_counts": {str(k): int(v) for k, v in sampled[label_col].value_counts().sort_index().to_dict().items()} if label_col in sampled else {},
    }
    return sampled, metadata


def _label_counts(df: pd.DataFrame, label_col: str) -> Dict[str, int]:
    if label_col not in df:
        return {}
    return {str(k): int(v) for k, v in df[label_col].value_counts().sort_index().to_dict().items()}


def _ddi_unique_drug_count(
    df: pd.DataFrame,
    *,
    drug_a_smiles_col: str,
    drug_b_smiles_col: str,
    drug_a_name_col: str,
    drug_b_name_col: str,
) -> int:
    a_values = _ddi_drug_key_series(df, drug_a_smiles_col, drug_a_name_col)
    b_values = _ddi_drug_key_series(df, drug_b_smiles_col, drug_b_name_col)
    values = pd.concat([a_values, b_values], ignore_index=True)
    values = values.dropna().astype(str).str.strip()
    values = values[values != ""]
    return int(values.nunique())


def _ddi_drug_key_series(df: pd.DataFrame, primary_col: str, fallback_col: str) -> pd.Series:
    if primary_col in df:
        values = df[primary_col].astype("string").str.strip()
        if fallback_col in df:
            fallback = df[fallback_col].astype("string").str.strip()
            values = values.mask(values.isna() | (values == ""), fallback)
        return values
    if fallback_col in df:
        return df[fallback_col].astype("string").str.strip()
    return pd.Series(dtype="string")


def _ddi_xy(
    df,
    label_col,
    input_agent,
    processor,
    builder,
    rules,
    dataset_name="DDI",
    *,
    drug_a_smiles_col="drug_a_smiles",
    drug_b_smiles_col="drug_b_smiles",
    drug_a_name_col="drug_a_name",
    drug_b_name_col="drug_b_name",
    feature_cache: Optional[Dict[str, SingleMoleculeFeatureBundle]] = None,
    molecule_cache: Optional[Dict[str, MoleculeRecord]] = None,
    persistent_feature_cache: Optional[PersistentSingleFeatureCache] = None,
    cache_stats: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[FeatureSchema], Dict[str, Any]]:
    x, y, schema, stats, _ = _ddi_xy_with_rows(
        df,
        label_col,
        input_agent,
        processor,
        builder,
        rules,
        dataset_name,
        drug_a_smiles_col=drug_a_smiles_col,
        drug_b_smiles_col=drug_b_smiles_col,
        drug_a_name_col=drug_a_name_col,
        drug_b_name_col=drug_b_name_col,
        feature_cache=feature_cache,
        molecule_cache=molecule_cache,
        persistent_feature_cache=persistent_feature_cache,
        cache_stats=cache_stats,
    )
    return x, y, schema, stats


def _ddi_xy_with_rows(
    df,
    label_col,
    input_agent,
    processor,
    builder,
    rules,
    dataset_name="DDI",
    *,
    drug_a_smiles_col="drug_a_smiles",
    drug_b_smiles_col="drug_b_smiles",
    drug_a_name_col="drug_a_name",
    drug_b_name_col="drug_b_name",
    feature_cache: Optional[Dict[str, SingleMoleculeFeatureBundle]] = None,
    molecule_cache: Optional[Dict[str, MoleculeRecord]] = None,
    persistent_feature_cache: Optional[PersistentSingleFeatureCache] = None,
    cache_stats: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[FeatureSchema], Dict[str, Any], List[Dict[str, Any]]]:
    x_matrix: Optional[np.ndarray] = None
    write_idx = 0
    y: List[Any] = []
    rows: List[Dict[str, Any]] = []
    schema: Optional[FeatureSchema] = None
    stats = {
        "total_rows": int(len(df)),
        "resolved_rows": 0,
        "skipped_rows": 0,
        "skip_reasons": Counter(),
    }
    for row_position, (_, row) in enumerate(df.iterrows(), start=1):
        a_raw, a_source_col = _first_nonempty_with_column(row, [drug_a_smiles_col, drug_a_name_col])
        b_raw, b_source_col = _first_nonempty_with_column(row, [drug_b_smiles_col, drug_b_name_col])
        if a_raw is None or b_raw is None:
            stats["skipped_rows"] += 1
            stats["skip_reasons"]["missing_drug_input"] += 1
            continue
        a_is_smiles = a_source_col == drug_a_smiles_col
        b_is_smiles = b_source_col == drug_b_smiles_col
        feat_a = _cached_single_ddi_feature(
            a_raw,
            input_agent=input_agent,
            processor=processor,
            builder=builder,
            rules=rules,
            dataset_name=dataset_name,
            raw_is_smiles=a_is_smiles,
            feature_cache=feature_cache,
            molecule_cache=molecule_cache,
            persistent_feature_cache=persistent_feature_cache,
            cache_stats=cache_stats,
        )
        feat_b = _cached_single_ddi_feature(
            b_raw,
            input_agent=input_agent,
            processor=processor,
            builder=builder,
            rules=rules,
            dataset_name=dataset_name,
            raw_is_smiles=b_is_smiles,
            feature_cache=feature_cache,
            molecule_cache=molecule_cache,
            persistent_feature_cache=persistent_feature_cache,
            cache_stats=cache_stats,
        )
        if feat_a is None or feat_b is None:
            stats["skipped_rows"] += 1
            stats["skip_reasons"]["unresolved_molecule"] += 1
            continue
        pair_vector, pair_schema = builder.build_ddi_pair_vector(feat_a, feat_b)
        schema = schema or pair_schema
        pair_vector = np.asarray(pair_vector, dtype=np.float32)
        if x_matrix is None:
            x_matrix = np.empty((len(df), len(pair_vector)), dtype=np.float32)
        elif x_matrix.shape[1] != len(pair_vector):
            raise ValueError(f"DDI pair feature dimension changed from {x_matrix.shape[1]} to {len(pair_vector)}.")
        x_matrix[write_idx] = pair_vector
        write_idx += 1
        y.append(row[label_col])
        rows.append(
            {
                "row_id": row.get("_agentm_row_id"),
                "drug_a_smiles": row.get(drug_a_smiles_col),
                "drug_b_smiles": row.get(drug_b_smiles_col),
                "drug_a_name": row.get(drug_a_name_col),
                "drug_b_name": row.get(drug_b_name_col),
                "label_true": row.get(label_col),
            }
        )
        stats["resolved_rows"] += 1
        if row_position % 20000 == 0:
            print(
                f"[ddi-train] feature rows processed={row_position}/{len(df)} "
                f"resolved={stats['resolved_rows']} skipped={stats['skipped_rows']}",
                flush=True,
            )
    stats["skip_reasons"] = dict(stats["skip_reasons"])
    if x_matrix is None:
        x_out = np.empty((0, 0), dtype=np.float32)
    else:
        x_out = x_matrix[:write_idx]
    return x_out, np.asarray(y), schema, stats, rows


def _cached_single_ddi_feature(
    raw_input: Any,
    *,
    input_agent: InputAgent,
    processor: MoleculeProcessingAgent,
    builder: FeatureBuilder,
    rules: List[Rule],
    dataset_name: str,
    raw_is_smiles: bool = False,
    feature_cache: Optional[Dict[str, SingleMoleculeFeatureBundle]] = None,
    molecule_cache: Optional[Dict[str, MoleculeRecord]] = None,
    persistent_feature_cache: Optional[PersistentSingleFeatureCache] = None,
    cache_stats: Optional[Dict[str, Any]] = None,
) -> Optional[SingleMoleculeFeatureBundle]:
    raw_key = str(raw_input).strip()
    molecule_key = f"smiles:{raw_key}" if raw_is_smiles else f"input:{raw_key}"
    record: MoleculeRecord
    if molecule_cache is not None and molecule_key in molecule_cache:
        record = molecule_cache[molecule_key]
        if cache_stats is not None:
            cache_stats["molecule_hits"] = int(cache_stats.get("molecule_hits", 0)) + 1
    else:
        record = processor.process_smiles(raw_key) if raw_is_smiles else processor.process_input(raw_key, input_agent=input_agent)
        if molecule_cache is not None:
            molecule_cache[molecule_key] = record
        if cache_stats is not None:
            cache_stats["molecule_misses"] = int(cache_stats.get("molecule_misses", 0)) + 1

    if not record.is_valid:
        return None
    feature_key = record.canonical_smiles or record.smiles or raw_key
    if feature_cache is not None and feature_key in feature_cache:
        if cache_stats is not None:
            cache_stats["feature_hits"] = int(cache_stats.get("feature_hits", 0)) + 1
        return feature_cache[feature_key]

    if persistent_feature_cache is not None:
        cached_bundle = persistent_feature_cache.load(
            record=record,
            builder=builder,
            rules=rules,
            task_type="ddi_prediction",
            dataset_name=dataset_name,
        )
        if cached_bundle is not None:
            if feature_cache is not None:
                feature_cache[feature_key] = cached_bundle
            return cached_bundle

    bundle = builder.build_single(record, rules=rules, task_type="ddi_prediction", dataset_name=dataset_name)
    if feature_cache is not None:
        feature_cache[feature_key] = bundle
    if persistent_feature_cache is not None:
        persistent_feature_cache.store(
            record=record,
            builder=builder,
            rules=rules,
            task_type="ddi_prediction",
            dataset_name=dataset_name,
            bundle=bundle,
        )
    if cache_stats is not None:
        cache_stats["feature_misses"] = int(cache_stats.get("feature_misses", 0)) + 1
    return bundle


def _prediction_row_metadata(row: pd.Series, *, smiles_col: str, label_col: str) -> Dict[str, Any]:
    data = {
        "row_id": row.get("_agentm_row_id"),
        "smiles": row.get(smiles_col),
        "label_true": row.get(label_col),
    }
    for key in ["name", "mol_id", "drug_name"]:
        if key in row:
            data[key] = row.get(key)
    return data


def _write_prediction_files(
    *,
    output_dir: Path,
    predictor: SklearnPredictor,
    mode: str,
    splits: Dict[str, Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]],
    label_col: str,
    smiles_col: str,
) -> Dict[str, str]:
    frames: List[pd.DataFrame] = []
    files: Dict[str, str] = {}
    for split, (x, y, rows) in splits.items():
        frame = _prediction_frame(predictor, mode, x, y, rows, split=split)
        path = output_dir / f"{split}_predictions.csv"
        frame.to_csv(path, index=False)
        files[split] = str(path)
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    combined_path = output_dir / "predictions.csv"
    combined.to_csv(combined_path, index=False)
    files["combined"] = str(combined_path)
    metadata = {
        "label_column": label_col,
        "smiles_column": smiles_col,
        "mode": mode,
        "files": files,
    }
    write_json(output_dir / "prediction_files.json", metadata)
    return files


def _prediction_frame(
    predictor: SklearnPredictor,
    mode: str,
    x: np.ndarray,
    y: np.ndarray,
    rows: List[Dict[str, Any]],
    *,
    split: str,
) -> pd.DataFrame:
    base = pd.DataFrame(rows)
    if base.empty:
        base = pd.DataFrame(columns=["row_id", "label_true"])
    base.insert(0, "split", split)
    if len(x) == 0:
        for column in ["score", "label_pred", "prediction"]:
            if column not in base:
                base[column] = []
        return base
    pred = predictor.model.predict(x)
    if mode == "classification":
        proba = predictor.predict_scores(x)
        if proba is not None:
            proba = np.asarray(proba, dtype=float)
            if proba.ndim == 2 and proba.shape[1] > 2:
                classes = list(getattr(predictor.model, "classes_", range(proba.shape[1])))
                max_idx = np.argmax(proba, axis=1)
                base["score"] = proba[np.arange(len(proba)), max_idx]
                base["probability"] = base["score"]
                base["label_pred"] = np.asarray(pred)
                base["top1_label"] = [classes[int(idx)] for idx in max_idx]
                top_k = min(3, proba.shape[1])
                top_indices = np.argsort(-proba, axis=1)[:, :top_k]
                base["top3_labels"] = [";".join(str(classes[int(idx)]) for idx in row) for row in top_indices]
                base["top3_probabilities"] = [";".join(f"{float(proba[i, int(idx)]):.6g}" for idx in row) for i, row in enumerate(top_indices)]
                base["label_true"] = np.asarray(y)
                return base
            scores = proba[:, 1] if proba.ndim == 2 and proba.shape[1] > 1 else proba.reshape(-1)
        else:
            scores = pred
        base["score"] = np.asarray(scores, dtype=float)
        base["probability"] = np.asarray(scores, dtype=float)
        base["label_pred"] = np.asarray(pred)
    else:
        base["prediction"] = np.asarray(pred, dtype=float)
        base["score"] = np.asarray(pred, dtype=float)
    base["label_true"] = np.asarray(y)
    return base


def _ddi_column_config(config: Dict[str, Any]) -> Tuple[str, str, str, str]:
    return (
        config.get("drug_a_smiles_column", "drug_a_smiles"),
        config.get("drug_b_smiles_column", "drug_b_smiles"),
        config.get("drug_a_name_column", "drug_a_name"),
        config.get("drug_b_name_column", "drug_b_name"),
    )


def _ddi_split_columns(df: pd.DataFrame, config: Dict[str, Any]) -> Tuple[str, str]:
    drug_a_smiles_col, drug_b_smiles_col, drug_a_name_col, drug_b_name_col = _ddi_column_config(config)
    drug_a_col = drug_a_smiles_col if drug_a_smiles_col in df else drug_a_name_col
    drug_b_col = drug_b_smiles_col if drug_b_smiles_col in df else drug_b_name_col
    return drug_a_col, drug_b_col


def _validate_ddi_columns(
    df: pd.DataFrame,
    label_col: str,
    drug_a_smiles_col: str,
    drug_b_smiles_col: str,
    drug_a_name_col: str,
    drug_b_name_col: str,
) -> None:
    if label_col not in df:
        raise ValueError(f"Missing DDI label column: {label_col}")
    has_a = drug_a_smiles_col in df or drug_a_name_col in df
    has_b = drug_b_smiles_col in df or drug_b_name_col in df
    if not has_a or not has_b:
        raise ValueError(
            "DDI data must include drug pair columns. "
            f"Expected {drug_a_smiles_col}/{drug_a_name_col} and {drug_b_smiles_col}/{drug_b_name_col}."
        )


def _first_nonempty(row: pd.Series, columns: List[str]) -> Optional[Any]:
    for column in columns:
        if column not in row:
            continue
        value = row.get(column)
        if pd.isna(value):
            continue
        text = str(value).strip()
        if text:
            return value
    return None


def _first_nonempty_with_column(row: pd.Series, columns: List[str]) -> Tuple[Optional[Any], Optional[str]]:
    for column in columns:
        if column not in row:
            continue
        value = row.get(column)
        if pd.isna(value):
            continue
        text = str(value).strip()
        if text:
            return value, column
    return None, None


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _safe_dataset_name(dataset_name: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in dataset_name)


def _training_samples(train_df: pd.DataFrame, smiles_col: str, label_col: str) -> pd.DataFrame:
    samples = train_df.copy()
    rename: Dict[str, str] = {}
    if smiles_col != "smiles":
        rename[smiles_col] = "smiles"
    if label_col != "label":
        rename[label_col] = "label"
    return samples.rename(columns=rename)


def _fallback_training_rules(task_type: str, dataset_name: str, train_samples: pd.DataFrame) -> List[Rule]:
    rule_library = RuleLibrary()
    return rule_library.merge_rules(
        rule_library.literature_rules(dataset_name, task_type),
        rule_library.data_inferred_rules(train_samples, task_type=task_type),
    )


def _discover_training_rules(
    *,
    config: Dict[str, Any],
    output_dir: Path,
    task_type: str,
    dataset_name: str,
    train_df: pd.DataFrame,
    smiles_col: str,
    label_col: str,
) -> Tuple[List[Rule], Dict[str, Any]]:
    train_samples = _training_samples(train_df, smiles_col, label_col)
    use_llm_rules = _as_bool(config.get("use_llm_rules"), default=False)
    precomputed = _load_single_rule_cache(config, task_type=task_type, dataset_name=dataset_name)
    if precomputed is not None:
        rules, metadata = precomputed
        metadata.update(
            {
                "mode": "precomputed",
                "use_llm_rules": use_llm_rules,
                "source_counts": dict(Counter(rule.source for rule in rules)),
            }
        )
        return rules, metadata

    if use_llm_rules:
        rule_agent = RuleDiscoveryAgent(
            output_dir=output_dir / "rules",
            llm_profile=config.get("llm_profile"),
            use_llm=True,
        )
        rules = rule_agent.discover(task_type, dataset_name, training_samples=train_samples)
        summary_path = output_dir / "rules" / f"{task_type}_{_safe_dataset_name(dataset_name)}_rules.json"
        metadata = {
            "mode": "llm4sd",
            "use_llm_rules": True,
            "rule_output_dir": str(output_dir / "rules"),
            "summary_path": str(summary_path),
            "llm_profile": rule_agent.llm_client.config.profile,
            "llm_provider": rule_agent.llm_client.config.provider,
            "llm_model": rule_agent.llm_client.config.model,
            "llm_available": rule_agent.llm_client.available,
            "llm_used": False,
            "llm_errors": [],
        }
        if summary_path.exists():
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                metadata["llm_used"] = bool(summary.get("llm_used"))
                metadata["llm_errors"] = list(summary.get("llm_errors") or [])
            except Exception as exc:
                metadata["summary_read_error"] = str(exc)
        metadata["source_counts"] = dict(Counter(rule.source for rule in rules))
        return rules, metadata

    rules = _fallback_training_rules(task_type, dataset_name, train_samples)
    metadata = {
        "mode": "fallback",
        "use_llm_rules": False,
        "source_counts": dict(Counter(rule.source for rule in rules)),
    }
    return rules, metadata


def _load_single_rule_cache(
    config: Dict[str, Any],
    *,
    task_type: str,
    dataset_name: str,
) -> Optional[Tuple[List[Rule], Dict[str, Any]]]:
    path_value = config.get("precomputed_rules_path") or config.get("single_rule_cache_path")
    if not path_value:
        return None
    path = Path(path_value)
    if not path.is_absolute():
        path = Path.cwd() / path
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        rule_payload = payload.get("rules") if isinstance(payload, dict) else payload
        if not isinstance(rule_payload, list):
            return None
        rules = [_rule_from_dict(item) for item in rule_payload]
        rules = [rule for rule in rules if rule.task_type == task_type or not rule.task_type]
        metadata = dict(payload.get("metadata") or {}) if isinstance(payload, dict) else {}
        metadata.update(
            {
                "rule_cache_hit": True,
                "rule_cache_path": str(path),
                "rule_cache_dataset_name": dataset_name,
                "rules_signature": rules_signature(rules),
            }
        )
        return rules, metadata
    except Exception as exc:
        failure_path = ensure_dir(Path(config.get("output_dir", "outputs")) / "rules") / "precomputed_rules.read_error.json"
        write_json(failure_path, {"cache_path": str(path), "error": str(exc)})
        return None


def _discover_ddi_rules(config: Dict[str, Any], output_dir: Path) -> Tuple[List[Rule], Dict[str, Any]]:
    dataset_name = config.get("dataset_name", "DDI")
    use_llm_rules = _as_bool(config.get("use_llm_rules"), default=False)
    cached = _load_ddi_rule_cache(config, dataset_name=dataset_name, use_llm_rules=use_llm_rules)
    if cached is not None:
        return cached

    if use_llm_rules:
        rule_agent = RuleDiscoveryAgent(output_dir=output_dir / "rules", llm_profile=config.get("llm_profile"), use_llm=True)
        rules = rule_agent.discover("ddi_prediction", dataset_name)
        metadata = {
            "mode": "llm4sd",
            "use_llm_rules": True,
            "rule_output_dir": str(output_dir / "rules"),
            "llm_profile": rule_agent.llm_client.config.profile,
            "llm_provider": rule_agent.llm_client.config.provider,
            "llm_model": rule_agent.llm_client.config.model,
            "llm_available": rule_agent.llm_client.available,
            "source_counts": dict(Counter(rule.source for rule in rules)),
            "rules_signature": rules_signature(rules),
        }
        _store_ddi_rule_cache(config, rules=rules, metadata=metadata, dataset_name=dataset_name, use_llm_rules=True)
        return rules, metadata

    rules = RuleLibrary().literature_rules("DDI", "ddi_prediction")
    metadata = {
        "mode": "fallback",
        "use_llm_rules": False,
        "source_counts": dict(Counter(rule.source for rule in rules)),
        "rules_signature": rules_signature(rules),
    }
    return rules, metadata


def _load_ddi_rule_cache(
    config: Dict[str, Any],
    *,
    dataset_name: str,
    use_llm_rules: bool,
) -> Optional[Tuple[List[Rule], Dict[str, Any]]]:
    cache_path = _ddi_rule_cache_path(config, dataset_name=dataset_name, use_llm_rules=use_llm_rules)
    if cache_path is None or not _as_bool(config.get("task_level_rule_cache_read"), default=True):
        return None
    if not cache_path.exists():
        return None
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        rules = [_rule_from_dict(item) for item in payload.get("rules", [])]
        metadata = dict(payload.get("metadata") or {})
        metadata.update(
            {
                "rule_cache_hit": True,
                "rule_cache_path": str(cache_path),
                "source_counts": dict(Counter(rule.source for rule in rules)),
                "rules_signature": rules_signature(rules),
            }
        )
        return rules, metadata
    except Exception as exc:
        failure_path = cache_path.parent / f"{cache_path.stem}.read_error.json"
        write_json(failure_path, {"cache_path": str(cache_path), "error": str(exc)})
        return None


def _store_ddi_rule_cache(
    config: Dict[str, Any],
    *,
    rules: List[Rule],
    metadata: Dict[str, Any],
    dataset_name: str,
    use_llm_rules: bool,
) -> Optional[Path]:
    cache_path = _ddi_rule_cache_path(config, dataset_name=dataset_name, use_llm_rules=use_llm_rules)
    if cache_path is None or not _as_bool(config.get("task_level_rule_cache_write"), default=True):
        return None
    payload = {
        "metadata": {**metadata, "rule_cache_hit": False, "rule_cache_path": str(cache_path)},
        "rules": [rule.to_dict() for rule in rules],
    }
    return write_json(cache_path, payload)


def _ddi_rule_cache_path(
    config: Dict[str, Any],
    *,
    dataset_name: str,
    use_llm_rules: bool,
) -> Optional[Path]:
    cache_dir = config.get("task_level_rule_cache_dir") or config.get("ddi_rule_cache_dir")
    if not cache_dir or not use_llm_rules:
        return None
    llm_profile = str(config.get("llm_profile") or "default")
    name = _safe_dataset_name(f"ddi_prediction_{dataset_name}_{llm_profile}_rules")
    return ensure_dir(cache_dir) / f"{name}.json"


def _rule_from_dict(data: Dict[str, Any]) -> Rule:
    return Rule(
        rule_id=str(data.get("rule_id") or data.get("feature_name") or "rule"),
        task_type=str(data.get("task_type") or "ddi_prediction"),
        source=str(data.get("source") or "cached"),
        text=str(data.get("text") or ""),
        feature_name=str(data.get("feature_name") or data.get("rule_id") or "rule_feature"),
        code=str(data.get("code") or ""),
        metadata=dict(data.get("metadata") or {}),
        is_executable=_as_bool(data.get("is_executable"), default=True),
        error=data.get("error"),
    )
