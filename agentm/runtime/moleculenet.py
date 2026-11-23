from __future__ import annotations

import hashlib
import json
import math
import random
import urllib.request
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import Crippen, Descriptors, Lipinski, MACCSkeys, QED, rdFingerprintGenerator, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)


DESCRIPTOR_NAMES = (
    "MolWt",
    "MolLogP",
    "TPSA",
    "NumHDonors",
    "NumHAcceptors",
    "NumRotatableBonds",
    "RingCount",
    "HeavyAtomCount",
    "FractionCSP3",
    "NumAromaticRings",
    "NumAliphaticRings",
    "FormalCharge",
    "QED",
)


@dataclass(frozen=True)
class MoleculeNetSpec:
    name: str
    filename: str
    url: str
    problem_type: str
    smiles_column: str
    tasks: tuple[str, ...] | None = None
    task_exclude: tuple[str, ...] = ()
    target_units: str | None = None


MOLECULENET_SPECS: dict[str, MoleculeNetSpec] = {
    "bace": MoleculeNetSpec(
        name="bace",
        filename="bace.csv",
        url="https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/bace.csv",
        problem_type="classification",
        smiles_column="mol",
        tasks=("Class",),
    ),
    "bbbp": MoleculeNetSpec(
        name="bbbp",
        filename="BBBP.csv",
        url="https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv",
        problem_type="classification",
        smiles_column="smiles",
        tasks=("p_np",),
    ),
    "clintox": MoleculeNetSpec(
        name="clintox",
        filename="clintox.csv.gz",
        url="https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/clintox.csv.gz",
        problem_type="classification",
        smiles_column="smiles",
        tasks=("FDA_APPROVED", "CT_TOX"),
    ),
    "tox21": MoleculeNetSpec(
        name="tox21",
        filename="tox21.csv.gz",
        url="https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz",
        problem_type="classification",
        smiles_column="smiles",
        tasks=(
            "NR-AR",
            "NR-AR-LBD",
            "NR-AhR",
            "NR-Aromatase",
            "NR-ER",
            "NR-ER-LBD",
            "NR-PPAR-gamma",
            "SR-ARE",
            "SR-ATAD5",
            "SR-HSE",
            "SR-MMP",
            "SR-p53",
        ),
    ),
    "toxcast": MoleculeNetSpec(
        name="toxcast",
        filename="toxcast_data.csv.gz",
        url="https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/toxcast_data.csv.gz",
        problem_type="classification",
        smiles_column="smiles",
        tasks=None,
        task_exclude=("smiles",),
    ),
    "sider": MoleculeNetSpec(
        name="sider",
        filename="sider.csv.gz",
        url="https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/sider.csv.gz",
        problem_type="classification",
        smiles_column="smiles",
        tasks=(
            "Hepatobiliary disorders",
            "Metabolism and nutrition disorders",
            "Product issues",
            "Eye disorders",
            "Investigations",
            "Musculoskeletal and connective tissue disorders",
            "Gastrointestinal disorders",
            "Social circumstances",
            "Immune system disorders",
            "Reproductive system and breast disorders",
            "Neoplasms benign, malignant and unspecified (incl cysts and polyps)",
            "General disorders and administration site conditions",
            "Endocrine disorders",
            "Surgical and medical procedures",
            "Vascular disorders",
            "Blood and lymphatic system disorders",
            "Skin and subcutaneous tissue disorders",
            "Congenital, familial and genetic disorders",
            "Infections and infestations",
            "Respiratory, thoracic and mediastinal disorders",
            "Psychiatric disorders",
            "Renal and urinary disorders",
            "Pregnancy, puerperium and perinatal conditions",
            "Ear and labyrinth disorders",
            "Cardiac disorders",
            "Nervous system disorders",
            "Injury, poisoning and procedural complications",
        ),
    ),
    "lipophilicity": MoleculeNetSpec(
        name="lipophilicity",
        filename="Lipophilicity.csv",
        url="https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/Lipophilicity.csv",
        problem_type="regression",
        smiles_column="smiles",
        tasks=("exp",),
    ),
    "freesolv": MoleculeNetSpec(
        name="freesolv",
        filename="SAMPL.csv",
        url="https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/SAMPL.csv",
        problem_type="regression",
        smiles_column="smiles",
        tasks=("expt",),
        target_units="kcal/mol",
    ),
    "esol": MoleculeNetSpec(
        name="esol",
        filename="delaney-processed.csv",
        url="https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv",
        problem_type="regression",
        smiles_column="smiles",
        tasks=("measured log solubility in mols per litre",),
    ),
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_dataset(spec: MoleculeNetSpec, raw_dir: str | Path) -> Path:
    target = Path(raw_dir) / spec.filename
    if target.exists() and target.stat().st_size:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    try:
        urllib.request.urlretrieve(spec.url, partial)
        partial.replace(target)
    finally:
        if partial.exists():
            partial.unlink()
    return target


def resolve_tasks(frame: pd.DataFrame, spec: MoleculeNetSpec) -> list[str]:
    if spec.tasks is not None:
        missing = sorted(set(spec.tasks) - set(frame.columns))
        if missing:
            raise ValueError(f"{spec.name}: missing task columns {missing}")
        return list(spec.tasks)
    excluded = set(spec.task_exclude) | {spec.smiles_column}
    tasks = [str(column) for column in frame.columns if column not in excluded]
    if not tasks:
        raise ValueError(f"{spec.name}: no task columns found")
    return tasks


def load_dataset(
    name: str,
    raw_dir: str | Path,
    *,
    download: bool = False,
) -> tuple[pd.DataFrame, MoleculeNetSpec, list[str], dict[str, Any]]:
    key = name.lower()
    if key not in MOLECULENET_SPECS:
        raise KeyError(f"Unknown MoleculeNet dataset: {name}")
    spec = MOLECULENET_SPECS[key]
    path = Path(raw_dir) / spec.filename
    if not path.exists():
        if not download:
            raise FileNotFoundError(f"Missing {path}; rerun with --download")
        path = download_dataset(spec, raw_dir)
    frame = pd.read_csv(path)
    tasks = resolve_tasks(frame, spec)
    source_rows = len(frame)
    smiles = frame[spec.smiles_column].fillna("").astype(str)
    molecules = [Chem.MolFromSmiles(value) for value in smiles]
    valid = np.asarray([mol is not None for mol in molecules], dtype=bool)
    invalid_row_ids = np.flatnonzero(~valid).astype(int).tolist()
    frame = frame.loc[valid].copy().reset_index(drop=False).rename(columns={"index": "source_row_id"})
    frame["smiles"] = frame[spec.smiles_column].astype(str)
    frame["canonical_smiles"] = [
        Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True) for mol in np.asarray(molecules, dtype=object)[valid]
    ]
    numeric_tasks = frame.loc[:, tasks].apply(pd.to_numeric, errors="coerce")
    frame = pd.concat([frame.drop(columns=tasks), numeric_tasks], axis=1)
    if spec.name == "freesolv":
        target_std = float(frame[tasks[0]].std(skipna=True))
        if target_std < 2.0:
            raise ValueError(
                "FreeSolv target appears standardized; expected experimental "
                "hydration free energy in kcal/mol from SAMPL.csv"
            )
    frame["row_id"] = np.arange(len(frame), dtype=int)
    keep = ["row_id", "source_row_id", "smiles", "canonical_smiles", *tasks]
    manifest = {
        "dataset": spec.name,
        "problem_type": spec.problem_type,
        "source_url": spec.url,
        "source_path": str(path),
        "source_sha256": sha256_file(path),
        "source_rows": source_rows,
        "valid_rows": len(frame),
        "invalid_rows": len(invalid_row_ids),
        "invalid_source_row_ids": invalid_row_ids,
        "tasks": tasks,
        "n_tasks": len(tasks),
        "target_units": spec.target_units,
        "labeled_values": {task: int(frame[task].notna().sum()) for task in tasks},
    }
    return frame[keep], spec, tasks, manifest


@lru_cache(maxsize=200_000)
def murcko_scaffold(smiles: str, fallback_idx: int | None = None) -> str:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return f"invalid_{fallback_idx}" if fallback_idx is not None else "invalid"
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)


def scaffold_groups(smiles: Sequence[str]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {}
    for idx, value in enumerate(smiles):
        groups.setdefault(murcko_scaffold(value, idx), []).append(idx)
    return groups


def split_indices(
    smiles: Sequence[str],
    split_type: str,
    *,
    seed: int = 13,
    frac_train: float = 0.8,
    frac_valid: float = 0.1,
) -> dict[str, np.ndarray]:
    frac_test = 1.0 - frac_train - frac_valid
    if min(frac_train, frac_valid, frac_test) < 0 or not np.isclose(frac_train + frac_valid + frac_test, 1.0):
        raise ValueError("Split fractions must be non-negative and sum to 1")
    groups_by_scaffold = scaffold_groups(smiles)
    groups = [sorted(group) for group in groups_by_scaffold.values()]
    if split_type == "scaffold":
        groups.sort(key=lambda group: (len(group), group[0]), reverse=True)
        train: list[int] = []
        valid: list[int] = []
        test: list[int] = []
        train_cutoff = frac_train * len(smiles)
        valid_cutoff = (frac_train + frac_valid) * len(smiles)
        for group in groups:
            if len(train) + len(group) > train_cutoff:
                if len(train) + len(valid) + len(group) > valid_cutoff:
                    test.extend(group)
                else:
                    valid.extend(group)
            else:
                train.extend(group)
    elif split_type == "random_scaffold":
        random_state = np.random.RandomState(seed)
        random_state.shuffle(groups)
        n_valid = int(math.floor(frac_valid * len(smiles)))
        n_test = int(math.floor(frac_test * len(smiles)))
        train, valid, test = [], [], []
        for group in groups:
            if len(valid) + len(group) <= n_valid:
                valid.extend(group)
            elif len(test) + len(group) <= n_test:
                test.extend(group)
            else:
                train.extend(group)
    else:
        raise ValueError(f"Unsupported split type: {split_type}")
    result = {
        "train": np.asarray(train, dtype=np.int64),
        "valid": np.asarray(valid, dtype=np.int64),
        "test": np.asarray(test, dtype=np.int64),
    }
    _validate_split(result, smiles)
    return result


def _validate_split(split: dict[str, np.ndarray], smiles: Sequence[str]) -> None:
    row_sets = {key: set(indices.tolist()) for key, indices in split.items()}
    if row_sets["train"] & row_sets["valid"] or row_sets["train"] & row_sets["test"] or row_sets["valid"] & row_sets["test"]:
        raise AssertionError("Row overlap detected between splits")
    if set.union(*row_sets.values()) != set(range(len(smiles))):
        raise AssertionError("Split does not cover every row exactly once")
    scaffolds = np.asarray([murcko_scaffold(value, idx) for idx, value in enumerate(smiles)], dtype=object)
    scaffold_sets = {key: set(scaffolds[indices].tolist()) for key, indices in split.items()}
    if (
        scaffold_sets["train"] & scaffold_sets["valid"]
        or scaffold_sets["train"] & scaffold_sets["test"]
        or scaffold_sets["valid"] & scaffold_sets["test"]
    ):
        raise AssertionError("Scaffold overlap detected between splits")


def split_audit(
    split: dict[str, np.ndarray],
    smiles: Sequence[str],
    labels: np.ndarray,
    tasks: Sequence[str],
) -> dict[str, Any]:
    scaffolds = np.asarray([murcko_scaffold(value, idx) for idx, value in enumerate(smiles)], dtype=object)
    payload: dict[str, Any] = {"scaffold_overlap": {}, "partitions": {}}
    scaffold_sets = {key: set(scaffolds[idx].tolist()) for key, idx in split.items()}
    for left, right in (("train", "valid"), ("train", "test"), ("valid", "test")):
        payload["scaffold_overlap"][f"{left}_{right}"] = len(scaffold_sets[left] & scaffold_sets[right])
    for part, indices in split.items():
        y = labels[indices]
        task_counts = {}
        for task_idx, task in enumerate(tasks):
            values = y[:, task_idx]
            finite = values[np.isfinite(values)]
            task_counts[task] = {
                "labeled": int(len(finite)),
                "positive": int(np.sum(finite == 1)) if len(finite) else 0,
                "negative": int(np.sum(finite == 0)) if len(finite) else 0,
            }
        payload["partitions"][part] = {
            "rows": int(len(indices)),
            "scaffolds": int(len(scaffold_sets[part])),
            "tasks": task_counts,
        }
    return payload


def _descriptor_vector(mol: Chem.Mol) -> np.ndarray:
    charge = sum(atom.GetFormalCharge() for atom in mol.GetAtoms())
    return np.asarray(
        [
            Descriptors.MolWt(mol),
            Crippen.MolLogP(mol),
            rdMolDescriptors.CalcTPSA(mol),
            Lipinski.NumHDonors(mol),
            Lipinski.NumHAcceptors(mol),
            Lipinski.NumRotatableBonds(mol),
            rdMolDescriptors.CalcNumRings(mol),
            mol.GetNumHeavyAtoms(),
            rdMolDescriptors.CalcFractionCSP3(mol),
            rdMolDescriptors.CalcNumAromaticRings(mol),
            rdMolDescriptors.CalcNumAliphaticRings(mol),
            charge,
            QED.qed(mol),
        ],
        dtype=np.float32,
    )


def featurize_smiles(smiles: Sequence[str]) -> np.ndarray:
    features = np.zeros((len(smiles), len(DESCRIPTOR_NAMES) + 2048 + 167), dtype=np.float32)
    morgan = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    for idx, value in enumerate(smiles):
        mol = Chem.MolFromSmiles(str(value))
        if mol is None:
            raise ValueError(f"Invalid SMILES at feature row {idx}: {value}")
        features[idx, : len(DESCRIPTOR_NAMES)] = _descriptor_vector(mol)
        morgan_array = np.zeros((2048,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(morgan.GetFingerprint(mol), morgan_array)
        maccs_array = np.zeros((167,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(MACCSkeys.GenMACCSKeys(mol), maccs_array)
        start = len(DESCRIPTOR_NAMES)
        features[idx, start : start + 2048] = morgan_array
        features[idx, start + 2048 :] = maccs_array
    return features


def load_or_build_features(
    frame: pd.DataFrame,
    cache_path: str | Path,
    *,
    force: bool = False,
) -> np.ndarray:
    target = Path(cache_path)
    smiles_hash = hashlib.sha256("\n".join(frame["canonical_smiles"].astype(str)).encode("utf-8")).hexdigest()
    if target.exists() and not force:
        with np.load(target, allow_pickle=False) as cached:
            if str(cached["smiles_sha256"]) == smiles_hash:
                return cached["features"].astype(np.float32, copy=False)
    target.parent.mkdir(parents=True, exist_ok=True)
    features = featurize_smiles(frame["canonical_smiles"].astype(str).tolist())
    np.savez_compressed(target, features=features, smiles_sha256=np.asarray(smiles_hash))
    return features


ALL_FEATURES_CACHE_SCHEMA = "agentm_all_features"
ADMET_MOLECULENET_DATASETS = {"clintox", "sider", "tox21", "toxcast"}
DATASET_DISPLAY_NAMES = {
    "bace": "BACE",
    "bbbp": "BBBP",
    "clintox": "ClinTox",
    "esol": "ESOL",
    "freesolv": "FreeSolv",
    "lipophilicity": "Lipophilicity",
    "sider": "SIDER",
    "tox21": "Tox21",
    "toxcast": "ToxCast",
}


def moleculenet_task_type(dataset: str) -> str:
    return "admet_prediction" if dataset.lower() in ADMET_MOLECULENET_DATASETS else "property_prediction"


def _all_feature_dependencies() -> tuple[Any, Any]:
    from agentm.runtime.compat import ensure_import_paths

    ensure_import_paths()
    from agentm.features.builder import FeatureBuilder
    from agentm.schema import MoleculeRecord

    return FeatureBuilder, MoleculeRecord


def load_or_build_all_features(
    frame: pd.DataFrame,
    cache_path: str | Path,
    *,
    dataset: str,
    artifacts_dir: str | Path,
    chemberta_model_path: str | Path,
    imagemol_checkpoint_path: str | Path,
    encoder_device: str = "auto",
    force: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build label-independent AgentM multimodal features once per dataset.

    Split-specific, training-derived rule features are appended separately by
    ``build_split_rule_features`` so validation and test labels never influence
    feature construction.
    """

    target = Path(cache_path)
    smiles_values = frame["canonical_smiles"].astype(str).tolist()
    smiles_hash = hashlib.sha256("\n".join(smiles_values).encode("utf-8")).hexdigest()
    requested = {
        "cache_schema": ALL_FEATURES_CACHE_SCHEMA,
        "chemberta_model_path": str(Path(chemberta_model_path).resolve()),
        "imagemol_checkpoint_path": str(Path(imagemol_checkpoint_path).resolve()),
        "visual_semantic_provider": "fallback_rdkit",
    }
    if target.exists() and not force:
        with np.load(target, allow_pickle=False) as cached:
            metadata = json.loads(str(cached["metadata_json"]))
            if (
                str(cached["smiles_sha256"]) == smiles_hash
                and all(metadata.get(key) == value for key, value in requested.items())
            ):
                return cached["features"].astype(np.float32, copy=False), metadata

    FeatureBuilder, MoleculeRecord = _all_feature_dependencies()
    builder = FeatureBuilder(
        output_dir=artifacts_dir,
        feature_set="all_features",
        imagemol_checkpoint_path=str(imagemol_checkpoint_path),
        use_smiles_encoder=True,
        smiles_encoder_model_name=str(chemberta_model_path),
        smiles_encoder_device=encoder_device,
        smiles_local_files_only=True,
        use_mllm_visual_semantic=False,
        generate_grad_cam=False,
    )
    smiles_encoder_name = str(getattr(builder.smiles_encoder, "encoder_name", "disabled"))
    imagemol_encoder_name = str(getattr(builder.imagemol_encoder, "encoder_name", "disabled"))
    if smiles_encoder_name != "deepchem_chemberta2":
        raise RuntimeError(f"ChemBERTa-2 unavailable; refusing mock all_features: {smiles_encoder_name}")
    if imagemol_encoder_name != "chengf_lab_imagemol_resnet18":
        raise RuntimeError(f"ImageMol unavailable; refusing mock all_features: {imagemol_encoder_name}")

    unique_vectors: dict[str, np.ndarray] = {}
    rows: list[np.ndarray] = []
    feature_names: list[str] | None = None
    blocks: dict[str, dict[str, Any]] | None = None
    task_type = moleculenet_task_type(dataset)
    for idx, smiles in enumerate(smiles_values):
        vector = unique_vectors.get(smiles)
        if vector is None:
            record = MoleculeRecord(
                original_input=smiles,
                smiles=smiles,
                canonical_smiles=smiles,
                source=f"moleculenet:{dataset}",
                is_valid=True,
            )
            bundle = builder.build_single(
                record,
                rules=(),
                task_type=task_type,
                dataset_name=DATASET_DISPLAY_NAMES.get(dataset.lower(), dataset),
            )
            vector = np.asarray(bundle.feature_vector, dtype=np.float32)
            unique_vectors[smiles] = vector
            if feature_names is None:
                feature_names = list(bundle.feature_schema.feature_names)
                blocks = dict(bundle.feature_schema.blocks)
        rows.append(vector)
        if (idx + 1) % 100 == 0 or idx + 1 == len(smiles_values):
            print(
                f"[features] dataset={dataset} rows={idx + 1}/{len(smiles_values)} "
                f"unique={len(unique_vectors)}",
                flush=True,
            )

    features = np.stack(rows).astype(np.float32, copy=False)
    block_dims = {name: int(payload.get("dim", 0)) for name, payload in (blocks or {}).items()}
    expected_dims = {"rdkit": 13, "fingerprint": 2215, "smiles": 384, "imagemol": 512, "visual_semantic": 64}
    for name, expected in expected_dims.items():
        if block_dims.get(name) != expected:
            raise RuntimeError(f"Unexpected {name} dimension: {block_dims.get(name)} != {expected}")
    if features.shape[1] != 3188:
        raise RuntimeError(f"Unexpected base all_features dimension: {features.shape[1]} != 3188")

    metadata = {
        **requested,
        "feature_set": "all_features",
        "base_total": int(features.shape[1]),
        "feature_names": feature_names or [],
        "feature_blocks": block_dims,
        "smiles_encoder": smiles_encoder_name,
        "imagemol_encoder": imagemol_encoder_name,
        "n_rows": int(len(features)),
        "n_unique_smiles": int(len(unique_vectors)),
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.stem + ".tmp.npz")
    np.savez(
        temporary,
        features=features,
        smiles_sha256=np.asarray(smiles_hash),
        metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False)),
    )
    temporary.replace(target)
    return features, metadata


def _rule_endpoint_name(task: str) -> str:
    cleaned = "".join(character.lower() if character.isalnum() else "_" for character in str(task))
    cleaned = "_".join(part for part in cleaned.split("_") if part)
    digest = hashlib.sha1(str(task).encode("utf-8")).hexdigest()[:8]
    return f"{cleaned[:36] or 'task'}_{digest}"


def build_split_rule_features(
    frame: pd.DataFrame,
    tasks: Sequence[str],
    train_indices: np.ndarray,
    base_features: np.ndarray,
    *,
    dataset: str,
    problem_type: str,
    max_rule_features: int = 23,
) -> tuple[np.ndarray, list[Any], dict[str, Any]]:
    """Create deterministic endpoint-aware rules using training labels only."""

    from agentm.runtime.compat import ensure_import_paths

    ensure_import_paths()
    from agentm.rules.compiler import OPS, RuleCompiler
    from agentm.rules.library import RuleLibrary

    task_type = moleculenet_task_type(dataset)
    display_name = DATASET_DISPLAY_NAMES.get(dataset.lower(), dataset)
    library = RuleLibrary()
    compiler = RuleCompiler()
    literature = [rule for rule in library.literature_rules(display_name, task_type) if rule.is_executable]
    selected: list[Any] = list(literature[:max_rule_features])
    remaining_slots = max(0, int(max_rule_features) - len(selected))

    descriptor_values = base_features[:, : len(DESCRIPTOR_NAMES)].astype(np.float64, copy=False)
    candidates: list[tuple[float, Any]] = []
    train_labels = frame.iloc[train_indices][list(tasks)].to_numpy(dtype=np.float64)
    train_descriptors = descriptor_values[train_indices]
    for task_idx, task in enumerate(tasks):
        labels = train_labels[:, task_idx]
        finite = np.isfinite(labels)
        if finite.sum() < 4:
            continue
        values = train_descriptors[finite]
        target = labels[finite]
        if problem_type == "classification":
            high_mask = target >= 0.5
            low_mask = target < 0.5
        else:
            threshold_y = float(np.median(target))
            high_mask = target >= threshold_y
            low_mask = target < threshold_y
        if high_mask.sum() < 2 or low_mask.sum() < 2:
            continue
        endpoint = _rule_endpoint_name(task)
        for descriptor_idx, descriptor in enumerate(DESCRIPTOR_NAMES):
            high_mean = float(values[high_mask, descriptor_idx].mean())
            low_mean = float(values[low_mask, descriptor_idx].mean())
            pooled_scale = float(values[:, descriptor_idx].std())
            if not np.isfinite(high_mean + low_mean + pooled_scale) or pooled_scale < 1e-8:
                continue
            effect = abs(high_mean - low_mean) / pooled_scale
            if effect < 1e-6:
                continue
            operator = ">" if high_mean > low_mean else "<"
            threshold_x = float((high_mean + low_mean) / 2.0)
            spec = {
                "operation": "descriptor_threshold",
                "descriptor": descriptor,
                "operator": operator,
                "threshold": threshold_x,
                "text": (
                    f"Training-derived {display_name}/{task} rule: {descriptor} {operator} "
                    f"{threshold_x:.6g}."
                ),
                "endpoint": str(task),
                "standardized_effect": float(effect),
                "training_only": True,
            }
            rule = compiler.from_spec(spec, task_type=task_type, source="training_data_inferred")
            rule.feature_name = f"rule_{endpoint}__{rule.feature_name.removeprefix('rule_')}"
            candidates.append((float(effect), rule))

    candidates.sort(key=lambda item: (-item[0], item[1].feature_name))
    selected.extend(rule for _, rule in candidates[:remaining_slots])
    feature_names = [rule.feature_name for rule in selected]
    if len(feature_names) != len(set(feature_names)):
        raise RuntimeError("Duplicate rule feature names after endpoint qualification")

    rule_matrix = np.zeros((len(frame), len(selected)), dtype=np.float32)
    molecules: list[Chem.Mol] | None = None
    descriptor_index = {name: idx for idx, name in enumerate(DESCRIPTOR_NAMES)}
    for rule_idx, rule in enumerate(selected):
        metadata = rule.metadata
        operation = metadata.get("operation")
        if operation == "descriptor_threshold":
            values = descriptor_values[:, descriptor_index[str(metadata["descriptor"])]]
            operator = str(metadata["operator"])
            if operator not in OPS:
                raise ValueError(f"Unsupported rule operator: {operator}")
            rule_matrix[:, rule_idx] = OPS[operator](values, float(metadata["threshold"])).astype(np.float32)
        elif operation == "substructure":
            if molecules is None:
                molecules = [Chem.MolFromSmiles(value) for value in frame["canonical_smiles"].astype(str)]
            pattern = Chem.MolFromSmarts(str(metadata["smarts"]))
            if pattern is None:
                raise ValueError(f"Invalid rule SMARTS: {metadata['smarts']}")
            rule_matrix[:, rule_idx] = np.asarray(
                [float(molecule.HasSubstructMatch(pattern)) for molecule in molecules],
                dtype=np.float32,
            )
        else:
            raise ValueError(f"Unsupported rule operation: {operation}")

    manifest = {
        "dataset": dataset,
        "task_type": task_type,
        "max_rule_features": int(max_rule_features),
        "n_literature_rules": int(min(len(literature), len(selected))),
        "n_training_derived_rules": int(max(0, len(selected) - len(literature))),
        "n_rule_features": int(len(selected)),
        "training_rows": int(len(train_indices)),
        "training_labels_only": True,
        "validation_or_test_labels_used": False,
        "feature_names": feature_names,
        "rules": [rule.to_dict() for rule in selected],
    }
    return rule_matrix, selected, manifest


def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float]:
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return float("nan"), float("nan")
    return float(roc_auc_score(y_true, y_score)), float(average_precision_score(y_true, y_score))


def classification_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    tasks: Sequence[str],
    *,
    threshold: float = 0.5,
) -> tuple[dict[str, Any], pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for idx, task in enumerate(tasks):
        mask = np.isfinite(y_true[:, idx]) & np.isfinite(y_score[:, idx])
        truth = y_true[mask, idx].astype(int)
        score = y_score[mask, idx]
        pred = (score >= threshold).astype(int)
        roc_auc, pr_auc = _safe_auc(truth, score)
        rows.append(
            {
                "task": task,
                "n": int(mask.sum()),
                "n_positive": int(truth.sum()),
                "n_negative": int(len(truth) - truth.sum()),
                "roc_auc": roc_auc,
                "pr_auc": pr_auc,
                "accuracy": float(accuracy_score(truth, pred)) if len(truth) else float("nan"),
                "balanced_accuracy": (
                    float(balanced_accuracy_score(truth, pred)) if len(np.unique(truth)) == 2 else float("nan")
                ),
                "f1": float(f1_score(truth, pred, zero_division=0)) if len(truth) else float("nan"),
            }
        )
    per_task = pd.DataFrame(rows)
    metric_columns = ("roc_auc", "pr_auc", "accuracy", "balanced_accuracy", "f1")
    aggregate = {
        f"{metric}_macro": float(per_task[metric].mean(skipna=True)) for metric in metric_columns
    }
    aggregate.update(
        {
            "n_tasks": len(tasks),
            "n_tasks_roc_auc": int(per_task["roc_auc"].notna().sum()),
            "n_labeled_test": int(np.isfinite(y_true).sum()),
            "threshold": threshold,
        }
    )
    if len(tasks) == 1:
        aggregate.update({metric: float(per_task.iloc[0][metric]) for metric in metric_columns})
    return aggregate, per_task


def regression_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    tasks: Sequence[str],
) -> tuple[dict[str, Any], pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for idx, task in enumerate(tasks):
        mask = np.isfinite(y_true[:, idx]) & np.isfinite(y_score[:, idx])
        truth = y_true[mask, idx]
        pred = y_score[mask, idx]
        rows.append(
            {
                "task": task,
                "n": int(mask.sum()),
                "rmse": float(math.sqrt(mean_squared_error(truth, pred))) if len(truth) else float("nan"),
                "mae": float(mean_absolute_error(truth, pred)) if len(truth) else float("nan"),
                "r2": float(r2_score(truth, pred)) if len(truth) >= 2 else float("nan"),
            }
        )
    per_task = pd.DataFrame(rows)
    aggregate = {
        "rmse_macro": float(per_task["rmse"].mean(skipna=True)),
        "mae_macro": float(per_task["mae"].mean(skipna=True)),
        "r2_macro": float(per_task["r2"].mean(skipna=True)),
        "n_tasks": len(tasks),
        "n_labeled_test": int(np.isfinite(y_true).sum()),
    }
    if len(tasks) == 1:
        aggregate.update({metric: float(per_task.iloc[0][metric]) for metric in ("rmse", "mae", "r2")})
    return aggregate, per_task


class MaskedMLP:
    def __init__(
        self,
        *,
        problem_type: str,
        hidden_dims: tuple[int, ...] = (512, 256),
        dropout: float = 0.2,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        batch_size: int = 256,
        max_epochs: int = 100,
        patience: int = 15,
        seed: int = 13,
        device: str = "auto",
    ):
        self.problem_type = problem_type
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.seed = seed
        self.device_requested = device
        self.device = "cpu"
        self.model: Any = None
        self.x_mean: np.ndarray | None = None
        self.x_scale: np.ndarray | None = None
        self.y_mean: np.ndarray | None = None
        self.y_scale: np.ndarray | None = None
        self.history: list[dict[str, Any]] = []
        self.best_epoch = -1
        self.best_score = float("-inf")
        self.pos_weight: np.ndarray | None = None

    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_valid: np.ndarray,
        y_valid: np.ndarray,
        tasks: Sequence[str],
    ) -> "MaskedMLP":
        import torch
        from torch import nn

        self._seed_everything(torch)
        if self.device_requested == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = self.device_requested
        self.x_mean = np.zeros(x_train.shape[1], dtype=np.float32)
        self.x_scale = np.ones(x_train.shape[1], dtype=np.float32)
        n_desc = len(DESCRIPTOR_NAMES)
        self.x_mean[:n_desc] = x_train[:, :n_desc].mean(axis=0)
        self.x_scale[:n_desc] = x_train[:, :n_desc].std(axis=0)
        self.x_scale[self.x_scale < 1e-8] = 1.0
        train_x = self._transform_x(x_train)
        valid_x = self._transform_x(x_valid)
        train_y = y_train.astype(np.float32, copy=True)
        valid_y = y_valid.astype(np.float32, copy=True)
        if self.problem_type == "regression":
            self.y_mean = np.nanmean(train_y, axis=0).astype(np.float32)
            self.y_scale = np.nanstd(train_y, axis=0).astype(np.float32)
            self.y_scale[self.y_scale < 1e-8] = 1.0
            train_y = (train_y - self.y_mean) / self.y_scale
            valid_y = (valid_y - self.y_mean) / self.y_scale
        else:
            positives = np.nansum(train_y == 1, axis=0)
            negatives = np.nansum(train_y == 0, axis=0)
            self.pos_weight = np.clip(negatives / np.maximum(positives, 1), 1.0, 50.0).astype(np.float32)

        layers: list[nn.Module] = []
        last_dim = train_x.shape[1]
        for hidden in self.hidden_dims:
            layers.extend((nn.Linear(last_dim, hidden), nn.ReLU(), nn.LayerNorm(hidden), nn.Dropout(self.dropout)))
            last_dim = hidden
        layers.append(nn.Linear(last_dim, train_y.shape[1]))
        self.model = nn.Sequential(*layers).to(self.device)
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        x_tensor = torch.from_numpy(train_x)
        y_tensor = torch.from_numpy(train_y)
        best_state: dict[str, Any] | None = None
        stale = 0
        generator = torch.Generator(device="cpu").manual_seed(self.seed)
        for epoch in range(self.max_epochs):
            self.model.train()
            order = torch.randperm(len(x_tensor), generator=generator)
            train_losses = []
            for start in range(0, len(order), self.batch_size):
                batch = order[start : start + self.batch_size]
                bx = x_tensor[batch].to(self.device)
                by = y_tensor[batch].to(self.device)
                optimizer.zero_grad(set_to_none=True)
                logits = self.model(bx)
                loss = self._masked_loss(logits, by, torch)
                loss.backward()
                optimizer.step()
                train_losses.append(float(loss.detach().cpu()))
            valid_prediction = self.predict(x_valid)
            if self.problem_type == "classification":
                valid_metrics, _ = classification_metrics(y_valid, valid_prediction, tasks)
                score = valid_metrics["roc_auc_macro"]
                if not np.isfinite(score):
                    score = -self._evaluation_loss(valid_x, valid_y, torch)
            else:
                valid_metrics, _ = regression_metrics(y_valid, valid_prediction, tasks)
                score = -valid_metrics["rmse_macro"]
            self.history.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": float(np.mean(train_losses)),
                    "validation_score": float(score),
                }
            )
            if score > self.best_score + 1e-6:
                self.best_score = float(score)
                self.best_epoch = epoch + 1
                best_state = {key: value.detach().cpu().clone() for key, value in self.model.state_dict().items()}
                stale = 0
            else:
                stale += 1
                if stale >= self.patience:
                    break
        if best_state is None:
            raise RuntimeError("Training did not produce a checkpoint")
        self.model.load_state_dict(best_state)
        return self

    def _seed_everything(self, torch: Any) -> None:
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

    def _transform_x(self, values: np.ndarray) -> np.ndarray:
        if self.x_mean is None or self.x_scale is None:
            raise RuntimeError("Feature scaler is not fitted")
        return ((values.astype(np.float32) - self.x_mean) / self.x_scale).astype(np.float32)

    def _masked_loss(self, logits: Any, labels: Any, torch: Any) -> Any:
        mask = torch.isfinite(labels)
        targets = torch.nan_to_num(labels, nan=0.0)
        if self.problem_type == "classification":
            pos_weight = torch.as_tensor(self.pos_weight, device=logits.device)
            raw = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, targets, pos_weight=pos_weight, reduction="none"
            )
        else:
            raw = (logits - targets).square()
        per_task_count = mask.sum(dim=0)
        active = per_task_count > 0
        per_task = (raw * mask).sum(dim=0) / per_task_count.clamp_min(1)
        return per_task[active].mean()

    def _evaluation_loss(self, x: np.ndarray, y: np.ndarray, torch: Any) -> float:
        self.model.eval()
        with torch.no_grad():
            logits = self.model(torch.from_numpy(x).to(self.device))
            return float(self._masked_loss(logits, torch.from_numpy(y).to(self.device), torch).cpu())

    def predict(self, values: np.ndarray) -> np.ndarray:
        import torch

        if self.model is None:
            raise RuntimeError("Model is not fitted")
        self.model.eval()
        transformed = self._transform_x(values)
        batches = []
        with torch.no_grad():
            for start in range(0, len(transformed), self.batch_size):
                batch = torch.from_numpy(transformed[start : start + self.batch_size]).to(self.device)
                batches.append(self.model(batch).cpu().numpy())
        prediction = np.concatenate(batches, axis=0)
        if self.problem_type == "classification":
            prediction = 1.0 / (1.0 + np.exp(-np.clip(prediction, -40, 40)))
        else:
            prediction = prediction * self.y_scale + self.y_mean
        return prediction.astype(np.float32)

    def checkpoint(self) -> dict[str, Any]:
        return {
            "problem_type": self.problem_type,
            "hidden_dims": self.hidden_dims,
            "dropout": self.dropout,
            "seed": self.seed,
            "device": self.device,
            "x_mean": self.x_mean,
            "x_scale": self.x_scale,
            "y_mean": self.y_mean,
            "y_scale": self.y_scale,
            "pos_weight": self.pos_weight,
            "best_epoch": self.best_epoch,
            "best_score": self.best_score,
            "state_dict": self.model.state_dict() if self.model is not None else None,
        }


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def write_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(_jsonable(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def orient_clintox_risk_labels(
    frame: pd.DataFrame,
    dataset: str,
    tasks: Sequence[str],
    manifest: dict[str, Any],
    *,
    enabled: bool,
) -> tuple[pd.DataFrame, list[str], dict[str, Any], dict[str, Any]]:
    if not enabled:
        return frame, list(tasks), manifest, {"name": "original", "positive_events": list(tasks)}
    if dataset.lower() != "clintox":
        raise ValueError("--clintox-risk-labels is only valid for the ClinTox dataset")
    expected = ["FDA_APPROVED", "CT_TOX"]
    if list(tasks) != expected:
        raise ValueError(f"Unexpected ClinTox tasks for risk conversion: {list(tasks)}")
    oriented = frame.copy()
    oriented["FDA_NOT_APPROVED"] = 1.0 - pd.to_numeric(oriented["FDA_APPROVED"], errors="coerce")
    risk_tasks = ["FDA_NOT_APPROVED", "CT_TOX"]
    risk_manifest = dict(manifest)
    risk_manifest["source_tasks"] = list(tasks)
    risk_manifest["tasks"] = risk_tasks
    risk_manifest["n_tasks"] = len(risk_tasks)
    risk_manifest["labeled_values"] = {
        task: int(oriented[task].notna().sum()) for task in risk_tasks
    }
    transform = {
        "name": "clintox_risk_orientation",
        "positive_events": risk_tasks,
        "transforms": {
            "FDA_NOT_APPROVED": "1 - FDA_APPROVED",
            "CT_TOX": "identity",
        },
        "source_label_preserved_in_frame": True,
    }
    risk_manifest["label_orientation"] = transform
    return oriented, risk_tasks, risk_manifest, transform


def run_one_benchmark(
    dataset: str,
    split_type: str,
    *,
    raw_dir: str | Path,
    processed_dir: str | Path,
    output_root: str | Path,
    seed: int = 13,
    device: str = "auto",
    max_epochs: int = 100,
    patience: int = 15,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    hidden_dims: tuple[int, ...] = (512, 256),
    dropout: float = 0.2,
    download: bool = False,
    force_features: bool = False,
    feature_set: str = "agentm_structural",
    chemberta_model_path: str | Path | None = None,
    imagemol_checkpoint_path: str | Path | None = None,
    max_rule_features: int = 23,
    clintox_risk_labels: bool = False,
) -> dict[str, Any]:
    import torch

    frame, spec, tasks, manifest = load_dataset(dataset, raw_dir, download=download)
    frame, tasks, manifest, label_orientation = orient_clintox_risk_labels(
        frame,
        spec.name,
        tasks,
        manifest,
        enabled=clintox_risk_labels,
    )
    labels = frame[tasks].to_numpy(dtype=np.float32)
    split = split_indices(frame["canonical_smiles"].tolist(), split_type, seed=seed)
    output_dir = Path(output_root) / spec.name / split_type / f"seed_{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    if feature_set == "agentm_structural":
        features = load_or_build_features(
            frame,
            Path(processed_dir) / spec.name / "agentm_structural_features.npz",
            force=force_features,
        )
        feature_metadata = {
            "feature_set": "agentm_structural",
            "feature_blocks": {
                "rdkit_descriptors": len(DESCRIPTOR_NAMES),
                "morgan_radius2": 2048,
                "maccs": 167,
                "total": int(features.shape[1]),
            },
        }
    elif feature_set == "all_features":
        if chemberta_model_path is None or imagemol_checkpoint_path is None:
            raise ValueError("all_features requires ChemBERTa-2 and ImageMol checkpoint paths")
        base_features, base_metadata = load_or_build_all_features(
            frame,
            Path(processed_dir) / spec.name / "agentm_all_features_base.npz",
            dataset=spec.name,
            artifacts_dir=Path(processed_dir) / spec.name / "all_features_artifacts",
            chemberta_model_path=chemberta_model_path,
            imagemol_checkpoint_path=imagemol_checkpoint_path,
            encoder_device="auto" if device == "cuda" else device,
            force=force_features,
        )
        rule_features, _, rule_manifest = build_split_rule_features(
            frame,
            tasks,
            split["train"],
            base_features,
            dataset=spec.name,
            problem_type=spec.problem_type,
            max_rule_features=max_rule_features,
        )
        features = np.concatenate([base_features, rule_features], axis=1)
        write_json(output_dir / "rule_manifest.json", rule_manifest)
        feature_metadata = {
            **base_metadata,
            "feature_set": "all_features",
            "feature_blocks": {
                **dict(base_metadata["feature_blocks"]),
                "endpoint_rules": int(rule_features.shape[1]),
                "total": int(features.shape[1]),
            },
            "rule_protocol": {
                "training_labels_only": True,
                "validation_or_test_labels_used": False,
                "manifest": str(output_dir / "rule_manifest.json"),
            },
        }
    else:
        raise ValueError(f"Unsupported feature_set: {feature_set}")
    audit = split_audit(split, frame["canonical_smiles"].tolist(), labels, tasks)
    model = MaskedMLP(
        problem_type=spec.problem_type,
        hidden_dims=hidden_dims,
        dropout=dropout,
        learning_rate=learning_rate,
        batch_size=batch_size,
        max_epochs=max_epochs,
        patience=patience,
        seed=seed,
        device=device,
    )
    model.fit(
        features[split["train"]],
        labels[split["train"]],
        features[split["valid"]],
        labels[split["valid"]],
        tasks,
    )
    test_score = model.predict(features[split["test"]])
    test_true = labels[split["test"]]
    if spec.problem_type == "classification":
        metrics, per_task = classification_metrics(test_true, test_score, tasks)
    else:
        metrics, per_task = regression_metrics(test_true, test_score, tasks)

    assignment = frame[["row_id", "source_row_id", "smiles", "canonical_smiles"]].copy()
    assignment["scaffold"] = [
        murcko_scaffold(value, idx) for idx, value in enumerate(frame["canonical_smiles"].tolist())
    ]
    assignment["split"] = ""
    for part, indices in split.items():
        assignment.loc[indices, "split"] = part
    assignment.to_csv(output_dir / "split_assignments.csv.gz", index=False, compression="gzip")
    per_task.to_csv(output_dir / "per_task_metrics.csv", index=False)
    pd.DataFrame(model.history).to_csv(output_dir / "training_history.csv", index=False)
    np.savez_compressed(
        output_dir / "test_predictions.npz",
        row_id=frame.iloc[split["test"]]["row_id"].to_numpy(),
        tasks=np.asarray(tasks),
        y_true=test_true,
        y_score=test_score,
    )
    torch.save(model.checkpoint(), output_dir / "model.pt")
    config = {
        "dataset": spec.name,
        "split_type": split_type,
        "seed": seed,
        "label_orientation": label_orientation,
        "feature_set": feature_set,
        "feature_blocks": feature_metadata["feature_blocks"],
        "feature_metadata": feature_metadata,
        "model": "masked_mlp",
        "hidden_dims": list(hidden_dims),
        "dropout": dropout,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "max_epochs": max_epochs,
        "patience": patience,
        "device_requested": device,
        "device_used": model.device,
        "torch_version": torch.__version__,
    }
    result = {
        "dataset": spec.name,
        "problem_type": spec.problem_type,
        "split_type": split_type,
        "seed": seed,
        "label_orientation": label_orientation["name"],
        "n_rows": len(frame),
        "n_tasks": len(tasks),
        "n_train": int(len(split["train"])),
        "n_valid": int(len(split["valid"])),
        "n_test": int(len(split["test"])),
        "feature_set": feature_set,
        "model": "masked_mlp",
        "best_epoch": model.best_epoch,
        "validation_score": model.best_score,
        "metrics": metrics,
        "output_dir": str(output_dir),
    }
    write_json(output_dir / "data_manifest.json", manifest)
    write_json(output_dir / "split_audit.json", audit)
    write_json(output_dir / "config.json", config)
    write_json(output_dir / "result.json", result)
    return result


def summarize_results(results: Iterable[dict[str, Any]], output_root: str | Path) -> pd.DataFrame:
    materialized = list(results)
    rows = []
    for result in materialized:
        row = {key: value for key, value in result.items() if key != "metrics"}
        row.update(result["metrics"])
        rows.append(row)
    summary = pd.DataFrame(rows).sort_values(["dataset", "split_type", "seed"]).reset_index(drop=True)
    if "feature_set" not in summary:
        summary["feature_set"] = "unspecified"
    for metric in (
        "roc_auc_macro",
        "pr_auc_macro",
        "balanced_accuracy_macro",
        "rmse_macro",
        "mae_macro",
        "r2_macro",
    ):
        if metric not in summary.columns:
            summary[metric] = np.nan
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    summary.to_csv(root / "benchmark_summary.csv", index=False)
    write_json(root / "benchmark_summary.json", materialized)
    classification = summary.loc[
        summary["problem_type"] == "classification",
        [
            "dataset",
            "split_type",
            "seed",
            "n_tasks",
            "n_test",
            "roc_auc_macro",
            "pr_auc_macro",
            "balanced_accuracy_macro",
        ],
    ].copy()
    regression = summary.loc[
        summary["problem_type"] == "regression",
        ["dataset", "split_type", "seed", "n_test", "rmse_macro", "mae_macro", "r2_macro"],
    ].copy()
    markdown = [
        "# MoleculeNet scaffold / random-scaffold benchmark",
        "",
        "- Canonical DeepChem MoleculeNet raw tables",
        "- 80/10/10 molecule split; zero Bemis–Murcko scaffold overlap",
        f"- Feature sets: {', '.join(sorted(set(summary['feature_set'].astype(str))))}",
        "- Unified masked MLP; missing multi-task labels are excluded from loss and metrics",
        "- FreeSolv RMSE and MAE are reported in kcal/mol from the unstandardized experimental target",
        "",
        "## Classification",
        "",
        _markdown_table(classification),
        "",
        "## Regression",
        "",
        _markdown_table(regression),
        "",
    ]
    (root / "benchmark_summary.md").write_text("\n".join(markdown), encoding="utf-8")
    return summary


def _markdown_table(frame: pd.DataFrame) -> str:
    columns = [str(column) for column in frame.columns]

    def render(value: Any) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.4f}"
        return str(value).replace("|", r"\|")

    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for values in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(render(value) for value in values) + " |")
    return "\n".join(lines)
