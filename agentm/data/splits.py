from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


def _validate_fracs(frac_train: float, frac_valid: float) -> None:
    frac_test = 1.0 - frac_train - frac_valid
    if not np.isclose(frac_train + frac_valid + frac_test, 1.0):
        raise ValueError("Split fractions must sum to 1.0.")
    if min(frac_train, frac_valid, frac_test) < 0:
        raise ValueError("Split fractions must be non-negative.")


def _generate_scaffold(smiles: str, fallback_idx: int) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return f"invalid_{fallback_idx}"
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)


def _scaffold_groups(df: pd.DataFrame, smiles_column: str) -> Dict[str, List[int]]:
    scaffolds: Dict[str, List[int]] = {}
    for idx, smiles in enumerate(df[smiles_column].astype(str)):
        scaffold = _generate_scaffold(smiles, idx)
        scaffolds.setdefault(scaffold, []).append(idx)
    return scaffolds


def random_split(df: pd.DataFrame, seed: int = 13, frac_train: float = 0.8, frac_valid: float = 0.1) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    _validate_fracs(frac_train, frac_valid)
    shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n = len(shuffled)
    n_train = int(n * frac_train)
    n_valid = int(n * frac_valid)
    return shuffled.iloc[:n_train], shuffled.iloc[n_train : n_train + n_valid], shuffled.iloc[n_train + n_valid :]


def scaffold_split(
    df: pd.DataFrame,
    smiles_column: str = "smiles",
    seed: int = 13,
    frac_train: float = 0.8,
    frac_valid: float = 0.1,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """DeepChem-style deterministic Bemis-Murcko scaffold split.

    Scaffold groups are sorted from largest to smallest, then greedily assigned
    to train, validation, and test. ``seed`` is accepted for API compatibility.
    """
    _validate_fracs(frac_train, frac_valid)
    _ = seed
    scaffolds = _scaffold_groups(df, smiles_column)
    scaffolds = {key: sorted(value) for key, value in scaffolds.items()}
    groups = [
        scaffold_set
        for _, scaffold_set in sorted(
            scaffolds.items(),
            key=lambda item: (len(item[1]), item[1][0]),
            reverse=True,
        )
    ]
    train_idx, valid_idx, test_idx = [], [], []
    n = len(df)
    train_cutoff = frac_train * n
    valid_cutoff = (frac_train + frac_valid) * n
    for group in groups:
        if len(train_idx) + len(group) > train_cutoff:
            if len(train_idx) + len(valid_idx) + len(group) > valid_cutoff:
                test_idx.extend(group)
            else:
                valid_idx.extend(group)
        else:
            train_idx.extend(group)
    return df.iloc[train_idx], df.iloc[valid_idx], df.iloc[test_idx]


def random_scaffold_split(
    df: pd.DataFrame,
    smiles_column: str = "smiles",
    seed: int = 13,
    frac_train: float = 0.8,
    frac_valid: float = 0.1,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chainer-Chemistry/ImageMol-style random Bemis-Murcko scaffold split."""
    _validate_fracs(frac_train, frac_valid)
    scaffolds = _scaffold_groups(df, smiles_column)
    rng = np.random.RandomState(seed)
    groups = rng.permutation(np.array(list(scaffolds.values()), dtype=object))

    n_total_valid = int(np.floor(frac_valid * len(df)))
    n_total_test = int(np.floor((1.0 - frac_train - frac_valid) * len(df)))

    train_idx, valid_idx, test_idx = [], [], []
    for group in groups:
        if len(valid_idx) + len(group) <= n_total_valid:
            valid_idx.extend(group)
        elif len(test_idx) + len(group) <= n_total_test:
            test_idx.extend(group)
        else:
            train_idx.extend(group)
    return df.iloc[train_idx], df.iloc[valid_idx], df.iloc[test_idx]


def cold_drug_split(
    df: pd.DataFrame,
    drug_a_col: str = "drug_a_smiles",
    drug_b_col: str = "drug_b_smiles",
    seed: int = 13,
    frac_train: float = 0.8,
    frac_valid: float = 0.1,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    _validate_fracs(frac_train, frac_valid)
    drugs = sorted(set(df[drug_a_col].astype(str)) | set(df[drug_b_col].astype(str)))
    rng = np.random.default_rng(seed)
    rng.shuffle(drugs)
    n_train = int(len(drugs) * frac_train)
    n_valid = int(len(drugs) * frac_valid)
    train_drugs = set(drugs[:n_train])
    valid_drugs = set(drugs[n_train : n_train + n_valid])

    train_mask = df[drug_a_col].astype(str).isin(train_drugs) & df[drug_b_col].astype(str).isin(train_drugs)
    valid_mask = df[drug_a_col].astype(str).isin(valid_drugs) | df[drug_b_col].astype(str).isin(valid_drugs)
    train = df[train_mask]
    valid = df[~train_mask & valid_mask]
    test = df[~train_mask & ~valid_mask]
    return train, valid, test
