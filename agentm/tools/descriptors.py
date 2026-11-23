from __future__ import annotations

from typing import Dict, List

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import Crippen, Descriptors, Lipinski, MACCSkeys, QED, rdMolDescriptors


DESCRIPTOR_NAMES = [
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
]


def compute_rdkit_descriptors(mol: Chem.Mol) -> Dict[str, float]:
    formal_charge = sum(atom.GetFormalCharge() for atom in mol.GetAtoms())
    values = {
        "MolWt": Descriptors.MolWt(mol),
        "MolLogP": Crippen.MolLogP(mol),
        "TPSA": rdMolDescriptors.CalcTPSA(mol),
        "NumHDonors": Lipinski.NumHDonors(mol),
        "NumHAcceptors": Lipinski.NumHAcceptors(mol),
        "NumRotatableBonds": Lipinski.NumRotatableBonds(mol),
        "RingCount": rdMolDescriptors.CalcNumRings(mol),
        "HeavyAtomCount": mol.GetNumHeavyAtoms(),
        "FractionCSP3": rdMolDescriptors.CalcFractionCSP3(mol),
        "NumAromaticRings": rdMolDescriptors.CalcNumAromaticRings(mol),
        "NumAliphaticRings": rdMolDescriptors.CalcNumAliphaticRings(mol),
        "FormalCharge": formal_charge,
        "QED": QED.qed(mol),
    }
    return {name: float(values[name]) for name in DESCRIPTOR_NAMES}


def morgan_fingerprint(mol: Chem.Mol, n_bits: int = 2048, radius: int = 2) -> List[int]:
    try:
        from rdkit.Chem import rdFingerprintGenerator

        generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
        fp = generator.GetFingerprint(mol)
    except Exception:
        fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr.astype(int).tolist()


def maccs_fingerprint(mol: Chem.Mol) -> List[int]:
    fp = MACCSkeys.GenMACCSKeys(mol)
    arr = np.zeros((fp.GetNumBits(),), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr.astype(int).tolist()


def tanimoto_from_bits(bits_a: List[int], bits_b: List[int]) -> float:
    a = np.asarray(bits_a, dtype=bool)
    b = np.asarray(bits_b, dtype=bool)
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 0.0
    return float(np.logical_and(a, b).sum() / union)


def descriptor_vector(descriptors: Dict[str, float]) -> List[float]:
    return [float(descriptors.get(name, 0.0)) for name in DESCRIPTOR_NAMES]

