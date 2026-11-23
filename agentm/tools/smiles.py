from __future__ import annotations

from typing import Optional, Tuple

from rdkit import Chem, rdBase


def _contains_cjk(text: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in text)


def mol_from_smiles(smiles: str) -> Optional[Chem.Mol]:
    if not smiles:
        return None
    with rdBase.BlockLogs():
        return Chem.MolFromSmiles(str(smiles).strip())


def is_valid_smiles(smiles: str) -> bool:
    return mol_from_smiles(smiles) is not None


def standardize_smiles(smiles: str) -> Tuple[Optional[str], Optional[str]]:
    mol = mol_from_smiles(smiles)
    if mol is None:
        return None, "Invalid SMILES"
    try:
        Chem.SanitizeMol(mol)
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True), None
    except Exception as exc:
        return None, f"RDKit sanitize failed: {exc}"


def likely_smiles(text: str) -> bool:
    stripped = (text or "").strip()
    if not stripped:
        return False
    if _contains_cjk(stripped):
        return False
    smiles_chars = set("BCNOFPSIHKclbrcnopsh[]=#()@+-\\/1234567890.%:")
    if any(ch not in smiles_chars for ch in stripped):
        return False
    if stripped.isalpha():
        smiles_letters = set("BCNOFPSIclbrH")
        if any(ch not in smiles_letters for ch in stripped):
            return False
    if is_valid_smiles(stripped):
        return True
    return len(stripped) > 2 and sum(ch in smiles_chars for ch in stripped) / len(stripped) > 0.65
