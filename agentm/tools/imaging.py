from __future__ import annotations

from pathlib import Path
from typing import Tuple

from rdkit import Chem
from rdkit.Chem import Draw

from agentm.utils.io import ensure_dir


def molecule_image_path(canonical_smiles: str, output_dir: str | Path) -> Path:
    safe = "".join(ch if ch.isalnum() else "_" for ch in canonical_smiles)[:80]
    return Path(output_dir) / f"{safe or 'molecule'}.png"


def generate_molecule_image(
    mol: Chem.Mol,
    canonical_smiles: str,
    output_dir: str | Path,
    size: Tuple[int, int] = (224, 224),
) -> str:
    ensure_dir(output_dir)
    path = molecule_image_path(canonical_smiles, output_dir)
    image = Draw.MolToImage(mol, size=size, kekulize=True)
    image.save(path)
    return str(path)

