from __future__ import annotations

from agentm.features.visual_semantic import VisualSemanticExtractor
from agentm.schema import MoleculeRecord, VisualSemanticBundle
from agentm.tools.descriptors import compute_rdkit_descriptors
from agentm.tools.smiles import mol_from_smiles


class VisualSemanticAgent:
    def __init__(
        self,
        extractor: VisualSemanticExtractor | None = None,
        use_mllm_visual_semantic: bool = False,
        mllm_profile: str | None = None,
    ):
        self.extractor = extractor or VisualSemanticExtractor(
            use_mllm=use_mllm_visual_semantic,
            mllm_profile=mllm_profile,
        )

    def describe(self, record: MoleculeRecord, task_type: str, image_path: str | None = None) -> VisualSemanticBundle:
        mol = mol_from_smiles(record.canonical_smiles or record.smiles or "")
        if mol is None:
            raise ValueError("VisualSemanticAgent requires a valid molecule.")
        descriptors = compute_rdkit_descriptors(mol)
        return self.extractor.extract(
            mol,
            descriptors,
            task_type=task_type,
            image_path=image_path,
            smiles=record.canonical_smiles or record.smiles,
        )
