from __future__ import annotations

from pathlib import Path

from agentm.features.imagemol import ImageMolEncoder, load_imagemol_encoder
from agentm.schema import ImageFeatureBundle, MoleculeRecord
from agentm.tools.descriptors import compute_rdkit_descriptors
from agentm.tools.imaging import generate_molecule_image
from agentm.tools.smiles import mol_from_smiles
from agentm.utils.io import ensure_dir


class ImageRepresentationAgent:
    def __init__(
        self,
        output_dir: str | Path = "outputs/images",
        encoder: ImageMolEncoder | None = None,
        imagemol_checkpoint_path: str | None = None,
    ):
        self.output_dir = ensure_dir(output_dir)
        self.encoder = encoder or load_imagemol_encoder(imagemol_checkpoint_path)

    def represent(self, record: MoleculeRecord) -> ImageFeatureBundle:
        mol = mol_from_smiles(record.canonical_smiles or record.smiles or "")
        if mol is None:
            raise ValueError("ImageRepresentationAgent requires a valid molecule.")
        descriptors = compute_rdkit_descriptors(mol)
        image_path = generate_molecule_image(mol, record.canonical_smiles or "molecule", self.output_dir)
        embedding = self.encoder.encode(image_path, metadata=descriptors)
        grad_cam_path = self.encoder.grad_cam(image_path)
        return ImageFeatureBundle(
            image_path=image_path,
            embedding=embedding,
            encoder_name=self.encoder.encoder_name,
            checkpoint_path=self.encoder.checkpoint_path,
            grad_cam_path=grad_cam_path,
        )
