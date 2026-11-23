from __future__ import annotations

from agentm.features.smiles import SmilesEncoder, load_smiles_encoder
from agentm.schema import MoleculeRecord, SmilesFeatureBundle


class SmilesRepresentationAgent:
    """Generate a SMILES foundation-model embedding for one molecule."""

    def __init__(
        self,
        encoder: SmilesEncoder | None = None,
        *,
        use_smiles_encoder: bool = True,
        model_name: str | None = None,
        device: str = "auto",
        pooling: str = "cls",
        local_files_only: bool = True,
    ):
        self.encoder = encoder or load_smiles_encoder(
            model_name=model_name,
            use_smiles_encoder=use_smiles_encoder,
            device=device,
            pooling=pooling,
            local_files_only=local_files_only,
        )

    def represent(self, record: MoleculeRecord) -> SmilesFeatureBundle:
        canonical = record.canonical_smiles or record.smiles or ""
        if self.encoder is None:
            raise ValueError("SMILES encoder is disabled.")
        embedding = self.encoder.encode(canonical)
        return SmilesFeatureBundle(
            smiles=canonical,
            embedding=embedding,
            encoder_name=self.encoder.encoder_name,
            model_name=self.encoder.model_name,
            pooling=self.encoder.pooling,
            metadata=dict(self.encoder.metadata),
        )
