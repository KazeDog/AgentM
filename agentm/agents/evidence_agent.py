from __future__ import annotations

from typing import Any, Dict, List, Optional

from agentm.schema import DDIPairFeatureBundle, SingleMoleculeFeatureBundle
from agentm.tools.evidence import local_evidence_for_ddi, local_evidence_for_molecule
from agentm.tools.external_evidence import ExternalEvidenceClient


class EvidenceAgent:
    def __init__(self, *, use_external: bool = False, external_client: ExternalEvidenceClient | None = None):
        self.use_external = bool(use_external)
        self.external_client = external_client or ExternalEvidenceClient(enabled=use_external)

    def retrieve_single(self, bundle: SingleMoleculeFeatureBundle, *, dataset_name: Optional[str] = None) -> List[Dict[str, Any]]:
        evidence = local_evidence_for_molecule(bundle.rdkit_descriptors, bundle.llm_rule_features.values)
        if self.use_external:
            evidence.extend(
                self.external_client.retrieve_single(
                    self._external_query_name(bundle),
                    dataset_name=dataset_name,
                )
            )
        return evidence

    def retrieve_ddi(self, pair: DDIPairFeatureBundle) -> List[Dict[str, Any]]:
        evidence: List[Dict[str, Any]] = []
        evidence.extend(self.retrieve_single(pair.drug_a_features, dataset_name="DDI"))
        evidence.extend(self.retrieve_single(pair.drug_b_features, dataset_name="DDI"))
        evidence.extend(
            local_evidence_for_ddi(
                pair.drug_a_features.molecule.name,
                pair.drug_b_features.molecule.name,
            )
        )
        evidence.append({"source": "feature_similarity", "claim": "Pair fingerprint similarity was computed.", "value": pair.fingerprint_similarity})
        if self.use_external:
            evidence.extend(
                self.external_client.retrieve_ddi(
                    pair.drug_a_features.molecule.name,
                    pair.drug_b_features.molecule.name,
                )
            )
        return evidence

    def _external_query_name(self, bundle: SingleMoleculeFeatureBundle) -> str | None:
        molecule = bundle.molecule
        for value in [molecule.name, molecule.original_input]:
            text = (value or "").strip()
            if text and not _looks_like_smiles(text):
                return text
        return None


def _looks_like_smiles(text: str) -> bool:
    smiles_chars = set("BCNOFPSIHKclbrcnopsh[]=#()@+-\\/1234567890.%:")
    stripped = text.strip()
    return bool(stripped) and all(ch in smiles_chars for ch in stripped)
