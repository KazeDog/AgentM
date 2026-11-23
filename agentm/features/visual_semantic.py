from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from rdkit import Chem

from agentm.mllm.provider import MLLMClient, build_mllm_client
from agentm.schema import VisualSemanticBundle
from agentm.features.visual_semantic_cache import VisualSemanticCache


FUNCTIONAL_GROUPS: Dict[str, str] = {
    "carboxylic_acid": "C(=O)[OH]",
    "amide": "C(=O)N",
    "ester": "C(=O)O[#6]",
    "amine": "[NX3;H2,H1,H0;!$(NC=O)]",
    "phenol": "c[OH]",
    "ether": "[OD2]([#6])[#6]",
    "nitro": "[NX3](=O)=O",
    "sulfonamide": "S(=O)(=O)N",
    "halogen": "[F,Cl,Br,I]",
}


class VisualSemanticExtractor:
    def __init__(
        self,
        provider: str = "fallback_rdkit",
        embedding_dim: int = 64,
        mllm_client: Optional[MLLMClient] = None,
        mllm_profile: Optional[str] = None,
        use_mllm: bool = False,
        cache_dir: Optional[str | Path] = None,
        cache_read: bool = True,
        cache_write: bool = True,
        cache_only: bool = False,
    ):
        self.provider = provider
        self.embedding_dim = embedding_dim
        self.use_mllm = use_mllm
        self.mllm_client = mllm_client or (build_mllm_client(profile_name=mllm_profile) if use_mllm else None)
        self.cache_read = cache_read
        self.cache_write = cache_write
        self.cache_only = cache_only
        cache_profile = mllm_profile
        if self.mllm_client is not None:
            cache_profile = self.mllm_client.config.profile
        self.cache = VisualSemanticCache(cache_dir, profile=cache_profile) if cache_dir else None

    def extract(
        self,
        mol: Chem.Mol,
        descriptors: Dict[str, float],
        task_type: str,
        image_path: Optional[str | Path] = None,
        smiles: Optional[str] = None,
        dataset_name: Optional[str] = None,
    ) -> VisualSemanticBundle:
        canonical_smiles = smiles or Chem.MolToSmiles(mol, canonical=True)
        if self.use_mllm and self.cache is not None and self.cache_read:
            cached = self.cache.load(smiles=canonical_smiles, task_type=task_type, dataset_name=dataset_name)
            if cached is not None:
                return cached
            if self.cache_only:
                fallback = self._extract_fallback(mol, descriptors, task_type)
                fallback.provider = "fallback_rdkit_after_mllm_cache_miss"
                fallback.metadata.update(
                    {
                        "mllm_cache_miss": True,
                        "cache_only": True,
                        "mllm_profile": self.mllm_client.config.profile if self.mllm_client else None,
                        "mllm_model": self.mllm_client.config.model if self.mllm_client else None,
                    }
                )
                return fallback
        if self.use_mllm and self.mllm_client is not None and self.mllm_client.available and image_path:
            try:
                bundle = self._extract_mllm(
                    image_path=Path(image_path),
                    smiles=canonical_smiles,
                    task_type=task_type,
                    dataset_name=dataset_name,
                    descriptors=descriptors,
                )
                if self.cache is not None and self.cache_write:
                    cache_path = self.cache.store(
                        bundle,
                        smiles=canonical_smiles,
                        task_type=task_type,
                        dataset_name=dataset_name,
                        image_path=image_path,
                    )
                    bundle.metadata["cache_path"] = str(cache_path)
                    bundle.metadata["cache_write"] = True
                return bundle
            except Exception as exc:
                fallback = self._extract_fallback(mol, descriptors, task_type)
                fallback.provider = "fallback_rdkit_after_mllm_error"
                fallback.metadata["mllm_error"] = str(exc)
                if self.mllm_client is not None:
                    fallback.metadata["mllm_profile"] = self.mllm_client.config.profile
                    fallback.metadata["mllm_model"] = self.mllm_client.config.model
                return fallback
        return self._extract_fallback(mol, descriptors, task_type)

    def _extract_fallback(self, mol: Chem.Mol, descriptors: Dict[str, float], task_type: str) -> VisualSemanticBundle:
        functional_groups = self._functional_groups(mol)
        ring_systems = self._ring_systems(mol, descriptors)
        hbond_features = {
            "donors": int(descriptors.get("NumHDonors", 0)),
            "acceptors": int(descriptors.get("NumHAcceptors", 0)),
            "tpsa": round(float(descriptors.get("TPSA", 0.0)), 3),
        }
        hydrophobic_regions = self._hydrophobic_regions(descriptors)
        ionizable_centers = self._ionizable_centers(functional_groups, descriptors)
        admet_alerts = self._admet_alerts(functional_groups, descriptors)
        cues = self._task_cues(task_type, functional_groups, descriptors)
        summary = self._summary(functional_groups, ring_systems, hbond_features, admet_alerts)
        embedding = self._embed(summary, functional_groups + ring_systems + admet_alerts + cues)
        return VisualSemanticBundle(
            functional_groups=functional_groups,
            ring_systems=ring_systems,
            hbond_features=hbond_features,
            hydrophobic_regions=hydrophobic_regions,
            ionizable_centers=ionizable_centers,
            admet_alerts=admet_alerts,
            task_relevant_visual_cues=cues,
            summary=summary,
            embedding=embedding,
            provider=self.provider,
            metadata={"source": "rdkit_fallback"},
        )

    def _extract_mllm(
        self,
        *,
        image_path: Path,
        smiles: str,
        task_type: str,
        dataset_name: Optional[str],
        descriptors: Dict[str, float],
    ) -> VisualSemanticBundle:
        assert self.mllm_client is not None
        prompt = self._mllm_prompt(smiles=smiles, task_type=task_type, dataset_name=dataset_name, descriptors=descriptors)
        response = self.mllm_client.image_json(image_path=image_path, prompt=prompt, temperature=0.1)
        parsed = self._parse_mllm_response(response)
        summary = str(parsed.get("summary") or "")
        tokens = (
            parsed["functional_groups"]
            + parsed["ring_systems"]
            + parsed["hydrophobic_regions"]
            + parsed["ionizable_centers"]
            + parsed["admet_alerts"]
            + parsed["task_relevant_visual_cues"]
        )
        embedding = self._embed(summary, tokens)
        return VisualSemanticBundle(
            functional_groups=parsed["functional_groups"],
            ring_systems=parsed["ring_systems"],
            hbond_features=parsed["hbond_features"],
            hydrophobic_regions=parsed["hydrophobic_regions"],
            ionizable_centers=parsed["ionizable_centers"],
            admet_alerts=parsed["admet_alerts"],
            task_relevant_visual_cues=parsed["task_relevant_visual_cues"],
            summary=summary,
            embedding=embedding,
            provider=f"mllm:{self.mllm_client.config.profile}",
            metadata={
                "source": "mllm",
                "mllm_profile": self.mllm_client.config.profile,
                "mllm_provider": self.mllm_client.config.provider,
                "mllm_model": self.mllm_client.config.model,
                "raw_response": response,
            },
        )

    def _mllm_prompt(self, *, smiles: str, task_type: str, dataset_name: Optional[str], descriptors: Dict[str, float]) -> str:
        descriptor_subset = {
            key: round(float(descriptors.get(key, 0.0)), 4)
            for key in ["MolWt", "MolLogP", "TPSA", "NumHDonors", "NumHAcceptors", "NumRotatableBonds", "RingCount", "QED"]
        }
        return f"""
You are a medicinal chemist analyzing an RDKit 2D molecular depiction.

SMILES: {smiles}
Task type: {task_type}
Dataset/task name: {dataset_name or "unknown"}
RDKit descriptor hints: {json.dumps(descriptor_subset, sort_keys=True)}

Return valid JSON only. Do not include markdown fences.
Use concise controlled vocabulary where possible. If uncertain, return an empty list instead of guessing.

Required JSON schema:
{{
  "functional_groups": ["..."],
  "ring_systems": ["..."],
  "hbond_features": {{"donors": 0, "acceptors": 0, "tpsa": 0.0}},
  "hydrophobic_regions": ["..."],
  "ionizable_centers": ["..."],
  "admet_alerts": ["..."],
  "task_relevant_visual_cues": ["..."],
  "summary": "one short sentence"
}}
""".strip()

    def _parse_mllm_response(self, response: str) -> Dict[str, Any]:
        text = response.strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:].strip()
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            text = text[start : end + 1]
        data = json.loads(text)
        return {
            "functional_groups": self._list_of_str(data.get("functional_groups")),
            "ring_systems": self._list_of_str(data.get("ring_systems")),
            "hbond_features": data.get("hbond_features") if isinstance(data.get("hbond_features"), dict) else {},
            "hydrophobic_regions": self._list_of_str(data.get("hydrophobic_regions")),
            "ionizable_centers": self._list_of_str(data.get("ionizable_centers")),
            "admet_alerts": self._list_of_str(data.get("admet_alerts")),
            "task_relevant_visual_cues": self._list_of_str(data.get("task_relevant_visual_cues")),
            "summary": str(data.get("summary") or ""),
        }

    def _list_of_str(self, value: Any) -> List[str]:
        if not isinstance(value, list):
            return []
        return [str(item).strip() for item in value if str(item).strip()]

    def _functional_groups(self, mol: Chem.Mol) -> List[str]:
        groups: List[str] = []
        for name, smarts in FUNCTIONAL_GROUPS.items():
            pattern = Chem.MolFromSmarts(smarts)
            if pattern is not None and mol.HasSubstructMatch(pattern):
                groups.append(name)
        return groups

    def _ring_systems(self, mol: Chem.Mol, descriptors: Dict[str, float]) -> List[str]:
        rings: List[str] = []
        aromatic = int(descriptors.get("NumAromaticRings", 0))
        aliphatic = int(descriptors.get("NumAliphaticRings", 0))
        if aromatic:
            rings.append(f"aromatic_rings:{aromatic}")
        if aliphatic:
            rings.append(f"aliphatic_rings:{aliphatic}")
        if mol.GetRingInfo().NumRings() >= 3:
            rings.append("polycyclic")
        return rings

    def _hydrophobic_regions(self, descriptors: Dict[str, float]) -> List[str]:
        regions: List[str] = []
        if descriptors.get("MolLogP", 0.0) >= 3:
            regions.append("high_logp_hydrophobic")
        if descriptors.get("NumAromaticRings", 0.0) >= 2:
            regions.append("multi_aromatic_hydrophobic")
        return regions

    def _ionizable_centers(self, groups: List[str], descriptors: Dict[str, float]) -> List[str]:
        centers: List[str] = []
        if "amine" in groups:
            centers.append("basic_amine")
        if "carboxylic_acid" in groups or "phenol" in groups:
            centers.append("acidic_center")
        if descriptors.get("FormalCharge", 0.0) != 0:
            centers.append("formal_charge")
        return centers

    def _admet_alerts(self, groups: List[str], descriptors: Dict[str, float]) -> List[str]:
        alerts: List[str] = []
        if descriptors.get("MolWt", 0.0) > 500:
            alerts.append("high_molecular_weight")
        if descriptors.get("MolLogP", 0.0) > 5:
            alerts.append("high_logp")
        if descriptors.get("TPSA", 0.0) > 140:
            alerts.append("high_tpsa")
        if "nitro" in groups:
            alerts.append("nitro_toxicity_alert")
        if descriptors.get("NumHDonors", 0.0) > 5 or descriptors.get("NumHAcceptors", 0.0) > 10:
            alerts.append("lipinski_hbond_alert")
        return alerts

    def _task_cues(self, task_type: str, groups: List[str], descriptors: Dict[str, float]) -> List[str]:
        cues: List[str] = []
        if task_type == "ddi_prediction":
            if descriptors.get("MolLogP", 0.0) > 4:
                cues.append("possible_cyp_binding_lipophilicity")
            if "amine" in groups:
                cues.append("possible_transporter_or_cyp_ionizable_center")
        elif task_type == "admet_prediction":
            cues.extend(self._admet_alerts(groups, descriptors))
        else:
            if descriptors.get("TPSA", 0.0) < 90:
                cues.append("low_polar_surface_area")
            if descriptors.get("QED", 0.0) > 0.5:
                cues.append("drug_like_qed")
        return cues

    def _summary(
        self,
        groups: List[str],
        rings: List[str],
        hbond: Dict[str, float],
        alerts: List[str],
    ) -> str:
        return (
            f"Detected {len(groups)} functional group types ({', '.join(groups) or 'none'}), "
            f"ring systems ({', '.join(rings) or 'none'}), "
            f"H-bond donors={hbond.get('donors', 0)}, acceptors={hbond.get('acceptors', 0)}, "
            f"TPSA={hbond.get('tpsa', 0)}. "
            f"ADMET alerts: {', '.join(alerts) or 'none'}."
        )

    def _embed(self, summary: str, tokens: List[str]) -> List[float]:
        vec = np.zeros(self.embedding_dim, dtype=float)
        for token in tokens + summary.lower().split():
            digest = hashlib.sha1(token.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "big") % self.embedding_dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vec[idx] += sign
        norm = float(np.linalg.norm(vec)) or 1.0
        return (vec / norm).astype(float).tolist()
