from __future__ import annotations

from typing import Any, Dict, List, Tuple


KNOWN_DDI_HINTS: Dict[Tuple[str, str], Dict[str, Any]] = {
    ("simvastatin", "ketoconazole"): {
        "interaction_type": "CYP3A4 inhibition",
        "mechanism": "Ketoconazole is a strong CYP3A4 inhibitor and may increase simvastatin exposure.",
        "severity": "high",
    },
    ("warfarin", "aspirin"): {
        "interaction_type": "bleeding risk",
        "mechanism": "Both drugs can increase bleeding risk through anticoagulant or antiplatelet effects.",
        "severity": "high",
    },
    ("clopidogrel", "omeprazole"): {
        "interaction_type": "CYP2C19 inhibition",
        "mechanism": "Omeprazole can inhibit CYP2C19 and may reduce clopidogrel activation.",
        "severity": "moderate",
    },
    ("dolutegravir", "magaldrate"): {
        "interaction_type": "polyvalent-cation chelation / absorption reduction",
        "mechanism": "Magaldrate provides aluminum and magnesium cations that can chelate dolutegravir and reduce gastrointestinal absorption if taken together.",
        "severity": "high",
    },
}


def local_evidence_for_molecule(descriptors: Dict[str, float], rule_hits: Dict[str, float]) -> List[Dict[str, Any]]:
    evidence: List[Dict[str, Any]] = []
    if descriptors.get("MolWt", 0.0) > 500:
        evidence.append({"source": "rdkit_descriptor", "claim": "High molecular weight may reduce permeability.", "value": descriptors["MolWt"]})
    if descriptors.get("TPSA", 0.0) > 140:
        evidence.append({"source": "rdkit_descriptor", "claim": "High TPSA is commonly associated with poor passive permeability.", "value": descriptors["TPSA"]})
    if descriptors.get("MolLogP", 0.0) > 5:
        evidence.append({"source": "rdkit_descriptor", "claim": "High logP can increase lipophilicity-driven ADMET risk.", "value": descriptors["MolLogP"]})
    for name, value in rule_hits.items():
        if value:
            evidence.append({"source": "llm_rule_feature", "claim": f"Rule matched: {name}", "value": value})
    return evidence


def local_evidence_for_ddi(name_a: str | None, name_b: str | None) -> List[Dict[str, Any]]:
    if not name_a or not name_b:
        return []
    key = tuple(sorted((name_a.lower(), name_b.lower())))
    for pair, info in KNOWN_DDI_HINTS.items():
        if tuple(sorted(pair)) == key:
            return [{"source": "local_ddi_hint", **info}]
    return []
