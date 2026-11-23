#!/usr/bin/env python
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = ROOT
RESOURCE_ROOT = WORKSPACE_ROOT
for path in [ROOT, RESOURCE_ROOT]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from agentm.tools.name_resolver import LOCAL_NAME_TO_SMILES
from agentm.tools.smiles import standardize_smiles


SOURCE_CASE_FILE = ROOT / "configs" / "reference_cases_50.jsonl"
OUTPUT_CASE_FILE = ROOT / "configs" / "molecular_safety_cases.jsonl"

PREFERRED_SMILES = {
    "metformin": "CN(C)C(=N)N=C(N)N",
    "ketoconazole": "CC(=O)N1CCN(CC1)C2=CC=C(C=C2)OC[C@@H]3CO[C@@](O3)(CN4C=CN=C4)C5=C(C=C(C=C5)Cl)Cl",
    "caffeine": "Cn1cnc2n(C)c(=O)n(C)c(=O)c12",
}

ADMET_ADDITIONS = [
    ("midazolam_cyp3a4", "midazolam", "CYP3A4_Veith", "cyp3a4", "Additional CYP3A4 case for benzodiazepine substrate/inhibition coverage."),
    ("quinidine_pgp", "quinidine", "Pgp_Broccatelli", "pgp", "Additional P-glycoprotein transporter case for antiarrhythmic coverage."),
    ("paclitaxel_pgp", "paclitaxel", "Pgp_Broccatelli", "pgp", "Additional P-glycoprotein transporter case for oncology-drug coverage."),
    ("rosuvastatin_lipophilicity", "rosuvastatin", "Lipophilicity_AstraZeneca", "lipophilicity", "Additional lipophilicity case for statin chemical-space coverage."),
    ("metformin_caco2", "metformin", "Caco2_Wang", "caco2", "Additional Caco-2 permeability case for polar antidiabetic-drug coverage."),
    ("caffeine_hia", "caffeine", "HIA_Hou", "hia", "Additional human intestinal absorption case for xanthine coverage."),
    ("naltrexone_bbbp", "naltrexone", "BBB_Martins", "bbbp", "Additional BBBP case for CNS-relevant opioid-antagonist coverage."),
    ("acetaminophen_ames", "acetaminophen", "AMES", "ames", "Additional Ames mutagenicity case for analgesic safety coverage."),
    ("ketoconazole_herg", "ketoconazole", "hERG", "herg", "Additional hERG cardiac-safety case for azole antifungal coverage."),
    ("clopidogrel_cyp2c19", "clopidogrel", "CYP2C19_Veith", "cyp2c19", "Additional CYP2C19 case for antiplatelet metabolism coverage."),
]

DDI_ADDITIONS = [
    ("naproxen", "warfarin", "anticoagulant/NSAID bleeding-risk pair."),
    ("diclofenac", "warfarin", "anticoagulant/NSAID bleeding-risk pair."),
    ("aspirin", "clopidogrel", "dual antiplatelet/bleeding-risk pair."),
    ("aspirin", "ibuprofen", "NSAID/antiplatelet pharmacodynamic pair."),
    ("clopidogrel", "warfarin", "antiplatelet/anticoagulant bleeding-risk pair."),
    ("clopidogrel", "naproxen", "antiplatelet/NSAID bleeding-risk pair."),
    ("clopidogrel", "ibuprofen", "antiplatelet/NSAID bleeding-risk pair."),
    ("clopidogrel", "diclofenac", "antiplatelet/NSAID bleeding-risk pair."),
    ("aspirin", "naproxen", "NSAID/antiplatelet pharmacodynamic pair."),
    ("aspirin", "diclofenac", "NSAID/antiplatelet pharmacodynamic pair."),
    ("ketoconazole", "midazolam", "CYP3A inhibitor-substrate pair."),
    ("clarithromycin", "midazolam", "CYP3A inhibitor-substrate pair."),
    ("amiodarone", "midazolam", "antiarrhythmic/CYP3A substrate cardiac-risk pair."),
    ("amiodarone", "simvastatin", "antiarrhythmic/statin metabolism-risk pair."),
    ("ketoconazole", "warfarin", "azole/anticoagulant metabolism-risk pair."),
    ("clarithromycin", "warfarin", "macrolide/anticoagulant metabolism-risk pair."),
    ("quinidine", "warfarin", "antiarrhythmic/anticoagulant metabolism-risk pair."),
    ("fluoxetine", "warfarin", "SSRI/anticoagulant bleeding and metabolism-risk pair."),
    ("omeprazole", "warfarin", "PPI/anticoagulant metabolism-risk pair."),
    ("ketoconazole", "cyclosporin A", "azole/immunosuppressant CYP3A interaction pair."),
    ("clarithromycin", "cyclosporin A", "macrolide/immunosuppressant CYP3A interaction pair."),
    ("fluoxetine", "cyclosporin A", "SSRI/immunosuppressant metabolism-risk pair."),
    ("amiodarone", "digoxin", "P-glycoprotein/cardiac glycoside interaction pair."),
    ("quinidine", "digoxin", "P-glycoprotein/cardiac glycoside interaction pair."),
    ("ketoconazole", "digoxin", "P-glycoprotein inhibitor/cardiac glycoside pair."),
    ("clarithromycin", "digoxin", "macrolide/P-glycoprotein cardiac glycoside pair."),
    ("omeprazole", "digoxin", "PPI/cardiac glycoside transporter or absorption-risk pair."),
    ("cyclosporin A", "digoxin", "transporter-mediated cardiac glycoside pair."),
    ("propranolol", "digoxin", "cardiac-rate/cardiac glycoside pharmacodynamic pair."),
    ("metoprolol", "amiodarone", "beta-blocker/antiarrhythmic cardiac-risk pair."),
    ("metoprolol", "quinidine", "CYP2D6 and cardiac-risk pair."),
    ("fluoxetine", "propranolol", "CYP2D6 inhibitor/beta-blocker exposure pair."),
    ("simvastatin", "cyclosporin A", "statin/immunosuppressant transporter-metabolism pair."),
    ("rosuvastatin", "cyclosporin A", "statin/immunosuppressant transporter pair."),
    ("paclitaxel", "ketoconazole", "oncology drug/CYP3A inhibitor pair."),
    ("paclitaxel", "clarithromycin", "oncology drug/macrolide inhibitor pair."),
    ("paclitaxel", "amiodarone", "oncology drug/antiarrhythmic metabolism-risk pair."),
    ("dolutegravir", "magaldrate", "integrase inhibitor/polyvalent-cation chelation pair."),
    ("dolutegravir", "metformin", "transporter-mediated antiretroviral/antidiabetic pair."),
    ("abacavir", "dolutegravir", "antiretroviral coadministration pair."),
]


def main() -> int:
    argparse.ArgumentParser(description="Build the 100 molecular safety cases from the reference cases.").parse_args()
    records = read_jsonl(SOURCE_CASE_FILE)
    output = list(records)
    output.extend(admet_records())
    output.extend(ddi_records(DDI_ADDITIONS, existing=output))
    validate(output)
    OUTPUT_CASE_FILE.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in output) + "\n", encoding="utf-8")
    summary = {
        "output_case_file": str(OUTPUT_CASE_FILE),
        "source_case_file": str(SOURCE_CASE_FILE),
        "n_cases": len(output),
        "category_counts": dict(Counter(row["category"] for row in output)),
        "task_counts": dict(Counter(row["task_type"] for row in output)),
        "dataset_counts": dict(Counter(row["dataset_name"] for row in output)),
        "n_new_admet_cases": len(ADMET_ADDITIONS),
        "n_new_ddi_cases": len(DDI_ADDITIONS),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            payload = json.loads(line)
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def admet_records() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for name, drug, dataset, endpoint, notes in ADMET_ADDITIONS:
        smiles = PREFERRED_SMILES.get(drug, LOCAL_NAME_TO_SMILES[drug])
        canonical, error = standardize_smiles(smiles)
        if error or not canonical:
            raise ValueError(f"Invalid SMILES for {drug}: {error}")
        rows.append(
            {
                "name": name,
                "request": smiles,
                "task_type": "admet_prediction",
                "dataset_name": dataset,
                "category": "admet",
                "expected_endpoint": endpoint,
                "notes": notes,
            }
        )
    return rows


def ddi_records(pairs: Sequence[tuple[str, str, str]], *, existing: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    existing_keys = {
        pair_key(*ddi_drugs_from_request(str(row.get("request") or "")))
        for row in existing
        if row.get("task_type") == "ddi_prediction"
    }
    rows: List[Dict[str, Any]] = []
    for drug_a, drug_b, note in pairs:
        key = pair_key(drug_a, drug_b)
        if key in existing_keys:
            raise ValueError(f"Duplicate DDI pair already present: {drug_a}, {drug_b}")
        existing_keys.add(key)
        rows.append(
            {
                "name": f"{slug(drug_a)}_{slug(drug_b)}_ddi",
                "request": f"{drug_a}, {drug_b}",
                "task_type": "ddi_prediction",
                "dataset_name": "DDI",
                "category": "ddi",
                "expected_endpoint": "ddi",
                "notes": f"Balanced100 DDI addition; {note}",
            }
        )
    return rows


def validate(records: Sequence[Mapping[str, Any]]) -> None:
    if len(records) != 100:
        raise ValueError(f"Expected 100 cases, found {len(records)}")
    counts = Counter(row.get("category") for row in records)
    if counts != {"admet": 50, "ddi": 50}:
        raise ValueError(f"Expected 50 ADMET and 50 DDI cases, found {dict(counts)}")
    names = [str(row.get("name") or "") for row in records]
    duplicates = sorted(item for item, count in Counter(names).items() if count > 1)
    if duplicates:
        raise ValueError(f"Duplicate case names: {duplicates}")
    ddi_keys = [
        pair_key(*ddi_drugs_from_request(str(row.get("request") or "")))
        for row in records
        if row.get("task_type") == "ddi_prediction"
    ]
    duplicate_pairs = sorted(item for item, count in Counter(ddi_keys).items() if count > 1)
    if duplicate_pairs:
        raise ValueError(f"Duplicate DDI pairs: {duplicate_pairs}")
    required = ["name", "request", "task_type", "dataset_name", "category", "expected_endpoint", "notes"]
    missing = [(row.get("name"), key) for row in records for key in required if not row.get(key)]
    if missing:
        raise ValueError(f"Missing required fields: {missing[:10]}")


def ddi_drugs_from_request(request: str) -> tuple[str, str]:
    parts = [part.strip() for part in request.replace(";", ",").replace("+", ",").split(",") if part.strip()]
    if len(parts) < 2:
        return "", ""
    return parts[0], parts[1]


def pair_key(left: str, right: str) -> str:
    return "__".join(sorted([slug(left), slug(right)]))


def slug(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value or "").lower()).strip("_")


if __name__ == "__main__":
    raise SystemExit(main())
