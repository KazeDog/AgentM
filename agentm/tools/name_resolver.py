from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

import requests

from agentm.tools.smiles import standardize_smiles


NETWORK_LOOKUP_TRUE_VALUES = {"1", "true", "yes", "on"}
NETWORK_LOOKUP_FALSE_VALUES = {"0", "false", "no", "off"}
DEFAULT_PUBCHEM_TIMEOUT_SECONDS = 15

LOCAL_NAME_TO_SMILES: Dict[str, str] = {
    "acetaminophen": "CC(=O)NC1=CC=C(O)C=C1",
    "paracetamol": "CC(=O)NC1=CC=C(O)C=C1",
    "aspirin": "CC(=O)OC1=CC=CC=C1C(=O)O",
    "caffeine": "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
    "ibuprofen": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
    "naproxen": "COC1=CC2=C(C=C1)C=C(C=C2)C(C)C(=O)O",
    "warfarin": "CC(=O)CC(C1=CC=CC=C1)C2=C(O)C3=CC=CC=C3OC2=O",
    "simvastatin": "CCC(C)(C)C(=O)OC1CC(C=C2C1C(C(C=C2)C)CCC3CC(CC(=O)O3)O)C",
    "ketoconazole": "CC(=O)N1CCN(CC1)C2=CC=C(OC3COC(CN4C=CN=C4)(O3)C5=C(Cl)C=C(Cl)C=C5)C=C2",
    "erythromycin": "CCC1C(C(C(C(O1)OC2CC(C(C(O2)C)O)(C)OC)C)OC3C(C(C(C(O3)C)N(C)C)O)C)O)C(C)C(=O)C(C)C(C(C)C(C(=O)O1)C)O",
    "clopidogrel": "COC1=CC=CC=C1C(C2=CC=CS2)N3CCC(C3)C(=O)OC",
    "omeprazole": "COC1=CC2=C(C=C1)N=C(N2)S(=O)CC3=NC=C(C(=C3C)OC)C",
    "quinidine": "COC1=CC2=C(C=CN=C2C=C1)[C@@H](O)C[C@H]3CC[N@]4C[C@@H]3C=C[C@H]4C",
    "metformin": "CNC(=N)N=C(N)N",
    "midazolam": "CC1=NOC2=C1C(=O)N(C3=CC=CC=C3Cl)C4=C2C=C(C=C4)Cl",
    "propranolol": "CC(C)NCC(O)COc1cccc2ccccc12",
    "naltrexone": "O=C1CC[C@@]2(O)[C@H]3Cc4ccc(O)c5c4[C@@]2(CCN3CC2CC2)[C@H]1O5",
    "abacavir": "Nc1nc(NC2CC2)c2ncn([C@@H]3C=C[C@H](CO)C3)c2n1",
    "dolutegravir": "Fc1ccc(c(c1)F)CNC(=O)c1cn2C[C@@H]3OCC[C@H](N3C(=O)c2c(c1=O)O)C",
    "magaldrate": "O.[OH-].[OH-].[OH-].[OH-].[OH-].[OH-].[OH-].[OH-].[OH-].[OH-].[OH-].[OH-].[OH-].[OH-].[OH-].[OH-].[OH-].[OH-].[OH-].[OH-].[OH-].[OH-].[OH-].[OH-].[OH-].[OH-].[OH-].[OH-].[OH-].[OH-].[OH-].[O-]S(=O)(=O)[O-].[O-]S(=O)(=O)[O-].[Mg+2].[Mg+2].[Mg+2].[Mg+2].[Mg+2].[Mg+2].[Mg+2].[Mg+2].[Mg+2].[Mg+2].[Al+3].[Al+3].[Al+3].[Al+3].[Al+3]",
    "rosuvastatin": "CC(C)C1=NC(=NC(C2=CC=C(F)C=C2)=C1\\C=C\\[C@@H](O)C[C@@H](O)CC(=O)O)N(C)S(C)(=O)=O",
    "paclitaxel": "CC1=C2[C@@]([C@]([C@H]([C@@H]3[C@]4([C@H](OC4)C[C@@H]([C@]3(C(=O)[C@@H]2OC(=O)C)C)O)OC(=O)C)OC(=O)c5ccccc5)(C[C@@H]1OC(=O)[C@H](O)[C@@H](NC(=O)c6ccccc6)c7ccccc7)O)(C)C",
}


def network_lookup_enabled(default: bool = True) -> bool:
    value = os.getenv("AGENTM_ENABLE_NETWORK_LOOKUP")
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in NETWORK_LOOKUP_TRUE_VALUES:
        return True
    if normalized in NETWORK_LOOKUP_FALSE_VALUES:
        return False
    return default


def resolve_name_to_smiles(name: str, allow_network: Optional[bool] = None) -> Tuple[Optional[str], str]:
    key = (name or "").strip().lower()
    if not key:
        return None, "empty_name"
    if key in LOCAL_NAME_TO_SMILES:
        canonical, err = standardize_smiles(LOCAL_NAME_TO_SMILES[key])
        return canonical, "local_dictionary" if not err else f"local_dictionary_error:{err}"

    if allow_network is None:
        allow_network = network_lookup_enabled(default=True)
    if not allow_network:
        return None, "not_found_local_network_disabled"

    try:
        timeout = float(os.getenv("AGENTM_PUBCHEM_TIMEOUT", str(DEFAULT_PUBCHEM_TIMEOUT_SECONDS)))
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{requests.utils.quote(key)}/property/CanonicalSMILES/JSON"
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        data = response.json()
        props = data.get("PropertyTable", {}).get("Properties", [])
        if not props:
            return None, "pubchem_no_result"
        canonical, err = standardize_smiles(props[0].get("CanonicalSMILES", ""))
        return canonical, "pubchem" if not err else f"pubchem_error:{err}"
    except Exception as exc:
        return None, f"pubchem_lookup_failed:{exc}"
