"""Bounded, request-only contracts for same-molecule endpoint additions.

No benchmark gold is accepted. Unknown or contradictory addition requests fail
closed; other kinds of request retain their existing routing behavior.
"""
from __future__ import annotations

import re
from typing import Any, Mapping
from rdkit import Chem, rdBase
from agentm.runtime.resources import DATASET_ALIASES

_DATASETS = (
    'BBB_Martins', 'hERG', 'DILI', 'HIA_Hou', 'Pgp_Broccatelli', 'AMES',
    'ClinTox', 'CYP1A2_Veith', 'CYP2C9_Veith', 'CYP2C19_Veith', 'CYP2D6_Veith',
    'CYP3A4_Veith', 'Lipophilicity_AstraZeneca', 'HydrationFreeEnergy_FreeSolv',
    'Solubility_AqSolDB', 'Caco2_Wang', 'DDI', 'Tox21', 'SIDER', 'BACE', 'HIV',
)
ENDPOINT_IDENTITIES = {s.casefold(): s for s in _DATASETS}
ENDPOINT_IDENTITIES.update(DATASET_ALIASES)
ENDPOINT_IDENTITIES.update({'hia':'HIA_Hou', 'pgp':'Pgp_Broccatelli', 'caco2':'Caco2_Wang',
    **{s:'CYP'+s[3:].upper()+'_Veith' for s in ('cyp1a2','cyp2c9','cyp2c19','cyp2d6','cyp3a4')}})


def endpoint_identity(value: Any) -> str | None:
    """Explicit aliases only: absent/unknown names must not select a default."""
    return ENDPOINT_IDENTITIES.get(value.strip().casefold()) if isinstance(value, str) else None


def canonical_structure(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    with rdBase.BlockLogs():
        mol = Chem.MolFromSmiles(value)
    return Chem.MolToSmiles(mol, isomericSmiles=True) if mol is not None else None


def endpoint_addition_contract(*, request: str, task_type: str | None,
                               dataset_name: str | None,
                               previous: Mapping[str, Any] | None) -> dict | None:
    # Deliberately bounded grammar, not a permissive NLP parser. Unsupported
    # additions must not silently degrade to single-endpoint execution.
    if not re.search(r'\badditionally\s+evaluate\b', request, re.I):
        return None
    match = re.fullmatch(r'For the same molecule\s+([^,\s]+),\s*additionally evaluate\s+(.+?)\.?', request.strip(), re.I)
    if not match or task_type != 'admet_prediction' or not previous:
        raise ValueError('endpoint_addition_requires_explicit_same_molecule_admet_history')
    molecule, terms = match.groups()
    structure = canonical_structure(molecule)
    prior_objects = previous.get('molecules') or []
    if not structure or len(prior_objects) != 1 or not isinstance(prior_objects[0], Mapping):
        raise ValueError('endpoint_addition_requires_one_valid_scientific_object')
    prior = prior_objects[0]
    # A resolver may annotate a locally normalized identity. The explicit
    # repeated input must match that same previously resolved input/structure.
    prior_keys = {canonical_structure(prior.get(k)) for k in ('original_input','smiles','canonical_smiles')}
    if structure not in prior_keys or prior.get('is_valid') is False:
        raise ValueError('endpoint_addition_object_changed')
    names = re.split(r'\s*(?:,|\band\b)\s*', terms.rstrip('.'), flags=re.I)
    requested = [endpoint_identity(n) for n in names]
    if not requested or any(n is None or n == 'DDI' for n in requested):
        raise ValueError('endpoint_addition_unknown_or_non_admet_endpoint')
    root = endpoint_identity(dataset_name)
    if root not in requested:
        raise ValueError('endpoint_addition_root_not_requested')
    retained = []
    if previous.get('task_type') != 'admet_prediction':
        raise ValueError('endpoint_addition_history_task_mismatch')
    for pred in previous.get('predictions') or []:
        ep = endpoint_identity(pred.get('dataset_name'))
        if pred.get('task_type') != 'admet_prediction' or not ep or ep == 'DDI':
            raise ValueError('endpoint_addition_history_endpoint_invalid')
        scoped = (pred.get('details') or {}).get('open_task_molecules')
        if scoped is not None and (len(scoped) != 1 or canonical_structure(scoped[0]) not in prior_keys - {None}):
            raise ValueError('endpoint_addition_history_prediction_object_mismatch')
        if ep not in retained:
            retained.append(ep)
    if not retained:
        raise ValueError('endpoint_addition_history_has_no_prediction')
    endpoints = list(dict.fromkeys(retained + requested))
    return {
        'schema':'agentm_endpoint_addition_contract',
        'open_task_type':'multi_endpoint_admet', 'request':request,
        'root_task_type':'admet_prediction', 'root_dataset_name':root,
        'scientific_objects':[molecule], 'retained_endpoints':retained,
        'requested_additional_endpoints':list(dict.fromkeys(requested)),
        'history_policy':'recompute_all_endpoints_no_prediction_or_citation_copy',
        'subtasks':[{'subtask_id':f'single::{i+1}::{ep}', 'kind':'single',
                     'task_type':'admet_prediction', 'dataset_name':ep,
                     'molecules':[molecule], 'evidence_scope':'single_drug'}
                    for i,ep in enumerate(endpoints)],
    }
