"""Opt-in structured target checks; legacy evaluation is returned unchanged.

These checks concern output presence and active object/endpoint state, not label
accuracy, independent evidence support or full claim/citation correctness.
"""
from __future__ import annotations

import re
from typing import Any, Mapping
from agentm.runtime.task_contracts import canonical_structure, endpoint_identity
from .offline_evaluator import evaluate_open_task, evaluate_multi_turn
from .run_matrix import _open_task_contract

PROTOCOL = 'agentm_structured_quality'


def object_identity(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    # Names are compared explicitly, never treated as equivalent to an
    # unrelated structure just because a display name happens to match.
    return canonical_structure(value) or 'name:' + value.strip().casefold()


def active_objects(payload: Mapping) -> set[str]:
    return {object_identity(m.get('canonical_smiles') or m.get('smiles') or m.get('original_input'))
            for m in payload.get('molecules', []) if isinstance(m, Mapping)} - {None}


def prediction_matches(pred: Mapping, target: Mapping, payload: Mapping, *, scoped: bool) -> bool:
    ep = endpoint_identity(target.get('dataset_name'))
    if not ep or pred.get('task_type') != target.get('task_type') or endpoint_identity(pred.get('dataset_name')) != ep:
        return False
    if not any(pred.get(k) is not None for k in ('label', 'score', 'probability')):
        return False
    expected = {object_identity(x) for x in target.get('molecules', [])}
    if not expected or None in expected:
        return False
    details = pred.get('details') or {}
    if scoped and details.get('open_task_subtask_id') != target.get('subtask_id'):
        return False
    if 'open_task_molecules' in details:
        actual = {object_identity(x) for x in details['open_task_molecules']}
    else:
        # An unscoped multi-output object binding is ambiguous, not success.
        if scoped or len(payload.get('predictions', [])) != 1:
            return False
        actual = active_objects(payload)
    return None not in actual and actual == expected


def target_present(target: Mapping, context: Mapping, report: Mapping, *, scoped: bool) -> bool:
    return all(any(prediction_matches(p, target, payload, scoped=scoped)
                   for p in payload.get('predictions', []) if isinstance(p, Mapping))
               for payload in (context, report))


def evaluate_open_task(*, case: Mapping, context: Mapping, report: Mapping) -> dict:
    legacy = evaluate_open_task(case=case, context=context, report=report)
    contract = _open_task_contract(case) or {}
    details = dict(legacy['objective_details'])
    bindings = {}
    for objective in details:
        if not objective.startswith('predict:'):
            continue
        tokens = objective.split(':')[1:]
        endpoint = endpoint_identity(tokens[-1])
        targets = [s for s in contract.get('subtasks', []) if endpoint and endpoint_identity(s.get('dataset_name')) == endpoint]
        if len(tokens) > 1:
            obj = object_identity(':'.join(tokens[:-1]))
            targets = [s for s in targets if obj in {object_identity(x) for x in s['molecules']}]
        details[objective] = bool(targets) and all(target_present(t,context,report,scoped=True) for t in targets)
        bindings[objective] = [t['subtask_id'] for t in targets]
    return {'protocol':PROTOCOL, 'legacy':legacy, 'objective_details':details,
            'objective_coverage':sum(details.values())/len(details) if details else 0.0,
            'expected_subtask_bindings':bindings,
            'nonprediction_objectives_basis':'unchanged_legacy_checks_not_full_semantic_validation',
            'scope':'structured_prediction_presence_not_scientific_accuracy', 'formal_statistics_eligible':False}


def _values(value):
    if isinstance(value, Mapping):
        for item in value.values():
            yield from _values(item)
    elif isinstance(value, list):
        for item in value:
            yield from _values(item)
    elif isinstance(value, str):
        yield value


def evaluate_multi_turn(*, case: Mapping, turn_contexts: list[Mapping], final_report: Mapping,
                           identity_map: Mapping[str, str] | None = None) -> dict:
    legacy = evaluate_multi_turn(case=case,turn_contexts=turn_contexts,final_report=final_report)
    context = turn_contexts[-1] if turn_contexts else {}
    previous = turn_contexts[-2] if len(turn_contexts) > 1 else {}
    gold = case['gold_plan']; kind = case.get('multi_turn_type')
    # Name-to-structure mappings must come from a separately audited local
    # registry, not the evaluated report's own display-name assertions.
    declared = gold.get('expected_final_scientific_objects', [])
    identities = identity_map or {}
    expected_objects = {canonical_structure(x) or canonical_structure(identities.get(str(x).casefold())) for x in declared}
    objects_ok = bool(expected_objects) and None not in expected_objects and all(
        active_objects(p) == expected_objects for p in (context,final_report))
    # Output is explicitly narrower than a global state/citation correctness score.
    result = {'protocol':PROTOCOL,'legacy':legacy,'tool_completion_coverage':legacy['objective_coverage'],
              'expected_objects_present':objects_ok, 'scientific_target_coverage':None,
              'evaluation_valid':bool(expected_objects) and None not in expected_objects,
              'active_object_endpoint_consistency':None, 'citation_quality':'not_regraded',
              'formal_statistics_eligible':False}
    if not result['evaluation_valid']:
        result['not_evaluable_reason']='missing_independent_expected_object_structure_mapping'
        return result
    if kind == 'add_two_endpoints':
        endpoints = gold.get('turn_level_expected_state', [{}])[-1].get('endpoints', [])
        targets = [{'task_type':'admet_prediction','dataset_name':e,
                    'molecules':gold.get('expected_final_scientific_objects', [])} for e in endpoints]
        hits = {t['dataset_name']:target_present(t,context,final_report,scoped=False) for t in targets}
        result.update(target_details=hits, scientific_target_coverage=sum(hits.values())/len(hits) if hits else 0.0,
                      active_object_endpoint_consistency=objects_ok and bool(hits) and all(hits.values()))
        return result
    active = {k:context.get(k) for k in ('molecules','predictions','evidence','citation_map','reasoning')}
    active['final_report'] = {k:final_report.get(k) for k in ('molecules','predictions','evidence','citation_map','explanation')}
    value_strings = list(_values(active))
    same_entity = (kind == 'resolve_specific_structure'
                   and case.get('task_semantics') == 'same_entity_name_to_structure_confirmation_not_proven_ambiguity_resolution'
                   and objects_ok and active_objects(previous) == expected_objects)
    previous_names = {str(previous.get('request','')).strip().casefold()}
    previous_names.update(str(m.get('name','')).strip().casefold() for m in previous.get('molecules', []))
    checks = {}
    for term in gold.get('stale_records_to_remove', []):
        ep = endpoint_identity(term)
        if same_entity and str(term).casefold() in previous_names:
            checks[term] = {'removed_or_compatible':True,'basis':'same_entity_structure_confirmed_alias_allowed'}
        elif ep:
            absent = all(endpoint_identity(p.get('dataset_name')) != ep
                         for payload in (context,final_report) for p in payload.get('predictions', []))
            checks[term] = {'removed_or_compatible':absent,'basis':'active_prediction_endpoint_not_descriptor_word'}
        else:
            identity = object_identity(term)
            in_objects = any(identity in active_objects(p) for p in (context,final_report))
            pattern = re.compile(r'(?<!\w)' + re.escape(str(term)) + r'(?!\w)', re.I)
            in_values = any(pattern.search(s) for s in value_strings)
            checks[term] = {'removed_or_compatible':not in_objects and not in_values,
                            'basis':'active_objects_and_value_text_excluding_keys_and_history'}
    result.update(stale_record_checks=checks,
                  active_object_endpoint_consistency=objects_ok and all(x['removed_or_compatible'] for x in checks.values()),
                  state_scope='structured_active_objects_endpoints_and_conservative_old_object_text_not_full_claim_semantics')
    return result
