#!/usr/bin/env python
"""Offline-only candidate preparation; deliberately not an execution launcher.

Outputs are append-only. Candidate gold and retrieval applicability still need
independent preflight; generating a JSONL is not acceptance of an experiment.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import random
from unittest.mock import patch

from .common import PROJECT_ROOT, RELIABILITY_ROOT, file_sha256, load_config, load_jsonl, stable_hash, write_jsonl
from .build_stress_manifest import _fault_schedule, PREDICTION_TARGET_FAULTS
from .gold_rules import gold_for_case, validate_stress_case


def entity_key(text):
    from rdkit import Chem
    from rdkit import rdBase
    with rdBase.BlockLogs():
        mol = Chem.MolFromSmiles(str(text))
    return Chem.MolToSmiles(mol, isomericSmiles=True) if mol is not None else ' '.join(str(text).casefold().split())


def object_key(row):
    request = str(row.get('request', ''))
    if row.get('task_type') == 'ddi_prediction':
        return stable_hash(sorted(entity_key(s.strip()) for s in request.split(',')))
    if str(row.get('task_type', '')).startswith('reaction_'):
        request = request.split(' for ', 1)[-1]
    return stable_hash(entity_key(request))


def case_key(row):
    return stable_hash([row.get('task_type'), row.get('dataset_name'), object_key(row)])


def seal(row):
    row['preparation_status'] = 'candidate_not_execution_authorized'
    row['case_hash'] = stable_hash({k: v for k, v in row.items() if k != 'case_hash'})
    return row


def unique(rows, key=case_key):
    seen = set()
    result = []
    for row in rows:
        identity = key(row)
        if identity not in seen:
            seen.add(identity)
            result.append(row)
    return result


def new_reactions(config, originals):
    """Use deterministic, previously unused test rows, never model outcomes."""
    from scripts.run_reaction_benchmark import materialize_case
    original_requests = {object_key(materialize_case(r, config['paths']['reaction_data_root'])) for r in originals}
    seen = set(original_requests)
    result = []
    for task in ('reaction_classification', 'reaction_yield_prediction', 'reaction_condition_retrieval', 'reaction_profile'):
        templates = unique([r for r in originals if r['task_type'] == task],
                           key=lambda r: (r['source_dataset'], r.get('split')))
        # Original cases are spread over the split, including its end. Scan
        # from row zero and exclude their actual objects, not max(index)+1.
        offset = 0
        selected = 0
        for ordinal in range(1000 * len(templates)):
            template = templates[ordinal % len(templates)]
            index = offset + ordinal // len(templates)
            candidate = {**template, 'part': 'test', 'row_index': index,
                         'name': f"scale_{task}_{template['source_dataset']}_{template.get('split', 'test')}_{index:05d}"}
            candidate.pop('request', None)
            try:
                candidate = materialize_case(candidate, config['paths']['reaction_data_root'])
            except IndexError:
                continue
            identity = object_key(candidate)
            if identity in seen:
                continue
            seen.add(identity)
            candidate['selection_protocol'] = 'first_unused_test_rows_excluding_all_original_reaction_objects_no_result_access'
            result.append(candidate)
            selected += 1
            if selected == 20:
                break
        # Record shortages in the report. Repeated rows with the same actual
        # runtime request cannot be relabelled as independent new cases.
    return result


def expanded_stress(named, scientific, reactions, old):
    rng = random.Random(2026)
    excluded = {object_key(r['base_case']) for r in old}
    used = set(excluded)
    result = []
    pools = {}
    for group in ('admet', 'ddi'):
        for pool, source in (('named_direct_evidence', named), ('held_out_prediction_gold', scientific)):
            rows = unique([r for r in source if r.get('category') == group], object_key)
            rng.shuffle(rows)
            pools[group, pool] = rows
    classified = [r for r in reactions if r['task_type'] == 'reaction_classification']
    yields = [r for r in reactions if r['task_type'] == 'reaction_yield_prediction']
    pools['reaction_prediction', 'reaction_benchmark'] = [r for i in range(max(len(classified), len(yields)))
                                                       for block in (classified, yields) for r in block[i:i+1]]
    pools['reaction_evidence', 'reaction_benchmark'] = [r for r in reactions if r['task_type'] in ('reaction_condition_retrieval', 'reaction_profile')]
    missing = []
    for group in ('admet', 'ddi', 'reaction_prediction', 'reaction_evidence'):
        for index, fault in enumerate(_fault_schedule(group)):
            pool = ('held_out_prediction_gold' if fault in PREDICTION_TARGET_FAULTS else 'named_direct_evidence') if group in ('admet', 'ddi') else 'reaction_benchmark'
            base = next((r for r in pools[group, pool] if object_key(r) not in used), None)
            if base is None:
                missing.append({'group': group, 'fault': fault, 'pool': pool})
                continue
            used.add(object_key(base))
            case_id = f'scale_stress_{group}_{index:02d}_{fault}'
            gold = gold_for_case(stress_case_id=case_id, base_case=base, task_group=group, fault=fault, seed=2026 + 10000 + len(result))
            row = {**gold, 'case_id': case_id, 'dataset_group': 'stress', 'base_case': deepcopy(base),
                   'base_case_hash': stable_hash(base), 'base_case_pool': pool, 'primary_fault_count': 1,
                   'requires_successful_base_execution': True, 'stress_protocol': 'scale_reflection_candidate',
                   'primary_analysis': False, 'evidence_applicability': 'pending_zero_token_preflight',
                   'cluster_id': object_key(base)}
            validate_stress_case(row)
            result.append(seal(row))
    return result, missing


def expanded_multi(named, old):
    from .build_multi_turn_manifest import _plan
    from .build_open_task_manifest import ENDPOINT_CYCLE
    admet = unique([r for r in named if r['category'] == 'admet'], object_key)
    old_objects = {entity_key(v) for r in old for v in r['gold_plan']['expected_final_scientific_objects']}
    unused = [r for r in admet if entity_key(r['request']) not in old_objects]
    ddi = [r for r in named if r['category'] == 'ddi']
    old_pairs = {entity_key(t['request']) for r in old for t in r['turns'] if t['task_type'] == 'ddi_prediction'}
    ddi = [r for r in ddi if entity_key(r['request']) not in old_pairs]
    if len(unused) < 10 or len(ddi) < 5:
        raise ValueError('Insufficient new molecular objects for multi-turn candidates')
    rows = []
    for kind in ('add_two_endpoints', 'resolve_specific_structure', 'single_to_ddi', 'withdraw_and_replace'):
        for index in range(5):
            base = unused[index]
            first = {'turn': 1, 'request': base['request'], 'task_type': base['task_type'], 'dataset_name': base['dataset_name']}
            stale = []
            if kind == 'add_two_endpoints':
                additions = [e for e in ENDPOINT_CYCLE if e[0] != base['dataset_name']][:2]
                second = {'turn': 2, 'request': f"For the same molecule {base['request']}, additionally evaluate {additions[0][1]} and {additions[1][1]}.", 'task_type': 'admet_prediction', 'dataset_name': additions[0][0]}
                objectives = [f"retain:{base['expected_endpoint']}"] + [f'add:{e[1]}' for e in additions]
                states = [{'turn': 1, 'endpoints': [base['expected_endpoint']]}, {'turn': 2, 'endpoints': [base['expected_endpoint']] + [e[1] for e in additions]}]
                objects = [base['request']]
                tools = ['resolve_molecule', 'predict_single', 'verify_prediction_grounding']
            elif kind == 'resolve_specific_structure':
                # Name and supplied structure refer to the SAME curated source row.
                first['request'] = base['name'].rsplit('_', 1)[0].replace('_', ' ')
                second = {**first, 'turn': 2, 'request': base['request']}
                objectives = ['replace_ambiguous_entity', 'recompute_prediction', 'refresh_citations']
                states = [{'turn': 1, 'entity': first['request']}, {'turn': 2, 'entity': second['request']}]
                objects = [second['request']]
                stale = [first['request']]
                tools = ['resolve_molecule', 'predict_single', 'verify_citation_grounding']
            elif kind == 'single_to_ddi':
                pair = ddi[index]['request'].split(',')
                first = {'turn': 1, 'request': pair[0].strip(), 'task_type': 'admet_prediction', 'dataset_name': 'hERG'}
                second = {'turn': 2, 'request': ddi[index]['request'], 'task_type': 'ddi_prediction', 'dataset_name': 'DDI'}
                objects = [p.strip() for p in pair]
                objectives = ['retain_single_drug_result', 'add_ddi_prediction', 'separate_pair_evidence']
                states = [{'turn': 1, 'objects': [objects[0]]}, {'turn': 2, 'objects': objects}]
                tools = ['resolve_molecule', 'predict_single', 'predict_ddi', 'verify_citation_grounding']
            else:
                new = unused[index + 5]
                second = {'turn': 2, 'request': new['request'], 'task_type': new['task_type'], 'dataset_name': new['dataset_name'], 'instruction': 'Withdraw the previous target, replace it with this molecule, and retrieve direct endpoint evidence.'}
                objectives = ['withdraw_previous_target', 'replace_scientific_object', 'refresh_prediction', 'refresh_direct_evidence']
                objects = [new['request']]
                stale = [first['request'], base['expected_endpoint']]
                states = [{'turn': 1, 'entity': first['request']}, {'turn': 2, 'entity': new['request'], 'withdrawn': first['request']}]
                tools = ['resolve_molecule', 'predict_single', 'retrieve_elasticsearch_hybrid', 'verify_citation_grounding']
            case_id = f'scale_multi_{kind}_{index:02d}'
            row = {'case_id': case_id, 'name': case_id, 'dataset_group': 'multi_turn', 'multi_turn_type': kind,
                   'turns': [first, second], 'cluster_id': stable_hash([entity_key(t['request']) for t in (first, second)]),
                   'gold_plan': _plan(objectives=objectives, tools=tools, dependencies=[], states=states, stale=stale,
                                      citation_updates=['refresh citations for final objects; retain no withdrawn-object claims'], final_objects=objects),
                   'gold_review_status': 'pending_independent_contract_review'}
            rows.append(seal(row))
    return rows


def prepare(output):
    if output.exists():
        raise FileExistsError(f'Append-only output already exists: {output}')
    config = load_config(RELIABILITY_ROOT / 'configs/base.yaml')
    originals = {g: load_jsonl(RELIABILITY_ROOT / 'manifests' / f'{f}.jsonl') for g, f in
                 [('clean', 'clean'), ('stress', 'stress_controlled'), ('open_task', 'open_tasks'), ('multi_turn', 'multi_turn_tasks')]}
    named = load_jsonl(PROJECT_ROOT / config['paths']['molecular_safety_manifest'])
    scientific = load_jsonl(PROJECT_ROOT / config['paths']['molecular_safety_scientific_manifest'])
    raw_reactions = load_jsonl(PROJECT_ROOT / config['paths']['reaction_manifest'])
    reactions = new_reactions(config, raw_reactions)
    old_clean = {case_key(r) for r in originals['clean']}
    clean = [seal({**r, 'case_id': 'scale_clean_' + r['name'], 'dataset_group': 'clean', 'cluster_id': object_key(r),
                   'source_pool': 'existing_held_out_scientific_manifest', 'evidence_applicability': 'pending_zero_token_preflight'})
             for r in unique(scientific) if case_key(r) not in old_clean]
    stress, missing = expanded_stress(named, scientific, reactions, originals['stress'])
    multi = expanded_multi(named, originals['multi_turn'])
    old_open = {object_key(r['base_case']) for r in originals['open_task']}
    open_pool = [r for r in named if object_key(r) not in old_open]
    output.mkdir(parents=True)
    write_jsonl(output / 'molecular_open_source.jsonl', open_pool)
    write_jsonl(output / 'reaction_source_candidates.jsonl', reactions)
    from .build_open_task_manifest import build as build_open
    open_config = deepcopy(config)
    open_config['paths']['molecular_safety_manifest'] = str(output / 'molecular_open_source.jsonl')
    open_config['paths']['reaction_manifest'] = str(output / 'reaction_source_candidates.jsonl')
    opened = build_open(open_config)
    for r in opened:
        r['name'] = r['case_id'] = 'scale_' + r['case_id']
        r['cluster_id'] = object_key(r['base_case'])
        r['gold_review_status'] = 'pending_independent_contract_review'
        seal(r)
    candidates = {'clean': clean, 'stress': stress, 'open_task': opened, 'multi_turn': multi}
    audit = {}
    for group, rows in candidates.items():
        write_jsonl(output / f'{group}_additional_candidates.jsonl', rows)
        # Original rows are embedded unchanged; new candidates are explicitly marked.
        write_jsonl(output / f'{group}_combined_candidates.jsonl', originals[group] + rows)
        audit[group] = {'original': len(originals[group]), 'additional_candidates': len(rows),
                        'combined_candidate_cases': len(originals[group]) + len(rows),
                        'additional_unique_cluster_ids': len({r['cluster_id'] for r in rows})}
    conditions = []
    for group, rows in candidates.items():
        for row in originals[group] + rows:
            rounds = range(4) if group == 'stress' else ((0, 1) if group == 'clean' else (0,))
            for orchestrator in ('heuristic', 'llm'):
                for depth in rounds:
                    identity = {'case_id': row['case_id'], 'dataset_group': group, 'orchestrator': orchestrator,
                                'reflection_rounds_requested': depth, 'enable_reflection': depth > 0, 'seed': 2026,
                                'temperature': 0.0, 'execution_authorized': False,
                                'requires_new_source_config_model_index_freeze': True}
                    identity['condition_id'] = 'scale_' + stable_hash(identity, length=20)
                    conditions.append(identity)
    write_jsonl(output / 'condition_matrix_DRAFT_NOT_RUNNABLE.jsonl', conditions)
    original_hashes = {str(RELIABILITY_ROOT / 'manifests' / f): file_sha256(RELIABILITY_ROOT / 'manifests' / f)
                       for f in ('clean.jsonl', 'stress_controlled.jsonl', 'open_tasks.jsonl', 'multi_turn_tasks.jsonl')}
    report = {'schema': 'scale_reflection_preparation', 'status': 'BLOCKED_FOR_LLM_EXECUTION',
              'external_llm_calls': 0, 'external_llm_tokens': 0, 'network_access': 'forbidden_by_socket_guard',
              'case_counts': audit, 'stress_missing_slots': missing, 'original_manifest_hashes': original_hashes,
              'conditions': len(conditions), 'llm_conditions': sum(r['orchestrator'] == 'llm' for r in conditions),
              'stress_fault_counts_new': dict(Counter(r['injection_type'] for r in stress)),
              'stress_task_counts_new': dict(Counter(r['task_type'] for r in stress)),
              'reaction_candidate_pool_counts': dict(Counter(r['task_type'] for r in reactions)),
              'budgets': {'authorized_calls': 0, 'authorized_tokens': 0, 'proposed_per_condition_calls': None,
                          'proposed_per_condition_tokens': None, 'reason': 'must profile true 2/3-round paths before setting caps'},
              'blockers': ['Independent gold/contract and direct-evidence applicability review pending',
                           'Fresh source, effective config, model resources and retrieval-index freeze required',
                           'Current result identity/evaluator groups by reflection boolean; round-aware adapter and acceptance required',
                           'Full offline graph replay, equal tool budgets and per-round immutable snapshots required',
                           'No live retrieval/model-resource preflight or LLM canary authorized/performed'],
              'interpretation': ['Candidate case count is not completed results or independent molecule count',
                                 'Clean additions reuse an existing scientific pool, not previously unseen-to-project data',
                                 'Within/cross-family shared molecules must be clustered; names-versus-SMILES aliases need resolver audit',
                                 'Heuristic and LLM differ in orchestration, not necessarily in the reflection tool implementation',
                                 'Do not select cases by observed improvement or claim more rounds will improve performance']}
    (output / 'preflight_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=PROJECT_ROOT / 'outputs/agentm_reliability/scale_reflection/preparation')
    args = parser.parse_args()
    with patch('socket.socket.connect', side_effect=RuntimeError('Network forbidden in offline preparation')), \
         patch('socket.create_connection', side_effect=RuntimeError('Network forbidden in offline preparation')):
        prepare(args.output.resolve())


if __name__ == '__main__':
    main()
