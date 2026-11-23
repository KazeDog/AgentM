"""Append-only semantic revision and evidence policy proposal; never a launcher.

Reads candidate inputs and prior zero-API preflight records only, not scientific
outcomes. Local resolver agreement is a consistency check, not chemical truth.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
from unittest.mock import patch

from .audit_scale_contracts import multi_checks, resolved_key
from .common import file_sha256, load_jsonl, source_tree_hash, stable_hash, write_json, write_jsonl
from .gold_rules import FAULT_TYPES
from .prepare_scale_reflection import entity_key, seal


def identity_check(name):
    from agentm.tools.name_resolver import resolve_name_to_smiles
    from agentm.runtime.tools.scientific import ScientificToolRuntime
    dictionary, _ = resolve_name_to_smiles(name, allow_network=False)
    runtime = object.__new__(ScientificToolRuntime)
    seed = runtime._local_identity_seed(query=name)
    seeded = seed.get('canonical_smiles') if seed else None
    conflict = bool(dictionary and seeded and entity_key(dictionary) != entity_key(seeded))
    actual = resolved_key(name)
    # Divergent legacy dictionaries are recorded, but an executed, correct
    # post-processor override must not be misreported as runtime ambiguity.
    unresolved_conflict = bool(conflict and actual != entity_key(seeded))
    return {'name': name, 'dictionary_structure': dictionary, 'seed_structure': seeded,
            'runtime_resolved_structure': actual,
            'dictionary_seed_disagreement': conflict,
            'effective_resolution_stage': 'post_processor_seed_override' if conflict and not unresolved_conflict else 'offline_runtime',
            'status': 'conflicting_local_maps' if unresolved_conflict else ('local_mapping_available' if actual else 'missing_local_mapping'),
            'external_identity_verified': False}


def revise_multi(original):
    row = deepcopy(original)
    before = stable_hash({'turns': row['turns'], 'gold_plan': row['gold_plan']})
    kind, turns, gold = row['multi_turn_type'], row['turns'], row['gold_plan']
    reasons, identities = [], []
    if kind == 'add_two_endpoints':
        retained = gold['turn_level_expected_state'][0]['endpoints'][0]
        if len(set(gold['turn_level_expected_state'][-1]['endpoints'])) != 3:
            options = [('BBB_Martins', 'bbbp'), ('hERG', 'herg'), ('DILI', 'dili'), ('HIA_Hou', 'hia'), ('Pgp_Broccatelli', 'pgp')]
            additions = [x for x in options if x[1] != retained][:2]
            turns[1]['request'] = f"For the same molecule {turns[0]['request']}, additionally evaluate {additions[0][1]} and {additions[1][1]}."
            turns[1]['dataset_name'] = additions[0][0]
            gold['required_objectives'] = [f'retain:{retained}'] + [f'add:{x[1]}' for x in additions]
            gold['required_outputs'] = list(gold['required_objectives'])
            gold['turn_level_expected_state'][-1]['endpoints'] = [retained] + [x[1] for x in additions]
            reasons.append('request_two_distinct_new_endpoints')
    elif kind == 'resolve_specific_structure':
        identity = identity_check(turns[0]['request'])
        identities.append(identity)
        intended = identity['runtime_resolved_structure']
        if intended and entity_key(turns[1]['request']) != entity_key(intended):
            turns[1]['request'] = entity_key(intended)
            gold['turn_level_expected_state'][-1]['entity'] = turns[1]['request']
            gold['expected_final_scientific_objects'] = [turns[1]['request']]
            reasons.append('replace_unrelated_structure_with_same_named_entity_candidate')
        # An unambiguous drug name is not proof of ambiguity. This case tests
        # confirmation of identity and state/citation refresh, not a new target.
        turns[1]['instruction'] = 'Confirm this explicit structure for the same named molecule; keep the endpoint and refresh predictions and citations for the confirmed structure.'
        gold['required_objectives'] = ['confirm_same_entity_structure', 'recompute_prediction', 'refresh_citations']
        gold['required_outputs'] = list(gold['required_objectives'])
        gold['required_citation_updates'] = ['revalidate name-linked citations against the confirmed structure; remove only incompatible mappings']
        row['task_semantics'] = 'same_entity_name_to_structure_confirmation_not_proven_ambiguity_resolution'
        reasons.append('state_same_entity_confirmation_explicitly')
    elif kind == 'single_to_ddi':
        pair = [x.strip() for x in turns[1]['request'].split(',')]
        identities.extend(identity_check(x) for x in pair)
        if resolved_key(turns[0]['request']) not in [resolved_key(x) for x in pair]:
            turns[0]['request'] = pair[0]
            gold['turn_level_expected_state'][0]['objects'] = [pair[0]]
            reasons.append('start_with_a_member_of_the_later_ddi_pair')
    row['parent_case_id'] = original['case_id']
    row['parent_case_content_hash'] = stable_hash(original)
    row['cohort'] = 'additional' if original['case_id'].startswith('scale_') else 'revised_original'
    row['case_id'] = 'scale_contract__' + original['case_id']
    row['name'] = row['case_id']
    row['contract_revision'] = reasons
    row['stimulus_or_gold_changed'] = before != stable_hash({'turns': turns, 'gold_plan': gold})
    row['identity_preflight'] = identities
    check = multi_checks(row)
    blockers = list(check['issues'])
    blockers.extend(f"identity:{x['name']}:{x['status']}" for x in identities if x['status'] != 'local_mapping_available')
    row['contract_preflight'] = {'passed': not blockers, 'blockers': blockers,
                                 'scope': 'local_contract_consistency_only_not_external_identity_validation'}
    row['execution_authorized'] = False
    return seal(row)


def evidence_policy():
    prerequisites = {
        'citation_mismatch': ['baseline_citation_maps_to_retained_record', 'original_mapping_available_to_evaluator_only'],
        'entity_or_endpoint_mismatch': ['baseline_record_matches_target_and_endpoint', 'mismatch_injected_into_one_record'],
        'irrelevant_or_weak_evidence': ['weak_or_irrelevant_record_present_after_injection', 'no_upgrade_to_direct_support'],
        'missing_field_or_overclaim': ['successful_baseline_prediction', 'one_field_or_claim_mutated'],
        'conflicting_evidence': ['two_distinct_records_with_opposing_claims_after_injection', 'both_record_ids_retained'],
        'plausible_but_wrong_prediction': ['independent_heldout_label_available', 'wrong_value_differs_from_label', 'label_hidden_from_agent'],
        'prediction_tool_or_service_failure': ['failure_injected_at_tool_boundary', 'no_fabricated_prediction'],
        'no_direct_evidence_or_ambiguous_entity': ['missing_direct_evidence_or_unresolved_entity_after_injection', 'abstention_not_counted_as_pipeline_failure'],
    }
    return {fault: {
        'fault': fault, 'controlled_protocol_prerequisites': prerequisites[fault],
        'direct_external_evidence_required_for_protocol_metric': False,
        'protocol_scope': ('controlled_conflict_handling_not_natural_literature_disagreement' if fault == 'conflicting_evidence' else 'controlled_artifact_and_state_handling_not_external_scientific_truth'),
        'scientific_claim_gate': ['independent_entity_endpoint_matched_source_body', 'unchanged_directness_classifier', 'not_qc_or_synthetic_injection', 'preregistered_claim_specific_evaluation'],
        'supported_prediction_accuracy_requires': 'independent_heldout_labels_not_rag_presence',
        'absence_is_intended_condition': fault == 'no_direct_evidence_or_ambiguous_entity',
        'evaluation_rule_changed': False, 'policy_status': 'proposal_not_execution_authorization',
    } for fault in FAULT_TYPES}


def evidence_inventory(row, root, policies):
    case_id = row['case_id']
    path = root / 'hybrid_preflight' / f'{case_id}.json'
    record = json.loads(path.read_text()) if path.exists() else None
    provenance = record.get('provenance', {}) if record else {}
    return {'case_id': case_id, 'case_content_hash': stable_hash(row), 'fault': row['injection_type'],
            'policy': policies[row['injection_type']],
            'cached_molecular_hybrid_record': str(path) if record else None,
            'cached_record_sha256': file_sha256(path) if record else None,
            'observed_direct_evidence_count': provenance.get('direct_evidence_count'),
            'observed_qc_grounding_count': provenance.get('qc_grounding_count'),
            'observed_external_evidence_count': provenance.get('external_evidence_count'),
            'record_status': 'cached_preflight_available' if record else 'not_checked_by_this_inventory',
            'protocol_applicability': 'pending_baseline_and_post_injection_offline_replay',
            'external_scientific_repair_claim_ready': False,
            'note': 'Reaction resource preflight is separate and is not relabelled as molecular direct evidence.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    prep = args.root / 'preparation'
    source_paths = list(prep.glob('*.jsonl')) + [prep / 'preflight_report.json']
    hashes = {str(p): file_sha256(p) for p in source_paths}
    with patch('socket.socket.connect', side_effect=AssertionError('Offline revision forbids network')), patch('socket.create_connection', side_effect=AssertionError('Offline revision forbids network')):
        rows = [revise_multi(r) for r in load_jsonl(prep / 'multi_turn_combined_candidates.jsonl')]
        policies = evidence_policy()
        inventory = [evidence_inventory(r, args.root, policies) for r in load_jsonl(prep / 'stress_combined_candidates.jsonl')]
    signatures = Counter(stable_hash(r['turns']) for r in rows)
    if any(n != 1 for n in signatures.values()):
        raise AssertionError('Duplicate conversation stimuli; do not silently replace cases')
    if hashes != {str(p): file_sha256(p) for p in source_paths}:
        raise AssertionError('Input changed during revision')
    args.output.mkdir(parents=True)
    write_jsonl(args.output / 'multi_turn_revised_candidates.jsonl', rows, overwrite=False)
    write_json(args.output / 'stress_evidence_policy_PROPOSED.json', policies, overwrite=False)
    write_jsonl(args.output / 'stress_evidence_inventory.jsonl', inventory, overwrite=False)
    summary = {'external_llm_calls': 0, 'external_llm_tokens': 0, 'network_blocked': True,
               'source_tree_hash': source_tree_hash(), 'input_hashes': hashes,
               'multi_turn_count': len(rows), 'multi_turn_types': dict(Counter(r['multi_turn_type'] for r in rows)),
               'locally_consistent': sum(r['contract_preflight']['passed'] for r in rows),
               'changed_stimulus_or_gold': sum(r['stimulus_or_gold_changed'] for r in rows),
               'shadowed_dictionary_disagreements': sorted({x['name'] for r in rows for x in r['identity_preflight'] if x.get('dictionary_seed_disagreement')}),
               'blocked_cases': [{'case_id': r['case_id'], **r['contract_preflight']} for r in rows if not r['contract_preflight']['passed']],
               'stress_policy_count': len(policies), 'stress_inventory_count': len(inventory),
               'cached_molecular_preflight_count': sum(r['cached_molecular_hybrid_record'] is not None for r in inventory),
               'all_execution_unauthorized': all(r['execution_authorized'] is False for r in rows),
               'ready_for_llm_execution': False,
               'remaining': ['baseline_and_post_injection_real_tool_replay', 'freeze_new_source_config_model_index_and_budget_before_canary'],
               'historical_manifests_results_and_papers_unchanged': True,
               'previous_depth_jobs': 'draft does not reference these revised case identities; must regenerate after gates pass'}
    write_json(args.output / 'summary.json', summary, overwrite=False)
    print(json.dumps({k:v for k,v in summary.items() if k != 'input_hashes'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
