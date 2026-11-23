"""Offline semantic contract checks; no prediction outcomes are read."""
import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
from unittest.mock import patch
from .common import load_jsonl, write_json
from .prepare_scale_reflection import entity_key


def resolved_key(value):
    # Exercise the real offline path, including the post-processor seed
    # override. Looking only at the underlying dictionary produces false
    # mismatches (e.g. metformin is corrected by _apply_local_identity_seed).
    from agentm.runtime.tools.scientific import ScientificToolRuntime
    from agentm.agents.molecule_processing_agent import MoleculeProcessingAgent
    runtime = object.__new__(ScientificToolRuntime)
    runtime.processor = MoleculeProcessingAgent()
    with patch.dict(os.environ, {'AGENTM_ENABLE_NETWORK_LOOKUP': '0'}), \
         patch('socket.socket.connect', side_effect=AssertionError('Identity audit forbids network')):
        molecule = runtime._process_molecule_mention(value)
    return entity_key(molecule.canonical_smiles) if molecule.is_valid else None


def multi_checks(row):
    turns = row['turns']
    gold = row['gold_plan']
    kind = row['multi_turn_type']
    issues = []
    if len(turns) != 2 or [t['turn'] for t in turns] != [1, 2]:
        issues.append('two_ordered_turns_required')
        return {'case_id': row['case_id'], 'type': kind, 'checks_passed': False, 'issues': issues,
                'scope': 'deterministic_task_contract_consistency_not_expert_report_quality'}
    if kind == 'add_two_endpoints':
        state = gold['turn_level_expected_state'][-1]
        if len(set(state.get('endpoints', []))) != 3:
            issues.append('claimed_two_new_endpoints_but_final_endpoints_not_three_distinct')
        additions = [s.removeprefix('add:') for s in gold['required_objectives'] if s.startswith('add:')]
        if len(set(additions)) != 2:
            issues.append('addition_objectives_not_two_distinct')
    elif kind == 'single_to_ddi':
        pair = [resolved_key(s.strip()) for s in turns[1]['request'].split(',')]
        single = resolved_key(turns[0]['request'])
        if single is None or None in pair:
            # A name-vs-structure mismatch is not proof of different entities.
            issues.append('single_drug_membership_in_final_pair_requires_alias_resolution')
        elif single not in pair:
            issues.append('initial_single_drug_absent_from_added_ddi_pair')
    elif kind == 'withdraw_and_replace':
        if entity_key(turns[0]['request']) == entity_key(turns[1]['request']):
            issues.append('replacement_did_not_change_target')
        if turns[0]['request'] not in gold.get('stale_records_to_remove', []):
            issues.append('withdrawn_target_missing_from_stale_gold')
    elif kind == 'resolve_specific_structure':
        # Existing runtime-local seed table supplies an independent code-side
        # identity mapping; missing seeds stay unresolved, not auto-passed.
        resolved = resolved_key(turns[0]['request'])
        if not resolved:
            issues.append('name_structure_mapping_not_in_runtime_local_resolvers')
        elif resolved != entity_key(turns[1]['request']):
            issues.append('name_and_specific_structure_are_different_entities')
    return {'case_id': row['case_id'], 'type': kind, 'checks_passed': not issues, 'issues': issues,
            'scope': 'deterministic_task_contract_consistency_not_expert_report_quality'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preparation', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    with patch('socket.socket.connect', side_effect=RuntimeError('Offline contract audit forbids network')):
        rows = load_jsonl(args.preparation / 'multi_turn_combined_candidates.jsonl')
        checked = [dict(multi_checks(row), cohort='additional' if row['case_id'].startswith('scale_') else 'original') for row in rows]
    summary = {'external_llm_calls': 0, 'external_llm_tokens': 0, 'rows': checked,
               'cohort_counts': dict(Counter(r['cohort'] for r in checked)),
               'passed_by_cohort': dict(Counter(r['cohort'] for r in checked if r['checks_passed'])),
               'issue_counts': dict(Counter(s for r in checked for s in r['issues'])),
               'policy': 'flag_only_original_manifests_and_results_not_modified'}
    write_json(args.output, summary, overwrite=False)
    print(json.dumps({k: v for k, v in summary.items() if k != 'rows'}, indent=2))


if __name__ == '__main__':
    main()
