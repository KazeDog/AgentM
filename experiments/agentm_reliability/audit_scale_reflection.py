"""Run network-blocked regressions and persist an append-only test report."""
from __future__ import annotations
import argparse
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch
from .common import RELIABILITY_ROOT, file_sha256, load_jsonl, source_tree_hash, stable_hash


def audit_preparation(root):
    from collections import Counter
    from .prepare_scale_reflection import object_key, case_key
    from .gold_rules import validate_stress_case
    preflight = json.loads((root / 'preflight_report.json').read_text())
    checks = {}
    for path, digest in preflight['original_manifest_hashes'].items():
        checks['preserved:' + Path(path).name] = file_sha256(path) == digest
    combined = {}
    for group, required in [('clean', 100), ('stress', 64), ('open_task', 30), ('multi_turn', 20)]:
        new = load_jsonl(root / f'{group}_additional_candidates.jsonl')
        rows = load_jsonl(root / f'{group}_combined_candidates.jsonl')
        old = rows[:-len(new)]
        combined[group] = rows
        checks[group + ':new_count'] = len(new) == required
        checks[group + ':unique_case_ids'] = len({r['case_id'] for r in rows}) == len(rows)
        checks[group + ':candidate_hashes'] = all(r['case_hash'] == stable_hash({k: v for k, v in r.items() if k != 'case_hash'}) for r in new)
        if group == 'clean':
            checks['clean:disjoint_task_endpoint_objects'] = not ({case_key(r) for r in new} & {case_key(r) for r in old})
        if group in ('stress', 'open_task'):
            old_keys = {object_key(r['base_case']) for r in old}
            new_keys = {object_key(r['base_case']) for r in new}
            checks[group + ':disjoint_base_objects'] = not (old_keys & new_keys)
            checks[group + ':unique_new_base_objects'] = len(new_keys) == len(new)
        if group == 'stress':
            for row in new:
                validate_stress_case(row)
            checks['stress:balanced_faults'] = sorted(Counter(r['injection_type'] for r in new).values()) == [8] * 8
            checks['stress:balanced_groups'] = sorted(Counter(r['task_type'] for r in new).values()) == [16] * 4
        if group == 'multi_turn':
            additions = [r for r in new if r['multi_turn_type'] == 'add_two_endpoints']
            checks['multi_turn:distinct_requested_endpoints'] = all(len(set(r['gold_plan']['turn_level_expected_state'][1]['endpoints'])) == 3 for r in additions)
    conditions = load_jsonl(root / 'condition_matrix_DRAFT_NOT_RUNNABLE.jsonl')
    checks['matrix:no_authorized_execution'] = all(r['execution_authorized'] is False for r in conditions)
    checks['matrix:unique_conditions'] = len({r['condition_id'] for r in conditions}) == len(conditions) == 2024
    cells = Counter((r['dataset_group'], r['case_id']) for r in conditions)
    checks['matrix:complete_depth_pairing'] = all(cells[group, r['case_id']] == {'clean': 4, 'stress': 8, 'open_task': 2, 'multi_turn': 2}[group]
                                               for group, rows in combined.items() for r in rows)
    return {'passed': all(checks.values()), 'checks': checks,
            'scope': 'structural_candidate_audit_not_real_RAG_or_scientific_acceptance'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--preparation', type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    stream = io.StringIO()
    with patch('socket.socket.connect', side_effect=AssertionError('Offline audit: network forbidden')), \
         patch('socket.create_connection', side_effect=AssertionError('Offline audit: network forbidden')):
        suite = unittest.defaultTestLoader.discover(str(RELIABILITY_ROOT / 'tests'),
                                                  top_level_dir=str(RELIABILITY_ROOT.parents[1]))
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
        preparation = audit_preparation(args.preparation) if args.preparation else None
    passed = result.wasSuccessful() and (preparation is None or preparation['passed'])
    payload = {'passed': passed, 'candidate_audit': preparation, 'tests_run': result.testsRun,
               'failures': [(str(t), e) for t, e in result.failures],
               'errors': [(str(t), e) for t, e in result.errors],
               'skipped': [(str(t), e) for t, e in result.skipped],
               'external_llm_calls': 0, 'external_llm_tokens': 0,
               'network_blocked': True, 'source_tree_hash_after': source_tree_hash(),
               'original_stress_sha256': file_sha256(RELIABILITY_ROOT / 'manifests/stress_controlled.jsonl'),
               'log': stream.getvalue()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n')
    print(stream.getvalue())
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
