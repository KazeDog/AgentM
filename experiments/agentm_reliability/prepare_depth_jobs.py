"""Export unauthorized depth-aware draft jobs and case-binding checks only."""
from collections import Counter
from copy import deepcopy
import argparse
import json
from pathlib import Path
from unittest.mock import patch

from .common import PROJECT_ROOT, RELIABILITY_ROOT, load_config, load_jsonl, source_tree_hash, write_json, write_jsonl
from .reflection_depth import depth_job, validate_depth_job, bound_model_path


def prepare(preparation, source_audit, output):
    if output.exists():
        raise FileExistsError(output)
    resources = json.loads((source_audit / 'gold_source_audit.json').read_text())['resources']
    config = load_config(RELIABILITY_ROOT / 'configs/stress_single_build.yaml')
    config['outputs']['root'] = str(output / 'NEVER_STARTED_raw')
    config['paths']['outputs_root'] = config['outputs']['root']
    # Proposal, NOT a frozen/authorized change to the previous experiment.
    config['runtime']['max_rounds'] = 64
    source_hash = source_tree_hash()
    jobs = []
    bindings = {}
    errors = []
    for group in ('clean', 'stress', 'open_task', 'multi_turn'):
        manifest = preparation / f'{group}_combined_candidates.jsonl'
        for case in load_jsonl(manifest):
            base = case.get('base_case', case)
            binding = None
            if base.get('model_path'):
                actual = str(PROJECT_ROOT / base['model_path'])
                verified = resources.get(actual)
                if not verified or not verified['exists'] or not verified['sha256']:
                    errors.append({'case_id': case['case_id'], 'error': 'no_verified_model_binding'})
                    continue
                binding = {'declared_path': base['model_path'], 'path': actual, 'sha256': verified['sha256']}
                bindings[case['case_id']] = binding
            per_group = deepcopy(config)
            per_group['dataset_group'] = group
            per_group['manifest'] = str(manifest)
            for orchestrator in ('heuristic', 'llm'):
                for depth in (range(4) if group == 'stress' else ((0, 1) if group == 'clean' else (0,))):
                    job = depth_job(case, per_group, orchestrator=orchestrator, depth=depth,
                                    source_hash=source_hash, manifest=manifest, model_binding=binding)
                    validate_depth_job(job)
                    if binding:
                        # Do not silently substitute a registry/default model.
                        bound_model_path(job)
                    jobs.append(job)
    output.mkdir(parents=True)
    write_jsonl(output / 'unauthorized_depth_jobs.jsonl', jobs)
    write_json(output / 'verified_model_bindings.json', bindings, overwrite=False)
    report = {'status': 'DRAFT_NOT_AUTHORIZED', 'jobs': len(jobs),
              'unique_run_ids': len({j['run_id'] for j in jobs}),
              'depth_counts': dict(Counter(j['reflection_depth'] for j in jobs)),
              'group_counts': dict(Counter(j['dataset_group'] for j in jobs)),
              'model_bound_cases': len(bindings), 'model_binding_errors': errors,
              'all_execution_authorized_false': all(j['execution_authorized'] is False for j in jobs),
              'all_api_budgets_zero': all(j['config']['runtime']['max_llm_calls_per_run'] == j['config']['runtime']['max_total_tokens_per_run'] == 0 for j in jobs),
              'external_llm_calls': 0, 'external_llm_tokens': 0, 'source_tree_hash': source_hash,
              'multi_turn_conditions_exported': sum(j['dataset_group'] == 'multi_turn' for j in jobs),
              'remaining_gates': ['Real hybrid retrieval applicability', 'Semantic gold acceptance for 40 conversation cases',
                                  'Actual tool replay for common tool-step and API-budget sizing',
                                  'Final model/index/config/source freeze and explicit execution authorization']}
    write_json(output / 'summary.json', report, overwrite=False)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preparation', type=Path, required=True)
    parser.add_argument('--source-audit', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    with patch('socket.socket.connect', side_effect=RuntimeError('Draft job export forbids network')):
        prepare(args.preparation.resolve(), args.source_audit.resolve(), args.output.resolve())


if __name__ == '__main__':
    main()
