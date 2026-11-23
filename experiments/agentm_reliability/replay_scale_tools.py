"""Bounded zero-LLM diagnostic replay, isolated from formal experiment results."""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import json
import os
from pathlib import Path
import signal
import socket
from unittest.mock import patch

from .common import (PROJECT_ROOT, RELIABILITY_ROOT, file_sha256, load_config, load_jsonl,
                     read_json, source_tree_hash, write_json, write_jsonl)
from .gold_rules import FAULT_TYPES
from .reflection_depth import depth_job, bound_model_path, audit_depth_protocol
from .preflight_scale_sources import INDEX


def select_cases(stress, multi):
    """Selection uses input strata only, never scientific outcomes."""
    groups = ('admet', 'ddi', 'reaction_prediction', 'reaction_evidence')
    chosen, counts = [], Counter()
    for fault in FAULT_TYPES:
        candidates = [r for r in stress if r['injection_type'] == fault]
        row = min(candidates, key=lambda r: (counts[r['task_type']], groups.index(r['task_type']), r['case_id']))
        chosen.append(row)
        counts[row['task_type']] += 1
    if set(counts) != set(groups):
        raise ValueError('Diagnostic selection did not cover all task groups')
    dialogues = [min((r for r in multi if r['multi_turn_type'] == kind), key=lambda r:r['case_id'])
                 for kind in sorted({r['multi_turn_type'] for r in multi})]
    if len(chosen) != 8 or len(dialogues) != 4:
        raise ValueError('Unexpected diagnostic strata')
    return chosen, dialogues


def index_identity():
    import requests
    response = requests.get(f'http://127.0.0.1:9200/{INDEX}/_settings', timeout=10)
    response.raise_for_status()
    settings = response.json()[INDEX]['settings']['index']
    count = requests.get(f'http://127.0.0.1:9200/{INDEX}/_count', timeout=10)
    count.raise_for_status()
    value = {'name': INDEX, 'uuid': settings['uuid'], 'write_block': settings.get('blocks', {}).get('write'),
             'document_count': count.json()['count']}
    if value != {'name': INDEX, 'uuid': 'kEDoA3LJSFCgKIASo1itzA', 'write_block': 'true', 'document_count': 1011}:
        raise RuntimeError(f'Frozen index identity mismatch: {value}')
    return value


def prepare(root, output, *, authorize_execution=False):
    if output.exists():
        raise FileExistsError(output)
    stress_path = root / 'preparation/stress_additional_candidates.jsonl'
    multi_path = root / 'contract_revision/multi_turn_revised_candidates.jsonl'
    selected, dialogues = select_cases(load_jsonl(stress_path), load_jsonl(multi_path))
    resources = read_json(root / 'source_audit/gold_source_audit.json')['resources']
    config = load_config(RELIABILITY_ROOT / 'configs/stress_single_build.yaml')
    config['outputs']['root'] = str(output / 'diagnostic_runs_NOT_FORMAL')
    config['paths']['outputs_root'] = config['outputs']['root']
    config['runtime'].update(max_rounds=64, max_llm_calls_per_run=0, max_total_tokens_per_run=0,
                             use_llm_reasoning=False, require_llm_reasoning=False,
                             require_llm_orchestrator=False, llm_profile=None,
                             llm_orchestrator_max_retries=0, es_max_retries=0,
                             es_request_timeout_seconds=15)
    config['experiment']['name'] = 'scale_zero_api_diagnostic_not_formal'
    source = source_tree_hash()
    jobs = []
    for case in selected + dialogues:
        group = case['dataset_group']
        manifest = stress_path if group == 'stress' else multi_path
        base = case.get('base_case', case)
        binding = None
        if base.get('model_path'):
            path = str(PROJECT_ROOT / base['model_path'])
            verified = resources[path]
            binding = {'declared_path': base['model_path'], 'path': path, 'sha256': verified['sha256']}
        per_case = deepcopy(config)
        per_case.update(dataset_group=group, manifest=str(manifest))
        for depth in (range(4) if group == 'stress' else (0,)):
            job = depth_job(case, per_case, orchestrator='heuristic', depth=depth,
                            source_hash=source, manifest=manifest, model_binding=binding)
            if binding:
                bound_model_path(job)
            job['execution_authorized'] = authorize_execution
            job['authorization_scope'] = ('user_authorized_zero_API_diagnostic_only_not_formal_queue' if authorize_execution else 'prepared_only_NOT_AUTHORIZED')
            job['formal_statistics_eligible'] = False
            jobs.append(job)
    output.mkdir(parents=True)
    write_jsonl(output / 'diagnostic_jobs.jsonl', jobs, overwrite=False)
    for job in jobs:
        write_json(output / 'jobs' / f"{job['run_id']}.json", job, overwrite=False)
    write_json(output / 'plan.json', {
        'selection': 'first_input_case_per_fault_greedy_balance_task_groups_no_outcome_access',
        'stress_cases': [r['case_id'] for r in selected], 'stress_task_groups': dict(Counter(r['task_type'] for r in selected)),
        'multi_turn_cases': [r['case_id'] for r in dialogues], 'job_count': len(jobs),
        'stress_runs': 32, 'dialogue_runs': 4, 'max_tool_steps_per_turn': 64,
        'per_job_timeout_seconds': 600, 'max_llm_calls': 0, 'max_llm_tokens': 0,
        'stop_on_first_pipeline_error': True, 'no_automatic_retry': True,
        'execution_authorized':authorize_execution,
        'source_tree_hash': source, 'input_hashes': {str(p):file_sha256(p) for p in (stress_path,multi_path)},
        'formal_statistics_eligible': False,
        'scope': 'bounded_heuristic_tool_probe_not_full_candidate_acceptance_or_LLM_validation',
    }, overwrite=False)
    return jobs


def audit(job, raw):
    from .validate_real_rag_run import validate_raw_run
    from .validation import run_invariant_violations
    record = read_json(raw / 'run_record.json', {})
    result = validate_raw_run(raw)
    errors = list(result['reasons'])
    if record.get('source_tree_hash') != job['source_tree_hash']:
        errors.append('source_hash_mismatch')
    if record.get('llm_call_count', 0) or record.get('total_tokens', 0):
        errors.append('nonzero_API_usage')
    contexts = sorted(raw.glob('turns/turn_*/global_context.json')) if job['dataset_group'] == 'multi_turn' else [raw / 'runtime/global_context.json']
    if not contexts:
        errors.append('missing_contexts')
    audits = []
    for path in contexts:
        context = read_json(path, {})
        hook_path = path.parent / 'reflection_depth_snapshots.json' if job['dataset_group'] == 'multi_turn' else raw / 'hook_record.json'
        hook = read_json(hook_path, {})
        depth = audit_depth_protocol(job, context, hook)
        errors.extend(depth['errors'])
        errors.extend(run_invariant_violations(job, context))
        calls = context.get('tool_calls', [])
        if len(calls) > 64:
            errors.append('tool_step_limit_exceeded')
        audits.append({'context':str(path), 'tool_calls':len(calls), 'depth_protocol':depth,
                       'snapshot_count':len(hook.get('reflection_snapshots', []))})
    if job['dataset_group'] == 'stress':
        from .prediction_injection_audit import audit_prediction_injection
        hook = read_json(raw / 'hook_record.json', {})
        if hook.get('applied') is not True:
            errors.append('injection_not_applied')
        if not hook.get('pre_injection_context') or not hook.get('post_injection_context'):
            errors.append('missing_injection_snapshots')
        injection_contract = audit_prediction_injection(hook)
        errors.extend(injection_contract['errors'])
    else:
        injection_contract = None
    model_resources = []
    for value in record.get('runtime_model_paths', []) or []:
        path = Path(value)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        model_resources.append({'path':str(path), 'exists':path.is_file(),
                                'sha256':file_sha256(path) if path.is_file() else None})
        if not path.is_file():
            errors.append('runtime_model_path_missing')
    return {'run_id':job['run_id'], 'case_id':job['case_id'], 'depth':job['reflection_depth'],
            'passed':not errors, 'errors':sorted(set(errors)), 'real_rag':result, 'contexts':audits,
            'scientific_quality_not_used_as_pipeline_gate':True,
            'prediction_injection_contract':injection_contract,
            'model_resources':model_resources,
            'reported_scientific_status':record.get('final_status'), 'formal_statistics_eligible':False}


def execute(root, output):
    from .run_matrix import run_worker
    from .multi_turn_runner import run_worker as multi_worker
    jobs = prepare(root, output, authorize_execution=True)
    before = index_identity()
    write_json(output / 'index_before.json', before, overwrite=False)
    rows = []
    def timeout(*_):
        raise TimeoutError('Zero-API diagnostic exceeded 600-second bound')
    signal.signal(signal.SIGALRM, timeout)
    for job in jobs:
        print(json.dumps({'starting':job['run_id'], 'API_budget':0}), flush=True)
        job_path = output / 'jobs' / f"{job['run_id']}.json"
        signal.alarm(600)
        try:
            code = (multi_worker if job['dataset_group'] == 'multi_turn' else run_worker)(job_path)
            raw = output / 'diagnostic_runs_NOT_FORMAL/raw_runs' / job['run_id']
            checked = audit(job, raw)
            checked['worker_exit_code'] = code
            if code:
                checked['passed'] = False
                checked['errors'].append('worker_failed')
        except Exception as exc:
            checked = {'run_id':job['run_id'], 'passed':False, 'errors':[f'{type(exc).__name__}: {exc}']}
        finally:
            signal.alarm(0)
        rows.append(checked)
        write_json(output / 'acceptance' / f"{job['run_id']}.json", checked, overwrite=False)
        print(json.dumps(checked), flush=True)
        if not checked['passed']:
            break
    after = index_identity()
    write_json(output / 'index_after.json', after, overwrite=False)
    summary = {'planned_jobs':len(jobs), 'executed_jobs':len(rows), 'passed_jobs':sum(r['passed'] for r in rows),
               'stopped_on_pipeline_failure':any(not r['passed'] for r in rows),
               'index_identity_unchanged':before == after, 'source_hash_unchanged':source_tree_hash() == jobs[0]['source_tree_hash'],
               'external_llm_calls':0, 'external_llm_tokens':0,
               'formal_statistics_eligible':False, 'results':rows,
               'not_proven':['full_candidate_coverage', 'LLM_orchestrator_behavior', 'sufficient_future_LLM_token_budget', 'scientific_improvement']}
    write_json(output / 'summary.json', summary, overwrite=False)
    print(json.dumps({k:v for k,v in summary.items() if k != 'results'}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    os.environ.update(AGENTM_ENABLE_NETWORK_LOOKUP='0', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                      AGENTM_ES_EMBEDDING_DEVICE='cpu', TOKENIZERS_PARALLELISM='false', CUDA_VISIBLE_DEVICES='')
    import torch
    torch.set_num_threads(2)
    original = socket.socket.connect
    def connect(sock, address):
        if not isinstance(address, tuple) or address[:2] != ('127.0.0.1', 9200):
            raise RuntimeError('Zero-API replay permits only read-only local ES access')
        return original(sock, address)
    with patch('socket.socket.connect', connect):
        if args.prepare_only:
            jobs = prepare(args.root.resolve(),args.output.resolve())
            print(json.dumps({'prepared_jobs':len(jobs),'execution_authorized':False,'external_llm_calls':0}))
        else:
            execute(args.root.resolve(), args.output.resolve())


if __name__ == '__main__':
    main()
