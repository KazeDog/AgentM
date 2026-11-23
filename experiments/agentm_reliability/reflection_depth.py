"""Depth-aware jobs and independent protocol audit, opt-in for new experiments.

This module never launches a run or grants execution authorization. Historical
boolean-factorial jobs keep their existing identities and hook behavior.
"""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from typing import Any, Mapping
from pathlib import Path

from .common import file_sha256, run_id_for, stable_hash
from .injections import ControlledInjectionHook
from agentm.runtime.compat import to_jsonable

PROTOCOL = 'fixed_reflection_depth'
STALE = ('reasoning_stale', 'report_stale', 'citation_verification_stale', 'schema_verification_stale')


def checked_depth(value):
    if type(value) is not int or value not in (0, 1, 2, 3):
        raise ValueError('reflection_depth must be an integer in {0,1,2,3}')
    return value


def depth_job(case, config, *, orchestrator, depth, source_hash, manifest, model_binding=None):
    depth = checked_depth(depth)
    if orchestrator not in ('heuristic', 'llm'):
        raise ValueError('unknown orchestrator')
    effective = deepcopy(config)
    effective.pop('_meta', None)
    effective['runtime']['max_reflection_rounds'] = depth
    # Candidate exports are non-runnable even if passed to the old worker.
    effective['runtime']['max_llm_calls_per_run'] = 0
    effective['runtime']['max_total_tokens_per_run'] = 0
    if model_binding is not None:
        effective['runtime']['verified_case_model_binding'] = deepcopy(model_binding)
    effective['matrix'] = {'orchestrators': [orchestrator], 'reflection': [depth > 0]}
    common = deepcopy(effective)
    common['runtime'].pop('max_reflection_rounds')
    common.pop('matrix')
    job = {'case': deepcopy(case), 'case_id': case['case_id'], 'dataset_group': case['dataset_group'],
           'orchestrator': orchestrator, 'reflection_enabled': depth > 0, 'reflection_depth': depth,
           'reflection_protocol': PROTOCOL, 'execution_authorized': False,
           'seed': int(effective['experiment']['seed']), 'repeat_index': 0,
           'source_tree_hash': source_hash, 'source_manifest': str(manifest), 'config': effective,
           'config_hash': stable_hash(effective), 'common_config_hash': stable_hash(common),
           'runtime_guard_protocol': 'pre_llm_real_rag_fail_closed'}
    job['run_id'] = run_id_for(job, job['config_hash'])
    return job


def validate_depth_job(job, *, executing=False):
    depth = checked_depth(job.get('reflection_depth'))
    if job.get('reflection_protocol') != PROTOCOL:
        raise ValueError('missing/unknown reflection protocol')
    if job.get('reflection_enabled') is not (depth > 0):
        raise ValueError('reflection boolean/depth mismatch')
    config = job['config']
    if config['runtime'].get('max_reflection_rounds') != depth:
        raise ValueError('runtime/job reflection depth mismatch')
    if job.get('config_hash') != stable_hash({k: v for k, v in config.items() if k != '_meta'}):
        raise ValueError('effective config hash mismatch')
    if job.get('run_id') != run_id_for(job, job['config_hash']):
        raise ValueError('depth job run ID mismatch')
    if executing and job.get('execution_authorized') is not True:
        raise PermissionError('depth candidate has no explicit execution authorization')
    if executing:
        from .queue_applicability import validate_queue_applicability
        validate_queue_applicability(job)


def bound_model_path(job):
    binding = job['config']['runtime'].get('verified_case_model_binding')
    if binding is None:
        return None
    base = job['case'].get('base_case', job['case'])
    if binding.get('declared_path') != base.get('model_path'):
        raise ValueError('model binding does not belong to this case')
    path = Path(binding['path'])
    if not path.is_absolute() or not path.is_file() or file_sha256(path) != binding.get('sha256'):
        raise ValueError('model resource binding hash/path mismatch')
    return str(path)


def condition_key(record):
    if record.get('reflection_protocol') != PROTOCOL:
        raise ValueError('do not infer depth from historical reflection-on records')
    return (record['dataset_group'], record['case_id'], record['orchestrator'],
            checked_depth(record['reflection_depth']), record['seed'], record.get('repeat_index', 0))


class DepthSnapshotHook(ControlledInjectionHook):
    """Deep-copied, hash-chained snapshots; does not change scientific feedback."""
    def __init__(self, case, depth):
        super().__init__(case)
        self.depth = checked_depth(depth)
        self.snapshots = []
        self.active_round = 0

    def _capture(self, phase, number, context):
        payload = to_jsonable(deepcopy({key: context.get(key) for key in (
            'final_report', 'verifier_report', 'reflection', 'predictions', 'reaction_predictions',
            'evidence', 'citation_map', 'confidence', 'objective_status',
        )}))
        metadata = context.get('metadata', {})
        payload['protocol_state'] = {k: deepcopy(metadata.get(k)) for k in (
            'reflection_rounds', 'reflection_feedback_applied_round', 'reflection_retrieval_requested',
            'reflection_retrieval_done', 'loop_status', *STALE,
        )}
        item = {'sequence': len(self.snapshots), 'phase': phase, 'reflection_round': number,
                'previous_sha256': self.snapshots[-1]['sha256'] if self.snapshots else None,
                'completed_tool_count_at_hook': len(context.get('tool_calls', [])), 'payload': payload}
        item['sha256'] = stable_hash(item)
        self.snapshots.append(item)

    def before_tool(self, *, action, context):
        if action.tool == 'reflect_on_report':
            if self.active_round:
                self._capture('round_end', self.active_round, context)
            self.active_round = int(context.get('metadata', {}).get('reflection_rounds', 0)) + 1
            self._capture('round_input', self.active_round, context)
        return super().before_tool(action=action, context=context)

    def after_tool(self, *, action, result, context):
        super().after_tool(action=action, result=result, context=context)
        if action.tool == 'reflect_on_report':
            self._capture('reflection_output', self.active_round, context)
        elif action.tool == 'apply_reflection_feedback':
            self._capture('feedback_applied', self.active_round, context)

    def after_finalize(self, *, context):
        super().after_finalize(context=context)
        if self.active_round:
            self._capture('round_end', self.active_round, context)
            self.active_round = 0

    def export(self):
        payload = super().export()
        payload.update(reflection_protocol=PROTOCOL, reflection_depth=self.depth,
                       reflection_snapshots=deepcopy(self.snapshots))
        return payload


def audit_depth_protocol(job, context, hook_record):
    """Hard execution gates only; never upgrades independent science quality."""
    errors = []
    try:
        validate_depth_job(job)
    except (ValueError, KeyError) as exc:
        errors.append(str(exc))
    depth = checked_depth(job.get('reflection_depth'))
    calls = [r for r in context.get('tool_calls', []) if r.get('tool') in ('reflect_on_report', 'apply_reflection_feedback')]
    expected = ['reflect_on_report', 'apply_reflection_feedback'] * depth
    if [r['tool'] for r in calls] != expected:
        errors.append('reflection_tool_sequence_mismatch')
    if any(r.get('status') != 'completed' or r.get('error') for r in calls):
        errors.append('reflection_tool_not_completed')
    metadata = context.get('metadata', {})
    if int(metadata.get('reflection_rounds', 0)) != depth or int(metadata.get('reflection_feedback_applied_round', 0)) != depth:
        errors.append('reflection_counters_mismatch')
    if metadata.get('max_reflection_rounds') != depth or metadata.get('enable_reflection') is not (depth > 0):
        errors.append('runtime_reflection_condition_mismatch')
    def pending(state):
        return any(state.get(k) for k in STALE) or (state.get('reflection_retrieval_requested') and not state.get('reflection_retrieval_done'))
    if pending(metadata):
        errors.append('unfinished_refresh_or_retrieval')
    snapshots = hook_record.get('reflection_snapshots', [])
    if hook_record.get('reflection_protocol') != PROTOCOL or hook_record.get('reflection_depth') != depth:
        errors.append('hook_protocol_mismatch')
    phases = [(n, p) for n in range(1, depth + 1) for p in ('round_input', 'reflection_output', 'feedback_applied', 'round_end')]
    if [(r.get('reflection_round'), r.get('phase')) for r in snapshots] != phases:
        errors.append('snapshot_round_sequence_mismatch')
    previous = None
    for index, item in enumerate(snapshots):
        if item.get('sequence') != index or item.get('previous_sha256') != previous or item.get('sha256') != stable_hash({k: v for k, v in item.items() if k != 'sha256'}):
            errors.append('snapshot_hash_chain_invalid')
        state = item.get('payload', {}).get('protocol_state', {})
        n = item.get('reflection_round')
        if item.get('phase') == 'round_end' and (state.get('reflection_feedback_applied_round') != n or state.get('reflection_rounds') != n or pending(state)):
            errors.append('round_ended_before_feedback_refresh_completed')
        previous = item.get('sha256')
    return {'passed': not errors, 'errors': sorted(set(errors)), 'reflection_depth': depth,
            'scope': 'reflection_protocol_only_not_scientific_quality', 'snapshot_count': len(snapshots)}


def paired_depth_rows(records, *, depths=(0, 1, 2, 3)):
    """Strict independent-analysis input: one accepted run per case/depth.

    Scientific failure rows are retained. Missing/duplicate runs and mixed
    source/resource/index/common-config signatures are rejected, not averaged.
    """
    groups = defaultdict(dict)
    for row in records:
        group, case, orch, depth, seed, repeat = condition_key(row)
        if row.get('protocol_accepted') is not True:
            raise ValueError('unaccepted protocol record')
        key = (group, case, orch, seed, repeat)
        if depth in groups[key]:
            raise ValueError('duplicate depth condition; explicit replacement lineage required')
        groups[key][depth] = row
    for by_depth in groups.values():
        if set(by_depth) != set(depths):
            raise ValueError('incomplete paired depth block')
        for field in ('source_tree_hash', 'common_config_hash', 'model_resource_hash', 'evidence_index_hash', 'case_hash'):
            values = {r.get(field) for r in by_depth.values()}
            if None in values or '' in values or len(values) != 1:
                raise ValueError(f'paired provenance mismatch: {field}')
    return [{**dict(zip(('dataset_group', 'case_id', 'orchestrator', 'seed', 'repeat_index'), key)),
             'depth_records': [value[d] for d in depths]} for key, value in sorted(groups.items())]
