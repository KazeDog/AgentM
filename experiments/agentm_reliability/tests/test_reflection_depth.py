from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.agentm_reliability.reflection_depth import (
    PROTOCOL, DepthSnapshotHook, audit_depth_protocol, checked_depth, condition_key,
    depth_job, paired_depth_rows, validate_depth_job, bound_model_path,
)
from experiments.agentm_reliability.common import run_id_for, file_sha256
from experiments.agentm_reliability.canonicalize_runs import run_identity
from agentm.runtime.tools.registry import AgentAction, ToolResult


def make_job(depth=2):
    return depth_job({'case_id': 'fixture', 'dataset_group': 'stress'},
                     {'runtime': {}, 'experiment': {'seed': 2026}},
                     orchestrator='llm', depth=depth, source_hash='fixture_source', manifest='fixture.jsonl')


def complete_fixture(depth):
    job = make_job(depth)
    hook = DepthSnapshotHook(job['case'], depth)
    context = {'metadata': {'enable_reflection': depth > 0, 'max_reflection_rounds': depth},
               'final_report': {'text': 'before'}, 'verifier_report': {'summary': {'status': 'fail'}}, 'tool_calls': []}
    for number in range(1, depth + 1):
        for tool in ('reflect_on_report', 'apply_reflection_feedback'):
            action = AgentAction('ReflectionAgent', tool)
            hook.before_tool(action=action, context=context)
            if tool == 'reflect_on_report':
                context['metadata']['reflection_rounds'] = number
                context['reflection'] = {'round': number}
            else:
                context['metadata']['reflection_feedback_applied_round'] = number
            result = ToolResult(tool, 'completed')
            hook.after_tool(action=action, result=result, context=context)
            context['tool_calls'].append(result.to_dict())
        context['final_report'] = {'text': f'round {number}'}
    hook.after_finalize(context=context)
    return job, context, hook


class ReflectionDepthTests(unittest.TestCase):
    def test_strict_depth(self):
        for value in (True, -1, 4, 2.0, '2', None):
            with self.assertRaises(ValueError):
                checked_depth(value)

    def test_identity_separates_depths_and_preserves_common_configuration(self):
        jobs = [make_job(i) for i in range(4)]
        self.assertEqual(len({j['run_id'] for j in jobs}), 4)
        self.assertEqual(len({j['common_config_hash'] for j in jobs}), 1)
        for j in jobs:
            validate_depth_job(j)
            self.assertIn(f"reflection-depth-{j['reflection_depth']}", j['run_id'])

    def test_depth_part_of_identity_even_if_same_config_hash(self):
        jobs = [make_job(i) for i in range(4)]
        self.assertEqual(len({run_id_for(j, 'fixed') for j in jobs}), 4)

    def test_legacy_records_not_inferred_or_silently_merged(self):
        with self.assertRaises(ValueError):
            condition_key({'reflection_enabled': True})
        with self.assertRaises(ValueError):
            run_identity(make_job())
        self.assertEqual(len(run_identity({'reflection_enabled': True})), 6)

    def test_drafts_rejected_before_worker_side_effects(self):
        from experiments.agentm_reliability.run_matrix import run_worker
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'job.json'
            path.write_text(json.dumps(make_job()))
            with patch('experiments.agentm_reliability.run_matrix.ensure_output_layout') as layout:
                with self.assertRaises(PermissionError):
                    run_worker(path)
                layout.assert_not_called()

    def test_conversation_drafts_rejected_before_worker_side_effects(self):
        from experiments.agentm_reliability.multi_turn_runner import run_worker
        job = make_job(0)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'job.json'
            path.write_text(json.dumps(job))
            with patch('experiments.agentm_reliability.multi_turn_runner.ensure_output_layout') as layout:
                with self.assertRaises(PermissionError):
                    run_worker(path)
                layout.assert_not_called()

    def test_runtime_depth_and_hash_disagreement_rejected(self):
        job = make_job()
        job['config']['runtime']['max_reflection_rounds'] = 1
        with self.assertRaises(ValueError):
            validate_depth_job(job)

    def test_model_binding_rejects_changed_resource(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'model.pkl'
            path.write_bytes(b'fixture model bytes, never deserialized')
            job = make_job()
            job['case']['model_path'] = 'outputs/old/model.pkl'
            job['config']['runtime']['verified_case_model_binding'] = {
                'declared_path': job['case']['model_path'], 'path': str(path), 'sha256': file_sha256(path)}
            self.assertEqual(bound_model_path(job), str(path))
            path.write_bytes(b'changed')
            with self.assertRaises(ValueError):
                bound_model_path(job)

    def test_all_depths_snapshot_protocol(self):
        for depth in range(4):
            job, context, hook = complete_fixture(depth)
            self.assertTrue(audit_depth_protocol(job, context, hook.export())['passed'])
            self.assertEqual(len(hook.snapshots), depth * 4)

    def test_snapshots_do_not_alias_live_report_or_export(self):
        _, context, hook = complete_fixture(2)
        context['final_report']['text'] = 'mutated'
        exported = hook.export()
        self.assertEqual(exported['reflection_snapshots'][-1]['payload']['final_report']['text'], 'round 2')
        exported['reflection_snapshots'][-1]['payload']['final_report']['text'] = 'changed copy'
        self.assertEqual(hook.export()['reflection_snapshots'][-1]['payload']['final_report']['text'], 'round 2')

    def test_tampering_missing_feedback_and_stale_report_rejected(self):
        job, context, hook = complete_fixture(3)
        altered = hook.export()
        altered['reflection_snapshots'][0]['payload']['final_report']['text'] = 'tampered'
        self.assertFalse(audit_depth_protocol(job, context, altered)['passed'])
        context['tool_calls'].pop()
        self.assertFalse(audit_depth_protocol(job, context, hook.export())['passed'])
        job, context, hook = complete_fixture(3)
        context['metadata']['report_stale'] = True
        self.assertFalse(audit_depth_protocol(job, context, hook.export())['passed'])

    def records(self):
        return [{**make_job(d), 'protocol_accepted': True, 'scientific_success': False,
                 'model_resource_hash': 'model', 'evidence_index_hash': 'index', 'case_hash': 'case'} for d in range(4)]

    def test_pairing_keeps_scientific_failures(self):
        rows = paired_depth_rows(self.records())
        self.assertEqual(len(rows), 1)
        self.assertTrue(all(r['scientific_success'] is False for r in rows[0]['depth_records']))

    def test_pairing_rejects_missing_duplicate_and_mixed_provenance(self):
        records = self.records()
        for variant in (records[:-1], records + [records[0]]):
            with self.assertRaises(ValueError):
                paired_depth_rows(variant)
        records[2]['source_tree_hash'] = 'different_build'
        with self.assertRaises(ValueError):
            paired_depth_rows(records)


if __name__ == '__main__':
    unittest.main()
