from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.agentm_reliability.common import file_sha256, stable_hash
from experiments.agentm_reliability.reflection_depth import depth_job, validate_depth_job
from experiments.agentm_reliability.queue_applicability import PROTOCOL, DIAGNOSTIC_SCOPE


class QueueApplicabilityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.case = {'case_id': 'fixture', 'dataset_group': 'stress', 'request': 'unchanged'}
        self.manifest = self.root / 'manifest.jsonl'
        self.manifest.write_text(json.dumps(self.case) + '\n')
        self.ledger_path = self.root / 'ledger.json'
        self.ledger = {'protocol': PROTOCOL, 'source_tree_hash': 'source',
            'manifest_hashes': {str(self.manifest): file_sha256(self.manifest)},
            'formal_freeze_ready': False, 'diagnostic_probe_permitted': True,
            'cases': [{'case_id': 'fixture', 'case_sha256': stable_hash(self.case),
                       'status': 'pending_probe', 'formal_eligible': False}]}
        config = {'experiment': {'seed': 2026}, 'runtime': {
            'use_llm_reasoning': False, 'require_llm_reasoning': False,
            'require_llm_orchestrator': False, 'llm_profile': None}}
        self.job = depth_job(self.case, config, orchestrator='heuristic', depth=0,
                             source_hash='source', manifest=self.manifest)
        self.job.update(execution_authorized=True, formal_statistics_eligible=False,
                        authorization_scope=DIAGNOSTIC_SCOPE)
        self.bind()
        self.source_patch = patch('experiments.agentm_reliability.queue_applicability.source_tree_hash', return_value='source')
        self.source_patch.start()
        self.addCleanup(self.source_patch.stop)

    def bind(self):
        self.ledger_path.write_text(json.dumps(self.ledger))
        self.job['applicability_binding'] = {'path': str(self.ledger_path), 'sha256': file_sha256(self.ledger_path)}

    def rejected(self, job=None):
        with self.assertRaises(PermissionError):
            validate_depth_job(job or self.job, executing=True)

    def test_pending_zero_API_diagnostic_allowed(self):
        validate_depth_job(self.job, executing=True)

    def test_missing_ledger_rejected_even_when_authorized(self):
        self.job.pop('applicability_binding')
        self.rejected()

    def test_NA_and_blocked_rejected_in_all_arms(self):
        for status in ('not_applicable', 'blocked', 'unknown'):
            self.ledger['cases'][0]['status'] = status
            self.bind()
            for depth in range(4):
                for orchestrator in ('heuristic', 'llm'):
                    job = depth_job(self.case, self.job['config'], orchestrator=orchestrator,
                                    depth=depth, source_hash='source', manifest=self.manifest)
                    job.update({k: deepcopy(self.job[k]) for k in ('execution_authorized',
                        'formal_statistics_eligible', 'authorization_scope', 'applicability_binding')})
                    self.rejected(job)

    def test_changed_ledger_rejected(self):
        self.ledger_path.write_text('{}')
        self.rejected()

    def test_source_mismatch_rejected(self):
        self.ledger['source_tree_hash'] = 'other'
        self.bind()
        self.rejected()

    def test_manifest_mismatch_rejected(self):
        self.manifest.write_text('{}\n')
        self.rejected()

    def test_changed_case_content_rejected(self):
        self.job['case']['request'] = 'changed'
        from experiments.agentm_reliability.common import run_id_for
        self.job['run_id'] = run_id_for(self.job, self.job['config_hash'])
        self.rejected()

    def test_duplicate_or_missing_case_rejected(self):
        row = deepcopy(self.ledger['cases'][0])
        for cases in ([], [row, row]):
            self.ledger['cases'] = cases
            self.bind()
            self.rejected()

    def test_diagnostic_cannot_enable_API_or_formal_statistics(self):
        variants = [({'orchestrator': 'llm'}, {}), ({'formal_statistics_eligible': True}, {}),
                    ({'authorization_scope': 'formal'}, {}),
                    ({}, {'max_llm_calls_per_run': 1}), ({}, {'max_total_tokens_per_run': 1}),
                    ({}, {'use_llm_reasoning': True}), ({}, {'llm_profile': 'enabled'})]
        from experiments.agentm_reliability.common import run_id_for
        for fields, runtime in variants:
            job = deepcopy(self.job)
            job.update(fields)
            job['config']['runtime'].update(runtime)
            job['config_hash'] = stable_hash(job['config'])
            job['run_id'] = run_id_for(job, job['config_hash'])
            self.rejected(job)

    def test_component_eligible_not_formally_frozen_rejected(self):
        self.ledger['cases'][0]['status'] = 'eligible'
        self.bind()
        self.job['authorization_scope'] = 'formal'
        self.rejected()

    def test_both_workers_reject_NA_before_any_output(self):
        self.ledger['cases'][0]['status'] = 'not_applicable'
        self.bind()
        path = self.root / 'job.json'
        path.write_text(json.dumps(self.job))
        from experiments.agentm_reliability import run_matrix, multi_turn_runner
        for module in (run_matrix, multi_turn_runner):
            with patch.object(module, 'ensure_output_layout') as layout:
                with self.assertRaises(PermissionError):
                    module.run_worker(path)
                layout.assert_not_called()

    def test_read_only_historical_audit_does_not_require_new_binding(self):
        self.job.pop('applicability_binding')
        validate_depth_job(self.job, executing=False)


if __name__ == '__main__':
    unittest.main()
