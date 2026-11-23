from copy import deepcopy
import unittest
from unittest.mock import patch

from experiments.agentm_reliability.common import RELIABILITY_ROOT, load_jsonl, stable_hash
from experiments.agentm_reliability.gold_rules import FAULT_TYPES
from experiments.agentm_reliability.revise_scale_contracts import evidence_policy, identity_check, revise_multi


class ScaleContractRevisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = load_jsonl(RELIABILITY_ROOT / 'manifests/multi_turn_tasks.jsonl')

    def test_distinct_endpoints_and_input_immutable(self):
        old = next(x for x in self.rows if x['case_id'] == 'multi_turn_add_endpoints_00')
        before = deepcopy(old)
        new = revise_multi(old)
        self.assertEqual(old, before)
        self.assertEqual(len(set(new['gold_plan']['turn_level_expected_state'][-1]['endpoints'])), 3)
        self.assertNotEqual(new['case_id'], old['case_id'])
        self.assertEqual(new['parent_case_content_hash'], stable_hash(old))
        self.assertFalse(new['execution_authorized'])

    def test_name_to_structure_explicitly_same_entity(self):
        old = next(x for x in self.rows if x['case_id'] == 'multi_turn_resolve_specific_structure_00')
        new = revise_multi(old)
        self.assertTrue(new['contract_preflight']['passed'])
        self.assertIn('confirm_same_entity_structure', new['gold_plan']['required_objectives'])
        self.assertNotIn('replace_ambiguous_entity', new['gold_plan']['required_objectives'])

    def test_single_is_member_of_pair(self):
        for old in self.rows:
            if old['multi_turn_type'] != 'single_to_ddi':
                continue
            new = revise_multi(old)
            self.assertIn(new['turns'][0]['request'], [s.strip() for s in new['turns'][1]['request'].split(',')])
            self.assertEqual(new['gold_plan']['turn_level_expected_state'][0]['objects'], [new['turns'][0]['request']])

    def test_resolver_conflict_cannot_pass(self):
        old = next(x for x in self.rows if x['case_id'] == 'multi_turn_resolve_specific_structure_03')
        fake = {'name': 'metformin', 'dictionary_structure': 'C', 'seed_structure': 'CC',
                'runtime_resolved_structure': 'C', 'status': 'conflicting_local_maps', 'external_identity_verified': False}
        with patch('experiments.agentm_reliability.revise_scale_contracts.identity_check', return_value=fake):
            new = revise_multi(old)
        self.assertFalse(new['contract_preflight']['passed'])
        self.assertTrue(any('conflicting_local_maps' in x for x in new['contract_preflight']['blockers']))

    def test_missing_identity_cannot_pass(self):
        with patch('socket.socket.connect', side_effect=AssertionError('no network')):
            identity = identity_check('unregistered-molecule-test-00000000')
        self.assertEqual(identity['status'], 'missing_local_mapping')
        self.assertFalse(identity['external_identity_verified'])

    def test_real_runtime_applies_seed_after_dictionary(self):
        from experiments.agentm_reliability.audit_scale_contracts import resolved_key
        from experiments.agentm_reliability.prepare_scale_reflection import entity_key
        self.assertEqual(resolved_key('metformin'), entity_key('CN(C)C(=N)N=C(N)N'))
        identity = identity_check('metformin')
        self.assertTrue(identity['dictionary_seed_disagreement'])
        self.assertEqual(identity['effective_resolution_stage'], 'post_processor_seed_override')
        self.assertEqual(identity['status'], 'local_mapping_available')

    def test_isoniazid_resolves_offline_and_matches_independent_source(self):
        import csv
        from experiments.agentm_reliability.audit_scale_contracts import resolved_key
        from experiments.agentm_reliability.prepare_scale_reflection import entity_key
        from experiments.agentm_reliability.common import PROJECT_ROOT, file_sha256
        source = PROJECT_ROOT / 'data/moleculenet/raw/BBBP.csv'
        self.assertEqual(file_sha256(source), 'd07a38487aeac5cee5508413e468043ef3097451d2a112701c2d60be9ec6b662')
        with source.open() as stream:
            matches = [r for r in csv.DictReader(stream) if r['name'].lower() == 'isoniazid']
        self.assertEqual(len(matches), 1)
        self.assertEqual(resolved_key('isoniazid'), entity_key(matches[0]['smiles']))

    def test_evidence_policy_preserves_scientific_boundary(self):
        policies = evidence_policy()
        self.assertEqual(set(policies), set(FAULT_TYPES))
        for policy in policies.values():
            self.assertFalse(policy['evaluation_rule_changed'])
            self.assertIn('not_qc_or_synthetic_injection', policy['scientific_claim_gate'])
            self.assertFalse(policy['direct_external_evidence_required_for_protocol_metric'])
        self.assertTrue(policies['no_direct_evidence_or_ambiguous_entity']['absence_is_intended_condition'])
        self.assertIn('independent_heldout_label_available', policies['plausible_but_wrong_prediction']['controlled_protocol_prerequisites'])


if __name__ == '__main__':
    unittest.main()
