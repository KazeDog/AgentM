import unittest
from copy import deepcopy
from agentm.runtime.compat import ensure_import_paths
ensure_import_paths()
from agentm.schema import PredictionResult
from experiments.agentm_reliability.injections import apply_injection
from experiments.agentm_reliability.prediction_injection_audit import audit_prediction_injection


class PredictionInjectionAuditTests(unittest.TestCase):
    def hook(self, fault, prediction):
        context = {'task_type':'ddi_prediction', 'predictions':[prediction]}
        mutation = apply_injection(context, fault, truth_reference={'value':49})
        return {'case':{'injection_type':fault, 'gold_truth_reference':{'value':49}}, 'mutation_record':mutation}

    def test_fixed_dataclass_wrong_prediction_passes_contract(self):
        prediction = PredictionResult('ddi_prediction', 'DrugBank', 49, .9, .9, 'sgd_classifier', {'classes':[1,49,51]})
        audit = audit_prediction_injection(self.hook('plausible_but_wrong_prediction', prediction))
        self.assertTrue(audit['passed'], audit)

    def test_fixed_dataclass_field_removal_passes_contract(self):
        prediction = PredictionResult('admet_prediction', 'DILI', 1, .9, .9, 'random_forest')
        audit = audit_prediction_injection(self.hook('missing_field_or_overclaim', prediction))
        self.assertTrue(audit['passed'], audit)

    def test_legacy_broken_wrapper_still_rejected(self):
        hook = {'case':{'injection_type':'plausible_but_wrong_prediction','gold_truth_reference':{'value':49}},
                'mutation_record':{'before':["PredictionResult(label=49)"],
                                   'after':[{'value':'plausible_alternative::PredictionResult(label=49)'}]}}
        audit = audit_prediction_injection(hook)
        self.assertFalse(audit['passed'])
        self.assertIn('mutated_prediction_missing_structured_label', audit['errors'])

    def test_rejects_serialized_repr_in_snapshot(self):
        hook = self.hook('missing_field_or_overclaim', {'label':1, 'score':.9})
        hook['mutation_record']['before'] = ["PredictionResult(label=1)"]
        self.assertIn('baseline_prediction_snapshot_not_structured', audit_prediction_injection(hook)['errors'])

    def test_valid_single_field_removal(self):
        self.assertTrue(audit_prediction_injection(self.hook('missing_field_or_overclaim', {'label':1,'score':.9,'model_type':'test'}))['passed'])

    def test_wrong_multiclass_label_must_be_in_vocab_and_not_gold(self):
        first = {'label':1,'details':{'classes':[1,49,51]}}
        last = deepcopy(first)
        hook = {'case':{'injection_type':'plausible_but_wrong_prediction','gold_truth_reference':{'value':49}},
                'mutation_record':{'before':[first], 'after':[last]}}
        last['label'] = 99
        self.assertIn('injected_prediction_outside_class_vocabulary', audit_prediction_injection(hook)['errors'])
        last['label'] = 49
        self.assertIn('injected_prediction_equals_gold_label', audit_prediction_injection(hook)['errors'])
        last['label'] = 51
        self.assertTrue(audit_prediction_injection(hook)['passed'])


if __name__ == '__main__':
    unittest.main()
