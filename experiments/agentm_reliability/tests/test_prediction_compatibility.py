from copy import deepcopy
import unittest
from unittest.mock import Mock, patch
import numpy as np

from agentm.runtime.reaction import _classify_with_model
from agentm.runtime.tools import AgentAction, ToolResult
from experiments.agentm_reliability.injections import ControlledInjectionHook, apply_injection
from experiments.agentm_reliability.prediction_mutations import mutate_prediction, InjectionNotApplicable
from experiments.agentm_reliability.prediction_injection_audit import audit_prediction_injection
from experiments.agentm_reliability.regression_injection import numeric_candidates, regression_profile


class PredictionCompatibilityTests(unittest.TestCase):
    def profile(self):
        return {'iqr':2.,'minimum_separation':1.,'candidate_values':[-2.,-1.,0.,1.,2.],
                'unit':'dataset_native_target_units','profile_sha256':'test-profile',
                'protocol':'admet_regression_train_quantiles'}

    def prediction(self):
        return {'task_type':'admet_prediction','dataset_name':'test_regression',
                'label':.5,'score':.5,'probability':None,'model_type':'random_forest','details':{}}

    def test_reaction_vocabulary_comes_from_fitted_model(self):
        classifier=Mock(classes_=np.array([0,4,7]))
        classifier.predict.return_value=np.array([4])
        classifier.decision_function.return_value=np.array([[1.,2.,0.]])
        model={'artifact_type':'sklearn_pickle','artifact':{'vectorizer':Mock(),'model':classifier}}
        prediction=_classify_with_model({'reaction_smiles':'C>>C'},model)
        self.assertEqual(prediction['classes'],['0','4','7'])
        self.assertEqual(prediction['label'],'4')
        self.assertEqual(prediction['class_vocabulary_source'],'fitted_estimator.classes_')

    def test_reaction_training_examples_vocab_no_gold(self):
        model={'examples':[{'label':'A','tokens':['C']},{'label':'B','tokens':['N']}],'majority_label':'A'}
        self.assertEqual(_classify_with_model({'reaction_smiles':'C>>C'},model)['classes'],['A','B'])

    def test_numeric_regression_synced_and_audited(self):
        p=self.prediction(); before=deepcopy(p)
        with patch('experiments.agentm_reliability.prediction_mutations.regression_profile',return_value=self.profile()), \
             patch('experiments.agentm_reliability.prediction_injection_audit.regression_profile',return_value=self.profile()):
            c={'predictions':[p],'task_type':'admet_prediction'}
            m=apply_injection(c,'plausible_but_wrong_prediction',seed=9,truth_reference={'value':.8})
            hook={'case':{'injection_type':'plausible_but_wrong_prediction','gold_truth_reference':{'value':.8}},'mutation_record':m}
            self.assertTrue(audit_prediction_injection(hook)['passed'])
            self.assertEqual(c['predictions'][0]['score'],c['predictions'][0]['label'])
            self.assertIsNone(c['predictions'][0]['probability'])
            self.assertEqual(p,before)
            self.assertNotIn('minimum_separation',c['predictions'][0])
            c['predictions'][0]['label']=.8
            hook['mutation_record']['after']=deepcopy(c['predictions'])
            self.assertFalse(audit_prediction_injection(hook)['passed'])

    def test_regression_determinism(self):
        with patch('experiments.agentm_reliability.prediction_mutations.regression_profile',return_value=self.profile()):
            kwargs=dict(seed=12,truth_reference={'value':.8})
            self.assertEqual(mutate_prediction(self.prediction(),'plausible_but_wrong_prediction',**kwargs),
                             mutate_prediction(self.prediction(),'plausible_but_wrong_prediction',**kwargs))

    def test_regression_invalid_numbers_and_schema(self):
        for change in ({'label':float('nan')},{'score':float('inf')},{'score':.2},{'probability':.9},{'label':True}):
            with self.subTest(change=change),self.assertRaises(ValueError):
                numeric_candidates(dict(self.prediction(),**change),.8,self.profile())

    def test_regression_degenerate_scale_rejected(self):
        with self.assertRaises(ValueError):
            numeric_candidates(self.prediction(),.8,dict(self.profile(),minimum_separation=0.))

    def test_regression_exhausted_candidates_is_not_applicable(self):
        profile=dict(self.profile(),candidate_values=[.4,.5,.6])
        with patch('experiments.agentm_reliability.prediction_mutations.regression_profile',return_value=profile):
            with self.assertRaises(InjectionNotApplicable):
                mutate_prediction(self.prediction(),'plausible_but_wrong_prediction',seed=1,truth_reference={'value':.8})

    def test_profile_hash_mismatch_blocks(self):
        p=dict(self.prediction(),dataset_name='LD50_Zhu')
        with patch('experiments.agentm_reliability.regression_injection.file_sha256',return_value='changed'):
            with self.assertRaisesRegex(ValueError,'hash_mismatch'):
                regression_profile(p)

    def test_binary_not_applicable_is_not_counted_as_success(self):
        case={'injection_type':'plausible_but_wrong_prediction','gold_truth_reference':{'value':1}}
        hook=ControlledInjectionHook(case)
        p={'task_type':'admet_prediction','dataset_name':'DILI','label':0,'score':.2,'probability':.2}
        c={'task_type':'admet_prediction','predictions':[p],'metadata':{}}
        hook.after_tool(action=AgentAction('GaugeAgent','predict_single'),result=ToolResult('predict_single','completed'),context=c)
        self.assertEqual(c['predictions'],[p])
        self.assertFalse(hook.applied)
        self.assertEqual(hook.mutation_record['status'],'not_applicable')
        a=audit_prediction_injection(hook.export())
        self.assertTrue(a['not_applicable_valid'])
        self.assertFalse(a['passed'])
        self.assertFalse(a['eligible_for_injection_metrics'])
        self.assertFalse(a['count_as_success'])
        self.assertIn('controlled_injection_error',c['metadata'])
        hook.mutation_record['after'][0]['label']=1
        self.assertFalse(audit_prediction_injection(hook.export())['not_applicable_valid'])

    def test_forged_not_applicable_rejected(self):
        p={'label':1,'details':{'classes':[0,1]}}
        hook={'case':{'injection_type':'plausible_but_wrong_prediction','gold_truth_reference':{'value':1}},
              'mutation_record':{'status':'not_applicable','not_applicable_reason':'no_distinct_wrong_class_available',
                                 'changed':False,'before':[p],'after':[deepcopy(p)]}}
        self.assertFalse(audit_prediction_injection(hook)['not_applicable_valid'])


if __name__=='__main__':
    unittest.main()
