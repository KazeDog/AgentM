from copy import deepcopy
import json
import unittest

from agentm.runtime.compat import ensure_import_paths
ensure_import_paths()
from agentm.schema import PredictionResult
from agentm.runtime.tools import AgentAction, ToolResult
from experiments.agentm_reliability.common import stable_hash
from experiments.agentm_reliability.injections import apply_injection, ControlledInjectionHook
from experiments.agentm_reliability.prediction_injection_audit import audit_prediction_injection
from experiments.agentm_reliability.prediction_mutations import mutate_prediction
from experiments.agentm_reliability.reflection_depth import DepthSnapshotHook


class StructuredPredictionMutationTests(unittest.TestCase):
    def prediction(self):
        return PredictionResult('ddi_prediction','DrugBank',49,.91,.91,'sgd_classifier',
                                {'classes':[1,49,51],'label_mode':'multiclass_interaction_type'})

    def test_multiclass_label_confidence_and_derived_semantics(self):
        original = self.prediction()
        before = deepcopy(original)
        new, details = mutate_prediction(original,'plausible_but_wrong_prediction',seed=7,truth_reference={'value':49})
        self.assertIn(new.label,[1,51])
        self.assertEqual(new.score,.91)
        self.assertEqual(new.probability,.91)
        self.assertEqual(new.details['interaction_type_class'],new.label)
        self.assertEqual(original,before)
        self.assertNotIn('gold',new)
        self.assertEqual(new.to_dict(),dict(new))
        self.assertEqual(details['mutation_protocol'],'structured_prediction')

    def test_seed_reproducible_and_class_order_independent(self):
        p=self.prediction().to_dict()
        a,_=mutate_prediction(p,'plausible_but_wrong_prediction',seed=9,truth_reference={'value':49})
        p['details']['classes'].reverse()
        b,_=mutate_prediction(p,'plausible_but_wrong_prediction',seed=9,truth_reference={'value':49})
        self.assertEqual(a.label,b.label)

    def test_binary_already_wrong_fails_without_modifying_input(self):
        c={'task_type':'admet_prediction','predictions':[PredictionResult('admet_prediction','DILI',0,.8,.8)]}
        before=deepcopy(c)
        with self.assertRaisesRegex(ValueError,'no_distinct_wrong_class'):
            apply_injection(c,'plausible_but_wrong_prediction',truth_reference={'value':1})
        self.assertEqual(c,before)

    def test_missing_gold_and_repr_rejected(self):
        for value, truth in [(self.prediction(),None),('PredictionResult(label=49)',{'value':49})]:
            with self.assertRaises(ValueError):
                mutate_prediction(value,'plausible_but_wrong_prediction',seed=1,truth_reference=truth)

    def test_missing_field_only_removes_model_type(self):
        new,_=mutate_prediction(self.prediction(),'missing_field_or_overclaim',seed=1)
        expected=self.prediction().to_dict();expected.pop('model_type')
        self.assertEqual(new,expected)
        self.assertEqual(new.label,49)
        self.assertFalse(hasattr(new,'model_type'))

    def test_reaction_alias_is_synchronized(self):
        p={'task_type':'reaction_classification','label':'A','score':.9,'model_type':'linear','details':{'classes':['A','B','C'],'label':'A'}}
        context={'task_type':'reaction_classification','reaction_predictions':[p],'predictions':[p]}
        apply_injection(context,'plausible_but_wrong_prediction',truth_reference={'value':'A'})
        self.assertIs(context['predictions'][0],context['reaction_predictions'][0])
        self.assertNotEqual(context['predictions'][0]['label'],'A')

    def test_yield_numeric_and_category_consistent(self):
        p={'task_type':'reaction_yield_prediction','label':'high_yield','score':85.,'probability':None,'details':{'predicted_yield':85.}}
        c={'task_type':'reaction_yield_prediction','reaction_predictions':[p],'predictions':[p]}
        mutation=apply_injection(c,'plausible_but_wrong_prediction',truth_reference={'value':90.})
        audit=audit_prediction_injection({'case':{'injection_type':'plausible_but_wrong_prediction','gold_truth_reference':{'value':90.}},'mutation_record':mutation})
        self.assertTrue(audit['passed'],audit)

    def test_context_and_depth_snapshots_are_json_structured_and_immutable(self):
        case={'injection_type':'plausible_but_wrong_prediction','gold_truth_reference':{'value':49}}
        hook=DepthSnapshotHook(case,1)
        c={'task_type':'ddi_prediction','predictions':[self.prediction()],'metadata':{}}
        hook.after_tool(action=AgentAction('GaugeAgent','predict_ddi'),result=ToolResult('predict_ddi','completed'),context=c)
        self.assertTrue(hook.applied)
        self.assertIsInstance(hook.pre_injection_context['predictions'][0],dict)
        before=stable_hash(hook.export())
        c['predictions'][0]['label']=999
        self.assertEqual(stable_hash(hook.export()),before)
        hook._capture('round_input',1,c)
        exported=hook.export()
        loaded=json.loads(json.dumps(exported))
        item=loaded['reflection_snapshots'][0]
        self.assertEqual(item['sha256'],stable_hash({k:v for k,v in item.items() if k!='sha256'}))

    def test_failed_injection_is_not_marked_applied(self):
        hook=ControlledInjectionHook({'injection_type':'plausible_but_wrong_prediction'})
        c={'task_type':'ddi_prediction','predictions':[self.prediction()],'metadata':{}}
        args={'action':AgentAction('GaugeAgent','predict_ddi'),'result':ToolResult('predict_ddi','completed'),'context':c}
        hook.after_tool(**args)
        self.assertFalse(hook.applied)
        self.assertIn('independent_gold',hook.mutation_record['error'])
        original=deepcopy(hook.mutation_record)
        hook.after_tool(**args)
        self.assertEqual(hook.mutation_record,original)

    def test_invalid_injection_aborts_before_another_planner_call(self):
        from unittest.mock import Mock
        from agentm.runtime.graph.nodes import MultiAgentGraphNodes
        nodes=object.__new__(MultiAgentGraphNodes)
        nodes.orchestrator=Mock()
        with self.assertRaisesRegex(RuntimeError,'controlled_injection_invalid'):
            nodes.orchestrator_loop({'metadata':{'controlled_injection_error':'missing_valid_class_vocabulary'}})
        nodes.orchestrator.next_action.assert_not_called()

    def test_confidence_survives_guard_repair_to_plain_dictionary(self):
        from agentm.runtime.guards.verifier import repair_prediction_record_integrity
        from agentm.runtime.tools.scientific import ScientificToolRuntime
        p=self.prediction()
        c={'task_type':'ddi_prediction','predictions':[p], 'tool_calls':[{'tool':'predict_ddi','status':'completed','observation':{'prediction':p.to_dict()}}]}
        apply_injection(c,'missing_field_or_overclaim')
        self.assertTrue(repair_prediction_record_integrity(c)['repaired'])
        runtime=object.__new__(ScientificToolRuntime)
        runtime.assess_confidence({},c)
        self.assertEqual(c['confidence']['score'],.91)
        self.assertEqual(c['confidence']['model_type'],'sgd_classifier')
