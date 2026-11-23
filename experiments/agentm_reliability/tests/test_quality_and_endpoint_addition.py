from copy import deepcopy
import unittest
from unittest.mock import Mock

from agentm.runtime.task_contracts import endpoint_addition_contract, endpoint_identity
from agentm.runtime.controller import AgentMMultiAgentController
from agentm.runtime.agents.orchestrator import DeterministicMultiAgentPlanner, LLMOrchestratorPlanner
from agentm.runtime.tools.registry import MultiAgentToolRegistry
from agentm.runtime.state import initial_context
from experiments.agentm_reliability.quality_evaluator import (
    prediction_matches, evaluate_open_task, evaluate_multi_turn)


def history():
    return {'request':'CCO','task_type':'admet_prediction','dataset_name':'hERG',
            'molecules':[{'original_input':'CCO','canonical_smiles':'CCO','name':'ethanol','is_valid':True}],
            'predictions':[{'task_type':'admet_prediction','dataset_name':'hERG','label':1}]}


def contract(**changes):
    args = dict(request='For the same molecule OCC, additionally evaluate dili and hia.',
                task_type='admet_prediction',dataset_name='DILI',previous=history())
    args.update(changes)
    return endpoint_addition_contract(**args)


def target():
    return {'task_type':'admet_prediction','dataset_name':'bbbp','molecules':['CCO'],'subtask_id':'bbb1'}


def prediction():
    return {'task_type':'admet_prediction','dataset_name':'BBB_Martins','label':1,
            'details':{'open_task_subtask_id':'bbb1','open_task_molecules':['OCC']}}


class EndpointAdditionTests(unittest.TestCase):
    def test_three_endpoints_recomputed_without_gold_or_history_copy(self):
        c = contract()
        self.assertEqual([s['dataset_name'] for s in c['subtasks']], ['hERG','DILI','HIA_Hou'])
        self.assertNotIn('predictions',c)
        self.assertNotIn('gold',str(c))
        self.assertIn('recompute_all',c['history_policy'])

    def test_unknown_endpoint_fails_closed(self):
        with self.assertRaises(ValueError): contract(request='For the same molecule CCO, additionally evaluate dili and invented.')

    def test_changed_molecule_rejected(self):
        with self.assertRaises(ValueError): contract(request='For the same molecule CCN, additionally evaluate dili and hia.')

    def test_contradictory_root_rejected(self):
        with self.assertRaises(ValueError): contract(dataset_name='AMES')

    def test_no_history_rejected(self):
        with self.assertRaises(ValueError): contract(previous=None)

    def test_unknown_grammar_not_silently_single_endpoint(self):
        with self.assertRaises(ValueError): contract(request='Please additionally evaluate everything.')

    def test_unknown_prior_endpoint_rejected(self):
        p=history();p['predictions'][0]['dataset_name']='unknown'
        with self.assertRaises(ValueError): contract(previous=p)

    def test_wrong_prior_object_scope_rejected(self):
        p=history();p['predictions'][0]['details']={'open_task_molecules':['CCN']}
        with self.assertRaises(ValueError): contract(previous=p)

    def test_normal_withdrawal_unchanged(self):
        self.assertIsNone(contract(request='Withdraw the previous target and assess CCN.'))

    def test_duplicate_endpoint_deduplicated(self):
        c=contract(request='For the same molecule CCO, additionally evaluate dili and DILI.')
        self.assertEqual(len(c['subtasks']),2)

    def test_empty_endpoint_no_default(self):
        self.assertIsNone(endpoint_identity(''))
        self.assertIsNone(endpoint_identity(None))

    def test_controller_derives_contract_before_graph_without_model_or_api(self):
        controller=AgentMMultiAgentController.__new__(AgentMMultiAgentController)
        for key in ('llm_profile','model_registry_path','previous_run_path'):
            setattr(controller,key,None)
        for key in ('use_llm_orchestrator','enable_fallbacks','enable_reflection','auto_model',
                    'require_real_model','require_real_rag','require_llm_orchestrator','use_llm_reasoning',
                    'require_llm_reasoning','use_external_evidence'):
            setattr(controller,key,False)
        controller.mode='full';controller.retrieval_backend='es';controller.max_rounds=64
        controller.max_reflection_rounds=0;controller.llm_orchestrator_max_retries=0
        controller.llm_orchestrator_retry_backoff_seconds=0;controller.use_rag_evidence=True;controller.min_evidence=1
        controller.graph=Mock();controller.graph.invoke.side_effect=lambda context:context
        controller.run('For the same molecule CCO, additionally evaluate dili and hia.',
                       task_type='admet_prediction',dataset_name='DILI',previous_run_context=history())
        context=controller.graph.invoke.call_args.args[0]
        self.assertEqual(len(context['metadata']['task_contract']['subtasks']),3)
        self.assertEqual(context['predictions'],[])
        self.assertEqual(context['previous_run_context'],history())

    def _simulate(self, llm):
        class ForbiddenClient:
            available=True
            def chat_json(self,*a,**kw):raise AssertionError('No API permitted')
        ctx=initial_context('same molecule',task_type='admet_prediction',dataset_name='DILI',
                            max_rounds=64,enable_reflection=False)
        ctx['metadata']['task_contract']=contract()
        planner=(LLMOrchestratorPlanner(registry=MultiAgentToolRegistry(),llm_client=ForbiddenClient(),temperature=0)
                 if llm else DeterministicMultiAgentPlanner())
        predictions=[]
        for i in range(64):
            action=planner.next_action(ctx)
            if action.tool=='finish':break
            if action.tool=='predict_single':
                predictions.append(action.args['dataset_name'])
                ctx['predictions'].append({'task_type':'admet_prediction','dataset_name':action.args['dataset_name'],
                    'label':1,'details':{'open_task_subtask_id':action.args['subtask_id']}})
            if action.tool=='build_runtime_report':ctx['final_report']={'predictions':deepcopy(ctx['predictions'])}
            ctx['tool_calls'].append({'agent':action.agent,'tool':action.tool,'args':dict(action.args),'status':'completed'})
        self.assertEqual(predictions,['hERG','DILI','HIA_Hou'])
        self.assertEqual(action.tool,'finish')
        self.assertLessEqual(i,24)

    def test_heuristic_simulates_all_three_endpoints(self):self._simulate(False)
    def test_llm_protocol_simulates_all_three_without_client_call(self):self._simulate(True)


class StructuredQualityTests(unittest.TestCase):
    def test_endpoint_alias_scoped_prediction_present(self):
        self.assertTrue(prediction_matches(prediction(),target(),{},scoped=True))

    def test_wrong_task_rejected(self):
        p=prediction();p['task_type']='ddi_prediction'
        self.assertFalse(prediction_matches(p,target(),{},scoped=True))

    def test_missing_dataset_rejected(self):
        p=prediction();p.pop('dataset_name')
        self.assertFalse(prediction_matches(p,target(),{},scoped=True))

    def test_wrong_endpoint_rejected(self):
        p=prediction();p['dataset_name']='hERG'
        self.assertFalse(prediction_matches(p,target(),{},scoped=True))

    def test_other_molecule_same_endpoint_rejected(self):
        p=prediction();p['details']['open_task_molecules']=['CCN']
        self.assertFalse(prediction_matches(p,target(),{},scoped=True))

    def test_wrong_subtask_rejected(self):
        p=prediction();p['details']['open_task_subtask_id']='another'
        self.assertFalse(prediction_matches(p,target(),{},scoped=True))

    def test_missing_output_rejected(self):
        p=prediction();p.pop('label')
        self.assertFalse(prediction_matches(p,target(),{},scoped=True))

    def _state(self, stale, *, same=False):
        case={'multi_turn_type':'resolve_specific_structure' if same else 'withdraw_and_replace',
              'turns':[{},{}], 'gold_plan':{'expected_final_scientific_objects':['CCO'],
                    'stale_records_to_remove':stale,'required_tools':[]}}
        if same:case['task_semantics']='same_entity_name_to_structure_confirmation_not_proven_ambiguity_resolution'
        ctx=history();ctx['request']='CCO';ctx['orchestrator_plan']={'steps':[]}
        return case,ctx

    def test_names_key_does_not_imply_ames_residue(self):
        case,ctx=self._state(['ames']);ctx['evidence']=[{'drug_names':['ethanol']}]
        r=evaluate_multi_turn(case=case,turn_contexts=[history(),ctx],final_report=ctx)
        self.assertFalse(r['legacy']['state_consistency'])
        self.assertTrue(r['active_object_endpoint_consistency'])

    def test_real_old_endpoint_is_not_ignored(self):
        case,ctx=self._state(['ames']);ctx['predictions'][0]['dataset_name']='AMES'
        self.assertFalse(evaluate_multi_turn(case=case,turn_contexts=[history(),ctx],final_report=ctx)['active_object_endpoint_consistency'])

    def test_descriptor_not_old_endpoint(self):
        case,ctx=self._state(['lipophilicity']);ctx['reasoning']={'text':'moderate lipophilicity'}
        self.assertTrue(evaluate_multi_turn(case=case,turn_contexts=[history(),ctx],final_report=ctx)['active_object_endpoint_consistency'])

    def test_same_entity_name_allowed_with_structure_proof(self):
        case,ctx=self._state(['ethanol'],same=True)
        self.assertTrue(evaluate_multi_turn(case=case,turn_contexts=[history(),ctx],final_report=ctx)['active_object_endpoint_consistency'])

    def test_same_name_wrong_structure_fails(self):
        case,ctx=self._state(['ethanol'],same=True);ctx['molecules'][0]['canonical_smiles']='CCN'
        self.assertFalse(evaluate_multi_turn(case=case,turn_contexts=[history(),ctx],final_report=ctx)['active_object_endpoint_consistency'])

    def test_stale_object_in_evidence_fails(self):
        case,ctx=self._state(['CCN']);ctx['evidence']=[{'claim':'CCN still supports this prediction'}]
        self.assertFalse(evaluate_multi_turn(case=case,turn_contexts=[history(),ctx],final_report=ctx)['active_object_endpoint_consistency'])

    def test_history_is_not_final_target_coverage(self):
        case,ctx=self._state([]);case['multi_turn_type']='add_two_endpoints'
        case['gold_plan']['turn_level_expected_state']=[{}, {'endpoints':['herg','dili','hia']}]
        ctx['previous_run_context']={'predictions':[{'task_type':'admet_prediction','dataset_name':'DILI','label':1}]}
        r=evaluate_multi_turn(case=case,turn_contexts=[history(),ctx],final_report=ctx)
        self.assertEqual(r['scientific_target_coverage'],1/3)
        self.assertFalse(r['active_object_endpoint_consistency'])

    def test_report_must_contain_prediction_too(self):
        from experiments.agentm_reliability.quality_evaluator import target_present
        ctx={'predictions':[prediction()]}
        self.assertFalse(target_present(target(),ctx,{},scoped=True))

    def test_named_expected_object_needs_independent_mapping(self):
        case,ctx=self._state([]);case['gold_plan']['expected_final_scientific_objects']=['ethanol']
        result=evaluate_multi_turn(case=case,turn_contexts=[history(),ctx],final_report=ctx)
        self.assertFalse(result['evaluation_valid'])
        self.assertIsNone(result['active_object_endpoint_consistency'])

    def test_named_target_with_audited_mapping(self):
        case,ctx=self._state([]);case['gold_plan']['expected_final_scientific_objects']=['ethanol']
        result=evaluate_multi_turn(case=case,turn_contexts=[history(),ctx],final_report=ctx,identity_map={'ethanol':'CCO'})
        self.assertTrue(result['evaluation_valid'])
        self.assertTrue(result['active_object_endpoint_consistency'])

    def test_named_target_cannot_trust_report_self_assertion(self):
        case,ctx=self._state([]);case['gold_plan']['expected_final_scientific_objects']=['ethanol']
        ctx['molecules'][0]['canonical_smiles']='CCN'
        result=evaluate_multi_turn(case=case,turn_contexts=[history(),ctx],final_report=ctx,identity_map={'ethanol':'CCO'})
        self.assertFalse(result['active_object_endpoint_consistency'])


if __name__=='__main__':unittest.main()
