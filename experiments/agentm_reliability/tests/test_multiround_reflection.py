"""Offline protocol tests: no controller, model initialization or API client."""
from copy import deepcopy
import unittest
from unittest.mock import patch

from agentm.runtime.agents.orchestrator import DeterministicMultiAgentPlanner
from agentm.runtime.tools.scientific import ScientificToolRuntime
from agentm.runtime.agents.orchestrator import OrchestratorAgent
from . import test_orchestrator_factorial as factorial


class MultiRoundReflectionTests(unittest.TestCase):
    def setUp(self):
        self.guard = patch('socket.socket.connect', side_effect=AssertionError('Network forbidden in offline tests'))
        self.guard.start()
        self.addCleanup(self.guard.stop)
        self.planner = DeterministicMultiAgentPlanner()
        self.runtime = object.__new__(ScientificToolRuntime)
        self.runtime.min_evidence = 1
        self.runtime.use_llm_reasoning = False
        self.runtime.reasoning_llm_client = None

    def context(self, rounds=3):
        return {
            'request': 'offline protocol fixture', 'task_type': 'admet_prediction',
            'final_report': {'request': 'offline protocol fixture'},
            'verifier_report': {'summary': {'status': 'pass'}},
            'evidence': [{'source': 'fixture', 'citation_id': 'C1'}],
            'metadata': {'enable_reflection': rounds > 0, 'max_reflection_rounds': rounds,
                         'mode': 'deep', 'max_rounds': 64, 'use_llm_reasoning': False},
        }

    def test_each_round_must_apply_before_next_reflection(self):
        for rounds in (1, 2, 3):
            with self.subTest(rounds=rounds):
                context = self.context(rounds)
                for number in range(1, rounds + 1):
                    self.assertTrue(self.planner._should_reflect(context))
                    self.runtime.reflect_on_report({}, context)
                    self.assertEqual(context['reflection']['round'], number)
                    self.assertFalse(self.planner._should_reflect(context))
                    self.assertTrue(self.planner._should_apply_reflection(context))
                    self.runtime.apply_reflection_feedback({}, context)
                    self.assertEqual(context['metadata']['reflection_feedback_applied_round'], number)
                    self.assertFalse(self.planner._should_apply_reflection(context))
                self.assertFalse(self.planner._should_reflect(context))

    def test_new_round_rearms_retrieval(self):
        context = self.context()
        context['metadata']['reflection_retrieval_done'] = True
        context['reflection'] = {'round': 2, 'requires_retrieval_refinement': True}
        self.runtime.apply_reflection_feedback({}, context)
        self.assertTrue(self.planner._needs_reflection_retrieval(context))

    def test_deferred_round_does_not_keep_old_request(self):
        context = self.context()
        context['metadata'].update(reflection_retrieval_requested=True, reflection_retrieval_done=False,
                                   retrieval_outcome='no_direct_evidence')
        context['reflection'] = {'round': 2, 'requires_retrieval_refinement': True}
        self.runtime.apply_reflection_feedback({}, context)
        self.assertFalse(self.planner._needs_reflection_retrieval(context))

    def test_zero_rounds_never_reflect(self):
        self.assertFalse(self.planner._should_reflect(self.context(0)))

    def test_retrieval_and_refresh_must_finish_before_next_round(self):
        for flag in ('reasoning_stale', 'report_stale', 'citation_verification_stale', 'schema_verification_stale'):
            context = self.context()
            context['metadata'][flag] = True
            original = deepcopy(context)
            self.assertFalse(self.planner._should_reflect(context), flag)
            self.assertEqual(context, original)
        context = self.context()
        context['metadata']['reflection_retrieval_requested'] = True
        self.assertFalse(self.planner._should_reflect(context))

    def test_last_round_cannot_finalize_before_report_refresh(self):
        orchestrator = OrchestratorAgent(registry=factorial._registry(), use_llm=True, llm_client=factorial._WriteArtifactsClient())
        context = factorial.OrchestratorFactorialConditionTests()._verified_reflection_context()
        context['metadata'].update(max_reflection_rounds=3, reflection_rounds=3,
                                   reflection_feedback_applied_round=3, report_stale=True)
        context['reflection'] = {'round': 3}
        self.assertFalse(orchestrator._reflection_protocol_complete(context))
        action = orchestrator.next_action(context)
        self.assertEqual(action.tool, 'build_runtime_report')

    def test_planner_replays_all_rounds_in_order(self):
        for rounds in (0, 1, 2, 3):
            for use_llm in (False, True):
                with self.subTest(rounds=rounds, mocked_llm=use_llm):
                    orchestrator = OrchestratorAgent(registry=factorial._registry(), use_llm=use_llm, llm_client=factorial._WriteArtifactsClient())
                    context = factorial.OrchestratorFactorialConditionTests()._verified_reflection_context()
                    context['metadata'].update(max_reflection_rounds=rounds, enable_reflection=rounds > 0,
                                               max_rounds=64, mode='full')
                    context['verifier_report']['summary']['status'] = 'warning'
                    sequence = []
                    for _ in range(40):
                        action = orchestrator.next_action(context)
                        if action.tool in ('finish', 'write_artifacts'):
                            break
                        sequence.append(action.tool)
                        if action.tool in ('reflect_on_report', 'apply_reflection_feedback'):
                            getattr(self.runtime, action.tool)({}, context)
                        elif action.tool not in ('build_runtime_report', 'verify_report_schema'):
                            self.fail(f'Unexpected fixture action: {action.tool}')
                        context['tool_calls'].append({'tool': action.tool, 'agent': action.agent, 'status': 'completed'})
                    else:
                        self.fail('Planner did not terminate within offline step cap')
                    expected = ['reflect_on_report', 'apply_reflection_feedback', 'build_runtime_report', 'verify_report_schema'] * rounds
                    self.assertEqual(sequence, expected)
                    self.assertTrue(orchestrator._reflection_protocol_complete(context))


if __name__ == '__main__':
    unittest.main()
