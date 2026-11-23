import unittest
from copy import deepcopy
from experiments.agentm_reliability.gold_rules import FAULT_TYPES
from experiments.agentm_reliability.replay_scale_tools import select_cases


class ScaleToolReplayTests(unittest.TestCase):
    def test_selection_is_deterministic_outcome_blind_and_balanced(self):
        stress = [{'case_id':f'{group}_{fault}', 'task_type':group, 'injection_type':fault}
                  for group in ('admet','ddi','reaction_prediction','reaction_evidence')
                  for fault in FAULT_TYPES
                  if group != 'reaction_evidence' or fault not in ('missing_field_or_overclaim','plausible_but_wrong_prediction','prediction_tool_or_service_failure')]
        multi = [{'case_id':f'{kind}_{i}', 'multi_turn_type':kind} for kind in ('add','resolve','pair','withdraw') for i in range(2)]
        a, b = select_cases(stress, multi)
        changed = deepcopy(stress)
        for row in changed:
            row['scientific_result'] = 'arbitrary_failure'
        c, d = select_cases(list(reversed(changed)), list(reversed(multi)))
        self.assertEqual([r['case_id'] for r in a], [r['case_id'] for r in c])
        self.assertEqual(b, d)
        self.assertEqual(len(a), 8)
        self.assertEqual(len({r['task_type'] for r in a}), 4)
        self.assertEqual(set(r['injection_type'] for r in a), set(FAULT_TYPES))

    def test_missing_strata_rejected(self):
        with self.assertRaises(ValueError):
            select_cases([], [])


if __name__ == '__main__':
    unittest.main()
