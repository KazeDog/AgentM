from __future__ import annotations

import unittest

from agentm.runtime.tools.scientific import ScientificToolRuntime


CAFFEINE = "Cn1cnc2n(C)c(=O)n(C)c(=O)c12"


class MultiTurnMoleculeArgumentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.runtime = ScientificToolRuntime.__new__(ScientificToolRuntime)

    def test_request_grounded_smiles_argument_precedes_bad_parser_mention(self) -> None:
        context = {
            "request": (
                f"For the same molecule {CAFFEINE}, additionally evaluate "
                "dili and hia."
            ),
            "molecule_mentions": [f"For the same {CAFFEINE}"],
        }

        mentions = self.runtime._mentions(
            {"smiles": CAFFEINE},
            context,
            target_count=1,
        )

        self.assertEqual(mentions, [CAFFEINE])

    def test_request_grounded_molecule_smiles_is_supported(self) -> None:
        context = {
            "request": f"Evaluate DILI for {CAFFEINE}",
            "molecule_mentions": ["Evaluate DILI"],
        }

        mentions = self.runtime._mentions(
            {"molecule_smiles": CAFFEINE},
            context,
            target_count=1,
        )

        self.assertEqual(mentions, [CAFFEINE])

    def test_scalar_plural_field_is_not_split_into_characters(self) -> None:
        context = {"request": f"Evaluate hERG for {CAFFEINE}"}

        mentions = self.runtime._mentions(
            {"molecules": CAFFEINE},
            context,
            target_count=1,
        )

        self.assertEqual(mentions, [CAFFEINE])

    def test_ungrounded_scalar_does_not_override_context_mention(self) -> None:
        context = {
            "request": "Evaluate hERG for metformin",
            "molecule_mentions": ["metformin"],
        }

        mentions = self.runtime._mentions(
            {"smiles": CAFFEINE},
            context,
            target_count=1,
        )

        self.assertEqual(mentions, ["metformin"])


if __name__ == "__main__":
    unittest.main()
