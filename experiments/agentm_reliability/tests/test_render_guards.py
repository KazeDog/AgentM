from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import pandas as pd

from experiments.agentm_reliability.render_outputs import _primary_clean, _setting_summary, _write_tables


class RenderGuardTests(unittest.TestCase):
    def test_primary_clean_excludes_stability_repeats(self) -> None:
        frame = pd.DataFrame(
            [
                {"run_id": "primary", "analysis_role": "primary", "repeat_index": 0},
                {"run_id": "repeat", "analysis_role": "stability_repeat", "repeat_index": 1},
            ]
        )
        selected = _primary_clean(frame)
        self.assertEqual(selected["run_id"].tolist(), ["primary"])

    def test_partial_table_contains_provisional_banner(self) -> None:
        empty = pd.DataFrame()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tables = root / "tables"
            tables.mkdir()
            _write_tables(
                {"tables": tables},
                {"clean": empty, "stress": empty, "open_task": empty, "multi_turn": empty},
                empty,
                protocol_note="PROVISIONAL PARTIAL-DATA OUTPUT: test",
            )
            text = (tables / "table_clean_regression.md").read_text(encoding="utf-8")
        self.assertIn("PROVISIONAL PARTIAL-DATA OUTPUT", text)

    def test_stress_summary_uses_metric_specific_applicability_denominators(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "case_id": "recoverable",
                    "orchestrator": "heuristic",
                    "reflection_enabled": False,
                    "repaired": True,
                    "repair_applicable": True,
                    "safe_abstention": False,
                    "safe_abstention_applicable": False,
                },
                {
                    "case_id": "unrecoverable",
                    "orchestrator": "heuristic",
                    "reflection_enabled": False,
                    "repaired": False,
                    "repair_applicable": False,
                    "safe_abstention": True,
                    "safe_abstention_applicable": True,
                },
            ]
        )
        row = _setting_summary(frame, ["repaired", "safe_abstention"]).iloc[0]
        self.assertEqual(row["repaired n"], 1)
        self.assertEqual(row["repaired"], 1.0)
        self.assertEqual(row["safe_abstention n"], 1)
        self.assertEqual(row["safe_abstention"], 1.0)


if __name__ == "__main__":
    unittest.main()
