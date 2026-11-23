from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
CASE_FILE = ROOT / "configs" / "molecular_safety_scientific_cases.jsonl"


class ScientificCasefileTests(unittest.TestCase):
    def test_casefile_contains_only_held_out_gold_cases(self):
        rows = [
            json.loads(line)
            for line in CASE_FILE.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.assertEqual(len(rows), 100)
        self.assertEqual(
            Counter(row["category"] for row in rows),
            {"admet": 50, "ddi": 50},
        )
        self.assertTrue(all(row["gold_split"] == "test" for row in rows))
        self.assertTrue(all(row.get("gold_value") is not None for row in rows))
        self.assertTrue(
            all(
                row["scientific_mode"] in {"classification", "regression"}
                for row in rows
            )
        )

    def test_ddi_cases_are_ordered_double_cold_cases(self):
        rows = [
            json.loads(line)
            for line in CASE_FILE.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        ddi = [row for row in rows if row["category"] == "ddi"]
        self.assertEqual(len(ddi), 50)
        self.assertTrue(all(row["split_strategy"] == "cold_drug" for row in ddi))
        self.assertTrue(all(row["training_drug_overlap_count"] == 0 for row in ddi))
        self.assertTrue(
            all("cold_drug_all_features/model.pkl" in row["model_path"] for row in ddi)
        )


if __name__ == "__main__":
    unittest.main()
