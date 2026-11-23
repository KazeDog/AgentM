from __future__ import annotations

import unittest
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from agentm.runtime.moleculenet import (
    MOLECULENET_SPECS,
    classification_metrics,
    orient_clintox_risk_labels,
    regression_metrics,
    scaffold_groups,
    split_indices,
    summarize_results,
)


class MoleculeNetBenchmarkTest(unittest.TestCase):
    def test_registry_covers_requested_datasets(self):
        self.assertEqual(
            set(MOLECULENET_SPECS),
            {
                "bace",
                "bbbp",
                "clintox",
                "tox21",
                "toxcast",
                "sider",
                "lipophilicity",
                "freesolv",
                "esol",
            },
        )
        self.assertEqual(len(MOLECULENET_SPECS["tox21"].tasks or ()), 12)
        self.assertEqual(len(MOLECULENET_SPECS["sider"].tasks or ()), 27)
        self.assertEqual(MOLECULENET_SPECS["freesolv"].tasks, ("expt",))
        self.assertEqual(MOLECULENET_SPECS["freesolv"].target_units, "kcal/mol")
        self.assertIsNone(MOLECULENET_SPECS["toxcast"].tasks)

    def test_scaffold_splits_are_disjoint_and_deterministic(self):
        smiles = [
            "c1ccccc1",
            "Cc1ccccc1",
            "Oc1ccccc1",
            "C1CCCCC1",
            "CC1CCCCC1",
            "c1ccncc1",
            "CC",
            "CCC",
            "CCCC",
            "CCO",
            "CCN",
            "CO",
        ]
        for split_type in ("scaffold", "random_scaffold"):
            first = split_indices(smiles, split_type, seed=13, frac_train=0.6, frac_valid=0.2)
            second = split_indices(smiles, split_type, seed=13, frac_train=0.6, frac_valid=0.2)
            for part in ("train", "valid", "test"):
                np.testing.assert_array_equal(first[part], second[part])
            groups = scaffold_groups(smiles)
            row_to_scaffold = {row: scaffold for scaffold, rows in groups.items() for row in rows}
            scaffold_sets = {
                part: {row_to_scaffold[row] for row in indices.tolist()} for part, indices in first.items()
            }
            self.assertFalse(scaffold_sets["train"] & scaffold_sets["valid"])
            self.assertFalse(scaffold_sets["train"] & scaffold_sets["test"])
            self.assertFalse(scaffold_sets["valid"] & scaffold_sets["test"])

    def test_masked_classification_metrics_skip_missing_and_single_class_tasks(self):
        truth = np.asarray([[0, 1], [1, np.nan], [1, 1], [0, np.nan]], dtype=float)
        score = np.asarray([[0.1, 0.8], [0.9, 0.2], [0.8, 0.7], [0.2, 0.4]], dtype=float)
        aggregate, per_task = classification_metrics(truth, score, ["a", "b"])
        self.assertAlmostEqual(aggregate["roc_auc_macro"], 1.0)
        self.assertEqual(aggregate["n_tasks_roc_auc"], 1)
        self.assertTrue(np.isnan(per_task.loc[per_task["task"] == "b", "roc_auc"].iloc[0]))

    def test_regression_metrics(self):
        truth = np.asarray([[1.0], [2.0], [np.nan], [4.0]])
        score = np.asarray([[1.0], [3.0], [9.0], [4.0]])
        aggregate, _ = regression_metrics(truth, score, ["y"])
        self.assertAlmostEqual(aggregate["mae"], 1.0 / 3.0)
        self.assertAlmostEqual(aggregate["rmse"], np.sqrt(1.0 / 3.0))

    def test_clintox_risk_orientation_flips_only_fda_label(self):
        frame = pd.DataFrame(
            {
                "FDA_APPROVED": [1.0, 0.0, np.nan],
                "CT_TOX": [0.0, 1.0, 1.0],
            }
        )
        manifest = {"tasks": ["FDA_APPROVED", "CT_TOX"], "n_tasks": 2}
        oriented, tasks, oriented_manifest, transform = orient_clintox_risk_labels(
            frame,
            "clintox",
            ["FDA_APPROVED", "CT_TOX"],
            manifest,
            enabled=True,
        )
        np.testing.assert_allclose(
            oriented["FDA_NOT_APPROVED"].to_numpy(),
            np.asarray([0.0, 1.0, np.nan]),
            equal_nan=True,
        )
        np.testing.assert_allclose(oriented["CT_TOX"], frame["CT_TOX"])
        self.assertEqual(tasks, ["FDA_NOT_APPROVED", "CT_TOX"])
        self.assertEqual(oriented_manifest["tasks"], tasks)
        self.assertEqual(transform["name"], "clintox_risk_orientation")

    def test_summary_supports_classification_only_invocation(self):
        result = {
            "dataset": "sider",
            "problem_type": "classification",
            "split_type": "scaffold",
            "seed": 13,
            "n_tasks": 27,
            "n_test": 143,
            "metrics": {
                "roc_auc_macro": 0.61,
                "pr_auc_macro": 0.65,
                "balanced_accuracy_macro": 0.56,
            },
        }
        with TemporaryDirectory() as tmp:
            summary = summarize_results([result], tmp)
            self.assertEqual(len(summary), 1)


if __name__ == "__main__":
    unittest.main()
