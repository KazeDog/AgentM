from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from experiments.agentm_reliability.common import load_config, write_json
from experiments.agentm_reliability.run_matrix import _extract_model_paths
from experiments.agentm_reliability.validation import REQUIRED_RESULT_FIELDS, run_invariant_violations, scientific_resource_signature, validate_result_record


class ResultSchemaTests(unittest.TestCase):
    def _record(self) -> dict:
        record = {field: None for field in REQUIRED_RESULT_FIELDS}
        record.update({"run_id": "run", "case_id": "case", "orchestrator": "heuristic", "dataset_group": "clean", "reflection_enabled": False})
        return record

    def test_required_schema_rejects_missing_field(self) -> None:
        record = self._record()
        record.pop("config_hash")
        with self.assertRaisesRegex(ValueError, "config_hash"):
            validate_result_record(record)

    def test_raw_json_write_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "raw.json"
            write_json(path, {"first": True}, overwrite=False)
            with self.assertRaises(FileExistsError):
                write_json(path, {"second": True}, overwrite=False)

    def test_reflection_false_forbids_reflection_agent_call(self) -> None:
        job = {"reflection_enabled": False, "dataset_group": "clean"}
        context = {"tool_calls": [{"agent": "ReflectionAgent", "tool": "reflect_on_report", "status": "completed"}]}
        self.assertIn("reflection_false_called_ReflectionAgent", run_invariant_violations(job, context))

    def test_orchestrators_share_scientific_resource_signature(self) -> None:
        config = load_config("experiments/agentm_reliability/configs/base.yaml")
        case = {"request": "CCO", "feature_set": "all_features", "model_path": "model.pkl"}
        heuristic = scientific_resource_signature(config, case)
        llm_config = deepcopy(config)
        llm_config["runtime"]["use_llm_orchestrator"] = True
        llm = scientific_resource_signature(llm_config, case)
        self.assertEqual(heuristic, llm)

    def test_actual_model_path_is_extracted_from_prediction_trace(self) -> None:
        context = {
            "tool_calls": [
                {
                    "agent": "GaugeAgent",
                    "tool": "predict_single",
                    "observation": {"model_path": "/models/actual.pkl"},
                }
            ]
        }
        self.assertEqual(_extract_model_paths(context), ["/models/actual.pkl"])


if __name__ == "__main__":
    unittest.main()
