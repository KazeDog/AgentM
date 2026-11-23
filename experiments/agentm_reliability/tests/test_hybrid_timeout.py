from __future__ import annotations

import os
from unittest.mock import patch
import unittest

from agentm.runtime.retrieval.hybrid import HybridRetriever
from experiments.agentm_reliability.common import load_config


class HybridTimeoutTests(unittest.TestCase):
    def test_timeout_and_retry_are_explicitly_configurable(self) -> None:
        with patch.dict(
            os.environ,
            {"AGENTM_ES_REQUEST_TIMEOUT": "75", "AGENTM_ES_MAX_RETRIES": "2"},
            clear=False,
        ):
            retriever = HybridRetriever(use_dense=False)
        self.assertEqual(retriever.request_timeout_seconds, 75.0)
        self.assertEqual(retriever.max_retries, 2)

    def test_recovery_config_records_one_pre_llm_transport_retry(self) -> None:
        config = load_config("experiments/agentm_reliability/configs/base.yaml")
        self.assertEqual(config["runtime"]["es_request_timeout_seconds"], 60)
        self.assertEqual(config["runtime"]["es_max_retries"], 1)


if __name__ == "__main__":
    unittest.main()
