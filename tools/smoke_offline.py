"""Exercise report writing on an unsupported request, with no models or network."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from offline_guard import activate, verify_import_roots, NETWORK_ATTEMPTS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error("Use a new or empty output directory")
    activate()
    imports = verify_import_roots()
    from agentm.runtime import AgentMMultiAgentController
    controller = AgentMMultiAgentController(
        output_dir=output, feature_set="rdkit_only", mode="fast",
        use_llm_orchestrator=False, use_llm_reasoning=False,
        use_rag_evidence=False, use_external_evidence=False,
        enable_fallbacks=False, enable_reflection=False,
        auto_model=False, max_rounds=8,
    )
    report = controller.run("What is the weather tomorrow?")
    assert isinstance(report, dict) and report, "Expected a structured report"
    assert not NETWORK_ATTEMPTS, "Offline smoke unexpectedly attempted networking"
    assert list(output.rglob("*.json")), "Report artifacts were not written"
    (output / "submission_smoke_summary.json").write_text(json.dumps({
        "passed": True, "api_calls_sent": 0, "network_attempts": 0,
        "scope": "unsupported-request guard and report serialization",
        "import_roots": imports,
    }, indent=2) + "\n")
    print("Offline smoke passed. Artifacts: " + str(output))


if __name__ == "__main__":
    main()
