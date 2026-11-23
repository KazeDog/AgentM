"""Run fixture-based regression tests without network or paid model access."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import unittest

from offline_guard import activate, verify_import_roots, NETWORK_ATTEMPTS, ROOT


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, help="Write machine-readable validation result")
    parser.add_argument("--modules", nargs="+", help="Optional explicit unittest modules")
    args = parser.parse_args()
    report_path = args.report.resolve() if args.report else None
    activate()
    imports = verify_import_roots()
    sys.path.insert(0, str(ROOT / "tests"))
    sys.path.insert(0, str(ROOT / "experiments" / "agentm_reliability" / "tests"))
    default_modules = [
        "test_state_guard_verifier", "test_prediction_semantics", "test_scientific_metrics",
        "test_moleculenet", "test_result_schema",
        "test_evidence_provenance", "test_conflict_evidence", "test_injections",
        "test_offline_evaluator", "test_render_guards", "test_runtime_guards",
        "test_multiround_reflection", "test_structured_prediction_mutations",
        "test_llm_terminal_stop", "test_orchestrator_factorial", "test_llm_retry_and_safe_abstention",
        "test_resources_objectives.ResourceAndObjectiveTests.test_require_llm_orchestrator_does_not_heuristic_fallback",
    ]
    modules = args.modules or default_modules
    # Preserve package context for tests with relative imports.
    reliability_tests = ROOT / "experiments" / "agentm_reliability" / "tests"
    modules = [
        "experiments.agentm_reliability.tests." + name
        if "." not in name and (reliability_tests / (name + ".py")).exists() else name
        for name in modules
    ]
    suite = unittest.TestLoader().loadTestsFromNames(modules)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    summary = {
        "scope": "synthetic-fixture offline regression",
        "default_profile_exclusion": "Resource-dependent integration tests require external assets. Required-LLM/reflection control is included in the offline profile. See KNOWN_ISSUES.md",
        "modules": modules, "tests_run": result.testsRun,
        "failures": [{"test": str(t), "traceback": s} for t, s in result.failures],
        "errors": [{"test": str(t), "traceback": s} for t, s in result.errors],
        "skipped": [{"test": str(t), "reason": s} for t, s in result.skipped],
        "passed": result.wasSuccessful(), "network_operations_blocked": len(NETWORK_ATTEMPTS),
        "api_calls_sent": 0, "gpu_visible": False, "import_roots": imports,
    }
    if report_path:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k not in {"failures", "errors", "modules"}}, indent=2))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
