#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agentm.runtime import AgentMMultiAgentController


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AgentM guarded multi-agent runtime.")
    parser.add_argument("--request", required=True)
    parser.add_argument(
        "--task-type",
        default=None,
        choices=[
            "property_prediction",
            "admet_prediction",
            "ddi_prediction",
            "reaction_profile",
            "reaction_classification",
            "reaction_yield_prediction",
            "reaction_condition_retrieval",
        ],
    )
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--feature-set", default="rdkit_only")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--auto-model", action="store_true", help="Resolve a core real model automatically from task/dataset/feature set.")
    parser.add_argument("--model-registry", default=None, help="Optional JSON model registry with custom path priorities.")
    parser.add_argument("--require-real-model", action="store_true", help="Fail strict prediction if no compatible real model.pkl is resolved.")
    parser.add_argument("--require-real-rag", action="store_true", help="Fail strict evidence retrieval if real RAG resources/evidence are unavailable.")
    parser.add_argument("--require-llm-orchestrator", action="store_true", help="Fail if LLM orchestration is unavailable; do not use heuristic fallback.")
    parser.add_argument("--require-llm-reasoning", action="store_true", help="Fail if LLM expert reasoning is unavailable or fails.")
    parser.add_argument("--min-evidence", type=int, default=1, help="Minimum grounded evidence items required by objective checks.")
    parser.add_argument("--mode", default="full", choices=["fast", "full", "deep"])
    parser.add_argument("--retrieval-backend", default="auto", choices=["auto", "es", "local"])
    parser.add_argument("--es-url", default=None)
    parser.add_argument(
        "--enable-network-lookup",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable PubChem/DailyMed/ChEMBL lookups for name/SMILES/CID alignment.",
    )
    parser.add_argument("--pubchem-timeout", type=float, default=None, help="Timeout seconds for PubChem/DailyMed/ChEMBL lookup requests.")
    parser.add_argument("--max-rounds", type=int, default=16)
    parser.add_argument("--max-reflection-rounds", type=int, default=1)
    parser.add_argument("--previous-run", default=None, help="Optional path to a previous multi_agent_report.json/open report.")
    parser.add_argument(
        "--enable-reflection",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use ReflectionAgent to feed the first-pass report/verifier output back into finalization.",
    )
    parser.add_argument(
        "--use-llm-orchestrator",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use LLM OrchestratorAgent when available; fallback is deterministic.",
    )
    parser.add_argument(
        "--use-llm-reasoning",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use LLM ReasoningAgent/ReflectionAgent expert reasoning when available.",
    )
    parser.add_argument(
        "--enable-fallbacks",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable zero-shot/model/retrieval/LLM fallback behavior. Disabled by default.",
    )
    parser.add_argument("--output-dir", default="outputs/multi_agent")
    parser.add_argument("--reaction-data-root", default="data/reaction_sources")
    parser.add_argument("--reaction-model-dir", default=None)
    parser.add_argument("--reaction-index-dir", default=None)
    parser.add_argument("--full-json", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.enable_network_lookup is not None:
        os.environ["AGENTM_ENABLE_NETWORK_LOOKUP"] = "1" if args.enable_network_lookup else "0"
    if args.pubchem_timeout is not None:
        os.environ["AGENTM_PUBCHEM_TIMEOUT"] = str(args.pubchem_timeout)
    controller = AgentMMultiAgentController(
        output_dir=args.output_dir,
        feature_set=args.feature_set,
        model_path=args.model_path,
        auto_model=args.auto_model,
        model_registry_path=args.model_registry,
        require_real_model=args.require_real_model,
        require_real_rag=args.require_real_rag,
        require_llm_orchestrator=args.require_llm_orchestrator,
        require_llm_reasoning=args.require_llm_reasoning,
        min_evidence=args.min_evidence,
        mode=args.mode,
        retrieval_backend=args.retrieval_backend,
        es_url=args.es_url,
        max_rounds=args.max_rounds,
        use_llm_orchestrator=args.use_llm_orchestrator,
        use_llm_reasoning=args.use_llm_reasoning,
        enable_fallbacks=args.enable_fallbacks,
        enable_reflection=args.enable_reflection,
        max_reflection_rounds=args.max_reflection_rounds,
        previous_run_path=args.previous_run,
        reaction_data_root=args.reaction_data_root,
        reaction_model_dir=args.reaction_model_dir,
        reaction_index_dir=args.reaction_index_dir,
    )
    report = controller.run(args.request, task_type=args.task_type, dataset_name=args.dataset_name)
    if args.full_json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        prediction = (report.get("predictions") or [{}])[0]
        compact = {
            "status": "completed",
            "runtime": "agentm_guarded_multi_agent",
            "report_path": report.get("artifacts", {}).get("multi_agent_report_path"),
            "compat_report_path": report.get("artifacts", {}).get("compat_report_path"),
            "resource_resolution_path": report.get("artifacts", {}).get("resource_resolution_path"),
            "objective_status_path": report.get("artifacts", {}).get("objective_status_path"),
            "task_type": report.get("task_type"),
            "dataset_name": prediction.get("dataset_name"),
            "prediction": {
                "label": prediction.get("label"),
                "score": prediction.get("score"),
                "probability": prediction.get("probability"),
                "model_type": prediction.get("model_type"),
            },
            "verifier_status": (report.get("verifier_report") or {}).get("summary", {}).get("status"),
            "objective_status": (report.get("objective_status") or {}).get("status"),
            "reflection_status": (report.get("reflection") or {}).get("application_status"),
            "previous_run_loaded": bool(report.get("previous_run_context")),
            "planner_type": report.get("multi_agent", {}).get("planner_type"),
            "orchestrator_mode": report.get("multi_agent", {}).get("orchestrator_mode"),
            "n_tool_calls": len(report.get("tool_calls") or []),
            "llm_orchestrator_enabled": report.get("multi_agent", {}).get("llm_orchestrator_enabled"),
            "llm_reasoning_enabled": report.get("multi_agent", {}).get("llm_reasoning_enabled"),
            "reasoning_mode": report.get("multi_agent", {}).get("reasoning_mode"),
            "fallbacks_enabled": report.get("multi_agent", {}).get("fallbacks_enabled"),
            "resource_resolution": report.get("resource_resolution", {}),
            "warnings": report.get("warnings", []),
        }
        print(json.dumps(compact, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
