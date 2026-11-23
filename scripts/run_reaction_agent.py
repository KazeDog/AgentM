#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agentm.runtime import AgentMMultiAgentController


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AgentM ReactionAgent.")
    parser.add_argument("--request", required=True, help="Natural-language request or reaction SMILES containing >>.")
    parser.add_argument(
        "--task-type",
        default="reaction_profile",
        choices=["reaction_profile", "reaction_classification", "reaction_yield_prediction", "reaction_condition_retrieval"],
    )
    parser.add_argument("--mode", default="full", choices=["fast", "full", "deep"])
    parser.add_argument("--max-rounds", type=int, default=16)
    parser.add_argument("--reaction-data-root", default="data/reaction_sources")
    parser.add_argument("--reaction-model-dir", default=None)
    parser.add_argument("--reaction-index-dir", default=None)
    parser.add_argument("--output-dir", default="outputs/reaction_agent")
    parser.add_argument("--full-json", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    controller = AgentMMultiAgentController(
        output_dir=args.output_dir,
        mode=args.mode,
        max_rounds=args.max_rounds,
        use_llm_orchestrator=False,
        use_llm_reasoning=False,
        reaction_data_root=args.reaction_data_root,
        reaction_model_dir=args.reaction_model_dir,
        reaction_index_dir=args.reaction_index_dir,
    )
    report = controller.run_reaction(args.request, task_type=args.task_type)
    if args.full_json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        compact = {
            "status": report.get("objective_status", {}).get("status") or "completed",
            "task_type": report.get("task_type"),
            "reaction_report_path": report.get("artifacts", {}).get("reaction_report_path"),
            "reaction_profile_path": report.get("artifacts", {}).get("reaction_profile_path"),
            "prediction": (report.get("reaction_predictions") or report.get("predictions") or [{}])[-1],
            "warnings": report.get("warnings", []),
        }
        print(json.dumps(compact, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
