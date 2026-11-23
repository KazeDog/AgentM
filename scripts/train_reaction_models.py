#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agentm.runtime.reaction import ReactionRuntime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train lightweight AgentM reaction baselines.")
    parser.add_argument("--task", required=True, choices=["schneider_classification", "buchwald_hartwig_yield"])
    parser.add_argument("--split", default="test1", help="Buchwald-Hartwig KG split: test1-test4.")
    parser.add_argument("--max-rows", default=None, help="Integer or 'smoke'.")
    parser.add_argument("--model-family", default="tfidf_linear", choices=["tfidf_linear", "token_knn"])
    parser.add_argument("--all-splits", action="store_true", help="Train Buchwald-Hartwig test1-test4 models.")
    parser.add_argument("--eval", action="store_true", help="After training, print the matching benchmark command.")
    parser.add_argument("--reaction-data-root", default="data/reaction_sources")
    parser.add_argument("--output-dir", default="outputs/reaction_models_full")
    return parser.parse_args()


def _max_rows(value: str | None) -> int | None:
    if value is None:
        return None
    if str(value).lower() == "smoke":
        return 4096
    return int(value)


def main() -> int:
    args = parse_args()
    runtime = ReactionRuntime(output_dir=ROOT / "outputs" / "reaction_training", data_root=args.reaction_data_root)
    splits = ["test1", "test2", "test3", "test4"] if args.all_splits and args.task == "buchwald_hartwig_yield" else [args.split]
    results = []
    for split in splits:
        result = runtime.train_model(
            task=args.task,
            split=split,
            max_rows=_max_rows(args.max_rows),
            output_dir=ROOT / args.output_dir,
            model_family=args.model_family,
        )
        results.append({k: v for k, v in result.items() if k not in {"examples", "artifact"}})
    payload = {"results": results}
    if args.eval:
        payload["eval_command"] = (
            "python scripts/run_reaction_model_benchmark.py "
            f"--model-family {args.model_family} --output-dir {args.output_dir}"
        )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
