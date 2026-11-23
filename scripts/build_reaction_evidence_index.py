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
    parser = argparse.ArgumentParser(description="Build local AgentM reaction evidence indices.")
    parser.add_argument("--dataset", required=True, choices=["mit", "orderly", "all"])
    parser.add_argument("--max-records", default=None, help="Integer record cap or 'full'.")
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--reaction-data-root", default="data/reaction_sources")
    parser.add_argument("--output-dir", default="data/reaction_indices")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    runtime = ReactionRuntime(output_dir=ROOT / "outputs" / "reaction_index_build", data_root=args.reaction_data_root)
    manifest = runtime.build_indices(dataset=args.dataset, max_records=_max_records(args.max_records), output_dir=ROOT / args.output_dir, shards=args.shards)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


def _max_records(value: str | None) -> int | None:
    if value is None:
        return None
    if str(value).lower() == "full":
        return None
    return int(value)


if __name__ == "__main__":
    raise SystemExit(main())
