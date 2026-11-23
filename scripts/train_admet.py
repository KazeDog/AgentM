#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agentm.evaluation.training import train_tabular
from agentm.utils.io import load_yaml


def main() -> int:
    parser = argparse.ArgumentParser(description="Train an ADMET prediction model.")
    parser.add_argument("--config", default="configs/admet_example.yaml")
    parser.add_argument("--seed", type=int, default=None, help="Override random_seed from the YAML config.")
    parser.add_argument("--output-dir", default=None, help="Override output_dir from the YAML config.")
    parser.add_argument(
        "--persistent-feature-cache-dir",
        default=None,
        help="Cache deterministic single-molecule feature bundles across repeated seeds.",
    )
    parser.add_argument(
        "--precomputed-rules-path",
        default=None,
        help="Reuse a rules_used.json file so repeated seeds use exactly the same rule features.",
    )
    args = parser.parse_args()
    config = load_yaml(ROOT / args.config if not Path(args.config).is_absolute() else args.config)
    config["task_type"] = "admet_prediction"
    if args.seed is not None:
        config["random_seed"] = int(args.seed)
    if args.output_dir is not None:
        config["output_dir"] = args.output_dir
    if args.persistent_feature_cache_dir is not None:
        config["persistent_feature_cache_dir"] = args.persistent_feature_cache_dir
    if args.precomputed_rules_path is not None:
        config["precomputed_rules_path"] = args.precomputed_rules_path
    result = train_tabular(config)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
