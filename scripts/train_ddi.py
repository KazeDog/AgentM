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
    parser = argparse.ArgumentParser(description="Train a DDI prediction model.")
    parser.add_argument("--config", default="configs/ddi_example.yaml")
    args = parser.parse_args()
    config = load_yaml(ROOT / args.config if not Path(args.config).is_absolute() else args.config)
    config["task_type"] = "ddi_prediction"
    result = train_tabular(config)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

