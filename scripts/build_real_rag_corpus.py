#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agentm.tools.real_rag_corpus import RealRAGCorpusBuilder, build_config_from_args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a real PubMed/PubChem/DailyMed RAG corpus for TDC ADMET/Tox tasks.")
    parser.add_argument("--tasks", action="append", default=["all"], help="Task names, comma-separated values, 'recommended', or 'all'.")
    parser.add_argument("--output-dir", default="data/rag_corpus/tdc_real")
    parser.add_argument("--data-root", default="data/tdc")
    parser.add_argument("--max-pubmed", type=int, default=20)
    parser.add_argument("--max-compounds", type=int, default=8)
    parser.add_argument("--max-dailymed-labels", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--request-delay", type=float, default=0.1)
    parser.add_argument("--random-seed", type=int, default=13)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = build_config_from_args(args)
    result = RealRAGCorpusBuilder(config).build()
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
