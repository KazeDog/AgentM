#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agentm.defaults import DEFAULT_RAG_EMBEDDING_MODEL_ID, DEFAULT_RAG_EMBEDDING_MODEL_PATH


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download the RAG embedding model into checkpoints.")
    parser.add_argument("--model-id", default=DEFAULT_RAG_EMBEDDING_MODEL_ID)
    parser.add_argument("--output-dir", default=DEFAULT_RAG_EMBEDDING_MODEL_PATH)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModel.from_pretrained(args.model_id)
    tokenizer.save_pretrained(output_dir)
    model.save_pretrained(output_dir)
    payload = {
        "model_id": args.model_id,
        "output_dir": str(output_dir),
        "backend": "transformers_mean_pool_or_sentence_transformer",
    }
    (output_dir / "agentm_rag_embedding_model.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
