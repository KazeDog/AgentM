#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

try:
    from .common import RELIABILITY_ROOT, load_config, load_jsonl, resolve_path, stable_hash, write_jsonl
except ImportError:
    from common import RELIABILITY_ROOT, load_config, load_jsonl, resolve_path, stable_hash, write_jsonl


def build(config: dict) -> list[dict]:
    source = resolve_path(config["paths"]["molecular_safety_manifest"])
    rows = load_jsonl(source, required_fields=["name", "request", "task_type", "dataset_name", "category"])
    if len(rows) != 100:
        raise ValueError(f"Expected exactly 100 molecular-safety cases, found {len(rows)} in {source}")
    built = []
    for index, row in enumerate(rows):
        built.append(
            {
                **row,
                "case_id": str(row["name"]),
                "dataset_group": "clean",
                "case_index": index,
                "source_manifest": str(source),
                "source_case_hash": stable_hash(row),
            }
        )
    return built


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the immutable clean molecular-safety manifest.")
    parser.add_argument("config", nargs="?", default=str(RELIABILITY_ROOT / "configs" / "base.yaml"))
    parser.add_argument("--output", default=str(RELIABILITY_ROOT / "manifests" / "clean.jsonl"))
    args = parser.parse_args()
    rows = build(load_config(args.config))
    path = write_jsonl(Path(args.output), rows)
    print(f"wrote {len(rows)} clean cases to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
