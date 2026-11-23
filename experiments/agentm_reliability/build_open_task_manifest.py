#!/usr/bin/env python
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

try:
    from .common import RELIABILITY_ROOT, load_config, load_jsonl, stable_hash, write_jsonl
except ImportError:
    from common import RELIABILITY_ROOT, load_config, load_jsonl, stable_hash, write_jsonl


ENDPOINT_CYCLE = [
    ("BBB_Martins", "bbbp"),
    ("hERG", "herg"),
    ("DILI", "dili"),
    ("HIA_Hou", "hia"),
    ("Pgp_Broccatelli", "pgp"),
]


def _gold(
    *,
    objectives: list[str],
    tools: list[str],
    dependencies: list[list[str]],
    outputs: list[str],
    objects: list[str],
    alternatives: list[list[str]] | None = None,
) -> dict:
    return {
        "required_objectives": objectives,
        "required_tools": tools,
        "forbidden_tools": ["run_zero_shot_fallback"],
        "required_dependencies": dependencies,
        "required_outputs": outputs,
        "allowed_alternative_plans": alternatives or [],
        "turn_level_expected_state": [],
        "stale_records_to_remove": [],
        "required_citation_updates": ["final claims must map only to final scientific objects"],
        "expected_final_scientific_objects": objects,
    }


def build(config: dict) -> list[dict]:
    molecular = load_jsonl(config["paths"]["molecular_safety_manifest"])
    reactions = load_jsonl(config["paths"]["reaction_manifest"])
    from scripts.run_reaction_benchmark import materialize_case

    reactions = [materialize_case(row, str(config["paths"]["reaction_data_root"])) for row in reactions]
    admet = [row for row in molecular if row.get("category") == "admet"]
    ddi = [row for row in molecular if row.get("category") == "ddi"]

    unique_by_request: dict[str, dict] = {}
    for row in admet:
        unique_by_request.setdefault(str(row["request"]), row)
    molecules = list(unique_by_request.values())[:10]
    if len(molecules) < 10 or len(ddi) < 10:
        raise ValueError("Insufficient molecular-safety source cases for 30 open tasks")

    rows: list[dict] = []
    for index, base in enumerate(molecules):
        endpoints = [ENDPOINT_CYCLE[(index + offset) % len(ENDPOINT_CYCLE)] for offset in range(3)]
        endpoint_text = ", ".join(label for _, label in endpoints)
        case_id = f"open_multi_endpoint_admet_{index:02d}"
        request = (
            f"For molecule {base['request']}, evaluate the {endpoint_text} endpoints, compare evidence strength, "
            "and keep each prediction distinct from external evidence."
        )
        gold = _gold(
            objectives=[f"predict:{label}" for _, label in endpoints] + ["compare_evidence_strength"],
            tools=["resolve_molecule", "predict_single", "retrieve_elasticsearch_hybrid", "verify_citation_grounding"],
            dependencies=[["resolve_molecule", "predict_single"], ["predict_single", "verify_prediction_grounding"]],
            outputs=["three endpoint-specific predictions", "endpoint-specific evidence comparison", "citation map"],
            objects=[str(base["request"])],
        )
        rows.append(
            {
                "case_id": case_id,
                "name": case_id,
                "dataset_group": "open_task",
                "open_task_type": "multi_endpoint_admet",
                "request": request,
                "task_type": "admet_prediction",
                "dataset_name": endpoints[0][0],
                "requested_endpoints": [name for name, _ in endpoints],
                "base_case": base,
                "gold_plan": gold,
            }
        )

    for index, base in enumerate(ddi[:10]):
        pair = [part.strip() for part in str(base["request"]).split(",", 1)]
        while len(pair) < 2:
            pair.append("unspecified_partner")
        case_id = f"open_ddi_plus_individual_safety_{index:02d}"
        request = (
            f"Evaluate the DDI risk for {pair[0]} and {pair[1]}; also assess hERG risk for {pair[0]} and DILI risk "
            f"for {pair[1]}. Keep pair-level and single-drug evidence separate."
        )
        gold = _gold(
            objectives=["predict:ddi", f"predict:{pair[0]}:herg", f"predict:{pair[1]}:dili", "separate_pair_and_single_evidence"],
            tools=["resolve_molecule", "predict_ddi", "predict_single", "retrieve_elasticsearch_hybrid", "verify_citation_grounding"],
            dependencies=[["resolve_molecule", "predict_ddi"], ["predict_ddi", "verify_prediction_grounding"]],
            outputs=["DDI prediction", "two individual safety predictions", "separated citation map"],
            objects=pair,
        )
        rows.append(
            {
                "case_id": case_id,
                "name": case_id,
                "dataset_group": "open_task",
                "open_task_type": "ddi_plus_individual_safety",
                "request": request,
                "task_type": "ddi_prediction",
                "dataset_name": "DDI",
                "base_case": base,
                "gold_plan": gold,
            }
        )

    reaction_sources = [
        row for row in reactions if row.get("task_type") in {"reaction_profile", "reaction_yield_prediction", "reaction_classification"}
    ][:10]
    if len(reaction_sources) < 10:
        raise ValueError("Insufficient reaction source cases for open tasks")
    for index, base in enumerate(reaction_sources):
        case_id = f"open_reaction_multi_tool_{index:02d}"
        reaction_request = str(base["request"]).split(" for ", 1)[-1]
        request = (
            f"For reaction {reaction_request}, classify the reaction, predict yield when supported, retrieve condition evidence, "
            "and analyse template evidence. Preserve uncertainty if a tool is temporarily unavailable."
        )
        gold = _gold(
            objectives=["reaction_classification", "yield_prediction", "condition_evidence", "template_evidence"],
            tools=[
                "parse_reaction",
                "classify_reaction_type",
                "predict_buchwald_hartwig_yield",
                "retrieve_orderly_condition_evidence",
                "retrieve_mit_template_evidence",
            ],
            dependencies=[["parse_reaction", "classify_reaction_type"], ["parse_reaction", "retrieve_mit_template_evidence"]],
            outputs=["reaction class", "yield prediction or explicit non-applicability", "condition evidence", "template evidence"],
            objects=[reaction_request],
            alternatives=[["parse_reaction", "build_reaction_profile", "verify_report_schema"]],
        )
        rows.append(
            {
                "case_id": case_id,
                "name": case_id,
                "dataset_group": "open_task",
                "open_task_type": "reaction_multi_tool",
                "request": request,
                "task_type": "reaction_profile",
                "dataset_name": str(base.get("dataset_name") or "reaction"),
                "base_case": base,
                "gold_plan": gold,
                "transient_fault": {"kind": "first_retrieval_no_result", "tool": "retrieve_mit_template_evidence", "recoverable": True},
            }
        )

    for index, row in enumerate(rows):
        row["case_index"] = index
        row["case_hash"] = stable_hash({key: value for key, value in row.items() if key != "case_hash"})
    if len(rows) != 30:
        raise AssertionError(f"Expected 30 open tasks, built {len(rows)}")
    counts = defaultdict(int)
    for row in rows:
        counts[row["open_task_type"]] += 1
    if set(counts.values()) != {10}:
        raise AssertionError(f"Open-task types are not balanced: {dict(counts)}")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Build 30 deterministic open-composition tasks.")
    parser.add_argument("config", nargs="?", default=str(RELIABILITY_ROOT / "configs" / "base.yaml"))
    parser.add_argument("--output", default=str(RELIABILITY_ROOT / "manifests" / "open_tasks.jsonl"))
    args = parser.parse_args()
    rows = build(load_config(args.config))
    path = write_jsonl(Path(args.output), rows)
    print(f"wrote {len(rows)} open tasks to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
