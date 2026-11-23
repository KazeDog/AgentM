#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

try:
    from .common import RELIABILITY_ROOT, load_config, load_jsonl, stable_hash, write_jsonl
except ImportError:
    from common import RELIABILITY_ROOT, load_config, load_jsonl, stable_hash, write_jsonl


def _plan(
    *,
    objectives: list[str],
    tools: list[str],
    dependencies: list[list[str]],
    states: list[dict],
    stale: list[str],
    citation_updates: list[str],
    final_objects: list[str],
) -> dict:
    return {
        "required_objectives": objectives,
        "required_tools": tools,
        "forbidden_tools": ["run_zero_shot_fallback"],
        "required_dependencies": dependencies,
        "required_outputs": objectives,
        "allowed_alternative_plans": [],
        "turn_level_expected_state": states,
        "stale_records_to_remove": stale,
        "required_citation_updates": citation_updates,
        "expected_final_scientific_objects": final_objects,
    }


def build(config: dict) -> list[dict]:
    molecular = load_jsonl(config["paths"]["molecular_safety_manifest"])
    admet = [row for row in molecular if row.get("category") == "admet"]
    ddi = [row for row in molecular if row.get("category") == "ddi"]
    unique: list[dict] = []
    seen: set[str] = set()
    for row in admet:
        if row["request"] not in seen:
            seen.add(row["request"])
            unique.append(row)
    if len(unique) < 15 or len(ddi) < 5:
        raise ValueError("Insufficient cases for multi-turn manifest")

    rows: list[dict] = []
    endpoints = [("BBB_Martins", "bbbp"), ("hERG", "herg"), ("DILI", "dili"), ("HIA_Hou", "hia"), ("Pgp_Broccatelli", "pgp")]

    for index in range(5):
        base = unique[index]
        additions = [endpoints[(index + 1) % 5], endpoints[(index + 2) % 5]]
        turns = [
            {"turn": 1, "request": base["request"], "task_type": "admet_prediction", "dataset_name": base["dataset_name"]},
            {
                "turn": 2,
                "request": f"For the same molecule {base['request']}, additionally evaluate {additions[0][1]} and {additions[1][1]}.",
                "task_type": "admet_prediction",
                "dataset_name": additions[0][0],
            },
        ]
        case_id = f"multi_turn_add_endpoints_{index:02d}"
        rows.append(
            {
                "case_id": case_id,
                "name": case_id,
                "dataset_group": "multi_turn",
                "multi_turn_type": "add_two_endpoints",
                "turns": turns,
                "gold_plan": _plan(
                    objectives=[f"retain:{base['expected_endpoint']}", f"add:{additions[0][1]}", f"add:{additions[1][1]}"],
                    tools=["resolve_molecule", "predict_single", "verify_prediction_grounding"],
                    dependencies=[["resolve_molecule", "predict_single"]],
                    states=[{"turn": 1, "endpoints": [base["expected_endpoint"]]}, {"turn": 2, "endpoints": [base["expected_endpoint"], additions[0][1], additions[1][1]]}],
                    stale=[],
                    citation_updates=["add citations for new endpoints without relabeling the retained endpoint"],
                    final_objects=[base["request"]],
                ),
            }
        )

    common_names = ["aspirin", "caffeine", "warfarin", "metformin", "ketoconazole"]
    for index, (name, specific) in enumerate(zip(common_names, unique[5:10])):
        turns = [
            {"turn": 1, "request": name, "task_type": "admet_prediction", "dataset_name": "hERG"},
            {"turn": 2, "request": specific["request"], "task_type": "admet_prediction", "dataset_name": "hERG"},
        ]
        case_id = f"multi_turn_resolve_specific_structure_{index:02d}"
        rows.append(
            {
                "case_id": case_id,
                "name": case_id,
                "dataset_group": "multi_turn",
                "multi_turn_type": "resolve_specific_structure",
                "turns": turns,
                "gold_plan": _plan(
                    objectives=["replace_ambiguous_entity", "recompute_prediction", "refresh_citations"],
                    tools=["resolve_molecule", "predict_single", "verify_citation_grounding"],
                    dependencies=[["resolve_molecule", "predict_single"]],
                    states=[{"turn": 1, "entity": name}, {"turn": 2, "entity": specific["request"]}],
                    stale=[name],
                    citation_updates=["remove citations attached only to the ambiguous name"],
                    final_objects=[specific["request"]],
                ),
            }
        )

    for index, pair_case in enumerate(ddi[:5]):
        pair = [part.strip() for part in str(pair_case["request"]).split(",", 1)]
        single = unique[10 + index]
        turns = [
            {"turn": 1, "request": single["request"], "task_type": "admet_prediction", "dataset_name": single["dataset_name"]},
            {"turn": 2, "request": pair_case["request"], "task_type": "ddi_prediction", "dataset_name": "DDI"},
        ]
        case_id = f"multi_turn_add_ddi_{index:02d}"
        rows.append(
            {
                "case_id": case_id,
                "name": case_id,
                "dataset_group": "multi_turn",
                "multi_turn_type": "single_to_ddi",
                "turns": turns,
                "gold_plan": _plan(
                    objectives=["retain_single_drug_result", "add_ddi_prediction", "separate_pair_evidence"],
                    tools=["predict_single", "predict_ddi", "verify_citation_grounding"],
                    dependencies=[["resolve_molecule", "predict_ddi"]],
                    states=[{"turn": 1, "objects": [single["request"]]}, {"turn": 2, "objects": pair}],
                    stale=[],
                    citation_updates=["pair-level citations must not be represented as single-drug evidence"],
                    final_objects=pair,
                ),
            }
        )

    for index in range(5):
        old = unique[index]
        new = unique[(index + 6) % len(unique)]
        turns = [
            {"turn": 1, "request": old["request"], "task_type": "admet_prediction", "dataset_name": old["dataset_name"]},
            {
                "turn": 2,
                "request": new["request"],
                "task_type": "admet_prediction",
                "dataset_name": new["dataset_name"],
                "instruction": "Withdraw the previous target, replace it with this molecule, and retrieve more direct endpoint evidence.",
            },
        ]
        case_id = f"multi_turn_withdraw_and_replace_{index:02d}"
        rows.append(
            {
                "case_id": case_id,
                "name": case_id,
                "dataset_group": "multi_turn",
                "multi_turn_type": "withdraw_and_replace",
                "turns": turns,
                "gold_plan": _plan(
                    objectives=["withdraw_previous_target", "replace_scientific_object", "refresh_prediction", "refresh_direct_evidence"],
                    tools=["resolve_molecule", "predict_single", "retrieve_elasticsearch_hybrid", "verify_citation_grounding"],
                    dependencies=[["resolve_molecule", "predict_single"], ["retrieve_elasticsearch_hybrid", "verify_citation_grounding"]],
                    states=[{"turn": 1, "entity": old["request"]}, {"turn": 2, "entity": new["request"], "withdrawn": old["request"]}],
                    stale=[old["request"], str(old.get("expected_endpoint") or "")],
                    citation_updates=["remove old-object citation mappings", "add new-object endpoint citations"],
                    final_objects=[new["request"]],
                ),
            }
        )

    for index, row in enumerate(rows):
        row["case_index"] = index
        row["case_hash"] = stable_hash({key: value for key, value in row.items() if key != "case_hash"})
    if len(rows) != 20:
        raise AssertionError(f"Expected 20 multi-turn tasks, built {len(rows)}")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Build 20 deterministic multi-turn state-update tasks.")
    parser.add_argument("config", nargs="?", default=str(RELIABILITY_ROOT / "configs" / "base.yaml"))
    parser.add_argument("--output", default=str(RELIABILITY_ROOT / "manifests" / "multi_turn_tasks.jsonl"))
    args = parser.parse_args()
    rows = build(load_config(args.config))
    path = write_jsonl(Path(args.output), rows)
    print(f"wrote {len(rows)} multi-turn tasks to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
