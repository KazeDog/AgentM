#!/usr/bin/env python
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import random
from pathlib import Path

try:
    from .common import RELIABILITY_ROOT, load_config, load_jsonl, stable_hash, write_jsonl
    from .gold_rules import FAULT_TYPES, gold_for_case, validate_stress_case
except ImportError:
    from common import RELIABILITY_ROOT, load_config, load_jsonl, stable_hash, write_jsonl
    from gold_rules import FAULT_TYPES, gold_for_case, validate_stress_case


def _materialize_reactions(rows: list[dict], data_root: str) -> list[dict]:
    from scripts.run_reaction_benchmark import materialize_case

    return [materialize_case(row, data_root) for row in rows]


def _sample(rows: list[dict], count: int, rng: random.Random) -> list[dict]:
    if len(rows) < count:
        raise ValueError(f"Need {count} source cases, found {len(rows)}")
    candidates = list(rows)
    rng.shuffle(candidates)
    return candidates[:count]


STRESS_PROTOCOL = "compatible_successful_targets"
PREDICTION_TARGET_FAULTS = {
    "missing_field_or_overclaim",
    "plausible_but_wrong_prediction",
    "prediction_tool_or_service_failure",
}


def _fault_schedule(task_group: str) -> list[str]:
    """Return a balanced schedule whose mutation target exists in the task.

    ADMET and DDI cases expose both predictions and evidence, so they receive
    two instances of every fault.  Pure reaction-evidence tasks do not expose a
    prediction record or prediction tool.  Prediction-only faults are therefore
    concentrated in reaction-prediction cases and the remaining evidence faults
    are counterbalanced so that every fault still has eight cases globally.
    """

    if task_group == "admet":
        return list(FAULT_TYPES) * 2
    if task_group == "ddi":
        schedule = list(FAULT_TYPES) * 2
        return schedule[2:] + schedule[:2]
    if task_group == "reaction_prediction":
        return [
            "missing_field_or_overclaim",
            "plausible_but_wrong_prediction",
            "prediction_tool_or_service_failure",
            "entity_or_endpoint_mismatch",
            "missing_field_or_overclaim",
            "plausible_but_wrong_prediction",
            "prediction_tool_or_service_failure",
            "irrelevant_or_weak_evidence",
            "missing_field_or_overclaim",
            "plausible_but_wrong_prediction",
            "prediction_tool_or_service_failure",
            "conflicting_evidence",
            "missing_field_or_overclaim",
            "plausible_but_wrong_prediction",
            "prediction_tool_or_service_failure",
            "no_direct_evidence_or_ambiguous_entity",
        ]
    if task_group == "reaction_evidence":
        return [
            "citation_mismatch",
            "entity_or_endpoint_mismatch",
            "irrelevant_or_weak_evidence",
            "conflicting_evidence",
            "no_direct_evidence_or_ambiguous_entity",
            "citation_mismatch",
            "entity_or_endpoint_mismatch",
            "irrelevant_or_weak_evidence",
            "conflicting_evidence",
            "no_direct_evidence_or_ambiguous_entity",
            "citation_mismatch",
            "entity_or_endpoint_mismatch",
            "irrelevant_or_weak_evidence",
            "conflicting_evidence",
            "no_direct_evidence_or_ambiguous_entity",
            "citation_mismatch",
        ]
    raise ValueError(f"Unknown stress task group: {task_group}")


def _safety_sources(
    *,
    task_group: str,
    named_rows: list[dict],
    scientific_rows: list[dict],
    rng: random.Random,
) -> list[tuple[dict, str]]:
    """Match evidence faults to named RAG cases and prediction faults to held-out gold cases."""

    schedule = _fault_schedule(task_group)
    evidence_count = sum(fault not in PREDICTION_TARGET_FAULTS for fault in schedule)
    prediction_count = len(schedule) - evidence_count
    evidence_cases = iter(_sample(named_rows, evidence_count, rng))
    prediction_cases = iter(_sample(scientific_rows, prediction_count, rng))
    assigned: list[tuple[dict, str]] = []
    for fault in schedule:
        if fault in PREDICTION_TARGET_FAULTS:
            assigned.append((next(prediction_cases), "held_out_prediction_gold"))
        else:
            assigned.append((next(evidence_cases), "named_direct_evidence"))
    return assigned


def build(config: dict) -> list[dict]:
    seed = int(config["experiment"]["seed"])
    rng = random.Random(seed)
    named = load_jsonl(config["paths"]["molecular_safety_manifest"])
    scientific = load_jsonl(config["paths"]["molecular_safety_scientific_manifest"])
    reactions = load_jsonl(config["paths"]["reaction_manifest"])
    reactions = _materialize_reactions(reactions, str(config["paths"]["reaction_data_root"]))

    grouped = {
        "admet": _safety_sources(
            task_group="admet",
            named_rows=[row for row in named if row.get("category") == "admet"],
            scientific_rows=[row for row in scientific if row.get("category") == "admet"],
            rng=rng,
        ),
        "ddi": _safety_sources(
            task_group="ddi",
            named_rows=[row for row in named if row.get("category") == "ddi"],
            scientific_rows=[row for row in scientific if row.get("category") == "ddi"],
            rng=rng,
        ),
        "reaction_prediction": [
            (row, "reaction_benchmark")
            for row in _sample(
                [row for row in reactions if row.get("task_type") in {"reaction_classification", "reaction_yield_prediction"}],
                16,
                rng,
            )
        ],
        "reaction_evidence": [
            (row, "reaction_benchmark")
            for row in _sample(
                [row for row in reactions if row.get("task_type") in {"reaction_condition_retrieval", "reaction_profile"}],
                16,
                rng,
            )
        ],
    }

    built: list[dict] = []
    for group_index, (task_group, source_rows) in enumerate(grouped.items()):
        fault_schedule = _fault_schedule(task_group)
        for local_index, ((base, base_pool), fault) in enumerate(zip(source_rows, fault_schedule)):
            mutation_seed = seed + group_index * 1000 + local_index
            case_id = f"stress_{task_group}_{local_index:02d}_{fault}"
            gold = gold_for_case(
                stress_case_id=case_id,
                base_case=base,
                task_group=task_group,
                fault=fault,
                seed=mutation_seed,
            )
            row = {
                **gold,
                "case_id": case_id,
                "dataset_group": "stress",
                "case_index": len(built),
                "base_case": deepcopy(base),
                "base_case_hash": stable_hash(base),
                "base_case_pool": base_pool,
                "primary_fault_count": 1,
                "requires_successful_base_execution": True,
                "stress_protocol": STRESS_PROTOCOL,
                "primary_analysis": True,
                "protocol_revision_reason": (
                    "fault target is applicable to the selected task type; evidence faults use named cases with "
                    "direct entity/endpoint matches in the frozen evidence index; prediction faults use held-out labels"
                ),
            }
            validate_stress_case(row)
            built.append(row)

    if len(built) != 64:
        raise AssertionError(f"Expected 64 stress cases, built {len(built)}")
    faults = Counter(row["injection_type"] for row in built)
    groups = Counter(row["task_type"] for row in built)
    if set(faults.values()) != {8} or set(groups.values()) != {16}:
        raise AssertionError(f"Unbalanced stress manifest: faults={faults}, groups={groups}")
    return built


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a deterministic 64-case controlled stress manifest.")
    parser.add_argument("config", nargs="?", default=str(RELIABILITY_ROOT / "configs" / "base.yaml"))
    parser.add_argument("--output", default=str(RELIABILITY_ROOT / "manifests" / "stress_controlled.jsonl"))
    args = parser.parse_args()
    rows = build(load_config(args.config))
    path = write_jsonl(Path(args.output), rows)
    print(f"wrote {len(rows)} stress cases to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
