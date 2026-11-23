from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

try:
    from .common import canonical_json, require_fields, stable_hash
except ImportError:
    from common import canonical_json, require_fields, stable_hash


FAULT_TYPES = (
    "citation_mismatch",
    "entity_or_endpoint_mismatch",
    "irrelevant_or_weak_evidence",
    "missing_field_or_overclaim",
    "conflicting_evidence",
    "plausible_but_wrong_prediction",
    "prediction_tool_or_service_failure",
    "no_direct_evidence_or_ambiguous_entity",
)

RECOVERABILITY = {
    "citation_mismatch": "recoverable",
    "entity_or_endpoint_mismatch": "recoverable",
    "irrelevant_or_weak_evidence": "recoverable",
    "missing_field_or_overclaim": "recoverable",
    "conflicting_evidence": "recoverable",
    "plausible_but_wrong_prediction": "unrecoverable",
    "prediction_tool_or_service_failure": "unrecoverable",
    "no_direct_evidence_or_ambiguous_entity": "unrecoverable",
}

INJECTION_STAGE = {
    "citation_mismatch": "citation_map",
    "entity_or_endpoint_mismatch": "evidence_record",
    "irrelevant_or_weak_evidence": "evidence_record",
    "missing_field_or_overclaim": "prediction_record",
    "conflicting_evidence": "evidence_record",
    "plausible_but_wrong_prediction": "prediction_record",
    "prediction_tool_or_service_failure": "tool_result",
    "no_direct_evidence_or_ambiguous_entity": "evidence_record",
}

EXPECTED_ACTION = {
    "citation_mismatch": "repair",
    "entity_or_endpoint_mismatch": "retrieve",
    "irrelevant_or_weak_evidence": "weaken_claim",
    "missing_field_or_overclaim": "repair",
    "conflicting_evidence": "preserve_conflict",
    "plausible_but_wrong_prediction": "mark_prediction",
    "prediction_tool_or_service_failure": "abstain",
    "no_direct_evidence_or_ambiguous_entity": "abstain",
}

FINAL_STATUS = {
    "citation_mismatch": "completed",
    "entity_or_endpoint_mismatch": "completed",
    "irrelevant_or_weak_evidence": "partial",
    "missing_field_or_overclaim": "completed",
    "conflicting_evidence": "partial",
    "plausible_but_wrong_prediction": "partial",
    "prediction_tool_or_service_failure": "failed_safely",
    "no_direct_evidence_or_ambiguous_entity": "failed_safely",
}


def gold_for_case(*, stress_case_id: str, base_case: Mapping[str, Any], task_group: str, fault: str, seed: int) -> dict[str, Any]:
    if fault not in FAULT_TYPES:
        raise ValueError(f"Unsupported injection type: {fault}")
    truth = _truth_reference(base_case)
    return {
        "stress_case_id": stress_case_id,
        "base_case_id": str(base_case.get("name") or base_case.get("case_id")),
        "task_type": task_group,
        "injection_type": fault,
        "recoverability": RECOVERABILITY[fault],
        "injection_stage": INJECTION_STAGE[fault],
        "injected_object_id": f"{INJECTION_STAGE[fault]}:primary",
        "gold_detection_target": f"{fault}:{INJECTION_STAGE[fault]}",
        "gold_expected_action": EXPECTED_ACTION[fault],
        "gold_final_status": FINAL_STATUS[fault],
        "gold_supported_claims": [],
        "gold_citation_map": {},
        "gold_conflicting_evidence_ids": [],
        "gold_truth_reference": truth,
        "gold_required_uncertainty": fault
        in {
            "irrelevant_or_weak_evidence",
            "conflicting_evidence",
            "plausible_but_wrong_prediction",
            "prediction_tool_or_service_failure",
            "no_direct_evidence_or_ambiguous_entity",
        },
        "mutation_seed": int(seed),
    }


def _truth_reference(case: Mapping[str, Any]) -> dict[str, Any] | None:
    if case.get("gold_value") is not None:
        return {
            "kind": "held_out_dataset_label",
            "value": case.get("gold_value"),
            "source": case.get("gold_source"),
            "row_id": case.get("gold_row_id"),
            "split": case.get("gold_split"),
        }
    if case.get("source_label") is not None:
        return {"kind": "source_dataset_label", "value": case.get("source_label"), "source": case.get("source_dataset")}
    if case.get("source_yield") is not None:
        return {"kind": "source_dataset_value", "value": case.get("source_yield"), "source": case.get("source_dataset")}
    if case.get("source_dataset") and case.get("row_index") is not None:
        return {
            "kind": "source_row_reference",
            "source": case.get("source_dataset"),
            "split": case.get("split"),
            "part": case.get("part"),
            "row_index": case.get("row_index"),
        }
    return None


def evidence_id(item: Mapping[str, Any], index: int = 0) -> str:
    nested = item.get("value") if isinstance(item.get("value"), Mapping) else {}
    return str(
        item.get("citation_id")
        or item.get("chunk_id")
        or nested.get("chunk_id")
        or item.get("source_path")
        or nested.get("source_path")
        or f"evidence_{index}_{stable_hash(item, length=8)}"
    )


def citation_map_from_evidence(evidence: list[Any]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for index, item in enumerate(evidence):
        if not isinstance(item, Mapping):
            continue
        identifier = evidence_id(item, index)
        mapping[str(item.get("citation_id") or f"claim_{index + 1}")] = identifier
    return mapping


def citation_links_from_evidence(evidence: list[Any]) -> dict[str, str]:
    """Return the claim-to-source links emitted by the runtime verifier.

    ``citation_map_from_evidence`` predates the verifier's citation payload and
    uses the evidence identifier as its target.  The verifier instead prefers a
    concrete chunk id, then a source path, and only then the citation id.  Gold
    generated for a controlled post-retrieval mutation must use the latter
    semantics or a perfectly preserved citation is incorrectly scored as both a
    false positive and a false negative.
    """

    mapping: dict[str, str] = {}
    for index, item in enumerate(evidence):
        if not isinstance(item, Mapping):
            continue
        nested = item.get("value") if isinstance(item.get("value"), Mapping) else {}
        claim_id = str(item.get("citation_id") or f"claim_{index + 1}")
        target = (
            item.get("chunk_id")
            or nested.get("chunk_id")
            or item.get("source_path")
            or nested.get("source_path")
            or item.get("citation_id")
        )
        if target is not None:
            mapping[claim_id] = str(target)
    return mapping


def validate_stress_case(case: Mapping[str, Any]) -> None:
    required = [
        "stress_case_id",
        "base_case_id",
        "task_type",
        "injection_type",
        "recoverability",
        "injection_stage",
        "injected_object_id",
        "gold_detection_target",
        "gold_expected_action",
        "gold_final_status",
        "gold_supported_claims",
        "gold_citation_map",
        "gold_conflicting_evidence_ids",
        "gold_truth_reference",
        "gold_required_uncertainty",
        "mutation_seed",
        "base_case",
    ]
    require_fields(case, required, label=str(case.get("stress_case_id") or "stress case"))
    if case["injection_type"] not in FAULT_TYPES:
        raise ValueError(f"Invalid injection_type: {case['injection_type']}")
    if case["recoverability"] != RECOVERABILITY[case["injection_type"]]:
        raise ValueError("Recoverability conflicts with the registered gold rule")


def resolve_runtime_gold(case: Mapping[str, Any], pre_context: Mapping[str, Any]) -> dict[str, Any]:
    gold = deepcopy(dict(case))
    evidence = [item for item in pre_context.get("evidence", []) or [] if isinstance(item, Mapping)]
    if not gold.get("gold_citation_map"):
        # Match the verifier/report payload: claims link to a concrete chunk or
        # source path, falling back to the citation id only when neither exists.
        gold["gold_citation_map"] = citation_links_from_evidence(evidence)
    gold["gold_supported_claims"] = [
        str(item.get("claim") or item.get("snippet") or "") for item in evidence if item.get("claim") or item.get("snippet")
    ]
    if gold.get("injection_type") == "conflicting_evidence":
        base_ids = [evidence_id(item, index) for index, item in enumerate(evidence[:1])]
        gold["gold_conflicting_evidence_ids"] = base_ids + [f"injected_conflict_{stable_hash(gold, length=8)}"]
    gold["runtime_gold_hash"] = stable_hash({key: value for key, value in gold.items() if key != "runtime_gold_hash"})
    return gold


def reconcile_runtime_gold_after_injection(
    gold: Mapping[str, Any],
    post_context: Mapping[str, Any],
    mutation_record: Mapping[str, Any],
) -> dict[str, Any]:
    """Align runtime gold with the actual object produced by an injection.

    The stress manifest is frozen before runtime.  Mutated citation ids are
    therefore unknowable until the injection has run.  This helper derives only
    mutation-local identifiers and links; it never changes labels, expected
    actions, recoverability, or final-status gold.
    """

    resolved = deepcopy(dict(gold))
    if resolved.get("injection_type") == "conflicting_evidence":
        evidence = [item for item in post_context.get("evidence", []) or [] if isinstance(item, Mapping)]
        resolved["gold_citation_map"] = citation_links_from_evidence(evidence)
        details = mutation_record.get("details") if isinstance(mutation_record.get("details"), Mapping) else {}
        identifiers = [str(item) for item in details.get("conflicting_evidence_ids", []) or [] if item]
        if not identifiers:
            identifiers = [evidence_id(item, index) for index, item in enumerate(evidence)]
        resolved["gold_conflicting_evidence_ids"] = identifiers
        resolved["gold_supported_claims"] = [
            str(item.get("claim") or item.get("snippet") or "")
            for item in evidence
            if item.get("claim") or item.get("snippet")
        ]
    elif resolved.get("injection_type") == "no_direct_evidence_or_ambiguous_entity":
        # The intended scientifically correct final state contains no claim that
        # requires a citation.  Retaining the pre-injection citation map would
        # incorrectly penalize a safe abstention for removing the injected-away
        # evidence.
        resolved["gold_citation_map"] = {}
        resolved["gold_supported_claims"] = []
        resolved["gold_conflicting_evidence_ids"] = []
    resolved["runtime_gold_hash"] = stable_hash(
        {key: value for key, value in resolved.items() if key != "runtime_gold_hash"}
    )
    return resolved


def mutation_diff_keys(before: Mapping[str, Any], after: Mapping[str, Any]) -> list[str]:
    keys = set(before) | set(after)
    return sorted(key for key in keys if canonical_json(before.get(key)) != canonical_json(after.get(key)))
