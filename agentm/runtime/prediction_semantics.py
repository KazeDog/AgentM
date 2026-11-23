from __future__ import annotations

import csv
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional, Sequence


MULTICLASS_LABEL_MODES = {
    "multiclass",
    "multi_class",
    "multiclass_interaction_type",
    "interaction_type_multiclass",
}

_BINARY_ABSENCE_PATTERNS = [
    re.compile(r"\bno\s+(?:clear|obvious)\s+ddi\s+(?:risk|interaction)(?:\s+signal)?\b", re.IGNORECASE),
    re.compile(r"\bno\s+ddi\s+risk(?:/interaction)?\b", re.IGNORECASE),
    re.compile(r"\bdoes\s+not\s+show\s+(?:a\s+)?(?:clear|obvious)\s+drug[- ]drug\s+interaction", re.IGNORECASE),
    re.compile(r"未显示(?:明确|明显).*药物相互作用"),
    re.compile(r"(?:无|没有)(?:明确|明显).*药物相互作用"),
    re.compile(r"不存在.*药物相互作用"),
]


def prediction_value(prediction: Any, key: str, default: Any = None) -> Any:
    if isinstance(prediction, Mapping):
        return prediction.get(key, default)
    return getattr(prediction, key, default)


def prediction_details(prediction: Any) -> Mapping[str, Any]:
    details = prediction_value(prediction, "details", {})
    return details if isinstance(details, Mapping) else {}


def mutable_prediction_details(prediction: Any) -> MutableMapping[str, Any]:
    details = prediction_value(prediction, "details", None)
    if isinstance(details, MutableMapping):
        return details
    details = {}
    if isinstance(prediction, MutableMapping):
        prediction["details"] = details
    else:
        setattr(prediction, "details", details)
    return details


def is_multiclass_ddi_prediction(prediction: Any) -> bool:
    details = prediction_details(prediction)
    task_type = str(prediction_value(prediction, "task_type", "") or "").strip().lower()
    dataset_name = str(prediction_value(prediction, "dataset_name", "") or "").strip().lower()
    mode = str(details.get("label_mode") or "").strip().lower().replace("-", "_")
    classes = details.get("classes")
    has_many_classes = isinstance(classes, Sequence) and not isinstance(classes, (str, bytes)) and len(classes) > 2
    is_ddi = task_type == "ddi_prediction" or dataset_name in {"ddi", "drugbank"}
    return is_ddi and (mode in MULTICLASS_LABEL_MODES or mode.startswith("multiclass") or has_many_classes)


def binary_prediction_direction(prediction: Any) -> Optional[bool]:
    """Return a binary direction, or None when binary semantics do not apply."""

    if is_multiclass_ddi_prediction(prediction):
        return None
    details = prediction_details(prediction)
    label = prediction_value(prediction, "label")
    positive_label = details.get("positive_label", 1)
    negative_label = details.get("negative_label", 0)
    if _labels_equal(label, positive_label):
        return True
    if _labels_equal(label, negative_label):
        return False
    return None


def annotate_multiclass_ddi_prediction(
    prediction: Any,
    *,
    drug_names: Optional[Sequence[str]] = None,
    model_path: Optional[str | Path] = None,
    interaction_map_path: Optional[str | Path] = None,
) -> Any:
    """Attach interaction-class semantics and explicitly disable binary interpretation.

    Existing model artifacts may contain legacy positive/negative metadata even when
    their class set has 86 DrugBank interaction types. Preserve that metadata only
    under an explicitly ignored audit field, then expose the actual class ontology.
    """

    if not is_multiclass_ddi_prediction(prediction):
        return prediction
    details = mutable_prediction_details(prediction)
    label = prediction_value(prediction, "label")
    legacy = {
        key: details.get(key)
        for key in (
            "positive_label",
            "positive_label_meaning",
            "negative_label",
            "negative_label_meaning",
        )
        if details.get(key) is not None
    }
    if legacy:
        details.setdefault("legacy_binary_metadata_ignored", legacy)
    for key in (
        "positive_label",
        "positive_label_meaning",
        "negative_label",
        "negative_label_meaning",
    ):
        details.pop(key, None)

    map_path = resolve_drugbank_interaction_map_path(
        explicit_path=interaction_map_path,
        model_path=model_path,
    )
    mapping = load_drugbank_interaction_type_map(map_path) if map_path else {}
    template = mapping.get(_label_key(label))
    names = list(drug_names or details.get("open_task_molecules") or [])
    description = render_interaction_template(template, names) if template else ""
    details.update(
        {
            "label_mode": "multiclass_interaction_type",
            "binary_interpretation_applicable": False,
            "interaction_type_class": label,
            "interaction_type_template": template,
            "interaction_type_description": description,
            "interaction_type_mapping_source": str(map_path) if map_path else None,
        }
    )
    return prediction


def prediction_meaning(
    prediction: Any,
    *,
    drug_names: Optional[Sequence[str]] = None,
    chinese: bool = False,
) -> str:
    details = prediction_details(prediction)
    label = prediction_value(prediction, "label")
    if is_multiclass_ddi_prediction(prediction):
        description = str(details.get("interaction_type_description") or "").strip()
        if not description:
            template = str(details.get("interaction_type_template") or "").strip()
            description = render_interaction_template(template, drug_names or details.get("open_task_molecules") or []) if template else ""
        class_id = details.get("interaction_type_class", label)
        if chinese:
            base = f"DrugBank 相互作用类型类别 {class_id}"
            return f"{base}（类别定义：{description}）" if description else f"{base}（当前没有可用的类别文字映射）"
        base = f"DrugBank interaction-type class {class_id}"
        return f"{base} (class definition: {description})" if description else f"{base} (class description unavailable)"

    direction = binary_prediction_direction(prediction)
    if direction is True:
        return str(details.get("positive_label_meaning") or "positive")
    if direction is False:
        return str(details.get("negative_label_meaning") or "negative")
    if str(prediction_value(prediction, "task_type", "") or "") == "ddi_prediction":
        return f"DDI binary label {label}"
    return f"label={label}"


def audit_multiclass_ddi_narrative(context: Mapping[str, Any]) -> Dict[str, Any]:
    predictions = list(context.get("predictions", []) or [])
    multiclass = [prediction for prediction in predictions if is_multiclass_ddi_prediction(prediction)]
    text = _narrative_text(context)
    normalized_text = _space_normalize(text).lower()
    cases = []
    for prediction in multiclass:
        details = prediction_details(prediction)
        label = details.get("interaction_type_class", prediction_value(prediction, "label"))
        label_text = re.escape(str(label))
        class_pattern = re.compile(
            rf"(?:interaction[- ]type\s+class|interaction\s+type\s+category|相互作用类型(?:类别|类)|\bclass)\s*[:=#]?\s*{label_text}\b",
            re.IGNORECASE,
        )
        class_identifier_reported = bool(class_pattern.search(text))
        description = _space_normalize(str(details.get("interaction_type_description") or ""))
        description_reported = not description or description.lower() in normalized_text
        binary_absence_claims = [pattern.pattern for pattern in _BINARY_ABSENCE_PATTERNS if pattern.search(text)]
        issues = []
        if not class_identifier_reported:
            issues.append("interaction_type_class_not_reported")
        if not description_reported:
            issues.append("available_interaction_type_description_not_reported")
        if binary_absence_claims:
            issues.append("multiclass_result_reduced_to_binary_absence")
        cases.append(
            {
                "label": label,
                "class_identifier_reported": class_identifier_reported,
                "class_description_available": bool(description),
                "class_description_reported": description_reported,
                "binary_absence_claim_patterns": binary_absence_claims,
                "issues": issues,
                "passed": not issues,
            }
        )
    passed = all(case["passed"] for case in cases)
    warnings = [] if passed else ["multiclass_ddi_semantics_not_preserved"]
    return {
        "multiclass_ddi_semantics_preserved": passed,
        "n_multiclass_ddi_predictions": len(multiclass),
        "cases": cases,
        "warnings": warnings,
    }


def resolve_drugbank_interaction_map_path(
    *,
    explicit_path: Optional[str | Path] = None,
    model_path: Optional[str | Path] = None,
) -> Optional[Path]:
    candidates = []
    if explicit_path:
        candidates.append(Path(explicit_path))
    configured = os.getenv("AGENTM_DRUGBANK_INTERACTION_MAP_PATH")
    if configured:
        candidates.append(Path(configured))
    if model_path:
        resolved_model = Path(model_path).expanduser().resolve()
        for parent in [resolved_model.parent, *resolved_model.parents]:
            candidates.append(parent / "data" / "tdc" / "raw" / "drugbank.tab")
    project_root = Path(__file__).resolve().parents[2]
    candidates.extend(
        [
            project_root / "data" / "tdc" / "raw" / "drugbank.tab",
        ]
    )
    seen = set()
    for candidate in candidates:
        try:
            resolved = candidate.expanduser().resolve()
        except OSError:
            continue
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.is_file():
            return resolved
    return None


@lru_cache(maxsize=8)
def _load_drugbank_interaction_type_map(path: str) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            label = _label_key(row.get("Y"))
            description = str(row.get("Map") or "").strip()
            if not label or not description:
                continue
            previous = mapping.get(label)
            if previous is not None and previous != description:
                raise ValueError(f"Conflicting DrugBank descriptions for class {label}")
            mapping[label] = description
    return mapping


def load_drugbank_interaction_type_map(path: Optional[str | Path]) -> Dict[str, str]:
    if path is None:
        return {}
    try:
        return dict(_load_drugbank_interaction_type_map(str(Path(path).resolve())))
    except (OSError, ValueError, csv.Error):
        return {}


def render_interaction_template(template: Optional[str], drug_names: Iterable[Any]) -> str:
    text = str(template or "").strip()
    if not text:
        return ""
    names = [str(name).strip() for name in drug_names if str(name).strip()]
    if len(names) >= 1:
        text = text.replace("#Drug1", names[0])
    if len(names) >= 2:
        text = text.replace("#Drug2", names[1])
    return text


def _narrative_text(context: Mapping[str, Any]) -> str:
    values = []
    report = context.get("final_report")
    if isinstance(report, Mapping):
        values.append(report.get("explanation"))
        reasoning = report.get("reasoning")
        if isinstance(reasoning, Mapping):
            values.append(reasoning.get("structured_explanation"))
    reasoning = context.get("reasoning")
    if isinstance(reasoning, Mapping):
        values.append(reasoning.get("structured_explanation"))
    values.append(context.get("explanation"))
    return "\n".join(str(value) for value in values if value)


def _labels_equal(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return False
    try:
        return float(left) == float(right)
    except (TypeError, ValueError):
        return str(left) == str(right)


def _label_key(value: Any) -> str:
    if value is None:
        return ""
    try:
        numeric = float(value)
        if numeric.is_integer():
            return str(int(numeric))
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def _space_normalize(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()
