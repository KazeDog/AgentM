from __future__ import annotations

from collections import Counter
import math
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


TEST_SPLIT_NAMES = {"test", "held_out_test", "cold_test"}


def as_number(value: Any) -> Optional[float]:
    """Return a finite float for numeric JSON/CSV values, otherwise ``None``."""
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def label_key(value: Any) -> Optional[Any]:
    """Normalize equivalent numeric labels (for example, 1 and 1.0)."""
    number = as_number(value)
    if number is not None:
        rounded = round(number)
        return int(rounded) if math.isclose(number, rounded, abs_tol=1e-12) else number
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def evaluate_scientific_case(
    *,
    gold_value: Any,
    prediction_value: Any,
    mode: str,
    gold_split: str = "test",
    allow_non_test: bool = False,
) -> Dict[str, Any]:
    """Evaluate one prediction while enforcing a held-out-test boundary by default."""
    normalized_mode = str(mode or "").strip().lower()
    normalized_split = str(gold_split or "").strip().lower()
    result: Dict[str, Any] = {
        "scientific_mode": normalized_mode,
        "gold_value": gold_value,
        "prediction_value": prediction_value,
        "gold_split": normalized_split or None,
        "scientific_evaluable": False,
        "scientific_status": "pending",
        "scientific_correct": None,
        "absolute_error": None,
    }
    if gold_value is None:
        result["scientific_status"] = "missing_gold"
        return result
    if prediction_value is None:
        result["scientific_status"] = "missing_prediction"
        return result
    if not allow_non_test and normalized_split not in TEST_SPLIT_NAMES:
        result["scientific_status"] = "excluded_non_test"
        return result

    if normalized_mode == "classification":
        gold_label = label_key(gold_value)
        predicted_label = label_key(prediction_value)
        if gold_label is None:
            result["scientific_status"] = "invalid_gold"
            return result
        if predicted_label is None:
            result["scientific_status"] = "invalid_prediction"
            return result
        result["gold_value"] = gold_label
        result["prediction_value"] = predicted_label
        result["scientific_correct"] = bool(gold_label == predicted_label)
    elif normalized_mode == "regression":
        gold_number = as_number(gold_value)
        prediction_number = as_number(prediction_value)
        if gold_number is None:
            result["scientific_status"] = "invalid_gold"
            return result
        if prediction_number is None:
            result["scientific_status"] = "invalid_prediction"
            return result
        result["gold_value"] = gold_number
        result["prediction_value"] = prediction_number
        result["absolute_error"] = abs(prediction_number - gold_number)
    else:
        result["scientific_status"] = "unsupported_mode"
        return result

    result["scientific_evaluable"] = True
    result["scientific_status"] = "evaluated"
    return result


def classification_metrics(y_true: Sequence[Any], y_pred: Sequence[Any]) -> Dict[str, Any]:
    pairs = [
        (gold, predicted)
        for gold_raw, predicted_raw in zip(y_true, y_pred)
        if (gold := label_key(gold_raw)) is not None and (predicted := label_key(predicted_raw)) is not None
    ]
    if not pairs:
        return {"n": 0, "status": "not_evaluable"}

    true_values = [gold for gold, _ in pairs]
    predicted_values = [predicted for _, predicted in pairs]
    labels = sorted(set(true_values) | set(predicted_values), key=lambda item: str(item))
    true_counts = Counter(true_values)
    predicted_counts = Counter(predicted_values)
    correct_counts = Counter(gold for gold, predicted in pairs if gold == predicted)
    per_label: Dict[str, Dict[str, Any]] = {}
    precisions: List[float] = []
    recalls: List[float] = []
    f1_values: List[float] = []
    weighted_f1_numerator = 0.0

    for label in labels:
        true_positive = correct_counts[label]
        precision = _safe_ratio(true_positive, predicted_counts[label])
        recall = _safe_ratio(true_positive, true_counts[label])
        f1 = _f1(precision, recall)
        precisions.append(precision)
        recalls.append(recall)
        f1_values.append(f1)
        weighted_f1_numerator += f1 * true_counts[label]
        per_label[str(label)] = {
            "support": true_counts[label],
            "predicted": predicted_counts[label],
            "precision": _round(precision),
            "recall": _round(recall),
            "f1": _round(f1),
        }

    true_labels = [label for label in labels if true_counts[label] > 0]
    balanced_accuracy = _mean(
        _safe_ratio(correct_counts[label], true_counts[label]) for label in true_labels
    )
    n_correct = sum(1 for gold, predicted in pairs if gold == predicted)
    accuracy = n_correct / len(pairs)
    ci_low, ci_high = wilson_interval(n_correct, len(pairs))
    return {
        "n": len(pairs),
        "status": "evaluated",
        "n_correct": n_correct,
        "accuracy": _round(accuracy),
        "accuracy_wilson_95ci": [_round(ci_low), _round(ci_high)],
        "balanced_accuracy": _round(balanced_accuracy),
        "macro_precision": _round(_mean(precisions)),
        "macro_recall": _round(_mean(recalls)),
        "macro_f1": _round(_mean(f1_values)),
        "weighted_f1": _round(weighted_f1_numerator / len(pairs)),
        "n_labels_true": len(set(true_values)),
        "n_labels_predicted": len(set(predicted_values)),
        "per_label": per_label,
    }


def regression_metrics(y_true: Sequence[Any], y_pred: Sequence[Any]) -> Dict[str, Any]:
    pairs: List[Tuple[float, float]] = []
    for gold_raw, predicted_raw in zip(y_true, y_pred):
        gold = as_number(gold_raw)
        predicted = as_number(predicted_raw)
        if gold is not None and predicted is not None:
            pairs.append((gold, predicted))
    if not pairs:
        return {"n": 0, "status": "not_evaluable"}

    errors = [predicted - gold for gold, predicted in pairs]
    absolute_errors = [abs(error) for error in errors]
    squared_errors = [error * error for error in errors]
    gold_values = [gold for gold, _ in pairs]
    gold_mean = _mean(gold_values)
    total_sum_squares = sum((gold - gold_mean) ** 2 for gold in gold_values)
    residual_sum_squares = sum(squared_errors)
    r2 = None if total_sum_squares <= 0 else 1.0 - residual_sum_squares / total_sum_squares
    return {
        "n": len(pairs),
        "status": "evaluated",
        "mae": _round(_mean(absolute_errors)),
        "rmse": _round(math.sqrt(_mean(squared_errors))),
        "r2": _round(r2),
    }


def summarize_scientific_rows(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    evaluated = [row for row in rows if bool(row.get("scientific_evaluable"))]
    classification_rows = [
        row for row in evaluated if str(row.get("scientific_mode") or "").lower() == "classification"
    ]
    regression_rows = [
        row for row in evaluated if str(row.get("scientific_mode") or "").lower() == "regression"
    ]
    status_counts = Counter(str(row.get("scientific_status") or "missing") for row in rows)
    split_counts = Counter(str(row.get("gold_split") or "missing") for row in rows)
    result: Dict[str, Any] = {
        "coverage": {
            "n_cases": len(rows),
            "n_evaluable": len(evaluated),
            "evaluable_rate": _round(_safe_ratio(len(evaluated), len(rows))) if rows else None,
            "status_counts": dict(status_counts),
            "gold_split_counts": dict(split_counts),
        },
        "classification": _classification_summary_for_rows(classification_rows),
        "regression": _regression_summary_for_rows(regression_rows),
    }

    result["by_category"] = _group_summaries(evaluated, "category")
    result["by_dataset"] = _group_summaries(evaluated, "dataset_name")
    return result


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> Tuple[float, float]:
    if total <= 0:
        return (math.nan, math.nan)
    proportion = successes / total
    denominator = 1.0 + (z * z / total)
    center = (proportion + z * z / (2.0 * total)) / denominator
    half_width = (
        z
        * math.sqrt((proportion * (1.0 - proportion) / total) + (z * z / (4.0 * total * total)))
        / denominator
    )
    return max(0.0, center - half_width), min(1.0, center + half_width)


def _group_summaries(rows: Sequence[Mapping[str, Any]], key: str) -> Dict[str, Any]:
    groups: Dict[str, List[Mapping[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row.get(key) or "missing"), []).append(row)
    output: Dict[str, Any] = {}
    for name, group in sorted(groups.items()):
        classification_rows = [
            row for row in group if str(row.get("scientific_mode") or "").lower() == "classification"
        ]
        regression_rows = [
            row for row in group if str(row.get("scientific_mode") or "").lower() == "regression"
        ]
        output[name] = {
            "n": len(group),
            "classification": _classification_summary_for_rows(classification_rows),
            "regression": _regression_summary_for_rows(regression_rows),
        }
    return output


def _classification_summary_for_rows(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    metrics = classification_metrics(
        [row.get("gold_value") for row in rows],
        [row.get("prediction_value") for row in rows],
    )
    datasets = sorted({str(row.get("dataset_name") or "missing") for row in rows})
    if len(datasets) <= 1 or metrics.get("status") != "evaluated":
        return metrics
    return {
        "n": metrics["n"],
        "status": "heterogeneous_datasets_not_pooled",
        "n_correct": metrics["n_correct"],
        "exact_match_accuracy": metrics["accuracy"],
        "accuracy_wilson_95ci": metrics["accuracy_wilson_95ci"],
        "datasets": datasets,
        "reason": (
            "Macro precision/recall/F1 are not pooled across endpoints with different "
            "label semantics. Use by_dataset metrics."
        ),
    }


def _regression_summary_for_rows(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    metrics = regression_metrics(
        [row.get("gold_value") for row in rows],
        [row.get("prediction_value") for row in rows],
    )
    datasets = sorted({str(row.get("dataset_name") or "missing") for row in rows})
    if len(datasets) <= 1 or metrics.get("status") != "evaluated":
        return metrics
    return {
        "n": metrics["n"],
        "status": "heterogeneous_datasets_not_pooled",
        "datasets": datasets,
        "reason": (
            "MAE/RMSE/R2 are not pooled across endpoints with different units and scales. "
            "Use by_dataset metrics."
        ),
    }


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _f1(precision: float, recall: float) -> float:
    return 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0


def _mean(values: Iterable[float]) -> float:
    materialized = list(values)
    return sum(materialized) / len(materialized) if materialized else 0.0


def _round(value: Optional[float], digits: int = 6) -> Optional[float]:
    if value is None or not math.isfinite(value):
        return None
    return round(float(value), digits)
