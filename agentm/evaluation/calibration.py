from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple
import warnings

import numpy as np
import pandas as pd

from agentm.models.tasks import task_mode
from agentm.utils.io import ensure_dir, write_json


def evaluate_calibration_from_files(
    *,
    predictions_path: str | Path,
    output_dir: str | Path,
    task_type: str,
    dataset_name: str,
    mode: Optional[str] = None,
    n_bins: int = 10,
    decision_threshold: float = 0.5,
) -> Dict[str, Any]:
    predictions = pd.read_csv(predictions_path)
    return evaluate_calibration(
        predictions,
        output_dir=output_dir,
        task_type=task_type,
        dataset_name=dataset_name,
        mode=mode,
        n_bins=n_bins,
        decision_threshold=decision_threshold,
        source_predictions_path=str(predictions_path),
    )


def evaluate_calibration(
    predictions: pd.DataFrame,
    *,
    output_dir: str | Path,
    task_type: str,
    dataset_name: str,
    mode: Optional[str] = None,
    n_bins: int = 10,
    decision_threshold: float = 0.5,
    source_predictions_path: Optional[str] = None,
) -> Dict[str, Any]:
    out = ensure_dir(output_dir)
    inferred_mode = mode or task_mode(task_type, dataset_name)
    if inferred_mode == "classification":
        result = _evaluate_classification(
            predictions,
            output_dir=out,
            task_type=task_type,
            dataset_name=dataset_name,
            n_bins=n_bins,
            decision_threshold=decision_threshold,
        )
    elif inferred_mode == "regression":
        result = _evaluate_regression(
            predictions,
            output_dir=out,
            task_type=task_type,
            dataset_name=dataset_name,
        )
    else:
        raise ValueError(f"Unsupported calibration mode: {inferred_mode}")

    result["task_type"] = task_type
    result["dataset_name"] = dataset_name
    result["mode"] = inferred_mode
    result["source_predictions_path"] = source_predictions_path
    result["output_dir"] = str(out)
    write_json(out / "calibration_result.json", result)
    _write_markdown_summary(out / "calibration_summary.md", result)
    return result


def zero_shot_uncertainty_summary(prediction: Mapping[str, Any]) -> Dict[str, Any]:
    details = prediction.get("details") if isinstance(prediction.get("details"), Mapping) else {}
    score = prediction.get("score")
    try:
        score_value = float(score)
        margin = abs(score_value - 0.5)
    except Exception:
        score_value = None
        margin = None
    return {
        "calibration_status": "not_calibrated",
        "calibration_method": "none",
        "uncertainty_method": "zero_shot_margin_and_evidence_contributions",
        "calibrated_probability": None,
        "zero_shot_score": details.get("zero_shot_score", score_value),
        "margin_from_threshold": margin,
        "confidence_level": details.get("confidence_level", "unknown"),
        "warning": "Zero-shot agent scores are ranking/risk scores, not calibrated probabilities.",
    }


def _evaluate_classification(
    predictions: pd.DataFrame,
    *,
    output_dir: Path,
    task_type: str,
    dataset_name: str,
    n_bins: int,
    decision_threshold: float,
) -> Dict[str, Any]:
    frame = _clean_classification_predictions(predictions)
    if _is_multiclass_prediction_frame(frame):
        return _evaluate_multiclass_confidence(
            frame,
            output_dir=output_dir,
            task_type=task_type,
            dataset_name=dataset_name,
            n_bins=n_bins,
        )
    valid = frame[frame["split"].astype(str) == "valid"].copy()
    test = frame[frame["split"].astype(str) == "test"].copy()
    if test.empty:
        test = frame.copy()
    raw_test = _classification_metric_block(test["label_true"].to_numpy(), test["score"].to_numpy(), n_bins=n_bins, threshold=decision_threshold)
    raw_valid = _classification_metric_block(valid["label_true"].to_numpy(), valid["score"].to_numpy(), n_bins=n_bins, threshold=decision_threshold) if not valid.empty else None

    methods: Dict[str, Any] = {
        "raw": {
            "status": "ok",
            "valid_metrics": raw_valid,
            "test_metrics": raw_test,
            "test_probabilities_column": "score",
        }
    }
    calibrated_columns = {"raw": test["score"].to_numpy(dtype=float)}

    platt = _fit_platt(valid)
    if platt is not None:
        valid_prob = platt.predict_proba(valid[["score"]].to_numpy(dtype=float))[:, 1]
        test_prob = platt.predict_proba(test[["score"]].to_numpy(dtype=float))[:, 1]
        methods["platt"] = {
            "status": "ok",
            "valid_metrics": _classification_metric_block(valid["label_true"].to_numpy(), valid_prob, n_bins=n_bins, threshold=decision_threshold),
            "test_metrics": _classification_metric_block(test["label_true"].to_numpy(), test_prob, n_bins=n_bins, threshold=decision_threshold),
            "test_probabilities_column": "platt_probability",
        }
        calibrated_columns["platt"] = test_prob
        test["platt_probability"] = test_prob
    else:
        methods["platt"] = {"status": "skipped", "reason": _calibrator_skip_reason(valid)}

    isotonic = _fit_isotonic(valid)
    if isotonic is not None:
        valid_prob = isotonic.predict(valid["score"].to_numpy(dtype=float))
        test_prob = isotonic.predict(test["score"].to_numpy(dtype=float))
        methods["isotonic"] = {
            "status": "ok",
            "valid_metrics": _classification_metric_block(valid["label_true"].to_numpy(), valid_prob, n_bins=n_bins, threshold=decision_threshold),
            "test_metrics": _classification_metric_block(test["label_true"].to_numpy(), test_prob, n_bins=n_bins, threshold=decision_threshold),
            "test_probabilities_column": "isotonic_probability",
        }
        calibrated_columns["isotonic"] = test_prob
        test["isotonic_probability"] = test_prob
    else:
        methods["isotonic"] = {"status": "skipped", "reason": _calibrator_skip_reason(valid, min_rows=8)}

    recommended = _recommended_classification_method(methods)
    reliability_path = _plot_reliability_diagram(
        output_dir / "reliability_diagram.png",
        test["label_true"].to_numpy(),
        calibrated_columns,
        n_bins=n_bins,
        title=f"{dataset_name} calibration",
    )
    histogram_path = _plot_confidence_histogram(
        output_dir / "confidence_histogram.png",
        calibrated_columns.get(recommended, calibrated_columns["raw"]),
        title=f"{dataset_name} confidence",
    )
    selective = _selective_prediction_curve(
        test["label_true"].to_numpy(),
        calibrated_columns.get(recommended, calibrated_columns["raw"]),
        threshold=decision_threshold,
    )
    selective_path = output_dir / "selective_prediction.csv"
    pd.DataFrame(selective).to_csv(selective_path, index=False)
    test_path = output_dir / "test_predictions_with_calibration.csv"
    test.to_csv(test_path, index=False)

    return {
        "calibration_status": "evaluated",
        "calibration_scope": "supervised_classification",
        "n_total_predictions": int(len(frame)),
        "n_valid": int(len(valid)),
        "n_test": int(len(test)),
        "n_bins": int(n_bins),
        "decision_threshold": float(decision_threshold),
        "methods": methods,
        "recommended_method": recommended,
        "recommended_method_note": "Selected by lowest valid Brier score when available; otherwise raw probabilities are kept.",
        "artifacts": {
            "reliability_diagram_path": str(reliability_path) if reliability_path else None,
            "confidence_histogram_path": str(histogram_path) if histogram_path else None,
            "selective_prediction_path": str(selective_path),
            "test_predictions_with_calibration_path": str(test_path),
        },
    }


def _evaluate_multiclass_confidence(
    frame: pd.DataFrame,
    *,
    output_dir: Path,
    task_type: str,
    dataset_name: str,
    n_bins: int,
) -> Dict[str, Any]:
    valid = frame[frame["split"].astype(str) == "valid"].copy()
    test = frame[frame["split"].astype(str) == "test"].copy()
    if test.empty:
        test = frame.copy()

    raw_valid = _multiclass_confidence_metric_block(valid, n_bins=n_bins) if not valid.empty else None
    raw_test = _multiclass_confidence_metric_block(test, n_bins=n_bins)
    methods: Dict[str, Any] = {
        "raw_top_label_confidence": {
            "status": "ok",
            "valid_metrics": raw_valid,
            "test_metrics": raw_test,
            "test_probabilities_column": "score",
        }
    }
    calibrated_columns = {"raw_top_label_confidence": test["score"].to_numpy(dtype=float)}

    platt = _fit_confidence_calibrator(valid, method="platt")
    if platt is not None:
        valid_prob = platt.predict_proba(valid[["score"]].to_numpy(dtype=float))[:, 1]
        test_prob = platt.predict_proba(test[["score"]].to_numpy(dtype=float))[:, 1]
        methods["platt_top_label"] = {
            "status": "ok",
            "valid_metrics": _multiclass_confidence_metric_block(valid.assign(score=valid_prob), n_bins=n_bins),
            "test_metrics": _multiclass_confidence_metric_block(test.assign(score=test_prob), n_bins=n_bins),
            "test_probabilities_column": "platt_top_label_confidence",
        }
        calibrated_columns["platt_top_label"] = test_prob
        test["platt_top_label_confidence"] = test_prob
    else:
        methods["platt_top_label"] = {"status": "skipped", "reason": _confidence_calibrator_skip_reason(valid)}

    isotonic = _fit_confidence_calibrator(valid, method="isotonic")
    if isotonic is not None:
        valid_prob = isotonic.predict(valid["score"].to_numpy(dtype=float))
        test_prob = isotonic.predict(test["score"].to_numpy(dtype=float))
        methods["isotonic_top_label"] = {
            "status": "ok",
            "valid_metrics": _multiclass_confidence_metric_block(valid.assign(score=valid_prob), n_bins=n_bins),
            "test_metrics": _multiclass_confidence_metric_block(test.assign(score=test_prob), n_bins=n_bins),
            "test_probabilities_column": "isotonic_top_label_confidence",
        }
        calibrated_columns["isotonic_top_label"] = test_prob
        test["isotonic_top_label_confidence"] = test_prob
    else:
        methods["isotonic_top_label"] = {"status": "skipped", "reason": _confidence_calibrator_skip_reason(valid, min_rows=8)}

    recommended = _recommended_multiclass_method(methods)
    reliability_path = _plot_reliability_diagram(
        output_dir / "top_label_reliability_diagram.png",
        test["is_correct"].to_numpy(dtype=int),
        calibrated_columns,
        n_bins=n_bins,
        title=f"{dataset_name} top-label calibration",
    )
    histogram_path = _plot_confidence_histogram(
        output_dir / "top_label_confidence_histogram.png",
        calibrated_columns.get(recommended, calibrated_columns["raw_top_label_confidence"]),
        title=f"{dataset_name} top-label confidence",
    )
    selective = _multiclass_selective_prediction_curve(test, calibrated_columns.get(recommended, calibrated_columns["raw_top_label_confidence"]))
    selective_path = output_dir / "selective_prediction.csv"
    pd.DataFrame(selective).to_csv(selective_path, index=False)
    test_path = output_dir / "test_predictions_with_calibration.csv"
    test.to_csv(test_path, index=False)

    return {
        "calibration_status": "evaluated",
        "calibration_scope": "supervised_multiclass_top_label_confidence",
        "task_type": task_type,
        "dataset_name": dataset_name,
        "n_total_predictions": int(len(frame)),
        "n_valid": int(len(valid)),
        "n_test": int(len(test)),
        "n_classes_observed": int(frame["label_true"].nunique()),
        "n_bins": int(n_bins),
        "methods": methods,
        "recommended_method": recommended,
        "recommended_method_note": "Multiclass DDI calibration treats top-label confidence as confidence that the predicted event label is correct.",
        "artifacts": {
            "reliability_diagram_path": str(reliability_path) if reliability_path else None,
            "confidence_histogram_path": str(histogram_path) if histogram_path else None,
            "selective_prediction_path": str(selective_path),
            "test_predictions_with_calibration_path": str(test_path),
        },
    }


def _evaluate_regression(
    predictions: pd.DataFrame,
    *,
    output_dir: Path,
    task_type: str,
    dataset_name: str,
) -> Dict[str, Any]:
    _ = task_type
    _ = dataset_name
    frame = _clean_regression_predictions(predictions)
    valid = frame[frame["split"].astype(str) == "valid"].copy()
    test = frame[frame["split"].astype(str) == "test"].copy()
    if test.empty:
        test = frame.copy()
    residual_source = valid if not valid.empty else test
    residual_source_name = "valid" if not valid.empty else "test"
    residuals = residual_source["label_true"].to_numpy(dtype=float) - residual_source["prediction"].to_numpy(dtype=float)
    test_residuals = test["label_true"].to_numpy(dtype=float) - test["prediction"].to_numpy(dtype=float)
    intervals = _regression_intervals(test["prediction"].to_numpy(dtype=float), test["label_true"].to_numpy(dtype=float), residuals)
    interval_path = output_dir / "regression_interval_coverage.csv"
    pd.DataFrame(intervals).to_csv(interval_path, index=False)
    test_with_intervals = test.copy()
    for item in intervals:
        level = int(item["interval_level"] * 100)
        q = float(item["absolute_residual_quantile"])
        test_with_intervals[f"pi{level}_lower"] = test_with_intervals["prediction"] - q
        test_with_intervals[f"pi{level}_upper"] = test_with_intervals["prediction"] + q
    test_path = output_dir / "test_predictions_with_intervals.csv"
    test_with_intervals.to_csv(test_path, index=False)
    scatter_path = _plot_regression_predictions(output_dir / "regression_predicted_vs_observed.png", test)

    return {
        "calibration_status": "evaluated",
        "calibration_scope": "supervised_regression_intervals",
        "n_total_predictions": int(len(frame)),
        "n_valid": int(len(valid)),
        "n_test": int(len(test)),
        "residual_source_split": residual_source_name,
        "residual_source_warning": "test_residuals_used_for_interval_estimation" if residual_source_name == "test" else None,
        "point_metrics": _regression_metric_block(test["label_true"].to_numpy(), test["prediction"].to_numpy()),
        "residual_summary": _residual_summary(test_residuals),
        "prediction_intervals": intervals,
        "artifacts": {
            "interval_coverage_path": str(interval_path),
            "test_predictions_with_intervals_path": str(test_path),
            "predicted_vs_observed_path": str(scatter_path) if scatter_path else None,
        },
    }


def _clean_classification_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    frame = predictions.copy()
    if "split" not in frame:
        frame["split"] = "test"
    label_col = _first_existing(frame, ["label_true", "y_true", "label"])
    score_col = _first_existing(frame, ["probability", "score", "y_score"])
    if label_col is None or score_col is None:
        raise ValueError("Classification calibration requires label_true/y_true and score/probability columns.")
    frame["label_true"] = pd.to_numeric(frame[label_col], errors="coerce")
    frame["score"] = pd.to_numeric(frame[score_col], errors="coerce").clip(0.0, 1.0)
    pred_col = _first_existing(frame, ["label_pred", "prediction", "top1_label", "y_pred"])
    if pred_col is not None:
        frame["label_pred"] = pd.to_numeric(frame[pred_col], errors="coerce")
    frame = frame.dropna(subset=["label_true", "score"]).copy()
    frame["label_true"] = frame["label_true"].astype(int)
    if "label_pred" in frame:
        frame = frame.dropna(subset=["label_pred"]).copy()
        frame["label_pred"] = frame["label_pred"].astype(int)
        frame["is_correct"] = (frame["label_true"] == frame["label_pred"]).astype(int)
    return frame


def _clean_regression_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    frame = predictions.copy()
    if "split" not in frame:
        frame["split"] = "test"
    label_col = _first_existing(frame, ["label_true", "y_true", "label"])
    pred_col = _first_existing(frame, ["prediction", "y_pred", "score"])
    if label_col is None or pred_col is None:
        raise ValueError("Regression uncertainty evaluation requires label_true/y_true and prediction/score columns.")
    frame["label_true"] = pd.to_numeric(frame[label_col], errors="coerce")
    frame["prediction"] = pd.to_numeric(frame[pred_col], errors="coerce")
    return frame.dropna(subset=["label_true", "prediction"]).copy()


def _classification_metric_block(y_true: Iterable[Any], y_prob: Iterable[float], *, n_bins: int, threshold: float) -> Dict[str, Any]:
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        brier_score_loss,
        f1_score,
        log_loss,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    y = np.asarray(y_true, dtype=int)
    p = np.clip(np.asarray(y_prob, dtype=float), 1e-8, 1.0 - 1e-8)
    pred = (p >= threshold).astype(int)
    metrics: Dict[str, Any] = {
        "accuracy": _safe_float(accuracy_score(y, pred)) if len(y) else None,
        "f1": _safe_float(f1_score(y, pred, zero_division=0)) if len(y) else None,
        "precision": _safe_float(precision_score(y, pred, zero_division=0)) if len(y) else None,
        "recall": _safe_float(recall_score(y, pred, zero_division=0)) if len(y) else None,
        "brier_score": _safe_float(brier_score_loss(y, p)) if len(y) else None,
        "nll": None,
        "roc_auc": None,
        "aupr": None,
    }
    try:
        metrics["nll"] = _safe_float(log_loss(y, p, labels=[0, 1]))
    except Exception:
        pass
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            metrics["roc_auc"] = _safe_float(roc_auc_score(y, p))
        except Exception:
            pass
        try:
            metrics["aupr"] = _safe_float(average_precision_score(y, p))
        except Exception:
            pass
    bins, ece, mce = _calibration_bins(y, p, n_bins=n_bins)
    metrics["ece"] = ece
    metrics["mce"] = mce
    metrics["calibration_bins"] = bins
    return metrics


def _is_multiclass_prediction_frame(frame: pd.DataFrame) -> bool:
    if "label_pred" not in frame:
        return False
    try:
        labels = set(frame["label_true"].dropna().astype(int).unique().tolist())
        preds = set(frame["label_pred"].dropna().astype(int).unique().tolist())
    except Exception:
        return False
    values = labels | preds
    return len(values) > 2 or any(value not in {0, 1} for value in values)


def _multiclass_confidence_metric_block(frame: pd.DataFrame, *, n_bins: int) -> Dict[str, Any]:
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score

    y_true = frame["label_true"].to_numpy(dtype=int)
    y_pred = frame["label_pred"].to_numpy(dtype=int)
    confidence = np.clip(frame["score"].to_numpy(dtype=float), 1e-8, 1.0 - 1e-8)
    correct = (y_true == y_pred).astype(int)
    bins, ece, mce = _calibration_bins(correct, confidence, n_bins=n_bins)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        balanced_accuracy = _safe_float(balanced_accuracy_score(y_true, y_pred)) if len(y_true) else None
    return {
        "accuracy": _safe_float(accuracy_score(y_true, y_pred)) if len(y_true) else None,
        "balanced_accuracy": balanced_accuracy,
        "f1_macro": _safe_float(f1_score(y_true, y_pred, average="macro", zero_division=0)) if len(y_true) else None,
        "precision_macro": _safe_float(precision_score(y_true, y_pred, average="macro", zero_division=0)) if len(y_true) else None,
        "recall_macro": _safe_float(recall_score(y_true, y_pred, average="macro", zero_division=0)) if len(y_true) else None,
        "mean_top_label_confidence": _safe_float(np.mean(confidence)) if len(confidence) else None,
        "brier_score_top_label_correctness": _safe_float(np.mean((confidence - correct) ** 2)) if len(correct) else None,
        "nll_top_label_correctness": _safe_float(-np.mean(correct * np.log(confidence) + (1 - correct) * np.log(1 - confidence))) if len(correct) else None,
        "ece": ece,
        "mce": mce,
        "calibration_bins": bins,
    }


def _fit_confidence_calibrator(valid: pd.DataFrame, *, method: str):
    if len(valid) < 4 or "is_correct" not in valid or valid["is_correct"].nunique() < 2:
        return None
    if method == "platt":
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression(max_iter=1000)
        model.fit(valid[["score"]].to_numpy(dtype=float), valid["is_correct"].to_numpy(dtype=int))
        return model
    if method == "isotonic":
        if len(valid) < 8:
            return None
        from sklearn.isotonic import IsotonicRegression

        model = IsotonicRegression(out_of_bounds="clip")
        model.fit(valid["score"].to_numpy(dtype=float), valid["is_correct"].to_numpy(dtype=int))
        return model
    raise ValueError(f"Unsupported confidence calibration method: {method}")


def _confidence_calibrator_skip_reason(valid: pd.DataFrame, *, min_rows: int = 4) -> str:
    if valid.empty:
        return "missing_valid_split"
    if len(valid) < min_rows:
        return f"valid_split_too_small_for_multiclass_confidence_calibration_min_{min_rows}"
    if "is_correct" not in valid:
        return "missing_label_pred_for_top_label_correctness"
    if valid["is_correct"].nunique() < 2:
        return "valid_split_has_single_correctness_class"
    return "not_available"


def _recommended_multiclass_method(methods: Mapping[str, Any]) -> str:
    candidates = []
    for name, payload in methods.items():
        if payload.get("status") != "ok" or not payload.get("valid_metrics"):
            continue
        brier = payload["valid_metrics"].get("brier_score_top_label_correctness")
        if brier is not None:
            candidates.append((float(brier), name))
    if not candidates:
        return "raw_top_label_confidence"
    return sorted(candidates)[0][1]


def _multiclass_selective_prediction_curve(frame: pd.DataFrame, confidence: np.ndarray) -> List[Dict[str, Any]]:
    conf = np.asarray(confidence, dtype=float)
    correct = frame["is_correct"].to_numpy(dtype=int)
    rows = []
    for coverage in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]:
        n_keep = max(1, int(np.ceil(len(frame) * coverage))) if len(frame) else 0
        order = np.argsort(-conf)
        idx = order[:n_keep]
        if n_keep == 0:
            accuracy = None
            mean_conf = None
        else:
            accuracy = float(correct[idx].mean())
            mean_conf = float(conf[idx].mean())
        rows.append({"coverage": float(coverage), "n_kept": int(n_keep), "accuracy": accuracy, "mean_confidence": mean_conf})
    return rows


def _calibration_bins(y_true: np.ndarray, y_prob: np.ndarray, *, n_bins: int) -> Tuple[List[Dict[str, Any]], float, float]:
    if len(y_true) == 0:
        return [], 0.0, 0.0
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows: List[Dict[str, Any]] = []
    ece = 0.0
    mce = 0.0
    for i in range(n_bins):
        left = edges[i]
        right = edges[i + 1]
        if i == n_bins - 1:
            mask = (y_prob >= left) & (y_prob <= right)
        else:
            mask = (y_prob >= left) & (y_prob < right)
        count = int(mask.sum())
        if count == 0:
            rows.append({"bin_left": float(left), "bin_right": float(right), "count": 0, "mean_confidence": None, "empirical_accuracy": None, "gap": None})
            continue
        confidence = float(y_prob[mask].mean())
        accuracy = float(y_true[mask].mean())
        gap = abs(accuracy - confidence)
        ece += (count / len(y_true)) * gap
        mce = max(mce, gap)
        rows.append(
            {
                "bin_left": float(left),
                "bin_right": float(right),
                "count": count,
                "mean_confidence": confidence,
                "empirical_accuracy": accuracy,
                "gap": gap,
            }
        )
    return rows, _safe_float(ece), _safe_float(mce)


def _fit_platt(valid: pd.DataFrame):
    if len(valid) < 4 or valid["label_true"].nunique() < 2:
        return None
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(max_iter=1000)
    model.fit(valid[["score"]].to_numpy(dtype=float), valid["label_true"].to_numpy(dtype=int))
    return model


def _fit_isotonic(valid: pd.DataFrame):
    if len(valid) < 8 or valid["label_true"].nunique() < 2:
        return None
    from sklearn.isotonic import IsotonicRegression

    model = IsotonicRegression(out_of_bounds="clip")
    model.fit(valid["score"].to_numpy(dtype=float), valid["label_true"].to_numpy(dtype=int))
    return model


def _calibrator_skip_reason(valid: pd.DataFrame, *, min_rows: int = 4) -> str:
    if valid.empty:
        return "missing_valid_split"
    if len(valid) < min_rows:
        return f"valid_split_too_small_for_calibration_min_{min_rows}"
    if valid["label_true"].nunique() < 2:
        return "valid_split_has_single_class"
    return "not_available"


def _recommended_classification_method(methods: Mapping[str, Any]) -> str:
    candidates = []
    for name, payload in methods.items():
        if payload.get("status") != "ok" or not payload.get("valid_metrics"):
            continue
        brier = payload["valid_metrics"].get("brier_score")
        if brier is not None:
            candidates.append((float(brier), name))
    if not candidates:
        return "raw"
    return sorted(candidates)[0][1]


def _selective_prediction_curve(y_true: np.ndarray, y_prob: np.ndarray, *, threshold: float) -> List[Dict[str, Any]]:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(y_prob, dtype=float)
    confidence = np.maximum(p, 1.0 - p)
    pred = (p >= threshold).astype(int)
    rows = []
    for coverage in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]:
        n_keep = max(1, int(np.ceil(len(y) * coverage))) if len(y) else 0
        order = np.argsort(-confidence)
        idx = order[:n_keep]
        if n_keep == 0:
            accuracy = None
            mean_conf = None
        else:
            accuracy = float((pred[idx] == y[idx]).mean())
            mean_conf = float(confidence[idx].mean())
        rows.append({"coverage": float(coverage), "n_kept": int(n_keep), "accuracy": accuracy, "mean_confidence": mean_conf})
    return rows


def _regression_metric_block(y_true: Iterable[float], y_pred: Iterable[float]) -> Dict[str, Optional[float]]:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    if len(y) == 0:
        return {"rmse": None, "mae": None, "r2": None}
    return {
        "rmse": _safe_float(np.sqrt(mean_squared_error(y, p))),
        "mae": _safe_float(mean_absolute_error(y, p)),
        "r2": _safe_float(r2_score(y, p)) if len(y) > 1 else None,
    }


def _regression_intervals(prediction: np.ndarray, y_true: np.ndarray, calibration_residuals: np.ndarray) -> List[Dict[str, Any]]:
    abs_residuals = np.abs(np.asarray(calibration_residuals, dtype=float))
    rows = []
    for level in [0.80, 0.90, 0.95]:
        q = float(np.quantile(abs_residuals, level)) if len(abs_residuals) else 0.0
        lower = prediction - q
        upper = prediction + q
        coverage = float(((y_true >= lower) & (y_true <= upper)).mean()) if len(y_true) else None
        rows.append(
            {
                "interval_level": level,
                "absolute_residual_quantile": q,
                "empirical_coverage": coverage,
                "mean_interval_width": 2.0 * q,
                "n_test": int(len(y_true)),
            }
        )
    return rows


def _residual_summary(residuals: np.ndarray) -> Dict[str, Optional[float]]:
    if len(residuals) == 0:
        return {"mean": None, "std": None, "median_abs": None, "q90_abs": None}
    abs_res = np.abs(residuals)
    return {
        "mean": _safe_float(np.mean(residuals)),
        "std": _safe_float(np.std(residuals)),
        "median_abs": _safe_float(np.median(abs_res)),
        "q90_abs": _safe_float(np.quantile(abs_res, 0.90)),
    }


def _plot_reliability_diagram(path: Path, y_true: np.ndarray, probability_by_method: Mapping[str, np.ndarray], *, n_bins: int, title: str) -> Optional[Path]:
    try:
        os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "agentm_matplotlib"))
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None
    fig, ax = plt.subplots(figsize=(5.0, 4.0), dpi=150)
    ax.plot([0, 1], [0, 1], color="#555555", linestyle="--", linewidth=1.0, label="ideal")
    for name, probs in probability_by_method.items():
        bins, _, _ = _calibration_bins(np.asarray(y_true, dtype=int), np.asarray(probs, dtype=float), n_bins=n_bins)
        x = [row["mean_confidence"] for row in bins if row["count"]]
        y = [row["empirical_accuracy"] for row in bins if row["count"]]
        if x:
            ax.plot(x, y, marker="o", linewidth=1.5, label=name)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Empirical positive rate")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def _plot_confidence_histogram(path: Path, y_prob: np.ndarray, *, title: str) -> Optional[Path]:
    try:
        os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "agentm_matplotlib"))
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None
    confidence = np.maximum(y_prob, 1.0 - y_prob)
    fig, ax = plt.subplots(figsize=(5.0, 3.2), dpi=150)
    ax.hist(confidence, bins=np.linspace(0.5, 1.0, 11), color="#4C78A8", edgecolor="white")
    ax.set_xlabel("Prediction confidence")
    ax.set_ylabel("Count")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def _plot_regression_predictions(path: Path, test: pd.DataFrame) -> Optional[Path]:
    try:
        os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "agentm_matplotlib"))
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None
    if test.empty:
        return None
    y = test["label_true"].to_numpy(dtype=float)
    p = test["prediction"].to_numpy(dtype=float)
    lo = float(min(y.min(), p.min()))
    hi = float(max(y.max(), p.max()))
    fig, ax = plt.subplots(figsize=(4.2, 4.2), dpi=150)
    ax.scatter(y, p, s=18, alpha=0.75, color="#4C78A8")
    ax.plot([lo, hi], [lo, hi], color="#555555", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Observed")
    ax.set_ylabel("Predicted")
    ax.set_title("Predicted vs observed")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def _write_markdown_summary(path: Path, result: Mapping[str, Any]) -> None:
    lines = [
        f"# Calibration / Uncertainty Summary: {result.get('dataset_name')}",
        "",
        f"- task_type: `{result.get('task_type')}`",
        f"- mode: `{result.get('mode')}`",
        f"- status: `{result.get('calibration_status')}`",
    ]
    if result.get("mode") == "classification":
        recommended = result.get("recommended_method")
        methods = result.get("methods") or {}
        test_metrics = ((methods.get(recommended) or {}).get("test_metrics") or {}) if recommended else {}
        lines.extend(
            [
                f"- recommended_method: `{recommended}`",
                f"- test ECE: `{_fmt(test_metrics.get('ece'))}`",
                f"- test Brier: `{_fmt(test_metrics.get('brier_score'))}`",
                f"- test NLL: `{_fmt(test_metrics.get('nll'))}`",
            ]
        )
    else:
        point = result.get("point_metrics") or {}
        lines.extend(
            [
                f"- RMSE: `{_fmt(point.get('rmse'))}`",
                f"- MAE: `{_fmt(point.get('mae'))}`",
                f"- R2: `{_fmt(point.get('r2'))}`",
            ]
        )
    artifacts = result.get("artifacts") or {}
    if artifacts:
        lines.append("")
        lines.append("## Artifacts")
        for key, value in artifacts.items():
            if value:
                lines.append(f"- {key}: `{value}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize_calibration_results(results: List[Mapping[str, Any]], output_dir: str | Path) -> Dict[str, Any]:
    out = ensure_dir(output_dir)
    rows = []
    for result in results:
        row = {
            "run_name": _run_name_from_calibration_result(result),
            "dataset_name": result.get("dataset_name"),
            "task_type": result.get("task_type"),
            "mode": result.get("mode"),
            "status": result.get("calibration_status"),
            "n_test": result.get("n_test"),
            "training_result_path": result.get("training_result_path"),
            "source_predictions_path": result.get("source_predictions_path"),
            "calibration_output_dir": result.get("output_dir"),
        }
        if result.get("mode") == "classification":
            recommended = result.get("recommended_method")
            method = ((result.get("methods") or {}).get(recommended) or {}) if recommended else {}
            metrics = method.get("test_metrics") or {}
            row.update(
                {
                    "recommended_method": recommended,
                    "calibration_scope": result.get("calibration_scope"),
                    "ece": metrics.get("ece"),
                    "mce": metrics.get("mce"),
                    "brier_score": metrics.get("brier_score", metrics.get("brier_score_top_label_correctness")),
                    "nll": metrics.get("nll", metrics.get("nll_top_label_correctness")),
                    "roc_auc": metrics.get("roc_auc"),
                    "aupr": metrics.get("aupr"),
                    "accuracy": metrics.get("accuracy"),
                    "balanced_accuracy": metrics.get("balanced_accuracy"),
                    "f1_macro": metrics.get("f1_macro"),
                    "mean_top_label_confidence": metrics.get("mean_top_label_confidence"),
                }
            )
        elif result.get("mode") == "regression":
            metrics = result.get("point_metrics") or {}
            intervals = {str(int(item["interval_level"] * 100)): item for item in result.get("prediction_intervals", [])}
            row.update(
                {
                    "rmse": metrics.get("rmse"),
                    "mae": metrics.get("mae"),
                    "r2": metrics.get("r2"),
                    "coverage_90": (intervals.get("90") or {}).get("empirical_coverage"),
                    "width_90": (intervals.get("90") or {}).get("mean_interval_width"),
                }
            )
        else:
            row["error"] = result.get("error")
        rows.append(row)
    table = pd.DataFrame(rows)
    csv_path = out / "calibration_summary.csv"
    md_path = out / "calibration_summary.md"
    table.to_csv(csv_path, index=False)
    md_path.write_text(_dataframe_to_markdown(table) + "\n", encoding="utf-8")
    payload = {"n_results": len(results), "summary_csv_path": str(csv_path), "summary_markdown_path": str(md_path)}
    write_json(out / "calibration_summary.json", payload)
    return payload


def _run_name_from_calibration_result(result: Mapping[str, Any]) -> str:
    training_path = result.get("training_result_path")
    if training_path:
        try:
            return Path(str(training_path)).parent.name
        except Exception:
            pass
    output_dir = result.get("output_dir")
    if output_dir:
        try:
            output_path = Path(str(output_dir))
            return output_path.name or output_path.parent.name
        except Exception:
            pass
    return str(result.get("dataset_name") or "unknown_run")


def _first_existing(frame: pd.DataFrame, columns: List[str]) -> Optional[str]:
    for column in columns:
        if column in frame:
            return column
    return None


def _dataframe_to_markdown(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "| empty |\n|---|"
    columns = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in frame.iterrows():
        values = []
        for column in frame.columns:
            value = row[column]
            if pd.isna(value):
                text = ""
            elif isinstance(value, float):
                text = f"{value:.6g}"
            else:
                text = str(value)
            values.append(text.replace("|", "\\|"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _safe_float(value: Any) -> Optional[float]:
    try:
        result = float(value)
    except Exception:
        return None
    if not np.isfinite(result):
        return None
    return result


def _fmt(value: Any) -> str:
    try:
        if value is None:
            return "n/a"
        return f"{float(value):.4f}"
    except Exception:
        return str(value)
