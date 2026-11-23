from __future__ import annotations

from typing import Dict
import warnings

import numpy as np


def classification_metrics(y_true, y_score, y_pred=None, classes=None) -> Dict[str, float]:
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        balanced_accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
        top_k_accuracy_score,
    )

    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=float)
    if y_pred is None:
        if y_score.ndim == 2:
            y_pred = np.argmax(y_score, axis=1)
        else:
            y_pred = (y_score >= 0.5).astype(int)
    y_pred = np.asarray(y_pred)
    labels = np.unique(y_true)
    is_multiclass = len(labels) > 2 or y_score.ndim == 2 and y_score.shape[1] > 2
    metrics = {"accuracy": float(accuracy_score(y_true, y_pred))}
    if is_multiclass:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            balanced_accuracy = float(balanced_accuracy_score(y_true, y_pred))
        metrics.update(
            {
                "balanced_accuracy": balanced_accuracy,
                "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
                "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
                "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
                "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
                "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
                "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
            }
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if y_score.ndim == 2:
                try:
                    metrics["roc_auc_ovr_macro"] = float(roc_auc_score(y_true, y_score, multi_class="ovr", average="macro"))
                except Exception:
                    metrics["roc_auc_ovr_macro"] = float("nan")
                try:
                    class_labels = np.asarray(classes if classes is not None else np.arange(y_score.shape[1]))
                    top_k = min(3, y_score.shape[1])
                    top_idx = np.argsort(-y_score, axis=1)[:, :top_k]
                    top_labels = class_labels[top_idx]
                    metrics["top3_accuracy"] = float(np.mean([truth in set(row) for truth, row in zip(y_true, top_labels)]))
                except Exception:
                    try:
                        metrics["top3_accuracy"] = float(top_k_accuracy_score(y_true, y_score, k=min(3, y_score.shape[1]), labels=classes))
                    except Exception:
                        metrics["top3_accuracy"] = float("nan")
        return metrics

    metrics.update(
        {
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        }
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
        except Exception:
            metrics["roc_auc"] = float("nan")
        try:
            metrics["aupr"] = float(average_precision_score(y_true, y_score))
        except Exception:
            metrics["aupr"] = float("nan")
    return metrics


def regression_metrics(y_true, y_pred) -> Dict[str, float]:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from scipy.stats import spearmanr

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mse = mean_squared_error(y_true, y_pred)
    spearman = spearmanr(y_true, y_pred).statistic
    return {
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "spearman": float(spearman),
    }
