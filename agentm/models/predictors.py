from __future__ import annotations

import pickle
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from agentm.schema import PredictionResult


class HeuristicPredictor:
    """Backward-compatible wrapper around ZeroShotPredictionAgent.

    Older code imported HeuristicPredictor directly. Keep that import working,
    but delegate all behavior to the explicit zero-shot predictor.
    """

    def __init__(self):
        from agentm.agents.zero_shot_prediction_agent import ZeroShotPredictionAgent

        self._delegate = ZeroShotPredictionAgent()

    def predict_single(
        self,
        bundle: Any,
        task_type: str,
        dataset_name: str,
    ) -> PredictionResult:
        return self._delegate.predict_single(bundle, task_type=task_type, dataset_name=dataset_name)

    def predict_ddi(self, pair_features: Any, dataset_name: str = "DDI") -> PredictionResult:
        return self._delegate.predict_ddi(pair_features, dataset_name=dataset_name)


class SklearnPredictor:
    def __init__(
        self,
        model: Any,
        mode: str,
        model_type: str,
        feature_schema: Optional[Dict[str, Any]] = None,
        rules: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.model = model
        self.mode = mode
        self.model_type = model_type
        self.feature_schema = feature_schema
        self.rules = rules or []
        self.metadata = metadata or {}

    @classmethod
    def train(
        cls,
        x: np.ndarray,
        y: np.ndarray,
        mode: str,
        model_type: str = "random_forest",
        random_seed: int = 13,
        model_params: Optional[Dict[str, Any]] = None,
    ) -> "SklearnPredictor":
        params = dict(model_params or {})
        if mode == "regression":
            if model_type == "linear" or model_type == "ridge":
                from sklearn.linear_model import Ridge

                model = Ridge(random_state=random_seed)
            else:
                from sklearn.ensemble import RandomForestRegressor

                model = RandomForestRegressor(
                    n_estimators=int(params.get("n_estimators", 100)),
                    random_state=random_seed,
                    n_jobs=int(params.get("n_jobs", -1)),
                )
        else:
            if model_type == "logistic_regression":
                from sklearn.linear_model import LogisticRegression

                model = LogisticRegression(max_iter=int(params.get("max_iter", 1000)), random_state=random_seed)
            elif model_type in {"sgd", "sgd_classifier", "linear_sgd"}:
                from sklearn.linear_model import SGDClassifier
                from sklearn.pipeline import make_pipeline
                from sklearn.preprocessing import MaxAbsScaler, StandardScaler

                estimator = SGDClassifier(
                    loss=str(params.get("loss", "log_loss")),
                    max_iter=int(params.get("max_iter", 100)),
                    tol=float(params.get("tol", 1e-3)),
                    random_state=random_seed,
                    class_weight="balanced",
                    n_jobs=int(params.get("n_jobs", -1)),
                    average=bool(params.get("average", False)),
                )
                scaler_name = str(params.get("scaler", params.get("feature_scaler", "maxabs"))).strip().lower()
                if scaler_name in {"none", "false", "off", "no"}:
                    model = estimator
                elif scaler_name in {"standard", "standard_scaler", "zscore"}:
                    model = make_pipeline(StandardScaler(), estimator)
                elif scaler_name in {"standard_no_mean", "standard_with_mean_false"}:
                    model = make_pipeline(StandardScaler(with_mean=False), estimator)
                elif scaler_name in {"maxabs", "max_abs", "max_abs_scaler"}:
                    model = make_pipeline(MaxAbsScaler(), estimator)
                else:
                    raise ValueError(f"Unsupported SGD feature scaler: {scaler_name}")
            else:
                from sklearn.ensemble import RandomForestClassifier

                model = RandomForestClassifier(
                    n_estimators=int(params.get("n_estimators", 100)),
                    random_state=random_seed,
                    class_weight=params.get("class_weight", "balanced"),
                    n_jobs=int(params.get("n_jobs", -1)),
                )
        model.fit(x, y)
        return cls(model=model, mode=mode, model_type=model_type)

    def predict_vector(self, vector: Iterable[float], task_type: str, dataset_name: str) -> PredictionResult:
        values = list(vector)
        if self.feature_schema and "feature_names" in self.feature_schema:
            expected = len(self.feature_schema["feature_names"])
            if len(values) != expected:
                raise ValueError(f"Feature dimension mismatch: got {len(values)}, expected {expected}.")
        x = np.asarray([values], dtype=float)
        pred = self.model.predict(x)[0]
        probability = None
        score = float(pred)
        if self.mode == "classification":
            scores_matrix = self.predict_scores(x)
            if scores_matrix is not None:
                proba = np.asarray(scores_matrix[0], dtype=float)
                classes = list(getattr(self.model, "classes_", range(len(proba))))
                if len(proba) > 2:
                    best_idx = int(np.argmax(proba))
                    probability = float(proba[best_idx])
                    score = probability
                    pred = classes[best_idx]
                    threshold = None
                else:
                    probability = float(proba[1]) if len(proba) > 1 else float(proba[0])
                    score = probability
                    threshold = float(self.metadata.get("decision_threshold", 0.5))
                    pred = int(probability >= threshold)
            else:
                threshold = float(self.metadata.get("decision_threshold", 0.5))
                pred = int(pred)
        else:
            threshold = None
        details = {}
        if self.mode == "classification":
            details = {
                "decision_threshold": threshold,
                "raw_probability": probability,
                "calibrated_probability": self.metadata.get("calibrated_probability"),
                "calibration_status": self.metadata.get("calibration_status", "not_evaluated"),
                "calibration_method": self.metadata.get("calibration_method", "none"),
                "calibration_warning": self.metadata.get(
                    "calibration_warning",
                    "Supervised model probability has not been post-hoc calibrated unless a calibration artifact is explicitly supplied.",
                ),
            }
            classes = list(getattr(self.model, "classes_", [])) if hasattr(self.model, "classes_") else []
            if len(classes) > 2:
                details.update(
                    {
                        "label_mode": self.metadata.get("label_mode", "multiclass"),
                        "classes": [c.item() if hasattr(c, "item") else c for c in classes],
                        "binary_interpretation_applicable": False,
                    }
                )
            else:
                details.update(
                    {
                        "positive_label": self.metadata.get("positive_label", 1),
                        "positive_label_meaning": self.metadata.get("positive_label_meaning"),
                        "negative_label": self.metadata.get("negative_label", 0),
                        "negative_label_meaning": self.metadata.get("negative_label_meaning"),
                        "binary_interpretation_applicable": True,
                    }
                )
        return PredictionResult(
            task_type=task_type,
            dataset_name=dataset_name,
            label=pred.item() if hasattr(pred, "item") else pred,
            score=score,
            probability=probability,
            model_type=self.model_type,
            details=details,
        )

    def predict_scores(self, x: np.ndarray) -> Optional[np.ndarray]:
        """Return finite classification scores suitable for metrics and reports.

        Some sklearn one-vs-rest linear classifiers can emit NaN probabilities for
        rare multiclass cases when all per-class probabilities underflow before
        normalization. In that case, fall back to decision-function scores and a
        stable sigmoid/softmax transform so top-k metrics remain meaningful.
        """

        if self.mode != "classification":
            return None
        if hasattr(self.model, "predict_proba"):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    proba = np.asarray(self.model.predict_proba(x), dtype=float)
                if _is_valid_probability_matrix(proba):
                    return proba
            except Exception:
                pass
        if hasattr(self.model, "decision_function"):
            try:
                decision = np.asarray(self.model.decision_function(x), dtype=float)
                return _decision_to_probability_matrix(decision)
            except Exception:
                pass
        return None

    def save(
        self,
        path: str | Path,
        feature_schema: Optional[Dict[str, Any]] = None,
        rules: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as f:
            pickle.dump(
                {
                    "model": self.model,
                    "mode": self.mode,
                    "model_type": self.model_type,
                    "feature_schema": feature_schema if feature_schema is not None else self.feature_schema,
                    "rules": rules if rules is not None else self.rules,
                    "metadata": metadata if metadata is not None else self.metadata,
                },
                f,
            )
        return target

    @classmethod
    def load(cls, path: str | Path) -> "SklearnPredictor":
        with Path(path).open("rb") as f:
            payload = pickle.load(f)
        return cls(
            model=payload["model"],
            mode=payload["mode"],
            model_type=payload["model_type"],
            feature_schema=payload.get("feature_schema"),
            rules=payload.get("rules"),
            metadata=payload.get("metadata"),
        )


def _is_valid_probability_matrix(proba: np.ndarray) -> bool:
    if proba.size == 0 or not np.all(np.isfinite(proba)):
        return False
    if proba.ndim == 1:
        return True
    row_sums = np.sum(proba, axis=1)
    return bool(np.all(np.isfinite(row_sums)) and np.all(row_sums > 0))


def _decision_to_probability_matrix(decision: np.ndarray) -> np.ndarray:
    if decision.ndim == 1:
        positive = _sigmoid(decision)
        return np.column_stack([1.0 - positive, positive])
    shifted = decision - np.max(decision, axis=1, keepdims=True)
    exp = np.exp(np.clip(shifted, -745.0, 0.0))
    denom = np.sum(exp, axis=1, keepdims=True)
    denom = np.where(denom <= 0, 1.0, denom)
    return exp / denom


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    out = np.empty_like(values, dtype=float)
    positive = values >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    out[~positive] = exp_values / (1.0 + exp_values)
    return out
