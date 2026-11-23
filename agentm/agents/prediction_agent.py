from __future__ import annotations

from pathlib import Path
from typing import Optional

from agentm.agents.zero_shot_prediction_agent import ZeroShotPredictionAgent
from agentm.models.predictors import SklearnPredictor
from agentm.schema import DDIPairFeatureBundle, PredictionResult, Rule, SingleMoleculeFeatureBundle


class PredictionAgent:
    def __init__(self, model_path: Optional[str] = None):
        self.zero_shot = ZeroShotPredictionAgent()
        self.heuristic = self.zero_shot
        self.model = SklearnPredictor.load(model_path) if model_path and Path(model_path).exists() else None
        self.training_rules = []
        if self.model is not None:
            self.training_rules = [Rule(**item) for item in self.model.rules]

    def predict_single(
        self,
        bundle: SingleMoleculeFeatureBundle,
        task_type: str,
        dataset_name: str,
    ) -> PredictionResult:
        if self.model is not None:
            self._ensure_model_compatible(task_type=task_type, dataset_name=dataset_name)
            return self.model.predict_vector(bundle.feature_vector, task_type=task_type, dataset_name=dataset_name)
        return self.zero_shot.predict_single(bundle, task_type=task_type, dataset_name=dataset_name)

    def predict_ddi(self, pair: DDIPairFeatureBundle, dataset_name: str = "DDI") -> PredictionResult:
        if self.model is not None:
            self._ensure_model_compatible(task_type="ddi_prediction", dataset_name=dataset_name)
            return self.model.predict_vector(pair.feature_vector, task_type="ddi_prediction", dataset_name=dataset_name)
        return self.zero_shot.predict_ddi(pair, dataset_name=dataset_name)

    def _ensure_model_compatible(self, task_type: str, dataset_name: str) -> None:
        if self.model is None:
            return
        metadata = self.model.metadata or {}
        trained_task = metadata.get("task_type")
        trained_dataset = metadata.get("dataset_name")
        errors = []
        if trained_task and not _compatible_task_type(str(trained_task), str(task_type), str(trained_dataset or ""), str(dataset_name)):
            errors.append(f"trained task_type={trained_task}, requested task_type={task_type}")
        if trained_dataset and _normalize_dataset_name(str(trained_dataset)) != _normalize_dataset_name(str(dataset_name)):
            errors.append(f"trained dataset_name={trained_dataset}, requested dataset_name={dataset_name}")
        if errors:
            raise ValueError("Loaded model is incompatible with the requested prediction: " + "; ".join(errors))


def _normalize_dataset_name(dataset_name: str) -> str:
    aliases = {
        "bbbp": "bbb_martins",
        "cyp1a2 inhibition": "cyp1a2_veith",
        "cyp2c9 inhibition": "cyp2c9_veith",
        "cyp2c19 inhibition": "cyp2c19_veith",
        "cyp2d6 inhibition": "cyp2d6_veith",
        "cyp3a4 inhibition": "cyp3a4_veith",
        "esol": "solubility_aqsoldb",
        "freesolv": "hydrationfreeenergy_freesolv",
        "lipophilicity": "lipophilicity_astrazeneca",
    }
    key = str(dataset_name).strip().lower()
    return aliases.get(key, key)


def _compatible_task_type(trained_task: str, requested_task: str, trained_dataset: str, requested_dataset: str) -> bool:
    if trained_task == requested_task:
        return True
    dataset_matches = _normalize_dataset_name(trained_dataset) == _normalize_dataset_name(requested_dataset)
    molecular_single_tasks = {"property_prediction", "admet_prediction"}
    return dataset_matches and trained_task in molecular_single_tasks and requested_task in molecular_single_tasks
