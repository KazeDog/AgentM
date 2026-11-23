from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from agentm.schema import DDIPairFeatureBundle, PredictionResult, SingleMoleculeFeatureBundle
from agentm.tools.evidence import local_evidence_for_ddi


ZERO_SHOT_MODEL_TYPE = "zero_shot_rule_semantic_fusion"


@dataclass
class EvidenceContribution:
    source: str
    feature: str
    direction: str
    weight: float
    value: Any
    rationale: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "feature": self.feature,
            "direction": self.direction,
            "weight": self.weight,
            "value": self.value,
            "rationale": self.rationale,
        }


class ZeroShotPredictionAgent:
    """General-purpose prediction from rules, descriptors, and visual semantics.

    This agent does not learn a task-specific decision boundary. It produces a
    non-calibrated score from auditable evidence contributions and makes that
    limitation explicit in PredictionResult.details.
    """

    def predict_single(
        self,
        bundle: SingleMoleculeFeatureBundle,
        task_type: str,
        dataset_name: str,
    ) -> PredictionResult:
        dataset = dataset_name or "unknown"
        if dataset in {"ESOL", "FreeSolv", "Lipophilicity"}:
            return self._predict_regression_like(bundle, task_type, dataset)
        if task_type == "admet_prediction":
            return self._predict_admet(bundle, dataset)
        return self._predict_property(bundle, task_type, dataset)

    def predict_ddi(self, pair: DDIPairFeatureBundle, dataset_name: str = "DDI") -> PredictionResult:
        contributions: List[EvidenceContribution] = []
        base = 0.2
        sim = pair.fingerprint_similarity
        admet = pair.admet_risk_features
        cyp = pair.cyp_related_features
        mechanism = pair.llm_ddi_mechanism_features

        self._add_if(contributions, "fingerprint_similarity", "morgan_tanimoto", "positive", 0.18 * sim.get("morgan_tanimoto", 0.0), sim.get("morgan_tanimoto"), "Higher molecular similarity can increase shared-target or shared-ADMET interaction plausibility.")
        self._add_if(contributions, "fingerprint_similarity", "maccs_tanimoto", "positive", 0.10 * sim.get("maccs_tanimoto", 0.0), sim.get("maccs_tanimoto"), "MACCS similarity contributes weak pair-level plausibility.")
        self._add_if(contributions, "pair_admet", "admet_alert_overlap_count", "positive", min(0.18, 0.08 * admet.get("admet_alert_overlap_count", 0.0)), admet.get("admet_alert_overlap_count"), "Overlapping ADMET alerts can indicate shared risk mechanisms.")
        self._add_if(contributions, "pair_cyp", "both_cyp_lipophilic", "positive", 0.18 if cyp.get("both_cyp_lipophilic") else 0.0, cyp.get("both_cyp_lipophilic"), "Both molecules are lipophilic enough to raise CYP-mediated interaction concern.")
        self._add_if(contributions, "pair_cyp", "aromatic_ring_total", "positive", 0.04 if cyp.get("aromatic_ring_total", 0.0) >= 3 else 0.0, cyp.get("aromatic_ring_total"), "Aromatic-rich pairs can be compatible with CYP binding.")
        self._add_if(contributions, "llm_mechanism_features", "llm_mechanism_cyp", "positive", 0.12 if mechanism.get("llm_mechanism_cyp") else 0.0, mechanism.get("llm_mechanism_cyp"), "Visual/semantic features mention CYP-like mechanism cues.")
        self._add_if(contributions, "llm_mechanism_features", "llm_mechanism_transporter", "positive", 0.08 if mechanism.get("llm_mechanism_transporter") else 0.0, mechanism.get("llm_mechanism_transporter"), "Transporter-like semantic cues raise possible DDI concern.")
        contributions.extend(self._pair_representation_context(pair))

        name_a = pair.drug_a_features.molecule.name
        name_b = pair.drug_b_features.molecule.name
        local_hints = local_evidence_for_ddi(name_a, name_b)
        if local_hints:
            severity = str(local_hints[0].get("severity", "")).lower()
            hint_weight = 0.35 if severity == "high" else 0.22 if severity == "moderate" else 0.12
            contributions.append(
                EvidenceContribution(
                    source="local_ddi_hint",
                    feature="known_pair_hint",
                    direction="positive",
                    weight=hint_weight,
                    value=local_hints[0],
                    rationale="A local curated DDI hint matched this drug pair.",
                )
            )

        score = self._score(base, contributions)
        label = int(score >= 0.5)
        return PredictionResult(
            task_type="ddi_prediction",
            dataset_name=dataset_name,
            label=label,
            score=score,
            probability=None,
            model_type=ZERO_SHOT_MODEL_TYPE,
            details=self._details(
                score=score,
                positive_meaning="DDI risk/interaction",
                negative_meaning="no DDI risk/interaction",
                contributions=contributions,
                extra={
                    "fingerprint_similarity": sim,
                    "admet_risk_features": admet,
                    "cyp_related_features": cyp,
                    "llm_ddi_mechanism_features": mechanism,
                    "local_ddi_hints": local_hints,
                },
            ),
        )

    def _predict_property(self, bundle: SingleMoleculeFeatureBundle, task_type: str, dataset_name: str) -> PredictionResult:
        if dataset_name == "BBBP":
            base = 0.42
            positive = "BBB permeable"
            negative = "BBB non-permeable"
            contributions = self._bbbp_contributions(bundle)
        elif dataset_name == "BACE":
            base = 0.35
            positive = "BACE-active-like"
            negative = "BACE-inactive-like"
            contributions = self._bace_contributions(bundle)
        elif dataset_name == "HIV":
            base = 0.30
            positive = "HIV-active-like"
            negative = "HIV-inactive-like"
            contributions = self._hiv_contributions(bundle)
        else:
            base = 0.45
            positive = f"{dataset_name} positive class"
            negative = f"{dataset_name} negative class"
            contributions = self._generic_property_contributions(bundle)

        score = self._score(base, contributions)
        label = int(score >= 0.5)
        return PredictionResult(
            task_type=task_type,
            dataset_name=dataset_name,
            label=label,
            score=score,
            probability=None,
            model_type=ZERO_SHOT_MODEL_TYPE,
            details=self._details(score=score, positive_meaning=positive, negative_meaning=negative, contributions=contributions, extra=self._single_extra(bundle)),
        )

    def _predict_admet(self, bundle: SingleMoleculeFeatureBundle, dataset_name: str) -> PredictionResult:
        base = 0.18
        contributions = self._admet_contributions(bundle, dataset_name)
        score = self._score(base, contributions)
        label = int(score >= 0.5)
        positive = f"{dataset_name} risk/positive"
        negative = f"low {dataset_name} risk/negative"
        return PredictionResult(
            task_type="admet_prediction",
            dataset_name=dataset_name,
            label=label,
            score=score,
            probability=None,
            model_type=ZERO_SHOT_MODEL_TYPE,
            details=self._details(score=score, positive_meaning=positive, negative_meaning=negative, contributions=contributions, extra=self._single_extra(bundle)),
        )

    def _predict_regression_like(self, bundle: SingleMoleculeFeatureBundle, task_type: str, dataset_name: str) -> PredictionResult:
        d = bundle.rdkit_descriptors
        if dataset_name == "ESOL":
            value = 0.16 - 0.63 * d.get("MolLogP", 0.0) - 0.0062 * d.get("MolWt", 0.0) + 0.066 * d.get("NumRotatableBonds", 0.0)
            units = "approximate log solubility"
        elif dataset_name == "Lipophilicity":
            value = d.get("MolLogP", 0.0)
            units = "approximate logD/logP-like lipophilicity"
        else:
            value = -0.02 * d.get("TPSA", 0.0) - 0.3 * d.get("MolLogP", 0.0)
            units = "rough hydration free-energy tendency"
        contributions = self._generic_property_contributions(bundle)
        details = self._details(
            score=float(value),
            positive_meaning=None,
            negative_meaning=None,
            contributions=contributions,
            extra={**self._single_extra(bundle), "units": units, "zero_shot_regression_estimate": float(value)},
        )
        return PredictionResult(
            task_type=task_type,
            dataset_name=dataset_name,
            label=float(value),
            score=float(value),
            probability=None,
            model_type=ZERO_SHOT_MODEL_TYPE,
            details=details,
        )

    def _bbbp_contributions(self, bundle: SingleMoleculeFeatureBundle) -> List[EvidenceContribution]:
        d = bundle.rdkit_descriptors
        visual = bundle.visual_semantics
        groups = set(visual.functional_groups if visual else [])
        centers = set(visual.ionizable_centers if visual else [])
        contributions: List[EvidenceContribution] = []
        self._threshold(contributions, "rdkit_descriptor", "TPSA", d.get("TPSA"), positive_if=lambda v: v < 90, pos_weight=0.16, neg_weight=-0.12, rationale_pos="Low polar surface area supports passive BBB permeability.", rationale_neg="High polar surface area reduces passive BBB permeability.")
        self._threshold(contributions, "rdkit_descriptor", "MolWt", d.get("MolWt"), positive_if=lambda v: v < 450, pos_weight=0.10, neg_weight=-0.10, rationale_pos="Moderate molecular weight supports permeability.", rationale_neg="High molecular weight can reduce permeability.")
        self._threshold(contributions, "rdkit_descriptor", "NumHDonors", d.get("NumHDonors"), positive_if=lambda v: v < 3, pos_weight=0.08, neg_weight=-0.08, rationale_pos="Few hydrogen bond donors support BBB permeability.", rationale_neg="Many hydrogen bond donors reduce membrane permeability.")
        self._threshold(contributions, "rdkit_descriptor", "MolLogP", d.get("MolLogP"), positive_if=lambda v: 1.0 <= v <= 5.0, pos_weight=0.08, neg_weight=-0.06, rationale_pos="Moderate lipophilicity supports membrane partitioning.", rationale_neg="Very low or extreme lipophilicity weakens the BBB heuristic.")
        if "carboxylic_acid" in groups or "acidic_center" in centers:
            contributions.append(EvidenceContribution("visual_semantics", "acidic_or_carboxylate_center", "negative", -0.18, True, "Ionizable acidic groups often reduce CNS penetration."))
        contributions.extend(self._matched_rule_contributions(bundle, positive_weight=0.025, negative_tokens=["risk", "above_500", "above_5", "above_140", "high_tpsa", "high_mw", "nitro"]))
        contributions.extend(self._single_representation_context(bundle))
        return contributions

    def _bace_contributions(self, bundle: SingleMoleculeFeatureBundle) -> List[EvidenceContribution]:
        d = bundle.rdkit_descriptors
        contributions: List[EvidenceContribution] = []
        self._threshold(contributions, "rdkit_descriptor", "MolLogP", d.get("MolLogP"), positive_if=lambda v: v > 2, pos_weight=0.10, neg_weight=-0.04, rationale_pos="Hydrophobicity can support BACE pocket binding.", rationale_neg="Low hydrophobicity weakens BACE pocket binding heuristics.")
        self._threshold(contributions, "rdkit_descriptor", "MolWt", d.get("MolWt"), positive_if=lambda v: v < 600, pos_weight=0.06, neg_weight=-0.08, rationale_pos="Molecular weight is within a broad drug-like BACE range.", rationale_neg="Very high molecular weight is less drug-like.")
        contributions.extend(self._matched_rule_contributions(bundle, positive_weight=0.02))
        contributions.extend(self._single_representation_context(bundle))
        return contributions

    def _hiv_contributions(self, bundle: SingleMoleculeFeatureBundle) -> List[EvidenceContribution]:
        d = bundle.rdkit_descriptors
        visual = bundle.visual_semantics
        groups = set(visual.functional_groups if visual else [])
        contributions: List[EvidenceContribution] = []
        if "halogen" in groups:
            contributions.append(EvidenceContribution("visual_semantics", "halogen", "positive", 0.08, True, "Halogens can modulate antiviral pharmacophore potency and permeability."))
        self._threshold(contributions, "rdkit_descriptor", "NumAromaticRings", d.get("NumAromaticRings"), positive_if=lambda v: v > 1, pos_weight=0.08, neg_weight=0.0, rationale_pos="Multiple aromatic rings can indicate hydrophobic pharmacophore content.", rationale_neg="")
        contributions.extend(self._matched_rule_contributions(bundle, positive_weight=0.02))
        contributions.extend(self._single_representation_context(bundle))
        return contributions

    def _generic_property_contributions(self, bundle: SingleMoleculeFeatureBundle) -> List[EvidenceContribution]:
        d = bundle.rdkit_descriptors
        contributions: List[EvidenceContribution] = []
        self._threshold(contributions, "rdkit_descriptor", "QED", d.get("QED"), positive_if=lambda v: v >= 0.5, pos_weight=0.08, neg_weight=-0.04, rationale_pos="Drug-like QED supports a generic property-positive tendency.", rationale_neg="Low QED weakens a generic property-positive tendency.")
        contributions.extend(self._matched_rule_contributions(bundle, positive_weight=0.02))
        contributions.extend(self._single_representation_context(bundle))
        return contributions

    def _admet_contributions(self, bundle: SingleMoleculeFeatureBundle, dataset_name: str) -> List[EvidenceContribution]:
        d = bundle.rdkit_descriptors
        visual = bundle.visual_semantics
        groups = set(visual.functional_groups if visual else [])
        alerts = set(visual.admet_alerts if visual else [])
        contributions: List[EvidenceContribution] = []
        self._threshold(contributions, "rdkit_descriptor", "MolWt", d.get("MolWt"), positive_if=lambda v: v > 500, pos_weight=0.16, neg_weight=-0.02, rationale_pos="High molecular weight is a general ADMET risk signal.", rationale_neg="Molecular weight is below the high-risk threshold.")
        self._threshold(contributions, "rdkit_descriptor", "MolLogP", d.get("MolLogP"), positive_if=lambda v: v > 5, pos_weight=0.16, neg_weight=-0.02, rationale_pos="High lipophilicity is a general ADMET risk signal.", rationale_neg="Lipophilicity is below the high-risk threshold.")
        self._threshold(contributions, "rdkit_descriptor", "TPSA", d.get("TPSA"), positive_if=lambda v: v > 140, pos_weight=0.12, neg_weight=-0.01, rationale_pos="High TPSA may cause permeability or exposure issues.", rationale_neg="TPSA is below the high-risk threshold.")
        if "nitro" in groups or "nitro_toxicity_alert" in alerts:
            contributions.append(EvidenceContribution("visual_semantics", "nitro_toxicity_alert", "positive", 0.20, True, "Nitro groups are common structural toxicity alerts."))
        if "halogen" in groups and dataset_name in {"DILI", "hERG"}:
            contributions.append(EvidenceContribution("visual_semantics", "halogen", "positive", 0.06, True, "Halogens can contribute to lipophilicity and some ADMET liabilities."))
        if dataset_name == "hERG":
            if d.get("MolLogP", 0.0) > 3:
                contributions.append(EvidenceContribution("rdkit_descriptor", "MolLogP_hERG", "positive", 0.12, d.get("MolLogP"), "Higher lipophilicity is a common hERG risk correlate."))
            if "amine" in groups:
                contributions.append(EvidenceContribution("visual_semantics", "basic_amine", "positive", 0.12, True, "Basic amines can be hERG risk motifs."))
        if dataset_name.startswith("CYP") or dataset_name in {"DILI"}:
            if d.get("MolLogP", 0.0) > 4 or d.get("NumAromaticRings", 0.0) >= 2:
                contributions.append(EvidenceContribution("rdkit_descriptor", "cyp_like_lipophilic_aromatic", "positive", 0.10, {"MolLogP": d.get("MolLogP"), "NumAromaticRings": d.get("NumAromaticRings")}, "Lipophilic/aromatic molecules can raise CYP-related ADMET concern."))
        contributions.extend(self._matched_rule_contributions(bundle, positive_weight=0.025, risk_task=True))
        contributions.extend(self._single_representation_context(bundle))
        return contributions

    def _matched_rule_contributions(
        self,
        bundle: SingleMoleculeFeatureBundle,
        positive_weight: float,
        *,
        risk_task: bool = False,
        negative_tokens: Optional[List[str]] = None,
    ) -> List[EvidenceContribution]:
        contributions: List[EvidenceContribution] = []
        negative_tokens = negative_tokens or []
        for name, value in sorted(bundle.llm_rule_features.values.items()):
            if not value:
                continue
            lowered = name.lower()
            negative = (not risk_task) and any(token in lowered for token in negative_tokens)
            direction = "negative" if negative else "positive"
            weight = -positive_weight if negative else positive_weight
            rationale = "A computable LLM4SD-style rule matched this molecule."
            if risk_task:
                rationale += " It is interpreted as a task risk cue."
            elif negative:
                rationale += " It is interpreted as an unfavorable cue for the positive class."
            contributions.append(EvidenceContribution("llm_rule_feature", name, direction, weight, value, rationale))
        return contributions[:8]

    def _single_representation_context(self, bundle: SingleMoleculeFeatureBundle) -> List[EvidenceContribution]:
        contributions: List[EvidenceContribution] = []
        if bundle.image is not None and bundle.imagemol_embedding:
            contributions.append(
                EvidenceContribution(
                    source="imagemol",
                    feature="embedding_available",
                    direction="neutral",
                    weight=0.0,
                    value={
                        "encoder_name": bundle.image.encoder_name,
                        "checkpoint_path": bundle.image.checkpoint_path,
                        "embedding_dim": len(bundle.imagemol_embedding),
                    },
                    rationale="ImageMol embedding is available as a generic representation, but no supervised zero-shot boundary is calibrated from it.",
                )
            )
        if bundle.smiles is not None and bundle.smiles_embedding:
            contributions.append(
                EvidenceContribution(
                    source="smiles_encoder",
                    feature="embedding_available",
                    direction="neutral",
                    weight=0.0,
                    value={
                        "encoder_name": bundle.smiles.encoder_name,
                        "model_name": bundle.smiles.model_name,
                        "pooling": bundle.smiles.pooling,
                        "embedding_dim": len(bundle.smiles_embedding),
                    },
                    rationale="SMILES foundation-model embedding is available as a generic representation, but no supervised zero-shot boundary is calibrated from it.",
                )
            )
        if bundle.visual_semantics is not None:
            contributions.append(
                EvidenceContribution(
                    source="visual_semantics",
                    feature="semantic_summary_available",
                    direction="neutral",
                    weight=0.0,
                    value={
                        "provider": bundle.visual_semantics.provider,
                        "summary": bundle.visual_semantics.summary,
                    },
                    rationale="Visual semantic interpretation supplies structural context for the rule-based score.",
                )
            )
        return contributions

    def _pair_representation_context(self, pair: DDIPairFeatureBundle) -> List[EvidenceContribution]:
        image_dims = [
            len(pair.drug_a_features.imagemol_embedding),
            len(pair.drug_b_features.imagemol_embedding),
        ]
        smiles_dims = [
            len(pair.drug_a_features.smiles_embedding),
            len(pair.drug_b_features.smiles_embedding),
        ]
        contributions: List[EvidenceContribution] = []
        if any(image_dims):
            contributions.append(
                EvidenceContribution(
                    source="imagemol_pair",
                    feature="pair_embeddings_available",
                    direction="neutral",
                    weight=0.0,
                    value={
                        "drug_a_embedding_dim": image_dims[0],
                        "drug_b_embedding_dim": image_dims[1],
                        "drug_a_encoder": pair.drug_a_features.image.encoder_name if pair.drug_a_features.image else None,
                        "drug_b_encoder": pair.drug_b_features.image.encoder_name if pair.drug_b_features.image else None,
                    },
                    rationale="ImageMol pair embeddings are retained as representation context; zero-shot DDI scoring uses interpretable pair cues unless a trained model is supplied.",
                )
            )
        if any(smiles_dims):
            contributions.append(
                EvidenceContribution(
                    source="smiles_encoder_pair",
                    feature="pair_embeddings_available",
                    direction="neutral",
                    weight=0.0,
                    value={
                        "drug_a_embedding_dim": smiles_dims[0],
                        "drug_b_embedding_dim": smiles_dims[1],
                        "drug_a_encoder": pair.drug_a_features.smiles.encoder_name if pair.drug_a_features.smiles else None,
                        "drug_b_encoder": pair.drug_b_features.smiles.encoder_name if pair.drug_b_features.smiles else None,
                    },
                    rationale="SMILES pair embeddings are retained as representation context; zero-shot DDI scoring uses interpretable pair cues unless a trained model is supplied.",
                )
            )
        return contributions

    def _threshold(
        self,
        contributions: List[EvidenceContribution],
        source: str,
        feature: str,
        value: Optional[float],
        *,
        positive_if,
        pos_weight: float,
        neg_weight: float,
        rationale_pos: str,
        rationale_neg: str,
    ) -> None:
        if value is None:
            return
        if positive_if(float(value)):
            if pos_weight:
                contributions.append(EvidenceContribution(source, feature, "positive", pos_weight, float(value), rationale_pos))
        elif neg_weight:
            contributions.append(EvidenceContribution(source, feature, "negative", neg_weight, float(value), rationale_neg))

    def _add_if(
        self,
        contributions: List[EvidenceContribution],
        source: str,
        feature: str,
        direction: str,
        weight: float,
        value: Any,
        rationale: str,
    ) -> None:
        if abs(weight) > 1e-12:
            contributions.append(EvidenceContribution(source, feature, direction, float(weight), value, rationale))

    def _score(self, base: float, contributions: List[EvidenceContribution]) -> float:
        return float(np.clip(base + sum(item.weight for item in contributions), 0.01, 0.99))

    def _confidence_level(self, score: float, contributions: List[EvidenceContribution]) -> str:
        margin = abs(score - 0.5)
        total_weight = sum(abs(item.weight) for item in contributions)
        if margin >= 0.25 and total_weight >= 0.25:
            return "high"
        if margin >= 0.12 and total_weight >= 0.15:
            return "medium"
        return "low"

    def _details(
        self,
        *,
        score: float,
        positive_meaning: Optional[str],
        negative_meaning: Optional[str],
        contributions: List[EvidenceContribution],
        extra: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {
            "mode": "zero_shot_agent",
            "zero_shot_score": float(score),
            "calibrated_probability": None,
            "calibration_status": "not_calibrated",
            "calibration_method": "none",
            "uncertainty_method": "zero_shot_margin_and_evidence_contributions",
            "confidence_level": self._confidence_level(float(score), contributions),
            "decision_threshold": 0.5,
            "positive_label": 1 if positive_meaning else None,
            "positive_label_meaning": positive_meaning,
            "negative_label": 0 if negative_meaning else None,
            "negative_label_meaning": negative_meaning,
            "warning": "No task-specific supervised model was used; score is not a calibrated probability.",
            "evidence_contributions": [item.to_dict() for item in contributions],
            **extra,
        }

    def _single_extra(self, bundle: SingleMoleculeFeatureBundle) -> Dict[str, Any]:
        visual = bundle.visual_semantics
        image = bundle.image
        return {
            "rdkit_descriptors": bundle.rdkit_descriptors,
            "matched_rules": {key: value for key, value in bundle.llm_rule_features.values.items() if value},
            "smiles_encoder": {
                "encoder_name": bundle.smiles.encoder_name if bundle.smiles else None,
                "model_name": bundle.smiles.model_name if bundle.smiles else None,
                "pooling": bundle.smiles.pooling if bundle.smiles else None,
                "embedding_dim": len(bundle.smiles_embedding),
                "used_as_generic_representation": bool(bundle.smiles_embedding),
                "direct_zero_shot_boundary": "not available without supervised calibration" if bundle.smiles_embedding else None,
                "metadata": bundle.smiles.metadata if bundle.smiles else {},
            },
            "visual_semantics": visual.to_dict() if visual else None,
            "imagemol": {
                "encoder_name": image.encoder_name if image else None,
                "checkpoint_path": image.checkpoint_path if image else None,
                "embedding_dim": len(bundle.imagemol_embedding),
                "used_as_generic_representation": True,
                "direct_zero_shot_boundary": "not available without supervised calibration",
            },
        }
