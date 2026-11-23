from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


def _clean_json(value: Any) -> Any:
    """Convert common scientific Python objects into JSON-friendly values."""
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean_json(v) for v in value]
    return value


@dataclass
class MoleculeRecord:
    original_input: str
    smiles: Optional[str] = None
    canonical_smiles: Optional[str] = None
    name: Optional[str] = None
    source: str = "user"
    is_valid: bool = False
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return _clean_json(asdict(self))


@dataclass
class Rule:
    rule_id: str
    task_type: str
    source: str
    text: str
    feature_name: str
    code: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    is_executable: bool = True
    error: Optional[str] = None

    def signature(self) -> str:
        return f"{self.feature_name}:{self.metadata.get('operation')}:{self.metadata.get('descriptor')}:{self.metadata.get('operator')}:{self.metadata.get('threshold')}:{self.metadata.get('smarts')}"

    def to_dict(self) -> Dict[str, Any]:
        return _clean_json(asdict(self))


@dataclass
class RuleFeatureVector:
    values: Dict[str, float] = field(default_factory=dict)
    executed_rules: List[str] = field(default_factory=list)
    failed_rules: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return _clean_json(asdict(self))


@dataclass
class ImageFeatureBundle:
    image_path: str
    embedding: List[float]
    encoder_name: str
    checkpoint_path: Optional[str] = None
    grad_cam_path: Optional[str] = None
    grad_cam: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return _clean_json(asdict(self))


@dataclass
class SmilesFeatureBundle:
    smiles: str
    embedding: List[float]
    encoder_name: str
    model_name: Optional[str] = None
    pooling: str = "cls"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return _clean_json(asdict(self))


@dataclass
class VisualSemanticBundle:
    functional_groups: List[str] = field(default_factory=list)
    ring_systems: List[str] = field(default_factory=list)
    hbond_features: Dict[str, Any] = field(default_factory=dict)
    hydrophobic_regions: List[str] = field(default_factory=list)
    ionizable_centers: List[str] = field(default_factory=list)
    admet_alerts: List[str] = field(default_factory=list)
    task_relevant_visual_cues: List[str] = field(default_factory=list)
    summary: str = ""
    embedding: List[float] = field(default_factory=list)
    provider: str = "fallback_rdkit"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_text(self) -> str:
        parts = [
            f"functional_groups={','.join(self.functional_groups)}",
            f"ring_systems={','.join(self.ring_systems)}",
            f"hbond_features={self.hbond_features}",
            f"hydrophobic_regions={','.join(self.hydrophobic_regions)}",
            f"ionizable_centers={','.join(self.ionizable_centers)}",
            f"admet_alerts={','.join(self.admet_alerts)}",
            f"task_relevant_visual_cues={','.join(self.task_relevant_visual_cues)}",
            f"summary={self.summary}",
        ]
        return "\n".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        return _clean_json(asdict(self))


@dataclass
class FeatureSchema:
    feature_set: str
    feature_names: List[str]
    schema_name: str = "agentm_molecular_features"
    blocks: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return _clean_json(asdict(self))


@dataclass
class SingleMoleculeFeatureBundle:
    molecule: MoleculeRecord
    rdkit_descriptors: Dict[str, float]
    morgan_fingerprint: List[int]
    maccs_fingerprint: List[int]
    llm_rule_features: RuleFeatureVector
    smiles_embedding: List[float]
    imagemol_embedding: List[float]
    visual_semantic_embedding: List[float]
    visual_semantic_text: str
    feature_vector: List[float]
    feature_schema: FeatureSchema
    smiles: Optional[SmilesFeatureBundle] = None
    image: Optional[ImageFeatureBundle] = None
    visual_semantics: Optional[VisualSemanticBundle] = None

    def to_dict(self) -> Dict[str, Any]:
        return _clean_json(asdict(self))


@dataclass
class DDIPairFeatureBundle:
    drug_a_features: SingleMoleculeFeatureBundle
    drug_b_features: SingleMoleculeFeatureBundle
    concat_features: List[float]
    absolute_difference: List[float]
    elementwise_product: List[float]
    fingerprint_similarity: Dict[str, float]
    admet_risk_features: Dict[str, float]
    cyp_related_features: Dict[str, float]
    toxicity_overlap_features: Dict[str, float]
    llm_ddi_mechanism_features: Dict[str, float]
    feature_vector: List[float]
    feature_schema: FeatureSchema

    def to_dict(self) -> Dict[str, Any]:
        return _clean_json(asdict(self))


@dataclass
class PredictionResult:
    task_type: str
    dataset_name: str
    label: Optional[Any]
    score: float
    probability: Optional[float] = None
    model_type: str = "zero_shot_rule_semantic_fusion"
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return _clean_json(asdict(self))


@dataclass
class AgentReport:
    request: str
    task_type: str
    molecules: List[MoleculeRecord]
    predictions: List[PredictionResult]
    evidence: List[Dict[str, Any]]
    explanation: str
    artifacts: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    plan: Dict[str, Any] = field(default_factory=dict)
    trace: List[Dict[str, Any]] = field(default_factory=list)
    reasoning: Dict[str, Any] = field(default_factory=dict)
    narrative_report: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return _clean_json(asdict(self))
