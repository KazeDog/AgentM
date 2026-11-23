from __future__ import annotations

from agentm.features.builder import FeatureBuilder
from agentm.schema import DDIPairFeatureBundle, SingleMoleculeFeatureBundle


class DDIReasoningAgent:
    def __init__(self, feature_builder: FeatureBuilder):
        self.feature_builder = feature_builder

    def build_pair(self, drug_a: SingleMoleculeFeatureBundle, drug_b: SingleMoleculeFeatureBundle) -> DDIPairFeatureBundle:
        return self.feature_builder.build_ddi_pair(drug_a, drug_b)

