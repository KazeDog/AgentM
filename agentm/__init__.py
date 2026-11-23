"""agentM: small-molecule multimodal agent toolkit."""

from agentm.schema import (
    AgentReport,
    DDIPairFeatureBundle,
    ImageFeatureBundle,
    MoleculeRecord,
    PredictionResult,
    Rule,
    RuleFeatureVector,
    SingleMoleculeFeatureBundle,
    SmilesFeatureBundle,
    VisualSemanticBundle,
)

__all__ = [
    "AgentMMultiAgentController",
    "AgentReport",
    "DDIPairFeatureBundle",
    "ImageFeatureBundle",
    "MoleculeRecord",
    "PredictionResult",
    "Rule",
    "RuleFeatureVector",
    "SingleMoleculeFeatureBundle",
    "SmilesFeatureBundle",
    "VisualSemanticBundle",
]


def __getattr__(name: str):
    """Expose the guarded controller without eagerly loading optional models."""
    if name == "AgentMMultiAgentController":
        from agentm.runtime.controller import AgentMMultiAgentController
        return AgentMMultiAgentController
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
