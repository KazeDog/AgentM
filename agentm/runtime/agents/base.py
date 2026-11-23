from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass
class SpecializedAgent:
    name: str
    allowed_tools: tuple[str, ...]
    role: str

    def allowlist(self) -> list[str]:
        return list(self.allowed_tools)


MOLECULE_AGENT = SpecializedAgent(
    name="MoleculeAgent",
    allowed_tools=("route_task", "resolve_molecule", "resolve_ddi_pair"),
    role="Parse and resolve molecules or DDI pairs.",
)
RETRIEVE_AGENT = SpecializedAgent(
    name="RetrieveAgent",
    allowed_tools=(
        "retrieve_local_rag",
        "retrieve_elasticsearch_hybrid",
        "retrieve_external_evidence",
        "expand_evidence_query",
        "retrieve_mit_template_evidence",
        "retrieve_orderly_condition_evidence",
    ),
    role="Retrieve molecule-level, reaction-specific, ES, local, and optional external evidence with citation traces.",
)
GAUGE_AGENT = SpecializedAgent(
    name="GaugeAgent",
    allowed_tools=("discover_rules", "build_features", "predict_single", "predict_ddi", "assess_confidence", "run_zero_shot_fallback"),
    role="Build features, predict ADMET/DDI outcomes, assess confidence, and own prediction fields.",
)
REASONING_AGENT = SpecializedAgent(
    name="ReasoningAgent",
    allowed_tools=("build_structured_explanation", "summarize_evidence_support"),
    role="Integrate prediction and evidence into grounded reasoning.",
)
VERIFIER_AGENT = SpecializedAgent(
    name="VerifierAgent",
    allowed_tools=("verify_prediction_grounding", "verify_citation_grounding", "verify_report_schema"),
    role="Verify prediction grounding, citation grounding, and report schema.",
)
REFLECTION_AGENT = SpecializedAgent(
    name="ReflectionAgent",
    allowed_tools=("reflect_on_report", "apply_reflection_feedback"),
    role="Use the current report and verifier output as the next planning input.",
)
REPORTER_AGENT = SpecializedAgent(
    name="ReporterAgent",
    allowed_tools=("build_runtime_report", "build_compatible_report", "write_artifacts"),
    role="Build reports and write artifacts.",
)


def specialized_agents() -> Iterable[SpecializedAgent]:
    return [MOLECULE_AGENT, RETRIEVE_AGENT, GAUGE_AGENT, REASONING_AGENT, VERIFIER_AGENT, REFLECTION_AGENT, REPORTER_AGENT]
