from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

from agentm.utils.io import write_text


class CaseStudyReportBuilder:
    """Build a compact paper-style case report from an agent JSON report."""

    def build(self, report: Mapping[str, Any]) -> str:
        prediction = _first(report.get("predictions", [])) or {}
        artifacts = report.get("artifacts", {}) if isinstance(report.get("artifacts"), Mapping) else {}
        reasoning = report.get("reasoning", {}) if isinstance(report.get("reasoning"), Mapping) else {}
        lines = [
            "# agentM Case Study",
            "",
            "## Task",
            f"- Request: {report.get('request')}",
            f"- Task type: `{report.get('task_type')}`",
            f"- Dataset/task name: `{prediction.get('dataset_name')}`",
            f"- Model: `{prediction.get('model_type')}`",
            f"- Prediction: `{prediction.get('label')}`",
            f"- Score: `{_fmt(prediction.get('score'))}`",
            f"- Probability: `{_fmt(prediction.get('probability'))}`",
        ]
        meaning = _prediction_meaning(prediction)
        if meaning:
            lines.append(f"- Meaning: {meaning}")

        lines.extend(["", "## Molecules"])
        for idx, molecule in enumerate(report.get("molecules", []) or [], start=1):
            lines.extend(
                [
                    f"### Molecule {idx}",
                    f"- Input: `{molecule.get('original_input')}`",
                    f"- Name: `{molecule.get('name')}`",
                    f"- Source: `{molecule.get('source')}`",
                    f"- Canonical SMILES: `{molecule.get('canonical_smiles')}`",
                ]
            )

        lines.extend(["", "## Multimodal Feature Status"])
        for item in self._feature_status(artifacts):
            lines.append(f"- {item}")

        contributions = _top_contributions(prediction)
        lines.extend(["", "## Top Prediction Contributions"])
        if contributions:
            lines.append("| Source | Feature | Direction | Weight | Rationale |")
            lines.append("|---|---|---:|---:|---|")
            for item in contributions:
                lines.append(
                    "| {source} | {feature} | {direction} | {weight} | {rationale} |".format(
                        source=_pipe(item.get("source")),
                        feature=_pipe(item.get("feature")),
                        direction=_pipe(item.get("direction")),
                        weight=_fmt(item.get("weight")),
                        rationale=_pipe(item.get("rationale")),
                    )
                )
        else:
            lines.append("- No zero-shot contribution table was attached by the predictor.")

        scoring = artifacts.get("evidence_grounded_scoring")
        lines.extend(["", "## Evidence-Grounded Scoring"])
        if isinstance(scoring, Mapping):
            lines.extend(
                [
                    f"- Evidence-grounded score: `{_fmt(scoring.get('evidence_grounded_score'))}`",
                    f"- Support score: `{_fmt(scoring.get('support_score'))}`",
                    f"- Contradiction score: `{_fmt(scoring.get('contradiction_score'))}`",
                    f"- Source reliability: `{_fmt(scoring.get('source_reliability_score'))}`",
                    f"- Coverage level: `{scoring.get('coverage_level')}`",
                ]
            )
        else:
            lines.append("- Not available for this report.")

        lines.extend(["", "## Evidence Table"])
        evidence = list(report.get("evidence", []) or [])
        if evidence:
            lines.append("| Source | Direction | Claim | Value |")
            lines.append("|---|---|---|---|")
            for item in evidence[:20]:
                lines.append(
                    "| {source} | {direction} | {claim} | {value} |".format(
                        source=_pipe(item.get("source")),
                        direction=_pipe(item.get("evidence_direction", "n/a")),
                        claim=_pipe(item.get("claim") or item.get("mechanism") or item.get("interaction_type") or ""),
                        value=_pipe(_short_value(item.get("value"))),
                    )
                )
            if len(evidence) > 20:
                lines.append(f"| ... | ... | {len(evidence) - 20} more evidence items in JSON report | |")
        else:
            lines.append("- No evidence items were retrieved.")

        lines.extend(["", "## RAG Evidence Table"])
        rag_context = artifacts.get("rag_context") or artifacts.get("retrieved_context") or []
        rag_evidence = artifacts.get("rag_evidence") if isinstance(artifacts.get("rag_evidence"), Mapping) else {}
        rag_query = rag_evidence.get("query") if isinstance(rag_evidence, Mapping) else None
        if rag_query:
            lines.append(f"- Query: `{_pipe(rag_query)}`")
        if isinstance(rag_context, list) and rag_context:
            lines.append("| Rank | Score | Source | Chunk | Snippet |")
            lines.append("|---:|---:|---|---|---|")
            for item in rag_context[:10]:
                if not isinstance(item, Mapping):
                    continue
                lines.append(
                    "| {rank} | {score} | {source} | {chunk} | {snippet} |".format(
                        rank=item.get("rank") or "n/a",
                        score=_fmt(item.get("score")),
                        source=_pipe(item.get("source_path")),
                        chunk=_pipe(item.get("chunk_id")),
                        snippet=_pipe(str(item.get("snippet") or "")[:220]),
                    )
                )
        else:
            lines.append("- No RAG context was retrieved.")
        if artifacts.get("rag_evidence_markdown_path"):
            lines.append(f"- Full RAG evidence: `{artifacts['rag_evidence_markdown_path']}`")

        robustness = artifacts.get("failure_mode_assessment")
        lines.extend(["", "## Robustness Notes"])
        if isinstance(robustness, Mapping):
            lines.extend(
                [
                    f"- Status: `{robustness.get('robustness_status')}`",
                    f"- Warnings: `{robustness.get('n_warnings')}`",
                    f"- Fallbacks: `{robustness.get('n_fallbacks')}`",
                    f"- Low confidence: `{robustness.get('low_confidence_prediction')}`",
                ]
            )
            for issue in robustness.get("issues", [])[:8]:
                lines.append(f"- Issue: {issue}")
        else:
            lines.append("- No robustness assessment was attached.")

        lines.extend(["", "## Agent Reasoning"])
        lines.append(str(reasoning.get("summary") or report.get("explanation") or "n/a"))

        lines.extend(["", "## Reproducibility"])
        for key in ["reproducibility_log_path", "agent_plan_path", "agent_trace_path", "markdown_report_path"]:
            if artifacts.get(key):
                lines.append(f"- {key}: `{artifacts[key]}`")
        return "\n".join(lines).rstrip() + "\n"

    def _feature_status(self, artifacts: Mapping[str, Any]) -> List[str]:
        schema = artifacts.get("feature_schema") or {}
        blocks = schema.get("blocks", {}) if isinstance(schema, Mapping) else {}
        items = [f"Feature set: `{schema.get('feature_set', 'unknown')}`"]
        items.append(f"Feature dimension: `{len(schema.get('feature_names', [])) if isinstance(schema, Mapping) else 'unknown'}`")
        for block_name in [
            "rdkit",
            "rdkit_descriptors",
            "fingerprint",
            "morgan_fingerprint",
            "maccs_fingerprint",
            "llm_rules",
            "imagemol",
            "visual_semantic",
            "smiles",
        ]:
            block = blocks.get(block_name) if isinstance(blocks, Mapping) else None
            if block is not None:
                items.append(f"{block_name}: enabled, dim={block.get('dim')}")
        if artifacts.get("image_path"):
            items.append(f"RDKit image: `{artifacts.get('image_path')}`")
        if artifacts.get("grad_cam_path"):
            items.append(f"Image explanation: `{artifacts.get('grad_cam_path')}`")
        smiles = artifacts.get("smiles_features") or artifacts.get("drug_a_smiles_features")
        if isinstance(smiles, Mapping) and smiles.get("encoder_name"):
            items.append(f"SMILES encoder: `{smiles.get('encoder_name')}`")
        visual = artifacts.get("visual_semantics")
        if isinstance(visual, Mapping) and visual.get("provider"):
            items.append(f"Visual semantic provider: `{visual.get('provider')}`")
        return items


def write_case_study_report(report: Mapping[str, Any], path: str | Path) -> Path:
    return write_text(path, CaseStudyReportBuilder().build(report))


def _first(values: Iterable[Any]) -> Optional[Any]:
    for value in values:
        return value
    return None


def _prediction_meaning(prediction: Mapping[str, Any]) -> Optional[str]:
    details = prediction.get("details", {}) if isinstance(prediction.get("details"), Mapping) else {}
    if prediction.get("label") == details.get("positive_label"):
        return details.get("positive_label_meaning")
    if prediction.get("label") == details.get("negative_label"):
        return details.get("negative_label_meaning")
    if prediction.get("task_type") == "ddi_prediction":
        return "DDI risk/interaction" if prediction.get("label") == 1 else "no DDI risk/interaction"
    return None


def _top_contributions(prediction: Mapping[str, Any], limit: int = 10) -> List[Dict[str, Any]]:
    details = prediction.get("details", {}) if isinstance(prediction.get("details"), Mapping) else {}
    items = details.get("evidence_contributions", [])
    if not isinstance(items, list):
        return []
    return sorted(
        [item for item in items if isinstance(item, Mapping)],
        key=lambda item: abs(float(item.get("weight") or 0.0)),
        reverse=True,
    )[:limit]


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.4f}"
    except Exception:
        return str(value)


def _pipe(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ").strip()


def _short_value(value: Any, limit: int = 160) -> str:
    text = str(value)
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."
