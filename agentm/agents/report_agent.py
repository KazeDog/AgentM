from __future__ import annotations

from typing import Any, Dict, List

from agentm.schema import AgentReport, MoleculeRecord, PredictionResult


class ReportAgent:
    def build(
        self,
        request: str,
        task_type: str,
        molecules: List[MoleculeRecord],
        predictions: List[PredictionResult],
        evidence: List[Dict[str, Any]],
        artifacts: Dict[str, Any],
        warnings: List[str] | None = None,
        plan: Dict[str, Any] | None = None,
        trace: List[Dict[str, Any]] | None = None,
        reasoning: Dict[str, Any] | None = None,
        narrative_report: Dict[str, Any] | None = None,
    ) -> AgentReport:
        narrative_report = narrative_report or {}
        explanation = narrative_report.get("plain_language_summary") or self._explain(task_type, predictions, evidence)
        return AgentReport(
            request=request,
            task_type=task_type,
            molecules=molecules,
            predictions=predictions,
            evidence=evidence,
            explanation=explanation,
            artifacts=artifacts,
            warnings=warnings or [],
            plan=plan or {},
            trace=trace or [],
            reasoning=reasoning or {},
            narrative_report=narrative_report,
        )

    def _explain(self, task_type: str, predictions: List[PredictionResult], evidence: List[Dict[str, Any]]) -> str:
        if not predictions:
            return "No prediction was produced."
        pred = predictions[0]
        score_name = "zero-shot score" if pred.model_type == "zero_shot_rule_semantic_fusion" else "score"
        calibration_note = " This is not a calibrated probability because no task-specific supervised model was used." if pred.model_type == "zero_shot_rule_semantic_fusion" else ""
        if task_type == "ddi_prediction":
            level = "high" if (pred.probability or pred.score) >= 0.5 else "low"
            return f"DDI risk is estimated as {level} with {score_name} {pred.score:.3f}. Evidence combines fingerprint similarity, ADMET overlap, CYP-like features, and local interaction hints.{calibration_note}"
        return (
            f"{pred.dataset_name} prediction produced label/value {pred.label} with {score_name} {pred.score:.3f}. "
            f"The report combines RDKit descriptors, LLM-style rule features, ImageMol-style image embedding, and visual semantics.{calibration_note}"
        )

    def to_markdown(self, report: AgentReport) -> str:
        prediction = report.predictions[0] if report.predictions else None
        lines = [
            "# agentM Report",
            "",
            f"**Request:** {report.request}",
            f"**Task:** {report.task_type}",
        ]
        if prediction is not None:
            details = prediction.details or {}
            predicted_meaning = self._prediction_meaning(prediction)
            lines.extend(
                [
                    f"**Dataset:** {prediction.dataset_name}",
                    f"**Prediction:** {prediction.label}",
                    f"**Score:** {prediction.score:.4f}",
                    f"**Probability:** {prediction.probability:.4f}" if prediction.probability is not None else "**Probability:** n/a",
                    f"**Model:** {prediction.model_type}",
                    f"**Meaning:** {predicted_meaning or 'n/a'}",
                ]
            )
            if "decision_threshold" in details:
                lines.append(f"**Decision Threshold:** {details.get('decision_threshold')}")
        lines.extend(["", "## Molecules"])
        for idx, molecule in enumerate(report.molecules, start=1):
            lines.extend(
                [
                    f"- Molecule {idx}",
                    f"  - original_input: `{molecule.original_input}`",
                    f"  - name: `{molecule.name}`",
                    f"  - source: `{molecule.source}`",
                    f"  - canonical_smiles: `{molecule.canonical_smiles}`",
                    f"  - valid: `{molecule.is_valid}`",
                ]
            )

        narrative = report.narrative_report or {}
        if narrative:
            lines.extend(["", "## Human-Readable Summary"])
            if narrative.get("headline"):
                lines.append(f"**{narrative['headline']}**")
            if narrative.get("plain_language_summary"):
                lines.extend(["", str(narrative["plain_language_summary"])])
            if narrative.get("key_reasons"):
                lines.extend(["", "### Key Reasons"])
                lines.extend(f"- {reason}" for reason in narrative.get("key_reasons", []))
            if narrative.get("evidence_summary"):
                lines.extend(["", "### Evidence Summary", str(narrative["evidence_summary"])])
            if narrative.get("uncertainty"):
                lines.extend(["", "### Uncertainty", str(narrative["uncertainty"])])
            if narrative.get("recommended_use"):
                lines.extend(["", "### Recommended Use", str(narrative["recommended_use"])])

        lines.extend(["", "## Interpretation", report.explanation])
        summary = report.reasoning.get("summary") if report.reasoning else None
        if summary:
            lines.extend(["", "## Agent Reasoning", summary])

        scoring = (report.artifacts or {}).get("evidence_grounded_scoring", {})
        if scoring:
            lines.extend(
                [
                    "",
                    "## Evidence-Grounded Score",
                    f"- evidence_grounded_score: `{scoring.get('evidence_grounded_score')}`",
                    f"- support_score: `{scoring.get('support_score')}`",
                    f"- contradiction_score: `{scoring.get('contradiction_score')}`",
                    f"- source_reliability_score: `{scoring.get('source_reliability_score')}`",
                    f"- recommendation: `{scoring.get('recommendation')}`",
                ]
            )

        lines.extend(["", "## Evidence"])
        if report.evidence:
            for item in report.evidence[:12]:
                claim = item.get("claim") or item.get("mechanism") or item.get("interaction_type") or str(item)
                source = item.get("source", "unknown")
                direction = item.get("evidence_direction")
                value = item.get("value")
                value_text = f" value={value}" if value is not None else ""
                direction_text = f" direction={direction}" if direction else ""
                lines.append(f"- `{source}`{direction_text}: {claim}{value_text}")
            if len(report.evidence) > 12:
                lines.append(f"- ... {len(report.evidence) - 12} more evidence items in JSON report.")
        else:
            lines.append("- No evidence items were retrieved.")

        if report.warnings:
            lines.extend(["", "## Warnings"])
            lines.extend(f"- {warning}" for warning in report.warnings)

        lines.extend(["", "## Agent Plan"])
        for step in report.plan.get("steps", []) if report.plan else []:
            lines.append(f"- `{step.get('step_id')}` via `{step.get('tool_name')}`: {step.get('status')}")

        artifacts = report.artifacts or {}
        lines.extend(["", "## Artifacts"])
        for key in ["image_path", "grad_cam_path", "drug_a_image_path", "drug_b_image_path", "agent_plan_path", "agent_trace_path"]:
            if artifacts.get(key):
                lines.append(f"- {key}: `{artifacts[key]}`")
        return "\n".join(lines).rstrip() + "\n"

    def _prediction_meaning(self, prediction: PredictionResult) -> str | None:
        details = prediction.details or {}
        if prediction.label == details.get("positive_label"):
            return details.get("positive_label_meaning")
        if prediction.label == details.get("negative_label"):
            return details.get("negative_label_meaning")
        if prediction.task_type == "ddi_prediction":
            return "DDI risk/interaction" if prediction.label == 1 else "no DDI risk/interaction"
        return None
