from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
import re
from typing import Any, Dict, List, Mapping, Optional

from agentm.llm.provider import LLMClient, build_llm_client
from agentm.schema import MoleculeRecord, PredictionResult
from agentm.utils.io import ensure_dir, write_json, write_text


@dataclass
class NarrativeReport:
    headline: str
    plain_language_summary: str
    key_reasons: List[str]
    evidence_summary: str
    uncertainty: str
    recommended_use: str
    structured_sections: Dict[str, Any] = field(default_factory=dict)
    evidence_citations: List[Dict[str, Any]] = field(default_factory=list)
    claim_evidence_alignment: List[Dict[str, Any]] = field(default_factory=list)
    claim_level_citations: List[Dict[str, Any]] = field(default_factory=list)
    provider: str = "template_fallback"
    language: str = "en"
    evidence_context: Dict[str, Any] = field(default_factory=dict)
    artifacts: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class EvidenceRAGContextBuilder:
    """Build compact grounding context from prediction, evidence, trace, and optional RAG artifacts."""

    def build(
        self,
        *,
        request: str,
        task_type: str,
        molecules: List[MoleculeRecord],
        predictions: List[PredictionResult],
        evidence: List[Dict[str, Any]],
        artifacts: Dict[str, Any],
        reasoning: Dict[str, Any],
        trace: List[Dict[str, Any]],
        warnings: List[str],
    ) -> Dict[str, Any]:
        prediction = predictions[0].to_dict() if predictions else {}
        evidence_items = [self._compact_evidence_item(item, idx) for idx, item in enumerate(evidence[:24], start=1)]
        source_counts = Counter(str(item.get("source", "unknown")) for item in evidence)
        trace_summary = self._trace_summary(trace)
        rag_context = self._extract_rag_context(artifacts)
        context = {
            "request": request,
            "task_type": task_type,
            "prediction": prediction,
            "molecules": [self._compact_molecule(molecule) for molecule in molecules],
            "evidence_items": evidence_items,
            "evidence_groups": self._evidence_groups(evidence_items),
            "evidence_source_counts": dict(source_counts),
            "evidence_grounded_scoring": artifacts.get("evidence_grounded_scoring", {}),
            "evidence_quality": artifacts.get("evidence_quality", {}),
            "reasoning_summary": (reasoning or {}).get("summary", ""),
            "confidence_signals": (reasoning or {}).get("confidence_signals", []),
            "trace_summary": trace_summary,
            "warnings": list(warnings),
            "rag_context": rag_context,
            "rag_summary": self._rag_summary(rag_context),
            "visual_context": self._visual_context(artifacts),
            "pair_context": self._pair_context(artifacts),
            "model_context": self._model_context(prediction),
            "top_feature_contributions": self._top_feature_contributions(prediction),
            "artifact_paths": self._artifact_paths(artifacts),
        }
        context["context_mode"] = "rag_context" if rag_context else "evidence_context"
        return context

    def _compact_molecule(self, molecule: MoleculeRecord) -> Dict[str, Any]:
        return {
            "original_input": molecule.original_input,
            "name": molecule.name,
            "source": molecule.source,
            "canonical_smiles": molecule.canonical_smiles,
            "is_valid": molecule.is_valid,
        }

    def _compact_evidence_item(self, item: Mapping[str, Any], idx: int) -> Dict[str, Any]:
        claim = item.get("claim") or item.get("mechanism") or item.get("interaction_type") or item.get("title") or ""
        if not claim:
            claim = self._compact_value({k: v for k, v in item.items() if k not in {"value"}}, max_len=180)
        value = item.get("value") if isinstance(item.get("value"), Mapping) else {}
        return {
            "id": f"E{idx}",
            "source": str(item.get("source", "unknown")),
            "claim": str(claim)[:500],
            "direction": item.get("evidence_direction"),
            "weight": item.get("evidence_weight"),
            "source_reliability": item.get("source_reliability"),
            "url": item.get("url") or item.get("link") or value.get("url"),
            "pmid": item.get("pmid") or value.get("pmid"),
            "cid": item.get("cid") or value.get("cid"),
            "pmids": item.get("pmids") or value.get("pmids"),
            "setid": item.get("setid") or value.get("setid"),
            "source_path": value.get("source_path"),
            "chunk_id": value.get("chunk_id"),
            "doc_id": value.get("doc_id"),
            "query": value.get("query"),
            "rank": value.get("rank"),
            "retrieval_score": value.get("score"),
            "snippet": str(value.get("snippet") or "")[:500],
            "pair_key": value.get("pair_key"),
            "value": self._compact_value(item.get("value"), max_len=300),
            "rationale": str(item.get("evidence_scoring_rationale") or "")[:300],
            "evidence_type": item.get("evidence_type") or (item.get("evidence_quality") or {}).get("evidence_type"),
            "quality_score": item.get("evidence_quality_score") or (item.get("evidence_quality") or {}).get("quality_score"),
            "support_level": item.get("support_level") or (item.get("evidence_quality") or {}).get("support_level"),
            "quality": item.get("evidence_quality") if isinstance(item.get("evidence_quality"), Mapping) else {},
        }

    def _evidence_groups(self, evidence_items: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        groups = {"support": [], "contradiction": [], "neutral": []}
        for item in evidence_items:
            direction = str(item.get("direction") or "neutral")
            if direction not in groups:
                direction = "neutral"
            groups[direction].append(item)
        return {key: value[:8] for key, value in groups.items()}

    def _trace_summary(self, trace: List[Dict[str, Any]]) -> Dict[str, Any]:
        statuses = Counter(str(item.get("status", "unknown")) for item in trace)
        fallback_steps = [
            {
                "step_id": item.get("step_id"),
                "tool_name": item.get("tool_name"),
                "fallback_tool": item.get("fallback_tool"),
                "error": item.get("error"),
            }
            for item in trace
            if item.get("fallback_tool") or str(item.get("status", "")).startswith("fallback")
        ]
        failed_steps = [
            {
                "step_id": item.get("step_id"),
                "tool_name": item.get("tool_name"),
                "error": item.get("error"),
            }
            for item in trace
            if item.get("status") == "failed"
        ]
        return {
            "n_steps": len(trace),
            "statuses": dict(statuses),
            "fallback_steps": fallback_steps[:8],
            "failed_steps": failed_steps[:8],
            "last_step": trace[-1].get("step_id") if trace else None,
        }

    def _extract_rag_context(self, artifacts: Dict[str, Any]) -> List[Dict[str, Any]]:
        raw = artifacts.get("rag_context") or artifacts.get("retrieved_context")
        if raw is None and isinstance(artifacts.get("rag_answer"), dict):
            raw = artifacts["rag_answer"].get("retrieved_context")
        if not isinstance(raw, list):
            return []
        context = []
        for idx, item in enumerate(raw[:8], start=1):
            if not isinstance(item, dict):
                continue
            context.append(
                {
                    "id": f"R{idx}",
                    "chunk_id": item.get("chunk_id"),
                    "source_path": item.get("source_path"),
                    "score": item.get("score"),
                    "snippet": str(item.get("snippet") or item.get("text") or "")[:800],
                }
            )
        return context

    def _rag_summary(self, rag_context: List[Dict[str, Any]]) -> Dict[str, Any]:
        source_counts = Counter()
        examples = []
        for item in rag_context:
            source_path = str(item.get("source_path") or "")
            if "dailymed" in source_path.lower():
                source = "DailyMed"
            elif "pubmed" in source_path.lower():
                source = "PubMed"
            elif "pubchem" in source_path.lower():
                source = "PubChem"
            else:
                source = "local_rag"
            source_counts[source] += 1
            examples.append(
                {
                    "source": source,
                    "source_path": source_path,
                    "chunk_id": item.get("chunk_id"),
                    "score": item.get("score"),
                    "snippet": str(item.get("snippet") or "")[:260],
                }
            )
        return {"n_context": len(rag_context), "source_counts": dict(source_counts), "examples": examples[:5]}

    def _visual_context(self, artifacts: Dict[str, Any]) -> Dict[str, Any]:
        visual = artifacts.get("visual_semantics") if isinstance(artifacts.get("visual_semantics"), dict) else {}
        return {
            "summary": visual.get("summary"),
            "functional_groups": visual.get("functional_groups", []),
            "admet_alerts": visual.get("admet_alerts", []),
            "provider": visual.get("provider"),
        }

    def _pair_context(self, artifacts: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "fingerprint_similarity": artifacts.get("fingerprint_similarity"),
            "admet_risk_features": artifacts.get("admet_risk_features"),
            "cyp_related_features": artifacts.get("cyp_related_features"),
            "toxicity_overlap_features": artifacts.get("toxicity_overlap_features"),
        }

    def _model_context(self, prediction: Mapping[str, Any]) -> Dict[str, Any]:
        details = prediction.get("details") if isinstance(prediction.get("details"), Mapping) else {}
        return {
            "model_type": prediction.get("model_type"),
            "label": prediction.get("label"),
            "score": prediction.get("score"),
            "probability": prediction.get("probability"),
            "calibrated_probability": details.get("calibrated_probability"),
            "calibration_status": details.get("calibration_status"),
            "calibration_method": details.get("calibration_method"),
            "confidence_level": details.get("confidence_level"),
            "decision_threshold": details.get("decision_threshold"),
            "positive_label_meaning": details.get("positive_label_meaning"),
            "negative_label_meaning": details.get("negative_label_meaning"),
            "warning": details.get("warning"),
        }

    def _top_feature_contributions(self, prediction: Mapping[str, Any]) -> List[Dict[str, Any]]:
        details = prediction.get("details") if isinstance(prediction.get("details"), Mapping) else {}
        rows = details.get("evidence_contributions") if isinstance(details.get("evidence_contributions"), list) else []
        compact = []
        for item in rows:
            if not isinstance(item, Mapping):
                continue
            compact.append(
                {
                    "source": item.get("source"),
                    "feature": item.get("feature"),
                    "direction": item.get("direction"),
                    "weight": item.get("weight"),
                    "value": item.get("value"),
                    "rationale": str(item.get("rationale") or "")[:260],
                }
            )
        compact.sort(key=lambda row: abs(float(row.get("weight") or 0.0)), reverse=True)
        return compact[:8]

    def _artifact_paths(self, artifacts: Dict[str, Any]) -> Dict[str, str]:
        keys = [
            "agent_trace_path",
            "agent_plan_path",
            "rag_evidence_path",
            "rag_evidence_markdown_path",
            "case_study_report_path",
            "reproducibility_log_path",
        ]
        return {key: str(artifacts[key]) for key in keys if artifacts.get(key)}

    def _compact_value(self, value: Any, *, max_len: int) -> str:
        if value is None:
            return ""
        try:
            text = json.dumps(value, ensure_ascii=False, sort_keys=True)
        except Exception:
            text = str(value)
        text = re.sub(r"\s+", " ", text).strip()
        return text[:max_len] + ("..." if len(text) > max_len else "")


class NarrativeReportAgent:
    """Turn structured agent outputs into grounded, human-readable explanations."""

    def __init__(
        self,
        output_dir: str | Path = "outputs/narrative",
        *,
        llm_profile: Optional[str] = None,
        use_llm: bool = False,
        llm_client: Optional[LLMClient] = None,
        language: str = "auto",
    ):
        self.output_dir = ensure_dir(output_dir)
        self.context_builder = EvidenceRAGContextBuilder()
        self.use_llm = bool(use_llm)
        self.llm_client = llm_client if llm_client is not None else (build_llm_client(llm_profile) if use_llm else None)
        self.language = language

    def build(
        self,
        *,
        request: str,
        task_type: str,
        molecules: List[MoleculeRecord],
        predictions: List[PredictionResult],
        evidence: List[Dict[str, Any]],
        artifacts: Dict[str, Any],
        reasoning: Dict[str, Any],
        trace: List[Dict[str, Any]],
        warnings: List[str],
    ) -> Dict[str, Any]:
        language = self._language(request)
        context = self.context_builder.build(
            request=request,
            task_type=task_type,
            molecules=molecules,
            predictions=predictions,
            evidence=evidence,
            artifacts=artifacts,
            reasoning=reasoning,
            trace=trace,
            warnings=warnings,
        )
        fallback = self._template_report(context, language=language)
        report = fallback
        if self.use_llm and self.llm_client and self.llm_client.available:
            try:
                report = self._llm_report(context, fallback, language=language)
            except Exception as exc:
                fallback.warnings.append(f"LLM narrative failed; used template fallback: {exc}")
                report = fallback
        self._write_report(report)
        return report.to_dict()

    def _template_report(self, context: Dict[str, Any], *, language: str) -> NarrativeReport:
        prediction = context.get("prediction") or {}
        details = prediction.get("details") or {}
        task_type = context.get("task_type")
        dataset_name = prediction.get("dataset_name") or "the requested task"
        label = prediction.get("label")
        raw_score = prediction.get("probability")
        if raw_score is None:
            raw_score = prediction.get("score")
        score = self._fmt_score(raw_score)
        meaning = self._prediction_meaning(prediction)
        model_type = prediction.get("model_type") or "unknown model"
        scoring = context.get("evidence_grounded_scoring") or {}
        quality = context.get("evidence_quality") or {}
        rec = scoring.get("recommendation")
        evidence_items = context.get("evidence_items") or []
        top_sources = ", ".join(sorted(context.get("evidence_source_counts", {}).keys())[:6]) or "local model signals"
        uncertainty = self._uncertainty_text(context, language=language)
        reasons = self._key_reasons(context, language=language)
        structured_sections = self._structured_sections(context, language=language)
        evidence_citations = self._evidence_citations(context)
        claim_alignment = self._claim_evidence_alignment(context, language=language)
        claim_level_citations = self._claim_level_citations(context, claim_alignment)
        headline, direct_summary = self._direct_answer(context, language=language, meaning=meaning, label=label, score=score)

        if language == "zh":
            if task_type == "ddi_prediction":
                summary = (
                    f"{direct_summary} "
                    f"agentM 综合了 pair-level 分子相似性、ADMET/CYP/toxicity 线索、DDI 规则提示和检索证据；"
                    f"因此输出更适合作为“需要复核的相互作用风险提示”，而不是单一规则判断。"
                )
            else:
                summary = (
                    f"{direct_summary} "
                    f"agentM 综合了 RDKit 描述符、分子指纹、LLM4SD 风格规则、图像/视觉语义表征和可用证据；"
                    f"其中结构线索用于解释，检索证据用于支撑或提醒不确定性。"
                )
            evidence_summary = self._evidence_summary_text(context, language=language, fallback_sources=top_sources, recommendation=rec)
            recommended_use = (
                "建议把该结果用于候选分子/药物组合的优先级排序、文献复核和实验设计参考；"
                "涉及用药、临床或监管决策时，必须结合原始说明书、文献全文、专业药师/医生判断和具体患者信息。"
            )
        else:
            if task_type == "ddi_prediction":
                summary = (
                    f"{direct_summary} "
                    "agentM combined pair-level molecular similarity, ADMET/CYP/toxicity cues, DDI rule hints, and retrieved evidence, "
                    "so it should be read as an interaction-risk signal for review rather than a single-rule verdict."
                )
            else:
                summary = (
                    f"{direct_summary} "
                    "agentM combined RDKit descriptors, molecular fingerprints, LLM4SD-style rules, image/visual semantics, and retrieved evidence."
                )
            evidence_summary = self._evidence_summary_text(context, language=language, fallback_sources=top_sources, recommendation=rec)
            recommended_use = (
                "Use this result for molecule or drug-pair prioritization, evidence review, and follow-up experimental planning. "
                "For medication, clinical, or regulatory decisions, review the primary labels/literature and consult qualified professionals."
            )

        calibration_status = (prediction.get("details") or {}).get("calibration_status") if isinstance(prediction.get("details"), Mapping) else None
        if model_type == "zero_shot_rule_semantic_fusion":
            summary += " " + (
                "该结论来自 zero-shot 规则/语义融合，不能等同于经过任务特异监督校准的概率。"
                if language == "zh"
                else "This conclusion comes from zero-shot rule/semantic fusion and is not a task-specific calibrated probability."
            )
        elif calibration_status in {None, "not_evaluated", "not_calibrated"} and prediction.get("probability") is not None:
            summary += " " + (
                "该概率尚未经过本任务的后验校准评估。"
                if language == "zh"
                else "This probability has not yet been post-hoc calibrated for this task."
            )

        return NarrativeReport(
            headline=headline,
            plain_language_summary=summary,
            key_reasons=reasons,
            evidence_summary=evidence_summary,
            uncertainty=uncertainty,
            recommended_use=recommended_use,
            structured_sections=structured_sections,
            evidence_citations=evidence_citations,
            claim_evidence_alignment=claim_alignment,
            claim_level_citations=claim_level_citations,
            provider="template_fallback",
            language=language,
            evidence_context=context,
            warnings=[],
        )

    def _llm_report(self, context: Dict[str, Any], fallback: NarrativeReport, *, language: str) -> NarrativeReport:
        assert self.llm_client is not None
        prompt = {
            "instruction": (
                "Rewrite the structured agent output into a concise human-readable biomedical report. "
                "The headline and first sentence must answer the user's question directly before mentioning dataset labels. "
                "Do not change the prediction label, score, model type, evidence counts, warnings, or uncertainty. "
                "Do not invent citations, mechanisms, databases, or evidence. If evidence is weak, say so."
            ),
            "language": "Chinese" if language == "zh" else "English",
            "required_json_fields": [
                "headline",
                "plain_language_summary",
                "key_reasons",
                "evidence_summary",
                "uncertainty",
                "recommended_use",
                "structured_sections",
            ],
            "fallback_template": fallback.to_dict(),
            "grounding_context": context,
        }
        response = self.llm_client.chat_json(
            [
                {
                    "role": "system",
                    "content": (
                        "You are a careful scientific report writer. "
                        "Return JSON only. Use only the provided grounding context."
                    ),
                },
                {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
            ],
            temperature=0.0,
        )
        payload = self._parse_json(response)
        merged = fallback.to_dict()
        for key in ["headline", "plain_language_summary", "evidence_summary", "uncertainty", "recommended_use"]:
            if payload.get(key):
                merged[key] = str(payload[key])
        if isinstance(payload.get("key_reasons"), list) and payload["key_reasons"]:
            merged["key_reasons"] = [str(item) for item in payload["key_reasons"][:8]]
        if isinstance(payload.get("structured_sections"), dict):
            merged["structured_sections"] = payload["structured_sections"]
        if not merged.get("claim_evidence_alignment"):
            merged["claim_evidence_alignment"] = fallback.claim_evidence_alignment
        merged["provider"] = f"llm_narrative:{self.llm_client.config.profile}"
        merged["language"] = language
        merged["evidence_context"] = context
        merged["warnings"] = list(fallback.warnings)
        return NarrativeReport(**{key: merged[key] for key in NarrativeReport.__dataclass_fields__})

    def _parse_json(self, text: str) -> Dict[str, Any]:
        raw = text.strip()
        if raw.startswith("```"):
            raw = raw.strip("`")
            if raw.lower().startswith("json"):
                raw = raw[4:].strip()
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            raw = raw[start : end + 1]
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError("Narrative LLM response must be a JSON object.")
        return data

    def _direct_answer(
        self,
        context: Mapping[str, Any],
        *,
        language: str,
        meaning: str,
        label: Any,
        score: str,
    ) -> tuple[str, str]:
        prediction = context.get("prediction") or {}
        task_type = context.get("task_type")
        dataset_name = str(prediction.get("dataset_name") or "")
        subject = self._subject_text(context, language=language)
        positive = self._is_positive_prediction(prediction)

        if task_type == "ddi_prediction":
            if language == "zh":
                if positive is False:
                    headline = f"{subject} 未显示明显相互作用风险"
                    summary = f"结论：{subject} 的相互作用风险信号较低。"
                else:
                    headline = f"{subject} 可能存在相互作用风险"
                    summary = f"结论：{subject} 可能存在需要复核的相互作用风险。"
            else:
                if positive is False:
                    headline = f"{subject} shows low apparent DDI risk"
                    summary = f"Conclusion: {subject} shows a low apparent interaction-risk signal."
                else:
                    headline = f"{subject} may have DDI risk"
                    summary = f"Conclusion: {subject} may have an interaction risk that should be reviewed."
            return headline, summary

        if language == "zh":
            endpoint = self._zh_endpoint(dataset_name)
            if positive is True:
                positive_text, _ = self._zh_endpoint_decision(dataset_name)
                statement = self._zh_join_decision(subject, positive_text)
                headline = statement
                summary = f"结论：{statement}。"
            elif positive is False:
                _, negative_text = self._zh_endpoint_decision(dataset_name)
                statement = self._zh_join_decision(subject, negative_text)
                headline = statement
                summary = f"结论：{statement}。"
            elif endpoint:
                headline = f"{subject}的{endpoint}估计值为 {score}"
                summary = f"结论：{subject}的{endpoint}估计值约为 {score}。"
            else:
                headline = f"{subject}的预测结果为：{meaning or label}"
                summary = f"结论：{subject}的预测结果为 `{meaning or label}`。"
            return headline, summary

        endpoint = self._en_endpoint(dataset_name)
        if positive is True:
            positive_text, _ = self._en_endpoint_decision(dataset_name)
            headline = f"{subject} {positive_text}"
            summary = f"Conclusion: {subject} {positive_text}."
        elif positive is False:
            _, negative_text = self._en_endpoint_decision(dataset_name)
            headline = f"{subject} {negative_text}"
            summary = f"Conclusion: {subject} {negative_text}."
        elif endpoint:
            headline = f"{subject} has an estimated {endpoint} of {score}"
            summary = f"Conclusion: {subject} has an estimated {endpoint} of about {score}."
        else:
            headline = f"{subject} is predicted as: {meaning or label}"
            summary = f"Conclusion: {subject} is predicted as `{meaning or label}`."
        return headline, summary

    def _subject_text(self, context: Mapping[str, Any], *, language: str) -> str:
        molecules = context.get("molecules") or []
        names = []
        for molecule in molecules[:2]:
            if not isinstance(molecule, Mapping):
                continue
            value = molecule.get("name") or molecule.get("original_input") or molecule.get("canonical_smiles")
            if value:
                names.append(str(value))
        if len(names) >= 2:
            return f"{names[0]} 和 {names[1]}" if language == "zh" else f"{names[0]} and {names[1]}"
        if names:
            return names[0]
        return "该分子" if language == "zh" else "the molecule"

    def _is_positive_prediction(self, prediction: Mapping[str, Any]) -> Optional[bool]:
        details = prediction.get("details") if isinstance(prediction.get("details"), Mapping) else {}
        label = prediction.get("label")
        if "positive_label" in details and label == details.get("positive_label"):
            return True
        if "negative_label" in details and label == details.get("negative_label"):
            return False
        if label in {1, "1", True}:
            return True
        if label in {0, "0", False}:
            return False
        return None

    def _zh_endpoint(self, dataset_name: str) -> str:
        key = dataset_name.lower()
        if key in {"dili"}:
            return "肝毒性风险"
        if "bbb" in key:
            return "血脑屏障透过性"
        if key == "herg":
            return "hERG/QT 相关心脏毒性风险"
        if "ames" in key:
            return "Ames 致突变风险"
        if "clintox" in key:
            return "临床毒性风险"
        if key.startswith("cyp") or "cyp" in key:
            return f"{self._cyp_name(dataset_name)} 抑制风险"
        if "bioavailability" in key:
            return "口服生物利用度"
        if "hia" in key:
            return "肠道吸收倾向"
        if "caco2" in key:
            return "Caco-2 通透性"
        if "pgp" in key:
            return "P-gp 转运相关风险"
        if "solubility" in key or "esol" in key:
            return "水溶性"
        if "lipophilicity" in key:
            return "脂溶性"
        if "clearance" in key:
            return "清除率"
        if "half_life" in key or "half-life" in key:
            return "半衰期"
        if "ld50" in key:
            return "急性毒性风险"
        return dataset_name

    def _zh_endpoint_decision(self, dataset_name: str) -> tuple[str, str]:
        key = dataset_name.lower()
        if key in {"dili"}:
            return "可能存在肝毒性风险", "肝毒性风险较低"
        if "bbb" in key:
            return "可能通过血脑屏障", "不太可能通过血脑屏障"
        if key == "herg":
            return "可能有 hERG/QT 相关心脏毒性风险", "hERG/QT 相关心脏毒性风险较低"
        if "ames" in key:
            return "可能有 Ames 致突变风险", "Ames 致突变风险较低"
        if "clintox" in key:
            return "可能有临床毒性风险", "临床毒性风险较低"
        if key.startswith("cyp") or "cyp" in key:
            cyp = self._cyp_name(dataset_name)
            return f"可能抑制 {cyp}", f"不太可能抑制 {cyp}"
        if "bioavailability" in key:
            return "可能有较好的口服生物利用度", "口服生物利用度可能偏低"
        if "hia" in key:
            return "可能有较好的肠道吸收", "肠道吸收可能偏低"
        if "caco2" in key:
            return "可能有较好的 Caco-2 通透性", "Caco-2 通透性可能偏低"
        if "pgp" in key:
            return "可能涉及 P-gp 转运", "P-gp 转运风险信号较低"
        if "ld50" in key:
            return "可能有较高急性毒性风险", "急性毒性风险较低"
        endpoint = self._zh_endpoint(dataset_name)
        return f"可能属于 {endpoint} 阳性", f"更像 {endpoint} 阴性或低风险"

    def _zh_join_decision(self, subject: str, decision: str) -> str:
        if decision.startswith(("可能", "不太可能", "不太会", "更像")):
            return f"{subject} {decision}"
        return f"{subject} 的{decision}"

    def _en_endpoint(self, dataset_name: str) -> str:
        key = dataset_name.lower()
        if key == "dili":
            return "DILI/hepatotoxicity risk"
        if "bbb" in key:
            return "blood-brain barrier permeability"
        if key == "herg":
            return "hERG/QT cardiac-liability risk"
        if "ames" in key:
            return "Ames mutagenicity risk"
        if "clintox" in key:
            return "clinical toxicity risk"
        if key.startswith("cyp") or "cyp" in key:
            return f"{self._cyp_name(dataset_name)} inhibition risk"
        return dataset_name

    def _en_endpoint_decision(self, dataset_name: str) -> tuple[str, str]:
        key = dataset_name.lower()
        if key == "dili":
            return "may have DILI/hepatotoxicity risk", "has a low DILI/hepatotoxicity risk signal"
        if "bbb" in key:
            return "is likely to cross the blood-brain barrier", "is less likely to cross the blood-brain barrier"
        if key == "herg":
            return "may have hERG/QT cardiac-liability risk", "has a low hERG/QT cardiac-liability signal"
        if "ames" in key:
            return "may have Ames mutagenicity risk", "has a low Ames mutagenicity risk signal"
        if "clintox" in key:
            return "may have clinical toxicity risk", "has a low clinical-toxicity risk signal"
        if key.startswith("cyp") or "cyp" in key:
            cyp = self._cyp_name(dataset_name)
            return f"may inhibit {cyp}", f"is less likely to inhibit {cyp}"
        endpoint = self._en_endpoint(dataset_name)
        return f"is predicted positive for {endpoint}", f"is predicted negative or low-risk for {endpoint}"

    def _cyp_name(self, dataset_name: str) -> str:
        match = re.search(r"CYP\d+[A-Z]?\d*", dataset_name, flags=re.IGNORECASE)
        return match.group(0).upper() if match else "CYP"

    def _write_report(self, report: NarrativeReport) -> None:
        json_path = self.output_dir / "narrative_report.json"
        md_path = self.output_dir / "narrative_report.md"
        report.artifacts.update({"narrative_report_path": str(json_path), "narrative_markdown_path": str(md_path)})
        write_json(json_path, report.to_dict())
        lines = [
            "# Narrative Report",
            "",
            f"## {report.headline}",
            "",
            report.plain_language_summary,
            "",
            "## Key Reasons",
        ]
        lines.extend(f"- {reason}" for reason in report.key_reasons)
        if report.structured_sections:
            lines.extend(["", "## Decision Basis"])
            for item in report.structured_sections.get("decision_basis", []) or []:
                lines.append(f"- {item}")
            lines.extend(["", "## Evidence Chain"])
            for item in report.structured_sections.get("evidence_chain", []) or []:
                lines.append(f"- {item}")
            limitations = report.structured_sections.get("limitations", []) or []
            if limitations:
                lines.extend(["", "## Limitations"])
                lines.extend(f"- {item}" for item in limitations)
            next_steps = report.structured_sections.get("next_steps", []) or []
            if next_steps:
                lines.extend(["", "## Suggested Follow-Up"])
                lines.extend(f"- {item}" for item in next_steps)
        if report.claim_evidence_alignment:
            lines.extend(["", "## Claim-Evidence Alignment"])
            lines.append("| Claim | Support | Uncertainty | Evidence IDs | Citations |")
            lines.append("|---|---|---|---|---|")
            for row in report.claim_evidence_alignment[:12]:
                citations = row.get("citations") if isinstance(row, Mapping) else []
                citation_labels = ", ".join(str(item.get("label") or item.get("citation_id") or "") for item in citations[:4] if isinstance(item, Mapping))
                lines.append(
                    "| {claim} | {support} | {uncertainty} | {evidence_ids} | {citations} |".format(
                        claim=self._pipe_md(row.get("claim")),
                        support=self._pipe_md(row.get("support_level")),
                        uncertainty=self._pipe_md(row.get("uncertainty")),
                        evidence_ids=self._pipe_md(", ".join(str(item) for item in row.get("evidence_ids", []) or []) or "none"),
                        citations=self._pipe_md(citation_labels or "none"),
                    )
                )
        if report.claim_level_citations:
            lines.extend(["", "## Claim-Level Citations"])
            lines.append("| Claim | Citation | Source | Pointer | Score |")
            lines.append("|---|---|---|---|---:|")
            for row in report.claim_level_citations[:20]:
                for citation in row.get("citations", [])[:5]:
                    pointer = citation.get("url") or citation.get("pmid") or citation.get("cid") or citation.get("setid") or citation.get("chunk_id") or citation.get("source_path")
                    lines.append(
                        "| {claim} | {label} | {source} | {pointer} | {score} |".format(
                            claim=self._pipe_md(row.get("claim")),
                            label=self._pipe_md(citation.get("label")),
                            source=self._pipe_md(citation.get("source")),
                            pointer=self._pipe_md(pointer),
                            score=self._pipe_md(self._fmt_score(citation.get("retrieval_score"))),
                        )
                    )
        lines.extend(["", "## Evidence", report.evidence_summary])
        if report.evidence_citations:
            lines.extend(["", "## Source Pointers"])
            for citation in report.evidence_citations[:10]:
                label = citation.get("label") or citation.get("source") or "source"
                target = citation.get("url") or citation.get("source_path") or citation.get("chunk_id") or ""
                lines.append(f"- {label}: {target}")
        lines.extend(["", "## Uncertainty", report.uncertainty, "", "## Recommended Use", report.recommended_use])
        if report.warnings:
            lines.extend(["", "## Warnings"])
            lines.extend(f"- {warning}" for warning in report.warnings)
        write_text(md_path, "\n".join(lines).rstrip() + "\n")

    def _key_reasons(self, context: Dict[str, Any], *, language: str) -> List[str]:
        reasons: List[str] = []
        prediction = context.get("prediction") or {}
        scoring = context.get("evidence_grounded_scoring") or {}
        visual = context.get("visual_context") or {}
        pair = context.get("pair_context") or {}
        evidence_items = context.get("evidence_items") or []
        groups = context.get("evidence_groups") or {}
        contributions = context.get("top_feature_contributions") or []
        top_support = groups.get("support", [])[:2]
        top_contra = groups.get("contradiction", [])[:1]
        if language == "zh":
            reasons.append(f"预测模型 `{prediction.get('model_type')}` 给出了标签 `{self._prediction_meaning(prediction) or prediction.get('label')}`。")
            if contributions:
                reasons.append("主要模型特征贡献：" + "；".join(self._contribution_text(item, language=language) for item in contributions[:3]) + "。")
            if scoring:
                reasons.append(f"证据一致性评分为 {self._fmt_score(scoring.get('evidence_grounded_score'))}，建议 `{scoring.get('recommendation')}`。")
            if top_support:
                reasons.append("支持性证据：" + "；".join(self._evidence_line(item, language=language) for item in top_support) + "。")
            if top_contra:
                reasons.append("需要复核的相反证据：" + "；".join(self._evidence_line(item, language=language) for item in top_contra) + "。")
            if visual.get("summary"):
                reasons.append(f"视觉/结构语义提示：{visual.get('summary')}")
            if pair.get("fingerprint_similarity"):
                reasons.append(f"DDI pair 层面计算了指纹相似度和 ADMET/CYP/toxicity overlap 线索。")
            if evidence_items:
                reasons.append(f"证据层纳入了 {len(evidence_items)} 条本地或外部证据，来源包括 {', '.join(sorted(context.get('evidence_source_counts', {}).keys())[:5])}。")
        else:
            reasons.append(f"The `{prediction.get('model_type')}` predictor produced label `{self._prediction_meaning(prediction) or prediction.get('label')}`.")
            if contributions:
                reasons.append("Main model feature contributions: " + "; ".join(self._contribution_text(item, language=language) for item in contributions[:3]) + ".")
            if scoring:
                reasons.append(f"The evidence-grounded score is {self._fmt_score(scoring.get('evidence_grounded_score'))} with recommendation `{scoring.get('recommendation')}`.")
            if top_support:
                reasons.append("Supporting evidence: " + "; ".join(self._evidence_line(item, language=language) for item in top_support) + ".")
            if top_contra:
                reasons.append("Contradictory evidence to review: " + "; ".join(self._evidence_line(item, language=language) for item in top_contra) + ".")
            if visual.get("summary"):
                reasons.append(f"Visual/structural semantic cue: {visual.get('summary')}")
            if pair.get("fingerprint_similarity"):
                reasons.append("The DDI pair layer computed fingerprint similarity plus ADMET, CYP, and toxicity-overlap cues.")
            if evidence_items:
                reasons.append(f"The evidence layer included {len(evidence_items)} local or external evidence items from {', '.join(sorted(context.get('evidence_source_counts', {}).keys())[:5])}.")
        return reasons[:8]

    def _structured_sections(self, context: Dict[str, Any], *, language: str) -> Dict[str, Any]:
        prediction = context.get("prediction") or {}
        scoring = context.get("evidence_grounded_scoring") or {}
        groups = context.get("evidence_groups") or {}
        trace = context.get("trace_summary") or {}
        rag = context.get("rag_summary") or {}
        quality = context.get("evidence_quality") or {}
        contributions = context.get("top_feature_contributions") or []
        task_type = context.get("task_type")
        decision_basis: List[str] = []
        evidence_chain: List[str] = []
        limitations: List[str] = []
        next_steps: List[str] = []

        if language == "zh":
            decision_basis.append(f"预测标签为 `{self._prediction_meaning(prediction) or prediction.get('label')}`，模型类型为 `{prediction.get('model_type')}`。")
            if prediction.get("probability") is not None:
                decision_basis.append(f"模型概率为 {self._fmt_score(prediction.get('probability'))}。")
                status = (prediction.get("details") or {}).get("calibration_status")
                if status in {None, "not_evaluated", "not_calibrated"}:
                    decision_basis.append("该概率当前标记为未校准或未评估校准。")
            else:
                decision_basis.append(f"模型分数为 {self._fmt_score(prediction.get('score'))}，当前不是校准概率。")
            for item in contributions[:4]:
                decision_basis.append(self._contribution_text(item, language=language))
            if task_type == "ddi_prediction" and (context.get("pair_context") or {}).get("fingerprint_similarity"):
                decision_basis.append(f"DDI pair 指纹相似度：{(context.get('pair_context') or {}).get('fingerprint_similarity')}。")

            evidence_chain.append(
                f"证据分组：支持 {len(groups.get('support', []))} 条，反对 {len(groups.get('contradiction', []))} 条，中性 {len(groups.get('neutral', []))} 条。"
            )
            if scoring:
                evidence_chain.append(
                    f"证据一致性评分 {self._fmt_score(scoring.get('evidence_grounded_score'))}，覆盖度 {self._fmt_score(scoring.get('coverage_score'))}，建议 `{scoring.get('recommendation')}`。"
                )
            if quality:
                evidence_chain.append(
                    f"证据质量：平均相关性 {self._fmt_score(quality.get('average_relevance_score'))}，平均质量分 {self._fmt_score(quality.get('average_quality_score'))}，直接 pair 证据 {quality.get('direct_pair_evidence_count', 0)} 条，丢弃低相关证据 {quality.get('n_dropped_evidence', 0)} 条。"
                )
            if rag.get("n_context"):
                evidence_chain.append(f"RAG 检索到 {rag.get('n_context')} 个片段，来源分布：{rag.get('source_counts')}。")
            for item in (groups.get("support") or [])[:3]:
                evidence_chain.append("支持证据：" + self._evidence_line(item, language=language))
            for item in (groups.get("contradiction") or [])[:2]:
                evidence_chain.append("反向证据：" + self._evidence_line(item, language=language))

            if prediction.get("model_type") == "zero_shot_rule_semantic_fusion":
                limitations.append("当前使用 zero-shot 规则/语义融合分数，不能解释为经过监督校准的概率。")
            elif (prediction.get("details") or {}).get("calibration_status") in {None, "not_evaluated", "not_calibrated"} and prediction.get("probability") is not None:
                limitations.append("监督模型输出的是原始概率；如需把概率当作风险概率，应先运行 calibration evaluator。")
            if trace.get("fallback_steps"):
                limitations.append(f"workflow 使用了 {len(trace.get('fallback_steps') or [])} 个 fallback，需要查看 trace。")
            if context.get("warnings"):
                limitations.append(f"workflow 有 {len(context.get('warnings') or [])} 条 warning。")
            if not rag.get("n_context"):
                limitations.append("本次没有可用 RAG 片段，解释主要依赖模型和本地证据。")
            if task_type == "ddi_prediction" and quality and not quality.get("direct_pair_evidence_count"):
                limitations.append("未找到明确同时覆盖两个药物的 direct pair evidence；当前 DDI 解释主要依赖机制级或单药标签证据。")

            next_steps.extend(self._next_steps(context, language=language))
        else:
            decision_basis.append(f"The predicted label is `{self._prediction_meaning(prediction) or prediction.get('label')}` from `{prediction.get('model_type')}`.")
            if prediction.get("probability") is not None:
                decision_basis.append(f"The model probability is {self._fmt_score(prediction.get('probability'))}.")
                status = (prediction.get("details") or {}).get("calibration_status")
                if status in {None, "not_evaluated", "not_calibrated"}:
                    decision_basis.append("This probability is currently marked as uncalibrated or not calibration-evaluated.")
            else:
                decision_basis.append(f"The model score is {self._fmt_score(prediction.get('score'))}; it is not a calibrated probability.")
            for item in contributions[:4]:
                decision_basis.append(self._contribution_text(item, language=language))
            if task_type == "ddi_prediction" and (context.get("pair_context") or {}).get("fingerprint_similarity"):
                decision_basis.append(f"DDI pair fingerprint similarity: {(context.get('pair_context') or {}).get('fingerprint_similarity')}.")

            evidence_chain.append(
                f"Evidence groups: {len(groups.get('support', []))} supporting, {len(groups.get('contradiction', []))} contradictory, {len(groups.get('neutral', []))} neutral."
            )
            if scoring:
                evidence_chain.append(
                    f"Evidence-grounded score {self._fmt_score(scoring.get('evidence_grounded_score'))}, coverage {self._fmt_score(scoring.get('coverage_score'))}, recommendation `{scoring.get('recommendation')}`."
                )
            if quality:
                evidence_chain.append(
                    f"Evidence quality: average relevance {self._fmt_score(quality.get('average_relevance_score'))}, average quality {self._fmt_score(quality.get('average_quality_score'))}, direct pair evidence {quality.get('direct_pair_evidence_count', 0)}, dropped low-relevance evidence {quality.get('n_dropped_evidence', 0)}."
                )
            if rag.get("n_context"):
                evidence_chain.append(f"RAG retrieved {rag.get('n_context')} snippets with source distribution {rag.get('source_counts')}.")
            for item in (groups.get("support") or [])[:3]:
                evidence_chain.append("Supporting evidence: " + self._evidence_line(item, language=language))
            for item in (groups.get("contradiction") or [])[:2]:
                evidence_chain.append("Contradictory evidence: " + self._evidence_line(item, language=language))

            if prediction.get("model_type") == "zero_shot_rule_semantic_fusion":
                limitations.append("The score comes from zero-shot rule/semantic fusion and is not a supervised calibrated probability.")
            elif (prediction.get("details") or {}).get("calibration_status") in {None, "not_evaluated", "not_calibrated"} and prediction.get("probability") is not None:
                limitations.append("The supervised model output is a raw probability; run the calibration evaluator before interpreting it as a calibrated risk.")
            if trace.get("fallback_steps"):
                limitations.append(f"The workflow used {len(trace.get('fallback_steps') or [])} fallback step(s); inspect the trace.")
            if context.get("warnings"):
                limitations.append(f"The workflow emitted {len(context.get('warnings') or [])} warning(s).")
            if not rag.get("n_context"):
                limitations.append("No RAG snippets were available; the explanation relies on model and local evidence.")
            if task_type == "ddi_prediction" and quality and not quality.get("direct_pair_evidence_count"):
                limitations.append("No direct pair evidence covering both drugs was found; DDI explanation relies on mechanism-level or single-drug label evidence.")

            next_steps.extend(self._next_steps(context, language=language))
        return {
            "decision_basis": decision_basis[:8],
            "evidence_chain": evidence_chain[:10],
            "claim_evidence_alignment": self._claim_evidence_alignment(context, language=language),
            "limitations": limitations[:8],
            "next_steps": next_steps[:6],
        }

    def _evidence_summary_text(
        self,
        context: Dict[str, Any],
        *,
        language: str,
        fallback_sources: str,
        recommendation: Any,
    ) -> str:
        evidence_items = context.get("evidence_items") or []
        groups = context.get("evidence_groups") or {}
        rag = context.get("rag_summary") or {}
        scoring = context.get("evidence_grounded_scoring") or {}
        quality = context.get("evidence_quality") or {}
        if language == "zh":
            parts = [
                f"本次报告整理了 {len(evidence_items)} 条主要证据，来源包括 {fallback_sources}。",
                f"其中支持 {len(groups.get('support', []))} 条、反对 {len(groups.get('contradiction', []))} 条、中性 {len(groups.get('neutral', []))} 条。",
            ]
            if rag.get("n_context"):
                parts.append(f"RAG 证据检索到 {rag.get('n_context')} 个片段，来源分布为 {rag.get('source_counts')}。")
            if recommendation:
                parts.append(f"证据一致性建议为 `{recommendation}`。")
            if scoring.get("source_reliability_score") is not None:
                parts.append(f"平均来源可靠性评分约为 {self._fmt_score(scoring.get('source_reliability_score'))}。")
            if quality:
                parts.append(
                    f"质量层平均相关性 {self._fmt_score(quality.get('average_relevance_score'))}，重排后保留 {quality.get('n_kept_evidence', len(evidence_items))} 条证据。"
                )
            return " ".join(parts)

        parts = [
            f"The report summarized {len(evidence_items)} main evidence items from {fallback_sources}.",
            f"Evidence grouping: {len(groups.get('support', []))} supporting, {len(groups.get('contradiction', []))} contradictory, {len(groups.get('neutral', []))} neutral.",
        ]
        if rag.get("n_context"):
            parts.append(f"RAG retrieved {rag.get('n_context')} snippets with source distribution {rag.get('source_counts')}.")
        if recommendation:
            parts.append(f"The evidence-grounded recommendation is `{recommendation}`.")
        if scoring.get("source_reliability_score") is not None:
            parts.append(f"The mean source reliability score is about {self._fmt_score(scoring.get('source_reliability_score'))}.")
        if quality:
            parts.append(
                f"The quality layer average relevance is {self._fmt_score(quality.get('average_relevance_score'))}, with {quality.get('n_kept_evidence', len(evidence_items))} evidence items kept after reranking."
            )
        return " ".join(parts)

    def _claim_evidence_alignment(self, context: Dict[str, Any], *, language: str) -> List[Dict[str, Any]]:
        prediction = context.get("prediction") or {}
        groups = context.get("evidence_groups") or {}
        evidence_items = context.get("evidence_items") or []
        quality = context.get("evidence_quality") or {}
        task_type = context.get("task_type")
        support = sorted(groups.get("support", []) or [], key=lambda item: float(item.get("quality_score") or 0.0), reverse=True)
        contradiction = sorted(groups.get("contradiction", []) or [], key=lambda item: float(item.get("quality_score") or 0.0), reverse=True)
        neutral = sorted(groups.get("neutral", []) or [], key=lambda item: float(item.get("quality_score") or 0.0), reverse=True)
        direct = [item for item in evidence_items if item.get("evidence_type") == "direct_pair_evidence"]
        mechanism = [item for item in evidence_items if item.get("evidence_type") == "mechanism_evidence"]
        endpoint_or_literature = [
            item
            for item in evidence_items
            if item.get("evidence_type") in {"mechanism_evidence", "background_literature", "single_drug_label_evidence", "compound_property_evidence"}
        ]

        if language == "zh":
            prediction_claim = f"模型预测为 `{self._prediction_meaning(prediction) or prediction.get('label')}`。"
            direct_claim = "存在明确同时覆盖两个药物的 direct pair evidence。" if direct else "未找到明确同时覆盖两个药物的 direct pair evidence。"
            endpoint_claim = "存在 endpoint/literature evidence 可用于解释该单分子结论。" if endpoint_or_literature else "未找到明确 endpoint/literature evidence；当前解释主要依赖模型与结构证据。"
            if mechanism:
                mechanism_claim = "存在机制级或说明书级证据可用于解释该结论。"
            elif support:
                mechanism_claim = "存在结构/规则级证据可用于解释该结论，但不是外部机制文献证据。"
            else:
                mechanism_claim = "机制级证据不足。"
            uncertainty_claim = "证据不足或间接证据较多时，需要保留不确定性。"
        else:
            prediction_claim = f"The model prediction is `{self._prediction_meaning(prediction) or prediction.get('label')}`."
            direct_claim = "Direct pair evidence covering both drugs is available." if direct else "No direct pair evidence covering both drugs was found."
            endpoint_claim = "Endpoint/literature evidence is available for this single-molecule conclusion." if endpoint_or_literature else "No endpoint/literature evidence was found; the explanation mainly relies on model and structural evidence."
            if mechanism:
                mechanism_claim = "Mechanism-level or label-level evidence is available for interpretation."
            elif support:
                mechanism_claim = "Structural/rule-level evidence is available for interpretation, but it is not external mechanism literature."
            else:
                mechanism_claim = "Mechanism-level evidence is insufficient."
            uncertainty_claim = "Uncertainty should be retained when evidence is sparse or indirect."

        rows = [
            {
                "claim": prediction_claim,
                "evidence_ids": [item.get("id") for item in support[:3] if item.get("id")],
                "support_level": _alignment_support_level(support, contradiction),
                "uncertainty": _alignment_uncertainty(context, direct=direct, support=support, contradiction=contradiction),
                "evidence_type": "prediction_support",
            },
        ]
        if task_type == "ddi_prediction":
            rows.append(
                {
                    "claim": direct_claim,
                    "evidence_ids": [item.get("id") for item in direct[:3] if item.get("id")],
                    "support_level": "direct" if direct else "insufficient_direct_evidence",
                    "uncertainty": "low" if direct else "moderate_to_high",
                    "evidence_type": "direct_pair_evidence",
                }
            )
        else:
            rows.append(
                {
                    "claim": endpoint_claim,
                    "evidence_ids": [item.get("id") for item in endpoint_or_literature[:3] if item.get("id")],
                    "support_level": "endpoint_supported" if endpoint_or_literature else "insufficient_external_evidence",
                    "uncertainty": "moderate" if endpoint_or_literature else "moderate_to_high",
                    "evidence_type": "endpoint_literature_evidence",
                }
            )
        rows.extend(
            [
                {
                    "claim": mechanism_claim,
                    "evidence_ids": [item.get("id") for item in (mechanism or support or neutral)[:3] if item.get("id")],
                    "support_level": "mechanistic_indirect" if mechanism else "structural_rule_support" if support else "insufficient_evidence",
                    "uncertainty": "moderate" if mechanism or support else "high",
                    "evidence_type": "mechanism_evidence",
                },
                {
                    "claim": uncertainty_claim,
                    "evidence_ids": [item.get("id") for item in contradiction[:2] + neutral[:2] if item.get("id")],
                    "support_level": "uncertainty_statement",
                    "uncertainty": _alignment_uncertainty(context, direct=direct, support=support, contradiction=contradiction),
                    "evidence_type": "uncertainty",
                },
            ]
        )
        if quality.get("has_conflicting_evidence"):
            rows.append(
                {
                    "claim": "存在相互冲突的证据，需要人工复核。" if language == "zh" else "Conflicting evidence is present and should be manually reviewed.",
                    "evidence_ids": [item.get("id") for item in contradiction[:3] if item.get("id")],
                    "support_level": "conflicting_evidence",
                    "uncertainty": "high",
                    "evidence_type": "conflict",
                }
            )
        evidence_by_id = {str(item.get("id")): item for item in evidence_items if item.get("id")}
        for row in rows:
            row["evidence_ids"] = [item for item in row.get("evidence_ids", []) if item]
            row["evidence_count"] = len(row["evidence_ids"])
            row["citations"] = [
                citation
                for evidence_id in row["evidence_ids"]
                for citation in [self._citation_for_evidence(evidence_by_id.get(str(evidence_id)))]
                if citation
            ][:6]
        return rows

    def _evidence_citations(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        citations: List[Dict[str, Any]] = []
        for item in context.get("evidence_items", []) or []:
            citation = self._citation_for_evidence(item)
            if citation:
                citations.append(citation)
        for item in context.get("rag_context", []) or []:
            if item.get("source_path"):
                citations.append(
                    {
                        "label": f"RAG {item.get('id')} {item.get('chunk_id')}",
                        "source": "rag_context",
                        "source_path": item.get("source_path"),
                        "chunk_id": item.get("chunk_id"),
                        "retrieval_score": item.get("score"),
                        "snippet": str(item.get("snippet") or "")[:500],
                    }
                )
        deduped = []
        seen = set()
        for item in citations:
            key = (item.get("url"), item.get("source_path"), item.get("chunk_id"))
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)
        return deduped[:12]

    def _claim_level_citations(self, context: Dict[str, Any], claim_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        citation_rows = []
        evidence_by_id = {
            str(item.get("id")): item
            for item in context.get("evidence_items", []) or []
            if item.get("id")
        }
        for idx, row in enumerate(claim_rows, start=1):
            citations = list(row.get("citations") or [])
            if not citations:
                citations = [
                    citation
                    for evidence_id in row.get("evidence_ids", []) or []
                    for citation in [self._citation_for_evidence(evidence_by_id.get(str(evidence_id)))]
                    if citation
                ]
            citation_rows.append(
                {
                    "claim_id": f"C{idx}",
                    "claim": row.get("claim"),
                    "support_level": row.get("support_level"),
                    "uncertainty": row.get("uncertainty"),
                    "evidence_type": row.get("evidence_type"),
                    "evidence_ids": list(row.get("evidence_ids") or []),
                    "citations": citations[:8],
                    "citation_count": len(citations[:8]),
                }
            )
        return citation_rows

    def _citation_for_evidence(self, item: Optional[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
        if not item:
            return None
        source = str(item.get("source") or "source")
        label_parts = [source, str(item.get("id") or "")]
        pmid = item.get("pmid")
        pmids = item.get("pmids")
        if pmid:
            label_parts.append(f"PMID:{pmid}")
        elif isinstance(pmids, list) and pmids:
            label_parts.append(f"PMID:{pmids[0]}")
        if item.get("cid"):
            label_parts.append(f"CID:{item.get('cid')}")
        if item.get("setid"):
            label_parts.append(f"setid:{item.get('setid')}")
        if item.get("chunk_id"):
            label_parts.append(str(item.get("chunk_id")))
        has_pointer = any(item.get(key) for key in ["url", "source_path", "chunk_id", "pmid", "pmids", "cid", "setid"])
        if not has_pointer:
            return None
        return {
            "citation_id": str(item.get("id") or ""),
            "label": " ".join(part for part in label_parts if part),
            "source": source,
            "url": item.get("url"),
            "pmid": pmid,
            "pmids": pmids,
            "cid": item.get("cid"),
            "setid": item.get("setid"),
            "source_path": item.get("source_path"),
            "chunk_id": item.get("chunk_id"),
            "doc_id": item.get("doc_id"),
            "query": item.get("query"),
            "rank": item.get("rank"),
            "retrieval_score": item.get("retrieval_score"),
            "evidence_direction": item.get("direction"),
            "quality_score": item.get("quality_score"),
            "support_level": item.get("support_level"),
            "snippet": str(item.get("snippet") or item.get("claim") or "")[:500],
        }

    def _contribution_text(self, item: Mapping[str, Any], *, language: str) -> str:
        feature = item.get("feature") or "feature"
        direction = item.get("direction") or "neutral"
        weight = self._fmt_score(item.get("weight"))
        rationale = str(item.get("rationale") or "").strip()
        if language == "zh":
            return f"`{feature}` 为 {direction} 方向贡献，权重约 {weight}" + (f"（{rationale}）" if rationale else "")
        return f"`{feature}` contributed in the {direction} direction with weight about {weight}" + (f" ({rationale})" if rationale else "")

    def _evidence_line(self, item: Mapping[str, Any], *, language: str) -> str:
        source = item.get("source") or "unknown"
        claim = re.sub(r"\s+", " ", str(item.get("claim") or "")).strip()
        claim = claim[:220] + ("..." if len(claim) > 220 else "")
        pointer = item.get("pmid") or item.get("cid") or item.get("setid") or item.get("chunk_id") or item.get("source_path")
        if language == "zh":
            return f"{source}: {claim}" + (f"（{pointer}）" if pointer else "")
        return f"{source}: {claim}" + (f" ({pointer})" if pointer else "")

    def _next_steps(self, context: Dict[str, Any], *, language: str) -> List[str]:
        task_type = context.get("task_type")
        prediction = context.get("prediction") or {}
        zero_shot = prediction.get("model_type") == "zero_shot_rule_semantic_fusion"
        if language == "zh":
            if task_type == "ddi_prediction":
                steps = [
                    "优先核对 DailyMed/药品说明书中的 Drug Interactions、Warnings 和 Clinical Pharmacology 原文。",
                    "若用于研究结论，建议补充 PubMed 全文阅读和人工药理机制审查。",
                ]
                if zero_shot:
                    steps.append("若要做批量 DDI 分类性能评估，需要引入监督 DDI 数据并校准模型概率。")
                return steps
            steps = [
                "对关键候选分子复核原始文献、训练数据标签和适用域。",
                "对接近阈值或 zero-shot 的结果，建议优先做实验验证或用任务特异模型复算。",
            ]
            if (context.get("rag_summary") or {}).get("n_context"):
                steps.append("查看 RAG source pointers，确认检索片段是否真正支撑预测方向。")
            return steps

        if task_type == "ddi_prediction":
            steps = [
                "Review the primary DailyMed label sections for Drug Interactions, Warnings, and Clinical Pharmacology.",
                "For research conclusions, read the linked PubMed records and perform manual pharmacology review.",
            ]
            if zero_shot:
                steps.append("For batch DDI classification, add supervised DDI data and probability calibration.")
            return steps
        steps = [
            "Review the original literature, training labels, and applicability domain for high-priority molecules.",
            "For near-threshold or zero-shot outputs, prioritize experimental validation or task-specific model reruns.",
        ]
        if (context.get("rag_summary") or {}).get("n_context"):
            steps.append("Inspect the RAG source pointers to verify that retrieved snippets genuinely support the predicted direction.")
        return steps

    def _uncertainty_text(self, context: Dict[str, Any], *, language: str) -> str:
        prediction = context.get("prediction") or {}
        details = prediction.get("details") or {}
        warnings = context.get("warnings") or []
        scoring = context.get("evidence_grounded_scoring") or {}
        low_conf = any(item.get("name") == "low_confidence_prediction" and item.get("value") for item in context.get("confidence_signals", []))
        zero_shot = prediction.get("model_type") == "zero_shot_rule_semantic_fusion"
        pieces: List[str] = []
        if language == "zh":
            if zero_shot:
                pieces.append("当前使用的是 zero-shot 规则/语义融合分数，因此不能解释为经过监督校准的概率。")
            elif details.get("calibration_status") in {None, "not_evaluated", "not_calibrated"} and prediction.get("probability") is not None:
                pieces.append("当前监督模型概率尚未完成校准评估，概率数值需要谨慎解释。")
            if low_conf or details.get("confidence_level") == "low":
                pieces.append("预测置信度偏低或接近阈值，建议结合更多证据复核。")
            if scoring.get("recommendation"):
                pieces.append(f"证据层建议：{scoring.get('recommendation')}。")
            if warnings:
                pieces.append(f"workflow 产生了 {len(warnings)} 条 warning，需查看完整 JSON/trace。")
            return " ".join(pieces) or "未发现明显 workflow 异常，但仍建议把结果作为计算预测而非确定结论。"

        if zero_shot:
            pieces.append("The current score comes from zero-shot rule/semantic fusion and is not a supervised calibrated probability.")
        elif details.get("calibration_status") in {None, "not_evaluated", "not_calibrated"} and prediction.get("probability") is not None:
            pieces.append("The supervised-model probability has not yet been calibration-evaluated, so interpret its numeric value cautiously.")
        if low_conf or details.get("confidence_level") == "low":
            pieces.append("The prediction is low-confidence or near the decision threshold, so follow-up evidence review is recommended.")
        if scoring.get("recommendation"):
            pieces.append(f"The evidence layer recommends: {scoring.get('recommendation')}.")
        if warnings:
            pieces.append(f"The workflow emitted {len(warnings)} warning(s); inspect the JSON report and trace.")
        return " ".join(pieces) or "No obvious workflow failure was detected, but the result should still be treated as a computational prediction."

    def _prediction_meaning(self, prediction: Mapping[str, Any]) -> str:
        details = prediction.get("details") or {}
        label = prediction.get("label")
        if label == details.get("positive_label"):
            return str(details.get("positive_label_meaning") or label)
        if label == details.get("negative_label"):
            return str(details.get("negative_label_meaning") or label)
        if prediction.get("task_type") == "ddi_prediction":
            return "DDI risk/interaction" if label == 1 else "no DDI risk/interaction"
        return str(label)

    def _language(self, request: str) -> str:
        if self.language in {"zh", "en"}:
            return self.language
        return "zh" if re.search(r"[\u4e00-\u9fff]", request or "") else "en"

    def _fmt_score(self, value: Any) -> str:
        try:
            number = float(value)
        except Exception:
            return "n/a"
        if not math.isfinite(number):
            return "n/a"
        return f"{number:.3f}"

    def _pipe_md(self, value: Any) -> str:
        return str(value or "").replace("|", "\\|").replace("\n", " ").strip()


def _alignment_support_level(support: List[Mapping[str, Any]], contradiction: List[Mapping[str, Any]]) -> str:
    if contradiction and not support:
        return "conflicting_evidence"
    if support:
        best = max(float(item.get("quality_score") or 0.0) for item in support)
        if best >= 0.75:
            return "strong"
        if best >= 0.50:
            return "moderate"
        return "weak"
    return "insufficient_evidence"


def _alignment_uncertainty(
    context: Mapping[str, Any],
    *,
    direct: List[Mapping[str, Any]],
    support: List[Mapping[str, Any]],
    contradiction: List[Mapping[str, Any]],
) -> str:
    warnings = context.get("warnings") or []
    prediction = context.get("prediction") or {}
    details = prediction.get("details") if isinstance(prediction.get("details"), Mapping) else {}
    if contradiction or details.get("confidence_level") == "low":
        return "high"
    if not support or not direct or warnings:
        return "moderate"
    return "low"
