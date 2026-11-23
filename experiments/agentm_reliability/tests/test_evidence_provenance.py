from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import unittest

from experiments.agentm_reliability.common import write_json
from experiments.agentm_reliability.evidence_provenance import (
    audit_evidence_provenance,
    classify_evidence_layer,
    evidence_is_direct,
    evidence_is_relevant,
)
from experiments.agentm_reliability.offline_evaluator import evaluate_report_quality, run_all


QC = {
    "source": "paper_qc_curated",
    "source_path": "/local/tasks/DILI/aspirin_paper_qc.md",
    "chunk_id": "qc-1",
    "claim": "Aspirin DILI entity-endpoint retrieval quality-control grounding.",
    "compound_names": ["aspirin"],
    "endpoint": "dili",
    "endpoint_terms": ["dili", "drug-induced liver injury"],
    "evidence_scope": "compound_endpoint",
    "compound_specific": True,
    "endpoint_specific": True,
    "metadata": {"evidence_type": "compound_endpoint_grounding"},
}


class EvidenceProvenanceTests(unittest.TestCase):
    def test_reaction_profile_schema_requires_profile_not_numeric_prediction(self) -> None:
        context = {
            "task_type": "reaction_profile",
            "reaction_profile": {"reaction_class": "Suzuki coupling"},
            "evidence": [{"source": "orderly_condition", "source_path": "row-1"}],
        }
        report = {
            "request": "CCBr.B(O)O>>CC",
            "task_type": "reaction_profile",
            "reaction_profile": {"reaction_class": "Suzuki coupling"},
            "explanation": {"summary": "Condition profile assembled from retrieved examples."},
            "tool_calls": [{"tool": "retrieve_orderly_conditions", "status": "completed"}],
            "retrieval_trace": [{"provider": "RetrieveAgent", "hits": 1}],
            "objective_status": {},
            "evidence": [{"source": "orderly_condition", "source_path": "row-1"}],
            "citation_map": [],
        }

        quality = evaluate_report_quality(case={}, context=context, report=report)

        self.assertEqual(quality["report_schema_completeness"], 1.0)
        self.assertEqual(json.loads(quality["report_schema_missing_json"]), [])

    def test_qc_grounding_is_relevant_alignment_but_never_external_or_direct(self) -> None:
        self.assertEqual(classify_evidence_layer(QC), "benchmark_qc_grounding")
        self.assertTrue(evidence_is_relevant(QC))
        self.assertFalse(evidence_is_direct(QC))
        audit = audit_evidence_provenance({"evidence": [QC], "retrieval_trace": []})
        self.assertTrue(audit["qc_only"])
        self.assertFalse(audit["external_evidence_coverage"])
        self.assertFalse(audit["direct_evidence_coverage"])

    def test_authoritative_label_is_external_but_not_direct_without_endpoint_content(self) -> None:
        label = {
            "source": "dailymed",
            "source_path": "https://dailymed.nlm.nih.gov/drugInfo.cfm?setid=1",
            "title": "Prochlorperazine tablet",
            "text": "Prochlorperazine is indicated for severe nausea and vomiting.",
            "compound_names": ["prochlorperazine"],
            "endpoint": "bioavailability",
            "endpoint_terms": ["bioavailability", "systemic exposure"],
            "evidence_scope": "compound_endpoint",
            "compound_specific": True,
            "endpoint_specific": True,
            "direct_evidence": True,
        }
        self.assertEqual(classify_evidence_layer(label), "authoritative_drug_label")
        self.assertFalse(evidence_is_direct(label))
        self.assertFalse(evidence_is_relevant(label))

    def test_dailymed_task_header_cannot_satisfy_endpoint_directness(self) -> None:
        label = {
            "source": "dailymed",
            "source_path": "https://dailymed.nlm.nih.gov/drugInfo.cfm?setid=1",
            "title": "Prochlorperazine tablet",
            "text": (
                "- source: DailyMed - task: Bioavailability_Ma - query: prochlorperazine "
                "- setid: 1 - url: https://dailymed.nlm.nih.gov/drugInfo.cfm?setid=1 "
                "- retrieved_at: 2026-05-31T09:48:27+00:00 ### Selected label sections "
                "#### INDICATIONS Prochlorperazine is indicated for severe nausea and vomiting."
            ),
            "task": "Bioavailability_Ma",
            "query": "prochlorperazine",
            "setid": "1",
            "url": "https://dailymed.nlm.nih.gov/drugInfo.cfm?setid=1",
            "retrieved_at": "2026-05-31T09:48:27+00:00",
            "compound_names": ["prochlorperazine"],
            "endpoint": "bioavailability",
            "endpoint_terms": ["bioavailability", "oral bioavailability", "systemic exposure"],
            "evidence_scope": "compound_endpoint",
            "compound_specific": True,
            "endpoint_specific": True,
        }
        self.assertFalse(evidence_is_direct(label))
        audit = audit_evidence_provenance({"evidence": [label], "retrieval_trace": []})
        self.assertTrue(audit["external_evidence_coverage"])
        self.assertFalse(audit["direct_evidence_coverage"])

    def test_dailymed_source_body_can_satisfy_endpoint_directness(self) -> None:
        label = {
            "source": "dailymed",
            "source_path": "https://dailymed.nlm.nih.gov/drugInfo.cfm?setid=2",
            "title": "Example Drug tablet",
            "text": (
                "- source: DailyMed - task: Bioavailability_Ma - query: example drug "
                "- setid: 2 - url: https://dailymed.nlm.nih.gov/drugInfo.cfm?setid=2 "
                "### Selected label sections #### CLINICAL PHARMACOLOGY "
                "Example Drug has an absolute oral bioavailability of 72 percent."
            ),
            "task": "Bioavailability_Ma",
            "query": "example drug",
            "setid": "2",
            "url": "https://dailymed.nlm.nih.gov/drugInfo.cfm?setid=2",
            "compound_names": ["example drug"],
            "endpoint": "bioavailability",
            "endpoint_terms": ["bioavailability", "oral bioavailability", "systemic exposure"],
            "evidence_scope": "compound_endpoint",
            "compound_specific": True,
            "endpoint_specific": True,
        }
        self.assertTrue(evidence_is_direct(label))

    def test_controlled_peer_named_after_qc_source_remains_injection(self) -> None:
        injected = {
            **QC,
            "source": "controlled_conflict_peer::paper_qc_curated",
            "controlled_injection": True,
            "source_path": "controlled://conflict-peer",
        }

        self.assertEqual(classify_evidence_layer(injected), "controlled_injection")
        audit = audit_evidence_provenance({"evidence": [injected], "retrieval_trace": []})
        self.assertEqual(audit["qc_grounding_count"], 0)
        self.assertFalse(audit["qc_only"])
        self.assertFalse(audit["direct_evidence_coverage"])

    def test_target_specific_pubmed_record_is_external_and_direct(self) -> None:
        paper = {
            "source": "pubmed",
            "source_path": "https://pubmed.ncbi.nlm.nih.gov/123/",
            "pmid": "123",
            "title": "Aspirin and drug-induced liver injury",
            "text": "This study directly evaluates aspirin-associated drug-induced liver injury (DILI).",
            "compound_names": ["aspirin"],
            "endpoint": "dili",
            "endpoint_terms": ["dili", "drug-induced liver injury"],
            "evidence_scope": "compound_endpoint",
            "compound_specific": True,
            "endpoint_specific": True,
        }
        self.assertEqual(classify_evidence_layer(paper), "external_literature")
        self.assertTrue(evidence_is_direct(paper))
        audit = audit_evidence_provenance({"evidence": [QC, paper], "retrieval_trace": []})
        self.assertTrue(audit["external_evidence_coverage"])
        self.assertTrue(audit["direct_evidence_coverage"])
        self.assertFalse(audit["qc_only"])

    def test_report_quality_audit_ignores_internal_verifier_pass(self) -> None:
        context = {
            "metadata": {},
            "task_type": "admet_prediction",
            "molecules": [{"original_input": "CCO", "smiles": "CCO"}],
            "predictions": [{"label": 1, "model_type": "random_forest"}],
            "evidence": [QC],
            "citation_map": [],
            "reasoning": {"structured_explanation": "The model predicts a risk signal; direct evidence is insufficient."},
            "warnings": [],
            "tool_calls": [{"tool": "predict_single", "status": "completed"}],
            "retrieval_trace": [{"retrieved_context": [QC]}],
        }
        report = {
            "request": "CCO",
            "task_type": "admet_prediction",
            "molecules": context["molecules"],
            "predictions": context["predictions"],
            "evidence": context["evidence"],
            "explanation": "The model predicts a risk signal; direct evidence is insufficient.",
            "tool_calls": context["tool_calls"],
            "retrieval_trace": context["retrieval_trace"],
            "citation_map": [],
            "objective_status": {},
        }
        case = {"request": "CCO", "task_type": "admet_prediction"}
        context["verifier_report"] = {"summary": {"status": "pass"}}
        passed = evaluate_report_quality(case=case, context=context, report=report)
        context["verifier_report"] = {"summary": {"status": "fail"}}
        failed = evaluate_report_quality(case=case, context=context, report=report)
        self.assertEqual(passed, failed)

    def test_report_quality_audits_final_citation_map_and_uncited_claims(self) -> None:
        context = {"evidence": [QC], "metadata": {}, "task_type": "general_drug_profile"}
        report = {
            "request": "aspirin profile",
            "task_type": "general_drug_profile",
            "explanation": {
                "claims": [
                    {"claim_id": "linked", "claim": "QC grounding was retained."},
                    {"claim_id": "uncited", "claim": "A second factual statement."},
                ]
            },
            "citation_map": [{"claim_id": "linked", "chunk_id": "qc-1"}],
        }

        quality = evaluate_report_quality(case={}, context=context, report=report)

        self.assertEqual(quality["explicit_claim_link_count"], 2)
        self.assertEqual(quality["broken_or_missing_citation_link_count"], 1)
        self.assertEqual(quality["structural_unsupported_link_rate"], 0.5)
        self.assertIsNone(quality["unsupported_claim_rate"])
        self.assertEqual(
            quality["unsupported_claim_metric_basis"],
            "not_estimable_without_claim_support_gold",
        )

        gold_quality = evaluate_report_quality(
            case={},
            context=context,
            report=report,
            group_evaluation={
                "citation_tp": 1,
                "citation_fp": 1,
                "citation_precision": 0.5,
                "citation_precision_applicable": True,
            },
        )
        self.assertEqual(gold_quality["claim_count_requiring_support"], 2)
        self.assertEqual(gold_quality["unsupported_claim_count_independent"], 1)
        self.assertEqual(gold_quality["unsupported_claim_rate"], 0.5)
        self.assertEqual(gold_quality["unsupported_claim_metric_basis"], "stress_case_gold_citation_map")

    def test_offline_downstream_writes_quality_and_provenance_tables(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "out"
            raw = output / "raw_runs" / "run-1"
            context = {
                "metadata": {},
                "task_type": "admet_prediction",
                "molecules": [{"original_input": "CCO", "smiles": "CCO"}],
                "predictions": [{"label": 1, "score": 0.7, "model_type": "random_forest"}],
                "evidence": [QC],
                "citation_map": [],
                "reasoning": {"structured_explanation": "The model predicts a signal; external direct evidence is insufficient."},
                "warnings": [],
                "tool_calls": [
                    {"agent": "GaugeAgent", "tool": "predict_single", "status": "completed"},
                    {"agent": "RetrieveAgent", "tool": "retrieve_elasticsearch_hybrid", "status": "completed"},
                ],
                "retrieval_trace": [
                    {
                        "provider": "elasticsearch_hybrid",
                        "backend": "elasticsearch_hybrid_bm25_dense",
                        "retrieved_context": [QC],
                    }
                ],
            }
            report = {
                "request": "CCO",
                "task_type": "admet_prediction",
                "molecules": context["molecules"],
                "predictions": context["predictions"],
                "evidence": context["evidence"],
                "explanation": context["reasoning"]["structured_explanation"],
                "tool_calls": context["tool_calls"],
                "retrieval_trace": context["retrieval_trace"],
                "citation_map": [],
                "objective_status": {},
            }
            write_json(
                raw / "run_record.json",
                {
                    "run_id": "run-1",
                    "case_id": "case-1",
                    "dataset_group": "clean",
                    "task_type": "admet_prediction",
                    "orchestrator": "heuristic",
                    "reflection_enabled": False,
                    "seed": 2026,
                    "repeat_index": 0,
                    "completed": True,
                    "runtime_retrieval_backend": "elasticsearch_hybrid_bm25_dense",
                    "retrieval_call_count": 1,
                },
                overwrite=False,
            )
            write_json(
                raw / "job.json",
                {"case": {"case_id": "case-1", "request": "CCO", "task_type": "admet_prediction"}},
                overwrite=False,
            )
            write_json(raw / "runtime" / "global_context.json", context, overwrite=False)
            write_json(raw / "runtime" / "multi_agent_report.json", report, overwrite=False)
            write_json(raw / "runtime" / "retrieval_trace.json", context["retrieval_trace"], overwrite=False)

            counts = run_all(output)
            self.assertEqual(counts["report_quality"], 1)
            with (output / "metrics" / "report_quality_case_level.csv").open(encoding="utf-8") as handle:
                quality = list(csv.DictReader(handle))
            with (output / "metrics" / "evidence_provenance_case_level.csv").open(encoding="utf-8") as handle:
                provenance = list(csv.DictReader(handle))
            self.assertEqual(len(quality), 1)
            self.assertEqual(len(provenance), 1)
            self.assertEqual(quality[0]["qc_only"], "True")
            self.assertEqual(quality[0]["external_evidence_coverage"], "False")


if __name__ == "__main__":
    unittest.main()
