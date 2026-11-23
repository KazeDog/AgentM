from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

from agentm.runtime.guards.verifier import (
    repair_evidence_record_integrity,
    repair_prediction_record_integrity,
    verify_citations,
)
from agentm.runtime.retrieval.hybrid import HybridRetrievalResult
from agentm.runtime.objectives import evaluate_task_objectives
from agentm.runtime.tools.scientific import ScientificToolRuntime
from experiments.agentm_reliability.common import source_tree_hash, write_json
from experiments.agentm_reliability.offline_evaluator import evaluate_report_quality
from experiments.agentm_reliability.multi_turn_runner import run_worker as run_multi_turn_worker
from experiments.agentm_reliability.run_matrix import run_worker
from experiments.agentm_reliability.runtime_guards import (
    ApiBudget,
    ApiBudgetExceeded,
    RealRAGPreflightError,
    api_budget_guard,
    effective_api_budget,
    run_real_rag_preflight,
)
from experiments.agentm_reliability.validate_real_rag_run import validate_raw_run
from experiments.agentm_reliability.validation import run_invariant_violations
from experiments.agentm_reliability.validation import runtime_retrieval_backend


class _FakeRetriever:
    def __init__(self, result: HybridRetrievalResult):
        self.result = result
        self.request_timeout_seconds = 60.0
        self.max_retries = 1

    def retrieve(self, *_args, **_kwargs):
        return self.result


class _FakeReactionRuntime:
    def resolve_resources(self):
        return {"all_required_present": True}

    def parse_reaction(self, _args, context):
        context["reaction"] = {"reaction_smiles": context["reaction_request"]}
        return SimpleNamespace(status="completed", error=None, observation={})

    def _retrieve(self, context, source):
        evidence = [{"citation_id": source, "source": source, "claim": "evidence"}]
        context.setdefault("evidence", []).extend(evidence)
        context.setdefault("retrieval_trace", []).append(
            {
                "provider": "RetrieveAgent",
                "backend": "local_reaction_data",
                "evidence_items": evidence,
                "warnings": [],
            }
        )
        return SimpleNamespace(
            status="completed",
            error=None,
            observation={"n_evidence_items": 1, "evidence_items": evidence},
        )

    def retrieve_mit_template_evidence(self, _args, context):
        return self._retrieve(context, "mit_reaction_template")

    def retrieve_orderly_condition_evidence(self, _args, context):
        return self._retrieve(context, "orderly_condition")


def _result(*, valid: bool) -> HybridRetrievalResult:
    context = [{"chunk_id": "chunk-1"}] if valid else []
    evidence = [{"chunk_id": "chunk-1", "claim": "evidence"}] if valid else []
    return HybridRetrievalResult(
        provider="elasticsearch_hybrid" if valid else "hybrid_retriever",
        backend="elasticsearch_hybrid_bm25_dense" if valid else "elasticsearch",
        query="query",
        retrieved_context=context,
        evidence_items=evidence,
        warnings=[] if valid else ["connection timed out"],
        metadata={},
    )


class RuntimeGuardTests(unittest.TestCase):
    def test_reaction_entity_mismatch_is_restored_from_retrieval_trace(self) -> None:
        trusted = {
            "citation_id": "orderly_condition_2",
            "source": "orderly_condition",
            "source_path": "source://orderly",
            "claim": "Similar reaction condition context.",
            "reaction_smiles": "CC>>CO",
            "reaction_specific": True,
            "score": 0.7,
        }
        injected = {
            **trusted,
            "claim": "unrelated comparator entity and non-target endpoint",
            "snippet": "unrelated comparator entity and non-target endpoint",
            "endpoint": "non_target_endpoint",
            "compound_specific": False,
            "endpoint_specific": False,
        }
        context = {
            "task_type": "reaction_condition_retrieval",
            "evidence": [injected],
            "retrieval_trace": [{"backend": "local_reaction_data", "evidence_items": [trusted]}],
            "tool_calls": [],
            "warnings": [],
            "metadata": {},
        }

        audit = repair_evidence_record_integrity(context)

        self.assertTrue(audit["detected"])
        self.assertTrue(audit["repaired"])
        self.assertEqual(context["evidence"], [trusted])
        self.assertIn(
            "evidence record entity mismatch or endpoint mismatch detected and repaired from retrieval trace",
            context["warnings"],
        )

    def test_explicit_background_only_record_is_classified_as_weak_evidence(self) -> None:
        trusted = {
            "citation_id": "mit_template_5",
            "source": "mit_reaction_template",
            "claim": "Reaction-specific template evidence.",
            "reaction_specific": True,
            "score": 0.75,
        }
        background = "General background information; it does not directly test the requested entity, endpoint, drug pair, or reaction."
        injected = {
            **trusted,
            "claim": background,
            "snippet": background,
            "evidence_scope": "background_only",
            "compound_specific": False,
            "endpoint_specific": False,
            "reaction_specific": False,
            "value": {
                "snippet": background,
                "compound_specific": False,
                "endpoint_specific": False,
            },
        }
        context = {
            "task_type": "reaction_yield_prediction",
            "evidence": [injected],
            "retrieval_trace": [{"backend": "local_reaction_data", "evidence_items": [trusted]}],
            "tool_calls": [],
            "warnings": [],
            "metadata": {},
        }

        audit = repair_evidence_record_integrity(context)

        self.assertTrue(audit["detected"])
        self.assertTrue(audit["repaired"])
        self.assertEqual(audit["records"][0]["category"], "irrelevant_or_weak_evidence")
        self.assertEqual(context["evidence"], [trusted])
        self.assertIn(
            "irrelevant or weak evidence detected and repaired from retrieval trace; "
            "the resulting conclusion should be interpreted cautiously",
            context["warnings"],
        )

    def test_evidence_integrity_guard_preserves_new_conflicting_peer(self) -> None:
        trusted = {
            "citation_id": "cite_1",
            "source": "curated",
            "claim": "Target evidence.",
        }
        context = {
            "task_type": "admet_prediction",
            "evidence": [
                {**trusted, "evidence_direction": "support"},
                {
                    "citation_id": "injected_conflict_peer",
                    "source": "controlled_conflict_peer::curated",
                    "claim": "Comparable evidence reports the opposite direction.",
                    "evidence_direction": "contradiction",
                },
            ],
            "retrieval_trace": [{"evidence_items": [trusted]}],
            "tool_calls": [],
            "warnings": [],
            "metadata": {},
        }

        audit = repair_evidence_record_integrity(context)

        self.assertFalse(audit["detected"])
        self.assertEqual(len(context["evidence"]), 2)
        self.assertEqual(context["evidence"][1]["citation_id"], "injected_conflict_peer")

    def test_reaction_verifier_does_not_promote_explicit_mismatch(self) -> None:
        context = {
            "task_type": "reaction_condition_retrieval",
            "evidence": [
                {
                    "citation_id": "bad",
                    "source": "orderly_condition",
                    "source_path": "source://orderly",
                    "claim": "unrelated comparator entity and non-target endpoint",
                    "reaction_specific": True,
                    "compound_specific": False,
                    "endpoint_specific": False,
                }
            ],
        }

        audit = verify_citations(context)

        self.assertTrue(audit["citation_grounded"])
        self.assertFalse(audit["evidence_relevance_grounded"])
        self.assertFalse(audit["compound_specific_grounded"])
        self.assertFalse(audit["endpoint_specific_grounded"])
        self.assertEqual(audit["n_low_relevance"], 1)
        self.assertEqual(audit["citation_map"][0]["citation_id"], "bad")

    def test_prediction_overclaim_wrapper_is_repaired_only_from_grounded_trace(self) -> None:
        prediction = {
            "task_type": "admet_prediction",
            "dataset_name": "AMES",
            "label": 1,
            "score": 0.53,
            "probability": 0.53,
            "model_type": "random_forest",
        }
        context = {
            "task_type": "admet_prediction",
            "predictions": [{"value": dict(prediction), "claim_status": "experimentally_confirmed"}],
            "tool_calls": [
                {
                    "agent": "GaugeAgent",
                    "tool": "predict_single",
                    "status": "completed",
                    "observation": {"prediction": dict(prediction)},
                }
            ],
            "warnings": [],
            "metadata": {},
        }

        audit = repair_prediction_record_integrity(context)

        self.assertTrue(audit["detected"])
        self.assertTrue(audit["repaired"])
        self.assertEqual(context["predictions"], [prediction])
        self.assertIn("prediction record missing field or overclaim detected", context["warnings"])
        self.assertIn("prediction record repaired from grounded model-tool output", context["warnings"])

    def test_prediction_value_conflict_is_never_silently_repaired(self) -> None:
        trusted = {
            "task_type": "admet_prediction",
            "dataset_name": "AMES",
            "label": 1,
            "score": 0.8,
            "model_type": "random_forest",
        }
        changed = {**trusted, "label": 0, "claim_status": "experimentally_confirmed"}
        context = {
            "task_type": "admet_prediction",
            "predictions": [changed],
            "tool_calls": [
                {
                    "agent": "GaugeAgent",
                    "tool": "predict_single",
                    "status": "completed",
                    "observation": {"prediction": trusted},
                }
            ],
            "warnings": [],
            "metadata": {},
        }

        audit = repair_prediction_record_integrity(context)

        self.assertTrue(audit["detected"])
        self.assertFalse(audit["repaired"])
        self.assertEqual(context["predictions"], [changed])

    def test_no_direct_evidence_report_uses_caution_not_strong_support(self) -> None:
        runtime = ScientificToolRuntime.__new__(ScientificToolRuntime)
        qc_record = {
            "source": "paper_qc_curated",
            "chunk_id": "tasks/DrugBank/cyclosporin_loxoprofen_ddi_paper_qc.md::1",
            "source_path": "/corpus/paper_qc/cyclosporin_loxoprofen_ddi.md",
            "claim": "Entity-endpoint QC grounding without an experimental DDI label.",
            "citation_id": "cite_1",
            "evidence_direction": "neutral",
        }
        context = {
            "request": "Cyclosporine + Loxoprofen DDI",
            "task_type": "ddi_prediction",
            "dataset_name": "DrugBank",
            "metadata": {
                "retrieval_outcome": "no_direct_evidence",
                "direct_evidence_available": False,
                "qc_grounding_only": True,
            },
            "molecules": [{"name": "Cyclosporine"}, {"name": "Loxoprofen"}],
            "predictions": [
                {
                    "task_type": "ddi_prediction",
                    "dataset_name": "DrugBank",
                    "label": 0,
                    "score": 0.1,
                    "probability": 0.1,
                    "model_type": "sgd_classifier",
                    "details": {
                        "positive_label": 1,
                        "negative_label": 0,
                        "negative_label_meaning": "no DDI risk/interaction",
                    },
                }
            ],
            "confidence": {"score": 0.1, "margin": 0.4, "low_confidence": False},
            "evidence": [qc_record],
            "citation_map": [
                {
                    "claim_id": "cite_1",
                    "chunk_id": qc_record["chunk_id"],
                    "source_path": qc_record["source_path"],
                }
            ],
        }

        explanation = runtime._deterministic_explanation(context)
        lowered = explanation.lower()

        self.assertIn("no direct evidence", lowered)
        self.assertIn("interpreted cautiously", lowered)
        for phrase in ("fairly clear support", "clear support", "strong support", "well supported"):
            self.assertNotIn(phrase, lowered)
        quality = evaluate_report_quality(
            case={},
            context=context,
            report={"explanation": explanation, "citation_map": context["citation_map"]},
        )
        self.assertTrue(quality["prediction_fact_separation_accuracy"])
        self.assertTrue(quality["uncertainty_expression_accuracy"])
        self.assertFalse(quality["forced_conclusion_rate"])

    def test_strict_real_rag_never_calls_local_fallback(self) -> None:
        runtime = ScientificToolRuntime.__new__(ScientificToolRuntime)
        runtime.hybrid_retriever = _FakeRetriever(_result(valid=False))
        runtime.retrieval_backend = "es"
        runtime.rag_top_k = 3
        runtime.require_real_rag = True
        runtime.min_evidence = 1
        fallback_calls: list[str] = []
        runtime._run_local_rag_as_hybrid_fallback = lambda *_args, **_kwargs: fallback_calls.append("called")
        context = {
            "request": "aspirin",
            "task_type": "admet_prediction",
            "dataset_name": "DILI",
            "metadata": {"require_real_rag": True, "enable_fallbacks": False, "min_evidence": 1},
            "molecules": [],
        }
        result = runtime.retrieve_elasticsearch_hybrid({}, context)
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.error, "strict_real_rag_dense_hybrid_result_required")
        self.assertEqual(fallback_calls, [])

        rejected = runtime.retrieve_elasticsearch_hybrid({"backend": "local"}, context)
        self.assertEqual(rejected.status, "failed")
        self.assertEqual(rejected.error, "strict_real_rag_elasticsearch_backend_required")
        self.assertEqual(fallback_calls, [])

        direct_local = runtime.retrieve_local_rag({}, context)
        self.assertEqual(direct_local.status, "failed")
        self.assertEqual(direct_local.error, "strict_real_rag_local_tool_disabled")
        self.assertEqual(fallback_calls, [])

    def test_strict_real_rag_rejects_bm25_only_evidence(self) -> None:
        runtime = ScientificToolRuntime.__new__(ScientificToolRuntime)
        runtime.hybrid_retriever = _FakeRetriever(
            HybridRetrievalResult(
                provider="elasticsearch_hybrid",
                backend="elasticsearch_bm25",
                query="query",
                retrieved_context=[{"chunk_id": "chunk-1"}],
                evidence_items=[{"chunk_id": "chunk-1", "claim": "evidence"}],
                warnings=["dense_knn_unavailable"],
                metadata={},
            )
        )
        runtime.retrieval_backend = "es"
        runtime.rag_top_k = 3
        runtime.require_real_rag = True
        runtime.min_evidence = 1
        context = {
            "request": "aspirin",
            "task_type": "admet_prediction",
            "dataset_name": "DILI",
            "metadata": {"require_real_rag": True, "enable_fallbacks": False, "min_evidence": 1},
            "molecules": [],
        }
        result = runtime.retrieve_elasticsearch_hybrid({}, context)
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.error, "strict_real_rag_dense_hybrid_result_required")
        self.assertEqual(context.get("evidence", []), [])
        self.assertNotIn("retrieval_backend_used", context["metadata"])

    def test_verified_empty_dense_result_is_recorded_without_fallback(self) -> None:
        runtime = ScientificToolRuntime.__new__(ScientificToolRuntime)
        runtime.hybrid_retriever = _FakeRetriever(
            HybridRetrievalResult(
                provider="elasticsearch_hybrid",
                backend="elasticsearch_hybrid_bm25_dense",
                query="query",
                retrieved_context=[],
                evidence_items=[],
                warnings=[],
                metadata={"dense_enabled": True, "dense_status": "completed"},
            )
        )
        runtime.retrieval_backend = "es"
        runtime.rag_top_k = 3
        runtime.require_real_rag = True
        runtime.min_evidence = 1
        context = {
            "request": "novel held-out molecule",
            "task_type": "admet_prediction",
            "dataset_name": "DILI",
            "metadata": {"require_real_rag": True, "enable_fallbacks": False, "min_evidence": 1},
            "molecules": [],
        }
        result = runtime.retrieve_elasticsearch_hybrid({}, context)
        self.assertEqual(result.status, "completed")
        self.assertFalse(result.error)
        self.assertEqual(result.observation["retrieval_outcome"], "no_direct_evidence")
        self.assertEqual(context["metadata"]["retrieval_backend_used"], "elasticsearch_hybrid_bm25_dense")
        self.assertEqual(context["metadata"]["retrieval_outcome"], "no_direct_evidence")
        self.assertFalse(context["metadata"]["direct_evidence_available"])
        self.assertEqual(context.get("evidence", []), [])
        self.assertIn("no_direct_evidence_for_target", context["warnings"])

    def test_verified_qc_grounding_is_not_promoted_to_direct_evidence(self) -> None:
        qc_record = {
            "source": "paper_qc_curated",
            "chunk_id": "tasks/HIA_Hou/metformin_hia_paper_qc.md::1",
            "source_path": "/corpus/paper_qc/metformin_hia.md",
            "claim": "Metformin HIA entity-endpoint grounding; no experimental endpoint label.",
            "value": {
                "source": "paper_qc_curated",
                "chunk_id": "tasks/HIA_Hou/metformin_hia_paper_qc.md::1",
                "source_path": "/corpus/paper_qc/metformin_hia.md",
                "text": "Metformin HIA entity-endpoint grounding; this does not assert an experimental endpoint label.",
                "metadata": {
                    "task_type": "admet_prediction",
                    "dataset_name": "HIA_Hou",
                    "evidence_type": "compound_endpoint_grounding",
                },
                "compound_names": ["metformin"],
                "endpoint": "hia",
                "endpoint_terms": ["hia", "human intestinal absorption"],
                "evidence_scope": "compound_endpoint",
                "compound_specific": True,
                "endpoint_specific": True,
            },
        }
        runtime = ScientificToolRuntime.__new__(ScientificToolRuntime)
        runtime.hybrid_retriever = _FakeRetriever(
            HybridRetrievalResult(
                provider="elasticsearch_hybrid",
                backend="elasticsearch_hybrid_bm25_dense",
                query="metformin HIA",
                retrieved_context=[qc_record["value"]],
                evidence_items=[qc_record],
                warnings=[],
                metadata={"dense_enabled": True, "dense_status": "completed"},
            )
        )
        runtime.retrieval_backend = "es"
        runtime.rag_top_k = 3
        runtime.require_real_rag = True
        runtime.min_evidence = 1
        context = {
            "request": "metformin",
            "task_type": "admet_prediction",
            "dataset_name": "HIA_Hou",
            "metadata": {"require_real_rag": True, "enable_fallbacks": False, "min_evidence": 1},
            "molecules": [{"name": "Metformin", "canonical_smiles": "CN(C)C(=N)N"}],
        }
        result = runtime.retrieve_elasticsearch_hybrid({}, context)
        self.assertEqual(result.status, "completed")
        self.assertEqual(context["metadata"]["retrieval_outcome"], "no_direct_evidence")
        self.assertTrue(context["metadata"]["real_rag_execution_valid"])
        self.assertTrue(context["metadata"]["qc_grounding_only"])
        self.assertFalse(context["metadata"]["direct_evidence_available"])
        self.assertEqual(len(context.get("evidence", [])), 1)
        self.assertIn("no_direct_evidence_for_target", context["warnings"])

    def test_dailymed_task_header_is_external_background_not_direct_evidence(self) -> None:
        chunk = {
            "source": "dailymed",
            "chunk_id": "tasks/Bioavailability_Ma/dailymed_labels.md::1",
            "source_path": "https://dailymed.nlm.nih.gov/drugInfo.cfm?setid=1",
            "title": "Prochlorperazine tablet",
            "text": (
                "- source: DailyMed - task: Bioavailability_Ma - query: prochlorperazine "
                "- setid: 1 - url: https://dailymed.nlm.nih.gov/drugInfo.cfm?setid=1 "
                "- retrieved_at: 2026-05-31T09:48:27+00:00 ### Selected label sections "
                "#### INDICATIONS Prochlorperazine is indicated for severe nausea and vomiting."
            ),
            "compound_names": ["prochlorperazine"],
            "endpoint": "bioavailability",
            "endpoint_terms": ["bioavailability", "oral bioavailability", "systemic exposure"],
            "evidence_scope": "compound_endpoint",
            "compound_specific": True,
            "endpoint_specific": True,
        }
        evidence = {
            "source": "dailymed",
            "claim": "DailyMed Bioavailability_Ma retrieval record",
            "value": chunk,
            "chunk_id": chunk["chunk_id"],
            "source_path": chunk["source_path"],
            "citation_id": "cite_1",
            "evidence_direction": "neutral",
        }
        runtime = ScientificToolRuntime.__new__(ScientificToolRuntime)
        runtime.hybrid_retriever = _FakeRetriever(
            HybridRetrievalResult(
                provider="elasticsearch_hybrid",
                backend="elasticsearch_hybrid_bm25_dense",
                query="prochlorperazine bioavailability",
                retrieved_context=[chunk],
                evidence_items=[evidence],
                warnings=[],
                metadata={"dense_enabled": True, "dense_status": "completed"},
            )
        )
        runtime.retrieval_backend = "es"
        runtime.rag_top_k = 3
        runtime.require_real_rag = True
        runtime.min_evidence = 1
        context = {
            "request": "prochlorperazine",
            "task_type": "admet_prediction",
            "dataset_name": "Bioavailability_Ma",
            "metadata": {"require_real_rag": True, "enable_fallbacks": False, "min_evidence": 1},
            "molecules": [{"name": "Prochlorperazine"}],
        }
        result = runtime.retrieve_elasticsearch_hybrid({}, context)
        self.assertEqual(result.status, "completed")
        self.assertTrue(context["metadata"]["real_rag_execution_valid"])
        self.assertEqual(context["metadata"]["retrieval_outcome"], "no_direct_evidence")
        self.assertTrue(context["metadata"]["external_evidence_available"])
        self.assertFalse(context["metadata"]["qc_grounding_only"])
        self.assertFalse(context["metadata"]["direct_evidence_available"])
        self.assertEqual(len(context.get("evidence", [])), 1)
        self.assertIn("no_direct_evidence_for_target", context["warnings"])

    def test_objectives_separate_real_rag_execution_from_direct_evidence(self) -> None:
        context = {
            "metadata": {
                "min_evidence": 1,
                "require_real_model": True,
                "require_real_rag": True,
                "real_rag_execution_valid": True,
                "retrieval_backend_used": "elasticsearch_hybrid_bm25_dense",
                "retrieval_outcome": "no_direct_evidence",
                "direct_evidence_available": False,
                "enable_fallbacks": False,
                "enable_reflection": False,
            },
            "predictions": [{"model_type": "random_forest"}],
            "confidence": {"low_confidence": False},
            "evidence": [{"citation_id": "cite_1", "source": "paper_qc_curated"}],
            "verifier_report": {
                "prediction": {"prediction_grounded": True},
                "citations": {
                    "citation_grounded": True,
                    "evidence_relevance_grounded": True,
                    "compound_specific_grounded": True,
                    "endpoint_specific_grounded": True,
                    "n_unsupported": 0,
                },
                "schema": {"schema_complete": True},
            },
        }
        status = evaluate_task_objectives(context)
        self.assertEqual(status["status"], "warning")
        self.assertTrue(status["checks"]["real_rag_required"])
        self.assertFalse(status["checks"]["direct_evidence_available"])
        self.assertNotIn("real_rag_required", status["missing_objectives"])
        self.assertIn("direct_evidence_available", status["missing_objectives"])

    def test_preflight_accepts_only_exact_nonempty_dense_hybrid(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "preflight.json"
            retriever = _FakeRetriever(_result(valid=True))
            result = run_real_rag_preflight(
                retriever,
                case={"request": "aspirin", "task_type": "admet_prediction"},
                runtime={
                    "require_real_rag": True,
                    "retrieval_backend": "es",
                    "es_request_timeout_seconds": 12,
                    "es_max_retries": 0,
                },
                output_path=output,
                label="test",
            )
            self.assertTrue(result["passed"])
            self.assertEqual(retriever.request_timeout_seconds, 12.0)
            self.assertEqual(retriever.max_retries, 0)
            self.assertTrue(output.exists())

    def test_preflight_fails_closed_without_llm(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "preflight.json"
            with self.assertRaisesRegex(RealRAGPreflightError, "real_rag_preflight_failed"):
                run_real_rag_preflight(
                    _FakeRetriever(_result(valid=False)),
                    case={"request": "aspirin"},
                    runtime={"require_real_rag": True, "retrieval_backend": "es"},
                    output_path=output,
                    label="test",
                )
            payload = json.loads(output.read_text())
            self.assertFalse(payload["passed"])
            self.assertIn("retrieved_context_empty", payload["reasons"])

    def test_reaction_preflight_uses_local_reaction_sources_not_elasticsearch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            retriever = mock.Mock()
            result = run_real_rag_preflight(
                retriever,
                case={
                    "task_type": "reaction_prediction",
                    "base_case": {
                        "request": "CCO>>CC=O",
                        "task_type": "reaction_yield_prediction",
                        "dataset_name": "buchwald_hartwig_kg:test1",
                    },
                },
                runtime={"require_real_rag": True, "retrieval_backend": "es"},
                output_path=Path(tmp) / "preflight.json",
                label="reaction-test",
                reaction_runtime=_FakeReactionRuntime(),
            )
            self.assertTrue(result["passed"], result["reasons"])
            self.assertEqual(result["backend"], "local_reaction_data")
            self.assertEqual(result["qualifying_retrieval_count"], 2)
            retriever.retrieve.assert_not_called()

    def test_api_budget_blocks_next_call_at_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            usage = Path(tmp) / "runtime" / "api_usage"
            usage.mkdir(parents=True)
            (usage / "api_calls.jsonl").write_text(
                json.dumps({"total_tokens": 5000}) + "\n",
                encoding="utf-8",
            )
            budget = ApiBudget(Path(tmp), max_calls=1, max_total_tokens=10000)
            with self.assertRaisesRegex(ApiBudgetExceeded, "calls=1"):
                budget.check_before_call()

    def test_heuristic_budget_is_zero_and_blocks_before_provider_call(self) -> None:
        self.assertEqual(effective_api_budget({"orchestrator": "heuristic"}), (0, 0))
        with tempfile.TemporaryDirectory() as tmp:
            from agentm.llm.provider import LLMClient, LLMConfig

            client = LLMClient(LLMConfig(api_key="must-not-be-used"))
            with api_budget_guard(tmp, max_calls=0, max_total_tokens=0):
                with self.assertRaisesRegex(ApiBudgetExceeded, "calls=0"):
                    client.chat_json([{"role": "user", "content": "must not leave process"}])
            self.assertEqual(ApiBudget(Path(tmp), max_calls=0, max_total_tokens=0).usage()["calls"], 0)

    def test_guarded_run_invariant_requires_exact_runtime_real_rag(self) -> None:
        job = {"runtime_guard_protocol": "pre_llm_real_rag_fail_closed"}
        valid_context = {
            "metadata": {"retrieval_backend_used": "elasticsearch_hybrid_bm25_dense"},
            "retrieval_trace": [
                {
                    "provider": "elasticsearch_hybrid",
                    "backend": "elasticsearch_hybrid_bm25_dense",
                    "retrieved_context": [{"chunk_id": "chunk"}],
                    "warnings": [],
                }
            ],
        }
        self.assertEqual(run_invariant_violations(job, valid_context), [])

        invalid_context = {
            "metadata": {"retrieval_backend_used": "local_rag"},
            "retrieval_trace": [
                {
                    "provider": "hybrid_retriever",
                    "backend": "local_rag",
                    "retrieved_context": [{"chunk_id": "chunk"}],
                    "warnings": ["Connection refused"],
                }
            ],
        }
        violations = run_invariant_violations(job, invalid_context)
        self.assertIn("runtime_retrieval_backend=local_rag", violations)
        self.assertIn("retrieval_trace_dependency_error", violations)
        self.assertIn(
            "retrieval_trace_missing_verified_elasticsearch_hybrid_attempt",
            violations,
        )

    def test_guarded_invariant_accepts_verified_empty_dense_attempt(self) -> None:
        job = {"runtime_guard_protocol": "pre_llm_real_rag_fail_closed"}
        context = {
            "metadata": {},
            "retrieval_trace": [
                {
                    "provider": "elasticsearch_hybrid",
                    "backend": "elasticsearch_hybrid_bm25_dense",
                    "retrieved_context": [],
                    "evidence_items": [],
                    "warnings": [],
                    "metadata": {"dense_enabled": True, "dense_status": "completed"},
                }
            ],
        }
        self.assertEqual(run_invariant_violations(job, context), [])
        self.assertEqual(runtime_retrieval_backend(context), "elasticsearch_hybrid_bm25_dense")

    def test_guarded_reaction_invariant_accepts_recovered_local_retrieval(self) -> None:
        job = {
            "runtime_guard_protocol": "pre_llm_real_rag_fail_closed",
            "dataset_group": "stress",
            "case": {
                "primary_fault_count": 1,
                "task_type": "reaction_evidence",
                "base_case": {"task_type": "reaction_classification"},
            },
        }
        context = {
            "resource_resolution": {"reaction": {"all_required_present": True}},
            "retrieval_trace": [
                {
                    "provider": "RetrieveAgent",
                    "backend": "local_reaction_data",
                    "evidence_items": [],
                    "warnings": ["temporary no result"],
                },
                {
                    "provider": "RetrieveAgent",
                    "backend": "local_reaction_data",
                    "evidence_items": [{"citation_id": "mit-1"}],
                    "warnings": [],
                },
            ],
        }
        self.assertEqual(run_invariant_violations(job, context), [])
        self.assertEqual(runtime_retrieval_backend(context), "local_reaction_data")

    def test_completed_guarded_raw_run_requires_preflight_and_budget_compliance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "run"
            write_json(
                raw / "job.json",
                {
                    "runtime_guard_protocol": "pre_llm_real_rag_fail_closed",
                    "config": {
                        "runtime": {
                            "require_real_rag": True,
                            "max_llm_calls_per_run": 5,
                            "max_total_tokens_per_run": 10000,
                        }
                    },
                },
                overwrite=False,
            )
            write_json(
                raw / "run_record.json",
                {
                    "run_id": "run",
                    "case_id": "case",
                    "dataset_group": "clean",
                    "completed": True,
                    "runtime_retrieval_backend": "elasticsearch_hybrid_bm25_dense",
                    "runtime_preflight_passed": True,
                    "retrieval_call_count": 1,
                    "llm_call_count": 1,
                    "total_tokens": 5000,
                    "llm_budget_exceeded": False,
                },
                overwrite=False,
            )
            write_json(
                raw / "runtime" / "retrieval_trace.json",
                [
                    {
                        "provider": "elasticsearch_hybrid",
                        "backend": "elasticsearch_hybrid_bm25_dense",
                        "retrieved_context": [{"chunk_id": "chunk"}],
                        "warnings": [],
                    }
                ],
                overwrite=False,
            )
            result = validate_raw_run(raw)
            self.assertTrue(result["valid"], result["reasons"])

    def test_worker_preflight_failure_records_zero_llm_calls(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "output"
            job_path = Path(tmp) / "job.json"
            job = {
                "run_id": "guarded-run",
                "case_id": "case",
                "dataset_group": "clean",
                "orchestrator": "llm",
                "reflection_enabled": False,
                "seed": 2026,
                "repeat_index": 0,
                "source_tree_hash": source_tree_hash(),
                "source_manifest": "test",
                "config_hash": "config-hash",
                "runtime_guard_protocol": "pre_llm_real_rag_fail_closed",
                "case": {
                    "case_id": "case",
                    "dataset_group": "clean",
                    "request": "aspirin",
                    "task_type": "admet_prediction",
                    "dataset_name": "DILI",
                },
                "config": {
                    "experiment": {
                        "temperature": 0.0,
                        "timeout_seconds": 30,
                        "llm_timeout_seconds": 5,
                        "max_retries": 0,
                    },
                    "runtime": {
                        "require_real_rag": True,
                        "retrieval_backend": "es",
                        "feature_set": "all_features",
                        "max_llm_calls_per_run": 2,
                        "max_total_tokens_per_run": 1000,
                    },
                    "paths": {
                        "reaction_model_dir": None,
                        "reaction_index_dir": None,
                        "reaction_data_root": None,
                    },
                    "outputs": {"root": str(output)},
                },
            }
            write_json(job_path, job, overwrite=False)
            run_called = False

            class FakeController:
                def __init__(self, **_kwargs):
                    self.runtime = SimpleNamespace(hybrid_retriever=_FakeRetriever(_result(valid=False)))

                def run(self, *_args, **_kwargs):
                    nonlocal run_called
                    run_called = True
                    raise AssertionError("controller.run must not execute after failed preflight")

            import agentm.runtime

            with mock.patch.object(agentm.runtime, "AgentMMultiAgentController", FakeController):
                exit_code = run_worker(job_path)
            record = json.loads((output / "raw_runs" / "guarded-run" / "run_record.json").read_text())
            self.assertEqual(exit_code, 2)
            self.assertFalse(run_called)
            self.assertEqual(record["llm_call_count"], 0)
            self.assertFalse(record["runtime_preflight_passed"])
            self.assertIn("real_rag_preflight_failed", record["error"])

    def test_multi_turn_preflights_every_turn_before_first_llm_call(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "output"
            job_path = Path(tmp) / "job.json"
            job = {
                "run_id": "guarded-multi",
                "case_id": "multi-case",
                "dataset_group": "multi_turn",
                "orchestrator": "llm",
                "reflection_enabled": False,
                "seed": 2026,
                "repeat_index": 0,
                "source_tree_hash": source_tree_hash(),
                "source_manifest": "test",
                "config_hash": "config-hash",
                "runtime_guard_protocol": "pre_llm_real_rag_fail_closed",
                "case": {
                    "case_id": "multi-case",
                    "turns": [
                        {"request": "aspirin", "task_type": "admet_prediction", "dataset_name": "DILI"},
                        {"request": "warfarin", "task_type": "admet_prediction", "dataset_name": "AMES"},
                    ],
                },
                "config": {
                    "experiment": {
                        "temperature": 0.0,
                        "timeout_seconds": 30,
                        "llm_timeout_seconds": 5,
                        "max_retries": 0,
                    },
                    "runtime": {
                        "require_real_rag": True,
                        "retrieval_backend": "es",
                        "feature_set": "all_features",
                        "max_llm_calls_per_run": 4,
                        "max_total_tokens_per_run": 2000,
                    },
                    "paths": {
                        "reaction_model_dir": None,
                        "reaction_index_dir": None,
                        "reaction_data_root": None,
                    },
                    "outputs": {"root": str(output)},
                    "multi_turn_tasks": {"turns_per_case": 2},
                },
            }
            write_json(job_path, job, overwrite=False)
            run_calls = 0
            constructed = 0

            class FakeController:
                def __init__(self, **_kwargs):
                    nonlocal constructed
                    constructed += 1
                    self.runtime = SimpleNamespace(
                        hybrid_retriever=_FakeRetriever(_result(valid=constructed == 1))
                    )

                def run(self, *_args, **_kwargs):
                    nonlocal run_calls
                    run_calls += 1
                    raise AssertionError("no turn may run until all turn preflights pass")

            import agentm.runtime

            with mock.patch.object(agentm.runtime, "AgentMMultiAgentController", FakeController):
                exit_code = run_multi_turn_worker(job_path)
            record = json.loads((output / "raw_runs" / "guarded-multi" / "run_record.json").read_text())
            self.assertEqual(exit_code, 2)
            self.assertEqual(constructed, 2)
            self.assertEqual(run_calls, 0)
            self.assertEqual(record["llm_call_count"], 0)
            self.assertEqual(record["turn_count"], 0)
            self.assertFalse(record["runtime_preflight_passed"])


if __name__ == "__main__":
    unittest.main()
