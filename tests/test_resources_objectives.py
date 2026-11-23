from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from agentm.runtime import AgentMMultiAgentController
from agentm.runtime.objectives import evaluate_task_objectives
from agentm.runtime.retrieval.evidence_schema import endpoint_from_dataset, entity_terms_from_context, iter_structured_evidence_documents
from agentm.runtime.resources import RESOURCE_ROOT, resolve_model_resource, resolve_rag_resources
from agentm.runtime.tools.scientific import LOCAL_COMPOUND_IDENTITY_SEEDS, ScientificToolRuntime
from agentm.tools.smiles import standardize_smiles


class UnavailableLLMClient:
    available = False

    def chat_json(self, messages, temperature=0.1):
        raise RuntimeError("unavailable")


class ResourceAndObjectiveTests(unittest.TestCase):
    def test_model_resolver_prefers_feature_compatible_admet_model(self):
        resolution = resolve_model_resource(
            task_type="admet_prediction",
            dataset_name="DILI",
            feature_set="rdkit_only",
            auto_model=True,
        )
        self.assertTrue(resolution["selected_path"].endswith("outputs/tdc_ablation/DILI/rdkit_only/model.pkl"))
        self.assertEqual(resolution["selected_reason"], "compatible")
        self.assertTrue(resolution["schema_compatible"])
        mismatched = [item for item in resolution["candidates"] if item["path"].endswith("outputs/tdc/DILI/model.pkl")]
        self.assertTrue(mismatched)
        self.assertIn("feature_set", mismatched[0]["compatibility_status"])

    def test_model_resolver_prefers_feature_compatible_ddi_model(self):
        resolution = resolve_model_resource(
            task_type="ddi_prediction",
            dataset_name="DDI",
            feature_set="rdkit_only",
            auto_model=True,
        )
        self.assertTrue(resolution["selected_path"].endswith("outputs/drugbank_ddi_full_low_token/random_rdkit_only/model.pkl"))
        self.assertEqual(resolution["selected_reason"], "compatible")

    def test_local_identity_seeds_cover_50_case_ddi_pairs(self):
        required = [
            "clarithromycin",
            "fluoxetine",
            "metoprolol",
            "amiodarone",
            "ciprofloxacin",
            "tizanidine",
            "ibuprofen",
        ]
        for key in required:
            with self.subTest(key=key):
                seed = LOCAL_COMPOUND_IDENTITY_SEEDS.get(key)
                self.assertIsNotNone(seed)
                self.assertTrue(seed.get("pubchem_cid"))
                canonical, error = standardize_smiles(seed.get("canonical_smiles", ""))
                self.assertIsNone(error)
                self.assertTrue(canonical)

    def test_rag_resolver_uses_absolute_sibling_resource_paths(self):
        resolution = resolve_rag_resources(task_type="admet_prediction")
        selected = Path(resolution["selected_corpus_dirs"][0])
        self.assertTrue(selected.is_absolute())
        self.assertEqual(selected, RESOURCE_ROOT / "data" / "rag_corpus" / "tdc_real")
        self.assertTrue(resolution["all_required_present"])

    def test_endpoint_mapping_normalizes_tdc_names(self):
        self.assertEqual(endpoint_from_dataset("BBB_Martins"), "bbbp")
        self.assertEqual(endpoint_from_dataset("hERG"), "herg")
        self.assertEqual(endpoint_from_dataset("DILI"), "dili")

    def test_entity_terms_include_resolved_identity_metadata(self):
        terms = entity_terms_from_context(
            {
                "molecules": [
                    {
                        "original_input": "CC(=O)OC1=CC=CC=C1C(=O)O",
                        "canonical_smiles": "CC(=O)OC1=CC=CC=C1C(=O)O",
                        "metadata": {
                            "pubchem_cid": 2244,
                            "pubchem_title": "Aspirin",
                            "pubchem_synonyms": ["ACETYLSALICYLIC ACID"],
                        },
                    }
                ]
            }
        )
        self.assertIn("aspirin", terms)
        self.assertIn("2244", terms)
        self.assertIn("acetylsalicylic acid", terms)

    def test_structured_evidence_index_extracts_compound_and_endpoint_metadata(self):
        docs = list(
            iter_structured_evidence_documents(
                [RESOURCE_ROOT / "data" / "rag_corpus" / "tdc_real" / "tasks" / "BBB_Martins"],
                limit=80,
            )
        )
        self.assertTrue(docs)
        caffeine_docs = [
            doc
            for doc in docs
            if doc["source"] == "pubchem" and "caffeine" in {term.lower() for term in doc.get("compound_synonyms", []) + doc.get("compound_names", [])}
        ]
        self.assertTrue(caffeine_docs)
        doc = caffeine_docs[0]
        self.assertEqual(doc["task_type"], "admet_prediction")
        self.assertEqual(doc["endpoint"], "bbbp")
        self.assertTrue(doc["compound_specific"])
        self.assertFalse(doc["endpoint_specific"])
        self.assertEqual(doc["evidence_scope"], "compound_background")
        self.assertIn("bbbp", doc["endpoint_terms"])
        self.assertIn("caffeine", doc["entity_terms"])

    def test_curated_admet_cards_are_compound_endpoint_grounded(self):
        root = Path(__file__).resolve().parents[1] / "data" / "rag_corpus" / "admet_curated"
        docs = list(iter_structured_evidence_documents([root]))
        curated_docs = [doc for doc in docs if doc.get("source") == "curated_admet"]
        if not curated_docs:
            self.skipTest("runtime curated ADMET corpus is optional in the ReactionAgent first-phase package.")
        self.assertGreaterEqual(len(docs), 15)
        self.assertGreaterEqual(len(curated_docs), 15)
        self.assertTrue(all(doc.get("compound_specific") for doc in curated_docs))
        self.assertTrue(all(doc.get("endpoint_specific") for doc in curated_docs))
        self.assertTrue(all(doc.get("evidence_scope") == "compound_endpoint" for doc in curated_docs))

    def test_curated_aspirin_bbbp_note_passes_retained_evidence_filter(self):
        root = Path(__file__).resolve().parents[1]
        note_path = root / "data" / "rag_corpus" / "admet_curated" / "tasks" / "BBB_Martins" / "aspirin_bbbp_curated_evidence.md"
        self.assertTrue(note_path.exists())
        note_text = note_path.read_text(encoding="utf-8")
        self.assertIn("PMID: 40922832", note_text)
        self.assertIn("PMID: 16489364", note_text)
        with tempfile.TemporaryDirectory() as tmp:
            runtime = ScientificToolRuntime(output_dir=tmp, feature_set="rdkit_only", require_real_rag=True)
            context = {
                "task_type": "property_prediction",
                "dataset_name": "BBB_Martins",
                "metadata": {"min_evidence": 1, "require_real_rag": True},
                "molecules": [
                    {
                        "name": "aspirin",
                        "canonical_smiles": "CC(=O)Oc1ccccc1C(=O)O",
                        "metadata": {
                            "pubchem_cid": "2244",
                            "pubchem_title": "Aspirin",
                            "pubchem_synonyms": ["acetylsalicylic acid"],
                        },
                    }
                ],
                "evidence": [],
                "warnings": [],
            }
            added = runtime._append_evidence(
                context,
                [
                    {
                        "source": "rag_literature",
                        "citation_id": "aspirin_bbbp_curated",
                        "source_path": str(note_path),
                        "claim": note_text,
                        "value": {
                            "snippet": note_text,
                            "source_path": str(note_path),
                            "pmid": "40922832",
                            "url": "https://pubmed.ncbi.nlm.nih.gov/40922832/",
                        },
                    }
                ],
            )
        self.assertEqual(added, 1)
        self.assertEqual(context["evidence"][0]["citation_id"], "aspirin_bbbp_curated")
        self.assertTrue(context["evidence"][0]["retained"])
        self.assertGreaterEqual(context["evidence"][0]["retention_score"], 6.0)
        self.assertIn("aspirin", context["evidence"][0]["retention_matches"]["compound"])
        self.assertIn("bbbp", context["evidence"][0]["retention_matches"]["endpoint"])

    def test_structured_evidence_index_extracts_ddi_pair_metadata(self):
        pair_dir = RESOURCE_ROOT / "data" / "rag_corpus" / "ddi_real" / "pairs" / "db00691__db00705"
        docs = list(iter_structured_evidence_documents([pair_dir], limit=20))
        self.assertTrue(docs)
        self.assertTrue(any(doc.get("endpoint") == "ddi" for doc in docs))
        self.assertTrue(any("moexipril" in doc.get("entity_terms", []) for doc in docs))
        self.assertTrue(any(doc.get("ddi_pair_key") == "db00691__db00705" for doc in docs))

    def test_structured_evidence_index_preserves_backfill_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task_dir = root / "tasks" / "hERG"
            task_dir.mkdir(parents=True)
            payload = {
                "source": "chembl",
                "evidence_type": "bioactivity",
                "title": "Caffeine ChEMBL profile",
                "text": "Caffeine bioactivity record relevant to hERG cardiac safety context.",
                "task_type": "admet_prediction",
                "dataset_name": "hERG",
                "endpoint": "herg",
                "compound_names": ["Caffeine"],
                "compound_synonyms": ["1,3,7-Trimethylxanthine"],
                "compound_cids": ["2519"],
                "chembl_ids": ["CHEMBL113"],
                "canonical_smiles": ["Cn1cnc2n(C)c(=O)n(C)c(=O)c12"],
            }
            (task_dir / "backfill_caffeine_herg.jsonl").write_text(json.dumps(payload) + "\n", encoding="utf-8")
            docs = list(iter_structured_evidence_documents([root]))
        self.assertEqual(len(docs), 1)
        doc = docs[0]
        self.assertEqual(doc["source"], "chembl")
        self.assertEqual(doc["endpoint"], "herg")
        self.assertIn("caffeine", doc["entity_terms"])
        self.assertIn("chembl113", {term.lower() for term in doc["entity_terms"]})
        self.assertTrue(doc["compound_specific"])
        self.assertTrue(doc["endpoint_specific"])

    def test_structured_evidence_index_does_not_promote_identity_docs_to_endpoint_specific(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task_dir = root / "tasks" / "DILI"
            task_dir.mkdir(parents=True)
            payload = {
                "source": "pubchem",
                "evidence_type": "compound_identity",
                "title": "Aspirin: PubChem identity",
                "text": "Aspirin identity from PubChem. CID 2244. Formula C9H8O4.",
                "task_type": "admet_prediction",
                "dataset_name": "DILI",
                "endpoint": "dili",
                "compound_names": ["Aspirin"],
                "compound_cids": ["2244"],
            }
            (task_dir / "backfill_aspirin_dili.jsonl").write_text(json.dumps(payload) + "\n", encoding="utf-8")
            docs = list(iter_structured_evidence_documents([root]))
        self.assertEqual(len(docs), 1)
        self.assertTrue(docs[0]["compound_specific"])
        self.assertFalse(docs[0]["endpoint_specific"])
        self.assertEqual(docs[0]["evidence_scope"], "compound_background")

    def test_objective_evaluator_detects_missing_evidence_and_zero_shot(self):
        context = {
            "metadata": {"min_evidence": 2, "require_real_model": True, "require_real_rag": True, "enable_fallbacks": False},
            "predictions": [{"model_type": "zero_shot_rule_semantic_fusion"}],
            "confidence": {"low_confidence": False},
            "evidence": [{"citation_id": "cite_1", "source_path": "/tmp/source"}],
            "verifier_report": {
                "prediction": {"prediction_grounded": True},
                "citations": {"citation_grounded": True, "n_unsupported": 0},
                "schema": {"schema_complete": True},
            },
            "reflection": {"application_status": "reflection_applied"},
        }
        status = evaluate_task_objectives(context)
        self.assertEqual(status["status"], "fail")
        self.assertIn("real_model_required", status["missing_objectives"])
        self.assertIn("real_rag_required", status["missing_objectives"])

    def test_objective_evaluator_requires_relevant_compound_specific_evidence(self):
        context = {
            "metadata": {"min_evidence": 1, "enable_fallbacks": True},
            "predictions": [{"model_type": "random_forest"}],
            "confidence": {"low_confidence": False},
            "evidence": [{"citation_id": "cite_1", "source_path": "/tmp/source"}],
            "verifier_report": {
                "prediction": {"prediction_grounded": True},
                "citations": {
                    "citation_grounded": True,
                    "evidence_relevance_grounded": False,
                    "compound_specific_grounded": False,
                    "n_unsupported": 0,
                },
                "schema": {"schema_complete": True},
            },
            "reflection": {"application_status": "reflection_applied"},
        }
        status = evaluate_task_objectives(context)
        self.assertEqual(status["status"], "warning")
        self.assertIn("evidence_relevance_grounded", status["missing_objectives"])
        self.assertIn("compound_specific_grounded", status["missing_objectives"])

    def test_curated_clopidogrel_omeprazole_corpus_is_pair_grounded(self):
        root = Path(__file__).resolve().parents[1] / "data" / "rag_corpus" / "ddi_curated"
        docs = list(iter_structured_evidence_documents([root]))
        if not docs:
            self.skipTest("runtime curated DDI corpus is optional in the ReactionAgent first-phase package.")
        pair_docs = [doc for doc in docs if doc.get("ddi_pair_key") == "clopidogrel__omeprazole"]
        self.assertGreaterEqual(len(pair_docs), 1)
        self.assertTrue(all(doc.get("evidence_scope") == "ddi_pair" for doc in pair_docs))
        self.assertTrue(all(doc.get("compound_specific") for doc in pair_docs))
        self.assertTrue(all(doc.get("endpoint_specific") for doc in pair_docs))

    def test_retained_evidence_filter_keeps_pair_specific_evidence_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            runtime = ScientificToolRuntime(output_dir=tmp, feature_set="rdkit_only", require_real_rag=True)
            context = {
                "task_type": "ddi_prediction",
                "dataset_name": "DDI",
                "metadata": {"min_evidence": 1, "require_real_rag": True},
                "molecules": [
                    {"name": "clopidogrel", "synonyms": ["clopidogrel"]},
                    {"name": "omeprazole", "synonyms": ["omeprazole"]},
                ],
                "evidence": [],
                "warnings": [],
            }
            added = runtime._append_evidence(
                context,
                [
                    {
                        "source": "pubmed",
                        "citation_id": "bad",
                        "source_path": "/tmp/background.md",
                        "claim": "Aspirin and warfarin are discussed in generic drug interaction background.",
                    },
                    {
                        "source": "dailymed",
                        "citation_id": "good",
                        "source_path": "/tmp/clopidogrel_omeprazole.md",
                        "claim": "Clopidogrel and omeprazole have a CYP2C19 drug-drug interaction affecting clopidogrel activation.",
                    },
                ],
            )
            self.assertEqual(added, 1)
            self.assertEqual(context["evidence"][0]["citation_id"], "good")
            self.assertEqual(context["artifacts"]["retained_evidence_filter"][0]["raw_count"], 2)
            self.assertEqual(context["artifacts"]["retained_evidence_filter"][0]["retained_count"], 1)
            self.assertIn("retained_evidence_filtered:1/2", context["warnings"])

    def test_objective_refresh_removes_stale_objective_warnings(self):
        with tempfile.TemporaryDirectory() as tmp:
            runtime = ScientificToolRuntime(output_dir=tmp, feature_set="rdkit_only", min_evidence=1)
            context = {
                "task_type": "property_prediction",
                "metadata": {"min_evidence": 1, "enable_fallbacks": True},
                "predictions": [{"model_type": "random_forest"}],
                "confidence": {"low_confidence": False},
                "evidence": [{"citation_id": "cite_1", "source_path": "/tmp/source"}],
                "verifier_report": {
                    "prediction": {"prediction_grounded": True},
                    "citations": {
                        "citation_grounded": True,
                        "evidence_relevance_grounded": False,
                        "compound_specific_grounded": False,
                        "n_unsupported": 0,
                    },
                    "schema": {"schema_complete": True},
                    "summary": {
                        "warnings": [
                            "objective_missing:citation_grounded",
                            "objective_missing:reflection_applied",
                            "low_relevance_evidence:1",
                        ]
                    },
                },
                "reflection": {"application_status": "reflection_applied"},
                "warnings": [
                    "objective_missing:citation_grounded",
                    "objective_missing:reflection_applied",
                    "low_relevance_evidence:1",
                ],
            }
            runtime._refresh_objective_status(context)
            self.assertEqual(context["objective_status"]["missing_objectives"], ["evidence_relevance_grounded", "compound_specific_grounded"])
            self.assertNotIn("objective_missing:citation_grounded", context["warnings"])
            self.assertNotIn("objective_missing:reflection_applied", context["warnings"])
            self.assertIn("objective_missing:evidence_relevance_grounded", context["warnings"])
            self.assertIn("objective_missing:compound_specific_grounded", context["warnings"])

    def test_reflection_requests_refinement_for_low_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            runtime = ScientificToolRuntime(output_dir=tmp, feature_set="rdkit_only", min_evidence=2)
            context = {
                "request": "aspirin DILI",
                "task_type": "admet_prediction",
                "dataset_name": "DILI",
                "metadata": {"min_evidence": 2, "reflection_rounds": 0},
                "evidence": [{"citation_id": "cite_1", "source_path": "/tmp/source"}],
                "confidence": {"low_confidence": True},
                "verifier_report": {
                    "summary": {"status": "warning"},
                    "citations": {"citation_grounded": True, "n_unsupported": 0},
                },
                "warnings": [],
            }
            result = runtime.reflect_on_report({}, context)
            self.assertEqual(result.status, "completed")
            self.assertTrue(context["reflection"]["needs_refinement"])
            self.assertTrue(context["reflection"]["requires_retrieval_refinement"])
            self.assertEqual(context["reflection"]["recommendation"], "expand_evidence")

    def test_require_llm_orchestrator_does_not_heuristic_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            controller = AgentMMultiAgentController(
                output_dir=tmp,
                feature_set="rdkit_only",
                use_llm_orchestrator=True,
                orchestrator_llm_client=UnavailableLLMClient(),
                enable_fallbacks=True,
                require_llm_orchestrator=True,
                max_rounds=2,
            )
            report = controller.run("CC(=O)OC1=CC=CC=C1C(=O)O", task_type="admet_prediction", dataset_name="DILI")
            self.assertFalse(report["predictions"])
            self.assertEqual(report["multi_agent"]["planner_type"], "llm_failed")
            self.assertFalse(any(call.get("agent") == "MoleculeAgent" for call in report["tool_calls"]))
            self.assertTrue(any("llm_task_graph_planning_failed_required" in warning for warning in report["warnings"]))

    def test_strict_auto_model_single_prediction_uses_real_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            controller = AgentMMultiAgentController(
                output_dir=tmp,
                feature_set="rdkit_only",
                auto_model=True,
                require_real_model=True,
                min_evidence=0,
                mode="fast",
                retrieval_backend="local",
                use_llm_orchestrator=False,
                enable_fallbacks=False,
            )
            report = controller.run("CC(=O)OC1=CC=CC=C1C(=O)O", task_type="admet_prediction", dataset_name="DILI")
            prediction = report["predictions"][0]
            self.assertEqual(prediction["model_type"], "random_forest")
            self.assertFalse(any(call.get("status") == "fallback_completed" for call in report["tool_calls"]))
            self.assertEqual(report["objective_status"]["status"], "pass")
            self.assertTrue((Path(tmp) / "resource_resolution.json").exists())
            self.assertTrue((Path(tmp) / "objective_status.json").exists())


if __name__ == "__main__":
    unittest.main()
