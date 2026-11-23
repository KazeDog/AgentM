from __future__ import annotations

import csv
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from agentm.runtime.retrieval.evidence_schema import endpoint_from_dataset, endpoint_terms, iter_structured_evidence_documents
from scripts.backfill_evidence_cases import PublicDatabaseCache, _admet_query_from_target, _case_status, _ddi_queries_from_target, _new_http_session
from scripts.audit_evidence_relevance_cases import audit_case, parse_case
from scripts.audit_hybrid_retrieval_cases import audit_local_retrieval_case
from scripts.build_evidence_manifest import attach_coverage, iter_admet_targets, iter_ddi_targets
from scripts.run_evidence_acceptance_suite import build_acceptance_summary


class EvidenceManifestTests(unittest.TestCase):
    def test_endpoint_hints_cover_full_admet_datasets(self):
        self.assertEqual(endpoint_from_dataset("Caco2_Wang"), "caco2")
        self.assertEqual(endpoint_from_dataset("Clearance_Hepatocyte_AZ"), "clearance_hepatocyte")
        self.assertEqual(endpoint_from_dataset("HydrationFreeEnergy_FreeSolv"), "hydration_free_energy")
        self.assertEqual(endpoint_from_dataset("TWOSIDES", task_type="ddi_prediction"), "ddi")

    def test_cyp_endpoint_terms_are_isoform_specific(self):
        terms = endpoint_terms("cyp1a2")
        self.assertIn("cyp1a2 inhibition", terms)
        self.assertIn("cytochrome p450 1a2", terms)
        self.assertNotIn("metabolism", terms)
        self.assertNotIn("cyp inhibition", terms)

    def test_loaded_endpoint_aliases_are_recomputed_strictly(self):
        with tempfile.TemporaryDirectory() as tmp:
            corpus = Path(tmp)
            dataset = corpus / "tasks" / "CYP1A2_Veith"
            dataset.mkdir(parents=True)
            _write_jsonl(
                dataset / "legacy.jsonl",
                [
                    {
                        "title": "Legacy CYP row",
                        "text": "Aspirin background profile.",
                        "endpoint_aliases": ["cyp inhibition", "metabolism"],
                    }
                ],
            )
            docs = list(iter_structured_evidence_documents([corpus]))
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0]["endpoint"], "cyp1a2")
        self.assertEqual(docs[0]["endpoint_aliases"], endpoint_terms("cyp1a2"))
        self.assertNotIn("metabolism", docs[0]["endpoint_aliases"])
        self.assertNotIn("cyp inhibition", docs[0]["endpoint_aliases"])

    def test_admet_manifest_filters_placeholder_drug_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            tdc = Path(tmp)
            dataset = tdc / "AMES"
            dataset.mkdir()
            _write_csv(
                dataset / "AMES.csv",
                ["smiles", "label", "tdc_task", "tdc_collection", "task_mode", "drug_id"],
                [["CCO", "1", "AMES", "Tox", "classification", "Drug 3"]],
            )
            targets = list(iter_admet_targets(tdc))
        self.assertEqual(len(targets), 1)
        self.assertEqual(targets[0]["endpoint"], "ames")
        self.assertEqual(targets[0]["drug_name"], "")
        self.assertNotIn("drug 3", targets[0]["entity_terms"])

    def test_ddi_manifest_deduplicates_unordered_pairs(self):
        with tempfile.TemporaryDirectory() as tmp:
            tdc = Path(tmp)
            dataset = tdc / "DrugBank"
            dataset.mkdir()
            _write_csv(
                dataset / "DrugBank.csv",
                ["drug_a_smiles", "drug_b_smiles", "drug_a_name", "drug_b_name", "label", "drug_a_id", "drug_b_id"],
                [
                    ["CCO", "CCC", "DB001", "DB002", "1", "DB001", "DB002"],
                    ["CCC", "CCO", "DB002", "DB001", "2", "DB002", "DB001"],
                ],
            )
            targets = list(iter_ddi_targets(tdc))
        self.assertEqual(len(targets), 1)
        self.assertEqual(targets[0]["pair_key"], "db001__db002")
        self.assertEqual(targets[0]["label_counts"], {"1": 1, "2": 1})

    def test_coverage_requires_entity_and_endpoint_grounding(self):
        target = {
            "target_id": "case_1",
            "kind": "admet_case",
            "dataset_name": "DILI",
            "endpoint": "dili",
            "entity_terms": ["aspirin", "2244"],
        }
        evidence = [
            {
                "endpoint": "dili",
                "source": "pubchem",
                "entity_terms": ["aspirin", "2244"],
                "compound_specific": True,
                "endpoint_specific": False,
                "evidence_scope": "compound_background",
                "chunk_id": "pubchem::1",
            },
            {
                "endpoint": "dili",
                "source": "pubmed",
                "entity_terms": ["aspirin"],
                "compound_specific": True,
                "endpoint_specific": True,
                "evidence_scope": "compound_endpoint",
                "chunk_id": "pubmed::1",
            },
        ]
        row = attach_coverage(target, evidence)
        self.assertEqual(row["status"], "covered")
        self.assertEqual(row["entity_endpoint_documents"], 1)
        self.assertEqual(row["entity_background_documents"], 1)

    def test_structured_evidence_preserves_canonical_smiles_list_without_repr(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task_dir = root / "tasks" / "DILI"
            task_dir.mkdir(parents=True)
            payload = {
                "source": "pubchem",
                "evidence_type": "compound_identity",
                "title": "Aspirin: PubChem identity",
                "text": "Aspirin identity from PubChem. CID 2244.",
                "task_type": "admet_prediction",
                "dataset_name": "DILI",
                "endpoint": "dili",
                "compound_names": ["Aspirin"],
                "compound_cids": ["2244"],
                "canonical_smiles": ["CC(=O)OC1=CC=CC=C1C(=O)O"],
            }
            (task_dir / "backfill_aspirin_dili.jsonl").write_text(json.dumps(payload) + "\n", encoding="utf-8")
            docs = list(iter_structured_evidence_documents([root]))
        self.assertEqual(docs[0]["canonical_smiles"], ["CC(=O)OC1=CC=CC=C1C(=O)O"])
        self.assertNotIn("['CC(=O)OC1=CC=CC=C1C(=O)O']", docs[0]["entity_terms"])

    def test_structured_evidence_loader_skips_generated_index_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task_dir = root / "tasks" / "DILI"
            task_dir.mkdir(parents=True)
            payload = {
                "source": "pubchem",
                "evidence_type": "compound_identity",
                "title": "Aspirin: PubChem identity",
                "text": "Aspirin identity from PubChem. CID 2244.",
                "task_type": "admet_prediction",
                "dataset_name": "DILI",
                "endpoint": "dili",
                "compound_names": ["Aspirin"],
            }
            (task_dir / "backfill_aspirin_dili.jsonl").write_text(json.dumps(payload) + "\n", encoding="utf-8")
            (root / "structured_evidence.jsonl").write_text(json.dumps({**payload, "title": "generated"}) + "\n", encoding="utf-8")
            (root / "evidence_relevance_audit.json").write_text(json.dumps({**payload, "title": "audit"}) + "\n", encoding="utf-8")
            docs = list(iter_structured_evidence_documents([root]))
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0]["title"], "Aspirin: PubChem identity")

    def test_structured_evidence_recomputes_stale_ddi_scope_from_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pair_dir = root / "pairs" / "aspirin__warfarin"
            pair_dir.mkdir(parents=True)
            payload = {
                "source": "pubmed",
                "evidence_type": "literature",
                "title": "Aspirin and warfarin bleeding risk",
                "text": "Patients receiving aspirin and warfarin dual therapy had bleeding risk.",
                "task_type": "ddi_prediction",
                "dataset_name": "DDI",
                "endpoint": "ddi",
                "compound_names": ["aspirin", "warfarin"],
                "pair_drug_names": ["aspirin", "warfarin"],
                "compound_specific": True,
                "endpoint_specific": False,
                "evidence_scope": "ddi_pair_background",
            }
            (pair_dir / "backfill_pair.jsonl").write_text(json.dumps(payload) + "\n", encoding="utf-8")
            docs = list(iter_structured_evidence_documents([root]))
        self.assertEqual(len(docs), 1)
        self.assertTrue(docs[0]["compound_specific"])
        self.assertTrue(docs[0]["endpoint_specific"])
        self.assertEqual(docs[0]["evidence_scope"], "ddi_pair")
        self.assertIn("aspirin", docs[0]["text_entity_terms"])
        self.assertIn("warfarin", docs[0]["text_entity_terms"])

    def test_ddi_pair_metadata_uses_manifest_names_and_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pair_dir = root / "pairs" / "db00945_db00682"
            pair_dir.mkdir(parents=True)
            (pair_dir / "pair_manifest.json").write_text(
                json.dumps(
                    {
                        "pair_key": "db00945_db00682",
                        "drug_a": "CC(=O)OC1=CC=CC=C1C(=O)O",
                        "drug_b": "CC(=O)CC(C1=CC=CC=C1)C1=C(O)C2=CC=CC=C2OC1=O",
                        "evidence_drug_a": "Aspirin",
                        "evidence_drug_b": "Warfarin",
                    }
                ),
                encoding="utf-8",
            )
            payload = {
                "source": "dailymed",
                "evidence_type": "drug_label",
                "title": "Aspirin and warfarin bleeding risk",
                "text": "Aspirin and warfarin coadministration may increase bleeding risk.",
                "task_type": "ddi_prediction",
                "dataset_name": "DDI",
                "endpoint": "ddi",
                "compound_names": ["Aspirin"],
                "pair_drug_names": ["CC(=O)OC1=CC=CC=C1C(=O)O", "CC(=O)CC(C1=CC=CC=C1)C1=C(O)C2=CC=CC=C2OC1=O"],
            }
            (pair_dir / "backfill_pair.jsonl").write_text(json.dumps(payload) + "\n", encoding="utf-8")
            docs = list(iter_structured_evidence_documents([root]))
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0]["pair_drug_names"], ["Aspirin", "Warfarin"])
        self.assertEqual(docs[0]["pair_drug_ids"], ["db00945", "db00682"])
        self.assertIn("CC(=O)OC1=CC=CC=C1C(=O)O", docs[0]["pair_drug_smiles"])
        self.assertEqual(docs[0]["evidence_scope"], "ddi_pair")
        self.assertIn("aspirin", docs[0]["text_entity_terms"])
        self.assertIn("warfarin", docs[0]["text_entity_terms"])

    def test_structured_evidence_does_not_use_pair_metadata_as_text_grounding(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pair_dir = root / "pairs" / "aspirin__warfarin"
            pair_dir.mkdir(parents=True)
            payload = {
                "source": "dailymed",
                "evidence_type": "drug_label",
                "title": "aspirin label",
                "text": "Aspirin may increase bleeding risk when used with other anticoagulant therapy.",
                "task_type": "ddi_prediction",
                "dataset_name": "DDI",
                "endpoint": "ddi",
                "compound_names": ["aspirin"],
                "pair_drug_names": ["aspirin", "warfarin"],
                "compound_specific": True,
                "endpoint_specific": True,
                "evidence_scope": "ddi_pair",
            }
            (pair_dir / "backfill_pair.jsonl").write_text(json.dumps(payload) + "\n", encoding="utf-8")
            docs = list(iter_structured_evidence_documents([root]))
        self.assertEqual(len(docs), 1)
        self.assertIn("aspirin", docs[0]["text_entity_terms"])
        self.assertNotIn("warfarin", docs[0]["text_entity_terms"])
        self.assertFalse(docs[0]["compound_specific"])
        self.assertTrue(docs[0]["endpoint_specific"])
        self.assertEqual(docs[0]["evidence_scope"], "endpoint_background")

    def test_dailymed_markdown_task_header_is_not_endpoint_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task_dir = root / "tasks" / "Bioavailability_Ma"
            task_dir.mkdir(parents=True)
            (task_dir / "dailymed_labels.md").write_text(
                """# DailyMed Label Evidence for Bioavailability_Ma

- source: DailyMed
- task: Bioavailability_Ma

## prochlorperazine: PROCHLORPERAZINE TABLET

- source: DailyMed
- task: Bioavailability_Ma
- query: prochlorperazine
- setid: one

### Selected label sections

#### INDICATIONS

Prochlorperazine is indicated for severe nausea and vomiting.
""",
                encoding="utf-8",
            )
            docs = list(iter_structured_evidence_documents([root]))
        self.assertEqual(len(docs), 1)
        self.assertTrue(docs[0]["compound_specific"])
        self.assertFalse(docs[0]["endpoint_specific"])
        self.assertEqual(docs[0]["evidence_scope"], "compound_background")
        self.assertNotIn("task: Bioavailability_Ma", docs[0]["source_body"])

    def test_dailymed_markdown_source_body_can_be_endpoint_specific(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task_dir = root / "tasks" / "Bioavailability_Ma"
            task_dir.mkdir(parents=True)
            (task_dir / "dailymed_labels.md").write_text(
                """# DailyMed Label Evidence for Bioavailability_Ma

- source: DailyMed
- task: Bioavailability_Ma

## example drug: EXAMPLE DRUG TABLET

- source: DailyMed
- task: Bioavailability_Ma
- query: example drug
- setid: two

### Selected label sections

#### CLINICAL PHARMACOLOGY

Example Drug has an absolute oral bioavailability of 72 percent.
""",
                encoding="utf-8",
            )
            docs = list(iter_structured_evidence_documents([root]))
        self.assertEqual(len(docs), 1)
        self.assertTrue(docs[0]["compound_specific"])
        self.assertTrue(docs[0]["endpoint_specific"])
        self.assertEqual(docs[0]["evidence_scope"], "compound_endpoint")

    def test_backfill_manifest_query_selection_avoids_db_and_cid_placeholders(self):
        admet = {"drug_name": "", "drug_id": "Drug 7", "smiles": "CCO", "target_id": "case_1"}
        self.assertEqual(_admet_query_from_target(admet), "CCO")
        ddi = {
            "drug_a": {"name": "DB001", "drug_id": "DB001", "smiles": "CCO"},
            "drug_b": {"name": "CID000002173", "drug_id": "CID000002173", "smiles": "CCC"},
        }
        self.assertEqual(_ddi_queries_from_target(ddi), ("CCO", "CCC"))


    def test_backfill_case_status_is_strict_about_empty_failures(self):
        self.assertEqual(_case_status([], [{"source": "pubchem", "error": "timeout"}]), "failed_no_documents")
        self.assertEqual(_case_status([], []), "no_documents")
        self.assertEqual(_case_status([{"source": "pubchem"}], []), "completed")
        self.assertEqual(_case_status([{"source": "pubchem"}], [{"source": "pubmed"}]), "completed_with_warnings")

    def test_backfill_http_session_can_ignore_environment_proxy(self):
        session = _new_http_session(http_retries=1, http_backoff_factor=0.1, no_env_proxy=True)
        self.assertFalse(session.trust_env)
        proxied = _new_http_session(http_retries=1, http_backoff_factor=0.1, no_env_proxy=False)
        self.assertTrue(proxied.trust_env)

    def test_public_database_cache_tracks_positive_and_negative_hits(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = PublicDatabaseCache(Path(tmp))
            self.assertIsNone(cache.get("pubchem", "aspirin"))
            cache.set("pubchem", "aspirin", {"ok": True, "payload": {"cid": 2244}})
            self.assertEqual(cache.get("pubchem", "aspirin")["payload"]["cid"], 2244)
            cache.set("pubchem", "missing", {"ok": False, "error": "HTTP 404"})
            self.assertFalse(cache.get("pubchem", "missing")["ok"])
        self.assertEqual(cache.stats["hits"], 2)
        self.assertEqual(cache.stats["negative_hits"], 1)
        self.assertEqual(cache.stats["writes"], 2)










    def test_evidence_relevance_audit_passes_compound_endpoint_admet_case(self):
        docs = [
            {
                "chunk_id": "aspirin_dili_1",
                "source": "pubmed",
                "endpoint": "dili",
                "endpoint_terms": ["dili", "drug-induced liver injury"],
                "entity_terms": ["aspirin", "2244"],
                "compound_specific": True,
                "endpoint_specific": True,
                "evidence_scope": "compound_endpoint",
                "text": "aspirin drug-induced liver injury evidence",
            }
        ]
        row = audit_case(docs, parse_case("aspirin|2244:dili"), min_docs=1)
        self.assertEqual(row["status"], "pass")
        self.assertEqual(row["n_compound_endpoint_grounded"], 1)

    def test_evidence_relevance_audit_requires_both_drugs_for_ddi(self):
        docs = [
            {
                "chunk_id": "aspirin_only",
                "source": "dailymed",
                "endpoint": "ddi",
                "endpoint_terms": ["ddi", "drug-drug interaction"],
                "entity_terms": ["aspirin", "2244"],
                "compound_specific": True,
                "endpoint_specific": True,
                "text": "aspirin drug-drug interaction warning",
            },
            {
                "chunk_id": "pair",
                "source": "pubmed",
                "endpoint": "ddi",
                "endpoint_terms": ["ddi", "drug-drug interaction"],
                "entity_terms": ["aspirin", "2244", "warfarin", "54678486"],
                "pair_drug_names": ["aspirin", "warfarin"],
                "compound_specific": True,
                "endpoint_specific": True,
                "evidence_scope": "ddi_pair",
                "text": "aspirin and warfarin drug-drug interaction evidence",
            },
        ]
        row = audit_case(docs, parse_case("aspirin|2244+warfarin|54678486:ddi"), min_docs=1)
        self.assertEqual(row["status"], "pass")
        self.assertEqual(row["examples"][0]["chunk_id"], "pair")

    def test_evidence_relevance_audit_strict_returned_flags_ungrounded_doc(self):
        docs = [
            {
                "chunk_id": "grounded",
                "source": "pubmed",
                "endpoint": "herg",
                "endpoint_terms": ["herg"],
                "entity_terms": ["caffeine"],
                "compound_specific": True,
                "endpoint_specific": True,
                "text": "caffeine hERG cardiac arrhythmia evidence",
            },
            {
                "chunk_id": "wrong_drug",
                "source": "pubmed",
                "endpoint": "herg",
                "endpoint_terms": ["herg"],
                "entity_terms": ["aspirin"],
                "compound_specific": True,
                "endpoint_specific": True,
                "text": "aspirin hERG cardiac arrhythmia evidence",
            },
        ]
        row = audit_case(docs, parse_case("caffeine:herg"), min_docs=1, strict_returned=True)
        self.assertEqual(row["status"], "returned_evidence_not_grounded")
        self.assertEqual(row["n_returned_not_grounded"], 1)

    def test_hybrid_retrieval_audit_filters_wrong_endpoint_and_keeps_grounded_hit(self):
        docs = [
            {
                "chunk_id": "aspirin_identity",
                "source": "pubchem",
                "endpoint": "dili",
                "endpoint_terms": ["dili"],
                "entity_terms": ["aspirin", "2244"],
                "compound_specific": True,
                "endpoint_specific": False,
                "text": "aspirin identity",
            },
            {
                "chunk_id": "aspirin_dili",
                "source": "pubmed",
                "endpoint": "dili",
                "endpoint_terms": ["dili", "drug-induced liver injury"],
                "entity_terms": ["aspirin", "2244"],
                "compound_specific": True,
                "endpoint_specific": True,
                "evidence_scope": "compound_endpoint",
                "text": "aspirin drug-induced liver injury evidence",
            },
            {
                "chunk_id": "aspirin_herg",
                "source": "pubmed",
                "endpoint": "herg",
                "endpoint_terms": ["herg"],
                "entity_terms": ["aspirin", "2244"],
                "compound_specific": True,
                "endpoint_specific": True,
                "text": "aspirin hERG evidence",
            },
        ]
        row = audit_local_retrieval_case(docs, parse_case("aspirin|2244:dili"), top_k=5, min_docs=1)
        self.assertEqual(row["status"], "pass")
        self.assertEqual(row["n_filter_passed"], 1)
        self.assertEqual(row["top_hits"][0]["chunk_id"], "aspirin_dili")

    def test_hybrid_retrieval_audit_requires_both_ddi_pair_sides(self):
        docs = [
            {
                "chunk_id": "aspirin_only",
                "source": "dailymed",
                "endpoint": "ddi",
                "endpoint_terms": ["ddi"],
                "entity_terms": ["aspirin", "2244"],
                "compound_specific": True,
                "endpoint_specific": True,
                "evidence_scope": "compound_endpoint",
                "text": "aspirin drug interaction warning",
            },
            {
                "chunk_id": "aspirin_warfarin",
                "source": "pubmed",
                "endpoint": "ddi",
                "endpoint_terms": ["ddi", "drug-drug interaction"],
                "entity_terms": ["aspirin", "2244", "warfarin", "54678486"],
                "pair_drug_names": ["aspirin", "warfarin"],
                "compound_specific": True,
                "endpoint_specific": True,
                "evidence_scope": "ddi_pair",
                "text": "aspirin warfarin drug-drug interaction evidence",
            },
        ]
        row = audit_local_retrieval_case(docs, parse_case("aspirin|2244+warfarin|54678486:ddi"), top_k=5, min_docs=1)
        self.assertEqual(row["status"], "pass")
        self.assertEqual(row["n_filter_passed"], 1)
        self.assertEqual(row["top_hits"][0]["chunk_id"], "aspirin_warfarin")




    def test_evidence_acceptance_suite_passes_dense_grounded_cases(self):
        args = Namespace(min_docs=1, top_k=5, require_dense=True, strict_scope=True, require_citation=True)
        rows = [
            {
                "label": "aspirin:ames",
                "endpoint": "ames",
                "status": "pass",
                "n_entity_related": 1,
                "n_endpoint_related": 1,
                "n_compound_endpoint_grounded": 1,
                "n_returned_not_grounded": 0,
                "retrieval_backend": "elasticsearch_hybrid_bm25_dense",
                "retrieval_metadata": {"dense_status": "completed"},
                "source_counts": {"pubmed": 1},
                "evidence_scope_counts": {"compound_endpoint": 1},
                "top_hits": [{"chunk_id": "hit", "source": "pubmed", "source_path": "https://pubmed.ncbi.nlm.nih.gov/1/", "evidence_scope": "compound_endpoint"}],
            },
            {
                "label": "aspirin+warfarin:ddi",
                "endpoint": "ddi",
                "status": "pass",
                "n_entity_related": 1,
                "n_endpoint_related": 1,
                "n_compound_endpoint_grounded": 1,
                "n_returned_not_grounded": 0,
                "retrieval_backend": "elasticsearch_hybrid_bm25_dense",
                "retrieval_metadata": {"dense_status": "completed"},
                "source_counts": {"pubmed": 1},
                "evidence_scope_counts": {"ddi_pair": 1},
                "top_hits": [{"chunk_id": "pair", "source": "pubmed", "source_path": "https://pubmed.ncbi.nlm.nih.gov/2/", "evidence_scope": "ddi_pair"}],
            },
        ]
        summary = build_acceptance_summary(rows, backend_info={"backend": "elasticsearch_hybrid"}, args=args)
        self.assertEqual(summary["status"], "pass")
        self.assertEqual(summary["metrics"]["dense_completed_cases"], 2)

    def test_evidence_acceptance_suite_rejects_missing_dense_or_wrong_scope(self):
        args = Namespace(min_docs=1, top_k=5, require_dense=True, strict_scope=True, require_citation=True)
        rows = [
            {
                "label": "aspirin+warfarin:ddi",
                "endpoint": "ddi",
                "status": "pass",
                "n_compound_endpoint_grounded": 1,
                "n_returned_not_grounded": 0,
                "retrieval_backend": "elasticsearch_bm25",
                "retrieval_metadata": {"dense_status": "missing_dense_vector_mapping"},
                "top_hits": [{"chunk_id": "single_side", "evidence_scope": "compound_endpoint"}],
            }
        ]
        summary = build_acceptance_summary(rows, backend_info={"backend": "elasticsearch_hybrid"}, args=args)
        self.assertEqual(summary["status"], "needs_attention")
        failures = summary["cases"][0]["failures"]
        self.assertTrue(any("dense_status" in item for item in failures))
        self.assertTrue(any("unexpected_scope" in item for item in failures))
        self.assertTrue(any("missing_source" in item for item in failures))


def _write_csv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    unittest.main()
