from __future__ import annotations

import os
from types import SimpleNamespace
import unittest

from agentm.runtime.agents.orchestrator import LLMOrchestratorPlanner
from agentm.runtime.graph.nodes import MultiAgentGraphNodes
from agentm.runtime.guards import GuardPolicy, GuardViolation, verify_citations, verify_prediction_grounding, verify_report_schema
from agentm.runtime.retrieval import HybridRetriever
from agentm.runtime.state import initial_context
from agentm.runtime.tools import AgentAction, MultiAgentToolRegistry, ToolResult, ToolSpec
from agentm.runtime.tools.scientific import ScientificToolRuntime
from agentm.agents.input_agent import InputAgent


class AlwaysWriteArtifactsOrchestrator:
    def next_action(self, context):
        return AgentAction("ReporterAgent", "write_artifacts", rationale="Final write.")


class StateGuardVerifierTests(unittest.TestCase):
    def _light_scientific_runtime(self):
        runtime = ScientificToolRuntime.__new__(ScientificToolRuntime)
        runtime.input_agent = InputAgent()
        return runtime

    def test_global_context_initializes_json_friendly_fields(self):
        context = initial_context("aspirin DILI", task_type="admet_prediction", dataset_name="DILI")
        self.assertEqual(context["metadata"]["runtime"], "agentm_guarded_multi_agent")
        self.assertEqual(context["task_graph"]["status"], "uninitialized")
        self.assertEqual(context["predictions"], [])
        self.assertEqual(context["agent_messages"], [])

    def test_agent_allowlist_blocks_cross_agent_prediction_tool(self):
        policy = GuardPolicy(agent_allowlists={"RetrieveAgent": ["retrieve_local_rag"], "GaugeAgent": ["predict_single"]})
        context = initial_context("aspirin")
        context["molecules"] = [object()]
        context["features"] = [object()]
        with self.assertRaises(GuardViolation):
            policy.validate_action({"agent": "RetrieveAgent", "tool": "predict_single", "args": {}}, context)

    def test_guard_blocks_fabricated_prediction_fields(self):
        policy = GuardPolicy(agent_allowlists={"ReasoningAgent": ["build_structured_explanation"]})
        context = initial_context("aspirin")
        context["predictions"] = [object()]
        with self.assertRaises(GuardViolation):
            policy.validate_action(
                {"agent": "ReasoningAgent", "tool": "build_structured_explanation", "args": {"label": 1, "score": 0.9}},
                context,
            )

    def test_guard_blocks_fabricated_citation_fields(self):
        policy = GuardPolicy(agent_allowlists={"ReasoningAgent": ["build_structured_explanation"]})
        context = initial_context("aspirin")
        context["predictions"] = [object()]
        with self.assertRaises(GuardViolation):
            policy.validate_action(
                {"agent": "ReasoningAgent", "tool": "build_structured_explanation", "args": {"citation_id": "fake"}},
                context,
            )

    def test_general_profile_requires_external_database_lookup_before_downstream_steps(self):
        policy = GuardPolicy(
            agent_allowlists={
                "GaugeAgent": ["build_features"],
                "RetrieveAgent": ["retrieve_external_evidence", "retrieve_local_rag"],
                "ReporterAgent": ["build_runtime_report"],
            }
        )
        context = initial_context("阿司匹林 这个药物怎样？", task_type="general_drug_profile", dataset_name="general")
        context["metadata"]["enable_fallbacks"] = True
        context["molecules"] = [{"name": "aspirin", "is_valid": True}]
        with self.assertRaises(GuardViolation) as blocked:
            policy.validate_action({"agent": "GaugeAgent", "tool": "build_features", "args": {}}, context)
        self.assertEqual(str(blocked.exception), "general_profile_requires_retrieve_external_evidence")
        context["tool_calls"] = [{"agent": "RetrieveAgent", "tool": "retrieve_external_evidence", "status": "failed"}]
        policy.validate_action({"agent": "GaugeAgent", "tool": "build_features", "args": {}}, context)

    def test_general_profile_guard_blocks_endpoint_prediction_tools(self):
        policy = GuardPolicy(
            agent_allowlists={
                "GaugeAgent": ["discover_rules", "predict_single", "predict_ddi", "assess_confidence", "run_zero_shot_fallback"],
                "VerifierAgent": ["verify_prediction_grounding"],
            }
        )
        context = initial_context("阿司匹林 这个药物怎样？", task_type="general_drug_profile", dataset_name="general")
        context["metadata"]["enable_fallbacks"] = True
        context["molecules"] = [{"name": "aspirin", "is_valid": True}]
        context["features"] = [{"rdkit": {}}]
        context["tool_calls"] = [{"agent": "RetrieveAgent", "tool": "retrieve_external_evidence", "status": "completed"}]
        for agent, tool in [
            ("GaugeAgent", "discover_rules"),
            ("GaugeAgent", "predict_single"),
            ("GaugeAgent", "predict_ddi"),
            ("GaugeAgent", "assess_confidence"),
            ("GaugeAgent", "run_zero_shot_fallback"),
            ("VerifierAgent", "verify_prediction_grounding"),
        ]:
            with self.subTest(tool=tool), self.assertRaises(GuardViolation) as blocked:
                policy.validate_action({"agent": agent, "tool": tool, "args": {}}, context)
            self.assertEqual(str(blocked.exception), "general_profile_endpoint_prediction_not_requested")

    def test_intent_gate_blocks_casual_chinese_request_before_molecule_resolution(self):
        runtime = self._light_scientific_runtime()
        context = initial_context("今天怎么样？")
        result = runtime.route_task({}, context)
        self.assertEqual(result.status, "completed")
        self.assertEqual(context["task_type"], "out_of_domain")
        self.assertEqual(context["dataset_name"], "none")
        self.assertEqual(context["molecule_mentions"], [])
        node_ids = {node["id"] for node in context["task_graph"]["nodes"]}
        self.assertNotIn("resolve_molecule", node_ids)
        self.assertEqual(context["metadata"]["intent_gate"]["status"], "out_of_domain")

    def test_intent_gate_requires_entity_for_pronoun_drug_prompt(self):
        runtime = self._light_scientific_runtime()
        context = initial_context("这个药安全吗？")
        runtime.route_task({}, context)
        self.assertEqual(context["task_type"], "missing_entity")
        self.assertEqual(context["molecule_mentions"], [])
        self.assertEqual(context["metadata"]["intent_gate"]["reason"], "scientific_intent_without_entity")

    def test_intent_gate_keeps_known_general_profile_in_domain(self):
        runtime = self._light_scientific_runtime()
        request = runtime._normalize_request_for_parsing("阿司匹林怎么样？")
        parsed = runtime.input_agent.parse(request)
        self.assertEqual(runtime._preflight_scientific_intent(request, explicit_task=False, explicit_dataset=False)["status"], "in_domain")
        self.assertTrue(
            runtime._is_general_drug_profile_request(
                request,
                parsed_task=parsed.task_type,
                parsed_dataset=parsed.dataset_name,
                explicit_task=False,
                explicit_dataset=False,
            )
        )

    def test_intent_gate_keeps_endpoint_prediction_in_domain(self):
        runtime = self._light_scientific_runtime()
        request = runtime._normalize_request_for_parsing("aspirin BBBP")
        preflight = runtime._preflight_scientific_intent(request, explicit_task=False, explicit_dataset=False)
        self.assertEqual(preflight["status"], "in_domain")

    def test_route_task_prefers_explicit_seeded_drugs_over_instruction_words(self):
        runtime = self._light_scientific_runtime()
        runtime._update_resource_resolution = lambda context: None
        runtime._task_graph_for = lambda context: {"status": "planned", "nodes": [], "edges": []}
        request = (
            "Evaluate the DDI risk for fluoxetine and metoprolol; also assess hERG risk for "
            "fluoxetine and DILI risk for metoprolol. Keep pair-level and single-drug evidence separate."
        )
        context = initial_context(request, task_type="ddi_prediction", dataset_name="DDI")

        result = runtime.route_task({}, context)

        self.assertEqual(result.status, "completed")
        self.assertEqual(context["molecule_mentions"], ["fluoxetine", "metoprolol"])
        resolution = context["metadata"]["molecule_mention_resolution"]
        self.assertEqual(resolution["strategy"], "explicit_local_identity_seed")
        self.assertEqual(resolution["selected_mentions"], ["fluoxetine", "metoprolol"])

    def test_local_seed_scan_deduplicates_brand_and_generic_mentions(self):
        runtime = self._light_scientific_runtime()
        mentions = runtime._local_seed_mentions_in_text(
            "Assess Prozac with metoprolol, then restate the fluoxetine endpoint."
        )
        self.assertEqual(mentions, ["fluoxetine", "metoprolol"])

    def test_verifier_requires_gauge_prediction_trace(self):
        context = initial_context("aspirin")
        context["predictions"] = [{"label": 0, "score": 0.1}]
        context["tool_calls"] = [{"agent": "ReasoningAgent", "tool": "build_structured_explanation", "status": "completed"}]
        report = verify_prediction_grounding(context)
        self.assertFalse(report["prediction_grounded"])
        context["tool_calls"].append({"agent": "GaugeAgent", "tool": "predict_single", "status": "completed"})
        self.assertTrue(verify_prediction_grounding(context)["prediction_grounded"])

    def test_verifier_checks_citation_map_against_retrieved_context(self):
        context = initial_context("aspirin")
        context["artifacts"] = {"retrieved_context": [{"chunk_id": "c1", "source": "pubmed"}]}
        context["evidence"] = [{"claim": "claim", "source": "local_rag", "chunk_id": "c1"}]
        report = verify_citations(context)
        self.assertTrue(report["citation_grounded"])
        self.assertEqual(report["citation_map"][0]["claim_id"], "claim_1")

    def test_verifier_flags_low_relevance_and_missing_compound_specific_evidence(self):
        context = initial_context("aspirin BBBP", task_type="property_prediction", dataset_name="BBBP")
        context["molecules"] = [{"name": "aspirin"}]
        context["artifacts"] = {"retrieved_context": [{"chunk_id": "c1", "source": "pubmed"}]}
        context["evidence"] = [
            {
                "claim": "A kinase inhibitor was studied in an unrelated oncology model.",
                "source": "rag_literature",
                "value": {"chunk_id": "c1", "snippet": "A kinase inhibitor was studied in an unrelated oncology model."},
            }
        ]
        report = verify_citations(context)
        self.assertTrue(report["citation_grounded"])
        self.assertFalse(report["evidence_relevance_grounded"])
        self.assertFalse(report["compound_specific_grounded"])
        self.assertFalse(report["endpoint_specific_grounded"])
        self.assertIn("low_relevance_evidence:1", report["warnings"])
        self.assertIn("compound_specific_evidence_missing", report["warnings"])
        self.assertIn("endpoint_specific_evidence_missing", report["warnings"])

    def test_verifier_ignores_query_metadata_for_evidence_relevance(self):
        context = initial_context("aspirin BBBP", task_type="property_prediction", dataset_name="BBBP")
        context["molecules"] = [{"name": "aspirin"}]
        context["artifacts"] = {"retrieved_context": [{"chunk_id": "c1", "source": "pubmed"}]}
        context["evidence"] = [
            {
                "claim": "A kinase inhibitor was studied in an unrelated oncology model.",
                "source": "rag_literature",
                "value": {
                    "chunk_id": "c1",
                    "snippet": "A kinase inhibitor was studied in an unrelated oncology model.",
                    "query": "aspirin BBBP property_prediction evidence",
                    "dataset_name": "BBBP",
                },
            }
        ]
        report = verify_citations(context)
        self.assertTrue(report["citation_grounded"])
        self.assertFalse(report["evidence_relevance_grounded"])
        self.assertFalse(report["compound_specific_grounded"])
        self.assertFalse(report["endpoint_specific_grounded"])

    def test_verifier_separates_compound_and_endpoint_grounding(self):
        context = initial_context("aspirin BBBP", task_type="property_prediction", dataset_name="BBBP")
        context["molecules"] = [{"name": "aspirin"}]
        context["artifacts"] = {"retrieved_context": [{"chunk_id": "c1", "source": "pubmed"}]}
        context["evidence"] = [
            {
                "claim": "Aspirin has anti-inflammatory pharmacology but this record only discusses analgesic use.",
                "source": "rag_literature",
                "chunk_id": "c1",
                "source_path": "/tmp/source",
            }
        ]
        report = verify_citations(context)
        self.assertTrue(report["compound_specific_grounded"])
        self.assertFalse(report["endpoint_specific_grounded"])
        self.assertIn("endpoint_specific_evidence_missing", report["warnings"])

    def test_verifier_accepts_structured_compound_endpoint_metadata(self):
        context = initial_context("aspirin BBBP", task_type="property_prediction", dataset_name="BBBP")
        context["molecules"] = [{"name": "aspirin"}]
        context["artifacts"] = {"retrieved_context": [{"chunk_id": "c1", "source": "pubmed"}]}
        context["evidence"] = [
            {
                "claim": "Aspirin BBBP evidence from a structured evidence index.",
                "source": "elasticsearch",
                "chunk_id": "c1",
                "source_path": "/tmp/source",
                "value": {
                    "chunk_id": "c1",
                    "snippet": "Aspirin has BBBP evidence in the structured index.",
                    "compound_specific": True,
                    "endpoint_specific": True,
                    "endpoint": "bbbp",
                    "entity_terms": ["aspirin"],
                    "endpoint_terms": ["bbbp", "blood-brain barrier"],
                },
            }
        ]
        report = verify_citations(context)
        self.assertTrue(report["compound_specific_grounded"])
        self.assertTrue(report["endpoint_specific_grounded"])
        self.assertEqual(report["warnings"], [])

    def test_verifier_accepts_top_level_structured_evidence_metadata(self):
        context = initial_context("aspirin BBBP", task_type="property_prediction", dataset_name="BBBP")
        context["molecules"] = [{"name": "aspirin"}]
        context["artifacts"] = {"retrieved_context": [{"chunk_id": "c1", "source": "pubmed"}]}
        context["evidence"] = [
            {
                "claim": "Aspirin BBBP evidence from a curated card.",
                "source": "rag_literature",
                "chunk_id": "c1",
                "source_path": "/tmp/source",
                "compound_specific": True,
                "endpoint_specific": True,
                "endpoint": "bbbp",
                "entity_terms": ["aspirin"],
                "endpoint_terms": ["bbbp", "blood-brain barrier"],
                "value": {"chunk_id": "c1", "snippet": "Aspirin and BBBP are both present in this evidence."},
            }
        ]
        report = verify_citations(context)
        self.assertTrue(report["compound_specific_grounded"])
        self.assertTrue(report["endpoint_specific_grounded"])
        self.assertEqual(report["warnings"], [])

    def test_verifier_does_not_accept_endpoint_terms_without_endpoint_specific_flag(self):
        context = initial_context("aspirin DILI", task_type="admet_prediction", dataset_name="DILI")
        context["molecules"] = [{"name": "aspirin"}]
        context["artifacts"] = {"retrieved_context": [{"chunk_id": "c1", "source": "pubchem"}]}
        context["evidence"] = [
            {
                "claim": "Aspirin identity from PubChem. CID 2244.",
                "source": "pubchem",
                "chunk_id": "c1",
                "source_path": "/tmp/pubchem",
                "value": {
                    "chunk_id": "c1",
                    "snippet": "Aspirin identity from PubChem. CID 2244.",
                    "compound_specific": True,
                    "endpoint_specific": False,
                    "endpoint": "dili",
                    "entity_terms": ["aspirin", "2244"],
                    "endpoint_terms": ["dili", "hepatotoxicity"],
                },
            }
        ]
        report = verify_citations(context)
        self.assertTrue(report["compound_specific_grounded"])
        self.assertFalse(report["endpoint_specific_grounded"])
        self.assertIn("endpoint_specific_evidence_missing", report["warnings"])

    def test_verifier_flags_mixed_non_endpoint_specific_evidence(self):
        context = initial_context("aspirin DILI", task_type="admet_prediction", dataset_name="DILI")
        context["molecules"] = [{"name": "aspirin"}]
        context["artifacts"] = {"retrieved_context": [{"chunk_id": "c1"}, {"chunk_id": "c2"}]}
        context["evidence"] = [
            {
                "claim": "Aspirin identity from PubChem. CID 2244.",
                "source": "pubchem",
                "chunk_id": "c1",
                "source_path": "/tmp/pubchem",
                "value": {"chunk_id": "c1", "compound_specific": True, "endpoint_specific": False},
            },
            {
                "claim": "Aspirin hepatotoxicity evidence discusses DILI.",
                "source": "pubmed",
                "chunk_id": "c2",
                "source_path": "/tmp/pubmed",
                "value": {"chunk_id": "c2", "compound_specific": True, "endpoint_specific": True, "endpoint": "dili"},
            },
        ]
        report = verify_citations(context)
        self.assertFalse(report["endpoint_specific_grounded"])
        self.assertEqual(report["n_non_endpoint_specific"], 1)
        self.assertIn("non_endpoint_specific_evidence:1", report["warnings"])

    def test_report_schema_verifier_core_fields(self):
        context = initial_context("aspirin")
        context["final_report"] = {field: None for field in ["request", "task_type", "molecules", "predictions", "evidence", "explanation", "artifacts", "warnings", "trace", "reasoning"]}
        self.assertTrue(verify_report_schema(context)["schema_complete"])

    def test_registry_exposes_agent_allowlists(self):
        registry = MultiAgentToolRegistry()
        registry.register(ToolSpec("route_task", "MoleculeAgent", "route", {}, lambda args, state: ToolResult("route_task", "completed")))
        self.assertEqual(registry.allowlist_for_agent("MoleculeAgent"), ["route_task"])
        self.assertIn("MoleculeAgent", registry.agent_allowlists())

    def test_general_profile_llm_plan_removes_endpoint_prediction_nodes(self):
        registry = MultiAgentToolRegistry()
        for agent in ["MoleculeAgent", "RetrieveAgent", "GaugeAgent", "ReasoningAgent", "ReporterAgent", "VerifierAgent"]:
            registry.register(ToolSpec(f"{agent}_noop", agent, "noop", {}, lambda args, state: ToolResult("noop", "completed")))
        planner = LLMOrchestratorPlanner(registry=registry, llm_client=object())
        context = initial_context("阿司匹林 这个药物怎样？", task_type="general_drug_profile", dataset_name="general")
        payload = {
            "objective": "general aspirin profile",
            "task_graph": {
                "nodes": [
                    {"id": "route", "agent": "MoleculeAgent", "goal": "Parse request intent and identify target molecule."},
                    {"id": "retrieve", "agent": "RetrieveAgent", "goal": "Retrieve public database context for aspirin."},
                    {"id": "predict", "agent": "GaugeAgent", "goal": "Run single-molecule prediction to derive pharmacological and safety attributes."},
                    {"id": "reason", "agent": "ReasoningAgent", "goal": "Synthesize evidence into an explanation."},
                    {"id": "verify_prediction", "agent": "VerifierAgent", "goal": "Verify that predictions originate from valid GaugeAgent tool traces."},
                    {"id": "verify_schema", "agent": "VerifierAgent", "goal": "Verify report schema."},
                    {"id": "write", "agent": "ReporterAgent", "goal": "Write final artifacts."},
                ],
                "edges": [["route", "retrieve"], ["retrieve", "predict"], ["predict", "reason"]],
            },
        }
        plan = planner._normalize_plan(payload, context)
        nodes = plan["task_graph"]["nodes"]
        goals = " ".join(node["goal"].lower() for node in nodes)
        self.assertIn("build_features", {node["id"] for node in nodes})
        self.assertNotIn("single-molecule prediction", goals)
        self.assertFalse(any(node["id"] == "verify_prediction" for node in nodes))
        self.assertEqual(plan["task_graph"]["edges"], [[nodes[i]["id"], nodes[i + 1]["id"]] for i in range(len(nodes) - 1)])

    def test_hybrid_retriever_merges_bm25_and_dense_hits(self):
        retriever = HybridRetriever(use_dense=False)
        bm25_hits = [
            {"_id": "a", "_score": 3.0, "_source": {"chunk_id": "a", "text": "aspirin liver"}},
            {"_id": "b", "_score": 2.0, "_source": {"chunk_id": "b", "text": "DILI"}},
        ]
        dense_hits = [
            {"_id": "b", "_score": 1.9, "_source": {"chunk_id": "b", "text": "DILI"}},
            {"_id": "c", "_score": 1.5, "_source": {"chunk_id": "c", "text": "toxicity"}},
        ]
        merged = retriever._merge_ranked_hits(bm25_hits, dense_hits, top_k=3)
        self.assertEqual(merged[0]["_source"]["chunk_id"], "b")
        self.assertEqual(merged[0]["_agentm_channel"], "hybrid")
        self.assertIn("bm25", merged[0]["_agentm_component_scores"])
        self.assertIn("dense", merged[0]["_agentm_component_scores"])

    def test_hybrid_retriever_uses_env_index_name_when_not_explicit(self):
        old_value = os.environ.get("AGENTM_ES_INDEX")
        try:
            os.environ.pop("AGENTM_ES_INDEX", None)
            self.assertEqual(HybridRetriever(use_dense=False).index_name, "agentm_evidence_paper_qc_dense")
            os.environ["AGENTM_ES_INDEX"] = "agentm_evidence_paper_qc_dense"
            self.assertEqual(HybridRetriever(use_dense=False).index_name, "agentm_evidence_paper_qc_dense")
            self.assertEqual(HybridRetriever(index_name="explicit_index", use_dense=False).index_name, "explicit_index")
        finally:
            if old_value is None:
                os.environ.pop("AGENTM_ES_INDEX", None)
            else:
                os.environ["AGENTM_ES_INDEX"] = old_value

    def test_hybrid_retriever_detects_dense_mapping_through_single_index_alias(self):
        class Indices:
            @staticmethod
            def get_mapping(*, index):
                self.assertEqual(index, "evidence_alias")
                return {
                    "evidence_concrete": {
                        "mappings": {
                            "properties": {"embedding": {"type": "dense_vector"}}
                        }
                    }
                }

        class Client:
            indices = Indices()

        retriever = HybridRetriever(index_name="evidence_alias", use_dense=True)
        self.assertTrue(retriever._index_has_dense_vector(Client()))

    def test_hybrid_retriever_general_profile_excludes_ddi_scopes(self):
        retriever = HybridRetriever(use_dense=False)
        filters, _ = retriever._filters_and_boosts(
            {
                "profile_type": "general_drug_profile",
                "drug_names": ["aspirin"],
                "entity_terms": ["aspirin", "2244"],
            }
        )
        exclusion_filter = next(
            item
            for item in filters
            if item.get("bool", {}).get("must_not")
            and {"term": {"endpoint": "ddi"}} in item["bool"]["must_not"]
        )
        self.assertIn({"terms": {"evidence_scope": ["ddi_pair", "ddi_pair_background"]}}, exclusion_filter["bool"]["must_not"])
        ddi_source = {"endpoint": "ddi", "evidence_scope": "compound_endpoint", "entity_terms": ["aspirin"]}
        profile_source = {"endpoint": "", "evidence_scope": "compound_background", "entity_terms": ["aspirin"]}
        metadata = {"profile_type": "general_drug_profile", "entity_terms": ["aspirin"]}
        self.assertLess(
            retriever._metadata_rerank_bonus(ddi_source, metadata),
            retriever._metadata_rerank_bonus(profile_source, metadata),
        )

    def test_hybrid_retriever_reranks_compound_endpoint_hits(self):
        retriever = HybridRetriever(use_dense=False)
        bm25_hits = [
            {
                "_id": "background",
                "_score": 9.0,
                "_source": {"chunk_id": "background", "text": "blood brain barrier background", "endpoint": "bbbp"},
            },
            {
                "_id": "aspirin",
                "_score": 1.0,
                "_source": {
                    "chunk_id": "aspirin",
                    "text": "aspirin blood brain barrier evidence",
                    "endpoint": "bbbp",
                    "entity_terms": ["aspirin"],
                    "endpoint_terms": ["bbbp", "blood-brain barrier"],
                    "evidence_scope": "compound_endpoint",
                    "compound_specific": True,
                    "endpoint_specific": True,
                },
            },
        ]
        merged = retriever._merge_ranked_hits(bm25_hits, [], metadata={"endpoint": "bbbp", "entity_terms": ["aspirin"]}, top_k=2)
        self.assertEqual(merged[0]["_source"]["chunk_id"], "aspirin")
        self.assertGreater(merged[0]["_agentm_metadata_rerank_bonus"], 0.0)
        self.assertGreater(merged[0]["_agentm_rerank_score"], merged[1]["_agentm_rerank_score"])

    def test_hybrid_retriever_filters_wrong_compound_specific_hits(self):
        retriever = HybridRetriever(use_dense=False)
        filters, _ = retriever._filters_and_boosts(
            {
                "task_type": "admet_prediction",
                "endpoint": "herg",
                "drug_names": ["caffeine"],
                "entity_terms": ["caffeine", "2519"],
            }
        )
        entity_filter = next(item for item in filters if "bool" in item)
        should = entity_filter["bool"]["should"]
        self.assertNotIn({"term": {"compound_specific": False}}, should)
        self.assertIn({"terms": {"text_entity_terms": ["caffeine", "2519"]}}, should)
        self.assertEqual(entity_filter["bool"]["minimum_should_match"], 1)
        self.assertIn({"term": {"endpoint_specific": True}}, filters)

    def test_hybrid_retriever_requires_each_ddi_entity_group(self):
        retriever = HybridRetriever(use_dense=False)
        filters, _ = retriever._filters_and_boosts(
            {
                "task_type": "ddi_prediction",
                "endpoint": "ddi",
                "entity_groups": [["aspirin", "2244"], ["warfarin", "54678486"]],
            }
        )
        group_filter = next(item for item in filters if "bool" in item and "must" in item["bool"])
        self.assertEqual(len(group_filter["bool"]["must"]), 2)
        first_group = group_filter["bool"]["must"][0]["bool"]
        second_group = group_filter["bool"]["must"][1]["bool"]
        self.assertEqual(first_group["minimum_should_match"], 1)
        self.assertEqual(second_group["minimum_should_match"], 1)
        self.assertEqual(first_group["should"], [{"terms": {"text_entity_terms": ["aspirin", "2244"]}}])
        self.assertEqual(second_group["should"], [{"terms": {"text_entity_terms": ["warfarin", "54678486"]}}])

    def test_hybrid_retriever_reranks_ddi_pair_over_single_side_evidence(self):
        retriever = HybridRetriever(use_dense=False)
        bm25_hits = [
            {
                "_id": "aspirin_only",
                "_score": 9.0,
                "_source": {
                    "chunk_id": "aspirin_only",
                    "text": "aspirin drug interaction warning",
                    "endpoint": "ddi",
                    "endpoint_terms": ["ddi"],
                    "entity_terms": ["aspirin", "2244"],
                    "compound_specific": True,
                    "endpoint_specific": True,
                    "evidence_scope": "compound_endpoint",
                },
            },
            {
                "_id": "pair",
                "_score": 1.0,
                "_source": {
                    "chunk_id": "pair",
                    "text": "aspirin warfarin drug interaction evidence",
                    "endpoint": "ddi",
                    "endpoint_terms": ["ddi"],
                    "entity_terms": ["aspirin", "2244", "warfarin", "54678486"],
                    "pair_drug_names": ["aspirin", "warfarin"],
                    "compound_specific": True,
                    "endpoint_specific": True,
                    "evidence_scope": "ddi_pair",
                },
            },
        ]
        merged = retriever._merge_ranked_hits(
            bm25_hits,
            [],
            metadata={"endpoint": "ddi", "entity_groups": [["aspirin", "2244"], ["warfarin", "54678486"]]},
            top_k=2,
        )
        self.assertEqual(merged[0]["_source"]["chunk_id"], "pair")

    def test_scientific_runtime_retrieval_metadata_groups_ddi_molecules(self):
        runtime = object.__new__(ScientificToolRuntime)
        context = initial_context("aspirin warfarin DDI", task_type="ddi_prediction", dataset_name="DrugBank")
        context["molecules"] = [
            SimpleNamespace(
                name="aspirin",
                original_input="aspirin",
                canonical_smiles="CC(=O)OC1=CC=CC=C1C(=O)O",
                smiles="CC(=O)OC1=CC=CC=C1C(=O)O",
                synonyms=["acetylsalicylic acid"],
                metadata={"pubchem_cid": "2244", "chembl_id": "CHEMBL25"},
            ),
            SimpleNamespace(
                name="warfarin",
                original_input="warfarin",
                canonical_smiles="CC(=O)CC(C1=CC=CC=C1)C2=C(O)OC3=CC=CC=C3C2=O",
                smiles="CC(=O)CC(C1=CC=CC=C1)C2=C(O)OC3=CC=CC=C3C2=O",
                synonyms=[],
                metadata={"pubchem_cid": "54678486"},
            ),
        ]
        metadata = runtime._retrieval_metadata(context)
        self.assertEqual(metadata["endpoint"], "ddi")
        self.assertEqual(len(metadata["entity_groups"]), 2)
        self.assertIn("aspirin", metadata["entity_groups"][0])
        self.assertIn("2244", metadata["entity_groups"][0])
        self.assertIn("warfarin", metadata["entity_groups"][1])
        self.assertIn("54678486", metadata["entity_groups"][1])

    def test_orchestrator_loop_stops_after_successful_artifact_write(self):
        registry = MultiAgentToolRegistry()

        def write_handler(args, context):
            context["final_report"] = {"multi_agent": {"loop_status": context["metadata"].get("loop_status")}}
            return ToolResult("write_artifacts", "completed", {"artifact_count": 1})

        registry.register(ToolSpec("write_artifacts", "ReporterAgent", "write", {}, write_handler))
        nodes = MultiAgentGraphNodes(
            registry=registry,
            guard=GuardPolicy(agent_allowlists=registry.agent_allowlists(), max_rounds=5),
            orchestrator=AlwaysWriteArtifactsOrchestrator(),
            output_dir=".",
        )
        context = initial_context("aspirin")
        result = nodes.orchestrator_loop(context)
        self.assertEqual(result["metadata"]["loop_status"], "finished")
        self.assertEqual(len(result["tool_calls"]), 1)
        self.assertEqual(result["tool_calls"][0]["tool"], "write_artifacts")
        self.assertEqual(result["final_report"]["multi_agent"]["loop_status"], "finished")
        self.assertEqual(result["final_report"]["tool_calls"][0]["tool"], "write_artifacts")
        finalized = nodes.guarded_finalize(result)
        self.assertEqual(finalized["metadata"]["loop_status"], "finalized")
        self.assertEqual(finalized["final_report"]["multi_agent"]["loop_status"], "finalized")


if __name__ == "__main__":
    unittest.main()
