from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path
from threading import Lock
from unittest.mock import patch

from agentm.runtime.web.demo_server import (
    INDEX_HTML,
    WebDemoConfig,
    WebRunManager,
    artifact_url,
    compact_report,
    create_app,
    run_agent_request,
    _run_public_summary,
    _trace_timeline,
)


class WebDemoTests(unittest.TestCase):
    def test_timeline_uses_trace_when_tool_calls_are_empty(self):
        report = {
            "tool_calls": [],
            "trace": [{"step_id": "resolve", "status": "completed", "tool_name": "resolve_molecule"}],
            "plan": {"steps": [{"step_id": "resolve", "description": "Resolve molecule"}]},
        }
        result = _trace_timeline(report)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["tool_name"], "resolve_molecule")
        self.assertEqual(result[0]["status"], "completed")
        self.assertEqual(result[0]["description"], "Resolve molecule")

    def test_timeline_accepts_a_report_without_execution_records(self):
        self.assertEqual(_trace_timeline({}), [])

    def test_index_html_keeps_style_elements(self):
        self.assertIn("Recent Runs", INDEX_HTML)
        self.assertIn("Start Prediction", INDEX_HTML)
        self.assertIn('input name="run-mode" type="radio" value="fast"', INDEX_HTML)
        self.assertIn('input name="run-mode" type="radio" value="full"', INDEX_HTML)
        self.assertIn('input name="run-mode" type="radio" value="custom"', INDEX_HTML)
        self.assertIn("history-item", INDEX_HTML)
        self.assertIn("confidenceText", INDEX_HTML)
        self.assertIn('data-view="workflow"', INDEX_HTML)
        self.assertIn("function renderWorkflow(data)", INDEX_HTML)
        self.assertIn("workflowNodeGrid(lane.nodes || [])", INDEX_HTML)
        self.assertIn("Graph Structure", INDEX_HTML)
        self.assertIn("function workflowTopologyGraph(topology)", INDEX_HTML)
        self.assertIn("workflow-svg", INDEX_HTML)
        self.assertIn("workflow-arrow-orange", INDEX_HTML)
        self.assertIn("repair/backtrack route", INDEX_HTML)
        self.assertIn("SMILES encoder", INDEX_HTML)
        self.assertIn("Runtime-added", INDEX_HTML)
        self.assertIn("Release Index", INDEX_HTML)
        self.assertIn("Endpoint prediction", INDEX_HTML)
        self.assertIn("Completed", INDEX_HTML)
        self.assertIn("Prediction model", INDEX_HTML)
        self.assertIn("Citation", INDEX_HTML)
        self.assertIn("Claim-level verification", INDEX_HTML)
        self.assertIn("Bounded reflection", INDEX_HTML)
        self.assertIn("20-50 cases", INDEX_HTML)
        self.assertIn("function releaseInfo(data)", INDEX_HTML)
        self.assertIn("function capabilityInline(data)", INDEX_HTML)
        self.assertIn("function capabilityTable(data)", INDEX_HTML)
        self.assertIn("Capabilities", INDEX_HTML)
        self.assertIn("reaction_profile", INDEX_HTML)
        self.assertIn("reaction_yield_prediction", INDEX_HTML)
        self.assertIn("ReactionAgent", INDEX_HTML)
        self.assertIn("Reaction", INDEX_HTML)
        self.assertIn("function moleculeImageGallery(data)", INDEX_HTML)
        self.assertIn("Molecule Visualization", INDEX_HTML)
        self.assertIn("Key substructure", INDEX_HTML)
        self.assertIn("AgentM", INDEX_HTML)

    def test_fastapi_app_registers_expected_routes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            app = create_app(WebDemoConfig(project_root=root, output_dir=root / "outputs" / "web_demo"))
            paths = {route.path for route in app.routes}
        self.assertIn("/api/health", paths)
        self.assertIn("/api/runs", paths)
        self.assertIn("/api/runs/{run_id}", paths)
        self.assertIn("/api/runs/{run_id}/result", paths)
        self.assertIn("/artifacts/{encoded_path:path}", paths)

    def test_artifact_url_only_allows_output_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_dir = root / "outputs" / "web_demo"
            allowed = output_dir / "run" / "multi_agent_report.json"
            allowed.parent.mkdir(parents=True)
            allowed.write_text("{}", encoding="utf-8")
            release_manifest = root / "outputs" / "paper_evidence_qc" / "evidence_qc_readout.json"
            release_manifest.parent.mkdir(parents=True)
            release_manifest.write_text("{}", encoding="utf-8")
            blocked = root / "secret.txt"
            blocked.write_text("secret", encoding="utf-8")
            config = WebDemoConfig(project_root=root, output_dir=output_dir)
            self.assertIsNotNone(artifact_url(str(allowed), config))
            self.assertIsNotNone(artifact_url(str(release_manifest), config))
            self.assertIsNone(artifact_url(str(blocked), config))

    def test_compact_report_exposes_fields_and_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "outputs" / "web_demo" / "run1"
            run_dir.mkdir(parents=True)
            report_path = run_dir / "multi_agent_report.json"
            report_path.write_text("{}", encoding="utf-8")
            global_context_path = run_dir / "global_context.json"
            tool_calls_path = run_dir / "tool_calls.json"
            citation_map_path = run_dir / "citation_map.json"
            verifier_report_path = run_dir / "verifier_report.json"
            reflection_path = run_dir / "reflection.json"
            for path in [global_context_path, tool_calls_path, citation_map_path, verifier_report_path, reflection_path]:
                path.write_text("{}", encoding="utf-8")
            (run_dir / "web_runtime_config.json").write_text(
                json.dumps(
                    {
                        "feature_set": "all_features",
                        "use_smiles_encoder": True,
                        "use_mllm_visual_semantic": True,
                    }
                ),
                encoding="utf-8",
            )
            release_dir = root / "outputs" / "paper_evidence_qc"
            release_dir.mkdir(parents=True)
            (release_dir / "evidence_qc_readout.json").write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "rows": [
                            {
                                "stage": "Entity-normalized paper-QC dense index",
                                "backend": "Elasticsearch BM25 + dense + metadata rerank",
                                "documents": 1052,
                                "dense_completed": "50/50",
                                "strict_pass": "50/50",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            (release_dir / "es_acceptance_summary.json").write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "metrics": {"n_cases": 50, "case_status_counts": {"pass": 50}, "dense_completed_cases": 50},
                        "requirements": {"strict_scope": True, "require_citation": True},
                    }
                ),
                encoding="utf-8",
            )
            config = WebDemoConfig(project_root=root, output_dir=root / "outputs" / "web_demo")
            report = {
                "request": "aspirin DILI",
                "task_type": "admet_prediction",
                "molecules": [{"original_input": "aspirin", "canonical_smiles": "CC", "is_valid": True}],
                "predictions": [{"task_type": "admet_prediction", "dataset_name": "DILI", "label": 0, "score": 0.17, "model_type": "random_forest"}],
                "evidence": [{"source": "elasticsearch_hybrid", "claim": "DILI evidence", "chunk_id": "c1", "citation_id": "cite_1"}],
                "tool_calls": [
                    {"round": 1, "agent": "GaugeAgent", "tool": "predict_single", "status": "completed", "duration_sec": 0.01},
                    {"round": 2, "agent": "ReporterAgent", "tool": "write_artifacts", "status": "completed", "duration_sec": 0.01},
                ],
                "explanation": "grounded explanation",
                "reasoning": {"structured_explanation": "grounded explanation"},
                "multi_agent": {"planner_type": "llm", "reasoning_mode": "llm_expert", "loop_status": "running"},
                "verifier_report": {"summary": {"status": "pass"}},
                "objective_status": {"status": "pass"},
                "artifacts": {
                    "multi_agent_report_path": str(report_path),
                    "global_context_path": str(global_context_path),
                    "tool_calls_path": str(tool_calls_path),
                    "citation_map_path": str(citation_map_path),
                    "verifier_report_path": str(verifier_report_path),
                    "reflection_path": str(reflection_path),
                    "feature_schema": {
                        "feature_set": "all_features",
                        "metadata": {
                            "block_sources": {
                                "smiles": {"enabled": True, "dim": 384},
                                "imagemol": {"encoder": "disabled", "dim": 0},
                                "visual_semantic": {"source": "disabled", "dim": 0},
                            }
                        },
                    },
                    "task_graph": {
                        "status": "planned",
                        "nodes": [
                            {"id": "route_task", "agent": "MoleculeAgent", "goal": "route request"},
                            {"id": "predict_single", "agent": "GaugeAgent", "goal": "run prediction"},
                        ],
                        "edges": [["route_task", "predict_single"]],
                    },
                },
                "warnings": [],
            }
            compact = compact_report(report, run_dir=run_dir, config=config)
        self.assertEqual(compact["multi_agent"]["planner_type"], "llm")
        self.assertEqual(compact["objective_status"]["status"], "pass")
        self.assertEqual(compact["verifier_report"]["summary"]["status"], "pass")
        self.assertEqual(compact["quality_summary"]["verifier_status"], "pass")
        self.assertEqual(compact["quality_summary"]["objective_status"], "pass")
        self.assertTrue(compact["quality_summary"]["completed"])
        self.assertEqual(compact["quality_summary"]["completed_status"], "yes")
        self.assertEqual(compact["quality_summary"]["model_status"], "available")
        self.assertEqual(compact["quality_summary"]["prediction_mode"], "tool_grounded")
        self.assertEqual(compact["release_info"]["index_name"], "agentm_evidence_paper_qc_dense")
        self.assertEqual(compact["release_info"]["document_count"], 1052)
        self.assertEqual(compact["release_info"]["expanded_acceptance"], "50/50")
        self.assertEqual(compact["release_info"]["acceptance_cases"], 50)
        self.assertEqual(compact["release_info"]["acceptance_passing_cases"], 50)
        self.assertEqual(compact["capability_summary"]["feature_set_requested"], "all_features")
        capability_by_key = {item["key"]: item for item in compact["capability_summary"]["items"]}
        self.assertEqual(capability_by_key["smiles_encoder"]["status"], "requested_and_used")
        self.assertEqual(capability_by_key["imagemol"]["status"], "requested_not_used")
        self.assertEqual(capability_by_key["mllm_visual_semantic"]["status"], "requested_not_used")
        self.assertEqual(compact["evidence_cards"][0]["source_group"], "elasticsearch_hybrid")
        self.assertEqual(compact["trace_timeline"][0]["tool_name"], "predict_single")
        self.assertEqual(compact["workflow_graph"]["summary"]["planned_steps"], 2)
        self.assertEqual(compact["workflow_graph"]["summary"]["executed_steps"], 2)
        self.assertEqual(compact["workflow_graph"]["summary"]["runtime_added_steps"], 1)
        self.assertGreater(compact["workflow_graph"]["summary"]["available_tools"], 10)
        self.assertGreater(compact["workflow_graph"]["summary"]["inactive_tools"], 1)
        self.assertIn("LangGraph", compact["workflow_graph"]["runtime"])
        self.assertEqual(compact["workflow_graph"]["langgraph"]["nodes"][2]["status"], "completed")
        self.assertEqual(compact["workflow_graph"]["planned"]["nodes"][0]["label"], "1. MoleculeAgent")
        self.assertIn("full_topology", compact["workflow_graph"])
        self.assertIn("topology_mermaid", compact["workflow_graph"])
        self.assertTrue(any(node["status"] == "inactive" for node in compact["workflow_graph"]["full_topology"]["nodes"]))
        status_by_tool = {node["tool"]: node["status"] for node in compact["workflow_graph"]["full_topology"]["nodes"]}
        origin_by_tool = {node["tool"]: node.get("origin") for node in compact["workflow_graph"]["full_topology"]["nodes"]}
        self.assertEqual(status_by_tool["resolve_ddi_pair"], "inactive")
        self.assertEqual(origin_by_tool["predict_single"], "planned")
        self.assertEqual(origin_by_tool["write_artifacts"], "runtime_added")
        self.assertEqual(compact["explanation"], "grounded explanation")
        self.assertIsNone(compact["reasoning_summary"])
        self.assertIn("multi_agent_report_path", {item["key"] for item in compact["files"]})
        labels_by_key = {item["key"]: item["label"] for item in compact["files"]}
        self.assertEqual(labels_by_key["global_context_path"], "Global context (provenance state)")
        self.assertEqual(labels_by_key["tool_calls_path"], "Tool calls (audit trail)")
        self.assertEqual(labels_by_key["citation_map_path"], "Citation map (claim links)")
        self.assertEqual(labels_by_key["verifier_report_path"], "Verifier report (claim-level checks)")
        self.assertEqual(labels_by_key["reflection_path"], "Reflection (bounded repair)")
        self.assertIn("release_qc_readout_path", {item["key"] for item in compact["files"]})
        self.assertIn("release_acceptance_summary_path", {item["key"] for item in compact["files"]})
        image_keys = {item["key"] for item in compact["images"]}
        self.assertIn("image_path", image_keys)
        self.assertIn("key_substructure_image_path", image_keys)
        self.assertIn("key_substructure_metadata_path", {item["key"] for item in compact["files"]})
        self.assertTrue(all(item["exists"] for item in compact["images"]))

    def test_compact_report_infers_artifact_write_for_existing_reports(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "outputs" / "web_demo" / "run1"
            run_dir.mkdir(parents=True)
            report_path = run_dir / "multi_agent_report.json"
            report_path.write_text("{}", encoding="utf-8")
            compact = compact_report(
                {
                    "request": "aspirin",
                    "task_type": "general_drug_profile",
                    "molecules": [],
                    "predictions": [],
                    "evidence": [],
                    "tool_calls": [{"round": 1, "agent": "ReporterAgent", "tool": "build_runtime_report", "status": "completed"}],
                    "artifacts": {"multi_agent_report_path": str(report_path)},
                    "warnings": [],
                },
                run_dir=run_dir,
                config=WebDemoConfig(project_root=root, output_dir=root / "outputs" / "web_demo"),
            )
        status_by_tool = {node["tool"]: node["status"] for node in compact["workflow_graph"]["full_topology"]["nodes"]}
        origin_by_tool = {node["tool"]: node.get("origin") for node in compact["workflow_graph"]["full_topology"]["nodes"]}
        self.assertEqual(status_by_tool["write_artifacts"], "completed")
        self.assertEqual(origin_by_tool["write_artifacts"], "runtime_added")
        self.assertTrue(any(node["tool"] == "write_artifacts" and node.get("inferred") for node in compact["workflow_graph"]["execution"]["nodes"]))
        self.assertIsNotNone(compact["general_profile_notice"])
        self.assertEqual(compact["quality_summary"]["prediction_mode"], "not_requested")

    def test_compact_report_exposes_reaction_fields_and_topology(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "outputs" / "web_demo" / "reaction_run"
            run_dir.mkdir(parents=True)
            reaction_report = run_dir / "reaction_report.json"
            reaction_report.write_text("{}", encoding="utf-8")
            compact = compact_report(
                {
                    "request": "Predict yield for BrC.CC>>CCC",
                    "task_type": "reaction_yield_prediction",
                    "dataset_name": "buchwald_hartwig_kg",
                    "reaction": {"reaction_smiles": "BrC.CC>>CCC", "reactants": ["BrC", "CC"], "products": ["CCC"]},
                    "reaction_profile": {"summary": "reaction profile", "template_evidence_count": 2, "condition_evidence_count": 3},
                    "reaction_predictions": [
                        {
                            "task_type": "reaction_yield_prediction",
                            "dataset_name": "buchwald_hartwig_kg:test1",
                            "label": "moderate_yield",
                            "score": 55.2,
                            "model_type": "reaction_tfidf_ridge_regressor",
                        }
                    ],
                    "predictions": [],
                    "evidence": [{"source": "orderly_condition", "claim": "similar condition", "citation_id": "orderly_1"}],
                    "tool_calls": [
                        {"round": 1, "agent": "ReactionAgent", "tool": "parse_reaction", "status": "completed"},
                        {"round": 2, "agent": "ReactionAgent", "tool": "predict_buchwald_hartwig_yield", "status": "completed"},
                    ],
                    "artifacts": {"reaction_report_path": str(reaction_report), "multi_agent_report_path": str(reaction_report)},
                    "warnings": [],
                },
                run_dir=run_dir,
                config=WebDemoConfig(project_root=root, output_dir=root / "outputs" / "web_demo"),
            )
        self.assertEqual(compact["reaction_prediction"]["label"], "moderate_yield")
        self.assertEqual(compact["prediction"]["model_type"], "reaction_tfidf_ridge_regressor")
        self.assertEqual(compact["quality_summary"]["prediction_mode"], "tool_grounded")
        self.assertIn("ReactionAgent", {lane["agent"] for lane in compact["workflow_graph"]["full_topology"]["lanes"]})
        self.assertIn("reaction_report_path", {item["key"] for item in compact["files"]})

    def test_topology_edges_mark_only_actual_adjacent_path_active(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "outputs" / "web_demo" / "run1"
            run_dir.mkdir(parents=True)
            report_path = run_dir / "multi_agent_report.json"
            report_path.write_text("{}", encoding="utf-8")
            compact = compact_report(
                {
                    "request": "aspirin 这个药物怎样？",
                    "task_type": "general_drug_profile",
                    "molecules": [{"name": "aspirin", "is_valid": True}],
                    "predictions": [],
                    "evidence": [],
                    "tool_calls": [
                        {"round": 1, "agent": "MoleculeAgent", "tool": "route_task", "status": "completed"},
                        {"round": 2, "agent": "MoleculeAgent", "tool": "resolve_molecule", "status": "completed"},
                        {"round": 3, "agent": "RetrieveAgent", "tool": "retrieve_external_evidence", "status": "completed"},
                        {"round": 4, "agent": "GaugeAgent", "tool": "build_features", "status": "completed"},
                        {"round": 5, "agent": "RetrieveAgent", "tool": "retrieve_elasticsearch_hybrid", "status": "completed"},
                        {"round": 6, "agent": "ReasoningAgent", "tool": "build_structured_explanation", "status": "completed"},
                    ],
                    "artifacts": {"multi_agent_report_path": str(report_path)},
                    "warnings": [],
                },
                run_dir=run_dir,
                config=WebDemoConfig(project_root=root, output_dir=root / "outputs" / "web_demo"),
            )
        active_by_edge = {
            (edge["source"], edge["target"]): edge.get("active")
            for edge in compact["workflow_graph"]["full_topology"]["edges"]
        }
        kind_by_edge = {
            (edge["source"], edge["target"]): edge.get("kind")
            for edge in compact["workflow_graph"]["full_topology"]["edges"]
        }
        self.assertTrue(active_by_edge[("RetrieveAgent.retrieve_external_evidence", "GaugeAgent.build_features")])
        self.assertTrue(active_by_edge[("GaugeAgent.build_features", "RetrieveAgent.retrieve_elasticsearch_hybrid")])
        self.assertTrue(active_by_edge[("RetrieveAgent.retrieve_elasticsearch_hybrid", "ReasoningAgent.build_structured_explanation")])
        self.assertFalse(active_by_edge[("RetrieveAgent.retrieve_external_evidence", "ReasoningAgent.build_structured_explanation")])
        self.assertFalse(active_by_edge[("GaugeAgent.build_features", "ReasoningAgent.build_structured_explanation")])
        self.assertEqual(kind_by_edge[("ReflectionAgent.apply_reflection_feedback", "RetrieveAgent.retrieve_elasticsearch_hybrid")], "backtrack")
        self.assertEqual(kind_by_edge[("VerifierAgent.verify_citation_grounding", "RetrieveAgent.expand_evidence_query")], "backtrack")

    def test_public_run_summary_exposes_status(self):
        summary = _run_public_summary(
            {
                "task_type": "admet_prediction",
                "prediction": {"dataset_name": "DILI", "label": 0, "score": 0.17},
                "molecules": [{}],
                "evidence": [{}],
                "warnings": [],
                "objective_status": {"status": "pass"},
                "verifier_report": {"summary": {"status": "pass"}},
                "multi_agent": {"planner_type": "llm", "reasoning_mode": "llm_expert"},
            }
        )
        self.assertEqual(summary["completed_status"], "yes")
        self.assertEqual(summary["model_status"], "available")
        self.assertEqual(summary["objective_status"], "pass")
        self.assertEqual(summary["verifier_status"], "pass")
        self.assertEqual(summary["citation_grounded"], None)
        self.assertEqual(summary["evidence_index"], None)
        self.assertEqual(summary["planner_type"], "llm")

    def test_run_manager_completes_async_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = WebDemoConfig(project_root=root, output_dir=root / "outputs" / "web_demo")

            def fake_run(payload, config, run_lock, run_dir=None, progress_callback=None):
                if progress_callback:
                    progress_callback({"event_type": "step_completed", "step_id": "route_task", "tool_name": "route_task", "status": "completed"})
                return {"status": "ok", "run_id": Path(run_dir).name, "task_type": "admet_prediction", "prediction": {"label": 0}, "warnings": []}

            with patch("agentm.runtime.web.demo_server.run_agent_request", side_effect=fake_run):
                manager = WebRunManager(config=config, run_lock=Lock())
                run = manager.start({"request": "aspirin DILI", "run_mode": "fast"})
                current = _wait_for_status(manager, run.run_id, "completed")
        self.assertIsNotNone(current)
        self.assertEqual(current.status, "completed")
        self.assertTrue(current.step_events)

    def test_full_mode_maps_to_llm_multi_agent_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_dir = root / "outputs" / "web_demo"
            run_dir = output_dir / "run_full"
            run_dir.mkdir(parents=True)
            captured = {}

            class FakeController:
                def __init__(self, **kwargs):
                    captured.update(kwargs)

                def run(self, request, task_type=None, dataset_name=None):
                    return {
                        "request": request,
                        "task_type": task_type,
                        "molecules": [],
                        "predictions": [{"task_type": task_type, "dataset_name": dataset_name, "label": 0}],
                        "evidence": [],
                        "tool_calls": [],
                        "artifacts": {},
                        "warnings": [],
                    }

            with patch("agentm.runtime.web.demo_server.AgentMMultiAgentController", FakeController):
                result = run_agent_request(
                    {"request": "aspirin DILI", "run_mode": "full", "task_type": "admet_prediction", "dataset_name": "DILI"},
                    WebDemoConfig(project_root=root, output_dir=output_dir),
                    Lock(),
                    run_dir=run_dir,
                )
        self.assertEqual(result["run_mode"], "full")
        self.assertEqual(captured["mode"], "full")
        self.assertEqual(captured["retrieval_backend"], "auto")
        self.assertTrue(captured["use_llm_orchestrator"])
        self.assertTrue(captured["use_llm_reasoning"])
        self.assertFalse(captured["require_llm_reasoning"])
        self.assertTrue(captured["enable_fallbacks"])

    def test_full_mode_honors_explicit_mllm_visual_semantic_opt_in(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_dir = root / "outputs" / "web_demo"
            run_dir = output_dir / "run_full_mllm"
            run_dir.mkdir(parents=True)
            captured = {}

            class FakeController:
                def __init__(self, **kwargs):
                    captured.update(kwargs)

                def run(self, request, task_type=None, dataset_name=None):
                    return {
                        "request": request,
                        "task_type": task_type,
                        "molecules": [],
                        "predictions": [],
                        "evidence": [],
                        "tool_calls": [],
                        "artifacts": {},
                        "warnings": [],
                    }

            with patch("agentm.runtime.web.demo_server.AgentMMultiAgentController", FakeController):
                run_agent_request(
                    {
                        "request": "aspirin BBBP",
                        "run_mode": "full",
                        "task_type": "property_prediction",
                        "dataset_name": "BBBP",
                        "use_mllm_visual_semantic": True,
                    },
                    WebDemoConfig(project_root=root, output_dir=output_dir),
                    Lock(),
                    run_dir=run_dir,
                )
            self.assertEqual(captured["mode"], "full")
            self.assertEqual(captured["feature_set"], "all_features")
            self.assertTrue(captured["use_mllm_visual_semantic"])
            self.assertTrue(captured["use_smiles_encoder"])
            runtime_config = json.loads((run_dir / "web_runtime_config.json").read_text(encoding="utf-8"))
            self.assertEqual(runtime_config["feature_set"], "all_features")
            self.assertTrue(runtime_config["use_mllm_visual_semantic"])
            self.assertTrue(runtime_config["use_smiles_encoder"])


def _wait_for_status(manager: WebRunManager, run_id: str, status: str):
    deadline = time.time() + 5
    current = None
    while time.time() < deadline:
        current = manager.get(run_id)
        if current and current.status == status:
            return current
        time.sleep(0.05)
    return current


if __name__ == "__main__":
    unittest.main()
