from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from agentm.runtime import AgentMMultiAgentController
from agentm.runtime.state import initial_context
from agentm.runtime.tools.scientific import ScientificToolRuntime


class FakeOrchestratorClient:
    available = True

    def __init__(self) -> None:
        self.action_calls = 0

    def chat_json(self, messages, temperature=0.1):
        system = messages[0]["content"]
        if "Build a FROGENT-style multi-agent task graph" in system:
            return """
            {
              "objective": "LLM planned ADMET evidence task",
              "task_graph": {
                "nodes": [
                  {"id": "route", "agent": "MoleculeAgent", "goal": "infer task"},
                  {"id": "gauge", "agent": "GaugeAgent", "goal": "ground prediction"},
                  {"id": "retrieve", "agent": "RetrieveAgent", "goal": "collect citations"},
                  {"id": "verify", "agent": "VerifierAgent", "goal": "check grounding"},
                  {"id": "reflect", "agent": "ReflectionAgent", "goal": "feed report back"}
                ],
                "edges": [["route", "gauge"], ["gauge", "retrieve"], ["retrieve", "verify"], ["verify", "reflect"]]
              },
              "execution_strategy": "LLM proposes, guard verifies, fallback repairs.",
              "stop_conditions": ["verifier_pass", "reflection_applied"]
            }
            """
        import json

        self.action_calls += 1
        if self.action_calls == 1:
            return json.dumps({"agent": "GaugeAgent", "tool": "predict_single", "args": {}, "rationale": "Deliberately invalid first proposal for guard revision."})
        payload = json.loads(messages[1]["content"])
        completed = " ".join(payload["context"].get("completed_actions", []))
        task_type = payload["context"].get("task_type")
        if "route_task:completed" not in completed:
            action = {"agent": "MoleculeAgent", "tool": "route_task"}
        elif task_type == "ddi_prediction" and "resolve_ddi_pair:completed" not in completed:
            action = {"agent": "MoleculeAgent", "tool": "resolve_ddi_pair"}
        elif task_type != "ddi_prediction" and "resolve_molecule:completed" not in completed:
            action = {"agent": "MoleculeAgent", "tool": "resolve_molecule"}
        elif "discover_rules:completed" not in completed:
            action = {"agent": "GaugeAgent", "tool": "discover_rules"}
        elif "build_features:completed" not in completed:
            action = {"agent": "GaugeAgent", "tool": "build_features"}
        elif "predict_single:completed" not in completed and "predict_single:fallback_completed" not in completed:
            action = {"agent": "GaugeAgent", "tool": "predict_single"}
        elif "assess_confidence:completed" not in completed:
            action = {"agent": "GaugeAgent", "tool": "assess_confidence"}
        elif "retrieve_local_rag:completed" not in completed:
            action = {"agent": "RetrieveAgent", "tool": "retrieve_local_rag"}
        elif "build_structured_explanation:completed" not in completed:
            action = {"agent": "ReasoningAgent", "tool": "build_structured_explanation"}
        elif "build_runtime_report:completed" not in completed:
            action = {"agent": "ReporterAgent", "tool": "build_runtime_report"}
        elif "verify_prediction_grounding:completed" not in completed:
            action = {"agent": "VerifierAgent", "tool": "verify_prediction_grounding"}
        elif "verify_citation_grounding:completed" not in completed:
            action = {"agent": "VerifierAgent", "tool": "verify_citation_grounding"}
        elif "verify_report_schema:completed" not in completed:
            action = {"agent": "VerifierAgent", "tool": "verify_report_schema"}
        elif "reflect_on_report:completed" not in completed:
            action = {"agent": "ReflectionAgent", "tool": "reflect_on_report"}
        elif "apply_reflection_feedback:completed" not in completed:
            action = {"agent": "ReflectionAgent", "tool": "apply_reflection_feedback"}
        else:
            action = {"agent": "OrchestratorAgent", "tool": "complete_task"}
        action.setdefault("args", {})
        action.setdefault("rationale", "Fake LLM selected the next guarded action.")
        return json.dumps(action)


class FakeReasoningClient:
    available = True

    def chat_json(self, messages, temperature=0.1):
        import json

        system = messages[0]["content"]
        if "ReflectionAgent" in system:
            return json.dumps(
                {
                    "recommendation": "finalize",
                    "needs_refinement": False,
                    "risk_assessment": "Verifier passed and evidence is sufficient.",
                    "refinement_queries": [],
                }
            )
        return json.dumps(
            {
                "summary": "Tool-grounded prediction is supported by retrieved evidence cite_1.",
                "prediction_interpretation": "The label and score come from GaugeAgent.",
                "evidence_support": [
                    {"citation_id": "cite_1", "claim": "Retrieved DILI evidence supports the interpretation.", "support": "supportive"},
                    {"citation_id": "fake_cite", "claim": "This citation should be filtered.", "support": "supportive"},
                ],
                "limitations": ["Only retrieved evidence is considered."],
                "unsupported_claims": [],
                "citation_ids": ["cite_1", "fake_cite"],
            }
        )


class GeneralProfilePredictingClient:
    available = True

    def chat_json(self, messages, temperature=0.1):
        import json

        system = messages[0]["content"]
        if "Build a FROGENT-style multi-agent task graph" in system:
            return json.dumps(
                {
                    "objective": "General aspirin profile",
                    "task_graph": {
                        "nodes": [
                            {"id": "resolve", "agent": "MoleculeAgent", "goal": "Resolve aspirin."},
                            {"id": "retrieve", "agent": "RetrieveAgent", "goal": "Retrieve public database context."},
                            {"id": "features", "agent": "GaugeAgent", "goal": "Construct structural descriptors."},
                            {"id": "reason", "agent": "ReasoningAgent", "goal": "Explain the general profile."},
                            {"id": "report", "agent": "ReporterAgent", "goal": "Build report."},
                        ],
                        "edges": [["resolve", "retrieve"], ["retrieve", "features"], ["features", "reason"], ["reason", "report"]],
                    },
                    "execution_strategy": "Use public profile evidence and descriptors.",
                    "stop_conditions": ["schema_checked"],
                },
                ensure_ascii=False,
            )
        if "repairing a guard-blocked action" in system:
            return json.dumps(
                {
                    "agent": "ReasoningAgent",
                    "tool": "build_structured_explanation",
                    "args": {},
                    "rationale": "Endpoint prediction was not requested; explain the general profile instead.",
                },
                ensure_ascii=False,
            )
        payload = json.loads(messages[1]["content"])
        completed = " ".join(payload["context"].get("completed_actions", []))
        if "route_task:completed" not in completed:
            action = {"agent": "MoleculeAgent", "tool": "route_task"}
        elif "resolve_molecule:completed" not in completed:
            action = {"agent": "MoleculeAgent", "tool": "resolve_molecule"}
        elif "retrieve_external_evidence:completed" not in completed:
            action = {"agent": "RetrieveAgent", "tool": "retrieve_external_evidence"}
        elif "build_features:completed" not in completed:
            action = {"agent": "GaugeAgent", "tool": "build_features"}
        elif "build_structured_explanation:completed" not in completed:
            action = {"agent": "GaugeAgent", "tool": "predict_single"}
        elif "build_runtime_report:completed" not in completed:
            action = {"agent": "ReporterAgent", "tool": "build_runtime_report"}
        elif "verify_citation_grounding:completed" not in completed:
            action = {"agent": "VerifierAgent", "tool": "verify_citation_grounding"}
        elif "verify_report_schema:completed" not in completed:
            action = {"agent": "VerifierAgent", "tool": "verify_report_schema"}
        elif "reflect_on_report:completed" not in completed:
            action = {"agent": "ReflectionAgent", "tool": "reflect_on_report"}
        elif "apply_reflection_feedback:completed" not in completed:
            action = {"agent": "ReflectionAgent", "tool": "apply_reflection_feedback"}
        else:
            action = {"agent": "OrchestratorAgent", "tool": "complete_task"}
        action.setdefault("args", {})
        action.setdefault("rationale", "Fake LLM selected the next guarded action.")
        return json.dumps(action, ensure_ascii=False)


class MultiAgentControllerSmokeTests(unittest.TestCase):
    def test_single_molecule_smoke_writes_multi_agent_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            controller = AgentMMultiAgentController(
                output_dir=tmp,
                feature_set="rdkit_only",
                mode="fast",
                retrieval_backend="local",
                use_llm_orchestrator=False,
                enable_fallbacks=True,
            )
            report = controller.run(
                "CC(=O)OC1=CC=CC=C1C(=O)O",
                task_type="admet_prediction",
                dataset_name="DILI",
            )
            self.assertEqual(report["task_type"], "admet_prediction")
            self.assertTrue(report["predictions"])
            self.assertEqual(report["verifier_report"]["summary"]["status"], "warning")
            self.assertEqual(report["multi_agent"]["loop_status"], "finalized")
            self.assertIn("real_model_prediction", report["objective_status"]["missing_objectives"])
            self.assertIn("MoleculeAgent", {call["agent"] for call in report["tool_calls"]})
            self.assertIn("GaugeAgent", {call["agent"] for call in report["tool_calls"]})
            self.assertIn("VerifierAgent", {call["agent"] for call in report["tool_calls"]})
            for filename in [
                "multi_agent_report.json",
                "single_molecule_report.json",
                "global_context.json",
                "orchestrator_plan.json",
                "task_graph.json",
                "llm_plan_proposals.json",
                "plan_revision_trace.json",
                "multi_agent_trace.json",
                "agent_messages.json",
                "tool_calls.json",
                "retrieval_trace.json",
                "citation_map.json",
                "verifier_report.json",
                "reflection.json",
                "reproducibility.json",
                "report.md",
            ]:
                self.assertTrue((Path(tmp) / filename).exists(), filename)
            self.assertIn("ReflectionAgent", {call["agent"] for call in report["tool_calls"]})
            self.assertTrue(report["reflection"])
            self.assertTrue(report["reasoning"].get("reflection_feedback", {}).get("used_as_next_input"))

    def test_general_drug_question_does_not_default_to_bbbp_prediction(self):
        with tempfile.TemporaryDirectory() as tmp:
            controller = AgentMMultiAgentController(
                output_dir=tmp,
                feature_set="rdkit_only",
                mode="fast",
                retrieval_backend="local",
                use_llm_orchestrator=False,
                use_llm_reasoning=False,
                enable_fallbacks=True,
            )
            report = controller.run("阿司匹林 这个药物怎样？")
            self.assertEqual(report["task_type"], "general_drug_profile")
            self.assertEqual(report["dataset_name"], "general")
            self.assertFalse(report["predictions"])
            self.assertEqual(report["molecules"][0]["name"], "aspirin")
            self.assertTrue(report["molecules"][0]["is_valid"])
            self.assertNotIn("BBBP", report["explanation"])
            self.assertNotIn("可检索到", report["explanation"])
            self.assertTrue(any(call["tool"] == "retrieve_external_evidence" for call in report["tool_calls"]))
            self.assertTrue(report["artifacts"].get("database_profile"))
            self.assertIn("分子结构已解析", report["explanation"])
            self.assertFalse(any(call["tool"] == "predict_single" for call in report["tool_calls"]))
            self.assertFalse(any(str(warning).startswith("objective_missing:real_model_prediction") for warning in report["warnings"]))

    def test_llm_general_profile_prediction_attempt_is_guard_revised(self):
        with tempfile.TemporaryDirectory() as tmp:
            controller = AgentMMultiAgentController(
                output_dir=tmp,
                feature_set="rdkit_only",
                mode="fast",
                retrieval_backend="local",
                use_llm_orchestrator=True,
                use_llm_reasoning=False,
                orchestrator_llm_client=GeneralProfilePredictingClient(),
                enable_fallbacks=True,
            )
            report = controller.run("阿司匹林 这个药物怎样？")
            self.assertEqual(report["task_type"], "general_drug_profile")
            self.assertFalse(report["predictions"])
            self.assertFalse(any(call["tool"] == "predict_single" for call in report["tool_calls"]))
            self.assertTrue(any("general_profile_endpoint_prediction_not_requested" in str(warning) for warning in report["warnings"]))
            self.assertTrue(any(call["tool"] == "build_structured_explanation" for call in report["tool_calls"]))

    def test_general_profile_sentence_summarizes_dailymed_and_chembl(self):
        with tempfile.TemporaryDirectory() as tmp:
            runtime = ScientificToolRuntime(output_dir=tmp, feature_set="rdkit_only")
            context = {
                "request": "atorvastatin 这个药物怎样？",
                "artifacts": {
                    "database_profile": [
                        {
                            "source": "pubchem",
                            "retrieval_status": "success",
                            "title": "Atorvastatin",
                            "pubchem_cid": 60823,
                            "molecular_formula": "C33H35FN2O5",
                            "molecular_weight": "558.6",
                            "synonyms": ["atorvastatin", "Cardyl"],
                        },
                        {
                            "source": "dailymed",
                            "retrieval_status": "success",
                            "label_entries": [{"title": "ATORVASTATIN CALCIUM tablet"}],
                        },
                        {
                            "source": "chembl",
                            "retrieval_status": "success",
                            "chembl_id": "CHEMBL1487",
                            "mechanisms": [{"mechanism_of_action": "HMG-CoA reductase inhibitor", "target_name": "HMG-CoA reductase"}],
                            "indications": [{"mesh_heading": "Hypercholesterolemia"}],
                        },
                    ]
                },
            }
            sentence = runtime._database_profile_sentence(context, chinese=True)
        self.assertIn("Atorvastatin", sentence)
        self.assertIn("PubChem CID 60823", sentence)
        self.assertIn("药品标签资料", sentence)
        self.assertIn("生物活性资料", sentence)
        self.assertIn("HMG-CoA reductase", sentence)
        self.assertIn("Hypercholesterolemia", sentence)
        self.assertNotIn("DailyMed 标签中包含", sentence)
        self.assertNotIn("ChEMBL ID 为", sentence)
        self.assertNotIn("可检索到", sentence)

    def test_discover_rules_prefers_cached_training_rules(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_path = root / "model.pkl"
            model_path.write_text("placeholder", encoding="utf-8")
            rules_dir = root / "rules"
            rules_dir.mkdir()
            rules_path = rules_dir / "admet_prediction_BBB_Martins_rules.json"
            rules_path.write_text(
                json.dumps(
                    {
                        "rules": [
                            {
                                "rule_id": "cached_1",
                                "task_type": "admet_prediction",
                                "source": "cached",
                                "text": "TPSA below 90",
                                "feature_name": "rule_TPSA_lt_90p0",
                                "code": "def rule_cached_1(mol, descriptors):\n    return 1.0\n",
                                "metadata": {"operation": "descriptor_threshold", "descriptor": "TPSA", "operator": "<", "threshold": 90.0},
                                "is_executable": True,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            runtime = ScientificToolRuntime(output_dir=root / "run", feature_set="all_features")
            runtime._update_resource_resolution = lambda context: None
            runtime._selected_model_path = lambda context: str(model_path)
            context = initial_context("aspirin BBBP", task_type="admet_prediction", dataset_name="BBB_Martins")
            result = runtime.discover_rules({}, context)
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.observation["source"], "cached_training_rules")
        self.assertEqual(context["rules"][0].feature_name, "rule_TPSA_lt_90p0")
        self.assertEqual(context["artifacts"]["rules_source_path"], str(rules_path))

    def test_feature_alignment_loads_cached_rules_before_building_features(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_path = root / "model.pkl"
            model_path.write_text("placeholder", encoding="utf-8")
            features_dir = root / "features"
            features_dir.mkdir()
            (features_dir / "feature_schema.json").write_text(
                json.dumps({"feature_names": ["rdkit_MolWt", "llm_rule_rule_a", "llm_rule_rule_b", "smiles_0"]}),
                encoding="utf-8",
            )
            rules_dir = root / "rules"
            rules_dir.mkdir()
            (rules_dir / "admet_prediction_BBB_Martins_rules.json").write_text(
                json.dumps(
                    {
                        "rules": [
                            {"rule_id": "a", "task_type": "admet_prediction", "feature_name": "rule_a", "text": "A", "code": "def rule_a(mol, descriptors):\n    return 1.0\n"},
                            {"rule_id": "c", "task_type": "admet_prediction", "feature_name": "rule_c", "text": "C", "code": "def rule_c(mol, descriptors):\n    return 1.0\n"},
                            {"rule_id": "b", "task_type": "admet_prediction", "feature_name": "rule_b", "text": "B", "code": "def rule_b(mol, descriptors):\n    return 1.0\n"},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            runtime = ScientificToolRuntime(output_dir=root / "run", feature_set="all_features")
            runtime._update_resource_resolution = lambda context: None
            runtime._selected_model_path = lambda context: str(model_path)
            context = initial_context("aspirin BBBP", task_type="admet_prediction", dataset_name="BBB_Martins")
            context["rules"] = [runtime._rule_from_mapping({"feature_name": "rule_c", "text": "wrong cached subset"})]
            runtime._ensure_cached_rules_for_feature_alignment(context)
        self.assertEqual([rule.feature_name for rule in context["rules"]], ["rule_a", "rule_b"])
        self.assertEqual(context["artifacts"]["rules_aligned_to_model_schema"]["expected"], 2)
        self.assertEqual(context["artifacts"]["rules_aligned_to_model_schema"]["selected"], 2)

    def test_cached_rules_use_selected_model_metadata_task_type(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_path = root / "model.pkl"
            model_path.write_text("placeholder", encoding="utf-8")
            rules_dir = root / "rules"
            rules_dir.mkdir()
            rules_path = rules_dir / "admet_prediction_BBB_Martins_rules.json"
            rules_path.write_text(
                json.dumps({"rules": [{"rule_id": "a", "task_type": "admet_prediction", "feature_name": "rule_a", "text": "A"}]}),
                encoding="utf-8",
            )
            runtime = ScientificToolRuntime(output_dir=root / "run", feature_set="all_features")
            runtime._update_resource_resolution = lambda context: None
            runtime._selected_model_path = lambda context: str(model_path)
            context = initial_context("aspirin BBBP", task_type="property_prediction", dataset_name="BBB_Martins")
            context["resource_resolution"] = {
                "model": {
                    "selected_path": str(model_path),
                    "candidates": [
                        {
                            "path": str(model_path),
                            "selected": True,
                            "metadata_task_type": "admet_prediction",
                            "metadata_dataset_name": "BBB_Martins",
                        }
                    ],
                }
            }
            result = runtime.discover_rules({}, context)
        self.assertEqual(result.status, "completed")
        self.assertEqual(context["rules"][0].feature_name, "rule_a")
        self.assertEqual(context["artifacts"]["rules_source_path"], str(rules_path))

    def test_explicit_endpoint_still_runs_prediction(self):
        with tempfile.TemporaryDirectory() as tmp:
            controller = AgentMMultiAgentController(
                output_dir=tmp,
                feature_set="rdkit_only",
                mode="fast",
                retrieval_backend="local",
                use_llm_orchestrator=False,
                use_llm_reasoning=False,
                enable_fallbacks=True,
            )
            report = controller.run("阿司匹林 BBBP")
            self.assertIn(report["task_type"], {"property_prediction", "admet_prediction"})
            self.assertEqual(report["dataset_name"], "BBBP")
            self.assertTrue(report["predictions"])
            self.assertTrue(any(call["tool"] == "predict_single" for call in report["tool_calls"]))

    def test_unresolved_drug_gets_human_readable_no_prediction_message(self):
        with tempfile.TemporaryDirectory() as tmp:
            controller = AgentMMultiAgentController(
                output_dir=tmp,
                feature_set="rdkit_only",
                mode="fast",
                retrieval_backend="local",
                use_llm_orchestrator=False,
                use_llm_reasoning=False,
                enable_fallbacks=True,
            )
            report = controller.run("azintamied 这个药物怎样？")
            self.assertFalse(report["predictions"])
            self.assertIn("没有生成可靠药物概览", report["explanation"])
            self.assertIn("不能构建分子特征", report["explanation"])
            self.assertNotIn("No tool-grounded prediction", report["explanation"])

    def test_ddi_smoke_writes_compat_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            controller = AgentMMultiAgentController(
                output_dir=tmp,
                feature_set="rdkit_only",
                mode="fast",
                retrieval_backend="local",
                use_llm_orchestrator=False,
                enable_fallbacks=True,
            )
            report = controller.run(
                "CC(=O)OC1=CC=CC=C1C(=O)O, CC(=O)CC(c1ccccc1)c1c(O)c2ccccc2oc1=O",
                task_type="ddi_prediction",
                dataset_name="DDI",
            )
            self.assertEqual(report["task_type"], "ddi_prediction")
            self.assertTrue(report["predictions"])
            self.assertTrue((Path(tmp) / "ddi_report.json").exists())

    def test_max_round_guard_still_finalizes_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            controller = AgentMMultiAgentController(
                output_dir=tmp,
                feature_set="rdkit_only",
                mode="fast",
                retrieval_backend="local",
                max_rounds=3,
                use_llm_orchestrator=False,
                enable_fallbacks=True,
            )
            report = controller.run(
                "CC(=O)OC1=CC=CC=C1C(=O)O",
                task_type="admet_prediction",
                dataset_name="DILI",
            )
            self.assertFalse(report["predictions"])
            self.assertTrue(any("guard_blocked" in warning for warning in report["warnings"]))
            self.assertTrue((Path(tmp) / "multi_agent_report.json").exists())

    def test_default_controller_uses_llm_orchestrator(self):
        with tempfile.TemporaryDirectory() as tmp:
            controller = AgentMMultiAgentController(output_dir=tmp, feature_set="rdkit_only")
            self.assertTrue(controller.use_llm_orchestrator)

    def test_fake_llm_orchestrator_generates_plan_and_guard_revision(self):
        with tempfile.TemporaryDirectory() as tmp:
            controller = AgentMMultiAgentController(
                output_dir=tmp,
                feature_set="rdkit_only",
                mode="fast",
                retrieval_backend="local",
                use_llm_orchestrator=True,
                orchestrator_llm_client=FakeOrchestratorClient(),
                enable_fallbacks=True,
            )
            report = controller.run(
                "CC(=O)OC1=CC=CC=C1C(=O)O",
                task_type="admet_prediction",
                dataset_name="DILI",
            )
            self.assertEqual(report["multi_agent"]["planner_type"], "llm")
            self.assertEqual(report["multi_agent"]["orchestrator_mode"], "llm_orchestrated")
            self.assertTrue(any(item["kind"] == "task_graph" and item["status"] == "accepted" for item in report["llm_plan_proposals"]))
            self.assertTrue(any(item.get("stage") == "action_guard" for item in report["plan_revision_trace"]))
            self.assertTrue(any(item["kind"] == "action" and item["status"] == "accepted_terminal" for item in report["llm_plan_proposals"]))
            self.assertFalse(any("llm_orchestrator_action_rejected" in str(warning) for warning in report["warnings"]))
            self.assertTrue(any("guard_revised" in warning for warning in report["warnings"]))
            self.assertTrue(report["predictions"])
            self.assertEqual(report["verifier_report"]["summary"]["status"], "warning")
            self.assertIn("real_model_prediction", report["objective_status"]["missing_objectives"])

    def test_previous_run_report_can_be_used_as_next_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = AgentMMultiAgentController(
                output_dir=Path(tmp) / "first",
                feature_set="rdkit_only",
                mode="fast",
                retrieval_backend="local",
                use_llm_orchestrator=False,
                enable_fallbacks=True,
            )
            first_report = first.run(
                "CC(=O)OC1=CC=CC=C1C(=O)O",
                task_type="admet_prediction",
                dataset_name="DILI",
            )
            second = AgentMMultiAgentController(
                output_dir=Path(tmp) / "second",
                feature_set="rdkit_only",
                mode="fast",
                retrieval_backend="local",
                use_llm_orchestrator=False,
                enable_fallbacks=True,
                previous_run_path=first_report["artifacts"]["multi_agent_report_path"],
            )
            second_report = second.run(
                "Cn1cnc2n(C)c(=O)n(C)c(=O)c12",
                task_type="admet_prediction",
                dataset_name="hERG",
            )
            self.assertTrue(second_report["previous_run_context"])
            self.assertTrue(second_report["previous_run_context"]["loaded"])
            self.assertNotIn("molecules", second_report["previous_run_context"])
            self.assertNotIn("evidence", second_report["previous_run_context"])
            self.assertEqual(second_report["reflection"]["previous_run_summary"]["task_type"], "admet_prediction")
            self.assertTrue(second_report["reasoning"]["reflection_feedback"]["used_as_next_input"])

    def test_fallbacks_disabled_by_default_do_not_zero_shot_predict(self):
        with tempfile.TemporaryDirectory() as tmp:
            controller = AgentMMultiAgentController(
                output_dir=tmp,
                feature_set="rdkit_only",
                mode="fast",
                retrieval_backend="local",
                use_llm_orchestrator=False,
            )
            report = controller.run(
                "CC(=O)OC1=CC=CC=C1C(=O)O",
                task_type="admet_prediction",
                dataset_name="DILI",
            )
            self.assertFalse(report["predictions"])
            self.assertFalse(report["multi_agent"]["fallbacks_enabled"])
            self.assertTrue(any("fallback_disabled" in warning for warning in report["warnings"]))
            self.assertFalse(any(call.get("status") == "fallback_completed" for call in report["tool_calls"]))

    def test_llm_reasoning_filters_unretrieved_citations(self):
        with tempfile.TemporaryDirectory() as tmp:
            runtime = ScientificToolRuntime(
                output_dir=tmp,
                feature_set="rdkit_only",
                use_llm_reasoning=True,
                reasoning_llm_client=FakeReasoningClient(),
            )
            context = initial_context("aspirin DILI", task_type="admet_prediction", dataset_name="DILI")
            context["predictions"] = [{"label": 0, "score": 0.17, "probability": 0.17, "model_type": "random_forest"}]
            context["confidence"] = {"score": 0.17, "low_confidence": False}
            context["evidence"] = [{"citation_id": "cite_1", "claim": "DILI literature evidence", "source": "local_rag", "chunk_id": "c1"}]
            result = runtime.build_structured_explanation({}, context)
            self.assertEqual(result.status, "completed")
            self.assertEqual(context["metadata"]["reasoning_mode"], "llm_expert")
            self.assertEqual(context["reasoning"]["llm_structured_explanation"]["citation_ids"], ["cite_1"])
            self.assertTrue(any("fake_cite" in warning for warning in context["warnings"]))

    def test_llm_reasoning_humanizes_internal_scores(self):
        class ScoreTalkingClient:
            available = True

            def chat_json(self, messages, temperature=0.1):
                import json

                return json.dumps(
                    {
                        "summary": "模型输出置信度分数为0.84，BBBP label=1。",
                        "prediction_interpretation": "综合评分0.84，预测结果为1。",
                        "evidence_support": [],
                        "limitations": ["该分数不是校准概率。"],
                        "unsupported_claims": [],
                        "citation_ids": [],
                    },
                    ensure_ascii=False,
                )

        with tempfile.TemporaryDirectory() as tmp:
            runtime = ScientificToolRuntime(
                output_dir=tmp,
                feature_set="rdkit_only",
                use_llm_reasoning=True,
                reasoning_llm_client=ScoreTalkingClient(),
            )
            context = initial_context("请预测 aspirin 是否能通过血脑屏障", task_type="admet_prediction", dataset_name="BBBP")
            context["predictions"] = [
                {
                    "label": 1,
                    "score": 0.84,
                    "probability": None,
                    "model_type": "zero_shot_rule_semantic_fusion",
                    "details": {"positive_label": 1, "positive_label_meaning": "BBB permeable"},
                }
            ]
            context["confidence"] = {"score": 0.84, "low_confidence": True}
            result = runtime.build_structured_explanation({}, context)
            explanation = context["reasoning"]["structured_explanation"]
            self.assertEqual(result.status, "completed")
            self.assertNotIn("0.84", explanation)
            self.assertNotIn("label=1", explanation)
            self.assertNotIn("分数", explanation)
            self.assertIn("整体判断强度", explanation)

    def test_llm_reasoning_removes_false_missing_feature_claim(self):
        class MissingFeatureClaimClient:
            available = True

            def chat_json(self, messages, temperature=0.1):
                import json

                return json.dumps(
                    {
                        "summary": "因未输入具体的分子描述符向量，该结构解释仅为理论映射。",
                        "prediction_interpretation": "当前任务未提供阿司匹林的具体分子指纹或描述符输入。",
                        "evidence_support": [],
                        "limitations": ["当前分析未提供模型输入特征。"],
                        "unsupported_claims": [],
                        "citation_ids": [],
                    },
                    ensure_ascii=False,
                )

        with tempfile.TemporaryDirectory() as tmp:
            runtime = ScientificToolRuntime(
                output_dir=tmp,
                feature_set="all_features",
                use_llm_reasoning=True,
                reasoning_llm_client=MissingFeatureClaimClient(),
            )
            context = initial_context("请预测 aspirin 是否能通过血脑屏障", task_type="admet_prediction", dataset_name="BBBP")
            context["predictions"] = [{"label": 0, "score": 0.48, "probability": 0.48, "model_type": "random_forest"}]
            context["confidence"] = {"score": 0.48, "low_confidence": True}
            context["features"] = [object()]
            context["artifacts"] = {"feature_schema": {"total_dim": 3208, "feature_names": ["rdkit_MolWt"], "blocks": {"rdkit": {"dim": 13}}}}
            result = runtime.build_structured_explanation({}, context)
            explanation = context["reasoning"]["structured_explanation"]
            self.assertEqual(result.status, "completed")
            self.assertNotIn("未输入具体的分子描述符向量", explanation)
            self.assertNotIn("未提供阿司匹林的具体分子指纹或描述符输入", explanation)
            self.assertIn("已由 GaugeAgent 构建分子描述符", explanation)
            self.assertIn("llm_reasoning_feature_absence_claim_corrected", context["warnings"])

    def test_llm_reasoning_rewrites_model_first_bbbp_conclusion(self):
        class MechanicalConclusionClient:
            available = True

            def chat_json(self, messages, temperature=0.1):
                import json

                return json.dumps(
                    {
                        "summary": "基于随机森林模型对BBBP数据集的预测，阿司匹林被判定为难以有效穿透血脑屏障。",
                        "prediction_interpretation": "模型综合了分子描述符、化学指纹、规则逻辑及序列与图像语义等多维结构特征进行推断。模型可能捕捉到了特定空间构型对中枢渗透的阻碍作用。",
                        "evidence_support": [],
                        "limitations": ["模型概率未经过后处理校准。"],
                        "unsupported_claims": ["提供的文献片段不支持针对阿司匹林的具体结构依据推论。"],
                        "citation_ids": [],
                    },
                    ensure_ascii=False,
                )

        with tempfile.TemporaryDirectory() as tmp:
            runtime = ScientificToolRuntime(
                output_dir=tmp,
                feature_set="all_features",
                use_llm_reasoning=True,
                reasoning_llm_client=MechanicalConclusionClient(),
            )
            context = initial_context("请预测 aspirin 是否能通过血脑屏障，任务是 BBBP", task_type="property_prediction", dataset_name="BBBP")
            context["molecules"] = [{"name": "阿司匹林", "canonical_smiles": "CC(=O)OC1=CC=CC=C1C(=O)O"}]
            context["predictions"] = [
                {
                    "task_type": "property_prediction",
                    "dataset_name": "BBB_Martins",
                    "label": 0,
                    "score": 0.48,
                    "probability": 0.48,
                    "model_type": "random_forest",
                    "details": {
                        "positive_label": 1,
                        "positive_label_meaning": "blood-brain barrier penetration",
                        "negative_label": 0,
                        "negative_label_meaning": "not blood-brain barrier penetration",
                    },
                }
            ]
            context["confidence"] = {"score": 0.48, "low_confidence": True}
            context["artifacts"] = {
                "feature_schema": {
                    "metadata": {
                        "descriptor_snapshot": {
                            "MolWt": 180.16,
                            "MolLogP": 1.31,
                            "TPSA": 63.6,
                            "NumHDonors": 1,
                            "NumHAcceptors": 3,
                        }
                    }
                }
            }
            result = runtime.build_structured_explanation({}, context)
            explanation = context["reasoning"]["structured_explanation"]
            self.assertEqual(result.status, "completed")
            self.assertTrue(explanation.startswith("结论：阿司匹林 不属于血脑屏障易透过分子"))
            self.assertNotIn("AgentM", explanation.split("。", 1)[0])
            self.assertNotIn("基于随机森林模型", explanation.split("。", 1)[0])
            self.assertNotIn("序列与图像语义", explanation)
            self.assertNotIn("空间构型", explanation)
            self.assertIn("方法说明：该判断来自 BBB_Martins 任务专用监督模型（Random Forest）", explanation)
            self.assertNotIn("LLM", explanation)
            self.assertIn("分子量=180", explanation)
            self.assertNotIn("{'name'", explanation)
            self.assertIn("羧酸", explanation)
            self.assertIn("llm_reasoning_ungrounded_feature_attribution_corrected", context["warnings"])

    def test_llm_reasoning_drops_bad_english_fragments_and_trims_sentence_boundary(self):
        class BadEnglishReasoningClient:
            available = True

            def chat_json(self, messages, temperature=0.1):
                import json

                return json.dumps(
                    {
                        "summary": (
                            "Aspirin demonstrates limited potential for passive blood-brain barrier permeability at therapeutic doses. "
                            "Salicylate can still be CNS-relevant under specific exposure conditions."
                        ),
                        "prediction_interpretation": (
                            "The computational evaluation assigns a low the prediction of BBB penetration, reflecting the compound's position near the classification boundary. "
                            "The underlying algorithm integrates multiple structured molecular representations to derive this outcome, operating within a framework calibrated to distinguish permeable from non-permeable chemotypes based on training data distributions."
                        ),
                        "evidence_support": [],
                        "limitations": [
                            "The retrieved evidence supports BBBP context but does not prove therapeutic-dose free BBB passage."
                        ],
                        "unsupported_claims": [],
                        "citation_ids": [],
                    }
                )

        with tempfile.TemporaryDirectory() as tmp:
            runtime = ScientificToolRuntime(
                output_dir=tmp,
                feature_set="all_features",
                use_llm_reasoning=True,
                reasoning_llm_client=BadEnglishReasoningClient(),
            )
            context = initial_context(
                "Evaluate the BBB permeability of aspirin and explain the evidence.",
                task_type="property_prediction",
                dataset_name="BBBP",
            )
            context["molecules"] = [{"name": "aspirin", "canonical_smiles": "CC(=O)OC1=CC=CC=C1C(=O)O"}]
            context["predictions"] = [
                {
                    "task_type": "property_prediction",
                    "dataset_name": "BBB_Martins",
                    "label": 0,
                    "score": 0.452,
                    "probability": 0.452,
                    "model_type": "random_forest",
                    "details": {
                        "positive_label": 1,
                        "positive_label_meaning": "blood-brain barrier penetration",
                        "negative_label": 0,
                        "negative_label_meaning": "not blood-brain barrier penetration",
                    },
                }
            ]
            context["confidence"] = {"score": 0.452, "low_confidence": True}
            context["artifacts"] = {
                "feature_schema": {
                    "metadata": {
                        "descriptor_snapshot": {
                            "MolWt": 180.16,
                            "MolLogP": 1.31,
                            "TPSA": 63.6,
                            "NumHDonors": 1,
                            "NumHAcceptors": 3,
                        }
                    }
                }
            }
            result = runtime.build_structured_explanation({}, context)
            explanation = context["reasoning"]["structured_explanation"]
            self.assertEqual(result.status, "completed")
            self.assertIn("Salicylate can still be CNS-relevant", explanation)
            self.assertNotIn("assigns a low the prediction", explanation)
            self.assertNotIn("underlying algorithm integrates", explanation)
            self.assertNotIn("calibrated to distinguish", explanation)
            self.assertNotRegex(explanation, r"\b(structured\s+molecu|molecu)$")
            self.assertRegex(explanation.rstrip(), r"[.!?]$")
            self.assertIn("llm_reasoning_low_quality_sentence_dropped", context["warnings"])

    def test_llm_reasoning_uses_reader_facing_english_style(self):
        class SystemReportStyleClient:
            available = True

            def chat_json(self, messages, temperature=0.1):
                import json

                return json.dumps(
                    {
                        "summary": (
                            "Aspirin demonstrates limited or borderline blood-brain barrier (BBB) permeability at therapeutic doses. "
                            "Although its relatively compact molecular size might initially suggest potential for central nervous system access, "
                            "its carboxylic acid group and elevated polar surface area hinder passive transcellular diffusion."
                        ),
                        "prediction_interpretation": (
                            "The computational assessment classifies aspirin as a non-penetrant compound regarding BBB crossing. "
                            "This classification carries low confidence, reflecting a borderline prediction near the decision boundary."
                        ),
                        "evidence_support": [],
                        "limitations": [
                            "The prediction relies on a supervised machine learning framework trained on curated BBB datasets."
                        ],
                        "unsupported_claims": [
                            "The model's exact decision pathway or specific feature weighting cannot be isolated or verified."
                        ],
                        "citation_ids": [],
                    }
                )

        with tempfile.TemporaryDirectory() as tmp:
            runtime = ScientificToolRuntime(
                output_dir=tmp,
                feature_set="all_features",
                use_llm_reasoning=True,
                reasoning_llm_client=SystemReportStyleClient(),
            )
            context = initial_context(
                "Evaluate the BBB permeability of aspirin and explain the evidence.",
                task_type="property_prediction",
                dataset_name="BBBP",
            )
            context["molecules"] = [{"name": "aspirin", "canonical_smiles": "CC(=O)OC1=CC=CC=C1C(=O)O"}]
            context["predictions"] = [
                {
                    "task_type": "property_prediction",
                    "dataset_name": "BBB_Martins",
                    "label": 0,
                    "score": 0.477,
                    "probability": 0.477,
                    "model_type": "random_forest",
                    "details": {
                        "positive_label": 1,
                        "positive_label_meaning": "blood-brain barrier penetration",
                        "negative_label": 0,
                        "negative_label_meaning": "not blood-brain barrier penetration",
                    },
                }
            ]
            context["confidence"] = {"score": 0.477, "low_confidence": True}
            context["artifacts"] = {
                "feature_schema": {
                    "metadata": {
                        "descriptor_snapshot": {
                            "MolWt": 180.16,
                            "MolLogP": 1.31,
                            "TPSA": 63.6,
                            "NumHDonors": 1,
                            "NumHAcceptors": 3,
                        }
                    }
                }
            }
            result = runtime.build_structured_explanation({}, context)
            explanation = context["reasoning"]["structured_explanation"]
            self.assertEqual(result.status, "completed")
            self.assertTrue(explanation.startswith("Conclusion: aspirin is predicted to have limited or borderline BBB permeability"))
            self.assertNotIn("AgentM", explanation)
            self.assertNotIn("LLM", explanation)
            self.assertNotIn("tool-grounded", explanation)
            self.assertNotIn("Additional note:", explanation)
            self.assertNotIn("Prediction source:", explanation)
            self.assertNotIn("Evidence and limitations:", explanation)
            self.assertNotIn("exact decision pathway", explanation)
            self.assertIn("Caution: This is a model-assisted assessment", explanation)
            self.assertIn("Method note: this assessment used a task-specific supervised BBB_Martins model", explanation)

    def test_zero_shot_fast_conclusion_is_complete_not_mechanical(self):
        with tempfile.TemporaryDirectory() as tmp:
            runtime = ScientificToolRuntime(
                output_dir=tmp,
                feature_set="rdkit_only",
                use_llm_reasoning=False,
            )
            context = initial_context("请预测 aspirin 是否能通过血脑屏障，任务是 BBBP", task_type="property_prediction", dataset_name="BBBP")
            context["metadata"]["mode"] = "fast"
            context["molecules"] = [{"name": "aspirin", "canonical_smiles": "CC(=O)OC1=CC=CC=C1C(=O)O"}]
            context["predictions"] = [
                {
                    "task_type": "property_prediction",
                    "dataset_name": "BBB_Martins",
                    "label": 1,
                    "score": 0.52,
                    "probability": 0.52,
                    "model_type": "zero_shot_rule_semantic_fusion",
                    "details": {
                        "positive_label": 1,
                        "positive_label_meaning": "blood-brain barrier penetration",
                        "negative_label": 0,
                        "negative_label_meaning": "not blood-brain barrier penetration",
                    },
                }
            ]
            context["confidence"] = {"score": 0.52, "low_confidence": True}
            context["evidence"] = [
                {"source": "rag_literature", "citation_id": f"cite_{idx}", "claim": f"background {idx}"}
                for idx in range(1, 6)
            ]
            context["artifacts"] = {
                "feature_schema": {
                    "metadata": {
                        "descriptor_snapshot": {
                            "MolWt": 180.16,
                            "MolLogP": 1.31,
                            "TPSA": 63.6,
                            "NumHDonors": 1,
                            "NumHAcceptors": 3,
                        }
                    }
                }
            }

            result = runtime.build_structured_explanation({}, context)
            explanation = context["reasoning"]["structured_explanation"]
            self.assertEqual(result.status, "completed")
            self.assertTrue(explanation.startswith("结论：aspirin"))
            self.assertIn("方法说明：本次使用规则/语义融合预测路径", explanation)
            self.assertIn("结构依据：分子量=180", explanation)
            self.assertIn("证据概况：本次参考了来自 rag_literature 的相关资料", explanation)
            self.assertNotIn("只能作为方向性筛查提示", explanation)
            self.assertNotIn("初筛", explanation)
            self.assertNotIn("AgentM", explanation.split("。", 1)[0])
            self.assertNotIn("RetrieveAgent", explanation)
            self.assertNotIn("VerifierAgent", explanation)

    def test_endpoint_specific_conclusions_avoid_mechanical_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            runtime = ScientificToolRuntime(output_dir=tmp, feature_set="rdkit_only", use_llm_reasoning=False)

            dili = initial_context("请评估 aspirin 的 DILI 风险", task_type="admet_prediction", dataset_name="DILI")
            dili["molecules"] = [{"name": "aspirin", "canonical_smiles": "CC(=O)OC1=CC=CC=C1C(=O)O"}]
            dili["predictions"] = [{"dataset_name": "DILI", "label": 1, "score": 0.86, "model_type": "random_forest", "details": {"positive_label": 1}}]
            dili["artifacts"] = {"feature_schema": {"metadata": {"descriptor_snapshot": {"MolWt": 180.16, "MolLogP": 1.31, "TPSA": 63.6}}}}
            runtime.build_structured_explanation({}, dili)
            dili_text = dili["reasoning"]["structured_explanation"]
            self.assertIn("药物性肝损伤风险信号", dili_text)
            self.assertIn("DILI 风险通常受暴露、代谢、反应性和剂量背景共同影响", dili_text)

            herg = initial_context("请评估 caffeine 的 hERG 风险", task_type="admet_prediction", dataset_name="hERG")
            herg["molecules"] = [{"name": "caffeine", "canonical_smiles": "Cn1cnc2n(C)c(=O)n(C)c(=O)c12"}]
            herg["predictions"] = [{"dataset_name": "hERG", "label": 0, "score": 0.18, "model_type": "random_forest", "details": {"positive_label": 1}}]
            herg["artifacts"] = {"feature_schema": {"metadata": {"descriptor_snapshot": {"MolWt": 194.19, "MolLogP": -0.07, "TPSA": 61.8}}}}
            runtime.build_structured_explanation({}, herg)
            herg_text = herg["reasoning"]["structured_explanation"]
            self.assertIn("未显示明显 hERG 相关心脏安全性风险信号", herg_text)
            self.assertIn("不等同于通道阻断机制的直接实验证据", herg_text)

            ddi = initial_context("aspirin 和 warfarin 是否有相互作用", task_type="ddi_prediction", dataset_name="DDI")
            ddi["molecules"] = [{"name": "aspirin"}, {"name": "warfarin"}]
            ddi["predictions"] = [{"task_type": "ddi_prediction", "dataset_name": "DDI", "label": 1, "score": 0.91, "model_type": "random_forest"}]
            runtime.build_structured_explanation({}, ddi)
            ddi_text = ddi["reasoning"]["structured_explanation"]
            self.assertIn("潜在药物相互作用风险信号", ddi_text)
            self.assertIn("相互作用依据", ddi_text)

            for text in [dili_text, herg_text, ddi_text]:
                self.assertNotIn("label=", text)
                self.assertNotIn("GaugeAgent", text)
                self.assertNotIn("RetrieveAgent", text)
                self.assertNotIn("VerifierAgent", text)

    def test_bbbp_structure_sentence_uses_current_molecule(self):
        with tempfile.TemporaryDirectory() as tmp:
            runtime = ScientificToolRuntime(output_dir=tmp, feature_set="rdkit_only")
            context = initial_context("atorvastatin BBBP", task_type="admet_prediction", dataset_name="BBBP")
            context["molecules"] = [{"name": "atorvastatin"}]
            context["predictions"] = [
                {
                    "label": 0,
                    "score": 0.84,
                    "dataset_name": "BBBP",
                    "details": {
                        "rdkit_descriptors": {
                            "MolWt": 559,
                            "TPSA": 112,
                            "MolLogP": 6.31,
                            "NumHDonors": 4,
                            "NumHAcceptors": 5,
                        }
                    },
                }
            ]
            sentence = runtime._structure_sentence(context, chinese=True)
        self.assertIn("atorvastatin", sentence)
        self.assertIn("分子量偏大", sentence)
        self.assertIn("TPSA 偏高", sentence)
        self.assertNotIn("aspirin", sentence)

    def test_required_llm_reasoning_fails_when_unavailable(self):
        with tempfile.TemporaryDirectory() as tmp:
            runtime = ScientificToolRuntime(
                output_dir=tmp,
                feature_set="rdkit_only",
                use_llm_reasoning=True,
                require_llm_reasoning=True,
                reasoning_llm_client=None,
            )
            runtime.reasoning_llm_client = None
            context = initial_context(
                "aspirin DILI",
                task_type="admet_prediction",
                dataset_name="DILI",
                require_llm_reasoning=True,
            )
            context["predictions"] = [{"label": 0, "score": 0.17, "model_type": "random_forest"}]
            result = runtime.build_structured_explanation({}, context)
            self.assertEqual(result.status, "failed")
            self.assertIn("llm_reasoning_required_but_unavailable", result.error)


if __name__ == "__main__":
    unittest.main()
