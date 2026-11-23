from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import requests

from agentm.runtime.artifacts import report_payload, write_artifacts
from agentm.runtime.compat import ensure_import_paths, to_jsonable
from agentm.runtime.guards.verifier import (
    merge_verifier_reports,
    verify_citations,
    verify_prediction_grounding,
    verify_prediction_semantics,
    verify_report_schema,
)
from agentm.runtime.objectives import ZERO_SHOT_MODEL_TYPE, evaluate_task_objectives
from agentm.runtime.prediction_semantics import (
    annotate_multiclass_ddi_prediction,
    binary_prediction_direction,
    is_multiclass_ddi_prediction,
    prediction_meaning,
    prediction_value,
)
from agentm.runtime.reaction import (
    DEFAULT_REACTION_DATA_ROOT,
    ReactionRuntime,
    infer_reaction_task_type,
    is_reaction_request,
    is_reaction_task,
    reaction_profile_summary,
)
from agentm.runtime.retrieval import HybridRetriever
from agentm.runtime.retrieval.evidence_schema import compact_term, endpoint_terms, infer_endpoint_from_context, entity_terms_from_context, normalize_text, unique_strings
from agentm.runtime.resources import default_resource_path, resolve_model_resource, resolve_rag_resources
from agentm.runtime.tools.registry import MultiAgentToolRegistry, ToolResult, ToolSpec
from experiments.agentm_reliability.evidence_provenance import audit_evidence_provenance

ensure_import_paths()

from agentm.agents.ddi_rag_evidence_agent import DDIRAGEvidenceAgent
from agentm.agents.evidence_agent import EvidenceAgent
from agentm.agents.input_agent import InputAgent
from agentm.agents.molecule_processing_agent import MoleculeProcessingAgent
from agentm.agents.prediction_agent import PredictionAgent
from agentm.agents.rag_evidence_agent import RAGEvidenceAgent
from agentm.agents.rule_discovery_agent import RuleDiscoveryAgent
from agentm.defaults import (
    DEFAULT_CHEMBERTA2_MODEL_PATH,
    DEFAULT_DDI_RAG_CORPUS_DIR,
    DEFAULT_FEATURE_SET,
    DEFAULT_IMAGEMOL_CHECKPOINT_PATH,
    DEFAULT_MLLM_PROFILE,
    DEFAULT_RAG_BACKEND,
    DEFAULT_RAG_CORPUS_DIR,
    DEFAULT_RAG_EMBEDDING_MODEL_PATH,
    DEFAULT_RAG_TOP_K,
)
from agentm.features.builder import FeatureBuilder
from agentm.schema import MoleculeRecord, Rule
from agentm.tools.smiles import standardize_smiles
from agentm.utils.io import ensure_dir


GENERAL_TASK_TYPE = "general_drug_profile"
GENERAL_DATASET_NAME = "general"
OUT_OF_DOMAIN_TASK_TYPE = "out_of_domain"
MISSING_ENTITY_TASK_TYPE = "missing_entity"
NON_SCIENCE_DATASET_NAME = "none"
NON_SCIENCE_TASK_TYPES = {OUT_OF_DOMAIN_TASK_TYPE, MISSING_ENTITY_TASK_TYPE}
RUNTIME_PROJECT_ROOT = Path(__file__).resolve().parents[3]
CURATED_ADMET_RAG_CORPUS_DIR = RUNTIME_PROJECT_ROOT / "data" / "rag_corpus" / "admet_curated"
CURATED_DDI_RAG_CORPUS_DIR = RUNTIME_PROJECT_ROOT / "data" / "rag_corpus" / "ddi_curated"


DRUG_TEXT_ALIASES = {
    "阿司匹林": "aspirin",
    "乙酰水杨酸": "aspirin",
    "咖啡因": "caffeine",
    "布洛芬": "ibuprofen",
    "萘普生": "naproxen",
    "华法林": "warfarin",
    "辛伐他汀": "simvastatin",
    "酮康唑": "ketoconazole",
    "红霉素": "erythromycin",
    "氯吡格雷": "clopidogrel",
    "奥美拉唑": "omeprazole",
    "奎尼丁": "quinidine",
    "二甲双胍": "metformin",
    "咪达唑仑": "midazolam",
    "普萘洛尔": "propranolol",
    "纳曲酮": "naltrexone",
    "阿巴卡韦": "abacavir",
    "度鲁特韦": "dolutegravir",
    "瑞舒伐他汀": "rosuvastatin",
    "紫杉醇": "paclitaxel",
    "对乙酰氨基酚": "acetaminophen",
    "扑热息痛": "acetaminophen",
}


DRUG_SPELLING_ALIASES = {
    "asa": "aspirin",
}


LOCAL_COMPOUND_IDENTITY_SEEDS = {
    "isoniazid": {
        "title": "Isoniazid",
        "pubchem_cid": "3767",
        "canonical_smiles": "NNC(=O)c1ccncc1",
        "synonyms": ["isoniazid", "54-85-3"],
        "identity_source": "https://pubchem.ncbi.nlm.nih.gov/compound/3767",
        "identity_crosscheck": "MoleculeNet raw BBBP.csv row num=279, name=isoniazid; SMILES only, no label used",
        "identity_source_file_sha256": "d07a38487aeac5cee5508413e468043ef3097451d2a112701c2d60be9ec6b662",
    },
    "aspirin": {
        "title": "Aspirin",
        "pubchem_cid": "2244",
        "canonical_smiles": "CC(=O)OC1=CC=CC=C1C(=O)O",
        "synonyms": ["aspirin", "acetylsalicylic acid", "50-78-2"],
    },
    "caffeine": {
        "title": "Caffeine",
        "pubchem_cid": "2519",
        "canonical_smiles": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
        "synonyms": ["caffeine", "58-08-2", "1,3,7-trimethylxanthine"],
    },
    "simvastatin": {
        "title": "Simvastatin",
        "pubchem_cid": "54454",
        "canonical_smiles": "CCC(C)(C)C(=O)O[C@H]1C[C@H](C=C2[C@H]1[C@H]([C@H](C=C2)C)CC[C@@H]3C[C@H](CC(=O)O3)O)C",
        "synonyms": ["simvastatin", "79902-63-9", "zocor"],
    },
    "ketoconazole": {
        "title": "Ketoconazole",
        "pubchem_cid": "47576",
        "canonical_smiles": "CC(=O)N1CCN(CC1)C2=CC=C(C=C2)OC[C@@H]3CO[C@@](O3)(CN4C=CN=C4)C5=C(C=C(C=C5)Cl)Cl",
        "synonyms": ["ketoconazole", "65277-42-1", "nizoral"],
    },
    "warfarin": {
        "title": "Warfarin",
        "pubchem_cid": "54678486",
        "canonical_smiles": "CC(=O)CC(C1=CC=CC=C1)C2=C(O)C3=CC=CC=C3OC2=O",
        "synonyms": ["warfarin", "coumadin"],
    },
    "omeprazole": {
        "title": "Omeprazole",
        "pubchem_cid": "4594",
        "canonical_smiles": "COC1=CC2=C(NC(=N2)S(=O)CC3=NC=C(C)C(OC)=C3C)C=C1",
        "synonyms": ["omeprazole", "prilosec"],
    },
    "propranolol": {
        "title": "Propranolol",
        "pubchem_cid": "4946",
        "canonical_smiles": "CC(C)NCC(O)COC1=CC=CC2=CC=CC=C12",
        "synonyms": ["propranolol", "inderal"],
    },
    "diclofenac": {
        "title": "Diclofenac",
        "pubchem_cid": "3033",
        "canonical_smiles": "O=C(O)CC1=CC=CC=C1NC2=C(Cl)C=CC=C2Cl",
        "synonyms": ["diclofenac", "voltaren"],
    },
    "ibuprofen": {
        "title": "Ibuprofen",
        "pubchem_cid": "3672",
        "canonical_smiles": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
        "synonyms": ["ibuprofen", "advil", "motrin", "15687-27-1"],
    },
    "metformin": {
        "title": "Metformin",
        "pubchem_cid": "4091",
        "canonical_smiles": "CN(C)C(=N)N=C(N)N",
        "synonyms": ["metformin", "glucophage"],
    },
    "prochlorperazine": {
        "title": "Prochlorperazine",
        "pubchem_cid": "4917",
        "canonical_smiles": "CN1CCN(CC1)CCCN2C3=CC=CC=C3SC4=C2C=C(C=C4)Cl",
        "synonyms": ["prochlorperazine", "compazine"],
    },
    "atenolol": {
        "title": "Atenolol",
        "pubchem_cid": "2249",
        "canonical_smiles": "CC(C)NCC(O)COC1=CC=C(CC(=O)N)C=C1",
        "synonyms": ["atenolol", "tenormin"],
    },
    "digoxin": {
        "title": "Digoxin",
        "pubchem_cid": "2724385",
        "canonical_smiles": "CC1OC(OC2CC(OC3CC(OC4CCC5(C)C(CCC6C5CC(O)C5(C)C(C7=CC(=O)OC7)CCC65O)O4)O3)O2)C(O)C1O",
        "synonyms": ["digoxin", "lanoxin"],
    },
    "naproxen": {
        "title": "Naproxen",
        "pubchem_cid": "156391",
        "canonical_smiles": "COC1=CC=C2C=C(C=CC2=C1)C(C)C(=O)O",
        "synonyms": ["naproxen", "aleve"],
    },
    "cyclosporine": {
        "title": "Cyclosporine",
        "pubchem_cid": "5284373",
        "canonical_smiles": r"CC[C@@H]1NC(=O)[C@H]([C@H](O)[C@H](C)C\C=C\C)N(C)C(=O)[C@H](C(C)C)N(C)C(=O)[C@H](CC(C)C)N(C)C(=O)[C@H](CC(C)C)N(C)C(=O)[C@@H](C)NC(=O)[C@H](C)NC(=O)[C@H](CC(C)C)N(C)C(=O)[C@@H](NC(=O)[C@H](CC(C)C)N(C)C(=O)CN(C)C1=O)C(C)C",
        "synonyms": ["cyclosporine", "cyclosporin", "cyclosporin a", "cyclosporine a", "59865-13-3"],
    },
    "loxoprofen": {
        "title": "Loxoprofen",
        "pubchem_cid": "3965",
        "canonical_smiles": "CC(C1=CC=C(C=C1)CC2CCCC2=O)C(=O)O",
        "synonyms": ["loxoprofen", "68767-14-6"],
    },
    "pentobarbital": {
        "title": "Pentobarbital",
        "pubchem_cid": "4737",
        "canonical_smiles": "CCCC(C)C1(CC)C(=O)NC(=O)NC1=O",
        "synonyms": ["pentobarbital", "pentobarbitone", "76-74-4"],
    },
    "alprazolam": {
        "title": "Alprazolam",
        "pubchem_cid": "2118",
        "canonical_smiles": "CC1=NN=C2CN=C(C3=CC=CC=C3)C3=C(C=CC(Cl)=C3)N12",
        "synonyms": ["alprazolam", "xanax", "28981-97-7"],
    },
    "clarithromycin": {
        "title": "Clarithromycin",
        "pubchem_cid": "84029",
        "canonical_smiles": "CC[C@@H]1[C@@]([C@@H]([C@H](C(=O)[C@@H](C[C@@]([C@@H]([C@H]([C@@H]([C@H](C(=O)O1)C)O[C@H]2C[C@@]([C@H]([C@@H](O2)C)O)(C)OC)C)O[C@H]3[C@@H]([C@H](C[C@H](O3)C)N(C)C)O)(C)OC)C)C)O)(C)O",
        "synonyms": ["clarithromycin", "biaxin", "81103-11-9"],
    },
    "fluoxetine": {
        "title": "Fluoxetine",
        "pubchem_cid": "3386",
        "canonical_smiles": "CNCCC(C1=CC=CC=C1)OC2=CC=C(C=C2)C(F)(F)F",
        "synonyms": ["fluoxetine", "prozac", "54910-89-3"],
    },
    "metoprolol": {
        "title": "Metoprolol",
        "pubchem_cid": "4171",
        "canonical_smiles": "CC(C)NCC(COC1=CC=C(C=C1)CCOC)O",
        "synonyms": ["metoprolol", "lopressor", "toprol", "37350-58-6"],
    },
    "amiodarone": {
        "title": "Amiodarone",
        "pubchem_cid": "2157",
        "canonical_smiles": "CCCCC1=C(C2=CC=CC=C2O1)C(=O)C3=CC(=C(C(=C3)I)OCCN(CC)CC)I",
        "synonyms": ["amiodarone", "cordarone", "1951-25-3"],
    },
    "ciprofloxacin": {
        "title": "Ciprofloxacin",
        "pubchem_cid": "2764",
        "canonical_smiles": "C1CC1N2C=C(C(=O)C3=CC(=C(C=C32)N4CCNCC4)F)C(=O)O",
        "synonyms": ["ciprofloxacin", "cipro", "85721-33-1"],
    },
    "tizanidine": {
        "title": "Tizanidine",
        "pubchem_cid": "5487",
        "canonical_smiles": "C1CN=C(N1)NC2=C(C=CC3=NSN=C32)Cl",
        "synonyms": ["tizanidine", "zanaflex", "51322-75-9"],
    },
}


def _evidence_key(item: Dict[str, Any]) -> tuple[str, str, str]:
    source = str(item.get("source") or item.get("source_group") or item.get("provider") or "")
    locator = str(
        item.get("citation_id")
        or item.get("pmid")
        or item.get("doi")
        or item.get("chunk_id")
        or item.get("source_path")
        or item.get("title")
        or ""
    )
    claim = str(item.get("claim") or item.get("snippet") or item.get("mechanism") or "")[:500]
    return (source, locator, claim)


class ScientificToolRuntime:
    def __init__(
        self,
        *,
        output_dir: str | Path,
        feature_set: str = DEFAULT_FEATURE_SET,
        model_path: Optional[str] = None,
        auto_model: bool = False,
        model_registry_path: Optional[str] = None,
        require_real_model: bool = False,
        require_real_rag: bool = False,
        min_evidence: int = 1,
        use_external_evidence: bool = False,
        rag_corpus_dirs: Optional[List[str]] = None,
        rag_top_k: int = DEFAULT_RAG_TOP_K,
        rag_backend: str = DEFAULT_RAG_BACKEND,
        rag_embedding_model_name: str = DEFAULT_RAG_EMBEDDING_MODEL_PATH,
        ddi_rag_corpus_dir: str = DEFAULT_DDI_RAG_CORPUS_DIR,
        use_smiles_encoder: bool = False,
        use_mllm_visual_semantic: bool = False,
        mllm_profile: Optional[str] = DEFAULT_MLLM_PROFILE,
        imagemol_checkpoint_path: Optional[str] = DEFAULT_IMAGEMOL_CHECKPOINT_PATH,
        retrieval_backend: str = "auto",
        es_url: Optional[str] = None,
        es_index_name: Optional[str] = None,
        llm_profile: Optional[str] = None,
        use_llm_reasoning: bool = True,
        require_llm_reasoning: bool = False,
        reasoning_llm_client: Any | None = None,
        reaction_data_root: Optional[str | Path] = None,
        reaction_model_dir: Optional[str | Path] = None,
        reaction_index_dir: Optional[str | Path] = None,
    ) -> None:
        self.output_dir = ensure_dir(output_dir)
        self.feature_set = feature_set
        inferred_use_smiles_encoder = bool(use_smiles_encoder) or feature_set in {"all_features", "smiles_only"}
        self.model_path = model_path
        self.auto_model = bool(auto_model)
        self.model_registry_path = model_registry_path
        self.require_real_model = bool(require_real_model)
        self.require_real_rag = bool(require_real_rag)
        self.min_evidence = int(min_evidence)
        self.retrieval_backend = retrieval_backend
        self.llm_profile = llm_profile
        self.use_llm_reasoning = bool(use_llm_reasoning)
        self.require_llm_reasoning = bool(require_llm_reasoning)
        self.reasoning_llm_client = reasoning_llm_client
        self.rag_corpus_dirs = [default_resource_path(path) for path in (rag_corpus_dirs or [DEFAULT_RAG_CORPUS_DIR])]
        if CURATED_ADMET_RAG_CORPUS_DIR.exists() and CURATED_ADMET_RAG_CORPUS_DIR not in self.rag_corpus_dirs:
            self.rag_corpus_dirs.append(CURATED_ADMET_RAG_CORPUS_DIR)
        self.rag_top_k = rag_top_k
        self.input_agent = InputAgent()
        self.processor = MoleculeProcessingAgent()
        self.rule_agent = RuleDiscoveryAgent(output_dir=self.output_dir / "rules", llm_profile=llm_profile, use_llm=False)
        self.feature_builder = FeatureBuilder(
            output_dir=self.output_dir / "features",
            feature_set=feature_set,
            imagemol_checkpoint_path=imagemol_checkpoint_path,
            use_smiles_encoder=inferred_use_smiles_encoder,
            smiles_encoder_model_name=DEFAULT_CHEMBERTA2_MODEL_PATH,
            smiles_local_files_only=True,
            use_mllm_visual_semantic=use_mllm_visual_semantic,
            mllm_profile=mllm_profile,
            generate_grad_cam=False,
        )
        self.prediction_agent = PredictionAgent(model_path=None)
        self._prediction_agents: Dict[str, PredictionAgent] = {}
        self.evidence_agent = EvidenceAgent(use_external=use_external_evidence)
        self.rag_evidence_agent = RAGEvidenceAgent(
            output_dir=self.output_dir / "rag",
            corpus_dirs=self.rag_corpus_dirs,
            top_k=rag_top_k,
            rag_backend=rag_backend,
            embedding_model_name=default_resource_path(rag_embedding_model_name),
            embedding_local_files_only=True,
            use_llm=False,
        )
        self.curated_admet_rag_evidence_agent = RAGEvidenceAgent(
            output_dir=self.output_dir / "rag_curated_admet",
            corpus_dirs=[CURATED_ADMET_RAG_CORPUS_DIR],
            top_k=rag_top_k,
            rag_backend=rag_backend,
            embedding_model_name=default_resource_path(rag_embedding_model_name),
            embedding_local_files_only=True,
            use_llm=False,
        )
        self.ddi_rag_evidence_agent = DDIRAGEvidenceAgent(
            output_dir=self.output_dir / "ddi_rag",
            corpus_dir=default_resource_path(ddi_rag_corpus_dir),
            build_on_demand=False,
            top_k=rag_top_k,
            rag_backend=rag_backend,
            embedding_model_name=default_resource_path(rag_embedding_model_name),
            embedding_local_files_only=True,
        )
        self.curated_ddi_rag_evidence_agent = DDIRAGEvidenceAgent(
            output_dir=self.output_dir / "ddi_rag_curated",
            corpus_dir=CURATED_DDI_RAG_CORPUS_DIR,
            build_on_demand=False,
            top_k=rag_top_k,
            rag_backend=rag_backend,
            embedding_model_name=default_resource_path(rag_embedding_model_name),
            embedding_local_files_only=True,
        )
        self.hybrid_retriever = HybridRetriever(
            es_url=es_url,
            index_name=es_index_name,
            top_k=rag_top_k,
            embedding_model_name=default_resource_path(rag_embedding_model_name),
        )
        self.reaction_runtime = ReactionRuntime(
            output_dir=self.output_dir,
            data_root=reaction_data_root or DEFAULT_REACTION_DATA_ROOT,
            model_dir=reaction_model_dir,
            index_dir=reaction_index_dir,
        )
        if self.use_llm_reasoning and self.reasoning_llm_client is None:
            try:
                from agentm.llm.provider import build_llm_client

                self.reasoning_llm_client = build_llm_client(profile_name=llm_profile)
            except Exception:
                self.reasoning_llm_client = None

    def registry(self) -> MultiAgentToolRegistry:
        registry = MultiAgentToolRegistry()
        specs = [
            ("MoleculeAgent", "route_task", "Parse and route request into task, dataset, and molecule mentions.", self.route_task),
            ("MoleculeAgent", "resolve_molecule", "Resolve one molecule into a core MoleculeRecord.", self.resolve_molecule),
            ("MoleculeAgent", "resolve_ddi_pair", "Resolve two molecules and prepare a DDI pair request.", self.resolve_ddi_pair),
            ("GaugeAgent", "discover_rules", "Discover or load task rules using core RuleDiscoveryAgent.", self.discover_rules),
            ("GaugeAgent", "build_features", "Build core feature bundles and optional DDI pair features.", self.build_features),
            ("GaugeAgent", "predict_single", "Run core single-molecule prediction.", self.predict_single),
            ("GaugeAgent", "predict_ddi", "Run core DDI prediction.", self.predict_ddi),
            ("GaugeAgent", "assess_confidence", "Assess confidence and need for evidence expansion.", self.assess_confidence),
            ("GaugeAgent", "run_zero_shot_fallback", "Explicitly run core zero-shot fallback prediction.", self.run_zero_shot_fallback),
            ("RetrieveAgent", "retrieve_local_rag", "Retrieve local RAG evidence using core RAG agents.", self.retrieve_local_rag),
            ("RetrieveAgent", "retrieve_elasticsearch_hybrid", "Retrieve ES BM25 evidence, falling back to local RAG.", self.retrieve_elasticsearch_hybrid),
            ("RetrieveAgent", "retrieve_external_evidence", "Retrieve core structured/external evidence when configured.", self.retrieve_external_evidence),
            ("RetrieveAgent", "expand_evidence_query", "Create an expanded evidence query for deep/repair retrieval.", self.expand_evidence_query),
            ("RetrieveAgent", "retrieve_mit_template_evidence", "Retrieve MIT product/template evidence for similar reactions.", self.reaction_runtime.retrieve_mit_template_evidence),
            ("RetrieveAgent", "retrieve_orderly_condition_evidence", "Retrieve ORDerly solvent/reagent/temperature/time/yield condition evidence.", self.reaction_runtime.retrieve_orderly_condition_evidence),
            ("ReactionAgent", "parse_reaction", "Parse reaction SMILES into reactants and products.", self.reaction_runtime.parse_reaction),
            ("ReactionAgent", "classify_reaction_type", "Run Schneider reaction type classification.", self.reaction_runtime.classify_reaction_type),
            ("ReactionAgent", "predict_buchwald_hartwig_yield", "Run Buchwald-Hartwig KG yield prediction.", self.reaction_runtime.predict_buchwald_hartwig_yield),
            ("ReactionAgent", "build_reaction_profile", "Build a structured reaction profile from reaction predictions and evidence.", self.reaction_runtime.build_reaction_profile),
            ("ReasoningAgent", "build_structured_explanation", "Build a grounded structured explanation from prediction and evidence.", self.build_structured_explanation),
            ("ReasoningAgent", "summarize_evidence_support", "Summarize evidence support and unsupported areas.", self.summarize_evidence_support),
            ("VerifierAgent", "verify_prediction_grounding", "Verify prediction came from GaugeAgent tool trace.", self.verify_prediction_grounding),
            ("VerifierAgent", "verify_citation_grounding", "Verify evidence citations are traceable.", self.verify_citation_grounding),
            ("VerifierAgent", "verify_report_schema", "Verify final report has core-compatible core fields.", self.verify_report_schema),
            ("ReflectionAgent", "reflect_on_report", "Use the current report and verifier output as the next planning input.", self.reflect_on_report),
            ("ReflectionAgent", "apply_reflection_feedback", "Apply reflection feedback and request bounded refinement when useful.", self.apply_reflection_feedback),
            ("ReporterAgent", "build_runtime_report", "Build the runtime multi-agent report payload.", self.build_runtime_report),
            ("ReporterAgent", "build_compatible_report", "Build core-compatible single/DDI report payload.", self.build_compatible_report),
            ("ReporterAgent", "write_artifacts", "Write runtime and compatibility artifacts.", self.write_artifacts),
        ]
        for agent, name, description, handler in specs:
            registry.register(
                ToolSpec(
                    agent=agent,
                    name=name,
                    description=description,
                    input_schema={"type": "object", "additionalProperties": True},
                    handler=handler,
                )
            )
        return registry

    def route_task(self, args: Dict[str, Any], context: Dict[str, Any]) -> ToolResult:
        normalized_request = self._normalize_request_for_parsing(context.get("request", ""))
        explicit_task = bool(context.get("task_type"))
        explicit_dataset = bool(context.get("dataset_name"))
        preflight = self._preflight_scientific_intent(
            normalized_request,
            explicit_task=explicit_task,
            explicit_dataset=explicit_dataset,
        )
        if preflight.get("status") in NON_SCIENCE_TASK_TYPES:
            context["task_type"] = str(preflight["status"])
            context["dataset_name"] = NON_SCIENCE_DATASET_NAME
            context["normalized_request_for_parsing"] = normalized_request
            context["molecule_mentions"] = []
            metadata = context.setdefault("metadata", {})
            metadata["intent"] = context["task_type"]
            metadata["intent_gate"] = preflight
            metadata["prediction_requested"] = False
            context.setdefault("warnings", []).append(f"intent_gate:{context['task_type']}")
            context.setdefault("reasoning", {})["routing"] = {
                "intent": context["task_type"],
                "reason": preflight.get("reason"),
                "suggestion": preflight.get("suggestion"),
            }
            self._update_resource_resolution(context)
            routed_graph = self._task_graph_for(context)
            context.setdefault("orchestrator_plan", {})["routed_task_graph"] = routed_graph
            if context.get("orchestrator_plan", {}).get("planner_type") != "llm":
                context["task_graph"] = routed_graph
            return ToolResult(
                "route_task",
                "completed",
                {
                    "task_type": context["task_type"],
                    "dataset_name": context["dataset_name"],
                    "molecules": [],
                    "intent_gate": preflight,
                },
            )
        if is_reaction_request(normalized_request, context.get("task_type") or None):
            context["task_type"] = infer_reaction_task_type(normalized_request, context.get("task_type") or None)
            context["dataset_name"] = context.get("dataset_name") or self._reaction_dataset_name(context["task_type"])
            context["reaction_request"] = normalized_request
            context["molecule_mentions"] = []
            context.setdefault("metadata", {})["intent"] = "reaction"
            context.setdefault("reasoning", {})["routing"] = {
                "intent": context["task_type"],
                "reason": "Reaction markers or an explicit reaction task were detected.",
            }
            self._update_resource_resolution(context)
            routed_graph = self._task_graph_for(context)
            context.setdefault("orchestrator_plan", {})["routed_task_graph"] = routed_graph
            if context.get("orchestrator_plan", {}).get("planner_type") != "llm":
                context["task_graph"] = routed_graph
            return ToolResult(
                "route_task",
                "completed",
                {"task_type": context["task_type"], "dataset_name": context["dataset_name"], "reaction": True},
            )
        parsed = self.input_agent.parse(
            normalized_request,
            task_type=context.get("task_type") or None,
            dataset_name=context.get("dataset_name") or None,
        )
        is_general_profile = self._is_general_drug_profile_request(
            normalized_request,
            parsed_task=parsed.task_type,
            parsed_dataset=parsed.dataset_name,
            explicit_task=explicit_task,
            explicit_dataset=explicit_dataset,
        )
        context["task_type"] = GENERAL_TASK_TYPE if is_general_profile else parsed.task_type
        context["dataset_name"] = GENERAL_DATASET_NAME if is_general_profile else parsed.dataset_name
        context["normalized_request_for_parsing"] = normalized_request
        parsed_mentions = [self._normalize_molecule_mention(item) for item in parsed.molecules]
        seeded_mentions = self._local_seed_mentions_in_text(normalized_request)
        required_mentions = 2 if context["task_type"] == "ddi_prediction" else 1
        if len(seeded_mentions) >= required_mentions:
            context["molecule_mentions"] = seeded_mentions[:required_mentions]
            context.setdefault("metadata", {})["molecule_mention_resolution"] = {
                "strategy": "explicit_local_identity_seed",
                "parsed_mentions": parsed_mentions,
                "selected_mentions": list(context["molecule_mentions"]),
            }
        else:
            context["molecule_mentions"] = parsed_mentions
        if is_general_profile:
            metadata = context.setdefault("metadata", {})
            metadata["intent"] = GENERAL_TASK_TYPE
            metadata["prediction_requested"] = False
            context.setdefault("reasoning", {})["routing"] = {
                "intent": GENERAL_TASK_TYPE,
                "reason": "No explicit ADMET endpoint, property dataset, or DDI intent was requested.",
            }
        self._update_resource_resolution(context)
        routed_graph = self._task_graph_for(context)
        context.setdefault("orchestrator_plan", {})["routed_task_graph"] = routed_graph
        if context.get("orchestrator_plan", {}).get("planner_type") != "llm":
            context["task_graph"] = routed_graph
        return ToolResult(
            "route_task",
            "completed",
            {"task_type": context["task_type"], "dataset_name": context["dataset_name"], "molecules": context["molecule_mentions"]},
        )

    def resolve_molecule(self, args: Dict[str, Any], context: Dict[str, Any]) -> ToolResult:
        self._activate_open_subtask(args, context, expected_kind="single")
        mentions = self._mentions(args, context, target_count=1)
        molecule = self._process_molecule_mention(mentions[0] if mentions else context.get("request", ""))
        context["molecules"] = [molecule]
        self._record_open_subtask_state(context, "molecules", [molecule.to_dict()])
        return ToolResult("resolve_molecule", "completed", {"n_molecules": 1, "molecules": [molecule.to_dict()]})

    def resolve_ddi_pair(self, args: Dict[str, Any], context: Dict[str, Any]) -> ToolResult:
        self._activate_open_subtask(args, context, expected_kind="ddi")
        mentions = self._mentions(args, context, target_count=2)
        if len(mentions) < 2:
            context.setdefault("warnings", []).append("ddi_request_has_less_than_two_molecules")
        while len(mentions) < 2:
            mentions.append(context.get("request", ""))
        molecules = [self._process_molecule_mention(value) for value in mentions[:2]]
        context["molecules"] = molecules
        self._record_open_subtask_state(context, "molecules", [molecule.to_dict() for molecule in molecules])
        return ToolResult("resolve_ddi_pair", "completed", {"n_molecules": len(molecules), "molecules": [m.to_dict() for m in molecules]})

    def discover_rules(self, args: Dict[str, Any], context: Dict[str, Any]) -> ToolResult:
        self._activate_open_subtask(args, context)
        cached_rules = self._load_cached_training_rules(context)
        if cached_rules is not None:
            cached_rules = self._filter_rules_to_expected_schema(cached_rules, context)
            context["rules"] = cached_rules
            return ToolResult("discover_rules", "completed", {"n_rules": len(cached_rules), "source": "cached_training_rules"})
        try:
            rules = self.rule_agent.discover(context.get("task_type", "property_prediction"), context.get("dataset_name", "BBBP"))
            status = "completed"
        except Exception as exc:
            if not self._fallbacks_enabled(context):
                context.setdefault("warnings", []).append(f"rule_discovery_failed_fallback_disabled: {exc}")
                return ToolResult("discover_rules", "failed", error=str(exc))
            rules = []
            status = "fallback_completed"
            self._fallback(context, "discover_rules", f"rule_discovery_failed_used_empty_rules: {exc}")
        context["rules"] = rules
        return ToolResult("discover_rules", status, {"n_rules": len(rules)})

    def _load_cached_training_rules(self, context: Dict[str, Any]) -> Optional[List[Rule]]:
        self._update_resource_resolution(context)
        selected = self._selected_model_path(context)
        if not selected:
            return None
        model_path = Path(selected)
        candidates = []
        model_resolution = (context.get("resource_resolution") or {}).get("model") or {}
        selected_candidate = self._selected_model_candidate(model_resolution, selected)
        selected_metadata = selected_candidate.get("metadata") if isinstance(selected_candidate.get("metadata"), dict) else {}
        rule_metadata = selected_metadata.get("rule_metadata") if isinstance(selected_metadata.get("rule_metadata"), dict) else {}
        # Every current core training run persists the exact rule payload next to the
        # model, including runs that reused another job's rule cache and therefore
        # do not own a local ``rules/`` directory (for example, cold-drug DDI).
        candidates.append(model_path.parent / "rules_used.json")
        if rule_metadata.get("summary_path"):
            candidates.append(Path(default_resource_path(rule_metadata["summary_path"])))
        metadata_task_type = (
            model_resolution.get("metadata_task_type")
            or selected_candidate.get("metadata_task_type")
            or selected_metadata.get("task_type")
            or context.get("task_type")
            or "property_prediction"
        )
        metadata_dataset = (
            model_resolution.get("metadata_dataset_name")
            or selected_candidate.get("metadata_dataset_name")
            or selected_metadata.get("dataset_name")
            or context.get("dataset_name")
            or "BBBP"
        )
        for task_type, dataset_name in [
            (metadata_task_type, metadata_dataset),
            (context.get("task_type") or "property_prediction", context.get("dataset_name") or "BBBP"),
        ]:
            safe_dataset = "".join(ch if str(ch).isalnum() else "_" for ch in str(dataset_name))
            candidates.append(model_path.parent / "rules" / f"{task_type}_{safe_dataset}_rules.json")
            candidates.append(model_path.parent / "rules" / f"{task_type}_{dataset_name}_rules.json")
        for path in candidates:
            if not path.exists():
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                rule_payload = payload.get("rules") if isinstance(payload, dict) else payload
                if not isinstance(rule_payload, list):
                    continue
                rules = [self._rule_from_mapping(item) for item in rule_payload if isinstance(item, dict)]
                if rules:
                    context.setdefault("artifacts", {})["rules_source_path"] = str(path)
                    return rules
            except Exception as exc:
                context.setdefault("warnings", []).append(f"cached_training_rules_load_failed:{path}: {exc}")
        return None

    def _selected_model_candidate(self, model_resolution: Dict[str, Any], selected_path: str) -> Dict[str, Any]:
        selected = Path(selected_path)
        for item in model_resolution.get("candidates", []) or []:
            if not isinstance(item, dict):
                continue
            if item.get("selected") or (item.get("path") and Path(str(item.get("path"))) == selected):
                return item
        return {}

    def _ensure_cached_rules_for_feature_alignment(self, context: Dict[str, Any]) -> None:
        expected = self._expected_model_rule_names(context)
        if not expected:
            return
        existing = context.get("rules") or []
        existing_names = {str(getattr(rule, "feature_name", "") or "") for rule in existing}
        if len(existing_names.intersection(expected)) == len(expected):
            return
        cached = self._load_cached_training_rules(context)
        if cached is None:
            context.setdefault("warnings", []).append(f"cached_training_rules_missing_for_model_schema:{len(expected)}")
            return
        aligned = self._filter_rules_to_expected_schema(cached, context, expected_rule_names=expected)
        if len(aligned) < len(expected):
            context.setdefault("warnings", []).append(f"cached_training_rules_incomplete:{len(aligned)}<{len(expected)}")
        context["rules"] = aligned

    def _expected_model_rule_names(self, context: Dict[str, Any]) -> List[str]:
        self._update_resource_resolution(context)
        selected = self._selected_model_path(context)
        if not selected:
            return []
        model_path = Path(selected)
        feature_names: List[str] = []
        schema_path = model_path.parent / "features" / "feature_schema.json"
        if schema_path.exists():
            try:
                payload = json.loads(schema_path.read_text(encoding="utf-8"))
                raw_names = payload.get("feature_names") if isinstance(payload, dict) else []
                if isinstance(raw_names, list):
                    feature_names = [str(name) for name in raw_names]
            except Exception as exc:
                context.setdefault("warnings", []).append(f"model_feature_schema_read_failed:{schema_path}: {exc}")
        if not feature_names:
            try:
                from agentm.models.predictors import SklearnPredictor

                model = SklearnPredictor.load(model_path)
                schema = model.feature_schema or {}
                raw_names = schema.get("feature_names") if isinstance(schema, dict) else []
                if isinstance(raw_names, list):
                    feature_names = [str(name) for name in raw_names]
            except Exception as exc:
                context.setdefault("warnings", []).append(f"model_feature_schema_load_failed:{model_path}: {exc}")
        expected = [name.replace("llm_rule_", "", 1) for name in feature_names if name.startswith("llm_rule_")]
        context.setdefault("artifacts", {})["expected_rule_feature_names"] = expected
        return expected

    def _filter_rules_to_expected_schema(
        self,
        rules: List[Rule],
        context: Dict[str, Any],
        *,
        expected_rule_names: Optional[List[str]] = None,
    ) -> List[Rule]:
        expected = expected_rule_names if expected_rule_names is not None else self._expected_model_rule_names(context)
        if not expected:
            return rules
        by_name: Dict[str, Rule] = {}
        for rule in rules:
            name = str(getattr(rule, "feature_name", "") or "")
            if name in expected and name not in by_name:
                by_name[name] = rule
        filtered = [by_name[name] for name in expected if name in by_name]
        context.setdefault("artifacts", {})["rules_aligned_to_model_schema"] = {
            "expected": len(expected),
            "loaded": len(rules),
            "selected": len(filtered),
            "missing": [name for name in expected if name not in by_name],
        }
        return filtered or rules

    def _rule_from_mapping(self, item: Dict[str, Any]) -> Rule:
        return Rule(
            rule_id=str(item.get("rule_id") or item.get("id") or item.get("feature_name") or "cached_rule"),
            task_type=str(item.get("task_type") or "property_prediction"),
            source=str(item.get("source") or "cached_training_rules"),
            text=str(item.get("text") or item.get("feature_name") or ""),
            feature_name=str(item.get("feature_name") or item.get("rule_id") or "rule_cached"),
            code=str(item.get("code") or ""),
            metadata=item.get("metadata") if isinstance(item.get("metadata"), dict) else {},
            is_executable=bool(item.get("is_executable", True)),
            error=item.get("error"),
        )

    def build_features(self, args: Dict[str, Any], context: Dict[str, Any]) -> ToolResult:
        self._activate_open_subtask(args, context)
        self._ensure_cached_rules_for_feature_alignment(context)
        try:
            return self._build_features_with(self.feature_builder, context, status="completed")
        except Exception as exc:
            if not self._fallbacks_enabled(context):
                context.setdefault("warnings", []).append(f"feature_builder_failed_fallback_disabled: {exc}")
                return ToolResult("build_features", "failed", error=str(exc))
            self._fallback(context, "build_features", f"feature_builder_failed_used_rdkit_only: {exc}")
            fallback_builder = FeatureBuilder(output_dir=self.output_dir / "features_fallback", feature_set="rdkit_only")
            return self._build_features_with(fallback_builder, context, status="fallback_completed")

    def predict_single(self, args: Dict[str, Any], context: Dict[str, Any]) -> ToolResult:
        self._activate_open_subtask(args, context, expected_kind="single")
        bundle = (context.get("features") or [None])[0]
        agent = self._prediction_agent_for(context)
        selected_model = self._selected_model_path(context)
        if self._real_model_required(context) and not selected_model:
            context.setdefault("warnings", []).append("predict_single_real_model_required_but_missing")
            return ToolResult("predict_single", "failed", {"resource_resolution": context.get("resource_resolution", {})}, error="real_model_required_but_missing")
        try:
            prediction_dataset = self._prediction_dataset_name(context, default=context.get("dataset_name", "BBBP"))
            prediction = agent.predict_single(
                bundle,
                task_type=context.get("task_type", "property_prediction"),
                dataset_name=prediction_dataset,
            )
            self._annotate_prediction_dataset_alias(prediction, context, prediction_dataset)
            if self._is_zero_shot_prediction(prediction) and not self._fallbacks_enabled(context):
                context.setdefault("warnings", []).append("predict_single_returned_zero_shot_fallback_disabled")
                return ToolResult("predict_single", "failed", error="zero_shot_prediction_disabled")
            if self._real_model_required(context) and self._is_zero_shot_prediction(prediction):
                context.setdefault("warnings", []).append("predict_single_returned_zero_shot_real_model_required")
                return ToolResult("predict_single", "failed", error="real_model_required")
            status = "completed"
        except Exception as exc:
            if not self._fallbacks_enabled(context):
                context.setdefault("warnings", []).append(f"predict_single_failed_fallback_disabled: {exc}")
                return ToolResult("predict_single", "failed", error=str(exc))
            prediction = agent.zero_shot.predict_single(
                bundle,
                task_type=context.get("task_type", "property_prediction"),
                dataset_name=context.get("dataset_name", "BBBP"),
            )
            status = "fallback_completed"
            self._fallback(context, "predict_single", f"predict_single_failed_used_zero_shot: {exc}")
        self._append_open_task_prediction(context, prediction)
        self._record_open_subtask_state(
            context,
            "prediction",
            {**to_jsonable(prediction.to_dict()), "model_path": selected_model},
        )
        return ToolResult("predict_single", status, {"prediction": prediction.to_dict(), "model_type": prediction.model_type, "model_path": selected_model})

    def predict_ddi(self, args: Dict[str, Any], context: Dict[str, Any]) -> ToolResult:
        self._activate_open_subtask(args, context, expected_kind="ddi")
        pair = context.get("ddi_pair")
        agent = self._prediction_agent_for(context)
        selected_model = self._selected_model_path(context)
        if self._real_model_required(context) and not selected_model:
            context.setdefault("warnings", []).append("predict_ddi_real_model_required_but_missing")
            return ToolResult("predict_ddi", "failed", {"resource_resolution": context.get("resource_resolution", {})}, error="real_model_required_but_missing")
        try:
            prediction_dataset = self._prediction_dataset_name(context, default=context.get("dataset_name", "DDI"))
            prediction = agent.predict_ddi(pair, dataset_name=prediction_dataset)
            self._annotate_prediction_dataset_alias(prediction, context, prediction_dataset)
            annotate_multiclass_ddi_prediction(
                prediction,
                drug_names=self._molecule_names(context),
                model_path=selected_model,
            )
            if self._is_zero_shot_prediction(prediction) and not self._fallbacks_enabled(context):
                context.setdefault("warnings", []).append("predict_ddi_returned_zero_shot_fallback_disabled")
                return ToolResult("predict_ddi", "failed", error="zero_shot_prediction_disabled")
            if self._real_model_required(context) and self._is_zero_shot_prediction(prediction):
                context.setdefault("warnings", []).append("predict_ddi_returned_zero_shot_real_model_required")
                return ToolResult("predict_ddi", "failed", error="real_model_required")
            status = "completed"
        except Exception as exc:
            if not self._fallbacks_enabled(context):
                context.setdefault("warnings", []).append(f"predict_ddi_failed_fallback_disabled: {exc}")
                return ToolResult("predict_ddi", "failed", error=str(exc))
            prediction = agent.zero_shot.predict_ddi(pair, dataset_name=context.get("dataset_name", "DDI"))
            status = "fallback_completed"
            self._fallback(context, "predict_ddi", f"predict_ddi_failed_used_zero_shot: {exc}")
        self._append_open_task_prediction(context, prediction)
        self._record_open_subtask_state(
            context,
            "prediction",
            {**to_jsonable(prediction.to_dict()), "model_path": selected_model},
        )
        return ToolResult("predict_ddi", status, {"prediction": prediction.to_dict(), "model_type": prediction.model_type, "model_path": selected_model})

    def run_zero_shot_fallback(self, args: Dict[str, Any], context: Dict[str, Any]) -> ToolResult:
        if not self._fallbacks_enabled(context):
            context.setdefault("warnings", []).append("run_zero_shot_fallback_disabled")
            return ToolResult("run_zero_shot_fallback", "failed", error="fallback_disabled")
        agent = self._prediction_agent_for(context)
        if context.get("task_type") == "ddi_prediction":
            prediction = agent.zero_shot.predict_ddi(context.get("ddi_pair"), dataset_name=context.get("dataset_name", "DDI"))
        else:
            prediction = agent.zero_shot.predict_single(
                (context.get("features") or [None])[0],
                task_type=context.get("task_type", "property_prediction"),
                dataset_name=context.get("dataset_name", "BBBP"),
            )
        context["predictions"] = [prediction]
        self._fallback(context, "run_zero_shot_fallback", "explicit_zero_shot_fallback_used")
        return ToolResult("run_zero_shot_fallback", "fallback_completed", {"prediction": prediction.to_dict(), "model_type": prediction.model_type})

    def assess_confidence(self, args: Dict[str, Any], context: Dict[str, Any]) -> ToolResult:
        self._activate_open_subtask(args, context)
        prediction = self._active_prediction(context)
        probability = prediction_value(prediction, 'probability')
        score = float(probability if probability is not None else prediction_value(prediction, 'score', 0.0))
        margin = abs(score - 0.5)
        model_type = prediction_value(prediction, 'model_type', '')
        low_confidence = margin < 0.12 or model_type == "zero_shot_rule_semantic_fusion"
        confidence = {"score": score, "margin": margin, "low_confidence": low_confidence, "model_type": model_type}
        context["confidence"] = confidence
        context.setdefault("reasoning", {})["confidence"] = confidence
        return ToolResult("assess_confidence", "completed", confidence)

    def retrieve_external_evidence(self, args: Dict[str, Any], context: Dict[str, Any]) -> ToolResult:
        self._activate_open_subtask(args, context)
        try:
            if context.get("task_type") == GENERAL_TASK_TYPE:
                profiles, evidence = self._retrieve_general_database_profiles(context)
                added = self._append_evidence(context, evidence)
                context.setdefault("artifacts", {})["database_profile"] = profiles
                context.setdefault("retrieval_trace", []).append(
                    {
                        "provider": "database_profile",
                        "backend": "pubchem" if any(item.get("source") == "pubchem" for item in profiles) else "resolver_metadata",
                        "query": self._rag_query(context),
                        "evidence_items": evidence,
                        "retrieved_context": profiles,
                        "warnings": [],
                    }
                )
                return ToolResult(
                    "retrieve_external_evidence",
                    "completed",
                    {"n_database_profiles": len(profiles), "n_evidence_items": len(evidence), "n_unique_evidence_added": added},
                )
            if context.get("task_type") == "ddi_prediction" and context.get("ddi_pair") is not None:
                evidence = self.evidence_agent.retrieve_ddi(context["ddi_pair"])
            elif context.get("features"):
                evidence = self.evidence_agent.retrieve_single(context["features"][0], dataset_name=context.get("dataset_name"))
            else:
                evidence = []
            evidence = self._normalize_evidence(evidence, source_default="structured_evidence")
            context.setdefault("evidence", []).extend(evidence)
            return ToolResult("retrieve_external_evidence", "completed", {"n_evidence_items": len(evidence)})
        except Exception as exc:
            context.setdefault("warnings", []).append(f"external_evidence_failed: {exc}")
            return ToolResult("retrieve_external_evidence", "failed", error=str(exc))

    def retrieve_elasticsearch_hybrid(self, args: Dict[str, Any], context: Dict[str, Any]) -> ToolResult:
        self._activate_open_subtask(args, context)
        query = args.get("query") or self._rag_query(context)
        metadata = self._retrieval_metadata(context)
        requested_backend = args.get("backend") or context.get("metadata", {}).get("retrieval_backend_requested") or self.retrieval_backend
        if self._real_rag_required(context) and requested_backend != "es":
            self._append_warning(context, f"strict_real_rag_rejected_backend:{requested_backend}")
            return ToolResult(
                "retrieve_elasticsearch_hybrid",
                "failed",
                {"requested_backend": requested_backend},
                error="strict_real_rag_elasticsearch_backend_required",
            )
        if requested_backend == "local":
            return self._run_local_rag_as_hybrid_fallback(query, context, "requested_local_backend")
        result = self.hybrid_retriever.retrieve(
            query,
            metadata=metadata,
            backend=requested_backend,
            top_k=args.get("top_k") or self.rag_top_k,
        )
        result_payload = result.to_dict()
        context.setdefault("retrieval_trace", []).append(result_payload)
        result_metadata = result.metadata if isinstance(result.metadata, dict) else {}
        provenance = audit_evidence_provenance(
            {
                "evidence": [
                    *(item for item in self._active_subtask_evidence(context) if isinstance(item, Mapping)),
                    *(item for item in result.evidence_items or [] if isinstance(item, Mapping)),
                ]
            }
        )
        direct_evidence_available = bool(provenance.get("direct_evidence_coverage"))
        qc_grounding_only = bool(provenance.get("qc_only"))
        verified_dense_attempt = bool(
            result.provider == "elasticsearch_hybrid"
            and result.backend == "elasticsearch_hybrid_bm25_dense"
            and not result.warnings
            and (
                bool(result.retrieved_context)
                or (
                    bool(result_metadata.get("dense_enabled"))
                    and str(result_metadata.get("dense_status") or "") == "completed"
                )
            )
        )
        if verified_dense_attempt:
            # Backend execution and scientific evidence availability are
            # distinct. Empty evidence remains explicit and never activates a
            # fallback path.
            context.setdefault("metadata", {})["retrieval_backend_used"] = result.backend
            context["metadata"]["real_rag_execution_valid"] = True
            context["metadata"]["retrieval_outcome"] = (
                "evidence_found" if direct_evidence_available else "no_direct_evidence"
            )
            context["metadata"]["direct_evidence_available"] = direct_evidence_available
            context["metadata"]["external_evidence_available"] = bool(
                provenance.get("external_evidence_coverage")
            )
            context["metadata"]["qc_grounding_only"] = qc_grounding_only
            context["metadata"]["evidence_provenance_schema"] = provenance.get(
                "evidence_provenance_schema"
            )
            context["metadata"]["direct_evidence_count"] = int(
                provenance.get("direct_evidence_count", 0) or 0
            )
            context["metadata"]["qc_grounding_count"] = int(
                provenance.get("qc_grounding_count", 0) or 0
            )
            self._record_open_subtask_state(
                context,
                "retrieval",
                {
                    "outcome": context["metadata"]["retrieval_outcome"],
                    "direct_evidence_available": direct_evidence_available,
                    "qc_grounding_only": qc_grounding_only,
                    "backend": result.backend,
                },
            )
        if self._real_rag_required(context):
            strict_reasons = []
            if result.provider != "elasticsearch_hybrid":
                strict_reasons.append(f"provider={result.provider or 'missing'}")
            if result.backend != "elasticsearch_hybrid_bm25_dense":
                strict_reasons.append(f"backend={result.backend or 'missing'}")
            if result.warnings:
                strict_reasons.append("retrieval_warnings_present")
            if not strict_reasons and not verified_dense_attempt:
                strict_reasons.append("dense_hybrid_execution_unverified")
            if strict_reasons:
                for warning in result.warnings:
                    self._append_warning(context, f"hybrid_retrieval: {warning}")
                self._append_warning(
                    context,
                    "strict_real_rag_dense_hybrid_result_rejected:" + ";".join(strict_reasons),
                )
                return ToolResult(
                    "retrieve_elasticsearch_hybrid",
                    "failed",
                    result_payload,
                    error="strict_real_rag_dense_hybrid_result_required",
                )
        if result.evidence_items:
            evidence = self._normalize_evidence(result.evidence_items, source_default="elasticsearch")
            added = self._append_evidence(context, evidence)
            context.setdefault("artifacts", {})["retrieved_context"] = result.retrieved_context
            context.setdefault("metadata", {})["retrieval_backend_used"] = result.backend
            observation = result.to_dict()
            observation["retrieval_outcome"] = context.get("metadata", {}).get("retrieval_outcome")
            observation["direct_evidence_available"] = direct_evidence_available
            observation["qc_grounding_only"] = qc_grounding_only
            observation["n_unique_evidence_added"] = added
            observation["n_retained_evidence_total"] = self._current_grounded_evidence_count(context)
            if self._real_rag_required(context) and observation["n_retained_evidence_total"] < self._min_evidence(context):
                self._append_warning(context, f"hybrid_retrieval_retained_min_evidence_not_met:{observation['n_retained_evidence_total']}<{self._min_evidence(context)}")
                return ToolResult(
                    "retrieve_elasticsearch_hybrid",
                    "failed",
                    observation,
                    error="strict_real_rag_retained_min_evidence_not_met",
                )
            if self._real_rag_required(context) and not direct_evidence_available:
                self._append_warning(context, "no_direct_evidence_for_target")
            return ToolResult("retrieve_elasticsearch_hybrid", "completed", observation)
        if self._real_rag_required(context) and verified_dense_attempt:
            # A completed target-specific query with zero direct hits is a
            # valid retrieval outcome, not a transport failure.  Downstream
            # reporting must keep the model prediction separate from facts and
            # explicitly disclose the evidence gap.
            self._append_warning(context, "no_direct_evidence_for_target")
            context.setdefault("artifacts", {})["retrieved_context"] = result.retrieved_context
            observation = result.to_dict()
            observation.update(
                {
                    "retrieval_outcome": "no_direct_evidence",
                    "n_unique_evidence_added": 0,
                    "n_retained_evidence_total": self._current_grounded_evidence_count(context),
                }
            )
            return ToolResult("retrieve_elasticsearch_hybrid", "completed", observation)
        for warning in result.warnings:
            self._append_warning(context, f"hybrid_retrieval: {warning}")
        if self._real_rag_required(context):
            return ToolResult(
                "retrieve_elasticsearch_hybrid",
                "failed",
                result.to_dict(),
                error="strict_real_rag_elasticsearch_result_required",
            )
        if result.backend == "local_fallback_required" and requested_backend == "auto":
            self._append_warning(context, "hybrid_retrieval_auto_selected_local_real_rag")
            return self._run_local_rag_as_hybrid_fallback(query, context, "auto_selected_local_real_rag")
        if not self._fallbacks_enabled(context):
            return ToolResult("retrieve_elasticsearch_hybrid", "failed", result.to_dict(), error="hybrid_retrieval_failed_fallback_disabled")
        return self._run_local_rag_as_hybrid_fallback(query, context, "fallback_enabled")

    def _run_local_rag_as_hybrid_fallback(self, query: str, context: Dict[str, Any], reason: str) -> ToolResult:
        local_result = self.retrieve_local_rag({"query": query}, context)
        observation = dict(local_result.observation or {})
        observation.update(
            {
                "backend": observation.get("backend") or "local_rag",
                "fallback_tool": "retrieve_local_rag",
                "fallback_reason": reason,
                "fallback_status": local_result.status,
                "fallback_error": local_result.error,
            }
        )
        return ToolResult(
            "retrieve_elasticsearch_hybrid",
            local_result.status,
            observation,
            error=local_result.error,
        )

    def retrieve_local_rag(self, args: Dict[str, Any], context: Dict[str, Any]) -> ToolResult:
        self._activate_open_subtask(args, context)
        if self._real_rag_required(context):
            context.setdefault("warnings", []).append("strict_real_rag_local_tool_disabled")
            return ToolResult(
                "retrieve_local_rag",
                "failed",
                error="strict_real_rag_local_tool_disabled",
            )
        self._update_resource_resolution(context)
        rag_resolution = context.get("resource_resolution", {}).get("rag", {})
        if self._real_rag_required(context) and not rag_resolution.get("all_required_present", False):
            context.setdefault("warnings", []).append("retrieve_local_rag_real_resources_missing")
            return ToolResult("retrieve_local_rag", "failed", {"resource_resolution": rag_resolution}, error="real_rag_resources_missing")
        query = args.get("query") or self._rag_query(context)
        metadata = self._retrieval_metadata(context)
        try:
            if context.get("task_type") == "ddi_prediction" and context.get("ddi_pair") is not None:
                result = self.ddi_rag_evidence_agent.retrieve_for_pair(
                    context["ddi_pair"],
                    query=query,
                    context_metadata=metadata,
                    top_k=self.rag_top_k,
                )
                if not (result.get("evidence_items") or []):
                    curated_result = self.curated_ddi_rag_evidence_agent.retrieve_for_pair(
                        context["ddi_pair"],
                        query=query,
                        context_metadata=metadata,
                        top_k=self.rag_top_k,
                    )
                    if curated_result.get("evidence_items"):
                        context.setdefault("retrieval_trace", []).append(result)
                        curated_result.setdefault("warnings", []).append("curated_ddi_pair_corpus_used")
                        result = curated_result
            else:
                result = self.rag_evidence_agent.retrieve(
                    query=query,
                    context_metadata=metadata,
                    corpus_dirs=self.rag_corpus_dirs,
                    top_k=self.rag_top_k,
                )
            evidence = self._normalize_evidence(result.get("evidence_items", []) or [], source_default="local_rag")
            added = self._append_evidence(context, evidence)
            context.setdefault("artifacts", {})["rag_evidence"] = result
            context["artifacts"]["rag_context"] = result.get("retrieved_context", [])
            context["artifacts"]["retrieved_context"] = result.get("retrieved_context", [])
            context.setdefault("retrieval_trace", []).append(result)
            for warning in result.get("warnings", []) or []:
                self._append_warning(context, f"local_rag: {warning}")
            if (
                context.get("task_type") != "ddi_prediction"
                and self._real_rag_required(context)
                and self._current_grounded_evidence_count(context) < self._min_evidence(context)
            ):
                curated_result = self.curated_admet_rag_evidence_agent.retrieve(
                    query=query,
                    context_metadata=metadata,
                    corpus_dirs=[CURATED_ADMET_RAG_CORPUS_DIR],
                    top_k=self.rag_top_k,
                )
                curated_evidence = self._normalize_evidence(curated_result.get("evidence_items", []) or [], source_default="local_rag")
                added += self._append_evidence(context, curated_evidence)
                context["artifacts"]["admet_curated_rag_evidence"] = curated_result
                context["artifacts"]["rag_context"] = list(context["artifacts"].get("rag_context") or []) + list(curated_result.get("retrieved_context", []) or [])
                context["artifacts"]["retrieved_context"] = list(context["artifacts"].get("retrieved_context") or []) + list(curated_result.get("retrieved_context", []) or [])
                context.setdefault("retrieval_trace", []).append(curated_result)
                for warning in curated_result.get("warnings", []) or []:
                    self._append_warning(context, f"local_rag_curated_admet: {warning}")
                if curated_result.get("evidence_items"):
                    self._append_warning(context, "local_rag_curated_admet_supplement_used")
            context.setdefault("metadata", {})["retrieval_backend_used"] = result.get("backend") or "local_rag"
            retained_total = self._current_grounded_evidence_count(context)
            if self._real_rag_required(context) and len(evidence) < self._min_evidence(context) and retained_total < self._min_evidence(context):
                self._append_warning(context, f"retrieve_local_rag_min_evidence_not_met:{retained_total}<{self._min_evidence(context)}")
                return ToolResult(
                    "retrieve_local_rag",
                    "failed",
                    {"n_evidence_items": len(evidence), "n_unique_evidence_added": added, "n_retained_evidence_total": retained_total, "backend": result.get("backend"), "provider": result.get("provider")},
                    error="min_evidence_not_met",
                )
            if self._real_rag_required(context) and retained_total < self._min_evidence(context):
                self._append_warning(context, f"retrieve_local_rag_retained_min_evidence_not_met:{retained_total}<{self._min_evidence(context)}")
                return ToolResult(
                    "retrieve_local_rag",
                    "failed",
                    {"n_evidence_items": len(evidence), "n_unique_evidence_added": added, "n_retained_evidence_total": retained_total, "backend": result.get("backend"), "provider": result.get("provider")},
                    error="retained_min_evidence_not_met",
                )
            return ToolResult("retrieve_local_rag", "completed", {"n_evidence_items": len(evidence), "n_unique_evidence_added": added, "n_retained_evidence_total": retained_total, "backend": result.get("backend"), "provider": result.get("provider")})
        except Exception as exc:
            self._append_warning(context, f"local_rag_failed: {exc}")
            return ToolResult("retrieve_local_rag", "failed", error=str(exc))

    def expand_evidence_query(self, args: Dict[str, Any], context: Dict[str, Any]) -> ToolResult:
        base = args.get("query") or self._rag_query(context)
        names = " ".join(self._molecule_names(context))
        expanded = f"{base} {names} mechanism adverse event clinical evidence contraindication pharmacokinetic interaction"
        context.setdefault("reasoning", {})["expanded_evidence_query"] = expanded
        context.setdefault("metadata", {})["evidence_expansion_rounds"] = int(context.get("metadata", {}).get("evidence_expansion_rounds", 0)) + 1
        return ToolResult("expand_evidence_query", "completed", {"query": expanded})

    def build_structured_explanation(self, args: Dict[str, Any], context: Dict[str, Any]) -> ToolResult:
        metadata = context.setdefault("metadata", {})
        metadata["use_llm_reasoning"] = bool(metadata.get("use_llm_reasoning", self.use_llm_reasoning))
        metadata["require_llm_reasoning"] = bool(metadata.get("require_llm_reasoning", self.require_llm_reasoning))
        if metadata.get("safe_abstention_required") and not (
            context.get("reaction_predictions") or context.get("predictions")
        ):
            explanation = self._safe_abstention_explanation(context)
            context.setdefault("reasoning", {})["structured_explanation"] = explanation
            metadata["reasoning_mode"] = "safe_abstention_template"
            return ToolResult(
                "build_structured_explanation",
                "completed",
                {"explanation": explanation, "reasoning_mode": "safe_abstention_template"},
            )
        if context.get("task_type") in NON_SCIENCE_TASK_TYPES:
            explanation = self._non_science_explanation(context)
            context.setdefault("reasoning", {})["structured_explanation"] = explanation
            metadata["reasoning_mode"] = "intent_gate_template"
            return ToolResult("build_structured_explanation", "completed", {"explanation": explanation, "reasoning_mode": "intent_gate_template"})
        if context.get("task_type") == GENERAL_TASK_TYPE:
            explanation = self._general_drug_profile_explanation(context)
            context.setdefault("reasoning", {})["structured_explanation"] = explanation
            metadata["reasoning_mode"] = "general_profile_template"
            return ToolResult("build_structured_explanation", "completed", {"explanation": explanation, "reasoning_mode": "general_profile_template"})
        contract = self._open_task_contract(context)
        if contract and contract.get("open_task_type") in {"multi_endpoint_admet", "ddi_plus_individual_safety"}:
            self._restore_open_task_root(context)
            explanation = self._composite_open_task_explanation(context, contract)
            context.setdefault("reasoning", {})["structured_explanation"] = explanation
            metadata["reasoning_mode"] = "composite_task_template"
            return ToolResult(
                "build_structured_explanation",
                "completed",
                {"explanation": explanation, "reasoning_mode": "composite_task_template"},
            )
        if is_reaction_task(context.get("task_type")):
            profile = context.get("reaction_profile") or {}
            explanation = profile.get("summary") or reaction_profile_summary(
                context,
                context.get("reaction", {}),
                context.get("reaction_predictions") or context.get("predictions") or [],
                context.get("artifacts", {}).get("reaction_template_evidence", []),
                context.get("artifacts", {}).get("reaction_condition_evidence", []),
            )
            context.setdefault("reasoning", {})["structured_explanation"] = explanation
            metadata["reasoning_mode"] = "reaction_profile_template"
            return ToolResult("build_structured_explanation", "completed", {"explanation": explanation, "reasoning_mode": "reaction_profile_template"})
        if is_multiclass_ddi_prediction(self._prediction_summary(context)):
            explanation = self._deterministic_explanation(context)
            context.setdefault("reasoning", {})["structured_explanation"] = explanation
            metadata["reasoning_mode"] = "multiclass_semantics_template"
            return ToolResult(
                "build_structured_explanation",
                "completed",
                {"explanation": explanation, "reasoning_mode": "multiclass_semantics_template"},
            )
        if metadata["use_llm_reasoning"]:
            if self._llm_reasoning_available():
                try:
                    payload = self._llm_json(self._llm_reasoning_messages(context))
                    explanation_payload = self._sanitize_llm_explanation(payload, context)
                    explanation = explanation_payload.get("summary") or self._deterministic_explanation(context)
                    context.setdefault("reasoning", {})["llm_structured_explanation"] = explanation_payload
                    context["reasoning"]["structured_explanation"] = explanation
                    metadata["reasoning_mode"] = "llm_expert"
                    return ToolResult(
                        "build_structured_explanation",
                        "completed",
                        {"explanation": explanation, "reasoning_mode": "llm_expert", "citation_ids": explanation_payload.get("citation_ids", [])},
                    )
                except Exception as exc:
                    context.setdefault("warnings", []).append(f"llm_reasoning_failed: {exc}")
                    if metadata["require_llm_reasoning"] and not self._fallbacks_enabled(context):
                        return ToolResult("build_structured_explanation", "failed", error=f"llm_reasoning_failed:{exc}")
            elif metadata["require_llm_reasoning"] and not self._fallbacks_enabled(context):
                context.setdefault("warnings", []).append("llm_reasoning_required_but_unavailable")
                return ToolResult("build_structured_explanation", "failed", error="llm_reasoning_required_but_unavailable")
        explanation = self._deterministic_explanation(context)
        context.setdefault("reasoning", {})["structured_explanation"] = explanation
        metadata["reasoning_mode"] = "deterministic_template"
        return ToolResult("build_structured_explanation", "completed", {"explanation": explanation, "reasoning_mode": "deterministic_template"})

    def _safe_abstention_explanation(self, context: Dict[str, Any]) -> str:
        failure = context.get("metadata", {}).get("prediction_failure", {}) or {}
        tool = str(failure.get("tool") or "the prediction tool")
        if failure.get("failure_kind") == "no_direct_evidence_or_ambiguous_entity":
            if self._looks_chinese(context.get("request")):
                return (
                    "本次任务执行安全拒答：检索后仍缺少目标对象与终点的直接证据，或科学对象存在无法消除的歧义，"
                    "因此没有生成预测标签、分数或概率。系统没有用后续检索记录、LLM 文本或背景资料替代缺失证据；"
                    "在对象得到澄清并获得直接证据前，结果保持不确定。"
                )
            return (
                "This task ended in a safe abstention because no direct evidence for the target object and endpoint was available, "
                "or the scientific entity remained ambiguous; therefore no prediction label, score, or probability was produced. "
                "Later retrieval records, LLM text, and background material were not allowed to replace the missing evidence. "
                "The outcome remains uncertain until the entity is clarified and direct evidence is available."
            )
        if self._looks_chinese(context.get("request")):
            return (
                f"本次任务未完成：{tool} 或其依赖服务不可用，因此没有生成预测标签、分数或概率。"
                "系统未用 LLM 文本或背景证据替代预测结果；已检索到的证据仅作为失败上下文保留，不能视为模型预测或实验真值。"
                "在预测工具恢复并重新运行前，应将结果视为明确的安全拒答。"
            )
        return (
            f"The task was not completed because {tool} or its dependent service was unavailable, so no prediction label, score, or probability was produced. "
            "The system did not substitute LLM text or background evidence for a prediction; any retrieved evidence is retained only as failure context and is neither a model output nor experimental truth. "
            "This result is an explicit safe abstention until the prediction tool is restored and the task is rerun."
        )

    def _deterministic_explanation(self, context: Dict[str, Any]) -> str:
        if context.get("task_type") == GENERAL_TASK_TYPE:
            return self._general_drug_profile_explanation(context)
        prediction = self._prediction_summary(context)
        chinese = self._looks_chinese(context.get("request"))
        evidence_count = len({_evidence_key(item) for item in context.get("evidence", []) or [] if isinstance(item, dict)})
        if chinese:
            conclusion = self._front_conclusion(context, chinese=True)
            method = self._prediction_source_sentence(context, chinese=True)
            structure = self._structure_sentence(context, chinese=True)
            evidence = self._evidence_sentence(context, evidence_count, chinese=True)
            return " ".join(part for part in [conclusion, structure, evidence, method] if part)
        explanation = " ".join(
            part
            for part in [
                self._front_conclusion(context, chinese=False),
                self._structure_sentence(context, chinese=False),
                self._evidence_sentence(context, evidence_count, chinese=False),
                self._prediction_source_sentence(context, chinese=False),
            ]
            if part
        )
        return explanation.strip()

    def _front_conclusion(self, context: Dict[str, Any], *, chinese: bool) -> str:
        prediction = self._prediction_summary(context)
        dataset_name = prediction.get("dataset_name") or context.get("dataset_name") or "the requested task"
        meaning = self._prediction_meaning(prediction, chinese=chinese)
        mol = self._molecule_phrase(context)
        value = prediction.get("probability") if prediction.get("probability") is not None else prediction.get("score")
        strength = self._qualitative_prediction_strength(value, chinese=chinese)
        boundary = self._near_prediction_boundary(context, prediction)
        no_direct_evidence = (
            str((context.get("metadata", {}) or {}).get("retrieval_outcome") or "")
            == "no_direct_evidence"
        )
        dataset_key = str(dataset_name or "").lower()
        if is_multiclass_ddi_prediction(prediction):
            class_meaning = prediction_meaning(
                prediction,
                drug_names=self._molecule_names(context),
                chinese=chinese,
            )
            if chinese:
                evidence_boundary = "；当前缺少直接证据，类别定义不能视为临床或实验事实" if no_direct_evidence else ""
                return (
                    f"结论：模型将 {mol} 归为 {class_meaning}，预测类别的判断强度为{strength}；"
                    f"这是多分类相互作用类型结果，不是二元的有/无相互作用判断{evidence_boundary}。"
                )
            evidence_boundary = "; no direct evidence was available, so the class definition is not a clinical or experimental fact" if no_direct_evidence else ""
            return (
                f"Conclusion: the model assigns {mol} to {class_meaning}, with {strength} top-class support. "
                f"This is a multiclass interaction-type result, not a binary interaction/no-interaction outcome{evidence_boundary}."
            )
        if chinese:
            if dataset_key in {"bbbp", "bbb_martins"}:
                if self._is_positive_prediction(prediction):
                    direction = f"{mol} 具有一定血脑屏障通透倾向"
                else:
                    direction = f"{mol} 不属于血脑屏障易透过分子"
                if no_direct_evidence:
                    return self._no_direct_evidence_conclusion(direction, chinese=True)
                if boundary:
                    return f"结论：{direction}，但该判断接近分类边界，置信度有限。"
                return f"结论：{direction}，整体判断强度为{strength}。"
            endpoint_phrase = self._endpoint_conclusion_phrase(context, prediction, chinese=True)
            if endpoint_phrase:
                if no_direct_evidence:
                    return self._no_direct_evidence_conclusion(endpoint_phrase, chinese=True)
                if boundary:
                    return f"结论：{endpoint_phrase}，但该判断接近分类边界，需谨慎解读。"
                return f"结论：{endpoint_phrase}，整体判断强度为{strength}。"
            target = f"在 {dataset_name} 任务中表现为“{meaning}”" if meaning else f"在 {dataset_name} 任务中给出当前方向性判断"
            if no_direct_evidence:
                return self._no_direct_evidence_conclusion(f"{mol} {target}", chinese=True)
            if boundary:
                return f"结论：{mol} {target}，但该判断接近分类边界，需谨慎解读。"
            return f"结论：{mol} {target}，整体判断强度为{strength}。"
        if dataset_key in {"bbbp", "bbb_martins"}:
            direction = "is predicted to have some BBB permeability tendency" if self._is_positive_prediction(prediction) else "is predicted to have limited or borderline BBB permeability"
            if no_direct_evidence:
                return self._no_direct_evidence_conclusion(f"{mol} {direction}", chinese=False)
            if boundary:
                return f"Conclusion: {mol} {direction}; because the result is close to the decision boundary, it should be interpreted cautiously."
            return f"Conclusion: {mol} {direction}, with {strength} support."
        endpoint_phrase = self._endpoint_conclusion_phrase(context, prediction, chinese=False)
        if endpoint_phrase:
            if no_direct_evidence:
                return self._no_direct_evidence_conclusion(endpoint_phrase, chinese=False)
            if boundary:
                return f"Conclusion: {endpoint_phrase}; because the result is close to the decision boundary, it should be interpreted cautiously."
            return f"Conclusion: {endpoint_phrase}, with {strength} support."
        target = f"as {meaning}" if meaning else f"for {dataset_name}"
        if no_direct_evidence:
            return self._no_direct_evidence_conclusion(f"{mol} is assessed {target}", chinese=False)
        if boundary:
            return f"Conclusion: {mol} is assessed {target}, but the result is close to the decision boundary and should be interpreted cautiously."
        return f"Conclusion: {mol} is assessed {target}, with {strength} support."

    def _no_direct_evidence_conclusion(self, direction: str, *, chinese: bool) -> str:
        """Keep model confidence distinct from evidentiary support."""

        if chinese:
            return f"结论：{direction}；该结果仅为模型预测，缺少直接证据，需谨慎解读。"
        return (
            f"Conclusion: {direction}, as a model prediction; no direct evidence was available, "
            "so it should be interpreted cautiously."
        )

    def _endpoint_conclusion_phrase(self, context: Dict[str, Any], prediction: Dict[str, Any], *, chinese: bool) -> str:
        dataset = str(prediction.get("dataset_name") or context.get("dataset_name") or "").lower()
        task_type = str(context.get("task_type") or prediction.get("task_type") or "").lower()
        mol = self._molecule_phrase(context)
        if task_type == "ddi_prediction" or dataset == "ddi":
            if is_multiclass_ddi_prediction(prediction):
                meaning = prediction_meaning(
                    prediction,
                    drug_names=self._molecule_names(context),
                    chinese=chinese,
                )
                if chinese:
                    return f"{mol} 的模型结果为 {meaning}；这不是二元的有/无相互作用判断"
                return f"the model assigns {mol} to {meaning}; this is not a binary interaction/no-interaction result"
            positive = self._is_positive_prediction(prediction)
            if chinese:
                return f"{mol} 存在潜在药物相互作用风险信号" if positive else f"{mol} 未显示明确药物相互作用风险信号"
            return f"{mol} carries a potential drug-drug interaction signal" if positive else f"{mol} does not show a clear drug-drug interaction signal"
        positive = self._is_positive_prediction(prediction)
        if dataset in {"dili", "dili_huang"}:
            if chinese:
                return f"{mol} 出现药物性肝损伤风险信号" if positive else f"{mol} 未显示明显药物性肝损伤风险信号"
            return f"{mol} shows a drug-induced liver injury risk signal" if positive else f"{mol} does not show an obvious drug-induced liver injury risk signal"
        if dataset in {"herg", "herg_central", "herg_karim"}:
            if chinese:
                return f"{mol} 出现 hERG 相关心脏安全性风险信号" if positive else f"{mol} 未显示明显 hERG 相关心脏安全性风险信号"
            return f"{mol} shows an hERG-related cardiac safety signal" if positive else f"{mol} does not show an obvious hERG-related cardiac safety signal"
        return ""

    def _prediction_source_sentence(self, context: Dict[str, Any], *, chinese: bool) -> str:
        prediction = self._prediction_summary(context)
        model_type = str(prediction.get("model_type") or "")
        dataset_name = prediction.get("dataset_name") or context.get("dataset_name") or "the requested task"
        if model_type == ZERO_SHOT_MODEL_TYPE:
            if chinese:
                return "方法说明：本次使用规则/语义融合预测路径，结论依据当前结构特征与证据整合形成；未调用任务专用监督模型。"
            return "Method note: this run used a rule/semantic fusion prediction path and integrates the available structural features and evidence; no task-specific supervised model was loaded."
        model_label = self._display_model_type(model_type)
        if is_multiclass_ddi_prediction(prediction):
            if chinese:
                return (
                    f"方法说明：该判断来自 {dataset_name} 多分类相互作用类型监督模型（{model_label}）；"
                    "所示置信度是预测类别的模型置信度，不是发生任意药物相互作用的概率。"
                )
            return (
                f"Method note: this assessment used a supervised multiclass interaction-type {dataset_name} model ({model_label}); "
                "the displayed confidence is confidence in the predicted class, not the probability of any drug-drug interaction."
            )
        if chinese:
            if model_label:
                return f"方法说明：该判断来自 {dataset_name} 任务专用监督模型（{model_label}），正文解释结合了结构特征与检索证据。"
            return f"方法说明：该判断来自 {dataset_name} 任务专用监督模型，正文解释结合了结构特征与检索证据。"
        if model_label:
            return f"Method note: this assessment used a task-specific supervised {dataset_name} model ({model_label}); the narrative summarizes the model output with structural features and retrieved evidence."
        return f"Method note: this assessment used a task-specific supervised {dataset_name} model; the narrative summarizes the model output with structural features and retrieved evidence."

    def _display_model_type(self, model_type: str) -> str:
        labels = {
            "random_forest": "Random Forest",
            "sgd_classifier": "SGD classifier",
            "zero_shot_rule_semantic_fusion": "zero-shot rule/semantic fusion",
        }
        return labels.get(str(model_type or ""), str(model_type or "").replace("_", " ").strip())

    def _near_prediction_boundary(self, context: Dict[str, Any], prediction: Dict[str, Any]) -> bool:
        if is_multiclass_ddi_prediction(prediction):
            return False
        confidence = context.get("confidence", {}) if isinstance(context.get("confidence"), dict) else {}
        if confidence.get("low_confidence") is True:
            return True
        value = prediction.get("probability") if prediction.get("probability") is not None else prediction.get("score")
        try:
            return abs(float(value) - 0.5) < 0.12
        except (TypeError, ValueError):
            return False

    def _is_positive_prediction(self, prediction: Dict[str, Any]) -> bool:
        direction = binary_prediction_direction(prediction)
        return bool(direction) if direction is not None else False

    def _general_drug_profile_explanation(self, context: Dict[str, Any]) -> str:
        chinese = self._looks_chinese(context.get("request"))
        mol = self._molecule_phrase(context)
        invalid = self._invalid_molecule_names(context)
        if invalid:
            names = ", ".join(invalid)
            if chinese:
                return (
                    f"本次没有生成可靠药物概览，因为系统未能把输入中的药物名解析为有效分子结构：{names}。"
                    "没有有效 SMILES 时，AgentM 不能构建分子特征，也不会让 LLM 直接编造药物性质或预测结论。"
                    "请检查药物拼写，或改用英文通用名/标准 SMILES；如果需要解析不在本地词典中的药物，请在 Custom 模式开启 network lookup。"
                )
            return (
                f"No reliable drug profile was generated because the molecule name could not be resolved to a valid structure: {names}. "
                "Without a valid SMILES, AgentM cannot build molecular features, and the guard prevents the LLM from inventing drug properties or predictions. "
                "Please check the spelling, provide a standard English drug name or SMILES, or enable network lookup in Custom mode."
            )
        evidence_count = len({_evidence_key(item) for item in context.get("evidence", []) or [] if isinstance(item, dict)})
        database = self._database_profile_sentence(context, chinese=chinese)
        structure = self._general_structure_sentence(context, chinese=chinese)
        evidence = self._general_evidence_sentence(context, evidence_count, chinese=chinese)
        if chinese:
            return " ".join(
                part
                for part in [
                    database,
                    structure,
                    evidence,
                    "如果需要进一步做定量风险评估，可以继续指定具体方向，例如肝损伤风险、hERG 心脏安全性、血脑屏障通透性或药物相互作用。",
                ]
                if part
            )
        return " ".join(
            part
            for part in [
                database,
                structure,
                evidence,
                "For a quantitative risk assessment, specify a direction such as DILI, hERG cardiac safety, blood-brain-barrier permeability, or drug-drug interaction.",
            ]
            if part
        )

    def _non_science_explanation(self, context: Dict[str, Any]) -> str:
        gate = context.get("metadata", {}).get("intent_gate", {}) or {}
        chinese = self._looks_chinese(context.get("request"))
        task_type = context.get("task_type")
        suggestion = str(gate.get("suggestion") or "").strip()
        if task_type == MISSING_ENTITY_TASK_TYPE:
            if chinese:
                return (
                    "本次没有进入 ADMET、DDI 或化学反应预测流程，因为请求像是药物/分子相关问题，但没有给出可解析的具体药物名、化合物名或 SMILES。"
                    "请补充明确实体，例如“阿司匹林 BBBP”“阿司匹林和华法林相互作用”或一个标准 SMILES。"
                )
            return (
                "The workflow did not enter ADMET, DDI, or reaction prediction because the request appears scientific but does not provide a resolvable drug name, compound name, or SMILES. "
                "Please provide a concrete entity, for example `aspirin BBBP`, an interaction pair, or a standard SMILES."
            )
        if chinese:
            base = "本次没有进入 ADMET、DDI 或化学反应预测流程，因为输入没有明确的药物、分子、SMILES、ADMET/DDI 端点或反应任务。"
            return base + "请改写为具体科学任务，例如“阿司匹林 BBBP”“阿司匹林和华法林相互作用”或“判断这个反应类型”。"
        base = "The workflow did not enter ADMET, DDI, or reaction prediction because the input does not specify a drug, molecule, SMILES, ADMET/DDI endpoint, or reaction task."
        return base + " " + (suggestion or "Please rewrite it as a concrete scientific task, such as `aspirin BBBP`, a drug-interaction pair, or a reaction classification request.")

    def _database_profile_sentence(self, context: Dict[str, Any], *, chinese: bool) -> str:
        profiles = context.get("artifacts", {}).get("database_profile", []) or []
        if not profiles:
            if chinese:
                return "目前只能根据已解析的分子结构做基础介绍，尚没有足够的外部资料来补充药品标签、机制或用途背景。"
            return "Only the resolved molecular structure is available at the moment, so the profile cannot yet add drug-label, mechanism, or use-context details."
        profile = self._first_profile(profiles, {"pubchem", "resolver_metadata"}) or profiles[0]
        dailymed = self._first_profile(profiles, {"dailymed"}, require_success=True)
        chembl = self._first_profile(profiles, {"chembl"}, require_success=True)
        name = profile.get("title") or profile.get("name") or profile.get("original_input")
        if not name and chembl:
            name = chembl.get("pref_name")
        name = name or self._molecule_phrase(context)
        cid = profile.get("pubchem_cid")
        formula = profile.get("molecular_formula")
        weight = profile.get("molecular_weight")
        synonyms = [str(item) for item in profile.get("synonyms", []) or [] if item][:4]
        source = profile.get("source") or "database"
        status_ok = profile.get("retrieval_status") == "success"
        label_entries = [item for item in dailymed.get("label_entries", []) or [] if isinstance(item, dict)] if dailymed else []
        mechanisms = [item for item in chembl.get("mechanisms", []) or [] if isinstance(item, dict)] if chembl else []
        indications = [item for item in chembl.get("indications", []) or [] if isinstance(item, dict)] if chembl else []
        mechanism_texts = self._mechanism_texts(mechanisms, limit=2)
        indication_texts = self._indication_texts(indications, limit=3)
        label_phrase_zh = self._label_context_phrase(label_entries, name, chinese=True)
        label_phrase_en = self._label_context_phrase(label_entries, name, chinese=False)
        if chinese:
            identity_parts = [str(name)]
            detail_parts = []
            if source == "pubchem" and cid:
                detail_parts.append(f"PubChem CID {cid}")
            elif cid:
                detail_parts.append(f"关联 CID {cid}")
            if formula:
                detail_parts.append(f"分子式 {formula}")
            if weight:
                detail_parts.append(f"分子量约 {self._format_score(weight)}")
            if detail_parts:
                identity_parts.append("，".join(detail_parts))
            if synonyms:
                identity_parts.append("也常写作 " + "、".join(synonyms))
            intro = "（".join([identity_parts[0], "；".join(identity_parts[1:]) + "）"]) if len(identity_parts) > 1 else identity_parts[0]
            if not status_ok:
                intro = f"{name} 的分子结构已解析，但外部资料有限"
            sentences = [f"{intro}。"]
            context_parts = []
            if label_phrase_zh:
                context_parts.append(label_phrase_zh)
            if mechanism_texts:
                context_parts.append("生物活性资料支持其与 " + "、".join(mechanism_texts) + " 相关")
            if indication_texts:
                context_parts.append("临床或适应症背景常涉及 " + "、".join(indication_texts))
            if context_parts:
                sentences.append("；".join(context_parts) + "。")
            return "".join(sentences)
        identity_parts = [str(name)]
        detail_parts = []
        if source == "pubchem" and cid:
            detail_parts.append(f"PubChem CID {cid}")
        elif cid:
            detail_parts.append(f"linked CID {cid}")
        if formula:
            detail_parts.append(f"formula {formula}")
        if weight:
            detail_parts.append(f"molecular weight about {self._format_score(weight)}")
        if detail_parts:
            identity_parts.append(", ".join(detail_parts))
        if synonyms:
            identity_parts.append("also listed as " + ", ".join(synonyms))
        intro = f"{identity_parts[0]} ({'; '.join(identity_parts[1:])})" if len(identity_parts) > 1 else identity_parts[0]
        if not status_ok:
            intro = f"The molecular structure for {name} was resolved, but external profile details are limited"
        sentences = [f"{intro}."]
        context_parts = []
        if label_phrase_en:
            context_parts.append(label_phrase_en)
        if mechanism_texts:
            context_parts.append("bioactivity records connect it with " + ", ".join(mechanism_texts))
        if indication_texts:
            context_parts.append("clinical or indication context includes " + ", ".join(indication_texts))
        if context_parts:
            sentences.append("; ".join(context_parts) + ".")
        return " ".join(sentences)

    def _mechanism_texts(self, mechanisms: List[Dict[str, Any]], *, limit: int) -> List[str]:
        texts = []
        seen = set()
        for item in mechanisms:
            label = item.get("mechanism_of_action") or item.get("target_name") or item.get("action_type")
            if not label:
                continue
            text = str(label)
            key = text.lower()
            if key in seen:
                continue
            seen.add(key)
            texts.append(text)
            if len(texts) >= limit:
                break
        return texts

    def _indication_texts(self, indications: List[Dict[str, Any]], *, limit: int) -> List[str]:
        texts = []
        seen = set()
        for item in indications:
            label = item.get("mesh_heading") or item.get("efo_term")
            if not label:
                continue
            text = str(label)
            key = text.lower()
            if key in seen:
                continue
            seen.add(key)
            texts.append(text)
            if len(texts) >= limit:
                break
        return texts

    def _label_context_phrase(self, entries: List[Dict[str, Any]], name: Any, *, chinese: bool) -> str:
        if not entries:
            return ""
        titles = [str(item.get("title")) for item in entries if item.get("title")]
        if not titles:
            return "药品标签资料中有相关制剂信息" if chinese else "drug-label information includes related products"
        generic = str(name or "").lower()
        combination = any("," in title or " AND " in title.upper() or "/" in title for title in titles)
        tablet = any("TABLET" in title.upper() for title in titles)
        capsule = any("CAPSULE" in title.upper() for title in titles)
        injection = any("INJECTION" in title.upper() for title in titles)
        formulation_bits = []
        if tablet:
            formulation_bits.append("片剂" if chinese else "tablets")
        if capsule:
            formulation_bits.append("胶囊" if chinese else "capsules")
        if injection:
            formulation_bits.append("注射剂" if chinese else "injections")
        if chinese:
            if combination:
                product = "复方制剂"
            elif formulation_bits:
                product = "、".join(formulation_bits)
            else:
                product = "相关制剂"
            return f"药品标签资料显示，它可见于{product}中"
        if combination:
            product = "combination-product"
        elif formulation_bits:
            product = "/".join(formulation_bits)
        else:
            product = "related product"
        prefix = f"{name} " if generic and generic not in product.lower() else ""
        return f"drug-label information includes {prefix}{product} entries"

    def _first_profile(self, profiles: List[Dict[str, Any]], sources: set[str], *, require_success: bool = False) -> Dict[str, Any]:
        for profile in profiles:
            if not isinstance(profile, dict):
                continue
            if profile.get("source") not in sources:
                continue
            if require_success and profile.get("retrieval_status") != "success":
                continue
            return profile
        return {}

    def _invalid_molecule_names(self, context: Dict[str, Any]) -> List[str]:
        invalid = []
        for item in context.get("molecules", []) or []:
            if isinstance(item, dict):
                is_valid = item.get("is_valid", True)
                name = item.get("original_input") or item.get("name") or "unknown"
            else:
                is_valid = getattr(item, "is_valid", True)
                name = getattr(item, "original_input", None) or getattr(item, "name", None) or "unknown"
            if not is_valid:
                invalid.append(str(name))
        return invalid

    def _general_evidence_sentence(self, context: Dict[str, Any], evidence_count: int, *, chinese: bool) -> str:
        if chinese:
            if evidence_count == 0:
                return "这份概览目前主要依据分子结构解析和 RDKit 描述符计算，尚缺少可进入报告的外部资料。"
            return "这份概览整合了公共化合物资料、药品标签资料、生物活性记录和 RDKit 结构描述符；它用于药物背景介绍，不等同于专项 ADMET/DDI 定量模型评估。"
        if evidence_count == 0:
            return "This profile is based mainly on the resolved molecular structure and RDKit descriptor calculation; no external evidence entered the report."
        return "This profile integrates public compound facts, drug-label information, bioactivity records, and RDKit structural descriptors; it is background context rather than a task-specific ADMET/DDI quantitative model assessment."

    def _general_structure_sentence(self, context: Dict[str, Any], *, chinese: bool) -> str:
        descriptors = self._feature_descriptors(context)
        if not descriptors:
            return "结构概览：当前报告没有可用的 RDKit 描述符明细。" if chinese else "Structural profile: no detailed RDKit descriptor summary was available."
        fields = [
            ("MolWt", "分子量", "molecular weight"),
            ("TPSA", "TPSA", "TPSA"),
            ("MolLogP", "LogP", "LogP"),
            ("NumHDonors", "氢键供体", "H-bond donors"),
            ("NumHAcceptors", "氢键受体", "H-bond acceptors"),
            ("NumRotatableBonds", "可旋转键", "rotatable bonds"),
        ]
        parts = []
        for key, zh_name, en_name in fields:
            if key in descriptors:
                value = self._format_score(descriptors.get(key))
                if chinese:
                    if zh_name == "分子量":
                        parts.append(f"{zh_name}约 {value}")
                    elif zh_name in {"氢键供体", "氢键受体", "可旋转键"}:
                        parts.append(f"{zh_name} {value} 个")
                    else:
                        parts.append(f"{zh_name} 为 {value}")
                else:
                    parts.append(f"{en_name}={value}")
        notes_zh = []
        notes_en = []
        try:
            mol_wt = float(descriptors.get("MolWt"))
        except (TypeError, ValueError):
            mol_wt = None
        try:
            logp = float(descriptors.get("MolLogP"))
        except (TypeError, ValueError):
            logp = None
        try:
            tpsa = float(descriptors.get("TPSA"))
        except (TypeError, ValueError):
            tpsa = None
        if mol_wt is not None and mol_wt > 500:
            notes_zh.append("分子量偏大，可能影响口服吸收和组织分布")
            notes_en.append("high molecular weight may affect oral absorption and tissue distribution")
        if logp is not None and logp > 5:
            notes_zh.append("LogP 偏高，提示疏水性较强并可能带来溶解度或代谢方面的不确定性")
            notes_en.append("high LogP suggests strong hydrophobicity and possible solubility or metabolism uncertainty")
        if tpsa is not None and tpsa > 90:
            notes_zh.append("TPSA 偏高，通常会降低被动跨膜或中枢暴露倾向")
            notes_en.append("high TPSA often lowers passive membrane permeability or CNS exposure tendency")
        if chinese:
            tail = "；".join(notes_zh) if notes_zh else "这些描述符没有显示特别突出的简单结构警讯，但仍需结合具体适应症和剂量背景解释"
            return f"从结构上看，它的{self._join_chinese_phrases(parts)}。{tail}。"
        tail = "; ".join(notes_en) if notes_en else "these descriptors do not show a strong simple structural alert, but interpretation still depends on indication and dose context"
        return f"Structurally, it has {'; '.join(parts)}. {tail}."

    def _join_chinese_phrases(self, parts: List[str]) -> str:
        if not parts:
            return "关键 RDKit 描述符暂不可用"
        if len(parts) == 1:
            return parts[0]
        return "，".join(parts[:-1]) + "，" + parts[-1]

    def _feature_descriptors(self, context: Dict[str, Any]) -> Dict[str, Any]:
        feature = (context.get("features") or [None])[0]
        if feature is None:
            return {}
        if isinstance(feature, dict):
            descriptors = feature.get("rdkit_descriptors")
        else:
            descriptors = getattr(feature, "rdkit_descriptors", None)
        return descriptors if isinstance(descriptors, dict) else {}

    def _prediction_meaning(
        self,
        prediction: Dict[str, Any],
        *,
        drug_names: Optional[List[str]] = None,
        chinese: bool = False,
    ) -> str:
        details = prediction.get("details") if isinstance(prediction.get("details"), dict) else {}
        return prediction_meaning(
            prediction,
            drug_names=drug_names or details.get("open_task_molecules") or None,
            chinese=chinese,
        )

    def _molecule_phrase(self, context: Dict[str, Any]) -> str:
        names = self._molecule_names(context)
        if names:
            return " + ".join(str(name) for name in names)
        return "the molecule"

    def _structure_sentence(self, context: Dict[str, Any], *, chinese: bool) -> str:
        prediction = self._prediction_summary(context)
        if str(context.get("task_type") or prediction.get("task_type") or "").lower() == "ddi_prediction":
            if is_multiclass_ddi_prediction(prediction):
                if chinese:
                    return (
                        "相互作用依据：当前输出来自两种药物的结构表征与配对特征；模型给出的是 DrugBank 相互作用类型类别，"
                        "而不是二元的有/无相互作用结论。类别定义用于解释模型标签，不构成该机制的临床或实验证明。"
                    )
                return (
                    "Interaction basis: the output is derived from the two drugs' structural representations and pair features. "
                    "The model predicts a DrugBank interaction type rather than a binary interaction/no-interaction outcome; "
                    "the class definition explains the model label but does not establish the mechanism clinically or experimentally."
                )
            if chinese:
                return "相互作用依据：当前判断结合两种药物的结构表征、配对特征与检索资料形成；若证据没有指向明确药代或药效机制，结论应表述为相互作用风险信号，而不是确定机制。"
            return "Interaction basis: the current judgment integrates the two drugs' structural representations, pair features, and retrieved evidence; when evidence does not establish a specific pharmacokinetic or pharmacodynamic mechanism, the report frames the result as an interaction risk signal rather than a proven mechanism."
        descriptors = self._descriptor_snapshot(context, prediction)
        if not descriptors:
            return "结构依据：当前报告没有可用的 RDKit 描述符明细。" if chinese else "No detailed RDKit descriptor summary was available."
        fields = [
            ("MolWt", "分子量", "molecular weight"),
            ("TPSA", "TPSA", "TPSA"),
            ("MolLogP", "LogP", "LogP"),
            ("NumHDonors", "氢键供体", "H-bond donors"),
            ("NumHAcceptors", "氢键受体", "H-bond acceptors"),
        ]
        parts = []
        for key, zh_name, en_name in fields:
            if key in descriptors:
                parts.append(f"{zh_name}={self._format_score(descriptors.get(key))}" if chinese else f"{en_name}={self._format_score(descriptors.get(key))}")
        if chinese:
            tail = "这些结构特征构成当前判断的主要结构依据。"
            if str(prediction.get("dataset_name") or "").lower() in {"bbbp", "bbb_martins"}:
                tail = self._bbbp_descriptor_tail(context, descriptors, chinese=True)
            elif str(prediction.get("dataset_name") or "").lower() in {"dili", "dili_huang"}:
                tail = "DILI 风险通常受暴露、代谢、反应性和剂量背景共同影响，因此这些结构描述符应作为风险评估线索，而不是单一机制证明。"
            elif str(prediction.get("dataset_name") or "").lower() in {"herg", "herg_central", "herg_karim"}:
                tail = "hERG 风险与疏水性、芳香性、极性和离子化特征等因素相关；当前结构描述符提供风险线索，但不等同于通道阻断机制的直接实验证据。"
            return f"结构依据：{'; '.join(parts)}。{tail}"
        tail = "These descriptors are the main structural basis for the current judgment."
        if str(prediction.get("dataset_name") or "").lower() in {"bbbp", "bbb_martins"}:
            tail = self._bbbp_descriptor_tail(context, descriptors, chinese=False)
        elif str(prediction.get("dataset_name") or "").lower() in {"dili", "dili_huang"}:
            tail = "DILI risk can depend on exposure, metabolism, reactivity, and dose context, so these descriptors should be treated as risk-assessment clues rather than proof of a single mechanism."
        elif str(prediction.get("dataset_name") or "").lower() in {"herg", "herg_central", "herg_karim"}:
            tail = "hERG risk can depend on hydrophobicity, aromaticity, polarity, and ionization-related features; the descriptors provide risk clues, not direct experimental evidence of channel blockade."
        return f"Structural basis: {'; '.join(parts)}. {tail}"

    def _bbbp_descriptor_tail(self, context: Dict[str, Any], descriptors: Dict[str, Any], *, chinese: bool) -> str:
        mol = self._molecule_phrase(context)

        def as_float(key: str) -> Optional[float]:
            value = descriptors.get(key)
            try:
                return float(value)
            except (TypeError, ValueError):
                return None

        mol_wt = as_float("MolWt")
        tpsa = as_float("TPSA")
        hbd = as_float("NumHDonors")
        caution_zh = []
        caution_en = []
        if mol_wt is not None and mol_wt > 500:
            caution_zh.append("分子量偏大")
            caution_en.append("high molecular weight")
        elif mol_wt is not None and mol_wt > 450:
            caution_zh.append("分子量接近常用 BBB 经验阈值")
            caution_en.append("molecular weight near common BBB heuristic limits")
        if tpsa is not None and tpsa > 90:
            caution_zh.append("TPSA 偏高")
            caution_en.append("high TPSA")
        elif tpsa is not None and tpsa > 70:
            caution_zh.append("TPSA 中等偏高")
            caution_en.append("moderately high TPSA")
        if hbd is not None and hbd > 2:
            caution_zh.append("氢键供体较多")
            caution_en.append("multiple H-bond donors")
        if caution_zh:
            if chinese:
                return f"对 BBBP 来说，{'; '.join(caution_zh)}通常不利于被动透过 BBB，因此 {mol} 的判断需要结合转运体、离子化状态和实验背景谨慎解释。"
            return f"For BBBP, {'; '.join(caution_en)} usually argues against passive BBB penetration, so the assessment for {mol} should be interpreted with transporter, ionization, and experimental context in mind."
        if self._has_carboxylic_acid_signal(context):
            if chinese:
                return f"对 BBBP 来说，较小分子量和中等脂溶性本身并不排斥中枢暴露；但羧酸等可电离极性基团、TPSA 和氢键特征可能削弱被动跨越 BBB 的能力，因此 {mol} 的边界性判断应作为筛查提示。"
            return f"For BBBP, low molecular weight and moderate lipophilicity do not by themselves rule out CNS exposure; however, ionizable polar groups such as carboxylic acid, TPSA, and H-bonding features can weaken passive BBB crossing, so the borderline assessment for {mol} should be treated as screening evidence."
        if chinese:
            return f"对 BBBP 来说，较低到中等的极性表面积、适中的分子量和有限氢键供体通常更有利于被动透过 BBB；仍需结合 {mol} 的离子化状态和实验背景解释。"
        return f"For BBBP, lower-to-moderate polar surface area, moderate molecular weight, and few H-bond donors generally favor passive BBB penetration; ionization and experimental context still matter for {mol}."

    def _descriptor_snapshot(self, context: Dict[str, Any], prediction: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        prediction = prediction or self._prediction_summary(context)
        details = prediction.get("details") if isinstance(prediction.get("details"), dict) else {}
        descriptors = details.get("rdkit_descriptors") if isinstance(details.get("rdkit_descriptors"), dict) else {}
        if descriptors:
            return descriptors
        schema = context.get("artifacts", {}).get("feature_schema") or {}
        metadata = schema.get("metadata") if isinstance(schema, dict) else {}
        snapshot = metadata.get("descriptor_snapshot") if isinstance(metadata, dict) else {}
        if isinstance(snapshot, dict) and snapshot:
            return snapshot
        for feature in context.get("features", []) or []:
            for attr in ("descriptor_snapshot", "rdkit_descriptors", "descriptors"):
                value = getattr(feature, attr, None)
                if isinstance(value, dict) and value:
                    return value
            value = getattr(feature, "metadata", None)
            if isinstance(value, dict):
                snapshot = value.get("descriptor_snapshot") or value.get("rdkit_descriptors")
                if isinstance(snapshot, dict) and snapshot:
                    return snapshot
        return {}

    def _has_carboxylic_acid_signal(self, context: Dict[str, Any]) -> bool:
        for molecule in context.get("molecules", []) or []:
            if isinstance(molecule, dict):
                smiles = str(molecule.get("canonical_smiles") or molecule.get("smiles") or "").lower()
            else:
                smiles = str(getattr(molecule, "canonical_smiles", None) or getattr(molecule, "smiles", "") or "").lower()
            if "c(=o)o" in smiles or "c(=o)[o-]" in smiles:
                return True
        for rule in context.get("rules", []) or []:
            text = ""
            if isinstance(rule, dict):
                text = " ".join(str(rule.get(key) or "") for key in ("feature_name", "name", "text"))
            else:
                text = " ".join(str(getattr(rule, key, "") or "") for key in ("feature_name", "name", "text"))
            if "carboxylic_acid" in text or "carboxylic acid" in text.lower():
                return True
        return False

    def _evidence_sentence(self, context: Dict[str, Any], evidence_count: int, *, chinese: bool) -> str:
        evidence = [item for item in context.get("evidence", []) or [] if isinstance(item, dict)]
        metadata = context.get("metadata", {}) or {}
        directions = {"support": 0, "contradiction": 0, "neutral": 0}
        sources = set()
        for item in evidence:
            direction = str(item.get("evidence_direction") or "neutral")
            if direction not in directions:
                direction = "neutral"
            directions[direction] += 1
            source = item.get("source") or item.get("provider")
            if source:
                sources.add(str(source))
        has_conflict = bool(directions["support"] and directions["contradiction"])
        if has_conflict:
            source_text = ", ".join(sorted(sources)) if sources else "retrieved context"
            source_text_en = ", ".join(self._display_evidence_source(source) for source in sorted(sources)) if sources else "retrieved context"
            if chinese:
                direct_note = "；同时缺少可将这些记录提升为实验事实的直接证据" if metadata.get("retrieval_outcome") == "no_direct_evidence" else ""
                return (
                    f"证据概况：来自 {source_text} 的记录存在明确冲突，其中支持 {directions['support']} 条、"
                    f"相反 {directions['contradiction']} 条、中性 {directions['neutral']} 条{direct_note}。"
                    "相反方向被完整保留，当前结论具有不确定性，需谨慎解读；引用可在 Evidence/Trace 中追溯。"
                )
            direct_note = "; neither direction is promoted to experimental fact because direct evidence is unavailable" if metadata.get("retrieval_outcome") == "no_direct_evidence" else ""
            return (
                f"Evidence conflict: records from {source_text_en} point in opposite directions: "
                f"{directions['support']} support, {directions['contradiction']} contradict, and {directions['neutral']} are neutral"
                f"{direct_note}. Both directions are preserved, so the conclusion remains uncertain and should be interpreted cautiously. "
                "Citations are traceable in the Evidence/Trace views."
            )
        if metadata.get("retrieval_outcome") == "no_direct_evidence":
            if metadata.get("qc_grounding_only"):
                if chinese:
                    return "证据概况：真实 dense-hybrid 检索已完成并命中实体–端点 QC 对齐记录，但该记录不含实验终点标签，仅属于 QC 对齐信息，缺少直接证据支持当前预测。"
                return "Evidence: the real dense-hybrid query completed and found an entity–endpoint QC grounding record, but that record contains no experimental endpoint label and is QC grounding rather than direct evidence for the prediction."
            if chinese:
                return "证据概况：真实 dense-hybrid 检索已完成，但未找到该对象与端点的直接证据；当前内容只能作为模型预测解读，不能表述为实验事实。"
            return "Evidence: the real dense-hybrid query completed but found no direct evidence for this object and endpoint; the result is a model prediction only and must not be presented as experimental fact."
        source_text = ", ".join(sorted(sources)) if sources else "retrieved context"
        source_text_en = ", ".join(self._display_evidence_source(source) for source in sorted(sources)) if sources else "retrieved context"
        if chinese:
            if evidence_count == 0:
                return "证据概况：当前没有可纳入最终报告的外部证据，结论主要依据结构特征和预测路径形成。"
            if directions["support"] or directions["contradiction"]:
                balance = f"其中支持 {directions['support']} 条、相反 {directions['contradiction']} 条、中性 {directions['neutral']} 条。"
            else:
                balance = "这些资料主要提供任务相关背景，尚未形成直接支持或反驳该判断的证据链。"
            return f"证据概况：本次参考了来自 {source_text} 的相关资料；{balance}引用可在 Evidence/Trace 中追溯。"
        if evidence_count == 0:
            return "Evidence: no external evidence entered the final report; the conclusion is based mainly on the structural features and prediction path."
        if directions["support"] or directions["contradiction"]:
            balance = f"{directions['support']} support, {directions['contradiction']} contradict, and {directions['neutral']} are neutral."
        else:
            balance = "the retrieved evidence mainly provides task background and was not scored as direct support or contradiction."
        return f"Evidence context: relevant material came from {source_text_en}; {balance} Citations are traceable in the Evidence/Trace views."

    def _display_evidence_source(self, source: str) -> str:
        labels = {
            "curated_admet_evidence": "curated ADMET evidence",
            "rag_literature": "retrieved literature",
            "local_rag": "local RAG evidence",
            "elasticsearch_hybrid": "hybrid evidence index",
            "elasticsearch_hybrid_bm25_dense": "hybrid evidence index",
        }
        return labels.get(str(source or ""), str(source or "").replace("_", " "))

    def _format_score(self, value: Any) -> str:
        if isinstance(value, (int, float)):
            return f"{float(value):.3g}"
        return str(value)

    def _qualitative_prediction_strength(self, value: Any, *, chinese: bool) -> str:
        try:
            score = float(value)
        except (TypeError, ValueError):
            return "需要谨慎解读" if chinese else "uncertain"
        distance = abs(score - 0.5)
        if distance >= 0.35:
            return "较明确" if chinese else "fairly clear"
        if distance >= 0.2:
            return "中等" if chinese else "moderate"
        return "偏谨慎" if chinese else "cautious"

    def _looks_chinese(self, text: Any) -> bool:
        return any("\u4e00" <= char <= "\u9fff" for char in str(text or ""))

    def summarize_evidence_support(self, args: Dict[str, Any], context: Dict[str, Any]) -> ToolResult:
        evidence = context.get("evidence", []) or []
        unique_evidence = [item for item in evidence if isinstance(item, dict)]
        support = {
            "n_evidence_items": len({_evidence_key(item) for item in unique_evidence}),
            "sources": sorted({str(item.get("source", "unknown")) for item in unique_evidence}),
            "has_retrieved_context": bool(context.get("artifacts", {}).get("retrieved_context")),
        }
        context.setdefault("reasoning", {})["evidence_support"] = support
        return ToolResult("summarize_evidence_support", "completed", support)

    def verify_prediction_grounding(self, args: Dict[str, Any], context: Dict[str, Any]) -> ToolResult:
        report = verify_prediction_grounding(context)
        context.setdefault("verifier_report", {})["prediction"] = report
        context.setdefault("warnings", []).extend(report.get("warnings", []))
        return ToolResult("verify_prediction_grounding", "completed", report)

    def verify_citation_grounding(self, args: Dict[str, Any], context: Dict[str, Any]) -> ToolResult:
        report = verify_citations(context)
        context["citation_map"] = report.get("citation_map", [])
        context.setdefault("verifier_report", {})["citations"] = report
        context.setdefault("warnings", []).extend(report.get("warnings", []))
        return ToolResult("verify_citation_grounding", "completed", report)

    def verify_report_schema(self, args: Dict[str, Any], context: Dict[str, Any]) -> ToolResult:
        report = verify_report_schema(context)
        semantics_report = verify_prediction_semantics(context)
        context.setdefault("verifier_report", {})["schema"] = report
        context["verifier_report"]["prediction_semantics"] = semantics_report
        summary = merge_verifier_reports(
            context["verifier_report"].get("prediction", {}),
            context["verifier_report"].get("citations", {}),
            semantics_report,
            report,
        )
        objectives = evaluate_task_objectives(context)
        context["objective_status"] = objectives
        context["verifier_report"]["objectives"] = objectives
        context["warnings"] = [
            warning
            for warning in context.get("warnings", []) or []
            if not str(warning).startswith("objective_missing:")
        ]
        if objectives.get("status") != "pass":
            objective_warnings = [f"objective_missing:{name}" for name in objectives.get("missing_objectives", [])]
            summary.setdefault("warnings", []).extend(objective_warnings)
            context.setdefault("warnings", []).extend(objective_warnings)
            summary["status"] = "fail" if objectives.get("status") == "fail" else "warning"
        summary["objectives"] = objectives
        context["verifier_report"]["summary"] = summary
        context.setdefault("warnings", []).extend(report.get("warnings", []))
        return ToolResult("verify_report_schema", "completed", {"summary": summary})

    def reflect_on_report(self, args: Dict[str, Any], context: Dict[str, Any]) -> ToolResult:
        verifier = context.get("verifier_report", {}) or {}
        summary = verifier.get("summary", {}) or {}
        citations = verifier.get("citations", {}) or {}
        confidence = context.get("confidence", {}) or {}
        previous = context.get("previous_run_context", {}) or {}
        report = context.get("final_report", {}) or {}
        warnings = list(context.get("warnings", []) or [])
        evidence_count = len({_evidence_key(item) for item in context.get("evidence", []) or [] if isinstance(item, dict)})
        min_evidence = self._min_evidence(context)
        unsupported_claims = int(citations.get("n_unsupported", 0) or 0)
        low_confidence = bool(confidence.get("low_confidence"))
        insufficient_evidence = evidence_count < min_evidence
        needs_refinement = bool(
            unsupported_claims
            or insufficient_evidence
            or low_confidence
            or summary.get("status") == "warning"
            or summary.get("status") == "fail"
            or (previous and previous.get("task_type") == context.get("task_type") and previous.get("request") != context.get("request"))
        )
        recommendation = "finalize"
        if unsupported_claims or insufficient_evidence:
            recommendation = "expand_evidence"
        elif low_confidence:
            recommendation = "keep_warning"
        if previous:
            recommendation = "compare_with_previous_run" if recommendation == "finalize" else recommendation
        reflection = {
            "round": int(context.get("metadata", {}).get("reflection_rounds", 0)) + 1,
            "input_report_summary": {
                "request": report.get("request") or context.get("request"),
                "task_type": report.get("task_type") or context.get("task_type"),
                "n_predictions": len(report.get("predictions", []) or context.get("predictions", []) or []),
                "n_evidence": evidence_count,
                "min_evidence": min_evidence,
                "verifier_status": summary.get("status"),
                "warnings": warnings[-8:],
            },
            "previous_run_summary": self._previous_run_summary(previous),
            "needs_refinement": needs_refinement,
            "requires_retrieval_refinement": bool(needs_refinement and (unsupported_claims or insufficient_evidence)),
            "requires_reasoning_refresh": bool(needs_refinement or previous),
            "recommendation": recommendation,
            "next_input": {
                "verifier_report": verifier,
                "confidence": confidence,
                "unsupported_claims": unsupported_claims,
                "evidence_count": evidence_count,
                "min_evidence": min_evidence,
                "low_confidence": low_confidence,
                "previous_run_summary": self._previous_run_summary(previous),
            },
        }
        llm_reflection = self._build_llm_reflection(context, reflection)
        if llm_reflection:
            reflection["llm_reflection"] = llm_reflection
            if llm_reflection.get("refinement_queries"):
                reflection["next_input"]["llm_refinement_queries"] = llm_reflection.get("refinement_queries")
            if llm_reflection.get("recommendation") and recommendation == "finalize":
                reflection["recommendation"] = llm_reflection["recommendation"]
        context["reflection"] = reflection
        context.setdefault("reasoning", {})["self_reflection"] = reflection
        context.setdefault("metadata", {})["reflection_rounds"] = reflection["round"]
        return ToolResult("reflect_on_report", "completed", reflection)

    def apply_reflection_feedback(self, args: Dict[str, Any], context: Dict[str, Any]) -> ToolResult:
        reflection = context.get("reflection", {}) or {}
        metadata = context.setdefault("metadata", {})
        metadata["reflection_feedback_applied"] = True
        metadata["reflection_feedback_applied_round"] = int(reflection.get("round", 0) or 0)
        context.setdefault("reasoning", {})["reflection_feedback"] = {
            "used_as_next_input": True,
            "recommendation": reflection.get("recommendation"),
            "next_input": reflection.get("next_input", {}),
        }
        remaining_budget = int(metadata.get("max_rounds", 16)) - len(context.get("tool_calls", []) or [])
        can_refine = (
            metadata.get("mode") == "deep"
            and bool(reflection.get("requires_retrieval_refinement"))
            and metadata.get("retrieval_outcome") != "no_direct_evidence"
            and remaining_budget >= 5
        )
        # These flags belong to this feedback round, not the whole run.
        # A completed retrieval in round 1 must not suppress round 2.
        metadata["reflection_retrieval_requested"] = bool(can_refine)
        metadata["reflection_retrieval_done"] = False
        if can_refine:
            metadata["reflection_retrieval_requested"] = True
            metadata["reasoning_stale"] = True
            metadata["report_stale"] = True
            metadata["citation_verification_stale"] = True
            metadata["schema_verification_stale"] = True
            query = self._reflection_query(context)
            context.setdefault("reasoning", {})["expanded_evidence_query"] = query
            status = "refinement_requested"
        else:
            status = "refinement_deferred" if reflection.get("requires_retrieval_refinement") else "reflection_applied"
            if reflection.get("requires_reasoning_refresh"):
                metadata["report_stale"] = True
                metadata["schema_verification_stale"] = True
        reflection["application_status"] = status
        return ToolResult(
            "apply_reflection_feedback",
            "completed",
            {"application_status": status, "remaining_budget": remaining_budget, "reflection_retrieval_requested": bool(metadata.get("reflection_retrieval_requested"))},
        )

    def build_runtime_report(self, args: Dict[str, Any], context: Dict[str, Any]) -> ToolResult:
        context["final_report"] = report_payload(context)
        context["compatible_report"] = context["final_report"]
        return ToolResult("build_runtime_report", "completed", {"has_report": True})

    def build_compatible_report(self, args: Dict[str, Any], context: Dict[str, Any]) -> ToolResult:
        report = report_payload(context)
        report["trace"] = context.get("multi_agent_trace", [])
        report["reasoning"] = context.get("reasoning", {})
        context["compatible_report"] = report
        return ToolResult("build_compatible_report", "completed", {"task_type": report.get("task_type")})

    def write_artifacts(self, args: Dict[str, Any], context: Dict[str, Any]) -> ToolResult:
        if context.get("metadata", {}).get("safe_abstention_required"):
            context.setdefault("metadata", {})["safe_abstention_completed"] = not bool(
                context.get("reaction_predictions") or context.get("predictions")
            )
        self._refresh_objective_status(context)
        if not context.get("final_report"):
            context["final_report"] = report_payload(context)
        if not context.get("compatible_report"):
            context["compatible_report"] = context["final_report"]
        paths = write_artifacts(context, self.output_dir)
        context.setdefault("artifacts", {}).update(paths)
        context["final_report"] = report_payload(context)
        self._refresh_objective_status(context)
        paths = write_artifacts(context, self.output_dir)
        context.setdefault("artifacts", {}).update(paths)
        return ToolResult("write_artifacts", "completed", {"artifact_count": len(paths), "paths": paths})

    def _llm_reasoning_available(self) -> bool:
        return bool(self.reasoning_llm_client is not None and getattr(self.reasoning_llm_client, "available", False))

    def _llm_json(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        response = self.reasoning_llm_client.chat_json(messages, temperature=0.1)
        payload = json.loads(response)
        if not isinstance(payload, dict):
            raise ValueError("LLM reasoning response must be a JSON object.")
        return payload

    def _llm_reasoning_messages(self, context: Dict[str, Any]) -> List[Dict[str, str]]:
        payload = {
            "request": context.get("request"),
            "task_type": context.get("task_type"),
            "dataset_name": context.get("dataset_name"),
            "prediction": self._prediction_summary(context),
            "confidence": context.get("confidence", {}),
            "feature_grounding": self._feature_grounding_summary(context),
            "evidence": self._evidence_summary(context),
            "valid_citation_ids": sorted(self._valid_citation_ids(context)),
            "constraints": [
                "Do not create or modify prediction label, score, probability, model_type, or citations.",
                "Do not mention raw numeric score/probability or label ids in the user-facing narrative; translate them into qualitative language.",
                "Write the user-facing narrative as a pharmacology or medicinal-chemistry answer, not as an execution log.",
                "Do not mention AgentM, ReasoningAgent, GaugeAgent, RetrieveAgent, VerifierAgent, LLM, tool-grounded, agent trace, exact decision pathway, or feature weighting in the user-facing summary.",
                "Do not start the conclusion with the model type. Start with the biomedical interpretation; model provenance is handled separately by the runtime.",
                "If feature_grounding.has_features is true, do not say molecular descriptors, fingerprints, feature vectors, or model inputs were missing.",
                "Do not claim the model captured image semantics, sequence semantics, spatial conformation, or a specific hidden decision mechanism unless the provided evidence explicitly supports that claim.",
                "Use only citation_id values listed in valid_citation_ids.",
                "Put uncertainty or unsupported statements in unsupported_claims instead of summary.",
            ],
        }
        return [
            {
                "role": "system",
                "content": (
                    "You are AgentM ReasoningAgent, an ADMET/DDI evidence expert. "
                    "Use the same language as the user request. "
                    "Your summary must read like a concise biomedical answer for a human reader, not like a system report. "
                    "For the user-facing narrative, avoid raw internal fields such as score=0.84, probability=0.84, or label=1; "
                    "express them as qualitative confidence and biological interpretation. "
                    "Do not mention AgentM, internal agents, tool names, the LLM, exact decision pathways, or feature weighting in the user-facing text. "
                    "Do not open the conclusion with phrases such as 'based on Random Forest'; explain the biomedical result first. "
                    "If the provided feature_grounding says features were built, you must not claim that descriptors, fingerprints, "
                    "feature vectors, or model inputs were absent. "
                    "Avoid unsupported feature-attribution claims such as the model capturing image semantics, sequence semantics, spatial conformation, "
                    "or a specific hidden Random Forest mechanism unless those attributions are explicitly provided. "
                    "Return JSON only with keys summary, prediction_interpretation, evidence_support, "
                    "limitations, unsupported_claims, citation_ids. Keep all claims grounded in the provided evidence."
                ),
            },
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]

    def _sanitize_llm_explanation(self, payload: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        valid = self._valid_citation_ids(context)
        evidence_support = []
        citation_ids = []
        for item in payload.get("evidence_support", []) or []:
            if not isinstance(item, dict):
                continue
            citation_id = str(item.get("citation_id") or "")
            if citation_id and citation_id not in valid:
                context.setdefault("warnings", []).append(f"llm_reasoning_dropped_ungrounded_citation:{citation_id}")
                continue
            if citation_id:
                citation_ids.append(citation_id)
            evidence_support.append(
                {
                    "citation_id": citation_id,
                    "claim": self._humanize_internal_prediction_fields(str(item.get("claim") or ""), context)[:600],
                    "support": str(item.get("support") or item.get("supports") or "neutral")[:80],
                }
            )
        for citation_id in payload.get("citation_ids", []) or []:
            citation_text = str(citation_id)
            if citation_text in valid and citation_text not in citation_ids:
                citation_ids.append(citation_text)
            elif citation_text not in valid:
                context.setdefault("warnings", []).append(f"llm_reasoning_dropped_ungrounded_citation:{citation_text}")
        unsupported = payload.get("unsupported_claims", []) or []
        if not isinstance(unsupported, list):
            unsupported = [str(unsupported)]
        limitations = payload.get("limitations", []) or []
        if not isinstance(limitations, list):
            limitations = [str(limitations)]
        raw_summary = self._sanitize_feature_absence_claims(
            self._sanitize_ungrounded_feature_attribution(
                self._humanize_internal_prediction_fields(str(payload.get("summary") or "").strip(), context),
                context,
            ),
            context,
        )
        raw_summary = self._clean_llm_narrative_text(raw_summary, context, max_chars=700)
        prediction_interpretation = self._sanitize_feature_absence_claims(
            self._sanitize_ungrounded_feature_attribution(
                self._humanize_internal_prediction_fields(str(payload.get("prediction_interpretation") or ""), context),
                context,
            ),
            context,
        )
        prediction_interpretation = self._clean_llm_narrative_text(prediction_interpretation, context, max_chars=500)
        limitations = [
            self._clean_llm_narrative_text(
                self._sanitize_feature_absence_claims(
                    self._sanitize_ungrounded_feature_attribution(self._humanize_internal_prediction_fields(str(item), context), context),
                    context,
                ),
                context,
                max_chars=260,
            )
            for item in limitations
        ]
        unsupported = [
            self._clean_llm_narrative_text(
                self._sanitize_feature_absence_claims(
                    self._sanitize_ungrounded_feature_attribution(self._humanize_internal_prediction_fields(str(item), context), context),
                    context,
                ),
                context,
                max_chars=260,
            )
            for item in unsupported
        ]
        summary = self._compose_llm_summary(
            raw_summary or self._deterministic_explanation(context),
            prediction_interpretation,
            evidence_support,
            [str(item)[:400] for item in limitations],
            [str(item)[:400] for item in unsupported],
            context,
        )
        return {
            "summary": self._trim_to_complete_sentences(summary, max_chars=1800),
            "prediction_interpretation": prediction_interpretation,
            "evidence_support": evidence_support,
            "limitations": [str(item)[:400] for item in limitations],
            "unsupported_claims": [str(item)[:400] for item in unsupported],
            "citation_ids": citation_ids,
            "grounding_policy": "citations_filtered_to_retrieved_evidence_and_feature_claims_checked",
        }

    def _feature_grounding_summary(self, context: Dict[str, Any]) -> Dict[str, Any]:
        schema = context.get("artifacts", {}).get("feature_schema") or {}
        blocks = schema.get("blocks") if isinstance(schema, dict) else {}
        feature_names = schema.get("feature_names") if isinstance(schema, dict) else []
        return {
            "has_features": bool(context.get("features") or feature_names),
            "n_feature_bundles": len(context.get("features", []) or []),
            "total_dim": schema.get("total_dim") if isinstance(schema, dict) else None,
            "feature_dim": len(feature_names) if isinstance(feature_names, list) else None,
            "blocks": blocks if isinstance(blocks, dict) else {},
            "rules_source_path": context.get("artifacts", {}).get("rules_source_path"),
            "rules_aligned_to_model_schema": context.get("artifacts", {}).get("rules_aligned_to_model_schema"),
        }

    def _sanitize_feature_absence_claims(self, text: str, context: Dict[str, Any]) -> str:
        if not text or not self._feature_grounding_summary(context).get("has_features"):
            return text
        chinese = self._looks_chinese(context.get("request")) or self._looks_chinese(text)
        replacement = (
            "本次已由 GaugeAgent 构建分子描述符、指纹和模型特征；结构解释仍需结合模型特征与检索证据谨慎解读。"
            if chinese
            else "GaugeAgent built molecular descriptors, fingerprints, and model features for this run; the structural interpretation should still be read with the retrieved evidence."
        )
        patterns = [
            r"因?未(输入|提供|给出)(具体的)?(分子)?(描述符|特征|指纹|模型输入)(向量|输入|数据)?[^。；.]*[。；.]?",
            r"当前(任务|分析|运行)?未(输入|提供|给出)[^。；.]{0,40}(描述符|特征|指纹|模型输入)[^。；.]*[。；.]?",
            r"no\s+(molecular\s+)?(descriptors?|fingerprints?|feature vectors?|model inputs?)\s+(were\s+)?(provided|available|given)[^.]*\.",
            r"(descriptors?|fingerprints?|feature vectors?|model inputs?)\s+(were\s+)?missing[^.]*\.",
        ]
        cleaned = text
        changed = False
        for pattern in patterns:
            cleaned_next = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)
            if cleaned_next != cleaned:
                changed = True
            cleaned = cleaned_next
        if changed:
            context.setdefault("warnings", []).append("llm_reasoning_feature_absence_claim_corrected")
        cleaned = re.sub(rf"(?:{re.escape(replacement)}\s*){{2,}}", replacement + " ", cleaned)
        return cleaned.strip()

    def _sanitize_ungrounded_feature_attribution(self, text: str, context: Dict[str, Any]) -> str:
        if not text:
            return text
        chinese = self._looks_chinese(context.get("request")) or self._looks_chinese(text)
        replacement = (
            "模型使用本次构建并与所选模型 schema 对齐的分子特征进行推断；具体决策机制仍应结合特征、证据和模型限制谨慎解读。"
            if chinese
            else "The model used the molecular features built for this run and aligned to the selected model schema; the exact decision mechanism should still be interpreted cautiously with features, evidence, and model limitations."
        )
        patterns = [
            r"模型综合了[^。；.]{0,120}(序列|图像|视觉|空间构型|空间构象|隐藏机制|捕捉)[^。；.]*[。；.]?",
            r"模型可能捕捉到?了[^。；.]{0,120}(空间构型|空间构象|图像语义|序列语义|隐藏机制)[^。；.]*[。；.]?",
            r"model\s+(captured|captures|learned|learns)[^.]{0,120}(image|visual|sequence|spatial|conformation|hidden mechanism)[^.]*\.",
        ]
        cleaned = text
        changed = False
        for pattern in patterns:
            cleaned_next = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)
            if cleaned_next != cleaned:
                changed = True
            cleaned = cleaned_next
        if changed:
            context.setdefault("warnings", []).append("llm_reasoning_ungrounded_feature_attribution_corrected")
        cleaned = re.sub(rf"(?:{re.escape(replacement)}\s*){{2,}}", replacement + " ", cleaned)
        return cleaned.strip()

    def _compose_llm_summary(
        self,
        summary: str,
        prediction_interpretation: str,
        evidence_support: List[Dict[str, Any]],
        limitations: List[str],
        unsupported_claims: List[str],
        context: Dict[str, Any],
    ) -> str:
        chinese = self._looks_chinese(context.get("request"))
        if context.get("task_type") == GENERAL_TASK_TYPE or not self._prediction_summary(context):
            parts = [summary.strip()]
        else:
            evidence_count = len({_evidence_key(item) for item in context.get("evidence", []) or [] if isinstance(item, dict)})
            parts = [
                self._front_conclusion(context, chinese=chinese),
            ]
            if summary and not self._mechanical_or_model_first_summary(summary):
                parts.append(self._strip_user_facing_section_label(summary.strip()))
            if self._should_include_llm_prediction_interpretation(prediction_interpretation, context):
                parts.append(self._strip_user_facing_section_label(prediction_interpretation.strip()))
            parts.append(self._structure_sentence(context, chinese=chinese))
            if evidence_support:
                claims = [
                    self._strip_user_facing_section_label(str(item.get("claim") or "").strip())
                    for item in evidence_support
                    if item.get("claim")
                ]
                if claims:
                    prefix = "证据解读：" if chinese else "Evidence context: "
                    parts.append(prefix + "；".join(claims[:3]))
            else:
                parts.append(self._evidence_sentence(context, evidence_count, chinese=chinese))
            cautions = self._user_facing_cautions(limitations, unsupported_claims, context)
            if cautions:
                prefix = "解读限制：" if chinese else "Caution: "
                if chinese:
                    caution_text = "；".join(item.rstrip("。；; .") for item in cautions[:2] if item.strip())
                    parts.append(prefix + caution_text + "。")
                else:
                    parts.append(prefix + " ".join(cautions[:2]))
            parts.append(self._prediction_source_sentence(context, chinese=chinese))
        return self._trim_to_complete_sentences(
            self._humanize_internal_prediction_fields(" ".join(part for part in parts if part), context),
            max_chars=1800,
        )

    def _strip_user_facing_section_label(self, text: str) -> str:
        cleaned = str(text or "").strip()
        cleaned = re.sub(
            r"^(?:Conclusion|Additional note|Prediction interpretation|Evidence and limitations|Limitations|Caution|Method note)\s*[:：]\s*",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(r"^(?:结论|补充说明|预测解释|证据与限制|证据解读|限制|解读限制|方法说明)\s*[:：]\s*", "", cleaned)
        return cleaned.strip()

    def _user_facing_cautions(self, limitations: List[str], unsupported_claims: List[str], context: Dict[str, Any]) -> List[str]:
        chinese = self._looks_chinese(context.get("request"))
        standard = (
            "这是模型辅助判断，不应等同于直接的体内或体外实验测量。"
            if chinese
            else "This is a model-assisted assessment rather than a direct in vitro or in vivo permeability measurement."
        )
        cautions: List[str] = []
        for item in list(limitations or []) + list(unsupported_claims or []):
            cleaned = self._strip_user_facing_section_label(str(item or "").strip())
            if not cleaned:
                continue
            lowered = cleaned.lower()
            internal_terms_zh = [
                "具体决策机制",
                "特征权重",
                "隐藏决策",
                "AgentM",
                "ReasoningAgent",
                "GaugeAgent",
                "RetrieveAgent",
                "VerifierAgent",
                "tool-grounded",
            ]
            if any(
                phrase in lowered
                for phrase in [
                    "exact decision pathway",
                    "specific feature weighting",
                    "hidden decision",
                    "llm",
                    "agentm",
                    "reasoningagent",
                    "gaugeagent",
                    "retrieveagent",
                    "verifieragent",
                    "tool-grounded",
                ]
            ) or any(phrase in cleaned for phrase in internal_terms_zh):
                if standard not in cautions:
                    cautions.append(standard)
                continue
            model_terms_zh = ["模型概率", "后处理校准", "校准概率", "监督机器学习"]
            if any(
                phrase in lowered
                for phrase in [
                    "supervised machine learning framework",
                    "trained on curated",
                    "model probability",
                    "post-hoc calibrated",
                    "calibrated probability",
                ]
            ) or any(phrase in cleaned for phrase in model_terms_zh):
                if standard not in cautions:
                    cautions.append(standard)
                continue
            cautions.append(cleaned.rstrip("。；; .") + ("。" if chinese else "."))
        deduped: List[str] = []
        seen = set()
        for caution in cautions:
            key = normalize_text(caution).lower()
            if key and key not in seen:
                seen.add(key)
                deduped.append(caution)
        return deduped

    def _clean_llm_narrative_text(self, text: str, context: Dict[str, Any], *, max_chars: int) -> str:
        cleaned = self._repair_decimal_spacing(normalize_text(text))
        if not cleaned:
            return ""
        cleaned = self._drop_low_quality_llm_sentences(cleaned, context)
        return self._trim_to_complete_sentences(cleaned, max_chars=max_chars)

    def _drop_low_quality_llm_sentences(self, text: str, context: Dict[str, Any]) -> str:
        if not text:
            return ""
        sentences = self._sentence_units(text)
        if not sentences:
            return text.strip()
        kept: List[str] = []
        for sentence in sentences:
            lowered = sentence.lower()
            bad_patterns = [
                r"assigns\s+a\s+(?:low|high|moderate|medium)\s+the\s+prediction",
                r"\bthe\s+prediction\s+of\s+bbb\s+penetration\b",
                r"\bunderlying\s+algorithm\s+integrates\s+multiple\s+structured\s+molecular\s+representations\b",
                r"\bframework\s+calibrated\s+to\s+distinguish\b",
                r"\bhidden\s+decision\s+pathways?\b",
            ]
            if any(re.search(pattern, lowered, flags=re.IGNORECASE) for pattern in bad_patterns):
                context.setdefault("warnings", []).append("llm_reasoning_low_quality_sentence_dropped")
                continue
            kept.append(sentence.strip())
        return " ".join(kept).strip()

    def _should_include_llm_prediction_interpretation(self, text: str, context: Dict[str, Any]) -> bool:
        cleaned = str(text or "").strip()
        if not cleaned or self._mechanical_or_model_first_summary(cleaned):
            return False
        lowered = cleaned.lower()
        modelish_terms = [
            "computational evaluation",
            "underlying algorithm",
            "classifier",
            "random forest",
            "model",
            "feature vector",
            "molecular representation",
            "training data",
            "calibrated to distinguish",
            "模型使用",
            "模型综合",
            "具体决策机制",
            "特征权重",
            "模型输入",
            "schema",
        ]
        if any(term in lowered for term in modelish_terms) or any(term in cleaned for term in modelish_terms):
            return False
        if self._near_prediction_boundary(context, self._prediction_summary(context)) and any(
            phrase in lowered for phrase in ["close to the decision boundary", "near the classification boundary", "置信度有限", "分类边界"]
        ):
            return False
        return True

    def _trim_to_complete_sentences(self, text: str, *, max_chars: int) -> str:
        cleaned = self._repair_decimal_spacing(normalize_text(text))
        if not cleaned:
            return ""
        if len(cleaned) <= max_chars and self._ends_with_sentence_punctuation(cleaned):
            return cleaned
        candidate = cleaned if len(cleaned) <= max_chars else cleaned[:max_chars].rstrip()
        sentences = self._sentence_units(candidate)
        if sentences:
            joined = " ".join(sentence.strip() for sentence in sentences if sentence.strip()).strip()
            if joined:
                return joined
        if len(cleaned) <= max_chars:
            return cleaned.rstrip(" ,;:") + ("。" if self._looks_chinese(cleaned) else ".")
        return candidate.rstrip(" ,;:")

    def _sentence_units(self, text: str) -> List[str]:
        cleaned = self._repair_decimal_spacing(normalize_text(text))
        if not cleaned:
            return []
        matches = re.findall(r"[^。！？.!?]+[。！？.!?]", cleaned)
        return [match.strip() for match in matches if match.strip()]

    def _ends_with_sentence_punctuation(self, text: str) -> bool:
        return bool(re.search(r"[。！？.!?][\"')\]]?\s*$", str(text or "").strip()))

    def _repair_decimal_spacing(self, text: str) -> str:
        return re.sub(r"(?<=\d)\.\s+(?=\d)", ".", str(text or ""))

    def _mechanical_or_model_first_summary(self, text: str) -> bool:
        lowered = str(text or "").strip().lower()
        if not lowered:
            return False
        patterns = [
            r"^基于[^。；,.]{0,80}(模型|random forest|随机森林)",
            r"^(based on|using)\s+[^.]{0,80}(random forest|model)",
            r"\bcomputational evaluation\s+assigns\s+a\s+(?:low|high|moderate|medium)\s+the\s+prediction\b",
            r"(序列与图像语义|图像语义|空间构型|空间构象|hidden mechanism)",
        ]
        return any(re.search(pattern, lowered, flags=re.IGNORECASE) for pattern in patterns)

    def _humanize_internal_prediction_fields(self, text: str, context: Dict[str, Any]) -> str:
        if not text:
            return text
        chinese = self._looks_chinese(context.get("request")) or self._looks_chinese(text)
        prediction = self._prediction_summary(context)
        strength = self._qualitative_prediction_strength(
            prediction.get("probability") if prediction.get("probability") is not None else prediction.get("score"),
            chinese=chinese,
        )
        score_phrase = f"整体判断强度为{strength}" if chinese else f"the overall support is {strength}"
        label_meaning = self._prediction_meaning(prediction)
        if chinese:
            label_phrase = f"判断为“{label_meaning}”" if label_meaning else "给出当前方向性判断"
            replacements = [
                (r"(模型输出)?(置信度)?(综合)?(分数|评分|得分)\s*(为|=|:|约|≈)?\s*\d+(?:\.\d+)?", score_phrase),
                (r"该(分数|评分|得分)", "该判断"),
                (r"(score|probability)\s*(=|:|为)?\s*\d+(?:\.\d+)?", score_phrase),
                (r"(BBBP\s*)?label\s*(=|:|为)?\s*\d+", label_phrase),
                (r"预测结果为\s*\d+", label_phrase),
            ]
        else:
            label_phrase = f"interpreted as {label_meaning}" if label_meaning else "the model direction"
            replacements = [
                (r"\b(score|probability)\s*(=|:|is|of)?\s*\d+(?:\.\d+)?\b", score_phrase),
                (r"\b(the\s+)?(score|probability)\b", "the prediction"),
                (r"\blabel\s*(=|:|is|of)?\s*\d+\b", label_phrase),
                (r"\bmodel output\s+(score|probability)\s*(=|:|is|of)?\s*\d+(?:\.\d+)?\b", score_phrase),
            ]
        cleaned = text
        for pattern, repl in replacements:
            cleaned = re.sub(pattern, repl, cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s+([，。；,.])", r"\1", cleaned)
        cleaned = re.sub(r"\s{2,}", " ", cleaned)
        return cleaned.strip()

    def _build_llm_reflection(self, context: Dict[str, Any], base_reflection: Dict[str, Any]) -> Dict[str, Any]:
        metadata = context.get("metadata", {}) or {}
        if not metadata.get("use_llm_reasoning", self.use_llm_reasoning) or not self._llm_reasoning_available():
            return {}
        try:
            payload = self._llm_json(self._llm_reflection_messages(context, base_reflection))
        except Exception as exc:
            context.setdefault("warnings", []).append(f"llm_reflection_failed: {exc}")
            return {}
        refinement_queries = payload.get("refinement_queries", []) or []
        if not isinstance(refinement_queries, list):
            refinement_queries = [str(refinement_queries)]
        return {
            "recommendation": str(payload.get("recommendation") or base_reflection.get("recommendation") or "finalize"),
            "risk_assessment": str(payload.get("risk_assessment") or "")[:800],
            "refinement_queries": [str(query)[:300] for query in refinement_queries[:3]],
            "needs_refinement": bool(payload.get("needs_refinement", base_reflection.get("needs_refinement"))),
            "grounding_policy": "llm_reflection_is_advisory_guard_remains_authoritative",
        }

    def _llm_reflection_messages(self, context: Dict[str, Any], base_reflection: Dict[str, Any]) -> List[Dict[str, str]]:
        payload = {
            "request": context.get("request"),
            "task_type": context.get("task_type"),
            "dataset_name": context.get("dataset_name"),
            "prediction": self._prediction_summary(context),
            "confidence": context.get("confidence", {}),
            "base_reflection": base_reflection,
            "verifier_report": context.get("verifier_report", {}),
            "evidence": self._evidence_summary(context),
        }
        return [
            {
                "role": "system",
                "content": (
                    "You are AgentM ReflectionAgent. Return JSON only with keys recommendation, "
                    "needs_refinement, risk_assessment, refinement_queries. The response is advisory; "
                    "do not change predictions or invent citations."
                ),
            },
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]

    def _prediction_summary(self, context: Dict[str, Any]) -> Dict[str, Any]:
        prediction = self._active_prediction(context)
        if prediction is None:
            return {}
        payload = self._prediction_to_dict(prediction)
        return {key: payload.get(key) for key in ["task_type", "dataset_name", "label", "score", "probability", "model_type", "details"]}

    def _evidence_summary(self, context: Dict[str, Any], *, limit: int = 8) -> List[Dict[str, Any]]:
        summary = []
        for item in (context.get("evidence", []) or [])[:limit]:
            if not isinstance(item, dict):
                continue
            summary.append(
                {
                    "citation_id": item.get("citation_id"),
                    "chunk_id": item.get("chunk_id"),
                    "source": item.get("source"),
                    "source_path": item.get("source_path"),
                    "claim": str(item.get("claim") or item.get("snippet") or "")[:900],
                }
            )
        return summary

    def _valid_citation_ids(self, context: Dict[str, Any]) -> set[str]:
        citation_ids = set()
        for item in context.get("evidence", []) or []:
            if isinstance(item, dict) and item.get("citation_id"):
                citation_ids.add(str(item["citation_id"]))
        return citation_ids

    def _build_features_with(self, builder: FeatureBuilder, context: Dict[str, Any], *, status: str) -> ToolResult:
        rules = context.get("rules", []) or []
        task_type = context.get("task_type", "property_prediction")
        dataset_name = context.get("dataset_name", "BBBP")
        features = [
            builder.build_single(molecule, rules=rules, task_type=task_type, dataset_name=dataset_name)
            for molecule in context.get("molecules", [])
        ]
        context["features"] = features
        artifacts = context.setdefault("artifacts", {})
        if features:
            artifacts["feature_schema"] = features[0].feature_schema.to_dict()
        if task_type == "ddi_prediction" and len(features) >= 2:
            pair = builder.build_ddi_pair(features[0], features[1])
            context["ddi_pair"] = pair
            artifacts["ddi_feature_schema"] = pair.feature_schema.to_dict()
            return ToolResult("build_features", status, {"n_features": len(features), "ddi_pair": True, "feature_dim": len(pair.feature_vector)})
        return ToolResult("build_features", status, {"n_features": len(features), "ddi_pair": False, "feature_dim": len(features[0].feature_vector) if features else 0})

    def _open_task_contract(self, context: Mapping[str, Any]) -> Dict[str, Any]:
        contract = (context.get("metadata", {}) or {}).get("task_contract")
        return dict(contract) if isinstance(contract, Mapping) else {}

    def _active_open_subtask(self, context: Mapping[str, Any]) -> Dict[str, Any]:
        contract = self._open_task_contract(context)
        active_id = str((context.get("metadata", {}) or {}).get("active_open_subtask_id") or "")
        if not active_id:
            return {}
        for item in contract.get("subtasks", []) or []:
            if isinstance(item, Mapping) and str(item.get("subtask_id") or "") == active_id:
                return dict(item)
        return {}

    def _activate_open_subtask(
        self,
        args: Dict[str, Any],
        context: Dict[str, Any],
        *,
        expected_kind: str | None = None,
    ) -> Dict[str, Any]:
        """Activate a request-derived subtask and isolate its mutable inputs."""

        contract = self._open_task_contract(context)
        subtasks = [dict(item) for item in contract.get("subtasks", []) or [] if isinstance(item, Mapping)]
        if not subtasks:
            return {}
        requested_id = str(args.get("subtask_id") or "")
        requested_dataset = str(args.get("dataset_name") or args.get("endpoint") or "").lower()
        requested_molecules = [str(item).lower() for item in args.get("molecules", []) or [] if item]
        candidates = [item for item in subtasks if not expected_kind or item.get("kind") == expected_kind]
        selected: Dict[str, Any] = {}
        if requested_id:
            selected = next((item for item in candidates if str(item.get("subtask_id") or "") == requested_id), {})
        if not selected and requested_dataset:
            selected = next(
                (item for item in candidates if str(item.get("dataset_name") or "").lower() == requested_dataset),
                {},
            )
        if not selected and requested_molecules:
            selected = next(
                (
                    item
                    for item in candidates
                    if all(value in [str(molecule).lower() for molecule in item.get("molecules", []) or []] for value in requested_molecules)
                ),
                {},
            )
        active = self._active_open_subtask(context)
        if not selected and active and (not expected_kind or active.get("kind") == expected_kind):
            selected = active
        if not selected and expected_kind:
            resolve_tool = "resolve_ddi_pair" if expected_kind == "ddi" else "resolve_molecule"
            completed_ids = {
                str((call.get("args") or {}).get("subtask_id") or "")
                for call in context.get("tool_calls", []) or []
                if call.get("tool") == resolve_tool and call.get("status") in {"completed", "fallback_completed"}
            }
            selected = next(
                (item for item in candidates if str(item.get("subtask_id") or "") not in completed_ids),
                {},
            )
        if not selected and len(candidates) == 1:
            selected = candidates[0]
        if not selected:
            return {}

        subtask_id = str(selected["subtask_id"])
        metadata = context.setdefault("metadata", {})
        phase_changed = str(metadata.get("active_open_subtask_id") or "") != subtask_id
        args.setdefault("subtask_id", subtask_id)
        args.setdefault("task_type", selected.get("task_type"))
        args.setdefault("dataset_name", selected.get("dataset_name"))
        args.setdefault("molecules", list(selected.get("molecules", []) or []))
        if phase_changed:
            metadata["active_open_subtask_id"] = subtask_id
            metadata.setdefault("open_task_state_trace", []).append(
                {
                    "event": "activate_subtask",
                    "subtask_id": subtask_id,
                    "kind": selected.get("kind"),
                    "task_type": selected.get("task_type"),
                    "dataset_name": selected.get("dataset_name"),
                    "molecules": list(selected.get("molecules", []) or []),
                }
            )
            context["task_type"] = str(selected.get("task_type") or context.get("task_type") or "")
            context["dataset_name"] = str(selected.get("dataset_name") or context.get("dataset_name") or "")
            context["molecule_mentions"] = list(selected.get("molecules", []) or [])
            context["molecules"] = []
            context["rules"] = []
            context["features"] = []
            context["ddi_pair"] = None
            context["confidence"] = {}
            context["resource_resolution"] = {}
        return selected

    def _record_open_subtask_state(self, context: Dict[str, Any], key: str, value: Any) -> None:
        active = self._active_open_subtask(context)
        if not active:
            return
        subtask_id = str(active["subtask_id"])
        bucket = context.setdefault("metadata", {}).setdefault("open_task_subtask_results", {}).setdefault(
            subtask_id,
            {
                "subtask_id": subtask_id,
                "kind": active.get("kind"),
                "task_type": active.get("task_type"),
                "dataset_name": active.get("dataset_name"),
                "molecules": list(active.get("molecules", []) or []),
                "evidence_scope": active.get("evidence_scope"),
            },
        )
        bucket[key] = to_jsonable(value)

    def _tag_open_task_prediction(self, context: Dict[str, Any], prediction: Any) -> Any:
        active = self._active_open_subtask(context)
        if not active:
            return prediction
        details = prediction.get("details") if isinstance(prediction, dict) else getattr(prediction, "details", None)
        if not isinstance(details, dict):
            details = {}
            if isinstance(prediction, dict):
                prediction["details"] = details
            else:
                setattr(prediction, "details", details)
        details.update(
            {
                "open_task_subtask_id": active.get("subtask_id"),
                "open_task_kind": active.get("kind"),
                "open_task_dataset_name": active.get("dataset_name"),
                "open_task_molecules": list(active.get("molecules", []) or []),
                "open_task_evidence_scope": active.get("evidence_scope"),
            }
        )
        return prediction

    def _append_open_task_prediction(self, context: Dict[str, Any], prediction: Any) -> None:
        prediction = self._tag_open_task_prediction(context, prediction)
        if not self._open_task_contract(context):
            context["predictions"] = [prediction]
            return
        active = self._active_open_subtask(context)
        subtask_id = str(active.get("subtask_id") or "")
        retained = []
        for item in context.get("predictions", []) or []:
            details = item.get("details") if isinstance(item, Mapping) else getattr(item, "details", None)
            if subtask_id and isinstance(details, Mapping) and str(details.get("open_task_subtask_id") or "") == subtask_id:
                continue
            retained.append(item)
        retained.append(prediction)
        context["predictions"] = retained

    def _active_prediction(self, context: Dict[str, Any]) -> Any:
        predictions = list(context.get("predictions", []) or [])
        active = self._active_open_subtask(context)
        subtask_id = str(active.get("subtask_id") or "")
        if subtask_id:
            for item in reversed(predictions):
                details = item.get("details") if isinstance(item, Mapping) else getattr(item, "details", None)
                if isinstance(details, Mapping) and str(details.get("open_task_subtask_id") or "") == subtask_id:
                    return item
        return predictions[-1] if self._open_task_contract(context) and predictions else (predictions[0] if predictions else None)

    def _active_subtask_evidence(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        evidence = [item for item in context.get("evidence", []) or [] if isinstance(item, dict)]
        active = self._active_open_subtask(context)
        subtask_id = str(active.get("subtask_id") or "")
        if not subtask_id:
            return evidence
        return [item for item in evidence if str(item.get("open_task_subtask_id") or "") == subtask_id]

    def _restore_open_task_root(self, context: Dict[str, Any]) -> None:
        contract = self._open_task_contract(context)
        if not contract:
            return
        metadata = context.setdefault("metadata", {})
        metadata["active_open_subtask_id"] = ""
        metadata["open_task_execution_complete"] = True
        context["task_type"] = str(contract.get("root_task_type") or metadata.get("root_task_type") or context.get("task_type") or "")
        context["dataset_name"] = str(contract.get("root_dataset_name") or metadata.get("root_dataset_name") or context.get("dataset_name") or "")
        root_kind = "ddi" if context["task_type"] == "ddi_prediction" else "single"
        root_subtask = next(
            (item for item in contract.get("subtasks", []) or [] if isinstance(item, Mapping) and item.get("kind") == root_kind),
            None,
        )
        if isinstance(root_subtask, Mapping):
            saved = (metadata.get("open_task_subtask_results", {}) or {}).get(str(root_subtask.get("subtask_id")), {})
            if isinstance(saved, Mapping) and saved.get("molecules"):
                context["molecules"] = list(saved["molecules"])

    def _composite_open_task_explanation(self, context: Dict[str, Any], contract: Mapping[str, Any]) -> str:
        predictions = list(context.get("predictions", []) or [])
        by_subtask: Dict[str, Any] = {}
        for prediction in predictions:
            details = prediction.get("details") if isinstance(prediction, Mapping) else getattr(prediction, "details", None)
            if isinstance(details, Mapping) and details.get("open_task_subtask_id"):
                by_subtask[str(details["open_task_subtask_id"])] = prediction
        results = context.get("metadata", {}).get("open_task_subtask_results", {}) or {}
        clauses: List[str] = []
        missing: List[str] = []
        evidence_counts: Dict[str, int] = {}
        for subtask in contract.get("subtasks", []) or []:
            if not isinstance(subtask, Mapping):
                continue
            subtask_id = str(subtask.get("subtask_id") or "")
            prediction = by_subtask.get(subtask_id)
            molecules = " + ".join(str(item) for item in subtask.get("molecules", []) or [])
            endpoint = str(subtask.get("dataset_name") or "the requested endpoint")
            scoped_evidence = [
                item
                for item in context.get("evidence", []) or []
                if isinstance(item, Mapping) and str(item.get("open_task_subtask_id") or "") == subtask_id
            ]
            evidence_counts[subtask_id] = len(scoped_evidence)
            if prediction is None:
                missing.append(f"{molecules} {endpoint}")
                continue
            summary = self._prediction_to_dict(prediction)
            if subtask.get("kind") == "ddi":
                if is_multiclass_ddi_prediction(summary):
                    direction = self._prediction_meaning(
                        summary,
                        drug_names=[str(item) for item in subtask.get("molecules", []) or []],
                    )
                    direction = f"predicted {direction}; this is not a binary no-interaction result"
                else:
                    positive = self._is_positive_prediction(summary)
                    direction = "a potential DDI risk signal" if positive else "no clear DDI risk signal"
            elif endpoint.lower() == "herg":
                positive = self._is_positive_prediction(summary)
                direction = "an hERG-related safety signal" if positive else "no obvious hERG-related safety signal"
            elif endpoint.lower() == "dili":
                positive = self._is_positive_prediction(summary)
                direction = "a DILI risk signal" if positive else "no obvious DILI risk signal"
            else:
                direction = self._prediction_meaning(summary) or "a model-derived directional result"
            clauses.append(f"{molecules} / {endpoint}: {direction} (model prediction)")
        task_type = str(contract.get("open_task_type") or "")
        if task_type == "ddi_plus_individual_safety":
            scope_text = (
                "Pair-level evidence and single-drug evidence are kept separate by subtask-scoped citation identifiers; "
                "the DDI result is not used as evidence for either individual safety endpoint, and vice versa."
            )
        else:
            counts = [evidence_counts.get(str(item.get("subtask_id") or ""), 0) for item in contract.get("subtasks", []) or [] if isinstance(item, Mapping)]
            strongest = max(counts) if counts else 0
            weakest = min(counts) if counts else 0
            scope_text = (
                f"Evidence strength was compared endpoint by endpoint using scoped retrieval records (range {weakest}–{strongest} retained records); "
                "predictions and external evidence remain distinct."
            )
        completion = "All requested prediction subtasks were completed." if not missing else (
            "The following requested subtasks remain incomplete: " + "; ".join(missing) + "."
        )
        uncertainty = (
            "These are screening-model outputs rather than experimental facts; endpoints with no direct evidence remain uncertain and require cautious interpretation."
        )
        return " ".join(
            part
            for part in [
                "Composite assessment: " + "; ".join(clauses) + "." if clauses else "Composite assessment produced no model prediction.",
                scope_text,
                completion,
                uncertainty,
            ]
            if part
        )

    def _prediction_to_dict(self, prediction: Any) -> Dict[str, Any]:
        if isinstance(prediction, Mapping):
            return dict(prediction)
        if hasattr(prediction, "to_dict"):
            value = prediction.to_dict()
            return dict(value) if isinstance(value, Mapping) else {}
        return {
            "task_type": getattr(prediction, "task_type", None),
            "dataset_name": getattr(prediction, "dataset_name", None),
            "label": getattr(prediction, "label", None),
            "score": getattr(prediction, "score", None),
            "probability": getattr(prediction, "probability", None),
            "model_type": getattr(prediction, "model_type", None),
            "details": getattr(prediction, "details", None),
        }

    def _prediction_agent_for(self, context: Dict[str, Any]) -> PredictionAgent:
        self._update_resource_resolution(context)
        selected = self._selected_model_path(context)
        if not selected:
            return self.prediction_agent
        if selected not in self._prediction_agents:
            self._prediction_agents[selected] = PredictionAgent(model_path=selected)
        return self._prediction_agents[selected]

    def _prediction_dataset_name(self, context: Dict[str, Any], *, default: str) -> str:
        model_resolution = (context.get("resource_resolution") or {}).get("model") or {}
        selected = self._selected_model_path(context)
        selected_candidate = self._selected_model_candidate(model_resolution, selected) if selected else {}
        # core PredictionAgent validates against the training metadata name, e.g. DrugBank for DDI.
        return (
            selected_candidate.get("metadata_dataset_name")
            or model_resolution.get("metadata_dataset_name")
            or default
        )

    def _annotate_prediction_dataset_alias(self, prediction: Any, context: Dict[str, Any], prediction_dataset: str) -> None:
        requested_dataset = context.get("dataset_name")
        if requested_dataset and prediction_dataset and str(requested_dataset) != str(prediction_dataset):
            details = getattr(prediction, "details", None)
            if isinstance(details, dict):
                details.setdefault("requested_dataset_name", requested_dataset)
                details.setdefault("model_dataset_name", prediction_dataset)

    def _update_resource_resolution(self, context: Dict[str, Any]) -> None:
        metadata = context.setdefault("metadata", {})
        task_type = context.get("task_type") or "admet_prediction"
        if is_reaction_task(task_type):
            resolution = context.setdefault("resource_resolution", {})
            reaction_resolution = self.reaction_runtime.resolve_resources()
            context["reaction_resource_resolution"] = reaction_resolution
            resolution["reaction"] = reaction_resolution
            resolution["model"] = {
                "status": "reaction_model_resolved_by_reaction_agent",
                "selected_path": None,
                "candidate_paths": [],
                "schema_compatible": True,
            }
            resolution["rag"] = {"status": "not_used_for_reaction_first_phase", "all_required_present": True}
            context.setdefault("artifacts", {})["resource_resolution"] = resolution
            return
        if task_type in NON_SCIENCE_TASK_TYPES:
            resolution = context.setdefault("resource_resolution", {})
            resolution["model"] = {
                "status": "not_requested",
                "selected_path": None,
                "candidate_paths": [],
                "missing_paths": [],
                "schema_compatible": None,
                "reason": "intent_gate_blocked_scientific_prediction",
            }
            resolution["rag"] = {
                "status": "not_requested",
                "all_required_present": True,
                "reason": "intent_gate_blocked_scientific_retrieval",
            }
            context.setdefault("artifacts", {})["resource_resolution"] = resolution
            return
        if task_type == GENERAL_TASK_TYPE:
            resolution = context.setdefault("resource_resolution", {})
            resolution["model"] = {
                "status": "not_requested",
                "selected_path": None,
                "candidate_paths": [],
                "missing_paths": [],
                "schema_compatible": None,
                "reason": "general_drug_profile_does_not_request_endpoint_prediction",
            }
            resolution["rag"] = resolve_rag_resources(
                task_type="admet_prediction",
                rag_corpus_dirs=self.rag_corpus_dirs,
                ddi_rag_corpus_dir=getattr(self.ddi_rag_evidence_agent, "corpus_dir", None),
                rag_embedding_model_name=getattr(self.rag_evidence_agent, "embedding_model_name", None),
            )
            context.setdefault("artifacts", {})["resource_resolution"] = resolution
            return
        dataset_name = context.get("dataset_name") or ("DDI" if task_type == "ddi_prediction" else "BBB_Martins")
        resolution = context.setdefault("resource_resolution", {})
        resolution["model"] = resolve_model_resource(
            task_type=task_type,
            dataset_name=dataset_name,
            feature_set=self.feature_set,
            # A composite task may switch endpoint and task family within one
            # controller.  A root-task explicit model must not be reused for a
            # different subtask; each phase is resolved against the registry.
            explicit_model_path=None if self._active_open_subtask(context) else self.model_path,
            auto_model=bool(metadata.get("auto_model", self.auto_model)),
            model_registry_path=metadata.get("model_registry_path") or self.model_registry_path,
        )
        resolution["rag"] = resolve_rag_resources(
            task_type=task_type,
            rag_corpus_dirs=self.rag_corpus_dirs,
            ddi_rag_corpus_dir=getattr(self.ddi_rag_evidence_agent, "corpus_dir", None),
            rag_embedding_model_name=getattr(self.rag_evidence_agent, "embedding_model_name", None),
        )
        context.setdefault("artifacts", {})["resource_resolution"] = resolution

    def _refresh_objective_status(self, context: Dict[str, Any]) -> None:
        objectives = evaluate_task_objectives(context)
        context["objective_status"] = objectives
        verifier = context.setdefault("verifier_report", {})
        verifier["objectives"] = objectives
        summary = verifier.setdefault("summary", {})
        summary["objectives"] = objectives
        summary["warnings"] = [
            warning for warning in (summary.get("warnings", []) or []) if not str(warning).startswith("objective_missing:")
        ]
        context["warnings"] = [
            warning for warning in (context.get("warnings", []) or []) if not str(warning).startswith("objective_missing:")
        ]
        if objectives.get("status") == "pass":
            if not summary["warnings"]:
                summary["status"] = "pass"
        elif objectives.get("status") != "pass":
            objective_warnings = [f"objective_missing:{name}" for name in objectives.get("missing_objectives", [])]
            existing = list(summary.get("warnings", []) or [])
            for warning in objective_warnings:
                if warning not in existing:
                    existing.append(warning)
                if warning not in context["warnings"]:
                    context["warnings"].append(warning)
            summary["warnings"] = existing
            summary["status"] = "fail" if objectives.get("status") == "fail" else "warning"

    def _mentions(self, args: Dict[str, Any], context: Dict[str, Any], *, target_count: int) -> List[str]:
        """Resolve molecule arguments without discarding singular planner fields.

        LLM plans commonly express a single molecule as ``smiles`` or
        ``molecule_smiles``.  Previously only the plural ``molecules`` field
        was read, so an otherwise valid explicit SMILES could be ignored and
        the entire natural-language request would be sent to the resolver.
        Prefer explicit, request-grounded scalar fields, then the plural
        schema field, and finally parser-populated context mentions.
        """

        request = str(
            context.get("normalized_request_for_parsing")
            or context.get("request")
            or ""
        )
        plural = args.get("molecules")
        if isinstance(plural, (str, bytes)):
            plural_values = [plural]
        elif isinstance(plural, (list, tuple)):
            plural_values = list(plural)
        else:
            plural_values = []
        scalar_values = [
            args.get(key)
            for key in (
                "smiles",
                "molecule_smiles",
                "molecule_id",
                "molecule",
                "drug",
                "compound",
                "name",
            )
            if args.get(key) not in (None, "")
        ]
        explicit_values = scalar_values or plural_values
        grounded_explicit = [
            value
            for value in explicit_values
            if str(value).strip()
            and str(value).strip().lower() in request.lower()
        ]
        source_values = (
            grounded_explicit
            or plural_values
            or list(context.get("molecule_mentions") or [])
        )
        mentions = [self._normalize_molecule_mention(item) for item in source_values]
        if len(mentions) < target_count:
            request = request or self._normalize_request_for_parsing(context.get("request", ""))
            for sep in [",", ";", "\n", " and ", " + "]:
                if sep in request:
                    parts = [self._normalize_molecule_mention(part.strip()) for part in request.split(sep) if part.strip()]
                    if len(parts) >= target_count:
                        mentions = parts
                        break
        if not mentions:
            mentions = [self._normalize_molecule_mention(context.get("normalized_request_for_parsing") or context.get("request", ""))]
        return mentions

    def _process_molecule_mention(self, mention: str) -> MoleculeRecord:
        normalized = self._normalize_molecule_mention(mention)
        molecule = self.processor.process_input(normalized)
        if molecule.is_valid:
            molecule = self._apply_local_identity_seed(molecule, query=normalized)
            return self._enrich_molecule_identity(molecule)
        seeded = self._molecule_from_local_identity_seed(normalized, original=molecule)
        if seeded is not None:
            return self._enrich_molecule_identity(seeded)
        if not self._network_lookup_enabled():
            return molecule
        repaired = self._pubchem_field_compat_lookup(normalized, molecule)
        if repaired is not None:
            repaired = self._apply_local_identity_seed(repaired, query=normalized)
        return self._enrich_molecule_identity(repaired) if repaired else molecule

    def _enrich_molecule_identity(self, molecule: MoleculeRecord) -> MoleculeRecord:
        if not self._network_lookup_enabled() or not getattr(molecule, "is_valid", False):
            return molecule
        metadata = molecule.metadata if isinstance(molecule.metadata, dict) else {}
        if metadata.get("pubchem_identity_enriched"):
            return molecule
        profile = self._pubchem_profile_for_molecule(molecule)
        metadata["pubchem_identity_enriched"] = profile.get("retrieval_status")
        if profile.get("retrieval_status") != "success":
            metadata.setdefault("identity_resolution_trace", []).append(
                {"source": "pubchem", "status": profile.get("retrieval_status"), "error": profile.get("error")}
            )
            molecule.metadata = metadata
            return molecule
        title = profile.get("title")
        synonyms = [str(item) for item in profile.get("synonyms", []) or [] if item]
        metadata.update(
            {
                "pubchem_cid": profile.get("pubchem_cid"),
                "pubchem_title": title,
                "pubchem_iupac_name": profile.get("iupac_name"),
                "pubchem_synonyms": synonyms,
            }
        )
        current_name = getattr(molecule, "name", None) or ""
        if title and (not current_name or self._looks_like_smiles(current_name)):
            molecule.name = str(title)
        if profile.get("canonical_smiles"):
            molecule.canonical_smiles = profile.get("canonical_smiles")
        molecule.metadata = metadata
        return molecule

    def _molecule_from_local_identity_seed(self, query: str, *, original: MoleculeRecord) -> Optional[MoleculeRecord]:
        seed = self._local_identity_seed(query=query, molecule=None)
        if not seed:
            return None
        canonical, err = standardize_smiles(str(seed.get("canonical_smiles") or ""))
        if err is not None or not canonical:
            return None
        metadata = dict(getattr(original, "metadata", {}) or {})
        metadata.setdefault("resolution_trace", []).append({"stage": "local_identity_seed", "status": "success", "query": query})
        molecule = MoleculeRecord(
            original_input=query,
            smiles=str(seed.get("canonical_smiles") or ""),
            canonical_smiles=canonical,
            name=str(seed.get("title") or query),
            source="agentm_local_identity_seed",
            is_valid=True,
            errors=[],
            metadata=metadata,
        )
        return self._apply_local_identity_seed(molecule, query=query)

    def _apply_local_identity_seed(self, molecule: MoleculeRecord, *, query: str = "") -> MoleculeRecord:
        seed = self._local_identity_seed(query=query, molecule=molecule)
        if not seed:
            return molecule
        metadata = dict(getattr(molecule, "metadata", {}) or {})
        metadata.update(
            {
                "pubchem_identity_enriched": metadata.get("pubchem_identity_enriched") or "local_seed",
                "pubchem_cid": seed.get("pubchem_cid"),
                "pubchem_title": seed.get("title"),
                "pubchem_synonyms": list(seed.get("synonyms", [])),
                "local_identity_seed": True,
            }
        )
        current_name = getattr(molecule, "name", None) or ""
        if seed.get("title") and (not current_name or self._looks_like_smiles(current_name)):
            molecule.name = str(seed["title"])
        canonical, err = standardize_smiles(str(seed.get("canonical_smiles") or ""))
        if err is None and canonical:
            molecule.canonical_smiles = canonical
            molecule.smiles = str(seed.get("canonical_smiles") or molecule.smiles)
        trace = list(metadata.get("identity_resolution_trace", []) or [])
        trace.append({"source": "local_identity_seed", "status": "success", "title": seed.get("title"), "cid": seed.get("pubchem_cid")})
        metadata["identity_resolution_trace"] = trace
        molecule.metadata = metadata
        return molecule

    def _local_identity_seed(self, *, query: str = "", molecule: Optional[MoleculeRecord] = None) -> Optional[Dict[str, Any]]:
        values = [str(query or "").strip()]
        if molecule is not None:
            values.extend(
                str(getattr(molecule, key, "") or "").strip()
                for key in ["name", "original_input", "smiles", "canonical_smiles"]
            )
        text_values = [value.lower() for value in values if value]
        for key, seed in LOCAL_COMPOUND_IDENTITY_SEEDS.items():
            aliases = {key, str(seed.get("title", "")).lower(), *[str(item).lower() for item in seed.get("synonyms", [])]}
            if any(value and value in aliases for value in text_values):
                return dict(seed)
            seed_canonical, seed_err = standardize_smiles(str(seed.get("canonical_smiles") or ""))
            if seed_err is not None or not seed_canonical:
                continue
            for value in values:
                if not value or not self._looks_like_smiles(value):
                    continue
                canonical, err = standardize_smiles(value)
                if err is None and canonical == seed_canonical:
                    return dict(seed)
        return None

    def _retrieve_general_database_profiles(self, context: Dict[str, Any]) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        profiles: List[Dict[str, Any]] = []
        evidence: List[Dict[str, Any]] = []
        for idx, molecule in enumerate(context.get("molecules", []) or [], start=1):
            if not getattr(molecule, "is_valid", True):
                continue
            candidate_profiles = [
                self._pubchem_profile_for_molecule(molecule),
                self._dailymed_profile_for_molecule(molecule),
                self._chembl_profile_for_molecule(molecule),
            ]
            for source_idx, profile in enumerate([item for item in candidate_profiles if item], start=1):
                profiles.append(profile)
                if not self._profile_has_reportable_content(profile):
                    continue
                evidence.append(
                    {
                        "source": profile.get("source") or "database_profile",
                        "claim": self._database_profile_claim(profile),
                        "value": profile,
                        "citation_id": f"db_{idx}_{source_idx}",
                        "cid": profile.get("pubchem_cid"),
                        "source_path": profile.get("url"),
                        "evidence_direction": "neutral",
                        "evidence_type": "database_profile",
                    }
                )
        return profiles, evidence

    def _pubchem_profile_for_molecule(self, molecule: MoleculeRecord) -> Dict[str, Any]:
        metadata = dict(getattr(molecule, "metadata", {}) or {})
        cid = metadata.get("pubchem_cid")
        name = getattr(molecule, "name", None) or getattr(molecule, "original_input", None) or ""
        profile = self._resolver_profile(molecule)
        if not self._network_lookup_enabled():
            profile["retrieval_status"] = "network_lookup_disabled"
            return profile
        try:
            timeout = float(os.getenv("AGENTM_PUBCHEM_TIMEOUT", "8"))
        except ValueError:
            timeout = 8.0
        properties = self._pubchem_properties(cid=cid, name=name, timeout=timeout)
        if not properties and getattr(molecule, "canonical_smiles", None):
            properties = self._pubchem_properties(name=getattr(molecule, "canonical_smiles", ""), timeout=timeout)
        if not properties:
            profile["retrieval_status"] = "pubchem_lookup_empty"
            return profile
        prop = properties[0]
        cid = prop.get("CID") or cid
        synonyms = self._pubchem_synonyms(cid, timeout=timeout) if cid else []
        profile.update(
            {
                "source": "pubchem",
                "retrieval_status": "success",
                "pubchem_cid": cid,
                "title": prop.get("Title") or name,
                "iupac_name": prop.get("IUPACName"),
                "molecular_formula": prop.get("MolecularFormula"),
                "molecular_weight": prop.get("MolecularWeight"),
                "canonical_smiles": prop.get("CanonicalSMILES") or prop.get("SMILES") or prop.get("ConnectivitySMILES") or getattr(molecule, "canonical_smiles", None),
                "isomeric_smiles": prop.get("IsomericSMILES") or prop.get("SMILES"),
                "xlogp": prop.get("XLogP"),
                "tpsa": prop.get("TPSA"),
                "hbond_donor_count": prop.get("HBondDonorCount"),
                "hbond_acceptor_count": prop.get("HBondAcceptorCount"),
                "rotatable_bond_count": prop.get("RotatableBondCount"),
                "synonyms": synonyms[:8],
                "url": f"https://pubchem.ncbi.nlm.nih.gov/compound/{cid}" if cid else None,
            }
        )
        return profile

    def _resolver_profile(self, molecule: MoleculeRecord) -> Dict[str, Any]:
        metadata = dict(getattr(molecule, "metadata", {}) or {})
        cid = metadata.get("pubchem_cid")
        return {
            "source": "resolver_metadata",
            "retrieval_status": "resolver_only",
            "name": getattr(molecule, "name", None),
            "original_input": getattr(molecule, "original_input", None),
            "canonical_smiles": getattr(molecule, "canonical_smiles", None),
            "smiles": getattr(molecule, "smiles", None),
            "pubchem_cid": cid,
            "url": f"https://pubchem.ncbi.nlm.nih.gov/compound/{cid}" if cid else None,
            "resolver_source": getattr(molecule, "source", None),
        }

    def _dailymed_profile_for_molecule(self, molecule: MoleculeRecord) -> Dict[str, Any]:
        name = getattr(molecule, "name", None) or getattr(molecule, "original_input", None) or ""
        profile = {"source": "dailymed", "retrieval_status": "not_attempted", "query": name, "label_entries": []}
        if not name:
            profile["retrieval_status"] = "missing_query"
            return profile
        if not self._network_lookup_enabled():
            profile["retrieval_status"] = "network_lookup_disabled"
            return profile
        try:
            timeout = self._external_lookup_timeout()
            response = requests.get(
                "https://dailymed.nlm.nih.gov/dailymed/services/v2/spls.json",
                params={"drug_name": name, "pagesize": 5},
                timeout=timeout,
            )
            response.raise_for_status()
            raw_items = response.json().get("data", []) or []
            entries = []
            for item in raw_items[:5]:
                if not isinstance(item, dict):
                    continue
                setid = item.get("setid") or item.get("set_id")
                entries.append(
                    {
                        "setid": setid,
                        "title": item.get("title"),
                        "published_date": item.get("published_date"),
                        "labeler": item.get("labeler"),
                        "spl_version": item.get("spl_version"),
                        "url": f"https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid={setid}" if setid else None,
                    }
                )
            profile.update(
                {
                    "retrieval_status": "success" if entries else "no_results",
                    "label_entries": entries,
                    "url": entries[0].get("url") if entries else "https://dailymed.nlm.nih.gov/dailymed/",
                }
            )
            return profile
        except Exception as exc:
            profile.update({"retrieval_status": "failed", "error": str(exc)})
            return profile

    def _chembl_profile_for_molecule(self, molecule: MoleculeRecord) -> Dict[str, Any]:
        name = getattr(molecule, "name", None) or getattr(molecule, "original_input", None) or ""
        profile = {"source": "chembl", "retrieval_status": "not_attempted", "query": name}
        if not name:
            profile["retrieval_status"] = "missing_query"
            return profile
        if not self._network_lookup_enabled():
            profile["retrieval_status"] = "network_lookup_disabled"
            return profile
        try:
            timeout = self._external_lookup_timeout()
            search = requests.get(
                "https://www.ebi.ac.uk/chembl/api/data/molecule/search.json",
                params={"q": name, "limit": 3},
                timeout=timeout,
            )
            search.raise_for_status()
            molecules = search.json().get("molecules", []) or []
            if not molecules:
                profile["retrieval_status"] = "no_results"
                return profile
            chosen = molecules[0]
            chembl_id = chosen.get("molecule_chembl_id")
            mechanisms = self._chembl_mechanisms(chembl_id, timeout=timeout) if chembl_id else []
            indications = self._chembl_indications(chembl_id, timeout=timeout) if chembl_id else []
            profile.update(
                {
                    "retrieval_status": "success",
                    "chembl_id": chembl_id,
                    "pref_name": chosen.get("pref_name") or name,
                    "molecule_type": chosen.get("molecule_type"),
                    "max_phase": chosen.get("max_phase"),
                    "first_approval": chosen.get("first_approval"),
                    "therapeutic_flag": chosen.get("therapeutic_flag"),
                    "mechanisms": mechanisms[:5],
                    "indications": indications[:5],
                    "url": f"https://www.ebi.ac.uk/chembl/compound_report_card/{chembl_id}/" if chembl_id else None,
                }
            )
            return profile
        except Exception as exc:
            profile.update({"retrieval_status": "failed", "error": str(exc)})
            return profile

    def _chembl_mechanisms(self, chembl_id: str, *, timeout: float) -> List[Dict[str, Any]]:
        try:
            response = requests.get(
                "https://www.ebi.ac.uk/chembl/api/data/mechanism.json",
                params={"molecule_chembl_id": chembl_id, "limit": 5},
                timeout=timeout,
            )
            response.raise_for_status()
            rows = response.json().get("mechanisms", []) or []
            return [
                {
                    "mechanism_of_action": row.get("mechanism_of_action"),
                    "action_type": row.get("action_type"),
                    "target_name": row.get("target_pref_name"),
                    "target_chembl_id": row.get("target_chembl_id"),
                }
                for row in rows
                if isinstance(row, dict)
            ]
        except Exception:
            return []

    def _chembl_indications(self, chembl_id: str, *, timeout: float) -> List[Dict[str, Any]]:
        try:
            response = requests.get(
                "https://www.ebi.ac.uk/chembl/api/data/drug_indication.json",
                params={"molecule_chembl_id": chembl_id, "limit": 5},
                timeout=timeout,
            )
            response.raise_for_status()
            rows = response.json().get("drug_indications", []) or []
            return [
                {
                    "mesh_heading": row.get("mesh_heading"),
                    "efo_term": row.get("efo_term"),
                    "max_phase_for_ind": row.get("max_phase_for_ind"),
                }
                for row in rows
                if isinstance(row, dict)
            ]
        except Exception:
            return []

    def _pubchem_properties(self, *, cid: Any = None, name: str = "", timeout: float = 8.0) -> List[Dict[str, Any]]:
        fields = ",".join(
            [
                "Title",
                "IUPACName",
                "MolecularFormula",
                "MolecularWeight",
                "CanonicalSMILES",
                "IsomericSMILES",
                "XLogP",
                "TPSA",
                "HBondDonorCount",
                "HBondAcceptorCount",
                "RotatableBondCount",
                "SMILES",
                "ConnectivitySMILES",
            ]
        )
        if cid:
            url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{requests.utils.quote(str(cid), safe='')}/property/{fields}/JSON"
        elif name:
            namespace = "smiles" if any(ch in str(name) for ch in ["=", "#", "(", ")", "[", "]", "@", "\\", "/"]) else "name"
            url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/{namespace}/{requests.utils.quote(str(name), safe='')}/property/{fields}/JSON"
        else:
            return []
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        return (response.json().get("PropertyTable", {}) or {}).get("Properties", []) or []

    def _looks_like_smiles(self, value: Any) -> bool:
        text = str(value or "")
        return bool(text) and any(ch in text for ch in ["=", "#", "(", ")", "[", "]", "@", "\\", "/"]) and bool(re.search(r"[A-Za-z]", text))

    def _pubchem_synonyms(self, cid: Any, *, timeout: float = 8.0) -> List[str]:
        try:
            url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{requests.utils.quote(str(cid), safe='')}/synonyms/JSON"
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            information = (response.json().get("InformationList", {}) or {}).get("Information", []) or []
            synonyms = information[0].get("Synonym", []) if information else []
            return [str(item) for item in synonyms if item][:12]
        except Exception:
            return []

    def _database_profile_claim(self, profile: Dict[str, Any]) -> str:
        source = profile.get("source")
        if source == "dailymed":
            entries = [item for item in profile.get("label_entries", []) or [] if isinstance(item, dict)]
            titles = [str(item.get("title")) for item in entries if item.get("title")]
            if titles:
                return "DailyMed label entries: " + "; ".join(titles[:3]) + "."
            return "DailyMed profile lookup completed without label-entry details."
        if source == "chembl":
            name = profile.get("pref_name") or profile.get("query") or "molecule"
            parts = [f"ChEMBL profile for {name}"]
            if profile.get("chembl_id"):
                parts.append(f"ChEMBL ID {profile.get('chembl_id')}")
            mechanisms = [item for item in profile.get("mechanisms", []) or [] if isinstance(item, dict)]
            if mechanisms:
                labels = []
                for item in mechanisms[:3]:
                    label = item.get("mechanism_of_action") or item.get("target_name") or item.get("action_type")
                    if label:
                        labels.append(str(label))
                if labels:
                    parts.append("mechanisms " + ", ".join(labels))
            indications = [item for item in profile.get("indications", []) or [] if isinstance(item, dict)]
            if indications:
                labels = [str(item.get("mesh_heading") or item.get("efo_term")) for item in indications[:3] if item.get("mesh_heading") or item.get("efo_term")]
                if labels:
                    parts.append("indications " + ", ".join(labels))
            return "; ".join(parts) + "."
        name = profile.get("title") or profile.get("name") or profile.get("original_input") or "molecule"
        parts = [f"Database profile for {name}"]
        if profile.get("pubchem_cid"):
            parts.append(f"PubChem CID {profile.get('pubchem_cid')}")
        if profile.get("molecular_formula"):
            parts.append(f"formula {profile.get('molecular_formula')}")
        if profile.get("molecular_weight"):
            parts.append(f"molecular weight {profile.get('molecular_weight')}")
        if profile.get("iupac_name"):
            parts.append(f"IUPAC {profile.get('iupac_name')}")
        if profile.get("synonyms"):
            parts.append("synonyms " + ", ".join(profile.get("synonyms", [])[:4]))
        return "; ".join(parts) + "."

    def _profile_has_reportable_content(self, profile: Dict[str, Any]) -> bool:
        source = profile.get("source")
        if source in {"pubchem", "resolver_metadata"}:
            return bool(profile.get("name") or profile.get("title") or profile.get("pubchem_cid") or profile.get("canonical_smiles"))
        if source == "dailymed":
            return profile.get("retrieval_status") == "success" and bool(profile.get("label_entries"))
        if source == "chembl":
            return profile.get("retrieval_status") == "success" and bool(profile.get("chembl_id"))
        return profile.get("retrieval_status") == "success"

    def _external_lookup_timeout(self) -> float:
        try:
            return float(os.getenv("AGENTM_PUBCHEM_TIMEOUT", "8"))
        except ValueError:
            return 8.0

    def _network_lookup_enabled(self) -> bool:
        return str(os.getenv("AGENTM_ENABLE_NETWORK_LOOKUP", "0")).strip().lower() in {"1", "true", "yes", "on"}

    def _pubchem_field_compat_lookup(self, query: str, original: MoleculeRecord) -> Optional[MoleculeRecord]:
        query = str(query or "").strip()
        if not query:
            return None
        try:
            timeout = float(os.getenv("AGENTM_PUBCHEM_TIMEOUT", "8"))
        except ValueError:
            timeout = 8.0
        try:
            url = (
                "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
                f"{requests.utils.quote(query, safe='')}/property/CanonicalSMILES,IsomericSMILES,SMILES,ConnectivitySMILES/JSON"
            )
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            props = (response.json().get("PropertyTable", {}) or {}).get("Properties", []) or []
        except Exception as exc:
            original.metadata.setdefault("resolution_trace", []).append(
                {"stage": "pubchem_field_compat", "status": "failed", "error": str(exc)}
            )
            return None
        for prop in props:
            for field in ["CanonicalSMILES", "IsomericSMILES", "SMILES", "ConnectivitySMILES"]:
                candidate = str(prop.get(field) or "").strip()
                if not candidate:
                    continue
                canonical, err = standardize_smiles(candidate)
                if err is None and canonical:
                    trace = list(original.metadata.get("resolution_trace", []) or [])
                    trace.append(
                        {
                            "stage": "pubchem_field_compat",
                            "status": "success",
                            "field": field,
                            "cid": prop.get("CID"),
                        }
                    )
                    return MoleculeRecord(
                        original_input=query,
                        smiles=candidate,
                        canonical_smiles=canonical,
                        name=query,
                        source="pubchem_field_compat",
                        is_valid=True,
                        errors=[],
                        metadata={
                            **dict(original.metadata or {}),
                            "resolver": "DrugResolverAgent+AgentMPubChemCompat",
                            "resolution_trace": trace,
                            "pubchem_cid": prop.get("CID"),
                        },
                    )
        original.metadata.setdefault("resolution_trace", []).append(
            {"stage": "pubchem_field_compat", "status": "failed", "reason": "no_valid_smiles_field"}
        )
        return None

    def _normalize_request_for_parsing(self, request: str) -> str:
        text = str(request or "")
        for alias, canonical in sorted(DRUG_TEXT_ALIASES.items(), key=lambda item: len(item[0]), reverse=True):
            text = text.replace(alias, canonical)
        text = re.sub(r"\bASA\b", "aspirin", text, flags=re.IGNORECASE)
        return text

    def _local_seed_mentions_in_text(self, request: str) -> List[str]:
        """Return explicitly mentioned local identities in textual order.

        The generic input parser can mistake leading instruction words for molecule
        names in compound open-task prompts.  This scanner is deliberately limited
        to identities already present verbatim in the request and in the local
        identity table; it does not consult case manifests or expected answers.
        """

        text = str(request or "")
        matches: List[tuple[int, int, str]] = []
        for canonical, seed in LOCAL_COMPOUND_IDENTITY_SEEDS.items():
            aliases = {
                str(canonical).strip(),
                str(seed.get("title") or "").strip(),
                *[str(item).strip() for item in seed.get("synonyms", []) or []],
            }
            for alias in aliases:
                if not alias:
                    continue
                pattern = rf"(?<![A-Za-z0-9]){re.escape(alias)}(?![A-Za-z0-9])"
                for match in re.finditer(pattern, text, flags=re.IGNORECASE):
                    matches.append((match.start(), -len(alias), str(canonical)))

        selected: List[str] = []
        for _, _, canonical in sorted(matches):
            if canonical not in selected:
                selected.append(canonical)
        return selected

    def _normalize_molecule_mention(self, mention: Any) -> str:
        text = str(mention or "").strip(" .;:：，,。！？\"'“”‘’")
        for alias, canonical in sorted(DRUG_TEXT_ALIASES.items(), key=lambda item: len(item[0]), reverse=True):
            text = text.replace(alias, canonical)
        lowered = text.lower().strip()
        if lowered in DRUG_SPELLING_ALIASES:
            return DRUG_SPELLING_ALIASES[lowered]
        text = re.sub(r"\b(?:this|the)?\s*(?:drug|compound|molecule)\s*(?:怎么样|怎样|如何|好吗|安全吗)?\b", " ", text, flags=re.IGNORECASE)
        for phrase in [
            "这个药物安全吗",
            "这个药安全吗",
            "这个化合物安全吗",
            "这个分子安全吗",
            "这个药物怎样",
            "这个药物怎么样",
            "这个药物如何",
            "这个药怎么样",
            "这个药如何",
            "这个药怎样",
            "这个化合物怎样",
            "这个化合物怎么样",
            "这个化合物如何",
            "这个分子怎样",
            "这个分子怎么样",
            "这个分子如何",
            "药物怎样",
            "药物怎么样",
            "药物如何",
            "这个药物",
            "这个药",
            "这个化合物",
            "这个分子",
            "药物",
            "化合物",
            "分子",
            "怎么样",
            "怎样",
            "如何",
        ]:
            text = text.replace(phrase, " ")
        text = re.sub(r"\s+", " ", text).strip(" .;:：，,。！？\"'“”‘’")
        lowered = text.lower()
        if lowered in DRUG_SPELLING_ALIASES:
            return DRUG_SPELLING_ALIASES[lowered]
        return text

    def _is_general_drug_profile_request(
        self,
        request: str,
        *,
        parsed_task: str,
        parsed_dataset: str,
        explicit_task: bool,
        explicit_dataset: bool,
    ) -> bool:
        if explicit_task or explicit_dataset:
            return False
        lower = str(request or "").lower()
        if parsed_task == "ddi_prediction":
            return False
        if self._contains_specific_prediction_endpoint(lower):
            return False
        molecules = self._valid_candidate_mentions(self.input_agent.extract_molecule_mentions(request))
        has_entity_anchor = self._has_scientific_entity_anchor(request, molecules)
        has_domain_object = self._contains_domain_object_term(lower)
        if self._contains_general_profile_phrase(lower) and (has_entity_anchor or has_domain_object):
            return True
        if len(molecules) == 1 and parsed_dataset == "BBBP" and parsed_task == "property_prediction":
            return True
        return False

    def _preflight_scientific_intent(
        self,
        request: str,
        *,
        explicit_task: bool,
        explicit_dataset: bool,
    ) -> Dict[str, Any]:
        text = str(request or "").strip()
        lower = text.lower()
        if explicit_task or explicit_dataset:
            return {"status": "in_domain", "reason": "explicit_task_or_dataset"}
        if not text:
            return {
                "status": MISSING_ENTITY_TASK_TYPE,
                "reason": "empty_request",
                "suggestion": "Please provide a drug name, compound name, SMILES, ADMET/DDI endpoint, or reaction task.",
            }
        if is_reaction_request(text, None):
            return {"status": "in_domain", "reason": "reaction_intent_detected"}

        mentions = self._valid_candidate_mentions(self.input_agent.extract_molecule_mentions(text))
        has_entity_anchor = self._has_scientific_entity_anchor(text, mentions)
        endpoint_intent = self._contains_specific_prediction_endpoint(lower)
        profile_phrase = self._contains_general_profile_phrase(lower)
        domain_object = self._contains_domain_object_term(lower)
        domain_keyword = self._contains_drug_domain_keyword(lower)
        profile_intent = profile_phrase and (has_entity_anchor or domain_object or self._has_non_stopword_candidate(mentions))

        if endpoint_intent or profile_intent or domain_keyword:
            if has_entity_anchor or self._has_non_stopword_candidate(mentions):
                return {"status": "in_domain", "reason": "scientific_intent_detected"}
            return {
                "status": MISSING_ENTITY_TASK_TYPE,
                "reason": "scientific_intent_without_entity",
                "suggestion": "Please provide a concrete drug name, compound name, SMILES, or reaction SMILES.",
            }
        if has_entity_anchor:
            return {"status": "in_domain", "reason": "molecular_entity_detected"}
        return {
            "status": OUT_OF_DOMAIN_TASK_TYPE,
            "reason": "no_scientific_intent_or_entity_detected",
            "suggestion": "Please rewrite the input as a concrete ADMET, DDI, drug-profile, or reaction task.",
        }

    def _valid_candidate_mentions(self, mentions: Any) -> List[str]:
        candidates: List[str] = []
        for item in mentions or []:
            text = self._normalize_molecule_mention(item)
            if not text or self._is_intent_gate_stopword(text):
                continue
            if text not in candidates:
                candidates.append(text)
        return candidates

    def _has_scientific_entity_anchor(self, request: str, mentions: List[str]) -> bool:
        values = [request, *mentions]
        for value in values:
            text = str(value or "").strip()
            if not text or self._is_intent_gate_stopword(text):
                continue
            if self._looks_like_smiles(text) or self._local_identity_seed(query=text):
                return True
            if re.search(r"\b(?:CHEMBL\d+|CID\s*:?\s*\d+|\d{2,7}-\d{2}-\d)\b", text, flags=re.IGNORECASE):
                return True
        return False

    def _has_non_stopword_candidate(self, mentions: List[str]) -> bool:
        return any(not self._is_intent_gate_stopword(item) for item in mentions)

    def _is_intent_gate_stopword(self, value: Any) -> bool:
        text = re.sub(r"\s+", "", str(value or "").strip().lower())
        if not text:
            return True
        stopwords = {
            "今天",
            "昨天",
            "明天",
            "现在",
            "天气",
            "怎么样",
            "怎样",
            "如何",
            "好吗",
            "hello",
            "hi",
            "today",
            "tomorrow",
            "yesterday",
            "weather",
            "howareyou",
            "whatsup",
            "这个药",
            "这个药物",
            "这个化合物",
            "这个分子",
            "药",
            "药物",
            "化合物",
            "分子",
        }
        if text in stopwords:
            return True
        return bool(re.fullmatch(r"(?:今天|昨天|明天|现在)?(?:怎么样|怎样|如何|好吗)", text))

    def _contains_domain_object_term(self, lower_request: str) -> bool:
        object_terms = [
            "药",
            "药物",
            "药品",
            "化合物",
            "分子",
            "drug",
            "compound",
            "molecule",
            "smiles",
            "chemical",
            "medication",
        ]
        return any(term in lower_request for term in object_terms)

    def _contains_drug_domain_keyword(self, lower_request: str) -> bool:
        domain_terms = [
            "admet",
            "ddi",
            "toxicity",
            "toxic",
            "pharmacology",
            "pharmacokinetic",
            "metabolism",
            "safety",
            "side effect",
            "adverse",
            "reaction smiles",
            "反应",
            "毒性",
            "副作用",
            "不良反应",
            "代谢",
            "安全性",
            "用药",
        ]
        return any(term in lower_request for term in domain_terms) or self._contains_domain_object_term(lower_request)

    def _contains_specific_prediction_endpoint(self, lower_request: str) -> bool:
        endpoint_terms = [
            "bbbp",
            "bbb",
            "blood brain barrier",
            "blood-brain barrier",
            "brain penetration",
            "dili",
            "drug induced liver injury",
            "herg",
            "ames",
            "clintox",
            "tox21",
            "sider",
            "cyp1a2",
            "cyp2c9",
            "cyp2c19",
            "cyp2d6",
            "cyp3a4",
            "bace",
            "hiv",
            "esol",
            "freesolv",
            "lipophilicity",
            "solubility",
            "logp",
            "logd",
            "血脑屏障",
            "脑屏障",
            "入脑",
            "透过脑",
            "肝损伤",
            "肝毒性",
            "心脏毒性",
            "致突变",
            "溶解度",
            "脂溶性",
            "亲脂性",
            "相互作用",
            "联合用药",
            "合用",
            "联用",
        ]
        return any(term in lower_request for term in endpoint_terms)

    def _contains_general_profile_phrase(self, lower_request: str) -> bool:
        general_terms = [
            "怎么样",
            "怎样",
            "如何",
            "介绍",
            "概览",
            "概述",
            "总体",
            "一般",
            "general",
            "profile",
            "overview",
            "tell me about",
            "what about",
            "how is",
            "what is",
            "安全吗",
            "安全性",
            "用药建议",
            "这个药",
            "这个药物",
            "这个化合物",
            "这个分子",
        ]
        return any(term in lower_request for term in general_terms)

    def _reaction_dataset_name(self, task_type: str) -> str:
        if task_type == "reaction_classification":
            return "schneider"
        if task_type == "reaction_yield_prediction":
            return "buchwald_hartwig_kg"
        if task_type == "reaction_condition_retrieval":
            return "orderly_condition"
        return "reaction"

    def _task_graph_for(self, context: Dict[str, Any]) -> Dict[str, Any]:
        def finalize(nodes: List[Dict[str, Any]]) -> Dict[str, Any]:
            metadata = context.get("metadata", {}) or {}
            reflection_enabled = bool(metadata.get("enable_reflection", True)) and int(
                metadata.get("max_reflection_rounds", 1) or 0
            ) > 0
            if not reflection_enabled:
                nodes = [node for node in nodes if node.get("agent") != "ReflectionAgent"]
            edges = [[nodes[i]["id"], nodes[i + 1]["id"]] for i in range(len(nodes) - 1)]
            return {"status": "planned", "mode": metadata.get("mode"), "nodes": nodes, "edges": edges}

        if context.get("task_type") in NON_SCIENCE_TASK_TYPES:
            nodes = [
                {"id": "route_task", "agent": "MoleculeAgent"},
                {"id": "build_runtime_report", "agent": "ReporterAgent"},
                {"id": "verify_schema", "agent": "VerifierAgent"},
                {"id": "report", "agent": "ReporterAgent"},
            ]
            return finalize(nodes)
        if is_reaction_task(context.get("task_type")):
            nodes = [
                {"id": "route_task", "agent": "MoleculeAgent"},
                {"id": "parse_reaction", "agent": "ReactionAgent"},
                {"id": "retrieve_mit_template_evidence", "agent": "RetrieveAgent"},
                {"id": "retrieve_orderly_condition_evidence", "agent": "RetrieveAgent"},
                {"id": "classify_or_predict", "agent": "ReactionAgent"},
                {"id": "build_reaction_profile", "agent": "ReactionAgent"},
                {"id": "reason", "agent": "ReasoningAgent"},
                {"id": "verify_prediction", "agent": "VerifierAgent"},
                {"id": "verify_citations", "agent": "VerifierAgent"},
                {"id": "verify_schema", "agent": "VerifierAgent"},
                {"id": "reflect", "agent": "ReflectionAgent"},
                {"id": "report", "agent": "ReporterAgent"},
            ]
            return finalize(nodes)
        if context.get("task_type") == GENERAL_TASK_TYPE:
            nodes = [
                {"id": "route_task", "agent": "MoleculeAgent"},
                {"id": "resolve_molecule", "agent": "MoleculeAgent"},
                {"id": "retrieve_database_profile", "agent": "RetrieveAgent"},
                {"id": "build_features", "agent": "GaugeAgent"},
                {"id": "retrieve_optional_rag", "agent": "RetrieveAgent"},
                {"id": "reason", "agent": "ReasoningAgent"},
                {"id": "verify_citations", "agent": "VerifierAgent"},
                {"id": "verify_schema", "agent": "VerifierAgent"},
                {"id": "reflect", "agent": "ReflectionAgent"},
                {"id": "report", "agent": "ReporterAgent"},
            ]
            return finalize(nodes)
        is_ddi = context.get("task_type") == "ddi_prediction"
        resolve = "resolve_ddi_pair" if is_ddi else "resolve_molecule"
        predict = "predict_ddi" if is_ddi else "predict_single"
        nodes = [
            {"id": "route_task", "agent": "MoleculeAgent"},
            {"id": resolve, "agent": "MoleculeAgent"},
            {"id": "discover_rules", "agent": "GaugeAgent"},
            {"id": "build_features", "agent": "GaugeAgent"},
            {"id": predict, "agent": "GaugeAgent"},
            {"id": "retrieve_evidence", "agent": "RetrieveAgent"},
            {"id": "reason", "agent": "ReasoningAgent"},
            {"id": "verify", "agent": "VerifierAgent"},
            {"id": "reflect", "agent": "ReflectionAgent"},
            {"id": "report", "agent": "ReporterAgent"},
        ]
        return finalize(nodes)

    def _rag_query(self, context: Dict[str, Any]) -> str:
        names = " ".join(self._molecule_names(context))
        if context.get("task_type") == GENERAL_TASK_TYPE:
            return f"{names} drug profile pharmacology indication safety adverse effects contraindication metabolism evidence citation"
        prediction = self._active_prediction(context)
        label = prediction.get("label", "") if isinstance(prediction, Mapping) else getattr(prediction, "label", "")
        dataset = context.get("dataset_name", "")
        return f"{dataset} {context.get('task_type')} {names} prediction label {label} evidence citation"

    def _reflection_query(self, context: Dict[str, Any]) -> str:
        reflection = context.get("reflection", {}) or {}
        names = " ".join(self._molecule_names(context))
        unsupported = reflection.get("next_input", {}).get("unsupported_claims", 0)
        return (
            f"{context.get('dataset_name')} {context.get('task_type')} {names} "
            f"refine evidence after verifier unsupported_claims {unsupported} "
            "mechanism clinical citation safety risk"
        )

    def _retrieval_metadata(self, context: Dict[str, Any]) -> Dict[str, Any]:
        task_type = context.get("task_type")
        endpoint = infer_endpoint_from_context(context)
        entity_terms = entity_terms_from_context(context)
        if task_type == GENERAL_TASK_TYPE:
            return {
                "request": context.get("request"),
                "profile_type": GENERAL_TASK_TYPE,
                "drug_names": self._molecule_names(context),
                "entity_terms": entity_terms,
            }
        es_task_type = "admet_prediction" if task_type == "property_prediction" else task_type
        metadata = {
            "task_type": es_task_type,
            "dataset_name": context.get("dataset_name"),
            "endpoint": endpoint,
            "endpoint_terms": endpoint_terms(endpoint),
            "request": context.get("request"),
            "drug_names": self._molecule_names(context),
            "entity_terms": entity_terms,
        }
        entity_groups = self._molecule_entity_groups(context)
        if endpoint == "ddi" and len(entity_groups) >= 2:
            metadata["entity_groups"] = entity_groups[:2]
        return metadata

    def _molecule_names(self, context: Dict[str, Any]) -> List[str]:
        names = []
        for molecule in context.get("molecules", []) or []:
            if isinstance(molecule, Mapping):
                names.append(str(molecule.get("name") or molecule.get("original_input") or molecule.get("canonical_smiles") or molecule))
            else:
                names.append(getattr(molecule, "name", None) or getattr(molecule, "original_input", None) or str(molecule))
        return names

    def _molecule_entity_groups(self, context: Dict[str, Any]) -> List[List[str]]:
        groups: List[List[str]] = []
        for molecule in context.get("molecules", []) or []:
            values: List[Any] = []
            if isinstance(molecule, Mapping):
                values.extend(molecule.get(key) for key in ["name", "original_input", "canonical_smiles", "smiles"])
                values.extend(molecule.get("synonyms", []) or [])
                metadata = molecule.get("metadata") if isinstance(molecule.get("metadata"), Mapping) else {}
            else:
                values.extend(getattr(molecule, key, None) for key in ["name", "original_input", "canonical_smiles", "smiles"])
                values.extend(getattr(molecule, "synonyms", []) or [])
                metadata = getattr(molecule, "metadata", {}) or {}
            if isinstance(metadata, Mapping):
                values.extend(
                    metadata.get(key)
                    for key in [
                        "pubchem_cid",
                        "pubchem_title",
                        "pubchem_iupac_name",
                        "chembl_id",
                        "chembl_pref_name",
                    ]
                )
                values.extend(metadata.get("synonyms", []) or [])
                values.extend(metadata.get("pubchem_synonyms", []) or [])
            group = unique_strings(values, lower=True, limit=40)
            if group:
                groups.append(group)
        return groups

    def _normalize_evidence(self, evidence: Any, *, source_default: str) -> List[Dict[str, Any]]:
        items = to_jsonable(list(evidence or []))
        normalized: List[Dict[str, Any]] = []
        for idx, item in enumerate(items, start=1):
            if not isinstance(item, dict):
                item = {"claim": str(item)}
            item.setdefault("source", source_default)
            item.setdefault("claim", item.get("snippet") or item.get("mechanism") or item.get("interaction_type") or str(item)[:240])
            item.setdefault("citation_id", f"cite_{idx}")
            normalized.append(item)
        return normalized

    def _append_evidence(self, context: Dict[str, Any], evidence: List[Dict[str, Any]]) -> int:
        active = self._active_open_subtask(context)
        subtask_id = str(active.get("subtask_id") or "")
        if subtask_id:
            for item in evidence:
                item["open_task_subtask_id"] = subtask_id
                item["open_task_kind"] = active.get("kind")
                item["open_task_dataset_name"] = active.get("dataset_name")
                item["open_task_molecules"] = list(active.get("molecules", []) or [])
                item["open_task_evidence_scope"] = active.get("evidence_scope")
                citation_id = str(item.get("citation_id") or "")
                if citation_id and not citation_id.startswith(f"{subtask_id}::"):
                    item["citation_id"] = f"{subtask_id}::{citation_id}"
        evidence = self._retained_evidence(context, evidence)
        existing = {
            (str(item.get("open_task_subtask_id") or ""), _evidence_key(item))
            for item in context.get("evidence", []) or []
            if isinstance(item, dict)
        }
        added = 0
        bucket = context.setdefault("evidence", [])
        for item in evidence:
            key = (str(item.get("open_task_subtask_id") or ""), _evidence_key(item))
            if key in existing:
                continue
            existing.add(key)
            bucket.append(item)
            added += 1
        return added

    def _retained_evidence(self, context: Dict[str, Any], evidence: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not evidence:
            return []
        if context.get("task_type") == GENERAL_TASK_TYPE:
            return evidence
        scored = [self._score_evidence_for_retention(context, item) for item in evidence if isinstance(item, dict)]
        retained = [item for item in scored if item.get("retained")]
        retained.sort(key=lambda item: (float(item.get("retention_score", 0.0)), str(item.get("citation_id") or "")), reverse=True)
        retained = retained[: max(self.rag_top_k, self._min_evidence(context))]
        audit = {
            "task_type": context.get("task_type"),
            "dataset_name": context.get("dataset_name"),
            "raw_count": len(evidence),
            "retained_count": len(retained),
            "dropped_count": max(0, len(evidence) - len(retained)),
            "threshold": self._retention_threshold(context),
            "retained_citation_ids": [item.get("citation_id") for item in retained],
            "dropped": [
                {
                    "citation_id": item.get("citation_id"),
                    "source": item.get("source"),
                    "score": item.get("retention_score"),
                    "reasons": item.get("retention_reasons", []),
                    "claim": str(item.get("claim") or "")[:160],
                }
                for item in scored
                if not item.get("retained")
            ][:8],
        }
        context.setdefault("artifacts", {}).setdefault("retained_evidence_filter", []).append(audit)
        if audit["dropped_count"]:
            self._append_warning(context, f"retained_evidence_filtered:{audit['dropped_count']}/{audit['raw_count']}")
        return retained

    def _score_evidence_for_retention(self, context: Dict[str, Any], item: Dict[str, Any]) -> Dict[str, Any]:
        value = item.get("value") if isinstance(item.get("value"), Mapping) else {}
        text = " ".join(
            str(part or "")
            for part in [
                item.get("claim"),
                item.get("snippet"),
                item.get("mechanism"),
                item.get("interaction_type"),
                value.get("snippet"),
                value.get("title"),
                value.get("text"),
            ]
        )
        source = str(item.get("source") or value.get("source") or "")
        citation_ok = bool(
            item.get("chunk_id")
            or item.get("source_path")
            or item.get("pmid")
            or item.get("url")
            or value.get("chunk_id")
            or value.get("source_path")
            or value.get("pmid")
            or value.get("url")
            or source in {"pubchem", "dailymed", "chembl", "pubmed", "database_profile", "resolver_metadata"}
        )
        compound_ok, compound_matches = self._evidence_matches_compounds(context, item, value, text)
        endpoint_ok, endpoint_matches = self._evidence_matches_endpoint(context, item, value, text)
        scope = str(item.get("evidence_scope") or value.get("evidence_scope") or "")
        score = 0.0
        reasons: List[str] = []
        if citation_ok:
            score += 1.0
            reasons.append("citation")
        if compound_ok:
            score += 3.0
            reasons.append("compound")
        if endpoint_ok:
            score += 3.0
            reasons.append("endpoint")
        if scope in {"compound_endpoint", "ddi_pair"}:
            score += 1.0
            reasons.append(scope)
        if source in {"dailymed", "pubmed", "elasticsearch", "local_rag"}:
            score += 0.25
        threshold = self._retention_threshold(context)
        retained = bool(citation_ok and compound_ok and endpoint_ok and score >= threshold)
        annotated = dict(item)
        annotated["retention_score"] = round(score, 3)
        annotated["retention_reasons"] = reasons
        annotated["retention_matches"] = {"compound": compound_matches[:8], "endpoint": endpoint_matches[:8]}
        annotated["retained"] = retained
        return annotated

    def _retention_threshold(self, context: Dict[str, Any]) -> float:
        if self._real_rag_required(context):
            return 6.0
        return 5.0

    def _evidence_matches_compounds(
        self,
        context: Dict[str, Any],
        item: Mapping[str, Any],
        value: Mapping[str, Any],
        text: str,
    ) -> tuple[bool, List[str]]:
        groups = self._molecule_entity_groups(context)
        if not groups:
            return True, []
        field_terms = set()
        for mapping in [item, value]:
            for key in [
                "entity_terms",
                "text_entity_terms",
                "drug_names",
                "compound_names",
                "compound_synonyms",
                "compound_cids",
                "chembl_ids",
                "canonical_smiles",
                "pair_drug_names",
                "pair_drug_ids",
                "pair_drug_smiles",
                "drug_a",
                "drug_b",
            ]:
                raw = mapping.get(key) if isinstance(mapping, Mapping) else None
                if isinstance(raw, str):
                    field_terms.add(raw.lower())
                else:
                    field_terms.update(str(part).lower() for part in (raw or []) if part)
        normalized_text = self._normalized_match_text(text)
        matched: List[str] = []
        required_groups = groups[:2] if context.get("task_type") == "ddi_prediction" and len(groups) >= 2 else groups[:1]
        for group in required_groups:
            group_match = ""
            for term in group:
                if self._term_matches(term, normalized_text, field_terms):
                    group_match = term
                    break
            if not group_match:
                return False, matched
            matched.append(group_match)
        return True, matched

    def _evidence_matches_endpoint(
        self,
        context: Dict[str, Any],
        item: Mapping[str, Any],
        value: Mapping[str, Any],
        text: str,
    ) -> tuple[bool, List[str]]:
        endpoint = infer_endpoint_from_context(context)
        terms = endpoint_terms(endpoint)
        metadata_endpoint = compact_term(item.get("endpoint") or value.get("endpoint") or "")
        metadata_terms = set()
        for mapping in [item, value]:
            raw = mapping.get("endpoint_terms") if isinstance(mapping, Mapping) else None
            if isinstance(raw, str):
                metadata_terms.add(raw.lower())
            else:
                metadata_terms.update(str(part).lower() for part in (raw or []) if part)
        metadata_specific = bool(item.get("endpoint_specific") or value.get("endpoint_specific"))
        if compact_term(endpoint) == "ddi" or context.get("task_type") == "ddi_prediction":
            terms = unique_strings(
                [
                    *terms,
                    "drug-drug interaction",
                    "drug interaction",
                    "interaction",
                    "coadministration",
                    "cyp2c19",
                    "cyp",
                    "active metabolite",
                    "antiplatelet",
                ],
                lower=True,
            )
        normalized_text = self._normalized_match_text(text)
        matches = [
            term
            for term in terms
            if self._term_matches(term, normalized_text, metadata_terms)
            or (metadata_specific and compact_term(term) == metadata_endpoint)
        ]
        return bool(matches or metadata_specific), matches

    def _normalized_match_text(self, text: str) -> Dict[str, str]:
        lower = str(text or "").lower()
        normalized = re.sub(r"[^a-z0-9]+", " ", lower)
        compact = re.sub(r"[^a-z0-9]+", "", lower)
        return {"normalized": normalized, "compact": compact}

    def _term_matches(self, term: Any, normalized_text: Mapping[str, str], field_terms: set[str]) -> bool:
        raw = str(term or "").strip().lower()
        if not raw:
            return False
        norm = re.sub(r"[^a-z0-9]+", " ", raw).strip()
        compact = re.sub(r"[^a-z0-9]+", "", raw)
        if raw in field_terms or norm in field_terms or compact in {re.sub(r"[^a-z0-9]+", "", item) for item in field_terms}:
            return True
        if len(norm) >= 4 and norm in normalized_text.get("normalized", ""):
            return True
        if len(compact) >= 6 and compact in normalized_text.get("compact", ""):
            return True
        return False

    def _current_evidence_count(self, context: Dict[str, Any]) -> int:
        return len({_evidence_key(item) for item in self._active_subtask_evidence(context)})

    def _current_grounded_evidence_count(self, context: Dict[str, Any]) -> int:
        grounded_keys: set[str] = set()
        for item in self._active_subtask_evidence(context):
            if not isinstance(item, dict):
                continue
            value = item.get("value") if isinstance(item.get("value"), Mapping) else {}
            text = " ".join(
                str(part or "")
                for part in [
                    item.get("claim"),
                    item.get("snippet"),
                    item.get("mechanism"),
                    item.get("interaction_type"),
                    value.get("snippet"),
                    value.get("title"),
                    value.get("text"),
                ]
            )
            compound_ok, _ = self._evidence_matches_compounds(context, item, value, text)
            endpoint_ok, _ = self._evidence_matches_endpoint(context, item, value, text)
            citation_ok = bool(
                item.get("chunk_id")
                or item.get("source_path")
                or item.get("pmid")
                or item.get("url")
                or value.get("chunk_id")
                or value.get("source_path")
                or value.get("pmid")
                or value.get("url")
                or str(item.get("source") or value.get("source") or "") in {"pubchem", "dailymed", "chembl", "pubmed", "database_profile", "resolver_metadata"}
            )
            if citation_ok and compound_ok and endpoint_ok:
                grounded_keys.add(_evidence_key(item))
        return len(grounded_keys)

    def _append_warning(self, context: Dict[str, Any], warning: str) -> None:
        warnings = context.setdefault("warnings", [])
        if warning not in warnings:
            warnings.append(warning)

    def _previous_run_summary(self, previous: Dict[str, Any]) -> Dict[str, Any]:
        if not previous:
            return {}
        prediction = (previous.get("predictions") or [{}])[0]
        verifier = previous.get("verifier_report", {}) or {}
        return {
            "request": previous.get("request"),
            "task_type": previous.get("task_type"),
            "prediction": {
                "label": prediction.get("label"),
                "score": prediction.get("score"),
                "model_type": prediction.get("model_type"),
            },
            "verifier_status": (verifier.get("summary") or {}).get("status"),
            "n_evidence": len(previous.get("evidence", []) or []),
        }

    def _fallback(self, context: Dict[str, Any], stage: str, message: str) -> None:
        entry = {"stage": stage, "message": message}
        context.setdefault("fallbacks", []).append(entry)
        context.setdefault("warnings", []).append(message)

    def _fallbacks_enabled(self, context: Dict[str, Any]) -> bool:
        return bool(context.get("metadata", {}).get("enable_fallbacks", False))

    def _is_zero_shot_prediction(self, prediction: Any) -> bool:
        return str(getattr(prediction, "model_type", "")) == ZERO_SHOT_MODEL_TYPE

    def _selected_model_path(self, context: Dict[str, Any]) -> Optional[str]:
        return (context.get("resource_resolution", {}).get("model", {}) or {}).get("selected_path")

    def _real_model_required(self, context: Dict[str, Any]) -> bool:
        return bool(context.get("metadata", {}).get("require_real_model", self.require_real_model))

    def _real_rag_required(self, context: Dict[str, Any]) -> bool:
        return bool(context.get("metadata", {}).get("require_real_rag", self.require_real_rag))

    def _min_evidence(self, context: Dict[str, Any]) -> int:
        return int(context.get("metadata", {}).get("min_evidence", self.min_evidence) or 0)
