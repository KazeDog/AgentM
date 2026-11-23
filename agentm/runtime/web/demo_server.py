from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from datetime import datetime
import hashlib
import importlib.util
import json
import mimetypes
import os
from pathlib import Path
import re
import threading
import time
import traceback
from typing import Any, Dict, Iterator, List, Optional
from urllib.parse import quote, unquote
from uuid import uuid4

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel, ConfigDict

from agentm.runtime import AgentMMultiAgentController
from agentm.runtime.compat import RESOURCE_ROOT, RUNTIME_ROOT, ensure_import_paths, to_jsonable
from agentm.runtime.prediction_semantics import is_multiclass_ddi_prediction, prediction_meaning as scientific_prediction_meaning

ensure_import_paths()

from agentm.defaults import (
    DEFAULT_DDI_RAG_CORPUS_DIR,
    DEFAULT_LLM_PROFILE,
    DEFAULT_RAG_BACKEND,
    DEFAULT_RAG_CORPUS_DIR,
    DEFAULT_RAG_EMBEDDING_MODEL_PATH,
    DEFAULT_RAG_TOP_K,
)


WEB_PROGRESS_PRE_AGENT_STEPS = 2
WEB_PROGRESS_DEFAULT_TOTAL_STEPS = 17
RELEASE_INDEX_NAME = "agentm_evidence_paper_qc_dense"
RELEASE_EVIDENCE_DIR_NAME = "paper_evidence_qc"
RELEASE_SMOKE_DIR_NAME = "paper_evidence_qc"
RELEASE_ACCEPTANCE_SUMMARY_NAME = "es_acceptance_summary.json"
RELEASE_QC_READOUT_NAME = "evidence_qc_readout.json"
RELEASE_STRUCTURED_JSONL_NAME = "structured_evidence_qc.jsonl"
RELEASE_EXPANDED_ACCEPTANCE_STATUS = "50/50 paper-QC strict"


class RunPayload(BaseModel):
    model_config = ConfigDict(extra="allow")

    request: str
    run_mode: str = "custom"
    task_type: Optional[str] = None
    dataset_name: Optional[str] = None
    feature_set: Optional[str] = None
    model_path: Optional[str] = None
    mode: Optional[str] = None
    retrieval_backend: Optional[str] = None
    es_url: Optional[str] = field(default_factory=lambda: os.getenv("AGENTM_ES_URL", "http://127.0.0.1:9200"))
    auto_model: Optional[bool] = None
    require_real_model: Optional[bool] = None
    require_real_rag: Optional[bool] = None
    require_llm_orchestrator: Optional[bool] = None
    require_llm_reasoning: Optional[bool] = None
    use_llm_orchestrator: Optional[bool] = None
    use_llm_reasoning: Optional[bool] = None
    enable_fallbacks: Optional[bool] = None
    enable_reflection: Optional[bool] = None
    max_rounds: Optional[int] = None
    min_evidence: Optional[int] = None
    use_mllm_visual_semantic: Optional[bool] = None
    use_external_evidence: Optional[bool] = None
    use_narrative_llm: Optional[bool] = None
    narrative_language: Optional[str] = None
    use_rag_evidence: Optional[bool] = None
    allow_network_lookup: Optional[bool] = None

    def payload_dict(self) -> Dict[str, Any]:
        return self.model_dump()


@dataclass
class WebDemoConfig:
    host: str = "127.0.0.1"
    port: int = 7864
    project_root: Path = RUNTIME_ROOT
    resource_root: Path = RESOURCE_ROOT
    output_dir: Path = RUNTIME_ROOT / "outputs" / "web_demo"
    feature_set: str = "rdkit_only"
    model_path: Optional[str] = None
    auto_model: bool = True
    model_registry_path: Optional[str] = None
    require_real_model: bool = False
    require_real_rag: bool = False
    require_llm_orchestrator: bool = False
    require_llm_reasoning: bool = False
    min_evidence: int = 1
    mode: str = "fast"
    retrieval_backend: str = "local"
    es_url: Optional[str] = None
    max_rounds: int = 16
    temperature: float = 0.1
    llm_profile: Optional[str] = DEFAULT_LLM_PROFILE
    llm_timeout: int = 60
    use_llm_orchestrator: bool = True
    use_llm_reasoning: bool = True
    enable_fallbacks: bool = True
    enable_reflection: bool = True
    max_reflection_rounds: int = 1
    previous_run_path: Optional[str] = None
    use_smiles_encoder: bool = False
    use_mllm_visual_semantic: bool = False
    use_external_evidence: bool = False
    use_rag_evidence: bool = True
    rag_corpus_dirs: List[str] = field(default_factory=lambda: [_resource_path(DEFAULT_RAG_CORPUS_DIR)])
    rag_top_k: int = DEFAULT_RAG_TOP_K
    rag_backend: str = DEFAULT_RAG_BACKEND
    rag_embedding_model_name: str = field(default_factory=lambda: _resource_path(DEFAULT_RAG_EMBEDDING_MODEL_PATH))
    ddi_rag_corpus_dir: str = field(default_factory=lambda: _resource_path(DEFAULT_DDI_RAG_CORPUS_DIR))
    allow_network_lookup: bool = True
    pubchem_timeout: float = 5.0


@dataclass
class WebRun:
    run_id: str
    request: str
    created_at: str
    run_dir: Path
    run_mode: str = "custom"
    status: str = "queued"
    updated_at: str = ""
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    elapsed_sec: Optional[float] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    traceback_path: Optional[str] = None
    error_artifacts: List[Dict[str, Any]] = field(default_factory=list)
    events: List[Dict[str, Any]] = field(default_factory=list)
    step_events: List[Dict[str, Any]] = field(default_factory=list)
    current_step: Optional[Dict[str, Any]] = None
    progress: Dict[str, Any] = field(default_factory=dict)

    def public_status(self) -> Dict[str, Any]:
        failed_step = None
        if self.status == "failed" and self.current_step:
            failed_step = {
                "step_id": self.current_step.get("step_id"),
                "tool_name": self.current_step.get("tool_name"),
                "status": self.current_step.get("status"),
                "message": self.current_step.get("message") or self.current_step.get("error"),
            }
        payload = {
            "run_id": self.run_id,
            "request": self.request,
            "run_mode": self.run_mode,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at or self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "elapsed_sec": self.elapsed_sec,
            "run_dir": str(self.run_dir),
            "error": self.error,
            "traceback_path": self.traceback_path,
            "error_artifacts": self.error_artifacts,
            "failed_step": failed_step,
            "fallback_used": any(
                "fallback" in str(event.get("status") or "") or bool(event.get("fallback_tool"))
                for event in self.step_events
            ),
            "events": self.events[-20:],
            "step_events": self.step_events[-100:],
            "current_step": self.current_step,
            "progress": self.progress,
        }
        if self.result:
            payload["summary"] = _run_public_summary(self.result)
        return payload


class WebRunManager:
    def __init__(self, *, config: WebDemoConfig, run_lock: Optional[threading.Lock] = None):
        self.config = config
        self.run_lock = run_lock or threading.Lock()
        self._runs: Dict[str, WebRun] = {}
        self._lock = threading.Lock()

    def start(self, payload: Dict[str, Any]) -> WebRun:
        request = str(payload.get("request") or "").strip()
        if not request:
            raise ValueError("Please provide a molecule, drug name, pair, SMILES, or natural-language request.")
        run_dir = _next_run_dir(self.config.output_dir)
        run_mode = _normalize_run_mode(payload.get("run_mode"), default="custom")
        run = WebRun(
            run_id=run_dir.name,
            request=request,
            run_mode=run_mode,
            created_at=_now_iso(),
            updated_at=_now_iso(),
            run_dir=run_dir,
            events=[_event("queued", f"Run queued ({run_mode} mode).")],
            progress={"run_mode": run_mode, "completed_steps": 0, "running_steps": 0, "failed_steps": 0, "skipped_steps": 0},
        )
        with self._lock:
            self._runs[run.run_id] = run
        thread = threading.Thread(target=self._execute, args=(run.run_id, dict(payload)), daemon=True)
        thread.start()
        return run

    def list_runs(self) -> List[Dict[str, Any]]:
        with self._lock:
            runs = sorted(self._runs.values(), key=lambda item: item.created_at, reverse=True)
            return [run.public_status() for run in runs]

    def get(self, run_id: str) -> Optional[WebRun]:
        with self._lock:
            return self._runs.get(run_id)

    def _execute(self, run_id: str, payload: Dict[str, Any]) -> None:
        started = time.time()
        self._update(run_id, status="running", started_at=_now_iso(), events=[_event("running", "AgentMMultiAgentController started.")])
        run = self.get(run_id)
        if run is None:
            return
        try:
            result = run_agent_request(
                payload,
                self.config,
                self.run_lock,
                run_dir=run.run_dir,
                progress_callback=lambda event: self._record_agent_progress(run_id, event),
            )
            self._update(
                run_id,
                status="completed",
                result=result,
                finished_at=_now_iso(),
                elapsed_sec=time.time() - started,
                events=[_event("completed", "Run completed.")],
            )
        except Exception as exc:
            trace = traceback.format_exc()
            trace_path = run.run_dir / "web_error_traceback.txt"
            trace_path.write_text(trace, encoding="utf-8")
            error_json_path = run.run_dir / "web_error.json"
            error_payload = {"status": "error", "error": str(exc), "traceback_path": str(trace_path), "web_error_path": str(error_json_path)}
            error_json_path.write_text(json.dumps(error_payload, indent=2, ensure_ascii=False), encoding="utf-8")
            error_artifacts = [
                {
                    "key": "web_error_path",
                    "label": "Web error JSON",
                    "path": str(error_json_path),
                    "url": artifact_url(str(error_json_path), self.config),
                },
                {
                    "key": "traceback_path",
                    "label": "Traceback",
                    "path": str(trace_path),
                    "url": artifact_url(str(trace_path), self.config),
                },
            ]
            self._update(
                run_id,
                status="failed",
                error=str(exc),
                traceback_path=str(trace_path),
                error_artifacts=error_artifacts,
                finished_at=_now_iso(),
                elapsed_sec=time.time() - started,
                events=[_event("failed", str(exc))],
            )

    def _record_agent_progress(self, run_id: str, event: Dict[str, Any]) -> None:
        progress_event = _compact_progress_event(event)
        general_event = _event(f"agent_{progress_event.get('status') or 'running'}", _progress_message(progress_event))
        general_event.update(
            {
                "step_id": progress_event.get("step_id"),
                "tool_name": progress_event.get("tool_name"),
                "duration_sec": progress_event.get("duration_sec"),
            }
        )
        with self._lock:
            run = self._runs.get(run_id)
            if run is None:
                return
            run.step_events.append(progress_event)
            run.events.append(general_event)
            run.current_step = progress_event
            run.progress = _summarize_step_progress(run.step_events, run.run_mode)
            run.updated_at = _now_iso()

    def _update(self, run_id: str, **updates: Any) -> None:
        with self._lock:
            run = self._runs.get(run_id)
            if run is None:
                return
            for key, value in updates.items():
                if key == "events":
                    run.events.extend(value or [])
                else:
                    setattr(run, key, value)
            run.updated_at = _now_iso()


def create_app(config: Optional[WebDemoConfig] = None) -> FastAPI:
    config = config or WebDemoConfig()
    _ensure_dir(config.output_dir)
    app = FastAPI(title="AgentM Web Demo", version="AgentM")
    app.state.config = config
    app.state.run_manager = WebRunManager(config=config)

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        return INDEX_HTML

    @app.get("/api/health")
    async def health() -> Dict[str, Any]:
        return {
            "status": "ok",
            "service": "AgentM web demo",
            "runtime": "agentm_guarded_multi_agent",
            "output_dir": str(_resolved_output_dir(config)),
            "resource_root": str(config.resource_root),
            "resource_root_exists": config.resource_root.exists(),
            "langgraph_available": importlib.util.find_spec("langgraph") is not None,
        }

    @app.get("/api/runs")
    async def list_runs() -> Dict[str, Any]:
        return {"status": "ok", "runs": app.state.run_manager.list_runs()}

    @app.post("/api/runs", status_code=status.HTTP_202_ACCEPTED)
    async def start_run(payload: RunPayload) -> Dict[str, Any]:
        try:
            run = app.state.run_manager.start(payload.payload_dict())
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"status": "queued", "run": run.public_status()}

    @app.post("/api/run", status_code=status.HTTP_202_ACCEPTED)
    async def start_run_compat(payload: RunPayload) -> Dict[str, Any]:
        return await start_run(payload)

    @app.get("/api/runs/{run_id}")
    async def run_status(run_id: str) -> Dict[str, Any]:
        run = app.state.run_manager.get(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Run not found.")
        return {"status": "ok", "run": run.public_status()}

    @app.get("/api/runs/{run_id}/result")
    async def run_result(run_id: str) -> JSONResponse:
        run = app.state.run_manager.get(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Run not found.")
        if run.status != "completed" or run.result is None:
            return JSONResponse({"status": run.status, "run": run.public_status()}, status_code=status.HTTP_202_ACCEPTED)
        return JSONResponse(run.result)

    @app.get("/artifacts/{encoded_path:path}")
    async def artifact(encoded_path: str) -> FileResponse:
        candidate = _artifact_candidate(encoded_path, config)
        if not _is_allowed_artifact(candidate, config):
            raise HTTPException(status_code=403, detail="Artifact path is outside the demo output directory.")
        if not candidate.is_file():
            raise HTTPException(status_code=404, detail="Artifact not found.")
        media_type = mimetypes.guess_type(candidate.name)[0] or "application/octet-stream"
        return FileResponse(candidate, media_type=media_type, filename=candidate.name)

    return app


def run_agent_request(
    payload: Dict[str, Any],
    config: WebDemoConfig,
    run_lock: threading.Lock,
    run_dir: Optional[Path] = None,
    progress_callback: Optional[Any] = None,
) -> Dict[str, Any]:
    request = str(payload.get("request") or "").strip()
    if not request:
        raise ValueError("Please provide a molecule, drug name, pair, SMILES, or natural-language request.")

    run_mode = _normalize_run_mode(payload.get("run_mode"), default="custom")
    mode_defaults = _run_mode_defaults(run_mode)
    task_type = _empty_to_none(payload.get("task_type"))
    dataset_name = _empty_to_none(payload.get("dataset_name"))
    explicit_feature_set = _empty_to_none(payload.get("feature_set"))
    feature_set = explicit_feature_set or config.feature_set
    model_path = _empty_to_none(payload.get("model_path")) or config.model_path
    use_smiles_encoder = _bool(payload.get("use_smiles_encoder"), config.use_smiles_encoder)
    if mode_defaults:
        mode = str(mode_defaults["mode"])
        retrieval_backend = str(mode_defaults["retrieval_backend"])
        auto_model = bool(mode_defaults["auto_model"])
        require_real_model = bool(mode_defaults["require_real_model"])
        require_real_rag = bool(mode_defaults["require_real_rag"])
        require_llm_orchestrator = bool(mode_defaults["require_llm_orchestrator"])
        require_llm_reasoning = bool(mode_defaults["require_llm_reasoning"])
        use_llm_orchestrator = bool(mode_defaults["use_llm_orchestrator"])
        use_llm_reasoning = bool(mode_defaults["use_llm_reasoning"])
        enable_fallbacks = bool(mode_defaults["enable_fallbacks"])
        enable_reflection = bool(mode_defaults["enable_reflection"])
        min_evidence = int(mode_defaults["min_evidence"])
        use_mllm_visual_semantic = bool(mode_defaults["use_mllm_visual_semantic"])
        use_external_evidence = bool(mode_defaults["use_external_evidence"])
        use_rag_evidence = bool(mode_defaults["use_rag_evidence"])
        allow_network_lookup = bool(mode_defaults["allow_network_lookup"])
        if _bool(payload.get("use_mllm_visual_semantic"), False):
            use_mllm_visual_semantic = True
        if _bool(payload.get("use_external_evidence"), False):
            use_external_evidence = True
        if _bool(payload.get("use_rag_evidence"), False):
            use_rag_evidence = True
        if _bool(payload.get("use_narrative_llm"), False):
            use_llm_reasoning = True
        if _bool(payload.get("allow_network_lookup"), False):
            allow_network_lookup = True
    else:
        mode = str(payload.get("mode") or config.mode)
        retrieval_backend = str(payload.get("retrieval_backend") or config.retrieval_backend)
        auto_model = _bool(payload.get("auto_model"), config.auto_model)
        require_real_model = _bool(payload.get("require_real_model"), config.require_real_model)
        require_real_rag = _bool(payload.get("require_real_rag"), config.require_real_rag)
        require_llm_orchestrator = _bool(payload.get("require_llm_orchestrator"), config.require_llm_orchestrator)
        require_llm_reasoning = _bool(payload.get("require_llm_reasoning"), config.require_llm_reasoning)
        use_llm_orchestrator = _bool(payload.get("use_llm_orchestrator"), config.use_llm_orchestrator)
        use_llm_reasoning = _bool(payload.get("use_llm_reasoning"), config.use_llm_reasoning)
        enable_fallbacks = _bool(payload.get("enable_fallbacks"), config.enable_fallbacks)
        enable_reflection = _bool(payload.get("enable_reflection"), config.enable_reflection)
        min_evidence = int(payload.get("min_evidence") or config.min_evidence)
        use_mllm_visual_semantic = _bool(payload.get("use_mllm_visual_semantic"), config.use_mllm_visual_semantic)
        use_external_evidence = _bool(payload.get("use_external_evidence"), config.use_external_evidence)
        use_rag_evidence = _bool(payload.get("use_rag_evidence"), config.use_rag_evidence)
        allow_network_lookup = _bool(payload.get("allow_network_lookup"), config.allow_network_lookup)
        if _bool(payload.get("use_narrative_llm"), False):
            use_llm_reasoning = True
    if use_mllm_visual_semantic:
        use_smiles_encoder = True
    if use_mllm_visual_semantic and not explicit_feature_set and feature_set == "rdkit_only":
        feature_set = "all_features"
    if not use_rag_evidence:
        min_evidence = 0

    run_dir = run_dir or _next_run_dir(config.output_dir)
    runtime = replace(
        config,
        feature_set=feature_set,
        model_path=model_path,
        auto_model=auto_model,
        require_real_model=require_real_model,
        require_real_rag=require_real_rag,
        require_llm_orchestrator=require_llm_orchestrator,
        require_llm_reasoning=require_llm_reasoning,
        min_evidence=min_evidence,
        mode=mode,
        retrieval_backend=retrieval_backend,
        es_url=_empty_to_none(payload.get("es_url")) or config.es_url,
        max_rounds=int(payload.get("max_rounds") or config.max_rounds),
        use_llm_orchestrator=use_llm_orchestrator,
        use_llm_reasoning=use_llm_reasoning,
        enable_fallbacks=enable_fallbacks,
        enable_reflection=enable_reflection,
        use_smiles_encoder=use_smiles_encoder,
        use_mllm_visual_semantic=use_mllm_visual_semantic,
        use_external_evidence=use_external_evidence,
        use_rag_evidence=use_rag_evidence,
        allow_network_lookup=allow_network_lookup,
    )
    _write_runtime_config(run_dir, run_mode, request, task_type, dataset_name, runtime, bool(mode_defaults))
    env_updates = {
        "AGENTM_ENABLE_NETWORK_LOOKUP": "1" if allow_network_lookup else "0",
        "AGENTM_PUBCHEM_TIMEOUT": str(runtime.pubchem_timeout),
        "AGENTM_LLM_TIMEOUT": str(runtime.llm_timeout),
    }

    _call_progress(
        progress_callback,
        {
            "event_type": "step_started",
            "step_id": "wait_for_agent_slot",
            "tool_name": "web_run_lock",
            "status": "running",
            "message": "waiting",
        },
    )
    with run_lock:
        _call_progress(
            progress_callback,
            {
                "event_type": "step_completed",
                "step_id": "wait_for_agent_slot",
                "tool_name": "web_run_lock",
                "status": "completed",
                "message": "acquired",
            },
        )
        with temporary_env(env_updates):
            init_start = time.time()
            _call_progress(
                progress_callback,
                {
                    "event_type": "step_started",
                    "step_id": "initialize_agent",
                    "tool_name": "AgentMMultiAgentController",
                    "status": "running",
                    "message": "initializing",
                },
            )
            agent = AgentMMultiAgentController(
                output_dir=run_dir,
                feature_set=runtime.feature_set,
                model_path=runtime.model_path,
                auto_model=runtime.auto_model,
                model_registry_path=runtime.model_registry_path,
                require_real_model=runtime.require_real_model,
                require_real_rag=runtime.require_real_rag,
                require_llm_orchestrator=runtime.require_llm_orchestrator,
                require_llm_reasoning=runtime.require_llm_reasoning,
                min_evidence=runtime.min_evidence,
                mode=runtime.mode,
                retrieval_backend=runtime.retrieval_backend,
                es_url=runtime.es_url,
                max_rounds=runtime.max_rounds,
                temperature=runtime.temperature,
                llm_profile=runtime.llm_profile,
                use_llm_orchestrator=runtime.use_llm_orchestrator,
                use_llm_reasoning=runtime.use_llm_reasoning,
                enable_fallbacks=runtime.enable_fallbacks,
                enable_reflection=runtime.enable_reflection,
                max_reflection_rounds=runtime.max_reflection_rounds,
                previous_run_path=runtime.previous_run_path,
                use_external_evidence=runtime.use_external_evidence,
                use_rag_evidence=runtime.use_rag_evidence,
                use_smiles_encoder=runtime.use_smiles_encoder,
                use_mllm_visual_semantic=runtime.use_mllm_visual_semantic,
                rag_corpus_dirs=runtime.rag_corpus_dirs,
                rag_top_k=runtime.rag_top_k,
                rag_backend=runtime.rag_backend,
                rag_embedding_model_name=runtime.rag_embedding_model_name,
                ddi_rag_corpus_dir=runtime.ddi_rag_corpus_dir,
            )
            _call_progress(
                progress_callback,
                {
                    "event_type": "step_completed",
                    "step_id": "initialize_agent",
                    "tool_name": "AgentMMultiAgentController",
                    "status": "completed",
                    "duration_sec": round(time.time() - init_start, 6),
                    "message": "initialized",
                    "output_summary": {
                        "feature_set": runtime.feature_set,
                        "mode": runtime.mode,
                        "retrieval_backend": runtime.retrieval_backend,
                        "use_llm_orchestrator": runtime.use_llm_orchestrator,
                        "use_llm_reasoning": runtime.use_llm_reasoning,
                        "enable_fallbacks": runtime.enable_fallbacks,
                    },
                },
            )
            report = agent.run(request, task_type=task_type, dataset_name=dataset_name)
            for event in _progress_from_report(report):
                _call_progress(progress_callback, event)

    result = compact_report(report, run_dir=run_dir, config=runtime)
    result["run_mode"] = run_mode
    return result


def compact_report(report: Dict[str, Any], *, run_dir: Path, config: WebDemoConfig) -> Dict[str, Any]:
    prediction = _primary_prediction(report)
    artifacts = report.get("artifacts") or {}
    if not isinstance(artifacts, dict):
        artifacts = {}
    _ensure_molecule_visual_artifacts(report, artifacts=artifacts, run_dir=run_dir)
    task = report.get("task_type")
    narrative = report.get("narrative_report", {}) if isinstance(report.get("narrative_report"), dict) else {}
    evidence = _unique_evidence(report.get("evidence", []))
    warnings = _unique_strings(report.get("warnings", []))
    reasoning = report.get("reasoning") or {}
    verifier = report.get("verifier_report") or {}
    objective = report.get("objective_status") or {}
    multi_agent = report.get("multi_agent") or {}
    explanation = report.get("explanation")
    reasoning_summary = reasoning.get("structured_explanation") or reasoning.get("summary")
    if _same_display_text(explanation, reasoning_summary):
        reasoning_summary = None
    release_info = _release_info(report, config)
    capability_summary = _capability_summary(report, run_dir=run_dir)
    quality_summary = _quality_summary(
        task=task,
        prediction=prediction,
        evidence=evidence,
        verifier=verifier,
        objective=objective,
        release_info=release_info,
    )
    return {
        "status": "ok",
        "run_id": run_dir.name,
        "run_dir": str(run_dir),
        "request": report.get("request"),
        "task_type": task,
        "dataset_name": report.get("dataset_name") or prediction.get("dataset_name"),
        "reaction": report.get("reaction", {}),
        "reaction_profile": report.get("reaction_profile", {}),
        "reaction_predictions": report.get("reaction_predictions", []),
        "reaction_prediction": prediction if _is_reaction_task(task) else (report.get("reaction_predictions") or [{}])[-1] if report.get("reaction_predictions") else {},
        "molecules": [_compact_molecule(item) for item in report.get("molecules", [])],
        "prediction": {
            "task_type": prediction.get("task_type") or task,
            "dataset_name": prediction.get("dataset_name"),
            "label": prediction.get("label"),
            "score": prediction.get("score"),
            "probability": prediction.get("probability"),
            "model_type": prediction.get("model_type"),
            "predicted_meaning": _prediction_meaning(prediction),
            "details": prediction.get("details", {}),
        },
        "narrative_report": narrative,
        "structured_sections": narrative.get("structured_sections", {}),
        "evidence_citations": narrative.get("evidence_citations", []),
        "claim_level_citations": narrative.get("claim_level_citations", []),
        "explanation": explanation,
        "reasoning_summary": reasoning_summary,
        "llm_structured_explanation": reasoning.get("llm_structured_explanation", {}),
        "multi_agent": multi_agent,
        "objective_status": objective,
        "verifier_report": verifier,
        "quality_summary": quality_summary,
        "capability_summary": capability_summary,
        "release_info": release_info,
        "general_profile_notice": _general_profile_notice(task),
        "resource_resolution": report.get("resource_resolution", {}),
        "retrieval_trace": report.get("retrieval_trace", []),
        "citation_map": report.get("citation_map", []),
        "tool_calls": report.get("tool_calls", []),
        "reflection": report.get("reflection", {}),
        "evidence": evidence,
        "evidence_groups": _evidence_groups(evidence),
        "evidence_cards": _evidence_cards(evidence),
        "trace": report.get("trace", []),
        "trace_timeline": _trace_timeline(report),
        "workflow_graph": _workflow_graph(report),
        "plan_steps": _compact_task_graph_steps(artifacts.get("task_graph") or {}),
        "warnings": warnings,
        "images": _collect_images(artifacts, config),
        "files": _collect_artifact_files(artifacts, config) + _collect_release_artifact_files(config),
        "evidence_quality": artifacts.get("evidence_quality", verifier.get("citations", {})),
        "dropped_evidence": artifacts.get("dropped_evidence", []),
        "evidence_grounded_scoring": artifacts.get("evidence_grounded_scoring", {}),
        "calibration_uncertainty": _calibration_uncertainty(prediction),
        "rag_evidence": artifacts.get("rag_evidence", {}),
        "rag_context": artifacts.get("rag_context", []),
        "failure_mode_assessment": artifacts.get("failure_mode_assessment", {}),
        "graph_topology": artifacts.get("graph_topology", {}),
        "fallback_branches": artifacts.get("fallback_branches", []),
        "human_in_loop_points": artifacts.get("human_in_loop_points", []),
        "checkpointing": {
            "manifest_path": artifacts.get("checkpoint_manifest_path"),
            "latest_checkpoint_path": artifacts.get("latest_checkpoint_path"),
        },
        "reproducibility": {"path": artifacts.get("reproducibility_log_path")},
    }


def _progress_from_report(report: Dict[str, Any]) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    for call in report.get("tool_calls", []) or []:
        if not isinstance(call, dict):
            continue
        round_id = call.get("round")
        tool = call.get("tool") or "tool"
        events.append(
            {
                "event_type": "step_completed" if call.get("status") in {"completed", "fallback_completed"} else "step_failed",
                "step_id": f"{round_id}_{tool}" if round_id is not None else str(tool),
                "tool_name": tool,
                "graph_node": call.get("agent"),
                "description": call.get("rationale"),
                "status": call.get("status"),
                "duration_sec": call.get("duration_sec"),
                "error": call.get("error"),
                "fallback_used": call.get("fallback_used"),
                "output_summary": call.get("observation"),
            }
        )
    return events


def _run_public_summary(result: Dict[str, Any]) -> Dict[str, Any]:
    prediction = result.get("prediction") or {}
    reaction_prediction = result.get("reaction_prediction") or {}
    quality = result.get("quality_summary") or {}
    warnings = _unique_strings(result.get("warnings", []) or [])
    trace = result.get("trace_timeline", []) or result.get("trace", []) or []
    fallback_used = any(
        str(item.get("status") or "").startswith("fallback") or bool(item.get("fallback_tool"))
        for item in trace
        if isinstance(item, dict)
    ) or any("fallback" in str(warning).lower() for warning in warnings)
    confidence = result.get("calibration_uncertainty") or {}
    completed_status = quality.get("completed_status")
    if completed_status is None:
        completed_status = "yes" if result.get("status") in {None, "ok", "completed"} else str(result.get("status"))
    model_status = quality.get("model_status")
    if model_status is None:
        prediction_mode = quality.get("prediction_mode")
        if prediction.get("label") is not None or prediction.get("score") is not None or prediction.get("probability") is not None:
            model_status = "available"
        elif prediction_mode in {"not_requested", "reaction_profile"}:
            model_status = "not required"
    return {
        "task_type": result.get("task_type"),
        "dataset_name": result.get("dataset_name") or prediction.get("dataset_name"),
        "prediction": prediction,
        "reaction_prediction": reaction_prediction,
        "confidence": {
            "probability": prediction.get("probability"),
            "score": prediction.get("score"),
            "confidence_level": confidence.get("confidence_level"),
            "calibration_status": confidence.get("calibration_status"),
            "evidence_grounded_score": confidence.get("evidence_grounded_score"),
        },
        "molecule_count": len(result.get("molecules", []) or []),
        "evidence_count": len(_unique_evidence(result.get("evidence", []) or [])),
        "warning_count": len(warnings),
        "warnings": warnings,
        "fallback_used": fallback_used,
        "completed_status": completed_status,
        "model_status": model_status,
        "objective_status": (result.get("objective_status") or {}).get("status"),
        "verifier_status": ((result.get("verifier_report") or {}).get("summary") or {}).get("status"),
        "citation_grounded": quality.get("citation_grounded"),
        "evidence_index": (result.get("release_info") or {}).get("index_name"),
        "planner_type": (result.get("multi_agent") or {}).get("planner_type"),
        "reasoning_mode": (result.get("multi_agent") or {}).get("reasoning_mode"),
        "files": result.get("files", []) or [],
    }


def _release_info(report: Dict[str, Any], config: WebDemoConfig) -> Dict[str, Any]:
    manifest = _read_json_if_exists(_release_manifest_path(config))
    es_manifest = manifest.get("elasticsearch", {}) if isinstance(manifest.get("elasticsearch"), dict) else {}
    qc_readout = _read_json_if_exists(_release_qc_readout_path(config))
    final_qc_row = _final_qc_readout_row(qc_readout)
    acceptance_payload = _read_json_if_exists(_release_acceptance_summary_path(config))
    acceptance = acceptance_payload or (manifest.get("acceptance", {}) if isinstance(manifest.get("acceptance"), dict) else {})
    acceptance_metrics = acceptance.get("metrics", {}) if isinstance(acceptance.get("metrics"), dict) else {}
    acceptance_requirements = acceptance.get("requirements", {}) if isinstance(acceptance.get("requirements"), dict) else {}
    acceptance_counts = (
        acceptance_metrics.get("case_status_counts", {})
        if isinstance(acceptance_metrics.get("case_status_counts"), dict)
        else {}
    )
    source_snapshot = manifest.get("source_snapshot", {}) if isinstance(manifest.get("source_snapshot"), dict) else {}
    inferred_index = _infer_retrieval_index(report)
    document_count = es_manifest.get("document_count") or final_qc_row.get("documents")
    embedding_enabled = es_manifest.get("embedding_enabled")
    if embedding_enabled is None and final_qc_row:
        embedding_enabled = "dense" in str(final_qc_row.get("backend", "")).lower()
    n_embedded = es_manifest.get("n_embedded")
    if n_embedded is None and embedding_enabled and document_count:
        n_embedded = document_count
    index_name = (
        inferred_index
        or os.getenv("AGENTM_ES_INDEX")
        or es_manifest.get("index_name")
        or RELEASE_INDEX_NAME
    )
    top_level_cases = acceptance.get("cases")
    if isinstance(top_level_cases, list):
        top_level_cases = len(top_level_cases)
    acceptance_cases = acceptance_metrics.get("n_cases") or top_level_cases
    acceptance_passing_cases = acceptance_counts.get("pass") or acceptance.get("passing_cases")
    return {
        "index_name": index_name,
        "es_url": os.getenv("AGENTM_ES_URL") or es_manifest.get("es_url"),
        "document_count": document_count,
        "embedding_enabled": embedding_enabled,
        "embedding_dim": es_manifest.get("embedding_dim") or (768 if embedding_enabled else None),
        "n_embedded": n_embedded,
        "index_errors": es_manifest.get("index_errors", 0 if final_qc_row else None),
        "acceptance_status": acceptance.get("status") or qc_readout.get("status"),
        "acceptance_cases": acceptance_cases,
        "acceptance_passing_cases": acceptance_passing_cases,
        "strict_scope": acceptance.get("strict_scope", acceptance_requirements.get("strict_scope")),
        "require_citation": acceptance.get("require_citation", acceptance_requirements.get("require_citation")),
        "expanded_acceptance": final_qc_row.get("strict_pass") or RELEASE_EXPANDED_ACCEPTANCE_STATUS,
        "full_backfill_status": "paused" if source_snapshot else None,
        "full_backfill_completed": source_snapshot.get("full_backfill_completed"),
        "manifest_path": str(_release_manifest_path(config)),
        "qc_readout_path": str(_release_qc_readout_path(config)),
        "structured_jsonl_path": str(_release_structured_jsonl_path(config)),
        "release_summary_path": str(_release_smoke_summary_path(config)),
    }


def _final_qc_readout_row(qc_readout: Dict[str, Any]) -> Dict[str, Any]:
    rows = qc_readout.get("rows") if isinstance(qc_readout.get("rows"), list) else []
    for row in reversed(rows):
        if not isinstance(row, dict):
            continue
        stage = str(row.get("stage") or "").lower()
        backend = str(row.get("backend") or "").lower()
        if "dense" in stage or "dense" in backend:
            return row
    return rows[-1] if rows and isinstance(rows[-1], dict) else {}


def _quality_summary(
    *,
    task: Any,
    prediction: Dict[str, Any],
    evidence: List[Dict[str, Any]],
    verifier: Dict[str, Any],
    objective: Dict[str, Any],
    release_info: Dict[str, Any],
) -> Dict[str, Any]:
    citation_block = verifier.get("citations") if isinstance(verifier.get("citations"), dict) else {}
    summary_block = verifier.get("summary") if isinstance(verifier.get("summary"), dict) else {}
    citation_grounded = citation_block.get("citation_grounded")
    if citation_grounded is None:
        citation_grounded = summary_block.get("citation_grounded")
    if task == "general_drug_profile":
        prediction_mode = "not_requested"
    elif _is_reaction_task(task) and task in {"reaction_profile", "reaction_condition_retrieval"}:
        prediction_mode = "reaction_profile"
    else:
        prediction_mode = "missing"
    has_prediction = prediction.get("label") is not None or prediction.get("score") is not None or prediction.get("probability") is not None
    if has_prediction:
        prediction_mode = "tool_grounded"
    model_status = "available" if has_prediction else "not required" if prediction_mode in {"not_requested", "reaction_profile"} else "missing"
    return {
        "completed": True,
        "completed_status": "yes",
        "verifier_status": summary_block.get("status") or verifier.get("status"),
        "objective_status": objective.get("status"),
        "model_status": model_status,
        "citation_grounded": citation_grounded,
        "evidence_count": len(evidence),
        "unsupported_claims": citation_block.get("n_unsupported", summary_block.get("n_unsupported")),
        "prediction_mode": prediction_mode,
        "evidence_index": release_info.get("index_name"),
        "release_docs": release_info.get("document_count"),
        "dense_indexed": bool(release_info.get("embedding_enabled")),
        "expanded_acceptance": release_info.get("expanded_acceptance"),
        "full_backfill_status": release_info.get("full_backfill_status"),
    }


def _capability_summary(report: Dict[str, Any], *, run_dir: Path) -> Dict[str, Any]:
    runtime = _read_json_if_exists(run_dir / "web_runtime_config.json")
    artifacts = report.get("artifacts") if isinstance(report.get("artifacts"), dict) else {}
    schema = artifacts.get("feature_schema") if isinstance(artifacts.get("feature_schema"), dict) else {}
    schema_metadata = schema.get("metadata") if isinstance(schema.get("metadata"), dict) else {}
    block_sources = schema_metadata.get("block_sources") if isinstance(schema_metadata.get("block_sources"), dict) else {}
    prediction = (report.get("predictions") or [{}])[0] if isinstance(report.get("predictions"), list) else {}
    details = prediction.get("details") if isinstance(prediction.get("details"), dict) else {}
    feature_set = str(runtime.get("feature_set") or schema.get("feature_set") or "")

    smiles_block = block_sources.get("smiles") if isinstance(block_sources.get("smiles"), dict) else {}
    imagemol_block = block_sources.get("imagemol") if isinstance(block_sources.get("imagemol"), dict) else {}
    visual_block = block_sources.get("visual_semantic") if isinstance(block_sources.get("visual_semantic"), dict) else {}
    smiles_details = details.get("smiles_encoder") if isinstance(details.get("smiles_encoder"), dict) else {}
    imagemol_details = details.get("imagemol") if isinstance(details.get("imagemol"), dict) else {}
    visual_details = details.get("visual_semantics") if isinstance(details.get("visual_semantics"), dict) else {}

    smiles_requested = bool(runtime.get("use_smiles_encoder")) or feature_set in {"all_features", "smiles_only"}
    imagemol_requested = feature_set in {"all_features", "imagemol_only"}
    visual_requested = bool(runtime.get("use_mllm_visual_semantic")) or feature_set in {"all_features", "visual_semantic_only"}
    smiles_used = bool(smiles_block.get("enabled")) and int(smiles_block.get("dim") or 0) > 0
    smiles_used = smiles_used or int(smiles_details.get("embedding_dim") or 0) > 0
    imagemol_used = str(imagemol_block.get("encoder") or imagemol_details.get("encoder_name") or "").lower() not in {"", "disabled", "none"}
    imagemol_used = imagemol_used and (int(imagemol_block.get("dim") or imagemol_details.get("embedding_dim") or 0) > 0)
    visual_provider = str(visual_details.get("provider") or visual_block.get("source") or "").lower()
    visual_used = visual_provider not in {"", "disabled", "none"} and (
        int(visual_block.get("dim") or 0) > 0
        or bool(visual_details.get("summary"))
        or bool(visual_details.get("task_relevant_visual_cues"))
    )
    items = [
        _capability_item("smiles_encoder", "SMILES encoder", smiles_requested, smiles_used, smiles_block, smiles_details),
        _capability_item("imagemol", "ImageMol", imagemol_requested, imagemol_used, imagemol_block, imagemol_details),
        _capability_item("mllm_visual_semantic", "MLLM visual semantics", visual_requested, visual_used, visual_block, visual_details),
    ]
    return {
        "feature_set_requested": runtime.get("feature_set"),
        "feature_set_actual": schema.get("feature_set") or runtime.get("feature_set"),
        "items": items,
    }


def _capability_item(
    key: str,
    label: str,
    requested: bool,
    used: bool,
    schema_block: Dict[str, Any],
    details: Dict[str, Any],
) -> Dict[str, Any]:
    if requested and used:
        status = "requested_and_used"
    elif requested and not used:
        status = "requested_not_used"
    elif used:
        status = "used_by_schema"
    else:
        status = "not_requested"
    reason = ""
    if requested and not used:
        reason = "requested, but the selected feature schema/model did not expose a nonzero usable block"
    elif used:
        reason = "present in the selected feature schema or prediction details"
    return {
        "key": key,
        "label": label,
        "requested": bool(requested),
        "used": bool(used),
        "status": status,
        "reason": reason,
        "schema": schema_block,
        "details": details,
    }


def _general_profile_notice(task: Any) -> Optional[str]:
    if task != "general_drug_profile":
        return None
    return (
        "General drug profile: this run summarizes public drug information and structure descriptors. "
        "Endpoint-specific ADMET/DDI prediction is not run unless a task such as DILI, hERG, BBBP, or DDI is specified."
    )


def _is_reaction_task(task: Any) -> bool:
    return str(task or "").startswith("reaction_")


def _primary_prediction(report: Dict[str, Any]) -> Dict[str, Any]:
    task = report.get("task_type")
    if _is_reaction_task(task):
        predictions = report.get("reaction_predictions") or report.get("predictions") or []
        return predictions[-1] if predictions and isinstance(predictions[-1], dict) else {}
    predictions = report.get("predictions") or []
    return predictions[0] if predictions and isinstance(predictions[0], dict) else {}


def _infer_retrieval_index(report: Dict[str, Any]) -> Optional[str]:
    def walk(value: Any) -> Optional[str]:
        if isinstance(value, dict):
            index = value.get("index_name")
            if isinstance(index, str) and index.strip():
                return index
            for child in value.values():
                found = walk(child)
                if found:
                    return found
        if isinstance(value, list):
            for child in value:
                found = walk(child)
                if found:
                    return found
        return None

    return walk(report.get("retrieval_trace"))


def _read_json_if_exists(path: Path) -> Dict[str, Any]:
    try:
        if path.is_file():
            payload = json.loads(path.read_text(encoding="utf-8"))
            return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}
    return {}


def artifact_url(path: Any, config: WebDemoConfig) -> Optional[str]:
    if not path:
        return None
    artifact = _resolve_artifact_path(str(path), config.project_root)
    if not artifact.is_file() or not _is_allowed_artifact(artifact, config):
        return None
    try:
        rel = artifact.relative_to(config.project_root.resolve())
        return "/artifacts/" + quote(rel.as_posix())
    except ValueError:
        rel = artifact.relative_to(_resolved_output_dir(config))
        return "/artifacts/" + quote(str(Path(config.output_dir).name / rel))


def serve(config: WebDemoConfig) -> None:
    import uvicorn

    _ensure_dir(config.output_dir)
    print(f"AgentM web demo: http://{config.host}:{config.port}")
    print(f"writing runs to: {config.output_dir}")
    uvicorn.run(create_app(config), host=config.host, port=config.port)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the AgentM web demo.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7864)
    parser.add_argument("--output-dir", default=str(RUNTIME_ROOT / "outputs" / "web_demo"))
    parser.add_argument("--feature-set", default="rdkit_only")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--auto-model", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--model-registry", default=None)
    parser.add_argument("--require-real-model", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--require-real-rag", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--require-llm-orchestrator", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--require-llm-reasoning", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--min-evidence", type=int, default=1)
    parser.add_argument("--mode", default="fast", choices=["fast", "full", "deep"])
    parser.add_argument("--retrieval-backend", default="local", choices=["auto", "es", "local"])
    parser.add_argument("--es-url", default=os.getenv("AGENTM_ES_URL", "http://127.0.0.1:9200"))
    parser.add_argument("--max-rounds", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--llm-profile", default=DEFAULT_LLM_PROFILE)
    parser.add_argument("--llm-timeout", type=int, default=60)
    parser.add_argument("--use-llm-orchestrator", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-llm-reasoning", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--enable-fallbacks", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable-reflection", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-reflection-rounds", type=int, default=1)
    parser.add_argument("--previous-run", default=None)
    parser.add_argument("--use-smiles-encoder", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use-mllm-visual-semantic", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use-external-evidence", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use-rag-evidence", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rag-corpus-dir", action="append", default=None)
    parser.add_argument("--rag-top-k", type=int, default=DEFAULT_RAG_TOP_K)
    parser.add_argument("--rag-backend", default=DEFAULT_RAG_BACKEND, choices=["dense_faiss", "tfidf"])
    parser.add_argument("--rag-embedding-model-name", default=_resource_path(DEFAULT_RAG_EMBEDDING_MODEL_PATH))
    parser.add_argument("--ddi-rag-corpus-dir", default=_resource_path(DEFAULT_DDI_RAG_CORPUS_DIR))
    parser.add_argument("--allow-network-lookup", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pubchem-timeout", type=float, default=5.0)
    return parser


def config_from_args(args: argparse.Namespace) -> WebDemoConfig:
    output_dir = Path(args.output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = RUNTIME_ROOT / output_dir
    return WebDemoConfig(
        host=args.host,
        port=args.port,
        output_dir=output_dir,
        feature_set=args.feature_set,
        model_path=args.model_path,
        auto_model=args.auto_model,
        model_registry_path=args.model_registry,
        require_real_model=args.require_real_model,
        require_real_rag=args.require_real_rag,
        require_llm_orchestrator=args.require_llm_orchestrator,
        require_llm_reasoning=args.require_llm_reasoning,
        min_evidence=args.min_evidence,
        mode=args.mode,
        retrieval_backend=args.retrieval_backend,
        es_url=args.es_url,
        max_rounds=args.max_rounds,
        temperature=args.temperature,
        llm_profile=args.llm_profile,
        llm_timeout=args.llm_timeout,
        use_llm_orchestrator=args.use_llm_orchestrator,
        use_llm_reasoning=args.use_llm_reasoning,
        enable_fallbacks=args.enable_fallbacks,
        enable_reflection=args.enable_reflection,
        max_reflection_rounds=args.max_reflection_rounds,
        previous_run_path=args.previous_run,
        use_smiles_encoder=args.use_smiles_encoder,
        use_mllm_visual_semantic=args.use_mllm_visual_semantic,
        use_external_evidence=args.use_external_evidence,
        use_rag_evidence=args.use_rag_evidence,
        rag_corpus_dirs=args.rag_corpus_dir or [_resource_path(DEFAULT_RAG_CORPUS_DIR)],
        rag_top_k=args.rag_top_k,
        rag_backend=args.rag_backend,
        rag_embedding_model_name=args.rag_embedding_model_name,
        ddi_rag_corpus_dir=args.ddi_rag_corpus_dir,
        allow_network_lookup=args.allow_network_lookup,
        pubchem_timeout=args.pubchem_timeout,
    )


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    serve(config_from_args(args))
    return 0


def _write_runtime_config(
    run_dir: Path,
    run_mode: str,
    request: str,
    task_type: Optional[str],
    dataset_name: Optional[str],
    runtime: WebDemoConfig,
    mode_defaults_applied: bool,
) -> None:
    payload = {
        "run_mode": run_mode,
        "request": request,
        "task_type": task_type,
        "dataset_name": dataset_name,
        "mode_defaults_applied": mode_defaults_applied,
        "feature_set": runtime.feature_set,
        "mode": runtime.mode,
        "retrieval_backend": runtime.retrieval_backend,
        "auto_model": runtime.auto_model,
        "require_real_model": runtime.require_real_model,
        "require_real_rag": runtime.require_real_rag,
        "require_llm_orchestrator": runtime.require_llm_orchestrator,
        "require_llm_reasoning": runtime.require_llm_reasoning,
        "use_llm_orchestrator": runtime.use_llm_orchestrator,
        "use_llm_reasoning": runtime.use_llm_reasoning,
        "enable_fallbacks": runtime.enable_fallbacks,
        "enable_reflection": runtime.enable_reflection,
        "max_rounds": runtime.max_rounds,
        "min_evidence": runtime.min_evidence,
        "use_smiles_encoder": runtime.use_smiles_encoder,
        "use_mllm_visual_semantic": runtime.use_mllm_visual_semantic,
        "use_external_evidence": runtime.use_external_evidence,
        "use_rag_evidence": runtime.use_rag_evidence,
        "allow_network_lookup": runtime.allow_network_lookup,
        "llm_profile": runtime.llm_profile,
        "llm_timeout": runtime.llm_timeout,
    }
    if runtime.model_path is not None:
        payload["model_path"] = runtime.model_path
    if runtime.es_url is not None:
        payload["es_url"] = runtime.es_url
    payload = to_jsonable(payload)
    (run_dir / "web_runtime_config.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _prediction_meaning(prediction: Dict[str, Any]) -> Optional[str]:
    details = prediction.get("details") or {}
    label = prediction.get("label")
    if is_multiclass_ddi_prediction(prediction):
        return scientific_prediction_meaning(
            prediction,
            drug_names=details.get("open_task_molecules") or None,
        )
    if label == details.get("positive_label"):
        return details.get("positive_label_meaning")
    if label == details.get("negative_label"):
        return details.get("negative_label_meaning")
    if prediction.get("task_type") == "ddi_prediction":
        return "DDI risk/interaction" if label == 1 else "no DDI risk/interaction"
    if prediction.get("task_type") == "reaction_classification":
        return f"reaction class {label}" if label not in {None, ""} else None
    if prediction.get("task_type") == "reaction_yield_prediction":
        score = prediction.get("score")
        return f"{label} ({format(float(score), '.1f')}% predicted yield)" if isinstance(score, (int, float)) else str(label or "")
    return None


def _compact_molecule(molecule: Dict[str, Any]) -> Dict[str, Any]:
    metadata = molecule.get("metadata") if isinstance(molecule.get("metadata"), dict) else {}
    return {
        "original_input": molecule.get("original_input"),
        "name": molecule.get("name"),
        "source": molecule.get("source"),
        "smiles": molecule.get("smiles"),
        "canonical_smiles": molecule.get("canonical_smiles"),
        "is_valid": molecule.get("is_valid"),
        "errors": molecule.get("errors", []),
        "resolution_trace": metadata.get("resolution_trace", []),
    }


def _compact_plan_steps(plan: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [
        {
            "step_id": step.get("step_id"),
            "tool": step.get("tool_name"),
            "status": step.get("status"),
            "condition": step.get("condition"),
            "message": step.get("message"),
        }
        for step in (plan.get("steps") or [])
    ]


def _compact_task_graph_steps(task_graph: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [
        {
            "step_id": node.get("id"),
            "tool": node.get("id"),
            "agent": node.get("agent"),
            "goal": node.get("goal"),
            "status": task_graph.get("status"),
        }
        for node in (task_graph.get("nodes") or [])
        if isinstance(node, dict)
    ]


def _evidence_groups(evidence: List[Dict[str, Any]]) -> Dict[str, Any]:
    groups: Dict[str, Any] = {"support": [], "contradiction": [], "neutral": [], "by_source": {}, "by_direction": {"support": [], "contradiction": [], "neutral": []}}
    for item in evidence or []:
        direction = str(item.get("evidence_direction") or "neutral")
        key = direction if direction in groups else "neutral"
        groups[key].append(item)
        groups["by_direction"][key].append(item)
        source_group = _evidence_source_group(item)
        groups["by_source"].setdefault(source_group, []).append(item)
    return groups


def _evidence_cards(evidence: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    cards = []
    for index, item in enumerate(evidence or [], start=1):
        value = item.get("value") if isinstance(item.get("value"), dict) else {}
        details = item.get("details") if isinstance(item.get("details"), dict) else {}
        score = item.get("retrieval_score") or item.get("rerank_score") or item.get("quality_score") or item.get("evidence_weight") or value.get("score")
        url = item.get("url") or value.get("url") or value.get("pubmed_url") or value.get("pubchem_url") or value.get("dailymed_url")
        cards.append(
            {
                "index": index,
                "source_group": _evidence_source_group(item),
                "source": item.get("source") or value.get("source"),
                "direction": item.get("evidence_direction") or "neutral",
                "support_level": item.get("support_level") or item.get("evidence_direction") or "neutral",
                "query": item.get("query") or value.get("query") or details.get("query"),
                "score": score,
                "url": url,
                "identifier": item.get("pmid") or item.get("cid") or value.get("pmid") or value.get("cid"),
                "claim": item.get("claim") or item.get("mechanism") or item.get("interaction_type") or value.get("title"),
                "related_claim": item.get("related_claim") or item.get("claim") or details.get("related_claim"),
                "is_fallback": bool(item.get("is_fallback") or item.get("fallback") or value.get("fallback")),
                "is_low_quality": bool(item.get("dropped") or item.get("low_quality") or (isinstance(score, (int, float)) and float(score) < 0.25)),
                "raw": item,
            }
        )
    return cards


def _evidence_source_group(item: Dict[str, Any]) -> str:
    value = item.get("value") if isinstance(item.get("value"), dict) else {}
    source_text = " ".join(str(part or "").lower() for part in [item.get("source"), value.get("source"), value.get("url"), value.get("database"), item.get("provider")])
    if "elasticsearch" in source_text:
        return "elasticsearch_hybrid"
    if "rag" in source_text or "tdc_real" in source_text or "ddi_real" in source_text:
        return "rag_literature"
    if "pubmed" in source_text or item.get("pmid") or value.get("pmid"):
        return "pubmed"
    if "dailymed" in source_text:
        return "dailymed"
    if "pubchem" in source_text or item.get("cid") or value.get("cid"):
        return "pubchem"
    if "local" in source_text or "ddi_hint" in source_text:
        return "local_ddi_evidence"
    if "rule" in source_text or "model" in source_text or "prediction" in source_text:
        return "model_rule_evidence"
    return "other_evidence"


def _unique_strings(items: Any) -> List[str]:
    unique: List[str] = []
    seen: set[str] = set()
    for item in items or []:
        text = str(item)
        if text in seen:
            continue
        seen.add(text)
        unique.append(text)
    return unique


def _same_display_text(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return False
    left_text = " ".join(str(left).split())
    right_text = " ".join(str(right).split())
    return bool(left_text and left_text == right_text)


def _unique_evidence(items: Any) -> List[Dict[str, Any]]:
    unique: List[Dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for item in items or []:
        if not isinstance(item, dict):
            continue
        key = _evidence_key(item)
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


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


def _trace_timeline_from_steps(trace: List[Dict[str, Any]], plan: Dict[str, Any]) -> List[Dict[str, Any]]:
    by_step = {str(item.get("step_id") or item.get("graph_node") or ""): item for item in trace or []}
    timeline = []
    for plan_step in plan.get("steps") or []:
        step_id = str(plan_step.get("step_id") or plan_step.get("graph_node") or "")
        observed = by_step.get(step_id, {})
        timeline.append(
            {
                "step_id": step_id,
                "tool_name": observed.get("tool_name") or plan_step.get("tool_name"),
                "description": plan_step.get("description") or observed.get("description"),
                "status": observed.get("status") or plan_step.get("status"),
                "duration_sec": observed.get("duration_sec"),
                "fallback_tool": observed.get("fallback_tool") or observed.get("fallback_used"),
                "message": observed.get("message") or plan_step.get("message"),
                "output_summary": observed.get("output_summary") or {},
                "raw": observed or plan_step,
            }
        )
    seen = {item["step_id"] for item in timeline}
    for observed in trace or []:
        step_id = str(observed.get("step_id") or observed.get("graph_node") or "")
        if step_id and step_id not in seen:
            timeline.append({"step_id": step_id, "status": observed.get("status"), "raw": observed})
    return timeline


def _trace_timeline(report: Dict[str, Any]) -> List[Dict[str, Any]]:
    timeline = []
    for call in report.get("tool_calls", []) or []:
        if not isinstance(call, dict):
            continue
        observation = call.get("observation") if isinstance(call.get("observation"), dict) else {}
        timeline.append(
            {
                "step_id": f"{call.get('round')}_{call.get('tool')}",
                "tool_name": call.get("tool"),
                "agent": call.get("agent"),
                "description": call.get("rationale"),
                "status": call.get("status"),
                "duration_sec": call.get("duration_sec"),
                "fallback_tool": call.get("fallback_used"),
                "message": call.get("error") or call.get("rationale"),
                "output_summary": _compact_for_status(observation),
                "raw": call,
            }
        )
    if timeline:
        return timeline
    return _trace_timeline_from_steps(report.get("trace", []), report.get("plan") or {})


def _workflow_graph(report: Dict[str, Any]) -> Dict[str, Any]:
    artifacts = report.get("artifacts") if isinstance(report.get("artifacts"), dict) else {}
    orchestrator_plan = artifacts.get("orchestrator_plan") if isinstance(artifacts.get("orchestrator_plan"), dict) else {}
    task_graph = artifacts.get("task_graph") if isinstance(artifacts.get("task_graph"), dict) else {}
    if not task_graph:
        task_graph = orchestrator_plan.get("task_graph") if isinstance(orchestrator_plan.get("task_graph"), dict) else {}
    planned_nodes = _workflow_plan_nodes(task_graph)
    planned_edges = _workflow_plan_edges(task_graph.get("edges") if isinstance(task_graph, dict) else [])
    execution_nodes, execution_edges = _workflow_execution(report.get("tool_calls", []) or [])
    _workflow_add_inferred_artifact_write(report, execution_nodes, execution_edges)
    _workflow_annotate_execution_origins(planned_nodes, execution_nodes)
    topology = _workflow_full_topology(report, planned_nodes, execution_nodes)
    loop_status = _workflow_loop_status(report)
    langgraph_nodes = [
        {"id": "START", "label": "START", "kind": "system", "status": "completed"},
        {"id": "initialize_context", "label": "initialize_context", "kind": "langgraph", "status": "completed"},
        {"id": "orchestrator_loop", "label": "orchestrator_loop", "kind": "langgraph", "status": loop_status},
        {"id": "guarded_finalize", "label": "guarded_finalize", "kind": "langgraph", "status": "completed"},
        {"id": "END", "label": "END", "kind": "system", "status": "completed"},
    ]
    langgraph_edges = [
        {"source": "START", "target": "initialize_context"},
        {"source": "initialize_context", "target": "orchestrator_loop"},
        {"source": "orchestrator_loop", "target": "guarded_finalize"},
        {"source": "guarded_finalize", "target": "END"},
    ]
    graph = {
        "runtime": "LangGraph guarded multi-agent",
        "langgraph": {"nodes": langgraph_nodes, "edges": langgraph_edges},
        "planned": {"status": task_graph.get("status") if isinstance(task_graph, dict) else None, "nodes": planned_nodes, "edges": planned_edges},
        "execution": {"nodes": execution_nodes, "edges": execution_edges},
        "full_topology": topology,
        "summary": {
            "planned_steps": len(planned_nodes),
            "executed_steps": len(execution_nodes),
            "runtime_added_steps": len([node for node in execution_nodes if node.get("origin") == "runtime_added"]),
            "available_tools": len(topology.get("nodes", [])),
            "inactive_tools": len([node for node in topology.get("nodes", []) if node.get("status") == "inactive"]),
            "agents": sorted({str(node.get("agent")) for node in execution_nodes if node.get("agent")}),
        },
    }
    graph["mermaid"] = _workflow_mermaid(graph)
    graph["topology_mermaid"] = _workflow_topology_mermaid(topology)
    return graph


def _workflow_loop_status(report: Dict[str, Any]) -> str:
    raw = str((report.get("multi_agent") or {}).get("loop_status") or "").strip()
    if raw in {"finished", "finalized", "complete", "completed"}:
        return "completed"
    artifacts = report.get("artifacts") if isinstance(report.get("artifacts"), dict) else {}
    write_done = any(
        isinstance(call, dict) and call.get("tool") == "write_artifacts" and call.get("status") == "completed"
        for call in report.get("tool_calls", []) or []
    )
    if raw in {"", "ready", "running", "finishing"} and (write_done or artifacts.get("multi_agent_report_path")):
        return "completed"
    return raw or "completed"


def _workflow_plan_nodes(task_graph: Dict[str, Any]) -> List[Dict[str, Any]]:
    nodes = task_graph.get("nodes") if isinstance(task_graph, dict) else []
    return [
        {
            "id": f"plan_{index}",
            "raw_id": str(node.get("id") or f"planned_{index}"),
            "label": f"{index}. {node.get('agent') or node.get('id') or 'step'}",
            "agent": node.get("agent"),
            "goal": node.get("goal"),
            "kind": "planned",
            "status": task_graph.get("status"),
        }
        for index, node in enumerate(nodes or [], start=1)
        if isinstance(node, dict)
    ]


def _workflow_plan_edges(edges: Any) -> List[Dict[str, Any]]:
    normalized = []
    for edge in edges or []:
        if isinstance(edge, (list, tuple)) and len(edge) >= 2:
            normalized.append({"source": str(edge[0]), "target": str(edge[1])})
        elif isinstance(edge, dict) and edge.get("source") and edge.get("target"):
            normalized.append({"source": str(edge.get("source")), "target": str(edge.get("target"))})
    return normalized


def _workflow_execution(tool_calls: List[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []
    previous_id = ""
    for index, call in enumerate(tool_calls or [], start=1):
        if not isinstance(call, dict):
            continue
        tool = str(call.get("tool") or "tool")
        agent = str(call.get("agent") or "Agent")
        node_id = f"exec_{call.get('round') or index}_{tool}"
        nodes.append(
            {
                "id": node_id,
                "label": f"{agent}.{tool}",
                "agent": agent,
                "tool": tool,
                "status": call.get("status"),
                "duration_sec": call.get("duration_sec"),
                "fallback_used": call.get("fallback_used"),
                "rationale": call.get("rationale"),
                "error": call.get("error"),
                "kind": "executed",
            }
        )
        if previous_id:
            edges.append({"source": previous_id, "target": node_id})
        previous_id = node_id
    return nodes, edges


def _workflow_add_inferred_artifact_write(report: Dict[str, Any], nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]]) -> None:
    if any(node.get("tool") == "write_artifacts" for node in nodes):
        return
    if not _workflow_has_written_artifacts(report):
        return
    node_id = f"exec_{len(nodes) + 1}_write_artifacts"
    if nodes:
        edges.append({"source": nodes[-1]["id"], "target": node_id})
    nodes.append(
        {
            "id": node_id,
            "label": "ReporterAgent.write_artifacts",
            "agent": "ReporterAgent",
            "tool": "write_artifacts",
            "status": "completed",
            "duration_sec": None,
            "fallback_used": False,
            "rationale": "Inferred from persisted artifact paths.",
            "error": "",
            "kind": "executed",
            "inferred": True,
        }
    )


def _workflow_annotate_execution_origins(planned_nodes: List[Dict[str, Any]], execution_nodes: List[Dict[str, Any]]) -> None:
    planned_tools = _workflow_planned_tools(planned_nodes)
    for node in execution_nodes or []:
        tool = str(node.get("tool") or "")
        if not tool:
            continue
        if tool in planned_tools:
            node["origin"] = "planned"
            node["origin_note"] = "from orchestrator plan"
        else:
            node["origin"] = "runtime_added"
            node["origin_note"] = "runtime/guard added"


def _workflow_full_topology(report: Dict[str, Any], planned_nodes: List[Dict[str, Any]], execution_nodes: List[Dict[str, Any]]) -> Dict[str, Any]:
    catalog = _workflow_tool_catalog()
    executed_by_tool = {str(node.get("tool")): node for node in execution_nodes if node.get("tool")}
    if "write_artifacts" not in executed_by_tool and _workflow_has_written_artifacts(report):
        executed_by_tool["write_artifacts"] = {
            "agent": "ReporterAgent",
            "tool": "write_artifacts",
            "status": "completed",
            "duration_sec": None,
            "rationale": "Inferred from persisted artifact paths.",
        }
    planned_tools = _workflow_planned_tools(planned_nodes)
    nodes: List[Dict[str, Any]] = []
    for item in catalog:
        tool = item["tool"]
        executed = executed_by_tool.get(tool)
        planned = tool in planned_tools
        status = executed.get("status") if executed else ("planned_inactive" if planned else "inactive")
        origin_note = ""
        if executed:
            origin_note = "from orchestrator plan" if planned else "runtime/guard added"
        nodes.append(
            {
                **item,
                "id": f"{item['agent']}.{tool}",
                "label": tool,
                "status": status,
                "active": bool(executed),
                "planned": bool(planned),
                "origin": "planned" if executed and planned else ("runtime_added" if executed else ""),
                "origin_note": origin_note,
                "inferred": executed.get("inferred") if executed else False,
                "duration_sec": executed.get("duration_sec") if executed else None,
                "description": item.get("description"),
            }
        )
    active_edge_keys = _workflow_executed_topology_edge_keys(execution_nodes)
    topology_edges = []
    for edge in _workflow_topology_edges():
        key = (str(edge.get("source") or ""), str(edge.get("target") or ""))
        topology_edges.append({**edge, "active": key in active_edge_keys})
    lanes = []
    for agent in ["MoleculeAgent", "GaugeAgent", "RetrieveAgent", "ReactionAgent", "ReasoningAgent", "ReporterAgent", "VerifierAgent", "ReflectionAgent"]:
        lane_nodes = [node for node in nodes if node.get("agent") == agent]
        if lane_nodes:
            lanes.append({"agent": agent, "nodes": lane_nodes})
    return {
        "nodes": nodes,
        "edges": topology_edges,
        "lanes": lanes,
        "legend": {
            "completed": "Tool executed in this run.",
            "planned_inactive": "Mentioned in the plan but not executed.",
            "inactive": "Available branch/tool that was not activated.",
            "failed": "Tool executed and failed.",
        },
    }


def _workflow_executed_topology_edge_keys(execution_nodes: List[Dict[str, Any]]) -> set[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()
    previous = ""
    for node in execution_nodes or []:
        agent = str(node.get("agent") or "")
        tool = str(node.get("tool") or "")
        if not agent or not tool:
            previous = ""
            continue
        current = f"{agent}.{tool}"
        if previous:
            keys.add((previous, current))
        previous = current
    return keys


def _workflow_has_written_artifacts(report: Dict[str, Any]) -> bool:
    artifacts = report.get("artifacts") if isinstance(report.get("artifacts"), dict) else {}
    return bool(artifacts.get("multi_agent_report_path") or artifacts.get("compat_report_path") or artifacts.get("reproducibility_log_path"))


def _workflow_planned_tools(planned_nodes: List[Dict[str, Any]]) -> set[str]:
    planned: set[str] = set()
    for node in planned_nodes or []:
        raw = str(node.get("raw_id") or node.get("id") or "").strip().lower()
        goal = str(node.get("goal") or "").lower()
        agent = str(node.get("agent") or "")
        text = f"{raw} {goal}"
        for item in _workflow_tool_catalog():
            tool = item["tool"]
            if raw == tool.lower() or tool.lower() in text:
                planned.add(tool)
        if agent == "MoleculeAgent" and "resolve" in text:
            planned.add("resolve_ddi_pair" if "ddi" in text or "two" in text or "pair" in text else "resolve_molecule")
        if agent == "RetrieveAgent":
            if "external" in text or "database" in text or "pubchem" in text or "dailymed" in text or "chembl" in text:
                planned.add("retrieve_external_evidence")
            elif "elastic" in text or "hybrid" in text:
                planned.add("retrieve_elasticsearch_hybrid")
            elif "rag" in text or "local" in text or "evidence" in text:
                planned.add("retrieve_local_rag")
        if agent == "GaugeAgent":
            if "feature" in text:
                planned.add("build_features")
            if "ddi" in text and "predict" in text:
                planned.add("predict_ddi")
            elif "prediction" in text or "predict" in text or "single-molecule" in text:
                planned.add("predict_single")
            if "confidence" in text:
                planned.add("assess_confidence")
            if "rule" in text:
                planned.add("discover_rules")
        if agent == "ReactionAgent":
            if "parse" in text:
                planned.add("parse_reaction")
            if "mit" in text or "template" in text or "product" in text:
                planned.add("retrieve_mit_template_evidence")
            if "orderly" in text or "condition" in text:
                planned.add("retrieve_orderly_condition_evidence")
            if "class" in text or "schneider" in text:
                planned.add("classify_reaction_type")
            if "yield" in text or "buchwald" in text:
                planned.add("predict_buchwald_hartwig_yield")
            if "profile" in text or "report" in text or "summary" in text:
                planned.add("build_reaction_profile")
        if agent == "ReasoningAgent":
            if "summar" in text and "evidence" in text:
                planned.add("summarize_evidence_support")
            else:
                planned.add("build_structured_explanation")
        if agent == "ReporterAgent":
            if "compatible" in text and "core" in text:
                planned.add("build_compatible_report")
            elif "write" in text or "artifact" in text:
                planned.add("write_artifacts")
            else:
                planned.add("build_runtime_report")
        if agent == "VerifierAgent":
            if "prediction" in text:
                planned.add("verify_prediction_grounding")
            if "citation" in text or "evidence" in text:
                planned.add("verify_citation_grounding")
            if "schema" in text or "core" in text:
                planned.add("verify_report_schema")
        if agent == "ReflectionAgent":
            if "apply" in text:
                planned.add("apply_reflection_feedback")
            else:
                planned.add("reflect_on_report")
    return planned


def _workflow_tool_catalog() -> List[Dict[str, Any]]:
    return [
        {"agent": "MoleculeAgent", "tool": "route_task", "stage": "input", "description": "Parse request and infer task type."},
        {"agent": "MoleculeAgent", "tool": "resolve_molecule", "stage": "input", "description": "Resolve a single molecule or drug name."},
        {"agent": "MoleculeAgent", "tool": "resolve_ddi_pair", "stage": "input", "description": "Resolve two molecules for DDI."},
        {"agent": "GaugeAgent", "tool": "discover_rules", "stage": "prediction", "description": "Load task rules."},
        {"agent": "GaugeAgent", "tool": "build_features", "stage": "prediction", "description": "Build molecular features."},
        {"agent": "GaugeAgent", "tool": "predict_single", "stage": "prediction", "description": "Run single-molecule model."},
        {"agent": "GaugeAgent", "tool": "predict_ddi", "stage": "prediction", "description": "Run DDI model."},
        {"agent": "GaugeAgent", "tool": "assess_confidence", "stage": "prediction", "description": "Assess confidence."},
        {"agent": "GaugeAgent", "tool": "run_zero_shot_fallback", "stage": "prediction", "description": "Explicit fallback predictor."},
        {"agent": "RetrieveAgent", "tool": "retrieve_external_evidence", "stage": "evidence", "description": "PubChem/DailyMed/ChEMBL or structured evidence."},
        {"agent": "RetrieveAgent", "tool": "retrieve_local_rag", "stage": "evidence", "description": "Local RAG retrieval."},
        {"agent": "RetrieveAgent", "tool": "retrieve_elasticsearch_hybrid", "stage": "evidence", "description": "Elasticsearch hybrid retrieval."},
        {"agent": "RetrieveAgent", "tool": "expand_evidence_query", "stage": "evidence", "description": "Query expansion/repair."},
        {"agent": "ReactionAgent", "tool": "parse_reaction", "stage": "reaction", "description": "Parse reaction SMILES."},
        {"agent": "RetrieveAgent", "tool": "retrieve_mit_template_evidence", "stage": "evidence", "description": "Retrieve MIT template/product evidence."},
        {"agent": "RetrieveAgent", "tool": "retrieve_orderly_condition_evidence", "stage": "evidence", "description": "Retrieve ORDerly condition evidence."},
        {"agent": "ReactionAgent", "tool": "classify_reaction_type", "stage": "reaction", "description": "Classify reaction type."},
        {"agent": "ReactionAgent", "tool": "predict_buchwald_hartwig_yield", "stage": "reaction", "description": "Predict Buchwald-Hartwig yield."},
        {"agent": "ReactionAgent", "tool": "build_reaction_profile", "stage": "reaction", "description": "Build grounded reaction profile."},
        {"agent": "ReasoningAgent", "tool": "build_structured_explanation", "stage": "reasoning", "description": "Build grounded explanation."},
        {"agent": "ReasoningAgent", "tool": "summarize_evidence_support", "stage": "reasoning", "description": "Summarize support and gaps."},
        {"agent": "ReporterAgent", "tool": "build_runtime_report", "stage": "report", "description": "Build runtime report payload."},
        {"agent": "ReporterAgent", "tool": "build_compatible_report", "stage": "report", "description": "Build core-compatible report."},
        {"agent": "ReporterAgent", "tool": "write_artifacts", "stage": "report", "description": "Write artifacts."},
        {"agent": "VerifierAgent", "tool": "verify_prediction_grounding", "stage": "verify", "description": "Verify prediction record grounding."},
        {"agent": "VerifierAgent", "tool": "verify_citation_grounding", "stage": "verify", "description": "Verify claim-to-evidence citation links."},
        {"agent": "VerifierAgent", "tool": "verify_report_schema", "stage": "verify", "description": "Verify report schema and objective status."},
        {"agent": "ReflectionAgent", "tool": "reflect_on_report", "stage": "reflect", "description": "Read verifier feedback for bounded repair."},
        {"agent": "ReflectionAgent", "tool": "apply_reflection_feedback", "stage": "reflect", "description": "Apply bounded reflection feedback."},
    ]


def _workflow_topology_edges() -> List[Dict[str, Any]]:
    def edge(source: str, target: str, condition: str = "", kind: str = "forward") -> Dict[str, Any]:
        return {"source": source, "target": target, "condition": condition, "kind": kind}

    return [
        edge("MoleculeAgent.route_task", "MoleculeAgent.resolve_molecule", "single/general"),
        edge("MoleculeAgent.route_task", "MoleculeAgent.resolve_ddi_pair", "DDI"),
        edge("MoleculeAgent.route_task", "ReactionAgent.parse_reaction", "reaction"),
        edge("MoleculeAgent.resolve_molecule", "RetrieveAgent.retrieve_external_evidence", "general profile or external evidence"),
        edge("MoleculeAgent.resolve_molecule", "GaugeAgent.discover_rules", "endpoint prediction"),
        edge("MoleculeAgent.resolve_ddi_pair", "GaugeAgent.discover_rules", "DDI"),
        edge("GaugeAgent.discover_rules", "GaugeAgent.build_features"),
        edge("RetrieveAgent.retrieve_external_evidence", "GaugeAgent.build_features", "general profile"),
        edge("GaugeAgent.build_features", "GaugeAgent.predict_single", "single endpoint"),
        edge("GaugeAgent.build_features", "GaugeAgent.predict_ddi", "DDI"),
        edge("GaugeAgent.build_features", "RetrieveAgent.retrieve_local_rag", "general profile fast/local"),
        edge("GaugeAgent.build_features", "RetrieveAgent.retrieve_elasticsearch_hybrid", "general profile full/deep"),
        edge("GaugeAgent.build_features", "ReasoningAgent.build_structured_explanation", "general profile"),
        edge("GaugeAgent.predict_single", "GaugeAgent.assess_confidence"),
        edge("GaugeAgent.predict_ddi", "GaugeAgent.assess_confidence"),
        edge("GaugeAgent.predict_single", "GaugeAgent.run_zero_shot_fallback", "debug/demo fallback"),
        edge("GaugeAgent.assess_confidence", "RetrieveAgent.retrieve_local_rag", "fast/local"),
        edge("GaugeAgent.assess_confidence", "RetrieveAgent.retrieve_elasticsearch_hybrid", "full/deep"),
        edge("GaugeAgent.assess_confidence", "RetrieveAgent.retrieve_external_evidence", "external evidence enabled"),
        edge("RetrieveAgent.retrieve_local_rag", "RetrieveAgent.expand_evidence_query", "repair/deep", "repair"),
        edge("RetrieveAgent.retrieve_elasticsearch_hybrid", "RetrieveAgent.expand_evidence_query", "repair/deep", "repair"),
        edge("RetrieveAgent.expand_evidence_query", "RetrieveAgent.retrieve_elasticsearch_hybrid", "second retrieval", "repair"),
        edge("RetrieveAgent.expand_evidence_query", "RetrieveAgent.retrieve_local_rag", "local second retrieval", "repair"),
        edge("RetrieveAgent.retrieve_local_rag", "ReasoningAgent.build_structured_explanation"),
        edge("RetrieveAgent.retrieve_elasticsearch_hybrid", "ReasoningAgent.build_structured_explanation"),
        edge("RetrieveAgent.retrieve_external_evidence", "ReasoningAgent.build_structured_explanation", "general profile"),
        edge("ReactionAgent.parse_reaction", "RetrieveAgent.retrieve_mit_template_evidence"),
        edge("ReactionAgent.parse_reaction", "RetrieveAgent.retrieve_orderly_condition_evidence"),
        edge("RetrieveAgent.retrieve_mit_template_evidence", "RetrieveAgent.retrieve_orderly_condition_evidence", "condition context"),
        edge("RetrieveAgent.retrieve_orderly_condition_evidence", "ReactionAgent.classify_reaction_type", "classification task"),
        edge("RetrieveAgent.retrieve_orderly_condition_evidence", "ReactionAgent.predict_buchwald_hartwig_yield", "yield task"),
        edge("RetrieveAgent.retrieve_orderly_condition_evidence", "ReactionAgent.build_reaction_profile", "profile/condition task"),
        edge("ReactionAgent.classify_reaction_type", "ReactionAgent.build_reaction_profile"),
        edge("ReactionAgent.predict_buchwald_hartwig_yield", "ReactionAgent.build_reaction_profile"),
        edge("ReactionAgent.build_reaction_profile", "ReasoningAgent.build_structured_explanation"),
        edge("ReactionAgent.build_reaction_profile", "ReporterAgent.build_runtime_report", "reaction profile ready"),
        edge("ReasoningAgent.build_structured_explanation", "ReasoningAgent.summarize_evidence_support", "optional"),
        edge("ReasoningAgent.build_structured_explanation", "ReporterAgent.build_runtime_report"),
        edge("ReasoningAgent.summarize_evidence_support", "ReporterAgent.build_runtime_report"),
        edge("ReporterAgent.build_runtime_report", "VerifierAgent.verify_prediction_grounding", "prediction tasks"),
        edge("ReporterAgent.build_runtime_report", "VerifierAgent.verify_citation_grounding"),
        edge("ReporterAgent.build_runtime_report", "VerifierAgent.verify_report_schema"),
        edge("VerifierAgent.verify_report_schema", "VerifierAgent.verify_citation_grounding", "schema then citation check"),
        edge("VerifierAgent.verify_citation_grounding", "VerifierAgent.verify_report_schema", "citation then schema check"),
        edge("VerifierAgent.verify_prediction_grounding", "ReflectionAgent.reflect_on_report", "verifier feedback", "repair"),
        edge("VerifierAgent.verify_citation_grounding", "ReflectionAgent.reflect_on_report", "verifier feedback", "repair"),
        edge("VerifierAgent.verify_report_schema", "ReflectionAgent.reflect_on_report", "verifier feedback", "repair"),
        edge("VerifierAgent.verify_citation_grounding", "RetrieveAgent.expand_evidence_query", "unsupported citations", "backtrack"),
        edge("VerifierAgent.verify_report_schema", "ReporterAgent.build_runtime_report", "schema repair", "backtrack"),
        edge("ReflectionAgent.reflect_on_report", "ReflectionAgent.apply_reflection_feedback", "reflection loop", "repair"),
        edge("ReflectionAgent.apply_reflection_feedback", "RetrieveAgent.expand_evidence_query", "needs query refinement", "backtrack"),
        edge("ReflectionAgent.apply_reflection_feedback", "RetrieveAgent.retrieve_elasticsearch_hybrid", "retrieve more evidence", "backtrack"),
        edge("ReflectionAgent.apply_reflection_feedback", "RetrieveAgent.retrieve_local_rag", "retrieve more local evidence", "backtrack"),
        edge("ReflectionAgent.apply_reflection_feedback", "ReasoningAgent.build_structured_explanation", "rewrite reasoning", "backtrack"),
        edge("ReflectionAgent.apply_reflection_feedback", "ReporterAgent.build_runtime_report", "rewrite report", "backtrack"),
        edge("ReporterAgent.build_runtime_report", "ReporterAgent.build_compatible_report", "optional compatibility"),
        edge("ReporterAgent.build_compatible_report", "ReporterAgent.write_artifacts"),
        edge("VerifierAgent.verify_report_schema", "ReporterAgent.write_artifacts"),
        edge("ReflectionAgent.apply_reflection_feedback", "ReporterAgent.write_artifacts", "finalize"),
    ]


def _workflow_mermaid(graph: Dict[str, Any]) -> str:
    lines = ["flowchart LR"]
    for node in graph.get("langgraph", {}).get("nodes", []):
        lines.append(f"  {_mermaid_id(node.get('id'))}[\"{_mermaid_label(node.get('label'))}\"]")
    for edge in graph.get("langgraph", {}).get("edges", []):
        lines.append(f"  {_mermaid_id(edge.get('source'))} --> {_mermaid_id(edge.get('target'))}")
    execution = graph.get("execution", {}).get("nodes", [])
    if execution:
        lines.append("  subgraph AgentExecution[\"Agent/tool execution\"]")
        for node in execution:
            label = f"{node.get('agent')}.{node.get('tool')}\\n{node.get('status') or ''}"
            lines.append(f"    {_mermaid_id(node.get('id'))}[\"{_mermaid_label(label)}\"]")
        for edge in graph.get("execution", {}).get("edges", []):
            lines.append(f"    {_mermaid_id(edge.get('source'))} --> {_mermaid_id(edge.get('target'))}")
        lines.append("  end")
    return "\n".join(lines)


def _workflow_topology_mermaid(topology: Dict[str, Any]) -> str:
    nodes = topology.get("nodes", []) or []
    edges = topology.get("edges", []) or []
    lines = ["flowchart LR"]
    for node in nodes:
        node_id = _mermaid_id(node.get("id"))
        label = f"{node.get('agent')}.{node.get('tool')}\\n{node.get('status') or ''}"
        lines.append(f"  {node_id}[\"{_mermaid_label(label)}\"]")
    for edge in edges:
        source = _mermaid_id(edge.get("source"))
        target = _mermaid_id(edge.get("target"))
        condition = str(edge.get("condition") or "")
        if condition:
            lines.append(f"  {source} -- \"{_mermaid_label(condition)}\" --> {target}")
        else:
            lines.append(f"  {source} --> {target}")
    return "\n".join(lines)


def _mermaid_id(value: Any) -> str:
    text = "".join(char if char.isalnum() else "_" for char in str(value or "node"))
    return text or "node"


def _mermaid_label(value: Any) -> str:
    return str(value or "").replace("\\", "\\\\").replace('"', '\\"')


def _calibration_uncertainty(prediction: Dict[str, Any]) -> Dict[str, Any]:
    details = prediction.get("details") if isinstance(prediction.get("details"), dict) else {}
    return {
        "calibration_status": details.get("calibration_status", "not_evaluated"),
        "calibration_method": details.get("calibration_method", "none"),
        "calibrated_probability": details.get("calibrated_probability"),
        "raw_probability": details.get("raw_probability", prediction.get("probability")),
        "confidence_level": details.get("confidence_level"),
        "warning": details.get("calibration_warning") or details.get("warning"),
        "evidence_grounded_score": (details.get("evidence_grounded_scoring") or {}).get("evidence_grounded_score")
        if isinstance(details.get("evidence_grounded_scoring"), dict)
        else None,
    }


def _ensure_molecule_visual_artifacts(report: Dict[str, Any], *, artifacts: Dict[str, Any], run_dir: Path) -> None:
    context = _read_json_if_exists(Path(artifacts.get("global_context_path") or run_dir / "global_context.json"))
    molecule, molecule_index, source_label = _select_visual_molecule(report, context)
    smiles = str((molecule or {}).get("canonical_smiles") or (molecule or {}).get("smiles") or "").strip()
    if not smiles:
        return

    visual_dir = _ensure_dir(run_dir / "visualizations")
    slug = _safe_image_slug((molecule or {}).get("name") or (molecule or {}).get("original_input") or smiles)
    molecule_image = _existing_feature_image_path(context, molecule_index, run_dir)
    if molecule_image is None:
        molecule_image = visual_dir / f"{slug}_molecule.png"
        if not molecule_image.is_file():
            _draw_molecule_png(smiles, molecule_image, legend="Molecule structure")
    if molecule_image and molecule_image.is_file():
        artifacts.setdefault("image_path", str(molecule_image))
        artifacts["molecule_image_path"] = str(molecule_image)

    highlight = _select_key_substructure_highlight(smiles, context=context, molecule_index=molecule_index)
    highlight_image = visual_dir / f"{slug}_key_substructure.png"
    if not highlight_image.is_file():
        _draw_molecule_png(
            smiles,
            highlight_image,
            highlight_atoms=highlight.get("atom_indices") or [],
            legend=highlight.get("legend") or "Key substructure",
        )
    metadata = {
        "molecule": {
            "name": (molecule or {}).get("name"),
            "original_input": (molecule or {}).get("original_input"),
            "canonical_smiles": smiles,
            "source": source_label,
        },
        "visualization_type": "heuristic_substructure_highlight",
        "note": (
            "The highlighted atoms are selected from matched chemistry rules, visual-semantic cues, "
            "or common ADMET-relevant SMARTS patterns. They are not a calibrated atom-level model attribution."
        ),
        **highlight,
    }
    metadata_path = visual_dir / f"{slug}_key_substructure.json"
    if not metadata_path.is_file():
        metadata_path.write_text(json.dumps(to_jsonable(metadata), indent=2, ensure_ascii=False), encoding="utf-8")
    artifacts["key_substructure_image_path"] = str(highlight_image)
    artifacts["key_substructure_metadata_path"] = str(metadata_path)
    artifacts["key_substructure_highlight"] = metadata


def _select_visual_molecule(report: Dict[str, Any], context: Dict[str, Any]) -> tuple[Dict[str, Any], int, str]:
    molecules = report.get("molecules") if isinstance(report.get("molecules"), list) else []
    if not molecules:
        molecules = context.get("molecules") if isinstance(context.get("molecules"), list) else []
    for idx, molecule in enumerate(molecules or []):
        if isinstance(molecule, dict) and molecule.get("is_valid") is not False and (molecule.get("canonical_smiles") or molecule.get("smiles")):
            return molecule, idx, "molecule"

    reaction = report.get("reaction") if isinstance(report.get("reaction"), dict) else {}
    if not reaction:
        reaction = context.get("reaction") if isinstance(context.get("reaction"), dict) else {}
    for field_name, source_label in [("products", "reaction_product"), ("reactants", "reaction_reactant")]:
        values = reaction.get(field_name)
        if isinstance(values, list):
            for value in values:
                if str(value or "").strip():
                    return {"name": source_label.replace("_", " "), "canonical_smiles": str(value).strip(), "is_valid": True}, 0, source_label
    reaction_smiles = str(reaction.get("reaction_smiles") or "").strip()
    if ">>" in reaction_smiles:
        products = reaction_smiles.split(">>", 1)[1].split(".")
        if products and products[0].strip():
            return {"name": "reaction product", "canonical_smiles": products[0].strip(), "is_valid": True}, 0, "reaction_product"
    return {}, 0, ""


def _existing_feature_image_path(context: Dict[str, Any], molecule_index: int, run_dir: Path) -> Optional[Path]:
    features = context.get("features") if isinstance(context.get("features"), list) else []
    if not features:
        return None
    feature = features[min(max(molecule_index, 0), len(features) - 1)]
    if not isinstance(feature, dict):
        return None
    image = feature.get("image") if isinstance(feature.get("image"), dict) else {}
    path = image.get("image_path")
    if not path:
        return None
    candidate = Path(str(path))
    if not candidate.is_absolute():
        candidate = run_dir / candidate
    return candidate if candidate.is_file() else None


def _draw_molecule_png(
    smiles: str,
    path: Path,
    *,
    highlight_atoms: Optional[List[int]] = None,
    legend: str = "",
) -> None:
    try:
        from rdkit import Chem
        from rdkit.Chem import Draw, rdDepictor
    except Exception:
        return
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return
    try:
        rdDepictor.Compute2DCoords(mol)
    except Exception:
        pass
    highlight_atoms = sorted({int(atom) for atom in (highlight_atoms or []) if isinstance(atom, int) or str(atom).isdigit()})
    highlight_bonds = _highlight_bonds_for_atoms(mol, highlight_atoms)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        Draw.MolToFile(
            mol,
            str(path),
            size=(420, 320),
            highlightAtoms=highlight_atoms,
            highlightBonds=highlight_bonds,
            legend=legend,
        )
    except Exception:
        Draw.MolToFile(mol, str(path), size=(420, 320), legend=legend)


def _highlight_bonds_for_atoms(mol: Any, atom_indices: List[int]) -> List[int]:
    atom_set = set(atom_indices)
    bonds: List[int] = []
    for bond in mol.GetBonds():
        if bond.GetBeginAtomIdx() in atom_set and bond.GetEndAtomIdx() in atom_set:
            bonds.append(int(bond.GetIdx()))
    return bonds


def _select_key_substructure_highlight(smiles: str, *, context: Dict[str, Any], molecule_index: int) -> Dict[str, Any]:
    try:
        from rdkit import Chem
        from rdkit.Chem import rdDepictor
    except Exception:
        return _empty_highlight("RDKit unavailable")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return _empty_highlight("Invalid SMILES")
    try:
        rdDepictor.Compute2DCoords(mol)
    except Exception:
        pass

    matched_rule = _matched_substructure_rule(mol, context=context, molecule_index=molecule_index)
    if matched_rule:
        return matched_rule

    cue_match = _matched_visual_or_common_pattern(mol, context=context, molecule_index=molecule_index)
    if cue_match:
        return cue_match

    rings = list(mol.GetRingInfo().AtomRings())
    if rings:
        atoms = list(max(rings, key=len))
        return {
            "label": "primary ring scaffold",
            "legend": "Key scaffold: ring system",
            "selection_source": "scaffold_fallback",
            "substructure_score": 0.45,
            "atom_indices": atoms,
            "rationale": "No matched substructure rule was available; the main ring scaffold is highlighted for orientation.",
        }
    atoms = [atom.GetIdx() for atom in list(mol.GetAtoms())[: min(4, mol.GetNumAtoms())]]
    return {
        "label": "primary molecular scaffold",
        "legend": "Key scaffold",
        "selection_source": "scaffold_fallback",
        "substructure_score": 0.35,
        "atom_indices": atoms,
        "rationale": "No matched substructure rule was available; the first connected scaffold atoms are highlighted for orientation.",
    }


def _matched_substructure_rule(mol: Any, *, context: Dict[str, Any], molecule_index: int) -> Optional[Dict[str, Any]]:
    features = context.get("features") if isinstance(context.get("features"), list) else []
    feature = features[min(max(molecule_index, 0), len(features) - 1)] if features else {}
    rule_values = {}
    if isinstance(feature, dict):
        rule_block = feature.get("llm_rule_features") if isinstance(feature.get("llm_rule_features"), dict) else {}
        rule_values = rule_block.get("values") if isinstance(rule_block.get("values"), dict) else {}
    candidates: List[Dict[str, Any]] = []
    for rule in context.get("rules", []) or []:
        if not isinstance(rule, dict):
            continue
        metadata = rule.get("metadata") if isinstance(rule.get("metadata"), dict) else {}
        if metadata.get("operation") != "substructure" or not metadata.get("smarts"):
            continue
        feature_name = str(rule.get("feature_name") or "")
        value = float(rule_values.get(feature_name, 0.0) or 0.0) if rule_values else 0.0
        if value <= 0:
            continue
        match = _first_smarts_match(mol, str(metadata.get("smarts")))
        if not match:
            continue
        label = str(metadata.get("label") or feature_name.replace("rule_has_", "").replace("_", " ") or "substructure")
        candidates.append(
            {
                "label": label,
                "legend": f"Key substructure: {label.replace('_', ' ')}",
                "selection_source": "matched_rule",
                "substructure_score": value,
                "atom_indices": list(match),
                "rationale": str(rule.get("text") or metadata.get("text") or "Matched a task rule."),
                "feature_name": feature_name,
                "smarts": metadata.get("smarts"),
            }
        )
    if not candidates:
        return None
    priority = ["carboxylic", "nitro", "sulfonamide", "amide", "amine", "halogen", "phenol", "ester", "benzene"]
    candidates.sort(key=lambda item: next((idx for idx, term in enumerate(priority) if term in item["label"].lower()), len(priority)))
    return candidates[0]


def _matched_visual_or_common_pattern(mol: Any, *, context: Dict[str, Any], molecule_index: int) -> Optional[Dict[str, Any]]:
    cue_text = _visual_cue_text(context, molecule_index=molecule_index)
    patterns = _substructure_highlight_patterns()
    ordered = sorted(
        patterns,
        key=lambda item: 0 if item["label"].lower() in cue_text else 1,
    )
    for item in ordered:
        match = _first_smarts_match(mol, item["smarts"])
        if match:
            source = "visual_semantic_cue" if item["label"].lower() in cue_text else "common_substructure_pattern"
            return {
                "label": item["label"],
                "legend": f"Key substructure: {item['label']}",
                "selection_source": source,
                "substructure_score": 0.8 if source == "visual_semantic_cue" else 0.6,
                "atom_indices": list(match),
                "rationale": item["rationale"],
                "smarts": item["smarts"],
            }
    return None


def _visual_cue_text(context: Dict[str, Any], *, molecule_index: int) -> str:
    features = context.get("features") if isinstance(context.get("features"), list) else []
    feature = features[min(max(molecule_index, 0), len(features) - 1)] if features else {}
    if not isinstance(feature, dict):
        return ""
    parts: List[str] = []
    for key in ["visual_semantic_text"]:
        if feature.get(key):
            parts.append(str(feature.get(key)))
    visual = feature.get("visual_semantics") if isinstance(feature.get("visual_semantics"), dict) else {}
    for value in visual.values():
        if isinstance(value, list):
            parts.extend(str(item) for item in value)
        elif isinstance(value, dict):
            parts.extend(f"{k} {v}" for k, v in value.items())
        elif value is not None:
            parts.append(str(value))
    return " ".join(parts).lower()


def _substructure_highlight_patterns() -> List[Dict[str, str]]:
    return [
        {"label": "carboxylic acid", "smarts": "[CX3](=O)[OX2H1]", "rationale": "Acidic or ionizable groups often dominate permeability and safety interpretation."},
        {"label": "nitro", "smarts": "[NX3](=O)=O", "rationale": "Nitro groups are common structural alerts in safety screening."},
        {"label": "sulfonamide", "smarts": "S(=O)(=O)N", "rationale": "Sulfonamide motifs can influence polarity, ionization, and binding context."},
        {"label": "amide", "smarts": "C(=O)N", "rationale": "Amide groups contribute polarity and hydrogen-bonding constraints."},
        {"label": "amine", "smarts": "[NX3;H2,H1,H0;!$(NC=O)]", "rationale": "Amines can affect ionization, solubility, and transport."},
        {"label": "ester", "smarts": "C(=O)O[#6]", "rationale": "Ester groups are relevant to polarity and metabolic liability."},
        {"label": "phenol", "smarts": "c[OH]", "rationale": "Phenolic groups add polarity and hydrogen-bonding capacity."},
        {"label": "halogen", "smarts": "[F,Cl,Br,I]", "rationale": "Halogens can modulate lipophilicity and target interactions."},
        {"label": "aromatic ring", "smarts": "a1aaaaa1", "rationale": "Aromatic scaffolds often drive hydrophobic interactions and shape."},
    ]


def _first_smarts_match(mol: Any, smarts: str) -> tuple:
    try:
        from rdkit import Chem
        from rdkit import rdBase
    except Exception:
        return ()
    with rdBase.BlockLogs():
        pattern = Chem.MolFromSmarts(smarts)
    if pattern is None:
        return ()
    matches = mol.GetSubstructMatches(pattern)
    return matches[0] if matches else ()


def _empty_highlight(reason: str) -> Dict[str, Any]:
    return {
        "label": "key substructure unavailable",
        "legend": "Key substructure unavailable",
        "selection_source": "unavailable",
        "substructure_score": 0.0,
        "atom_indices": [],
        "rationale": reason,
    }


def _safe_image_slug(value: Any) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "").strip()).strip("_")
    if text:
        return text[:64]
    return hashlib.sha1(str(value or "").encode("utf-8")).hexdigest()[:12]


def _collect_images(artifacts: Dict[str, Any], config: WebDemoConfig) -> List[Dict[str, Any]]:
    labels = {
        "image_path": "Molecule visualization",
        "molecule_image_path": "Molecule visualization",
        "key_substructure_image_path": "Key substructure highlight",
        "grad_cam_path": "Grad-CAM",
        "drug_a_image_path": "Drug A",
        "drug_b_image_path": "Drug B",
        "drug_a_grad_cam_path": "Drug A Grad-CAM",
        "drug_b_grad_cam_path": "Drug B Grad-CAM",
    }
    images: List[Dict[str, Any]] = []
    seen_urls = set()
    highlight = artifacts.get("key_substructure_highlight") if isinstance(artifacts.get("key_substructure_highlight"), dict) else {}
    for key, label in labels.items():
        path = artifacts.get(key)
        if not path:
            continue
        url = artifact_url(path, config)
        if not url or url in seen_urls:
            continue
        seen_urls.add(url)
        item = {"key": key, "label": label, "path": str(path), "url": url, "exists": True}
        if key == "key_substructure_image_path":
            item["highlight"] = highlight
        images.append(item)
    return images


def _collect_artifact_files(artifacts: Dict[str, Any], config: WebDemoConfig) -> List[Dict[str, Any]]:
    labels = {
        "multi_agent_report_path": "Multi-agent report",
        "compat_report_path": "Compatibility report",
        "reaction_report_path": "Reaction report",
        "reaction_profile_path": "Reaction profile",
        "reaction_prediction_path": "Reaction prediction",
        "reaction_condition_evidence_path": "Reaction condition evidence",
        "reaction_template_evidence_path": "Reaction template evidence",
        "reaction_resource_resolution_path": "Reaction resource resolution",
        "key_substructure_metadata_path": "Key substructure highlight metadata",
        "global_context_path": "Global context (provenance state)",
        "orchestrator_plan_path": "Orchestrator plan",
        "task_graph_path": "Task graph",
        "llm_plan_proposals_path": "LLM plan proposals",
        "plan_revision_trace_path": "Plan revision trace",
        "multi_agent_trace_path": "Multi-agent trace",
        "agent_messages_path": "Agent messages",
        "tool_calls_path": "Tool calls (audit trail)",
        "retrieval_trace_path": "Retrieval trace",
        "citation_map_path": "Citation map (claim links)",
        "verifier_report_path": "Verifier report (claim-level checks)",
        "resource_resolution_path": "Resource resolution",
        "objective_status_path": "Objective status",
        "reflection_path": "Reflection (bounded repair)",
        "report_path": "JSON report",
        "markdown_report_path": "Markdown report",
        "case_study_report_path": "Case-study report",
        "narrative_markdown_path": "Narrative report",
        "narrative_report_path": "Narrative JSON",
        "rag_evidence_markdown_path": "RAG evidence",
        "rag_evidence_path": "RAG evidence JSON",
        "agent_plan_path": "Agent plan",
        "agent_trace_path": "Agent trace",
        "agent_state_summary_path": "State summary",
        "graph_topology_path": "Graph topology",
        "checkpoint_manifest_path": "Checkpoint manifest",
        "reproducibility_log_path": "Reproducibility log",
        "api_usage_log_path": "API usage log",
        "api_usage_summary_path": "API usage summary",
    }
    files: List[Dict[str, Any]] = []
    for key, label in labels.items():
        path = artifacts.get(key)
        if not path:
            continue
        url = artifact_url(path, config)
        files.append({"key": key, "label": label, "path": str(path), "url": url, "exists": bool(url)})
    return files


def _collect_release_artifact_files(config: WebDemoConfig) -> List[Dict[str, Any]]:
    files: List[Dict[str, Any]] = []
    labels = {
        "release_manifest_path": ("Release manifest", _release_manifest_path(config)),
        "release_readme_path": ("Release README", _release_readme_path(config)),
        "release_acceptance_summary_path": ("Release acceptance summary", _release_acceptance_summary_path(config)),
        "release_qc_readout_path": ("Evidence QC readout", _release_qc_readout_path(config)),
        "release_structured_jsonl_path": ("Structured evidence JSONL", _release_structured_jsonl_path(config)),
        "release_default_index_smoke_path": ("Default-index smoke", _release_default_smoke_path(config)),
        "release_smoke_summary_path": ("Release smoke summary", _release_smoke_summary_path(config)),
    }
    for key, (label, path) in labels.items():
        if not path.is_file():
            continue
        url = artifact_url(path, config)
        files.append({"key": key, "label": label, "path": str(path), "url": url, "exists": bool(url)})
    return files


def _release_evidence_dir(config: WebDemoConfig) -> Path:
    return (config.project_root / "outputs" / RELEASE_EVIDENCE_DIR_NAME).resolve()


def _release_smoke_dir(config: WebDemoConfig) -> Path:
    return (config.project_root / "outputs" / RELEASE_SMOKE_DIR_NAME).resolve()


def _release_manifest_path(config: WebDemoConfig) -> Path:
    return _release_evidence_dir(config) / "release_manifest.json"


def _release_readme_path(config: WebDemoConfig) -> Path:
    return _release_evidence_dir(config) / "README.md"


def _release_acceptance_summary_path(config: WebDemoConfig) -> Path:
    return _release_evidence_dir(config) / RELEASE_ACCEPTANCE_SUMMARY_NAME


def _release_qc_readout_path(config: WebDemoConfig) -> Path:
    return _release_evidence_dir(config) / RELEASE_QC_READOUT_NAME


def _release_structured_jsonl_path(config: WebDemoConfig) -> Path:
    return _release_evidence_dir(config) / RELEASE_STRUCTURED_JSONL_NAME


def _release_default_smoke_path(config: WebDemoConfig) -> Path:
    return _release_evidence_dir(config) / "acceptance_default_index_smoke.json"


def _release_smoke_summary_path(config: WebDemoConfig) -> Path:
    return _release_smoke_dir(config) / "release_smoke_summary.json"


def _artifact_candidate(encoded_path: str, config: WebDemoConfig) -> Path:
    rel = Path(unquote(encoded_path))
    if rel.is_absolute():
        return rel.resolve()
    candidate = (config.project_root / rel).resolve()
    if not candidate.exists():
        candidate = (_resolved_output_dir(config).parent / rel).resolve()
    return candidate


def _resolve_artifact_path(path: str, project_root: Path) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = project_root / candidate
    return candidate.resolve()


def _is_allowed_artifact(path: Path, config: WebDemoConfig) -> bool:
    resolved = path.resolve()
    if _is_relative_to(resolved, _resolved_output_dir(config)):
        return True
    release_allowed = {
        _release_manifest_path(config).resolve(),
        _release_readme_path(config).resolve(),
        _release_acceptance_summary_path(config).resolve(),
        _release_qc_readout_path(config).resolve(),
        _release_structured_jsonl_path(config).resolve(),
        _release_default_smoke_path(config).resolve(),
        _release_smoke_summary_path(config).resolve(),
    }
    return resolved in release_allowed


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _next_run_dir(output_dir: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return _ensure_dir(_resolved_path(output_dir, RUNTIME_ROOT) / f"{stamp}_{uuid4().hex[:8]}")


def _resolved_output_dir(config: WebDemoConfig) -> Path:
    return _resolved_path(config.output_dir, config.project_root)


def _resolved_path(path: Path | str, root: Path) -> Path:
    target = Path(path).expanduser()
    if not target.is_absolute():
        target = root / target
    return target.resolve()


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _resource_path(path: str) -> str:
    candidate = Path(path)
    if candidate.is_absolute():
        return str(candidate)
    return str((RESOURCE_ROOT / candidate).resolve())


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _event(status: str, message: str) -> Dict[str, Any]:
    return {"time": _now_iso(), "status": status, "message": message}


def _call_progress(progress_callback: Optional[Any], event: Dict[str, Any]) -> None:
    if progress_callback is None:
        return
    try:
        progress_callback(event)
    except Exception:
        pass


def _normalize_run_mode(value: Any, default: str = "custom") -> str:
    text = str(value or "").strip().lower()
    if text in {"fast", "full", "custom"}:
        return text
    return default


def _run_mode_defaults(run_mode: str) -> Dict[str, Any]:
    if run_mode == "fast":
        return {
            "mode": "fast",
            "retrieval_backend": "local",
            "auto_model": True,
            "require_real_model": False,
            "require_real_rag": False,
            "require_llm_orchestrator": False,
            "require_llm_reasoning": False,
            "use_llm_orchestrator": True,
            "use_llm_reasoning": False,
            "enable_fallbacks": True,
            "enable_reflection": True,
            "min_evidence": 0,
            "use_mllm_visual_semantic": False,
            "use_external_evidence": False,
            "use_rag_evidence": False,
            "allow_network_lookup": True,
        }
    if run_mode == "full":
        return {
            "mode": "full",
            "retrieval_backend": "auto",
            "auto_model": True,
            "require_real_model": False,
            "require_real_rag": False,
            "require_llm_orchestrator": False,
            "require_llm_reasoning": False,
            "use_llm_orchestrator": True,
            "use_llm_reasoning": True,
            "enable_fallbacks": True,
            "enable_reflection": True,
            "min_evidence": 1,
            "use_mllm_visual_semantic": False,
            "use_external_evidence": False,
            "use_rag_evidence": True,
            "allow_network_lookup": True,
        }
    return {}


def _compact_progress_event(event: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "time": _now_iso(),
        "event_type": event.get("event_type"),
        "step_id": event.get("step_id") or event.get("graph_node"),
        "tool_name": event.get("tool_name"),
        "graph_node": event.get("graph_node"),
        "description": event.get("description"),
        "status": event.get("status"),
        "duration_sec": event.get("duration_sec"),
        "message": event.get("message"),
        "error": event.get("error"),
        "fallback_tool": event.get("fallback_tool") or event.get("fallback_used"),
        "output_summary": _compact_for_status(event.get("output_summary")),
    }


def _compact_for_status(value: Any, max_chars: int = 360) -> Any:
    if value is None:
        return None
    if isinstance(value, dict):
        return {str(key): _compact_for_status(item, max_chars=max_chars) for key, item in list(value.items())[:12]}
    if isinstance(value, list):
        return [_compact_for_status(item, max_chars=max_chars) for item in value[:8]]
    if isinstance(value, (str, int, float, bool)):
        text = str(value)
        if len(text) > max_chars:
            return text[:max_chars] + "..."
        return value
    return str(value)[:max_chars]


def _summarize_step_progress(step_events: List[Dict[str, Any]], run_mode: str) -> Dict[str, Any]:
    latest_by_step: Dict[str, Dict[str, Any]] = {}
    planned_steps = 0
    for event in step_events:
        if event.get("event_type") == "plan_ready":
            output = event.get("output_summary") if isinstance(event.get("output_summary"), dict) else {}
            try:
                planned_steps = max(planned_steps, int(output.get("total_steps") or 0))
            except (TypeError, ValueError):
                pass
        step_id = str(event.get("step_id") or event.get("graph_node") or "")
        if step_id:
            latest_by_step[step_id] = event
    statuses = [str(event.get("status") or "") for event in latest_by_step.values()]
    expected_total_steps = max(
        planned_steps + WEB_PROGRESS_PRE_AGENT_STEPS if planned_steps else 0,
        len(latest_by_step),
        WEB_PROGRESS_DEFAULT_TOTAL_STEPS,
    )
    return {
        "run_mode": run_mode,
        "total_seen_steps": len(latest_by_step),
        "planned_agent_steps": planned_steps,
        "expected_total_steps": expected_total_steps,
        "completed_steps": sum(status in {"completed", "fallback_completed"} for status in statuses),
        "running_steps": sum(status in {"running", "fallback_running"} for status in statuses),
        "failed_steps": sum(status == "failed" for status in statuses),
        "skipped_steps": sum(status == "skipped" for status in statuses),
    }


def _progress_message(event: Dict[str, Any]) -> str:
    step = event.get("step_id") or event.get("tool_name") or "step"
    status = event.get("status") or "running"
    duration = event.get("duration_sec")
    if duration is None:
        return f"{step}: {status}"
    return f"{step}: {status} ({duration:.1f}s)"


def _empty_to_none(value: Any) -> Optional[str]:
    text = str(value or "").strip()
    return text or None


def _bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return bool(default)


@contextmanager
def temporary_env(updates: Dict[str, Optional[str]]) -> Iterator[None]:
    old_values = {key: os.environ.get(key) for key in updates}
    try:
        for key, value in updates.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        yield
    finally:
        for key, value in old_values.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


INDEX_HTML = (Path(__file__).with_name("templates") / "index.html").read_text(encoding="utf-8")


app = create_app()


if __name__ == "__main__":
    raise SystemExit(main())
