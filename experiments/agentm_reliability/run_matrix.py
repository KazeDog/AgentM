#!/usr/bin/env python
from __future__ import annotations

import argparse
from contextlib import nullcontext
from copy import deepcopy
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
import traceback
from typing import Any, Mapping

try:
    from .common import (
        CODE_COMMIT,
        PROJECT_ROOT,
        RELIABILITY_ROOT,
        ensure_output_layout,
        load_config,
        load_jsonl,
        publish_artifact,
        read_json,
        resolve_path,
        run_id_for,
        source_tree_hash,
        stable_hash,
        utc_now,
        write_csv,
        write_json,
    )
    from .injections import ControlledInjectionHook
    from .runtime_guards import api_budget_guard, effective_api_budget, run_real_rag_preflight
    from .validation import runtime_retrieval_backend, run_invariant_violations, validate_result_record
except ImportError:
    from common import (
        CODE_COMMIT,
        PROJECT_ROOT,
        RELIABILITY_ROOT,
        ensure_output_layout,
        load_config,
        load_jsonl,
        publish_artifact,
        read_json,
        resolve_path,
        run_id_for,
        source_tree_hash,
        stable_hash,
        utc_now,
        write_csv,
        write_json,
    )
    from injections import ControlledInjectionHook
    from runtime_guards import api_budget_guard, effective_api_budget, run_real_rag_preflight
    from validation import runtime_retrieval_backend, run_invariant_violations, validate_result_record


RESULT_FIELDS = [
    "run_id",
    "case_id",
    "dataset_group",
    "task_type",
    "orchestrator",
    "reflection_enabled",
    "seed",
    "started_at",
    "finished_at",
    "latency_seconds",
    "completed",
    "objective_passed",
    "verifier_passed",
    "reflection_step_count",
    "llm_call_count",
    "tool_call_count",
    "retrieval_call_count",
    "unsupported_claim_count",
    "external_service_error_count",
    "global_context_path",
    "final_report_path",
    "trace_path",
    "error_path",
    "code_commit",
    "source_tree_hash",
    "config_hash",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run append-only AgentM reliability experiment matrices.")
    parser.add_argument("config", nargs="?", default=str(RELIABILITY_ROOT / "configs" / "clean_matrix.yaml"))
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--include-stability-repeats", action="store_true")
    parser.add_argument("--in-process", action="store_true", help="Debug only; production uses timeout-isolated workers.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--run-one", default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def _effective_config_hash(config: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in config.items() if key != "_meta"}
    return stable_hash(payload)


def build_jobs(
    config: dict,
    *,
    selected: set[str],
    max_cases: int | None,
    include_repeats: bool,
    shard_index: int = 0,
    num_shards: int = 1,
) -> list[dict]:
    group = str(config.get("dataset_group") or "")
    if group not in {"clean", "stress", "open_task"}:
        raise ValueError(f"run_matrix supports clean, stress, or open_task; received {group!r}")
    manifest = resolve_path(config["manifest"])
    cases = load_jsonl(manifest, required_fields=["case_id", "dataset_group"])
    if selected:
        cases = [case for case in cases if str(case["case_id"]) in selected]
        missing = selected - {str(case["case_id"]) for case in cases}
        if missing:
            raise ValueError(f"Unknown case IDs: {sorted(missing)}")
    if max_cases is not None:
        cases = cases[: max(0, max_cases)]
    if num_shards < 1 or shard_index < 0 or shard_index >= num_shards:
        raise ValueError("Require num_shards >= 1 and 0 <= shard_index < num_shards")
    cases = cases[shard_index::num_shards]

    experiment = config["experiment"]
    seed = int(experiment["seed"])
    config_hash = _effective_config_hash(config)
    source_hash = source_tree_hash()
    jobs: list[dict] = []
    for case in cases:
        for orchestrator in config["matrix"]["orchestrators"]:
            for reflection in config["matrix"]["reflection"]:
                repeat_indices = [0]
                if include_repeats and group == "clean":
                    repeat_spec = config.get("stability_repeats", {})
                    selected_count = int(repeat_spec.get("selected_case_count", 20))
                    repeats = int(repeat_spec.get("repeats", 3))
                    if int(case.get("case_index", 0)) < selected_count:
                        repeat_indices = list(range(max(1, repeats)))
                for repeat_index in repeat_indices:
                    job = {
                        "case": case,
                        "case_id": str(case["case_id"]),
                        "dataset_group": group,
                        "orchestrator": str(orchestrator),
                        "reflection_enabled": bool(reflection),
                        "seed": seed,
                        "repeat_index": repeat_index,
                        "config": config,
                        "config_hash": config_hash,
                        "source_tree_hash": source_hash,
                        "source_manifest": str(manifest),
                        "runtime_guard_protocol": (
                            "pre_llm_real_rag_fail_closed"
                            if bool(config.get("runtime", {}).get("require_real_rag", True))
                            else None
                        ),
                    }
                    job["run_id"] = run_id_for(job, config_hash)
                    jobs.append(job)
    return jobs


def _raw_record(layout: Mapping[str, Path], run_id: str) -> dict | None:
    return read_json(layout["raw_runs"] / run_id / "run_record.json")


def run_parent(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    layout = ensure_output_layout(config.get("outputs", {}).get("root") or config["paths"]["outputs_root"])
    jobs = build_jobs(
        config,
        selected=set(args.case),
        max_cases=args.max_cases,
        include_repeats=args.include_stability_repeats,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
    )
    if args.dry_run:
        print(json.dumps({"n_jobs": len(jobs), "run_ids": [job["run_id"] for job in jobs]}, ensure_ascii=False, indent=2))
        return 0

    failures = 0
    timeout_seconds = int(config["experiment"].get("timeout_seconds", 300))
    for index, job in enumerate(jobs, start=1):
        run_id = job["run_id"]
        existing = _raw_record(layout, run_id)
        if existing is not None:
            if existing.get("config_hash") != job["config_hash"]:
                raise RuntimeError(f"Immutable run ID collision with a different config hash: {run_id}")
            if existing.get("source_tree_hash") != job["source_tree_hash"]:
                raise RuntimeError(f"Immutable run ID collision with a different source hash: {run_id}")
            print(json.dumps({"index": index, "n_jobs": len(jobs), "run_id": run_id, "status": "immutable_raw_run_exists"}))
            continue
        job_path = layout["_jobs"] / f"{run_id}.json"
        write_json(job_path, job)
        if args.in_process:
            code = run_worker(job_path)
            output = "in-process"
        else:
            command = [sys.executable, str(Path(__file__).resolve()), args.config, "--run-one", str(job_path)]
            try:
                completed = subprocess.run(
                    command,
                    cwd=str(PROJECT_ROOT),
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    timeout=timeout_seconds,
                    check=False,
                )
                code = int(completed.returncode)
                output = completed.stdout
            except subprocess.TimeoutExpired as exc:
                output = (exc.stdout or "") + (exc.stderr or "") if isinstance(exc.stdout, str) else ""
                _finalize_parent_failure(job, layout, kind="timeout", message=f"case_timeout_sec_exceeded:{timeout_seconds}")
                code = 124
            except Exception as exc:
                output = traceback.format_exc()
                _finalize_parent_failure(job, layout, kind="worker_launch_failure", message=str(exc))
                code = 2
        if code != 0 and _raw_record(layout, run_id) is None:
            tail = (output or "")[-4000:]
            _finalize_parent_failure(job, layout, kind="worker_nonzero_exit", message=f"exit_code={code}; output_tail={tail}")
        (layout["_jobs"] / f"{run_id}.stdout.log").write_text(output or "", encoding="utf-8")
        record = _raw_record(layout, run_id) or {}
        status = "completed" if record.get("completed") else str(record.get("failure_kind") or "failed")
        print(json.dumps({"index": index, "n_jobs": len(jobs), "run_id": run_id, "status": status, "worker_exit": code}))
        if code != 0 or not record.get("completed"):
            failures += 1
    _refresh_basic_metrics(layout, str(config.get("dataset_group")))
    print(json.dumps({"n_jobs": len(jobs), "n_failed_or_incomplete": failures, "output_root": str(layout["root"])}, indent=2))
    return 0 if failures == 0 else 2


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def _controller_kwargs(job: Mapping[str, Any], runtime_dir: Path, hook: ControlledInjectionHook) -> dict[str, Any]:
    config = job["config"]
    runtime = config["runtime"]
    paths = config["paths"]
    case = job["case"]
    base = case.get("base_case") if isinstance(case.get("base_case"), Mapping) else case
    model_path = base.get("model_path") if isinstance(base, Mapping) else None
    if model_path:
        model_path = str(resolve_path(model_path))
    if "reflection_depth" in job:
        from experiments.agentm_reliability.reflection_depth import bound_model_path
        model_path = bound_model_path(job) or model_path
    reaction_model_dir = resolve_path(paths["reaction_model_dir"]) if paths.get("reaction_model_dir") else None
    reaction_index_dir = resolve_path(paths["reaction_index_dir"]) if paths.get("reaction_index_dir") else None
    orchestrator = str(job["orchestrator"])
    return {
        "output_dir": runtime_dir,
        "feature_set": str(base.get("feature_set") or runtime.get("feature_set", "all_features")),
        "model_path": model_path,
        "auto_model": bool(runtime.get("auto_model", True)),
        "require_real_model": bool(runtime.get("require_real_model", True)),
        "require_real_rag": bool(runtime.get("require_real_rag", True)),
        "require_llm_orchestrator": bool(orchestrator == "llm" and runtime.get("require_llm_orchestrator", True)),
        "min_evidence": int(runtime.get("min_evidence", 1)),
        "mode": str(runtime.get("mode", "full")),
        "retrieval_backend": str(runtime.get("retrieval_backend", "auto")),
        "es_url": runtime.get("es_url"),
        "es_index_name": runtime.get("es_index_name"),
        "max_rounds": int(runtime.get("max_rounds", 20)),
        "temperature": float(config["experiment"].get("temperature", 0.0)),
        "llm_profile": runtime.get("llm_profile"),
        "use_llm_orchestrator": orchestrator == "llm",
        "llm_orchestrator_max_retries": int(runtime.get("llm_orchestrator_max_retries", 0)),
        "llm_orchestrator_retry_backoff_seconds": float(
            runtime.get("llm_orchestrator_retry_backoff_seconds", 0.0)
        ),
        "use_llm_reasoning": bool(runtime.get("use_llm_reasoning", False)),
        "require_llm_reasoning": bool(runtime.get("require_llm_reasoning", False)),
        "enable_fallbacks": bool(runtime.get("enable_fallbacks", False)),
        "enable_reflection": bool(job["reflection_enabled"]),
        "max_reflection_rounds": int(runtime.get("max_reflection_rounds", 1)),
        "rag_embedding_model_name": str(
            runtime.get("rag_embedding_model_name", "checkpoints/rag/scibert_scivocab_uncased")
        ),
        "reaction_data_root": paths.get("reaction_data_root"),
        "reaction_model_dir": reaction_model_dir,
        "reaction_index_dir": reaction_index_dir,
        "experiment_hook": hook,
    }


def _runtime_case(job: Mapping[str, Any]) -> Mapping[str, Any]:
    case = job["case"]
    if job["dataset_group"] == "stress":
        return case["base_case"]
    return case


def _open_task_contract(case: Mapping[str, Any]) -> dict[str, Any] | None:
    """Build a runtime composition contract without exposing benchmark gold."""

    open_type = str(case.get("open_task_type") or "")
    if not open_type:
        return None
    base = case.get("base_case") if isinstance(case.get("base_case"), Mapping) else {}
    root_task = str(case.get("task_type") or base.get("task_type") or "")
    root_dataset = str(case.get("dataset_name") or base.get("dataset_name") or "")
    contract: dict[str, Any] = {
        "schema": "agentm_open_task_contract",
        "open_task_type": open_type,
        "request": str(case.get("request") or ""),
        "root_task_type": root_task,
        "root_dataset_name": root_dataset,
        "subtasks": [],
    }
    if open_type == "multi_endpoint_admet":
        molecule = str(base.get("request") or "").strip()
        endpoints = [str(item) for item in case.get("requested_endpoints", []) or [] if item]
        contract["scientific_objects"] = [molecule]
        contract["subtasks"] = [
            {
                "subtask_id": f"single::{index + 1}::{endpoint}",
                "kind": "single",
                "task_type": "admet_prediction",
                "dataset_name": endpoint,
                "molecules": [molecule],
                "evidence_scope": "single_drug",
            }
            for index, endpoint in enumerate(endpoints)
        ]
    elif open_type == "ddi_plus_individual_safety":
        pair = [part.strip() for part in str(base.get("request") or "").split(",", 1) if part.strip()]
        while len(pair) < 2:
            pair.append("unspecified_partner")
        pair = pair[:2]
        contract["scientific_objects"] = pair
        # Resolve the single-drug objectives first.  This leaves the final
        # active molecule state as the pair and also satisfies the natural
        # dependency that an individual molecule be resolved before the
        # combined DDI conclusion is produced.
        contract["subtasks"] = [
            {
                "subtask_id": f"single::{pair[0]}::hERG",
                "kind": "single",
                "task_type": "admet_prediction",
                "dataset_name": "hERG",
                "molecules": [pair[0]],
                "evidence_scope": "single_drug",
            },
            {
                "subtask_id": f"single::{pair[1]}::DILI",
                "kind": "single",
                "task_type": "admet_prediction",
                "dataset_name": "DILI",
                "molecules": [pair[1]],
                "evidence_scope": "single_drug",
            },
            {
                "subtask_id": f"pair::{pair[0]}::{pair[1]}::DDI",
                "kind": "ddi",
                "task_type": "ddi_prediction",
                "dataset_name": "DDI",
                "molecules": pair,
                "evidence_scope": "pair_level",
            },
        ]
    elif open_type == "reaction_multi_tool":
        contract["scientific_objects"] = [str(base.get("request") or case.get("request") or "")]
        contract["operations"] = [
            "reaction_classification",
            "yield_prediction",
            "condition_evidence",
            "template_evidence",
        ]
    else:
        raise ValueError(f"Unsupported open_task_type: {open_type}")
    return contract


def _usage_context(runtime_dir: Path):
    try:
        from agentm.utils.api_usage import api_usage_context

        return api_usage_context(runtime_dir)
    except Exception:
        return nullcontext(runtime_dir / "api_usage")


def run_worker(job_path: str | Path) -> int:
    job = read_json(job_path)
    if not isinstance(job, dict):
        raise ValueError(f"Invalid worker job: {job_path}")
    if "reflection_depth" in job:
        from experiments.agentm_reliability.reflection_depth import validate_depth_job
        validate_depth_job(job, executing=True)
    config = job["config"]
    layout = ensure_output_layout(config.get("outputs", {}).get("root") or config["paths"]["outputs_root"])
    run_id = str(job["run_id"])
    raw_dir = layout["raw_runs"] / run_id
    if raw_dir.exists():
        return 0
    stage = layout["_staging"] / f"{run_id}.pending"
    if stage.exists():
        _finalize_parent_failure(job, layout, kind="abandoned_staging", message="pre-existing incomplete staging directory")
        return 2
    stage.mkdir(parents=True, exist_ok=False)
    runtime_dir = stage / "runtime"
    runtime_dir.mkdir()
    write_json(stage / "job.json", job, overwrite=False)
    source_hash = source_tree_hash()
    started_at = utc_now()
    start_clock = time.perf_counter()
    hook = ControlledInjectionHook(job["case"])
    if "reflection_depth" in job:
        from experiments.agentm_reliability.reflection_depth import DepthSnapshotHook
        hook = DepthSnapshotHook(job["case"], job["reflection_depth"])
    _seed_everything(int(job["seed"]) + int(job.get("repeat_index", 0)))
    os.environ["AGENTM_LLM_TIMEOUT"] = str(config["experiment"].get("llm_timeout_seconds", 60))
    try:
        if source_hash != job.get("source_tree_hash"):
            raise RuntimeError(
                "source_tree_changed_after_job_creation: "
                f"expected={job.get('source_tree_hash')} observed={source_hash}"
            )
        runtime_config = config["runtime"]
        max_llm_calls, max_total_tokens = effective_api_budget(job)
        with api_budget_guard(
            stage,
            max_calls=max_llm_calls,
            max_total_tokens=max_total_tokens,
        ):
            from agentm.runtime import AgentMMultiAgentController

            runtime_case = _runtime_case(job)
            controller = AgentMMultiAgentController(**_controller_kwargs(job, runtime_dir, hook))
            run_real_rag_preflight(
                controller.runtime.hybrid_retriever,
                case=runtime_case,
                runtime=runtime_config,
                output_path=runtime_dir / "real_rag_preflight.json",
                label=f"{run_id}:single_turn",
                reaction_runtime=getattr(controller.runtime, "reaction_runtime", None),
            )
            with _usage_context(runtime_dir):
                report = controller.run(
                    str(runtime_case["request"]),
                    task_type=str(runtime_case.get("task_type") or ""),
                    dataset_name=str(runtime_case.get("dataset_name") or "") or None,
                    task_contract=_open_task_contract(job["case"]) if job.get("dataset_group") == "open_task" else None,
                )
        finished_at = utc_now()
        latency = time.perf_counter() - start_clock
        context = read_json(runtime_dir / "global_context.json", default={}) or {}
        invariant_violations = run_invariant_violations(job, context)
        if "reflection_depth" in job:
            from experiments.agentm_reliability.reflection_depth import audit_depth_protocol
            depth_audit = audit_depth_protocol(job, context, hook.export())
            write_json(stage / "reflection_depth_acceptance.json", depth_audit, overwrite=False)
            invariant_violations.extend(depth_audit["errors"])
        if invariant_violations:
            raise RuntimeError("run invariant violation(s): " + ", ".join(invariant_violations))
        write_json(stage / "hook_record.json", hook.export(), overwrite=False)
        if hook.pre_injection_context is not None:
            write_json(stage / "pre_injection_context.json", hook.pre_injection_context, overwrite=False)
        if hook.post_injection_context is not None:
            write_json(stage / "post_injection_context.json", hook.post_injection_context, overwrite=False)
        if hook.runtime_gold is not None:
            write_json(stage / "runtime_gold.json", hook.runtime_gold, overwrite=False)
        record = _build_record(
            job,
            layout,
            context=context,
            report=report,
            started_at=started_at,
            finished_at=finished_at,
            latency=latency,
            source_hash=source_hash,
            hook=hook,
            failure=None,
        )
        write_json(stage / "run_record.json", record, overwrite=False)
        _commit_stage(stage, raw_dir)
        _publish_run(layout, run_id)
        return 0
    except Exception as exc:
        finished_at = utc_now()
        latency = time.perf_counter() - start_clock
        failure = {
            "failure_kind": "worker_exception",
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "started_at": started_at,
            "finished_at": finished_at,
        }
        write_json(stage / "error.json", failure, overwrite=False)
        context = read_json(runtime_dir / "global_context.json", default={}) or {}
        report = read_json(runtime_dir / "multi_agent_report.json", default={}) or {}
        record = _build_record(
            job,
            layout,
            context=context,
            report=report,
            started_at=started_at,
            finished_at=finished_at,
            latency=latency,
            source_hash=source_hash,
            hook=hook,
            failure=failure,
        )
        write_json(stage / "hook_record.json", hook.export(), overwrite=False)
        write_json(stage / "run_record.json", record, overwrite=False)
        _commit_stage(stage, raw_dir)
        _publish_run(layout, run_id)
        return 2


def _build_record(
    job: Mapping[str, Any],
    layout: Mapping[str, Path],
    *,
    context: Mapping[str, Any],
    report: Mapping[str, Any],
    started_at: str,
    finished_at: str,
    latency: float,
    source_hash: str,
    hook: ControlledInjectionHook,
    failure: Mapping[str, Any] | None,
) -> dict[str, Any]:
    run_id = str(job["run_id"])
    final_raw = layout["raw_runs"] / run_id
    tool_calls = list(context.get("tool_calls", []) or report.get("tool_calls", []) or [])
    verifier = context.get("verifier_report", {}) or report.get("verifier_report", {}) or {}
    objectives = context.get("objective_status", {}) or report.get("objective_status", {}) or {}
    citations = context.get("citation_map", []) or report.get("citation_map", []) or []
    predictions = context.get("reaction_predictions", []) or context.get("predictions", []) or report.get("predictions", []) or []
    summary = verifier.get("summary", {}) if isinstance(verifier, Mapping) else {}
    unsupported = verifier.get("unsupported_claims", []) if isinstance(verifier, Mapping) else []
    if not unsupported and isinstance(verifier, Mapping):
        unsupported_count = int((verifier.get("citations", {}) or {}).get("n_unsupported", 0) or 0)
    else:
        unsupported_count = len(unsupported or [])
    usage = read_json(final_raw / "runtime" / "api_usage" / "api_usage_summary.json", default=None)
    if usage is None:
        # The directory is still staged when this record is assembled.
        usage = read_json(layout["_staging"] / f"{run_id}.pending" / "runtime" / "api_usage" / "api_usage_summary.json", default={}) or {}
    error_calls = [call for call in tool_calls if call.get("status") in {"failed", "guard_blocked", "guard_revision_blocked"} or call.get("error")]
    external_markers = ("timeout", "unavailable", "service", "http", "elasticsearch", "pubchem", "llm", "connection")
    external_errors = sum(1 for call in error_calls if any(marker in str(call.get("error") or "").lower() for marker in external_markers))
    reflection_calls = [call for call in tool_calls if call.get("agent") == "ReflectionAgent"]
    retrieval_calls = [call for call in tool_calls if call.get("agent") == "RetrieveAgent" or "retrieve" in str(call.get("tool") or "")]
    runtime_model_paths = _extract_model_paths(context)
    preflight = read_json(
        layout["_staging"] / f"{run_id}.pending" / "runtime" / "real_rag_preflight.json",
        default={},
    ) or {}
    runtime_config = job["config"]["runtime"]
    max_llm_calls, max_total_tokens = effective_api_budget(job)
    context_exists = bool(context)
    completed = failure is None and bool(report) and context_exists
    case = job["case"]
    runtime_case = _runtime_case(job)
    record = {
        "run_id": run_id,
        "case_id": str(job["case_id"]),
        "dataset_group": str(job["dataset_group"]),
        "task_type": str(runtime_case.get("task_type") or ""),
        "orchestrator": str(job["orchestrator"]),
        "reflection_enabled": bool(job["reflection_enabled"]),
        "seed": int(job["seed"]),
        "repeat_index": int(job.get("repeat_index", 0)),
        "started_at": started_at,
        "finished_at": finished_at,
        "latency_seconds": round(float(latency), 6),
        "completed": completed,
        "objective_passed": objectives.get("status") == "pass",
        "verifier_passed": summary.get("status") == "pass",
        "model_record_created": bool(predictions),
        "citation_map_created": bool(citations),
        "reflection_step_count": len(reflection_calls),
        "report_changed_by_reflection": hook.report_changed_by_reflection(),
        "llm_call_count": int(usage.get("n_calls", 0) or 0),
        "llm_failed_call_count": int(usage.get("n_failed_calls", 0) or 0),
        "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
        "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
        "total_tokens": int(usage.get("total_tokens", 0) or 0),
        "tool_call_count": len(tool_calls),
        "retrieval_call_count": len(retrieval_calls),
        "unsupported_claim_count": unsupported_count,
        "external_service_error_count": external_errors,
        "final_report_hash": stable_hash(report) if report else None,
        "global_context_path": str(final_raw / "runtime" / "global_context.json") if context_exists else "",
        "final_report_path": str(final_raw / "runtime" / "multi_agent_report.json") if report else "",
        "trace_path": str(final_raw / "runtime" / "multi_agent_trace.json") if tool_calls else "",
        "error_path": str(final_raw / "error.json") if failure else "",
        "code_commit": CODE_COMMIT,
        "source_tree_hash": source_hash,
        "config_hash": str(job["config_hash"]),
        "source_manifest": str(job.get("source_manifest") or ""),
        "case_hash": stable_hash(case),
        "runtime_model_path": runtime_model_paths[0] if runtime_model_paths else runtime_case.get("model_path"),
        "runtime_model_paths": runtime_model_paths,
        "runtime_feature_set": runtime_case.get("feature_set") or job["config"]["runtime"].get("feature_set"),
        "runtime_retrieval_backend": runtime_retrieval_backend(context),
        "runtime_preflight_passed": bool(preflight.get("passed")),
        "runtime_preflight_path": str(final_raw / "runtime" / "real_rag_preflight.json") if preflight else "",
        "llm_call_budget": max_llm_calls,
        "llm_token_budget": max_total_tokens,
        "llm_budget_exceeded": "api_budget_exceeded" in str((failure or {}).get("error") or ""),
        "temperature": float(job["config"]["experiment"].get("temperature", 0.0)),
        "timeout_seconds": int(job["config"]["experiment"].get("timeout_seconds", 300)),
        "max_retries": int(job["config"]["experiment"].get("max_retries", 0)),
        "llm_orchestrator_max_retries": int(runtime_config.get("llm_orchestrator_max_retries", 0)),
        "llm_orchestrator_retry_backoff_seconds": float(
            runtime_config.get("llm_orchestrator_retry_backoff_seconds", 0.0)
        ),
        "injection_type": case.get("injection_type"),
        "injection_applied": hook.applied,
        "injection_record_path": str(final_raw / "hook_record.json") if (case.get("injection_type") or case.get("transient_fault")) else "",
        "failure_kind": failure.get("failure_kind") if failure else None,
        "error": failure.get("error") if failure else None,
    }
    if "reflection_depth" in job:
        record.update(reflection_depth=job["reflection_depth"], reflection_protocol=job["reflection_protocol"],
                      common_config_hash=job["common_config_hash"])
    missing = [field for field in RESULT_FIELDS if field not in record]
    if missing:
        raise AssertionError(f"run record missing required fields: {missing}")
    validate_result_record(record)
    return record


def _extract_model_paths(context: Mapping[str, Any]) -> list[str]:
    """Collect actual model artifacts recorded by executed prediction tools."""

    paths: list[str] = []

    def visit(value: Any) -> None:
        if isinstance(value, Mapping):
            model_path = value.get("model_path")
            if isinstance(model_path, str) and model_path and model_path not in paths:
                paths.append(model_path)
            selected_path = value.get("selected_path")
            if isinstance(selected_path, str) and selected_path and selected_path not in paths:
                paths.append(selected_path)
            for item in value.values():
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    for call in context.get("tool_calls", []) or []:
        if call.get("agent") in {"GaugeAgent", "ReactionAgent"}:
            visit(call.get("observation") or {})
    return paths


def _commit_stage(stage: Path, raw_dir: Path) -> None:
    if raw_dir.exists():
        raise FileExistsError(f"Refusing to overwrite immutable raw run: {raw_dir}")
    stage.rename(raw_dir)


def _publish_run(layout: Mapping[str, Path], run_id: str) -> None:
    raw = layout["raw_runs"] / run_id
    publish_artifact(raw / "runtime" / "global_context.json", layout["global_contexts"] / f"{run_id}.json")
    publish_artifact(raw / "runtime" / "multi_agent_report.json", layout["reports"] / f"{run_id}.json")
    publish_artifact(raw / "runtime" / "multi_agent_trace.json", layout["traces"] / f"{run_id}.json")
    publish_artifact(raw / "error.json", layout["errors"] / f"{run_id}.json")


def _finalize_parent_failure(job: Mapping[str, Any], layout: Mapping[str, Path], *, kind: str, message: str) -> None:
    run_id = str(job["run_id"])
    raw = layout["raw_runs"] / run_id
    if raw.exists():
        return
    stage = layout["_staging"] / f"{run_id}.pending"
    stage.mkdir(parents=True, exist_ok=True)
    now = utc_now()
    error = {"failure_kind": kind, "error": message, "traceback": "", "started_at": None, "finished_at": now}
    error_path = stage / "error.json"
    if not error_path.exists():
        write_json(error_path, error, overwrite=False)
    record_path = stage / "run_record.json"
    if not record_path.exists():
        final_raw = layout["raw_runs"] / run_id
        runtime_case = _runtime_case(job)
        record = {
            "run_id": run_id,
            "case_id": str(job["case_id"]),
            "dataset_group": str(job["dataset_group"]),
            "task_type": str(runtime_case.get("task_type") or ""),
            "orchestrator": str(job["orchestrator"]),
            "reflection_enabled": bool(job["reflection_enabled"]),
            "seed": int(job["seed"]),
            "repeat_index": int(job.get("repeat_index", 0)),
            "started_at": None,
            "finished_at": now,
            "latency_seconds": None,
            "completed": False,
            "objective_passed": False,
            "verifier_passed": False,
            "model_record_created": False,
            "citation_map_created": False,
            "reflection_step_count": 0,
            "report_changed_by_reflection": None,
            "llm_call_count": 0,
            "tool_call_count": 0,
            "retrieval_call_count": 0,
            "unsupported_claim_count": 0,
            "external_service_error_count": 1,
            "global_context_path": "",
            "final_report_path": "",
            "trace_path": "",
            "error_path": str(final_raw / "error.json"),
            "code_commit": CODE_COMMIT,
            "source_tree_hash": str(job.get("source_tree_hash") or source_tree_hash()),
            "config_hash": str(job["config_hash"]),
            "source_manifest": str(job.get("source_manifest") or ""),
            "case_hash": stable_hash(job["case"]),
            "temperature": float(job["config"]["experiment"].get("temperature", 0.0)),
            "timeout_seconds": int(job["config"]["experiment"].get("timeout_seconds", 300)),
            "max_retries": int(job["config"]["experiment"].get("max_retries", 0)),
            "injection_type": job["case"].get("injection_type"),
            "injection_applied": False,
            "failure_kind": kind,
            "error": message,
        }
        write_json(record_path, record, overwrite=False)
    _commit_stage(stage, raw)
    _publish_run(layout, run_id)


def _refresh_basic_metrics(layout: Mapping[str, Path], group: str) -> None:
    filenames = {
        "clean": "clean_case_level.csv",
        "stress": "stress_case_level.csv",
        "open_task": "open_task_case_level.csv",
    }
    if group not in filenames:
        return
    rows = []
    for path in sorted(layout["raw_runs"].glob("*/run_record.json")):
        record = read_json(path, default={}) or {}
        if record.get("dataset_group") == group:
            rows.append(record)
    write_csv(layout["metrics"] / filenames[group], rows)


def main() -> int:
    args = parse_args()
    if args.run_one:
        return run_worker(args.run_one)
    return run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
