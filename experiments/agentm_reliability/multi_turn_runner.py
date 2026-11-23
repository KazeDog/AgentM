#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
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
    from .run_matrix import _controller_kwargs, _extract_model_paths, _finalize_parent_failure, _seed_everything, _usage_context
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
    from run_matrix import _controller_kwargs, _extract_model_paths, _finalize_parent_failure, _seed_everything, _usage_context
    from runtime_guards import api_budget_guard, effective_api_budget, run_real_rag_preflight
    from validation import runtime_retrieval_backend, run_invariant_violations, validate_result_record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run append-only AgentM multi-turn state-update experiments.")
    parser.add_argument("config", nargs="?", default=str(RELIABILITY_ROOT / "configs" / "multi_turn_tasks.yaml"))
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--in-process", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--run-one", default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def build_jobs(config: dict, selected: set[str], max_cases: int | None, *, shard_index: int = 0, num_shards: int = 1) -> list[dict]:
    manifest = resolve_path(config["manifest"])
    cases = load_jsonl(manifest, required_fields=["case_id", "turns", "gold_plan"])
    if selected:
        cases = [case for case in cases if str(case["case_id"]) in selected]
        missing = selected - {str(case["case_id"]) for case in cases}
        if missing:
            raise ValueError(f"Unknown multi-turn cases: {sorted(missing)}")
    if max_cases is not None:
        cases = cases[: max(0, max_cases)]
    if num_shards < 1 or shard_index < 0 or shard_index >= num_shards:
        raise ValueError("Require num_shards >= 1 and 0 <= shard_index < num_shards")
    cases = cases[shard_index::num_shards]
    config_hash = stable_hash({key: value for key, value in config.items() if key != "_meta"})
    source_hash = source_tree_hash()
    jobs: list[dict] = []
    for case in cases:
        for orchestrator in config["matrix"]["orchestrators"]:
            for reflection in config["matrix"]["reflection"]:
                job = {
                    "case": case,
                    "case_id": str(case["case_id"]),
                    "dataset_group": "multi_turn",
                    "orchestrator": str(orchestrator),
                    "reflection_enabled": bool(reflection),
                    "seed": int(config["experiment"]["seed"]),
                    "repeat_index": 0,
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


def run_parent(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    layout = ensure_output_layout(config.get("outputs", {}).get("root") or config["paths"]["outputs_root"])
    jobs = build_jobs(config, set(args.case), args.max_cases, shard_index=args.shard_index, num_shards=args.num_shards)
    if args.dry_run:
        print(json.dumps({"n_jobs": len(jobs), "run_ids": [job["run_id"] for job in jobs]}, indent=2))
        return 0
    failures = 0
    timeout = int(config["experiment"].get("timeout_seconds", 300)) * int(config["multi_turn_tasks"].get("turns_per_case", 2))
    for index, job in enumerate(jobs, start=1):
        run_id = job["run_id"]
        raw = layout["raw_runs"] / run_id
        if (raw / "run_record.json").exists():
            existing = read_json(raw / "run_record.json", default={}) or {}
            if existing.get("config_hash") != job["config_hash"] or existing.get("source_tree_hash") != job["source_tree_hash"]:
                raise RuntimeError(f"Immutable run ID collision with different provenance: {run_id}")
            print(json.dumps({"index": index, "n_jobs": len(jobs), "run_id": run_id, "status": "immutable_raw_run_exists"}))
            continue
        job_path = layout["_jobs"] / f"{run_id}.json"
        write_json(job_path, job)
        if args.in_process:
            code, output = run_worker(job_path), "in-process"
        else:
            try:
                completed = subprocess.run(
                    [sys.executable, str(Path(__file__).resolve()), args.config, "--run-one", str(job_path)],
                    cwd=str(PROJECT_ROOT),
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    timeout=timeout,
                    check=False,
                )
                code, output = int(completed.returncode), completed.stdout
            except subprocess.TimeoutExpired:
                _finalize_parent_failure(job, layout, kind="timeout", message=f"multi_turn_timeout_sec_exceeded:{timeout}")
                code, output = 124, "multi-turn worker timed out"
            except Exception as exc:
                _finalize_parent_failure(job, layout, kind="worker_launch_failure", message=str(exc))
                code, output = 2, traceback.format_exc()
        if code != 0 and not (raw / "run_record.json").exists():
            _finalize_parent_failure(
                job,
                layout,
                kind="worker_nonzero_exit",
                message=f"exit_code={code}; output_tail={(output or '')[-4000:]}",
            )
        (layout["_jobs"] / f"{run_id}.stdout.log").write_text(output or "", encoding="utf-8")
        record = read_json(raw / "run_record.json", default={}) or {}
        print(json.dumps({"index": index, "n_jobs": len(jobs), "run_id": run_id, "completed": record.get("completed"), "worker_exit": code}))
        if code != 0 or not record.get("completed"):
            failures += 1
    _refresh_metrics(layout)
    print(json.dumps({"n_jobs": len(jobs), "n_failed_or_incomplete": failures, "output_root": str(layout["root"])}, indent=2))
    return 0 if failures == 0 else 2


def run_worker(job_path: str | Path) -> int:
    job = read_json(job_path)
    if isinstance(job, dict) and "reflection_depth" in job:
        from experiments.agentm_reliability.reflection_depth import validate_depth_job
        validate_depth_job(job, executing=True)
    config = job["config"]
    layout = ensure_output_layout(config.get("outputs", {}).get("root") or config["paths"]["outputs_root"])
    run_id = str(job["run_id"])
    raw = layout["raw_runs"] / run_id
    if raw.exists():
        return 0
    stage = layout["_staging"] / f"{run_id}.pending"
    if stage.exists():
        _finalize_parent_failure(job, layout, kind="abandoned_staging", message="pre-existing incomplete staging directory")
        return 2
    stage.mkdir(parents=True)
    write_json(stage / "job.json", job, overwrite=False)
    current_source_hash = source_tree_hash()
    turns_root = stage / "turns"
    turns_root.mkdir()
    started_at = utc_now()
    started = time.perf_counter()
    _seed_everything(int(job["seed"]))
    os.environ["AGENTM_LLM_TIMEOUT"] = str(config["experiment"].get("llm_timeout_seconds", 60))
    previous_context: dict[str, Any] | None = None
    turn_records: list[dict[str, Any]] = []
    turn_context_paths: list[str] = []
    final_context: dict[str, Any] = {}
    final_report: dict[str, Any] = {}
    final_hook: ControlledInjectionHook | None = None
    try:
        if current_source_hash != job.get("source_tree_hash"):
            raise RuntimeError(
                "source_tree_changed_after_job_creation: "
                f"expected={job.get('source_tree_hash')} observed={current_source_hash}"
            )
        runtime_config = config["runtime"]
        max_llm_calls, max_total_tokens = effective_api_budget(job)
        with api_budget_guard(
            stage,
            max_calls=max_llm_calls,
            max_total_tokens=max_total_tokens,
        ):
            from agentm.runtime import AgentMMultiAgentController

            prepared_turns: list[tuple[int, Mapping[str, Any], Path, ControlledInjectionHook, Any, str]] = []
            for turn_index, turn in enumerate(job["case"]["turns"], start=1):
                turn_dir = turns_root / f"turn_{turn_index:02d}"
                hook = ControlledInjectionHook({})
                if "reflection_depth" in job:
                    from experiments.agentm_reliability.reflection_depth import DepthSnapshotHook
                    hook = DepthSnapshotHook({}, job["reflection_depth"])
                turn_job = {**job, "case": turn}
                controller = AgentMMultiAgentController(**_controller_kwargs(turn_job, turn_dir, hook))
                request = str(turn["request"])
                if turn.get("instruction"):
                    request = f"{turn['instruction']}\n{request}"
                run_real_rag_preflight(
                    controller.runtime.hybrid_retriever,
                    case=turn,
                    runtime=runtime_config,
                    output_path=turn_dir / "real_rag_preflight.json",
                    label=f"{run_id}:turn_{turn_index:02d}",
                    reaction_runtime=getattr(controller.runtime, "reaction_runtime", None),
                )
                prepared_turns.append((turn_index, turn, turn_dir, hook, controller, request))

            for turn_index, turn, turn_dir, hook, controller, request in prepared_turns:
                final_hook = hook
                turn_started = utc_now()
                clock = time.perf_counter()
                with _usage_context(turn_dir):
                    report = controller.run(
                        request,
                        task_type=str(turn.get("task_type") or ""),
                        dataset_name=str(turn.get("dataset_name") or "") or None,
                        previous_run_context=previous_context,
                    )
                context = read_json(turn_dir / "global_context.json", default={}) or {}
                invariant_violations = run_invariant_violations(job, context)
                if "reflection_depth" in job:
                    from experiments.agentm_reliability.reflection_depth import audit_depth_protocol
                    audit = audit_depth_protocol(job, context, hook.export())
                    write_json(turn_dir / "reflection_depth_acceptance.json", audit, overwrite=False)
                    write_json(turn_dir / "reflection_depth_snapshots.json", hook.export(), overwrite=False)
                    invariant_violations.extend(audit["errors"])
                if invariant_violations:
                    raise RuntimeError("run invariant violation(s): " + ", ".join(invariant_violations))
                usage = read_json(turn_dir / "api_usage" / "api_usage_summary.json", default={}) or {}
                tool_calls = list(context.get("tool_calls", []) or [])
                turn_records.append(
                    {
                        "turn": turn_index,
                        "started_at": turn_started,
                        "finished_at": utc_now(),
                        "latency_seconds": round(time.perf_counter() - clock, 6),
                        "request": request,
                        "task_type": turn.get("task_type"),
                        "dataset_name": turn.get("dataset_name"),
                        "completed": bool(context and report),
                        "tool_call_count": len(tool_calls),
                        "reflection_step_count": sum(c.get("agent") == "ReflectionAgent" for c in tool_calls),
                        "retrieval_call_count": sum(
                            1
                            for call in tool_calls
                            if call.get("agent") == "RetrieveAgent" or "retrieve" in str(call.get("tool") or "")
                        ),
                        "llm_call_count": int(usage.get("n_calls", 0) or 0),
                        "llm_failed_call_count": int(usage.get("n_failed_calls", 0) or 0),
                        "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
                        "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
                        "total_tokens": int(usage.get("total_tokens", 0) or 0),
                        "objective_status": (context.get("objective_status", {}) or {}).get("status"),
                    }
                )
                final_context, final_report = context, report
                previous_context = context
                turn_context_paths.append(str(layout["raw_runs"] / run_id / "turns" / f"turn_{turn_index:02d}" / "global_context.json"))

        runtime = stage / "runtime"
        runtime.mkdir()
        for source_name, dest_name in [
            ("global_context.json", "global_context.json"),
            ("multi_agent_report.json", "multi_agent_report.json"),
            ("multi_agent_trace.json", "multi_agent_trace.json"),
        ]:
            source = turns_root / f"turn_{len(turn_records):02d}" / source_name
            if source.exists():
                shutil.copy2(source, runtime / dest_name)
        write_json(stage / "turn_records.json", turn_records, overwrite=False)
        write_json(stage / "turn_context_paths.json", turn_context_paths, overwrite=False)
        record = _build_record(job, layout, final_context, final_report, turn_records, started_at, time.perf_counter() - started, final_hook)
        write_json(stage / "run_record.json", record, overwrite=False)
        stage.rename(raw)
        _publish(layout, run_id)
        return 0
    except Exception as exc:
        error = {"failure_kind": "multi_turn_worker_exception", "error": str(exc), "traceback": traceback.format_exc(), "finished_at": utc_now()}
        write_json(stage / "error.json", error, overwrite=False)
        write_json(stage / "turn_records.json", turn_records, overwrite=False)
        write_json(stage / "turn_context_paths.json", turn_context_paths, overwrite=False)
        record = _build_record(job, layout, final_context, final_report, turn_records, started_at, time.perf_counter() - started, final_hook, failure=error)
        write_json(stage / "run_record.json", record, overwrite=False)
        stage.rename(raw)
        _publish(layout, run_id)
        return 2


def _build_record(
    job: Mapping[str, Any],
    layout: Mapping[str, Path],
    context: Mapping[str, Any],
    report: Mapping[str, Any],
    turns: list[Mapping[str, Any]],
    started_at: str,
    latency: float,
    hook: ControlledInjectionHook | None,
    failure: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    run_id = str(job["run_id"])
    final_raw = layout["raw_runs"] / run_id
    tool_calls = [call for turn in turns for call in []]  # counts are stored per turn below
    final_tools = context.get("tool_calls", []) or []
    total_tools = sum(int(turn.get("tool_call_count", 0)) for turn in turns)
    total_llm = sum(int(turn.get("llm_call_count", 0)) for turn in turns)
    total_failed_llm = sum(int(turn.get("llm_failed_call_count", 0)) for turn in turns)
    total_prompt_tokens = sum(int(turn.get("prompt_tokens", 0)) for turn in turns)
    total_completion_tokens = sum(int(turn.get("completion_tokens", 0)) for turn in turns)
    total_tokens = sum(int(turn.get("total_tokens", 0)) for turn in turns)
    total_retrieval = sum(int(turn.get("retrieval_call_count", 0)) for turn in turns)
    reflection_steps = sum(1 for call in final_tools if call.get("agent") == "ReflectionAgent")
    runtime_model_paths = _extract_model_paths(context)
    verifier = context.get("verifier_report", {}) or {}
    objectives = context.get("objective_status", {}) or {}
    preflights = [
        read_json(path, default={}) or {}
        for path in sorted((layout["_staging"] / f"{run_id}.pending" / "turns").glob("turn_*/real_rag_preflight.json"))
    ]
    runtime_config = job["config"]["runtime"]
    max_llm_calls, max_total_tokens = effective_api_budget(job)
    record = {
        "run_id": run_id,
        "case_id": str(job["case_id"]),
        "dataset_group": "multi_turn",
        "task_type": "multi_turn_state_update",
        "orchestrator": str(job["orchestrator"]),
        "reflection_enabled": bool(job["reflection_enabled"]),
        "seed": int(job["seed"]),
        "repeat_index": 0,
        "started_at": started_at,
        "finished_at": utc_now(),
        "latency_seconds": round(latency, 6),
        "completed": failure is None and len(turns) == len(job["case"]["turns"]) and all(turn.get("completed") for turn in turns),
        "objective_passed": objectives.get("status") == "pass",
        "verifier_passed": (verifier.get("summary", {}) or {}).get("status") == "pass",
        "model_record_created": bool(context.get("predictions") or context.get("reaction_predictions")),
        "citation_map_created": bool(context.get("citation_map")),
        "reflection_step_count": reflection_steps,
        "report_changed_by_reflection": hook.report_changed_by_reflection() if hook else None,
        "llm_call_count": total_llm,
        "llm_failed_call_count": total_failed_llm,
        "prompt_tokens": total_prompt_tokens,
        "completion_tokens": total_completion_tokens,
        "total_tokens": total_tokens,
        "tool_call_count": total_tools,
        "retrieval_call_count": total_retrieval,
        "unsupported_claim_count": int((verifier.get("citations", {}) or {}).get("n_unsupported", 0) or 0),
        "external_service_error_count": sum(1 for call in final_tools if call.get("status") == "failed"),
        "final_report_hash": stable_hash(report) if report else None,
        "global_context_path": str(final_raw / "runtime" / "global_context.json") if context else "",
        "final_report_path": str(final_raw / "runtime" / "multi_agent_report.json") if report else "",
        "trace_path": str(final_raw / "runtime" / "multi_agent_trace.json") if final_tools else "",
        "error_path": str(final_raw / "error.json") if failure else "",
        "code_commit": CODE_COMMIT,
        "source_tree_hash": str(job.get("source_tree_hash") or source_tree_hash()),
        "config_hash": str(job["config_hash"]),
        "source_manifest": str(job["source_manifest"]),
        "case_hash": stable_hash(job["case"]),
        "runtime_model_path": runtime_model_paths[0] if runtime_model_paths else None,
        "runtime_model_paths": runtime_model_paths,
        "runtime_feature_set": job["config"]["runtime"].get("feature_set"),
        "runtime_retrieval_backend": runtime_retrieval_backend(context),
        "runtime_preflight_passed": len(preflights) == len(job["case"]["turns"]) and all(
            bool(item.get("passed")) for item in preflights
        ),
        "runtime_preflight_paths": [
            str(final_raw / "turns" / f"turn_{index:02d}" / "real_rag_preflight.json")
            for index in range(1, len(preflights) + 1)
        ],
        "llm_call_budget": max_llm_calls,
        "llm_token_budget": max_total_tokens,
        "llm_budget_exceeded": "api_budget_exceeded" in str((failure or {}).get("error") or ""),
        "temperature": float(job["config"]["experiment"].get("temperature", 0.0)),
        "timeout_seconds": int(job["config"]["experiment"].get("timeout_seconds", 300)),
        "max_retries": int(job["config"]["experiment"].get("max_retries", 0)),
        "turn_count": len(turns),
        "expected_turn_count": len(job["case"]["turns"]),
        "failure_kind": failure.get("failure_kind") if failure else None,
        "error": failure.get("error") if failure else None,
    }
    if "reflection_depth" in job:
        record.update(reflection_depth=job["reflection_depth"], reflection_protocol=job["reflection_protocol"],
                      common_config_hash=job["common_config_hash"], reflection_depth_scope="per_conversation_turn",
                      reflection_step_count=sum(int(t.get("reflection_step_count", 0)) for t in turns))
    validate_result_record(record)
    return record


def _publish(layout: Mapping[str, Path], run_id: str) -> None:
    raw = layout["raw_runs"] / run_id
    publish_artifact(raw / "runtime" / "global_context.json", layout["global_contexts"] / f"{run_id}.json")
    publish_artifact(raw / "runtime" / "multi_agent_report.json", layout["reports"] / f"{run_id}.json")
    publish_artifact(raw / "runtime" / "multi_agent_trace.json", layout["traces"] / f"{run_id}.json")
    publish_artifact(raw / "error.json", layout["errors"] / f"{run_id}.json")


def _refresh_metrics(layout: Mapping[str, Path]) -> None:
    rows = []
    for path in sorted(layout["raw_runs"].glob("*/run_record.json")):
        row = read_json(path, default={}) or {}
        if row.get("dataset_group") == "multi_turn":
            rows.append(row)
    write_csv(layout["metrics"] / "multi_turn_case_level.csv", rows)


def main() -> int:
    args = parse_args()
    if args.run_one:
        return run_worker(args.run_one)
    return run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
