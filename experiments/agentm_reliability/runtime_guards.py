from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import json
from pathlib import Path
import threading
import time
from typing import Any, Iterator, Mapping

try:
    from .common import utc_now, write_json
except ImportError:
    from common import utc_now, write_json


REAL_RAG_PROVIDER = "elasticsearch_hybrid"
REAL_RAG_BACKEND = "elasticsearch_hybrid_bm25_dense"
REACTION_RAG_PROVIDER = "RetrieveAgent"
REACTION_RAG_BACKEND = "local_reaction_data"


class RealRAGPreflightError(RuntimeError):
    """Raised before any LLM call when strict real-RAG readiness is not proven."""


class ApiBudgetExceeded(RuntimeError):
    """Raised at an API-call boundary when the configured per-run budget is exhausted."""


def effective_api_budget(job: Mapping[str, Any]) -> tuple[int | None, int | None]:
    """Return the enforced budget; heuristic orchestration is strictly zero-LLM."""

    if str(job.get("orchestrator") or "") == "heuristic":
        return 0, 0
    runtime = (job.get("config", {}) or {}).get("runtime", {}) or {}
    max_calls = runtime.get("max_llm_calls_per_run")
    max_tokens = runtime.get("max_total_tokens_per_run")
    return (
        int(max_calls) if max_calls is not None else None,
        int(max_tokens) if max_tokens is not None else None,
    )


def is_reaction_case(case: Mapping[str, Any]) -> bool:
    base = case.get("base_case") if isinstance(case.get("base_case"), Mapping) else case
    values = {
        str(case.get("task_type") or "").lower(),
        str(base.get("task_type") or "").lower(),
    }
    return any(value.startswith("reaction") for value in values)


def _preflight_query(case: Mapping[str, Any]) -> str:
    parts = [
        str(case.get("dataset_name") or ""),
        str(case.get("task_type") or ""),
        str(case.get("expected_endpoint") or ""),
        str(case.get("instruction") or ""),
        str(case.get("request") or ""),
        "direct scientific evidence citation",
    ]
    return " ".join(part.strip() for part in parts if part and part.strip())


def run_real_rag_preflight(
    retriever: Any,
    *,
    case: Mapping[str, Any],
    runtime: Mapping[str, Any],
    output_path: str | Path,
    label: str,
    reaction_runtime: Any | None = None,
) -> dict[str, Any]:
    """Exercise the task-appropriate real retrieval path before any LLM call."""

    target = Path(output_path)
    required = bool(runtime.get("require_real_rag", True))
    requested_backend = str(runtime.get("retrieval_backend") or "")
    reaction_mode = is_reaction_case(case)
    payload: dict[str, Any] = {
        "generated_at": utc_now(),
        "label": label,
        "required": required,
        "requested_backend": requested_backend,
        "retrieval_mode": "reaction_data" if reaction_mode else "elasticsearch_dense_hybrid",
        "expected_provider": REACTION_RAG_PROVIDER if reaction_mode else REAL_RAG_PROVIDER,
        "expected_backend": REACTION_RAG_BACKEND if reaction_mode else REAL_RAG_BACKEND,
        "passed": False,
        "reasons": [],
    }
    if not required:
        payload.update({"passed": True, "skipped": True})
        write_json(target, payload, overwrite=False)
        return payload

    reasons: list[str] = []
    if reaction_mode:
        started = time.perf_counter()
        try:
            if reaction_runtime is None:
                reasons.append("reaction_runtime_missing")
            else:
                runtime_case = (
                    case.get("base_case")
                    if isinstance(case.get("base_case"), Mapping)
                    else case
                )
                resolution = reaction_runtime.resolve_resources()
                payload["reaction_resource_resolution"] = resolution
                if not bool(resolution.get("all_required_present")):
                    reasons.append("reaction_resources_incomplete")
                probe_context: dict[str, Any] = {
                    "request": str(runtime_case.get("request") or ""),
                    "reaction_request": str(runtime_case.get("request") or ""),
                    "task_type": str(runtime_case.get("task_type") or "reaction_profile"),
                    "dataset_name": str(runtime_case.get("dataset_name") or "reaction"),
                    "metadata": {},
                }
                parse_result = reaction_runtime.parse_reaction({}, probe_context)
                if str(getattr(parse_result, "status", "")) != "completed":
                    reasons.append(
                        "reaction_parse_failed:" + str(getattr(parse_result, "error", "unknown"))
                    )
                else:
                    probe_args = {
                        "top_k": int(runtime.get("preflight_top_k", 3)),
                        "max_rows": int(runtime.get("reaction_preflight_max_rows", 12000)),
                    }
                    template_result = reaction_runtime.retrieve_mit_template_evidence(
                        probe_args,
                        probe_context,
                    )
                    condition_result = reaction_runtime.retrieve_orderly_condition_evidence(
                        probe_args,
                        probe_context,
                    )
                    for name, result in (
                        ("mit_template", template_result),
                        ("orderly_condition", condition_result),
                    ):
                        observation = getattr(result, "observation", {}) or {}
                        if str(getattr(result, "status", "")) != "completed":
                            reasons.append(
                                f"{name}_retrieval_failed:{getattr(result, 'error', 'unknown')}"
                            )
                        if int(observation.get("n_evidence_items", 0) or 0) < 1:
                            reasons.append(f"{name}_evidence_empty")
                trace = [
                    item
                    for item in (probe_context.get("retrieval_trace", []) or [])
                    if isinstance(item, Mapping)
                ]
                evidence = list(probe_context.get("evidence", []) or [])
                warnings = [str(value) for value in (probe_context.get("warnings", []) or [])]
                qualifying_trace = [
                    item
                    for item in trace
                    if str(item.get("provider") or "") == REACTION_RAG_PROVIDER
                    and str(item.get("backend") or "") == REACTION_RAG_BACKEND
                    and bool(item.get("evidence_items"))
                    and not bool(item.get("warnings"))
                ]
                payload.update(
                    {
                        "provider": REACTION_RAG_PROVIDER,
                        "backend": REACTION_RAG_BACKEND,
                        "retrieved_context_count": len(trace),
                        "qualifying_retrieval_count": len(qualifying_trace),
                        "evidence_item_count": len(evidence),
                        "warnings": warnings,
                    }
                )
                if len(qualifying_trace) < 2:
                    reasons.append("reaction_retrieval_trace_missing_both_real_sources")
                if not evidence:
                    reasons.append("reaction_evidence_items_empty")
                if warnings and bool(runtime.get("preflight_reject_warnings", True)):
                    reasons.append("reaction_retrieval_warnings_present")
        except Exception as exc:
            reasons.append(f"reaction_retrieval_exception:{type(exc).__name__}:{exc}")
        payload["latency_seconds"] = round(time.perf_counter() - started, 6)
        payload["reasons"] = sorted(set(reasons))
        payload["passed"] = not reasons
        write_json(target, payload, overwrite=False)
        if reasons:
            raise RealRAGPreflightError(
                "real_rag_preflight_failed:" + ";".join(payload["reasons"])
            )
        return payload

    if requested_backend != "es":
        reasons.append(f"retrieval_backend={requested_backend or 'missing'}")

    if hasattr(retriever, "request_timeout_seconds"):
        retriever.request_timeout_seconds = float(runtime.get("es_request_timeout_seconds", 60.0))
    if hasattr(retriever, "max_retries"):
        retriever.max_retries = int(runtime.get("es_max_retries", 1))

    query = _preflight_query(case)
    payload["query"] = query
    started = time.perf_counter()
    try:
        result = retriever.retrieve(
            query,
            metadata={},
            backend="es",
            top_k=int(runtime.get("preflight_top_k", 3)),
        )
        result_payload = result.to_dict() if hasattr(result, "to_dict") else dict(result)
        provider = str(result_payload.get("provider") or "")
        backend = str(result_payload.get("backend") or "")
        context = list(result_payload.get("retrieved_context") or [])
        evidence = list(result_payload.get("evidence_items") or [])
        warnings = [str(value) for value in (result_payload.get("warnings") or [])]
        payload.update(
            {
                "provider": provider,
                "backend": backend,
                "retrieved_context_count": len(context),
                "evidence_item_count": len(evidence),
                "warnings": warnings,
                "retrieval_metadata": result_payload.get("metadata") or {},
            }
        )
        if provider != REAL_RAG_PROVIDER:
            reasons.append(f"provider={provider or 'missing'}")
        if backend != REAL_RAG_BACKEND:
            reasons.append(f"backend={backend or 'missing'}")
        if not context:
            reasons.append("retrieved_context_empty")
        if not evidence:
            reasons.append("evidence_items_empty")
        if warnings and bool(runtime.get("preflight_reject_warnings", True)):
            reasons.append("retrieval_warnings_present")
    except Exception as exc:
        reasons.append(f"retrieval_exception:{type(exc).__name__}:{exc}")
    payload["latency_seconds"] = round(time.perf_counter() - started, 6)
    payload["reasons"] = sorted(set(reasons))
    payload["passed"] = not reasons
    write_json(target, payload, overwrite=False)
    if reasons:
        raise RealRAGPreflightError("real_rag_preflight_failed:" + ";".join(payload["reasons"]))
    return payload


@dataclass
class ApiBudget:
    root: Path
    max_calls: int | None
    max_total_tokens: int | None

    def __post_init__(self) -> None:
        self._lock = threading.Lock()

    def usage(self) -> dict[str, int]:
        calls = 0
        total_tokens = 0
        for path in self.root.rglob("api_calls.jsonl"):
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except OSError:
                continue
            for line in lines:
                try:
                    record = json.loads(line)
                except Exception:
                    continue
                if not isinstance(record, Mapping):
                    continue
                calls += 1
                total_tokens += int(record.get("total_tokens", 0) or 0)
        return {"calls": calls, "total_tokens": total_tokens}

    def check_before_call(self) -> None:
        with self._lock:
            usage = self.usage()
            if self.max_calls is not None and usage["calls"] >= self.max_calls:
                raise ApiBudgetExceeded(
                    f"api_budget_exceeded:calls={usage['calls']}>=max_calls={self.max_calls}"
                )
            if self.max_total_tokens is not None and usage["total_tokens"] >= self.max_total_tokens:
                raise ApiBudgetExceeded(
                    "api_budget_exceeded:"
                    f"total_tokens={usage['total_tokens']}>=max_total_tokens={self.max_total_tokens}"
                )

    def check_after_call(self) -> None:
        with self._lock:
            usage = self.usage()
            if self.max_calls is not None and usage["calls"] > self.max_calls:
                raise ApiBudgetExceeded(
                    f"api_budget_exceeded:calls={usage['calls']}>max_calls={self.max_calls}"
                )
            if self.max_total_tokens is not None and usage["total_tokens"] > self.max_total_tokens:
                raise ApiBudgetExceeded(
                    "api_budget_exceeded:"
                    f"total_tokens={usage['total_tokens']}>max_total_tokens={self.max_total_tokens}"
                )


@contextmanager
def api_budget_guard(
    root: str | Path,
    *,
    max_calls: int | None,
    max_total_tokens: int | None,
) -> Iterator[ApiBudget]:
    """Guard LLM/MLLM client call boundaries without changing the shared package."""

    budget = ApiBudget(
        root=Path(root),
        max_calls=int(max_calls) if max_calls is not None else None,
        max_total_tokens=int(max_total_tokens) if max_total_tokens is not None else None,
    )
    if budget.max_calls is None and budget.max_total_tokens is None:
        yield budget
        return

    from agentm.runtime.compat import ensure_import_paths

    ensure_import_paths()
    from agentm.llm import provider as llm_provider
    from agentm.mllm import provider as mllm_provider

    targets = [
        (llm_provider.LLMClient, "chat_json"),
        (mllm_provider.MLLMClient, "image_json"),
    ]
    originals: list[tuple[Any, str, Any]] = []
    for cls, method_name in targets:
        original = getattr(cls, method_name)

        def guarded(client: Any, *args: Any, _original: Any = original, **kwargs: Any) -> Any:
            budget.check_before_call()
            try:
                result = _original(client, *args, **kwargs)
            except Exception:
                budget.check_after_call()
                raise
            budget.check_after_call()
            return result

        originals.append((cls, method_name, original))
        setattr(cls, method_name, guarded)
    try:
        yield budget
    finally:
        for cls, method_name, original in reversed(originals):
            setattr(cls, method_name, original)
