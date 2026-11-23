from __future__ import annotations

from contextlib import contextmanager
import contextvars
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import threading
import time
from typing import Any, Dict, Iterable, Iterator, Mapping, Optional
from uuid import uuid4

from agentm.utils.io import ensure_dir, write_json


_USAGE_DIR: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("agentm_api_usage_dir", default=None)
_LOCK = threading.Lock()


def current_api_usage_dir() -> Optional[Path]:
    value = _USAGE_DIR.get() or os.getenv("AGENTM_API_USAGE_DIR")
    return Path(value) if value else None


@contextmanager
def api_usage_context(output_dir: str | Path) -> Iterator[Path]:
    usage_dir = ensure_dir(Path(output_dir) / "api_usage")
    _initialize_files(usage_dir)
    token = _USAGE_DIR.set(str(usage_dir))
    try:
        yield usage_dir
    finally:
        write_usage_summary(usage_dir)
        _USAGE_DIR.reset(token)


def usage_artifact_paths(output_dir: str | Path) -> Dict[str, str]:
    usage_dir = Path(output_dir) / "api_usage"
    return {
        "api_usage_dir": str(usage_dir),
        "api_usage_log_path": str(usage_dir / "api_calls.jsonl"),
        "api_usage_summary_path": str(usage_dir / "api_usage_summary.json"),
    }


def record_api_call(
    *,
    provider: str,
    profile: str,
    model: str,
    call_type: str,
    status: str,
    started_at: float,
    messages: Optional[Iterable[Mapping[str, Any]]] = None,
    prompt: Optional[str] = None,
    response_text: Optional[str] = None,
    usage: Optional[Mapping[str, Any]] = None,
    endpoint: Optional[str] = None,
    temperature: Optional[float] = None,
    error: Optional[str] = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    usage_dir = current_api_usage_dir()
    if usage_dir is None:
        return None
    ensure_dir(usage_dir)

    prompt_chars = _text_length(messages=messages, prompt=prompt)
    output_chars = len(response_text or "")
    token_usage = _normalize_usage(usage)
    usage_source = "provider" if token_usage else "estimated"
    if not token_usage:
        token_usage = {
            "prompt_tokens": _estimate_tokens(prompt_chars),
            "completion_tokens": _estimate_tokens(output_chars),
            "total_tokens": _estimate_tokens(prompt_chars) + _estimate_tokens(output_chars),
        }
    cost = _estimate_cost(model=model, usage=token_usage)
    record = {
        "call_id": str(uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "provider": provider,
        "profile": profile,
        "model": model,
        "call_type": call_type,
        "status": status,
        "endpoint": _safe_endpoint(endpoint),
        "temperature": temperature,
        "latency_sec": round(max(0.0, time.perf_counter() - started_at), 6),
        "usage_source": usage_source,
        "prompt_chars": prompt_chars,
        "response_chars": output_chars,
        "prompt_tokens": int(token_usage.get("prompt_tokens") or 0),
        "completion_tokens": int(token_usage.get("completion_tokens") or 0),
        "total_tokens": int(token_usage.get("total_tokens") or 0),
        "estimated_cost": cost,
        "error": _compact_error(error),
        "extra": dict(extra or {}),
    }
    with _LOCK:
        _append_jsonl(usage_dir / "api_calls.jsonl", record)
        write_usage_summary(usage_dir)
    return record


def write_usage_summary(usage_dir: str | Path) -> Dict[str, Any]:
    path = Path(usage_dir)
    records = _read_jsonl(path / "api_calls.jsonl")
    by_provider: Dict[str, Dict[str, Any]] = {}
    by_model: Dict[str, Dict[str, Any]] = {}
    total = _empty_rollup()
    for record in records:
        _add_record(total, record)
        _add_record(by_provider.setdefault(str(record.get("provider") or "unknown"), _empty_rollup()), record)
        _add_record(by_model.setdefault(str(record.get("model") or "unknown"), _empty_rollup()), record)
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_calls": total["n_calls"],
        "n_completed_calls": total["n_completed_calls"],
        "n_failed_calls": total["n_failed_calls"],
        "prompt_tokens": total["prompt_tokens"],
        "completion_tokens": total["completion_tokens"],
        "total_tokens": total["total_tokens"],
        "usage_sources": total["usage_sources"],
        "estimated_costs": total["estimated_costs"],
        "by_provider": by_provider,
        "by_model": by_model,
        "notes": [
            "Provider token usage is used when present in the API response.",
            "If provider usage is absent, token counts are estimated from text length and marked usage_source=estimated.",
            "Cost is computed only when per-1K-token rates are configured via environment variables.",
        ],
    }
    write_json(path / "api_usage_summary.json", summary)
    return summary


def _initialize_files(usage_dir: Path) -> None:
    ensure_dir(usage_dir)
    log_path = usage_dir / "api_calls.jsonl"
    if not log_path.exists():
        log_path.write_text("", encoding="utf-8")
    summary_path = usage_dir / "api_usage_summary.json"
    if not summary_path.exists():
        write_usage_summary(usage_dir)


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def _read_jsonl(path: Path) -> list[Dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            value = json.loads(line)
        except Exception:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def _text_length(*, messages: Optional[Iterable[Mapping[str, Any]]], prompt: Optional[str]) -> int:
    if prompt is not None:
        return len(prompt)
    if not messages:
        return 0
    return len(json.dumps(list(messages), ensure_ascii=False, sort_keys=True))


def _estimate_tokens(chars: int) -> int:
    if chars <= 0:
        return 0
    return int(math.ceil(chars / 3.0))


def _normalize_usage(usage: Optional[Mapping[str, Any]]) -> Dict[str, int]:
    if not isinstance(usage, Mapping):
        return {}
    prompt = _first_int(usage, ["prompt_tokens", "input_tokens"])
    completion = _first_int(usage, ["completion_tokens", "output_tokens"])
    total = _first_int(usage, ["total_tokens"])
    if total is None and prompt is not None and completion is not None:
        total = prompt + completion
    if prompt is None and total is not None and completion is not None:
        prompt = max(0, total - completion)
    if completion is None and total is not None and prompt is not None:
        completion = max(0, total - prompt)
    if prompt is None and completion is None and total is None:
        return {}
    return {
        "prompt_tokens": int(prompt or 0),
        "completion_tokens": int(completion or 0),
        "total_tokens": int(total if total is not None else int(prompt or 0) + int(completion or 0)),
    }


def _first_int(mapping: Mapping[str, Any], keys: Iterable[str]) -> Optional[int]:
    for key in keys:
        value = mapping.get(key)
        if value is None:
            continue
        try:
            return int(value)
        except Exception:
            continue
    return None


def _estimate_cost(*, model: str, usage: Mapping[str, Any]) -> Dict[str, Any]:
    rates = _price_rates(model)
    if not rates:
        return {
            "currency": None,
            "input": None,
            "output": None,
            "total": None,
            "rate_source": "not_configured",
        }
    prompt = float(usage.get("prompt_tokens") or 0.0)
    completion = float(usage.get("completion_tokens") or 0.0)
    input_cost = prompt / 1000.0 * float(rates["input_per_1k"])
    output_cost = completion / 1000.0 * float(rates["output_per_1k"])
    return {
        "currency": rates["currency"],
        "input": round(input_cost, 8),
        "output": round(output_cost, 8),
        "total": round(input_cost + output_cost, 8),
        "rate_source": rates["rate_source"],
    }


def _price_rates(model: str) -> Dict[str, Any]:
    model_key = re.sub(r"[^A-Za-z0-9]+", "_", str(model or "unknown")).strip("_").upper()
    candidates = [
        f"AGENTM_API_PRICE_{model_key}",
        "AGENTM_API_PRICE_DEFAULT",
    ]
    for prefix in candidates:
        input_rate = os.getenv(f"{prefix}_INPUT_PER_1K")
        output_rate = os.getenv(f"{prefix}_OUTPUT_PER_1K")
        currency = os.getenv(f"{prefix}_CURRENCY", os.getenv("AGENTM_API_PRICE_CURRENCY", "CNY"))
        if input_rate is None or output_rate is None:
            continue
        try:
            return {
                "input_per_1k": float(input_rate),
                "output_per_1k": float(output_rate),
                "currency": currency,
                "rate_source": "env",
            }
        except Exception:
            continue
    return {}


def _safe_endpoint(endpoint: Optional[str]) -> Optional[str]:
    if not endpoint:
        return None
    text = str(endpoint)
    return re.sub(r"(api_key|key|token|secret)=([^&]+)", r"\1=<redacted>", text, flags=re.IGNORECASE)


def _compact_error(error: Optional[str]) -> Optional[str]:
    if not error:
        return None
    text = re.sub(r"\s+", " ", str(error)).strip()
    return text[:500] + ("..." if len(text) > 500 else "")


def _empty_rollup() -> Dict[str, Any]:
    return {
        "n_calls": 0,
        "n_completed_calls": 0,
        "n_failed_calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "usage_sources": {},
        "estimated_costs": {},
    }


def _add_record(rollup: Dict[str, Any], record: Mapping[str, Any]) -> None:
    rollup["n_calls"] += 1
    if record.get("status") == "completed":
        rollup["n_completed_calls"] += 1
    elif record.get("status") == "failed":
        rollup["n_failed_calls"] += 1
    rollup["prompt_tokens"] += int(record.get("prompt_tokens") or 0)
    rollup["completion_tokens"] += int(record.get("completion_tokens") or 0)
    rollup["total_tokens"] += int(record.get("total_tokens") or 0)
    source = str(record.get("usage_source") or "unknown")
    rollup["usage_sources"][source] = int(rollup["usage_sources"].get(source, 0)) + 1
    cost = record.get("estimated_cost") if isinstance(record.get("estimated_cost"), Mapping) else {}
    currency = cost.get("currency")
    total_cost = cost.get("total")
    if currency and total_cost is not None:
        rollup["estimated_costs"][currency] = round(float(rollup["estimated_costs"].get(currency, 0.0)) + float(total_cost), 8)
