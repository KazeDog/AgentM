from __future__ import annotations

import hashlib
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

from agentm.utils.io import ensure_dir, write_json


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SENSITIVE_TOKENS = ("key", "secret", "token", "password", "credential")

DEFAULT_PACKAGE_NAMES = {
    "numpy": "numpy",
    "pandas": "pandas",
    "rdkit": "rdkit",
    "scikit-learn": "sklearn",
    "torch": "torch",
    "torchvision": "torchvision",
    "transformers": "transformers",
    "sentence-transformers": "sentence_transformers",
    "faiss-cpu": "faiss",
    "requests": "requests",
    "Pillow": "PIL",
    "PyYAML": "yaml",
}

DEFAULT_ENV_FLAGS = (
    "DASHSCOPE_API_KEY",
    "QWEN_API_KEY",
    "MOONSHOT_API_KEY",
    "KIMI_API_KEY",
    "BAILIAN_API_KEY",
    "ALIBABA_CLOUD_ACCESS_KEY_ID",
    "ALIBABA_CLOUD_ACCESS_KEY_SECRET",
    "AGENTM_ENABLE_LLM_DRUG_RESOLVER",
    "AGENTM_ENABLE_EXTERNAL_EVIDENCE",
)


def write_reproducibility_log(
    output_dir: str | Path,
    *,
    config: Optional[Mapping[str, Any]] = None,
    artifacts: Optional[Mapping[str, Any]] = None,
    extra: Optional[Mapping[str, Any]] = None,
    filename: str = "reproducibility.json",
) -> Path:
    payload = collect_reproducibility_metadata(config=config, artifacts=artifacts, extra=extra)
    return write_json(ensure_dir(output_dir) / filename, payload)


def collect_reproducibility_metadata(
    *,
    config: Optional[Mapping[str, Any]] = None,
    artifacts: Optional[Mapping[str, Any]] = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    artifact_map = artifacts or {}
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(PROJECT_ROOT),
        "python": _python_metadata(),
        "platform": _platform_metadata(),
        "git": _git_metadata(PROJECT_ROOT),
        "packages": _package_versions(DEFAULT_PACKAGE_NAMES.keys()),
        "environment": _environment_flags(DEFAULT_ENV_FLAGS),
        "config": _redact(config or {}),
        "artifacts": {str(name): _artifact_metadata(value) for name, value in artifact_map.items()},
        "extra": _redact(extra or {}),
    }


def sha256_file(path: str | Path, *, max_bytes: Optional[int] = None) -> Optional[str]:
    target = _resolve_project_path(path)
    if not target.exists() or not target.is_file():
        return None
    digest = hashlib.sha256()
    remaining = max_bytes
    with target.open("rb") as handle:
        while True:
            size = 1024 * 1024 if remaining is None else min(1024 * 1024, remaining)
            if size <= 0:
                break
            chunk = handle.read(size)
            if not chunk:
                break
            digest.update(chunk)
            if remaining is not None:
                remaining -= len(chunk)
    return digest.hexdigest()


def _artifact_metadata(value: Any) -> Dict[str, Any]:
    if isinstance(value, Mapping):
        path = value.get("path") or value.get("file") or value.get("uri")
        metadata_payload = {str(k): _redact(v) for k, v in value.items() if k not in {"path", "file", "uri"}}
    else:
        path = value
        metadata_payload = {}
    if path is None:
        return {"path": None, "exists": False, "metadata": metadata_payload}
    target = _resolve_project_path(path)
    exists = target.exists()
    payload: Dict[str, Any] = {
        "path": str(path),
        "resolved_path": str(target),
        "exists": exists,
        "metadata": metadata_payload,
    }
    if exists and target.is_file():
        stat = target.stat()
        payload.update(
            {
                "size_bytes": int(stat.st_size),
                "sha256": sha256_file(target),
            }
        )
    return payload


def _resolve_project_path(path: str | Path) -> Path:
    target = Path(path).expanduser()
    if target.is_absolute():
        return target
    project_candidate = PROJECT_ROOT / target
    if project_candidate.exists():
        return project_candidate
    return (Path.cwd() / target).resolve()


def _python_metadata() -> Dict[str, Any]:
    return {
        "executable": sys.executable,
        "version": sys.version,
        "version_info": list(sys.version_info[:3]),
    }


def _platform_metadata() -> Dict[str, Any]:
    return {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }


def _git_metadata(project_root: Path) -> Dict[str, Any]:
    def run_git(args: Iterable[str]) -> Optional[str]:
        try:
            completed = subprocess.run(
                ["git", "-C", str(project_root), *args],
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            )
            return completed.stdout.strip()
        except Exception:
            return None

    commit = run_git(["rev-parse", "HEAD"])
    status = run_git(["status", "--short"])
    branch = run_git(["rev-parse", "--abbrev-ref", "HEAD"])
    return {
        "available": bool(commit),
        "commit": commit,
        "branch": branch,
        "dirty": bool(status),
        "status_short": status.splitlines() if status else [],
    }


def _package_versions(package_names: Iterable[str]) -> Dict[str, Optional[str]]:
    versions: Dict[str, Optional[str]] = {}
    for package_name in package_names:
        try:
            versions[package_name] = metadata.version(package_name)
        except metadata.PackageNotFoundError:
            versions[package_name] = None
    return versions


def _environment_flags(env_names: Iterable[str]) -> Dict[str, bool]:
    return {name: bool(os.environ.get(name)) for name in env_names}


def _redact(value: Any) -> Any:
    if isinstance(value, Mapping):
        redacted: Dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            if any(token in key_text.lower() for token in SENSITIVE_TOKENS):
                redacted[key_text] = item if item is None or item == "" else "***REDACTED***"
            else:
                redacted[key_text] = _redact(item)
        return redacted
    if isinstance(value, list):
        return [_redact(item) for item in value]
    if isinstance(value, tuple):
        return [_redact(item) for item in value]
    return value
