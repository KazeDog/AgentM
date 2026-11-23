from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
from typing import Any, Iterable, Mapping, Sequence

import yaml


RELIABILITY_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = RELIABILITY_ROOT.parents[1]
CODE_COMMIT = "unavailable-no-git-metadata"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def stable_hash(value: Any, *, length: int | None = None) -> str:
    digest = hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()
    return digest[:length] if length else digest


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_tree_hash() -> str:
    """Hash the audited runtime when no Git commit metadata is available."""

    # Hash the complete unified package and the experimental harness once.
    # Project-relative paths keep the identity stable when the repository moves.
    roots = [
        PROJECT_ROOT / "agentm",
        RELIABILITY_ROOT,
    ]
    digest = hashlib.sha256()
    for root in roots:
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix not in {".py", ".toml", ".yaml", ".yml", ".json", ".jsonl", ".md", ".sh", ".html"}:
                continue
            if "__pycache__" in path.parts or path.name.startswith("."):
                continue
            rel = path.relative_to(PROJECT_ROOT).as_posix()
            digest.update(rel.encode("utf-8"))
            digest.update(b"\0")
            digest.update(path.read_bytes())
            digest.update(b"\0")
    return digest.hexdigest()


def resolve_path(value: str | Path, *, base: Path = PROJECT_ROOT) -> Path:
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def _expand_env(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _expand_env(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_env(item) for item in value]
    if not isinstance(value, str):
        return value

    def repl(match: re.Match[str]) -> str:
        name, default = match.group(1), match.group(2)
        return os.getenv(name, default or "")

    return _ENV_PATTERN.sub(repl, value)


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = resolve_path(path, base=Path.cwd())
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Configuration must be a mapping: {config_path}")
    extends = payload.pop("extends", None)
    if extends:
        parent_path = resolve_path(extends, base=config_path.parent)
        parent = load_config(parent_path)
        payload = deep_merge(parent, payload)
    payload = _expand_env(payload)
    payload.setdefault("_meta", {})
    payload["_meta"].update(
        {
            "config_path": str(config_path),
            "loaded_at": utc_now(),
        }
    )
    hash_payload = {key: value for key, value in payload.items() if key != "_meta"}
    payload["_meta"]["config_hash"] = stable_hash(hash_payload)
    return payload


def load_jsonl(path: str | Path, *, required_fields: Sequence[str] = ()) -> list[dict[str, Any]]:
    target = resolve_path(path)
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(target.read_text(encoding="utf-8").splitlines(), start=1):
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        try:
            row = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at {target}:{line_no}: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"JSONL row must be an object at {target}:{line_no}")
        missing = [field for field in required_fields if field not in row]
        if missing:
            raise ValueError(f"Missing fields {missing} at {target}:{line_no}")
        rows.append(row)
    return rows


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]], *, overwrite: bool = True) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if overwrite else "x"
    with target.open(mode, encoding="utf-8") as handle:
        for row in rows:
            handle.write(canonical_json(dict(row)) + "\n")
    return target


def write_json(path: str | Path, payload: Any, *, overwrite: bool = True) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if overwrite else "x"
    with target.open(mode, encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True, default=str)
        handle.write("\n")
    return target


def read_json(path: str | Path, default: Any = None) -> Any:
    target = Path(path)
    if not target.exists():
        return default
    return json.loads(target.read_text(encoding="utf-8"))


def write_csv(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row}) if rows else []
    with target.open("w", encoding="utf-8", newline="") as handle:
        if fields:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    return target


def slug(value: Any) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value or "unknown")).strip("-._").lower()
    return text or "unknown"


def run_id_for(job: Mapping[str, Any], config_hash: str) -> str:
    identity = {
        "dataset_group": job.get("dataset_group"),
        "case_id": job.get("case_id") or job.get("name"),
        "orchestrator": job.get("orchestrator"),
        "reflection_enabled": bool(job.get("reflection_enabled")),
        "seed": int(job.get("seed", 0)),
        "repeat_index": int(job.get("repeat_index", 0)),
        "config_hash": config_hash,
        "case_hash": stable_hash(job.get("case") or {}),
        # Immutable results from an older implementation must not be reused
        # after a code change. Git metadata is unavailable in this workspace,
        # so the audited source-tree digest is the code identity.
        "source_tree_hash": job.get("source_tree_hash") or CODE_COMMIT,
    }
    if "reflection_depth" in job:
        depth = job["reflection_depth"]
        if type(depth) is not int or depth not in (0, 1, 2, 3):
            raise ValueError("invalid explicit reflection depth")
        identity["reflection_depth"] = depth
        identity["reflection_protocol"] = job.get("reflection_protocol")
    suffix = stable_hash(identity, length=12)
    return "__".join(
        [
            slug(identity["dataset_group"]),
            slug(identity["case_id"]),
            slug(identity["orchestrator"]),
            f"reflection-depth-{identity['reflection_depth']}" if "reflection_depth" in identity else (
                "reflection-on" if identity["reflection_enabled"] else "reflection-off"
            ),
            f"s{identity['seed']}",
            f"r{identity['repeat_index']}",
            suffix,
        ]
    )


def ensure_output_layout(root: str | Path) -> dict[str, Path]:
    output_root = resolve_path(root)
    names = [
        "raw_runs",
        "global_contexts",
        "reports",
        "traces",
        "errors",
        "metrics",
        "tables",
        "figures",
        "cases",
        "_staging",
        "_jobs",
    ]
    paths = {"root": output_root}
    output_root.mkdir(parents=True, exist_ok=True)
    for name in names:
        paths[name] = output_root / name
        paths[name].mkdir(parents=True, exist_ok=True)
    return paths


def publish_artifact(source: Path, destination: Path) -> None:
    """Publish a derived copy without mutating the immutable raw source."""

    if not source.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def require_fields(mapping: Mapping[str, Any], fields: Sequence[str], *, label: str) -> None:
    missing = [field for field in fields if field not in mapping]
    if missing:
        raise ValueError(f"{label} missing required field(s): {', '.join(missing)}")


def nested_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return canonical_json(value)
