from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def write_json(path: str | Path, obj: Any) -> Path:
    target = Path(path)
    ensure_dir(target.parent)
    target.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    return target


def read_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_yaml(path: str | Path) -> Dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to load YAML configs.") from exc
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return dict(data)


def write_text(path: str | Path, text: str) -> Path:
    target = Path(path)
    ensure_dir(target.parent)
    target.write_text(text, encoding="utf-8")
    return target

