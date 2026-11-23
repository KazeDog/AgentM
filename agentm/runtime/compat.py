from __future__ import annotations

import json
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_ROOT = PROJECT_ROOT
WORKSPACE_ROOT = PROJECT_ROOT
RESOURCE_ROOT = PROJECT_ROOT


def ensure_import_paths() -> None:
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))


def to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "to_dict"):
        return to_jsonable(value.to_dict())
    if hasattr(value, "tolist"):
        return to_jsonable(value.tolist())
    if is_dataclass(value):
        return to_jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(v) for v in value]
    return str(value)


def write_json(path: str | Path, payload: Any) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(to_jsonable(payload), indent=2, ensure_ascii=False), encoding="utf-8")
    return target


def write_text(path: str | Path, text: str) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    return target


ensure_import_paths()
