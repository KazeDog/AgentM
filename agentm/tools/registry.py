from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict


ToolHandler = Callable[..., Dict[str, Any]]


@dataclass
class ToolSpec:
    name: str
    handler: ToolHandler
    description: str = ""


class ToolRegistry:
    """Small explicit registry for agent-callable tools.

    The registry keeps orchestration code independent from the concrete agents
    that implement each operation. Handlers mutate the shared agent state and
    return compact JSON-friendly summaries for the execution trace.
    """

    def __init__(self):
        self._tools: Dict[str, ToolSpec] = {}

    def register(self, name: str, handler: ToolHandler, description: str = "") -> None:
        if name in self._tools:
            raise ValueError(f"Tool already registered: {name}")
        self._tools[name] = ToolSpec(name=name, handler=handler, description=description)

    def invoke(self, name: str, **kwargs: Any) -> Dict[str, Any]:
        if name not in self._tools:
            raise KeyError(f"Unknown tool: {name}")
        result = self._tools[name].handler(**kwargs)
        return result or {}

    def describe(self) -> Dict[str, Dict[str, str]]:
        return {
            name: {
                "name": spec.name,
                "description": spec.description,
            }
            for name, spec in sorted(self._tools.items())
        }

    def names(self) -> set[str]:
        return set(self._tools)
