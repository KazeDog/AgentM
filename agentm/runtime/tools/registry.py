from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Mapping


@dataclass
class AgentAction:
    agent: str
    tool: str
    args: Dict[str, Any] = field(default_factory=dict)
    rationale: str = ""

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "AgentAction":
        return cls(
            agent=str(payload.get("agent") or ""),
            tool=str(payload.get("tool") or ""),
            args=dict(payload.get("args") or {}),
            rationale=str(payload.get("rationale") or ""),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {"agent": self.agent, "tool": self.tool, "args": self.args, "rationale": self.rationale}


@dataclass
class ToolResult:
    tool: str
    status: str
    observation: Dict[str, Any] = field(default_factory=dict)
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"tool": self.tool, "status": self.status, "observation": self.observation, "error": self.error}


@dataclass
class ToolSpec:
    name: str
    agent: str
    description: str
    input_schema: Dict[str, Any]
    handler: Callable[[Dict[str, Any], Dict[str, Any]], ToolResult]

    def public_schema(self) -> Dict[str, Any]:
        return {
            "agent": self.agent,
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


class MultiAgentToolRegistry:
    def __init__(self) -> None:
        self._tools: Dict[str, ToolSpec] = {}
        self._agent_tools: Dict[str, set[str]] = {}

    def register(self, spec: ToolSpec) -> None:
        self._tools[spec.name] = spec
        self._agent_tools.setdefault(spec.agent, set()).add(spec.name)

    def execute(self, action: AgentAction, context: Dict[str, Any]) -> ToolResult:
        spec = self._tools.get(action.tool)
        if spec is None:
            return ToolResult(tool=action.tool, status="failed", error=f"unknown_tool: {action.tool}")
        if spec.agent != action.agent:
            return ToolResult(tool=action.tool, status="failed", error=f"tool_agent_mismatch: {action.agent}.{action.tool}")
        return spec.handler(action.args, context)

    def allowlist_for_agent(self, agent: str) -> list[str]:
        return sorted(self._agent_tools.get(agent, set()))

    def agent_allowlists(self) -> Dict[str, list[str]]:
        return {agent: sorted(tools) for agent, tools in sorted(self._agent_tools.items())}

    def agents(self) -> list[str]:
        return sorted(self._agent_tools)

    def schemas(self) -> list[Dict[str, Any]]:
        return [self._tools[name].public_schema() for name in sorted(self._tools)]

    def schemas_for_agent(self, agent: str) -> list[Dict[str, Any]]:
        return [self._tools[name].public_schema() for name in self.allowlist_for_agent(agent)]

    def all_tools(self) -> Iterable[str]:
        return sorted(self._tools)
