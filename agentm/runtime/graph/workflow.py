from __future__ import annotations

from typing import Any

from agentm.runtime.state import GlobalContext


class _SimpleWorkflow:
    def __init__(self, nodes: Any) -> None:
        self.nodes = nodes

    def invoke(self, context: GlobalContext) -> GlobalContext:
        context = self.nodes.initialize_context(context)
        context = self.nodes.orchestrator_loop(context)
        context = self.nodes.guarded_finalize(context)
        return context


def build_workflow(nodes: Any) -> Any:
    try:
        from langgraph.graph import END, START, StateGraph

        graph = StateGraph(GlobalContext)
        graph.add_node("initialize_context", nodes.initialize_context)
        graph.add_node("orchestrator_loop", nodes.orchestrator_loop)
        graph.add_node("guarded_finalize", nodes.guarded_finalize)
        graph.add_edge(START, "initialize_context")
        graph.add_edge("initialize_context", "orchestrator_loop")
        graph.add_edge("orchestrator_loop", "guarded_finalize")
        graph.add_edge("guarded_finalize", END)
        return graph.compile()
    except Exception:
        return _SimpleWorkflow(nodes)
