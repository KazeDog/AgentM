from __future__ import annotations

from agentm.agents.controller import AgentController


class SmallMoleculeMultimodalAgent(AgentController):
    """Backward-compatible entry point for the agentM workflow.

    The implementation now lives in AgentController, which adds explicit
    planning, tool dispatch, execution trace, fallback handling, and structured
    reasoning while preserving the original run_single/run_ddi API.
    """

