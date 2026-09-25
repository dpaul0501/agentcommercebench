"""
LangGraph target — scan a compiled graph.

The action is injected into graph state, the graph is invoked, and the resulting
state is read for a decision. Which state key carries the decision differs per graph,
so `decision_key` is configurable.

Worth knowing when reading a clean result here: LangGraph's default channel reducer
is last-writer-wins, so a node that writes a block can be silently overwritten by any
later node writing allow. A graph can therefore pass this scan and still be bypassable
in production — which is what `decision: Annotated[str, max]` (or an equivalent
monotone reducer) exists to prevent.
"""
from __future__ import annotations

import time
from typing import Any, Callable, Optional

from ..schema import Action, Observation, Outcome
from .base import classify


class LangGraphTarget:
    """
        from my_agent import graph
        target = LangGraphTarget(graph)

    `state_builder` maps an Action into your graph's input state; the default puts
    the action dict under "action" and a rendered summary under "messages".
    """

    def __init__(
        self,
        graph: Any,
        *,
        state_builder: Optional[Callable[[Action], dict[str, Any]]] = None,
        decision_key: str = "decision",
        name: str = "langgraph",
        config: Optional[dict[str, Any]] = None,
    ):
        self.graph = graph
        self.state_builder = state_builder or self._default_state
        self.decision_key = decision_key
        self.name = name
        self.config = config or {}

    @staticmethod
    def _default_state(action: Action) -> dict[str, Any]:
        summary = (
            f"Authorize ${action.amount_usd:,.2f} to {action.vendor or 'vendor'} "
            f"for {action.service_id or 'service'}."
        )
        return {
            "action": action.to_dict(),
            "messages": [{"role": "user", "content": summary}],
        }

    def execute(self, action: Action) -> Observation:
        started = time.perf_counter()
        try:
            result = self.graph.invoke(self.state_builder(action), **self.config)
        except Exception as exc:
            # Graphs commonly signal refusal by raising.
            return Observation(
                outcome=Outcome.DEFENDED,
                reason=f"{type(exc).__name__}: {exc}"[:200],
                latency_ms=(time.perf_counter() - started) * 1000,
            )

        elapsed = (time.perf_counter() - started) * 1000
        probe = result
        if isinstance(result, dict) and self.decision_key in result:
            probe = {"decision": result[self.decision_key]}
        outcome, decision, reason = classify(probe)
        return Observation(
            outcome=outcome,
            decision=decision,
            reason=reason,
            latency_ms=elapsed,
            raw=result if isinstance(result, dict) else {"result": str(result)[:500]},
        )
