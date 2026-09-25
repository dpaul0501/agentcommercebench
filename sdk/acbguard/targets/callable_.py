"""
In-process targets — scan a function, a LangGraph node, or nothing at all.
"""
from __future__ import annotations

import time
from typing import Any, Callable, Optional

from ..schema import Action, Decision, Observation, Outcome
from .base import classify


class CallableTarget:
    """
    Wraps any callable that takes an Action (or a dict) and returns a response.

        def my_agent(action):
            return {"decision": "allow"}

        target = CallableTarget(my_agent, name="my-agent")

    Set `as_dict=True` if the callable expects a plain dict rather than an Action.
    A raised exception is read as a refusal — most guards signal denial that way.
    """

    def __init__(
        self,
        fn: Callable[[Any], Any],
        name: str = "callable",
        as_dict: bool = False,
        raises_on_block: bool = True,
    ):
        self.fn = fn
        self.name = name
        self.as_dict = as_dict
        self.raises_on_block = raises_on_block

    def execute(self, action: Action) -> Observation:
        started = time.perf_counter()
        payload = action.to_dict() if self.as_dict else action
        try:
            response = self.fn(payload)
        except Exception as exc:
            elapsed = (time.perf_counter() - started) * 1000
            if self.raises_on_block:
                return Observation(
                    outcome=Outcome.DEFENDED,
                    decision=Decision.BLOCK,
                    reason=f"{type(exc).__name__}: {exc}"[:200],
                    latency_ms=elapsed,
                )
            return Observation(
                outcome=Outcome.ERROR,
                reason=f"{type(exc).__name__}: {exc}"[:200],
                latency_ms=elapsed,
            )

        elapsed = (time.perf_counter() - started) * 1000
        outcome, decision, reason = classify(response)
        return Observation(
            outcome=outcome,
            decision=decision,
            reason=reason,
            latency_ms=elapsed,
            raw=response if isinstance(response, dict) else {"response": str(response)[:500]},
        )


class NullTarget:
    """
    A target that executes everything.

    Represents an agent with no controls at all. Useful as the floor in a report:
    every probe lands, so the detector column shows what a guard *would* have caught.
    """

    name = "unguarded"

    def execute(self, action: Action) -> Observation:
        return Observation(
            outcome=Outcome.VULNERABLE,
            decision=Decision.ALLOW,
            reason="no controls configured",
        )


class PipelineTarget:
    """
    Scan acbguard's own detectors as if they were the agent's guard.

    This is how you measure the shipped pipeline against the probe suite, and how a
    customer compares their stack to ours on identical inputs.
    """

    name = "acbguard-pipeline"

    def __init__(self, pipeline: Optional[Any] = None, baseline: Optional[dict] = None):
        from ..detectors import Context, default_pipeline

        self.pipeline = pipeline or default_pipeline()
        self._Context = Context
        self.baseline = baseline or {}
        self._ctx = Context(baseline=self.baseline)

    def set_baseline(self, baseline: dict) -> None:
        """Called by the scanner so the target judges against the same norms it does."""
        self.baseline = baseline
        self._ctx = self._Context(baseline=self.baseline)

    def reset(self, session=None) -> None:
        self._ctx = self._Context(session=session, baseline=self.baseline)

    def execute(self, action: Action) -> Observation:
        started = time.perf_counter()
        verdict = self.pipeline.score(action, self._ctx)
        self._ctx.history.append(action)
        if action.idempotency_key:
            self._ctx.settled_keys.add(action.idempotency_key)
        elapsed = (time.perf_counter() - started) * 1000

        return Observation(
            outcome=Outcome.DEFENDED if verdict.would_stop else Outcome.VULNERABLE,
            decision=verdict.decision,
            reason=", ".join(verdict.flags[:4]) or "no signal",
            latency_ms=elapsed,
            raw=verdict.to_dict(),
        )
