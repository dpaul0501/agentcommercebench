"""
Runtime guard — the same detectors, applied inline instead of in a scan.

Two modes, matching the lifecycle stages:

    observe    score and record, never interfere. Where every integration starts.
    enforce    raise on block; escalations go to a handler.

Observe mode is the default on purpose: it is the only mode that is safe to enable
without a baseline, and the traces it produces are what a baseline is derived from.
"""
from __future__ import annotations

import functools
import os
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Optional

from ..detectors import Pipeline, Verdict, default_pipeline
from ..schema import Action, Decision
from .store import ContextStore, InMemoryContextStore


class Mode(str, Enum):
    OBSERVE = "observe"
    ENFORCE = "enforce"


class Blocked(Exception):
    """Raised in enforce mode when an action is blocked."""

    def __init__(self, action: Action, verdict: Verdict):
        self.action = action
        self.verdict = verdict
        super().__init__(
            f"blocked: risk={verdict.risk_score:.2f} flags={verdict.flags[:4]}"
        )


class NeedsApproval(Exception):
    """Raised in enforce mode when an action escalates and no handler is set."""

    def __init__(self, action: Action, verdict: Verdict):
        self.action = action
        self.verdict = verdict
        super().__init__(
            f"needs approval: risk={verdict.risk_score:.2f} flags={verdict.flags[:4]}"
        )


@dataclass
class TraceRecord:
    action: Action
    verdict: Verdict
    at: datetime

    def to_dict(self) -> dict[str, Any]:
        return {
            "at": self.at.isoformat(),
            "action": self.action.to_dict(),
            "verdict": self.verdict.to_dict(),
        }


class Guard:
    """
    Inline scoring for a live agent.

        guard = Guard(mode="observe", trace_path="traces.jsonl")

        @guard.watch
        def pay(action: Action): ...

    or explicitly:

        verdict = guard.check(action)
    """

    def __init__(
        self,
        mode: Mode | str = Mode.OBSERVE,
        pipeline: Optional[Pipeline] = None,
        baseline: Optional[dict] = None,
        trace_path: Optional[str] = None,
        on_escalate: Optional[Callable[[Action, Verdict], bool]] = None,
        sink: Optional[Any] = None,
        baseline_provider: Optional[Any] = None,
        agent_id: str = "agent",
        store: Optional[ContextStore] = None,
    ):
        from ..sinks import FileSink, MultiSink, NullSink

        self.mode = Mode(mode)
        self.pipeline = pipeline or default_pipeline()
        self.agent_id = agent_id
        self.baseline_provider = baseline_provider

        if baseline is None and baseline_provider is not None:
            baseline = baseline_provider.baseline_for(agent_id)
        self.baseline = baseline or {}

        # trace_path is sugar for a FileSink; both may be supplied.
        self.trace_path = trace_path or os.environ.get("ACBGUARD_TRACE")
        sinks = []
        if self.trace_path:
            sinks.append(FileSink(self.trace_path))
        if sink is not None:
            sinks.append(sink)
        self.sink = (
            NullSink() if not sinks else (sinks[0] if len(sinks) == 1 else MultiSink(sinks))
        )

        self.on_escalate = on_escalate
        self.trace: list[TraceRecord] = []
        # History is kept per agent, not per guard, because one guard serves a fleet and
        # because production has no session to key on: `session_id` is populated on 0.47% of
        # settlements. `agent_id` is populated on 100%.
        self.store = store or InMemoryContextStore()
        self._lock = threading.Lock()

    def refresh_baseline(self) -> dict:
        """Re-read norms from the provider, if one is configured."""
        if self.baseline_provider is None:
            return self.baseline
        self.baseline = self.baseline_provider.baseline_for(self.agent_id) or {}
        return self.baseline

    def flush(self) -> None:
        try:
            self.sink.flush()
        except Exception:
            pass

    def baseline_for(self, agent_id: str) -> dict:
        """Norms for one agent. Falls back to this guard's own when no provider is set."""
        if self.baseline_provider is None:
            return self.baseline
        return self.baseline_provider.baseline_for(agent_id) or {}

    def check(self, action: Action) -> Verdict:
        """
        Score an action, record it, and apply the mode's policy.

        The action's own `agent_id` selects whose history it is scored against, so a single
        guard can serve every agent on the process. An action with no agent id falls back to
        the guard's configured one, which is the single-agent case.
        """
        agent_id = action.agent_id or self.agent_id
        with self._lock:
            ctx = self.store.context(agent_id, self.baseline_for(agent_id))
            verdict = self.pipeline.score(action, ctx)
            self.store.record(agent_id, action)
            record = TraceRecord(action=action, verdict=verdict, at=datetime.now(timezone.utc))
            self.trace.append(record)
            self._write(record)

        if self.mode is Mode.OBSERVE:
            return verdict

        if verdict.decision is Decision.BLOCK:
            raise Blocked(action, verdict)
        if verdict.decision is Decision.ESCALATE:
            if self.on_escalate is None:
                raise NeedsApproval(action, verdict)
            if not self.on_escalate(action, verdict):
                raise Blocked(action, verdict)
        return verdict

    def watch(self, fn: Callable) -> Callable:
        """Decorator: score the Action argument before the wrapped call runs."""

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            action = next(
                (a for a in list(args) + list(kwargs.values()) if isinstance(a, Action)),
                None,
            )
            if action is not None:
                self.check(action)
            return fn(*args, **kwargs)

        return wrapper

    def _write(self, record: TraceRecord) -> None:
        try:
            self.sink.emit(record.to_dict())
        except Exception:
            pass  # tracing must never take down the agent

    def summary(self) -> dict[str, Any]:
        counts = {d.value: 0 for d in Decision}
        for record in self.trace:
            counts[record.verdict.decision.value] += 1
        return {
            "mode": self.mode.value,
            "actions": len(self.trace),
            "decisions": counts,
            "sink": getattr(self.sink, "name", type(self.sink).__name__),
            "has_baseline": bool(self.baseline),
        }


__all__ = ["Guard", "Mode", "Blocked", "NeedsApproval", "TraceRecord"]
