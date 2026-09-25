"""
Where history-level detectors keep their state.

Two facts about production decide the whole design of this module.

**There are no sessions.** `session_id` is populated on 0.47% of production settlements — one
in 213. A benchmark can group an agent's actions into tidy sessions because it created them;
a deployed detector cannot, because nothing upstream records the boundary. Any history keyed
on a session works in the harness and returns an empty history in production, silently, and
the detectors that depend on it degrade to nothing without erroring.

So history is keyed on **`agent_id`**, which is populated on 100% of settlements, and a
session boundary is treated as information that may simply be absent.

**It has to be bounded.** An in-process list grows until the process dies, and a set of every
idempotency key ever seen is an unbounded memory leak with a long fuse. Both are capped here,
and the caps are stated rather than implied, because a cap silently truncating an agent's
history changes what a detector can detect.

    store = InMemoryContextStore(max_actions=200, max_agents=10_000)
    ctx = store.context("agent-42", baseline)
    store.record("agent-42", action)

`InMemoryContextStore` is correct for one process. It is deliberately not correct for a fleet:
replay detection asks "has this key settled anywhere", and a per-process set cannot answer
that. `ContextStore` is the seam a Redis or Postgres implementation drops into, and
`RemoteContextStore` below documents what such an implementation has to guarantee.
"""
from __future__ import annotations

import threading
from collections import OrderedDict, deque
from typing import Any, Optional, Protocol, runtime_checkable

from ..detectors import Context
from ..schema import Action

DEFAULT_MAX_ACTIONS = 200
"""Actions retained per agent.

Chosen from what the history-level detectors actually reach for: the longest lookback any of
them uses is the agent's own spend distribution and its set of known counterparties, both of
which are stable well inside 200 payments. Velocity and ratcheting need tens, not hundreds.
Raising it costs memory linearly and buys nothing measurable; lowering it below ~50 starts to
truncate the counterparty set, which is what `LearnedDestinationDetector` reads.
"""

DEFAULT_MAX_AGENTS = 10_000
"""Agents held in memory before the least-recently-used is evicted.

An evicted agent is not an error — it comes back with an empty history, exactly as a
first-seen agent does, and the history-level detectors are already required to be silent
rather than confident when they have no history. That is the same cold-start path, not a new
failure mode.
"""

DEFAULT_MAX_SETTLED_KEYS = 5_000
"""Idempotency keys retained per agent, oldest evicted first."""

DEFAULT_SESSION_GAP_SECONDS = 1_800.0
"""
Inactivity gap that ends a derived session: 30 minutes.

Taken from web analytics, where a 30-minute inactivity timeout has been the default since
Google Analytics adopted it, and is used here for the same reason it works there — it is long
enough that a pause for thought does not split a sitting, and short enough that tomorrow's
work is not joined to today's.

It is deliberately NOT fitted to this benchmark. The generator draws within-session spans of
20-900s and between-session gaps of 1800s and up, so the two are separable by construction and
*any* threshold in that band scores perfectly. Fitting one here would be measuring the
generator, not the method. Recovery accuracy against the benchmark's true session labels is
therefore an upper bound on production accuracy, and is reported as one.

Agents are faster than people, so if production sessions turn out to be shorter this should
come down — the number to look at is the gap distribution's trough, once `authorized_at` is
grouped per agent over real history.
"""


@runtime_checkable
class ContextStore(Protocol):
    """
    The seam between in-process state and a real store.

    An implementation backed by Redis or Postgres must guarantee two things that the in-memory
    one cannot:

    1. `settled(key)` is true if the key settled **anywhere**, not merely in this process.
       Replay across two API servers is otherwise undetectable, and an attacker choosing which
       server to hit is not a sophisticated attack.
    2. `record` is atomic. Two concurrent requests carrying the same idempotency key must not
       both observe it as unseen.
    """

    def context(self, agent_id: str, baseline: Optional[dict] = None) -> Context: ...

    def record(self, agent_id: str, action: Action) -> None: ...


class InMemoryContextStore:
    """
    Bounded, agent-keyed, single-process history.

        store = InMemoryContextStore()
        ctx   = store.context(action.agent_id, baseline)
        verdict = pipeline.score(action, ctx)
        store.record(action.agent_id, action)

    Thread-safe. Correct for one process and explicitly not for a fleet — see `ContextStore`.
    """

    def __init__(
        self,
        max_actions: int = DEFAULT_MAX_ACTIONS,
        max_agents: int = DEFAULT_MAX_AGENTS,
        max_settled_keys: int = DEFAULT_MAX_SETTLED_KEYS,
        session_gap_seconds: float = DEFAULT_SESSION_GAP_SECONDS,
    ):
        self.max_actions = max_actions
        self.max_agents = max_agents
        self.max_settled_keys = max_settled_keys
        self.session_gap_seconds = session_gap_seconds
        self._history: OrderedDict[str, deque] = OrderedDict()
        self._settled: OrderedDict[str, OrderedDict] = OrderedDict()
        self._session: dict[str, tuple[str, Any]] = {}
        """agent_id -> (current session id, timestamp of its last action)."""
        self._lock = threading.Lock()

    # ── reads ────────────────────────────────────────────────────────────
    def context(self, agent_id: str, baseline: Optional[dict] = None) -> Context:
        """
        A `Context` for this agent, carrying its recent history.

        The returned `Context` has **no `session`**. That is deliberate and not an oversight:
        production cannot tell a detector where a session began, so a detector that behaves
        differently when `ctx.session` is set would behave differently in the harness than in
        deployment. Leaving it None keeps the two honest.
        """
        with self._lock:
            history = self._history.get(agent_id)
            settled = self._settled.get(agent_id)
            if history is not None:
                self._history.move_to_end(agent_id)
            return Context(
                session=None,
                history=list(history) if history else [],
                baseline=dict(baseline or {}),
                settled_keys=set(settled) if settled else set(),
                # The key this history is filed under is the authenticated agent, so an
                # action claiming a different one is claiming an identity it did not prove.
                principal_id=agent_id,
                session_id=(self._session.get(agent_id) or (None, None))[0],
            )

    def settled(self, agent_id: str, key: str) -> bool:
        with self._lock:
            keys = self._settled.get(agent_id)
            return bool(keys and key in keys)

    # ── session derivation ───────────────────────────────────────────────
    def session_id_for(self, agent_id: str, action: Action) -> Optional[str]:
        """
        The session this action belongs to: the one it carries, or one derived from the gap.

        Production sets `session_id` on 0.47% of settlements, so for almost every action this
        derives. We know the agent — that is on 100% of them — and we know when it acted, so a
        sitting is recoverable the way web analytics has recovered them for twenty years:
        consecutive actions by the same agent belong together until it goes quiet.

        A derived id is marked `derived:` so nothing downstream can mistake it for one the
        platform asserted. An action with no timestamp gets the agent's current session rather
        than starting a new one — absence of a timestamp is not evidence of a new sitting.
        """
        if action.session_id:
            return action.session_id

        current = self._session.get(agent_id)
        at = action.timestamp
        if current is None:
            new = f"derived:{agent_id}:0"
            self._session[agent_id] = (new, at)
            return new

        session_id, last_at = current
        if at is None or last_at is None:
            return session_id
        if (at - last_at).total_seconds() > self.session_gap_seconds:
            index = int(session_id.rsplit(":", 1)[-1]) + 1
            session_id = f"derived:{agent_id}:{index}"
        self._session[agent_id] = (session_id, at)
        return session_id

    # ── writes ───────────────────────────────────────────────────────────
    def record(self, agent_id: str, action: Action) -> None:
        with self._lock:
            self.session_id_for(agent_id, action)
            history = self._history.get(agent_id)
            if history is None:
                history = deque(maxlen=self.max_actions)
                self._history[agent_id] = history
                self._settled[agent_id] = OrderedDict()
            history.append(action)
            self._history.move_to_end(agent_id)

            if action.idempotency_key:
                keys = self._settled[agent_id]
                keys[action.idempotency_key] = True
                keys.move_to_end(action.idempotency_key)
                while len(keys) > self.max_settled_keys:
                    keys.popitem(last=False)

            while len(self._history) > self.max_agents:
                evicted, _ = self._history.popitem(last=False)
                self._settled.pop(evicted, None)
                self._session.pop(evicted, None)

    def forget(self, agent_id: str) -> None:
        """Drop one agent's history — for tests, and for a deletion request."""
        with self._lock:
            self._history.pop(agent_id, None)
            self._settled.pop(agent_id, None)
            self._session.pop(agent_id, None)

    # ── introspection ────────────────────────────────────────────────────
    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "agents": len(self._history),
                "actions": sum(len(h) for h in self._history.values()),
                "settled_keys": sum(len(k) for k in self._settled.values()),
                "max_actions_per_agent": self.max_actions,
                "max_agents": self.max_agents,
            }


__all__ = [
    "ContextStore",
    "InMemoryContextStore",
    "DEFAULT_SESSION_GAP_SECONDS",
    "DEFAULT_MAX_ACTIONS",
    "DEFAULT_MAX_AGENTS",
    "DEFAULT_MAX_SETTLED_KEYS",
]
