"""
History is per agent, bounded, and does not assume a session.

The production facts these pin, so a future change cannot quietly undo them:

  - `session_id` is populated on 0.47% of production settlements. A detector that needs a
    session boundary has one in the harness and not in deployment.
  - `agent_id` is populated on 100%, so it is the only key history can use.
  - One guard serves a fleet, so two agents must not see each other's history.
  - State must be bounded, or a long-lived process leaks until it dies.
"""

from acbguard import Guard
from acbguard.detectors import Pipeline
from acbguard.guard.store import InMemoryContextStore
from acbguard.schema import Action, ActionType


def pay(agent_id, amount=50_000, key=None, vendor="exa.ai"):
    return Action(action_type=ActionType.AUTHORIZE, agent_id=agent_id,
                  service_id="exa", amount_units=amount, vendor=vendor,
                  idempotency_key=key)


# ── agent keying ─────────────────────────────────────────────────────────

def test_two_agents_do_not_share_history():
    store = InMemoryContextStore()
    store.record("a", pay("a"))
    store.record("a", pay("a"))
    store.record("b", pay("b"))

    assert len(store.context("a").history) == 2
    assert len(store.context("b").history) == 1
    assert store.context("never-seen").history == []


def test_one_guard_serves_many_agents():
    """The action's own agent_id selects whose history it is scored against."""
    guard = Guard(mode="observe", pipeline=Pipeline([]))
    for _ in range(3):
        guard.check(pay("agent-1"))
    guard.check(pay("agent-2"))

    assert len(guard.store.context("agent-1").history) == 3
    assert len(guard.store.context("agent-2").history) == 1


def test_action_without_an_agent_id_falls_back_to_the_guard():
    """The single-agent case still works, which is how most integrations start."""
    guard = Guard(mode="observe", agent_id="solo", pipeline=Pipeline([]))
    guard.check(pay(None))
    assert len(guard.store.context("solo").history) == 1


# ── no session assumption ────────────────────────────────────────────────

def test_context_carries_no_session():
    """
    Production cannot tell a detector where a session began.

    A detector that behaves differently when `ctx.session` is set would behave differently in
    the harness than in deployment, and the harness is the one that would look better.
    """
    store = InMemoryContextStore()
    store.record("a", pay("a"))
    assert store.context("a").session is None


def test_history_crosses_session_boundaries():
    """
    Actions from what a harness would call two sessions land in one agent history.

    This is the behaviour that matters for ratcheting and for counterparty warm-up: both are
    slow, both cross any session boundary, and keying on a session would hide them.
    """
    store = InMemoryContextStore()
    for session in ("s1", "s2"):
        for _ in range(2):
            action = pay("a")
            action.session_id = session
            store.record("a", action)
    assert len(store.context("a").history) == 4


# ── bounds ───────────────────────────────────────────────────────────────

def test_history_is_bounded_per_agent():
    store = InMemoryContextStore(max_actions=10)
    for i in range(50):
        store.record("a", pay("a", amount=i + 1))
    history = store.context("a").history
    assert len(history) == 10
    # Oldest evicted, newest kept.
    assert history[-1].amount_units == 50


def test_settled_keys_are_bounded():
    store = InMemoryContextStore(max_settled_keys=5)
    for i in range(20):
        store.record("a", pay("a", key=f"k{i}"))
    assert len(store.context("a").settled_keys) == 5
    assert store.settled("a", "k19")
    assert not store.settled("a", "k0")


def test_agents_are_evicted_least_recently_used_first():
    """
    An evicted agent returns an empty history, which is the cold-start path, not a new one.

    History-level detectors are already required to be silent rather than confident when they
    have no history — that is what makes them safe for a customer's first day.
    """
    store = InMemoryContextStore(max_agents=2)
    store.record("a", pay("a"))
    store.record("b", pay("b"))
    store.context("a")                 # touch a, so b is now least recently used
    store.record("c", pay("c"))

    assert store.context("b").history == []
    assert len(store.context("a").history) == 1
    assert len(store.context("c").history) == 1


def test_stats_report_what_is_held():
    store = InMemoryContextStore()
    store.record("a", pay("a", key="k1"))
    stats = store.stats()
    assert stats["agents"] == 1
    assert stats["actions"] == 1
    assert stats["settled_keys"] == 1


def test_forget_drops_an_agent():
    store = InMemoryContextStore()
    store.record("a", pay("a", key="k1"))
    store.forget("a")
    assert store.context("a").history == []
    assert not store.settled("a", "k1")


# ── the detectors still work against it ──────────────────────────────────

def test_replay_is_caught_across_what_would_be_two_sessions():
    """
    The point of all of this: a key reused for a *different* request is replay, and it must
    still be caught when the two halves arrive in different sessions — because production
    cannot tell the guard that a session ended.
    """
    from acbguard.detectors.registry import RegistryDetector

    store = InMemoryContextStore()
    first = pay("a", key="same-key")
    first.request_fingerprint = "request-one"
    store.record("a", first)

    second = pay("a", key="same-key")
    second.request_fingerprint = "request-two"     # a DIFFERENT request, same key

    ctx = store.context("a")
    risk, flags = RegistryDetector().score(second, ctx)
    assert risk >= 0.9
    assert "idempotency_replay" in flags


# ── derived sessions ─────────────────────────────────────────────────────

def _at(seconds):
    from datetime import datetime, timedelta, timezone
    return datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=seconds)


def stamped(agent_id, seconds):
    action = pay(agent_id)
    action.session_id = None          # as production delivers it: 99.5% of the time
    action.timestamp = _at(seconds)
    return action


def test_a_session_is_derived_from_the_inactivity_gap():
    """
    We know the agent and we know when it acted, so a sitting is recoverable even though
    nothing recorded one. Consecutive actions belong together until the agent goes quiet.
    """
    store = InMemoryContextStore(session_gap_seconds=1800)
    ids = [store.session_id_for("a", stamped("a", t))
           for t in (0, 30, 90, 5_000, 5_030, 60_000)]

    assert ids[0] == ids[1] == ids[2]          # one sitting
    assert ids[3] == ids[4] != ids[2]          # a gap, so a new one
    assert ids[5] != ids[4]                    # and another
    assert len(set(ids)) == 3


def test_a_derived_session_is_marked_as_derived():
    """Nothing downstream may mistake an inferred boundary for an asserted one."""
    store = InMemoryContextStore()
    assert store.session_id_for("a", stamped("a", 0)).startswith("derived:")


def test_a_real_session_id_is_never_overridden():
    store = InMemoryContextStore()
    action = pay("a")
    action.session_id = "sess-from-the-platform"
    assert store.session_id_for("a", action) == "sess-from-the-platform"


def test_agents_get_independent_session_boundaries():
    store = InMemoryContextStore(session_gap_seconds=1800)
    store.session_id_for("a", stamped("a", 0))
    store.session_id_for("b", stamped("b", 0))
    # `a` goes quiet and comes back; `b` does not.
    a_second = store.session_id_for("a", stamped("a", 9_000))
    b_second = store.session_id_for("b", stamped("b", 60))
    assert a_second.endswith(":1")
    assert b_second.endswith(":0")


def test_a_missing_timestamp_does_not_start_a_new_session():
    """Absence of a timestamp is not evidence of a new sitting."""
    store = InMemoryContextStore()
    first = store.session_id_for("a", stamped("a", 0))
    blank = pay("a")
    blank.session_id = None
    blank.timestamp = None
    assert store.session_id_for("a", blank) == first
