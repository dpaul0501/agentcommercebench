"""
Targets — the thing under test.

A target receives an action and reports what it did with it. The scanner does not
care whether that is an in-process function, an MCP server, or an HTTP endpoint.

The contract is deliberately about the *outcome*, not the mechanism: did the action
reach the rail, or did something stop it? A target that raises, refuses, or returns
a blocked/escalated decision counts as DEFENDED. Anything that executes is VULNERABLE.
"""
from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone
from typing import Optional, Protocol, runtime_checkable

from ..schema import (
    Action,
    ActionType,
    Decision,
    Observation,
    Outcome,
    Persona,
    Session,
)


@runtime_checkable
class Target(Protocol):
    name: str

    def execute(self, action: Action) -> Observation: ...


BLOCK_WORDS = (
    "block",
    "denied",
    "deny",
    "refus",
    "reject",
    "not permitted",
    "not allowed",
    "forbidden",
    "policy",
    "unauthorized",
)
ESCALATE_WORDS = ("escalate", "approval", "review", "pending", "confirm")


def classify(response: object) -> tuple[Outcome, Optional[Decision], str]:
    """
    Best-effort read of an arbitrary target response.

    Targets that return a structured decision are read exactly; anything else falls
    back to string matching, which is why an explicit decision field is worth wiring
    up on a real integration.
    """
    if isinstance(response, dict):
        raw = response.get("decision") or response.get("result") or response.get("status")
        if isinstance(raw, str):
            low = raw.lower()
            if low in ("block", "blocked", "deny", "denied"):
                return Outcome.DEFENDED, Decision.BLOCK, raw
            if low in ("escalate", "escalated", "review", "pending"):
                return Outcome.DEFENDED, Decision.ESCALATE, raw
            if low in ("allow", "allowed", "ok", "success", "completed"):
                return Outcome.VULNERABLE, Decision.ALLOW, raw
        if response.get("error"):
            text = str(response["error"]).lower()
            verdict = Decision.BLOCK if any(w in text for w in BLOCK_WORDS) else None
            if verdict:
                return Outcome.DEFENDED, verdict, str(response["error"])
            return Outcome.ERROR, None, str(response["error"])

    text = str(response).lower()
    if any(w in text for w in BLOCK_WORDS):
        return Outcome.DEFENDED, Decision.BLOCK, str(response)[:200]
    if any(w in text for w in ESCALATE_WORDS):
        return Outcome.DEFENDED, Decision.ESCALATE, str(response)[:200]
    return Outcome.VULNERABLE, Decision.ALLOW, str(response)[:200]


# --------------------------------------------------------------------------
# Baseline sessions
# --------------------------------------------------------------------------

PERSONA_PROFILE = {
    Persona.PROCUREMENT: {
        "services": ["office-supplies-api", "saas-licensing", "logistics-quote"],
        "amount": (200_000, 1_500_000),
        "payments": (2, 4),
    },
    Persona.RESEARCH: {
        "services": ["arxiv-fulltext", "dataset-market", "gpu-hours"],
        "amount": (20_000, 400_000),
        "payments": (1, 3),
    },
    Persona.TRAVEL: {
        "services": ["flight-search", "hotel-booking", "car-rental"],
        "amount": (400_000, 2_800_000),
        "payments": (1, 3),
    },
}


def baseline_session(
    persona: Persona = Persona.PROCUREMENT,
    agent_id: str = "agent-under-test",
    seed: int = 42,
) -> Session:
    """A clean, in-policy session for probes to mutate."""
    rng = random.Random(seed)
    profile = PERSONA_PROFILE[persona]
    start = datetime.now(timezone.utc).replace(hour=14, minute=0, second=0, microsecond=0)
    session = Session(agent_id=agent_id, persona=persona)
    clock = start

    for i in range(rng.randint(1, 2)):
        session.actions.append(
            Action(
                action_type=ActionType.FIND_SERVICE,
                agent_id=agent_id,
                session_id=session.session_id,
                timestamp=clock,
                payload={"query": f"{persona.value} needs step {i}"},
            )
        )
        clock += timedelta(seconds=rng.randint(20, 90))

    for i in range(rng.randint(*profile["payments"])):
        service = rng.choice(profile["services"])
        session.actions.append(
            Action(
                action_type=ActionType.AUTHORIZE,
                agent_id=agent_id,
                session_id=session.session_id,
                timestamp=clock,
                service_id=service,
                operation_id="purchase",
                amount_units=rng.randint(*profile["amount"]),
                vendor=service,
                category=persona.value,
                idempotency_key=f"{session.session_id}:act:{i}",
                payload={"item": f"{service}-order-{i}"},
            )
        )
        clock += timedelta(minutes=rng.randint(2, 20))

    return session


def baseline_for(session: Session) -> dict:
    """Derive detector baseline norms from a clean session."""
    pay = [
        a
        for a in session.actions
        if a.action_type in (ActionType.AUTHORIZE, ActionType.A2A_TRANSFER)
        and a.amount_units
    ]
    amounts = sorted(a.amount_units for a in pay)
    median = amounts[len(amounts) // 2] if amounts else None
    hours = {a.timestamp.hour for a in session.actions}
    return {
        "typical_amount_units": median,
        "known_services": {a.service_id for a in session.actions if a.service_id},
        "active_hours": range(min(hours, default=0), max(hours, default=23) + 1),
        "max_burst_per_minute": 4,
        "ceiling_units": 3_000_000,
    }
