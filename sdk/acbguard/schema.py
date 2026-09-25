"""
Core types shared by probes, targets, and detectors.

The action model is deliberately commerce-shaped: an agent action that could move
money. It is a superset of what Platform's platform records, so a trace captured here
can be replayed against the platform and vice versa.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict, fields
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional
import uuid


MICRO_PER_USD = 1_000_000
"""Money is integer micro-units throughout. 1_000_000 == $1.00 == 1 USDC atom-scale."""


class ActionType(str, Enum):
    FIND_SERVICE = "find_service"
    GET_SERVICE = "get_service"
    AUTHORIZE = "authorize"
    A2A_TRANSFER = "a2a_transfer"
    SETTLE = "settle"


class Decision(str, Enum):
    ALLOW = "allow"
    ESCALATE = "escalate"
    BLOCK = "block"


class Persona(str, Enum):
    PROCUREMENT = "procurement"
    RESEARCH = "research"
    TRAVEL = "travel"


class Outcome(str, Enum):
    """What the target under test did with a probe."""

    DEFENDED = "defended"
    """Target blocked or escalated the action. Good."""

    VULNERABLE = "vulnerable"
    """Target executed the action. The attack landed."""

    ERROR = "error"
    """Target failed in a way we can't score (network, auth, malformed)."""

    SKIPPED = "skipped"
    """Probe not applicable to this target."""


def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class Action:
    """A single consequential action an agent attempts."""

    action_type: ActionType
    agent_id: str = "agent-under-test"
    session_id: str = field(default_factory=lambda: f"sess-{uuid.uuid4().hex[:12]}")
    action_id: str = field(default_factory=lambda: f"act-{uuid.uuid4().hex[:12]}")
    timestamp: datetime = field(default_factory=_now)

    service_id: Optional[str] = None
    operation_id: Optional[str] = None
    amount_units: Optional[int] = None
    vendor: Optional[str] = None
    category: Optional[str] = None
    endpoint: Optional[str] = None
    payee: Optional[str] = None
    request_fingerprint: Optional[str] = None
    """Hash of the request this payment is for. Production records one on every settlement.
    It is the only thing that distinguishes ONE purchase paid twice from TWO purchases of the
    same item at the same price — which are identical in amount, service and timing."""
    """Settlement destination actually used, for comparison against the registry."""
    idempotency_key: Optional[str] = None

    payload: dict[str, Any] = field(default_factory=dict)
    """The request body. Primary signal for payload-layer detection."""

    # ── Reasoning layer (L0) ─────────────────────────────────────────────
    # Present only when the integration can see the model, i.e. an SDK hook
    # around the model client or an observability feed. An MCP or rail
    # integration will always leave these empty.

    reasoning: Optional[str] = None
    """The model's visible thinking immediately before this action."""

    stated_intent: Optional[str] = None
    """What the agent said it was about to do, if it announced one."""

    context_sources: list[str] = field(default_factory=list)
    """Provenance of context the model saw: 'user', 'tool:<name>', 'system', 'memory'.
    Injected instructions arriving via 'tool:' are far more suspicious than the same
    text arriving via 'user' — the user is allowed to give instructions."""

    # Ground truth, set by probes.
    probe_id: Optional[str] = None
    is_attack: bool = False

    @property
    def has_reasoning(self) -> bool:
        return bool(self.reasoning or self.stated_intent)

    @property
    def amount_usd(self) -> float:
        return (self.amount_units or 0) / MICRO_PER_USD

    def clone(self, **overrides: Any) -> "Action":
        data = {**asdict(self), **overrides}
        data["action_type"] = ActionType(data["action_type"])
        data["timestamp"] = (
            data["timestamp"]
            if isinstance(data["timestamp"], datetime)
            else datetime.fromisoformat(data["timestamp"])
        )
        return Action(**data)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["action_type"] = self.action_type.value
        d["timestamp"] = self.timestamp.isoformat()
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Action":
        """Build an Action from a JSON record, ignoring keys this schema does not define.

        Records arriving from a trace file, a rail export or the published dataset carry
        fields belonging to whoever wrote them. Dropping the unknown ones is deliberate: a
        loader that raises on an extra column makes every producer's schema change a
        breaking change for every consumer.
        """
        known = {f.name for f in fields(cls)}
        data = {k: v for k, v in d.items() if k in known}
        if "action_type" in data:
            data["action_type"] = ActionType(data["action_type"])
        ts = data.get("timestamp")
        if isinstance(ts, str):
            data["timestamp"] = datetime.fromisoformat(ts)
        return cls(**data)


@dataclass
class Session:
    """An ordered run of actions by one agent."""

    agent_id: str = "agent-under-test"
    persona: Persona = Persona.PROCUREMENT
    session_id: str = field(default_factory=lambda: f"sess-{uuid.uuid4().hex[:12]}")
    actions: list[Action] = field(default_factory=list)
    is_clean: bool = True
    probe_id: Optional[str] = None

    def clone(self) -> "Session":
        return Session(
            agent_id=self.agent_id,
            persona=self.persona,
            session_id=self.session_id,
            actions=[a.clone() for a in self.actions],
            is_clean=self.is_clean,
            probe_id=self.probe_id,
        )

    def of_type(self, *types: ActionType) -> list[int]:
        """Indices of actions matching any given type."""
        return [i for i, a in enumerate(self.actions) if a.action_type in types]

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "agent_id": self.agent_id,
            "persona": self.persona.value,
            "is_clean": self.is_clean,
            "probe_id": self.probe_id,
            "actions": [a.to_dict() for a in self.actions],
        }


@dataclass
class Observation:
    """What the target did with one action."""

    outcome: Outcome
    decision: Optional[Decision] = None
    reason: Optional[str] = None
    latency_ms: float = 0.0
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "outcome": self.outcome.value,
            "decision": self.decision.value if self.decision else None,
            "reason": self.reason,
            "latency_ms": round(self.latency_ms, 1),
        }
