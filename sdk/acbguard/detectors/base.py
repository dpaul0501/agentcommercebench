"""
Detector contract and pipeline.

    detector(action, context) -> (risk_score, flags)
    risk_score in [0, 1]

    0.00 - 0.29  allow
    0.30 - 0.69  escalate
    0.70 - 1.00  block

The pipeline is deliberately a max, not a mean: any single layer must be able to
raise the verdict on its own. This mirrors the block-wins merge in the platform
engine — a decision can only ever get stricter as more evidence arrives.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, runtime_checkable

from ..schema import Action, Decision, Session

ESCALATE_AT = 0.30
BLOCK_AT = 0.70


@dataclass
class Context:
    """What a detector knows beyond the single action in front of it."""

    session: Optional[Session] = None
    history: list[Action] = field(default_factory=list)
    """Prior actions this run, oldest first."""
    baseline: dict[str, Any] = field(default_factory=dict)
    """Learned or declared norms for this agent: limits, known vendors, hours."""
    settled_keys: set[str] = field(default_factory=set)
    session_id: Optional[str] = None
    """
    The current sitting — asserted by the platform, or derived from an inactivity gap.

    A value beginning `derived:` was inferred from timestamps, not asserted by anything. Use
    it to group, never to authenticate: a derived boundary is a guess about attention, and the
    identity question is answered by `principal_id`.
    """
    principal_id: Optional[str] = None
    """
    The *authenticated* agent — who the caller proved they are, from the API key or token.

    Distinct from `action.agent_id`, which is only what the request claims. The gap between
    the two is the impersonation signal, and it is the one thing here that must not be read
    from a session: `session_id` is populated on 0.47% of production settlements, so a check
    keyed on `ctx.session.agent_id` fires in a harness and never in deployment. It scored
    1.00 against the identity-mismatch class here and 0.30 once history was keyed the way
    production keys it.
    """


@runtime_checkable
class Detector(Protocol):
    name: str

    def score(self, action: Action, ctx: Context) -> tuple[float, list[str]]: ...


@dataclass
class Verdict:
    risk_score: float
    flags: list[str]
    layer_scores: dict[str, float]
    escalate_at: float = ESCALATE_AT
    block_at: float = BLOCK_AT
    """
    Thresholds carried on the verdict so a pipeline can be calibrated as a whole.

    Fitting each detector to a 10% false-positive budget does NOT give a 10% pipeline: seven
    detectors firing independently at 10% compounded to 47% here. The budget is a property of
    the decision, so it has to be fitted on the aggregate score.
    """

    @property
    def decision(self) -> Decision:
        if self.risk_score >= self.block_at:
            return Decision.BLOCK
        if self.risk_score >= self.escalate_at:
            return Decision.ESCALATE
        return Decision.ALLOW

    @property
    def would_stop(self) -> bool:
        """True if this verdict prevents the action reaching the rail."""
        return self.decision in (Decision.BLOCK, Decision.ESCALATE)

    def to_dict(self) -> dict[str, Any]:
        return {
            "risk_score": round(self.risk_score, 3),
            "decision": self.decision.value,
            "flags": self.flags,
            "layers": {k: round(v, 3) for k, v in self.layer_scores.items()},
        }


class Pipeline:
    """Runs detectors over an action and takes the strictest verdict."""

    def __init__(self, detectors: list[Detector],
                 escalate_at: float = ESCALATE_AT, block_at: float = BLOCK_AT):
        self.escalate_at = escalate_at
        self.block_at = block_at
        self.detectors = list(detectors)

    def score(self, action: Action, ctx: Optional[Context] = None) -> Verdict:
        ctx = ctx or Context()
        layers: dict[str, float] = {}
        flags: list[str] = []
        # Names are keys, so two detectors sharing one would have the second silently
        # overwrite the first and `max` would never see it. Disambiguating here rather than
        # forbidding duplicates keeps wrappers and decorators usable, which is how the
        # collision arises in practice.
        seen: dict[str, int] = {}
        for det in self.detectors:
            name = det.name
            if name in seen:
                seen[name] += 1
                name = f"{name}#{seen[name]}"
            else:
                seen[name] = 0
            try:
                risk, det_flags = det.score(action, ctx)
            except Exception as exc:  # a broken detector must not swallow the action
                layers[name] = 0.0
                flags.append(f"{name}:error:{type(exc).__name__}")
                continue
            layers[name] = risk
            flags.extend(f"{name}:{f}" for f in det_flags)
        top = max(layers.values(), default=0.0)
        return Verdict(risk_score=top, flags=flags, layer_scores=layers,
                       escalate_at=self.escalate_at, block_at=self.block_at)

    def score_session(self, session: Session) -> list[Verdict]:
        """Score every action, accumulating history and settled keys as we go."""
        ctx = Context(session=session)
        verdicts = []
        for action in session.actions:
            verdicts.append(self.score(action, ctx))
            ctx.history.append(action)
            if action.idempotency_key:
                ctx.settled_keys.add(action.idempotency_key)
        return verdicts


def default_pipeline() -> Pipeline:
    """The layers that ship with acbguard."""
    from .payload import PayloadDetector
    from .behavioral import BehavioralDetector
    from .price import PriceDetector
    from .registry import RegistryDetector
    from .reasoning import ReasoningDetector
    from .catalog import CatalogDetector

    return Pipeline(
        [
            PayloadDetector(),
            BehavioralDetector(),
            PriceDetector(),
            RegistryDetector(),
            # Scores zero without a registry, so it is inert until one is supplied.
            CatalogDetector(),
            ReasoningDetector(),
        ]
    )
