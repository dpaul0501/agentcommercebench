"""
Baseline providers — where "normal for this agent" comes from.

The behavioral layer is only as good as its baseline. Without one it falls back to
weak structural signals, which is the cold-start position: real, unavoidable on day
one, and the reason observation has to precede enforcement.

Three sources, in increasing order of usefulness:

    StaticBaseline    declared by hand. Works immediately, no history needed.
    LearnedBaseline   fitted from observed traces. Needs traffic first.
    PlatformBaselines   fetched from the platform, fitted on your own history.
"""
from __future__ import annotations

import statistics
from typing import Any, Iterable, Optional, Protocol, runtime_checkable

DEFAULT_CEILING_UNITS = 3_000_000


@runtime_checkable
class BaselineProvider(Protocol):
    def baseline_for(self, agent_id: str) -> dict[str, Any]:
        """Norms for this agent. Return {} when nothing is known."""


class StaticBaseline:
    """Hand-declared norms. The honest cold-start answer."""

    name = "static"

    def __init__(self, baseline: Optional[dict[str, Any]] = None, **per_agent: dict):
        self.default = baseline or {}
        self.per_agent = per_agent

    def baseline_for(self, agent_id: str) -> dict[str, Any]:
        return {**self.default, **self.per_agent.get(agent_id, {})}


class NullBaseline:
    """Knows nothing. Behavioral detection degrades to structural signals only."""

    name = "null"

    def baseline_for(self, agent_id: str) -> dict[str, Any]:
        return {}


class LearnedBaseline:
    """
    Fits norms from observed traces.

    Unsupervised: it models what this agent *usually* does, not what is fraudulent.
    There are no ground-truth fraud labels in production, so anything claiming to be
    a trained fraud classifier here would be overstating it.

    Thresholds follow mean + k*stdev on log-scale amounts, which is the same shape
    used in the research track.
    """

    name = "learned"

    def __init__(
        self,
        k: float = 1.1,
        min_samples: int = 20,
        hard_k: float = 4.0,
        hard_margin: float = 3.0,
    ):
        self.k = k
        """Escalation threshold, in stdevs of log-amount. Soft: costs a review."""
        self.hard_k = hard_k
        """Hard-block threshold. Far out, because a wrong block is the costly error."""
        self.hard_margin = hard_margin
        """Hard ceiling is also at least this multiple of the largest amount seen."""
        self.min_samples = min_samples
        self._fitted: dict[str, dict[str, Any]] = {}

    def fit(self, records: Iterable[dict[str, Any]]) -> "LearnedBaseline":
        """
        Fit from trace records as written by a sink.

        Accepts either {"action": {...}} envelopes or bare action dicts.
        """
        by_agent: dict[str, list[dict]] = {}
        for record in records:
            action = record.get("action", record)
            if not isinstance(action, dict):
                continue
            agent = action.get("agent_id")
            if agent:
                by_agent.setdefault(agent, []).append(action)

        for agent, actions in by_agent.items():
            fitted = self._fit_one(actions)
            if fitted:
                self._fitted[agent] = fitted
        return self

    def _fit_one(self, actions: list[dict]) -> dict[str, Any]:
        amounts = [
            a["amount_units"]
            for a in actions
            if isinstance(a.get("amount_units"), (int, float)) and a["amount_units"] > 0
        ]
        services = {a.get("service_id") for a in actions if a.get("service_id")}
        hours = sorted(
            {
                int(a["timestamp"][11:13])
                for a in actions
                if isinstance(a.get("timestamp"), str) and len(a["timestamp"]) >= 13
            }
        )

        baseline: dict[str, Any] = {}
        if services:
            baseline["known_services"] = services
        if hours:
            baseline["active_hours"] = range(hours[0], hours[-1] + 1)

        if len(amounts) >= self.min_samples:
            median = statistics.median(amounts)
            baseline["typical_amount_units"] = int(median)
            baseline["samples"] = len(amounts)

            # Log-scale spread, so one large legitimate purchase does not drag the
            # limits up the way a raw mean would.
            import math

            logs = [math.log(a) for a in amounts]
            mu, sigma = statistics.mean(logs), (
                statistics.stdev(logs) if len(logs) > 1 else 0.0
            )

            # Two limits, and the distinction matters.
            #
            # soft_limit sits at mu + k*sigma. On a lognormal that is roughly the
            # 86th percentile at k=1.1 — so a meaningful share of perfectly ordinary
            # traffic exceeds it. That is fine for *escalation*, and would be a
            # disaster as a hard block.
            #
            # ceiling is the hard block, so it is deliberately set far above
            # anything observed. Hard-blocking legitimate traffic is the expensive
            # error; escalating it merely costs a review.
            baseline["soft_limit_units"] = int(math.exp(mu + self.k * sigma))
            baseline["ceiling_units"] = max(
                int(math.exp(mu + self.hard_k * sigma)),
                int(max(amounts) * self.hard_margin),
            )
        elif amounts:
            baseline["typical_amount_units"] = int(statistics.median(amounts))
            baseline["samples"] = len(amounts)
            baseline["cold_start"] = True

        return baseline

    def baseline_for(self, agent_id: str) -> dict[str, Any]:
        return dict(self._fitted.get(agent_id, {}))

    @property
    def agents(self) -> list[str]:
        return sorted(self._fitted)


def __getattr__(name: str):
    if name == "PlatformBaselines":
        from .platform import PlatformBaselines

        return PlatformBaselines
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "BaselineProvider",
    "StaticBaseline",
    "NullBaseline",
    "LearnedBaseline",
    "PlatformBaselines",
    "DEFAULT_CEILING_UNITS",
]
