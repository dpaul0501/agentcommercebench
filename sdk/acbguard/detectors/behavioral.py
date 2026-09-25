"""
Behavioral layer — judges the action against what this agent normally does.

Needs a baseline. With one it catches velocity, ratcheting, drift, and off-hours
activity; without one it can only fall back to weak structural signals, which is
exactly the cold-start position described in the lifecycle design.

Baseline keys (all optional):
    log_mu, log_sigma      float  fitted log-normal for this agent's spend; preferred
    typical_amount_units   int    median payment (fallback when no fit is available)
    known_services         set    services seen before
    active_hours           range  normal UTC hours
    max_burst_per_minute   int    payments/minute considered normal
"""
from __future__ import annotations

import math
from datetime import timedelta

from ..schema import ActionType, Action
from .base import Context

_PAY = (ActionType.AUTHORIZE, ActionType.A2A_TRANSFER)


class BehavioralDetector:
    name = "behavioral"

    def __init__(
        self,
        burst_window: timedelta = timedelta(minutes=1),
        recon_ratio: float = 6.0,
        z_escalate: float = 2.5,
        z_block: float = 4.0,
    ):
        self.burst_window = burst_window
        self.recon_ratio = recon_ratio
        self.z_escalate = z_escalate
        """Standard deviations above the agent's own log-mean before a payment is worth a
        review. Calibrate from clean training traffic — `benchmark.calibrate` fits it to a
        chosen false-positive budget rather than leaving it a guess."""
        self.z_block = z_block

    def score(self, action: Action, ctx: Context) -> tuple[float, list[str]]:
        risk = 0.0
        flags: list[str] = []
        base = ctx.baseline

        if action.action_type in _PAY:
            risk, flags = self._score_payment(action, ctx, base, risk, flags)

        # Reconnaissance: lots of lookups, almost no purchases.
        lookups = sum(
            1
            for a in ctx.history
            if a.action_type in (ActionType.FIND_SERVICE, ActionType.GET_SERVICE)
        )
        payments = sum(1 for a in ctx.history if a.action_type in _PAY)
        if lookups >= 10 and payments * self.recon_ratio < lookups:
            risk = max(risk, 0.45)
            flags.append("reconnaissance_pattern")

        return risk, flags

    def _score_payment(self, action, ctx, base, risk, flags):
        recent = [
            a
            for a in ctx.history
            if a.action_type in _PAY
            and abs((action.timestamp - a.timestamp).total_seconds())
            <= self.burst_window.total_seconds()
        ]
        burst_cap = base.get("max_burst_per_minute", 4)
        if len(recent) >= burst_cap:
            risk = max(risk, 0.65)
            flags.append(f"velocity_{len(recent)}_in_{int(self.burst_window.total_seconds())}s")

        # Prefer a z-score against the agent's own fitted log-normal when the baseline
        # carries one.
        #
        # Fixed ratio bands cannot work across agents whose spend has different spread. The
        # measured catalogue has per-category sigma from 1.0 to 3.4, and at sigma=2.8 a 3x
        # deviation from the median is the 35th percentile — utterly ordinary. Those bands
        # fired on 48% of clean sessions. A z-score asks the only question that transfers
        # between agents: how unusual is this *for this agent*.
        log_mu, log_sigma = base.get("log_mu"), base.get("log_sigma")
        if (log_mu is not None and log_sigma and action.amount_units
                and action.amount_units > 0):
            z = (math.log(action.amount_units) - log_mu) / log_sigma
            if z >= self.z_block:
                risk = max(risk, 0.65)
                flags.append(f"amount_z{z:.1f}")
            elif z >= self.z_escalate:
                risk = max(risk, 0.45)
                flags.append(f"amount_z{z:.1f}")
        else:
            typical = base.get("typical_amount_units")
            if typical and action.amount_units:
                ratio = action.amount_units / typical
                if ratio >= 8:
                    # Escalate, never block on this alone. The ratio is measured against a
                    # fitted median, so a thin or skewed clean sample makes an ordinary
                    # purchase from a dearer merchant look extreme — on a four-merchant
                    # catalog this hard-blocked 21% of legitimate traffic. The hard stop is
                    # `ceiling_units`, which is set far enough out that crossing it is
                    # genuinely abnormal; distributional evidence earns a review.
                    risk = max(risk, 0.65)
                    flags.append(f"amount_{ratio:.1f}x_typical")
                elif ratio >= 3:
                    risk = max(risk, 0.45)
                    flags.append(f"amount_{ratio:.1f}x_typical")

        # Monotonic growth across payments is ratcheting even when each step is small.
        prior = [a.amount_units for a in ctx.history if a.action_type in _PAY and a.amount_units]
        if len(prior) >= 3 and action.amount_units:
            series = prior[-3:] + [action.amount_units]
            if all(b > a for a, b in zip(series, series[1:])):
                risk = max(risk, 0.50)
                flags.append("amount_ratcheting")

        known = base.get("known_services")
        if known and action.service_id and action.service_id not in known:
            severity = 0.55 if (action.amount_units or 0) > 1_000_000 else 0.35
            risk = max(risk, severity)
            flags.append("unknown_service")

        hours = base.get("active_hours")
        if hours is not None and action.timestamp.hour not in hours:
            risk = max(risk, 0.40)
            flags.append(f"off_hours_{action.timestamp.hour:02d}h")

        if action.category == "off_persona":
            risk = max(risk, 0.50)
            flags.append("off_persona_category")

        return risk, flags
