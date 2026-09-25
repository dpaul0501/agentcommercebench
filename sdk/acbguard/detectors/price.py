"""
Price layer — the requested amount against what the operation should cost.

Needs a price oracle. Without one this layer only catches self-inconsistent
payloads (a quoted price that disagrees with the amount being authorized), which
is still enough for price-poisoning probes.
"""
from __future__ import annotations

from typing import Optional

from ..schema import ActionType, Action
from .base import Context

_PAY = (ActionType.AUTHORIZE, ActionType.A2A_TRANSFER)


class PriceDetector:
    name = "price"

    def __init__(self, oracle: Optional[dict[str, int]] = None, ceiling_units: Optional[int] = None):
        self.oracle = oracle or {}
        """service_id or 'service:operation' -> expected units."""
        self.ceiling_units = ceiling_units

    def _expected(self, action: Action) -> Optional[int]:
        if action.service_id and action.operation_id:
            keyed = self.oracle.get(f"{action.service_id}:{action.operation_id}")
            if keyed:
                return keyed
        if action.service_id:
            return self.oracle.get(action.service_id)
        return None

    def score(self, action: Action, ctx: Context) -> tuple[float, list[str]]:
        if action.action_type not in _PAY:
            return 0.0, []

        risk = 0.0
        flags: list[str] = []
        amount = action.amount_units

        # Payload asserts a price that disagrees with the authorized amount.
        quoted = action.payload.get("quoted_price_units")
        if isinstance(quoted, (int, float)) and amount:
            if quoted > amount * 2:
                risk = max(risk, 0.75)
                flags.append(f"payload_price_{quoted / max(amount, 1):.1f}x_amount")

        expected = self._expected(action)
        if expected and amount:
            ratio = amount / expected
            if ratio >= 10:
                risk = max(risk, 0.90)
                flags.append(f"amount_{ratio:.1f}x_oracle")
            elif ratio >= 5:
                risk = max(risk, 0.60)
                flags.append(f"amount_{ratio:.1f}x_oracle")

        # Two limits with different consequences. Exceeding the soft limit is common
        # in ordinary traffic and only warrants a review; the hard ceiling is set far
        # enough out that crossing it is genuinely abnormal, so it blocks.
        soft = ctx.baseline.get("soft_limit_units")
        if soft and amount and amount > soft:
            risk = max(risk, 0.45)
            flags.append("over_soft_limit")

        ceiling = self.ceiling_units or ctx.baseline.get("ceiling_units")
        if ceiling and amount and amount > ceiling:
            risk = max(risk, 0.95)
            flags.append("over_ceiling")

        return risk, flags
