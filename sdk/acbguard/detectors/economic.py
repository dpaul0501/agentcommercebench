"""
Economic detection — was too much taken, by a counterparty who is exactly who they claim to be?

Every other detector in this package asks a security question: is this request well-formed, is
this counterparty who it says it is, was the agent manipulated. Those catch impersonation,
substitution and injection.

They do not catch a genuine seller charging more than it should. Correct domain, correct
settlement address, correct protocol, real service delivered — and 26% more than the listed
price. Nothing about the identity is wrong, so nothing keyed on identity will ever fire.

**The single transaction cannot answer this.** A price is only wrong relative to something, and
the reference has to come from outside the payment:

    catalogue   what the service lists     -> overcharge, phantom fee
    population  what other agents pay      -> price discrimination
    history     what this service charged  -> price drift, silent increases
    session     what was first quoted      -> drip pricing

The agent's own history is deliberately *not* on that list. It is the reference every other
detector here uses, and it is the one a patient seller can move — charge this buyer a little
more each month and its baseline follows. Against a counterparty, the anchor has to be outside
the buyer. This is the same reason card issuers keep merchant-level and peer references
alongside cardholder history.

Silent downgrade — full price, cheap tier delivered — is not addressed here, because nothing
about the payment is wrong. Whether it is detectable at the payment layer at all is an open
question and should not be claimed either way.
"""
from __future__ import annotations

import statistics
from typing import Any, Optional

from ..schema import Action, ActionType
from .base import Context

_PAY = (ActionType.AUTHORIZE, ActionType.A2A_TRANSFER)


class PriceReference:
    """
    What a service *should* cost, from sources outside this transaction.

    Every field is optional. A reference with nothing in it makes the detector silent rather
    than wrong, which is the right failure mode for something that will run before any history
    exists.
    """

    def __init__(
        self,
        catalogue: Optional[dict[str, int]] = None,
        population: Optional[dict[str, list[int]]] = None,
        service_history: Optional[dict[str, list[int]]] = None,
    ):
        self.catalogue = catalogue or {}
        """service_id -> listed price in micro-units."""
        self.population = population or {}
        """service_id -> what OTHER agents have paid. The only reference that survives a
        seller patiently moving one buyer's expectations."""
        self.service_history = service_history or {}
        """service_id -> prices observed over time, independent of who paid them."""

    def peer_median(self, service_id: str) -> Optional[float]:
        prices = self.population.get(service_id or "")
        return statistics.median(prices) if prices and len(prices) >= 3 else None

    def historical_median(self, service_id: str) -> Optional[float]:
        prices = self.service_history.get(service_id or "")
        return statistics.median(prices) if prices and len(prices) >= 3 else None

    @classmethod
    def from_sessions(cls, sessions: list[dict[str, Any]],
                      catalogue: Optional[dict[str, int]] = None) -> "PriceReference":
        """
        Build the population reference from observed clean traffic.

        Prices are pooled across *all* agents on purpose. A per-agent view cannot see price
        discrimination, because the discriminated price is that agent's normal.
        """
        population: dict[str, list[int]] = {}
        for session in sessions:
            for action in session.get("actions", []):
                if action.get("action_type") != "authorize":
                    continue
                service, amount = action.get("service_id"), action.get("amount_units")
                if service and amount:
                    population.setdefault(service, []).append(int(amount))
        return cls(catalogue=catalogue, population=population)


class EconomicDetector:
    """
    Scores what was charged against what should have been charged.

        EconomicDetector(PriceReference(catalogue=..., population=...))

    Silent without a reference, so it is safe to include in any pipeline.
    """

    name = "economic"

    def __init__(self, reference: Optional[PriceReference] = None,
                 catalogue_tolerance: Optional[float] = None,
                 peer_tolerance: Optional[float] = None,
                 catalogue_block: Optional[float] = None,
                 peer_block: Optional[float] = None,
                 quote_tolerance: Optional[float] = None):
        self.reference = reference or PriceReference()
        self.catalogue_tolerance = catalogue_tolerance
        self.peer_tolerance = peer_tolerance
        """
        Where review starts. Fitted to a flag budget, and `None` until it has been.

        There is no sensible default. The previous ones — 1.10 against the catalogue, 1.25
        against peers — were guesses, and they were guesses that happened to sit inside the
        margins the benchmark's own overcharge classes draw from. A check that has not been
        calibrated stays silent instead of guessing; silence is a known quantity and a
        borrowed constant is not.
        """
        self.catalogue_block = catalogue_block
        self.peer_block = peer_block
        self.quote_tolerance = quote_tolerance
        """How far above the quoted price is still ordinary. Fitted like the rest: honest
        merchants quote in the 402 challenge and then charge close to it, but not exactly it,
        so this has a real distribution to sit inside rather than a gap."""
        """
        Where refusal starts, and a separate statistic on purpose.

        Prices genuinely rise. A single tolerance used for both review and refusal blocks the
        legitimate tail — with one threshold at the 90th percentile of clean price ratios,
        22% of honest sessions were refused. Review is cheap and refusal is not, so they are
        fitted to different budgets: the same soft-limit-and-ceiling split the amount checks
        already use.
        """

    def score(self, action: Action, ctx: Context) -> tuple[float, list[str]]:
        if action.action_type not in _PAY or not action.amount_units:
            return 0.0, []

        reference = self.reference
        if ctx.baseline.get("price_reference"):
            reference = ctx.baseline["price_reference"]

        risk = 0.0
        flags: list[str] = []
        service = action.service_id or ""
        amount = action.amount_units

        # 1. Against the listed price. Catches an overcharge and a padded fee alike — from the
        #    payment's point of view a fee that buys nothing and a price that is too high are
        #    the same event.
        listed = reference.catalogue.get(service) if self.catalogue_tolerance else None
        if listed:
            ratio = amount / listed
            if self.catalogue_block and ratio > self.catalogue_block:
                risk = max(risk, 0.75)
                flags.append(f"above_listed_{ratio:.2f}x")
            elif ratio > self.catalogue_tolerance:
                risk = max(risk, 0.55)
                flags.append(f"above_listed_{ratio:.2f}x")

        # 2. Against what other agents pay. The only check that sees price discrimination,
        #    because a discriminated price is by definition this agent's normal.
        peer = reference.peer_median(service) if self.peer_tolerance else None
        if peer:
            ratio = amount / peer
            if self.peer_block and ratio > self.peer_block:
                risk = max(risk, 0.80)
                flags.append(f"above_peers_{ratio:.2f}x")
            elif ratio > self.peer_tolerance:
                risk = max(risk, 0.60)
                flags.append(f"above_peers_{ratio:.2f}x")

        # 3. Against the service's own price history, which moves independently of any buyer.
        historical = reference.historical_median(service) if self.peer_tolerance else None
        if historical:
            ratio = amount / historical
            if ratio > self.peer_tolerance:
                risk = max(risk, 0.55)
                flags.append(f"above_history_{ratio:.2f}x")

        # 4. Against the opening quote in this session. Drip pricing is invisible to every
        #    other reference, because each individual price is defensible.
        quoted = action.payload.get("quoted_price_units") if action.payload else None
        if not quoted:
            for prior in ctx.history:
                if (prior.service_id == service and prior.action_type in _PAY
                        and prior.amount_units):
                    quoted = prior.amount_units
                    break
        if self.quote_tolerance and quoted and amount > quoted * self.quote_tolerance:
            risk = max(risk, 0.65)
            flags.append(f"above_opening_quote_{amount / quoted:.2f}x")

        return risk, flags


class DuplicateChargeDetector:
    """
    The same purchase paid for twice.

    Distinct from the idempotency-key check in `RegistryDetector`, which catches a *reused*
    key. This catches the opposite and more common case: a retry that regenerated its key, so
    the payment looks new to everyone and settles twice.

    That is the default behaviour of most agent frameworks, and a seller can induce it simply
    by failing the first attempt. Whether the cause is a hostile merchant or an honest bug, the
    money is equally gone — so this fires on both, and the difference is a question for
    whoever reads the flag.
    """

    name = "duplicate_charge"

    def __init__(self, window_seconds: float = 300.0):
        self.window_seconds = window_seconds

    def score(self, action: Action, ctx: Context) -> tuple[float, list[str]]:
        if action.action_type not in _PAY or not action.amount_units:
            return 0.0, []

        # The fingerprint is what makes this decidable. Two purchases of the same thing at
        # the same price are identical in amount, service and timing — an agent buying three
        # searches looks exactly like an agent paying three times for one. Only the request
        # identity separates them, and production records one on every settlement.
        fingerprint = action.request_fingerprint
        if not fingerprint:
            return 0.0, []

        for prior in ctx.history:
            if prior.action_type not in _PAY or not prior.amount_units:
                continue
            if prior.request_fingerprint != fingerprint:
                continue  # a different purchase, however similar it looks
            if prior.idempotency_key and action.idempotency_key \
                    and prior.idempotency_key == action.idempotency_key:
                continue  # correctly keyed retry — the registry layer owns this
            if action.timestamp and prior.timestamp:
                gap = abs((action.timestamp - prior.timestamp).total_seconds())
                if gap > self.window_seconds:
                    continue
            return 0.70, ["duplicate_charge", f"same_request:{fingerprint[:32]}"]
        return 0.0, []


__all__ = ["DuplicateChargeDetector", "EconomicDetector", "PriceReference"]
