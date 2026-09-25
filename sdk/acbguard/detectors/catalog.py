"""
D3/D5 — the counterparty is not who the catalog says it is.

The harness can already produce hostile *merchants*: a lookalike domain, a swapped settlement
address, a price well above the catalogued one. Nothing in this package could detect them,
which meant those attacks scored zero at L1 for want of a detector rather than for want of a
signal — and made the reasoning layer look better than it is by comparison.

Everything here compares the action against a registry of what the catalog promised. That is
the whole idea: these attacks are invisible in isolation, because a payment to a lookalike
domain is perfectly well-formed. They are only visible against a reference.

The registry is `{service_id: {"domain": ..., "payee": ..., "price_units": ...}}`. With no
registry the detector scores zero, so it is safe to include in any pipeline.
"""
from __future__ import annotations

import unicodedata
from typing import Any, Optional
from urllib.parse import urlparse

from ..schema import Action, ActionType
from .base import Context

_PAY = (ActionType.AUTHORIZE, ActionType.A2A_TRANSFER)


def domain_of(value: Optional[str]) -> str:
    """Host part of a URL, or the value itself when it is already a bare domain."""
    if not value:
        return ""
    value = value.strip()
    parsed = urlparse(value if "//" in value else f"//{value}")
    return (parsed.netloc or parsed.path).split("/")[0].split(":")[0].lower()


def levenshtein(a: str, b: str, cap: int = 3) -> int:
    """Edit distance, short-circuited at `cap` — we only care about near-misses."""
    if a == b:
        return 0
    if abs(len(a) - len(b)) > cap:
        return cap + 1
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1,
                               previous[j - 1] + (ca != cb)))
        if min(current) > cap:
            return cap + 1
        previous = current
    return previous[-1]


def confusable_skeleton(domain: str) -> str:
    """
    Fold a domain to its ASCII lookalike form.

    A homoglyph attack replaces a Latin letter with a visually identical character from
    another script, so the bytes differ while the rendering does not. Decomposing and
    stripping combining marks catches the accented cases; the explicit map covers the
    Cyrillic and Greek letters that carry no decomposition.
    """
    table = {"а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "х": "x", "у": "y",
             "ѕ": "s", "і": "i", "ј": "j", "һ": "h", "ԁ": "d", "ɡ": "g",
             "α": "a", "ε": "e", "ο": "o", "ρ": "p", "τ": "t", "υ": "u", "ν": "v"}
    folded = "".join(table.get(ch, ch) for ch in domain)
    decomposed = unicodedata.normalize("NFKD", folded)
    return "".join(c for c in decomposed if not unicodedata.combining(c)).lower()


class CatalogDetector:
    """
    Scores an action against the registered identity of the service it claims to pay.

        CatalogDetector({"exa": {"domain": "exa.ai", "payee": "0xEXA…",
                                 "price_units": 50_000}})
    """

    name = "catalog"

    def __init__(self, registry: Optional[dict[str, dict[str, Any]]] = None,
                 price_tolerance: float = 1.5):
        self.registry = registry or {}
        self.price_tolerance = price_tolerance
        """Multiple of the catalogued price above which the charge is flagged."""

    def score(self, action: Action, ctx: Context) -> tuple[float, list[str]]:
        if action.action_type not in _PAY or not action.service_id:
            return 0.0, []
        entry = self.registry.get(action.service_id) or ctx.baseline.get(
            "catalog", {}).get(action.service_id)
        if not entry:
            return 0.0, []

        risk = 0.0
        flags: list[str] = []
        expected = domain_of(entry.get("domain"))
        seen = domain_of(action.endpoint or action.vendor)

        if expected and seen and seen != expected:
            # Rank the explanations, most specific first — "homoglyph" tells an operator
            # far more than "mismatch", and they are not mutually exclusive.
            if not seen.isascii() and confusable_skeleton(seen) == expected:
                risk = max(risk, 0.95)
                flags.append(f"homoglyph_domain:{seen}")
            elif not seen.isascii():
                risk = max(risk, 0.90)
                flags.append(f"non_ascii_domain:{seen}")
            elif 0 < levenshtein(seen, expected) <= 2:
                risk = max(risk, 0.90)
                flags.append(f"typosquat_domain:{seen}")
            else:
                risk = max(risk, 0.85)
                flags.append(f"catalog_domain_mismatch:{seen}")

        registered_payee = entry.get("payee")
        if registered_payee and action.payee and action.payee != registered_payee:
            # Escalate, do not block.
            #
            # Measured on production traffic (2026-09-08): PAY_TO_ADDRESS_CHANGED fired on
            # 14 of 1,027 scored settlements — real payments, by real agents, that settled
            # normally. Services rotate their settlement addresses, so a changed payee is
            # routine rather than conclusive. The deployed detector scores it 50 against a
            # block threshold of 60 — warn, not block. Scoring it 0.95 here would have
            # blocked 14 legitimate payments, making it the largest single source of false
            # positives in this package.
            risk = max(risk, 0.60)
            flags.append("payee_changed")

        catalogued = entry.get("price_units")
        if catalogued and action.amount_units and action.amount_units > \
                catalogued * self.price_tolerance:
            ratio = action.amount_units / catalogued
            risk = max(risk, 0.80 if ratio < 5 else 0.90)
            flags.append(f"price_above_catalog_{ratio:.1f}x")

        return risk, flags


__all__ = ["CatalogDetector", "confusable_skeleton", "domain_of", "levenshtein"]
