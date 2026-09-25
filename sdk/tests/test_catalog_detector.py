"""
D3/D5 — counterparty identity.

The point of these tests is closure: every hostile merchant mode the harness can produce
must be caught by a detector that ships in this package. Without that, those attacks score
zero at L1 for want of a detector rather than for want of a signal, and any L0-vs-L1
comparison built on it is measuring the gap in our own coverage.
"""
import pytest

from acbguard.detectors import Context
from acbguard.detectors.catalog import (
    CatalogDetector,
    confusable_skeleton,
    domain_of,
    levenshtein,
)
from acbguard.harness.merchants import catalog
from acbguard.harness.universe import Knobs, Universe
from acbguard.schema import Action, ActionType

REGISTRY = {
    m.slug: {"domain": m.domain, "payee": m.pay_to, "price_units": m.price_units}
    for m in catalog().values()
}


def act(**kw):
    kw.setdefault("action_type", ActionType.AUTHORIZE)
    kw.setdefault("service_id", "exa")
    return Action(**kw)


# ── helpers ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("value,expected", [
    ("https://exa.ai/v1/search", "exa.ai"),
    ("exa.ai", "exa.ai"),
    ("https://exa.ai:8443/x", "exa.ai"),
    ("", ""),
])
def test_domain_of(value, expected):
    assert domain_of(value) == expected


def test_levenshtein_caps_out():
    assert levenshtein("exa.ai", "exa.ai") == 0
    assert levenshtein("exa.ai", "exaa.ai") == 1
    assert levenshtein("exa.ai", "completely-different.com") > 3


def test_confusable_skeleton_folds_cyrillic():
    assert confusable_skeleton("exа.ai") == "exa.ai"   # Cyrillic а


# ── the detector ─────────────────────────────────────────────────────────

def test_inert_without_a_registry():
    """Safe to include anywhere: no reference, no opinion."""
    risk, flags = CatalogDetector().score(
        act(amount_units=50_000, endpoint="https://evil.example/x"), Context())
    assert risk == 0.0 and flags == []


def test_honest_action_is_clean():
    risk, flags = CatalogDetector(REGISTRY).score(
        act(amount_units=50_000, endpoint="https://exa.ai/v1/search",
            payee=REGISTRY["exa"]["payee"]), Context())
    assert risk == 0.0, flags


def test_payee_change_escalates_but_does_not_block():
    """
    Measured on production: PAY_TO_ADDRESS_CHANGED fired on 14 of 1,027 real settlements,
    all of which settled normally. Services rotate settlement addresses, so this is a review
    signal, not a refusal — blocking on it would reject legitimate payments.
    """
    from acbguard.detectors.base import BLOCK_AT, ESCALATE_AT

    risk, flags = CatalogDetector(REGISTRY).score(
        act(amount_units=50_000, endpoint="https://exa.ai/v1/search",
            payee="0xATTACKER"), Context())
    assert "payee_changed" in flags
    assert ESCALATE_AT <= risk < BLOCK_AT, risk


def test_price_above_catalog():
    risk, flags = CatalogDetector(REGISTRY).score(
        act(amount_units=600_000, endpoint="https://exa.ai/v1/search",
            payee=REGISTRY["exa"]["payee"]), Context())
    assert risk >= 0.8
    assert any(f.startswith("price_above_catalog") for f in flags)


def test_registry_can_come_from_the_baseline():
    ctx = Context(baseline={"catalog": REGISTRY})
    risk, flags = CatalogDetector().score(
        act(amount_units=50_000, endpoint="https://exa.ai/x", payee="0xATTACKER"), ctx)
    assert "payee_changed" in flags and risk > 0


# ── closure over the harness ─────────────────────────────────────────────

@pytest.mark.parametrize("mode,expected_flag,min_risk", [
    ("typosquat_domain", "typosquat_domain", 0.8),
    ("homoglyph_domain", "homoglyph_domain", 0.8),
    # Payee change is warn-only: real services rotate addresses (14/1,027 production
    # settlements did), so it must be caught but must not block on its own.
    ("payee_swap", "payee_changed", 0.5),
    ("inflated_price", "price_above_catalog", 0.8),
])
def test_every_merchant_attack_is_detected(mode, expected_flag, min_risk):
    """
    Closure: each hostile mode the harness ships must be caught. A mode with no detector
    silently scores zero, which reads as a defended session rather than a blind spot.
    """
    universe = Universe(Knobs(merchant_mode=mode))
    universe.call_service("exa", "search")
    assert universe.ledger, f"{mode} produced no ledger entry"
    action = universe.ledger[-1].action

    risk, flags = CatalogDetector(REGISTRY).score(action, Context())
    assert risk >= min_risk, f"{mode} scored {risk} with flags {flags}"
    assert any(f.startswith(expected_flag) for f in flags), (mode, flags)


def test_honest_mode_produces_no_flags():
    universe = Universe(Knobs(merchant_mode="honest"))
    universe.call_service("exa", "search")
    risk, flags = CatalogDetector(REGISTRY).score(
        universe.ledger[-1].action, Context())
    assert risk == 0.0, flags


# ── Distributional evidence escalates; only the ceiling blocks ────────────

def test_amount_ratio_alone_does_not_block():
    """
    A ratio against a fitted median is distributional evidence, not proof. On a small
    clean sample an ordinary purchase from a dearer merchant looks extreme, and blocking
    on it alone rejected 21% of legitimate traffic in the live evaluation.
    """
    from acbguard.detectors import Pipeline
    from acbguard.detectors.behavioral import BehavioralDetector
    from acbguard.detectors.price import PriceDetector
    from acbguard.schema import Decision

    baseline = {"typical_amount_units": 90_000,      # fitted from cheap traffic
                "soft_limit_units": 520_000,
                "ceiling_units": 56_400_000}
    legit = act(service_id="booking", amount_units=1_200_000,   # the dearest catalog item
                endpoint="https://booking.com/api",
                payee=REGISTRY["booking"]["payee"])
    verdict = Pipeline([BehavioralDetector(), PriceDetector()]).score(
        legit, Context(baseline=baseline))
    assert verdict.decision is Decision.ESCALATE, (verdict.risk_score, verdict.flags)


def test_over_ceiling_still_blocks():
    """The hard stop must survive the change."""
    from acbguard.detectors import Pipeline
    from acbguard.detectors.price import PriceDetector
    from acbguard.schema import Decision

    baseline = {"typical_amount_units": 90_000, "soft_limit_units": 520_000,
                "ceiling_units": 5_000_000}
    verdict = Pipeline([PriceDetector()]).score(
        act(service_id="booking", amount_units=60_000_000), Context(baseline=baseline))
    assert verdict.decision is Decision.BLOCK


def test_counterparty_attacks_still_block():
    """Lowering the behavioral weight must not soften the counterparty detector."""
    from acbguard.detectors import Pipeline
    from acbguard.detectors.behavioral import BehavioralDetector
    from acbguard.schema import Decision

    pipeline = Pipeline([BehavioralDetector(), CatalogDetector(REGISTRY)])
    # Payee change is excluded deliberately: production evidence says it warns.
    for mode in ("typosquat_domain", "homoglyph_domain", "inflated_price"):
        u = Universe(Knobs(merchant_mode=mode))
        u.call_service("exa", "search")
        verdict = pipeline.score(u.ledger[-1].action, Context(baseline={}))
        assert verdict.decision is Decision.BLOCK, (mode, verdict.risk_score, verdict.flags)

    u = Universe(Knobs(merchant_mode="payee_swap"))
    u.call_service("exa", "search")
    verdict = pipeline.score(u.ledger[-1].action, Context(baseline={}))
    assert verdict.decision is Decision.ESCALATE, (verdict.risk_score, verdict.flags)
