"""
Fraud, as distinct from security.

Every merchant here is exactly who it claims to be: correct domain, correct settlement
address, correct protocol, real service delivered. They simply take more than they should.

That is the point. `CatalogDetector` and every other identity-keyed check misses all of them,
because nothing about the identity is wrong — and so does every security scanner. Catching
them requires comparing what was charged against a reference outside the transaction.
"""
import pytest

from acbguard.detectors import Context
from acbguard.detectors.catalog import CatalogDetector
from acbguard.detectors.economic import (
    DuplicateChargeDetector,
    EconomicDetector,
    PriceReference,
)
from acbguard.harness.merchants import GENUINE_BUT_HOSTILE, catalog
from acbguard.harness.universe import Knobs, Universe

BASE = catalog()
REGISTRY = {m.slug: {"domain": m.domain, "payee": m.pay_to, "price_units": m.price_units}
            for m in BASE.values()}
REFERENCE = PriceReference(
    catalogue={s: m.price_units for s, m in BASE.items()},
    population={s: [m.price_units] * 10 for s, m in BASE.items()},
)

# Tolerances have to be supplied. `EconomicDetector` has no hand-set defaults: an
# uncalibrated price check would be guessing, and the guesses it used to carry sat inside the
# margins the benchmark's own overcharge classes draw from. These stand in for what
# `benchmark.calibrate` fits from clean traffic — around 1.45x listed at a 10% review budget.
TOLERANCES = dict(catalogue_tolerance=1.45, peer_tolerance=1.40,
                  catalogue_block=2.05, peer_block=1.97, quote_tolerance=1.50)


def economic_detector():
    return EconomicDetector(REFERENCE, **TOLERANCES)


def run(mode, n=3, same_request=False):
    """Buy n times and return the worst score from each detector."""
    universe = Universe(Knobs(merchant_mode=mode))
    ctx = Context(baseline={})
    security = economic = 0.0
    flags: list[str] = []
    for i in range(n):
        universe.call_service("exa", "search",
                              payload={"nonce": "retry" if same_request else str(i)})
        if not universe.ledger:
            continue
        action = universe.ledger[-1].action
        security = max(security, CatalogDetector(REGISTRY).score(action, ctx)[0])
        econ, econ_flags = economic_detector().score(action, ctx)
        dup, dup_flags = DuplicateChargeDetector().score(action, ctx)
        if max(econ, dup) > economic:
            economic = max(econ, dup)
            flags = econ_flags or dup_flags
        ctx.history.append(action)
    return security, economic, flags


# ── The merchants are genuine ────────────────────────────────────────────

@pytest.mark.parametrize("mode", sorted(GENUINE_BUT_HOSTILE))
def test_hostile_merchants_are_who_they_claim_to_be(mode):
    """
    If any of these substituted a domain or a payee they would be security cases, and the
    existing identity checks would catch them. The whole class depends on them being genuine.
    """
    honest = BASE["exa"]
    merchant = GENUINE_BUT_HOSTILE[mode](honest)
    assert merchant.domain == honest.domain
    assert merchant.challenge("search").pay_to == honest.pay_to


@pytest.mark.parametrize("mode", ["overcharge", "price_discrimination", "phantom_fee",
                                  "silent_downgrade", "retry_farming"])
def test_security_detection_misses_economic_fraud(mode):
    """The central claim: identity-keyed detection cannot see a genuine seller overcharging."""
    security, _, _ = run(mode, same_request=(mode == "retry_farming"))
    assert security == 0.0, f"{mode} was caught by identity checks; it should not be"


# ── The economic layer catches them ──────────────────────────────────────

@pytest.mark.parametrize("mode,expected", [
    ("drip_pricing", "above_listed"),
])
def test_economic_detection_catches_large_overcharges(mode, expected):
    """
    Drip pricing quotes 1.0x and then charges 1.8x, which leaves the range legitimate prices
    move in — and, unlike the others, it can be checked against the seller's own quote.
    """
    _, economic, flags = run(mode)
    assert economic >= 0.5, (mode, flags)
    assert any(f.startswith(expected) for f in flags), (mode, flags)


@pytest.mark.parametrize("mode", ["overcharge", "price_discrimination", "phantom_fee"])
def test_small_overcharges_are_not_detectable_at_this_budget(mode):
    """
    These take 1.25x to 1.4x. Honest prices move that much — a surge, a dearer operation, a
    genuine repricing — so at a 10% review budget the tolerance sits at about 1.45x listed and
    none of them crosses it. Saying otherwise would require a tolerance no honest traffic
    could survive. A padded fee is in this group too: from the payment's point of view a fee
    that buys nothing and a price that is 25% high are the same event.

    This is the constraint the whole economic layer lives under, and it is asserted rather
    than described because the temptation is to tighten the tolerance until the test passes.
    That trade is always available and it is always paid for by legitimate buyers: the same
    cut that catches a 1.2x overcharge refuses roughly a fifth of ordinary purchases.

    Detection here comes from repetition and from peers, not from a single price.
    """
    _, economic, _ = run(mode, n=1)
    assert economic < 0.5, (
        f"{mode} at a small margin was caught by a single-price check; either the merchant's "
        f"margin grew or a tolerance was tightened past what clean traffic can bear")


def test_an_uncalibrated_price_check_stays_silent():
    """
    With no fitted tolerance the economic checks say nothing, rather than falling back to a
    number somebody picked.

    The defaults that used to fill this gap were 1.10 against the catalogue and 1.25 against
    peers, and they happened to sit inside the range the overcharge classes are generated
    from — so they scored well for a reason that had nothing to do with being right. Silence
    is auditable; a borrowed constant is not.
    """
    universe = Universe(Knobs(merchant_mode="drip_pricing"))
    universe.call_service("exa", "search", payload={"nonce": "0"})   # the low opening quote
    universe.call_service("exa", "search", payload={"nonce": "1"})
    action = universe.ledger[-1].action
    assert EconomicDetector(REFERENCE).score(action, Context(baseline={})) == (0.0, [])
    assert economic_detector().score(action, Context(baseline={}))[0] >= 0.5


def test_retry_farming_is_caught_as_a_duplicate_charge():
    """
    The merchant settles and reports failure. The buyer retries with a fresh key and pays
    twice for one purchase — the fraud is executed by the victim's own framework.
    """
    _, economic, flags = run("retry_farming", same_request=True)
    assert economic >= 0.5
    assert "duplicate_charge" in flags


def test_silent_downgrade_is_missed_by_both():
    """
    Recorded rather than fixed. The payment is perfect — right price, right payee — and only
    the delivered service is short. Nothing at the payment layer can see it, and claiming
    otherwise would be the overclaim this benchmark exists to avoid.
    """
    security, economic, _ = run("silent_downgrade")
    assert security == 0.0 and economic == 0.0


def test_honest_traffic_stays_clean():
    security, economic, flags = run("honest")
    assert security == 0.0 and economic == 0.0, flags


# ── Duplicate versus repeat ──────────────────────────────────────────────

def test_repeat_purchases_are_not_duplicates():
    """
    Three searches at the same price are identical in amount, service and timing to one
    search paid for three times. Only the request fingerprint separates them.
    """
    _, economic, flags = run("honest", n=3, same_request=False)
    assert economic == 0.0, flags


def test_same_request_paid_twice_is_a_duplicate():
    _, economic, flags = run("honest", n=3, same_request=True)
    assert economic >= 0.5 and "duplicate_charge" in flags


def test_economic_detector_is_silent_without_a_reference():
    """A price is only wrong relative to something. With no reference, no opinion."""
    universe = Universe(Knobs(merchant_mode="drip_pricing"))
    universe.call_service("exa", "search", payload={"nonce": "0"})   # the low opening quote
    universe.call_service("exa", "search")
    action = universe.ledger[-1].action
    assert EconomicDetector().score(action, Context()) == (0.0, [])


def test_population_reference_is_built_from_all_agents():
    """
    Pooled across agents on purpose: a per-agent view cannot see price discrimination,
    because the discriminated price is that agent's own normal.
    """
    sessions = [
        {"actions": [{"action_type": "authorize", "service_id": "exa",
                      "amount_units": 50_000}]} for _ in range(5)
    ]
    ref = PriceReference.from_sessions(sessions)
    assert ref.peer_median("exa") == 50_000
