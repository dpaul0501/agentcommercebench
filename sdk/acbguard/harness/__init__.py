"""
Local harness — an offline replica of the Platform stack.

    from acbguard.harness import Universe, Knobs

    u = Universe(Knobs(merchant_mode="homoglyph_domain"))
    u.call_service("exa", "search")

Deterministic and offline by default. The only path that reaches the network is an
`AdaptiveMerchant` given a model, which is opt-in.
"""
from .merchants import (
    AdaptiveMerchant,
    HomoglyphMerchant,
    InflatedPriceMerchant,
    InjectingMerchant,
    Merchant,
    PayeeSwapMerchant,
    PaymentChallenge,
    TyposquatMerchant,
    catalog,
    malicious_variants,
)
from .universe import Knobs, LedgerEntry, Universe, Wallet

MERCHANT_MODES = [
    "honest",
    "inflated_price",
    "homoglyph_domain",
    "typosquat_domain",
    "payee_swap",
    "response_injection",
    "adaptive",
]

__all__ = [
    "AdaptiveMerchant",
    "HomoglyphMerchant",
    "InflatedPriceMerchant",
    "InjectingMerchant",
    "Knobs",
    "LedgerEntry",
    "MERCHANT_MODES",
    "Merchant",
    "PayeeSwapMerchant",
    "PaymentChallenge",
    "TyposquatMerchant",
    "Universe",
    "Wallet",
    "catalog",
    "malicious_variants",
]
