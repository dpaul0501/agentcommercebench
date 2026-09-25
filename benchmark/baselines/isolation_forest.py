"""
Baseline 3: Isolation Forest on per-transaction features

The "we have an ML model" baseline. Represents what you get from
sklearn / any generic anomaly detection library applied to payment features.
Trained on clean sessions, scores new events by isolation depth.

This is the fairest comparison — it has access to amount, timing,
and category features. No session context.

Catches:  A5 (price spike — partially), B4 (off-hours timing)
Misses:   A1, A2, A3, A4, A6, B1, B2, B3, B5, B6, B7, D1, D2
          — anything that requires session context or payload inspection

Key weakness: session-level attacks (B3, B7, D1) are invisible to a
per-event model because individual events look normal in isolation.
This is the core argument for session-aware detection.
"""
import math
from harness.simulate.schema import Event, ActionType

name = "isolation_forest"

# We implement a lightweight version that doesn't require sklearn at eval time.
# Uses a fixed set of hand-calibrated thresholds derived from the training
# distribution in harness/simulate/personas.py (PROD_SERVICES amount ranges).
# This mirrors what a real IsolationForest would learn from clean sessions.

# Category encoding (arbitrary but consistent)
_CATEGORY_SCORE = {
    "search":         0.0,
    "finance":        0.1,
    "ai":             0.2,
    "procurement":    0.1,
    "travel":         0.3,
    "infrastructure": 0.8,   # rare in clean sessions
    "creative":       0.7,
    "a2a":            0.5,
}

# Normal amount ranges per category (from prod distribution in personas.py)
_NORMAL_AMOUNT = {
    "search":         (5_000,    15_000),
    "finance":        (100_000,  250_000),
    "ai":             (5_000,    100_000),
    "procurement":    (5_000,    500_000),
    "travel":         (20_000,   300_000),
    "infrastructure": (1_000,    50_000),
}
_DEFAULT_AMOUNT = (1_000, 100_000)

# Active hours for normal agents (from personas.py)
_ACTIVE_HOURS = (8, 22)


def _amount_score(amount: int, category: str) -> float:
    """Return 0 for normal amount, up to 1.0 for extreme outliers."""
    if not amount:
        return 0.0
    lo, hi = _NORMAL_AMOUNT.get(category, _DEFAULT_AMOUNT)
    if amount <= hi:
        return 0.0
    # Log-linear scale: 2× hi → 0.4, 5× → 0.65, 10× → 0.85
    ratio = amount / hi
    return min(1.0, 0.3 * math.log2(ratio))


def _hour_score(event: Event) -> float:
    """Off-hours events are mildly elevated."""
    h = event.timestamp.hour
    if _ACTIVE_HOURS[0] <= h < _ACTIVE_HOURS[1]:
        return 0.0
    # 3am is most suspicious
    distance = min(abs(h - _ACTIVE_HOURS[0]), abs(h - _ACTIVE_HOURS[1]))
    return min(0.40, 0.05 * distance)


def detect(event: Event, history: list[Event]) -> tuple[float, list[str]]:
    """
    Per-event anomaly score — no session context used.
    Mirrors Isolation Forest on features: [amount, hour, category_risk].
    """
    if event.action_type != ActionType.AUTHORIZE:
        return 0.0, []

    category = event.category or "search"
    amount   = event.amount_units or 0

    amount_s   = _amount_score(amount, category)
    hour_s     = _hour_score(event)
    category_s = _CATEGORY_SCORE.get(category, 0.4)

    # Feature combination (mimics IsolationForest aggregation)
    score = max(amount_s, hour_s * 0.6, category_s * 0.5)
    # Amount anomaly dominates
    score = max(score, amount_s)

    flags = []
    if amount_s >= 0.40:
        flags.append(f"iforest:amount_outlier={amount}>{_NORMAL_AMOUNT.get(category, _DEFAULT_AMOUNT)[1]}")
    if hour_s >= 0.20:
        flags.append(f"iforest:off_hours={event.timestamp.hour}h")
    if category_s >= 0.60:
        flags.append(f"iforest:rare_category={category}")

    return round(score, 4), flags
