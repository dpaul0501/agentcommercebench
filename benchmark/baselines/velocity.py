"""
Baseline 1: Velocity check

The simplest possible production defence — rate limiting on authorize calls.
Used by most payment systems as a first-pass filter.

Catches: B6 (velocity spike), partially B7 (policy probe)
Misses:  A1, A2, A3, A4, A5, A6, B1, B2, B3, B4, B5, D1, D2
         — anything that doesn't cause unusually high call frequency

This is the "we already have this in prod" baseline.
The benchmark shows it is necessary but nowhere near sufficient.
"""
from datetime import timedelta
from harness.simulate.schema import Event, ActionType

name = "velocity_check"

# Tune these to be "reasonable production defaults"
WINDOW_SECONDS   = 60
MAX_AUTH_PER_MIN = 5
BLOCK_THRESHOLD  = 10


def detect(event: Event, history: list[Event]) -> tuple[float, list[str]]:
    """
    Counts authorize calls in the last WINDOW_SECONDS.
    Returns score proportional to how far over the limit we are.
    """
    if event.action_type != ActionType.AUTHORIZE:
        return 0.0, []

    cutoff = event.timestamp - timedelta(seconds=WINDOW_SECONDS)
    recent_auths = sum(
        1 for e in history
        if e.action_type == ActionType.AUTHORIZE and e.timestamp >= cutoff
    )

    if recent_auths >= BLOCK_THRESHOLD:
        score = min(1.0, 0.70 + 0.03 * (recent_auths - BLOCK_THRESHOLD))
        return score, [f"velocity:auth_count={recent_auths}_per_{WINDOW_SECONDS}s"]
    if recent_auths >= MAX_AUTH_PER_MIN:
        score = 0.30 + 0.04 * (recent_auths - MAX_AUTH_PER_MIN)
        return min(score, 0.69), [f"velocity:elevated_auth={recent_auths}"]

    return 0.0, []
