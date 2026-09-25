"""
Distributional audit: is each attack class actually hard to separate from clean traffic?

The principle
-------------
A benchmark measures detection only when the attack and clean populations *overlap* in every
feature the attack is not about. If they are disjoint, any threshold placed in the gap scores
perfectly and never false-alarms — and it would do so even if the detector were broken. The
number then describes the generator, not the detector.

So for each attack class and each nuisance feature this reports:

    overlap        share of the attack distribution inside the clean support
    OVR            overlapping coefficient — area shared by the two densities, 0..1
    AUC            how well this single feature alone separates attack from clean

**AUC is the headline.** A nuisance feature with AUC near 1.0 is a giveaway: a detector can
score perfectly on that class without modelling the attack at all. A feature the attack is not
about should sit near 0.5.

A threshold is reported as `IN A GAP` when no clean sample and no attack sample fall on the
same side of it — the condition under which a reported false-positive rate carries no
information.

    python -m benchmark.distribution_audit
    python -m benchmark.distribution_audit --strict     # exit 1 if any nuisance AUC > 0.75
"""
from __future__ import annotations

import argparse
import collections
import statistics
import sys
from pathlib import Path
from typing import Any, Callable, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "sdk"))

# `benchmark.synthetic` is the retired v1 generator and reaches into the proprietary
# pipeline, so importing it at module scope made this script unrunnable from a clean
# checkout — the second command in the reproducibility section died on ModuleNotFoundError.
# The --v2 path reads sessions from a file and needs none of it, so the import is deferred
# to the one branch that does.

# Which features each attack is legitimately about. Read from `benchmark.config.ATTACKS`
# rather than duplicated here: the duplicate drifted the moment new classes were added, and
# the audit then reported a class's own defining feature as a giveaway.
#
# Anything not listed for a class is a nuisance feature for it, and must not separate it.

def _intended_map() -> dict[str, set]:
    try:
        from benchmark.config import ATTACKS
    except Exception:
        return {}
    return {a.code: set(a.intended_features) for a in ATTACKS}


# Feature names in this file vs the vocabulary the specs use.
FEATURE_ALIASES = {
    "amount_max": {"amount"},
    "amount_mean": {"amount"},
    "n_commits": {"n_actions", "n_commits"},
    "n_actions": {"n_actions", "n_commits"},
}

INTENDED = {
    **_intended_map(),
    # legacy classes, kept so the old generator can still be audited for comparison
    "B1": {"first_action"}, "B2": {"n_actions"}, "B3": {"mcc"},
    "B4": set(), "B5": set(), "B6": {"payload"}, "B7": {"mcc"},
    "B7s": {"mcc"}, "B8": {"amount"}, "B8s": {"n_actions", "amount"},
}


def amounts(session) -> list[float]:
    return [float(a.amount) for a in session.actions if getattr(a, "amount", None)]


def n_commits(session) -> int:
    """Payment actions. Accepts the old COMMIT naming and the v2 `authorize`."""
    return sum(1 for a in session.actions
               if "COMMIT" in str(getattr(a, "action_type", ""))
               or "authorize" in str(getattr(a, "action_type", "")).lower())


def first_action(session) -> str:
    return str(getattr(session.actions[0], "action_type", "")) if session.actions else ""


FEATURES: dict[str, Callable[[Any], Optional[float]]] = {
    "amount_max": lambda s: max(amounts(s)) if amounts(s) else None,
    "amount_mean": lambda s: statistics.fmean(amounts(s)) if amounts(s) else None,
    "n_commits": lambda s: float(n_commits(s)),
    "n_actions": lambda s: float(len(s.actions)),
}
# Which INTENDED key each feature belongs to, so we know when separation is legitimate.
FEATURE_TOPIC = {"amount_max": "amount", "amount_mean": "amount",
                 "n_commits": "n_commits", "n_actions": "n_commits"}


def auc(pos: list[float], neg: list[float]) -> float:
    """
    Probability a random attack scores above a random clean session, on this feature alone.

    0.5 means the feature is uninformative — which is what a nuisance feature should be.
    Computed by rank sum, ties counted as half.
    """
    if not pos or not neg:
        return float("nan")
    merged = sorted([(v, 1) for v in pos] + [(v, 0) for v in neg])
    ranks: dict[int, float] = {}
    i = 0
    while i < len(merged):
        j = i
        while j + 1 < len(merged) and merged[j + 1][0] == merged[i][0]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1
    rank_sum = sum(ranks[k] for k, (_, lab) in enumerate(merged) if lab == 1)
    n_pos, n_neg = len(pos), len(neg)
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def overlapping_coefficient(a: list[float], b: list[float], bins: int = 30) -> float:
    """Shared area under two histograms, 0 (disjoint) to 1 (identical)."""
    if not a or not b:
        return float("nan")
    lo, hi = min(a + b), max(a + b)
    if hi == lo:
        return 1.0
    width = (hi - lo) / bins

    def hist(xs):
        counts = collections.Counter(
            min(int((x - lo) / width), bins - 1) for x in xs)
        return {k: v / len(xs) for k, v in counts.items()}

    ha, hb = hist(a), hist(b)
    return sum(min(ha.get(k, 0.0), hb.get(k, 0.0)) for k in set(ha) | set(hb))


def in_support(values: list[float], reference: list[float]) -> float:
    """Share of `values` lying inside the min-max range of `reference`."""
    if not values or not reference:
        return float("nan")
    lo, hi = min(reference), max(reference)
    return sum(1 for v in values if lo <= v <= hi) / len(values)


class _Rec:
    """Adapts a v2 JSONL record to the shape this audit expects."""

    __slots__ = ("actions", "attack_type")

    def __init__(self, row):
        self.attack_type = row.get("probe_id") if not row.get("is_clean") else None
        self.actions = [_Act(a) for a in row["actions"]]


class _Act:
    __slots__ = ("amount", "action_type")

    def __init__(self, a):
        self.amount = a.get("amount_units")
        self.action_type = a.get("action_type", "")


def load_v2(path: str) -> list:
    import json
    with open(path) as fh:
        return [_Rec(json.loads(line)) for line in fh if line.strip()]


def audit(n_clean: int = 400, n_per_attack: int = 60, seed: int = 42,
          v2_path: Optional[str] = None) -> dict[str, Any]:
    if not v2_path:
        raise SystemExit(
            "--v2 <test.jsonl> is required. The v1 generator this once fell back to is "
            "retired and is not part of this distribution."
        )
    data = load_v2(v2_path)
    by_class: dict[str, list] = collections.defaultdict(list)
    for s in data:
        by_class[getattr(s, "attack_type", None) or "clean"].append(s)
    clean = by_class.pop("clean", [])

    report: dict[str, Any] = {"clean_n": len(clean), "classes": {}}
    for cls, sessions in sorted(by_class.items()):
        intended = INTENDED.get(cls, set())
        entry: dict[str, Any] = {"n": len(sessions), "features": {}}
        for fname, fn in FEATURES.items():
            pos = [v for v in (fn(s) for s in sessions) if v is not None]
            neg = [v for v in (fn(s) for s in clean) if v is not None]
            if not pos or not neg:
                continue
            # A feature is a nuisance unless the spec says this attack is about it.
            is_nuisance = not (FEATURE_ALIASES.get(fname, {fname}) & intended)
            entry["features"][fname] = {
                "auc": round(auc(pos, neg), 3),
                "ovr": round(overlapping_coefficient(pos, neg), 3),
                "in_clean_support": round(in_support(pos, neg), 3),
                "nuisance": is_nuisance,
                "attack_range": [round(min(pos), 1), round(max(pos), 1)],
                "clean_range": [round(min(neg), 1), round(max(neg), 1)],
            }
        report["classes"][cls] = entry
    return report


def worst_nuisance(report: dict[str, Any]) -> list[tuple[str, str, float]]:
    """Nuisance features that separate the classes anyway, worst first."""
    out = []
    for cls, entry in report["classes"].items():
        for fname, f in entry["features"].items():
            if f["nuisance"]:
                distance = abs(f["auc"] - 0.5)
                out.append((cls, fname, f["auc"], distance))
    return [(c, f, a) for c, f, a, _ in sorted(out, key=lambda r: -r[3])]


def render(report: dict[str, Any], threshold: float = 0.75) -> str:
    out = ["", f"  Distributional audit — clean n={report['clean_n']}", "  " + "─" * 74,
           "  AUC on a NUISANCE feature should be ~0.50. Higher means the class is separable",
           "  without modelling the attack at all.", ""]
    out.append(f"    {'class':<6} {'feature':<13} {'AUC':>6} {'OVR':>6} "
               f"{'in-clean':>9}  {'':<4} range")
    for cls, entry in report["classes"].items():
        for fname, f in entry["features"].items():
            tag = "    " if not f["nuisance"] else ("LEAK" if abs(f["auc"] - 0.5) > 0.25
                                                    else "    ")
            out.append(
                f"    {cls:<6} {fname:<13} {f['auc']:>6.3f} {f['ovr']:>6.3f} "
                f"{f['in_clean_support']:>9.3f}  {tag:<4} "
                f"attack {f['attack_range']} vs clean {f['clean_range']}")
        out.append("")

    # Even a feature the attack IS about must overlap clean traffic. If it does not, the
    # detection task is trivial: any threshold in the empty gap scores perfectly and can
    # never false-alarm, so the result describes the generator rather than the detector.
    trivial = []
    for cls, entry in report["classes"].items():
        for fname, f in entry["features"].items():
            if not f["nuisance"] and f["ovr"] < 0.10:
                trivial.append((cls, fname, f["ovr"], f["in_clean_support"]))

    out += ["  " + "─" * 74]
    if trivial:
        out.append(f"  {len(trivial)} intended feature(s) are DISJOINT from clean traffic — "
                   f"the task is trivial, not hard:")
        for cls, fname, ovr, ins in trivial:
            out.append(f"    {cls:<6} {fname:<13} overlap {ovr:.3f}, "
                       f"{100 * ins:.0f}% of attacks inside the clean range")
        out.append("")

    leaks = [(c, f, a) for c, f, a in worst_nuisance(report) if abs(a - 0.5) > 0.25]
    if leaks:
        out.append(f"  {len(leaks)} nuisance feature(s) separate a class they should not:")
        for cls, fname, a in leaks:
            out.append(f"    {cls:<6} {fname:<13} AUC {a:.3f}")
    else:
        out.append("  No nuisance feature separates any class. ")
    out.append("")
    return "\n".join(out)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--clean", type=int, default=400)
    p.add_argument("--per-attack", type=int, default=60)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--strict", action="store_true",
                   help="exit 1 if any nuisance feature separates a class")
    p.add_argument("--v2", required=True, help="audit a generated v2 split instead of the old generator")
    args = p.parse_args(argv)

    report = audit(args.clean, args.per_attack, args.seed, v2_path=args.v2)
    print(render(report))
    if args.strict:
        leaks = [x for x in worst_nuisance(report) if abs(x[2] - 0.5) > 0.25]
        trivial = [1 for e in report["classes"].values() for f in e["features"].values()
                   if not f["nuisance"] and f["ovr"] < 0.10]
        return 1 if (leaks or trivial) else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
