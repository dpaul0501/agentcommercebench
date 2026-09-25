"""
Comparing detectors that do not share an operating point, and layers that do not share a job.

Two questions this answers, neither of which `evaluate_v2` answers honestly today.

**How do you compare detectors at different false-positive rates?**

You mostly cannot, and the usual move is to pretend otherwise. `evaluate_v2` reports
`lift = recall - clean_flag_rate`, which quietly assumes one false positive costs exactly what
one catch is worth. In payments that is wrong by two orders of magnitude in whichever direction
you happen to be going. Three honest answers instead:

  1. `iso_fpr`    — for a detector with a tunable threshold, refit it to a shared budget and
                    compare recall at matched FPR. This is the only apples-to-apples recall.
  2. `frontier`   — for a detector with ONE operating point (a rule, a probe set, garak,
                    promptfoo), there is nothing to tune. Plot it, and claim superiority only
                    where one point dominates another on both axes. Where neither dominates,
                    say they are incomparable, because they are.
  3. `expected_cost` — the number a customer actually cares about, given an explicit cost
                    model. This makes incomparable detectors comparable, but only relative to
                    a stated set of prices, and the prices are an input, not a finding.

**How do you compare across layers?**

Only inside a class both layers can observe. `config.JURISDICTION` records who can see what,
and an out-of-jurisdiction cell is `n/a`, not zero. A reasoning judge scoring 0.00 on a
substituted payee is not weak — the agent never saw the substitution. Averaging that in is how
a layer gets blamed for jurisdiction it does not have.

Where a layer does have jurisdiction, the operationally meaningful number is not its standalone
recall but its **marginal contribution**: what does adding it to everything else buy? A layer
that catches 80% of a class the other layer already catches is worth nothing, and its recall
will not say so.

    python -m benchmark.compare --data benchmark/data/v2
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "sdk"))

from benchmark.config import JURISDICTION  # noqa: E402
from benchmark.evaluate_v2 import (  # noqa: E402
    apply_limits, build_price_reference, fit_baselines, l0, l1, load, score,
)

ROOT = Path(__file__).resolve().parent.parent

_ATTACK_SELECTION = """
    Attack recall is measured on attack sessions that carry NO defect.

    16% of attack sessions also contain a genuine retry-paid-twice loss, because they are
    built on clean traffic and clean traffic fails 20% of the time. The detector catches the
    duplicate charge — correctly — and the session is then scored as "attack detected", which
    it is not: nothing about the attack was seen. The inflation reached +0.24 on a single
    class, and three classes whose true L1 recall is 0.00 were reporting 0.10-0.24.

    A session containing both an attack and an unrelated loss cannot attribute the detection,
    so it is excluded rather than guessed at. Those sessions are still scored in the LOSS
    bucket, where catching them is the win it actually is.
"""


# ── the cost model ───────────────────────────────────────────────────────────

class CostModel:
    """
    What a decision is worth, in dollars. Every value is an input, not a result.

    The point of stating it is that the "best" detector is not a property of the detector. It
    depends on these three numbers, and a reader who disagrees with them should be able to
    substitute their own and get a different answer from the same measurements.

    The defaults encode one fact that is measured and one judgement that is not:

    **Measured.** Production's median settled payment is $0.007 and its p99 is $0.25
    (`grounding.json: settled_amount_units`). These are fractions of a cent.

    **Judgement.** A human review costs $1-5 of analyst time anywhere in payments. Taking the
    low end, reviewing one flagged payment costs roughly **140x the median payment's value**.

    That ratio is not a detail, it is the binding constraint on this whole problem. Per-payment
    human review is economically impossible at agent-commerce ticket sizes: a 10% review rate
    on median traffic costs 14x the entire payment volume being protected. So `escalate` cannot
    mean "a person looks at this payment". It can only mean "a person looks at this AGENT",
    after enough evidence accumulates to be worth $1 — which is an argument for history-level
    detection over request-level review, and the reason the review budget has to be set by
    business disruption rather than by analyst capacity.
    """

    def __init__(self, review_usd: float = 1.00,
                 blocked_legitimate_usd: float = 0.50,
                 median_payment_usd: float = 0.007):
        self.review_usd = review_usd
        """Analyst time for one escalated item. Judgement; the low end of the usual range."""
        self.blocked_legitimate_usd = blocked_legitimate_usd
        """
        Cost of refusing an honest payment — not the payment, the broken workflow.

        A blocked payment fails an agent mid-task, and what that costs is the retry, the
        abandoned job, and the customer's trust, none of which is the ticket value. Judgement,
        and the number most worth arguing about.
        """
        self.median_payment_usd = median_payment_usd
        """MEASURED: grounding.json settled_amount_units p50 = 7,000 micro-units."""

    @property
    def review_to_payment_ratio(self) -> float:
        return self.review_usd / self.median_payment_usd

    def cost(self, n_clean: int, n_attack: int, clean_flag_rate: float,
             clean_block_rate: float, recall: float,
             loss_per_missed_usd: Optional[float] = None) -> dict[str, float]:
        """
        Expected cost of running this detector over this traffic mix.

        Missed fraud is charged at the payment's value by default, which is deliberately
        conservative: real fraud loss includes chargeback handling and remediation, so this
        understates it and therefore understates the case for detection.
        """
        loss = self.median_payment_usd if loss_per_missed_usd is None else loss_per_missed_usd
        reviews = n_clean * clean_flag_rate
        blocks = n_clean * clean_block_rate
        missed = n_attack * (1.0 - recall)
        return {
            "review_cost": round(reviews * self.review_usd, 2),
            "blocked_legitimate_cost": round(blocks * self.blocked_legitimate_usd, 2),
            "fraud_loss": round(missed * loss, 2),
            "total": round(reviews * self.review_usd
                           + blocks * self.blocked_legitimate_usd
                           + missed * loss, 2),
            "protected_volume": round((n_clean + n_attack) * self.median_payment_usd, 2),
        }


# ── operating points and dominance ───────────────────────────────────────────

class OperatingPoint:
    """One detector at one setting: where it sits in (false positive, recall)."""

    def __init__(self, name: str, clean_flag: float, clean_block: float,
                 recall: float, tunable: bool):
        self.name = name
        self.clean_flag = clean_flag
        self.clean_block = clean_block
        self.recall = recall
        self.tunable = tunable
        """False for a rule set or a fixed probe list: there is no threshold to move."""

    def dominates(self, other: "OperatingPoint") -> bool:
        """Better on BOTH axes. Anything less is a trade, not a win."""
        return (self.recall >= other.recall and self.clean_flag <= other.clean_flag
                and (self.recall > other.recall or self.clean_flag < other.clean_flag))

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "clean_flag": round(self.clean_flag, 4),
                "clean_block": round(self.clean_block, 4),
                "recall": round(self.recall, 4), "tunable": self.tunable}


def frontier(points: list[OperatingPoint]) -> dict[str, Any]:
    """
    Which detectors are on the Pareto frontier, and which pairs are simply incomparable.

    Reporting incomparability is the honest part. Two detectors at different false-positive
    rates are not ranked by their recalls, and a table that ranks them anyway has quietly
    chosen a cost model without saying so.
    """
    on_frontier = [p for p in points
                   if not any(q.dominates(p) for q in points if q is not p)]
    incomparable = []
    for i, a in enumerate(points):
        for b in points[i + 1:]:
            if not a.dominates(b) and not b.dominates(a) and a.recall != b.recall:
                incomparable.append((a.name, b.name))
    return {
        "frontier": [p.name for p in on_frontier],
        "dominated": [p.name for p in points if p not in on_frontier],
        "incomparable_pairs": incomparable,
    }


# ── cross-layer ──────────────────────────────────────────────────────────────

def jurisdictional(per_class: dict[str, dict[str, float]]) -> dict[str, Any]:
    """
    Per-class layer scores with out-of-jurisdiction cells marked `n/a` rather than 0.

    Also reports, for each layer, the recall it gets *within its own jurisdiction* — the only
    figure that describes the layer rather than the benchmark's class mix.
    """
    rows, in_juris = [], defaultdict(list)
    for code in sorted(per_class):
        allowed = JURISDICTION.get(code, frozenset({"L0", "L1"}))
        row: dict[str, Any] = {"class": code, "jurisdiction": sorted(allowed) or ["none"]}
        for layer in ("L0", "L1"):
            if layer in allowed:
                row[layer] = round(per_class[code].get(layer, 0.0), 3)
                in_juris[layer].append(per_class[code].get(layer, 0.0))
            else:
                row[layer] = None
        row["combined"] = round(per_class[code].get("combined", 0.0), 3)
        rows.append(row)
    return {
        "per_class": rows,
        "recall_within_jurisdiction": {
            layer: round(sum(v) / len(v), 3) for layer, v in in_juris.items() if v
        },
        "classes_no_layer_can_see": [c for c, a in JURISDICTION.items() if not a],
    }


def marginal(per_class: dict[str, dict[str, float]]) -> dict[str, Any]:
    """
    What each layer adds that the other did not already have.

    A layer's standalone recall overstates its worth whenever the other layer catches the same
    sessions. Marginal contribution is `combined - other_layer_alone`, and it is the number to
    use when deciding whether a layer is worth provisioning.
    """
    out = {}
    for layer, other in (("L0", "L1"), ("L1", "L0")):
        gains = []
        for code, scores in per_class.items():
            if layer not in JURISDICTION.get(code, frozenset({"L0", "L1"})):
                continue
            gains.append({"class": code,
                          "alone": round(scores.get(layer, 0.0), 3),
                          "other_alone": round(scores.get(other, 0.0), 3),
                          "marginal": round(scores.get("combined", 0.0)
                                            - scores.get(other, 0.0), 3)})
        gains.sort(key=lambda g: -g["marginal"])
        out[layer] = {
            "classes": gains,
            "mean_marginal": round(sum(g["marginal"] for g in gains) / len(gains), 3)
            if gains else 0.0,
        }
    return out


# ── driver ───────────────────────────────────────────────────────────────────

def measure(data_dir: Path, budgets=(0.01, 0.05, 0.10, 0.20)) -> dict[str, Any]:
    train = load(data_dir / "train.jsonl")
    test = load(data_dir / "test.jsonl")
    cal_path = data_dir / "calibration.json"
    base_cal = json.loads(cal_path.read_text()) if cal_path.exists() else {}
    reference = build_price_reference(train, data_dir)

    # See `_ATTACK_SELECTION` for why defect-carrying attack sessions are excluded.
    attacks = [r for r in test
               if not r.get("is_clean") and r.get("probe_id") and not r.get("defects")]
    cleans = [r for r in test if r.get("is_clean") and not r.get("defects")]

    def run(pipeline, baselines):
        per = defaultdict(list)
        for r in attacks:
            per[r["probe_id"]].append(score(r, pipeline, baselines.get(r["agent_id"], {}))[0])
        flag = sum(score(r, pipeline, baselines.get(r["agent_id"], {}))[0] for r in cleans)
        block = sum(score(r, pipeline, baselines.get(r["agent_id"], {}))[1] for r in cleans)
        recalls = {k: sum(v) / len(v) for k, v in per.items()}
        pooled = sum(sum(v) for v in per.values()) / max(1, len(attacks))
        return recalls, pooled, flag / max(1, len(cleans)), block / max(1, len(cleans))

    # ── 1. iso-FPR: the same pipeline refitted to each budget ──
    from benchmark.calibrate import calibrate

    iso = []
    for budget in budgets:
        cal = calibrate(data_dir, flag_budget=budget,
                        block_budget=min(budget / 10, 0.01))
        baselines = fit_baselines(train)
        apply_limits(baselines, cal)
        _, pooled, flag, block = run(l1(cal, reference), baselines)
        iso.append({"budget": budget, "achieved_clean_flag": round(flag, 4),
                    "achieved_clean_block": round(block, 4), "pooled_recall": round(pooled, 4)})

    # ── 2. operating points, tunable and not ──
    baselines = fit_baselines(train)
    apply_limits(baselines, base_cal)
    l1_recalls, l1_pooled, l1_flag, l1_block = run(l1(base_cal, reference), baselines)

    points = [OperatingPoint("L1 pipeline (calibrated)", l1_flag, l1_block, l1_pooled, True)]
    per_class: dict[str, dict[str, float]] = defaultdict(dict)
    for code, v in l1_recalls.items():
        per_class[code]["L1"] = v

    for kind, tunable in (("pattern", False), ("d8", False)):
        judge = l0(kind)
        recalls, pooled, flag, block = run(judge, baselines)
        points.append(OperatingPoint(f"L0 {kind}", flag, block, pooled, tunable))
        for code, v in recalls.items():
            per_class[code][f"L0:{kind}"] = v

    # combined, and the L0 column used for the cross-layer view
    judge = l0("pattern")
    for code in per_class:
        per_class[code]["L0"] = per_class[code].get("L0:pattern", 0.0)
    combined_per = defaultdict(list)
    for r in attacks:
        b = baselines.get(r["agent_id"], {})
        hit = score(r, l1(base_cal, reference), b)[0] or score(r, judge, b)[0]
        combined_per[r["probe_id"]].append(hit)
    for code, v in combined_per.items():
        per_class[code]["combined"] = sum(v) / len(v)
    combined_flag = sum(
        bool(score(r, l1(base_cal, reference), baselines.get(r["agent_id"], {}))[0]
             or score(r, judge, baselines.get(r["agent_id"], {}))[0]) for r in cleans
    ) / max(1, len(cleans))
    combined_pooled = sum(sum(v) for v in combined_per.values()) / max(1, len(attacks))
    points.append(OperatingPoint("L0+L1 combined", combined_flag, l1_block,
                                 combined_pooled, True))

    # ── 3. expected cost at each point ──
    model = CostModel()
    costs = [{"name": p.name,
              **model.cost(len(cleans), len(attacks), p.clean_flag, p.clean_block, p.recall)}
             for p in points]

    return {
        "n_clean": len(cleans), "n_attack": len(attacks),
        "iso_fpr": iso,
        "operating_points": [p.to_dict() for p in points],
        "frontier": frontier(points),
        "cost_model": {
            "review_usd": model.review_usd,
            "blocked_legitimate_usd": model.blocked_legitimate_usd,
            "median_payment_usd": model.median_payment_usd,
            "review_to_payment_ratio": round(model.review_to_payment_ratio, 1),
        },
        "expected_cost": costs,
        "cross_layer": jurisdictional(per_class),
        "marginal": marginal(per_class),
    }


def render(r: dict[str, Any]) -> str:
    out = ["", "  Comparing detectors and layers", "  " + "─" * 72, ""]

    out += ["  1. SAME BUDGET, different recall  (the only apples-to-apples recall)", ""]
    out.append(f"     {'budget':>8} {'achieved FPR':>13} {'blocked':>9} {'pooled recall':>14}")
    for row in r["iso_fpr"]:
        out.append(f"     {row['budget']:>8.0%} {row['achieved_clean_flag']:>13.3f} "
                   f"{row['achieved_clean_block']:>9.3f} {row['pooled_recall']:>14.3f}")
    out += ["",
            "     Recall is a function of the budget, not a property of the detector. Quoting",
            "     one number without its false-positive rate says nothing."]

    out += ["", "  2. OPERATING POINTS  (a fixed detector has only one; nothing to tune)", ""]
    out.append(f"     {'detector':<28} {'clean flag':>11} {'recall':>8} {'tunable':>8}")
    for p in r["operating_points"]:
        out.append(f"     {p['name']:<28} {p['clean_flag']:>11.3f} {p['recall']:>8.3f} "
                   f"{str(p['tunable']):>8}")
    f = r["frontier"]
    out += ["", f"     on the frontier : {', '.join(f['frontier'])}",
            f"     dominated       : {', '.join(f['dominated']) or 'none'}"]
    if f["incomparable_pairs"]:
        out.append("     incomparable    : neither dominates, so neither is 'better' —")
        for a, b in f["incomparable_pairs"][:4]:
            out.append(f"                       {a}  vs  {b}")

    cm = r["cost_model"]
    out += ["", "  3. EXPECTED COST  (given a stated cost model, which is an input)", "",
            f"     review ${cm['review_usd']:.2f}   blocked-legitimate "
            f"${cm['blocked_legitimate_usd']:.2f}   median payment "
            f"${cm['median_payment_usd']:.4f}",
            f"     one review costs {cm['review_to_payment_ratio']:.0f}x the median payment", ""]
    out.append(f"     {'detector':<28} {'review':>10} {'blocked':>9} {'fraud loss':>11} "
               f"{'total':>9}")
    for c in r["expected_cost"]:
        out.append(f"     {c['name']:<28} {c['review_cost']:>10.2f} "
                   f"{c['blocked_legitimate_cost']:>9.2f} {c['fraud_loss']:>11.2f} "
                   f"{c['total']:>9.2f}")
    out += ["", f"     total payment volume protected: "
                f"${r['expected_cost'][0]['protected_volume']:.2f}",
            "     Review dominates every other term. At these ticket sizes per-payment human",
            "     review cannot pay for itself, whatever the detector's recall — so `escalate`",
            "     has to mean 'look at this AGENT once it is worth $1', not 'look at this",
            "     payment'. That is an argument for history-level detection, not a tuning "
            "choice."]

    cl = r["cross_layer"]
    out += ["", "  4. ACROSS LAYERS  (n/a = this layer cannot observe the class at all)", ""]
    out.append(f"     {'class':<7} {'jurisdiction':<12} {'L0':>7} {'L1':>7} {'combined':>9}")
    for row in cl["per_class"]:
        l0v = "  n/a" if row["L0"] is None else f"{row['L0']:.2f}"
        l1v = "  n/a" if row["L1"] is None else f"{row['L1']:.2f}"
        out.append(f"     {row['class']:<7} {'+'.join(row['jurisdiction']):<12} "
                   f"{l0v:>7} {l1v:>7} {row['combined']:>9.2f}")
    out += ["", "     recall WITHIN jurisdiction: "
            + "   ".join(f"{k} {v:.3f}" for k, v in
                         sorted(cl["recall_within_jurisdiction"].items()))]
    if cl["classes_no_layer_can_see"]:
        out.append(f"     no layer can see: {', '.join(cl['classes_no_layer_can_see'])} "
                   f"(control — should sit at the clean flag rate)")

    out += ["", "  5. MARGINAL CONTRIBUTION  (what a layer adds that the other lacked)", ""]
    for layer, m in sorted(r["marginal"].items()):
        out.append(f"     {layer}: mean marginal {m['mean_marginal']:+.3f}")
        for g in m["classes"][:3]:
            out.append(f"        {g['class']}  alone {g['alone']:.2f}  "
                       f"other alone {g['other_alone']:.2f}  marginal {g['marginal']:+.2f}")
    out += ["", "     A layer catching what the other already caught is worth nothing, and its",
            "     standalone recall will not say so.", ""]
    return "\n".join(out)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="benchmark/data/v2")
    p.add_argument("--json", action="store_true")
    args = p.parse_args(argv)

    report = measure(Path(args.data))
    print(json.dumps(report, indent=2, default=str) if args.json else render(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
