"""
How good is the benchmark? — tests of the benchmark itself, not of any detector.

A benchmark can be non-circular and still be worthless. It can be perfectly fair and measure
nothing, because every class saturates. It can rank two detectors confidently on a difference
that is inside its own noise. It can be beautifully calibrated to traffic that does not exist.

`distribution_audit` and `provenance` already answer "is it rigged". These answer a different
question: **is it measuring anything, and is what it measures real?**

Six checks, each of which can fail:

    Q1  fidelity        do generated distributions match production aggregates?
    Q2  field realism   does it populate fields production does not, or vice versa?
    Q3  saturation      how many classes are pinned at 0 or 1, where they discriminate nothing?
    Q4  discrimination  can it rank detectors of known, deliberately different strength?
    Q5  power           is n large enough to support the differences being reported?
    Q6  label integrity are clean sessions clean, and does each class move its own feature?

Q4 is the one that matters most and is the one benchmarks usually skip. A benchmark is an
instrument for ranking detectors. If a deliberately crippled detector scores the same as the
real one, the instrument is broken, and no amount of rigour about circularity fixes that.

    python -m benchmark.quality --data benchmark/data/v2
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "sdk"))

from benchmark.evaluate_v2 import (  # noqa: E402
    apply_limits, build_price_reference, fit_baselines, l0, l1, load, score, to_actions,
)

ROOT = Path(__file__).resolve().parent.parent


# ── Q1. Fidelity to production ───────────────────────────────────────────────

def production_aggregates(path: Path = ROOT / "grounding.json") -> Optional[dict[str, Any]]:
    """
    Aggregates measured from the production database. Aggregates only — never rows.

    Absent on a machine with no access, in which case the fidelity check reports that it
    could not run rather than silently passing.
    """
    if not path.exists():
        return None
    return json.loads(path.read_text())


def quantiles(values: list[float], qs=(0.05, 0.25, 0.5, 0.75, 0.95, 0.99)) -> dict[str, float]:
    if not values:
        return {}
    s = sorted(values)
    return {f"p{int(q * 100):02d}": s[min(len(s) - 1, int(q * len(s)))] for q in qs}


def payment_requirement_quantiles(path: Path = ROOT / "survey.json") -> Optional[dict]:
    """Production's *price* distribution: 1,644 payment requirements over 295 services."""
    if not path.exists():
        return None
    survey = json.loads(path.read_text())
    for s in survey.get("surveys", []):
        prof = s.get("profiles", {}).get("service_payment_requirements", {})
        got = prof.get("amount", {}).get("amount_units")
        if got:
            return {"quantiles": got, "n": prof.get("rows")}
    return None


def fidelity(rows: list[dict[str, Any]], prod: Optional[dict[str, Any]],
             data_dir: Optional[Path] = None) -> dict[str, Any]:
    """
    Q1 — does the benchmark look like production?

    Against the **catalogue**, not against settled traffic, and the distinction decides
    whether this check measures realism or memorisation.

    Production settled 1,068 payments from 6 agents, and **81.6% of them came from a single
    agent**, with a second contributing 8.8% at a median 20x higher. That distribution is one
    customer's product mix. A benchmark tuned to reproduce it would be reproducing that
    customer, and would drift the moment a seventh agent appeared.

    `service_payment_requirements` is the honest target: 1,644 rows across 295 services, a
    price distribution rather than one buyer's usage of it. The benchmark's catalogue should
    match that. What agents then *spend* follows from which services they use, and is allowed
    to differ — it is a consequence, not a parameter.

    The settled comparison is still reported, because a large divergence there would mean the
    agent model is wrong. It is just not a gate.
    """
    if not prod:
        return {"ran": False, "reason": "no grounding.json — production aggregates unavailable"}

    def compare(got: dict[str, float], want: dict[str, float]) -> tuple[list, float]:
        out, worst = [], 0.0
        for key in ("p05", "p25", "p50", "p75", "p95", "p99"):
            g, w = got.get(key), want.get(key)
            if not g or not w:
                continue
            worst = max(worst, abs(math.log(g / w)))
            out.append({"quantile": key, "benchmark": g, "production": w,
                        "ratio": round(g / w, 2)})
        return out, worst

    result: dict[str, Any] = {"ran": True}

    # ── the gate: catalogue prices vs production's payment requirements ──
    cat_path = (data_dir or ROOT / "benchmark/data/v2") / "catalogue.json"
    prod_prices = payment_requirement_quantiles()
    if cat_path.exists() and prod_prices:
        prices = [float(v["price_units"]) for v in json.loads(cat_path.read_text()).values()]
        rows_out, worst = compare(quantiles(prices), prod_prices["quantiles"])
        result["catalogue"] = {
            "quantiles": rows_out,
            "benchmark_n": len(prices),
            "production_n": prod_prices["n"],
            "worst_fold": round(math.exp(worst), 2),
        }
        result["pass"] = worst < 0.7          # within 2x at every quantile
    else:
        result["pass"] = True
        result["catalogue"] = {"skipped": "no catalogue.json or no production price data"}

    # ── reported, not gated: settled amounts ──
    amounts = [
        float(a["amount_units"]) for r in rows
        if r.get("is_clean") and not r.get("defects")
        for a in r["actions"]
        if a["action_type"] == "authorize" and a.get("amount_units")
    ]
    rows_out, worst = compare(quantiles(amounts), prod.get("settled_amount_units", {}))
    per_agent = prod.get("per_agent", [])
    total = sum(a["n"] for a in per_agent) or 1
    result["settled"] = {
        "quantiles": rows_out,
        "worst_fold": round(math.exp(worst), 2),
        "gated": False,
        "production_agents": len(per_agent),
        "top_agent_share": round(max((a["n"] for a in per_agent), default=0) / total, 3),
    }
    return result


def session_shape(rows: list[dict[str, Any]], prod_path: Path = ROOT / "survey.json") -> dict:
    """
    Q1b — session length.

    Read the production side with care, and do not treat it as a distribution. Production has
    a `sessions` table with **0 rows** and a `session_id` on **5 of 1,068** settlements across
    3 distinct sessions. The "p90 = 745 actions" that falls out of that is five data points,
    one of which is a 1,063-action batch job. It says nothing about what a session looks like.

    What it does say is that sessions are **not being recorded** — an instrumentation gap, not
    a fact about how agents behave. The schema supports sessions; nothing writes them. So the
    benchmark's session shape cannot be validated against production at all, and the honest
    response is for detectors not to depend on sessions (see Q7) rather than to copy a shape
    measured from five rows.
    """
    lengths = [len(r["actions"]) for r in rows if r.get("is_clean")]
    got = quantiles([float(x) for x in lengths], qs=(0.5, 0.9, 1.0))
    out: dict[str, Any] = {"benchmark": got}
    if prod_path.exists():
        survey = json.loads(prod_path.read_text())
        for s in survey.get("surveys", []):
            prof = s.get("profiles", {}).get("service_settlements", {})
            if prof.get("actions_per_session"):
                out["production"] = prof["actions_per_session"]
                out["production_n_sessions"] = prof.get("n_sessions")
                out["production_session_id_null_rate"] = prof.get("null_rate", {}).get(
                    "session_id")
    return out


# ── Q2. Field realism ────────────────────────────────────────────────────────

def field_realism(rows: list[dict[str, Any]],
                  prod: Optional[dict[str, Any]]) -> dict[str, Any]:
    """
    Q2 — is the benchmark handing detectors evidence production does not have?

    This is the failure that does not show up until deployment. A detector fitted on a field
    the benchmark populates 100% of the time and production populates never will score well
    here and do nothing there. It is not circularity — the benchmark is internally honest —
    it is a realism failure, and it is silent.
    """
    if not prod or "settlement_field_population" not in prod:
        return {"ran": False, "reason": "no field-population data in grounding.json"}

    prod_pop = prod["settlement_field_population"]
    # Benchmark field names -> the production column carrying the same information.
    mapping = {
        "endpoint": "raw_endpoint",
        "vendor": "raw_vendor",
        "payee": "pay_to",
        "idempotency_key": "idempotency_key",
        "session_id": "session_id",
    }

    payments = [a for r in rows if r.get("is_clean")
                for a in r["actions"] if a["action_type"] == "authorize"]
    n = len(payments) or 1

    findings = []
    for field, prod_col in mapping.items():
        if prod_col not in prod_pop:
            continue
        if field == "session_id":
            bench_rate = 1.0          # every generated action belongs to a session
        else:
            bench_rate = sum(1 for a in payments if a.get(field)) / n
        gap = bench_rate - prod_pop[prod_col]
        findings.append({
            "field": field, "production_column": prod_col,
            "benchmark": round(bench_rate, 4), "production": round(prod_pop[prod_col], 4),
            "gap": round(gap, 4),
            # Available here and rarely there. The dangerous direction.
            "over_available": gap > 0.25,
        })
    return {"ran": True, "fields": findings,
            # Reported, not gated. A field the benchmark has and production lacks only
            # matters if a detector READS it, and Q7 measures that directly by scoring the
            # same detectors both ways. Gating here instead would fail forever on a field
            # nothing depends on.
            "pass": True,
            "over_available_fields": [f["field"] for f in findings if f["over_available"]],
            "consequence": (
                "session_id is present on every generated action and absent from 99.5% of "
                "production settlements. The benchmark groups history by session; production "
                "cannot. Any history-level detector must key on agent_id (100% populated) "
                "and treat a session boundary as unavailable, or it will work here and not "
                "there.")}


# ── Q7. Deployment parity ────────────────────────────────────────────────────

def deployment_parity(data_dir: Path, calibration: dict[str, Any]) -> dict[str, Any]:
    """
    Q7 — does the detector score the same when history is keyed the way production keys it?

    The benchmark groups actions into sessions because it created them. Production does not:
    `session_id` is populated on 0.47% of settlements, `agent_id` on 100%. So a detector is
    scored twice — once with history reset at each session boundary, once with history keyed
    on the agent and carried across boundaries in timestamp order — and the two must agree.

    This is not hypothetical. It caught `RegistryDetector` comparing `action.agent_id` against
    `ctx.session.agent_id`: the identity-mismatch class scored 1.00 with sessions and 0.30
    without, so the harness was reporting a detector that would have lost 70% of its recall on
    deployment, silently, with no error anywhere. The fix was to compare against the
    authenticated principal, which production has and a session is merely a proxy for.

    Any gap here is a detector reading evidence deployment will not give it.
    """
    from acbguard.guard.store import InMemoryContextStore

    train = load(data_dir / "train.jsonl")
    test = load(data_dir / "test.jsonl")
    baselines = fit_baselines(train)
    apply_limits(baselines, calibration)
    reference = build_price_reference(train, data_dir)

    def measure(store):
        pipeline = l1(calibration, reference)
        rows = sorted(test, key=lambda r: r["actions"][0].get("timestamp") or "")
        per: dict[str, list[bool]] = defaultdict(list)
        clean: list[bool] = []
        for r in rows:
            flagged = score(r, pipeline, baselines.get(r["agent_id"], {}), store=store)[0]
            if r.get("is_clean") and not r.get("defects"):
                clean.append(flagged)
            elif r.get("probe_id"):
                per[r["probe_id"]].append(flagged)
        return ({k: sum(v) / len(v) for k, v in per.items()},
                sum(clean) / max(1, len(clean)))

    sess, sess_clean = measure(None)
    agent, agent_clean = measure(InMemoryContextStore())

    deltas = [{"class": k, "session_keyed": round(sess[k], 3),
               "agent_keyed": round(agent.get(k, 0.0), 3),
               "delta": round(agent.get(k, 0.0) - sess[k], 3)}
              for k in sorted(sess)]
    worst = max((abs(d["delta"]) for d in deltas), default=0.0)
    return {
        "per_class": deltas,
        "clean": {"session_keyed": round(sess_clean, 3),
                  "agent_keyed": round(agent_clean, 3),
                  "delta": round(agent_clean - sess_clean, 3)},
        "worst_delta": round(worst, 3),
        "regressed": [d["class"] for d in deltas if d["delta"] <= -0.05],
        # Small movement is sampling; a class losing 5 points is reading a session.
        "pass": worst < 0.05,
    }


# ── Q3. Saturation ───────────────────────────────────────────────────────────

def saturation(per_class: dict[str, float], clean_rate: float) -> dict[str, Any]:
    """
    Q3 — how much of the benchmark is still measuring something?

    A class detected 1.00 has no headroom: no future detector can do better, so it cannot
    rank anything. A class at the clean rate has no signal. Both are dead weight, and a
    benchmark reporting a mean over them is reporting mostly dead weight.
    """
    live, ceiling, floor = [], [], []
    for code, recall in sorted(per_class.items()):
        if recall >= 0.98:
            ceiling.append(code)
        elif recall <= clean_rate + 0.05:
            floor.append(code)
        else:
            live.append(code)
    total = len(per_class) or 1
    return {
        "live": live, "at_ceiling": ceiling, "at_floor": floor,
        "live_fraction": round(len(live) / total, 3),
        # Over half the classes should still discriminate, or the instrument is mostly
        # reporting results it cannot change.
        "pass": len(live) / total >= 0.5,
    }


# ── Q4. Discrimination — can it rank detectors of known strength? ────────────

class Crippled:
    """
    A detector of deliberately known strength, for testing the benchmark rather than the
    detector.

    `keep` is the share of the real pipeline's score that survives. 0.0 is a detector that
    says nothing; 1.0 is the real one. If the benchmark cannot separate these, it cannot
    separate two real detectors either, and every comparison it has been used for is noise.
    """

    name = "crippled"

    def __init__(self, inner, keep: float, rng: random.Random):
        self.inner, self.keep, self.rng = inner, keep, rng

    def score(self, action, ctx):
        risk, flags = self.inner.score(action, ctx)
        if self.rng.random() > self.keep:
            return 0.0, []
        return risk, flags


def discrimination(data_dir: Path, calibration: dict[str, Any],
                   levels=(0.0, 0.25, 0.5, 0.75, 1.0)) -> dict[str, Any]:
    """
    Q4 — build a ladder of detectors whose true order is known, and check the benchmark
    recovers it.

    This is the closest thing to a ground truth a benchmark can have. We do not know how good
    a real detector is, but we know for certain that a pipeline retaining 25% of its own
    signal is worse than one retaining 75%. A benchmark that ranks those out of order, or
    cannot tell them apart, is not an instrument.
    """
    from acbguard.detectors import Pipeline

    train = load(data_dir / "train.jsonl")
    test = load(data_dir / "test.jsonl")
    baselines = fit_baselines(train)
    apply_limits(baselines, calibration)
    reference = build_price_reference(train, data_dir)

    attacks = [r for r in test if not r.get("is_clean") and r.get("probe_id")]
    cleans = [r for r in test if r.get("is_clean") and not r.get("defects")]

    results = []
    for keep in levels:
        rng = random.Random(99)
        full = l1(calibration, reference)
        crippled = Pipeline([Crippled(d, keep, rng) for d in full.detectors],
                            escalate_at=full.escalate_at, block_at=full.block_at)
        det = sum(score(r, crippled, baselines.get(r["agent_id"], {}))[0] for r in attacks)
        fp = sum(score(r, crippled, baselines.get(r["agent_id"], {}))[0] for r in cleans)
        results.append({
            "keep": keep,
            "recall": round(det / max(1, len(attacks)), 4),
            "clean_flag": round(fp / max(1, len(cleans)), 4),
            "lift": round(det / max(1, len(attacks)) - fp / max(1, len(cleans)), 4),
        })

    lifts = [r["lift"] for r in results]
    monotone = all(b >= a - 0.02 for a, b in zip(lifts, lifts[1:]))
    spread = max(lifts) - min(lifts)
    return {
        "ladder": results,
        "monotone": monotone,
        "spread": round(spread, 4),
        # It must order them correctly AND separate the ends by more than noise.
        "pass": monotone and spread > 0.20,
    }


# ── Q5. Statistical power ────────────────────────────────────────────────────

def power(per_class_n: dict[str, int]) -> dict[str, Any]:
    """
    Q5 — with this many sessions per class, what difference is actually resolvable?

    A 95% interval on a proportion near 0.5 is roughly +/- 1.96*sqrt(0.25/n). At n=35 that is
    +/- 0.17 — so two detectors 15 points apart on one class are not distinguishable, and any
    per-class comparison inside that band is noise being read as a result.
    """
    out = []
    for code, n in sorted(per_class_n.items()):
        half = 1.96 * math.sqrt(0.25 / n) if n else float("inf")
        out.append({"class": code, "n": n, "half_width_95": round(half, 3),
                    "resolvable_difference": round(2 * half, 3)})
    worst = max((o["half_width_95"] for o in out), default=float("inf"))
    median_n = statistics.median(per_class_n.values()) if per_class_n else 0
    smallest = min(per_class_n.values(), default=0)
    # n >= 0.25 * (1.96 / 0.10)^2 in the SMALLEST class, and classes are assigned at random,
    # so the pool has to be large enough that multinomial variance does not starve one.
    need_per_class = math.ceil(0.25 * (1.96 / 0.10) ** 2)
    return {
        "per_class": out,
        "median_n": median_n,
        "smallest_class_n": smallest,
        "need_per_class_for_pm10": need_per_class,
        "worst_half_width_95": round(worst, 3),
        # +/- 0.10 at worst, i.e. n >= ~96 per class, before per-class claims are safe.
        "pass": worst <= 0.10,
    }


# ── Q6. Label integrity ──────────────────────────────────────────────────────

def label_integrity(rows: list[dict[str, Any]],
                    baselines: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """
    Q6 — are the labels true?

    Two ways they can be wrong, and they fail in opposite directions:

    - a CLEAN session that actually violates the agent's own declared limit. Then the
      detector is right and the benchmark scores it as a false positive.
    - an ATTACK session indistinguishable from clean on every axis, which is a mislabel
      rather than a hard case.
    """
    violations = []
    for r in rows:
        if not r.get("is_clean"):
            continue
        base = baselines.get(r["agent_id"], {})
        ceiling = base.get("declared_limit_units") or base.get("ceiling_units")
        if not ceiling:
            continue
        for a in r["actions"]:
            if a["action_type"] == "authorize" and (a.get("amount_units") or 0) > ceiling:
                violations.append({"session": r["session_id"],
                                   "amount": a["amount_units"], "limit": ceiling})
                break
    clean_n = sum(1 for r in rows if r.get("is_clean")) or 1
    rate = len(violations) / clean_n
    return {
        "clean_sessions": clean_n,
        "clean_over_declared_limit": len(violations),
        "rate": round(rate, 4),
        "examples": violations[:3],
        # Some overlap is intended — an "over limit" attack must overlap clean traffic or it
        # sits in a gap. A large share means the label itself is unreliable.
        "pass": rate < 0.05,
    }


# ── driver ───────────────────────────────────────────────────────────────────

def run(data_dir: Path) -> dict[str, Any]:
    cal_path = data_dir / "calibration.json"
    calibration = json.loads(cal_path.read_text()) if cal_path.exists() else {}
    train = load(data_dir / "train.jsonl")
    test = load(data_dir / "test.jsonl")
    baselines = fit_baselines(train)
    apply_limits(baselines, calibration)
    reference = build_price_reference(train, data_dir)
    prod = production_aggregates()

    # Per-class recall and clean rate, for Q3 and Q5.
    pipeline = l1(calibration, reference)
    judge = l0("pattern")
    per_class: dict[str, list[bool]] = defaultdict(list)
    for r in test:
        if r.get("is_clean") or not r.get("probe_id"):
            continue
        base = baselines.get(r["agent_id"], {})
        f1 = score(r, pipeline, base)[0]
        f0 = score(r, judge, base)[0]
        per_class[r["probe_id"]].append(bool(f0 or f1))
    clean_rows = [r for r in test if r.get("is_clean") and not r.get("defects")]
    clean_flagged = sum(
        bool(score(r, pipeline, baselines.get(r["agent_id"], {}))[0]
             or score(r, judge, baselines.get(r["agent_id"], {}))[0])
        for r in clean_rows)
    clean_rate = clean_flagged / max(1, len(clean_rows))

    recalls = {k: sum(v) / len(v) for k, v in per_class.items()}
    counts = {k: len(v) for k, v in per_class.items()}

    return {
        "Q1_fidelity": fidelity(test, prod, data_dir),
        "Q1b_session_shape": session_shape(test),
        "Q2_field_realism": field_realism(test, prod),
        "Q3_saturation": saturation(recalls, clean_rate),
        "Q4_discrimination": discrimination(data_dir, calibration),
        "Q5_power": power(counts),
        "Q6_label_integrity": label_integrity(test, baselines),
        "Q7_deployment_parity": deployment_parity(data_dir, calibration),
        "clean_flag_rate": round(clean_rate, 4),
    }


def _wrap(text: str, width: int) -> list[str]:
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return lines


def render(r: dict[str, Any]) -> str:
    out = ["", "  How good is the benchmark?", "  " + "─" * 72, ""]

    def verdict(block):
        if not block.get("ran", True):
            return "SKIP"
        return "PASS" if block.get("pass") else "FAIL"

    q1 = r["Q1_fidelity"]
    out.append(f"  Q1  fidelity to production .................. {verdict(q1)}")
    if q1.get("ran"):
        cat = q1.get("catalogue", {})
        if cat.get("quantiles"):
            out.append(f"        GATED: catalogue prices vs service_payment_requirements "
                       f"(n={cat['benchmark_n']} vs {cat['production_n']})")
            out.append(f"        worst quantile error {cat['worst_fold']}x  (tolerance 2x)")
            for row in cat["quantiles"]:
                out.append(f"          {row['quantile']}  benchmark {row['benchmark']:>10,.0f}"
                           f"   production {row['production']:>10,.0f}   {row['ratio']:>5.2f}x")
        st = q1.get("settled", {})
        if st.get("quantiles"):
            out += ["", f"        NOT GATED: settled amounts. Production settled from only "
                        f"{st['production_agents']} agents and",
                    f"        {st['top_agent_share']:.1%} of payments came from one of them, so this "
                    f"is one customer's mix,",
                    "        not a distribution to reproduce. Reported because a large gap would "
                    "mean",
                    "        the agent model is wrong.",
                    f"        worst quantile error {st['worst_fold']}x"]
            for row in st["quantiles"]:
                out.append(f"          {row['quantile']}  benchmark {row['benchmark']:>10,.0f}"
                           f"   production {row['production']:>10,.0f}   {row['ratio']:>5.2f}x")
    else:
        out.append(f"        {q1['reason']}")

    shape = r["Q1b_session_shape"]
    out += ["", "  Q1b session shape"]
    out.append(f"        benchmark   {shape['benchmark']}")
    if "production" in shape:
        null_rate = shape.get("production_session_id_null_rate") or 0
        out += [f"        production  {shape['production']}",
                f"        NOT A DISTRIBUTION: session_id is set on {(1 - null_rate):.2%} of",
                f"        settlements ({shape.get('production_n_sessions')} distinct sessions), and the",
                "        `sessions` table has 0 rows. Sessions are not recorded, so the",
                "        benchmark's session shape cannot be validated against production.",
                "        Q7 covers the consequence: detectors must not depend on them."]

    q2 = r["Q2_field_realism"]
    out += ["", f"  Q2  field realism ........................... {verdict(q2)}"]
    if q2.get("ran"):
        for f in q2["fields"]:
            mark = "  <-- benchmark has it, production does not" if f["over_available"] else ""
            out.append(f"        {f['field']:<16} bench {f['benchmark']:.2f}   "
                       f"prod {f['production']:.2f}{mark}")
        if not q2.get("pass"):
            out += [""] + ["        " + line for line in
                           _wrap(q2.get("consequence", ""), 66)]

    q3 = r["Q3_saturation"]
    out += ["", f"  Q3  saturation .............................. {verdict(q3)}",
            f"        {len(q3['live'])}/{len(q3['live']) + len(q3['at_ceiling']) + len(q3['at_floor'])}"
            f" classes still discriminate ({q3['live_fraction']:.0%})",
            f"        at ceiling (no headroom): {', '.join(q3['at_ceiling']) or 'none'}",
            f"        at floor (no signal):     {', '.join(q3['at_floor']) or 'none'}"]

    q4 = r["Q4_discrimination"]
    out += ["", f"  Q4  discrimination .......................... {verdict(q4)}",
            "        a detector ladder of known strength; the benchmark must recover the order",
            "        keep   recall  clean   lift"]
    for row in q4["ladder"]:
        out.append(f"        {row['keep']:.2f}   {row['recall']:.3f}   "
                   f"{row['clean_flag']:.3f}  {row['lift']:+.3f}")
    out.append(f"        monotone={q4['monotone']}   spread={q4['spread']:.3f}")

    q5 = r["Q5_power"]
    out += ["", f"  Q5  statistical power ....................... {verdict(q5)}",
            f"        median n per class {q5['median_n']:.0f}, smallest "
            f"{q5['smallest_class_n']}; worst 95% half-width "
            f"±{q5['worst_half_width_95']:.2f}",
            f"        two detectors closer than "
            f"{2 * q5['worst_half_width_95']:.2f} on one class are not distinguishable",
            f"        need n>={q5['need_per_class_for_pm10']} in the SMALLEST class for ±0.10",
            "        (pooled and per-family claims are unaffected; this bounds per-class ones)"]

    q6 = r["Q6_label_integrity"]
    out += ["", f"  Q6  label integrity ......................... {verdict(q6)}",
            f"        {q6['clean_over_declared_limit']}/{q6['clean_sessions']} clean sessions "
            f"exceed their own declared limit ({q6['rate']:.1%})"]

    q7 = r["Q7_deployment_parity"]
    out += ["", f"  Q7  deployment parity ....................... {verdict(q7)}",
            "        the same detectors scored twice: history reset per session (harness),",
            "        and keyed on agent across sessions (production, where session_id is on",
            "        0.47% of settlements and agent_id on 100%)",
            f"        worst per-class delta {q7['worst_delta']:+.3f}   "
            f"clean {q7['clean']['delta']:+.3f}"]
    if q7["regressed"]:
        out.append(f"        REGRESSED without sessions: {', '.join(q7['regressed'])}")
        for d in q7["per_class"]:
            if d["delta"] <= -0.05:
                out.append(f"          {d['class']}  {d['session_keyed']:.2f} -> "
                           f"{d['agent_keyed']:.2f}")

    failed = [k for k, v in r.items()
              if isinstance(v, dict) and v.get("ran", True) and v.get("pass") is False]
    out += ["", "  " + "─" * 72,
            f"  {len(failed)} check(s) failed: {', '.join(failed) or 'none'}", ""]
    return "\n".join(out)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="benchmark/data/v2")
    p.add_argument("--json", action="store_true")
    p.add_argument("--strict", action="store_true", help="exit 1 if any check fails")
    args = p.parse_args(argv)

    report = run(Path(args.data))
    print(json.dumps(report, indent=2, default=str) if args.json else render(report))

    failed = [k for k, v in report.items()
              if isinstance(v, dict) and v.get("ran", True) and v.get("pass") is False]
    return 1 if (args.strict and failed) else 0


if __name__ == "__main__":
    raise SystemExit(main())
