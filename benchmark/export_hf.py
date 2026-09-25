"""
Build the public Hugging Face release of Agentic Commerce Bench.

Two configs come out of here:

    benchmark/sessions.jsonl   a stratified sample of generated sessions, balanced across
                               every class and every observation surface
    production/actions.jsonl   a small slice of real settled traffic, de-identified

The production slice is the part that needs care, so the rules are stated here rather than
buried in the code below:

  * `b64_decoded` is DROPPED. Three rows carry it and none decode to intelligible text, so
    its content cannot be certified free of anything. An uncertifiable field does not ship.
  * `transaction_id` and `agent_id` are replaced by HMAC-SHA256 digests under a salt drawn
    fresh at export time and never written out. Identifiers stay linkable inside the file and
    are not reversible outside it, including by us.
  * Timestamps are shifted by one constant offset. Inter-arrival gaps survive, which is what
    a velocity check needs; the calendar dates do not.
  * `endpoint` is kept whole. These are public vendor APIs, not personal data, and the
    hostname is the merchant surface — the slice is inert without it.
  * `query_string` is kept. Every distinct value was read before release: pagination
    parameters, market-research queries, and one prompt-injection probe.

Run:
    python -m benchmark.export_hf --out dist/hf --prod prod_transactions_extracted.csv
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import hmac
import json
import os
import random
import secrets
import urllib.parse
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .config import JURISDICTION

# Anchor for the shifted production timestamps. Chosen, not measured.
EPOCH = datetime(2026, 1, 1, tzinfo=timezone.utc)

# Columns that never reach the release, with the reason each is held back.
PROD_DROP = {
    "b64_decoded": "content not certifiable as free of personal data",
}

SURFACE_NAMES = {
    frozenset({"L1"}): "wire",
    frozenset({"L0"}): "reasoning",
    frozenset({"L0", "L1"}): "reasoning+wire",
    frozenset(): "none",
}


def surface_of(probe_id: str | None) -> str:
    """Which observation surface can see this class at all."""
    if not probe_id:
        return "clean"
    return SURFACE_NAMES[JURISDICTION[probe_id[:2]]]


def pseudonym(value: str, salt: bytes, prefix: str) -> str:
    digest = hmac.new(salt, value.encode(), hashlib.sha256).hexdigest()[:12]
    return f"{prefix}{digest}"


# ─────────────────────────────────────────────────────────────────────────────
# benchmark config


def sample_sessions(path: Path, per_class: int, clean: int, seed: int) -> list[dict]:
    """Take `per_class` sessions of each attack class plus `clean` clean ones."""
    by_class: dict[str, list[dict]] = defaultdict(list)
    with path.open() as fh:
        for line in fh:
            s = json.loads(line)
            key = s["probe_id"][:2] if s["probe_id"] else "clean"
            by_class[key].append(s)

    rng = random.Random(seed)
    out: list[dict] = []
    for key in sorted(by_class):
        pool = by_class[key]
        want = clean if key == "clean" else per_class
        picked = rng.sample(pool, min(want, len(pool)))
        for s in picked:
            s["surface"] = surface_of(s["probe_id"])
            out.append(s)
    rng.shuffle(out)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# production config


ANOMALY_SHARE = 0.4
"""How much of the production slice is given to non-allow rows.

Production is 96% allows, so a uniform sample of thirty rows would carry one or two
anomalies and a slice made only of the rare rows would misrepresent the traffic entirely.
Neither is useful. Forty percent keeps allows in the majority, so the contrast between
normal and anomalous is visible, while guaranteeing enough rare rows to exercise a detector.
The slice is sized for inspection, not for fitting a baseline, and the card says so.
"""


def load_production(path: Path, want: int, salt: bytes, seed: int) -> list[dict]:
    """De-identify real traffic and keep a spread over decision and category."""
    rows = [r for r in csv.DictReader(path.open()) if r["created_at"].strip()]
    rows.sort(key=lambda r: r["created_at"])

    base = datetime.fromisoformat(rows[0]["created_at"]).replace(tzinfo=timezone.utc)
    shift = EPOCH - base

    def interesting(r: dict) -> bool:
        return r["decision"] != "allow" or r["heuristic_injection"] == "True"

    rng = random.Random(seed)

    def spread(pool: list[dict], quota: int, key: str) -> list[dict]:
        """Round-robin over `key` so one busy value cannot take the whole quota."""
        buckets: dict[str, list[dict]] = defaultdict(list)
        for r in pool:
            buckets[r[key]].append(r)
        order = sorted(buckets, key=lambda c: -len(buckets[c]))
        for c in order:
            rng.shuffle(buckets[c])
        out, i = [], 0
        while quota > 0 and any(buckets.values()):
            c = order[i % len(order)]
            if buckets[c]:
                out.append(buckets[c].pop())
                quota -= 1
            i += 1
        return out

    odd = [r for r in rows if interesting(r)]
    # The injection probe is the single most useful row in the set; never let sampling lose it.
    forced = [r for r in odd if r["heuristic_injection"] == "True"]
    rest = [r for r in odd if r not in forced]
    quota = max(len(forced), round(want * ANOMALY_SHARE))
    picked = forced + spread(rest, quota - len(forced), "decision")
    picked += spread([r for r in rows if not interesting(r)], want - len(picked), "category")

    picked.sort(key=lambda r: r["created_at"])

    out = []
    for r in picked:
        ts = datetime.fromisoformat(r["created_at"]).replace(tzinfo=timezone.utc) + shift
        qs = r["query_string"].strip()
        out.append(
            {
                "action_id": pseudonym(r["transaction_id"], salt, "act-"),
                "agent_id": pseudonym(r["agent_id"], salt, "agent-"),
                "action_type": "settle",
                "timestamp": ts.isoformat(),
                "operation_id": r["operation_id"] or None,
                "category": r["category"] or None,
                "amount_units": int(float(r["amount"])) if r["amount"].strip() else None,
                "endpoint": r["endpoint"] or None,
                "payload": dict(urllib.parse.parse_qsl(qs)) if qs else {},
                "context_sources": [],
                # Ground truth as the live engine recorded it, not as a detector scored it.
                "observed": {
                    "decision": r["decision"],
                    "rule_triggered": r["rule_triggered"] or None,
                    "receipt_status": r["receipt_status"] or None,
                    "heuristic_injection": r["heuristic_injection"] == "True",
                    "matched_keywords": [
                        k.strip() for k in r["matched_keywords"].split(",") if k.strip()
                    ],
                },
            }
        )
    return out


# ─────────────────────────────────────────────────────────────────────────────


def dataset_card(bench: list[dict], prod: list[dict]) -> str:
    surfaces = Counter(s["surface"] for s in bench)
    classes = Counter(s["probe_id"][:2] if s["probe_id"] else "clean" for s in bench)
    decisions = Counter(a["observed"]["decision"] for a in prod)
    cats = Counter(a["category"] for a in prod)

    def table(counter: Counter, head: str) -> str:
        lines = [f"| {head} | sessions |", "|---|---|"]
        lines += [f"| {k} | {v} |" for k, v in sorted(counter.items())]
        return "\n".join(lines)

    return f"""---
license: apache-2.0
task_categories:
- tabular-classification
tags:
- agents
- payments
- fraud-detection
- anomaly-detection
- benchmark
configs:
- config_name: benchmark
  data_files: benchmark/sessions.jsonl
- config_name: production
  data_files: production/actions.jsonl
---

# Agentic Commerce Bench

Twenty attack classes against agents that hold spend authority, plus a slice of real settled
agent traffic. Companion to the paper and to the `acbguard` detector stack.

An agent pays with a credential issued to it, acting on instructions it was genuinely given.
Every identity control stays silent, correctly, because nothing about the identity is wrong.
Six of the twenty classes here have that shape: the counterparty is exactly who it claims to
be and still takes more than it should.

## Configs

### `benchmark` — {len(bench)} generated sessions

Sampled evenly across all twenty classes and the clean population, then labelled by the
observation surface that can see the class at all.

{table(surfaces, "surface")}

A class is `wire` if only the settlement record carries the evidence, `reasoning` if only the
model's visible thinking does, and `reasoning+wire` if detection needs both. F6 silent
downgrade is the control: its surface is `none`, nothing we model can observe it, so it should
score at the clean flag rate. A detector that beats chance on F6 is reading the answer key.

{table(classes, "class")}

Each record is a session: an ordered list of actions for one agent, with `is_clean`,
`probe_id`, `defects` and `agent_spec`. Labels live on the session, never on an action, and
are stripped before any detector sees them.

### `production` — {len(prod)} real settled actions

Live agent payments, de-identified. Sampling is deliberately not uniform — every blocked,
escalated and injection-flagged row is included, because the rare rows carry the signal.

| decision | actions |
|---|---|
{chr(10).join(f"| {k} | {v} |" for k, v in sorted(decisions.items()))}

Categories: {", ".join(f"{k} ({v})" for k, v in cats.most_common())}.

`observed` carries what the live engine decided, which is ground truth about the deployed
system rather than about the attack. Use it to check a detector against production behaviour;
do not read it as a fraud label.

**De-identification.** `b64_decoded` is dropped, since none of its three values decode to
intelligible text and an uncertifiable field does not ship. Identifiers are HMAC-SHA256
digests under a salt drawn at export and never written down, so they stay linkable inside the
file and are not reversible outside it. Timestamps are shifted by one constant offset, which
preserves inter-arrival gaps and discards calendar dates. Endpoints are kept whole: they are
public vendor APIs, and the hostname is the merchant surface the data exists to show. Every
distinct query string was read before release.

## Use with the SDK

```bash
pip install acbguard
```

```python
import json
from acbguard import Guard
from acbguard.schema import Action

guard = Guard(mode="observe")
for line in open("production/actions.jsonl"):
    row = json.loads(line)
    verdict = guard.check(Action.from_dict(row))
    print(row["action_id"], round(verdict.risk_score, 2), verdict.decision)
```

`Action.from_dict` ignores keys the schema does not define, so the `observed` block rides
along in the raw record without reaching the detector. The same call loads the benchmark
config, whose sessions carry a list of actions in the identical shape.

Out of the box the pipeline is uncalibrated and nearly silent on this slice: it scores one
action, the prompt-injection probe, at 0.95. Production's own heuristic also saw that request
and marked it, and the recorded decision was still `allow` — detection was wired to a log and
not to the decision path. Thresholds must be fitted on clean traffic before any other score
means anything, and fitting each detector to a 10% false-positive budget does not give a 10%
pipeline: seven detectors firing independently compounded to 47%, so the budget is fitted on
the aggregate score.

## Limitations

The benchmark is synthetic. Parameters are grounded in production aggregates, but no real
customer session is in it. The production config is real and small, sized for inspection and
integration testing rather than for measuring a detector. Traffic settles over a single
irreversible microtransaction rail; conclusions that depend on irreversibility do not carry
to authorise-then-capture rails such as UCP.

## Citation

```bibtex
@misc{{acb2026,
  title  = {{Agentic Commerce Bench}},
  author = {{Anonymous Author(s)}},
  year   = {{2026}},
  url    = {{https://anonymous.4open.science/r/agentcommercebench}}
}}
```

Apache 2.0.
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="benchmark/data/v2", help="generated pool")
    ap.add_argument("--prod", default="prod_transactions_extracted.csv")
    ap.add_argument("--out", default="dist/hf")
    ap.add_argument("--per-class", type=int, default=25)
    ap.add_argument("--clean", type=int, default=500)
    ap.add_argument("--prod-rows", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = Path(args.out)
    (out / "benchmark").mkdir(parents=True, exist_ok=True)
    (out / "production").mkdir(parents=True, exist_ok=True)

    bench = sample_sessions(
        Path(args.data) / "test.jsonl", args.per_class, args.clean, args.seed
    )
    with (out / "benchmark" / "sessions.jsonl").open("w") as fh:
        for s in bench:
            fh.write(json.dumps(s) + "\n")

    salt = secrets.token_bytes(32)
    prod_path = Path(args.prod)
    prod: list[dict] = []
    if prod_path.exists():
        prod = load_production(prod_path, args.prod_rows, salt, args.seed)
        with (out / "production" / "actions.jsonl").open("w") as fh:
            for a in prod:
                fh.write(json.dumps(a) + "\n")
    else:
        print(f"warning: {prod_path} absent, production config skipped")

    (out / "README.md").write_text(dataset_card(bench, prod))

    # Refuse to ship a field that was meant to be held back.
    leaked = set()
    for a in prod:
        leaked |= set(a) & set(PROD_DROP)
    if leaked:
        raise SystemExit(f"held-back fields present in output: {sorted(leaked)}")

    print(f"benchmark: {len(bench)} sessions -> {out}/benchmark/sessions.jsonl")
    print(f"production: {len(prod)} actions -> {out}/production/actions.jsonl")
    print(f"card:       {out}/README.md")


if __name__ == "__main__":
    main()
