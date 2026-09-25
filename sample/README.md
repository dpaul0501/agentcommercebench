---
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

### `benchmark` — 1000 generated sessions

Sampled evenly across all twenty classes and the clean population, then labelled by the
observation surface that can see the class at all.

| surface | sessions |
|---|---|
| clean | 500 |
| none | 25 |
| reasoning | 25 |
| reasoning+wire | 75 |
| wire | 375 |

A class is `wire` if only the settlement record carries the evidence, `reasoning` if only the
model's visible thinking does, and `reasoning+wire` if detection needs both. F6 silent
downgrade is the control: its surface is `none`, nothing we model can observe it, so it should
score at the clean flag rate. A detector that beats chance on F6 is reading the answer key.

| class | sessions |
|---|---|
| A1 | 25 |
| A2 | 25 |
| A3 | 25 |
| A4 | 25 |
| E1 | 25 |
| E2 | 25 |
| E3 | 25 |
| E4 | 25 |
| F1 | 25 |
| F2 | 25 |
| F3 | 25 |
| F4 | 25 |
| F5 | 25 |
| F6 | 25 |
| S1 | 25 |
| S2 | 25 |
| S3 | 25 |
| S4 | 25 |
| S5 | 25 |
| S6 | 25 |
| clean | 500 |

Each record is a session: an ordered list of actions for one agent, with `is_clean`,
`probe_id`, `defects` and `agent_spec`. Labels live on the session, never on an action, and
are stripped before any detector sees them.

### `production` — 30 real settled actions

Live agent payments, de-identified. Sampling is deliberately not uniform — every blocked,
escalated and injection-flagged row is included, because the rare rows carry the signal.

| decision | actions |
|---|---|
| allow | 19 |
| block | 9 |
| escalate | 2 |

Categories: search (10), finance (6), service (4), ai (3), infrastructure (3), security (3), scrape (1).

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
@misc{acb2026,
  title  = {Agentic Commerce Bench},
  author = {Anonymous Author(s)},
  year   = {2026},
  url    = {https://anonymous.4open.science/r/agentcommercebench}
}
```

Apache 2.0.
