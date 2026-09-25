# acbguard

Adversarial testing and runtime guarding for AI agents that spend money.

Point it at an agent or an MCP server, and it tells you which attacks land, which
your controls stop, and, for the ones that land, whether they were detectable at all.

```
  acbguard scan — my-procurement-agent
  ──────────────────────────────────────────────────────────────
  Grade D    exposure 47.3/100    9/18 probes landed
  Of what landed, acbguard flags 77.8%
  3 caught only mid-attack — 11 action(s) executed first

  By family
    behavioral   ████████████████····  6/7 landed
    injection    █████···············  2/7 landed
    settlement   ██████████··········  2/4 landed
```

The core has **no dependencies**. It should be trivial to drop into an existing agent.

---

## Install

```bash
pip install acbguard              # everything below, zero dependencies
pip install "acbguard[mcp]"       # + MCP transports
```

HTTP, JSON-RPC, A2A, UCP, and every model target run on the standard library alone.

## Try it in ten seconds

```bash
acbguard probes                   # what it tests for
acbguard targets                  # what it can point at
acbguard scan unguarded           # an agent with no controls: everything lands
acbguard scan pipeline            # acbguard's own detectors, scanned honestly
```

## Scan anything

One protocol, `execute(action) -> Observation`, so a probe never knows what is
behind it. Every target is a URI:

```bash
acbguard scan mcp+https://api.example.com/mcp --auth "Bearer $TOKEN"
acbguard scan a2a+https://agent.example.com/a2a     # Agent2Agent
acbguard scan ucp+https://merchant.example.com/ucp  # Universal Commerce Protocol
acbguard scan https://api.example.com/authorize     # any REST endpoint
acbguard scan openai:gpt-4o-mini                    # any OpenAI-compatible model
acbguard scan anthropic:claude-sonnet-4-6
acbguard scan langgraph:myapp.graphs:checkout
acbguard scan module:myagent:handler --json report.json
```

`openai:` covers anything speaking that API (together.ai, Groq, Fireworks, vLLM,
LM Studio, Ollama) via `base_url`.

In Python:

```python
from acbguard import scan
from acbguard.targets import CallableTarget

def my_agent(action):
    if action.amount_units > 3_000_000:
        return {"decision": "block", "reason": "over limit"}
    return {"decision": "allow"}

report = scan(CallableTarget(my_agent, name="my-agent"))
print(report.render())
```

**Scanning a model** works differently from scanning a system: the proposed
transaction is rendered as a prompt, including whatever the probe planted, and the
model is asked to ALLOW, BLOCK, or ESCALATE. If injected text talks it into ALLOW,
the injection worked.

---

## Scan an agent, not just a session

The targets above receive a decision the agent already made. That tests a control. An
agent is a *configuration*: a model, a system prompt, tools, a task. Given those
you can run the loop and watch what it decides:

```python
from acbguard import AgentSpec, AgentTarget, ToolSpec, scan

spec = AgentSpec(
    model=my_model,                       # anything with .complete(system, user)
    task="Find a search provider and buy one query.",
    tools=[ToolSpec("call_service", "Call a service. This charges the wallet.",
                    {"service_id": "string", "amount_units": "integer"})],
)
print(scan(AgentTarget(spec)).render())
```

Two things follow from running the agent rather than replaying it. Probes can attack the
*configuration*, since a poisoned tool description is untrusted text reaching the model
with the authority of a system message, which is what A7 does. The agent's reasoning
becomes visible, which is the only way the reasoning detector has anything to score.

Every verdict is relative to a clean reference run. An attack landed when the agent spent
*more* than it does untouched, or paid someone the clean run never paid. Otherwise an
agent doing its job would grade F.

### Audit the config without running anything

The configuration alone says a lot, and reading it costs no tokens and no money:

```bash
acbguard audit agent.json --fail-over 20
```

`agent.json` can be the SDK's own shape or a raw MCP `tools/list` dump. Ten checks across
three groups. What the prompt establishes (spend ceiling, precedence over tool output, an
approval boundary), what the tools permit (unbounded amounts, free destinations,
non-idempotent charges, a fetch tool sharing context with a paying one), and instruction
text already sitting in a description, which is tool poisoning found at rest.

A check firing means the configuration *allows* something, not that the agent will do it.
`AgentTarget` is how you find out which.

### In CI

```bash
acbguard scan module:myagent:handler --fail-over 20
```

Exits non-zero when exposure goes above the threshold, so a regression in your
controls fails the build.

---

## What it tests

18 probes in three families. Each takes a clean, in-policy session and mutates it,
so what you are testing is your controls, not your happy path.

| Family | Probes | Signal | Catchable by |
|---|---|---|---|
| **injection** | A1–A7 | adversarial text in the payload | payload scanning |
| **behavioral** | B1–B7 | the action *sequence* is the attack | per-agent baselines |
| **settlement** | C1–C2, D1–D2 | identity, replay, circular value | durable registries |

The families are ordered by difficulty on purpose. Injection carries a text signal, so
any payload scanner should catch it, and a stack that fails this family fails everything.
Behavioral attacks contain no adversarial text at all: every action is well-formed and
in-policy, and only the shape of the sequence gives them away. Settlement attacks need
state the action does not carry, so no amount of inspecting the current request will
find them.

## Two numbers, kept apart

**Exposure** is what your target let through, severity-weighted, so a settlement
attack landing costs more than an injection landing.

**Coverage** is what acbguard's own detectors flagged on the same input. It answers
a different question: of the attacks that got past you, how many are addressable with
detection you could deploy today.

A third line appears when it applies:

```
  3 caught only mid-attack — 11 action(s) executed first
```

Behavioral detection is **structurally late**. A velocity burst or an amount ratchet is
only visible once the pattern has emerged, which means the early actions execute. The
report says so rather than counting those as clean saves.

---

## Runtime guard

The same detectors, applied inline.

```python
from acbguard import Guard

guard = Guard(mode="observe", trace_path="traces.jsonl")

@guard.watch
def authorize(action):
    ...
```

**Observe** scores and records without ever interfering. It is the default because it
is the only mode that is safe before you have a baseline, and the traces it writes are
what a baseline is derived from.

**Enforce** raises `Blocked` on a block, and routes escalations to `on_escalate`:

```python
guard = Guard(mode="enforce", on_escalate=lambda action, verdict: ask_a_human(action))
```

Detection quality depends on the baseline. Without one, the behavioral layer falls
back to weak structural signals. That is the cold-start position, which is real and
unavoidable on day one.

## Where traces go, where norms come from

Two interfaces, so acbguard never assumes a backend:

```python
class TraceSink(Protocol):        # where observations go
    def emit(self, record) -> None: ...

class BaselineProvider(Protocol):  # where norms come from
    def baseline_for(self, agent_id: str) -> dict: ...
```

Ships with `FileSink`, `MemorySink`, `MultiSink`, `NullSink`, all local. Every sink
**fails open** by contract: telemetry that can take down the agent it observes is
worse than no telemetry.

Norms come from `StaticBaseline` (declared by hand, works on day one),
`LearnedBaseline` (fitted from your own traces), or your own implementation.

### Instrument, then learn, then enforce

That order is a technical dependency rather than advice: you cannot fit a per-agent
baseline before you have that agent's traffic.

```bash
# 1. observe: nothing blocks, traces accumulate
ACBGUARD_TRACE=traces.jsonl python -m myagent

# 2. learn: fit norms from what actually happened
acbguard learn traces.jsonl --out baselines.json
```

```python
# 3. enforce, with norms that came from real traffic
guard = Guard(mode="enforce", baseline=json.load(open("baselines.json"))["agent-7"])
```

**Two limits, and the difference matters.** `LearnedBaseline` emits a `soft_limit`
at mean+1.1σ of log-amount, roughly the 86th percentile, so a meaningful slice of
perfectly ordinary traffic exceeds it. That is correct for *escalation* and would be
a disaster as a hard block. The hard `ceiling` sits far above anything observed.
Hard-blocking legitimate traffic is the expensive error; escalating it costs a review.

### Hosted backend (optional)

`PlatformSink` and `PlatformBaselines` ship as one implementation of those interfaces,
not a dependency. Nothing in the core imports them, and none of the above needs an
account.

```python
from acbguard.sinks import PlatformSink
guard = Guard(sink=PlatformSink(agent_key="gak_pub_...:gak_sec_..."))
```

Only **derived features** leave the machine: amounts, identifiers, timestamps,
detector flags. Raw payload text is reduced to a SHA-256 digest, enough to notice
the same payload twice, not enough to reconstruct it. Sending payloads is possible
but requires `send_payloads=True`. Outbound only; TLS enforced; batched and silent
on failure.

---

## Extending it

Custom probe:

```python
from acbguard.probes import probe

@probe(id="X1", family="injection", title="Our known bad pattern",
       description="...", expected_layer="payload")
def x1(session, rng):
    action = session.actions[-1]
    action.payload["note"] = "..."
    action.is_attack = True          # required: marks what the scanner scores
    return session
```

Custom detector, anything with a `name` and a `score`:

```python
class MyDetector:
    name = "mine"
    def score(self, action, ctx):
        return (0.8, ["reason"]) if bad(action) else (0.0, [])
```

Detectors compose by **max**, never mean: any one layer can raise the verdict on its
own, and a later layer can never talk an earlier one down. A detector that raises is
isolated and flagged rather than being allowed to swallow the action.

---

## Design notes

**Money is integer micro-units.** `1_000_000 == $1.00`, matching USDC's six decimals.
Never use floats for amounts.

**No LLM in the scoring path.** Every shipped detector is deterministic and runs in
well under a millisecond, so the same code can sit inline in a latency-sensitive
authorization path. Semantic analysis belongs in an offline lane, not here.

**Probes are pure.** They clone before mutating, so one baseline can be attacked
eighteen ways and the results compared like for like.

## License

Apache-2.0
