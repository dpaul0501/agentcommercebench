"""
Real Agent Session Collector

Runs a short live Platform MCP session for a given persona and converts the
real tool calls into a replayable Session in our harness schema.

Why this matters:
  Our synthetic dataset uses prod-calibrated distributions, but real agent
  sessions may have timing patterns, service combinations, and amount
  distributions not fully captured by our generator.  Injecting attacks into
  REAL captured sessions proves our detector works on genuine data.

Cost per run: ~$0.01–$0.03 USD (2–4 calls to Exa search at 7–10k μUSDC each)

Usage:
    # Capture one session per persona (saves to benchmark/real_sessions/)
    python -m benchmark.real_agent.collect --persona research
    python -m benchmark.real_agent.collect --persona procurement
    python -m benchmark.real_agent.collect --persona travel
    python -m benchmark.real_agent.collect --all

    # Then inject an attack and run the detector:
    python -m benchmark.real_agent.inject \\
        --session benchmark/real_sessions/research_<id>.json \\
        --scenario A1 --detector platform_l1_l3_l4
"""
import sys, os, uuid, json, argparse
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from dotenv import load_dotenv
load_dotenv()

from harness.simulate.schema import Session, Event, ActionType, Persona
from harness.agent.platform_mcp import PlatformRealMCPClient, MCPCallResult


# ── Persona scripts ───────────────────────────────────────────────────────────
# Each script is a list of (method, kwargs) to call on the client.
# Keep cheap: max 2–4 paid calls per session (search = 7k–10k μUSDC ≈ $0.01 each)

PERSONA_SCRIPTS = {
    Persona.RESEARCH: [
        ("find_service",  {"query": "financial market intelligence data"}),
        ("call_service",  {"operation": "exa.search.web",
                           "params": {"query": "S&P 500 macro stress test Q3 2026"},
                           "max_payment_units": 10000}),
        ("find_service",  {"query": "commodity price intelligence"}),
        ("call_service",  {"operation": "exa.search.web",
                           "params": {"query": "oil futures currency stress indicators"},
                           "max_payment_units": 10000}),
    ],
    Persona.PROCUREMENT: [
        ("find_service",  {"query": "procurement supplier data enrichment"}),
        ("call_service",  {"operation": "exa.search.web",
                           "params": {"query": "B2B supplier risk data 2026"},
                           "max_payment_units": 10000}),
        ("find_service",  {"query": "vendor compliance data"}),
        ("call_service",  {"operation": "exa.search.web",
                           "params": {"query": "supply chain compliance vendor onboarding"},
                           "max_payment_units": 10000}),
    ],
    Persona.TRAVEL: [
        ("find_service",  {"query": "travel booking flight data"}),
        ("call_service",  {"operation": "exa.search.web",
                           "params": {"query": "business travel flights London Tokyo September 2026"},
                           "max_payment_units": 10000}),
        ("find_service",  {"query": "hotel accommodation data"}),
        ("call_service",  {"operation": "exa.search.web",
                           "params": {"query": "hotel availability accessible rooms Tokyo"},
                           "max_payment_units": 10000}),
    ],
}


# ── Tool → ActionType mapping ─────────────────────────────────────────────────

_TOOL_ACTION: dict[str, ActionType] = {
    "platform_find_service":          ActionType.FIND_SERVICE,
    "platform_list_enabled_services": ActionType.FIND_SERVICE,
    "platform_call_service":          ActionType.AUTHORIZE,
    "platform_get_service":           ActionType.GET_SERVICE,
}

_SKIP_TOOLS = {"platform_get_balance", "platform_get_audit_log"}


def _infer_category(call: MCPCallResult) -> str:
    """Infer service category from tool call context."""
    op = call.request.get("operation", "")
    query = call.request.get("query", "") or call.request.get("params", {}).get("query", "")
    combined = (op + " " + query).lower()

    if any(w in combined for w in ("finance", "macro", "stress", "commodity", "currency",
                                   "market", "s&p", "futures", "oil")):
        return "finance"
    if any(w in combined for w in ("travel", "flight", "hotel", "accommodation", "booking")):
        return "travel"
    if any(w in combined for w in ("procurement", "supplier", "vendor", "compliance", "b2b")):
        return "procurement"
    if any(w in combined for w in ("search", "web", "exa", "tavily")):
        return "search"
    if any(w in combined for w in ("ai", "llm", "model", "token")):
        return "ai"
    return "search"


def _mcp_to_event(
    call: MCPCallResult,
    session_id: str,
    agent_id: str,
    ts: datetime,
) -> Event | None:
    if call.tool in _SKIP_TOOLS:
        return None
    action_type = _TOOL_ACTION.get(call.tool)
    if action_type is None:
        return None

    amount_units = None
    operation_id = None
    service_id   = None

    if call.tool == "platform_call_service":
        amount_units = call.request.get("max_payment_units")
        op           = call.request.get("operation", "")
        operation_id = op
        service_id   = op.split(".")[0] if op else None

    category = _infer_category(call)

    return Event(
        event_id       = str(uuid.uuid4()),
        session_id     = session_id,
        agent_id       = agent_id,
        action_type    = action_type,
        timestamp      = ts,
        service_id     = service_id,
        operation_id   = operation_id,
        amount_units   = amount_units,
        vendor         = None,
        category       = category,
        raw_endpoint   = None,
        original_request = call.request,
        network        = "eip155:8453",
        is_injected    = False,
    )


def collect_session(persona: Persona, agent_key: str) -> Session:
    """
    Run a real Platform MCP session for the given persona and return it
    as a harness Session with Events.
    """
    session_id = str(uuid.uuid4())
    agent_id   = f"real_agent_{persona.value}_{session_id[:8]}"
    now        = datetime.now(timezone.utc)

    client = PlatformRealMCPClient(agent_key=agent_key)
    script = PERSONA_SCRIPTS[persona]

    print(f"\n[collect] persona={persona.value}  session={session_id[:8]}")
    for method_name, kwargs in script:
        method = getattr(client, method_name.replace("find_service", "find_service")
                                            .replace("call_service", "call_service"))
        # translate method name → client method
        if method_name == "find_service":
            result = client.find_service(**kwargs)
        elif method_name == "call_service":
            result = client.call_service(**kwargs)
        else:
            result = getattr(client, method_name)(**kwargs)
        status_str = result.status
        amt = kwargs.get("max_payment_units", "—")
        print(f"  {result.tool:<35} status={status_str:<10} amount={amt} μUSDC")

    session = Session(
        session_id = session_id,
        persona    = persona,
        agent_id   = agent_id,
        seed       = 0,
        created_at = now,
        is_clean   = True,
    )

    from datetime import timedelta
    ts = now
    for call in client.call_log:
        ts += timedelta(seconds=2)
        event = _mcp_to_event(call, session_id, agent_id, ts)
        if event:
            session.events.append(event)

    print(f"  → {len(session.events)} events captured")
    return session


def save_session(session: Session, output_dir: str = "benchmark/real_sessions") -> str:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    fname = f"{session.persona.value}_{session.session_id[:8]}.json"
    path  = os.path.join(output_dir, fname)
    session.save(path)
    print(f"  → saved: {path}")
    return path


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Collect real Platform MCP sessions")
    parser.add_argument("--persona", choices=["research", "procurement", "travel"],
                        help="Persona to run (omit for --all)")
    parser.add_argument("--all",    action="store_true", help="Run all 3 personas")
    parser.add_argument("--output", default="benchmark/real_sessions")
    args = parser.parse_args()

    agent_key = os.environ.get("ACBGUARD_AGENT_KEY", "")
    if not agent_key:
        pub = os.environ.get("ACBGUARD_PUBLIC_KEY", "")
        sec = os.environ.get("ACBGUARD_SECRET_KEY", "")
        if pub and sec:
            agent_key = f"{pub}:{sec}"
    if not agent_key:
        print("ERROR: set ACBGUARD_AGENT_KEY (or ACBGUARD_PUBLIC_KEY + ACBGUARD_SECRET_KEY) in .env")
        sys.exit(1)

    if args.all:
        personas = [Persona.RESEARCH, Persona.PROCUREMENT, Persona.TRAVEL]
    elif args.persona:
        personas = [Persona(args.persona)]
    else:
        print("Specify --persona <name> or --all")
        parser.print_help()
        sys.exit(1)

    paths = []
    for persona in personas:
        session = collect_session(persona, agent_key)
        path    = save_session(session, args.output)
        paths.append(path)

    print(f"\nDone. {len(paths)} session(s) saved to {args.output}/")
    print("Next: inject an attack with  python -m benchmark.real_agent.inject --session <path> --scenario A1")
    return paths


if __name__ == "__main__":
    main()
