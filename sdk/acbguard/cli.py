"""
acbguard CLI.

    acbguard probes
    acbguard targets
    acbguard scan pipeline
    acbguard scan mcp+https://api.example.com/mcp --auth "Bearer $TOKEN"
    acbguard scan openai:gpt-4o-mini
    acbguard learn traces.jsonl
    acbguard audit agent.json --fail-over 20
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from . import __version__
from .probes import all_probes, families
from .schema import Persona
from .targets.registry import SCHEMES, from_uri


def cmd_probes(args: argparse.Namespace) -> int:
    probes = all_probes(families=args.family)
    width = max((len(p.id) for p in probes), default=4)
    current = None
    for p in probes:
        if p.family != current:
            current = p.family
            print(f"\n  {current}")
        print(f"    {p.id:<{width}}  {p.title}")
        print(f"    {'':<{width}}  {p.description}")
    print(f"\n  {len(probes)} probes across {len(families())} families\n")
    return 0


def cmd_targets(args: argparse.Namespace) -> int:
    width = max(len(s) for s, _ in SCHEMES)
    print("\n  Target URIs\n")
    for scheme, desc in SCHEMES:
        print(f"    {scheme:<{width}}   {desc}")
    print("\n  Example:  acbguard scan ucp+https://merchant.example.com/ucp\n")
    return 0


def cmd_scan(args: argparse.Namespace) -> int:
    from .scan import scan

    try:
        target = from_uri(args.target, auth=args.auth)
    except (ValueError, ImportError) as exc:
        print(f"  error: {exc}\n", file=sys.stderr)
        return 2

    report = scan(
        target,
        families=args.family,
        persona=Persona(args.persona),
        seed=args.seed,
    )
    print(report.render(verbose=args.verbose))

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(report.to_dict(), fh, indent=2)
        print(f"  report written to {args.json}\n")

    if args.fail_over is not None and report.exposure > args.fail_over:
        print(f"  FAIL: exposure {report.exposure} exceeds --fail-over {args.fail_over}\n")
        return 1
    return 0


def cmd_learn(args: argparse.Namespace) -> int:
    """Fit baselines from recorded traces — lifecycle stage 2."""
    from .baselines import LearnedBaseline
    from .sinks import FileSink

    records = FileSink(args.traces).read()
    if not records:
        print(f"  no usable records in {args.traces}\n", file=sys.stderr)
        return 2

    learned = LearnedBaseline(k=args.k, min_samples=args.min_samples).fit(records)
    if not learned.agents:
        print(f"  read {len(records)} records but found no agent_id to fit on\n",
              file=sys.stderr)
        return 2

    print(f"\n  Fitted {len(learned.agents)} agent(s) from {len(records)} records\n")
    out: dict = {}
    for agent in learned.agents:
        base = learned.baseline_for(agent)
        out[agent] = {
            k: (sorted(v) if isinstance(v, set) else
                [v.start, v.stop - 1] if isinstance(v, range) else v)
            for k, v in base.items()
        }
        typical = base.get("typical_amount_units")
        ceiling = base.get("ceiling_units")
        print(f"    {agent}")
        print(f"      samples:  {base.get('samples', 0)}"
              + ("   (cold start — below min-samples)" if base.get("cold_start") else ""))
        if typical:
            print(f"      typical:  ${typical / 1_000_000:,.2f}")
        if ceiling:
            print(f"      ceiling:  ${ceiling / 1_000_000:,.2f}")
        if base.get("known_services"):
            print(f"      services: {len(base['known_services'])}")
    print()

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2)
        print(f"  baselines written to {args.out}\n")
    return 0


def _load_spec(path: str, name: Optional[str] = None):
    """
    Build an AgentSpec from a JSON config file.

    Accepts the SDK's own shape and the MCP `tools/list` shape, because the second is what a
    user can actually get out of a running server without writing any code:

        {"tools": [{"name": "pay", "description": "...", "inputSchema": {"properties": {...}}}]}
    """
    from .agent import AgentSpec, ToolSpec

    with open(path) as fh:
        raw = json.load(fh)
    if isinstance(raw, list):                       # a bare tools/list result
        raw = {"tools": raw}

    tools = []
    for entry in raw.get("tools", []):
        schema = entry.get("inputSchema") or entry.get("parameters") or {}
        props = schema.get("properties", schema) if isinstance(schema, dict) else {}
        params = {
            k: (v.get("type", "string") if isinstance(v, dict) else str(v))
            + (f" (max {v['maximum']})" if isinstance(v, dict) and "maximum" in v else "")
            for k, v in props.items()
        }
        tools.append(ToolSpec(entry.get("name", "?"), entry.get("description", ""), params))

    kwargs = {"model": None, "tools": tools,
              "name": name or raw.get("name") or Path(path).stem}
    for key in ("task", "system_prompt"):
        if raw.get(key):
            kwargs[key] = raw[key]
    return AgentSpec(**kwargs)


def cmd_audit(args: argparse.Namespace) -> int:
    from .config_audit import audit

    report = audit(_load_spec(args.config, args.name))
    print(report.render())
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(report.to_dict(), fh, indent=2)
        print(f"  report written to {args.json}\n")
    if args.fail_over is not None and report.risk > args.fail_over:
        print(f"  FAIL: risk {report.risk} exceeds {args.fail_over}\n")
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="acbguard",
        description="Adversarial testing and runtime guarding for agents that spend money.",
    )
    parser.add_argument("--version", action="version", version=f"acbguard {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_probes = sub.add_parser("probes", help="list available probes")
    p_probes.add_argument("--family", action="append", help="filter by family (repeatable)")
    p_probes.set_defaults(func=cmd_probes)

    p_targets = sub.add_parser("targets", help="list supported target URI schemes")
    p_targets.set_defaults(func=cmd_targets)

    p_scan = sub.add_parser("scan", help="run probes against a target")
    p_scan.add_argument("target", help="target URI (see: acbguard targets)")
    p_scan.add_argument("--auth", help="Authorization header value")
    p_scan.add_argument("--family", action="append", help="only run this family (repeatable)")
    p_scan.add_argument("--persona", default="procurement",
                        choices=[p.value for p in Persona])
    p_scan.add_argument("--seed", type=int, default=42)
    p_scan.add_argument("--json", help="write the full report to this path")
    p_scan.add_argument("--fail-over", type=float, metavar="N",
                        help="exit 1 if exposure exceeds N (for CI)")
    p_scan.add_argument("-v", "--verbose", action="store_true")
    p_scan.set_defaults(func=cmd_scan)

    p_audit = sub.add_parser(
        "audit", help="statically audit an agent config (no model calls, no spend)")
    p_audit.add_argument("config", help="JSON agent config, or an MCP tools/list dump")
    p_audit.add_argument("--name", help="override the agent name in the report")
    p_audit.add_argument("--json", help="write the full report to this path")
    p_audit.add_argument("--fail-over", type=float, metavar="N",
                         help="exit 1 if risk exceeds N (for CI)")
    p_audit.set_defaults(func=cmd_audit)

    p_learn = sub.add_parser("learn", help="fit baselines from recorded traces")
    p_learn.add_argument("traces", help="path to a JSONL trace file")
    p_learn.add_argument("--out", help="write fitted baselines to this JSON path")
    p_learn.add_argument("--k", type=float, default=1.1,
                         help="stdev multiplier for the ceiling (default 1.1)")
    p_learn.add_argument("--min-samples", type=int, default=20)
    p_learn.set_defaults(func=cmd_learn)

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
