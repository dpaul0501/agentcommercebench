"""
Real Session Attack Injector

Loads a real captured session from collect.py, applies an attack injector,
runs the detector pipeline, and prints the result.

Usage:
    python -m benchmark.real_agent.inject \\
        --session benchmark/real_sessions/research_abc123.json \\
        --scenario A1

    python -m benchmark.real_agent.inject \\
        --session benchmark/real_sessions/research_abc123.json \\
        --scenario B3 --detector platform_+seq

    # Batch: inject all scenarios into all real sessions
    python -m benchmark.real_agent.inject --batch --dir benchmark/real_sessions
"""
import sys, os, argparse, json
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from harness.simulate.schema import Session
from harness.simulate.injectors import inject, ALL_SCENARIOS
from harness.simulate.replay import replay, ALLOW_THRESHOLD, ESCALATE_THRESHOLD

from benchmark.baselines import velocity, keyword, isolation_forest
from benchmark.baselines.llm_safety import detect as llm_safety_detect
from benchmark.detectors import adapter, sequence_model

PIPELINES = {
    "velocity_check":   [velocity.detect],
    "keyword_filter":   [keyword.detect],
    "llm_text_safety":  [llm_safety_detect],
    "isolation_forest": [isolation_forest.detect],
    "platform_l1_l3_l4":  [adapter.detect],
    "platform_+seq":      [adapter.detect, sequence_model.detect],
}


def run_one(session_path: str, scenario: str, detector_name: str = "platform_+seq") -> dict:
    """Load a real session, inject scenario, run detector, return result dict."""
    session = Session.load(session_path)
    print(f"\n{'='*60}")
    print(f"Session : {session_path}")
    print(f"Persona : {session.persona.value}")
    print(f"Events  : {len(session.events)}")
    print(f"Scenario: {scenario}")
    print(f"Detector: {detector_name}")

    attacked = inject(session, scenario=scenario, seed=42)
    n_injected = sum(1 for e in attacked.events if e.is_injected)
    print(f"Injected: {n_injected} event(s) marked")

    pipeline = PIPELINES.get(detector_name)
    if not pipeline:
        raise ValueError(f"Unknown detector: {detector_name}. Valid: {list(PIPELINES)}")

    # Annotate persona for sequence model
    for e in attacked.events:
        e._persona = attacked.persona.value

    result = replay(attacked, pipeline=pipeline, detector_name=detector_name, verbose=True)

    caught = result.true_positive
    print(f"\n{'CAUGHT ✓' if caught else 'MISSED ✗'}  session_risk={result.session_risk_score:.3f}")

    # Show per-event scores for injected events
    injected_idxs = [i for i, e in enumerate(attacked.events) if e.is_injected]
    print(f"\nInjected event scores:")
    for i in injected_idxs:
        er = result.event_results[i]
        e  = attacked.events[i]
        print(f"  [{i}] {e.action_type.value:<15} score={er.risk_score:.3f} "
              f"decision={er.decision.value:<10} flags={er.risk_flags[:2]}")

    return {
        "session_id":   session.session_id,
        "persona":      session.persona.value,
        "scenario":     scenario,
        "detector":     detector_name,
        "caught":       caught,
        "session_risk": result.session_risk_score,
    }


def batch_run(session_dir: str, detector_name: str = "platform_+seq") -> None:
    """Inject every scenario into every real session and print a summary table."""
    paths = list(Path(session_dir).glob("*.json"))
    if not paths:
        print(f"No sessions found in {session_dir}")
        return

    non_c_scenarios = [s for s in ALL_SCENARIOS if not s.startswith("C")]
    results = []

    for path in paths:
        session = Session.load(str(path))
        for scenario in non_c_scenarios:
            try:
                r = run_one(str(path), scenario, detector_name)
                results.append(r)
            except Exception as ex:
                results.append({
                    "session_id": session.session_id,
                    "persona": session.persona.value,
                    "scenario": scenario,
                    "detector": detector_name,
                    "caught": None,
                    "error": str(ex),
                })

    print(f"\n\n{'='*70}")
    print(f"BATCH RESULTS — detector: {detector_name}")
    print(f"{'='*70}")
    scenarios = sorted({r["scenario"] for r in results})
    print(f"{'Persona':<14} " + "  ".join(f"{s:>4}" for s in scenarios))
    print("-" * 70)

    by_session = {}
    for r in results:
        key = (r["persona"], r["session_id"][:8])
        by_session.setdefault(key, {})[r["scenario"]] = r.get("caught")

    for (persona, sid), scen_map in sorted(by_session.items()):
        row = f"{persona+'_'+sid:<14} "
        row += "  ".join(
            f"{'✓':>4}" if scen_map.get(s) else
            f"{'✗':>4}" if scen_map.get(s) is False else
            f"{'—':>4}"
            for s in scenarios
        )
        print(row)

    caught_total   = sum(1 for r in results if r.get("caught") is True)
    eligible_total = sum(1 for r in results if r.get("caught") is not None)
    if eligible_total:
        print(f"\nOverall catch rate: {caught_total}/{eligible_total} "
              f"= {100*caught_total/eligible_total:.0f}%")


def main():
    parser = argparse.ArgumentParser(description="Inject attacks into real captured sessions")
    parser.add_argument("--session",  help="Path to a single .json session file")
    parser.add_argument("--scenario", help="Attack scenario ID (e.g. A1, B3, D1)")
    parser.add_argument("--detector", default="platform_+seq",
                        choices=list(PIPELINES.keys()),
                        help="Detector pipeline to run")
    parser.add_argument("--batch",    action="store_true",
                        help="Inject all scenarios into all sessions in --dir")
    parser.add_argument("--dir",      default="benchmark/real_sessions",
                        help="Directory for --batch mode")
    args = parser.parse_args()

    if args.batch:
        batch_run(args.dir, args.detector)
    elif args.session and args.scenario:
        run_one(args.session, args.scenario, args.detector)
    else:
        print("Specify (--session + --scenario) or --batch")
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
