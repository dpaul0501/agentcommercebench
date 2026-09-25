"""
Scan orchestration.

For each probe: build a clean baseline, mutate it, replay it against the target, and
record both what the target did and what our detectors would have said.

A probe counts as landed only if an action the probe actually *marked as an attack*
executed. A target that happens to block an unrelated action in the session is not
credited with defending against this probe.
"""
from __future__ import annotations

from typing import Iterable, Optional

from ..detectors import Context, Pipeline, default_pipeline
from ..probes import Probe, all_probes
from ..schema import Decision, Outcome, Persona, Session
from ..targets.base import Target, baseline_for, baseline_session
from .report import Finding, Report


def _run_probe(
    probe: Probe,
    target: Target,
    baseline: Session,
    pipeline: Pipeline,
    norms: dict,
    seed: int,
) -> Finding:
    attacked = probe.apply(baseline, seed=seed)

    finding = Finding(
        probe_id=probe.id,
        family=probe.family,
        title=probe.title,
        description=probe.description,
        expected_layer=probe.expected_layer,
        outcome=Outcome.SKIPPED,
    )

    if hasattr(target, "reset"):
        target.reset(attacked)

    ctx = Context(session=attacked, baseline=norms)
    worst_detector = Decision.ALLOW
    top_risk = 0.0
    flags: list[str] = []

    landed = False
    defended = False
    errors = 0
    skips = 0

    for index, action in enumerate(attacked.actions):
        verdict = pipeline.score(action, ctx)
        ctx.history.append(action)
        if action.idempotency_key:
            ctx.settled_keys.add(action.idempotency_key)

        if action.is_attack:
            top_risk = max(top_risk, verdict.risk_score)
            flags.extend(verdict.flags)
            if verdict.decision is Decision.BLOCK:
                worst_detector = Decision.BLOCK
            elif verdict.decision is Decision.ESCALATE and worst_detector is Decision.ALLOW:
                worst_detector = Decision.ESCALATE

        observation = target.execute(action)

        if not action.is_attack:
            continue

        finding.attack_actions += 1

        if observation.outcome is Outcome.VULNERABLE:
            landed = True
            finding.landed_actions += 1
            if finding.landed_at is None:
                finding.landed_at = index
                finding.target_decision = observation.decision
                finding.target_reason = observation.reason
        elif observation.outcome is Outcome.DEFENDED:
            defended = True
            if finding.target_decision is None:
                finding.target_decision = observation.decision
                finding.target_reason = observation.reason
        elif observation.outcome is Outcome.ERROR:
            errors += 1
            finding.target_reason = finding.target_reason or observation.reason
        else:
            skips += 1

    finding.detector_decision = worst_detector
    finding.detector_risk = top_risk
    finding.detector_flags = list(dict.fromkeys(flags))

    if landed:
        finding.outcome = Outcome.VULNERABLE
    elif defended:
        finding.outcome = Outcome.DEFENDED
    elif errors:
        finding.outcome = Outcome.ERROR
    else:
        finding.outcome = Outcome.SKIPPED

    return finding


def scan(
    target: Target,
    *,
    probes: Optional[Iterable[Probe]] = None,
    families: Optional[Iterable[str]] = None,
    persona: Persona = Persona.PROCUREMENT,
    agent_id: str = "agent-under-test",
    pipeline: Optional[Pipeline] = None,
    seed: int = 42,
) -> Report:
    """
    Run the probe suite against a target.

        from acbguard import scan
        from acbguard.targets import CallableTarget

        report = scan(CallableTarget(my_agent))
        print(report.render())
    """
    selected = list(probes) if probes is not None else all_probes(families=families)
    pipeline = pipeline or default_pipeline()

    baseline = baseline_session(persona=persona, agent_id=agent_id, seed=seed)
    norms = baseline_for(baseline)

    # Targets that score internally must judge against the same norms we do, or the
    # comparison is against a strawman rather than a like-for-like configuration.
    if hasattr(target, "set_baseline"):
        target.set_baseline(norms)

    report = Report(target_name=getattr(target, "name", type(target).__name__))
    for probe in selected:
        report.findings.append(
            _run_probe(probe, target, baseline, pipeline, norms, seed)
        )
    return report
