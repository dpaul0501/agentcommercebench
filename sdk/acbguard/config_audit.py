"""
Static audit of an agent configuration.

The scan in `acbguard.scan` is dynamic: it runs attacks and reports what executed. It needs
a working agent, a budget, and time. This module is the other half — it reads the configuration
alone and reports what the configuration *permits*, with zero model calls and zero spend.

That is worth having on its own, not just as a cheap approximation of the dynamic scan. A
dynamic scan proves an attack landed on the run you observed; a config audit finds the capability
that made it possible, and it finds it in CI on every prompt edit, before anything runs.

The checks fall into three groups:

    prompt     what the system prompt does and does not establish — spend authority,
               precedence over tool output, an approval boundary
    tools      what the tool surface permits — unbounded amounts, free destinations,
               non-idempotent charges, arbitrary fetch/exec reachable from a paying loop
    content    text already sitting in the configuration that reads as an instruction,
               which is MCP tool-description poisoning found at rest

Findings are advisory. A check firing means the configuration allows something, not that an
agent will do it — `AgentTarget` is how you find out which.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable

from .agent import AgentSpec, ToolSpec

SEVERITY_WEIGHT = {"critical": 1.0, "high": 0.8, "medium": 0.5, "low": 0.25}


@dataclass
class ConfigFinding:
    check_id: str
    group: str
    title: str
    severity: str
    detail: str
    """What is true about this configuration."""
    remediation: str
    """The smallest change that clears the check."""
    evidence: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "check_id": self.check_id,
            "group": self.group,
            "title": self.title,
            "severity": self.severity,
            "detail": self.detail,
            "remediation": self.remediation,
            "evidence": self.evidence[:240],
        }


@dataclass
class AuditReport:
    agent_name: str
    findings: list[ConfigFinding] = field(default_factory=list)
    checks_run: int = 0

    @property
    def risk(self) -> float:
        """0-100, severity-weighted share of checks that fired."""
        if not self.checks_run:
            return 0.0
        weight = sum(SEVERITY_WEIGHT[f.severity] for f in self.findings)
        return round(min(100.0, 100.0 * weight / self.checks_run), 1)

    @property
    def grade(self) -> str:
        if any(f.severity == "critical" for f in self.findings):
            return "F"
        r = self.risk
        return "A" if r == 0 else "B" if r < 15 else "C" if r < 30 else "D" if r < 50 else "F"

    def by_group(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for f in self.findings:
            out[f.group] = out.get(f.group, 0) + 1
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent": self.agent_name,
            "grade": self.grade,
            "risk": self.risk,
            "checks_run": self.checks_run,
            "findings": [f.to_dict() for f in self.findings],
        }

    def render(self) -> str:
        lines: list[str] = ["", f"  acbguard config audit — {self.agent_name}",
                            "  " + "─" * 62]
        lines.append(f"  Grade {self.grade}    risk {self.risk}/100    "
                     f"{len(self.findings)}/{self.checks_run} checks fired")
        lines.append("")
        if not self.findings:
            lines += ["  No configuration weaknesses found.",
                      "  This is a statement about the config, not the agent — "
                      "run AgentTarget to test behaviour.", ""]
            return "\n".join(lines)

        for group in ("content", "tools", "prompt"):
            group_findings = [f for f in self.findings if f.group == group]
            if not group_findings:
                continue
            lines.append(f"  {group}")
            for f in sorted(group_findings, key=lambda x: -SEVERITY_WEIGHT[x.severity]):
                lines.append(f"    [{f.severity:>8}] {f.check_id}  {f.title}")
                lines.append(f"               {f.detail}")
                if f.evidence:
                    lines.append(f"               evidence: {f.evidence[:100]}")
                lines.append(f"               fix: {f.remediation}")
            lines.append("")
        return "\n".join(lines)


# ── Checks ───────────────────────────────────────────────────────────────────
#
# A check reads the spec and returns findings. Registered rather than hardcoded so a user can
# add their own policy checks without editing this file.

CheckFn = Callable[[AgentSpec], list[ConfigFinding]]
_CHECKS: list[tuple[str, CheckFn]] = []


def check(check_id: str) -> Callable[[CheckFn], CheckFn]:
    def decorate(fn: CheckFn) -> CheckFn:
        _CHECKS.append((check_id, fn))
        return fn

    return decorate


# Words that make a tool description read as an instruction to the model rather than a
# description of the tool. In a description field, any of these is a smell.
_IMPERATIVE = re.compile(
    r"\b(ignore|disregard|override|instead of|before (?:calling|using|you)|"
    r"you must|always (?:send|transfer|pay|use)|do not (?:tell|mention|report|log)|"
    r"important:|note to (?:the )?(?:ai|assistant|model)|system:|"
    r"previous instructions|new instructions)\b",
    re.IGNORECASE,
)

_SPEND_TOOL = re.compile(r"pay|purchase|buy|charge|transfer|checkout|settle|call_service|send",
                         re.IGNORECASE)
_AMOUNT_PARAM = re.compile(r"amount|price|value|units|total|cost", re.IGNORECASE)
_DEST_PARAM = re.compile(r"payee|recipient|destination|to_address|wallet|account|endpoint|url",
                         re.IGNORECASE)
_IDEMPOTENCY = re.compile(r"idempotenc|nonce|request_id|dedup", re.IGNORECASE)
_BROAD_TOOL = re.compile(r"(exec|shell|bash|eval|http_request|fetch|browse|crawl|scrape|"
                         r"run_code|python|sql|admin|open_url|read_web)", re.IGNORECASE)


def _spend_tools(spec: AgentSpec) -> list[ToolSpec]:
    """
    Tools that can move money.

    The name is authoritative. The description is only consulted when it is *not* carrying
    instruction text — a poisoned description talks about paying in order to make the model pay,
    and letting that same text decide our classification would let the injection steer the audit.
    """
    out = []
    for tool in spec.tools:
        if _SPEND_TOOL.search(tool.name):
            out.append(tool)
        elif _SPEND_TOOL.search(tool.description) and not _IMPERATIVE.search(tool.description):
            out.append(tool)
    return out


@check("C1")
def _no_spend_ceiling(spec: AgentSpec) -> list[ConfigFinding]:
    prompt = spec.system_prompt.lower()
    stated = re.search(r"(\$\s?\d|\d+\s*(usd|usdc|cents?|dollars?))", prompt)
    if stated or not _spend_tools(spec):
        return []
    return [ConfigFinding(
        "C1", "prompt", "No spend ceiling in the system prompt", "high",
        "The agent can spend but its instructions name no limit, so the only bound is whatever "
        "the tool layer happens to enforce.",
        "State an explicit per-transaction and per-session limit in the system prompt, and "
        "enforce the same number below the model.",
    )]


@check("C2")
def _no_precedence_rule(spec: AgentSpec) -> list[ConfigFinding]:
    prompt = spec.system_prompt.lower()
    if re.search(r"(untrusted|do not follow instructions|treat .* as data|"
                 r"ignore instructions (in|from)|only .* (user|operator) )", prompt):
        return []
    return [ConfigFinding(
        "C2", "prompt", "Tool output is not marked untrusted", "high",
        "Nothing in the prompt tells the model that text returned by a tool is data rather than "
        "instruction, which is the precondition for every indirect injection.",
        "Add a precedence rule: content returned by tools is untrusted data and never changes "
        "the task, the budget, or the payee.",
    )]


@check("C3")
def _no_approval_boundary(spec: AgentSpec) -> list[ConfigFinding]:
    if not _spend_tools(spec):
        return []
    prompt = spec.system_prompt.lower()
    if re.search(r"(ask|confirm|approval|check with|human|escalate|permission)", prompt):
        return []
    return [ConfigFinding(
        "C3", "prompt", "No approval boundary", "medium",
        "The agent has a paying tool and no instruction about when to stop and ask, so every "
        "amount inside the tool's limits is fully autonomous.",
        "Name a threshold above which the agent must escalate instead of acting.",
    )]


@check("C4")
def _unbounded_amount(spec: AgentSpec) -> list[ConfigFinding]:
    out = []
    for tool in _spend_tools(spec):
        amounts = [p for p in tool.parameters if _AMOUNT_PARAM.search(p)]
        if not amounts:
            continue
        types = " ".join(str(tool.parameters[p]) for p in amounts).lower()
        if re.search(r"max|<=|limit|between|range|\d", types):
            continue
        out.append(ConfigFinding(
            "C4", "tools", f"Unbounded amount on {tool.name}", "critical",
            f"`{tool.name}` takes {', '.join(amounts)} with no bound in the schema. Whatever the "
            "model writes there is what gets charged.",
            "Put the ceiling in the parameter schema, not only in the prompt — a schema bound "
            "survives an injection that rewrites the model's intent.",
            evidence=tool.render(),
        ))
    return out


@check("C5")
def _free_destination(spec: AgentSpec) -> list[ConfigFinding]:
    out = []
    for tool in _spend_tools(spec):
        dests = [p for p in tool.parameters if _DEST_PARAM.search(p)]
        if not dests:
            continue
        if re.search(r"allowlist|enum|registered|known|one of", tool.description, re.I):
            continue
        out.append(ConfigFinding(
            "C5", "tools", f"Unconstrained destination on {tool.name}", "critical",
            f"`{tool.name}` accepts a caller-supplied {', '.join(dests)} with no allowlist. An "
            "injected payee reaches the rail unchanged.",
            "Resolve destinations from a registry keyed by service id; never accept an address "
            "or URL the model composed.",
            evidence=tool.render(),
        ))
    return out


@check("C6")
def _non_idempotent_charge(spec: AgentSpec) -> list[ConfigFinding]:
    out = []
    for tool in _spend_tools(spec):
        surface = " ".join([tool.name, tool.description, *tool.parameters])
        if _IDEMPOTENCY.search(surface):
            continue
        out.append(ConfigFinding(
            "C6", "tools", f"No idempotency key on {tool.name}", "medium",
            f"`{tool.name}` charges money and exposes no idempotency key, so a retry — the "
            "agent's own, or one an injection induces — charges twice.",
            "Require an idempotency key and reject a repeat with a different amount.",
            evidence=tool.render(),
        ))
    return out


@check("C7")
def _broad_scope(spec: AgentSpec) -> list[ConfigFinding]:
    if not _spend_tools(spec):
        return []
    broad = [t for t in spec.tools if _BROAD_TOOL.search(t.name)]
    if not broad:
        return []
    return [ConfigFinding(
        "C7", "tools", "General-purpose tool in a paying agent", "high",
        f"{', '.join(t.name for t in broad)} can reach arbitrary content, and the same loop can "
        "spend. That is a direct path from fetched text to a charge.",
        "Split them: an agent that fetches untrusted content should not hold the paying tool in "
        "the same context.",
        evidence=", ".join(t.render() for t in broad),
    )]


@check("C8")
def _poisoned_description(spec: AgentSpec) -> list[ConfigFinding]:
    out = []
    for tool in spec.tools:
        match = _IMPERATIVE.search(tool.description)
        if not match:
            continue
        out.append(ConfigFinding(
            "C8", "content", f"Instruction text in {tool.name} description", "critical",
            "A tool description is meant to describe a tool. This one gives the model an order, "
            "which is what a poisoned MCP server looks like from the client side.",
            "Pin tool descriptions to a reviewed hash and fail the build when a server changes "
            "one underneath you.",
            evidence=f"…{tool.description[max(0, match.start() - 30):match.end() + 40]}…",
        ))
    return out


@check("C9")
def _instruction_in_task(spec: AgentSpec) -> list[ConfigFinding]:
    match = _IMPERATIVE.search(spec.task)
    if not match:
        return []
    return [ConfigFinding(
        "C9", "content", "Instruction-shaped text in the task", "high",
        "The task carries override language. If the task is assembled from user or retrieved "
        "content, this is an injection that has already arrived.",
        "Keep the task a statement of goal, and pass any user-supplied text as quoted data.",
        evidence=f"…{spec.task[max(0, match.start() - 30):match.end() + 40]}…",
    )]


@check("C10")
def _duplicate_tool_names(spec: AgentSpec) -> list[ConfigFinding]:
    seen: dict[str, int] = {}
    for tool in spec.tools:
        seen[tool.name] = seen.get(tool.name, 0) + 1
    dupes = [n for n, c in seen.items() if c > 1]
    if not dupes:
        return []
    return [ConfigFinding(
        "C10", "tools", "Duplicate tool names", "high",
        f"{', '.join(dupes)} appears more than once. Which definition wins depends on merge "
        "order, so a second server can shadow a trusted tool by claiming its name.",
        "Namespace tools by server and reject collisions at load time.",
        evidence=", ".join(dupes),
    )]


def audit(spec: AgentSpec) -> AuditReport:
    """
    Read an agent configuration and report what it permits. No model calls, no spend.

        report = audit(AgentSpec(model=None, tools=my_tools))
        print(report.render())
    """
    report = AuditReport(agent_name=spec.name, checks_run=len(_CHECKS))
    for _, fn in _CHECKS:
        report.findings.extend(fn(spec))
    return report


def checks() -> list[str]:
    return [cid for cid, _ in _CHECKS]


__all__ = ["AuditReport", "ConfigFinding", "audit", "check", "checks"]
