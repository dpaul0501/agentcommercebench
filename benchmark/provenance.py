"""
Provenance and confounding checks for every constant that affects a benchmark result.

The rule this enforces
----------------------
A detector is blind. It may only get its parameters from two places:

    LEARNED   fitted from training data the detector is allowed to see
    HUMAN     domain knowledge, a published standard, a policy someone set

Anything else is **confounded**: a value that exists because of how the benchmark was built.
A threshold nobody can trace is a cheat even when it happens to work, and a threshold copied
from the generator is a cheat that cannot fail.

The classic shape, from this repo's own history: the generator drew fraudulent amounts from
`uniform(3010, 4500)` and the rule blocked above `3000`, with clean traffic topping out at
`2800`. Detection was perfect and false positives were zero — not because the detector was
good, but because the two populations never overlapped and the threshold sat in the gap.

Two checks
----------
`check_shared_constants`  mechanical. Any literal appearing in both a generator module and a
detector module is reported. Mechanical checks find copied values; they cannot find a value
that was *chosen* with knowledge of the other side, which is why the ledger exists too.

`check_ledger`  every constant declared in PROVENANCE must have a source, a justification and,
where it interacts with a distribution, evidence that the supports overlap. A constant that is
not in the ledger is itself a finding.

    python -m benchmark.provenance
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parent.parent

# The active system. These are the files a published result comes from, and the only ones
# `--strict` judges.
GENERATOR_FILES = ["benchmark/generator.py", "benchmark/config.py"]
DETECTOR_FILES = [
    "sdk/acbguard/detectors/economic.py",
    "sdk/acbguard/detectors/registry.py",
    "sdk/acbguard/detectors/catalog.py",
    "sdk/acbguard/detectors/l0_judge.py",
    "benchmark/evaluate_v2.py",
]

# The retired v1 system, scanned separately. Its findings are the reason v2 exists, so they
# are kept rather than deleted — but they must not make the guard permanently red, or it
# stops being a guard.
V1_GENERATOR_FILES = ["benchmark/synthetic.py"]
V1_DETECTOR_FILES = [
    "benchmark/behavioral_ml.py",
    "benchmark/acp_baselines.py",
    "ach/verifiers/local_rules.py",
    "ach/verifiers/uncertainty.py",
]

# Values so common they carry no information about coupling.
BORING = {0, 1, 2, 3, -1, 10, 100, 1000, 0.0, 0.5, 1.0, 2.0, 100.0}


@dataclass
class Constant:
    """One parameter that can move a benchmark number."""

    name: str
    value: Any
    source: str          # LEARNED | HUMAN | CONFOUNDED | UNTRACED
    where: str
    justification: str
    overlap_evidence: Optional[str] = None
    """For a threshold sitting between two distributions: proof their supports overlap. A
    threshold in a gap cannot be wrong, so it measures nothing."""

    system: str = "v1"
    """v2 = the system results are published from; v1 = the retired one, kept as a record."""

    literals: tuple = ()
    """Numeric values this entry accounts for.

    The mechanical check reports every number appearing on both sides of the
    generator/detector boundary. Most are coincidences of a 0-1 scale, but a coincidence and
    a copied threshold look identical to a scanner, so the resolution is not to guess: a
    shared number passes only if some ledger entry claims it and says where it came from."""

    @property
    def ok(self) -> bool:
        return self.source in {"LEARNED", "HUMAN"}


# ── The ledger ───────────────────────────────────────────────────────────────
#
# Every entry is a claim that can be checked. Where the claim is that a value is confounded,
# that is a finding, not an excuse — the list below is the current state, not the target.

PROVENANCE: list[Constant] = [
    # ── v2: the active system ────────────────────────────────────────────────
    #
    # Every decision constant below is fitted by `benchmark/calibrate.py` on the CLEAN
    # TRAINING split only, to a stated false-positive budget. Recall on the test attacks is
    # not an input to any of them, and none is written down in the generator.

    Constant(
        name="escalate_at", value="fitted (0.70)", source="LEARNED", system="v2",
        where="benchmark/calibrate.py:pipeline_thresholds -> calibration.json",
        justification=(
            "The lowest pipeline score whose clean-session rate lands nearest a 10% review "
            "budget. Fitted on the aggregate score rather than per detector, because seven "
            "detectors each fitted to 10% compound to a 47% review rate — the "
            "multiple-comparisons problem arriving through the back door. The budget is a "
            "property of the decision, so it is fitted on the score the decision uses."),
        overlap_evidence=(
            "FULL. 6.8% of clean training sessions sit at or above it, as do 44% of test "
            "attacks; the two populations share the whole range. Clean traffic reaches the "
            "same maximum score (0.95) that attacks do."),
    ),
    Constant(
        name="block_at", value="fitted (0.95)", source="LEARNED", system="v2",
        where="benchmark/calibrate.py:pipeline_thresholds -> calibration.json",
        justification=(
            "Fitted to a 1% hard-refusal budget, and allowed a bounded overshoot: 1.18% of "
            "clean sessions sit at 0.95, so the strictly-inside rule steps to the next "
            "distinct score, where nothing of any kind lands — surrendering every block in "
            "the system to save 0.18 points of clean traffic. The achieved rate (1.2% clean, "
            "9.9% of attacks) is printed rather than assumed."),
        overlap_evidence=(
            "FULL, and this is the point: clean sessions reach this score. A block threshold "
            "no clean session can reach would be the v1 failure repeated."),
    ),
    Constant(
        name="z_escalate / z_block", value="fitted (1.56 / 2.12)", source="LEARNED",
        system="v2", where="benchmark/calibrate.py:clean_z_scores -> calibration.json",
        justification=(
            "Quantiles of clean amount z-scores against each agent's own fitted log-normal, "
            "so a limit is a multiple of that agent's own typical spend and never an absolute "
            "figure. Fitted per SESSION, not per payment: a session holds about four "
            "payments, so a 1%-of-payments cut leaves 3-5% of sessions above it, and the "
            "session is what gets interrupted."),
        overlap_evidence=(
            "FULL. Clean z reaches 2.60 in training; attacks are drawn from margins that "
            "overlap the legitimate tail by construction."),
    ),
    Constant(
        name="catalogue_tolerance / peer_tolerance", value="fitted (1.44 / 1.40)",
        source="LEARNED", system="v2",
        where="benchmark/calibrate.py:price_ratios -> calibration.json",
        justification=(
            "Session-max price ratios of clean traffic against the published catalogue and "
            "against what other agents paid, cut at the same 10% budget. Hand-setting this "
            "is the $3,000 mistake in another costume: 13.8% of legitimate purchases here "
            "sit above 1.15x listed, because real prices rise."),
        overlap_evidence=(
            "FULL. Clean catalogue ratios run to 2.04 at p99; the overcharging merchants "
            "draw 1.15-1.6x, inside the legitimate tail."),
    ),
    Constant(
        name="uncalibrated fallbacks", value="z 2.5/4.0, decide 0.30/0.70",
        source="HUMAN", system="v2", literals=(2.5, 4.0, 0.3, 0.7),
        where="benchmark/evaluate_v2.py:l1",
        justification=(
            "Used only when no calibration.json exists — the SDK's first run, before an agent "
            "has any history. 2.5 and 4 sigma are the conventional soft-limit and outlier "
            "cuts; 0.30 and 0.70 are the package's default review and refuse points. Every "
            "published number in this repo comes from the calibrated path, where all four are "
            "replaced by values fitted on clean training traffic."),
        overlap_evidence=(
            "Not applicable: these never decide a benchmark result. `--uncalibrated` exists "
            "to show what they cost, which is the point of reporting them."),
    ),
    Constant(
        name="sybil warm-up shape", value="spread <= 4x, payout >= 20x median",
        source="HUMAN", system="v2", literals=(4, 20),
        where="sdk/acbguard/detectors/registry.py",
        justification=(
            "Ratios against the counterparty's own prior payments, not amounts. The previous "
            "form — dust under $0.05, payout over $1.00 — was two absolute figures that are "
            "routine for some agents here and a month of spend for others, and both appeared "
            "verbatim in the generator. A ratio needs no scale and cannot be copied from an "
            "answer key, because the answer key has no ratios in it."),
        overlap_evidence=(
            "FULL. Clean sessions routinely pay the same counterparty several times with a "
            "spread under 4x, so the first condition alone fires on ordinary traffic; only "
            "the 20x step is unusual, and legitimate traffic reaches it."),
    ),
    Constant(
        name="catalog price severity split", value="0.80 below 5x, 0.90 above",
        source="HUMAN", system="v2", literals=(5, 0.8),
        where="sdk/acbguard/detectors/catalog.py",
        justification=(
            "A severity, not a threshold: both branches flag. The split says a 10x overcharge "
            "deserves more attention than a 2x one, which is a judgement about triage order "
            "and cannot change what is detected."),
        overlap_evidence="Not applicable: nothing is compared against 5 to decide detection.",
    ),
    Constant(
        name="rationale truncation", value=60, source="HUMAN", system="v2", literals=(60,),
        where="sdk/acbguard/detectors/l0_judge.py",
        justification=(
            "How many characters of the judge's stated reason are kept in a flag. A display "
            "length. It coincides with the generator's agent count, which is what a scanner "
            "looking only at values cannot tell apart from a threshold — hence this entry."),
        overlap_evidence="Not applicable: affects no decision.",
    ),
    Constant(
        name="L0 probe weights", value="0.55 - 0.95", source="HUMAN", system="v2",
        literals=(0.55, 0.7, 0.8, 0.95, 0.6, 0.65, 0.75, 0.85, 0.9),
        where="sdk/acbguard/detectors/l0_judge.py",
        justification=(
            "Confidence each probe carries when it fires, on the same 0-1 scale every "
            "detector in the package emits. They are outputs: nothing is compared against "
            "them to decide whether a probe matches. Their coincidence with probabilities in "
            "the generator's config is arithmetic on a bounded scale, not a shared constant."),
        overlap_evidence="Not applicable: emitted, never compared against data.",
    ),
    Constant(
        name="L0 probe set", value="PROBES", source="HUMAN", system="v2",
        where="sdk/acbguard/detectors/l0_judge.py",
        justification=(
            "Hand-written from how x402 works, not from the generator: paying before a result "
            "arrives IS the protocol, so payment-as-precondition cannot be a signal. Written "
            "against the train pool and scored on the disjoint test pool, with the reported "
            "false-positive rate measured on a third pool written by a model that was never "
            "shown the probes."),
        overlap_evidence=(
            "0 of 80 held-out legitimate notes fired, which bounds the rate at 3.8% (95%, "
            "rule of three) and does NOT establish zero. The in-benchmark clean rate is "
            "in-sample and is reported as such."),
    ),

    # ── v1: the retired system, kept as the record of why v2 exists ──────────
    Constant(
        name="per_transaction_limit",
        value=3000,
        source="CONFOUNDED",
        where="benchmark/synthetic.py:110 (wallet) / ach/verifiers/local_rules.py:59 (rule)",
        justification=(
            "Nothing derives this. It is a round number chosen when the generator was "
            "written, and the B8 attack builder then draws uniform(3010, 4500) — immediately "
            "above it. Clean traffic tops out at 2800 (_AMOUNT_RANGES_HIGHVAL['travel']), so "
            "the threshold sits in an empty gap between the two populations."
        ),
        overlap_evidence=(
            "NONE. clean max 2800 < 3010 min attack. The supports are disjoint, so no clean "
            "session can trigger the rule and no attack session can evade it."
        ),
    ),
    Constant(
        name="velocity_block_threshold",
        value=8,
        source="CONFOUNDED",
        where="ach/verifiers/local_rules.py:22 vs benchmark/synthetic.py:220",
        justification=(
            "The rule blocks at 8 commits; the B2 builder draws randint(9, 15). The threshold "
            "is one below the attack's minimum. No clean session has more than 1 commit."
        ),
        overlap_evidence="NONE. clean commit count is a point mass at 1; attacks are 9-15.",
    ),
    Constant(
        name="injection_keywords",
        value="_INJECTION_KEYWORDS",
        source="CONFOUNDED",
        where="benchmark/behavioral_ml.py:44 vs benchmark/synthetic.py:294",
        justification=(
            "The detector's keyword list is a superset of the substrings of the four strings "
            "the generator injects. 40/40 generated payloads match. The test set was written "
            "against the answer key."
        ),
        overlap_evidence="NONE. No clean session contains any keyword.",
    ),
    Constant(
        name="mcc_allowlist",
        value="_MCC_OK",
        source="CONFOUNDED",
        where="benchmark/synthetic.py:40 (both wallet policy and attack construction)",
        justification=(
            "The generator builds B7 attacks by drawing from _MCC_BAD_OPTIONS, constructed to "
            "be disjoint from _MCC_OK, and the wallet policy is _MCC_OK. Same object on both "
            "sides. As a policy conformance check this is legitimate; as a detection result "
            "it is circular."
        ),
        overlap_evidence="NONE, by construction of _MCC_BAD_OPTIONS.",
    ),
    Constant(
        name="threshold_sigmas",
        value=1.1,
        source="UNTRACED",
        where="benchmark/behavioral_ml.py:125",
        justification=(
            "Fitted as mu + k*sigma on clean training scores, which is the right shape — but k "
            "is 1.1 while the module docstring says 1.5 and predicts a 6-10% false-positive "
            "rate. The measured rate is 56%. Whatever produced 1.1, it is not what is "
            "documented, so it cannot be audited."
        ),
    ),
    Constant(
        name="clean_amount_ranges",
        value="_AMOUNT_RANGES",
        source="HUMAN",
        where="benchmark/synthetic.py:79",
        justification=(
            "Per-persona spend ranges chosen as plausible for the personas. Legitimate as a "
            "modelling choice; the problem is not this constant but that attack builders draw "
            "from different ranges, making the attack detectable by amount alone."
        ),
        overlap_evidence=(
            "PARTIAL. Attacks unrelated to money (B4, B5, B7s) draw from ranges whose means "
            "sit well above the clean per-persona means, so they are separable by amount."
        ),
    ),
    Constant(
        name="training_seed",
        value=7,
        source="HUMAN",
        where="benchmark/behavioral_ml.py:133",
        justification=(
            "Deliberately different from the evaluation seed (42) so training and test draws "
            "are independent. Verified: n_per_attack=0, so norms are fitted on clean data "
            "only. This one is correct."
        ),
    ),
    Constant(
        name="zero_variance_fallback",
        value=1.0,
        source="CONFOUNDED",
        where="benchmark/behavioral_ml.py:168",
        justification=(
            "When a feature has zero variance in training the code substitutes sigma=1.0. "
            "Clean commit count is constant at 1, so this manufactures z=(9-1)/1.0=8 for a "
            "velocity attack. The detector appears to have learned a distribution that does "
            "not exist. A zero-variance feature carries no information and must be dropped."
        ),
    ),
]


# ── Mechanical check ─────────────────────────────────────────────────────────

def severity_lines(tree: ast.AST) -> set[int]:
    """
    Lines where a number is a risk score the detector EMITS, not a threshold it compares to.

    `max(risk, 0.75)` and `return 0.70, [...]` are outputs. A copied threshold is dangerous
    because it is compared against generated data; a severity is never compared against
    anything, so a severity that happens to equal a number in the generator is a coincidence
    of the 0-1 scale, not a leak. Distinguishing them mechanically is the point — an
    allowlist of "numbers we decided are fine" would be exactly the kind of unauditable
    judgement this module exists to catch.
    """
    lines: set[int] = set()
    for node in ast.walk(tree):
        # max(risk, X) / min(risk, X)
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id in {"max", "min"} and len(node.args) == 2
                and isinstance(node.args[0], ast.Name) and node.args[0].id == "risk"
                and isinstance(node.args[1], ast.Constant)):
            lines.add(node.args[1].lineno)
        # return <score>, [flags...]
        if (isinstance(node, ast.Return) and isinstance(node.value, ast.Tuple)
                and node.value.elts and isinstance(node.value.elts[0], ast.Constant)):
            lines.add(node.value.elts[0].lineno)
    return lines


def literals(path: Path, skip_severities: bool = False) -> dict[Any, list[int]]:
    """Every numeric and string literal in a module, with the lines it appears on."""
    try:
        tree = ast.parse(path.read_text())
    except (OSError, SyntaxError):
        return {}
    skip = severity_lines(tree) if skip_severities else set()
    found: dict[Any, list[int]] = {}
    for node in ast.walk(tree):
        if node.lineno in skip if hasattr(node, "lineno") else False:
            continue
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float, str)):
            value = node.value
            if isinstance(value, str) and (len(value) < 4 or len(value) > 80):
                continue
            if isinstance(value, (int, float)) and value in BORING:
                continue
            found.setdefault(value, []).append(node.lineno)
    return found


# Field names, file paths and CLI flags. A detector reads the same fields the generator
# writes — that is the schema, not coupling, and a benchmark where the two disagreed would
# simply be broken. Only numbers can encode a threshold copied across the boundary.
SCHEMA_LIKE = str


def literal_kind(value: Any) -> str:
    """`schema` for a shared name or path, `numeric` for a shared number."""
    return "schema" if isinstance(value, SCHEMA_LIKE) else "numeric"


def check_shared_constants(root: Path = ROOT,
                           generator_files: Optional[list[str]] = None,
                           detector_files: Optional[list[str]] = None) -> list[dict[str, Any]]:
    """Literals appearing on both sides of the generator/detector boundary."""
    gen: dict[Any, list[str]] = {}
    generator_files = generator_files or GENERATOR_FILES
    detector_files = detector_files or DETECTOR_FILES
    for name in generator_files:
        for value, lines in literals(root / name).items():
            gen.setdefault(value, []).extend(f"{name}:{n}" for n in lines)

    findings: list[dict[str, Any]] = []
    for name in detector_files:
        path = root / name
        if not path.exists():
            continue
        for value, lines in literals(path, skip_severities=True).items():
            if value in gen:
                findings.append({
                    "value": value,
                    "kind": literal_kind(value),
                    "generator": sorted(set(gen[value]))[:4],
                    "detector": [f"{name}:{n}" for n in lines][:4],
                })
    return sorted(findings, key=lambda f: str(f["value"]))


def declared_literals() -> set:
    """Every numeric value some v2 ledger entry accounts for."""
    out: set = set()
    for c in PROVENANCE:
        if c.system == "v2":
            out.update(c.literals)
    return out


def undeclared_shared_numbers() -> list[dict[str, Any]]:
    """Shared numbers no ledger entry claims. These are the ones that fail the build."""
    declared = declared_literals()
    return [f for f in check_shared_constants()
            if f["kind"] == "numeric" and f["value"] not in declared]


def check_ledger(system: Optional[str] = None) -> list[Constant]:
    """Constants whose provenance is not LEARNED or HUMAN."""
    return [c for c in PROVENANCE
            if not c.ok and (system is None or c.system == system)]


def render() -> str:
    out: list[str] = ["", "  Parameter provenance", "  " + "─" * 70]
    for group in ("CONFOUNDED", "UNTRACED", "HUMAN", "LEARNED"):
        rows = [c for c in PROVENANCE if c.source == group]
        if not rows:
            continue
        out.append(f"\n  {group}  ({len(rows)})")
        for c in rows:
            out.append(f"    {c.name} = {c.value}")
            out.append(f"      {c.where}")
            out.append(f"      {c.justification}")
            if c.overlap_evidence:
                out.append(f"      support overlap: {c.overlap_evidence}")

    shared = check_shared_constants()
    numeric = [f for f in shared if f["kind"] == "numeric"]
    schema = [f for f in shared if f["kind"] == "schema"]
    out += ["", "  Numbers shared across the generator/detector boundary  (v2, active)",
            "  " + "─" * 70,
            "  Field names and paths are excluded: a detector must read the fields the",
            "  generator writes. Only a shared NUMBER can be a threshold copied from the",
            f"  answer key. ({len(schema)} shared names omitted.)", ""]
    undeclared = undeclared_shared_numbers()
    if not undeclared:
        out.append(f"    none undeclared  ({len(numeric)} shared, all accounted for in the "
                   f"ledger)")
    for f in undeclared:
        out.append(f"    {f['value']!r}  NOT DECLARED")
        out.append(f"      generator: {', '.join(f['generator'])}")
        out.append(f"      detector:  {', '.join(f['detector'])}")

    v1_shared = check_shared_constants(generator_files=V1_GENERATOR_FILES,
                                       detector_files=V1_DETECTOR_FILES)
    bad_v2 = check_ledger("v2")
    bad_v1 = check_ledger("v1")
    out += ["", "  " + "─" * 70,
            f"  v2 (active):  {len(bad_v2)} of "
            f"{len([c for c in PROVENANCE if c.system == 'v2'])} constants untraceable; "
            f"{len(numeric)} number(s) shared across the boundary, "
            f"{len(undeclared)} of them undeclared.",
            f"  v1 (retired): {len(bad_v1)} confounded constants, "
            f"{len([f for f in v1_shared if f['kind'] == 'numeric'])} shared numbers — kept "
            f"as the record of why v2 exists, not counted against the build.", ""]
    return "\n".join(out)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--json", action="store_true")
    p.add_argument("--strict", action="store_true",
                   help="exit 1 if anything is confounded or untraced (for CI)")
    args = p.parse_args(argv)

    if args.json:
        print(json.dumps({
            "constants": [vars(c) for c in PROVENANCE],
            "shared_literals": check_shared_constants(),
        }, indent=2, default=str))
    else:
        print(render())

    # Only the active system gates the build. The v1 entries are a record of a benchmark
    # that is no longer used to produce results; leaving them to fail strict forever would
    # make the guard permanently red, which is the same as having no guard.
    if args.strict and (check_ledger("v2") or undeclared_shared_numbers()):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
