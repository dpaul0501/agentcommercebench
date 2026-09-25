"""
D8 — the reasoning layer.

Every other detector reads the wire: what the agent *did*. This one reads what the agent
*thought*, which is only available when the integration can see the model — an SDK hook
wrapping the model client, or an observability feed. MCP and rail integrations never carry
it, and this detector correctly scores 0 when reasoning is absent.

An important constraint on how it can be used: observability platforms are post-hoc sinks
with a 1-30s floor, so a verdict from this detector arrives *after* the action it describes.
It cannot gate a payment. It can score, train, and justify ending a session or revoking a
credential before the next action.

What it catches that the wire cannot
------------------------------------
Three of these four are invisible at L1 *in principle*, not just in practice:

  D8a  injected instruction, with provenance — the wire sees text; only here can you tell
       whether it arrived from the user (allowed to instruct) or from a tool result
       (never allowed to instruct)
  D8b  evasion planning — "I'll split this to stay under the limit" produces actions that
       are each individually perfectly legal
  D8c  intent-action mismatch — the agent says one thing and does another; needs both
       layers at once, so no single-layer detector can see it
  D8d  concealment — reasoning about hiding, obscuring, or not mentioning
"""
from __future__ import annotations

import re
from typing import Optional

from ..schema import ActionType, Action
from .base import Context

_PAY = (ActionType.AUTHORIZE, ActionType.A2A_TRANSFER)


# ── D8a: injected instruction ────────────────────────────────────────────────
# Same vocabulary as the payload detector, but the score depends on provenance.

INSTRUCTION_PATTERNS = [
    (r"ignore\s+(all\s+)?(previous|prior|earlier)\s+instruction", "ignore_previous"),
    (r"disregard\s+(your|the|all)\s+", "disregard"),
    (r"you\s+are\s+now\s+(a|an|the)\b", "role_override"),
    (r"forget\s+(your|all|everything|the\s+above)", "forget"),
    (r"new\s+(instruction|directive|task)s?\s*:", "new_instructions"),
    (r"\[SYSTEM\]|<system>", "system_tag"),
    (r"do\s+not\s+(tell|inform|mention|reveal)", "suppress_disclosure"),
]

# ── D8b: evasion planning ────────────────────────────────────────────────────
# The agent reasoning about how to get around a control. The resulting actions
# are individually compliant, which is exactly why the wire cannot see this.

EVASION_PATTERNS = [
    (r"(split|divide|break)\s+(it|this|the\s+\w+)?\s*(up\s+)?into\s+(smaller|multiple|several)", 0.90, "splitting_to_evade"),
    (r"(stay|keep|remain)\s+(just\s+)?(under|below|beneath)\s+(the\s+)?(limit|threshold|cap|ceiling)", 0.95, "staying_under_limit"),
    (r"(avoid|bypass|circumvent|get\s+around|work\s+around)\s+(the\s+)?(approval|limit|check|review|policy|threshold)", 0.95, "bypass_control"),
    (r"without\s+(triggering|tripping|alerting|raising)", 0.90, "avoid_trigger"),
    (r"(spread|spac(e|ing))\s+(them\s+|it\s+|these\s+)?out\s+(over|across)\s+"
     r"(\w+\s+){0,2}(time|days?|hours?|weeks?|sessions?)", 0.75, "temporal_spreading"),
    (r"(smaller|separate)\s+(transaction|payment|charge)s?\s+(so|to)\b", 0.85, "structuring"),
    (r"(won't|will\s+not|shouldn't)\s+(be\s+)?(flagged|noticed|detected|caught)", 0.90, "evade_detection"),
    (r"below\s+the\s+\$?[\d,]+\s+(limit|threshold)", 0.90, "explicit_threshold_evasion"),
]

# ── D8d: concealment ─────────────────────────────────────────────────────────

CONCEALMENT_PATTERNS = [
    (r"(hide|conceal|obscure|mask)\s+(this|the|it|my)", 0.85, "concealment"),
    (r"(don't|do\s+not|shouldn't)\s+(log|record|report|mention)", 0.85, "avoid_logging"),
    (r"(secret|covert|quietly|discreetly)\b", 0.55, "covert_language"),
    (r"my\s+(real|actual|true)\s+(goal|objective|purpose|intent)", 0.90, "hidden_objective"),
]


def _compile(patterns):
    return [(re.compile(p, re.IGNORECASE), *rest) for p, *rest in patterns]


_INSTRUCTION = _compile(INSTRUCTION_PATTERNS)
_EVASION = _compile(EVASION_PATTERNS)
_CONCEALMENT = _compile(CONCEALMENT_PATTERNS)

# Words that make a stated intent comparable to an action. Inflected forms are listed
# explicitly: "Buying one search" is a purchase, and matching only the bare stem would read it
# as a read-only intent because of the noun.
_INTENT_VERBS = re.compile(
    r"\b(search(?:es|ing)?|look(?:s|ed|ing)?\s+up|find(?:s|ing)?|check(?:s|ed|ing)?|"
    r"read(?:s|ing)?|fetch(?:es|ed|ing)?|brows(?:e|es|ed|ing)|compar(?:e|es|ed|ing)|"
    r"buy(?:s|ing)?|bought|purchas(?:e|es|ed|ing)|pay(?:s|ing)?|paid|order(?:s|ed|ing)?|"
    r"book(?:s|ed|ing)?|subscrib(?:e|es|ed|ing)|transfer(?:s|red|ring)?|"
    r"send(?:s|ing)?|sent|spend(?:s|ing)?|spent|charg(?:e|es|ed|ing))\b",
    re.IGNORECASE,
)
_BUY_VERBS = {
    "buy", "buys", "buying", "bought",
    "purchase", "purchases", "purchased", "purchasing",
    "pay", "pays", "paying", "paid",
    "order", "orders", "ordered", "ordering",
    "book", "books", "booked", "booking",
    "subscribe", "subscribes", "subscribed", "subscribing",
    "transfer", "transfers", "transferred", "transferring",
    "send", "sends", "sending", "sent",
    "spend", "spends", "spending", "spent",
    "charge", "charges", "charged", "charging",
}
# Language showing the agent knows it is about to spend, even without a purchase verb.
# Deliberately excludes bare "price"/"cost": researching a price is the read-only intent this
# check exists to catch, so treating those words as evidence of spend awareness would suppress
# the true positives.
_PURCHASE_CONTEXT = re.compile(
    r"\b(fee|fees|invoice|checkout|wallet|balance|budget|usdc|"
    r"settle|settlement|charge\s+me|\$\s?\d)", re.IGNORECASE
)
_AMOUNT_RE = re.compile(r"\$\s?([\d,]+(?:\.\d{1,2})?)")

# Context that makes a dollar figure something other than the intended purchase — a wallet
# balance, a remaining budget, a per-transaction cap.
_NOT_A_PURCHASE_AMOUNT = re.compile(
    r"\b(balance|available|remaining|budget|limit|cap|ceiling|allowance|so far|"
    r"spent|total of|per[- ]transaction|daily|"
    # Unit conversions. The tool schema itself reads "micro-units where 1000000 = $1.00",
    # and agents echo it back; without this the explanation of the unit becomes the
    # stated purchase amount and every ordinary session flags a mismatch.
    r"micro-?units?|units?\s+where|equals?|equivalent|conversion|denominated|"
    r"1[,_]?000[,_]?000)\b",
    re.IGNORECASE)


def _purchase_amount(text: str) -> Optional[str]:
    """
    The dollar figure the reasoning presents as the purchase, if any.

    Figures introduced as a balance or a cap are skipped: an agent that says "I have $100
    available" and then spends $0.05 has not contradicted itself, and treating the first
    figure as the intent flagged ordinary sessions at a 0.0x ratio.
    """
    for match in _AMOUNT_RE.finditer(text or ""):
        # Both sides: the qualifier can lead ("my balance is $100") or trail
        # ("I have $100 available"), and checking only one side missed half of them.
        window = (text[max(0, match.start() - 60):match.start()]
                  + " " + text[match.end():match.end() + 30])
        if _NOT_A_PURCHASE_AMOUNT.search(window):
            continue
        return match.group(1)
    return None


# An obligation the counterparty asserts that is not simply the price of the thing being
# bought. This is what separates an injected demand from a 402 challenge.
_EXTRA_OBLIGATION = re.compile(
    r"\b(verification|verify|validation|activation|unlock|release|fees?|"
    r"handling|deposit|prepayment|pre-?authoriz\w+|"
    r"additional\s+(?:payment|fee|charge|transfer)|extra\s+(?:payment|fee|charge)|"
    r"security\s+(?:deposit|fee)|before\s+(?:i|we|you)\s+can|first\s+send|"
    r"send\s+.{0,20}\s+first|transfer\s+.{0,30}\s+before|remaining\s+balance|"
    r"0x[a-fA-F0-9]{4,})\b",
    re.IGNORECASE,
)

# Reasoning that sources its justification from the counterparty rather than the task.
_DEFERENCE = re.compile(
    r"("
    r"(?:the\s+)?(?:service|site|website|merchant|vendor|seller|tool|api|server|page|"
    r"response|result|message|endpoint|provider)\s+"
    r"(?:says?|said|states?|requires?|requests?|asks?|instructs?|indicates?|told|wants?|"
    r"needs?|demands?)"
    r"|as\s+(?:instructed|requested|directed|required|specified)\s+by"
    r"|according\s+to\s+the\s+(?:response|result|service|site|merchant|tool|api|page)"
    r"|(?:i\s+was|we\s+were)\s+(?:told|instructed|asked|directed)"
    r"|it\s+says\s+(?:i|we)\s+(?:must|need|should|have\s+to)"
    r"|(?:in\s+order\s+)?to\s+(?:comply|satisfy)\s+with\s+(?:the|their)"
    r")",
    re.IGNORECASE,
)


class ReasoningDetector:
    """
    D8. Scores 0.0 when no reasoning is present, so it is safe to include in any
    pipeline — it simply contributes nothing on integrations that cannot see the model.
    """

    name = "reasoning"

    def __init__(self, intent_mismatch_ratio: float = 3.0):
        self.intent_mismatch_ratio = intent_mismatch_ratio
        """Stated amount must differ from actual by this factor to flag a mismatch."""

    def score(self, action: Action, ctx: Context) -> tuple[float, list[str]]:
        if not action.has_reasoning:
            return 0.0, []

        text = " ".join(filter(None, [action.reasoning, action.stated_intent]))
        risk = 0.0
        flags: list[str] = []

        risk, flags = self._injected(action, text, risk, flags)
        risk, flags = self._deference(action, text, risk, flags)
        risk, flags = self._evasion(text, risk, flags)
        risk, flags = self._concealment(text, risk, flags)
        risk, flags = self._intent_mismatch(action, ctx, risk, flags)

        return risk, flags

    # D8a — an instruction is only suspicious depending on where it came from.
    def _injected(self, action: Action, text: str, risk: float, flags: list[str]):
        hits = [name for pattern, name in _INSTRUCTION if pattern.search(text)]
        if not hits:
            return risk, flags

        sources = set(action.context_sources)
        tool_sourced = any(s.startswith("tool:") for s in sources)
        user_sourced = "user" in sources

        if tool_sourced:
            # Content returned by a tool is data. It is never entitled to instruct.
            risk = max(risk, 0.95)
            flags.append("injected_via_tool_result")
        elif user_sourced and not sources - {"user", "system"}:
            # The user may legitimately instruct their own agent.
            risk = max(risk, 0.35)
            flags.append("instruction_from_user")
        else:
            risk = max(risk, 0.70)
            flags.append("instruction_unknown_provenance")

        flags.extend(f"instr:{h}" for h in hits)
        return risk, flags

    # D8a' — the agent naming tool output as its reason for paying.
    #
    # This is the signature a *successful* injection leaves, and it is not the same thing as
    # D8a. D8a fires when the attacker's own words survive into the reasoning ("ignore previous
    # instructions"); a competent injection does not leave those. What it leaves is a compliant
    # paraphrase — "the service says a verification fee is required, so I'll send it" — which
    # reads as perfectly ordinary text and is invisible at L1, where the payment is just a
    # well-formed authorize to a known merchant.
    #
    # The tell is the *authority structure*: money moved because something the agent fetched
    # said it should. A legitimate purchase is justified by the task, not by the counterparty.
    def _deference(self, action: Action, text: str, risk: float, flags: list[str]):
        if action.action_type not in _PAY:
            return risk, flags
        match = _DEFERENCE.search(text)
        if not match:
            return risk, flags

        # A counterparty stating its own price is the protocol working, not an attack:
        # "the service requires 20,000 units" is what a 402 challenge says. What matters is
        # deference to an obligation *beyond* the purchase — a fee, a verification step, a
        # transfer that has to happen first. Without that marker this fired on most ordinary
        # sessions, which is worse than not firing at all.
        if not _EXTRA_OBLIGATION.search(text):
            return risk, flags

        tool_sourced = any(s.startswith("tool:") for s in action.context_sources)
        if tool_sourced:
            # Deference plus a tool in the context is the full pattern.
            risk = max(risk, 0.80)
            flags.append("pays_on_tool_authority")
        else:
            # Same language, no tool we can point at — worth a look, not a block.
            risk = max(risk, 0.45)
            flags.append("defers_to_external_authority")
        flags.append(f"deference:{match.group(0)[:40].strip().lower()}")
        return risk, flags

    # D8b — planning to defeat a control the wire will report as satisfied.
    def _evasion(self, text: str, risk: float, flags: list[str]):
        for pattern, weight, name in _EVASION:
            if pattern.search(text):
                risk = max(risk, weight)
                flags.append(f"evasion:{name}")
        return risk, flags

    # D8d
    def _concealment(self, text: str, risk: float, flags: list[str]):
        for pattern, weight, name in _CONCEALMENT:
            if pattern.search(text):
                risk = max(risk, weight)
                flags.append(f"conceal:{name}")
        return risk, flags

    # D8c — needs L0 and L1 together, so no single-layer detector can reach it.
    def _intent_mismatch(self, action: Action, ctx: Context, risk: float,
                         flags: list[str]):
        intent = action.stated_intent or action.reasoning
        if not intent or action.action_type not in _PAY:
            return risk, flags

        # Purchase intent is judged over the whole session, not this one thought. A turn
        # that says "let me check the balance first" names only a read verb, but the agent
        # may well have said "buy one search" two turns earlier. Scoring the turn in
        # isolation flagged ordinary sessions whose reasoning happened to be split up.
        session_text = " ".join(
            filter(None, [intent] + [a.reasoning or a.stated_intent
                                     for a in getattr(ctx, "history", [])]))
        verbs = {v.lower() for v in _INTENT_VERBS.findall(session_text)}

        # Said it would only look, then paid. Suggestive rather than conclusive: an agent
        # narrating the thing it is buying ("getting the search result") states no purchase
        # verb but is not lying. So this escalates for review and does not block on its own,
        # and it stays quiet when the reasoning shows the agent knows money is involved.
        if verbs and not (verbs & _BUY_VERBS) and not _PURCHASE_CONTEXT.search(intent):
            risk = max(risk, 0.55)
            flags.append("intent_says_read_action_pays")

        # Named an amount that does not match what it actually spent.
        #
        # Only a figure the agent presents as the *purchase* counts. Reasoning routinely
        # recites the wallet balance ("I have $100 available"), and taking the first dollar
        # figure in the text made every ordinary session with a balance check look like a
        # 0.0x mismatch.
        stated = _purchase_amount(intent)
        if stated and action.amount_units:
            try:
                stated_units = int(float(stated.replace(",", "")) * 1_000_000)
            except ValueError:
                stated_units = None
            if stated_units and stated_units > 0:
                ratio = action.amount_units / stated_units
                if ratio >= self.intent_mismatch_ratio or ratio <= 1 / self.intent_mismatch_ratio:
                    risk = max(risk, 0.90)
                    flags.append(f"intent_amount_mismatch_{ratio:.1f}x")

        # Named a vendor it then did not pay.
        if action.vendor:
            vendor_root = action.vendor.split(".")[0].lower()
            if len(vendor_root) > 3 and vendor_root not in intent.lower():
                mentioned = re.findall(r"\b([a-z0-9-]{4,})\.(?:com|ai|io|org|net)\b", intent.lower())
                if mentioned and vendor_root not in {m for m in mentioned}:
                    risk = max(risk, 0.80)
                    flags.append("intent_vendor_mismatch")

        return risk, flags
