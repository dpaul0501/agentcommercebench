"""
Payload layer — scans request bodies for smuggled instructions.

Deterministic and fast (sub-millisecond): regex library, suspicious field names, and
a decode pass so base64-wrapped content is scanned as text rather than skipped.

No LLM call. An optional semantic scorer can be layered on separately for offline
analysis, but nothing in this module reaches the network.
"""
from __future__ import annotations

import base64
import binascii
import re
from typing import Any, Optional

from ..schema import Action
from .base import Context

# (pattern, risk, flag)
PATTERNS: list[tuple[str, float, str]] = [
    (r"ignore\s+(all\s+)?(previous|prior|earlier)\s+instruction", 0.95, "ignore_instructions"),
    (r"(system|instruction)\s*:\s*(transfer|send|pay|authorize)", 0.90, "embedded_system_cmd"),
    (r"override\s*(note|instruction|policy|limit|spend|approval)", 0.85, "override_keyword"),
    (r"transfer.{0,40}to\s+0x[a-fA-F0-9]{20,}", 0.90, "transfer_to_wallet"),
    (r"0x[a-fA-F0-9]{20,}Malicious", 0.95, "known_malicious_wallet"),
    (r"(skip|bypass|no need for)\s+(approval|escalation|confirmation)", 0.85, "approval_bypass"),
    (r"pre-?approved", 0.55, "claims_preapproved"),
    (r"do not (ask|request|require)\s+(for\s+)?confirmation", 0.80, "suppress_confirmation"),
    (r"place the same order\s+\d+\s+more times", 0.85, "recursive_purchase"),
    (r"repeat (this|the) (order|purchase|payment)", 0.75, "repeat_instruction"),
    (r"manual\s+override\s+required", 0.80, "manual_override"),
    (r"circular_chain", 0.80, "circular_chain_marker"),
    (r"approval is not required", 0.90, "asserts_no_approval"),
    (r"limits (were|have been) (raised|lifted|increased)", 0.85, "claims_limit_raised"),
]

COMPILED = [(re.compile(p, re.IGNORECASE), risk, flag) for p, risk, flag in PATTERNS]

SUSPICIOUS_FIELDS = {
    "override_note",
    "system_instruction",
    "transfer_to",
    "ship_to",
    "circular_chain",
    "purchase_for",
    "external_wallet",
    "tool_description",
}

BASE64_BLOB = re.compile(r"^[A-Za-z0-9+/]{40,}={0,2}$")


def _decode(value: str) -> Optional[str]:
    """Return decoded text only if this really looks like encoded prose."""
    candidate = value.strip()
    if not BASE64_BLOB.match(candidate) or len(candidate) % 4:
        return None
    try:
        decoded = base64.b64decode(candidate, validate=True).decode("utf-8")
    except (binascii.Error, ValueError, UnicodeDecodeError):
        return None
    # A hex wallet address or random id decodes to bytes, not sentences. Require
    # mostly-printable text with whitespace before treating it as smuggled content.
    if not decoded.isprintable() or " " not in decoded:
        return None
    return decoded


def _walk(value: Any, plain: list[str], decoded: list[str], keys: list[str]) -> None:
    if isinstance(value, dict):
        for k, v in value.items():
            keys.append(str(k))
            _walk(v, plain, decoded, keys)
    elif isinstance(value, (list, tuple)):
        for v in value:
            _walk(v, plain, decoded, keys)
    elif isinstance(value, str):
        plain.append(value)
        hidden = _decode(value)
        if hidden is not None:
            decoded.append(hidden)
    elif value is not None:
        plain.append(str(value))


def _match(text: str) -> list[tuple[float, str]]:
    return [
        (risk, flag) for pattern, risk, flag in COMPILED if pattern.search(text)
    ]


class PayloadDetector:
    name = "payload"

    def score(self, action: Action, ctx: Context) -> tuple[float, list[str]]:
        plain: list[str] = []
        decoded: list[str] = []
        keys: list[str] = []
        _walk(action.payload, plain, decoded, keys)

        risk = 0.0
        flags: list[str] = []

        for hit_risk, flag in _match("\n".join(plain)):
            risk = max(risk, hit_risk)
            flags.append(flag)

        # Only claim evasion when the *decoded* content is what matched.
        hidden_hits = _match("\n".join(decoded)) if decoded else []
        if hidden_hits:
            for hit_risk, flag in hidden_hits:
                risk = max(risk, hit_risk)
                if flag not in flags:
                    flags.append(flag)
            risk = max(risk, 0.90)
            flags.append("encoded_payload")

        hit_fields = SUSPICIOUS_FIELDS.intersection(keys)
        if hit_fields:
            risk = max(risk, 0.60)
            flags.extend(f"field:{f}" for f in sorted(hit_fields))

        return risk, flags
