"""
Hosted platform sink — the one hosted integration.

Two properties define it and neither is negotiable:

**Data minimisation.** Only derived features leave the machine by default: amounts,
identifiers, timestamps, detector flags. Raw payload text — the field most likely to
contain a customer's business data — is reduced to a SHA-256 digest, which is enough
to notice the same payload twice and not enough to reconstruct it. Sending raw
payloads is possible but requires saying so explicitly.

**Fail-open.** Buffered, batched, and silent on failure. If the platform is down the
agent does not notice. This is the opposite of the platform's own decision path,
which fails closed — and both directions are deliberate: refusing to authorise a
payment when state is missing is safe, while breaking an agent because telemetry
is unreachable is not.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
from typing import Any, Optional

from .._http import post

DEFAULT_ENDPOINT = "https://api.example.invalid"

SAFE_ACTION_FIELDS = {
    "action_id",
    "session_id",
    "agent_id",
    "action_type",
    "timestamp",
    "service_id",
    "operation_id",
    "amount_units",
    "category",
    "idempotency_key",
    "probe_id",
    "is_attack",
}


def _digest(value: Any) -> str:
    try:
        blob = json.dumps(value, sort_keys=True, default=str)
    except (TypeError, ValueError):
        blob = str(value)
    return "sha256:" + hashlib.sha256(blob.encode("utf-8")).hexdigest()[:32]


def redact(record: dict[str, Any], send_payloads: bool = False) -> dict[str, Any]:
    """Reduce a trace record to what is safe to transmit."""
    action = record.get("action", {}) or {}
    safe_action = {k: v for k, v in action.items() if k in SAFE_ACTION_FIELDS}

    payload = action.get("payload")
    if payload:
        if send_payloads:
            safe_action["payload"] = payload
        else:
            safe_action["payload_digest"] = _digest(payload)
            safe_action["payload_keys"] = sorted(
                k for k in payload if isinstance(k, str)
            )[:32]

    # Vendor can be a wallet address; keep the shape, drop the identifier.
    if action.get("vendor"):
        safe_action["vendor_digest"] = _digest(action["vendor"])

    out: dict[str, Any] = {"action": safe_action}
    if "verdict" in record:
        out["verdict"] = record["verdict"]
    if "at" in record:
        out["at"] = record["at"]
    return out


class PlatformSink:
    """
    Ships derived features to the Hosted platform.

        sink = PlatformSink(agent_key="gak_pub_...:gak_sec_...")

    Credentials come from `ACBGUARD_AGENT_KEY` when not passed. Outbound only: this
    never opens a listener and never accepts an inbound connection.
    """

    name = "platform"

    def __init__(
        self,
        agent_key: Optional[str] = None,
        endpoint: Optional[str] = None,
        *,
        batch_size: int = 50,
        send_payloads: bool = False,
        timeout: float = 10.0,
    ):
        self.agent_key = agent_key or os.environ.get("ACBGUARD_AGENT_KEY", "")
        self.endpoint = (
            endpoint or os.environ.get("ACBGUARD_ENDPOINT") or DEFAULT_ENDPOINT
        ).rstrip("/")
        self.batch_size = batch_size
        self.send_payloads = send_payloads
        self.timeout = timeout

        self._buffer: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self.dropped = 0
        self.sent = 0

        if not self.agent_key:
            raise ValueError(
                "PlatformSink needs an agent key (pass agent_key= or set ACBGUARD_AGENT_KEY)"
            )
        if not self.endpoint.startswith("https://") and "localhost" not in self.endpoint:
            raise ValueError(f"refusing to send traces over plaintext: {self.endpoint}")

    def emit(self, record: dict[str, Any]) -> None:
        try:
            safe = redact(record, send_payloads=self.send_payloads)
        except Exception:
            return
        with self._lock:
            self._buffer.append(safe)
            ready = len(self._buffer) >= self.batch_size
        if ready:
            self.flush()

    def flush(self) -> None:
        with self._lock:
            batch, self._buffer = self._buffer, []
        if not batch:
            return
        try:
            status, _ = post(
                f"{self.endpoint}/traces",
                json_body={"records": batch},
                headers={"Authorization": f"Bearer {self.agent_key}"},
                timeout=self.timeout,
            )
            if 200 <= status < 300:
                self.sent += len(batch)
            else:
                self.dropped += len(batch)
        except Exception:
            # Fail open: an unreachable platform must not surface to the agent.
            self.dropped += len(batch)

    def stats(self) -> dict[str, int]:
        return {"sent": self.sent, "dropped": self.dropped, "buffered": len(self._buffer)}
