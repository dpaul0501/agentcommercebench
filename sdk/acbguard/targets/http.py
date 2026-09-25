"""
HTTP target — any REST or JSON-RPC endpoint.

The base every wire protocol here is built on. An Action becomes a request via a
`request_builder`, and the response is classified into an Outcome. Both halves are
injectable, which is what lets one class cover REST, JSON-RPC, and the commerce
protocols without special-casing any of them.

Status codes are read as decisions: 402/403/409/422/429 are refusals, since a guard
that declines a payment overwhelmingly signals it that way.
"""
from __future__ import annotations

import time
from typing import Any, Callable, Optional

from .._http import post as http_post
from ..schema import Action, Decision, Observation, Outcome
from .base import classify

# Status codes a policy layer uses to say no.
REFUSAL_CODES = {401: "unauthorized", 402: "payment_required", 403: "forbidden",
                 409: "conflict", 422: "unprocessable", 429: "rate_limited"}


def default_request(action: Action) -> dict[str, Any]:
    """Flat JSON body. Overridable per target."""
    body: dict[str, Any] = {
        "action_type": action.action_type.value,
        "agent_id": action.agent_id,
        "session_id": action.session_id,
    }
    for field in ("service_id", "operation_id", "amount_units", "vendor",
                  "category", "idempotency_key"):
        value = getattr(action, field)
        if value is not None:
            body[field] = value
    if action.payload:
        body["payload"] = action.payload
    return body


class HTTPTarget:
    """
        target = HTTPTarget("https://api.example.com/authorize",
                            headers={"Authorization": "Bearer ..."})

    Point `request_builder` at your own function to match a bespoke API shape.
    """

    def __init__(
        self,
        url: str,
        *,
        headers: Optional[dict[str, str]] = None,
        request_builder: Optional[Callable[[Action], dict[str, Any]]] = None,
        response_classifier: Optional[Callable[[int, Any], tuple[Outcome, Optional[Decision], str]]] = None,
        name: Optional[str] = None,
        timeout: float = 30.0,
        method: str = "POST",
    ):
        self.url = url
        self.headers = headers or {}
        self.request_builder = request_builder or default_request
        self.response_classifier = response_classifier or self.classify_response
        self.name = name or url
        self.timeout = timeout
        self.method = method

    @staticmethod
    def classify_response(status: int, body: Any) -> tuple[Outcome, Optional[Decision], str]:
        if status in REFUSAL_CODES:
            return Outcome.DEFENDED, Decision.BLOCK, REFUSAL_CODES[status]
        if status >= 500:
            return Outcome.ERROR, None, f"server_error_{status}"
        if 200 <= status < 300:
            # A 2xx can still carry a refusal in the body.
            return classify(body)
        return Outcome.ERROR, None, f"http_{status}"

    def execute(self, action: Action) -> Observation:
        started = time.perf_counter()
        try:
            status, body = http_post(
                self.url,
                json_body=self.request_builder(action),
                headers=self.headers,
                timeout=self.timeout,
            )
        except Exception as exc:
            return Observation(
                outcome=Outcome.ERROR,
                reason=f"{type(exc).__name__}: {exc}"[:200],
                latency_ms=(time.perf_counter() - started) * 1000,
            )

        elapsed = (time.perf_counter() - started) * 1000
        outcome, decision, reason = self.response_classifier(status, body)
        return Observation(
            outcome=outcome,
            decision=decision,
            reason=reason,
            latency_ms=elapsed,
            raw=body if isinstance(body, dict) else {"body": str(body)[:500]},
        )


class JSONRPCTarget(HTTPTarget):
    """
    JSON-RPC 2.0 — the transport A2A and parts of UCP use.

    A JSON-RPC error object is a refusal signal; `error.message` carries the reason.
    """

    def __init__(self, url: str, method_name: str, **kw):
        self.method_name = method_name
        params_builder = kw.pop("params_builder", None) or default_request
        super().__init__(url, request_builder=self._envelope, **kw)
        self._params_builder = params_builder

    def _envelope(self, action: Action) -> dict[str, Any]:
        return {
            "jsonrpc": "2.0",
            "id": action.action_id,
            "method": self.method_name,
            "params": self._params_builder(action),
        }

    @staticmethod
    def classify_response(status: int, body: Any) -> tuple[Outcome, Optional[Decision], str]:
        if isinstance(body, dict) and "error" in body:
            err = body["error"] or {}
            message = err.get("message", str(err)) if isinstance(err, dict) else str(err)
            outcome, decision, _ = classify({"error": message})
            return outcome, decision or Decision.BLOCK, str(message)[:200]
        if isinstance(body, dict) and "result" in body:
            return classify(body["result"])
        return HTTPTarget.classify_response(status, body)
