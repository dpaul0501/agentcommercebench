"""
Minimal HTTP client built on the standard library.

Deliberately not `requests`. The core package must install with no dependencies, and
a security tool that drags in a transitive tree is a harder sell than one that does not.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Optional

USER_AGENT = "acbguard"


class HTTPError(Exception):
    def __init__(self, status: int, body: str):
        self.status = status
        self.body = body
        super().__init__(f"HTTP {status}: {body[:200]}")


def request(
    method: str,
    url: str,
    *,
    json_body: Optional[dict[str, Any]] = None,
    headers: Optional[dict[str, str]] = None,
    timeout: float = 30.0,
) -> tuple[int, Any]:
    """
    Returns (status, parsed_body). Parsed as JSON when possible, else raw text.

    Raises only on transport failure; HTTP error statuses come back as values so
    callers can classify a 403 as a refusal rather than a crash.
    """
    data = None
    hdrs = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    if json_body is not None:
        data = json.dumps(json_body).encode("utf-8")
        hdrs["Content-Type"] = "application/json"
    if headers:
        hdrs.update(headers)

    req = urllib.request.Request(url, data=data, headers=hdrs, method=method.upper())
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            return resp.status, _parse(raw)
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace") if exc.fp else str(exc)
        return exc.code, _parse(raw)


def _parse(raw: str) -> Any:
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return raw


def post(url: str, **kw) -> tuple[int, Any]:
    return request("POST", url, **kw)


def get(url: str, **kw) -> tuple[int, Any]:
    return request("GET", url, **kw)
