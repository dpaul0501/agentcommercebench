"""
Hosted platform baselines — norms fitted on your own history, served back.

Cached in-process and fail-soft: an unreachable platform yields the fallback rather
than an exception, because a guard that stops working when its config server blinks
is worse than one running on slightly stale norms.
"""
from __future__ import annotations

import os
import time
from typing import Any, Optional

from .._http import get


class PlatformBaselines:
    """
        provider = PlatformBaselines(agent_key="gak_pub_...:gak_sec_...")
        guard = Guard(baseline=provider.baseline_for("agent-1"))

    `fallback` is used whenever the platform has nothing or cannot be reached, so a
    cold start still gets declared limits rather than none.
    """

    name = "platform"

    def __init__(
        self,
        agent_key: Optional[str] = None,
        endpoint: Optional[str] = None,
        *,
        fallback: Optional[dict[str, Any]] = None,
        ttl_seconds: float = 300.0,
        timeout: float = 10.0,
    ):
        from ..sinks.platform import DEFAULT_ENDPOINT

        self.agent_key = agent_key or os.environ.get("ACBGUARD_AGENT_KEY", "")
        self.endpoint = (
            endpoint or os.environ.get("ACBGUARD_ENDPOINT") or DEFAULT_ENDPOINT
        ).rstrip("/")
        self.fallback = fallback or {}
        self.ttl = ttl_seconds
        self.timeout = timeout
        self._cache: dict[str, tuple[float, dict[str, Any]]] = {}

        if not self.agent_key:
            raise ValueError(
                "PlatformBaselines needs an agent key (pass agent_key= or set ACBGUARD_AGENT_KEY)"
            )

    def baseline_for(self, agent_id: str) -> dict[str, Any]:
        cached = self._cache.get(agent_id)
        if cached and (time.monotonic() - cached[0]) < self.ttl:
            return dict(cached[1])

        try:
            status, body = get(
                f"{self.endpoint}/agents/{agent_id}/baseline",
                headers={"Authorization": f"Bearer {self.agent_key}"},
                timeout=self.timeout,
            )
        except Exception:
            return dict(self.fallback)

        if not (200 <= status < 300) or not isinstance(body, dict):
            return dict(self.fallback)

        baseline = self._coerce({**self.fallback, **body})
        self._cache[agent_id] = (time.monotonic(), baseline)
        return dict(baseline)

    @staticmethod
    def _coerce(raw: dict[str, Any]) -> dict[str, Any]:
        """JSON has no sets or ranges; restore the shapes detectors expect."""
        out = dict(raw)
        if isinstance(out.get("known_services"), list):
            out["known_services"] = set(out["known_services"])
        hours = out.get("active_hours")
        if isinstance(hours, list) and len(hours) == 2:
            out["active_hours"] = range(int(hours[0]), int(hours[1]) + 1)
        return out

    def invalidate(self, agent_id: Optional[str] = None) -> None:
        if agent_id:
            self._cache.pop(agent_id, None)
        else:
            self._cache.clear()
