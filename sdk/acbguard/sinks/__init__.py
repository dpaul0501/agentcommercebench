"""
Trace sinks — where observations go.

acbguard never assumes a backend. A sink is anything with `emit`, so a trace can
land in a file, a queue, your own warehouse, or a hosted platform. The shipped
implementations are local; Platform is one option among them, not a dependency.

Every sink must fail open. Telemetry that can take down the agent it observes is
worse than no telemetry, so `emit` swallows its own errors by contract.
"""
from __future__ import annotations

import json
import threading
from typing import Any, Iterable, Optional, Protocol, runtime_checkable


@runtime_checkable
class TraceSink(Protocol):
    def emit(self, record: dict[str, Any]) -> None:
        """Record one observation. Must never raise."""

    def flush(self) -> None:
        """Best-effort delivery of anything buffered. Must never raise."""


class NullSink:
    """Discards everything. The default, so nothing leaves by accident."""

    name = "null"

    def emit(self, record: dict[str, Any]) -> None:
        return None

    def flush(self) -> None:
        return None


class MemorySink:
    """Keeps records in a list. For tests and short-lived analysis."""

    name = "memory"

    def __init__(self, cap: Optional[int] = None):
        self.records: list[dict[str, Any]] = []
        self.cap = cap
        self._lock = threading.Lock()

    def emit(self, record: dict[str, Any]) -> None:
        with self._lock:
            self.records.append(record)
            if self.cap and len(self.records) > self.cap:
                del self.records[: -self.cap]

    def flush(self) -> None:
        return None

    def __len__(self) -> int:
        return len(self.records)


class FileSink:
    """Append-only JSONL on local disk. Never leaves the machine."""

    name = "file"

    def __init__(self, path: str):
        self.path = path
        self._lock = threading.Lock()

    def emit(self, record: dict[str, Any]) -> None:
        try:
            line = json.dumps(record, default=str)
        except (TypeError, ValueError):
            return
        try:
            with self._lock, open(self.path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        except OSError:
            return

    def flush(self) -> None:
        return None

    def read(self) -> list[dict[str, Any]]:
        try:
            with open(self.path, encoding="utf-8") as fh:
                return [json.loads(line) for line in fh if line.strip()]
        except (OSError, json.JSONDecodeError):
            return []


class MultiSink:
    """Fan out to several sinks. One failing never affects the others."""

    name = "multi"

    def __init__(self, sinks: Iterable[TraceSink]):
        self.sinks = list(sinks)

    def emit(self, record: dict[str, Any]) -> None:
        for sink in self.sinks:
            try:
                sink.emit(record)
            except Exception:
                continue

    def flush(self) -> None:
        for sink in self.sinks:
            try:
                sink.flush()
            except Exception:
                continue


def __getattr__(name: str):
    if name == "PlatformSink":
        from .platform import PlatformSink

        return PlatformSink
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["TraceSink", "NullSink", "MemorySink", "FileSink", "MultiSink", "PlatformSink"]
