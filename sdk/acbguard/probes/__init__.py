"""
Probes — attack generators.

A probe takes a clean baseline session and returns a mutated copy carrying ground
truth. Probes are pure: they clone before mutating, so one baseline can be attacked
many ways and the results compared.

Register a custom probe with the @probe decorator:

    from acbguard.probes import probe, Probe

    @probe(id="X1", family="custom", title="My attack",
           description="...", expected_layer="payload")
    def my_probe(session, rng):
        session.actions[-1].payload["evil"] = True
        return session
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Iterable, Optional

from ..schema import Session

ProbeFn = Callable[[Session, random.Random], Session]


@dataclass(frozen=True)
class Probe:
    id: str
    family: str
    title: str
    description: str
    expected_layer: str
    """Which detection layer should catch this: payload | behavioral | price | registry."""
    fn: ProbeFn

    def apply(self, baseline: Session, seed: int = 42) -> Session:
        attacked = baseline.clone()
        attacked.is_clean = False
        attacked.probe_id = self.id
        result = self.fn(attacked, random.Random(seed))
        for action in result.actions:
            if action.is_attack:
                action.probe_id = self.id
        return result


_REGISTRY: dict[str, Probe] = {}


def probe(
    *, id: str, family: str, title: str, description: str, expected_layer: str
) -> Callable[[ProbeFn], ProbeFn]:
    def decorate(fn: ProbeFn) -> ProbeFn:
        if id in _REGISTRY:
            raise ValueError(f"Probe id {id!r} already registered")
        _REGISTRY[id] = Probe(
            id=id,
            family=family,
            title=title,
            description=description,
            expected_layer=expected_layer,
            fn=fn,
        )
        return fn

    return decorate


def get(probe_id: str) -> Probe:
    try:
        return _REGISTRY[probe_id]
    except KeyError:
        raise KeyError(
            f"Unknown probe {probe_id!r}. Available: {sorted(_REGISTRY)}"
        ) from None


def all_probes(
    families: Optional[Iterable[str]] = None, ids: Optional[Iterable[str]] = None
) -> list[Probe]:
    """All registered probes, optionally filtered by family or explicit ids."""
    selected = list(_REGISTRY.values())
    if families is not None:
        wanted = set(families)
        selected = [p for p in selected if p.family in wanted]
    if ids is not None:
        wanted = set(ids)
        selected = [p for p in selected if p.id in wanted]
    return sorted(selected, key=lambda p: p.id)


def families() -> list[str]:
    return sorted({p.family for p in _REGISTRY.values()})


# Importing these modules populates the registry via @probe.
from . import injection, behavioral, settlement  # noqa: E402,F401

__all__ = ["Probe", "probe", "get", "all_probes", "families"]
