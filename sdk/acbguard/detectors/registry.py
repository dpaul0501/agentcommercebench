"""
Registry layer — identity and settlement state.

This is the layer that needs memory the action itself does not carry: which
idempotency keys have already settled, which agent id owns this session, which
counterparties are new. Catching replay or impersonation without this state is
not a matter of a better model; the information is simply absent.

In-process here, so it holds only for the current run. In production this backs
onto durable stores, which is why these checks belong outside the sub-300ms
in-process decision path.
"""
from __future__ import annotations

import statistics

from ..schema import ActionType, Action
from .base import Context

_PAY = (ActionType.AUTHORIZE, ActionType.A2A_TRANSFER)


class RegistryDetector:
    name = "registry"

    def score(self, action: Action, ctx: Context) -> tuple[float, list[str]]:
        risk = 0.0
        flags: list[str] = []

        # Replay: this key already settled in this run.
        #
        # But a *correctly keyed retry* also reuses its key, and that is the entire purpose of
        # an idempotency key — the server returns the cached result instead of charging again.
        # Treating every reuse as replay punishes the agents that implement idempotency
        # properly: it blocked 294 of 312 clean sessions here, and the agents it blocked were
        # the well-behaved ones.
        #
        # Replay is reuse of a key for a *different* request. A retry repeats the same request
        # fingerprint, which production records on every settlement.
        if action.idempotency_key and action.idempotency_key in ctx.settled_keys:
            prior = next(
                (a for a in ctx.history
                 if a.idempotency_key == action.idempotency_key
                 and a.request_fingerprint),
                None,
            )
            same_request = (
                prior is not None
                and action.request_fingerprint is not None
                and prior.request_fingerprint == action.request_fingerprint
            )
            if not same_request:
                risk = max(risk, 0.95)
                flags.append("idempotency_replay")

        # Impersonation: the request claims an agent the caller did not authenticate as.
        #
        # Against the authenticated principal where one is supplied, and only then against the
        # session. A session is a harness convenience; the principal is what an API key
        # actually proves, and it is available on every production settlement.
        session_agent = ctx.principal_id or (ctx.session.agent_id if ctx.session else None)
        if session_agent and action.agent_id != session_agent:
            risk = max(risk, 0.85)
            flags.append(f"agent_id_mismatch:{action.agent_id}")

        if action.action_type in _PAY and action.vendor:
            # Circular routing: value returning to the originating agent.
            if action.vendor == session_agent:
                risk = max(risk, 0.80)
                flags.append("circular_settlement")

            chain = action.payload.get("circular_chain")
            if isinstance(chain, (list, tuple)) and session_agent in chain:
                risk = max(risk, 0.85)
                flags.append("circular_chain_declared")

            # Sybil: a counterparty warmed up on dust, then paid at scale.
            #
            # Expressed as a ratio against that counterparty's OWN prior payments, never as
            # an absolute figure. "Dust is under $0.05 and a payout is over $1.00" is a pair
            # of numbers nobody can justify: $1.00 is a routine purchase for some agents here
            # and a month of spend for others, and both figures also appear in the generator.
            # A relative rule needs no such constant and means the same thing at every scale.
            to_vendor = [
                a.amount_units for a in ctx.history
                if a.vendor == action.vendor and a.amount_units
            ]
            if len(to_vendor) >= 4 and action.amount_units:
                warmup = statistics.median(to_vendor)
                spread = max(to_vendor) / min(to_vendor)
                # A warm-up is small AND consistent: several payments of similar size, then
                # one far larger. A counterparty already transacting across a wide range has
                # no warm-up to speak of, and a big payment there is just a big payment.
                if spread <= 4 and action.amount_units >= warmup * 20:
                    risk = max(risk, 0.75)
                    flags.append(
                        f"sybil_warmup_then_payout:{action.amount_units / warmup:.0f}x")

        return risk, flags
