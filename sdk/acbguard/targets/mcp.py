"""
MCP target — scan an MCP server.

Works against any MCP server over stdio or streamable HTTP. Actions are mapped onto
tool calls; the mapping is configurable because tool names differ per server.

Requires the optional `mcp` extra:

    pip install "acbguard[mcp]"
"""
from __future__ import annotations

import asyncio
import time
from typing import Any, Optional

from ..schema import Action, ActionType, Observation, Outcome
from .base import classify

DEFAULT_TOOL_MAP = {
    ActionType.FIND_SERVICE: "find_service",
    ActionType.GET_SERVICE: "get_service",
    ActionType.AUTHORIZE: "call_service",
    ActionType.A2A_TRANSFER: "transfer",
    ActionType.SETTLE: "settle",
}


def _require_mcp():
    try:
        import mcp  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "MCPTarget needs the optional mcp extra: pip install 'acbguard[mcp]'"
        ) from exc


class MCPTarget:
    """
    Scan a live MCP server.

        target = MCPTarget.http("https://api.example.com/mcp", auth="Bearer ...")
        target = MCPTarget.stdio("python", ["-m", "my_server"])

    `tool_map` overrides which tool each action type calls; `arg_builder` overrides
    how an Action becomes tool arguments.
    """

    def __init__(
        self,
        *,
        url: Optional[str] = None,
        command: Optional[str] = None,
        args: Optional[list[str]] = None,
        auth: Optional[str] = None,
        tool_map: Optional[dict[ActionType, str]] = None,
        arg_builder: Optional[Any] = None,
        name: Optional[str] = None,
        timeout: float = 30.0,
    ):
        _require_mcp()
        if not url and not command:
            raise ValueError("MCPTarget needs either url= (HTTP) or command= (stdio)")
        self.url = url
        self.command = command
        self.args = args or []
        self.auth = auth
        self.tool_map = {**DEFAULT_TOOL_MAP, **(tool_map or {})}
        self.arg_builder = arg_builder or self._default_args
        self.timeout = timeout
        self.name = name or (url or f"{command} {' '.join(self.args)}")
        self._tools_cache: Optional[list[str]] = None

    @classmethod
    def http(cls, url: str, auth: Optional[str] = None, **kw) -> "MCPTarget":
        return cls(url=url, auth=auth, **kw)

    @classmethod
    def stdio(cls, command: str, args: Optional[list[str]] = None, **kw) -> "MCPTarget":
        return cls(command=command, args=args, **kw)

    @staticmethod
    def _default_args(action: Action) -> dict[str, Any]:
        args: dict[str, Any] = {}
        if action.service_id:
            args["service_id"] = action.service_id
        if action.operation_id:
            args["operation_id"] = action.operation_id
        if action.amount_units is not None:
            args["amount_units"] = action.amount_units
        if action.vendor:
            args["vendor"] = action.vendor
        if action.idempotency_key:
            args["idempotency_key"] = action.idempotency_key
        if action.payload:
            args.update(action.payload)
        return args

    async def _session(self):
        from mcp import ClientSession

        if self.url:
            from mcp.client.streamable_http import streamablehttp_client

            headers = {"Authorization": self.auth} if self.auth else None
            return streamablehttp_client(self.url, headers=headers), ClientSession
        from mcp import StdioServerParameters
        from mcp.client.stdio import stdio_client

        params = StdioServerParameters(command=self.command, args=self.args)
        return stdio_client(params), ClientSession

    async def _call(self, tool: str, args: dict[str, Any]) -> Any:
        from mcp import ClientSession

        transport, _ = await self._session()
        async with transport as streams:
            read, write = streams[0], streams[1]
            async with ClientSession(read, write) as session:
                await session.initialize()
                if self._tools_cache is None:
                    listed = await session.list_tools()
                    self._tools_cache = [t.name for t in listed.tools]
                if tool not in self._tools_cache:
                    return {"error": f"tool_not_found: {tool}", "_skip": True}
                result = await session.call_tool(tool, args)
                if getattr(result, "isError", False):
                    return {"error": _text(result)}
                return {"result": _text(result)}

    async def list_tools(self) -> list[str]:
        from mcp import ClientSession

        transport, _ = await self._session()
        async with transport as streams:
            read, write = streams[0], streams[1]
            async with ClientSession(read, write) as session:
                await session.initialize()
                listed = await session.list_tools()
                self._tools_cache = [t.name for t in listed.tools]
                return self._tools_cache

    def execute(self, action: Action) -> Observation:
        tool = self.tool_map.get(action.action_type)
        if not tool:
            return Observation(outcome=Outcome.SKIPPED, reason="no tool mapped")

        args = self.arg_builder(action)
        started = time.perf_counter()
        try:
            response = asyncio.run(asyncio.wait_for(self._call(tool, args), self.timeout))
        except Exception as exc:
            return Observation(
                outcome=Outcome.ERROR,
                reason=f"{type(exc).__name__}: {exc}"[:200],
                latency_ms=(time.perf_counter() - started) * 1000,
            )

        elapsed = (time.perf_counter() - started) * 1000
        if isinstance(response, dict) and response.get("_skip"):
            return Observation(
                outcome=Outcome.SKIPPED, reason=response["error"], latency_ms=elapsed
            )
        outcome, decision, reason = classify(response)
        return Observation(
            outcome=outcome,
            decision=decision,
            reason=reason,
            latency_ms=elapsed,
            raw=response if isinstance(response, dict) else {},
        )


def _text(result: Any) -> str:
    content = getattr(result, "content", None)
    if not content:
        return str(result)
    parts = []
    for block in content:
        parts.append(getattr(block, "text", None) or str(block))
    return "\n".join(parts)
