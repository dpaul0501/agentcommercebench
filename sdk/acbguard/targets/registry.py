"""
Target URIs — one string builds any target.

    unguarded                     an agent with no controls (the floor)
    pipeline                      acbguard's own detectors
    module:pkg.mod:attr           a python callable
    http://... | https://...      a REST endpoint
    mcp+http://host/mcp           MCP over streamable HTTP
    mcp+stdio:python -m server    MCP over stdio
    a2a+https://host/a2a          Agent2Agent (JSON-RPC)
    ucp+https://host/ucp          Universal Commerce Protocol
    openai:gpt-4o-mini            any OpenAI-compatible chat endpoint
    anthropic:claude-sonnet-4-6   Anthropic messages API
    langgraph:pkg.mod:graph       a compiled LangGraph

Keeps the CLI honest: every target is reachable from a config file or a CI job
without writing Python.
"""
from __future__ import annotations

import importlib
import sys
from typing import Any, Optional


def _load(spec: str) -> Any:
    if ":" not in spec:
        raise ValueError(f"expected 'package.module:attribute', got {spec!r}")
    mod_name, attr = spec.rsplit(":", 1)
    if "" not in sys.path:
        sys.path.insert(0, "")
    module = importlib.import_module(mod_name)
    try:
        return getattr(module, attr)
    except AttributeError:
        raise ValueError(f"{mod_name!r} has no attribute {attr!r}") from None


def from_uri(uri: str, *, auth: Optional[str] = None, **kw) -> Any:
    """Build a target from a URI string. See module docstring for the schemes."""
    headers = {"Authorization": auth} if auth else {}

    if uri in ("unguarded", "null"):
        from .callable_ import NullTarget

        return NullTarget()

    if uri in ("pipeline", "demo", "self"):
        from .callable_ import PipelineTarget

        return PipelineTarget()

    if uri.startswith("module:"):
        from .callable_ import CallableTarget

        spec = uri[len("module:") :]
        return CallableTarget(_load(spec), name=spec, **kw)

    if uri.startswith("langgraph:"):
        from .langgraph import LangGraphTarget

        spec = uri[len("langgraph:") :]
        return LangGraphTarget(_load(spec), name=f"langgraph:{spec}", **kw)

    if uri.startswith("openai:"):
        from .model import ModelTarget, OpenAIChatModel

        return ModelTarget(OpenAIChatModel(uri[len("openai:") :], **kw))

    if uri.startswith("anthropic:"):
        from .model import AnthropicChatModel, ModelTarget

        return ModelTarget(AnthropicChatModel(uri[len("anthropic:") :], **kw))

    if uri.startswith("mcp+stdio:"):
        from .mcp import MCPTarget

        parts = uri[len("mcp+stdio:") :].split()
        return MCPTarget.stdio(parts[0], parts[1:], **kw)

    if uri.startswith("mcp+"):
        from .mcp import MCPTarget

        return MCPTarget.http(uri[len("mcp+") :], auth=auth, **kw)

    if uri.startswith("a2a+"):
        from .protocols import A2ATarget

        return A2ATarget(uri[len("a2a+") :], headers=headers, **kw)

    if uri.startswith("ucp+"):
        from .protocols import UCPTarget

        return UCPTarget(uri[len("ucp+") :], headers=headers, **kw)

    if uri.startswith(("http://", "https://")):
        from .http import HTTPTarget

        return HTTPTarget(uri, headers=headers, **kw)

    raise ValueError(
        f"unrecognised target {uri!r}. Supported: unguarded, pipeline, module:, "
        "langgraph:, openai:, anthropic:, mcp+http(s)://, mcp+stdio:, a2a+, ucp+, http(s)://"
    )


SCHEMES = [
    ("unguarded", "an agent with no controls"),
    ("pipeline", "acbguard's own detectors"),
    ("module:pkg.mod:attr", "a python callable"),
    ("https://host/path", "a REST endpoint"),
    ("mcp+https://host/mcp", "MCP over HTTP"),
    ("mcp+stdio:CMD", "MCP over stdio"),
    ("a2a+https://host/a2a", "Agent2Agent"),
    ("ucp+https://host/ucp", "Universal Commerce Protocol"),
    ("openai:MODEL", "OpenAI-compatible chat endpoint"),
    ("anthropic:MODEL", "Anthropic messages API"),
    ("langgraph:pkg.mod:graph", "a compiled LangGraph"),
]
