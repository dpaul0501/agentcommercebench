"""
Targets under test.

Everything implements one protocol — `execute(action) -> Observation` — so a probe
does not know or care whether it is hitting a function, an MCP server, a UCP
checkout, an A2A agent, or a raw LLM.

Heavier targets are imported lazily so the core stays dependency-free.
"""
from .base import Target, baseline_for, baseline_session, classify
from .callable_ import CallableTarget, NullTarget, PipelineTarget
from .http import HTTPTarget, JSONRPCTarget
from .registry import SCHEMES, from_uri

_LAZY = {
    "MCPTarget": ("mcp", "MCPTarget"),
    "A2ATarget": ("protocols", "A2ATarget"),
    "UCPTarget": ("protocols", "UCPTarget"),
    "LangGraphTarget": ("langgraph", "LangGraphTarget"),
    "ModelTarget": ("model", "ModelTarget"),
    "ChatModel": ("model", "ChatModel"),
    "CallableModel": ("model", "CallableModel"),
    "OpenAIChatModel": ("model", "OpenAIChatModel"),
    "AnthropicChatModel": ("model", "AnthropicChatModel"),
}


def __getattr__(name: str):
    if name in _LAZY:
        module_name, attr = _LAZY[name]
        from importlib import import_module

        return getattr(import_module(f".{module_name}", __package__), attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "A2ATarget",
    "AnthropicChatModel",
    "CallableModel",
    "CallableTarget",
    "ChatModel",
    "HTTPTarget",
    "JSONRPCTarget",
    "LangGraphTarget",
    "MCPTarget",
    "ModelTarget",
    "NullTarget",
    "OpenAIChatModel",
    "PipelineTarget",
    "SCHEMES",
    "Target",
    "UCPTarget",
    "baseline_for",
    "baseline_session",
    "classify",
    "from_uri",
]
