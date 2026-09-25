"""
acbguard — adversarial testing and runtime guarding for agents that spend money.

    from acbguard import scan
    from acbguard.targets import CallableTarget

    report = scan(CallableTarget(my_agent))
    print(report.render())

Runtime:

    from acbguard import Guard
    guard = Guard(mode="observe", trace_path="traces.jsonl")
    guard.check(action)
"""
from .agent import AgentRun, AgentSpec, AgentTarget, ToolSpec
from .config_audit import AuditReport, ConfigFinding, audit
from .baselines import (
    BaselineProvider,
    LearnedBaseline,
    NullBaseline,
    StaticBaseline,
)
from .detectors import Context, Pipeline, Verdict, default_pipeline
from .guard import Blocked, Guard, Mode, NeedsApproval
from .guard.store import ContextStore, InMemoryContextStore
from .probes import Probe, all_probes, families, get as get_probe, probe
from .scan import Finding, Report, scan
from .sinks import FileSink, MemorySink, MultiSink, NullSink, TraceSink
from .targets import from_uri
from .schema import (
    MICRO_PER_USD,
    Action,
    ActionType,
    Decision,
    Observation,
    Outcome,
    Persona,
    Session,
)

__version__ = "0.1.0"

__all__ = [
    "Action",
    "ActionType",
    "AgentRun",
    "AgentSpec",
    "AgentTarget",
    "AuditReport",
    "BaselineProvider",
    "ConfigFinding",
    "ToolSpec",
    "audit",
    "Blocked",
    "ContextStore",
    "Context",
    "InMemoryContextStore",
    "Decision",
    "FileSink",
    "Finding",
    "Guard",
    "LearnedBaseline",
    "MICRO_PER_USD",
    "MemorySink",
    "Mode",
    "MultiSink",
    "NeedsApproval",
    "NullBaseline",
    "NullSink",
    "Observation",
    "Outcome",
    "Persona",
    "Pipeline",
    "Probe",
    "Report",
    "Session",
    "StaticBaseline",
    "TraceSink",
    "Verdict",
    "all_probes",
    "default_pipeline",
    "families",
    "from_uri",
    "get_probe",
    "probe",
    "scan",
    "__version__",
]
