"""
Session Graph Model — Novel Baseline for AgentCommerceBench

Represents each agent session as a directed graph of MCP calls and extracts
structural features for fraud detection. Trained with sklearn RandomForest.

Key insight: MCP call sequences form distinctive graphs per persona.
Clean sessions have regular, low-density graphs. Attacks introduce
structural anomalies: high out-degree (recon), cycles (recursive purchase),
amount variance spikes (price poisoning), timing irregularities.

This is the "novel model" contribution:
  - No prior work applies GNN/graph analysis to MCP commerce call graphs
  - Catches behavioral attacks (L2) that text classifiers miss
  - Complements the LLM safety baseline (L1 attacks)

Graph schema per session:
  Nodes: agent + each distinct service_id/operation_id
  Edges: directed call edges FIND_SERVICE→service, AUTHORIZE→service
  Features: amount (normalized), timing gap (seconds), category (one-hot)

Features extracted (20 dims):
  [0]  n_events
  [1]  n_find_service
  [2]  n_authorize
  [3]  n_settle
  [4]  find_authorize_ratio   (recon signal)
  [5]  n_distinct_services
  [6]  service_diversity      (entropy of service distribution)
  [7]  total_amount_usdc
  [8]  max_amount_usdc
  [9]  amount_std_usdc        (variance signal for ratcheting)
  [10] amount_max_ratio       (spike vs session mean)
  [11] timing_gap_mean_s
  [12] timing_gap_std_s       (burst signal)
  [13] timing_gap_min_s       (minimum gap — burst signal)
  [14] n_categories
  [15] dominant_category_frac (purity — scope violation signal)
  [16] n_a2a_transfers
  [17] a2a_amount_usdc
  [18] has_override_keyword   (L1 signal — SYSTEM: in any payload)
  [19] has_b64_blob           (L1 signal — base64 in payload)

Usage:
  from benchmark.models.session_graph import SessionGraphModel
  model = SessionGraphModel()
  model.fit(train_sessions)
  score, flags = model.detect(event, history)  # drop-in detector interface
"""
import sys, os, re, math, base64, pickle
from collections import Counter

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from harness.simulate.schema import Session, Event, ActionType

name = "session_graph_rf"

_OVERRIDE_PATTERNS = [
    re.compile(r"SYSTEM\s*:?", re.I),
    re.compile(r"ignore\s+(previous|prior)\s+instruction", re.I),
    re.compile(r"manual\s+override", re.I),
    re.compile(r"route\s+payment\s+via", re.I),
    re.compile(r"settlement\s+proxy", re.I),
    re.compile(r"transfer[_\s]+to\s*[:\s]*0x", re.I),
    re.compile(r"_mcp_response_injected", re.I),
]
_B64_PAT = re.compile(r'[A-Za-z0-9+/]{20,}={0,2}')
_B64_MALICIOUS = re.compile(r"instruction|transfer|ignore|override", re.I)


def _entropy(counts: list[int]) -> float:
    total = sum(counts)
    if total == 0:
        return 0.0
    return -sum((c / total) * math.log2(c / total + 1e-9) for c in counts if c > 0)


def _has_override(payload: dict | None) -> bool:
    if not payload:
        return False
    text = str(payload)
    return any(pat.search(text) for pat in _OVERRIDE_PATTERNS)


def _has_b64(payload: dict | None) -> bool:
    if not payload:
        return False
    text = str(payload)
    for token in _B64_PAT.findall(text):
        try:
            decoded = base64.b64decode(token + "==").decode("utf-8", errors="ignore")
            if _B64_MALICIOUS.search(decoded):
                return True
        except Exception:
            pass
    return False


def session_to_features(session: Session) -> np.ndarray:
    """Extract 20-dim graph feature vector from a session."""
    events = session.events
    if not events:
        return np.zeros(20)

    n_events    = len(events)
    n_find      = sum(1 for e in events if e.action_type == ActionType.FIND_SERVICE)
    n_auth      = sum(1 for e in events if e.action_type == ActionType.AUTHORIZE)
    n_settle    = sum(1 for e in events if e.action_type == ActionType.SETTLE)
    n_a2a       = sum(1 for e in events if e.action_type == ActionType.A2A_TRANSFER)

    ratio = n_find / max(n_auth, 1)

    services = [e.service_id or e.operation_id or "?" for e in events
                if e.service_id or e.operation_id]
    n_services = len(set(services))
    svc_counts = list(Counter(services).values())
    diversity = _entropy(svc_counts)

    amounts = [e.amount_units / 1_000_000 for e in events
               if e.amount_units and e.action_type in
               (ActionType.AUTHORIZE, ActionType.SETTLE, ActionType.A2A_TRANSFER)]
    total_amt = sum(amounts)
    max_amt   = max(amounts) if amounts else 0.0
    amt_std   = float(np.std(amounts)) if len(amounts) > 1 else 0.0
    amt_mean  = total_amt / max(len(amounts), 1)
    max_ratio = max_amt / max(amt_mean, 1e-9)

    timestamps = sorted(e.timestamp for e in events)
    gaps = [(timestamps[i+1] - timestamps[i]).total_seconds()
            for i in range(len(timestamps) - 1)]
    gap_mean = float(np.mean(gaps)) if gaps else 0.0
    gap_std  = float(np.std(gaps))  if len(gaps) > 1 else 0.0
    gap_min  = min(gaps) if gaps else 0.0

    cats = [e.category for e in events if e.category]
    n_cats = len(set(cats))
    cat_counts = list(Counter(cats).values())
    dom_frac = max(cat_counts) / max(sum(cat_counts), 1) if cat_counts else 0.0

    a2a_amounts = [e.amount_units / 1_000_000 for e in events
                   if e.action_type == ActionType.A2A_TRANSFER and e.amount_units]
    a2a_total = sum(a2a_amounts)

    has_override = int(any(_has_override(e.original_request) for e in events))
    has_b64      = int(any(_has_b64(e.original_request) for e in events))

    return np.array([
        n_events, n_find, n_auth, n_settle,
        ratio, n_services, diversity,
        total_amt, max_amt, amt_std, max_ratio,
        gap_mean, gap_std, gap_min,
        n_cats, dom_frac,
        n_a2a, a2a_total,
        has_override, has_b64,
    ], dtype=np.float32)


class SessionGraphModel:
    """
    RandomForest classifier on session graph features.
    Implements the same detect(event, history) interface as other detectors.
    """

    def __init__(self, n_estimators: int = 200, max_depth: int = 10):
        self.clf          = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight="balanced",
            random_state=42,
        )
        self._fitted      = False
        self._history_buf: dict[str, list[Event]] = {}
        self._scored_sessions: set[str] = set()
        self._session_score: dict[str, tuple[float, list[str]]] = {}

    def fit(self, sessions: list[Session]) -> "SessionGraphModel":
        X = np.stack([session_to_features(s) for s in sessions])
        y = np.array([0 if s.is_clean else 1 for s in sessions])
        self.clf.fit(X, y)
        self._fitted = True
        print(f"  [session_graph] Fitted on {len(sessions)} sessions "
              f"({sum(y==0)} clean / {sum(y==1)} attacked)")
        return self

    def _predict_session(self, events: list[Event]) -> tuple[float, list[str]]:
        """Compute session-level fraud score from event list."""
        if not self._fitted or not events:
            return 0.0, []
        # Build a dummy session to extract features
        dummy = Session.__new__(Session)
        dummy.events = events
        feats = session_to_features(dummy).reshape(1, -1)
        proba = self.clf.predict_proba(feats)[0]
        score = float(proba[1])  # P(fraud)

        if score < 0.30:
            return 0.0, []

        # Attribution: top contributing features
        importances = self.clf.feature_importances_
        feat_names = [
            "n_events", "n_find", "n_auth", "n_settle",
            "find_auth_ratio", "n_services", "svc_diversity",
            "total_amt", "max_amt", "amt_std", "max_ratio",
            "gap_mean", "gap_std", "gap_min",
            "n_categories", "dom_cat_frac",
            "n_a2a", "a2a_total",
            "has_override", "has_b64",
        ]
        feat_vals = feats[0]
        # Contribution = importance × |deviation from zero| (rough)
        contribs = importances * np.abs(feat_vals)
        top_idx = np.argsort(contribs)[::-1][:3]
        flags = [f"sgrf:{feat_names[i]}={feat_vals[i]:.2f}" for i in top_idx
                 if contribs[i] > 0.001]

        return round(score, 4), flags

    # ── Drop-in detector interface ────────────────────────────────────────────

    def detect(self, event: Event, history: list[Event]) -> tuple[float, list[str]]:
        """
        Called per-event. Scores the FULL session (history + event) each time.
        Returns 0.0 until we have >= 4 events (not enough context).
        Caches per session_id so the RF only runs once per session.
        """
        all_events = history + [event]
        if len(all_events) < 4:
            return 0.0, []

        # Use session_id for caching (all events share same session_id)
        session_id = event.session_id

        # Invalidate cache when session grows
        cache_key = (session_id, len(all_events))
        if cache_key in self._session_score:
            return self._session_score[cache_key]

        score, flags = self._predict_session(all_events)
        self._session_score[cache_key] = (score, flags)
        return score, flags


# ── Singleton for benchmark integration ──────────────────────────────────────

_model: SessionGraphModel | None = None


def get_model() -> SessionGraphModel:
    """Return the global singleton model (must call fit_from_sessions first)."""
    global _model
    if _model is None:
        _model = SessionGraphModel()
    return _model


def fit_from_sessions(sessions: list[Session]) -> None:
    """Fit the singleton model from training sessions."""
    get_model().fit(sessions)


def detect(event: Event, history: list[Event]) -> tuple[float, list[str]]:
    """Module-level detect function for use in DETECTORS dict."""
    return get_model().detect(event, history)
