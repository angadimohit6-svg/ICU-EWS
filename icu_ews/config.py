"""Application-wide constants, design tokens and paths for ICU-EWS."""

from __future__ import annotations

from pathlib import Path

APP_NAME = "ICU-EWS"
APP_LONG_NAME = "Intelligent ICU Early Warning System"
APP_TAGLINE = "Clinical deterioration intelligence"

ROOT = Path(__file__).resolve().parent.parent
ARTIFACT_DIR = ROOT / "artifacts"
BUNDLE_DIR = ARTIFACT_DIR / "bundles"
ALERT_STATE_FILE = ARTIFACT_DIR / "alert_state.json"
DEFAULT_BUNDLE = "demo-cohort-v1"

# ---------------------------------------------------------------------------
# Design tokens
# ---------------------------------------------------------------------------
PALETTE = {
    "bg": "#060b14",
    "bg_soft": "#081120",
    "surface": "#0b1524",
    "card": "#0f1a2b",
    "card_alt": "#132135",
    "elevated": "#16253c",
    "border": "#1b2a42",
    "border_strong": "#2a3d5c",
    "text": "#e8eefb",
    "text_muted": "#8ea3c0",
    "text_dim": "#5f728e",
    "cyan": "#22d3ee",
    "blue": "#3b82f6",
    "indigo": "#818cf8",
    "violet": "#a78bfa",
    "critical": "#f43f5e",
    "high": "#fb923c",
    "moderate": "#facc15",
    "low": "#22c55e",
    "info": "#38bdf8",
    "teal": "#2dd4bf",
    "pink": "#f472b6",
}

RISK_ORDER = ["critical", "high", "moderate", "low"]

RISK_META = {
    "critical": {
        "label": "Critical",
        "short": "Critical",
        "color": PALETTE["critical"],
        "soft": "rgba(244, 63, 94, 0.14)",
        "border": "rgba(244, 63, 94, 0.42)",
        "icon": "alert-octagon",
        "rank": 0,
        "note": "Immediate review recommended",
    },
    "high": {
        "label": "High risk",
        "short": "High",
        "color": PALETTE["high"],
        "soft": "rgba(251, 146, 60, 0.14)",
        "border": "rgba(251, 146, 60, 0.40)",
        "icon": "alert-triangle",
        "rank": 1,
        "note": "Escalated observation",
    },
    "moderate": {
        "label": "Moderate",
        "short": "Moderate",
        "color": PALETTE["moderate"],
        "soft": "rgba(250, 204, 21, 0.12)",
        "border": "rgba(250, 204, 21, 0.36)",
        "icon": "activity",
        "rank": 2,
        "note": "Heightened monitoring",
    },
    "low": {
        "label": "Low risk",
        "short": "Low",
        "color": PALETTE["low"],
        "soft": "rgba(34, 197, 94, 0.13)",
        "border": "rgba(34, 197, 94, 0.38)",
        "icon": "shield-check",
        "rank": 3,
        "note": "Routine observation",
    },
}

ALERT_META = {
    "critical": {"label": "Critical", "color": PALETTE["critical"], "soft": "rgba(244,63,94,0.14)"},
    "high": {"label": "High", "color": PALETTE["high"], "soft": "rgba(251,146,60,0.14)"},
    "moderate": {"label": "Moderate", "color": PALETTE["moderate"], "soft": "rgba(250,204,21,0.12)"},
    "info": {"label": "Informational", "color": PALETTE["info"], "soft": "rgba(56,189,248,0.12)"},
}

NAV_ITEMS = [
    {"key": "overview", "label": "Overview", "icon": "dashboard", "caption": "Command centre"},
    {"key": "monitoring", "label": "Patient Monitoring", "icon": "monitor-heart", "caption": "Bedside view"},
    {"key": "alerts", "label": "Critical Alerts", "icon": "bell-ring", "caption": "Alert management"},
    {"key": "prediction", "label": "Risk Prediction", "icon": "sparkles", "caption": "Live scoring"},
    {"key": "analytics", "label": "Clinical Analytics", "icon": "bar-chart", "caption": "Cohort insight"},
    {"key": "xai", "label": "Explainable AI", "icon": "brain", "caption": "Model reasoning"},
    {"key": "performance", "label": "Model Performance", "icon": "gauge", "caption": "Evaluation"},
    {"key": "reports", "label": "Research Reports", "icon": "file-text", "caption": "Exports"},
    {"key": "settings", "label": "Settings", "icon": "sliders", "caption": "Configuration"},
]

# Default operating thresholds on the predicted deterioration probability.
# Adjustable in Settings; persisted per session.
# Default operating thresholds on the predicted deterioration probability,
# calibrated for the current primary model and adjustable in Settings.
DEFAULT_THRESHOLDS = {"critical": 0.75, "high": 0.55, "moderate": 0.35}

PREDICTION_HORIZON_HOURS = 24
DEMO_BUNDLE_PREFIX = "demo"
DEFAULT_PAGE_SIZE = 6
PAGE_SIZE_OPTIONS = [4, 6, 8, 12]


def risk_from_probability(p: float, thresholds: dict[str, float]) -> str:
    """Map a predicted probability to a risk band using the active thresholds."""
    if p >= thresholds["critical"]:
        return "critical"
    if p >= thresholds["high"]:
        return "high"
    if p >= thresholds["moderate"]:
        return "moderate"
    return "low"
