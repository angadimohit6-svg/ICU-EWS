"""Application chrome: fixed top bar, sidebar navigation and routing entry."""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import streamlit as st

from .config import APP_LONG_NAME, APP_NAME, NAV_ITEMS
from .data import repository
from .icons import icon
from .ml import artifacts
from .state import navigate, switch_dataset, thresholds
from .ui.components import feed_item, md

_MATERIAL = {
    "overview": "dashboard",
    "monitoring": "monitor_heart",
    "alerts": "notifications_active",
    "prediction": "auto_awesome",
    "analytics": "bar_chart",
    "xai": "psychology",
    "performance": "speed",
    "reports": "description",
    "settings": "tune",
}

_NAV_GROUPS = [
    ("Operations", ["overview", "monitoring", "alerts"]),
    ("Intelligence", ["prediction", "analytics", "xai", "performance"]),
    ("Workspace", ["reports", "settings"]),
]

_NAV_BY_KEY = {item["key"]: item for item in NAV_ITEMS}


@st.fragment(run_every="30s")
def _clock() -> None:
    now = datetime.now()
    md(
        f'<div class="clock"><div class="clock-time">{now:%H:%M:%S}</div>'
        f'<div class="clock-date">{now:%a %d %b %Y}</div></div>'
    )


def _brand_html(summary: dict) -> str:
    return f"""
<div class="brand">
  <div class="brand-mark">{icon('heart-pulse', 22, '#7ff3ff')}</div>
  <div>
    <div class="brand-name">{APP_NAME}<span class="brand-live" style="color:#fbbf24;
      background:rgba(251,191,36,.1);border-color:rgba(251,191,36,.34);-webkit-text-fill-color:#fbbf24"
      title="Synthetic demonstration dataset. Not a live hospital feed.">
      <span class="pulse-dot" style="background:#fbbf24;box-shadow:0 0 0 0 rgba(251,191,36,.6)"></span>DEMO DATA</span></div>
    <div class="brand-sub">{APP_LONG_NAME}</div>
  </div>
</div>
"""


def _dataset_changed() -> None:
    switch_dataset(st.session_state["dataset_pick"])


def _run_search(query: str) -> None:
    q = (query or "").strip()
    if not q:
        return
    slug = st.session_state.dataset
    snap = repository.patient_snapshot(slug)
    needle = q.upper()
    exact = snap[snap["patient_id"].str.upper() == needle]
    match = exact if len(exact) else snap[snap["patient_id"].str.upper().str.contains(needle, regex=False)]
    if len(match):
        navigate("patient", pid=str(match.iloc[0]["patient_id"]))
    else:
        st.session_state["mon_search"] = q
        navigate("monitoring")


def topbar() -> None:
    slug = st.session_state.dataset
    summary = repository.dataset_summary(slug)
    bundles = artifacts.list_bundles()
    labels = {b["slug"]: (b.get("short_label") or b.get("label") or b["slug"]) for b in bundles}
    if slug not in labels:
        labels[slug] = slug
    options = list(labels)
    reference_ts = summary["latest_ts"]

    with st.container(key="topnav"):
        cols = st.columns([3.6, 1.1, 2.15, 2.95, 0.95, 0.95], gap="small")

        with cols[0]:
            md(_brand_html(summary))

        with cols[1]:
            _clock()

        with cols[2]:
            st.selectbox(
                "Dataset",
                options,
                index=options.index(slug) if slug in options else 0,
                format_func=lambda s: labels[s],
                key="dataset_pick",
                label_visibility="collapsed",
                on_change=_dataset_changed,
            )

        with cols[3]:
            with st.form("global_search", border=False):
                inner = st.columns([4.2, 1.35], gap="small")
                query = inner[0].text_input(
                    "Global patient search",
                    key="top_q",
                    placeholder="Search patient ID or bed…",
                    label_visibility="collapsed",
                )
                submitted = inner[1].form_submit_button("Search", icon=":material/search:", use_container_width=True)
            if submitted:
                _run_search(query)

        with cols[4]:
            alerts = repository.alerts(slug, thresholds())
            open_alerts = alerts[~alerts["resolved"]]
            active = open_alerts[
                (open_alerts["ts"] >= reference_ts - pd.Timedelta(hours=24))
                & (open_alerts["category"].isin(["critical", "high"]))
            ]
            unread = int(len(active))
            with st.container(key="bellwrap"):
                if unread:
                    md(f'<span class="bell-badge">{unread if unread < 100 else "99+"}</span>')
                with st.popover("Alerts", icon=":material/notifications:", use_container_width=True):
                    md(
                        f'<div class="pop-head"><span>{icon("bell-ring", 16, "#22d3ee")} Open alerts</span>'
                        f'<b>{len(active)}</b></div>'
                        f'<div class="pop-body dim">Critical and high alerts raised in the last 24 hours.</div>'
                    )
                    recent = active.sort_values("ts", ascending=False).head(4)
                    if recent.empty:
                        md('<div class="pop-empty">No unresolved alerts.</div>')
                    for _, alert in recent.iterrows():
                        if feed_item_clickable(alert, reference_ts):
                            navigate("patient", pid=str(alert["patient_id"]))
                    if st.button("Open alert centre", key="top_alerts_all", use_container_width=True):
                        navigate("alerts")

        with cols[5]:
            with st.popover("Profile", icon=":material/account_circle:", use_container_width=True):
                with st.container(key="profile_menu"):
                    md(
                        f'<div class="pop-head"><span>{icon("user", 16, "#22d3ee")} Session</span>'
                        f'<span class="chip" style="font-size:.56rem;padding:1px 7px">Demonstration</span></div>'
                        f'<div class="pop-profile-title">Clinical research workspace</div>'
                        f'<div class="pop-profile-row"><span>Dataset</span><b>{summary["label"]}</b></div>'
                        f'<div class="pop-profile-row"><span>Model</span>'
                        f'<b>{summary["training"].get("model_version", "—")}</b></div>'
                    )
                    c1, c2 = st.columns(2)
                    if c1.button("Settings", key="top_profile_settings", use_container_width=True):
                        navigate("settings")
                    if c2.button("Reports", key="top_profile_reports", use_container_width=True):
                        navigate("reports")


def feed_item_clickable(alert: pd.Series, reference_ts) -> bool:
    from .ui.components import clickable

    return clickable(feed_item(alert, reference_ts), key=f"feed_{alert['id']}", label="Open patient")


def sidebar(active: str) -> None:
    with st.sidebar:
        for caption, keys in _NAV_GROUPS:
            md(f'<div class="side-caption">{caption}</div>')
            for key in keys:
                item = _NAV_BY_KEY[key]
                if st.button(
                    item["label"],
                    key=f"nav_{key}",
                    icon=f":material/{_MATERIAL[key]}:",
                    use_container_width=True,
                ):
                    navigate(key)

        summary = repository.dataset_summary(st.session_state.dataset)
        model_version = summary["training"].get("model_version", "—")
        md(
            f'<div class="side-demo"><div class="side-demo-title">{icon("sparkles", 12)} Demonstration workspace</div>'
            f'<div class="side-demo-body">Synthetic ICU cohort · {summary["n_patients"]} patients · '
            f'{summary["event_count"]} documented deterioration events. Predictions come from models trained '
            f'locally on this cohort. No live hospital connection.</div></div>'
        )
        md(
            f'<div class="side-foot"><div class="side-foot-avatar">EW</div><div>'
            f'<div class="side-foot-name">Clinical Lead</div>'
            f'<div class="side-foot-role">Demo session · {model_version}</div></div></div>'
            f'<div class="side-version">{icon("lock", 11)} Local inference · no PHI</div>'
        )
