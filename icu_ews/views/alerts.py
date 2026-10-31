"""Critical alerts workspace."""

from __future__ import annotations

import math

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ..config import ALERT_META, PALETTE
from ..data import repository
from ..data import alerts as A
from ..state import navigate, thresholds
from ..ui import charts
from ..ui import components as C


_CATEGORY_ORDER = ["critical", "high", "moderate", "info"]
_CATEGORY_LABELS = {key: ALERT_META[key]["label"] for key in _CATEGORY_ORDER}
_LABEL_CATEGORY = {v: k for k, v in _CATEGORY_LABELS.items()}

_PERIODS = {"Last 24 hours": 1, "Last 7 days": 7, "Full cohort": None}

_SORTS = {
    "Severity · newest": "severity",
    "Newest first": "newest",
    "Oldest first": "oldest",
    "Patient ID": "pid",
}

_STATUS_TONE = {
    "Open": "red",
    "Acknowledged": "indigo",
    "Resolved": "green",
}


def _stat(label: str, value: str, sub: str, icon_name: str, color: str) -> str:
    return f"""
<div class="astat" style="--astat-color:{color}">
  <div class="astat-top"><span>{label}</span>{C.icon_html(icon_name, 14)}</div>
  <div class="astat-value">{value}</div>
  <div class="astat-sub">{sub}</div>
</div>"""


def _alert_card(alert: pd.Series, reference_ts) -> str:
    meta = ALERT_META.get(alert["category"], ALERT_META["info"])
    tone = _STATUS_TONE.get(alert["status"], "muted")
    acknowledged = alert.get("acknowledged_at")
    audit = ""
    if alert["resolved"]:
        audit = f'resolved {C.relative_time(pd.Timestamp(alert["resolved_at"]), pd.Timestamp.utcnow())}'
    elif acknowledged:
        audit = f'acknowledged {C.relative_time(pd.Timestamp(acknowledged), pd.Timestamp.utcnow())}'
    return f"""
<div class="alert-card" style="--alert-color:{meta['color']}">
  <div class="alert-top">
    <div style="min-width:0">
      <div style="display:flex;align-items:center;gap:.55rem;flex-wrap:wrap">
        {C.alert_badge(alert['category'])}
        <span class="alert-title">{alert['title']}</span>
      </div>
      <div class="alert-desc">{alert['detail']}</div>
    </div>
    <div style="text-align:right;flex:0 0 auto">
      <div class="mono" style="font-size:.92rem;font-weight:650;color:#f1f7ff">{alert['value']}</div>
      <div class="text-xs dim" style="margin-top:2px" title="{C.fmt_ts(alert['ts'], with_year=True)}">
        {C.relative_time(alert['ts'], reference_ts)}
      </div>
    </div>
  </div>
  <div class="alert-meta">
    <span class="pid mono" style="color:#9fb6d4">{alert['patient_id']}</span>
    <span>·</span><span>{alert.get('bed') or '—'}</span>
    <span>·</span><span>{alert['metric']} rule</span>
    <span>·</span><span>{C.fmt_ts(alert['ts'], with_year=True)}</span>
    <span style="margin-left:auto;display:inline-flex;align-items:center;gap:.4rem">{C.chip(alert['status'], tone, dot=True)}{audit}</span>
  </div>
</div>
"""


def _mix_donut(open_alerts: pd.DataFrame) -> go.Figure:
    counts = [int((open_alerts["category"] == key).sum()) for key in _CATEGORY_ORDER]
    total = sum(counts)
    fig = go.Figure(
        go.Pie(
            labels=[_CATEGORY_LABELS[key] for key in _CATEGORY_ORDER],
            values=counts,
            hole=0.7,
            sort=False,
            marker=dict(
                colors=[ALERT_META[key]["color"] for key in _CATEGORY_ORDER],
                line=dict(color="#0b1524", width=2),
            ),
            textinfo="none",
            hovertemplate="%{label}: %{value} alerts<extra></extra>",
        )
    )
    fig.update_layout(
        height=184,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        annotations=[
            dict(text=f"<b>{total}</b>", x=0.5, y=0.54, font=dict(size=25, color="#f2f8ff"), showarrow=False),
            dict(text="open", x=0.5, y=0.4, font=dict(size=10, color=PALETTE["text_dim"]), showarrow=False),
        ],
    )
    return fig


def _source_rows(open_alerts: pd.DataFrame) -> str:
    if open_alerts.empty:
        return '<div class="pop-empty">No open alerts.</div>'
    counts = open_alerts["metric"].value_counts().head(6)
    peak = int(counts.max()) or 1
    rows = []
    for metric, count in counts.items():
        rows.append(
            f'<div class="queue-row"><div style="flex:1;min-width:0">'
            f'<div class="queue-id" style="font-size:.72rem">{metric}</div></div>'
            f'<div class="queue-bar" style="max-width:110px"><span style="width:{count / peak * 100:.0f}%;'
            f'background:linear-gradient(90deg,{PALETTE["cyan"]}88,{PALETTE["cyan"]})"></span></div>'
            f'<span class="queue-prob" style="color:#bcd7f2">{int(count)}</span></div>'
        )
    return "".join(rows)


def _reset_filters() -> None:
    for key in ("al_search", "al_category", "al_status", "al_period", "al_sort", "al_page"):
        st.session_state.pop(key, None)


def _filter(frame: pd.DataFrame, reference_ts, category: str, status: str, period: str, query: str) -> pd.DataFrame:
    out = frame
    days = _PERIODS[period]
    if days is not None:
        out = out[out["ts"] >= pd.Timestamp(reference_ts) - pd.Timedelta(days=days)]
    if category != "All":
        out = out[out["category"] == _LABEL_CATEGORY[category]]
    if status != "All":
        out = out[out["status"] == status]
    if query:
        needle = query.strip()
        haystack = out[["patient_id", "bed", "title", "detail", "value"]].astype(str)
        out = out[haystack.apply(lambda col: col.str.contains(needle, case=False, regex=False)).any(axis=1)]
    return out


def _sort(frame: pd.DataFrame, key: str) -> pd.DataFrame:
    if key == "newest":
        return frame.sort_values("ts", ascending=False)
    if key == "oldest":
        return frame.sort_values("ts", ascending=True)
    if key == "pid":
        return frame.sort_values(["patient_id", "severity_rank"])
    return frame.sort_values(["severity_rank", "ts"], ascending=[True, False])


def render() -> None:
    ss = st.session_state
    slug = ss.dataset
    th = thresholds()
    summary = repository.dataset_summary(slug)
    reference_ts = summary["latest_ts"]
    frame = repository.alerts(slug, th)

    open_alerts = frame[~frame["resolved"]]
    critical_open = int((open_alerts["category"] == "critical").sum())
    high_open = int((open_alerts["category"] == "high").sum())
    acknowledged = int(((open_alerts["status"] == "Acknowledged")).sum())
    resolved = int(frame["resolved"].sum())
    resolution_rate = resolved / len(frame) if len(frame) else 0.0
    high_share = (critical_open + high_open) / len(open_alerts) if len(open_alerts) else 0.0

    C.page_header(
        "Critical alerts",
        "Alert queue derived from recorded vital signs, MEWS and model-scored deterioration risk at the latest "
        "assessment. Review, acknowledge and resolve alerts to track the demonstration workflow.",
        chips=(
            C.chip(f"{len(open_alerts)} open", "red", dot=True)
            + C.chip(f"{critical_open} critical", "red")
            + C.chip(f"{high_open} high", "amber")
            + C.chip(f"{resolved} resolved", "green")
            + C.chip(f"Snapshot {pd.Timestamp(reference_ts):%d %b %Y %H:%M}", "muted")
        ),
    )
    C.demo_banner(
        "These alerts are synthesised from the demonstration cohort and are not transmitted to any clinical "
        "system. Acknowledgement and resolution state is stored locally for this demo workspace only."
    )

    specs = [
        ("Open queue", str(len(open_alerts)), f"{high_share:.0%} are critical or high", "bell-ring", PALETTE["critical"]),
        ("Critical", str(critical_open), "immediate review band", "alert-octagon", PALETTE["critical"]),
        ("High", str(high_open), "escalated review band", "alert-triangle", PALETTE["high"]),
        ("In review", str(acknowledged), "acknowledged, not resolved", "eye", PALETTE["indigo"]),
        ("Resolved", f"{resolution_rate:.0%}", f"{resolved} of {len(frame)} alerts", "check-circle", PALETTE["low"]),
    ]
    for col, spec in zip(st.columns(5, gap="small"), specs):
        with col:
            C.md(_stat(*spec))

    with st.container(key="toolbar_alerts"):
        f1, f2, f3, f4, f5, f6 = st.columns([2.0, 2.2, 1.3, 1.55, 1.65, 0.8], gap="small")
        query = f1.text_input("Find alert", key="al_search", placeholder="Patient, bed, metric…")
        category = f2.selectbox(
            "Category", ["All", *[_CATEGORY_LABELS[k] for k in _CATEGORY_ORDER]], key="al_category"
        )
        status = f3.selectbox("Status", ["Open", "Acknowledged", "Resolved", "All"], key="al_status")
        period = f4.selectbox("Generated", list(_PERIODS), key="al_period")
        sort_label = f5.selectbox("Sort", list(_SORTS), key="al_sort")
        f6.markdown('<div style="height:.55rem"></div>', unsafe_allow_html=True)
        f6.button("Reset", key="al_reset", use_container_width=True, on_click=_reset_filters)

    filtered = _sort(_filter(frame, reference_ts, category, status, period, query), _SORTS[sort_label])

    main, rail = st.columns([1.95, 1], gap="large")

    with main:
        head, export = st.columns([2.7, 1], gap="small")
        with head:
            C.section_header(
                "Alert queue",
                f"{len(filtered)} alerts match the current filters · {status.lower()} · {period.lower()}",
                "bell-ring",
            )
        with export:
            st.download_button(
                "Export filtered CSV",
                data=filtered.drop(columns=["acknowledged_at", "resolved_at"], errors="ignore").to_csv(index=False),
                file_name="icu_ews_filtered_alerts.csv",
                mime="text/csv",
                key="al_export",
                use_container_width=True,
            )
        C.md(
            '<div class="panel-note" style="margin:-.3rem 0 .55rem">Relative ages are measured from the dataset '
            f'snapshot ({pd.Timestamp(reference_ts):%d %b %Y %H:%M}), not the wall clock.</div>'
        )

        if filtered.empty:
            C.empty_state(
                "No alerts match these filters",
                "Change the category, status or generated window, or clear the search term.",
                "bell-off",
            )
            C.md('<div class="panel-note">Acknowledgement and resolution state is retained per alert ID.</div>')
        else:
            size = 8
            total_pages = max(1, math.ceil(len(filtered) / size))
            page = min(max(int(ss.get("al_page", 1)), 1), total_pages)
            ss["al_page"] = page
            page_rows = filtered.iloc[(page - 1) * size : page * size]

            for _, alert in page_rows.iterrows():
                alert_id = alert["id"]
                C.md(_alert_card(alert, reference_ts))
                b1, b2, b3, _ = st.columns([1, 1, 1.35, 2.6], gap="small")
                is_resolved = bool(alert["resolved"])
                is_ack = bool(alert["acknowledged"])
                if is_resolved:
                    if b1.button("Reopen", key=f"al_reopen_{alert_id}", use_container_width=True):
                        A.set_resolved(alert_id, False)
                        A.set_acknowledged(alert_id, False)
                        st.rerun()
                else:
                    label = "Acknowledged" if is_ack else "Acknowledge"
                    if b1.button(label, key=f"al_ack_{alert_id}", disabled=is_ack, use_container_width=True):
                        A.set_acknowledged(alert_id, True)
                        st.rerun()
                if b2.button(
                    "Resolved" if is_resolved else "Resolve",
                    key=f"al_resolve_{alert_id}",
                    disabled=is_resolved,
                    use_container_width=True,
                ):
                    A.set_resolved(alert_id, True)
                    st.rerun()
                if b3.button("Open patient", key=f"al_open_{alert_id}", use_container_width=True):
                    navigate("patient", pid=str(alert["patient_id"]))

            C.pager("al", total_pages, page, len(page_rows), len(filtered))

    with rail:
        with C.panel("al_mix", "Alert mix", f"{len(open_alerts)} unresolved alerts by category", "pie-chart"):
            current = open_alerts
            if current.empty:
                C.md('<div class="pop-empty">All alerts are resolved.</div>')
            else:
                st.plotly_chart(_mix_donut(current), use_container_width=True, config=charts.PLOT_CONFIG)
                counts = [int((current["category"] == key).sum()) for key in _CATEGORY_ORDER]
                C.md(
                    C.legend_html(
                        [(_CATEGORY_LABELS[key], ALERT_META[key]["color"]) for key in _CATEGORY_ORDER],
                        {_CATEGORY_LABELS[key]: str(count) for key, count in zip(_CATEGORY_ORDER, counts)},
                    )
                )
            C.md(
                f'<div class="panel-note">Model-derived alerts use the active thresholds: critical ≥ {th["critical"]:.0%}, '
                f'high ≥ {th["high"]:.0%}. Thresholds are adjustable in Settings.</div>'
            )

        with C.panel("al_workflow", "Review workflow", "Acknowledged and resolved alerts across the full cohort", "workflow"):
            awaiting = len(open_alerts) - acknowledged
            C.md(
                C.stat_row("Awaiting review", str(awaiting), "inbox", PALETTE["critical"])
                + C.stat_row("Acknowledged", str(acknowledged), "eye", PALETTE["indigo"])
                + C.stat_row("Resolved", str(resolved), "check-circle", PALETTE["low"])
                + C.stat_row("Resolution rate", f"{resolution_rate:.0%}", "trending-up", PALETTE["cyan"])
            )
            C.md(
                f'<div class="panel-note" style="display:flex;align-items:center;gap:.5rem">'
                f'{C.completion_bar(resolution_rate, width=140)} <span>{resolved} / {len(frame)} alerts resolved</span></div>'
            )

        with C.panel("al_sources", "Top alert sources", "Unresolved alerts by measured metric", "activity"):
            C.md(_source_rows(open_alerts))

        with C.panel("al_recent", "Recently resolved", "Most recent resolutions in this workspace", "history"):
            recent = frame[frame["resolved"]].copy()
            if recent.empty:
                C.md('<div class="pop-empty">No alerts have been resolved yet.</div>')
            else:
                recent["_resolved"] = pd.to_datetime(recent["resolved_at"], utc=True)
                recent = recent.sort_values("_resolved", ascending=False).head(4)
                for _, alert in recent.iterrows():
                    C.md(
                        f'<div class="queue-row"><div style="flex:1;min-width:0">'
                        f'<div class="queue-id" style="font-size:.72rem">{alert["patient_id"]}</div>'
                        f'<div class="queue-sub">{alert["title"]}</div></div>'
                        f'<span class="dim text-xs">{C.fmt_ts(alert["_resolved"], with_year=False)}</span></div>'
                    )
            C.md('<div class="panel-note">Resolution timestamps use this workspace\'s local clock.</div>')
