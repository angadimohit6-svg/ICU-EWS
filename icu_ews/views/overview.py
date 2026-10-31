"""Executive monitoring overview."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import streamlit as st

from ..analytics import band_counts, kpi_metrics, measurement_alerts, period_window, with_bands
from ..config import PALETTE, RISK_META, RISK_ORDER
from ..data import repository
from ..state import navigate, thresholds
from ..ui import charts
from ..ui import components as C


_PERIODS = {
    "Last 24 hours": 1,
    "Last 7 days": 7,
    "Last 30 days": 30,
    "Full cohort": None,
}

_SORTS = {
    "Highest risk": "risk_desc",
    "Lowest risk": "risk_asc",
    "Recent prediction": "recent",
    "Patient ID": "pid",
    "Lowest completeness": "completeness",
}


def _filtered(snap: pd.DataFrame, start, end, band: str, query: str) -> pd.DataFrame:
    frame = snap[(snap["snapshot_ts"] >= start) & (snap["snapshot_ts"] <= end)]
    if band != "All":
        frame = frame[frame["risk_band"] == band.lower()]
    if query:
        frame = frame[frame["patient_id"].str.contains(query.strip(), case=False, regex=False)]
    return frame


def _sorted(frame: pd.DataFrame, key: str) -> pd.DataFrame:
    if key == "risk_asc":
        return frame.sort_values(["risk_prob", "patient_id"], ascending=[True, True])
    if key == "recent":
        return frame.sort_values("predicted_ts", ascending=False)
    if key == "pid":
        return frame.sort_values("patient_id")
    if key == "completeness":
        return frame.sort_values(["completeness", "risk_prob"], ascending=[True, False])
    return frame.sort_values(["risk_prob", "patient_id"], ascending=[False, True])


def _reset_filters() -> None:
    ss = st.session_state
    ss["ov_search"] = ""
    ss["ov_band"] = "All"
    ss["ov_period"] = "Last 7 days"
    ss["ov_sort"] = "Highest risk"
    ss["ov_page_size"] = 6
    ss["ov_page"] = 1


def _pager(total_pages: int, page: int, shown: int, total: int) -> None:
    with st.container(key="ov_pager"):
        left, mid, right = st.columns([1, 2.4, 1])
        if left.button("Previous", key="ov_prev", disabled=page <= 1, use_container_width=True):
            st.session_state["ov_page"] = page - 1
            st.rerun()
        mid.markdown(
            f'<div class="pager-info"><b>Page {page}</b> of {total_pages}'
            f'<span> · showing {shown} of {total} patients</span></div>',
            unsafe_allow_html=True,
        )
        if right.button("Next", key="ov_next", disabled=page >= total_pages, use_container_width=True):
            st.session_state["ov_page"] = page + 1
            st.rerun()


def render() -> None:
    ss = st.session_state
    slug = ss.dataset
    th = thresholds()
    summary = repository.dataset_summary(slug)
    snap = with_bands(repository.patient_snapshot(slug), th)
    reference_ts = summary["latest_ts"]
    primary = summary["training"].get("primary_model", "Primary model")
    brier = (summary.get("metrics") or {}).get(primary, {}).get("brier")

    period_label = ss.get("ov_period", "Last 7 days")
    days = _PERIODS.get(period_label, 7)
    start, end = period_window(reference_ts, days)

    C.page_header(
        "ICU command centre",
        "Executive monitoring overview of the demonstration cohort: model-scored deterioration risk, "
        "open clinical alerts and the highest-priority patients at the latest assessment.",
        chips=(
            C.chip(summary["label"], "cyan", dot=True)
            + C.chip(period_label, "muted")
            + C.chip(f"Snapshot {pd.Timestamp(reference_ts):%d %b %Y %H:%M}", "muted")
            + C.chip(f"{primary} · Brier {brier:.3f}" if brier else primary, "indigo")
        ),
    )
    C.demo_banner(
        "All patients, measurements and alerts shown here are synthetic. Risk scores are produced by "
        "models trained on this cohort; nothing is a live hospital feed or a clinical diagnosis."
    )

    cards = kpi_metrics(snap, th, reference_ts, days)
    cols = st.columns(4, gap="medium")
    for col, card in zip(cols, cards):
        with col:
            if C.kpi_card(card):
                mapping = {"monitored": "monitoring", "critical": "monitoring", "high": "monitoring", "deterioration": "analytics"}
                if card["key"] in {"critical", "high"}:
                    ss["mon_band"] = card["key"].capitalize()
                navigate(mapping[card["key"]])

    with st.container(key="toolbar_overview"):
        f1, f2, f3, f4, f5, f6 = st.columns([2.2, 2.5, 1.5, 1.7, 1.0, 0.85], gap="small")
        query = f1.text_input(
            "Find patient",
            key="ov_search",
            placeholder="Search pseudonymous ID…",
            label_visibility="visible",
        )
        band = f2.segmented_control(
            "Risk band",
            ["All", "Critical", "High", "Moderate", "Low"],
            key="ov_band",
            default="All",
        ) or "All"
        period = f3.selectbox("Period", list(_PERIODS), key="ov_period", index=1)
        sort_label = f4.selectbox("Sort", list(_SORTS), key="ov_sort", index=0)
        f5.selectbox("Cards per page", [4, 6, 8, 12], key="ov_page_size", index=1, format_func=lambda n: f"{n} / page")
        f6.markdown('<div style="height:.55rem"></div>', unsafe_allow_html=True)
        f6.button("Reset", key="ov_reset", use_container_width=True, on_click=_reset_filters)

    days = _PERIODS[period]
    start, end = period_window(reference_ts, days)
    cohort = snap[(snap["snapshot_ts"] >= start) & (snap["snapshot_ts"] <= end)]
    if query:
        # An explicit identifier search spans every assessment date so a known patient is never hidden by the period filter.
        filtered = _sorted(_filtered(snap, snap["snapshot_ts"].min(), snap["snapshot_ts"].max(), band, query), _SORTS[sort_label])
    else:
        filtered = _sorted(_filtered(snap, start, end, band, query), _SORTS[sort_label])

    size = int(ss.get("ov_page_size", 6))
    total_pages = max(1, math.ceil(len(filtered) / size)) if len(filtered) else 1
    page = min(max(int(ss.get("ov_page", 1)), 1), total_pages)
    ss["ov_page"] = page
    page_rows = filtered.iloc[(page - 1) * size : page * size]

    main, rail = st.columns([2.08, 1], gap="large")

    with main:
        C.section_header(
            "Patient cards",
            f"{len(filtered)} patients match the current filters"
            f"{' · searching all assessment dates' if query else ''} · sorted by {sort_label.lower()}",
            "grid",
        )
        if page_rows.empty:
            C.empty_state(
                "No patients match these filters",
                "Adjust the risk band, period or search term to widen the cohort.",
                "search",
            )
        else:
            for i in range(0, len(page_rows), 2):
                row = page_rows.iloc[i : i + 2]
                grid = st.columns(2, gap="medium")
                for col, (_, patient) in zip(grid, row.iterrows()):
                    with col:
                        if C.patient_card(patient, patient["risk_band"], reference_ts, brier):
                            navigate("patient", pid=str(patient["patient_id"]))
            _pager(total_pages, page, len(page_rows), len(filtered))

    with rail:
        counts = band_counts(cohort, th)
        with C.panel("risk_dist", "Risk distribution", f"{len(cohort)} patients in period · model thresholds", "target"):
            st.plotly_chart(charts.risk_donut(counts, height=196), use_container_width=True, config=charts.PLOT_CONFIG)
            C.md(
                C.legend_html(
                    [(RISK_META[b]["label"], RISK_META[b]["color"]) for b in RISK_ORDER],
                    {RISK_META[b]["label"]: str(counts[b]) for b in RISK_ORDER},
                )
            )
            C.md(
                f'<div class="panel-note">Critical ≥ {th["critical"]:.0%} · high ≥ {th["high"]:.0%} · '
                f'moderate ≥ {th["moderate"]:.0%} predicted probability</div>'
            )

        alerts = repository.alerts(slug, th)
        open_alerts = alerts[~alerts["resolved"]].sort_values("ts", ascending=False)
        with C.panel("critical_feed", "Critical alert feed", "Unresolved alerts, newest first", "bell-ring"):
            critical = (
                open_alerts[open_alerts["category"].isin(["critical", "high"])]
                .drop_duplicates("patient_id")
                .head(5)
            )
            if critical.empty:
                C.md('<div class="pop-empty">No unresolved critical or high alerts.</div>')
            for _, alert in critical.iterrows():
                if C.clickable(C.feed_item(alert, reference_ts), key=f"ov_alert_{alert['id']}", label="Open patient"):
                    navigate("patient", pid=str(alert["patient_id"]))
            if st.button("View all alerts", key="ov_all_alerts", use_container_width=True):
                navigate("alerts")

        high_risk = cohort.sort_values("risk_prob", ascending=False).head(6)
        with C.panel("high_queue", "High-risk queue", "Ranked by predicted 24h deterioration risk", "layers"):
            for rank, (_, patient) in enumerate(high_risk.iterrows(), start=1):
                if C.clickable(
                    C.queue_row(rank, patient, patient["risk_band"], reference_ts),
                    key=f"ov_queue_{patient['patient_id']}",
                    label="Open patient",
                ):
                    navigate("patient", pid=str(patient["patient_id"]))
            if st.button("Open monitoring view", key="ov_open_monitoring", use_container_width=True):
                navigate("monitoring")

        with C.panel("activity", "Recent prediction activity", "Latest model assessments", "activity"):
            recent = cohort.sort_values("predicted_ts", ascending=False).head(4)
            if recent.empty:
                C.md('<div class="pop-empty">No assessments in this period.</div>')
            for _, patient in recent.iterrows():
                if C.clickable(
                    C.activity_row(patient["patient_id"], patient["predicted_ts"], reference_ts, "risk assessment"),
                    key=f"ov_activity_{patient['patient_id']}",
                    label="Open patient",
                ):
                    navigate("patient", pid=str(patient["patient_id"]))

        measurements = measurement_alerts(cohort, limit=5)
        with C.panel("measurements", "Clinical measurement alerts", "Latest values beyond reference range", "thermometer"):
            if measurements.empty:
                C.md('<div class="pop-empty">All measured values are within range.</div>')
            for _, item in measurements.iterrows():
                if C.clickable(C.measurement_row(item), key=f"ov_meas_{item['patient_id']}_{item['label']}", label="Open patient"):
                    navigate("patient", pid=str(item["patient_id"]))
