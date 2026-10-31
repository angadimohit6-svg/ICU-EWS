"""Bedside patient monitoring workspace."""

from __future__ import annotations

import math

import pandas as pd
import streamlit as st

from ..analytics import band_counts, with_bands
from ..config import RISK_META, RISK_ORDER
from ..data import repository
from ..state import navigate, thresholds
from ..ui import components as C


_SORT = {
    "Highest risk": "risk_desc",
    "Lowest risk": "risk_asc",
    "Most recent prediction": "recent",
    "Highest MEWS": "mews",
    "Patient ID": "pid",
}


def _reset() -> None:
    ss = st.session_state
    ss["mon_search"] = ""
    ss["mon_band"] = "All"
    ss["mon_unit"] = "All units"
    ss["mon_sort"] = "Highest risk"
    ss["mon_period"] = "Latest"
    ss["mon_page"] = 1


def _sort(frame: pd.DataFrame, key: str) -> pd.DataFrame:
    if key == "risk_asc":
        return frame.sort_values("risk_prob", ascending=True)
    if key == "recent":
        return frame.sort_values("predicted_ts", ascending=False)
    if key == "mews":
        return frame.sort_values(["mews", "risk_prob"], ascending=[False, False])
    if key == "pid":
        return frame.sort_values("patient_id")
    return frame.sort_values("risk_prob", ascending=False)


def render() -> None:
    ss = st.session_state
    slug = ss.dataset
    th = thresholds()
    summary = repository.dataset_summary(slug)
    snap = with_bands(repository.patient_snapshot(slug), th)
    reference_ts = summary["latest_ts"]
    primary = summary["training"].get("primary_model", "Primary model")
    brier = (summary.get("metrics") or {}).get(primary, {}).get("brier")

    counts = band_counts(snap, th)
    C.page_header(
        "Patient monitoring",
        "Bedside view of every monitored patient with the latest physiology, model-scored risk and "
        "data completeness for the current assessment window.",
        chips=(
            "".join(
                C.chip(f"{RISK_META[b]['label']} · {counts[b]}", "red" if b == "critical" else ("amber" if b in {"high", "moderate"} else "green"))
                for b in RISK_ORDER
            )
            + C.chip(f"Snapshot {pd.Timestamp(reference_ts):%d %b %Y %H:%M}", "muted")
        ),
    )
    C.demo_banner(
        "Every card is generated from the synthetic demonstration cohort. Risk is the model's predicted "
        "probability of a documented deterioration event within 24 hours of the assessment."
    )

    with st.container(key="toolbar_monitoring"):
        f1, f2, f3, f4, f5, f6 = st.columns([2.1, 2.5, 1.25, 1.55, 1.0, 0.8], gap="small")
        query = f1.text_input("Find patient", key="mon_search", placeholder="Search ID or bed…")
        band = f2.segmented_control("Risk band", ["All", *[RISK_META[b]["short"] for b in RISK_ORDER]], key="mon_band") or "All"
        units = ["All units"] + sorted(snap["bed"].dropna().str.split("-").str[0].unique())
        unit = f3.selectbox("Unit", units, key="mon_unit")
        sort_label = f4.selectbox("Sort", list(_SORT), key="mon_sort")
        period = f5.selectbox("Assessment window", ["Latest", "Last 24h", "7 days"], key="mon_period")
        f6.markdown('<div style="height:.55rem"></div>', unsafe_allow_html=True)
        f6.button("Reset", key="mon_reset", use_container_width=True, on_click=_reset)

    frame = snap.copy()
    if period == "Last 24h":
        frame = frame[frame["snapshot_ts"] >= reference_ts - pd.Timedelta(hours=24)]
    elif period == "7 days":
        frame = frame[frame["snapshot_ts"] >= reference_ts - pd.Timedelta(days=7)]
    if band != "All":
        frame = frame[frame["risk_band"] == next(b for b in RISK_ORDER if RISK_META[b]["short"] == band)]
    if unit != "All units":
        frame = frame[frame["bed"].str.startswith(f"{unit}-", na=False)]
    if query:
        needle = query.strip()
        frame = frame[
            frame["patient_id"].str.contains(needle, case=False, regex=False)
            | frame["bed"].str.contains(needle, case=False, regex=False)
        ]
    frame = _sort(frame, _SORT[sort_label])

    size = 9
    total_pages = max(1, math.ceil(len(frame) / size)) if len(frame) else 1
    page = min(max(int(ss.get("mon_page", 1)), 1), total_pages)
    ss["mon_page"] = page
    page_rows = frame.iloc[(page - 1) * size : page * size]

    C.section_header(
        "Monitored patients",
        f"{len(frame)} patients · {size} cards per page · sorted by {sort_label.lower()}",
        "monitor-heart",
    )
    if page_rows.empty:
        C.empty_state("No patients match these filters", "Widen the risk band, unit or search term.", "search")
        return

    for start in range(0, len(page_rows), 3):
        chunk = page_rows.iloc[start : start + 3]
        grid = st.columns(3, gap="medium")
        for col, (_, patient) in zip(grid, chunk.iterrows()):
            with col:
                if C.patient_card(patient, patient["risk_band"], reference_ts, brier):
                    navigate("patient", pid=str(patient["patient_id"]))

    C.pager("mon", total_pages, page, len(page_rows), len(frame))
