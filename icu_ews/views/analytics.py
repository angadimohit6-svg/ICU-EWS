"""Clinical analytics workspace."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ..analytics import (
    admission_trend,
    correlation_matrix,
    subgroup_table,
    wilson_ci,
    with_bands,
)
from ..config import PALETTE, RISK_META, RISK_ORDER
from ..data import repository
from ..state import thresholds
from ..ui import charts
from ..ui import components as C

_VITALS = {
    "Heart rate": ("heart_rate", "bpm"),
    "Respiratory rate": ("respiratory_rate", "/min"),
    "SpO₂": ("spo2", "%"),
    "Systolic BP": ("systolic_bp", "mmHg"),
    "Temperature": ("temperature", "°C"),
    "MEWS": ("mews", "pts"),
}
_LABS = {
    "Lactate": ("lactate", "mmol/L"),
    "White cell count": ("wbc", "10⁹/L"),
    "Creatinine": ("creatinine", "mg/dL"),
    "C-reactive protein": ("crp", "mg/L"),
    "Platelets": ("platelets", "10⁹/L"),
    "Haemoglobin": ("hemoglobin", "g/dL"),
}
_CORR_LABELS = {
    "age": "Age",
    "los_hours": "LOS (h)",
    "heart_rate": "HR",
    "respiratory_rate": "RR",
    "spo2": "SpO₂",
    "systolic_bp": "SBP",
    "diastolic_bp": "DBP",
    "temperature": "Temp",
    "mews": "MEWS",
    "lactate": "Lactate",
    "wbc": "WBC",
    "creatinine": "Creatinine",
    "platelets": "Platelets",
    "hemoglobin": "Hb",
    "crp": "CRP",
    "bilirubin": "Bilirubin",
    "sodium": "Na⁺",
    "potassium": "K⁺",
    "risk_prob": "Predicted risk",
}


def _stat(label: str, value: str, sub: str, icon_name: str, color: str) -> str:
    return f"""
<div class="astat" style="--astat-color:{color}">
  <div class="astat-top"><span>{label}</span>{C.icon_html(icon_name, 14)}</div>
  <div class="astat-value">{value}</div>
  <div class="astat-sub">{sub}</div>
</div>"""


def _box(frame: pd.DataFrame, column: str, unit: str, th: dict[str, float], label: str = "") -> go.Figure:
    fig = go.Figure()
    for band in RISK_ORDER:
        values = frame.loc[frame["risk_band"] == band, column].dropna()
        if values.empty:
            continue
        meta = RISK_META[band]
        fig.add_trace(
            go.Box(
                y=values,
                name=meta["short"],
                marker_color=meta["color"],
                line_color=meta["color"],
                fillcolor=meta["soft"],
                boxpoints="outliers",
                whiskerwidth=0.5,
                hovertemplate=f"<b>{meta['short']}</b><br>%{{y:.1f}} {unit}<extra></extra>",
            )
        )
    charts.apply_theme(
        fig, height=330, legend=False, hovermode="closest", margins=(8, 8, 12, 8),
        y_title=f"{label} ({unit})" if label else unit,
    )
    fig.update_layout(boxgap=0.3, showlegend=False)
    return fig


def _reset_filters(min_date, max_date) -> None:
    ss = st.session_state
    ss["an_dates"] = (min_date, max_date)
    ss["an_types"] = []
    ss["an_sex"] = "All"
    ss["an_age"] = []
    ss["an_vital"] = "Heart rate"
    ss["an_lab"] = "Lactate"
    ss["an_by"] = "Admission type"
    ss["an_x"] = "Heart rate"
    ss["an_y"] = "SpO₂"


def render() -> None:
    slug = st.session_state.dataset
    th = thresholds()
    summary = repository.dataset_summary(slug)
    snap = with_bands(repository.patient_snapshot(slug), th)
    snap["admit_date"] = snap["admit_ts"].dt.date
    min_date, max_date = snap["admit_date"].min(), snap["admit_date"].max()

    C.page_header(
        "Clinical analytics",
        "Cohort-level description of admissions, physiology, laboratory values and documented outcomes. "
        "All statistics are recomputed from the active dataset and current filters.",
        chips=(
            C.chip(summary["label"], "cyan", dot=True)
            + C.chip(f"{len(snap)} patients", "muted")
            + C.chip(f"Target: {summary['training'].get('target', '24h deterioration')}", "indigo")
        ),
    )
    C.demo_banner(
        "The cohort is synthetic. Admission dates are simulated to create a twelve-day operational window; "
        "statistics describe this demonstration dataset only."
    )

    with st.container(key="toolbar_analytics"):
        c1, c2, c3, c4, c5 = st.columns([1.5, 1.8, 0.9, 1.5, 1.4], gap="small")
        dates = c1.date_input(
            "Admission dates", value=(min_date, max_date), min_value=min_date, max_value=max_date, key="an_dates"
        )
        types = c2.multiselect("Admission type", sorted(snap["admission_type"].dropna().unique()), key="an_types")
        sex = c3.segmented_control("Sex", ["All", "F", "M"], key="an_sex") or "All"
        age_groups = c4.multiselect(
            "Age group", [str(a) for a in snap["age_group"].cat.categories], key="an_age"
        )
        c5.markdown('<div style="height:.55rem"></div>', unsafe_allow_html=True)
        c5.button(
            "Reset filters",
            key="an_reset",
            use_container_width=True,
            on_click=_reset_filters,
            args=(min_date, max_date),
        )

    if isinstance(dates, (tuple, list)) and len(dates) == 2:
        start, end = dates
    else:
        start = end = dates
    frame = snap[(snap["admit_date"] >= start) & (snap["admit_date"] <= end)]
    if types:
        frame = frame[frame["admission_type"].isin(types)]
    if sex != "All":
        frame = frame[frame["sex"] == sex]
    if age_groups:
        frame = frame[frame["age_group"].astype(str).isin(age_groups)]

    if frame.empty:
        C.empty_state("No admissions in this selection", "Broaden the date range or clear admission filters.", "calendar")
        return

    events = int(frame["deterioration_24h"].sum())
    rate = events / len(frame)
    ci_low, ci_high = wilson_ci(events, len(frame))
    critical_share = float((frame["risk_prob"] >= th["critical"]).mean())
    tiles = st.columns(5, gap="medium")
    specs = [
        ("Admissions", f"{len(frame)}", f"{start:%d %b} – {end:%d %b %Y}", "users", PALETTE["cyan"]),
        ("Deterioration events", f"{events}", "documented within 24h of assessment", "alert-octagon", PALETTE["critical"]),
        ("Event rate", f"{rate*100:.1f}%", f"95% CI {ci_low:.1f}–{ci_high:.1f}%", "activity", PALETTE["indigo"]),
        ("Mean length of stay", f"{frame['los_hours'].mean():.0f} h", f"median {frame['los_hours'].median():.0f} h", "clock", PALETTE["teal"]),
        ("Critical band share", f"{critical_share*100:.1f}%", f"≥ {th['critical']:.0%} predicted risk", "shield-alert", PALETTE["high"]),
    ]
    for col, (label, value, sub, icon_name, color) in zip(tiles, specs):
        with col:
            C.md(_stat(label, value, sub, icon_name, color))

    tab_adm, tab_dist, tab_corr, tab_out = st.tabs(
        ["Admissions & demographics", "Physiology & laboratories", "Correlation structure", "Outcomes & subgroups"]
    )

    with tab_adm:
        left, right = st.columns([1.6, 1], gap="large")
        with left:
            with C.panel("an_trend", "Admission and deterioration trend", "Daily admissions with documented events", "bar-chart"):
                trend = admission_trend(frame)
                fig = go.Figure()
                fig.add_trace(
                    go.Bar(
                        x=trend["period"], y=trend["admissions"], name="Admissions",
                        marker=dict(color=PALETTE["cyan"], opacity=0.75, line=dict(width=0)),
                        hovertemplate="%{x|%d %b}<br><b>%{y}</b> admissions<extra></extra>",
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=trend["period"], y=trend["deteriorations"], name="Deterioration events",
                        mode="lines+markers", line=dict(color=PALETTE["critical"], width=2.2, shape="spline"),
                        marker=dict(size=7, color=PALETTE["critical"], line=dict(color="#071120", width=1.5)),
                        hovertemplate="%{x|%d %b}<br><b>%{y}</b> events<extra></extra>",
                    )
                )
                charts.apply_theme(fig, height=310, legend=True, margins=(8, 8, 30, 8), y_title="Patients / day")
                fig.update_yaxes(title=None)
                st.plotly_chart(fig, use_container_width=True, config=charts.PLOT_CONFIG)

            with C.panel("an_los", "Length of stay distribution", "Hours from admission to the latest observation", "clock"):
                fig = go.Figure(
                    go.Histogram(
                        x=frame["los_hours"], nbinsx=26, marker=dict(color=PALETTE["indigo"], opacity=0.75),
                        hovertemplate="%{x} h<br><b>%{y}</b> patients<extra></extra>",
                    )
                )
                charts.apply_theme(fig, height=260, hovermode="closest", margins=(8, 8, 10, 8), y_title="Patients")
                fig.update_yaxes(title=None)
                st.plotly_chart(fig, use_container_width=True, config=charts.PLOT_CONFIG)

        with right:
            with C.panel("an_age", "Age distribution", "Admissions by age band", "users"):
                counts = frame["age_group"].value_counts().reindex(snap["age_group"].cat.categories).fillna(0)
                fig = go.Figure(
                    go.Bar(
                        x=counts.values, y=[str(i) for i in counts.index], orientation="h",
                        marker=dict(color=PALETTE["cyan"], opacity=0.8),
                        hovertemplate="%{y} years<br><b>%{x}</b> patients<extra></extra>",
                    )
                )
                charts.apply_theme(fig, height=235, hovermode="closest", margins=(8, 8, 8, 8), xgrid=True, ygrid=False)
                fig.update_xaxes(title=None)
                st.plotly_chart(fig, use_container_width=True, config=charts.PLOT_CONFIG)

            with C.panel("an_sex", "Sex distribution", "All filtered admissions", "user"):
                counts = frame["sex"].value_counts()
                total = int(counts.sum())
                names = {"F": "Female", "M": "Male"}
                labels = [
                    f"{names.get(str(sex), str(sex))} · {int(n)} ({n / total:.0%})"
                    for sex, n in counts.items()
                ]
                fig = go.Figure(
                    go.Pie(
                        labels=labels, values=counts.values, hole=0.62,
                        marker=dict(colors=[PALETTE["cyan"], PALETTE["indigo"]], line=dict(color="#0b1524", width=3)),
                        textinfo="none",
                        hovertemplate="<b>%{label}</b><br>%{value} patients<extra></extra>",
                    )
                )
                fig.update_layout(
                    height=235, margin=dict(l=4, r=4, t=4, b=4), paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)", showlegend=True,
                    legend=dict(
                        orientation="h", yanchor="bottom", y=-0.14, xanchor="center", x=0.5,
                        font=dict(size=10, color=PALETTE["text_muted"]),
                    ),
                    annotations=[dict(
                        text=f"<b>{total}</b><br><span style='font-size:9px'>admissions</span>",
                        x=0.5, y=0.5, showarrow=False,
                        font=dict(size=17, color=PALETTE["text"]),
                    )],
                )
                st.plotly_chart(fig, use_container_width=True, config=charts.PLOT_CONFIG)

    with tab_dist:
        d1, d2 = st.columns(2, gap="large")
        with d1:
            with C.panel("an_vital", "Vital sign distribution", "By model risk band · current assessment", "heart-pulse"):
                vital = st.selectbox("Vital sign", list(_VITALS), key="an_vital")
                col, unit = _VITALS[vital]
                st.plotly_chart(_box(frame, col, unit, th, vital), use_container_width=True, config=charts.PLOT_CONFIG)
        with d2:
            with C.panel("an_lab", "Laboratory distribution", "By model risk band · latest panel", "flask"):
                lab = st.selectbox("Laboratory assay", list(_LABS), key="an_lab")
                col, unit = _LABS[lab]
                st.plotly_chart(_box(frame, col, unit, th, lab), use_container_width=True, config=charts.PLOT_CONFIG)

        with C.panel("an_scatter", "Physiology scatter", "Latest values coloured by predicted risk", "activity"):
            xcol, ycol = st.columns(2, gap="small")
            x_name = xcol.selectbox("X axis", list(_VITALS), index=0, key="an_x")
            y_name = ycol.selectbox("Y axis", list(_VITALS), index=2, key="an_y")
            xc, xu = _VITALS[x_name]
            yc, yu = _VITALS[y_name]
            data = frame.dropna(subset=[xc, yc])
            fig = go.Figure(
                go.Scatter(
                    x=data[xc], y=data[yc], mode="markers",
                    marker=dict(
                        size=7, color=data["risk_prob"], colorscale=[[0, PALETTE["low"]], [0.35, PALETTE["moderate"]],
                        [0.55, PALETTE["high"]], [0.75, PALETTE["critical"]], [1, "#fb7185"]],
                        cmin=0, cmax=1, opacity=0.85, line=dict(width=0),
                        colorbar=dict(
                            title=dict(text="Risk", side="top", font=dict(size=9)),
                            thickness=10, len=0.68, tickfont=dict(size=8),
                        ),
                    ),
                    customdata=np.stack([data["patient_id"]], axis=-1),
                    hovertemplate=f"%{{customdata[0]}}<br>{x_name} %{{x:.1f}} {xu}<br>{y_name} %{{y:.1f}} {yu}<br>Risk %{{marker.color:.0%}}<extra></extra>",
                )
            )
            charts.apply_theme(fig, height=330, hovermode="closest", margins=(8, 8, 10, 8), x_title=f"{x_name} ({xu})", y_title=f"{y_name} ({yu})")
            fig.update_xaxes(title=None)
            fig.update_yaxes(title=None)
            st.plotly_chart(fig, use_container_width=True, config=charts.PLOT_CONFIG)

    with tab_corr:
        with C.panel("an_corr", "Spearman correlation heatmap", "Physiology, laboratories and predicted risk", "grid"):
            matrix = correlation_matrix(frame)
            labels = [_CORR_LABELS.get(f, f) for f in matrix.columns]
            fig = go.Figure(
                go.Heatmap(
                    z=matrix.values, x=labels, y=labels, zmin=-1, zmax=1,
                    colorscale=[[0, "#2563eb"], [0.5, "#0b1524"], [1, "#f43f5e"]],
                    text=np.round(matrix.values, 2), texttemplate="%{text}", textfont=dict(size=7),
                    hovertemplate="%{y} ↔ %{x}<br>ρ = %{z:.2f}<extra></extra>",
                    colorbar=dict(
                        title=dict(text="Spearman ρ", side="top", font=dict(size=9)),
                        thickness=10, len=0.75, tickfont=dict(size=8),
                    ),
                )
            )
            fig.update_layout(height=580, margin=dict(l=8, r=8, t=8, b=8), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            fig.update_xaxes(tickangle=-40, tickfont=dict(size=9))
            fig.update_yaxes(tickfont=dict(size=9))
            st.plotly_chart(fig, use_container_width=True, config=charts.PLOT_CONFIG)
            C.md('<div class="panel-note">Rank correlation on the filtered cohort. Associations are descriptive and '
                 'do not establish causality.</div>')

    with tab_out:
        left, right = st.columns([1.2, 1], gap="large")
        with left:
            with C.panel("an_outcome", "Outcome distribution by risk band", "Documented deterioration within 24h", "target"):
                rows = []
                for band in RISK_ORDER:
                    group = frame[frame["risk_band"] == band]
                    rows.append({"band": band, "event": int(group["deterioration_24h"].sum()), "no_event": int((~group["deterioration_24h"].astype(bool)).sum())})
                outcome = pd.DataFrame(rows)
                fig = go.Figure()
                fig.add_trace(go.Bar(x=[RISK_META[b]["short"] for b in outcome["band"]], y=outcome["event"], name="Event documented",
                                     marker_color=PALETTE["critical"], hovertemplate="%{x}<br><b>%{y}</b> events<extra></extra>"))
                fig.add_trace(go.Bar(x=[RISK_META[b]["short"] for b in outcome["band"]], y=outcome["no_event"], name="No event",
                                     marker_color="rgba(142,163,192,0.35)", hovertemplate="%{x}<br><b>%{y}</b> without event<extra></extra>"))
                charts.apply_theme(fig, height=310, legend=True, hovermode="closest", margins=(8, 8, 30, 8), y_title="Patients")
                fig.update_layout(barmode="stack")
                fig.update_yaxes(title=None)
                st.plotly_chart(fig, use_container_width=True, config=charts.PLOT_CONFIG)
                rate_band = (outcome["event"] / (outcome["event"] + outcome["no_event"]).replace(0, np.nan) * 100).round(1)
                C.md(C.legend_html([(RISK_META[b]["short"], RISK_META[b]["color"]) for b in RISK_ORDER],
                                   {RISK_META[b]["short"]: f"{rate_band[i]:.0f}% event rate" for i, b in enumerate(RISK_ORDER)}))

        with right:
            with C.panel("an_subgroup", "Subgroup comparison", "Event rate with Wilson 95% interval", "layers"):
                by_label = st.selectbox("Group by", ["Admission type", "Age group", "Sex"], key="an_by")
                by_col = {"Admission type": "admission_type", "Age group": "age_group", "Sex": "sex"}[by_label]
                table = subgroup_table(frame, by_col, th)
                if table.empty:
                    C.empty_state("No subgroup data", "Adjust filters.", "filter")
                else:
                    fig = go.Figure(
                        go.Bar(
                            x=table[by_col].astype(str), y=table["event_rate"],
                            marker=dict(color=PALETTE["cyan"], opacity=0.8),
                            error_y=dict(type="data", array=table["ci_high"] - table["event_rate"],
                                         arrayminus=table["event_rate"] - table["ci_low"], color=PALETTE["text_dim"]),
                            text=[f"{v:.1f}%" for v in table["event_rate"]], textposition="outside",
                            hovertemplate="%{x}<br><b>%{y:.1f}%</b> event rate<extra></extra>",
                        )
                    )
                    charts.apply_theme(fig, height=310, hovermode="closest", margins=(8, 8, 18, 8), y_title="Event rate (%)")
                    fig.update_yaxes(title=None)
                    st.plotly_chart(fig, use_container_width=True, config=charts.PLOT_CONFIG)

        with C.panel("an_table", "Subgroup statistics", "Counts, intervals and mean model risk", "clipboard-list"):
            by_label = st.session_state.get("an_by", "Admission type")
            by_col = {"Admission type": "admission_type", "Age group": "age_group", "Sex": "sex"}[by_label]
            table = subgroup_table(frame, by_col, th)
            if not table.empty:
                display = table.rename(columns={
                    by_col: by_label,
                    "patients": "Patients", "events": "Events", "event_rate": "Event rate %",
                    "ci_low": "CI low", "ci_high": "CI high", "mean_risk": "Mean risk %",
                    "critical_share": "Critical %", "mean_los": "Mean LOS (h)",
                })
                st.dataframe(display, use_container_width=True, hide_index=True)
