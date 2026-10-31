"""Individual patient detail workspace."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ..analytics import with_bands
from ..config import PALETTE, RISK_META, risk_from_probability
from ..data import repository
from ..state import current_pid, navigate, thresholds
from ..ui import charts
from ..ui import components as C
from ..ml.explain import patient_explanation


_METRICS = {
    "HR": ("heart_rate", "HR", "bpm", PALETTE["critical"], (50, 100)),
    "RR": ("respiratory_rate", "RR", "/min", PALETTE["info"], (10, 24)),
    "SpO₂": ("spo2", "SpO₂", "%", PALETTE["cyan"], (94, 100)),
    "SBP": ("systolic_bp", "SBP", "mmHg", PALETTE["high"], (90, 140)),
    "Temp": ("temperature", "Temp", "°C", PALETTE["moderate"], (36.0, 38.0)),
    "MEWS": ("mews", "MEWS", "pts", PALETTE["indigo"], (0, 4)),
}

_LABS = {
    "Lactate": ("lactate", "mmol/L", (0.0, 2.0), PALETTE["high"]),
    "White cell count": ("wbc", "10⁹/L", (4.0, 12.0), PALETTE["cyan"]),
    "Creatinine": ("creatinine", "mg/dL", (0.6, 1.3), PALETTE["indigo"]),
    "C-reactive protein": ("crp", "mg/L", (0.0, 10.0), PALETTE["moderate"]),
    "Platelets": ("platelets", "10⁹/L", (150.0, 400.0), PALETTE["teal"]),
    "Haemoglobin": ("hemoglobin", "g/dL", (12.0, 17.0), PALETTE["blue"]),
}

_WINDOWS = {"12h": 12, "24h": 24, "48h": 48, "Full stay": None}


def _vital_tile(label: str, value: str, unit: str, icon_name: str, flag: str) -> str:
    return (
        f'<div class="vital {flag}"><div class="vital-top">{C.icon_html(icon_name, 12)}'
        f'<span class="vital-lbl">{label}</span></div>'
        f'<div class="vital-val">{value}<small>{unit}</small></div></div>'
    )


def _trend_chart(obs: pd.DataFrame, metric: str) -> go.Figure:
    col, label, unit, color, (low, high) = _METRICS[metric]
    data = obs[["ts", col]].dropna()
    fig = go.Figure()
    fig.add_hrect(y0=low, y1=high, fillcolor=PALETTE["low"], opacity=0.06, line_width=0, layer="below")
    for bound in (low, high):
        fig.add_hline(y=bound, line=dict(color=PALETTE["low"], width=1, dash="dot"), opacity=0.4)
    fig.add_trace(
        go.Scatter(
            x=data["ts"],
            y=data[col],
            mode="lines+markers",
            line=dict(color=color, width=2.2, shape="spline", smoothing=0.5),
            marker=dict(size=4.5, color=color, line=dict(color="#071120", width=1)),
            hovertemplate=f"%{{x|%d %b %H:%M}}<br><b>%{{y:.1f}}</b> {unit}<extra></extra>",
            name=label,
        )
    )
    charts.apply_theme(fig, height=260, margins=(8, 8, 14, 8), y_title=f"{label} ({unit})")
    fig.update_yaxes(title=None)
    return fig


def _bp_chart(obs: pd.DataFrame) -> go.Figure:
    data = obs.dropna(subset=["systolic_bp"])
    fig = go.Figure()
    fig.add_hrect(y0=90, y1=140, fillcolor=PALETTE["low"], opacity=0.05, line_width=0, layer="below")
    fig.add_trace(
        go.Scatter(
            x=data["ts"], y=data["systolic_bp"], mode="lines",
            line=dict(color=PALETTE["high"], width=2.2, shape="spline"),
            name="Systolic", hovertemplate="%{x|%d %b %H:%M}<br>SBP <b>%{y:.0f}</b> mmHg<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=data["ts"], y=data["diastolic_bp"], mode="lines",
            line=dict(color=PALETTE["blue"], width=1.6, dash="dot", shape="spline"),
            name="Diastolic", hovertemplate="%{x|%d %b %H:%M}<br>DBP <b>%{y:.0f}</b> mmHg<extra></extra>",
        )
    )
    charts.apply_theme(fig, height=260, legend=True, margins=(8, 8, 30, 8), y_title="mmHg")
    fig.update_yaxes(title=None)
    return fig


def _lab_chart(labs: pd.DataFrame, selected: list[str]) -> go.Figure:
    fig = go.Figure()
    for i, name in enumerate(selected):
        col, unit, (low, high), color = _LABS[name]
        data = labs[["ts", col]].dropna()
        fig.add_trace(
            go.Scatter(
                x=data["ts"], y=data[col], mode="lines+markers", name=name,
                line=dict(color=color, width=2, shape="spline"),
                marker=dict(size=6, color=color, line=dict(color="#071120", width=1.5)),
                hovertemplate=f"%{{x|%d %b %H:%M}}<br><b>%{{y:g}}</b> {unit}<extra></extra>",
            )
        )
    charts.apply_theme(fig, height=270, legend=True, margins=(8, 8, 32, 8))
    fig.update_yaxes(title=None)
    return fig


def _risk_panel(risk: pd.DataFrame, th: dict[str, float], probability: float, delta: float | None) -> None:
    band = risk_from_probability(probability, th)
    meta = RISK_META[band]
    C.md(
        f'<div class="risk-hero" style="--risk-color:{meta["color"]};--risk-soft:{meta["soft"]};--risk-border:{meta["border"]}">'
        f'<div class="risk-hero-top"><span>Predicted 24h deterioration risk</span>{C.risk_badge(band)}</div>'
        f'<div class="risk-hero-value">{probability:.0%}<span>{("+" if (delta or 0) >= 0 else "")}{delta*100:.0f} pp vs previous assessment</span></div>'
        f'<div class="risk-hero-bar"><span style="width:{probability*100:.0f}%"></span></div>'
        f'<div class="risk-hero-foot">Thresholds · critical ≥ {th["critical"]:.0%} · high ≥ {th["high"]:.0%} · '
        f'moderate ≥ {th["moderate"]:.0%}</div></div>'
    )


def render() -> None:
    slug = st.session_state.dataset
    th = thresholds()
    pid = current_pid() or st.session_state.get("last_pid")
    if not pid:
        C.page_header("Patient detail", "Select a patient from monitoring to open their assessment.")
        C.empty_state("No patient selected", "Open a patient card from the overview or monitoring workspace.", "user")
        if st.button("Go to patient monitoring", key="pd_goto_monitoring"):
            navigate("monitoring")
        return

    detail = repository.patient_detail(slug, pid)
    if not detail:
        C.page_header("Patient not found", f"No record for {pid} in the active dataset.")
        C.empty_state("Unknown patient", "The patient ID is not part of the active dataset.", "search")
        if st.button("Back to monitoring", key="pd_unknown_back"):
            navigate("monitoring")
        return

    st.session_state["last_pid"] = pid
    patient = detail["patient"]
    obs = detail["obs"]
    labs = detail["labs"]
    risk = detail["risk"]
    snap = with_bands(repository.patient_snapshot(slug), th)
    row = snap[snap["patient_id"] == pid]
    row = row.iloc[0] if len(row) else None
    band = row["risk_band"] if row is not None else "low"
    probability = float(row["risk_prob"]) if row is not None else float("nan")
    delta = float(row["risk_delta"]) if row is not None and pd.notna(row["risk_delta"]) else None
    meta = RISK_META[band]

    C.page_header(
        f"Patient {pid}",
        f"{int(patient['age'])}y {patient['sex']} · {patient['admission_type']} admission · "
        f"{patient['diagnosis_group']} · bed {patient['bed']}",
        chips=(
            C.risk_badge(band)
            + C.chip(f"Risk {probability:.0%}", "red" if band == "critical" else "amber")
            + C.chip(f"Predicted {C.fmt_ts(row['predicted_ts']) if row is not None else '—'}", "muted")
            + C.chip(f"Completeness {C.fmt_pct(row['completeness']) if row is not None else '—'}", "teal")
            + C.chip(f"MEWS {C.fmt_num(row['mews'], 0) if row is not None else '—'}", "indigo")
        ),
    )

    a1, a2, a3, a4 = st.columns([1, 1.15, 1.4, 3], gap="small")
    if a1.button("Back to monitoring", key="pd_back", use_container_width=True):
        navigate("monitoring")
    if a2.button("Explain this risk", key="pd_xai", use_container_width=True, icon=":material/psychology:"):
        navigate("xai", pid=pid)
    if a3.button("Open risk prediction", key="pd_predict", use_container_width=True, icon=":material/auto_awesome:"):
        navigate("prediction")
    a4.markdown(
        f'<div class="pd-note">{C.icon_html("info", 13)} Model output for decision support, not a diagnosis. '
        f'Assessment anchored at hour {int(patient["assessment_hour"])} of admission.</div>',
        unsafe_allow_html=True,
    )

    latest = obs.iloc[-1] if len(obs) else None
    tiles = [
        _vital_tile("Heart rate", C.fmt_num(latest["heart_rate"], 0), "bpm", "heart", C.clinical_flag("heart_rate", latest["heart_rate"]) if latest is not None else ""),
        _vital_tile("Respiratory rate", C.fmt_num(latest["respiratory_rate"], 0), "/min", "wind", C.clinical_flag("respiratory_rate", latest["respiratory_rate"]) if latest is not None else ""),
        _vital_tile("SpO₂", C.fmt_num(latest["spo2"], 0), "%", "droplet", C.clinical_flag("spo2", latest["spo2"]) if latest is not None else ""),
        _vital_tile("Blood pressure", f'{C.fmt_num(latest["systolic_bp"],0)}/{C.fmt_num(latest["diastolic_bp"],0)}', "mmHg", "activity", C.clinical_flag("systolic_bp", latest["systolic_bp"]) if latest is not None else ""),
        _vital_tile("Temperature", C.fmt_num(latest["temperature"], 1), "°C", "thermometer", C.clinical_flag("temperature", latest["temperature"]) if latest is not None else ""),
        _vital_tile("MEWS", C.fmt_num(latest["mews"], 0), "pts", "gauge", C.clinical_flag("mews", latest["mews"]) if latest is not None else ""),
    ]
    C.md(f'<div class="vitals-grid pd-vitals">{"".join(tiles)}</div>')

    left, right = st.columns([1.5, 1], gap="large")

    with left:
        with C.panel("pd_physiology", "Vital sign monitoring", "Hourly observations · select a metric and window", "heart-pulse"):
            c1, c2 = st.columns([2.4, 1.2], gap="small")
            metric = c1.segmented_control("Metric", list(_METRICS), key="px_metric") or "HR"
            window = c2.segmented_control("Window", list(_WINDOWS), key="px_window") or "24h"
            max_hour = int(obs["hour"].max()) if len(obs) else 0
            hours = _WINDOWS[window]
            obs_view = obs[obs["hour"] >= max_hour - hours] if hours else obs
            st.plotly_chart(_trend_chart(obs_view, metric), use_container_width=True, config=charts.PLOT_CONFIG)

    with right:
        with C.panel("pd_risk", "Risk trajectory", "Model-scored 24h risk at 6-hour anchors", "activity"):
            _risk_panel(risk, th, probability, delta)
            if len(risk):
                cutoff = risk["assessment_ts"].max() - pd.Timedelta(hours=(hours or 240))
                st.plotly_chart(charts.risk_trajectory(risk[risk["assessment_ts"] >= cutoff], th, height=232),
                                use_container_width=True, config=charts.PLOT_CONFIG)

    left2, right2 = st.columns([1.5, 1], gap="large")
    with left2:
        with C.panel("pd_labs", "Laboratory trends", "Select one or more assays", "flask"):
            lab_names = st.multiselect("Assays", list(_LABS), default=["Lactate"], key="px_labs", label_visibility="collapsed")
            lab_view = labs[labs["hour"] >= max_hour - hours] if hours and len(labs) else labs
            if lab_names and len(lab_view):
                st.plotly_chart(_lab_chart(lab_view, lab_names), use_container_width=True, config=charts.PLOT_CONFIG)
            else:
                C.empty_state("No laboratory values in this window", "Widen the observation window.", "flask")

        with C.panel("pd_bp", "Blood pressure", "Systolic and diastolic trend", "activity"):
            st.plotly_chart(_bp_chart(obs_view), use_container_width=True, config=charts.PLOT_CONFIG)

    with right2:
        with C.panel("pd_context", "Assessment context", "Static and administrative attributes", "clipboard-list"):
            completeness = float(row["completeness"]) if row is not None else 0.0
            hours_labs = max_hour - int(labs["hour"].max()) if len(labs) else None
            C.md(
                '<div class="ctx-grid">'
                + C.stat_row("Admission", f"{pd.Timestamp(patient['admit_ts']):%d %b %Y %H:%M}", "calendar")
                + C.stat_row("Length of stay", f"{patient['los_hours']:.0f} h", "clock")
                + C.stat_row("Assessment anchor", f"Hour {int(patient['assessment_hour'])}", "target")
                + C.stat_row("Comorbidities", str(int(patient["comorbidity_count"])), "stethoscope")
                + C.stat_row("Observations (8h)", str(int(row["obs_count_8h"])) if row is not None else "—", "activity")
                + C.stat_row("Hours since labs", f"{hours_labs} h" if hours_labs is not None else "—", "flask")
                + "</div>"
            )
            C.md(
                f'<div class="ctx-complete"><span>Data completeness</span>{C.completion_bar(completeness, width=150)}'
                f'<b>{C.fmt_pct(completeness)}</b></div>'
            )

        patient_alerts = repository.alerts(slug, th)
        patient_alerts = patient_alerts[patient_alerts["patient_id"] == pid].sort_values("ts", ascending=False)
        with C.panel("pd_alerts", "Patient alerts", f"{int((~patient_alerts['resolved']).sum())} unresolved", "bell-ring"):
            if patient_alerts.empty:
                C.md('<div class="pop-empty">No alerts generated for this patient.</div>')
            for _, alert in patient_alerts.head(4).iterrows():
                C.md(C.feed_item(alert, row["snapshot_ts"] if row is not None else patient["admit_ts"]))
            if st.button("Open alert centre", key="pd_open_alerts", use_container_width=True):
                navigate("alerts")

    with C.panel("pd_shap", "Influential features for this assessment", "SHAP additive contributions from the primary model", "brain"):
        try:
            result = repository.shap_patient(slug, pid)
            explanation = patient_explanation(result, 0, top_n=9)
            contrib = explanation["contributions"].sort_values("shap")
            colors = [PALETTE["critical"] if v > 0 else PALETTE["low"] for v in contrib["shap"]]
            fig = go.Figure(
                go.Bar(
                    x=contrib["shap"],
                    y=contrib["label"],
                    orientation="h",
                    marker=dict(color=colors, line=dict(width=0)),
                    text=[f"{v:+.3f}" for v in contrib["shap"]],
                    textposition="outside",
                    hovertemplate="<b>%{y}</b><br>SHAP %{x:.3f}<extra></extra>",
                )
            )
            charts.apply_theme(fig, height=310, hovermode="closest", margins=(8, 30, 10, 8), xgrid=True, ygrid=False)
            fig.update_xaxes(title=f"SHAP contribution ({'log-odds' if result.link == 'logit' else 'probability'})")
            st.plotly_chart(fig, use_container_width=True, config=charts.PLOT_CONFIG)
            label = "log-odds" if result.link == "logit" else "probability"
            C.md(
                f'<div class="panel-note">Base value {explanation["base_value"]:.3f} · additive output '
                f'{explanation["raw_output"]:.3f} ({label}) · model probability {explanation["probability"]:.1%}. '
                f'Positive values push risk up; negative values pull it down.</div>'
            )
            C.md(
                '<div class="shap-lists"><div><div class="shap-list-title">Increases risk</div>'
                + "".join(
                    f'<div class="shap-row"><span>{r.label}</span><b style="color:{PALETTE["critical"]}">+{r.shap:.3f}</b></div>'
                    for r in explanation["increasing"].itertuples()
                )
                + '</div><div><div class="shap-list-title">Decreases risk</div>'
                + "".join(
                    f'<div class="shap-row"><span>{r.label}</span><b style="color:{PALETTE["low"]}">{r.shap:.3f}</b></div>'
                    for r in explanation["decreasing"].itertuples()
                )
                + "</div></div>"
            )
        except Exception as exc:  # noqa: BLE001 - surface model errors honestly
            C.md(f'<div class="pop-empty">SHAP values unavailable for this assessment: {exc}</div>')
