"""Interactive risk prediction workspace.

Every score on this page is produced on demand by the trained primary model
from the feature row built for the selected assessment or the entered values.
No risk value is precomputed or hardcoded.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ..analytics import with_bands
from ..config import PALETTE, RISK_META, risk_from_probability
from ..data import repository
from ..data.mews import AVPU_SCORE, mews_components
from ..ml import features as F
from ..ml.explain import compute_shap, patient_explanation
from ..state import navigate, thresholds
from ..ui import charts
from ..ui import components as C


_VITAL_SPECS = [
    ("heart_rate", "Heart rate", "bpm", 30, 200, 1),
    ("respiratory_rate", "Respiratory rate", "/min", 5, 50, 1),
    ("spo2", "SpO₂", "%", 70, 100, 1),
    ("systolic_bp", "Systolic BP", "mmHg", 60, 200, 1),
    ("diastolic_bp", "Diastolic BP", "mmHg", 30, 130, 1),
    ("temperature", "Temperature", "°C", 33.0, 42.0, 0.1),
]

_LAB_SPECS = [
    ("lactate", "Lactate", "mmol/L", 0.0, 20.0, 0.1),
    ("wbc", "White cell count", "10⁹/L", 0.0, 60.0, 0.1),
    ("creatinine", "Creatinine", "mg/dL", 0.1, 12.0, 0.01),
    ("crp", "C-reactive protein", "mg/L", 0.0, 400.0, 1.0),
    ("platelets", "Platelets", "10⁹/L", 5.0, 800.0, 1.0),
    ("hemoglobin", "Haemoglobin", "g/dL", 3.0, 22.0, 0.1),
    ("bilirubin", "Bilirubin", "mg/dL", 0.0, 30.0, 0.1),
    ("sodium", "Sodium", "mmol/L", 100.0, 180.0, 1.0),
    ("potassium", "Potassium", "mmol/L", 1.5, 8.0, 0.1),
]

_ADMISSIONS = ["Medical", "Surgical", "Neurological", "Cardiac", "Respiratory"]
_AVPU_OPTIONS = ["A", "V", "P", "U"]


def _cls(value, lo, hi, fallback):
    if value is None or pd.isna(value):
        return fallback
    return float(min(max(float(value), lo), hi))


def _defaults(key: str, value) -> dict:
    return {} if key in st.session_state else {"value": value}


def _vital_default(key: str, lo, hi, step, base: pd.Series):
    value = _cls(base.get(key), lo, hi, (float(lo) + float(hi)) / 2)
    return int(round(value)) if float(step) >= 1 else round(value, 1)


def _scenario_inputs(base: pd.Series, tag: str) -> tuple[dict, dict]:
    """Render the editable scenario controls and return vitals and labs."""
    prefix = f"px_{tag}_"
    vitals: dict[str, float] = {}
    c1, c2 = st.columns(2, gap="small")
    for i, (key, label, unit, lo, hi, step) in enumerate(_VITAL_SPECS):
        col = c1 if i % 2 == 0 else c2
        default = _vital_default(key, lo, hi, step, base)
        if float(step) >= 1.0:
            lo, hi, step = int(lo), int(hi), int(step)
            fmt = "%d"
        else:
            lo, hi, step = float(lo), float(hi), float(step)
            fmt = "%.1f"
        vitals[key] = col.slider(
            f"{label} ({unit})", lo, hi, step=step, format=fmt, key=f"{prefix}{key}",
            **_defaults(f"{prefix}{key}", default),
        )
    avpu_default = int(_cls(base.get("avpu_worst"), 0, 3, 0))
    vitals["avpu_worst"] = c1.selectbox(
        "Consciousness (AVPU)", _AVPU_OPTIONS, index=avpu_default, key=f"{prefix}avpu"
    )

    st.markdown('<div class="divider" style="margin:.55rem 0"></div>', unsafe_allow_html=True)
    labs: dict[str, float] = {}
    l1, l2, l3 = st.columns(3, gap="small")
    cols = [l1, l2, l3]
    for i, (key, label, unit, lo, hi, step) in enumerate(_LAB_SPECS):
        default = _cls(base.get(key), lo, hi, (lo + hi) / 3)
        labs[key] = cols[i % 3].number_input(
            f"{label} ({unit})", min_value=lo, max_value=hi, step=step, key=f"{prefix}{key}",
            **_defaults(f"{prefix}{key}", default),
        )
    return vitals, labs


def _scenario_context(base: pd.Series, tag: str) -> dict:
    prefix = f"px_{tag}_ctx_"
    c1, c2, c3 = st.columns([1, 1.25, 1.1], gap="small")
    age = c1.number_input(
        "Age (years)", min_value=18, max_value=100, step=1, key=f"{prefix}age",
        **_defaults(f"{prefix}age", int(_cls(base.get("age"), 18, 100, 65))),
    )
    sex = c2.selectbox(
        "Sex", ["Female", "Male"], index=int(float(base.get("sex_male", 0) or 0) >= 0.5), key=f"{prefix}sex"
    )
    comorbidity = c3.number_input(
        "Comorbidities", min_value=0, max_value=15, step=1, key=f"{prefix}comorb",
        **_defaults(f"{prefix}comorb", int(_cls(base.get("comorbidity_count"), 0, 15, 2))),
    )
    c1, c2 = st.columns([1.2, 1], gap="small")
    admission_default = "Medical"
    for name in _ADMISSIONS[1:]:
        if float(base.get(f"adm_{name.lower()}", 0) or 0) >= 0.5:
            admission_default = name
            break
    admission = c1.selectbox(
        "Admission type", _ADMISSIONS, index=_ADMISSIONS.index(admission_default), key=f"{prefix}adm"
    )
    hour = c2.number_input(
        "Hours since ICU admission", min_value=8, max_value=72, step=1, key=f"{prefix}hour",
        **_defaults(f"{prefix}hour", int(_cls(base.get("assessment_hour"), 8, 72, 24))),
    )
    return {"age": age, "sex": sex, "comorbidity": comorbidity, "admission": admission, "hour": hour}


def _build_row(base: pd.Series, vitals: dict, labs: dict, context: dict) -> pd.Series:
    vital_inputs = {key: vitals[key] for key in ["heart_rate", "respiratory_rate", "spo2", "systolic_bp", "diastolic_bp", "temperature"]}
    row = F.apply_manual_override(base, vital_inputs, labs)
    row["systolic_bp"] = vitals["systolic_bp"]
    row["diastolic_bp"] = vitals["diastolic_bp"]
    row["map"] = (vitals["systolic_bp"] + 2 * vitals["diastolic_bp"]) / 3
    row["avpu_worst"] = float(AVPU_SCORE[vitals["avpu_worst"]])
    mews = mews_components(
        pd.Series({**vital_inputs, "avpu": vitals["avpu_worst"]})
    )["total"]
    row["mews"] = mews
    row["mews_max_8h"] = mews
    row["age"] = float(context["age"])
    row["sex_male"] = 1.0 if context["sex"] == "Male" else 0.0
    row["comorbidity_count"] = float(context["comorbidity"])
    row["assessment_hour"] = float(context["hour"])
    for name in _ADMISSIONS[1:]:
        row[f"adm_{name.lower()}"] = 1.0 if context["admission"] == name else 0.0
    return row


def _gauge(probability: float, band: str, th: dict[str, float]) -> go.Figure:
    meta = RISK_META[band]
    steps = [
        dict(range=[0, th["moderate"] * 100], color="rgba(34,197,94,.14)"),
        dict(range=[th["moderate"] * 100, th["high"] * 100], color="rgba(250,204,21,.13)"),
        dict(range=[th["high"] * 100, th["critical"] * 100], color="rgba(251,146,60,.15)"),
        dict(range=[th["critical"] * 100, 100], color="rgba(244,63,94,.16)"),
    ]
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number=dict(suffix="%", font=dict(size=40, color=meta["color"], family="Inter")),
            gauge=dict(
                axis=dict(
                    range=[0, 100], tickvals=[0, 25, 50, 75, 100], ticksuffix="%",
                    tickfont=dict(size=9, color=PALETTE["text_dim"]), tickcolor="rgba(142,163,192,.3)",
                ),
                bar=dict(color=meta["color"], thickness=0.26),
                bgcolor="rgba(0,0,0,0)",
                borderwidth=0,
                steps=steps,
                threshold=dict(line=dict(color="#e8eefb", width=2), thickness=0.8, value=probability * 100),
            ),
        )
    )
    fig.update_layout(
        height=208,
        margin=dict(l=18, r=18, t=8, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter"),
    )
    return fig


def _waterfall(shap, index: int) -> None:
    explanation = patient_explanation(shap, index, top_n=10)
    contrib = explanation["contributions"].sort_values("abs", ascending=False)
    labels = ["Base value"] + contrib["label"].tolist() + ["Model output"]
    measures = ["absolute"] + ["relative"] * len(contrib) + ["total"]
    values = [explanation["base_value"]] + contrib["shap"].tolist() + [explanation["raw_output"]]
    fig = go.Figure(
        go.Waterfall(
            orientation="h",
            measure=measures,
            y=labels,
            x=values,
            increasing=dict(marker=dict(color=PALETTE["critical"])),
            decreasing=dict(marker=dict(color=PALETTE["low"])),
            totals=dict(marker=dict(color=PALETTE["cyan"])),
            connector=dict(line=dict(color="rgba(142,163,192,.35)", width=1)),
            hovertemplate="<b>%{y}</b><br>%{x:.3f}<extra></extra>",
            text=[f"{v:+.3f}" if m == "relative" else f"{v:.3f}" for v, m in zip(values, measures)],
            textposition="outside",
        )
    )
    charts.apply_theme(fig, height=380, hovermode="closest", margins=(8, 36, 10, 8), xgrid=True, ygrid=False)
    unit = "log-odds" if shap.link == "logit" else "probability"
    fig.update_xaxes(title=f"Model output ({unit})")
    st.plotly_chart(fig, use_container_width=True, config=charts.PLOT_CONFIG)
    C.md(
        f'<div class="panel-note">Base value {explanation["base_value"]:.3f} → additive output '
        f'{explanation["raw_output"]:.3f} ({unit}) → probability {explanation["probability"]:.1%}. '
        f'SHAP values are recomputed for the exact scenario row shown.</div>'
    )


def _reset_scenario(tag: str, base: pd.Series) -> None:
    """Restore every scenario control to the value inherited from the assessment."""
    ss = st.session_state
    prefix = f"px_{tag}_"
    for key, _label, _unit, lo, hi, step in _VITAL_SPECS:
        ss[f"{prefix}{key}"] = _vital_default(key, lo, hi, step, base)
    ss[f"{prefix}avpu"] = _AVPU_OPTIONS[int(_cls(base.get("avpu_worst"), 0, 3, 0))]
    for key, _label, _unit, lo, hi, _step in _LAB_SPECS:
        ss[f"{prefix}{key}"] = _cls(base.get(key), lo, hi, (lo + hi) / 3)
    ctx = f"{prefix}ctx_"
    ss[f"{ctx}age"] = int(_cls(base.get("age"), 18, 100, 65))
    ss[f"{ctx}sex"] = "Male" if float(base.get("sex_male", 0) or 0) >= 0.5 else "Female"
    ss[f"{ctx}comorb"] = int(_cls(base.get("comorbidity_count"), 0, 15, 2))
    admission = "Medical"
    for name in _ADMISSIONS[1:]:
        if float(base.get(f"adm_{name.lower()}", 0) or 0) >= 0.5:
            admission = name
            break
    ss[f"{ctx}adm"] = admission
    ss[f"{ctx}hour"] = int(_cls(base.get("assessment_hour"), 8, 72, 24))
    st.rerun()


def render() -> None:
    slug = st.session_state.dataset
    th = thresholds()
    bundle = repository.load_bundle(slug)
    primary = bundle["primary_model"]
    summary = repository.dataset_summary(slug)
    snap = with_bands(repository.patient_snapshot(slug), th)

    C.page_header(
        "Risk prediction",
        "Score a monitored assessment or an investigator-defined scenario with the trained model. "
        "Entered values are scored on demand; every output includes the exact feature row used.",
        chips=(
            C.chip(f"Model · {primary}", "cyan", dot=True)
            + C.chip(f"{len(bundle['feature_names'])} features", "muted")
            + C.chip(f"Threshold · critical {th['critical']:.0%} / high {th['high']:.0%}", "muted")
        ),
    )
    C.demo_banner(
        "Hypothetical scenario inputs describe the demonstration cohort, not a real patient. Predictions are model "
        "outputs with no observed outcome; they are not a diagnosis or a validated clinical decision rule."
    )

    mode = st.segmented_control(
        "Scenario source", ["Monitored patient", "Manual entry"], key="px_mode"
    ) or "Monitored patient"

    if mode == "Monitored patient":
        pid = st.selectbox(
            "Assessment to score",
            snap.sort_values("risk_prob", ascending=False)["patient_id"].tolist(),
            key="px_pid",
            format_func=lambda p: f"{p} · {snap.loc[snap['patient_id'] == p, 'bed'].iloc[0]} "
            f"· {snap.loc[snap['patient_id'] == p, 'risk_prob'].iloc[0]:.0%} recorded risk",
        )
        detail = repository.patient_detail(slug, pid)
        patient = detail["patient"]
        base = repository.feature_row(slug, pid)
        recorded = float(snap.loc[snap["patient_id"] == pid, "risk_prob"].iloc[0])
        tag = pid
        source_note = (
            f"Assessment hour {int(patient['assessment_hour'])} · {int(patient['age'])}y {patient['sex']} · "
            f"{patient['admission_type']} admission · recorded risk {recorded:.0%}"
        )
    else:
        pid = ""
        median = bundle["X_train"].median(numeric_only=True)
        base = median.reindex(F.FEATURE_COLUMNS).fillna(median.median())
        recorded = float(np.median(bundle["proba"][primary]))
        tag = "manual"
        source_note = (
            f"Manual entry starts from the cohort median feature row · reference median test-set risk {recorded:.0%}"
        )

    left, right = st.columns([1.08, 1], gap="large")

    with left:
        with C.panel("px_inputs", "Scenario inputs", "Vital signs and laboratory values", "sliders"):
            top_l, top_r = st.columns([2.2, 1], gap="small")
            top_l.markdown(
                f'<div class="panel-note" style="margin:0 0 .6rem;font-size:.68rem">{C.icon_html("info", 12)} {source_note}</div>',
                unsafe_allow_html=True,
            )
            top_r.button(
                "Reset values", key=f"px_reset_{tag}", use_container_width=True,
                on_click=_reset_scenario, args=(tag, base),
            )
            vitals, labs = _scenario_inputs(base, tag)

        with C.panel("px_context", "Patient context", "Static attributes used by the model", "user"):
            context = _scenario_context(base, tag)
            C.md(
                '<div class="panel-note">8-hour mean, maximum and event-count features are rebuilt from the entered '
                'vital signs; slopes and features without a direct input are inherited from the selected assessment.</div>'
            )

    row = _build_row(base, vitals, labs, context)
    X = pd.DataFrame([row[F.FEATURE_COLUMNS].astype(float)])
    model = bundle["models"][primary]
    probability = float(model.predict_proba(X)[0, 1])
    band = risk_from_probability(probability, th)
    meta = RISK_META[band]
    delta = probability - recorded if pd.notna(recorded) else None

    with right:
        with C.panel("px_result", "Model prediction", "Predicted 24h deterioration risk for this feature row", "sparkles"):
            C.md(
                f'<div class="risk-hero" style="--risk-color:{meta["color"]};--risk-soft:{meta["soft"]};'
                f'--risk-border:{meta["border"]}">'
                f'<div class="risk-hero-top"><span>Scenario risk band</span>{C.risk_badge(band)}</div>'
                f'<div class="risk-hero-foot" style="border:none;padding:0;margin-top:.4rem">'
                f'Probability {probability:.1%} · threshold critical ≥ {th["critical"]:.0%} · '
                f'high ≥ {th["high"]:.0%} · moderate ≥ {th["moderate"]:.0%}</div></div>'
            )
            st.plotly_chart(_gauge(probability, band, th), use_container_width=True, config=charts.PLOT_CONFIG)
            C.md(
                C.stat_row("Recorded / reference risk", f"{recorded:.1%}", "activity", PALETTE["text_muted"])
                + C.stat_row(
                    "Change from reference",
                    f"{delta*100:+.1f} pp" if delta is not None else "—",
                    "trending-up" if (delta or 0) >= 0 else "trending-down",
                    PALETTE["critical"] if (delta or 0) > 0 else PALETTE["low"],
                )
                + C.stat_row("MEWS from entered values", f"{row['mews']:.0f}", "gauge", PALETTE["indigo"])
                + C.stat_row("Model", primary, "cpu", PALETTE["cyan"])
            )
            C.md(
                '<div class="panel-note">Single-row inference from the fitted pipeline. A negative change does not '
                'mean the patient is safe; interpret alongside the recorded observations and clinical judgement.</div>'
            )

            scenario = row[F.FEATURE_COLUMNS].astype(float).rename("value").to_frame()
            scenario.insert(0, "feature", scenario.index)
            scenario["label"] = [F.feature_label(f) for f in scenario["feature"]]
            scenario["scenario_score"] = probability
            scenario["scenario_band"] = band
            st.download_button(
                "Download scenario row (CSV)",
                data=scenario.reset_index(drop=True).to_csv(index=False),
                file_name="icu_ews_scenario_prediction.csv",
                mime="text/csv",
                key="px_download",
                use_container_width=True,
            )

    with C.panel("px_shap", "Why the model produced this score", "SHAP attribution for the scored feature row", "brain"):
        try:
            shap = compute_shap(model, X, bundle["feature_names"])
            explanation = patient_explanation(shap, 0, top_n=10)
            _waterfall(shap, 0)
            C.md(
                '<div class="shap-lists"><div><div class="shap-list-title">Increases predicted risk</div>'
                + "".join(
                    f'<div class="shap-row"><span>{r.label}</span>'
                    f'<b style="color:{PALETTE["critical"]}">+{r.shap:.3f}</b></div>'
                    for r in explanation["increasing"].itertuples()
                )
                + '</div><div><div class="shap-list-title">Decreases predicted risk</div>'
                + "".join(
                    f'<div class="shap-row"><span>{r.label}</span>'
                    f'<b style="color:{PALETTE["low"]}">{r.shap:.3f}</b></div>'
                    for r in explanation["decreasing"].itertuples()
                )
                + "</div></div>"
            )
        except Exception as exc:  # noqa: BLE001 - show the real failure
            C.md(f'<div class="pop-empty">SHAP attribution unavailable for this scenario: {exc}</div>')

        b1, b2, b3 = st.columns([1, 1.2, 1.4], gap="small")
        if pid and b1.button("Open patient detail", key="px_open_patient", use_container_width=True):
            navigate("patient", pid=pid)
        if b2.button("Open model explanation", key="px_open_xai", use_container_width=True):
            navigate("xai")
        b3.markdown(
            '<div class="panel-note" style="margin:0;text-align:right">Scores recompute on every input change.</div>',
            unsafe_allow_html=True,
        )
