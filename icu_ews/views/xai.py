"""Explainable AI workspace backed by real SHAP values."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ..analytics import with_bands
from ..config import PALETTE, RISK_META
from ..data import repository
from ..ml.explain import patient_explanation
from ..ml.features import FEATURE_META
from ..state import navigate, thresholds
from ..ui import charts
from ..ui import components as C

_GROUP_COLORS = {
    "Vital signs": PALETTE["cyan"],
    "Laboratory": PALETTE["indigo"],
    "Patient": PALETTE["teal"],
    "Acuity events": PALETTE["high"],
    "Trends": PALETTE["violet"],
    "Data quality": PALETTE["text_dim"],
}


def _importance(fig_height: int = 380) -> None:
    bundle = repository.load_bundle(st.session_state.dataset)
    importance = bundle["importance"][:15][::-1]
    meta = bundle["feature_meta"]
    colors = [_GROUP_COLORS.get(meta.get(item["feature"], {}).get("group"), PALETTE["cyan"]) for item in importance]
    fig = go.Figure(
        go.Bar(
            x=[item["share"] * 100 for item in importance],
            y=[item["label"] for item in importance],
            orientation="h",
            marker=dict(color=colors, opacity=0.85, line=dict(width=0)),
            text=[f"{item['share']*100:.1f}%" for item in importance],
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>%{x:.2f}% of importance<extra></extra>",
        )
    )
    charts.apply_theme(fig, height=fig_height, hovermode="closest", margins=(8, 34, 8, 8), xgrid=True, ygrid=False)
    fig.update_xaxes(title="Share of model importance (%)")
    st.plotly_chart(fig, use_container_width=True, config=charts.PLOT_CONFIG)
    groups = []
    for item in bundle["importance"]:
        group = meta.get(item["feature"], {}).get("group")
        if group and (group, _GROUP_COLORS.get(group)) not in groups:
            groups.append((group, _GROUP_COLORS.get(group, PALETTE["cyan"])))
    C.md(C.legend_html(groups))
    C.md(
        f'<div class="panel-note">Source: {bundle["importance_source"].lower()} from the '
        f'{bundle["primary_model"]} pipeline. Importance describes model behaviour on this cohort, not causality.</div>'
    )


def _beeswarm(shap) -> None:
    mean_abs = shap.mean_abs().head(14)
    index = [shap.feature_names.index(f) for f in mean_abs["feature"]]
    labels = [shap.feature_names[i] for i in index]
    rng = np.random.default_rng(11)
    fig = go.Figure()
    for row, (feat_idx, feat) in enumerate(zip(index, labels)):
        values = shap.values[:, feat_idx]
        feature_values = shap.X.iloc[:, feat_idx]
        norm = feature_values.rank(pct=True).to_numpy()
        jitter = rng.uniform(-0.27, 0.27, size=len(values))
        fig.add_trace(
            go.Scattergl(
                x=values,
                y=np.full(len(values), row) + jitter,
                mode="markers",
                marker=dict(
                    size=5.2,
                    color=norm,
                    colorscale=[[0, "#3b82f6"], [0.5, "#a78bfa"], [1, "#f43f5e"]],
                    cmin=0,
                    cmax=1,
                    opacity=0.82,
                    line=dict(width=0),
                    showscale=(row == 0),
                    colorbar=dict(thickness=9, len=0.5, y=0.78,
                                  title=dict(text="Feature value", side="top", font=dict(size=9)),
                                  tickfont=dict(size=8), ticks=""),
                ),
                customdata=np.stack(
                    [np.full(len(values), FEATURE_META.get(feat, {}).get("label", feat)),
                     shap.X.iloc[:, feat_idx].astype(float)],
                    axis=-1,
                ),
                hovertemplate="<b>%{customdata[0]}</b><br>SHAP %{x:.3f}<br>Value %{customdata[1]:.3g}<extra></extra>",
                showlegend=False,
            )
        )
    labels_clean = [shap.feature_names[i] for i in index]
    fig.update_yaxes(
        tickmode="array",
        tickvals=list(range(len(index))),
        ticktext=[FEATURE_META.get(f, {}).get("label", f) for f in labels_clean],
        range=[-0.6, len(index) - 0.4],
        title=None,
    )
    charts.apply_theme(fig, height=430, hovermode="closest", margins=(8, 8, 8, 8), xgrid=True, ygrid=False)
    fig.update_xaxes(title="SHAP value (impact on model output)")
    st.plotly_chart(fig, use_container_width=True, config=charts.PLOT_CONFIG)
    C.md(
        '<div class="panel-note">Each point is one held-out test patient. Position shows the SHAP contribution; '
        'colour shows that patient\'s value for the feature (blue low, red high).</div>'
    )


def _dependence(shap, feature: str) -> None:
    idx = shap.feature_names.index(feature)
    probs = np.array([shap.probability(i) for i in range(len(shap.values))])
    fig = go.Figure(
        go.Scatter(
            x=shap.X.iloc[:, idx],
            y=shap.values[:, idx],
            mode="markers",
            marker=dict(
                size=8,
                color=probs,
                colorscale=[[0, PALETTE["low"]], [0.4, PALETTE["moderate"]], [0.7, PALETTE["high"]], [1, PALETTE["critical"]]],
                cmin=0,
                cmax=1,
                opacity=0.88,
                line=dict(width=0),
                colorbar=dict(thickness=10, len=0.7, title=dict(text="Risk", font=dict(size=9)), tickfont=dict(size=8)),
            ),
            hovertemplate="Value %{x:.3g}<br>SHAP %{y:.3f}<br>Risk %{marker.color:.0%}<extra></extra>",
        )
    )
    fig.add_hline(y=0, line=dict(color="rgba(142,163,192,.45)", width=1, dash="dot"))
    label = FEATURE_META.get(feature, {}).get("label", feature)
    unit = FEATURE_META.get(feature, {}).get("unit", "")
    charts.apply_theme(fig, height=340, hovermode="closest", margins=(10, 10, 10, 10),
                       x_title=f"{label} ({unit})" if unit else label, y_title="SHAP contribution")
    fig.update_xaxes(title=None)
    fig.update_yaxes(title=None)
    st.plotly_chart(fig, use_container_width=True, config=charts.PLOT_CONFIG)


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
    fig.update_xaxes(title="Model output (log-odds)" if shap.link == "logit" else "Model output (probability)")
    st.plotly_chart(fig, use_container_width=True, config=charts.PLOT_CONFIG)
    unit = "log-odds" if shap.link == "logit" else "probability"
    C.md(
        f'<div class="panel-note">Base value {explanation["base_value"]:.3f} → additive output '
        f'{explanation["raw_output"]:.3f} ({unit}) → probability {explanation["probability"]:.1%}. '
        f'Contributions are additive and come directly from SHAP TreeExplainer for this assessment window.</div>'
    )


def render() -> None:
    slug = st.session_state.dataset
    th = thresholds()
    summary = repository.dataset_summary(slug)
    bundle = repository.load_bundle(slug)
    primary = bundle["primary_model"]

    C.page_header(
        "Explainable AI",
        "Global and patient-level model explanations computed with SHAP on the trained primary model. "
        "Values shown are model attributions, not clinical causes.",
        chips=(
            C.chip(f"Model · {primary}", "cyan", dot=True)
            + C.chip(f"{len(bundle['feature_names'])} features", "muted")
            + C.chip(f"Test cohort · {len(bundle['X_test'])} patients", "muted")
            + C.chip("SHAP TreeExplainer", "indigo")
        ),
    )
    C.demo_banner(
        "SHAP values are computed from the trained model and the synthetic cohort. They explain model output "
        "only and must not be read as causal clinical mechanisms."
    )

    shap_test = repository.shap_test(slug)
    link = "log-odds" if shap_test.link == "logit" else "probability"
    C.md(
        f'<div class="glass-strip">{C.icon_html("info", 14)} Additive explanation in <b>{link}</b> space · '
        f'{len(shap_test.values)} held-out assessments · base value {shap_test.base_value:.3f}</div>'
    )

    left, right = st.columns([1.15, 1], gap="large")
    with left:
        with C.panel("xai_importance", "Global feature importance", "Top 15 features · normalised importance share", "bar-chart"):
            _importance()
    with right:
        with C.panel("xai_beeswarm", "SHAP value distribution", "Top 14 features across held-out patients", "layers"):
            _beeswarm(shap_test)

    top_features = shap_test.mean_abs()["feature"].head(12).tolist()
    with C.panel("xai_dependence", "SHAP dependence", "Effect of one feature across patients", "activity"):
        c1, c2 = st.columns([2, 3], gap="small")
        feature = c1.selectbox(
            "Feature",
            top_features,
            key="xai_feature",
            format_func=lambda f: bundle["feature_meta"].get(f, {}).get("label", f),
        )
        _dependence(shap_test, feature)

    with C.panel("xai_local", "Individual explanation", "Waterfall for one assessment window", "brain"):
        snap = with_bands(repository.patient_snapshot(slug), th)
        options = snap.sort_values("risk_prob", ascending=False)["patient_id"].tolist()
        c1, c2 = st.columns([1.4, 2.6], gap="small")
        pid = c1.selectbox("Patient", options, key="xai_pid", index=0)
        patient_shap = repository.shap_patient(slug, pid)
        explanation = patient_explanation(patient_shap, 0, top_n=10)
        row = snap[snap["patient_id"] == pid].iloc[0]
        c2.markdown(
            f'<div class="xai-context">Predicted risk <b>{row["risk_prob"]:.0%}</b> · band '
            f'{C.risk_badge(row["risk_band"])} · completeness {C.fmt_pct(row["completeness"])} · '
            f'MEWS {C.fmt_num(row["mews"], 0)}</div>',
            unsafe_allow_html=True,
        )
        _waterfall(patient_shap, 0)

        table = explanation["all_contributions"].head(12).copy()
        table = table.rename(columns={"label": "Feature", "group": "Group", "value": "Value", "shap": "SHAP", "direction": "Direction"})
        table["SHAP"] = table["SHAP"].round(4)
        table["Value"] = table["Value"].round(3)
        st.dataframe(table[["Feature", "Group", "Value", "SHAP", "Direction"]], use_container_width=True, hide_index=True)

        inc = explanation["increasing"].head(5)
        dec = explanation["decreasing"].head(5)
        C.md(
            '<div class="shap-lists"><div><div class="shap-list-title">Increases predicted risk</div>'
            + "".join(f'<div class="shap-row"><span>{r.label}</span><b style="color:{PALETTE["critical"]}">+{r.shap:.3f}</b></div>' for r in inc.itertuples())
            + '</div><div><div class="shap-list-title">Decreases predicted risk</div>'
            + "".join(f'<div class="shap-row"><span>{r.label}</span><b style="color:{PALETTE["low"]}">{r.shap:.3f}</b></div>' for r in dec.itertuples())
            + "</div></div>"
        )
        b1, b2 = st.columns(2)
        if b1.button("Open patient detail", key="xai_open_patient", use_container_width=True):
            navigate("patient", pid=pid)
        if b2.button("Score a manual scenario", key="xai_open_predict", use_container_width=True):
            navigate("prediction")
