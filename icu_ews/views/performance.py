"""Model performance workspace: held-out evaluation only."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ..config import PALETTE
from ..data import repository
from ..ui import charts
from ..ui import components as C

_MODEL_COLORS = {
    "XGBoost": PALETTE["cyan"],
    "Random Forest": PALETTE["indigo"],
    "Logistic Regression": PALETTE["teal"],
}


def _metric_tile(label: str, value: str, sub: str, color: str) -> str:
    return f"""
<div class="astat" style="--astat-color:{color}">
  <div class="astat-top"><span>{label}</span>{C.icon_html('target', 13)}</div>
  <div class="astat-value">{value}</div>
  <div class="astat-sub">{sub}</div>
</div>"""


def _roc(bundle: dict) -> go.Figure:
    fig = go.Figure()
    for name, color in _MODEL_COLORS.items():
        roc = bundle["curves"][name]["roc"]
        auc = bundle["metrics"][name]["roc_auc"]
        fig.add_trace(
            go.Scatter(
                x=roc["fpr"], y=roc["tpr"], mode="lines", name=f"{name} · {auc:.3f}",
                line=dict(color=color, width=3 if name == bundle["primary_model"] else 1.8),
                hovertemplate="FPR %{x:.2f}<br>TPR %{y:.2f}<extra></extra>",
            )
        )
    fig.add_trace(
        go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="rgba(142,163,192,.4)", width=1, dash="dash"),
                   name="Chance", hoverinfo="skip")
    )
    charts.apply_theme(fig, height=360, legend=True, hovermode="closest", margins=(8, 8, 34, 8))
    fig.update_xaxes(title="False positive rate", range=[0, 1])
    fig.update_yaxes(title="True positive rate", range=[0, 1])
    return fig


def _pr(bundle: dict) -> go.Figure:
    fig = go.Figure()
    baseline = float(np.mean(bundle["y_test"]))
    for name, color in _MODEL_COLORS.items():
        pr = bundle["curves"][name]["pr"]
        auc = bundle["metrics"][name]["pr_auc"]
        fig.add_trace(
            go.Scatter(
                x=pr["recall"], y=pr["precision"], mode="lines", name=f"{name} · {auc:.3f}",
                line=dict(color=color, width=3 if name == bundle["primary_model"] else 1.8),
                hovertemplate="Recall %{x:.2f}<br>Precision %{y:.2f}<extra></extra>",
            )
        )
    fig.add_hline(y=baseline, line=dict(color="rgba(142,163,192,.4)", width=1, dash="dash"),
                  annotation_text=f"Baseline {baseline:.0%}", annotation_position="top left",
                  annotation_font=dict(size=9, color=PALETTE["text_dim"]))
    charts.apply_theme(fig, height=360, legend=True, hovermode="closest", margins=(8, 8, 34, 8))
    fig.update_xaxes(title="Recall", range=[0, 1])
    fig.update_yaxes(title="Precision", range=[0, 1])
    return fig


def _confusion(bundle: dict) -> go.Figure:
    m = bundle["metrics"][bundle["primary_model"]]
    z = [[m["tn"], m["fp"]], [m["fn"], m["tp"]]]
    labels = ["No event", "Event"]
    fig = go.Figure(
        go.Heatmap(
            z=z, x=labels, y=labels, zmin=0,
            colorscale=[[0, "rgba(15,26,43,.9)"], [0.5, "rgba(34,211,238,.35)"], [1, "rgba(34,211,238,.85)"]],
            text=[[f"<b>{v}</b>" for v in row] for row in z], texttemplate="%{text}",
            textfont=dict(size=20, color="#eaf7ff"), showscale=False,
            hovertemplate="Actual %{y}<br>Predicted %{x}<br>%{z} patients<extra></extra>",
        )
    )
    fig.update_layout(height=310, margin=dict(l=8, r=8, t=8, b=8), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)")
    fig.update_xaxes(title="Predicted", side="bottom")
    fig.update_yaxes(title="Actual", autorange="reversed")
    return fig


def _calibration(bundle: dict) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect",
                             line=dict(color="rgba(142,163,192,.45)", width=1, dash="dash"), hoverinfo="skip"))
    for name, color in _MODEL_COLORS.items():
        points = bundle["curves"][name]["calibration"]
        fig.add_trace(
            go.Scatter(
                x=points["mean_predicted"], y=points["fraction_positive"], mode="lines+markers", name=name,
                line=dict(color=color, width=3 if name == bundle["primary_model"] else 1.6, shape="spline"),
                marker=dict(size=6, color=color, line=dict(color="#071120", width=1.5)),
                hovertemplate="Predicted %{x:.0%}<br>Observed %{y:.0%}<extra></extra>",
            )
        )
    charts.apply_theme(fig, height=330, legend=True, hovermode="closest", margins=(8, 8, 34, 8))
    fig.update_xaxes(title="Mean predicted probability", range=[0, 1])
    fig.update_yaxes(title="Observed event rate", range=[0, 1])
    return fig


def _cv(bundle: dict) -> go.Figure:
    primary = bundle["primary_model"]
    cv = bundle["cv"][primary]
    folds = list(range(1, len(cv["roc_auc_folds"]) + 1))
    fig = go.Figure(
        go.Bar(
            x=[f"Fold {i}" for i in folds], y=cv["roc_auc_folds"],
            marker=dict(color=PALETTE["cyan"], opacity=0.8), name="Fold ROC-AUC",
            text=[f"{v:.3f}" for v in cv["roc_auc_folds"]], textposition="outside",
            hovertemplate="%{x}<br>AUC %{y:.3f}<extra></extra>",
        )
    )
    fig.add_hline(y=cv["roc_auc_mean"], line=dict(color=PALETTE["indigo"], width=1.4, dash="dot"),
                  annotation_text=f"Mean {cv['roc_auc_mean']:.3f}", annotation_position="top right",
                  annotation_font=dict(size=10, color=PALETTE["indigo"]))
    fig.add_hrect(y0=cv["roc_auc_mean"] - cv["roc_auc_std"], y1=cv["roc_auc_mean"] + cv["roc_auc_std"],
                  fillcolor=PALETTE["indigo"], opacity=0.07, line_width=0)
    charts.apply_theme(fig, height=290, hovermode="closest", margins=(8, 8, 12, 8), y_title="ROC-AUC")
    fig.update_yaxes(title=None, range=[max(0.5, min(cv["roc_auc_folds"]) - 0.08), 1.0])
    return fig


def _sweep(bundle: dict) -> go.Figure:
    sweep = pd.DataFrame(bundle["threshold_sweep"])
    fig = go.Figure()
    for col, color, label in [
        ("precision", PALETTE["cyan"], "Precision"),
        ("recall", PALETTE["critical"], "Recall / sensitivity"),
        ("specificity", PALETTE["low"], "Specificity"),
        ("f1", PALETTE["indigo"], "F1"),
    ]:
        fig.add_trace(
            go.Scatter(x=sweep["threshold"], y=sweep[col], mode="lines", name=label,
                       line=dict(color=color, width=2.2),
                       hovertemplate="%{x:.3f}<br>%{y:.3f}<extra></extra>")
        )
    operating = bundle["metrics"][bundle["primary_model"]]["threshold"]
    fig.add_vline(x=operating, line=dict(color="rgba(255,255,255,.55)", width=1.2, dash="dash"),
                  annotation_text=f"Operating {operating:.2f}", annotation_position="top left",
                  annotation_font=dict(size=9, color=PALETTE["text_muted"]))
    charts.apply_theme(fig, height=330, legend=True, hovermode="x unified", margins=(8, 8, 34, 8), xgrid=True)
    fig.update_xaxes(title="Classification threshold")
    fig.update_yaxes(title=None, range=[0, 1.02])
    return fig


def render() -> None:
    slug = st.session_state.dataset
    bundle = repository.load_bundle(slug)
    summary = repository.dataset_summary(slug)
    training = bundle["training"]
    primary = bundle["primary_model"]
    metrics = bundle["metrics"][primary]
    test_n = len(bundle["y_test"])
    positives = int(np.sum(bundle["y_test"]))

    C.page_header(
        "Model performance",
        "Held-out evaluation of the models trained on this cohort. All metrics are computed on the test "
        "partition, which was not used for tuning, threshold selection or model selection.",
        chips=(
            C.chip(training.get("model_version", "—"), "cyan", dot=True)
            + C.chip(f"Primary · {primary}", "indigo")
            + C.chip(f"Test n = {test_n} · {positives} events", "muted")
            + C.chip(f"Fingerprint {training.get('fingerprint', '—')}", "muted")
        ),
    )
    C.demo_banner(
        "Metrics describe performance on the synthetic demonstration cohort only. They are not evidence of "
        "clinical effectiveness and must be re-estimated on any real deployment cohort."
    )

    tile_specs = [
        ("ROC-AUC", f"{metrics['roc_auc']:.3f}", "discrimination · held-out test", PALETTE["cyan"]),
        ("PR-AUC", f"{metrics['pr_auc']:.3f}", f"baseline prevalence {positives/test_n:.1%}", PALETTE["indigo"]),
        ("Sensitivity", f"{metrics['sensitivity']:.1%}", f"recall at threshold {metrics['threshold']:.2f}", PALETTE["critical"]),
        ("Specificity", f"{metrics['specificity']:.1%}", "true negative rate", PALETTE["low"]),
        ("F1 score", f"{metrics['f1']:.3f}", "precision / recall balance", PALETTE["high"]),
        ("Brier score", f"{metrics['brier']:.3f}", "probability calibration · lower better", PALETTE["teal"]),
    ]
    for col, spec in zip(st.columns(6, gap="small"), tile_specs):
        with col:
            C.md(_metric_tile(*spec))

    left, right = st.columns([1.08, 1], gap="large")
    with left:
        with C.panel("pf_roc", "Receiver operating characteristic", "All candidate models · held-out test partition", "target"):
            st.plotly_chart(_roc(bundle), use_container_width=True, config=charts.PLOT_CONFIG)
        with C.panel("pf_pr", "Precision–recall curve", "Positive class is a documented 24h deterioration event", "activity"):
            st.plotly_chart(_pr(bundle), use_container_width=True, config=charts.PLOT_CONFIG)
    with right:
        with C.panel("pf_cm", "Confusion matrix", f"{primary} · operating threshold {metrics['threshold']:.3f}", "grid"):
            st.plotly_chart(_confusion(bundle), use_container_width=True, config=charts.PLOT_CONFIG)
            C.md(
                f'<div class="panel-note">TP {metrics["tp"]} · FP {metrics["fp"]} · FN {metrics["fn"]} · TN {metrics["tn"]} '
                f'on {test_n} held-out assessments.</div>'
            )
        with C.panel("pf_cal", "Calibration", "Observed event rate against predicted probability", "gauge"):
            st.plotly_chart(_calibration(bundle), use_container_width=True, config=charts.PLOT_CONFIG)

    with C.panel("pf_compare", "Model comparison", "Test-partition metrics at each model's out-of-fold Youden threshold", "layers"):
        rows = []
        for name, m in bundle["metrics"].items():
            rows.append({
                "Model": f"★ {name}" if name == primary else name,
                "Threshold": f"{m['threshold']:.3f}",
                "ROC-AUC": f"{m['roc_auc']:.3f}",
                "PR-AUC": f"{m['pr_auc']:.3f}",
                "Precision": f"{m['precision']:.1%}",
                "Recall": f"{m['recall']:.1%}",
                "F1": f"{m['f1']:.3f}",
                "Specificity": f"{m['specificity']:.1%}",
                "Brier": f"{m['brier']:.3f}",
                "CV ROC-AUC": f"{bundle['cv'][name]['roc_auc_mean']:.3f} ± {bundle['cv'][name]['roc_auc_std']:.3f}",
            })
        C.md(
            '<table class="mtable"><thead><tr>'
            + "".join(f"<th>{c}</th>" for c in rows[0])
            + "</tr></thead><tbody>"
            + "".join(
                "<tr>" + "".join(f'<td class="{"num" if i else ""}">{v}</td>' for i, v in enumerate(row.values())) + "</tr>"
                for row in rows
            )
            + "</tbody></table>"
        )
        C.md('<div class="panel-note">★ primary model, selected by highest mean cross-validated ROC-AUC among tree '
              'ensembles. Test metrics are reported once and were not used for selection.</div>')

    lower_left, lower_right = st.columns([1.05, 1], gap="large")
    with lower_left:
        with C.panel("pf_cv", "Cross-validation stability", f"{primary} · {training.get('cv_folds', 5)}-fold ROC-AUC on the training partition", "refresh"):
            st.plotly_chart(_cv(bundle), use_container_width=True, config=charts.PLOT_CONFIG)
    with lower_right:
        with C.panel("pf_sweep", "Threshold trade-off", "Test-partition metrics across decision thresholds", "sliders"):
            st.plotly_chart(_sweep(bundle), use_container_width=True, config=charts.PLOT_CONFIG)

    with C.panel("pf_method", "Evaluation methodology", "Why these numbers can be read as held-out estimates", "clipboard-check"):
        C.md(
            f'<div class="method-grid">'
            f'<div class="method-item"><span>Dataset</span><b>{training.get("dataset_label", slug)}</b></div>'
            f'<div class="method-item"><span>Patients</span><b>{training.get("n_patients", "—")} total · '
            f'{training.get("n_train", "—")} train · {training.get("n_test", "—")} test</b></div>'
            f'<div class="method-item"><span>Split</span><b>Stratified {int((1-training.get("test_size", 0.3))*100)}/{int(training.get("test_size", 0.3)*100)} '
            f'· seed {training.get("split_seed", "—")}</b></div>'
            f'<div class="method-item"><span>Target</span><b>{training.get("target", "24h deterioration")}</b></div>'
            f'<div class="method-item"><span>Selection</span><b>{training.get("primary_selection", "CV ROC-AUC")}</b></div>'
            f'<div class="method-item"><span>Trained</span><b>{pd.Timestamp(training.get("trained_at")).strftime("%d %b %Y %H:%M UTC") if training.get("trained_at") else "—"}</b></div>'
            f'</div>'
        )
        C.md(
            '<div class="panel-note">Hyperparameters were searched with randomised cross-validation on the training '
            'partition only. Operating thresholds come from out-of-fold predictions (Youden J). The test partition was '
            'scored once after training. Class imbalance is reported through PR-AUC and Brier score in addition to ROC-AUC.</div>'
        )
