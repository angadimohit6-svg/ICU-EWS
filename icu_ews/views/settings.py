"""Workspace settings: thresholds, dataset registry, workflow state."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from ..analytics import band_counts, with_bands
from ..config import APP_LONG_NAME, APP_NAME, DEFAULT_THRESHOLDS, PALETTE
from ..data import repository
from ..data import alerts as alerts_mod
from ..ml import artifacts
from ..state import navigate, switch_dataset, thresholds
from ..ui import components as C


def _sync_threshold_order() -> None:
    ss = st.session_state
    moderate = int(ss.get("set_moderate", round(DEFAULT_THRESHOLDS["moderate"] * 100)))
    high = int(ss.get("set_high", round(DEFAULT_THRESHOLDS["high"] * 100)))
    critical = int(ss.get("set_critical", round(DEFAULT_THRESHOLDS["critical"] * 100)))
    if high <= moderate:
        ss["set_high"] = min(96, moderate + 1)
    if critical <= high:
        ss["set_critical"] = min(99, high + 1)


def _reset_thresholds() -> None:
    st.session_state.thresholds = dict(DEFAULT_THRESHOLDS)
    st.session_state["set_moderate"] = round(DEFAULT_THRESHOLDS["moderate"] * 100)
    st.session_state["set_high"] = round(DEFAULT_THRESHOLDS["high"] * 100)
    st.session_state["set_critical"] = round(DEFAULT_THRESHOLDS["critical"] * 100)
    st.rerun()


def _clear_workflow() -> None:
    alerts_mod.reset_state()
    st.session_state["set_confirm_reset"] = False


def render() -> None:
    ss = st.session_state
    slug = ss.dataset
    summary = repository.dataset_summary(slug)
    bundle = repository.load_bundle(slug)
    training = bundle["training"]
    primary = bundle["primary_model"]
    current = dict(thresholds())

    C.page_header(
        "Settings",
        "Operating configuration for this demonstration workspace. Threshold changes apply immediately to every "
        "risk band, alert queue and cohort count in the application.",
        chips=(
            C.chip(f"{APP_NAME} · v1.0.0", "cyan", dot=True)
            + C.chip(summary["label"], "muted")
            + C.chip(f"Model · {primary}", "indigo")
            + C.chip("Demonstration mode", "amber")
        ),
    )
    C.demo_banner(
        "Settings never change the trained model. Band thresholds are product configuration for this demo and are "
        "not validated clinical cut-offs. All changes persist for this local session only."
    )

    left, right = st.columns([1.05, 1], gap="large")

    with left:
        with C.panel("set_thresholds", "Operating thresholds", "Predicted 24h deterioration probability bands", "sliders"):
            C.md(
                f'<div class="panel-note" style="margin:0 0 .55rem">Current model operating threshold from '
                f'out-of-fold Youden selection: <b>{bundle["thresholds"][primary]:.3f}</b>. Band cut-offs below '
                f'are independent of that classification threshold.</div>'
            )
            st.slider(
                "Moderate band ≥ (%)", 5, 90, step=1, format="%d%%",
                value=round(float(current["moderate"]) * 100), key="set_moderate", on_change=_sync_threshold_order,
            )
            st.slider(
                "High band ≥ (%)", 6, 96, step=1, format="%d%%",
                value=round(float(current["high"]) * 100), key="set_high", on_change=_sync_threshold_order,
            )
            st.slider(
                "Critical band ≥ (%)", 7, 99, step=1, format="%d%%",
                value=round(float(current["critical"]) * 100), key="set_critical", on_change=_sync_threshold_order,
            )
            desired = {
                "moderate": int(ss["set_moderate"]) / 100,
                "high": int(ss["set_high"]) / 100,
                "critical": int(ss["set_critical"]) / 100,
            }
            C.md(
                f'<div class="panel-note">Effective bands · moderate ≥ {desired["moderate"]:.0%} · '
                f'high ≥ {desired["high"]:.0%} · critical ≥ {desired["critical"]:.0%}</div>'
            )
            b1, b2 = st.columns([1.2, 1])
            b1.button("Reset to defaults", key="set_reset", use_container_width=True, on_click=_reset_thresholds)
            b2.markdown(
                f'<div class="panel-note" style="margin:.55rem 0 0;text-align:right">'
                f'Default {DEFAULT_THRESHOLDS["moderate"]:.0%} / {DEFAULT_THRESHOLDS["high"]:.0%} / '
                f'{DEFAULT_THRESHOLDS["critical"]:.0%}</div>',
                unsafe_allow_html=True,
            )

            if desired != current:
                ss.thresholds = desired
                st.rerun()

        snap = with_bands(repository.patient_snapshot(slug), ss.thresholds)
        counts = band_counts(snap, ss.thresholds)
        alerts = repository.alerts(slug, ss.thresholds)
        open_alerts = int((~alerts["resolved"]).sum())
        with C.panel("set_impact", "Band impact", "Recomputed from every assessment under the active thresholds", "target"):
            C.md(
                C.stat_row("Critical band", f"{counts['critical']} patients", "alert-octagon", PALETTE["critical"])
                + C.stat_row("High band", f"{counts['high']} patients", "alert-triangle", PALETTE["high"])
                + C.stat_row("Moderate band", f"{counts['moderate']} patients", "activity", PALETTE["moderate"])
                + C.stat_row("Low band", f"{counts['low']} patients", "shield-check", PALETTE["low"])
                + C.stat_row("Unresolved alerts", f"{open_alerts}", "bell-ring", PALETTE["info"])
            )
            C.md(
                f'<div class="panel-note">Probabilities are unchanged; only the band mapping, patient cards, KPI counts '
                f'and derived model alerts follow these cut-offs. {len(snap)} assessments scored.</div>'
            )

        acknowledged = int(((alerts["status"] == "Acknowledged")).sum())
        resolved = int((alerts["status"] == "Resolved").sum())
        with C.panel("set_workflow", "Review workflow state", "Acknowledgement and resolution records", "inbox"):
            C.md(
                C.stat_row("Total derived alerts", str(len(alerts)), "bell-ring")
                + C.stat_row("Acknowledged", str(acknowledged), "eye", PALETTE["indigo"])
                + C.stat_row("Resolved", str(resolved), "check-circle", PALETTE["low"])
                + C.stat_row("Storage", "Local demo workspace", "database")
            )
            confirm = st.checkbox(
                "I understand this clears every acknowledgement and resolution record for this workspace.",
                key="set_confirm_reset",
            )
            st.button(
                "Clear alert workflow state", key="set_clear_alerts", disabled=not confirm,
                use_container_width=True, on_click=_clear_workflow,
            )
            C.md('<div class="panel-note">Derived alerts remain; only review status is cleared.</div>')

    with right:
        bundles = artifacts.list_bundles()
        labels = {b["slug"]: (b.get("short_label") or b.get("label") or b["slug"]) for b in bundles}
        options = list(labels) or [slug]
        with C.panel("set_dataset", "Dataset registry", "Active artifact and cohort provenance", "database"):
            selected = st.selectbox(
                "Active dataset",
                options,
                index=options.index(slug) if slug in options else 0,
                format_func=lambda s: labels.get(s, s),
                key=f"set_dataset_{slug}",
            )
            if selected != slug:
                switch_dataset(selected)
                st.rerun()
            C.md(
                C.stat_row("Cohort", summary["label"], "users")
                + C.stat_row("Synthetic demo", "Yes" if summary["is_demo"] else "No", "shield")
                + C.stat_row("Generator", summary.get("generator") or "—", "cpu")
                + C.stat_row("Generator seed", str(summary.get("seed", "—")), "hash")
                + C.stat_row("Patients", str(summary["n_patients"]), "users")
                + C.stat_row("Observations", f"{summary['n_observations']:,}", "activity")
                + C.stat_row("Laboratory panels", f"{summary['n_labs']:,}", "flask")
                + C.stat_row(
                    "Snapshot window",
                    f"{pd.Timestamp(summary['earliest_snapshot']):%d %b} – "
                    f"{pd.Timestamp(summary['latest_ts']):%d %b %Y}",
                    "calendar",
                )
            )

        with C.panel("set_model", "Model registry", "Training lineage for the active artifact", "cpu"):
            metrics = bundle["metrics"][primary]
            C.md(
                C.stat_row("Primary model", primary, "layers")
                + C.stat_row("Model version", training.get("model_version", "—"), "tag")
                + C.stat_row("Artifact fingerprint", training.get("fingerprint", "—"), "fingerprint")
                + C.stat_row("Trained", C.fmt_ts(training.get("trained_at"), with_year=True), "clock")
                + C.stat_row(
                    "Train / test patients",
                    f"{training.get('n_train', '—')} / {training.get('n_test', '—')}",
                    "shuffle",
                )
                + C.stat_row("Features", str(training.get("n_features", len(bundle["feature_names"]))), "list")
                + C.stat_row("Held-out ROC-AUC", f"{metrics['roc_auc']:.3f}", "target", PALETTE["low"])
                + C.stat_row("Held-out Brier", f"{metrics['brier']:.3f}", "gauge", PALETTE["cyan"])
            )
            b1, b2 = st.columns(2)
            if b1.button("Open performance", key="set_open_perf", use_container_width=True):
                navigate("performance")
            if b2.button("Open model card", key="set_open_reports", use_container_width=True):
                navigate("reports")

        with C.panel("set_integrity", "Data integrity", "What this workspace does and does not do", "shield-check"):
            C.md(
                '<div class="method-grid" style="grid-template-columns:1fr">'
                '<div class="method-item"><span>Data source</span><b>Synthetic cohort generator · no PHI</b></div>'
                '<div class="method-item"><span>Inference</span><b>Local model artifacts · no external API</b></div>'
                '<div class="method-item"><span>Notifications</span><b>None sent · no hospital integration</b></div>'
                '<div class="method-item"><span>Clinical use</span><b>Demonstration only · not validated</b></div>'
                "</div>"
            )
            C.md(
                '<div class="panel-note">Predictions, SHAP values and evaluation metrics are computed from the active '
                'artifacts. Demonstration alerts are derived locally and are never transmitted.</div>'
            )

        with C.panel("set_maintenance", "Maintenance", "Deterministic rebuild pipeline", "terminal"):
            C.md(
                '<div class="method-grid" style="grid-template-columns:1fr">'
                '<div class="method-item"><span>Rebuild cohort</span><b>python scripts/build_dataset.py --slug demo-cohort-v1</b></div>'
                '<div class="method-item"><span>Retrain models</span><b>python scripts/train_model.py --slug demo-cohort-v1</b></div>'
                "</div>"
            )
            C.md(
                f'<div class="panel-note">{APP_NAME} · {APP_LONG_NAME}. Fixed seeds make the cohort and split '
                f'reproducible; thresholds and workflow state are session-scoped.</div>'
            )
