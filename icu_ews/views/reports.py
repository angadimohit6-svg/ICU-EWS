"""Research reports workspace: exports generated from the active artifacts."""

from __future__ import annotations

import io
import json
import zipfile
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version

import numpy as np
import pandas as pd
import streamlit as st

from ..analytics import band_counts, with_bands
from ..config import PALETTE, RISK_META, RISK_ORDER
from ..data import repository
from ..ml import features as F
from ..state import thresholds
from ..ui import components as C


def _stat(label: str, value: str, sub: str, icon_name: str, color: str) -> str:
    return f"""
<div class="astat" style="--astat-color:{color}">
  <div class="astat-top"><span>{label}</span>{C.icon_html(icon_name, 14)}</div>
  <div class="astat-value">{value}</div>
  <div class="astat-sub">{sub}</div>
</div>"""


def _lib_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not installed"


def _snapshot_frame(slug: str, th: dict[str, float]) -> pd.DataFrame:
    frame = with_bands(repository.patient_snapshot(slug), th).copy()
    columns = [
        "patient_id", "age", "sex", "admission_type", "diagnosis_group", "bed", "admit_ts",
        "assessment_hour", "snapshot_ts", "risk_prob", "risk_band", "risk_delta", "completeness",
        "mews", "heart_rate", "respiratory_rate", "spo2", "systolic_bp", "diastolic_bp", "temperature",
        "lactate", "wbc", "creatinine", "crp", "deterioration_24h",
    ]
    frame = frame[[c for c in columns if c in frame.columns]]
    for col in ("admit_ts", "snapshot_ts"):
        if col in frame:
            frame[col] = pd.to_datetime(frame[col]).dt.strftime("%Y-%m-%dT%H:%M:%S")
    return frame


def _metrics_frame(bundle: dict) -> pd.DataFrame:
    rows = []
    for name, m in bundle["metrics"].items():
        cv = bundle["cv"][name]
        rows.append(
            {
                "model": name,
                "primary": name == bundle["primary_model"],
                "threshold": m["threshold"],
                "roc_auc": m["roc_auc"],
                "pr_auc": m["pr_auc"],
                "accuracy": m["accuracy"],
                "precision": m["precision"],
                "recall": m["recall"],
                "f1": m["f1"],
                "sensitivity": m["sensitivity"],
                "specificity": m["specificity"],
                "brier": m["brier"],
                "tp": m["tp"], "fp": m["fp"], "fn": m["fn"], "tn": m["tn"],
                "cv_roc_auc_mean": cv["roc_auc_mean"],
                "cv_roc_auc_std": cv["roc_auc_std"],
            }
        )
    return pd.DataFrame(rows)


def _importance_frame(bundle: dict) -> pd.DataFrame:
    frame = pd.DataFrame(bundle["importance"])
    frame["group"] = [bundle["feature_meta"].get(f, {}).get("group", "") for f in frame["feature"]]
    frame["unit"] = [bundle["feature_meta"].get(f, {}).get("unit", "") for f in frame["feature"]]
    return frame[["feature", "label", "group", "unit", "gain", "share"]]


def _dictionary_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"feature": key, "label": meta["label"], "group": meta["group"], "unit": meta["unit"]}
            for key, meta in F.FEATURE_META.items()
        ]
    )


def _alert_frame(slug: str, th: dict[str, float]) -> pd.DataFrame:
    frame = repository.alerts(slug, th).copy()
    frame = frame[
        ["id", "patient_id", "bed", "category", "title", "detail", "metric", "value", "ts", "status", "acknowledged_at", "resolved_at"]
    ]
    frame["ts"] = pd.to_datetime(frame["ts"]).dt.strftime("%Y-%m-%dT%H:%M:%S")
    return frame


def _summary_payload(slug: str, summary: dict, bundle: dict, th: dict[str, float], counts: dict, alerts: pd.DataFrame) -> dict:
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "application": "ICU-EWS",
        "dataset": {
            "slug": slug,
            "label": summary["label"],
            "is_demo": summary["is_demo"],
            "description": summary["description"],
            "generated_at": summary["generated_at"],
            "generator": summary.get("generator"),
            "seed": summary.get("seed"),
            "n_patients": summary["n_patients"],
            "n_observations": summary["n_observations"],
            "n_labs": summary["n_labs"],
            "positive_rate": summary["positive_rate"],
            "event_count": summary["event_count"],
        },
        "training": summary["training"],
        "metrics": bundle["metrics"],
        "cv": bundle["cv"],
        "thresholds": {"applied": th, "model": bundle["thresholds"]},
        "risk_band_counts": counts,
        "alerts": {
            "total": int(len(alerts)),
            "open": int((alerts["status"] != "Resolved").sum()),
            "resolved": int((alerts["status"] == "Resolved").sum()),
        },
        "libraries": {
            name: _lib_version(name)
            for name in ("streamlit", "pandas", "numpy", "scikit-learn", "xgboost", "shap", "plotly")
        },
    }


def _markdown_report(slug: str, summary: dict, bundle: dict, th: dict[str, float], counts: dict, alerts: pd.DataFrame) -> str:
    training = bundle["training"]
    primary = bundle["primary_model"]
    metrics = bundle["metrics"]
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# ICU-EWS cohort analysis report",
        "",
        f"Generated {now} from artifact `{slug}` (fingerprint `{training.get('fingerprint', '—')}`).",
        "",
        "> Demonstration output. The cohort is synthetic and no real patient data is included. "
        "Metrics describe this cohort only and do not establish clinical effectiveness.",
        "",
        "## Dataset",
        "",
        f"- Label: {summary['label']}",
        f"- Patients: {summary['n_patients']}",
        f"- Observations: {summary['n_observations']}",
        f"- Laboratory panels: {summary['n_labs']}",
        f"- Documented 24h deterioration events: {summary['event_count']} ({summary['positive_rate']:.1%})",
        f"- Target: {training.get('target', '—')}",
        "",
        "## Model",
        "",
        f"- Primary model: {primary}",
        f"- Version: {training.get('model_version', '—')}",
        f"- Training / test patients: {training.get('n_train', '—')} / {training.get('n_test', '—')}",
        f"- Features: {training.get('n_features', len(bundle['feature_names']))}",
        f"- Split seed: {training.get('split_seed', '—')} · {training.get('cv_folds', '—')}-fold cross-validation",
        f"- Selection rule: {training.get('primary_selection', '—')}",
        "",
        "## Held-out performance",
        "",
        "| Model | Threshold | ROC-AUC | PR-AUC | Sensitivity | Specificity | F1 | Brier |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, m in metrics.items():
        star = " (primary)" if name == primary else ""
        lines.append(
            f"| {name}{star} | {m['threshold']:.3f} | {m['roc_auc']:.3f} | {m['pr_auc']:.3f} | "
            f"{m['sensitivity']:.1%} | {m['specificity']:.1%} | {m['f1']:.3f} | {m['brier']:.3f} |"
        )
    lines += [
        "",
        "Metrics are computed on the held-out test partition. Thresholds come from out-of-fold predictions "
        "on the training partition (Youden J).",
        "",
        "## Operating thresholds in this session",
        "",
        f"- Critical: {th['critical']:.0%}",
        f"- High: {th['high']:.0%}",
        f"- Moderate: {th['moderate']:.0%}",
        "",
        "## Risk band distribution",
        "",
        "| Band | Patients |",
        "| --- | ---: |",
    ]
    for band in RISK_ORDER:
        lines.append(f"| {RISK_META[band]['label']} | {counts.get(band, 0)} |")
    lines += [
        "",
        "## Alert summary",
        "",
        f"- Total derived alerts: {len(alerts)}",
        f"- Unresolved: {int((alerts['status'] != 'Resolved').sum())}",
        f"- Resolved: {int((alerts['status'] == 'Resolved').sum())}",
        "",
        "Alerts are derived from recorded measurements and model outputs for the demonstration cohort. "
        "No notification is transmitted to any clinical system.",
        "",
        "## Top features by model importance",
        "",
        f"Source: {bundle.get('importance_source', '—')} from the {primary} pipeline.",
        "",
        "| Feature | Group | Share |",
        "| --- | --- | ---: |",
    ]
    for item in bundle["importance"][:10]:
        group = bundle["feature_meta"].get(item["feature"], {}).get("group", "")
        lines.append(f"| {item['label']} | {group} | {item['share'] * 100:.2f}% |")
    lines += [
        "",
        "## Reproducibility",
        "",
        f"- Dataset generated: {summary['generated_at']}",
        f"- Model trained: {training.get('trained_at', '—')}",
        f"- Artifact fingerprint: `{training.get('fingerprint', '—')}`",
        "- Rebuild instructions: `python scripts/build_dataset.py --slug demo-cohort-v1 && "
        "python scripts/train_model.py --slug demo-cohort-v1` (see project README).",
        "",
    ]
    return "\n".join(lines)


def _zip_bundle(files: dict[str, bytes | str]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    return buffer.getvalue()


def _report_card(
    key: str,
    title: str,
    subtitle: str,
    icon_name: str,
    description: str,
    data,
    file_name: str,
    mime: str,
    rows: str = "",
) -> None:
    with C.panel(f"rep_{key}", title, subtitle, icon_name):
        C.md(f'<div class="rep-desc">{description}</div>{rows}')
        st.download_button(
            "Download",
            data=data,
            file_name=file_name,
            mime=mime,
            key=f"rep_dl_{key}",
            use_container_width=True,
        )


def render() -> None:
    slug = st.session_state.dataset
    th = thresholds()
    summary = repository.dataset_summary(slug)
    bundle = repository.load_bundle(slug)
    training = bundle["training"]
    primary = bundle["primary_model"]
    snap = with_bands(repository.patient_snapshot(slug), th)
    counts = band_counts(snap, th)
    alerts = repository.alerts(slug, th)
    metrics = bundle["metrics"][primary]
    stamp = datetime.now(timezone.utc)

    C.page_header(
        "Research reports",
        "Download cohort data, evaluation metrics, feature attributions and an analysis narrative. Every export "
        "is generated from the active artifact and the thresholds currently applied in this session.",
        chips=(
            C.chip(f"{summary['label']}", "cyan", dot=True)
            + C.chip(f"Fingerprint {training.get('fingerprint', '—')}", "muted")
            + C.chip(f"Model {training.get('model_version', '—')}", "indigo")
            + C.chip(f"Generated {stamp:%d %b %Y %H:%M} UTC", "muted")
        ),
    )
    C.demo_banner(
        "Reports describe the synthetic demonstration cohort only. Exports contain pseudonymous identifiers "
        "from the generated dataset; no real patient data, hospital identifiers or protected health information "
        "is present."
    )

    specs = [
        ("Cohort", str(summary["n_patients"]), f"{summary['n_observations']:,} observations · {summary['n_labs']:,} lab panels", "users", PALETTE["cyan"]),
        ("Events", str(summary["event_count"]), f"{summary['positive_rate']:.1%} documented 24h deterioration", "activity", PALETTE["critical"]),
        ("Features", str(training.get("n_features", len(bundle["feature_names"]))), f"{len(bundle['X_test'])} held-out assessments", "layers", PALETTE["indigo"]),
        ("Primary AUC", f"{metrics['roc_auc']:.3f}", f"{primary} · held-out test", "target", PALETTE["low"]),
    ]
    for col, spec in zip(st.columns(4, gap="small"), specs):
        with col:
            C.md(_stat(*spec))

    snapshot_csv = _snapshot_frame(slug, th)
    metrics_csv = _metrics_frame(bundle)
    importance_csv = _importance_frame(bundle)
    alerts_csv = _alert_frame(slug, th)
    dictionary_csv = _dictionary_frame()
    payload = _summary_payload(slug, summary, bundle, th, counts, alerts)
    markdown = _markdown_report(slug, summary, bundle, th, counts, alerts)

    main, rail = st.columns([1.72, 1], gap="large")

    with main:
        with C.panel("rep_bundle", "Analysis report bundle", "One archive with the full reproducible export set", "package"):
            C.md(
                f'<div class="rep-desc">A single ZIP containing the analysis narrative, cohort snapshot, held-out '
                f'metrics, feature attributions, alert log, data dictionary and a machine-readable summary. '
                f'Generated {stamp:%d %b %Y %H:%M UTC}.</div>'
                f'<div class="method-grid" style="margin:.65rem 0 .8rem">'
                f'<div class="method-item"><span>Narrative</span><b>README.md</b></div>'
                f'<div class="method-item"><span>Cohort</span><b>{len(snapshot_csv)} assessments</b></div>'
                f'<div class="method-item"><span>Metrics</span><b>{len(metrics_csv)} models</b></div>'
                f'<div class="method-item"><span>Attributions</span><b>{len(importance_csv)} features</b></div>'
                f'<div class="method-item"><span>Alerts</span><b>{len(alerts_csv)} derived alerts</b></div>'
                f'<div class="method-item"><span>Summary</span><b>cohort_summary.json</b></div>'
                f'</div>'
            )
            archive = _zip_bundle(
                {
                    "README.md": markdown,
                    "cohort_snapshot.csv": snapshot_csv.to_csv(index=False),
                    "model_metrics.csv": metrics_csv.to_csv(index=False),
                    "feature_importance.csv": importance_csv.to_csv(index=False),
                    "alert_log.csv": alerts_csv.to_csv(index=False),
                    "data_dictionary.csv": dictionary_csv.to_csv(index=False),
                    "cohort_summary.json": json.dumps(payload, indent=2, default=str),
                }
            )
            st.download_button(
                "Download report bundle (.zip)",
                data=archive,
                file_name=f"icu_ews_{slug}_report_bundle.zip",
                mime="application/zip",
                key="rep_dl_bundle",
                use_container_width=True,
            )

        C.section_header("Individual exports", "Each file can be downloaded separately", "file-text")
        left, right = st.columns(2, gap="medium")
        with left:
            _report_card(
                "snapshot", "Cohort snapshot", "Latest assessment per patient", "clipboard-list",
                "Pseudonymous ID, demographics, latest physiology, model probability, risk band and observed outcome.",
                snapshot_csv.to_csv(index=False), f"icu_ews_{slug}_cohort_snapshot.csv", "text/csv",
            )
            _report_card(
                "importance", "Feature attribution", "Global model importance", "brain",
                "Gain-based importance share for every model input, with feature group and unit.",
                importance_csv.to_csv(index=False), f"icu_ews_{slug}_feature_importance.csv", "text/csv",
            )
            _report_card(
                "dictionary", "Data dictionary", "Model input definitions", "book-open",
                f"{len(dictionary_csv)} documented features across patient, physiology, laboratory and data-quality groups.",
                dictionary_csv.to_csv(index=False), f"icu_ews_{slug}_data_dictionary.csv", "text/csv",
            )
        with right:
            _report_card(
                "metrics", "Model evaluation", "Held-out test metrics", "gauge",
                "Per-model threshold, discrimination, calibration and confusion-matrix counts with cross-validation stability.",
                metrics_csv.to_csv(index=False), f"icu_ews_{slug}_model_metrics.csv", "text/csv",
            )
            _report_card(
                "alerts", "Alert log", "Derived alert history", "bell-ring",
                "Every derived alert with category, source rule, measured value, timestamp and current workflow status.",
                alerts_csv.to_csv(index=False), f"icu_ews_{slug}_alert_log.csv", "text/csv",
            )
            _report_card(
                "summary", "Analysis narrative", "Markdown report", "file-text",
                "Dataset description, model configuration, evaluation table, threshold settings and importance summary.",
                markdown, f"icu_ews_{slug}_analysis_report.md", "text/markdown",
            )

    with rail:
        with C.panel("rep_repro", "Reproducibility", "Artifact lineage for this export", "fingerprint"):
            C.md(
                C.stat_row("Dataset", summary["label"], "database")
                + C.stat_row("Slug", slug, "tag")
                + C.stat_row("Generator", summary.get("generator") or "—", "cpu")
                + C.stat_row("Generator seed", str(summary.get("seed", "—")), "hash")
                + C.stat_row("Split seed", str(training.get("split_seed", "—")), "shuffle")
                + C.stat_row("CV folds", str(training.get("cv_folds", "—")), "refresh")
                + C.stat_row("Artifact fingerprint", training.get("fingerprint", "—"), "fingerprint")
            )
            C.md(
                f'<div class="panel-note">Trained {C.fmt_ts(training.get("trained_at"), with_year=True)} · '
                f'dataset generated {C.fmt_ts(summary.get("generated_at"), with_year=True)}.</div>'
            )

        with C.panel("rep_runtime", "Runtime", "Versions used to produce these files", "terminal"):
            C.md(
                "".join(
                    C.stat_row(name, _lib_version(name), "package")
                    for name in ("streamlit", "pandas", "numpy", "scikit-learn", "xgboost", "shap", "plotly")
                )
            )

        with C.panel("rep_notes", "Reporting notes", "How to read these exports", "info"):
            C.md(
                '<div class="panel-note" style="margin-top:0">'
                "• Metrics are reported once on the held-out partition and were not used for model or threshold selection.<br>"
                "• Risk bands depend on the session thresholds; the snapshot export records the bands in force at download time.<br>"
                "• Alert state reflects this workspace's acknowledgement and resolution actions.<br>"
                "• SHAP and importance values explain model behaviour, not clinical causality.<br>"
                "• Do not interpret any result as evidence of clinical effectiveness."
                "</div>"
            )
        with C.panel("rep_rebuild", "Rebuild locally", "Deterministic pipeline commands", "refresh"):
            C.md(
                '<div class="method-grid" style="grid-template-columns:1fr">'
                '<div class="method-item"><span>Dataset</span><b>python scripts/build_dataset.py --slug demo-cohort-v1</b></div>'
                '<div class="method-item"><span>Training</span><b>python scripts/train_model.py --slug demo-cohort-v1</b></div>'
                "</div>"
            )
            C.md('<div class="panel-note">Both steps use fixed seeds; rerunning reproduces the same cohort and split.</div>')
