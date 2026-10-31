"""Cached data access for the UI layer.

Streamlit-facing; business logic lives in :mod:`icu_ews.ml` and
:mod:`icu_ews.data`. All caches are keyed by dataset slug so switching the
active dataset never mixes results.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from ..ml import artifacts
from ..ml import features as F
from ..ml.explain import ShapResult, compute_shap
from . import alerts as alerts_mod
from .mews import add_mews

LAB_COLS = ["lactate", "wbc", "creatinine", "platelets", "hemoglobin", "crp", "bilirubin", "sodium", "potassium"]


@st.cache_data(show_spinner=False)
def load_dataset(slug: str) -> dict:
    frames = artifacts.load_frames(slug)
    meta = artifacts.read_meta(slug)
    return {**frames, "meta": meta}


@st.cache_resource(show_spinner=False)
def load_bundle(slug: str) -> dict:
    return artifacts.load_model_bundle(slug)


@st.cache_data(show_spinner=False)
def dataset_summary(slug: str) -> dict:
    ds = load_dataset(slug)
    patients = ds["patients"]
    return {
        "slug": slug,
        "label": ds["meta"].get("label", slug),
        "is_demo": bool(ds["meta"].get("is_demo", False)),
        "description": ds["meta"].get("description", ""),
        "generated_at": ds["meta"].get("generated_at"),
        "generator": ds["meta"].get("generator"),
        "seed": ds["meta"].get("seed"),
        "n_patients": int(len(patients)),
        "n_observations": int(len(ds["observations"])),
        "n_labs": int(len(ds["labs"])),
        "first_admit": patients["admit_ts"].min(),
        "latest_ts": patients["snapshot_ts"].max(),
        "earliest_snapshot": patients["snapshot_ts"].min(),
        "positive_rate": float(patients["deterioration_24h"].mean()),
        "event_count": int(patients["deterioration_24h"].sum()),
        "training": ds["meta"].get("training", {}),
        "metrics": ds["meta"].get("metrics", {}),
    }


@st.cache_data(show_spinner=False)
def patient_snapshot(slug: str) -> pd.DataFrame:
    ds = load_dataset(slug)
    patients = ds["patients"].copy()
    obs = add_mews(ds["observations"]).sort_values(["patient_id", "hour"])

    latest = obs.groupby("patient_id", as_index=False).tail(1)
    latest = latest.rename(columns={"hour": "vitals_hour", "ts": "vitals_ts"})
    latest = latest[
        [
            "patient_id",
            "vitals_hour",
            "vitals_ts",
            "heart_rate",
            "respiratory_rate",
            "spo2",
            "systolic_bp",
            "diastolic_bp",
            "map",
            "temperature",
            "avpu",
            "mews",
        ]
    ]

    win = obs.merge(patients[["patient_id", "assessment_hour"]], on="patient_id", how="left")
    win = win[win["hour"] > win["assessment_hour"] - 8].copy()
    win["_present"] = win[F.CORE_VITALS].notna().sum(axis=1)
    completeness = (
        win.groupby("patient_id")
        .agg(present=("_present", "sum"), rows=("hour", "size"))
        .reset_index()
    )
    completeness["completeness"] = (
        completeness["present"] / (completeness["rows"] * len(F.CORE_VITALS))
    ).clip(0, 1)
    completeness["obs_count_8h"] = completeness["rows"]
    completeness = completeness[["patient_id", "completeness", "obs_count_8h"]]

    labs = ds["labs"].sort_values(["patient_id", "hour"])
    lab_latest = labs.groupby("patient_id", as_index=False).tail(1)
    lab_latest = lab_latest.rename(columns={"hour": "labs_hour", "ts": "labs_ts"})[
        ["patient_id", "labs_hour", "labs_ts"] + LAB_COLS
    ]

    risk = load_bundle(slug)["risk_table"].sort_values(["patient_id", "anchor_hour"]).copy()
    risk["prev_prob"] = risk.groupby("patient_id")["risk_prob"].shift(1)
    risk_latest = risk.groupby("patient_id", as_index=False).tail(1)
    risk_latest = risk_latest[["patient_id", "risk_prob", "prev_prob", "assessment_ts"]].rename(
        columns={"assessment_ts": "predicted_ts"}
    )

    snap = (
        patients.merge(latest, on="patient_id", how="left")
        .merge(completeness, on="patient_id", how="left")
        .merge(lab_latest, on="patient_id", how="left")
        .merge(risk_latest, on="patient_id", how="left")
    )
    snap["risk_delta"] = snap["risk_prob"] - snap["prev_prob"]
    snap["completeness"] = snap["completeness"].fillna(0.0).clip(0, 1)
    snap["age_group"] = pd.cut(
        snap["age"], bins=[17, 44, 64, 74, 84, 100], labels=["18–44", "45–64", "65–74", "75–84", "85+"]
    )
    return snap.sort_values("risk_prob", ascending=False).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def risk_timeline(slug: str) -> pd.DataFrame:
    return load_bundle(slug)["risk_table"].sort_values(["patient_id", "anchor_hour"]).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def raw_alerts(slug: str, t_critical: float, t_high: float, t_moderate: float) -> pd.DataFrame:
    thresholds = {"critical": t_critical, "high": t_high, "moderate": t_moderate}
    return alerts_mod.build_alerts(patient_snapshot(slug), thresholds)


def alerts(slug: str, thresholds: dict[str, float]) -> pd.DataFrame:
    frame = raw_alerts(slug, thresholds["critical"], thresholds["high"], thresholds["moderate"]).copy()
    return alerts_mod.apply_state(frame)


def alert_counts_by_patient(slug: str, thresholds: dict[str, float]) -> pd.DataFrame:
    frame = alerts(slug, thresholds)
    open_alerts = frame[~frame["resolved"]]
    if open_alerts.empty:
        return pd.DataFrame(columns=["patient_id", "open_alerts", "critical_alerts", "high_alerts"])
    return (
        open_alerts.groupby("patient_id")
        .agg(
            open_alerts=("id", "count"),
            critical_alerts=("category", lambda s: int((s == "critical").sum())),
            high_alerts=("category", lambda s: int((s == "high").sum())),
        )
        .reset_index()
    )


@st.cache_data(show_spinner=False)
def patient_detail(slug: str, patient_id: str) -> dict:
    ds = load_dataset(slug)
    patients = ds["patients"].set_index("patient_id")
    if patient_id not in patients.index:
        return {}
    patient = patients.loc[patient_id]
    obs = add_mews(ds["observations"][ds["observations"]["patient_id"] == patient_id]).sort_values("hour")
    labs = ds["labs"][ds["labs"]["patient_id"] == patient_id].sort_values("hour")
    risk = risk_timeline(slug)
    risk = risk[risk["patient_id"] == patient_id].reset_index(drop=True)
    return {"patient": patient, "obs": obs, "labs": labs, "risk": risk}


@st.cache_data(show_spinner=False)
def feature_row(slug: str, patient_id: str) -> pd.Series:
    detail = patient_detail(slug, patient_id)
    patient = detail["patient"]
    row = F.patient_features_at(patient, detail["obs"], detail["labs"], int(patient["assessment_hour"]))
    return pd.Series(row)


@st.cache_data(show_spinner=False)
def shap_test(slug: str) -> ShapResult:
    bundle = load_bundle(slug)
    model = bundle["models"][bundle["primary_model"]]
    return compute_shap(model, bundle["X_test"], bundle["feature_names"])


@st.cache_data(show_spinner=False)
def shap_patient(slug: str, patient_id: str) -> ShapResult:
    bundle = load_bundle(slug)
    model = bundle["models"][bundle["primary_model"]]
    row = feature_row(slug, patient_id)
    X = pd.DataFrame([row[F.FEATURE_COLUMNS].astype(float)])
    return compute_shap(model, X, bundle["feature_names"])


@st.cache_data(show_spinner=False)
def patient_feature_matrix(slug: str) -> pd.DataFrame:
    bundle = load_bundle(slug)
    return bundle["X_test"].copy()


def thresholds_tuple(thresholds: dict[str, float]) -> tuple[float, float, float]:
    return (thresholds["critical"], thresholds["high"], thresholds["moderate"])


def probability_for_feature_row(slug: str, row: pd.Series) -> float:
    bundle = load_bundle(slug)
    model = bundle["models"][bundle["primary_model"]]
    X = pd.DataFrame([row[F.FEATURE_COLUMNS].astype(float)])
    return float(model.predict_proba(X)[0, 1])
