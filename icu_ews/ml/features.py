"""Feature engineering for the ICU-EWS deterioration model.

Features are built from a per-patient *assessment window*: all observations up
to the assessment hour. Nothing after the assessment hour is ever used, so the
same function powers training, live scoring and historical risk trajectories
without leakage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..data.mews import add_mews

WINDOW_HOURS = 8

# name -> (label, group, unit)
FEATURE_META: dict[str, dict[str, str]] = {
    "age": {"label": "Age", "group": "Patient", "unit": "years"},
    "sex_male": {"label": "Male sex", "group": "Patient", "unit": ""},
    "comorbidity_count": {"label": "Comorbidity burden", "group": "Patient", "unit": "conditions"},
    "adm_surgical": {"label": "Admission: surgical", "group": "Patient", "unit": ""},
    "adm_neurological": {"label": "Admission: neurological", "group": "Patient", "unit": ""},
    "adm_cardiac": {"label": "Admission: cardiac", "group": "Patient", "unit": ""},
    "adm_respiratory": {"label": "Admission: respiratory", "group": "Patient", "unit": ""},
    "assessment_hour": {"label": "Hours since ICU admission", "group": "Patient", "unit": "h"},
    "heart_rate": {"label": "Heart rate (latest)", "group": "Vital signs", "unit": "bpm"},
    "respiratory_rate": {"label": "Respiratory rate (latest)", "group": "Vital signs", "unit": "/min"},
    "spo2": {"label": "SpO₂ (latest)", "group": "Vital signs", "unit": "%"},
    "systolic_bp": {"label": "Systolic BP (latest)", "group": "Vital signs", "unit": "mmHg"},
    "diastolic_bp": {"label": "Diastolic BP (latest)", "group": "Vital signs", "unit": "mmHg"},
    "map": {"label": "Mean arterial pressure", "group": "Vital signs", "unit": "mmHg"},
    "temperature": {"label": "Temperature (latest)", "group": "Vital signs", "unit": "°C"},
    "mews": {"label": "MEWS score (latest)", "group": "Vital signs", "unit": "points"},
    "shock_index": {"label": "Shock index (HR / SBP)", "group": "Vital signs", "unit": ""},
    "avpu_worst": {"label": "Worst consciousness level", "group": "Vital signs", "unit": "AVPU"},
    "hr_mean_8h": {"label": "Heart rate — 8h mean", "group": "Trends", "unit": "bpm"},
    "rr_mean_8h": {"label": "Respiratory rate — 8h mean", "group": "Trends", "unit": "/min"},
    "spo2_mean_8h": {"label": "SpO₂ — 8h mean", "group": "Trends", "unit": "%"},
    "sbp_mean_8h": {"label": "Systolic BP — 8h mean", "group": "Trends", "unit": "mmHg"},
    "temp_mean_8h": {"label": "Temperature — 8h mean", "group": "Trends", "unit": "°C"},
    "spo2_min_8h": {"label": "SpO₂ — 8h minimum", "group": "Trends", "unit": "%"},
    "sbp_min_8h": {"label": "Systolic BP — 8h minimum", "group": "Trends", "unit": "mmHg"},
    "hr_max_8h": {"label": "Heart rate — 8h maximum", "group": "Trends", "unit": "bpm"},
    "rr_max_8h": {"label": "Respiratory rate — 8h maximum", "group": "Trends", "unit": "/min"},
    "temp_max_8h": {"label": "Temperature — 8h maximum", "group": "Trends", "unit": "°C"},
    "hr_slope_8h": {"label": "Heart rate trend", "group": "Trends", "unit": "bpm/h"},
    "spo2_slope_8h": {"label": "SpO₂ trend", "group": "Trends", "unit": "%/h"},
    "sbp_slope_8h": {"label": "Systolic BP trend", "group": "Trends", "unit": "mmHg/h"},
    "mews_max_8h": {"label": "MEWS — 8h maximum", "group": "Trends", "unit": "points"},
    "hypoxia_events": {"label": "SpO₂ < 92% episodes (8h)", "group": "Acuity events", "unit": "count"},
    "tachycardia_events": {"label": "HR > 120 episodes (8h)", "group": "Acuity events", "unit": "count"},
    "hypotension_events": {"label": "SBP < 100 episodes (8h)", "group": "Acuity events", "unit": "count"},
    "lactate": {"label": "Lactate (latest)", "group": "Laboratory", "unit": "mmol/L"},
    "wbc": {"label": "White cell count", "group": "Laboratory", "unit": "10⁹/L"},
    "creatinine": {"label": "Creatinine", "group": "Laboratory", "unit": "mg/dL"},
    "platelets": {"label": "Platelets", "group": "Laboratory", "unit": "10⁹/L"},
    "hemoglobin": {"label": "Hemoglobin", "group": "Laboratory", "unit": "g/dL"},
    "crp": {"label": "C-reactive protein", "group": "Laboratory", "unit": "mg/L"},
    "bilirubin": {"label": "Bilirubin", "group": "Laboratory", "unit": "mg/dL"},
    "sodium": {"label": "Sodium", "group": "Laboratory", "unit": "mmol/L"},
    "potassium": {"label": "Potassium", "group": "Laboratory", "unit": "mmol/L"},
    "hours_since_labs": {"label": "Hours since last labs", "group": "Data quality", "unit": "h"},
    "obs_count_8h": {"label": "Observations in 8h window", "group": "Data quality", "unit": "count"},
    "data_completeness": {"label": "Data completeness", "group": "Data quality", "unit": "fraction"},
}

FEATURE_COLUMNS: list[str] = list(FEATURE_META)

CORE_VITALS = ["heart_rate", "respiratory_rate", "spo2", "systolic_bp", "diastolic_bp", "temperature"]
AVPU_MAP = {"A": 0, "V": 1, "P": 2, "U": 3}


def feature_label(name: str) -> str:
    return FEATURE_META.get(name, {}).get("label", name)


def feature_unit(name: str) -> str:
    return FEATURE_META.get(name, {}).get("unit", "")


def _slope(frame: pd.DataFrame, col: str) -> float:
    sub = frame[["hour", col]].dropna()
    if len(sub) < 2:
        return np.nan
    dh = sub["hour"].iloc[-1] - sub["hour"].iloc[0]
    if dh <= 0:
        return np.nan
    return float((sub[col].iloc[-1] - sub[col].iloc[0]) / dh)


def _mean(frame: pd.DataFrame, col: str) -> float:
    return float(frame[col].mean()) if frame[col].notna().any() else np.nan


def _min(frame: pd.DataFrame, col: str) -> float:
    return float(frame[col].min()) if frame[col].notna().any() else np.nan


def _max(frame: pd.DataFrame, col: str) -> float:
    return float(frame[col].max()) if frame[col].notna().any() else np.nan


def _last(row: pd.Series, col: str) -> float:
    value = row.get(col, np.nan)
    return float(value) if pd.notna(value) else np.nan


def patient_features_at(
    patient: pd.Series,
    obs: pd.DataFrame,
    labs: pd.DataFrame,
    anchor_hour: int,
) -> dict[str, float]:
    """Compute one model input row for a patient at a given assessment hour."""
    window = obs[obs["hour"] <= anchor_hour]
    last8 = window[window["hour"] > anchor_hour - WINDOW_HOURS]
    last = window.iloc[-1] if not window.empty else pd.Series(dtype=float)
    lab_window = labs[labs["hour"] <= anchor_hour]
    lab_last = lab_window.iloc[-1] if not lab_window.empty else None

    feats: dict[str, float] = {
        "age": float(patient["age"]),
        "sex_male": 1.0 if str(patient["sex"]).upper().startswith("M") else 0.0,
        "comorbidity_count": float(patient["comorbidity_count"]),
        "adm_surgical": 1.0 if patient["admission_type"] == "Surgical" else 0.0,
        "adm_neurological": 1.0 if patient["admission_type"] == "Neurological" else 0.0,
        "adm_cardiac": 1.0 if patient["admission_type"] == "Cardiac" else 0.0,
        "adm_respiratory": 1.0 if patient["admission_type"] == "Respiratory" else 0.0,
        "assessment_hour": float(anchor_hour),
    }

    for col in ["heart_rate", "respiratory_rate", "spo2", "systolic_bp", "diastolic_bp", "map", "temperature"]:
        feats[col] = _last(last, col)

    hr = feats["heart_rate"]
    sbp = feats["systolic_bp"]
    feats["shock_index"] = float(hr / sbp) if pd.notna(hr) and pd.notna(sbp) and sbp > 0 else np.nan
    feats["mews"] = _last(last, "mews")
    avpu_vals = window["avpu"].map(AVPU_MAP).dropna()
    feats["avpu_worst"] = float(avpu_vals.max()) if not avpu_vals.empty else np.nan

    feats["hr_mean_8h"] = _mean(last8, "heart_rate")
    feats["rr_mean_8h"] = _mean(last8, "respiratory_rate")
    feats["spo2_mean_8h"] = _mean(last8, "spo2")
    feats["sbp_mean_8h"] = _mean(last8, "systolic_bp")
    feats["temp_mean_8h"] = _mean(last8, "temperature")
    feats["spo2_min_8h"] = _min(last8, "spo2")
    feats["sbp_min_8h"] = _min(last8, "systolic_bp")
    feats["hr_max_8h"] = _max(last8, "heart_rate")
    feats["rr_max_8h"] = _max(last8, "respiratory_rate")
    feats["temp_max_8h"] = _max(last8, "temperature")
    feats["mews_max_8h"] = _max(last8, "mews")
    feats["hr_slope_8h"] = _slope(last8, "heart_rate")
    feats["spo2_slope_8h"] = _slope(last8, "spo2")
    feats["sbp_slope_8h"] = _slope(last8, "systolic_bp")

    feats["hypoxia_events"] = float((last8["spo2"] < 92).sum())
    feats["tachycardia_events"] = float((last8["heart_rate"] > 120).sum())
    feats["hypotension_events"] = float((last8["systolic_bp"] < 100).sum())

    if lab_last is not None:
        for col in ["lactate", "wbc", "creatinine", "platelets", "hemoglobin", "crp", "bilirubin", "sodium", "potassium"]:
            feats[col] = _last(lab_last, col)
        feats["hours_since_labs"] = float(anchor_hour - int(lab_last["hour"]))
    else:
        for col in ["lactate", "wbc", "creatinine", "platelets", "hemoglobin", "crp", "bilirubin", "sodium", "potassium"]:
            feats[col] = np.nan
        feats["hours_since_labs"] = np.nan

    observed = window[CORE_VITALS].notna().sum().sum()
    expected = max(len(window) * len(CORE_VITALS), 1)
    feats["data_completeness"] = float(observed / expected)
    feats["obs_count_8h"] = float(len(last8))

    return {name: feats.get(name, np.nan) for name in FEATURE_COLUMNS}


def build_feature_frame(
    patients: pd.DataFrame,
    observations: pd.DataFrame,
    labs: pd.DataFrame,
    anchors: pd.DataFrame,
) -> pd.DataFrame:
    """Build feature rows for (patient_id, anchor_hour) pairs.

    Returns a DataFrame indexed like ``anchors`` with patient_id, anchor_hour
    and every feature column.
    """
    obs = add_mews(observations)
    obs_groups = {pid: g for pid, g in obs.groupby("patient_id", sort=False)}
    lab_groups = {pid: g for pid, g in labs.groupby("patient_id", sort=False)}
    patient_rows = {row["patient_id"]: row for _, row in patients.iterrows()}

    records: list[dict] = []
    for pid, anchor in anchors[["patient_id", "anchor_hour"]].itertuples(index=False):
        patient = patient_rows.get(pid)
        obs_p = obs_groups.get(pid)
        labs_p = lab_groups.get(pid, labs.iloc[0:0])
        if patient is None or obs_p is None:
            continue
        feats = patient_features_at(patient, obs_p, labs_p, int(anchor))
        feats.update({"patient_id": pid, "anchor_hour": int(anchor)})
        records.append(feats)

    return pd.DataFrame.from_records(records)


def training_frame(
    patients: pd.DataFrame,
    observations: pd.DataFrame,
    labs: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    anchors = patients[["patient_id", "assessment_hour"]].rename(columns={"assessment_hour": "anchor_hour"})
    frame = build_feature_frame(patients, observations, labs, anchors)
    frame = frame.merge(patients[["patient_id", "deterioration_24h"]], on="patient_id", how="left")
    X = frame[FEATURE_COLUMNS].astype(float)
    y = frame["deterioration_24h"].astype(int)
    meta = frame[["patient_id", "anchor_hour"]].copy()
    return X, y, meta


def risk_anchors(patients: pd.DataFrame, step: int = 6) -> pd.DataFrame:
    """Assessment hours used to build each patient's risk trajectory."""
    records = []
    for row in patients[["patient_id", "assessment_hour"]].itertuples(index=False):
        end = int(row.assessment_hour)
        hours = sorted(set(list(range(WINDOW_HOURS, end + 1, step)) + [end]))
        for h in hours:
            records.append({"patient_id": row.patient_id, "anchor_hour": h})
    return pd.DataFrame.from_records(records)


def apply_manual_override(base: pd.Series, vitals: dict[str, float], labs: dict[str, float]) -> pd.Series:
    """Overlay clinician-entered values onto a feature row.

    Latest values, 8-hour summaries and event counts are rebuilt from the
    entered values so the manual form remains internally consistent.
    """
    row = base.copy()
    for key, value in vitals.items():
        if key in row and pd.notna(value):
            row[key] = float(value)

    hr = vitals.get("heart_rate")
    spo2 = vitals.get("spo2")
    sbp = vitals.get("systolic_bp")
    if hr is not None and pd.notna(hr):
        row["hr_mean_8h"] = row["hr_max_8h"] = float(hr)
        row["tachycardia_events"] = float(1 if hr > 120 else 0)
    if spo2 is not None and pd.notna(spo2):
        row["spo2_mean_8h"] = row["spo2_min_8h"] = float(spo2)
        row["hypoxia_events"] = float(1 if spo2 < 92 else 0)
    if sbp is not None and pd.notna(sbp):
        row["sbp_mean_8h"] = row["sbp_min_8h"] = float(sbp)
        row["hypotension_events"] = float(1 if sbp < 100 else 0)
    if vitals.get("respiratory_rate") is not None:
        row["rr_mean_8h"] = row["rr_max_8h"] = float(vitals["respiratory_rate"])
    if vitals.get("temperature") is not None:
        row["temp_mean_8h"] = row["temp_max_8h"] = float(vitals["temperature"])
    if hr is not None and sbp is not None and pd.notna(hr) and pd.notna(sbp) and sbp > 0:
        row["shock_index"] = float(hr / sbp)
    for key, value in labs.items():
        if key in row and pd.notna(value):
            row[key] = float(value)
    return row
