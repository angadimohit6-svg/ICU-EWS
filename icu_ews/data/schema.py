"""Dataset schema contracts used by the bundle builder and the UI."""

from __future__ import annotations

import pandas as pd

PATIENT_COLUMNS = {
    "patient_id": "str",
    "age": "int",
    "sex": "str",
    "admission_type": "str",
    "diagnosis_group": "str",
    "comorbidity_count": "int",
    "admit_ts": "datetime",
    "los_hours": "int",
    "assessment_hour": "int",
    "snapshot_ts": "datetime",
    "deterioration_24h": "int",
    "event_hour": "float",
    "bed": "str",
}

OBSERVATION_COLUMNS = {
    "patient_id": "str",
    "hour": "int",
    "ts": "datetime",
    "heart_rate": "float",
    "respiratory_rate": "float",
    "spo2": "float",
    "systolic_bp": "float",
    "diastolic_bp": "float",
    "map": "float",
    "temperature": "float",
    "avpu": "str",
}

LAB_COLUMNS = {
    "patient_id": "str",
    "hour": "int",
    "ts": "datetime",
    "lactate": "float",
    "wbc": "float",
    "creatinine": "float",
    "platelets": "float",
    "hemoglobin": "float",
    "crp": "float",
    "bilirubin": "float",
    "sodium": "float",
    "potassium": "float",
}

VITAL_FIELDS = [
    "heart_rate",
    "respiratory_rate",
    "spo2",
    "systolic_bp",
    "diastolic_bp",
    "map",
    "temperature",
    "avpu",
]
LAB_FIELDS = [
    "lactate",
    "wbc",
    "creatinine",
    "platelets",
    "hemoglobin",
    "crp",
    "bilirubin",
    "sodium",
    "potassium",
]


def validate_bundle(patients: pd.DataFrame, observations: pd.DataFrame, labs: pd.DataFrame) -> list[str]:
    """Return a list of schema problems. An empty list means the bundle is usable."""
    problems: list[str] = []
    for name, frame, contract in (
        ("patients", patients, PATIENT_COLUMNS),
        ("observations", observations, OBSERVATION_COLUMNS),
        ("labs", labs, LAB_COLUMNS),
    ):
        if frame.empty:
            problems.append(f"{name}: table is empty")
            continue
        missing = [c for c in contract if c not in frame.columns]
        if missing:
            problems.append(f"{name}: missing columns {', '.join(missing)}")
    if problems:
        return problems

    pid = patients["patient_id"]
    if pid.duplicated().any():
        problems.append("patients: duplicate patient_id values")
    for name, frame in (("observations", observations), ("labs", labs)):
        unknown = set(frame["patient_id"]) - set(pid)
        if unknown:
            problems.append(f"{name}: {len(unknown)} patient_id values not present in patients table")
    for target in ("deterioration_24h",):
        if not patients[target].isin([0, 1]).all():
            problems.append(f"patients: {target} must be binary 0/1")
    if (patients["assessment_hour"] < 1).any():
        problems.append("patients: assessment_hour must be >= 1")
    if not pd.api.types.is_datetime64_any_dtype(patients["admit_ts"]):
        problems.append("patients: admit_ts must be a datetime column")
    return problems
