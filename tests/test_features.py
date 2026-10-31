"""Feature engineering contracts."""

from __future__ import annotations

import numpy as np
import pandas as pd

from icu_ews.data.mews import add_mews
from icu_ews.ml.features import FEATURE_COLUMNS, FEATURE_META, build_feature_frame


def test_feature_frame_exposes_the_canonical_feature_set(small_cohort):
    patients = small_cohort["patients"]
    anchors = patients[["patient_id", "assessment_hour"]].rename(
        columns={"assessment_hour": "anchor_hour"}
    )
    frame = build_feature_frame(
        patients, small_cohort["observations"], small_cohort["labs"], anchors
    )

    assert len(frame) == len(patients)
    for column in FEATURE_COLUMNS:
        assert column in frame.columns, f"missing feature {column}"
    assert len(FEATURE_COLUMNS) == 47
    assert len(FEATURE_META) >= len(FEATURE_COLUMNS)


def test_feature_values_are_numeric_and_within_plausible_ranges(small_cohort):
    patients = small_cohort["patients"]
    anchors = patients[["patient_id", "assessment_hour"]].rename(
        columns={"assessment_hour": "anchor_hour"}
    )
    frame = build_feature_frame(
        patients, small_cohort["observations"], small_cohort["labs"], anchors
    )
    features = frame[FEATURE_COLUMNS]
    numeric = features.apply(lambda s: np.issubdtype(s.dtype, np.number)).all()
    assert numeric

    finite = features.replace([np.inf, -np.inf], np.nan)
    assert finite["heart_rate"].dropna().between(20, 240).all()
    assert finite["spo2"].dropna().between(50, 100).all()
    assert finite["systolic_bp"].dropna().between(50, 240).all()
    assert finite["assessment_hour"].between(12, 36).all()
    assert finite["avpu_worst"].dropna().between(0, 3).all()
    assert finite["data_completeness"].between(0, 1).all()


def test_one_hot_admission_columns_are_mutually_consistent(small_cohort):
    patients = small_cohort["patients"]
    anchors = patients[["patient_id", "assessment_hour"]].rename(
        columns={"assessment_hour": "anchor_hour"}
    )
    frame = build_feature_frame(
        patients, small_cohort["observations"], small_cohort["labs"], anchors
    )
    indicators = ["adm_surgical", "adm_neurological", "adm_cardiac", "adm_respiratory"]
    # Medical is the reference category; at most one indicator is set per patient.
    assert frame[indicators].sum(axis=1).le(1).all()


def test_features_do_not_read_observations_after_the_anchor(small_cohort):
    patients = small_cohort["patients"]
    first = patients.iloc[0]
    obs = add_mews(small_cohort["observations"])
    labs = small_cohort["labs"]
    anchor = int(first["assessment_hour"])

    from icu_ews.ml.features import patient_features_at

    base = patient_features_at(first, obs, labs, anchor)

    # Append observations strictly after the anchor; feature values must not change.
    future = obs[obs["patient_id"] == first["patient_id"]].iloc[[-1]].copy()
    future["hour"] = anchor + 1
    future["ts"] = future["ts"] + pd.Timedelta(hours=1)
    future[["heart_rate", "spo2", "systolic_bp", "respiratory_rate"]] = 999.0
    tampered = patient_features_at(first, pd.concat([obs, future]), labs, anchor)

    assert base["heart_rate"] == tampered["heart_rate"]
    assert base["spo2"] == tampered["spo2"]
    assert base["sbp_mean_8h"] == tampered["sbp_mean_8h"]
    assert base["hr_max_8h"] == tampered["hr_max_8h"]
