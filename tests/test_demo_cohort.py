"""Synthetic cohort generation and schema contracts."""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest

from icu_ews.data.demo import CohortSpec, generate_cohort
from icu_ews.data.schema import validate_bundle


def test_generated_cohort_passes_schema_validation(small_cohort):
    problems = validate_bundle(
        small_cohort["patients"], small_cohort["observations"], small_cohort["labs"]
    )
    assert problems == []


def test_generation_is_deterministic_for_a_fixed_seed_and_timestamp(small_cohort):
    again = generate_cohort(
        CohortSpec(n_patients=40, seed=11, as_of=datetime(2026, 10, 21, 12, 0, 0))
    )
    pd.testing.assert_frame_equal(small_cohort["patients"], again["patients"])
    pd.testing.assert_frame_equal(small_cohort["observations"], again["observations"])
    pd.testing.assert_frame_equal(small_cohort["labs"], again["labs"])


def test_patient_identifiers_are_unique_and_pseudonymous(small_cohort):
    patients = small_cohort["patients"]
    assert patients["patient_id"].is_unique
    assert patients["patient_id"].str.fullmatch(r"ICU-\d{4}").all()
    assert set(patients["sex"]) <= {"M", "F"}


def test_observations_and_labs_reference_known_patients(small_cohort):
    patients = set(small_cohort["patients"]["patient_id"])
    assert set(small_cohort["observations"]["patient_id"]) <= patients
    assert set(small_cohort["labs"]["patient_id"]) <= patients


def test_vital_ranges_are_physiologically_plausible(small_cohort):
    obs = small_cohort["observations"]
    assert obs["spo2"].dropna().between(50, 100).all()
    assert obs["heart_rate"].dropna().between(20, 240).all()
    assert obs["respiratory_rate"].dropna().between(4, 60).all()
    assert obs["systolic_bp"].dropna().between(50, 240).all()
    assert obs["temperature"].dropna().between(32, 42).all()
    assert set(obs["avpu"].dropna()) <= {"A", "V", "P", "U"}


def test_target_definition_has_both_classes(small_cohort):
    target = small_cohort["patients"]["deterioration_24h"]
    assert set(target.unique()) <= {0, 1}
    # A 40-patient draw should not be degenerate; keep the check informative if it is.
    assert target.sum() > 0, "expected at least one documented event in the fixture cohort"


def test_schema_validation_reports_missing_columns(small_cohort):
    broken = small_cohort["patients"].drop(columns=["assessment_hour"])
    problems = validate_bundle(broken, small_cohort["observations"], small_cohort["labs"])
    assert any("assessment_hour" in problem for problem in problems)


def test_event_hour_is_consistent_with_target(small_cohort):
    patients = small_cohort["patients"]
    events = patients[patients["deterioration_24h"] == 1]
    assert events["event_hour"].notna().all()
    assert (events["event_hour"] >= events["assessment_hour"]).all()
    assert patients.loc[patients["deterioration_24h"] == 0, "event_hour"].isna().all()


@pytest.mark.parametrize(
    "column,low,high",
    [
        ("lactate", 0.2, 18.0),
        ("wbc", 0.5, 42.0),
        ("creatinine", 0.2, 6.5),
        ("crp", 0.0, 380.0),
        ("platelets", 14.0, 520.0),
        ("sodium", 118.0, 158.0),
        ("potassium", 2.4, 6.9),
    ],
)
def test_laboratory_values_are_within_reported_ranges(small_cohort, column, low, high):
    labs = small_cohort["labs"]
    assert labs[column].between(low, high).all()
