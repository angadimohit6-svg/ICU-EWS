"""Shared fixtures for the ICU-EWS test suite."""

from __future__ import annotations

from datetime import datetime

import pytest

from icu_ews.data.demo import CohortSpec, generate_cohort
from icu_ews.ml import artifacts

SLUG = "demo-cohort-v1"


@pytest.fixture(scope="session")
def small_cohort():
    """A deterministic, small synthetic cohort used across tests."""
    return generate_cohort(
        CohortSpec(n_patients=40, seed=11, as_of=datetime(2026, 10, 21, 12, 0, 0))
    )


@pytest.fixture(scope="session")
def model_bundle():
    if not artifacts.bundle_exists(SLUG):
        pytest.skip(f"Artifact bundle '{SLUG}' is not available")
    return artifacts.load_model_bundle(SLUG)
