"""SHAP explanations must be computed from the trained artifact and be additive."""

from __future__ import annotations

import numpy as np
import pytest

from icu_ews.ml.explain import compute_shap, patient_explanation

PRIMARY = "Random Forest"


def test_shap_values_are_computed_for_each_assessment(model_bundle):
    model = model_bundle["models"][PRIMARY]
    X = model_bundle["X_test"].head(8)
    result = compute_shap(model, X, model_bundle["feature_names"])

    assert result.values.shape == (8, 47)
    assert np.isfinite(result.values).all()
    assert result.link in {"identity", "logit"}
    assert result.method.startswith("shap.")


def test_shap_output_reconstructs_model_probability(model_bundle):
    model = model_bundle["models"][PRIMARY]
    X = model_bundle["X_test"].head(8)
    result = compute_shap(model, X, model_bundle["feature_names"])
    expected = model_bundle["proba"][PRIMARY][:8]

    for index in range(len(X)):
        assert result.probability(index) == pytest.approx(expected[index], abs=2e-3)


def test_patient_explanation_is_sorted_by_absolute_contribution(model_bundle):
    model = model_bundle["models"][PRIMARY]
    X = model_bundle["X_test"].head(4)
    result = compute_shap(model, X, model_bundle["feature_names"])
    explanation = patient_explanation(result, 0, top_n=10)

    contributions = explanation["contributions"]
    assert len(contributions) == 10
    assert contributions["abs"].is_monotonic_decreasing
    assert set(contributions["direction"]) <= {"increases", "decreases"}
    assert 0.0 <= explanation["probability"] <= 1.0
    assert explanation["base_value"] == pytest.approx(result.base_value)
    # Full attribution must reproduce the additive output.
    total = explanation["all_contributions"]["shap"].sum()
    assert explanation["raw_output"] == pytest.approx(explanation["base_value"] + total)


def test_global_importance_uses_mean_absolute_shap(model_bundle):
    model = model_bundle["models"][PRIMARY]
    X = model_bundle["X_test"].head(12)
    result = compute_shap(model, X, model_bundle["feature_names"])
    table = result.mean_abs()

    assert len(table) == 47
    assert table["mean_abs_shap"].is_monotonic_decreasing
    assert (table["mean_abs_shap"] >= 0).all()
