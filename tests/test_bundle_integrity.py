"""Artifact integrity: stored evaluation metrics must be reproducible from saved probabilities."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
)

PRIMARY = "Random Forest"


def test_bundle_has_expected_structure(model_bundle):
    assert set(model_bundle["models"]) == {"XGBoost", "Random Forest", "Logistic Regression"}
    assert model_bundle["primary_model"] == PRIMARY
    assert len(model_bundle["feature_names"]) == 47
    assert model_bundle["X_test"].shape == (126, 47)
    assert len(model_bundle["y_test"]) == 126
    assert set(np.unique(model_bundle["y_test"])) == {0, 1}
    assert set(model_bundle["proba"]) == set(model_bundle["models"])


def test_roc_auc_pr_auc_and_brier_match_saved_metrics(model_bundle):
    y_true = np.asarray(model_bundle["y_test"])
    for name, metrics in model_bundle["metrics"].items():
        proba = model_bundle["proba"][name]
        assert roc_auc_score(y_true, proba) == pytest.approx(metrics["roc_auc"], abs=1e-4)
        assert average_precision_score(y_true, proba) == pytest.approx(metrics["pr_auc"], abs=1e-4)
        assert brier_score_loss(y_true, proba) == pytest.approx(metrics["brier"], abs=1e-4)


def test_confusion_matrix_matches_saved_counts(model_bundle):
    y_true = np.asarray(model_bundle["y_test"])
    for name, metrics in model_bundle["metrics"].items():
        predicted = (model_bundle["proba"][name] >= metrics["threshold"]).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, predicted).ravel()
        assert (tp, fp, fn, tn) == (metrics["tp"], metrics["fp"], metrics["fn"], metrics["tn"])


def test_saved_probabilities_match_model_predictions(model_bundle):
    X_test = model_bundle["X_test"]
    for name, model in model_bundle["models"].items():
        expected = model.predict_proba(X_test)[:, 1]
        assert np.allclose(expected, model_bundle["proba"][name], atol=1e-6)


def test_preprocessing_is_refitted_from_the_training_matrix(model_bundle):
    """Preprocessors are rebuilt at load so artifacts survive scikit-learn upgrades."""
    X_train = model_bundle["X_train"]
    medians = X_train.median().to_numpy()
    for name, model in model_bundle["models"].items():
        if not hasattr(model, "named_steps"):
            continue
        imputer = model.named_steps["impute"]
        assert np.allclose(imputer.statistics_, medians)
        if "scale" in model.named_steps:
            scaler = model.named_steps["scale"]
            assert scaler.mean_.shape == (X_train.shape[1],)
            assert np.all(scaler.scale_ > 0)


def test_primary_threshold_comes_from_out_of_fold_youden(model_bundle):
    thresholds = model_bundle["thresholds"]
    cv = model_bundle["cv"][PRIMARY]
    # The bundle stores the operating threshold rounded to four decimals.
    assert thresholds[PRIMARY] == pytest.approx(cv["oof_threshold"], abs=1e-4)
    assert 0.0 < thresholds[PRIMARY] < 1.0


def test_cross_validation_summary_matches_folds(model_bundle):
    for name, cv in model_bundle["cv"].items():
        folds = cv["roc_auc_folds"]
        assert len(folds) == 5
        # Fold summaries are persisted rounded to three decimals.
        assert np.mean(folds) == pytest.approx(cv["roc_auc_mean"], abs=1e-3)
        assert np.std(folds) == pytest.approx(cv["roc_auc_std"], abs=1e-3)
        assert 0.0 <= min(folds) <= max(folds) <= 1.0


def test_importance_covers_every_feature(model_bundle):
    importance = model_bundle["importance"]
    assert len(importance) == len(model_bundle["feature_names"])
    assert {row["feature"] for row in importance} == set(model_bundle["feature_names"])
    shares = [row["share"] for row in importance]
    assert all(share >= 0 for share in shares)
    assert sum(shares) == pytest.approx(1.0, abs=1e-3)


def test_risk_table_covers_patients_and_anchors(model_bundle):
    risk = model_bundle["risk_table"]
    meta = model_bundle["meta_test"]
    assert set(meta["patient_id"]) <= set(risk["patient_id"])
    assert risk["risk_prob"].between(0, 1).all()
    assert len(risk) >= len(meta)
