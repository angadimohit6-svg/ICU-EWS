"""Training and evaluation pipeline.

All reported metrics are computed from a held-out stratified test split of the
active dataset. Operating thresholds are selected on out-of-fold predictions
from the training partition only (Youden's J), then applied once to the test
partition. Nothing in the evaluation is hand-set or synthetic.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import features as F

MODEL_VERSION = "ICU-EWS v1.0.0"
SPLIT_SEED = 42
TEST_SIZE = 0.30
CV_FOLDS = 5

XGB_PARAMS = {
    "n_estimators": 380,
    "max_depth": 4,
    "learning_rate": 0.045,
    "subsample": 0.9,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "reg_lambda": 1.2,
    "gamma": 0.1,
    "random_state": SPLIT_SEED,
    "n_jobs": 4,
    "eval_metric": "logloss",
    "tree_method": "hist",
}

XGB_SEARCH_SPACE = {
    "n_estimators": [220, 320, 420, 550],
    "max_depth": [2, 3, 4],
    "learning_rate": [0.03, 0.05, 0.08],
    "subsample": [0.75, 0.85, 1.0],
    "colsample_bytree": [0.5, 0.65, 0.8],
    "min_child_weight": [1, 3, 5],
    "reg_lambda": [0.3, 1.0, 2.5],
    "scale_pos_weight": [1.0, 2.0, 4.0],
}


@dataclass
class TrainingSpec:
    split_seed: int = SPLIT_SEED
    test_size: float = TEST_SIZE
    cv_folds: int = CV_FOLDS


def _models(xgb_params: dict | None = None) -> dict[str, object]:
    params = dict(XGB_PARAMS)
    if xgb_params:
        params.update(xgb_params)
    return {
        "XGBoost": xgb.XGBClassifier(**params),
        "Random Forest": Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=320,
                        max_depth=8,
                        min_samples_leaf=4,
                        max_features="sqrt",
                        random_state=SPLIT_SEED,
                        n_jobs=4,
                        class_weight="balanced_subsample",
                    ),
                ),
            ]
        ),
        "Logistic Regression": Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("model", LogisticRegression(max_iter=1200, C=0.4, class_weight="balanced")),
            ]
        ),
    }


def _proba(model, X: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(X)[:, 1]


def _threshold_from_oof(y_true: pd.Series, oof: np.ndarray) -> tuple[float, float]:
    fpr, tpr, thresholds = roc_curve(y_true, oof)
    specificity = 1 - fpr
    j = tpr + specificity - 1
    idx = int(np.argmax(j))
    threshold = float(np.clip(thresholds[idx], 0.05, 0.95))
    return threshold, float(j[idx])


def _metrics_at(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) else float("nan")
    return {
        "threshold": round(float(threshold), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_prob)), 4),
        "pr_auc": round(float(average_precision_score(y_true, y_prob)), 4),
        "accuracy": round(float((tp + tn) / max(tp + tn + fp + fn, 1)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "sensitivity": round(float(sensitivity), 4),
        "specificity": round(float(specificity), 4),
        "brier": round(float(brier_score_loss(y_true, y_prob)), 4),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


def _downsample(arr: np.ndarray, n: int = 240) -> list[float]:
    arr = np.asarray(arr, dtype=float)
    if len(arr) <= n:
        return [round(float(v), 5) for v in arr]
    idx = np.linspace(0, len(arr) - 1, n).astype(int)
    return [round(float(v), 5) for v in arr[idx]]


def _calibration(y_true: np.ndarray, y_prob: np.ndarray, bins: int = 8) -> dict:
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=bins, strategy="quantile")
    return {
        "mean_predicted": [round(float(v), 4) for v in prob_pred],
        "fraction_positive": [round(float(v), 4) for v in prob_true],
        "bins": bins,
    }


def train_bundle(
    patients: pd.DataFrame,
    observations: pd.DataFrame,
    labs: pd.DataFrame,
    dataset_meta: dict,
    spec: TrainingSpec | None = None,
) -> dict:
    spec = spec or TrainingSpec()
    X, y, meta = F.training_frame(patients, observations, labs)
    if len(X) < 40:
        raise ValueError("Not enough patients with observations to train a model.")

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X,
        y,
        np.arange(len(X)),
        test_size=spec.test_size,
        random_state=spec.split_seed,
        stratify=y,
    )
    meta_test = meta.iloc[idx_test].reset_index(drop=True)

    cv = StratifiedKFold(n_splits=spec.cv_folds, shuffle=True, random_state=spec.split_seed)

    # Hyperparameters for XGBoost are selected on the training partition only.
    search = RandomizedSearchCV(
        xgb.XGBClassifier(random_state=spec.split_seed, n_jobs=4, eval_metric="logloss", tree_method="hist"),
        param_distributions=XGB_SEARCH_SPACE,
        n_iter=16,
        scoring="roc_auc",
        cv=StratifiedKFold(n_splits=4, shuffle=True, random_state=spec.split_seed),
        random_state=spec.split_seed,
        n_jobs=4,
        refit=True,
    )
    search.fit(X_train, y_train)
    best_params = {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in search.best_params_.items()}
    best_params = {
        k: (int(v) if isinstance(v, (np.integer,)) else v) for k, v in best_params.items()
    }

    models = _models(best_params)

    metrics: dict[str, dict] = {}
    curves: dict[str, dict] = {}
    cv_results: dict[str, dict] = {}
    probabilities: dict[str, np.ndarray] = {}
    thresholds: dict[str, float] = {}

    for name, model in models.items():
        model.fit(X_train, y_train)

        oof = cross_val_predict(model, X_train, y_train, cv=cv, method="predict_proba", n_jobs=1)[:, 1]
        threshold, youden = _threshold_from_oof(y_train, oof)
        thresholds[name] = threshold

        cv_scores = cross_validate(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring={"roc_auc": "roc_auc", "pr_auc": "average_precision", "brier": "neg_brier_score"},
            n_jobs=1,
        )
        cv_results[name] = {
            "roc_auc_folds": [round(float(v), 4) for v in cv_scores["test_roc_auc"]],
            "pr_auc_folds": [round(float(v), 4) for v in cv_scores["test_pr_auc"]],
            "brier_folds": [round(float(-v), 4) for v in cv_scores["test_brier"]],
            "roc_auc_mean": round(float(cv_scores["test_roc_auc"].mean()), 4),
            "roc_auc_std": round(float(cv_scores["test_roc_auc"].std()), 4),
            "pr_auc_mean": round(float(cv_scores["test_pr_auc"].mean()), 4),
            "brier_mean": round(float(-cv_scores["test_brier"].mean()), 4),
            "youden_j_oof": round(youden, 4),
            "oof_threshold": round(threshold, 4),
        }

        proba = _proba(model, X_test)
        probabilities[name] = proba
        metrics[name] = _metrics_at(y_test.to_numpy(), proba, threshold)

        fpr, tpr, _ = roc_curve(y_test, proba)
        prec, rec, _ = precision_recall_curve(y_test, proba)
        curves[name] = {
            "roc": {"fpr": _downsample(fpr), "tpr": _downsample(tpr)},
            "pr": {"precision": _downsample(prec), "recall": _downsample(rec)},
            "calibration": _calibration(y_test.to_numpy(), proba),
        }

    # Primary model = best cross-validated ROC-AUC among the tree ensembles.
    tree_candidates = ["XGBoost", "Random Forest"]
    primary = max(tree_candidates, key=lambda n: cv_results[n]["roc_auc_mean"])
    primary_model = models[primary]

    if primary == "XGBoost":
        importance_map = primary_model.get_booster().get_score(importance_type="gain")
        importance = sorted(
            (
                {"feature": feat, "gain": float(gain), "label": F.feature_label(feat)}
                for feat, gain in importance_map.items()
            ),
            key=lambda d: d["gain"],
            reverse=True,
        )
        importance_source = "Booster gain"
    else:
        raw_importance = primary_model.named_steps["model"].feature_importances_
        importance = sorted(
            (
                {"feature": feat, "gain": float(value), "label": F.feature_label(feat)}
                for feat, value in zip(F.FEATURE_COLUMNS, raw_importance)
            ),
            key=lambda d: d["gain"],
            reverse=True,
        )
        importance_source = "Impurity importance"
    total_gain = sum(item["gain"] for item in importance) or 1.0
    for item in importance:
        item["share"] = round(item["gain"] / total_gain, 4)

    # threshold sweep on the held-out test partition for the primary model
    sweep_thresholds = np.round(np.arange(0.05, 0.951, 0.025), 3)
    sweep = []
    for t in sweep_thresholds:
        yp = (probabilities[primary] >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, yp, labels=[0, 1]).ravel()
        sweep.append(
            {
                "threshold": float(t),
                "precision": round(float(tp / max(tp + fp, 1)), 4),
                "recall": round(float(tp / max(tp + fn, 1)), 4),
                "specificity": round(float(tn / max(tn + fp, 1)), 4),
                "f1": round(float(2 * tp / max(2 * tp + fp + fn, 1)), 4),
            }
        )

    background = X_train.sample(n=min(160, len(X_train)), random_state=spec.split_seed)

    risk_anchors = F.risk_anchors(patients)
    risk_features = F.build_feature_frame(patients, observations, labs, risk_anchors)
    risk_proba = _proba(primary_model, risk_features[F.FEATURE_COLUMNS].astype(float))
    risk_table = risk_features[["patient_id", "anchor_hour"]].copy()
    risk_table["risk_prob"] = np.round(risk_proba, 6)
    risk_table = risk_table.merge(
        patients[["patient_id", "admit_ts"]], on="patient_id", how="left"
    )
    risk_table["assessment_ts"] = risk_table["admit_ts"] + pd.to_timedelta(risk_table["anchor_hour"], unit="h")
    risk_table = risk_table.drop(columns=["admit_ts"]).sort_values(["patient_id", "anchor_hour"])

    fingerprint = hashlib.sha1(
        "|".join(sorted(patients["patient_id"])).encode()
        + f"|{len(observations)}|{len(labs)}|{spec.split_seed}".encode()
    ).hexdigest()[:12]

    training_info = {
        "dataset_slug": dataset_meta.get("slug"),
        "dataset_label": dataset_meta.get("label"),
        "n_patients": int(len(patients)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "positive_rate_train": round(float(y_train.mean()), 4),
        "positive_rate_test": round(float(y_test.mean()), 4),
        "n_features": int(X.shape[1]),
        "test_size": spec.test_size,
        "split_seed": spec.split_seed,
        "cv_folds": spec.cv_folds,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_version": MODEL_VERSION,
        "primary_model": primary,
        "primary_selection": f"Highest {spec.cv_folds}-fold cross-validated ROC-AUC among tree ensembles",
        "fingerprint": fingerprint,
        "target": "Documented deterioration event within 24h of assessment",
        "xgb_params": best_params,
        "xgb_search": {
            "n_iter": 16,
            "cv_folds": 4,
            "scoring": "roc_auc",
            "best_cv_auc": round(float(search.best_score_), 4),
        },
    }

    return {
        "models": models,
        "primary_model": primary,
        "importance_source": importance_source,
        "hyperparameters": {"XGBoost": best_params},
        "metrics": metrics,
        "curves": curves,
        "cv": cv_results,
        "thresholds": thresholds,
        "threshold_sweep": sweep,
        "importance": importance,
        "feature_names": F.FEATURE_COLUMNS,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train.to_numpy(),
        "y_test": y_test.to_numpy(),
        "proba": probabilities,
        "meta_test": meta_test,
        "background": background,
        "risk_table": risk_table,
        "training": training_info,
        "feature_meta": {k: dict(v) for k, v in F.FEATURE_META.items()},
    }


def bundle_json(meta: dict, training: dict) -> dict:
    """Small human-readable sidecar written next to the joblib bundle."""
    return {
        **{k: v for k, v in meta.items()},
        "training": training,
    }


def dumps(obj) -> str:
    return json.dumps(obj, indent=2, default=str)
