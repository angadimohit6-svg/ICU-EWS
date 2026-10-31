"""SHAP explainability for the trained ICU-EWS model.

Values are produced by shap.TreeExplainer on the actual fitted tree ensemble
(including pipelines: preprocessing is applied first, then the tree model is
attributed). If the SHAP library cannot attribute a given XGBoost build, the
exact XGBoost TreeSHAP implementation (pred_contribs) is used instead; both
produce additive attributions of the same model output.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.pipeline import Pipeline

from .features import FEATURE_META


@dataclass
class ShapResult:
    values: np.ndarray
    base_value: float
    feature_names: list[str]
    X: pd.DataFrame
    method: str
    link: str  # "logit" (additive log-odds) or "identity" (additive probability)

    def row(self, index: int) -> np.ndarray:
        return self.values[index]

    def raw_output(self, index: int) -> float:
        return float(self.base_value + self.values[index].sum())

    def probability(self, index: int) -> float:
        raw = self.raw_output(index)
        if self.link == "identity":
            return float(raw)
        return float(1.0 / (1.0 + np.exp(-raw)))

    def mean_abs(self) -> pd.DataFrame:
        frame = pd.DataFrame(
            {
                "feature": self.feature_names,
                "mean_abs_shap": np.abs(self.values).mean(axis=0),
                "label": [FEATURE_META.get(f, {}).get("label", f) for f in self.feature_names],
            }
        )
        return frame.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


def _extract(values, base_value) -> tuple[np.ndarray, float]:
    if isinstance(values, list):
        values = values[1] if len(values) > 1 else values[0]
    values = np.asarray(values, dtype=float)
    if values.ndim == 3:
        values = values[:, :, 1]
    base = np.asarray(base_value, dtype=float).ravel()
    base_float = float(base[-1] if base.size > 1 else base[0])
    return values, base_float


def _prepare(model, X: pd.DataFrame) -> tuple[object, pd.DataFrame]:
    if isinstance(model, Pipeline):
        estimator = model.steps[-1][1]
        preprocess = Pipeline(model.steps[:-1])
        transformed = preprocess.transform(X)
        transformed = np.asarray(transformed, dtype=float)
        return estimator, pd.DataFrame(transformed, columns=X.columns, index=X.index)
    return model, X


def compute_shap(model, X: pd.DataFrame, feature_names: list[str]) -> ShapResult:
    Xf = X[feature_names].astype(float)
    estimator, Xt = _prepare(model, Xf)

    method = "shap.TreeExplainer"
    try:
        explainer = shap.TreeExplainer(estimator)
        raw = explainer.shap_values(Xt)
        values, base = _extract(raw, explainer.expected_value)
        if values.shape != Xt.shape:
            raise ValueError("unexpected SHAP shape")
    except Exception:
        booster = estimator.get_booster()
        contribs = booster.predict(xgb.DMatrix(Xt, feature_names=feature_names), pred_contribs=True)
        values = contribs[:, :-1]
        base = float(contribs[0, -1])
        method = "xgboost.TreeSHAP (pred_contribs)"

    # Determine whether attributions are in probability or log-odds space by
    # comparing the additive output with the model's own probability.
    model_proba = model.predict_proba(Xf)[:, 1]
    additive = base + values.sum(axis=1)
    if np.allclose(additive, model_proba, atol=2e-3):
        link = "identity"
    elif np.allclose(1.0 / (1.0 + np.exp(-additive)), model_proba, atol=2e-3):
        link = "logit"
    else:
        link = "logit"

    return ShapResult(
        values=np.asarray(values, dtype=float),
        base_value=float(base),
        feature_names=list(feature_names),
        X=Xf.reset_index(drop=True),
        method=method,
        link=link,
    )


def patient_explanation(result: ShapResult, index: int, top_n: int = 10) -> dict:
    """Additive breakdown for one assessment, sorted by absolute impact."""
    row_values = result.values[index]
    contributions = pd.DataFrame(
        {
            "feature": result.feature_names,
            "label": [FEATURE_META.get(f, {}).get("label", f) for f in result.feature_names],
            "group": [FEATURE_META.get(f, {}).get("group", "") for f in result.feature_names],
            "unit": [FEATURE_META.get(f, {}).get("unit", "") for f in result.feature_names],
            "shap": row_values,
            "value": result.X.iloc[index].values,
        }
    )
    contributions["abs"] = contributions["shap"].abs()
    contributions["direction"] = np.where(contributions["shap"] >= 0, "increases", "decreases")
    contributions = contributions.sort_values("abs", ascending=False).reset_index(drop=True)
    probability = result.probability(index)
    return {
        "contributions": contributions.head(top_n),
        "all_contributions": contributions,
        "base_value": result.base_value,
        "raw_output": result.raw_output(index),
        "probability": probability,
        "increasing": contributions[contributions["shap"] > 0].head(5),
        "decreasing": contributions[contributions["shap"] < 0].head(5),
    }
