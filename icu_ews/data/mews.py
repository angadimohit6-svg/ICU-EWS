"""Modified Early Warning Score (MEWS) — standard five-parameter implementation.

Reference table (Subbe et al., 2001): heart rate, systolic blood pressure,
respiratory rate, temperature and AVPU consciousness level. Scores are summed
to 0-14. Every component is computed from recorded bedside observations; no
imputed values are scored.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

AVPU_SCORE = {"A": 0, "V": 1, "P": 2, "U": 3}


def _hr_score(v: float) -> float:
    if pd.isna(v):
        return np.nan
    if v < 40:
        return 2
    if v <= 50:
        return 1
    if v <= 100:
        return 0
    if v <= 110:
        return 1
    if v <= 129:
        return 2
    return 3


def _sbp_score(v: float) -> float:
    if pd.isna(v):
        return np.nan
    if v <= 70:
        return 3
    if v <= 80:
        return 2
    if v <= 100:
        return 1
    if v <= 199:
        return 0
    return 2


def _rr_score(v: float) -> float:
    if pd.isna(v):
        return np.nan
    if v < 9:
        return 2
    if v <= 14:
        return 0
    if v <= 20:
        return 1
    if v <= 29:
        return 2
    return 3


def _temp_score(v: float) -> float:
    if pd.isna(v):
        return np.nan
    if v < 35.0:
        return 2
    if v <= 38.4:
        return 0
    return 2


def mews_components(row: pd.Series) -> dict[str, float]:
    """Return per-component MEWS points and the total (NaN if any component missing)."""
    parts = {
        "hr": _hr_score(row.get("heart_rate", np.nan)),
        "sbp": _sbp_score(row.get("systolic_bp", np.nan)),
        "rr": _rr_score(row.get("respiratory_rate", np.nan)),
        "temp": _temp_score(row.get("temperature", np.nan)),
        "avpu": AVPU_SCORE.get(str(row.get("avpu", "")), np.nan),
    }
    total = sum(parts.values()) if not any(pd.isna(v) for v in parts.values()) else np.nan
    return {**parts, "total": total}


def add_mews(df: pd.DataFrame) -> pd.DataFrame:
    """Vectorised MEWS over a long observation frame."""
    out = df.copy()
    hr = out["heart_rate"]
    sbp = out["systolic_bp"]
    rr = out["respiratory_rate"]
    temp = out["temperature"]
    avpu = out["avpu"].map(AVPU_SCORE).astype(float)

    out["mews_hr"] = np.select(
        [hr.isna(), hr < 40, hr <= 50, hr <= 100, hr <= 110, hr <= 129],
        [np.nan, 2, 1, 0, 1, 2],
        default=3,
    )
    out["mews_sbp"] = np.select(
        [sbp.isna(), sbp <= 70, sbp <= 80, sbp <= 100, sbp <= 199],
        [np.nan, 3, 2, 1, 0],
        default=2,
    )
    out["mews_rr"] = np.select(
        [rr.isna(), rr < 9, rr <= 14, rr <= 20, rr <= 29],
        [np.nan, 2, 0, 1, 2],
        default=3,
    )
    out["mews_temp"] = np.select(
        [temp.isna(), temp < 35.0, temp <= 38.4],
        [np.nan, 2, 0],
        default=2,
    )
    out["mews_avpu"] = avpu
    out["mews"] = out[["mews_hr", "mews_sbp", "mews_rr", "mews_temp", "mews_avpu"]].sum(
        axis=1, min_count=5
    )
    return out


def mews_category(score: float) -> str:
    if pd.isna(score):
        return "unknown"
    if score >= 6:
        return "critical"
    if score >= 4:
        return "high"
    if score == 3:
        return "moderate"
    return "low"
