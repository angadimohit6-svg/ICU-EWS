"""Reusable analytics computed from the active dataset (no UI code)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import PALETTE, RISK_ORDER, risk_from_probability


def with_bands(frame: pd.DataFrame, thresholds: dict[str, float]) -> pd.DataFrame:
    out = frame.copy()
    out["risk_band"] = [risk_from_probability(float(p), thresholds) for p in out["risk_prob"].fillna(0.0)]
    return out


def band_counts(frame: pd.DataFrame, thresholds: dict[str, float]) -> dict[str, int]:
    bands = frame["risk_prob"].fillna(0).map(lambda p: risk_from_probability(float(p), thresholds))
    counts = bands.value_counts().to_dict()
    return {band: int(counts.get(band, 0)) for band in RISK_ORDER}


def period_window(latest_ts: pd.Timestamp, days: int | None) -> tuple[pd.Timestamp, pd.Timestamp]:
    end = latest_ts
    start = end - pd.Timedelta(days=days) if days else end - pd.Timedelta(days=3650)
    return start, end


def previous_window(start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp]:
    span = end - start
    return start - span, start


def _change_pct(current: float, previous: float) -> float | None:
    if previous is None or previous == 0 or pd.isna(previous):
        return None
    return (current - previous) / previous * 100.0


def kpi_metrics(
    snapshot: pd.DataFrame,
    thresholds: dict[str, float],
    latest_ts: pd.Timestamp,
    period_days: int | None,
) -> list[dict]:
    start, end = period_window(latest_ts, period_days)
    prev_start, prev_end = previous_window(start, end)

    def in_window(frame: pd.DataFrame, lo, hi) -> pd.DataFrame:
        return frame[(frame["snapshot_ts"] >= lo) & (frame["snapshot_ts"] <= hi)]

    cohort = in_window(snapshot, start, end)
    prev = in_window(snapshot, prev_start, prev_end)

    def summarize(frame: pd.DataFrame) -> dict:
        counts = band_counts(frame, thresholds)
        return {
            "n": len(frame),
            "critical": counts["critical"],
            "high": counts["high"],
            "det_rate": float(frame["deterioration_24h"].mean() * 100) if len(frame) else np.nan,
        }

    cur, old = summarize(cohort), summarize(prev)

    daily = (
        snapshot.assign(day=snapshot["snapshot_ts"].dt.floor("D"))
        .groupby("day")
        .apply(
            lambda g: pd.Series(
                {
                    "n": len(g),
                    "critical": int((g["risk_prob"] >= thresholds["critical"]).sum()),
                    "high": int(((g["risk_prob"] >= thresholds["high"]) & (g["risk_prob"] < thresholds["critical"])).sum()),
                    "det_rate": float(g["deterioration_24h"].mean() * 100),
                }
            ),
            include_groups=False,
        )
        .reset_index()
        .sort_values("day")
    )
    trend = daily.tail(14)

    cards = [
        {
            "key": "monitored",
            "label": "Patients monitored",
            "value": cur["n"],
            "display": f"{cur['n']}",
            "unit": "patients",
            "icon": "users",
            "color": PALETTE["cyan"],
            "glow": "rgba(34, 211, 238, 0.13)",
            "delta_pct": _change_pct(cur["n"], old["n"]),
            "delta_label": "assessments",
            "foot": f"{cur['critical'] + cur['high']} require escalated review",
            "series": trend["n"].tolist(),
            "delta_mode": "pct",
        },
        {
            "key": "critical",
            "label": "Critical risk",
            "value": cur["critical"],
            "display": f"{cur['critical']}",
            "unit": "patients",
            "icon": "alert-octagon",
            "color": PALETTE["critical"],
            "glow": "rgba(244, 63, 94, 0.15)",
            "delta_pct": _change_pct(cur["critical"], old["critical"]),
            "delta_label": "caseload",
            "foot": f"{(cur['critical'] / cur['n'] * 100 if cur['n'] else 0):.1f}% of monitored patients",
            "series": trend["critical"].tolist(),
            "delta_mode": "pct",
        },
        {
            "key": "high",
            "label": "High risk",
            "value": cur["high"],
            "display": f"{cur['high']}",
            "unit": "patients",
            "icon": "alert-triangle",
            "color": PALETTE["high"],
            "glow": "rgba(251, 146, 60, 0.14)",
            "delta_pct": _change_pct(cur["high"], old["high"]),
            "delta_label": "caseload",
            "foot": f"{(cur['high'] / cur['n'] * 100 if cur['n'] else 0):.1f}% of monitored patients",
            "series": trend["high"].tolist(),
            "delta_mode": "pct",
        },
        {
            "key": "deterioration",
            "label": "Deterioration rate",
            "value": cur["det_rate"],
            "display": f"{cur['det_rate']:.1f}" if pd.notna(cur["det_rate"]) else "—",
            "unit": "%",
            "icon": "activity",
            "color": PALETTE["indigo"],
            "glow": "rgba(129, 140, 248, 0.14)",
            "delta_pct": (cur["det_rate"] - old["det_rate"]) if pd.notna(old["det_rate"]) else None,
            "delta_label": "prior period",
            "foot": f"{int(cohort['deterioration_24h'].sum())} documented events in period",
            "series": trend["det_rate"].round(2).tolist(),
            "delta_mode": "pp",
            "is_rate": True,
        },
    ]
    for card in cards:
        card["cohort"] = cohort
        card["period"] = (start, end)
    return cards


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (np.nan, np.nan)
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    margin = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, centre - margin) * 100, min(1.0, centre + margin) * 100)


def admission_trend(frame: pd.DataFrame, freq: str = "D") -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["period", "admissions", "deteriorations"])
    out = (
        frame.assign(period=frame["admit_ts"].dt.floor(freq))
        .groupby("period")
        .agg(admissions=("patient_id", "count"), deteriorations=("deterioration_24h", "sum"))
        .reset_index()
    )
    return out


def vitals_columns() -> list[str]:
    return ["heart_rate", "respiratory_rate", "spo2", "systolic_bp", "diastolic_bp", "temperature", "mews"]


def lab_columns() -> list[str]:
    return ["lactate", "wbc", "creatinine", "platelets", "hemoglobin", "crp", "bilirubin", "sodium", "potassium"]


def correlation_matrix(snapshot: pd.DataFrame) -> pd.DataFrame:
    cols = ["age", "los_hours"] + vitals_columns() + lab_columns() + ["risk_prob"]
    cols = [c for c in cols if c in snapshot.columns]
    data = snapshot[cols].apply(pd.to_numeric, errors="coerce")
    return data.corr(method="spearman").round(3)


def subgroup_table(frame: pd.DataFrame, by: str, thresholds: dict[str, float]) -> pd.DataFrame:
    rows = []
    for key, group in frame.groupby(by, dropna=False):
        n = len(group)
        events = int(group["deterioration_24h"].sum())
        low, high = wilson_ci(events, n)
        rows.append(
            {
                by: key,
                "patients": n,
                "events": events,
                "event_rate": round(events / n * 100, 1) if n else np.nan,
                "ci_low": round(low, 1) if pd.notna(low) else np.nan,
                "ci_high": round(high, 1) if pd.notna(high) else np.nan,
                "mean_risk": round(float(group["risk_prob"].mean() * 100), 1) if n else np.nan,
                "critical_share": round(
                    float((group["risk_prob"] >= thresholds["critical"]).mean() * 100), 1
                )
                if n
                else np.nan,
                "mean_los": round(float(group["los_hours"].mean()), 1) if n else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("patients", ascending=False).reset_index(drop=True)


def measurement_alerts(snapshot: pd.DataFrame, limit: int = 6) -> pd.DataFrame:
    """Latest out-of-range measurements across the cohort."""
    checks = [
        ("spo2", 92, "below", "SpO₂", "%", PALETTE["critical"]),
        ("systolic_bp", 100, "below", "Systolic BP", "mmHg", PALETTE["high"]),
        ("heart_rate", 120, "above", "Heart rate", "bpm", PALETTE["critical"]),
        ("respiratory_rate", 25, "above", "Respiratory rate", "/min", PALETTE["high"]),
        ("temperature", 38.5, "above", "Temperature", "°C", PALETTE["moderate"]),
        ("lactate", 4.0, "above", "Lactate", "mmol/L", PALETTE["high"]),
    ]
    rows = []
    for col, threshold, direction, label, unit, color in checks:
        if col not in snapshot.columns:
            continue
        series = pd.to_numeric(snapshot[col], errors="coerce")
        mask = series < threshold if direction == "below" else series > threshold
        subset = snapshot[mask].nlargest(2, "risk_prob") if direction == "above" else snapshot[mask].nsmallest(2, col)
        for _, row in subset.iterrows():
            rows.append(
                {
                    "patient_id": row["patient_id"],
                    "label": label,
                    "value": row[col],
                    "unit": unit,
                    "threshold": threshold,
                    "direction": direction,
                    "color": color,
                    "ts": row["snapshot_ts"],
                }
            )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("ts", ascending=False).head(limit).reset_index(drop=True)
