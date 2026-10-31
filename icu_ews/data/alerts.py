"""Clinical alert derivation and acknowledgement state.

Alerts are derived deterministically from recorded measurements and model
outputs in the active dataset. In demonstration mode they describe the
synthetic cohort only; no notification is transmitted anywhere.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from ..config import ALERT_STATE_FILE

SEVERITY_RANK = {"critical": 0, "high": 1, "moderate": 2, "info": 3}


def _alert_id(patient_id: str, rule: str) -> str:
    return hashlib.sha1(f"{patient_id}|{rule}".encode()).hexdigest()[:10]


def _add(
    items: list[dict],
    row: pd.Series,
    *,
    rule: str,
    category: str,
    title: str,
    detail: str,
    metric: str,
    value,
) -> None:
    offset = int(hashlib.sha1(f"{row.patient_id}|{rule}".encode()).hexdigest()[:3], 16) % 55
    ts = row["snapshot_ts"] - timedelta(minutes=offset)
    items.append(
        {
            "id": _alert_id(row.patient_id, rule),
            "patient_id": row.patient_id,
            "bed": row.get("bed", ""),
            "rule": rule,
            "category": category,
            "title": title,
            "detail": detail,
            "metric": metric,
            "value": value,
            "ts": ts,
            "risk_prob": row.get("risk_prob", np.nan),
        }
    )


def build_alerts(snapshot: pd.DataFrame, thresholds: dict[str, float]) -> pd.DataFrame:
    items: list[dict] = []
    for _, row in snapshot.iterrows():
        hr = row.get("heart_rate", np.nan)
        rr = row.get("respiratory_rate", np.nan)
        spo2 = row.get("spo2", np.nan)
        sbp = row.get("systolic_bp", np.nan)
        temp = row.get("temperature", np.nan)
        mews = row.get("mews", np.nan)
        avpu = row.get("avpu", "A")
        lactate = row.get("lactate", np.nan)
        creat = row.get("creatinine", np.nan)
        risk = row.get("risk_prob", np.nan)
        completeness = row.get("completeness", 1.0)

        if pd.notna(spo2):
            if spo2 < 88:
                _add(items, row, rule="spo2_critical", category="critical", title="Severe hypoxaemia",
                     detail=f"SpO₂ {spo2:.0f}% is below the critical threshold of 88%.", metric="SpO₂", value=f"{spo2:.0f}%")
            elif spo2 < 92:
                _add(items, row, rule="spo2_high", category="high", title="Low oxygen saturation",
                     detail=f"SpO₂ {spo2:.0f}% below 92% — supplemental oxygen review indicated.", metric="SpO₂", value=f"{spo2:.0f}%")
        if pd.notna(sbp):
            if sbp < 85:
                _add(items, row, rule="sbp_critical", category="critical", title="Significant hypotension",
                     detail=f"Systolic BP {sbp:.0f} mmHg is below 85 mmHg.", metric="Systolic BP", value=f"{sbp:.0f} mmHg")
            elif sbp < 100:
                _add(items, row, rule="sbp_high", category="high", title="Borderline hypotension",
                     detail=f"Systolic BP {sbp:.0f} mmHg below 100 mmHg.", metric="Systolic BP", value=f"{sbp:.0f} mmHg")
        if pd.notna(hr):
            if hr > 140 or hr < 40:
                _add(items, row, rule="hr_critical", category="critical", title="Critical heart rate",
                     detail=f"Heart rate {hr:.0f} bpm outside the safe monitoring range.", metric="Heart rate", value=f"{hr:.0f} bpm")
            elif hr > 120:
                _add(items, row, rule="hr_high", category="high", title="Marked tachycardia",
                     detail=f"Heart rate {hr:.0f} bpm above 120 bpm.", metric="Heart rate", value=f"{hr:.0f} bpm")
            elif hr > 110:
                _add(items, row, rule="hr_moderate", category="moderate", title="Tachycardia",
                     detail=f"Heart rate {hr:.0f} bpm above 110 bpm.", metric="Heart rate", value=f"{hr:.0f} bpm")
        if pd.notna(rr):
            if rr >= 32:
                _add(items, row, rule="rr_critical", category="critical", title="Severe tachypnoea",
                     detail=f"Respiratory rate {rr:.0f}/min at or above 32/min.", metric="Respiratory rate", value=f"{rr:.0f}/min")
            elif rr >= 25:
                _add(items, row, rule="rr_high", category="high", title="Tachypnoea",
                     detail=f"Respiratory rate {rr:.0f}/min at or above 25/min.", metric="Respiratory rate", value=f"{rr:.0f}/min")
            elif rr >= 21:
                _add(items, row, rule="rr_moderate", category="moderate", title="Raised respiratory rate",
                     detail=f"Respiratory rate {rr:.0f}/min above 20/min.", metric="Respiratory rate", value=f"{rr:.0f}/min")
        if pd.notna(temp):
            if temp >= 39.5:
                _add(items, row, rule="temp_high", category="high", title="High fever",
                     detail=f"Temperature {temp:.1f}°C above 39.5°C.", metric="Temperature", value=f"{temp:.1f}°C")
            elif temp >= 38.5:
                _add(items, row, rule="temp_moderate", category="moderate", title="Fever",
                     detail=f"Temperature {temp:.1f}°C above 38.5°C.", metric="Temperature", value=f"{temp:.1f}°C")
            elif temp < 35.0:
                _add(items, row, rule="temp_low", category="high", title="Hypothermia",
                     detail=f"Temperature {temp:.1f}°C below 35.0°C.", metric="Temperature", value=f"{temp:.1f}°C")
        if pd.notna(mews):
            if mews >= 7:
                _add(items, row, rule="mews_critical", category="critical", title="MEWS ≥ 7",
                     detail=f"MEWS {mews:.0f} — urgent senior review indicated by score trajectory.", metric="MEWS", value=f"{mews:.0f}")
            elif mews >= 5:
                _add(items, row, rule="mews_high", category="high", title="MEWS ≥ 5",
                     detail=f"MEWS {mews:.0f} — escalate frequency of observations.", metric="MEWS", value=f"{mews:.0f}")
            elif mews >= 4:
                _add(items, row, rule="mews_moderate", category="moderate", title="MEWS 4",
                     detail="MEWS 4 — increase monitoring frequency.", metric="MEWS", value="4")
        if str(avpu).upper() in ("P", "U"):
            _add(items, row, rule="avpu_critical", category="critical", title="Reduced consciousness",
                 detail=f"AVPU level '{avpu}' — unresponsive to voice.", metric="AVPU", value=str(avpu))
        if pd.notna(lactate):
            if lactate >= 4.0:
                _add(items, row, rule="lactate_high", category="high", title="Elevated lactate",
                     detail=f"Lactate {lactate:.1f} mmol/L at or above 4.0 mmol/L.", metric="Lactate", value=f"{lactate:.1f} mmol/L")
            elif lactate >= 2.0:
                _add(items, row, rule="lactate_moderate", category="moderate", title="Raised lactate",
                     detail=f"Lactate {lactate:.1f} mmol/L above 2.0 mmol/L.", metric="Lactate", value=f"{lactate:.1f} mmol/L")
        if pd.notna(creat) and creat >= 2.0:
            _add(items, row, rule="creatinine_moderate", category="moderate", title="Renal impairment",
                 detail=f"Creatinine {creat:.2f} mg/dL above 2.0 mg/dL.", metric="Creatinine", value=f"{creat:.2f} mg/dL")

        if pd.notna(risk):
            if risk >= thresholds["critical"]:
                _add(items, row, rule="model_critical", category="critical", title="Model: critical deterioration risk",
                     detail=f"Predicted 24h deterioration risk {risk:.0%} (critical band ≥ {thresholds['critical']:.0%}).",
                     metric="Model risk", value=f"{risk:.0%}")
            elif risk >= thresholds["high"]:
                _add(items, row, rule="model_high", category="high", title="Model: high deterioration risk",
                     detail=f"Predicted 24h deterioration risk {risk:.0%} (high band ≥ {thresholds['high']:.0%}).",
                     metric="Model risk", value=f"{risk:.0%}")
        if pd.notna(completeness) and completeness < 0.7:
            _add(items, row, rule="data_gap", category="info", title="Documentation gap",
                 detail=f"Only {completeness:.0%} of expected vital-sign fields recorded in the last 8 hours.",
                 metric="Completeness", value=f"{completeness:.0%}")

    if not items:
        return pd.DataFrame(
            columns=["id", "patient_id", "bed", "rule", "category", "title", "detail", "metric", "value", "ts", "risk_prob", "severity_rank"]
        )
    frame = pd.DataFrame(items)
    frame["severity_rank"] = frame["category"].map(SEVERITY_RANK).astype(int)
    frame = frame.sort_values(["severity_rank", "ts"], ascending=[True, False]).reset_index(drop=True)
    return frame


# ---------------------------------------------------------------------------
# acknowledgement / resolution state
# ---------------------------------------------------------------------------
def load_state() -> dict:
    if ALERT_STATE_FILE.exists():
        try:
            return json.loads(ALERT_STATE_FILE.read_text())
        except json.JSONDecodeError:
            return {}
    return {}


def save_state(state: dict) -> None:
    ALERT_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    ALERT_STATE_FILE.write_text(json.dumps(state, indent=2, default=str))


def reset_state() -> None:
    """Clear all acknowledgement and resolution records for this workspace."""
    if ALERT_STATE_FILE.exists():
        ALERT_STATE_FILE.unlink()


def set_acknowledged(alert_id: str, value: bool = True) -> None:
    state = load_state()
    entry = state.setdefault(alert_id, {})
    entry["acknowledged_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds") if value else None
    if not value:
        entry.pop("acknowledged_at", None)
    save_state(state)


def set_resolved(alert_id: str, value: bool = True) -> None:
    state = load_state()
    entry = state.setdefault(alert_id, {})
    entry["resolved_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds") if value else None
    if not value:
        entry.pop("resolved_at", None)
    save_state(state)


def apply_state(alerts: pd.DataFrame) -> pd.DataFrame:
    state = load_state()
    frame = alerts.copy()
    frame["acknowledged_at"] = frame["id"].map(lambda i: (state.get(i) or {}).get("acknowledged_at"))
    frame["resolved_at"] = frame["id"].map(lambda i: (state.get(i) or {}).get("resolved_at"))
    frame["acknowledged"] = frame["acknowledged_at"].notna()
    frame["resolved"] = frame["resolved_at"].notna()
    frame["status"] = np.select(
        [frame["resolved"], frame["acknowledged"]],
        ["Resolved", "Acknowledged"],
        default="Open",
    )
    return frame
