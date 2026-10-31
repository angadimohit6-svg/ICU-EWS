"""Synthetic ICU cohort generator used for the demonstration dataset.

The cohort is fully synthetic. It is generated once at bootstrap, clearly
labelled as demonstration data throughout the application, and never presented
as hospital data. Vitals, laboratory values, the MEWS score and the
deterioration outcome are derived from a latent severity process so that a
model trained on the cohort has a real (but imperfect) signal to learn.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

GENERATOR_VERSION = "demo-cohort/1.2"

ADMISSION_TYPES = [
    ("Medical", 0.34, 0.12),
    ("Surgical", 0.24, -0.10),
    ("Neurological", 0.14, 0.28),
    ("Cardiac", 0.14, 0.34),
    ("Respiratory", 0.14, 0.48),
]

DIAGNOSIS_BY_TYPE = {
    "Medical": [("Sepsis / infection", 0.46), ("Renal / metabolic", 0.28), ("Other medical", 0.26)],
    "Surgical": [("Post-operative care", 0.70), ("Trauma", 0.30)],
    "Neurological": [("Neurological event", 0.75), ("Trauma", 0.25)],
    "Cardiac": [("Cardiovascular event", 0.82), ("Post-operative care", 0.18)],
    "Respiratory": [("Respiratory failure", 0.78), ("Sepsis / infection", 0.22)],
}

UNITS = {
    "Medical": "MICU",
    "Surgical": "SICU",
    "Neurological": "NICU",
    "Cardiac": "CICU",
    "Respiratory": "RICU",
}


@dataclass(frozen=True)
class CohortSpec:
    n_patients: int = 420
    seed: int = 7
    as_of: datetime | None = None


def _sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-x))


def _ar1(rng: np.random.Generator, n: int, sd: float, phi: float = 0.8) -> np.ndarray:
    out = np.zeros(n)
    innovation = rng.normal(0, sd, n)
    for i in range(1, n):
        out[i] = phi * out[i - 1] + innovation[i]
    return out


def _pick(rng: np.random.Generator, options: list[tuple[str, float]]) -> str:
    labels = [o[0] for o in options]
    weights = np.array([o[1] for o in options], dtype=float)
    return str(rng.choice(labels, p=weights / weights.sum()))


def generate_cohort(spec: CohortSpec | None = None) -> dict[str, pd.DataFrame]:
    spec = spec or CohortSpec()
    rng = np.random.default_rng(spec.seed)
    as_of = (spec.as_of or datetime.now()).replace(minute=0, second=0, microsecond=0)

    type_labels = [t[0] for t in ADMISSION_TYPES]
    type_probs = np.array([t[1] for t in ADMISSION_TYPES], dtype=float)
    type_probs = type_probs / type_probs.sum()
    type_coef = {t[0]: t[2] for t in ADMISSION_TYPES}

    ids = rng.choice(np.arange(1000, 9999), size=spec.n_patients, replace=False)

    patients: list[dict] = []
    obs_rows: list[dict] = []
    lab_rows: list[dict] = []

    for i in range(spec.n_patients):
        pid = f"ICU-{int(ids[i])}"
        age = int(np.clip(rng.normal(66, 15), 18, 96))
        sex = "M" if rng.random() < 0.54 else "F"
        adm_type = str(rng.choice(type_labels, p=type_probs))
        diagnosis = _pick(rng, DIAGNOSIS_BY_TYPE[adm_type])
        comorbidities = int(np.clip(rng.poisson(1.9), 0, 7))
        static_risk = (
            0.045 * (age - 65) + 0.22 * comorbidities + type_coef[adm_type] + rng.normal(0, 0.25)
        )

        los_hours = int(np.clip(round(rng.gamma(3.0, 30.0)), 22, 260))
        assessment_hour = int(min(36, los_hours - 4))
        assessment_hour = max(assessment_hour, 12)
        n_obs = assessment_hour + 1

        z = float(rng.normal(0, 1))
        severity = float(_sigmoid(0.95 * z + 0.30 * static_risk - 0.15))
        toward_end = np.arange(n_obs) / max(n_obs - 1, 1)
        trend = toward_end * (0.25 + 1.05 * severity)

        hr_base = 74 + rng.normal(0, 7) + 5 * max(static_risk, 0) + 6 * (severity - 0.5)
        rr_base = 15.2 + rng.normal(0, 1.6) + 1.2 * max(static_risk, 0)
        spo2_base = 97.9 + rng.normal(0, 1.0) - 1.6 * max(static_risk, 0) - 2.0 * (severity - 0.5)
        sbp_base = 126 + rng.normal(0, 10) - 3 * max(static_risk, 0) - 4 * (severity - 0.5)
        temp_base = 36.8 + rng.normal(0, 0.22) + 0.15 * max(static_risk, 0)

        hr_spike = np.zeros(n_obs)
        spo2_spike = np.zeros(n_obs)
        rr_spike = np.zeros(n_obs)
        sbp_spike = np.zeros(n_obs)
        temp_spike = np.zeros(n_obs)

        n_events = 1 if rng.random() < (0.55 if severity > 0.6 else 0.12) else 0
        for _ in range(n_events):
            start = int(rng.integers(max(2, n_obs // 3), max(3, n_obs - 2)))
            length = int(rng.integers(2, 6))
            slice_ = slice(start, min(start + length, n_obs))
            hr_spike[slice_] += rng.uniform(14, 32)
            spo2_spike[slice_] -= rng.uniform(2.5, 6.0)
            rr_spike[slice_] += rng.uniform(2.5, 5.5)
            sbp_spike[slice_] -= rng.uniform(6, 16)
            temp_spike[slice_] += rng.uniform(0.2, 0.5)

        hr = hr_base + 30 * trend + _ar1(rng, n_obs, 2.2) + hr_spike
        rr = rr_base + 8 * trend + _ar1(rng, n_obs, 0.8) + rr_spike
        spo2 = spo2_base - 6.5 * trend + _ar1(rng, n_obs, 0.85) + spo2_spike
        sbp = sbp_base - 16 * trend + _ar1(rng, n_obs, 3.4) + sbp_spike
        temp = temp_base + 0.8 * trend + _ar1(rng, n_obs, 0.07) + temp_spike

        hr = np.clip(hr, 34, 205)
        rr = np.clip(rr, 5, 58)
        spo2 = np.clip(spo2, 64, 100)
        sbp = np.clip(sbp, 54, 226)
        dbp = np.clip(sbp * 0.64 + rng.normal(0, 4, n_obs), 24, 138)
        map_ = np.clip((sbp + 2 * dbp) / 3, 36, 165)
        temp = np.clip(temp, 33.0, 42.2)

        # AVPU consciousness level
        avpu: list[str] = []
        for h in range(n_obs):
            p_abn = float(_sigmoid(-5.4 + 9.5 * trend[h] * severity))
            u = rng.random()
            if u < p_abn * 0.10:
                avpu.append("U")
            elif u < p_abn * 0.32:
                avpu.append("P")
            elif u < p_abn:
                avpu.append("V")
            else:
                avpu.append("A")

        # observation gaps (monitor downtime) and per-field missingness
        gap_start = -1
        if rng.random() < 0.10:
            gap_start = int(rng.integers(2, max(3, n_obs - 3)))
            gap_len = int(rng.integers(2, 6))

        snapshot_ts = as_of - timedelta(hours=int(rng.integers(1, 24 * 12)))
        admit_ts = snapshot_ts - timedelta(hours=assessment_hour)

        hr = hr.round(0)
        rr = rr.round(0)
        spo2 = spo2.round(0)
        sbp = sbp.round(0)
        dbp = dbp.round(0)
        map_ = map_.round(0)
        temp = temp.round(1)

        for h in range(n_obs):
            is_gap = gap_start >= 0 and gap_start <= h < gap_start + gap_len
            row = {
                "patient_id": pid,
                "hour": h,
                "ts": admit_ts + timedelta(hours=h),
                "heart_rate": np.nan if is_gap or rng.random() < 0.02 else float(hr[h]),
                "respiratory_rate": np.nan if is_gap or rng.random() < 0.03 else float(rr[h]),
                "spo2": np.nan if is_gap or rng.random() < 0.03 else float(spo2[h]),
                "systolic_bp": np.nan if is_gap or rng.random() < 0.03 else float(sbp[h]),
                "diastolic_bp": np.nan if is_gap or rng.random() < 0.04 else float(dbp[h]),
                "map": np.nan if is_gap or rng.random() < 0.04 else float(map_[h]),
                "temperature": np.nan if is_gap or rng.random() < 0.09 else float(temp[h]),
                "avpu": "A" if is_gap or rng.random() < 0.05 else avpu[h],
            }
            obs_rows.append(row)

        # laboratory panel every ~12 hours
        lab_hours = sorted(
            {
                int(np.clip(round(h), 0, assessment_hour))
                for h in np.arange(0, assessment_hour + 1, 12) + rng.normal(0, 1.1, len(np.arange(0, assessment_hour + 1, 12)))
            }
        )
        for lh in lab_hours:
            sev_now = float(severity) * (0.55 + 0.45 * (lh / max(assessment_hour, 1)))
            if rng.random() < 0.04:
                continue
            lab_rows.append(
                {
                    "patient_id": pid,
                    "hour": lh,
                    "ts": admit_ts + timedelta(hours=lh),
                    "lactate": round(float(np.clip(0.8 + 3.0 * sev_now + rng.normal(0, 0.32), 0.4, 18)), 1),
                    "wbc": round(float(np.clip(6.2 + 7.0 * sev_now + rng.normal(0, 1.2), 0.9, 42)), 1),
                    "creatinine": round(float(np.clip(0.82 + 1.25 * sev_now + rng.normal(0, 0.22), 0.3, 6.5)), 2),
                    "platelets": int(np.clip(255 - 105 * sev_now + rng.normal(0, 28), 14, 520)),
                    "hemoglobin": round(float(np.clip(12.6 - 1.8 * sev_now + rng.normal(0, 0.8), 6.0, 18.0)), 1),
                    "crp": int(np.clip(6 + 95 * sev_now + rng.normal(0, 12), 0.5, 380)),
                    "bilirubin": round(float(np.clip(0.7 + 1.3 * sev_now + rng.normal(0, 0.25), 0.2, 12)), 1),
                    "sodium": int(np.clip(139 - 2 * sev_now + rng.normal(0, 3.1), 118, 158)),
                    "potassium": round(float(np.clip(4.0 + 0.4 * sev_now + rng.normal(0, 0.32), 2.4, 6.9)), 1),
                }
            )

        p_event = float(_sigmoid(1.5 * z + 0.70 * static_risk - 2.35))
        deteriorated = int(rng.random() < p_event)
        event_hour = (
            float(assessment_hour + int(rng.integers(2, 25))) if deteriorated else np.nan
        )

        patients.append(
            {
                "patient_id": pid,
                "age": age,
                "sex": sex,
                "admission_type": adm_type,
                "diagnosis_group": diagnosis,
                "comorbidity_count": comorbidities,
                "admit_ts": admit_ts,
                "los_hours": los_hours,
                "assessment_hour": assessment_hour,
                "snapshot_ts": snapshot_ts,
                "deterioration_24h": deteriorated,
                "event_hour": event_hour,
                "bed": f"{UNITS[adm_type]}-{int(rng.integers(1, 25)):02d}",
            }
        )

    patients_df = pd.DataFrame(patients)
    obs_df = pd.DataFrame(obs_rows)
    labs_df = pd.DataFrame(lab_rows)
    patients_df = patients_df.sort_values("snapshot_ts", ascending=False).reset_index(drop=True)

    return {"patients": patients_df, "observations": obs_df, "labs": labs_df}
