"""Risk-band mapping and deterministic alert derivation."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from icu_ews.config import DEFAULT_THRESHOLDS, risk_from_probability
from icu_ews.data import alerts as alerts_mod


@pytest.mark.parametrize(
    "probability,expected",
    [
        (0.99, "critical"),
        (0.75, "critical"),
        (0.74, "high"),
        (0.55, "high"),
        (0.54, "moderate"),
        (0.35, "moderate"),
        (0.34, "low"),
        (0.0, "low"),
    ],
)
def test_risk_band_boundaries_are_inclusive(probability, expected):
    assert risk_from_probability(probability, DEFAULT_THRESHOLDS) == expected


@pytest.fixture()
def snapshot():
    ts = pd.Timestamp("2026-10-21 18:00:00")
    rows = [
        {
            "patient_id": "ICU-0001",
            "bed": "MICU-01",
            "snapshot_ts": ts,
            "heart_rate": 150.0,
            "respiratory_rate": 34.0,
            "spo2": 85.0,
            "systolic_bp": 90.0,
            "temperature": 39.8,
            "mews": 8,
            "avpu": "P",
            "lactate": 4.5,
            "creatinine": 2.4,
            "risk_prob": 0.80,
            "completeness": 0.50,
        },
        {
            "patient_id": "ICU-0002",
            "bed": "MICU-02",
            "snapshot_ts": ts,
            "heart_rate": 80.0,
            "respiratory_rate": 15.0,
            "spo2": 98.0,
            "systolic_bp": 120.0,
            "temperature": 36.8,
            "mews": 0,
            "avpu": "A",
            "lactate": 1.0,
            "creatinine": 0.9,
            "risk_prob": 0.10,
            "completeness": 0.95,
        },
        {
            "patient_id": "ICU-0003",
            "bed": "SICU-03",
            "snapshot_ts": ts,
            "heart_rate": 95.0,
            "respiratory_rate": 18.0,
            "spo2": 90.0,
            "systolic_bp": 110.0,
            "temperature": 37.4,
            "mews": 2,
            "avpu": "A",
            "lactate": 1.6,
            "creatinine": 1.0,
            "risk_prob": 0.60,
            "completeness": 0.90,
        },
    ]
    return pd.DataFrame(rows)


def test_alerts_are_derived_from_recorded_values(snapshot):
    frame = alerts_mod.build_alerts(snapshot, DEFAULT_THRESHOLDS)
    rules = set(frame["rule"])
    assert {
        "spo2_critical",
        "sbp_high",
        "hr_critical",
        "rr_critical",
        "temp_high",
        "mews_critical",
        "avpu_critical",
        "lactate_high",
        "creatinine_moderate",
        "model_critical",
        "data_gap",
        "spo2_high",
        "model_high",
    } <= rules
    assert len(frame) == 13
    assert frame["id"].is_unique
    assert set(frame["patient_id"]) <= set(snapshot["patient_id"])
    assert set(frame["category"]) <= {"critical", "high", "moderate", "info"}
    assert frame.iloc[0]["category"] == "critical"
    assert (frame["ts"] <= snapshot["snapshot_ts"].max()).all()


def test_alert_model_rules_follow_active_thresholds(snapshot):
    strict = {"critical": 0.95, "high": 0.90, "moderate": 0.80}
    frame = alerts_mod.build_alerts(snapshot, strict)
    assert "model_critical" not in set(frame["rule"])
    assert "model_high" not in set(frame["rule"])
    assert len(frame) == 11  # two model alerts removed versus the default thresholds


def test_alert_workflow_state_roundtrip(tmp_path, monkeypatch, snapshot):
    state_file = tmp_path / "alert_state.json"
    monkeypatch.setattr(alerts_mod, "ALERT_STATE_FILE", state_file)

    frame = alerts_mod.build_alerts(snapshot, DEFAULT_THRESHOLDS)
    alert_id = frame.iloc[0]["id"]

    alerts_mod.set_acknowledged(alert_id, True)
    assert json.loads(state_file.read_text())[alert_id]["acknowledged_at"]

    with_state = alerts_mod.apply_state(frame)
    row = with_state[with_state["id"] == alert_id].iloc[0]
    assert bool(row["acknowledged"]) is True
    assert row["status"] == "Acknowledged"

    alerts_mod.set_resolved(alert_id, True)
    with_state = alerts_mod.apply_state(frame)
    row = with_state[with_state["id"] == alert_id].iloc[0]
    assert bool(row["resolved"]) is True
    assert row["status"] == "Resolved"

    alerts_mod.reset_state()
    assert not state_file.exists()
    reset = alerts_mod.apply_state(frame)
    assert not reset["acknowledged"].any()
    assert not reset["resolved"].any()


def test_alert_timestamps_are_relative_to_the_snapshot(snapshot):
    frame = alerts_mod.build_alerts(snapshot, DEFAULT_THRESHOLDS)
    oldest = snapshot["snapshot_ts"].min() - pd.Timedelta(minutes=55)
    assert frame["ts"].min() >= oldest
    assert frame["ts"].max() <= snapshot["snapshot_ts"].max()
