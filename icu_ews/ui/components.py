"""Reusable HTML/Streamlit UI components for the ICU-EWS design system."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

from ..config import ALERT_META, PALETTE, RISK_META
from ..icons import donut_svg, icon, sparkline

# ---------------------------------------------------------------------------
# formatting
# ---------------------------------------------------------------------------
def fmt_num(value, digits: int = 1, suffix: str = "", dash: str = "—") -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)) or pd.isna(value):
        return dash
    if isinstance(value, (int, np.integer)):
        return f"{int(value)}{suffix}"
    return f"{value:.{digits}f}{suffix}"


def fmt_pct(value, digits: int = 0, dash: str = "—") -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)) or pd.isna(value):
        return dash
    return f"{value * 100:.{digits}f}%"


def fmt_ts(ts, with_date: bool = True, with_year: bool = False) -> str:
    if ts is None or pd.isna(ts):
        return "—"
    ts = pd.Timestamp(ts)
    if with_date:
        fmt = "%d %b · %H:%M" if not with_year else "%d %b %Y · %H:%M"
    else:
        fmt = "%H:%M"
    return ts.strftime(fmt)


def relative_time(ts, reference) -> str:
    if ts is None or pd.isna(ts) or reference is None or pd.isna(reference):
        return "—"
    delta = pd.Timestamp(reference) - pd.Timestamp(ts)
    seconds = abs(delta.total_seconds())
    suffix = "ago" if delta.total_seconds() >= 0 else "ahead"
    if seconds < 90:
        return "just now" if suffix == "ago" else "in a moment"
    if seconds < 3600:
        return f"{int(seconds // 60)}m {suffix}"
    if seconds < 86400 * 2:
        hours = seconds / 3600
        return f"{hours:.0f}h {suffix}" if hours >= 10 else f"{hours:.1f}h {suffix}"
    return f"{int(seconds // 86400)}d {suffix}"


def time_tag(ts, reference) -> str:
    return f'<span title="{fmt_ts(ts, with_year=True)}">{relative_time(ts, reference)}</span>'


# ---------------------------------------------------------------------------
# primitives
# ---------------------------------------------------------------------------
def md(html: str) -> None:
    st.markdown(html, unsafe_allow_html=True)


def risk_badge(band: str, compact: bool = False) -> str:
    meta = RISK_META.get(band, RISK_META["low"])
    return (
        f'<span class="badge" style="color:{meta["color"]};background:{meta["soft"]};'
        f'border-color:{meta["border"]}"><span class="dot"></span>{meta["label"].upper()}</span>'
    )


def alert_badge(category: str) -> str:
    meta = ALERT_META.get(category, ALERT_META["info"])
    return (
        f'<span class="badge" style="color:{meta["color"]};background:{meta["soft"]};'
        f'border-color:{meta["color"]}55"><span class="dot"></span>{meta["label"].upper()}</span>'
    )


def chip(text: str, tone: str = "muted", dot: bool = False) -> str:
    dot_html = '<span class="dot"></span>' if dot else ""
    return f'<span class="chip chip-{tone}">{dot_html}{text}</span>'


def icon_html(name: str, size: int = 16, color: str = "currentColor") -> str:
    return icon(name, size=size, color=color)


def section_header(title: str, subtitle: str | None = None, icon_name: str | None = None) -> None:
    ico = f'<span class="ico">{icon(icon_name, 17)}</span>' if icon_name else ""
    sub = f'<div class="section-sub">{subtitle}</div>' if subtitle else ""
    md(f'<div class="section-head"><div><div class="section-title">{ico}{title}</div>{sub}</div></div>')


def page_header(title: str, subtitle: str, chips: str = "", right: str = "") -> None:
    md(
        f'<div class="page-head"><div><div class="page-title">{title}</div>'
        f'<div class="page-sub">{subtitle}</div><div class="page-meta">{chips}</div></div>'
        f'<div style="display:flex;gap:.5rem;align-items:center;flex-wrap:wrap">{right}</div></div>'
    )


def demo_banner(text: str) -> None:
    md(
        f'<div class="demo-banner">{icon("info", 15, "#fde68a")}'
        f'<div><b>DEMONSTRATION DATA.</b> {text}</div></div>'
    )


def empty_state(title: str, body: str, icon_name: str = "search") -> None:
    md(
        f'<div class="empty"><div class="empty-ico">{icon(icon_name, 26)}</div>'
        f'<div class="empty-title">{title}</div><div class="empty-body">{body}</div></div>'
    )


def stat_row(label: str, value: str, icon_name: str | None = None, color: str | None = None) -> str:
    ico = icon(icon_name, 13, color or PALETTE["text_dim"]) if icon_name else ""
    style = f' style="color:{color}"' if color else ""
    return (
        f'<div class="stat-row"><span class="stat-k">{ico}{label}</span>'
        f'<span class="stat-v"{style}>{value}</span></div>'
    )


def legend_html(items: list[tuple[str, str]], values: dict[str, str] | None = None) -> str:
    parts = []
    for label, color in items:
        value = f' · <b style="color:#dbe7f8">{values.get(label, "")}</b>' if values else ""
        parts.append(
            f'<span class="legend-item"><span class="legend-swatch" style="background:{color}"></span>'
            f"{label}{value}</span>"
        )
    return f'<div class="legend">{"".join(parts)}</div>'


def delta_html(delta: float | None, mode: str = "pct", label: str = "", higher_is_worse: bool = False) -> str:
    if delta is None or (isinstance(delta, float) and np.isnan(delta)):
        return f'<span class="kpi-delta" style="color:{PALETTE["text_dim"]}">{icon("minus", 12)} no prior period</span>'
    positive = delta >= 0
    good = (not positive) if higher_is_worse else positive
    color = PALETTE["low"] if good else PALETTE["critical"]
    arrow = "trending-up" if positive else "trending-down"
    if mode == "pp":
        text = f"{delta:+.1f} pp"
    else:
        text = f"{delta:+.0f}%"
    lbl = f'<span style="color:{PALETTE["text_dim"]};font-weight:500"> {label}</span>' if label else ""
    return f'<span class="kpi-delta" style="color:{color}">{icon(arrow, 13, color)}{text}{lbl}</span>'


def completion_bar(pct: float, width: int = 74) -> str:
    pct = max(0.0, min(1.0, float(pct)))
    color = PALETTE["low"] if pct >= 0.85 else (PALETTE["moderate"] if pct >= 0.7 else PALETTE["critical"])
    return (
        f'<span class="progress" style="width:{width}px">'
        f'<span style="width:{pct*100:.0f}%;background:linear-gradient(90deg,{color},{color}aa)"></span></span>'
    )


def avatar_initials(patient_id: str) -> str:
    digits = "".join(ch for ch in patient_id if ch.isdigit())[-3:]
    return digits or "EW"


def clinical_flag(metric: str, value: float) -> str:
    """CSS modifier for a measured value ('' | 'warn' | 'abn')."""
    if value is None or pd.isna(value):
        return ""
    if metric == "heart_rate":
        return "abn" if (value >= 140 or value < 40) else ("warn" if (value > 120 or value < 50) else "")
    if metric == "respiratory_rate":
        return "abn" if (value >= 32 or value < 9) else ("warn" if (value > 24 or value < 11) else "")
    if metric == "spo2":
        return "abn" if value < 88 else ("warn" if value < 92 else "")
    if metric == "systolic_bp":
        return "abn" if value < 85 else ("warn" if value < 100 else "")
    if metric == "temperature":
        return "abn" if (value >= 39.5 or value < 35.0) else ("warn" if (value >= 38.5 or value < 35.8) else "")
    if metric == "mews":
        return "abn" if value >= 7 else ("warn" if value >= 5 else "")
    return ""


# ---------------------------------------------------------------------------
# KPI cards
# ---------------------------------------------------------------------------
def kpi_card_html(card: dict) -> str:
    color = card["color"]
    glow = card["glow"]
    delta = card.get("delta_pct")
    mode = card.get("delta_mode", "pct")
    higher_is_worse = card["key"] in {"critical", "high", "deterioration"}
    spark = sparkline(card.get("series", []), color, width=118, height=38)
    dir_icon = "trending-up" if (delta or 0) >= 0 else "trending-down"
    if card["key"] == "deterioration" and delta is not None:
        dir_icon = "trending-up" if delta >= 0 else "trending-down"
    unit = card.get("unit", "")
    return f"""
<div class="kpi" style="--kpi-color:{color};--kpi-glow:{glow};--kpi-soft:{glow};
     --kpi-border:{color}55">
  <div class="kpi-top">
    <div class="kpi-label">{card['label']}</div>
    <div class="kpi-icon">{icon(card['icon'], 17, color)}</div>
  </div>
  <div class="kpi-value-row">
    <div class="kpi-value">{card['display']}<small>{unit}</small></div>
  </div>
  <div class="kpi-bottom">
    <div>
      {delta_html(delta, mode, card.get('delta_label', ''), higher_is_worse)}
      <div class="kpi-foot">{card.get('foot','')}</div>
    </div>
    <div class="kpi-spark">{spark}</div>
  </div>
</div>
"""


def kpi_card(card: dict) -> bool:
    with st.container(key=f"kpi_{card['key']}"):
        md(kpi_card_html(card))
        return st.button("Open", key=f"kpi_open_{card['key']}")


# ---------------------------------------------------------------------------
# patient card
# ---------------------------------------------------------------------------
_VITAL_LAYOUT = [
    ("heart_rate", "HR", "heart", "bpm"),
    ("respiratory_rate", "RR", "wind", "/min"),
    ("spo2", "SpO₂", "droplet", "%"),
    ("systolic_bp", "BP", "activity", "mmHg"),
    ("temperature", "Temp", "thermometer", "°C"),
    ("mews", "MEWS", "gauge", "pts"),
]


def patient_card_html(row: pd.Series, band: str, reference_ts, brier: float | None = None) -> str:
    meta = RISK_META[band]
    hr = row.get("heart_rate", np.nan)
    rr = row.get("respiratory_rate", np.nan)
    spo2 = row.get("spo2", np.nan)
    sbp = row.get("systolic_bp", np.nan)
    dbp = row.get("diastolic_bp", np.nan)
    temp = row.get("temperature", np.nan)
    mews = row.get("mews", np.nan)

    vitals = []
    for key, label, icon_name, unit in _VITAL_LAYOUT:
        if key == "systolic_bp":
            value = f"{fmt_num(sbp,0)}/{fmt_num(dbp,0)}" if pd.notna(sbp) else "—"
            flag = clinical_flag("systolic_bp", sbp)
            unit_html = ""
        elif key == "mews":
            value = fmt_num(mews, 0)
            flag = clinical_flag("mews", mews)
            unit_html = '<small>pts</small>'
        else:
            raw = {"heart_rate": hr, "respiratory_rate": rr, "spo2": spo2, "temperature": temp}.get(key)
            digits = 1 if key == "temperature" else 0
            value = fmt_num(raw, digits)
            flag = clinical_flag(key, raw)
            unit_html = f"<small>{unit}</small>"
        vitals.append(
            f'<div class="vital {flag}"><div class="vital-top">{icon(icon_name, 11)}'
            f'<span class="vital-lbl">{label}</span></div>'
            f'<div class="vital-val">{value}{unit_html}</div></div>'
        )

    labs = []
    lab_specs = [
        ("lactate", "Lactate", 2.0, 4.0, 1),
        ("wbc", "WBC", 12.0, 18.0, 1),
        ("creatinine", "Creat", 1.3, 2.0, 2),
        ("crp", "CRP", 50, 100, 0),
    ]
    for key, label, warn, high, digits in lab_specs:
        value = row.get(key, np.nan)
        cls = ""
        if pd.notna(value):
            cls = "hi" if value >= high else ("mid" if value >= warn else "")
        labs.append(
            f'<span class="lab-pill {cls}" title="Latest laboratory value">{label} '
            f"<b>{fmt_num(value, digits)}</b></span>"
        )

    p = row.get("risk_prob", np.nan)
    delta = row.get("risk_delta", np.nan)
    delta_html_txt = ""
    if pd.notna(delta) and abs(delta) >= 0.005:
        arrow = "up" if delta > 0 else "down"
        tone = PALETTE["critical"] if delta > 0 else PALETTE["low"]
        delta_html_txt = (
            f'<span style="color:{tone};font-size:.6rem;display:inline-flex;align-items:center;gap:2px">'
            f'{icon("arrow-up-right" if arrow=="up" else "arrow-down-right", 10, tone)}{abs(delta)*100:.0f}pp</span>'
        )

    completeness = row.get("completeness", 0.0)
    comp_color = PALETTE["low"] if completeness >= 0.85 else (
        PALETTE["moderate"] if completeness >= 0.7 else PALETTE["critical"]
    )
    calibration = (
        f'<span class="chip" style="font-size:.56rem;padding:1px 6px" '
        f'title="Brier score on the held-out test set — lower is better. Not a per-patient confidence interval.">'
        f"Brier {brier:.3f}</span>"
        if brier is not None
        else ""
    )
    predicted = row.get("predicted_ts", row.get("snapshot_ts"))
    return f"""
<div class="pcard" style="--risk-color:{meta['color']};--risk-glow:{meta['color']}22;--risk-border:{meta['border']}">
  <div class="pcard-head">
    <div style="min-width:0">
      <div class="pcard-id"><span style="width:26px;height:26px;border-radius:7px;display:inline-flex;align-items:center;
        justify-content:center;background:{meta['soft']};border:1px solid {meta['border']};color:{meta['color']};font-size:.6rem;font-weight:700">
        {avatar_initials(row['patient_id'])}</span>{row['patient_id']}</div>
      <div class="pcard-sub" style="margin-top:5px">{int(row['age'])}y {row['sex']} <span class="sep"></span>
        {row['admission_type']} <span class="sep"></span> {row.get('bed','')}</div>
    </div>
    <div class="pcard-prob">
      {risk_badge(band)}
      <div class="pcard-prob-val">{fmt_pct(p, 0)} {delta_html_txt}</div>
      <div class="pcard-prob-lbl">24h risk</div>
    </div>
  </div>
  <div class="vitals-grid">{''.join(vitals)}</div>
  <div class="pcard-labs">{''.join(labs)}</div>
  <div class="pcard-foot">
    <div class="completeness">{completion_bar(completeness)}
      <span style="color:{comp_color}">{fmt_pct(completeness,0)} complete</span>{calibration}</div>
    <div class="open-hint">{icon('clock', 11)} {time_tag(predicted, reference_ts)}</div>
  </div>
</div>
"""


def patient_card(row: pd.Series, band: str, reference_ts, brier: float | None = None) -> bool:
    pid = row["patient_id"]
    with st.container(key=f"pcard_{pid}"):
        md(patient_card_html(row, band, reference_ts, brier))
        return st.button("Open patient", key=f"pc_open_{pid}")


# ---------------------------------------------------------------------------
# risk intelligence pieces
# ---------------------------------------------------------------------------
def feed_item(alert: pd.Series, reference_ts) -> str:
    meta = ALERT_META.get(alert["category"], ALERT_META["info"])
    return f"""
<div class="feed-item">
  <div class="feed-ico" style="background:{meta['soft']};color:{meta['color']};border:1px solid {meta['color']}44">
    {icon('alert-octagon' if alert['category']=='critical' else 'alert-triangle', 13)}
  </div>
  <div style="min-width:0">
    <div class="feed-title">{alert['title']}</div>
    <div class="feed-meta"><span class="pid">{alert['patient_id']}</span>
      <span>·</span><span>{alert.get('value','')}</span>
      <span class="feed-time">{relative_time(alert['ts'], reference_ts)}</span>
    </div>
  </div>
</div>
"""


def queue_row(rank: int, row: pd.Series, band: str, reference_ts) -> str:
    meta = RISK_META[band]
    p = float(row.get("risk_prob", 0))
    return f"""
<div class="queue-row">
  <span class="rank">{rank:02d}</span>
  <div style="min-width:64px">
    <div class="queue-id">{row['patient_id']}</div>
    <div class="queue-sub">{row.get('bed','')} · {int(row['age'])}y</div>
  </div>
  <div class="queue-bar"><span style="width:{p*100:.0f}%;background:linear-gradient(90deg,{meta['color']}88,{meta['color']})"></span></div>
  <span class="queue-prob" style="color:{meta['color']}">{p*100:.0f}%</span>
</div>
"""


def activity_row(patient_id: str, ts, reference_ts, label: str) -> str:
    return (
        f'<div class="queue-row"><span class="rank" style="width:6px;height:6px;border-radius:50%;'
        f'background:{PALETTE["cyan"]};box-shadow:0 0 8px {PALETTE["cyan"]}"></span>'
        f'<div style="flex:1"><div class="queue-id" style="font-size:.72rem">{patient_id} '
        f'<span class="muted" style="font-family:Inter,sans-serif;font-weight:500">· {label}</span></div>'
        f'<div class="queue-sub">{fmt_ts(ts)}</div></div>'
        f'<span class="dim text-xs">{relative_time(ts, reference_ts)}</span></div>'
    )


def measurement_row(item: pd.Series) -> str:
    direction = "below" if item["direction"] == "below" else "above"
    unit = item["unit"]
    sep = "" if unit in {"%", "°C", "/min"} else " "
    threshold = fmt_num(item["threshold"], 1 if float(item["threshold"]) % 1 else 0)
    return f"""
<div class="queue-row">
  <div style="min-width:0;flex:1">
    <div class="queue-id" style="font-size:.72rem">{item['label']} <span class="muted"
      style="font-family:Inter,sans-serif;font-weight:500">{direction} {threshold}{sep}{unit}</span></div>
    <div class="queue-sub">{item['patient_id']}</div>
  </div>
  <span class="mono" style="color:{item['color']};font-size:.78rem;font-weight:600">
    {fmt_num(item['value'], 1 if float(item['value']) % 1 else 0)}{sep}{unit}</span>
</div>
"""


# ---------------------------------------------------------------------------
# panels
# ---------------------------------------------------------------------------
@contextmanager
def panel(key: str, title: str | None = None, subtitle: str | None = None, icon_name: str | None = None, tight: bool = False):
    container = st.container(key=f"panel_{key}")
    with container:
        if title:
            ico = f'<span class="ico">{icon(icon_name, 16)}</span>' if icon_name else ""
            sub = f'<div class="section-sub">{subtitle}</div>' if subtitle else ""
            md(f'<div class="section-head" style="margin-top:0"><div>'
               f'<div class="section-title" style="font-size:.9rem">{ico}{title}</div>{sub}</div></div>')
        yield container


def panel_toolbar(key: str) -> None:
    st.markdown(f'<div class="panel-toolbar-{key}"></div>', unsafe_allow_html=True)


def clickable(html: str, key: str, label: str = "Open") -> bool:
    """Render HTML with a transparent full-area button overlay (keyed row link)."""
    with st.container(key=f"click_{key}"):
        md(html)
        return st.button(label, key=f"cbtn_{key}")


def pager(namespace: str, total_pages: int, page: int, shown: int, total: int) -> None:
    """Prev / next controls bound to ``{namespace}_page`` in session state."""
    state_key = f"{namespace}_page"
    with st.container(key=f"{namespace}_pager"):
        left, mid, right = st.columns([1, 2.6, 1])
        if left.button("Previous", key=f"{namespace}_prev", disabled=page <= 1, use_container_width=True):
            st.session_state[state_key] = page - 1
            st.rerun()
        mid.markdown(
            f'<div class="pager-info"><b>Page {page}</b> of {total_pages}'
            f'<span> · showing {shown} of {total}</span></div>',
            unsafe_allow_html=True,
        )
        if right.button("Next", key=f"{namespace}_next", disabled=page >= total_pages, use_container_width=True):
            st.session_state[state_key] = page + 1
            st.rerun()
