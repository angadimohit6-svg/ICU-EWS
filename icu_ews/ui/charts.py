"""Plotly chart theme and shared chart builders for ICU-EWS."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from ..config import PALETTE, RISK_META, RISK_ORDER

GRID = "rgba(31, 48, 73, 0.55)"
AXIS_LINE = "rgba(42, 61, 92, 0.85)"
FONT_STACK = "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif"
MONO_STACK = "'JetBrains Mono', ui-monospace, Menlo, monospace"

SERIES_COLORS = [
    PALETTE["cyan"],
    PALETTE["blue"],
    PALETTE["indigo"],
    PALETTE["violet"],
    PALETTE["teal"],
    PALETTE["pink"],
    PALETTE["high"],
    PALETTE["moderate"],
]

PLOT_CONFIG = {"displayModeBar": False, "scrollZoom": False}
PLOT_CONFIG_BAR = {
    "displayModeBar": True,
    "displaylogo": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d", "toggleSpikelines"],
}


def apply_theme(
    fig: go.Figure,
    height: int = 320,
    legend: bool = False,
    hovermode: str = "x unified",
    margins: tuple[int, int, int, int] = (10, 10, 16, 10),
    x_title: str | None = None,
    y_title: str | None = None,
    xgrid: bool = False,
    ygrid: bool = True,
    hover_label: bool = True,
) -> go.Figure:
    left, right, top, bottom = margins
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT_STACK, size=11, color=PALETTE["text_muted"]),
        margin=dict(l=left, r=right, t=top, b=bottom),
        showlegend=legend,
        hovermode=hovermode,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            x=0,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=10, color=PALETTE["text_muted"]),
        ),
        hoverlabel=dict(
            bgcolor="#0c1727",
            bordercolor="#2a3d5c",
            font=dict(family=FONT_STACK, size=11, color="#e8eefb"),
        )
        if hover_label
        else None,
        dragmode=False,
    )
    fig.update_xaxes(
        gridcolor=GRID,
        griddash="dot",
        showgrid=xgrid,
        zeroline=False,
        linecolor=AXIS_LINE,
        tickfont=dict(size=10, color=PALETTE["text_dim"]),
        title=dict(text=x_title, font=dict(size=10, color=PALETTE["text_muted"])) if x_title else None,
        showspikes=False,
        automargin=True,
    )
    fig.update_yaxes(
        gridcolor=GRID,
        griddash="dot",
        showgrid=ygrid,
        zeroline=False,
        linecolor=AXIS_LINE,
        tickfont=dict(size=10, color=PALETTE["text_dim"]),
        title=dict(text=y_title, font=dict(size=10, color=PALETTE["text_muted"])) if y_title else None,
        automargin=True,
    )
    return fig


def risk_color(level: str) -> str:
    return RISK_META[level]["color"]


def risk_donut(counts: dict[str, int], height: int = 210) -> go.Figure:
    labels = [RISK_META[b]["label"] for b in RISK_ORDER]
    values = [counts.get(b, 0) for b in RISK_ORDER]
    colors = [RISK_META[b]["color"] for b in RISK_ORDER]
    fig = go.Figure(
        go.Pie(
            labels=labels,
            values=values,
            hole=0.68,
            sort=False,
            direction="clockwise",
            marker=dict(colors=colors, line=dict(color="#0b1524", width=3)),
            textinfo="none",
            hovertemplate="<b>%{label}</b><br>%{value} patients · %{percent}<extra></extra>",
        )
    )
    total = int(sum(values))
    fig.add_annotation(
        text=f"<b>{total}</b><br><span style='font-size:10px;color:{PALETTE['text_dim']}'>PATIENTS</span>",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(family=MONO_STACK, size=20, color=PALETTE["text"]),
    )
    fig.update_layout(
        height=height,
        margin=dict(l=4, r=4, t=4, b=4),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        hoverlabel=dict(bgcolor="#0c1727", bordercolor="#2a3d5c", font=dict(family=FONT_STACK, size=11, color="#e8eefb")),
    )
    return fig


def risk_trajectory(risk: pd.DataFrame, thresholds: dict[str, float], height: int = 280) -> go.Figure:
    fig = go.Figure()
    bounds = [
        (0, thresholds["moderate"], RISK_META["low"]["color"], 0.05),
        (thresholds["moderate"], thresholds["high"], RISK_META["moderate"]["color"], 0.05),
        (thresholds["high"], thresholds["critical"], RISK_META["high"]["color"], 0.05),
        (thresholds["critical"], 1.0, RISK_META["critical"]["color"], 0.06),
    ]
    for y0, y1, color, opacity in bounds:
        fig.add_hrect(y0=y0, y1=y1, fillcolor=color, opacity=opacity, line_width=0, layer="below")

    for level, label in (("critical", "Critical"), ("high", "High"), ("moderate", "Moderate")):
        fig.add_hline(
            y=thresholds[level],
            line=dict(color=RISK_META[level]["color"], width=1, dash="dot"),
            opacity=0.55,
            annotation_text=label,
            annotation_position="right",
            annotation_font=dict(size=9, color=RISK_META[level]["color"]),
        )

    fig.add_trace(
        go.Scatter(
            x=risk["assessment_ts"],
            y=risk["risk_prob"],
            mode="lines+markers",
            line=dict(color=PALETTE["cyan"], width=2.4, shape="spline", smoothing=0.6),
            marker=dict(size=5, color=PALETTE["cyan"], line=dict(color="#071120", width=1.5)),
            fill="tozeroy",
            fillcolor="rgba(34, 211, 238, 0.08)",
            customdata=risk["anchor_hour"],
            hovertemplate="Hour %{customdata}h<br><b>%{y:.1%}</b> predicted risk<extra></extra>",
            name="Predicted risk",
        )
    )
    apply_theme(fig, height=height, hovermode="x unified", margins=(10, 58, 14, 10), ygrid=True)
    fig.update_yaxes(tickformat=".0%", range=[0, 1], title="24h deterioration risk")
    fig.update_xaxes(title=None)
    return fig


def calibration_plot(points: dict, height: int = 300, label: str = "Primary model") -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line=dict(color="rgba(142, 163, 192, 0.45)", width=1, dash="dash"),
            name="Perfect calibration",
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=points["mean_predicted"],
            y=points["fraction_positive"],
            mode="lines+markers",
            line=dict(color=PALETTE["cyan"], width=2.2, shape="spline"),
            marker=dict(size=7, color=PALETTE["cyan"], line=dict(color="#071120", width=2)),
            name=label,
            hovertemplate="Predicted %{x:.0%}<br>Observed %{y:.0%}<extra></extra>",
        )
    )
    apply_theme(fig, height=height, legend=True, hovermode="closest", margins=(6, 10, 30, 8))
    fig.update_xaxes(tickformat=".0%", range=[0, 1], title="Mean predicted probability")
    fig.update_yaxes(tickformat=".0%", range=[0, 1], title="Observed event rate")
    return fig
