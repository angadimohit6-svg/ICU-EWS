"""Inline SVG icon set (Lucide-style, stroke based) and small SVG primitives.

Streamlit strips <script> tags but keeps inline SVG, so icons render reliably
inside custom HTML components.
"""

from __future__ import annotations

import itertools
from html import escape

_SPARK_IDS = itertools.count(1)

_ICONS: dict[str, list[str]] = {
    "heart-pulse": [
        "M19 14c1.49-1.46 3-3.21 3-5.5A5.5 5.5 0 0 0 16.5 3c-1.76 0-3 .5-4.5 2-1.5-1.5-2.74-2-4.5-2A5.5 5.5 0 0 0 2 8.5c0 2.3 1.51 4.04 3 5.5l7 7Z",
        "M3.2 12h5.3l.5-1 2 4.5 2-7 1.5 3.5h5.3",
    ],
    "heart": [
        "M19 14c1.49-1.46 3-3.21 3-5.5A5.5 5.5 0 0 0 16.5 3c-1.76 0-3 .5-4.5 2-1.5-1.5-2.74-2-4.5-2A5.5 5.5 0 0 0 2 8.5c0 2.3 1.51 4.04 3 5.5l7 7Z",
    ],
    "activity": ["M22 12h-4l-3 9L9 3l-3 9H2"],
    "users": [
        "M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2",
        "M9 7a4 4 0 1 0 0 .01",
        "M22 21v-2a4 4 0 0 0-3-3.87",
        "M16 3.13a4 4 0 0 1 0 7.75",
    ],
    "user": ["M19 21v-2a4 4 0 0 0-4-4H9a4 4 0 0 0-4 4v2", "M12 7a4 4 0 1 0 0 .01"],
    "alert-triangle": [
        "m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z",
        "M12 9v4",
        "M12 17h.01",
    ],
    "alert-octagon": [
        "M7.86 2h8.28L22 7.86v8.28L16.14 22H7.86L2 16.14V7.86L7.86 2Z",
        "M12 8v4",
        "M12 16h.01",
    ],
    "shield-check": [
        "M20 13c0 5-3.5 7.5-7.66 8.95a1 1 0 0 1-.67-.01C7.5 20.5 4 18 4 13V6a1 1 0 0 1 1-1c2 0 4.5-1.2 6.24-2.72a1.17 1.17 0 0 1 1.52 0C14.51 3.81 17 5 19 5a1 1 0 0 1 1 1z",
        "m9 12 2 2 4-4",
    ],
    "shield": [
        "M20 13c0 5-3.5 7.5-7.66 8.95a1 1 0 0 1-.67-.01C7.5 20.5 4 18 4 13V6a1 1 0 0 1 1-1c2 0 4.5-1.2 6.24-2.72a1.17 1.17 0 0 1 1.52 0C14.51 3.81 17 5 19 5a1 1 0 0 1 1 1z",
    ],
    "thermometer": ["M14 4v10.54a4 4 0 1 1-4 0V4a2 2 0 0 1 4 0Z"],
    "wind": [
        "M17.7 7.7a2.5 2.5 0 1 1 1.8 4.3H2",
        "M9.6 4.6A2 2 0 1 1 11 8H2",
        "M12.6 19.4A2 2 0 1 0 14 16H2",
    ],
    "droplet": [
        "M12 22a7 7 0 0 0 7-7c0-2-1-3.9-3-5.5s-3.5-4-4-6.5c-.5 2.5-2 4.9-4 6.5C6 11.1 5 13 5 15a7 7 0 0 0 7 7z"
    ],
    "gauge": ["m12 14 4-4", "M3.34 19a10 10 0 1 1 17.32 0"],
    "stethoscope": [
        "M4.8 2.3A.3.3 0 1 0 5 2H4a2 2 0 0 0-2 2v5a6 6 0 0 0 6 6 6 6 0 0 0 6-6V4a2 2 0 0 0-2-2h-1a.2.2 0 1 0 .3.3",
        "M8 15v1a6 6 0 0 0 6 6 6 6 0 0 0 6-6v-4",
        "M20 10a2 2 0 1 0 0 .01",
    ],
    "flask": ["M4.5 3h15", "M6 3v16a2 2 0 0 0 2 2h8a2 2 0 0 0 2-2V3", "M6 14h12"],
    "cpu": [
        "M4 4h16v16H4z",
        "M9 9h6v6H9z",
        "M9 2v2",
        "M15 2v2",
        "M9 20v2",
        "M15 20v2",
        "M2 9h2",
        "M2 15h2",
        "M20 9h2",
        "M20 15h2",
    ],
    "brain": [
        "M12 5a3 3 0 1 0-5.997.125 4 4 0 0 0-2.526 5.77 4 4 0 0 0 .556 6.588A4 4 0 1 0 12 18Z",
        "M12 5a3 3 0 1 1 5.997.125 4 4 0 0 1 2.526 5.77 4 4 0 0 1-.556 6.588A4 4 0 1 1 12 18Z",
        "M15 13a4.5 4.5 0 0 1-3-4 4.5 4.5 0 0 1-3 4",
    ],
    "file-text": [
        "M15 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7Z",
        "M14 2v4a2 2 0 0 0 2 2h4",
        "M10 9H8",
        "M16 13H8",
        "M16 17H8",
    ],
    "clipboard-list": [
        "M8 2h8v4H8z",
        "M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2",
        "M12 11h4",
        "M12 16h4",
        "M8 11h.01",
        "M8 16h.01",
    ],
    "dashboard": [
        "M3 3h7v9H3z",
        "M14 3h7v5h-7z",
        "M14 12h7v9h-7z",
        "M3 16h7v5H3z",
    ],
    "sliders": [
        "M4 21v-7",
        "M4 10V3",
        "M12 21v-9",
        "M12 8V3",
        "M20 21v-5",
        "M20 12V3",
        "M2 14h4",
        "M10 8h4",
        "M18 16h4",
    ],
    "search": ["M11 3a8 8 0 1 0 0 .01", "m21 21-4.3-4.3"],
    "bell": [
        "M10.268 21a2 2 0 0 0 3.464 0",
        "M3.262 15.326A1 1 0 0 0 4 17h16a1 1 0 0 0 .74-1.673C19.41 13.956 18 12.499 18 8A6 6 0 0 0 6 8c0 4.499-1.411 5.956-2.738 7.326",
    ],
    "bell-ring": [
        "M10.268 21a2 2 0 0 0 3.464 0",
        "M3.262 15.326A1 1 0 0 0 4 17h16a1 1 0 0 0 .74-1.673C19.41 13.956 18 12.499 18 8A6 6 0 0 0 6 8c0 4.499-1.411 5.956-2.738 7.326",
        "M22 8c0-2.3-.8-4.3-2-6",
        "M4 2C2.8 3.7 2 5.7 2 8",
    ],
    "chevron-right": ["m9 18 6-6-6-6"],
    "chevron-left": ["m15 18-6-6 6-6"],
    "chevron-down": ["m6 9 6 6 6-6"],
    "arrow-right": ["M5 12h14", "m12 5 7 7-7 7"],
    "arrow-up-right": ["M7 7h10v10", "M7 17 17 7"],
    "arrow-down-right": ["M7 7l10 10", "M17 7v10H7"],
    "trending-up": ["M22 7 13.5 15.5 8.5 10.5 2 17", "M16 7h6v6"],
    "trending-down": ["M22 17 13.5 8.5 8.5 13.5 2 7", "M16 17h6v-6"],
    "download": ["M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4", "M7 10l5 5 5-5", "M12 15V3"],
    "check": ["M20 6 9 17l-5-5"],
    "check-circle": ["M12 2a10 10 0 1 0 0 .01", "m9 12 2 2 4-4"],
    "clock": ["M12 2a10 10 0 1 0 0 .01", "M12 6v6l4 2"],
    "database": ["M12 2C7 2 3 3.3 3 5s4 3 9 3 9-1.3 9-3-4-3-9-3z", "M3 5v14c0 1.7 4 3 9 3s9-1.3 9-3V5", "M3 12c0 1.7 4 3 9 3s9-1.3 9-3"],
    "target": ["M12 2a10 10 0 1 0 0 .01", "M12 6a6 6 0 1 0 0 .01", "M12 10a2 2 0 1 0 0 .01"],
    "layers": [
        "M12.83 2.18a2 2 0 0 0-1.66 0L2.6 6.08a1 1 0 0 0 0 1.83l8.58 3.91a2 2 0 0 0 1.66 0l8.58-3.9a1 1 0 0 0 0-1.83Z",
        "m22 17.65-9.17 4.16a2 2 0 0 1-1.66 0L2 17.65",
        "m22 12.65-9.17 4.16a2 2 0 0 1-1.66 0L2 12.65",
    ],
    "filter": ["M22 3H2l8 9.46V19l4 2v-8.54L22 3z"],
    "calendar": ["M3 4h18v18H3z", "M16 2v4", "M8 2v4", "M3 10h18"],
    "refresh": [
        "M3 12a9 9 0 0 1 9-9 9.75 9.75 0 0 1 6.74 2.74L21 8",
        "M21 3v5h-5",
        "M21 12a9 9 0 0 1-9 9 9.75 9.75 0 0 1-6.74-2.74L3 16",
        "M8 16H3v5",
    ],
    "sparkles": [
        "m12 3-1.9 5.8a2 2 0 0 1-1.287 1.288L3 12l5.8 1.9a2 2 0 0 1 1.288 1.287L12 21l1.9-5.8a2 2 0 0 1 1.287-1.288L21 12l-5.8-1.9a2 2 0 0 1-1.288-1.287Z",
        "M5 3v4",
        "M19 17v4",
        "M3 5h4",
        "M17 19h4",
    ],
    "zap": [
        "M4 14a1 1 0 0 1-.78-1.63l9.9-10.2a.5.5 0 0 1 .86.46l-1.92 6.02A1 1 0 0 0 13 10h7a1 1 0 0 1 .78 1.63l-9.9 10.2a.5.5 0 0 1-.86-.46l1.92-6.02A1 1 0 0 0 11 14z",
    ],
    "eye": ["M2.062 12.348a1 1 0 0 1 0-.696 10.75 10.75 0 0 1 19.876 0 1 1 0 0 1 0 .696 10.75 10.75 0 0 1-19.876 0", "M12 9a3 3 0 1 0 0 .01"],
    "bar-chart": ["M3 3v16a2 2 0 0 0 2 2h16", "M18 17V9", "M13 17V5", "M8 17v-3"],
    "monitor": ["M2 3h20v14H2z", "M8 21h8", "M12 17v4"],
    "monitor-heart": ["M2 3h20v14H2z", "M4.5 11h3l1.5 3 2.5-6 2 4.5 1.5-1.5h4.5"],
    "info": ["M12 2a10 10 0 1 0 0 .01", "M12 16v-4", "M12 8h.01"],
    "wifi": ["M12 20h.01", "M2 8.82a15 15 0 0 1 20 0", "M5 12.86a10 10 0 0 1 14 0", "M8.5 16.43a5 5 0 0 1 7 0"],
    "lock": ["M3 11h18v11H3z", "M7 11V7a5 5 0 0 1 10 0v4"],
    "plus": ["M12 5v14", "M5 12h14"],
    "minus": ["M5 12h14"],
    "x": ["M18 6 6 18", "M6 6l12 12"],
    "circle-dot": ["M12 2a10 10 0 1 0 0 .01", "M12 11a1 1 0 1 0 0 .01"],
    "cross-medical": ["M9 3h6v6h6v6h-6v6H9v-6H3V9h6z"],
    "flask-conical": ["M10 2v7.5L4.5 20a1 1 0 0 0 .9 1.5h13.2a1 1 0 0 0 .9-1.5L14 9.5V2", "M8.5 2h7", "M6.5 15h11"],
    "clipboard-check": [
        "M8 2h8v4H8z",
        "M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2",
        "m9 14 2 2 4-4",
    ],
    "bookmark": ["M19 21l-7-5-7 5V3h14z"],
    "grid": ["M3 3h7v7H3z", "M14 3h7v7h-7z", "M14 14h7v7h-7z", "M3 14h7v7H3z"],
    "list": ["M8 6h13", "M8 12h13", "M8 18h13", "M3 6h.01", "M3 12h.01", "M3 18h.01"],
    "alarm-clock": ["M12 2a8 8 0 1 0 0 .01", "M12 6v6l4 2", "M5 3 2 6", "M22 6l-3-3"],
    "shield-alert": [
        "M20 13c0 5-3.5 7.5-7.66 8.95a1 1 0 0 1-.67-.01C7.5 20.5 4 18 4 13V6a1 1 0 0 1 1-1c2 0 4.5-1.2 6.24-2.72a1.17 1.17 0 0 1 1.52 0C14.51 3.81 17 5 19 5a1 1 0 0 1 1 1z",
        "M12 8v4",
        "M12 16h.01",
    ],
    "scale": ["M12 3v18", "M6 7h12", "m6 7-4 8h8z", "m18 7-4 8h8z", "M8 21h8"],
    "workflow": [
        "M3 3h6v6H3z",
        "M15 15h6v6h-6z",
        "M9 6h4a2 2 0 0 1 2 2v7",
        "M15 12h-4a2 2 0 0 0-2 2v1",
    ],
    "pie-chart": [
        "M21.21 15.89A10 10 0 1 1 8 2.83",
        "M22 12A10 10 0 0 0 12 2v10z",
    ],
    "history": ["M3 12a9 9 0 1 0 3-6.7L3 8", "M3 3v5h5", "M12 7v5l4 2"],
    "inbox": [
        "M22 12h-6l-2 3h-4l-2-3H2",
        "M5.45 5.11 2 12v6a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2v-6l-3.45-6.89A2 2 0 0 0 16.76 4H7.24a2 2 0 0 0-1.79 1.11z",
    ],
    "bell-off": [
        "M8.7 3A6 6 0 0 1 18 8c0 2.2.8 4 1.5 5.1",
        "M17 17H4a1 1 0 0 1-.74-1.67C4.59 13.96 6 12.5 6 8a6 6 0 0 1 .2-1.6",
        "M10.3 21a1.94 1.94 0 0 0 3.4 0",
        "m2 2 20 20",
    ],
    "package": [
        "m7.5 4.27 9 5.15",
        "M21 8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16Z",
        "m3.3 7 8.7 5 8.7-5",
        "M12 22V12",
    ],
    "fingerprint": [
        "M12 10a2 2 0 0 0-2 2c0 1.02-.1 2.51-.26 4",
        "M14 13.12c0 2.38 0 6.38-1 8.88",
        "M17.29 21.02c.12-.6.43-2.3.5-3.02",
        "M2 12a10 10 0 0 1 18-6",
        "M2 16h.01",
        "M21.8 16c.2-2 .131-5.354 0-6",
        "M5 19.5C5.5 18 6 15 6 12a6 6 0 0 1 .34-2",
        "M8.65 22c.21-.66.45-1.32.57-2",
        "M9 6.8a6 6 0 0 1 9 5.2v2",
    ],
    "tag": [
        "M12.586 2.586A2 2 0 0 0 11.172 2H4a2 2 0 0 0-2 2v7.172a2 2 0 0 0 .586 1.414l8.704 8.704a2.426 2.426 0 0 0 3.42 0l6.58-6.58a2.426 2.426 0 0 0 0-3.42z",
        "M7.5 7.5h.01",
    ],
    "hash": ["M4 9h16", "M4 15h16", "M10 3 8 21", "M16 3l-2 18"],
    "shuffle": [
        "m18 14 4 4-4 4",
        "m18 2 4 4-4 4",
        "M2 18h1.97a2 2 0 0 0 1.6-.8l.13-.17a2 2 0 0 1 1.6-.8h.7a2 2 0 0 1 1.6.8l3.5 4.66a2 2 0 0 0 1.6.8H22",
        "M2 6h2.97a2 2 0 0 1 1.6.8l.13.17a2 2 0 0 0 1.6.8h.7a2 2 0 0 0 1.6-.8l3.5-4.66a2 2 0 0 1 1.6-.8H22",
    ],
    "terminal": ["m4 17 6-6-6-6", "M12 19h8"],
    "book-open": [
        "M12 7v14",
        "M3 18a1 1 0 0 1-1-1V4a1 1 0 0 1 1-1h5a4 4 0 0 1 4 4 4 4 0 0 1 4-4h5a1 1 0 0 1 1 1v13a1 1 0 0 1-1 1h-6a3 3 0 0 0-3 3 3 3 0 0 0-3-3z",
    ],
    "sparkline": ["M3 17l6-6 4 4 8-8"],
}


def icon(name: str, size: int = 18, color: str = "currentColor", stroke_width: float = 1.7) -> str:
    """Return an inline SVG string. Unknown names fall back to a dot."""
    paths = _ICONS.get(name) or _ICONS["circle-dot"]
    body = "".join(f'<path d="{p}" />' for p in paths)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" '
        f'fill="none" stroke="{color}" stroke-width="{stroke_width}" stroke-linecap="round" '
        f'stroke-linejoin="round" aria-hidden="true">{body}</svg>'
    )


def sparkline(
    values: list[float],
    color: str,
    width: int = 132,
    height: int = 40,
    fill: bool = True,
    stroke_width: float = 2.0,
) -> str:
    """Render a smooth-ish sparkline as an inline SVG string from real values."""
    clean = [v for v in values if v is not None]
    if len(clean) < 2:
        return f'<svg width="{width}" height="{height}"></svg>'
    lo, hi = min(clean), max(clean)
    span = (hi - lo) or 1.0
    pad = 3
    step = (width - pad * 2) / (len(clean) - 1)
    pts = []
    for i, v in enumerate(clean):
        x = pad + i * step
        y = height - pad - ((v - lo) / span) * (height - pad * 2)
        pts.append((round(x, 2), round(y, 2)))
    line = " ".join(f"{x},{y}" for x, y in pts)
    area = f"{pad},{height - pad} " + line + f" {width - pad},{height - pad}"
    gid = f"sp{next(_SPARK_IDS)}"
    area_svg = (
        f'<polygon points="{area}" fill="url(#{gid})" stroke="none" />' if fill else ""
    )
    return (
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
        f'xmlns="http://www.w3.org/2000/svg" aria-hidden="true">'
        f'<defs><linearGradient id="{gid}" x1="0" y1="0" x2="0" y2="1">'
        f'<stop offset="0%" stop-color="{color}" stop-opacity="0.32" />'
        f'<stop offset="100%" stop-color="{color}" stop-opacity="0" />'
        f"</linearGradient></defs>"
        f'{area_svg}<polyline points="{line}" fill="none" stroke="{color}" '
        f'stroke-width="{stroke_width}" stroke-linecap="round" stroke-linejoin="round" />'
        f'<circle cx="{pts[-1][0]}" cy="{pts[-1][1]}" r="2.6" fill="{color}" />'
        f"</svg>"
    )


def donut_svg(segments: list[tuple[int, str]], size: int = 68, thickness: int = 7) -> str:
    """Compact donut from (value, color) segments, used inside KPI cards."""
    total = sum(v for v, _ in segments) or 1
    radius = (size - thickness) / 2
    circumference = 2 * 3.14159265 * radius
    offset = 0.0
    circles = []
    for value, color in segments:
        portion = value / total
        dash = max(circumference * portion - 2, 0)
        circles.append(
            f'<circle cx="{size/2}" cy="{size/2}" r="{radius}" fill="none" stroke="{color}" '
            f'stroke-width="{thickness}" stroke-dasharray="{dash:.2f} {circumference:.2f}" '
            f'stroke-dashoffset="{-offset:.2f}" stroke-linecap="round" transform="rotate(-90 {size/2} {size/2})" />'
        )
        offset += circumference * portion
    return (
        f'<svg width="{size}" height="{size}" viewBox="0 0 {size} {size}" aria-hidden="true">'
        f'<circle cx="{size/2}" cy="{size/2}" r="{radius}" fill="none" stroke="rgba(255,255,255,0.06)" '
        f'stroke-width="{thickness}" />{"".join(circles)}</svg>'
    )


def escape_html(text) -> str:
    return escape(str(text))
