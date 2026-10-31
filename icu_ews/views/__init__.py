"""Route dispatch for ICU-EWS views."""

from __future__ import annotations


def render(view: str) -> None:
    if view == "overview":
        from .overview import render as page
    elif view == "monitoring":
        from .monitoring import render as page
    elif view == "patient":
        from .patient import render as page
    elif view == "alerts":
        from .alerts import render as page
    elif view == "prediction":
        from .prediction import render as page
    elif view == "analytics":
        from .analytics import render as page
    elif view == "xai":
        from .xai import render as page
    elif view == "performance":
        from .performance import render as page
    elif view == "reports":
        from .reports import render as page
    else:
        from .settings import render as page
    page()
