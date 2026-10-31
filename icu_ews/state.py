"""Session state, dataset selection and navigation helpers."""

from __future__ import annotations

import streamlit as st

from .config import DEFAULT_BUNDLE, DEFAULT_THRESHOLDS
from .ml import artifacts

VALID_VIEWS = {
    "overview",
    "monitoring",
    "patient",
    "alerts",
    "prediction",
    "analytics",
    "xai",
    "performance",
    "reports",
    "settings",
}

# filter namespaces that must reset when the underlying cohort changes
_RESET_PREFIXES = ("ov_", "mon_", "px_", "al_", "an_", "xai_", "pf_")


def ensure_state() -> None:
    ss = st.session_state

    if "dataset" not in ss or not artifacts.bundle_exists(ss.dataset):
        bundles = artifacts.list_bundles()
        slugs = [b["slug"] for b in bundles]
        ss.dataset = DEFAULT_BUNDLE if DEFAULT_BUNDLE in slugs else (slugs[0] if slugs else "")

    ss.setdefault("thresholds", dict(DEFAULT_THRESHOLDS))
    ss.setdefault("view", "overview")
    ss.setdefault("ov_band", "All")
    ss.setdefault("ov_period", "Last 7 days")
    ss.setdefault("ov_sort", "Highest risk")
    ss.setdefault("ov_page_size", 6)

    qp_view = st.query_params.get("view")
    if qp_view in VALID_VIEWS:
        ss.view = qp_view
    elif ss.view not in VALID_VIEWS:
        ss.view = "overview"


def navigate(view: str, pid: str | None = None) -> None:
    st.session_state["view"] = view if view in VALID_VIEWS else "overview"
    st.query_params.clear()
    st.query_params["view"] = st.session_state["view"]
    if pid:
        st.query_params["pid"] = pid
    st.rerun()


def current_pid() -> str | None:
    return st.query_params.get("pid")


def active_nav(view: str) -> str:
    return "monitoring" if view == "patient" else view


def switch_dataset(slug: str) -> None:
    ss = st.session_state
    ss.dataset = slug
    ss["dataset_pick"] = slug
    for key in list(ss.keys()):
        if key.startswith(_RESET_PREFIXES):
            ss.pop(key, None)
    ss.view = "overview"
    st.query_params.clear()
    st.query_params["view"] = "overview"


def thresholds() -> dict[str, float]:
    return st.session_state.thresholds
