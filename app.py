"""ICU-EWS entry point."""

from __future__ import annotations

import streamlit as st

from icu_ews.shell import sidebar, topbar
from icu_ews.state import active_nav, ensure_state
from icu_ews.theme import inject_css
from icu_ews.views import render

st.set_page_config(
    page_title="ICU-EWS · ICU command centre",
    page_icon=":material/monitor_heart:",
    layout="wide",
    initial_sidebar_state="expanded",
)

ensure_state()
view = st.session_state.view
inject_css(active_nav(view))
topbar()
sidebar(active_nav(view))
render(view)
