"""Global design system for ICU-EWS.

A single injected stylesheet drives the whole application: Streamlit chrome
restyling, the fixed top navigation, sidebar navigation states, custom card
components and Plotly containers.
"""

from __future__ import annotations

import streamlit as st

from .config import PALETTE, RISK_META

_FONT_STACK = (
    "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif"
)
_MONO_STACK = "'JetBrains Mono', 'SF Mono', ui-monospace, Menlo, Consolas, monospace"

_CSS = f"""
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {{
  --bg: {PALETTE['bg']};
  --bg-soft: {PALETTE['bg_soft']};
  --surface: {PALETTE['surface']};
  --card: {PALETTE['card']};
  --card-alt: {PALETTE['card_alt']};
  --elevated: {PALETTE['elevated']};
  --border: {PALETTE['border']};
  --border-strong: {PALETTE['border_strong']};
  --text: {PALETTE['text']};
  --muted: {PALETTE['text_muted']};
  --dim: {PALETTE['text_dim']};
  --cyan: {PALETTE['cyan']};
  --blue: {PALETTE['blue']};
  --indigo: {PALETTE['indigo']};
  --violet: {PALETTE['violet']};
  --critical: {PALETTE['critical']};
  --high: {PALETTE['high']};
  --moderate: {PALETTE['moderate']};
  --low: {PALETTE['low']};
  --info: {PALETTE['info']};
  --teal: {PALETTE['teal']};
  --pink: {PALETTE['pink']};
  --radius: 16px;
  --radius-sm: 10px;
  --topbar-h: 76px;
  --sidebar-w: 272px;
  --font: {_FONT_STACK};
  --mono: {_MONO_STACK};
  --shadow-card: 0 18px 40px -24px rgba(0, 0, 0, 0.85);
  --shadow-pop: 0 24px 60px -18px rgba(0, 0, 0, 0.75);
}}

html, body, .stApp, [class*="st-"] {{
  font-family: var(--font);
  -webkit-font-smoothing: antialiased;
  text-rendering: optimizeLegibility;
}}
/* keep Streamlit's Material icon ligature font intact */
[data-testid="stIconMaterial"],
span[data-testid="stIconMaterial"],
[data-testid="stIconMaterial"] * {{
  font-family: "Material Symbols Rounded" !important;
  font-weight: normal !important;
  font-style: normal !important;
  font-feature-settings: "liga" !important;
  letter-spacing: normal !important;
  text-transform: none !important;
  white-space: nowrap;
  line-height: 1;
}}

/* ---------------------------------------------------------------- shell */
.stApp {{
  background:
    radial-gradient(1200px 720px at 88% -12%, rgba(34, 211, 238, 0.075), transparent 58%),
    radial-gradient(1000px 620px at -8% 4%, rgba(59, 130, 246, 0.09), transparent 55%),
    radial-gradient(900px 900px at 50% 120%, rgba(129, 140, 248, 0.06), transparent 60%),
    linear-gradient(180deg, #060b14 0%, #050910 100%);
  color: var(--text);
}}

[data-testid="stHeader"] {{ display: none !important; }}
[data-testid="stToolbar"], [data-testid="stDecoration"], [data-testid="stStatusWidget"] {{ display: none !important; }}
footer {{ display: none !important; }}
#MainMenu {{ visibility: hidden; }}

[data-testid="stAppViewContainer"] {{ background: transparent; }}

/* main column */
[data-testid="stMain"] {{ margin-left: var(--sidebar-w); width: calc(100% - var(--sidebar-w)); }}
.stAppViewContainer:has(section[data-testid="stSidebar"][aria-expanded="false"]) [data-testid="stMain"] {{
  margin-left: 0;
  width: 100%;
}}
[data-testid="stMainBlockContainer"], .block-container {{
  padding: calc(var(--topbar-h) + 1.6rem) 1.9rem 4.5rem 1.9rem !important;
  max-width: 1760px !important;
}}

/* ---------------------------------------------------------------- topbar */
.st-key-topnav {{
  position: fixed;
  top: 0; left: 0; right: 0;
  height: var(--topbar-h);
  z-index: 1000;
  padding: 0 1.35rem 0 1.5rem !important;
  background: rgba(6, 12, 22, 0.82);
  backdrop-filter: blur(20px) saturate(150%);
  -webkit-backdrop-filter: blur(20px) saturate(150%);
  border-bottom: 1px solid rgba(27, 42, 66, 0.9);
  box-shadow: 0 18px 40px -30px rgba(0, 0, 0, 0.9);
  gap: 0 !important;
}}
.st-key-topnav > div {{ gap: 0 !important; }}
.st-key-topnav [data-testid="stHorizontalBlock"] {{ align-items: center; gap: 1rem; }}
.st-key-topnav [data-testid="stElementContainer"] {{ margin: 0 !important; }}

.brand {{ display: flex; align-items: center; gap: 0.72rem; }}
.brand-mark {{
  width: 42px; height: 42px; border-radius: 12px;
  display: flex; align-items: center; justify-content: center;
  background: linear-gradient(140deg, rgba(34, 211, 238, 0.22), rgba(59, 130, 246, 0.22));
  border: 1px solid rgba(34, 211, 238, 0.35);
  box-shadow: 0 0 22px rgba(34, 211, 238, 0.22), inset 0 1px 0 rgba(255, 255, 255, 0.08);
  color: #7ff3ff;
}}
.brand-name {{
  font-size: 1.02rem; font-weight: 800; letter-spacing: 0.06em; line-height: 1.1;
  background: linear-gradient(90deg, #ffffff 10%, #a7e9ff 90%);
  -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent;
  display: flex; align-items: center; gap: 0.5rem;
}}
.brand-sub {{
  font-size: 0.63rem; color: var(--muted); letter-spacing: 0.13em;
  text-transform: uppercase; font-weight: 600; margin-top: 3px;
}}
.brand-live {{
  display: inline-flex; align-items: center; gap: 5px; margin-left: 6px;
  font-size: 0.56rem; letter-spacing: 0.12em; font-weight: 700;
  color: #6ee7b7; background: rgba(16, 185, 129, 0.12);
  border: 1px solid rgba(16, 185, 129, 0.3);
  padding: 2px 7px; border-radius: 999px; text-transform: uppercase;
  -webkit-text-fill-color: #6ee7b7;
}}
.pulse-dot {{
  width: 6px; height: 6px; border-radius: 50%; background: #34d399;
  box-shadow: 0 0 0 0 rgba(52, 211, 153, 0.7); animation: pulse 2.4s infinite;
}}
@keyframes pulse {{
  0% {{ box-shadow: 0 0 0 0 rgba(52, 211, 153, 0.55); }}
  70% {{ box-shadow: 0 0 0 7px rgba(52, 211, 153, 0); }}
  100% {{ box-shadow: 0 0 0 0 rgba(52, 211, 153, 0); }}
}}
.clock {{ display: flex; flex-direction: column; line-height: 1.15; text-align: right; }}
.clock-time {{ font-family: var(--mono); font-size: 0.92rem; font-weight: 600; color: #dbe9fb; letter-spacing: 0.03em; }}
.clock-date {{ font-size: 0.66rem; color: var(--muted); letter-spacing: 0.05em; text-transform: uppercase; }}

.topbar-divider {{ width: 1px; height: 32px; background: var(--border); }}

/* topbar widget sizing */
.st-key-topnav [data-baseweb="select"] > div,
.st-key-topnav [data-baseweb="input"] {{
  height: 40px; min-height: 40px; border-radius: 10px !important;
  background: rgba(13, 24, 38, 0.85) !important;
  border: 1px solid rgba(38, 58, 88, 0.9) !important;
}}
.st-key-topnav input {{ font-size: 0.82rem !important; }}
.st-key-topnav [data-testid="stTextInputRootElement"] {{ border: none !important; background: transparent !important; }}
.st-key-topnav .stButton > button,
.st-key-topnav [data-testid="stPopover"] button {{
  height: 40px; min-width: 42px; border-radius: 10px;
  background: rgba(13, 24, 38, 0.85);
  border: 1px solid rgba(38, 58, 88, 0.9);
}}

/* ---------------------------------------------------------------- sidebar */
section[data-testid="stSidebar"] {{
  position: fixed !important;
  top: var(--topbar-h); left: 0; bottom: 0;
  width: var(--sidebar-w) !important;
  min-width: var(--sidebar-w) !important;
  height: auto !important;
  background:
    radial-gradient(700px 400px at 0% 0%, rgba(34, 211, 238, 0.06), transparent 60%),
    linear-gradient(180deg, #08111e 0%, #060d17 100%) !important;
  border-right: 1px solid var(--border);
  z-index: 900;
}}
section[data-testid="stSidebar"] > div:first-child {{
  padding-top: 0.55rem;
  height: 100%;
}}
section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {{
  padding: 0.35rem 0.85rem 1.2rem 0.95rem;
}}
section[data-testid="stSidebar"] [data-testid="stSidebarHeader"] {{
  padding: 0.35rem 0.5rem 0 0.4rem;
  height: 2.3rem;
  min-height: 2.3rem;
}}
section[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"] button {{
  position: relative;
  visibility: visible !important;
  color: var(--dim);
  background: rgba(11, 21, 36, 0.55);
  border: 1px solid var(--border);
  border-radius: 8px;
  width: 28px; height: 28px;
  min-height: 28px;
  transition: color 0.15s ease, border-color 0.15s ease, background 0.15s ease;
}}
section[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"] button:hover {{
  color: var(--cyan);
  border-color: rgba(34, 211, 238, 0.45);
  background: rgba(34, 211, 238, 0.08);
}}
section[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"] button:hover::after {{
  content: "Collapse sidebar";
  position: absolute; left: calc(100% + 8px); top: 50%; transform: translateY(-50%);
  white-space: nowrap; padding: 4px 8px; border-radius: 6px;
  background: #0c1727; border: 1px solid var(--border-strong); color: var(--text);
  font-size: 0.65rem; font-weight: 600; z-index: 1200; pointer-events: none;
}}
header[data-testid="stHeader"]:has([data-testid="stExpandSidebarButton"]) {{
  display: block !important;
  position: fixed !important;
  top: calc(var(--topbar-h) + 10px) !important;
  left: 0.85rem !important;
  width: 34px !important;
  height: 34px !important;
  z-index: 1005 !important;
  background: transparent !important;
  border: none !important;
}}
[data-testid="stToolbar"]:has([data-testid="stExpandSidebarButton"]) {{
  display: flex !important;
  position: static !important;
  height: 34px !important;
  background: transparent !important;
}}
[data-testid="stExpandSidebarButton"] {{
  position: relative;
  width: 34px !important;
  height: 34px !important;
  border-radius: 10px !important;
  border: 1px solid var(--border-strong) !important;
  background: var(--card) !important;
  color: var(--text) !important;
  box-shadow: 0 12px 26px -20px rgba(0, 0, 0, 0.9);
  transition: color 0.15s ease, border-color 0.15s ease, background 0.15s ease;
}}
[data-testid="stExpandSidebarButton"]:hover {{
  color: var(--cyan) !important;
  border-color: rgba(34, 211, 238, 0.45) !important;
  background: #0e1e33 !important;
}}
[data-testid="stExpandSidebarButton"]:hover::after {{
  content: "Expand sidebar";
  position: absolute; left: calc(100% + 8px); top: 50%; transform: translateY(-50%);
  white-space: nowrap; padding: 4px 8px; border-radius: 6px;
  background: #0c1727; border: 1px solid var(--border-strong); color: var(--text);
  font-size: 0.65rem; font-weight: 600; z-index: 1200; pointer-events: none;
}}

.side-caption {{
  font-size: 0.6rem; letter-spacing: 0.16em; text-transform: uppercase;
  color: var(--dim); font-weight: 700; margin: 1.15rem 0 0.35rem 0.55rem;
}}

/* sidebar nav buttons */
section[data-testid="stSidebar"] .stButton > button {{
  width: 100%;
  justify-content: flex-start;
  gap: 0.65rem;
  padding: 0.5rem 0.72rem;
  min-height: 42px;
  border-radius: 11px;
  background: transparent;
  border: 1px solid transparent;
  color: #a9bdd8;
  font-size: 0.845rem;
  font-weight: 550;
  letter-spacing: 0.01em;
  transition: background 0.16s ease, color 0.16s ease, border-color 0.16s ease;
}}
section[data-testid="stSidebar"] .stButton > button:hover {{
  background: rgba(34, 211, 238, 0.07);
  color: #eaf7ff;
  border-color: rgba(34, 211, 238, 0.16);
  transform: none;
  box-shadow: none;
}}
section[data-testid="stSidebar"] .stButton > button p {{ text-align: left; font-size: 0.81rem; white-space: nowrap; }}
section[data-testid="stSidebar"] .stButton > button [data-testid="stIconMaterial"] {{
  font-size: 19px; color: var(--muted); transition: color 0.16s ease;
}}
section[data-testid="stSidebar"] .stButton > button:hover [data-testid="stIconMaterial"] {{ color: var(--cyan); }}
section[data-testid="stSidebar"] [data-testid="stElementContainer"] {{ margin-bottom: 0.15rem; }}

.nav-active button {{
  background: linear-gradient(90deg, rgba(34, 211, 238, 0.16) 0%, rgba(59, 130, 246, 0.07) 70%, transparent 100%) !important;
  border-color: rgba(34, 211, 238, 0.26) !important;
  color: #f2fdff !important;
  box-shadow: inset 3px 0 0 0 var(--cyan), 0 8px 22px -16px rgba(34, 211, 238, 0.6);
  font-weight: 650 !important;
}}
.nav-active button [data-testid="stIconMaterial"] {{ color: var(--cyan) !important; }}

.side-demo {{
  margin-top: 1rem;
  padding: 0.85rem 0.9rem;
  border-radius: 13px;
  border: 1px solid rgba(34, 211, 238, 0.2);
  background:
    linear-gradient(160deg, rgba(34, 211, 238, 0.1), rgba(59, 130, 246, 0.05) 60%, transparent 100%),
    rgba(11, 21, 36, 0.8);
}}
.side-demo-title {{ font-size: 0.68rem; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase; color: #7fe9f9; display: flex; align-items: center; gap: 6px; }}
.side-demo-body {{ font-size: 0.68rem; color: var(--muted); line-height: 1.5; margin-top: 6px; }}
.side-foot {{ display: flex; align-items: center; gap: 0.6rem; margin-top: 0.95rem; padding: 0.55rem 0.5rem 0 0.5rem; border-top: 1px solid var(--border); }}
.side-foot-avatar {{
  width: 30px; height: 30px; border-radius: 9px; display: flex; align-items: center; justify-content: center;
  background: linear-gradient(135deg, rgba(129, 140, 248, 0.28), rgba(34, 211, 238, 0.22));
  border: 1px solid rgba(129, 140, 248, 0.35); font-size: 0.64rem; font-weight: 700; color: #dfe8ff;
}}
.side-foot-name {{ font-size: 0.74rem; font-weight: 600; color: #d9e6f7; line-height: 1.2; }}
.side-foot-role {{ font-size: 0.62rem; color: var(--dim); }}

/* ---------------------------------------------------------------- widgets */
[data-testid="stWidgetLabel"] p {{
  font-size: 0.67rem !important;
  letter-spacing: 0.09em;
  text-transform: uppercase;
  color: var(--muted) !important;
  font-weight: 650 !important;
  margin-bottom: 0.25rem;
}}
[data-testid="stTextInput"] [data-baseweb="input"],
[data-testid="stDateInput"] [data-baseweb="input"],
[data-testid="stNumberInput"] [data-baseweb="input"],
[data-testid="stSelectbox"] [data-baseweb="select"] > div,
[data-testid="stMultiSelect"] [data-baseweb="select"] > div,
[data-baseweb="textarea"] {{
  background: rgba(11, 21, 36, 0.85) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-sm) !important;
  color: var(--text) !important;
  min-height: 40px;
  transition: border-color 0.15s ease, box-shadow 0.15s ease;
}}
[data-testid="stTextInput"] [data-baseweb="input"]:focus-within,
[data-testid="stDateInput"] [data-baseweb="input"]:focus-within,
[data-testid="stNumberInput"] [data-baseweb="input"]:focus-within,
[data-testid="stSelectbox"] [data-baseweb="select"] > div:focus-within,
[data-testid="stMultiSelect"] [data-baseweb="select"] > div:focus-within {{
  border-color: rgba(34, 211, 238, 0.6) !important;
  box-shadow: 0 0 0 3px rgba(34, 211, 238, 0.1);
}}
input, textarea {{ color: var(--text) !important; font-size: 0.84rem !important; }}
input::placeholder {{ color: var(--dim) !important; }}
[data-baseweb="select"] [data-testid="stMarkdownContainer"] p {{ font-size: 0.84rem; }}
[data-baseweb="popover"] [role="listbox"] {{
  background: #0c1727 !important;
  border: 1px solid var(--border-strong) !important;
  border-radius: 12px !important;
  box-shadow: var(--shadow-pop) !important;
}}
[data-baseweb="popover"] [role="option"] {{ font-size: 0.82rem; }}
[data-baseweb="popover"] [role="option"]:hover {{ background: rgba(34, 211, 238, 0.1) !important; }}

/* buttons */
.stButton > button, .stDownloadButton > button {{
  border-radius: var(--radius-sm);
  font-size: 0.8rem;
  font-weight: 600;
  border: 1px solid var(--border-strong);
  background: linear-gradient(180deg, rgba(23, 38, 61, 0.95), rgba(15, 26, 43, 0.95));
  color: #dbe8f9;
  transition: all 0.16s ease;
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.04);
}}
.stButton > button:hover, .stDownloadButton > button:hover {{
  border-color: rgba(34, 211, 238, 0.45);
  color: #f2fdff;
  background: linear-gradient(180deg, rgba(30, 49, 76, 0.95), rgba(18, 31, 51, 0.95));
  box-shadow: 0 10px 24px -16px rgba(34, 211, 238, 0.55), inset 0 1px 0 rgba(255, 255, 255, 0.05);
  transform: translateY(-1px);
}}
.stButton > button:focus-visible, .stDownloadButton > button:focus-visible {{
  outline: 2px solid rgba(34, 211, 238, 0.75);
  outline-offset: 2px;
}}
.stButton > button[kind="primary"], .stButton > button[data-testid="stBaseButton-primary"] {{
  background: linear-gradient(135deg, #22d3ee 0%, #38bdf8 45%, #3b82f6 100%);
  border: 1px solid rgba(125, 240, 255, 0.5);
  color: #04131f;
  font-weight: 750;
  letter-spacing: 0.01em;
  box-shadow: 0 12px 30px -14px rgba(34, 211, 238, 0.65);
}}
.stButton > button[kind="primary"]:hover, .stButton > button[data-testid="stBaseButton-primary"]:hover {{
  filter: brightness(1.08);
  color: #020e17;
}}
.stButton > button[kind="tertiary"], .stButton > button[data-testid="stBaseButton-tertiary"] {{
  background: transparent;
  border-color: transparent;
  box-shadow: none;
  color: var(--muted);
}}
.stButton > button[kind="tertiary"]:hover, .stButton > button[data-testid="stBaseButton-tertiary"]:hover {{
  background: rgba(34, 211, 238, 0.08);
  color: var(--text);
  transform: none;
  box-shadow: none;
}}

/* segmented control */
[data-testid="stButtonGroup"] {{ gap: 4px; }}
[data-testid="stBaseButton-segmented_control"],
[data-testid="stBaseButton-segmented_controlActive"] {{
  border-radius: 9px !important;
  border: 1px solid var(--border) !important;
  background: rgba(11, 21, 36, 0.7) !important;
  color: var(--muted) !important;
  font-size: 0.74rem !important;
  font-weight: 600 !important;
  padding: 0.25rem 0.7rem !important;
  min-height: 32px;
  box-shadow: none !important;
}}
[data-testid="stBaseButton-segmented_controlActive"] {{
  background: rgba(34, 211, 238, 0.14) !important;
  border-color: rgba(34, 211, 238, 0.45) !important;
  color: #b9f5ff !important;
}}

/* sliders, toggles, radios */
[data-testid="stSlider"] [role="slider"] {{ background: var(--cyan) !important; box-shadow: 0 0 0 4px rgba(34, 211, 238, 0.18) !important; }}
[data-testid="stSlider"] [data-baseweb="slider"] div[role="progressbar"] {{ background: linear-gradient(90deg, #22d3ee, #3b82f6) !important; }}
[data-testid="stToggle"] label span {{ background: rgba(34, 211, 238, 0.3) !important; }}
[data-testid="stCheckbox"] label span[data-baseweb="checkbox"] div {{ border-color: var(--border-strong) !important; }}

/* expander */
[data-testid="stExpander"] {{
  border: 1px solid var(--border) !important;
  border-radius: 13px !important;
  background: rgba(12, 22, 38, 0.6);
  overflow: hidden;
}}
[data-testid="stExpander"] summary {{
  font-size: 0.8rem;
  font-weight: 600;
  color: #c8d8ee;
  padding: 0.65rem 0.9rem;
}}
[data-testid="stExpander"] summary:hover {{ color: #eaf7ff; }}

/* dataframe */
[data-testid="stDataFrame"] {{
  border: 1px solid var(--border);
  border-radius: 13px;
  overflow: hidden;
  background: rgba(9, 17, 30, 0.6);
}}
[data-testid="stDataFrame"] [role="columnheader"] {{ background: #0d1827 !important; }}
[data-testid="stElementToolbar"] {{ display: none !important; }}

/* plotly wrapper */
[data-testid="stPlotlyChart"] {{ border-radius: 12px; overflow: hidden; }}
.js-plotly-plot .plotly .modebar {{ opacity: 0.25; }}
.js-plotly-plot .plotly .modebar:hover {{ opacity: 1; }}

/* tabs (if used) */
[data-testid="stTabs"] [data-baseweb="tab-list"] {{ gap: 4px; border-bottom: 1px solid var(--border); }}
[data-testid="stTabs"] [data-baseweb="tab"] {{
  font-size: 0.8rem; font-weight: 600; color: var(--muted);
  padding: 0.45rem 0.9rem; border-radius: 9px 9px 0 0;
}}
[data-testid="stTabs"] [aria-selected="true"] {{ color: #dff8ff; background: rgba(34, 211, 238, 0.08); }}
[data-testid="stTabs"] [data-baseweb="tab-highlight"] {{ background: var(--cyan); }}

/* popover */
div[data-baseweb="popover"] > div,
[data-testid="stPopoverBody"] {{
  background: #0b1626 !important;
  border: 1px solid var(--border-strong) !important;
  border-radius: 14px !important;
  box-shadow: var(--shadow-pop) !important;
  color: var(--text);
}}
[data-testid="stPopoverBody"] {{ padding: 0.9rem 1rem; }}

/* toast / status */
[data-testid="stToast"] {{
  background: #0d1a2c; border: 1px solid var(--border-strong); color: var(--text);
  border-radius: 12px; box-shadow: var(--shadow-pop);
}}
[data-testid="stAlert"] {{ border-radius: 12px; border: 1px solid var(--border-strong); background: rgba(15, 27, 45, 0.9); }}

/* spinner / skeleton */
.stSpinner > div {{ border-top-color: var(--cyan) !important; }}

/* scrollbars */
::-webkit-scrollbar {{ width: 9px; height: 9px; }}
::-webkit-scrollbar-track {{ background: transparent; }}
::-webkit-scrollbar-thumb {{ background: #22334e; border-radius: 8px; border: 2px solid #0a1220; }}
::-webkit-scrollbar-thumb:hover {{ background: #2f4666; }}

/* ---------------------------------------------------------------- panels */
.panel {{
  background: linear-gradient(180deg, rgba(17, 30, 49, 0.72) 0%, rgba(11, 21, 36, 0.72) 100%);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  box-shadow: var(--shadow-card);
}}
.key-panel {{
  background: linear-gradient(180deg, rgba(17, 30, 49, 0.72) 0%, rgba(11, 21, 36, 0.72) 100%);
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
  box-shadow: var(--shadow-card);
  padding: 1rem 1.05rem 1.05rem 1.05rem !important;
}}
.key-panel-tight {{ padding: 0.85rem 0.9rem 0.9rem 0.9rem !important; }}

.section-head {{ display: flex; align-items: flex-end; justify-content: space-between; gap: 1rem; margin: 0.2rem 0 0.55rem 0; }}
.section-title {{ display: flex; align-items: center; gap: 0.55rem; font-size: 0.98rem; font-weight: 700; color: #e9f1fd; letter-spacing: 0.01em; }}
.section-title .ico {{ color: var(--cyan); display: inline-flex; }}
.section-sub {{ font-size: 0.74rem; color: var(--muted); margin-top: 3px; line-height: 1.45; }}
.section-eyebrow {{ font-size: 0.62rem; letter-spacing: 0.16em; text-transform: uppercase; color: var(--dim); font-weight: 700; }}

/* page header */
.page-head {{ display: flex; align-items: flex-start; justify-content: space-between; gap: 1.2rem; margin-bottom: 1.15rem; flex-wrap: wrap; }}
.page-title {{ font-size: 1.42rem; font-weight: 750; letter-spacing: -0.015em; color: #f2f7ff; line-height: 1.15; }}
.page-sub {{ font-size: 0.8rem; color: var(--muted); margin-top: 0.25rem; line-height: 1.5; max-width: 74ch; }}
.page-meta {{ display: flex; align-items: center; gap: 0.5rem; flex-wrap: wrap; margin-top: 0.6rem; }}

/* chips / badges */
.chip {{
  display: inline-flex; align-items: center; gap: 5px;
  padding: 3px 9px; border-radius: 999px;
  font-size: 0.66rem; font-weight: 650; letter-spacing: 0.02em;
  border: 1px solid var(--border-strong); color: var(--muted);
  background: rgba(11, 21, 36, 0.65);
  white-space: nowrap;
}}
.chip .dot {{ width: 6px; height: 6px; border-radius: 50%; background: currentColor; }}
.chip-cyan {{ color: #8ceeff; border-color: rgba(34, 211, 238, 0.35); background: rgba(34, 211, 238, 0.1); }}
.chip-blue {{ color: #9cc3ff; border-color: rgba(59, 130, 246, 0.35); background: rgba(59, 130, 246, 0.1); }}
.chip-green {{ color: #86efac; border-color: rgba(34, 197, 94, 0.32); background: rgba(34, 197, 94, 0.1); }}
.chip-amber {{ color: #fde68a; border-color: rgba(250, 204, 21, 0.3); background: rgba(250, 204, 21, 0.09); }}
.chip-red {{ color: #fda4af; border-color: rgba(244, 63, 94, 0.32); background: rgba(244, 63, 94, 0.1); }}
.chip-muted {{ color: var(--muted); }}
.chip-indigo {{ color: #c7d2fe; border-color: rgba(129, 140, 248, 0.35); background: rgba(129, 140, 248, 0.1); }}
.chip-teal {{ color: #99f6e4; border-color: rgba(45, 212, 191, 0.32); background: rgba(45, 212, 191, 0.1); }}

.badge {{
  display: inline-flex; align-items: center; gap: 5px;
  font-size: 0.62rem; font-weight: 750; letter-spacing: 0.1em; text-transform: uppercase;
  padding: 3px 8px; border-radius: 7px; border: 1px solid transparent; white-space: nowrap;
}}
.badge .dot {{ width: 5px; height: 5px; border-radius: 50%; background: currentColor; }}

/* KPI cards */
.kpi {{
  position: relative; overflow: hidden;
  padding: 1rem 1.05rem 0.95rem 1.05rem;
  border-radius: var(--radius);
  border: 1px solid var(--border);
  background:
    radial-gradient(120% 140% at 100% 0%, var(--kpi-glow, rgba(34, 211, 238, 0.1)) 0%, transparent 55%),
    linear-gradient(180deg, rgba(19, 33, 53, 0.9) 0%, rgba(12, 22, 38, 0.92) 100%);
  box-shadow: var(--shadow-card);
  transition: transform 0.18s ease, border-color 0.18s ease, box-shadow 0.18s ease;
  min-height: 132px;
}}
.kpi:hover {{
  transform: translateY(-2px);
  border-color: var(--kpi-border, rgba(34, 211, 238, 0.35));
  box-shadow: 0 25px 45px -28px rgba(0, 0, 0, 0.9), 0 0 0 1px var(--kpi-glow, rgba(34, 211, 238, 0.08));
}}
.kpi-top {{ display: flex; align-items: center; justify-content: space-between; gap: 0.5rem; }}
.kpi-label {{ font-size: 0.66rem; letter-spacing: 0.12em; text-transform: uppercase; color: var(--muted); font-weight: 700; }}
.kpi-icon {{
  width: 34px; height: 34px; border-radius: 10px; display: flex; align-items: center; justify-content: center;
  background: var(--kpi-soft, rgba(34, 211, 238, 0.12));
  border: 1px solid var(--kpi-border, rgba(34, 211, 238, 0.28));
  color: var(--kpi-color, var(--cyan));
}}
.kpi-value-row {{ display: flex; align-items: baseline; gap: 0.55rem; margin-top: 0.45rem; }}
.kpi-value {{ font-size: 2.05rem; font-weight: 750; letter-spacing: -0.03em; color: #f4f9ff; line-height: 1; font-variant-numeric: tabular-nums; }}
.kpi-value small {{ font-size: 0.8rem; font-weight: 600; color: var(--muted); letter-spacing: 0; margin-left: 0.35rem; }}
.kpi-bottom {{ display: flex; align-items: flex-end; justify-content: space-between; gap: 0.5rem; margin-top: 0.6rem; }}
.kpi-delta {{ display: inline-flex; align-items: center; gap: 4px; font-size: 0.68rem; font-weight: 650; }}
.kpi-foot {{ font-size: 0.64rem; color: var(--dim); margin-top: 3px; letter-spacing: 0.01em; }}
.kpi-spark {{ opacity: 0.95; line-height: 0; }}

/* patient cards */
.pcard {{
  position: relative;
  border-radius: var(--radius);
  border: 1px solid var(--border);
  border-left: 3px solid var(--risk-color, var(--border-strong));
  background:
    radial-gradient(120% 130% at 100% 0%, var(--risk-glow, rgba(34, 211, 238, 0.06)) 0%, transparent 52%),
    linear-gradient(180deg, rgba(18, 32, 51, 0.92) 0%, rgba(12, 22, 38, 0.94) 100%);
  padding: 0.9rem 1rem 0.8rem 1rem;
  box-shadow: var(--shadow-card);
  transition: transform 0.16s ease, border-color 0.16s ease, box-shadow 0.16s ease;
  min-height: 268px;
}}
.pcard:hover {{
  transform: translateY(-2px);
  border-color: var(--risk-border, rgba(34, 211, 238, 0.4));
  box-shadow: 0 26px 50px -30px rgba(0, 0, 0, 0.95), 0 0 24px -12px var(--risk-glow, rgba(34, 211, 238, 0.3));
}}
.pcard-head {{ display: flex; align-items: flex-start; justify-content: space-between; gap: 0.6rem; }}
.pcard-id {{ font-family: var(--mono); font-size: 0.9rem; font-weight: 600; color: #eef6ff; letter-spacing: 0.01em; display: flex; align-items: center; gap: 0.45rem; }}
.pcard-sub {{ font-size: 0.66rem; color: var(--dim); margin-top: 3px; display: flex; align-items: center; gap: 0.4rem; flex-wrap: wrap; }}
.pcard-sub .sep {{ width: 3px; height: 3px; border-radius: 50%; background: var(--dim); }}
.pcard-prob {{ display: flex; flex-direction: column; align-items: flex-end; gap: 2px; }}
.pcard-prob-val {{ font-family: var(--mono); font-size: 1.02rem; font-weight: 700; color: var(--risk-color, var(--text)); line-height: 1; }}
.pcard-prob-lbl {{ font-size: 0.54rem; letter-spacing: 0.1em; text-transform: uppercase; color: var(--dim); font-weight: 700; }}

.vitals-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.42rem; margin-top: 0.7rem; }}
.vital {{
  background: rgba(8, 16, 28, 0.55);
  border: 1px solid rgba(31, 48, 73, 0.75);
  border-radius: 9px;
  padding: 0.38rem 0.45rem 0.4rem 0.45rem;
  line-height: 1.15;
}}
.vital-top {{ display: flex; align-items: center; gap: 4px; color: var(--dim); }}
.vital-lbl {{ font-size: 0.55rem; letter-spacing: 0.09em; text-transform: uppercase; font-weight: 700; }}
.vital-val {{ font-family: var(--mono); font-size: 0.86rem; font-weight: 600; color: #e6f0fe; margin-top: 3px; white-space: nowrap; }}
.vital-val small {{ font-size: 0.55rem; color: var(--dim); font-weight: 500; margin-left: 2px; }}
.vital.abn {{ border-color: rgba(244, 63, 94, 0.38); background: rgba(244, 63, 94, 0.07); }}
.vital.abn .vital-val {{ color: #fecdd3; }}
.vital.warn {{ border-color: rgba(251, 146, 60, 0.34); background: rgba(251, 146, 60, 0.06); }}
.vital.warn .vital-val {{ color: #fed7aa; }}

.pcard-labs {{ display: flex; align-items: center; gap: 0.55rem; margin-top: 0.6rem; flex-wrap: wrap; }}
.lab-pill {{ display: inline-flex; align-items: baseline; gap: 4px; font-size: 0.62rem; color: var(--muted); }}
.lab-pill b {{ font-family: var(--mono); font-size: 0.72rem; color: #cfe0f5; font-weight: 600; }}
.lab-pill.hi b {{ color: #fda4af; }}
.lab-pill.mid b {{ color: #fde68a; }}

.pcard-foot {{ display: flex; align-items: center; justify-content: space-between; gap: 0.6rem; margin-top: 0.66rem; padding-top: 0.6rem; border-top: 1px dashed rgba(31, 48, 73, 0.8); }}
.completeness {{ display: flex; align-items: center; gap: 0.45rem; font-size: 0.62rem; color: var(--muted); }}
.progress {{ width: 74px; height: 4px; border-radius: 4px; background: rgba(255, 255, 255, 0.07); overflow: hidden; }}
.progress > span {{ display: block; height: 100%; border-radius: 4px; background: linear-gradient(90deg, #22d3ee, #34d399); }}
.open-hint {{ font-size: 0.64rem; color: var(--dim); display: inline-flex; align-items: center; gap: 3px; }}

/* overlay click target over a keyed card (patient cards / KPI cards) */
div[class*="st-key-pcard_"], div[class*="st-key-kpi_"] {{ position: relative; }}
div[class*="st-key-pcard_"]:hover .pcard,
div[class*="st-key-kpi_"]:hover .kpi {{
  transform: translateY(-2px);
  border-color: var(--risk-border, rgba(34, 211, 238, 0.4));
}}
div[class*="st-key-pc_open_"], div[class*="st-key-kpi_open_"] {{
  position: absolute !important;
  inset: 0;
  margin: 0 !important;
  padding: 0 !important;
  z-index: 6;
  width: auto !important;
  height: auto !important;
}}
div[class*="st-key-pc_open_"] .stButton,
div[class*="st-key-kpi_open_"] .stButton {{ width: 100%; height: 100%; }}
div[class*="st-key-pc_open_"] button,
div[class*="st-key-kpi_open_"] button {{
  width: 100% !important;
  height: 100% !important;
  min-height: 100% !important;
  opacity: 0 !important;
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
  transform: none !important;
  cursor: pointer;
}}
div[class*="st-key-pc_open_"] button:focus-visible,
div[class*="st-key-kpi_open_"] button:focus-visible {{
  opacity: 1 !important;
  outline: 2px solid rgba(34, 211, 238, 0.85);
  outline-offset: 2px;
  background: rgba(34, 211, 238, 0.06) !important;
}}
div[class*="st-key-pc_open_"] button p,
div[class*="st-key-kpi_open_"] button p {{ font-size: 0; }}

/* generic clickable row overlay */
div[class*="st-key-click_"] {{ position: relative; }}
div[class*="st-key-click_"]:hover .queue-row,
div[class*="st-key-click_"]:hover .feed-item {{ background: rgba(14, 25, 42, 0.85); border-color: var(--border-strong); }}
div[class*="st-key-cbtn_"] {{
  position: absolute !important;
  inset: 0;
  margin: 0 !important;
  padding: 0 !important;
  z-index: 6;
  width: auto !important;
  height: auto !important;
}}
div[class*="st-key-cbtn_"] .stButton {{ width: 100%; height: 100%; }}
div[class*="st-key-cbtn_"] button {{
  width: 100% !important; height: 100% !important; min-height: 100% !important;
  opacity: 0 !important; background: transparent !important; border: none !important;
  box-shadow: none !important; transform: none !important; cursor: pointer;
}}
div[class*="st-key-cbtn_"] button p {{ font-size: 0; }}
div[class*="st-key-cbtn_"] button:focus-visible {{
  opacity: 1 !important;
  outline: 2px solid rgba(34, 211, 238, 0.8);
  outline-offset: 1px;
}}

/* filter toolbars */
div[class*="st-key-toolbar_"] {{
  background: linear-gradient(180deg, rgba(13, 24, 38, 0.8), rgba(10, 19, 33, 0.72));
  border: 1px solid rgba(31, 48, 73, 0.85);
  border-radius: 14px;
  padding: 0.72rem 0.9rem 0.55rem;
  margin-bottom: 0.85rem;
}}
div[class*="st-key-toolbar_"] [data-testid="stWidgetLabel"] p {{ font-size: 0.61rem !important; }}

/* keyed panel containers */
div[class*="st-key-panel_"] {{
  background: linear-gradient(180deg, rgba(15, 26, 43, 0.92) 0%, rgba(10, 19, 33, 0.86) 100%);
  border: 1px solid rgba(31, 48, 73, 0.9);
  border-radius: var(--radius);
  padding: 1rem 1.05rem 1.05rem;
  box-shadow: var(--shadow-card);
}}
div[class*="st-key-panel_"] > div[data-testid="stVerticalBlock"] {{ gap: 0.55rem; }}

/* risk intelligence feed */
.feed {{ display: flex; flex-direction: column; gap: 0.5rem; }}
.feed-item {{
  display: flex; gap: 0.6rem; align-items: flex-start;
  padding: 0.55rem 0.6rem; border-radius: 10px;
  border: 1px solid rgba(31, 48, 73, 0.8);
  background: rgba(9, 17, 30, 0.6);
  transition: border-color 0.15s ease, background 0.15s ease;
}}
.feed-item:hover {{ border-color: var(--border-strong); background: rgba(14, 25, 42, 0.8); }}
.feed-ico {{ width: 26px; height: 26px; border-radius: 8px; display: flex; align-items: center; justify-content: center; flex: 0 0 auto; }}
.feed-title {{ font-size: 0.74rem; font-weight: 600; color: #dde9fa; line-height: 1.3; }}
.feed-meta {{ font-size: 0.63rem; color: var(--dim); margin-top: 2px; display: flex; align-items: center; gap: 6px; flex-wrap: wrap; }}
.feed-meta .pid {{ font-family: var(--mono); color: #9fb6d4; }}
.feed-time {{ margin-left: auto; font-size: 0.6rem; color: var(--dim); white-space: nowrap; font-variant-numeric: tabular-nums; }}

.queue-row {{
  display: flex; align-items: center; gap: 0.6rem;
  padding: 0.5rem 0.55rem; border-radius: 10px;
  border: 1px solid transparent; transition: background 0.15s ease, border-color 0.15s ease;
}}
.queue-row:hover {{ background: rgba(14, 25, 42, 0.8); border-color: rgba(31, 48, 73, 0.9); }}
.rank {{ width: 20px; text-align: center; font-family: var(--mono); font-size: 0.66rem; color: var(--dim); font-weight: 600; }}
.queue-id {{ font-family: var(--mono); font-size: 0.76rem; color: #e3edfb; font-weight: 600; }}
.queue-sub {{ font-size: 0.6rem; color: var(--dim); margin-top: 1px; }}
.queue-bar {{ flex: 1; height: 5px; border-radius: 4px; background: rgba(255, 255, 255, 0.06); overflow: hidden; }}
.queue-bar > span {{ display: block; height: 100%; border-radius: 4px; }}
.queue-prob {{ font-family: var(--mono); font-size: 0.72rem; font-weight: 600; width: 44px; text-align: right; }}

/* stat rows inside panels */
.stat-row {{ display: flex; align-items: center; justify-content: space-between; gap: 0.6rem; padding: 0.42rem 0; border-bottom: 1px dashed rgba(31, 48, 73, 0.65); }}
.stat-row:last-child {{ border-bottom: none; }}
.stat-k {{ font-size: 0.7rem; color: var(--muted); display: flex; align-items: center; gap: 6px; }}
.stat-v {{ font-family: var(--mono); font-size: 0.8rem; font-weight: 600; color: #e6f0fe; }}

/* legend */
.legend {{ display: flex; align-items: center; gap: 0.75rem; flex-wrap: wrap; }}
.legend-item {{ display: inline-flex; align-items: center; gap: 5px; font-size: 0.64rem; color: var(--muted); }}
.legend-swatch {{ width: 8px; height: 8px; border-radius: 2px; }}

/* empty state */
.empty {{
  border: 1px dashed var(--border-strong);
  border-radius: 14px;
  padding: 1.7rem 1.2rem;
  text-align: center;
  background: rgba(9, 17, 30, 0.45);
}}
.empty-ico {{ color: var(--dim); margin-bottom: 0.5rem; }}
.empty-title {{ font-size: 0.84rem; font-weight: 650; color: #cfdcee; }}
.empty-body {{ font-size: 0.72rem; color: var(--dim); margin-top: 4px; line-height: 1.5; }}

/* alert detail card */
.alert-card {{
  border: 1px solid var(--border);
  border-left: 3px solid var(--alert-color, var(--border-strong));
  border-radius: 13px;
  background: linear-gradient(180deg, rgba(18, 32, 51, 0.85), rgba(11, 21, 36, 0.9));
  padding: 0.75rem 0.9rem;
}}
.alert-top {{ display: flex; align-items: flex-start; justify-content: space-between; gap: 0.8rem; }}
.alert-title {{ font-size: 0.82rem; font-weight: 650; color: #e9f2ff; line-height: 1.3; }}
.alert-desc {{ font-size: 0.7rem; color: var(--muted); margin-top: 4px; line-height: 1.5; }}
.alert-meta {{ display: flex; align-items: center; gap: 0.55rem; flex-wrap: wrap; margin-top: 0.5rem; font-size: 0.62rem; color: var(--dim); }}

/* misc */
.glass-strip {{
  display: flex; align-items: center; gap: 0.7rem; flex-wrap: wrap;
  padding: 0.55rem 0.8rem;
  border: 1px solid rgba(34, 211, 238, 0.18);
  background: linear-gradient(90deg, rgba(34, 211, 238, 0.08), rgba(59, 130, 246, 0.04) 55%, transparent);
  border-radius: 11px;
  font-size: 0.7rem; color: var(--muted);
}}
.glass-strip b {{ color: #dff7ff; font-weight: 600; }}
.demo-banner {{
  display: flex; align-items: flex-start; gap: 0.65rem;
  padding: 0.65rem 0.85rem;
  border-radius: 12px;
  border: 1px solid rgba(250, 204, 21, 0.26);
  background: linear-gradient(90deg, rgba(250, 204, 21, 0.09), rgba(250, 204, 21, 0.02) 70%);
  color: #fde68a;
  font-size: 0.7rem;
  line-height: 1.5;
}}
.demo-banner b {{ color: #fef3c7; letter-spacing: 0.04em; }}
.divider {{ height: 1px; background: var(--border); margin: 0.9rem 0; }}
.hr-space {{ height: 0.9rem; }}
.mono {{ font-family: var(--mono); }}
.muted {{ color: var(--muted); }}
.dim {{ color: var(--dim); }}
.text-sm {{ font-size: 0.74rem; }}
.text-xs {{ font-size: 0.66rem; }}
.nowrap {{ white-space: nowrap; }}
.right {{ text-align: right; }}

/* metric table */
.mtable {{ width: 100%; border-collapse: separate; border-spacing: 0; font-size: 0.74rem; }}
.mtable th {{
  text-align: left; font-size: 0.6rem; letter-spacing: 0.1em; text-transform: uppercase;
  color: var(--dim); font-weight: 700; padding: 0.5rem 0.6rem; border-bottom: 1px solid var(--border);
  white-space: nowrap;
}}
.mtable td {{ padding: 0.55rem 0.6rem; border-bottom: 1px solid rgba(27, 42, 66, 0.6); color: #dbe7f8; }}
.mtable tr:last-child td {{ border-bottom: none; }}
.mtable tr:hover td {{ background: rgba(34, 211, 238, 0.035); }}
.mtable td.num {{ font-family: var(--mono); text-align: right; font-variant-numeric: tabular-nums; }}

/* ----------------------------------------------------------- misc chrome */
.st-key-topnav [data-testid="stForm"] {{ border: none; padding: 0; background: transparent; }}
.st-key-topnav [data-testid="stFormSubmitButton"] button {{ min-width: 40px; padding: 0 6px; }}
.st-key-topnav [data-testid="stFormSubmitButton"] button p {{ font-size: 0.72rem; white-space: nowrap; }}
.st-key-topnav [data-testid="stPopover"] button p {{ font-size: 0.72rem; white-space: nowrap; }}
.st-key-topnav [data-testid="stPopover"] button {{ padding: 0 8px; gap: 5px; }}

.st-key-bellwrap {{ position: relative; }}
.bell-badge {{
  position: absolute; top: -4px; right: -4px; z-index: 20;
  min-width: 17px; height: 17px; padding: 0 4px; border-radius: 999px;
  background: var(--critical); color: #fff; font-size: 0.56rem; font-weight: 700;
  display: inline-flex; align-items: center; justify-content: center;
  border: 2px solid #081120; box-shadow: 0 0 12px rgba(244, 63, 94, 0.55);
}}

.pop-head {{ display: flex; align-items: center; justify-content: space-between; gap: .5rem;
  font-size: .76rem; font-weight: 650; color: #e8f2ff; padding-bottom: .45rem;
  border-bottom: 1px solid var(--border); margin-bottom: .5rem; }}
.pop-head span {{ display: inline-flex; align-items: center; gap: .4rem; }}
.pop-head b {{ font-family: var(--mono); color: var(--cyan); }}
.pop-body {{ font-size: .72rem; color: var(--muted); line-height: 1.5; margin-bottom: .3rem;
  white-space: normal; overflow-wrap: anywhere; }}
.pop-body.dim {{ color: var(--dim); }}
.pop-empty {{ font-size: .72rem; color: var(--dim); padding: .5rem 0; }}
.st-key-profile_menu {{ width: 264px; }}
.pop-profile-title {{ font-size: .74rem; font-weight: 600; color: var(--text); margin: .1rem 0 .45rem; }}
.pop-profile-row {{ display: flex; align-items: baseline; justify-content: space-between; gap: .7rem;
  padding: .26rem 0; border-bottom: 1px dashed rgba(58, 84, 122, 0.35); }}
.pop-profile-row:last-of-type {{ border-bottom: none; padding-bottom: .1rem; }}
.pop-profile-row span {{ font-size: .68rem; color: var(--dim); flex: 0 0 auto; }}
.pop-profile-row b {{ font-size: .7rem; font-weight: 600; color: #dce9fa; text-align: right;
  overflow-wrap: anywhere; }}

.panel-note {{ font-size: .62rem; color: var(--dim); margin-top: .5rem; line-height: 1.45; }}
.pager-info {{ text-align: center; font-size: .72rem; color: var(--muted); padding-top: .35rem; }}
.pager-info b {{ color: #e6effc; }}
.pager-info span {{ color: var(--dim); }}

.xai-context {{
  display: flex; align-items: center; gap: .45rem; flex-wrap: wrap; justify-content: flex-end;
  font-size: .7rem; color: var(--muted); padding-top: .6rem;
}}
.xai-context b {{ color: #eef6ff; font-family: var(--mono); }}

.method-grid {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: .55rem .9rem; }}
.method-item {{ display: flex; flex-direction: column; gap: .15rem; padding: .45rem .55rem;
  border: 1px solid rgba(31,48,73,.7); border-radius: 9px; background: rgba(9,17,30,.45); }}
.method-item span {{ font-size: .58rem; letter-spacing: .1em; text-transform: uppercase; color: var(--dim); font-weight: 700; }}
.method-item b {{ font-size: .72rem; color: #dbe7f8; font-weight: 600; }}

/* analytics stat tiles */
.astat {{
  background: linear-gradient(180deg, rgba(15,26,43,.92), rgba(10,19,33,.86));
  border: 1px solid rgba(31,48,73,.9); border-radius: 14px; padding: .8rem .9rem .75rem;
  box-shadow: var(--shadow-card); min-height: 104px;
}}
.astat-top {{ display: flex; align-items: center; justify-content: space-between; gap: .5rem;
  font-size: .62rem; letter-spacing: .09em; text-transform: uppercase; color: var(--muted); font-weight: 650; }}
.astat-top svg {{ color: var(--astat-color); }}
.astat-value {{ font-size: 1.55rem; font-weight: 750; letter-spacing: -.02em; color: #f2f8ff; margin-top: .3rem; line-height: 1; }}
.astat-sub {{ font-size: .62rem; color: var(--dim); margin-top: .35rem; line-height: 1.35; }}

/* report cards */
.rep-desc {{ font-size: .72rem; color: var(--muted); line-height: 1.55; margin-bottom: .3rem; }}

/* patient detail */
.pd-vitals {{ grid-template-columns: repeat(6, minmax(0, 1fr)); margin: 0.2rem 0 1rem; }}
.pd-vitals .vital {{ padding: 0.62rem 0.7rem; min-height: 66px; }}
.pd-vitals .vital-val {{ font-size: 1.02rem; }}
.pd-note {{ display: flex; align-items: center; gap: .45rem; font-size: .66rem; color: var(--dim);
  padding-top: .5rem; line-height: 1.4; }}
.risk-hero {{
  border: 1px solid var(--risk-border, var(--border)); border-radius: 13px;
  background: linear-gradient(150deg, var(--risk-soft, rgba(34,211,238,.08)), rgba(9,17,30,.35));
  padding: .85rem .95rem;
}}
.risk-hero-top {{ display: flex; align-items: center; justify-content: space-between; gap: .6rem;
  font-size: .68rem; letter-spacing: .06em; text-transform: uppercase; color: var(--muted); font-weight: 650; }}
.risk-hero-value {{ font-family: var(--mono); font-size: 2rem; font-weight: 750; color: var(--risk-color);
  margin-top: .35rem; line-height: 1.05; display: flex; align-items: baseline; gap: .55rem; }}
.risk-hero-value span {{ font-family: var(--font); font-size: .64rem; font-weight: 500; color: var(--dim); letter-spacing: 0; }}
.risk-hero-bar {{ height: 6px; border-radius: 6px; background: rgba(255,255,255,.07); overflow: hidden; margin-top: .55rem; }}
.risk-hero-bar span {{ display: block; height: 100%; border-radius: 6px; background: linear-gradient(90deg, var(--risk-color)88, var(--risk-color)); }}
.risk-hero-foot {{ font-size: .6rem; color: var(--dim); margin-top: .5rem; line-height: 1.45; }}

.ctx-grid {{ display: flex; flex-direction: column; gap: .1rem; }}
.ctx-complete {{ display: flex; align-items: center; gap: .55rem; margin-top: .7rem; padding-top: .65rem;
  border-top: 1px dashed rgba(31,48,73,.8); font-size: .66rem; color: var(--muted); }}
.ctx-complete b {{ color: var(--text); font-family: var(--mono); margin-left: auto; }}
.shap-lists {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1.1rem; margin-top: .7rem; }}
.shap-list-title {{ font-size: .6rem; letter-spacing: .12em; text-transform: uppercase; color: var(--dim); font-weight: 700; margin-bottom: .35rem; }}
.shap-row {{ display: flex; justify-content: space-between; gap: .7rem; font-size: .7rem; color: #cfdcee;
  padding: .28rem 0; border-bottom: 1px solid rgba(27,42,66,.55); }}
.shap-row b {{ font-family: var(--mono); font-weight: 600; }}

/* nav small label */
.side-version {{ font-size: 0.6rem; color: var(--dim); margin-top: 0.6rem; display: flex; align-items: center; gap: 5px; }}

/* responsive */
@media (max-width: 1280px) {{
  [data-testid="stMainBlockContainer"], .block-container {{ padding-left: 1.15rem !important; padding-right: 1.15rem !important; }}
  .kpi-value {{ font-size: 1.8rem; }}
}}
@media (max-width: 900px) {{
  .st-key-topnav {{ padding: 0 0.7rem !important; }}
  .brand-sub {{ display: none; }}
  .kpi {{ min-height: 118px; }}
  [data-testid="stMain"] {{ margin-left: 0 !important; width: 100% !important; }}
}}

@media (prefers-reduced-motion: reduce) {{
  *, *::before, *::after {{ animation-duration: 0.001ms !important; transition-duration: 0.001ms !important; }}
}}
"""


def inject_css(active_route: str) -> None:
    """Inject the stylesheet plus the active-nav rule for the current route."""
    active_rule = f"""
.st-key-nav_{active_route} button {{
  background: linear-gradient(90deg, rgba(34, 211, 238, 0.16) 0%, rgba(59, 130, 246, 0.07) 70%, transparent 100%) !important;
  border-color: rgba(34, 211, 238, 0.26) !important;
  color: #f2fdff !important;
  box-shadow: inset 3px 0 0 0 var(--cyan), 0 8px 22px -16px rgba(34, 211, 238, 0.6);
  font-weight: 650 !important;
}}
.st-key-nav_{active_route} button [data-testid="stIconMaterial"] {{ color: var(--cyan) !important; }}
"""
    st.markdown(f"<style>{_CSS}{active_rule}</style>", unsafe_allow_html=True)
