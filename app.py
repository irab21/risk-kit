"""
Risk Kit Dashboard: portfolio risk explained in plain English.

Run locally:   streamlit run app.py
All calculations come from risk_kit.py; this file only handles inputs, charts and explanations.
"""
import io
import os
import datetime as dt

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import norm

import advisor
import market
import commentary as cmt
import holdings_import as hi
import mutual_funds as mf
import peers
import stocks
import transactions as txm
import risk_kit as rk

st.set_page_config(page_title="Risk Kit Dashboard", page_icon=":material/monitoring:", layout="wide")

# -----------------------------------------------------------------------------
# Colors (validated categorical palette; status colors are reserved for good/bad)
# -----------------------------------------------------------------------------
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6")
SERIES = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET]
GOOD, CRITICAL = "#0ca30c", "#d03b3b"
INK, INK_2, MUTED, GRID = "#0f172a", "#4a5468", "#8a93a6", "#eeebe4"
SURFACE = "#ffffff"
NAVY, BORDER = "#0c1a3a", "#e7e2d8"
NAVY_2, GOLD, GOLD_SOFT, IVORY = "#1b3270", "#b98a2f", "#f4ead3", "#f6f4ef"
POS, NEG = "#0f7a4f", "#b42318"
FONT, SERIF = "'Plus Jakarta Sans', -apple-system, 'Segoe UI', Roboto, sans-serif", "'Fraunces', Georgia, serif"

PRESETS = {
    "India · Balanced mutual funds": {"MF:120716": 35, "MF:122639": 25, "MF:118989": 10, "MF:118663": 10,
                                      "MF:120692": 20},
    "India · All-equity mutual funds": {"MF:120716": 40, "MF:122639": 35, "MF:118989": 25},
    "India · Blue-chip stocks": {"RELIANCE.NS": 25, "HDFCBANK.NS": 25, "INFY.NS": 20, "TCS.NS": 15, "ITC.NS": 15},
    "US · Diversified (stocks, bonds, gold)": {"SPY": 40, "QQQ": 20, "AGG": 30, "GLD": 10},
    "US · Classic 60/40": {"SPY": 60, "AGG": 40},
    "US · Tech heavy": {"QQQ": 50, "AAPL": 15, "MSFT": 15, "NVDA": 10, "AGG": 10},
}
PRESET_NAMES = {"MF:120716": "UTI Nifty 50 Index", "MF:122639": "Parag Parikh Flexi Cap",
                "MF:118989": "HDFC Mid Cap", "MF:118663": "Nippon India Gold Savings",
                "MF:120692": "ICICI Pru Corporate Bond"}

st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600;9..144,700&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
  .stApp, .stMarkdown, .stMarkdown p, .stMarkdown li, label, input, textarea, button p,
  [data-testid="stMetricValue"], [data-testid="stMetricLabel"] p, [data-baseweb="tab"] p, [data-testid="stCaptionContainer"] p
    {{ font-family: {FONT} !important; }}
  h1, h2, h3, h4, h5, .serif {{ font-family: {SERIF} !important; }}
  .stApp {{ background: {IVORY}; }}
  [data-testid="stMetricValue"], .num {{ font-variant-numeric: tabular-nums; }}
  .block-container {{ padding-top: 1.4rem; max-width: 1320px; }}
  header[data-testid="stHeader"] {{ background: transparent; }}
  [data-testid="stAppDeployButton"], [data-testid="stMainMenu"], #MainMenu, footer {{ display: none !important; }}
  [data-testid="stHeaderActionElements"] {{ display: none; }}
  [data-testid="stSidebar"] {{ min-width: 360px; max-width: 360px; background: #fff; border-right: 1px solid {BORDER}; }}
  .stMarkdown p, .stMarkdown li {{ color: {INK_2}; line-height: 1.6; }}
  .stMarkdown strong, .stMarkdown b {{ color: {INK}; }}
  h3 {{ font-size: 1.35rem !important; font-weight: 600 !important; color: {INK} !important; letter-spacing: -.01em;
        margin-top: 1.3rem !important; }}
  h5 {{ font-size: 1.15rem !important; font-weight: 600 !important; color: {INK} !important; }}

  /* header */
  .hero {{ position: relative; overflow: hidden; border-radius: 18px; padding: 26px 30px; margin-bottom: 14px;
           background: radial-gradient(120% 140% at 100% 0%, rgba(185,138,47,.30) 0%, rgba(185,138,47,0) 45%),
                       linear-gradient(125deg, {NAVY} 0%, #132657 60%, {NAVY_2} 100%);
           box-shadow: 0 10px 30px -12px rgba(12,26,58,.45);
           display: flex; flex-wrap: wrap; gap: 20px 36px; justify-content: space-between; align-items: flex-end; }}
  .hero .eyebrow {{ font-size: .7rem; letter-spacing: .2em; text-transform: uppercase; color: #e2c27d; font-weight: 700; }}
  .hero .title {{ font-family: {SERIF}; color: #fff; font-size: 2rem; font-weight: 600; margin: .3rem 0 .35rem;
                  letter-spacing: -.01em; line-height: 1.15; }}
  .hero .sub {{ color: #c3cde3; font-size: .9rem; }}
  .hero .kpis {{ display: flex; flex-wrap: wrap; background: rgba(255,255,255,.06); border: 1px solid rgba(255,255,255,.12);
                 border-radius: 14px; padding: 10px 4px; }}
  .hero .kpi {{ padding: 2px 18px; border-left: 1px solid rgba(255,255,255,.12); }}
  .hero .kpi:first-child {{ border-left: none; }}
  .hero .kpi .l {{ font-size: .66rem; text-transform: uppercase; letter-spacing: .12em; color: #9fb0d6; font-weight: 700; }}
  .hero .kpi .v {{ font-size: 1.15rem; font-weight: 700; color: #fff; margin-top: 3px; font-variant-numeric: tabular-nums; }}
  .hero .kpi .v.pos {{ color: #86e0b0; }} .hero .kpi .v.neg {{ color: #ffb1a3; }}

  /* tabs */
  .stTabs [data-baseweb="tab-list"] {{ gap: 4px; background: #fff; border: 1px solid {BORDER}; border-radius: 14px;
                                       padding: 5px; flex-wrap: wrap; row-gap: 4px; box-shadow: 0 1px 2px rgba(15,23,42,.04); }}
  .stTabs [data-baseweb="tab"] {{ height: 38px; padding: 0 12px; border-radius: 10px; background: transparent; }}
  .stTabs [data-baseweb="tab"] p {{ font-weight: 600; font-size: .84rem; color: {INK_2}; }}
  .stTabs [data-baseweb="tab"]:hover {{ background: {IVORY}; }}
  .stTabs [aria-selected="true"] {{ background: {NAVY} !important; }}
  .stTabs [aria-selected="true"] p {{ color: #fff !important; }}
  .stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] {{ display: none; }}

  /* cards */
  [data-testid="stMetric"] {{ background: #fff; border: 1px solid {BORDER}; border-radius: 14px; padding: 14px 16px;
                              box-shadow: 0 1px 2px rgba(15,23,42,.04); }}
  [data-testid="stMetricLabel"] p {{ font-size: .74rem !important; text-transform: uppercase; letter-spacing: .06em;
                                     color: {INK_2} !important; font-weight: 600 !important; }}
  [data-testid="stMetricValue"] {{ font-size: 1.55rem; font-weight: 700; color: {INK}; }}
  [data-testid="stPlotlyChart"] {{ background: #fff; border: 1px solid {BORDER}; border-radius: 16px; padding: 8px 10px 4px;
                                   box-shadow: 0 1px 2px rgba(15,23,42,.04); }}
  [data-testid="stExpander"] details {{ background: #fff; border: 1px solid {BORDER}; border-radius: 14px; }}
  [data-testid="stExpander"] summary p {{ font-weight: 600; color: {INK}; }}
  [data-testid="stDataFrame"] {{ border: 1px solid {BORDER}; border-radius: 12px; overflow: hidden; }}
  .chart-title {{ font-family: {SERIF}; font-weight: 600; font-size: 1.12rem; color: {INK}; margin: 1.2rem 0 .5rem; }}
  .small-note {{ color: {MUTED}; font-size: .84rem; }}

  .headline {{ background: #fff; border: 1px solid {BORDER}; border-radius: 14px; padding: 18px 20px; margin: .3rem 0 .6rem;
               font-size: 1.04rem; line-height: 1.65; color: {INK}; }}
  .grade {{ display: inline-block; font-weight: 700; font-size: .82rem; padding: .22rem .7rem; border-radius: 999px;
            margin-bottom: .55rem; letter-spacing: .02em; }}
  .grade.good {{ background: #e7f6ec; color: #0a6b2c; }} .grade.ok {{ background: #fdf3dc; color: #7a5300; }}
  .grade.bad {{ background: #fdecea; color: #a1281e; }}
  .point {{ background: #fff; border: 1px solid {BORDER}; border-radius: 12px; padding: 14px 16px; margin-bottom: 12px;
            line-height: 1.55; color: {INK_2}; font-size: .93rem; }}
  .point b {{ display: flex; align-items: center; gap: 8px; margin-bottom: .3rem; color: {INK}; font-size: .97rem; }}
  .point b::before {{ content: ""; width: 9px; height: 9px; border-radius: 50%; flex: none; background: var(--dot); }}
  .point.good {{ --dot: {GOOD}; }} .point.ok {{ --dot: #eda100; }} .point.bad {{ --dot: {CRITICAL}; }}
  .case {{ background: #fff; border: 1px solid {BORDER}; border-top: 4px solid var(--c); border-radius: 12px;
           padding: 16px 18px; line-height: 1.6; color: {INK_2}; font-size: .92rem; }}
  .case.good {{ --c: {GOOD}; }} .case.mid {{ --c: {BLUE}; }} .case.bad {{ --c: {CRITICAL}; }}
  .case .lbl {{ font-size: .72rem; text-transform: uppercase; letter-spacing: .08em; font-weight: 700; color: {INK_2}; }}
  .case .big {{ display: block; font-size: 1.6rem; font-weight: 800; color: {INK}; margin: 2px 0; letter-spacing: -.01em; }}

  .prio {{ display: inline-block; font-size: .68rem; font-weight: 700; letter-spacing: .06em; text-transform: uppercase;
           padding: .14rem .5rem; border-radius: 5px; }}
  .prio.High {{ background: #fdecea; color: #a1281e; }} .prio.Medium {{ background: #fdf3dc; color: #7a5300; }}
  .prio.Low {{ background: #eef2f7; color: #344054; }}
  .action {{ background: #fff; border: 1px solid {BORDER}; border-radius: 12px; padding: 14px 18px; margin-bottom: 10px; }}
  .action .head {{ display: flex; align-items: center; gap: 10px; margin-bottom: 4px; }}
  .action .t {{ font-weight: 700; color: {INK}; font-size: 1rem; }}
  .action .why {{ color: {INK_2}; font-size: .92rem; }}
  .action .how {{ color: {INK}; font-size: .92rem; margin-top: 4px; }}
  .action.done {{ opacity: .55; }} .action.done .t {{ text-decoration: line-through; }}
  .action .impact {{ margin-left: auto; font-size: .78rem; font-weight: 700; color: #05603a; background: #ecfdf3;
                     border-radius: 999px; padding: 2px 10px; white-space: nowrap; }}
  .when {{ font-size: .78rem; font-weight: 700; letter-spacing: .07em; text-transform: uppercase; color: {INK_2};
           margin: 18px 0 8px; border-bottom: 1px solid {BORDER}; padding-bottom: 4px; }}
  .action .how::before {{ content: "→ "; color: #1d4ed8; font-weight: 700; }}
  .side-title {{ font-family: {SERIF}; font-size: 1.45rem; font-weight: 600; color: {INK}; letter-spacing: -.01em;
                 margin: .2rem 0 .4rem; }}
  .side-brand {{ font-size: .68rem; letter-spacing: .2em; text-transform: uppercase; color: {GOLD}; font-weight: 800; }}
  .panel {{ background: #fff; border: 1px solid {BORDER}; border-radius: 14px; padding: 16px 18px 6px; height: 100%; }}
  .panel .ph {{ font-size: .74rem; text-transform: uppercase; letter-spacing: .1em; font-weight: 700; margin-bottom: 8px; }}
  .panel.good .ph {{ color: #0a6b2c; }} .panel.bad .ph {{ color: #a1281e; }}
  .panel ul {{ list-style: none; margin: 0; padding: 0; }}
  .panel li {{ margin: 0 0 12px; padding-left: 16px; position: relative; line-height: 1.5; }}
  .panel li::before {{ content: ""; position: absolute; left: 0; top: .5em; width: 7px; height: 7px; border-radius: 50%; }}
  .panel.good li::before {{ background: {GOOD}; }} .panel.bad li::before {{ background: #eda100; }}
  .panel li b {{ display: block; color: {INK}; font-size: .95rem; }}
  .panel li span {{ color: {INK_2}; font-size: .88rem; }}
  .nextstep {{ display: flex; flex-wrap: wrap; align-items: baseline; gap: 4px 12px; background: #fff; border: 1px solid {BORDER};
               border-radius: 12px; padding: 12px 16px; margin-bottom: 8px; }}
  .nextstep .t {{ font-weight: 700; color: {INK}; }}
  .nextstep .how {{ color: {INK_2}; font-size: .9rem; flex-basis: 100%; }}
  .stTabs .stTabs [data-baseweb="tab-list"] {{ background: transparent; border: none; padding: 0; gap: 6px;
                                              box-shadow: none; margin-bottom: 6px; }}
  .stTabs .stTabs [data-baseweb="tab"] {{ height: 32px; border: 1px solid {BORDER}; background: #fff; border-radius: 999px;
                                         padding: 0 14px; }}
  .stTabs .stTabs [aria-selected="true"] {{ background: {GOLD_SOFT} !important; border-color: {GOLD}; }}
  .stTabs .stTabs [aria-selected="true"] p {{ color: {NAVY} !important; }}
  .keep {{ display: inline-block; background: #e7f6ec; color: #0a6b2c; border-radius: 999px; padding: .2rem .7rem;
           font-size: .84rem; font-weight: 600; margin: 0 6px 6px 0; }}

  .mood {{ background: #fff; border: 1px solid {BORDER}; border-radius: 14px; padding: 18px 20px; margin-bottom: 14px;
           display: grid; grid-template-columns: minmax(260px, 1.3fr) 2fr; gap: 18px; }}
  @media (max-width: 800px) {{ .mood {{ grid-template-columns: 1fr; }} }}
  .mood .eyebrow {{ font-size: .7rem; text-transform: uppercase; letter-spacing: .12em; font-weight: 700; color: {MUTED}; }}
  .mood .label {{ font-size: 1.9rem; font-weight: 800; letter-spacing: -.02em; margin: 2px 0 6px; }}
  .mood .label.good {{ color: #0a6b2c; }} .mood .label.ok {{ color: #8a5d00; }} .mood .label.bad {{ color: #a1281e; }}
  .mood ul {{ margin: 0; padding-left: 1.1rem; color: {INK_2}; font-size: .9rem; line-height: 1.6; }}
  .tiles {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; align-content: start; }}
  .tile {{ background: #f8f9fb; border: 1px solid {BORDER}; border-radius: 10px; padding: 10px 12px; }}
  .tile .l {{ font-size: .68rem; text-transform: uppercase; letter-spacing: .08em; color: {MUTED}; font-weight: 700; }}
  .tile .v {{ font-size: 1.05rem; font-weight: 700; color: {INK}; margin-top: 2px; }}
  .tile .d {{ font-size: .8rem; color: {INK_2}; }}
  .means {{ grid-column: 1 / -1; border-top: 1px solid {BORDER}; padding-top: 12px; }}
  .means b {{ color: {INK}; }}
  .backdrop {{ background: #fff; border: 1px solid {BORDER}; border-left: 4px solid var(--c); border-radius: 0 12px 12px 0;
               padding: 12px 16px; margin: 4px 0 14px; color: {INK_2}; font-size: .92rem; }}
  .backdrop.good {{ --c: {GOOD}; }} .backdrop.ok {{ --c: #eda100; }} .backdrop.bad {{ --c: {CRITICAL}; }}

  /* summary */
  .sec {{ margin: 1.6rem 0 .7rem; display: flex; align-items: baseline; justify-content: space-between; gap: 12px; flex-wrap: wrap; }}
  .sec .t {{ font-family: {SERIF}; font-size: 1.35rem; font-weight: 600; color: {INK}; letter-spacing: -.01em; }}
  .sec .s {{ color: {MUTED}; font-size: .85rem; }}
  .verdict {{ display: grid; grid-template-columns: minmax(220px, 280px) 1fr; gap: 0; background: #fff; border: 1px solid {BORDER};
              border-radius: 18px; overflow: hidden; box-shadow: 0 1px 2px rgba(15,23,42,.04); margin: .2rem 0 1rem; }}
  .verdict .vl {{ padding: 22px 24px; background: linear-gradient(160deg, var(--soft) 0%, #fff 100%); border-right: 1px solid {BORDER}; }}
  .verdict.good {{ --c: {POS}; --soft: #e8f5ee; }} .verdict.ok {{ --c: #8a5d00; --soft: #fbf2dc; }}
  .verdict.bad {{ --c: {NEG}; --soft: #fdecea; }}
  .verdict .eyebrow {{ font-size: .68rem; letter-spacing: .16em; text-transform: uppercase; color: {INK_2}; font-weight: 700; }}
  .verdict .word {{ font-family: {SERIF}; font-size: 2.6rem; font-weight: 600; color: var(--c); line-height: 1.1; margin: 4px 0 12px; }}
  .meter {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 5px; }}
  .meter span {{ height: 7px; border-radius: 99px; background: #e9e5dc; }}
  .meter span.on {{ background: var(--c); }}
  .meter-l {{ display: flex; justify-content: space-between; font-size: .7rem; color: {MUTED}; margin-top: 5px; font-weight: 600; }}
  .verdict .vr {{ padding: 22px 26px; display: flex; flex-direction: column; justify-content: center; gap: 14px; }}
  .verdict .vr p {{ margin: 0; font-size: 1.06rem; line-height: 1.65; color: {INK}; }}
  .tally {{ display: flex; gap: 8px; flex-wrap: wrap; }}
  .tally span {{ font-size: .8rem; font-weight: 700; padding: 4px 11px; border-radius: 99px; }}
  .tally .good {{ background: #e8f5ee; color: {POS}; }} .tally .ok {{ background: #fbf2dc; color: #7a5300; }}
  .tally .bad {{ background: #fdecea; color: {NEG}; }}
  @media (max-width: 760px) {{ .verdict {{ grid-template-columns: 1fr; }} .verdict .vl {{ border-right: none; border-bottom: 1px solid {BORDER}; }} }}

  .stats {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: .4rem; }}
  @media (max-width: 900px) {{ .stats {{ grid-template-columns: repeat(2, 1fr); }} }}
  .stat {{ background: #fff; border: 1px solid {BORDER}; border-radius: 16px; padding: 16px 18px;
           box-shadow: 0 1px 2px rgba(15,23,42,.04); position: relative; }}
  .stat .l {{ font-size: .7rem; text-transform: uppercase; letter-spacing: .1em; color: {INK_2}; font-weight: 700; }}
  .stat .v {{ font-size: 1.75rem; font-weight: 800; color: {INK}; margin: 6px 0 8px; letter-spacing: -.02em;
              font-variant-numeric: tabular-nums; }}
  .stat .d {{ display: inline-block; font-size: .76rem; font-weight: 700; padding: 3px 9px; border-radius: 99px; }}
  .stat .d.up {{ background: #e8f5ee; color: {POS}; }} .stat .d.down {{ background: #fdecea; color: {NEG}; }}
  .stat .d.flat {{ background: {IVORY}; color: {INK_2}; }}
  .stat .f {{ font-size: .8rem; color: {MUTED}; margin-top: 8px; }}

  .duo {{ display: grid; grid-template-columns: 1fr 1fr; gap: 14px; align-items: start; }}
  @media (max-width: 900px) {{ .duo {{ grid-template-columns: 1fr; }} }}
  .duo .panel {{ padding: 18px 20px 8px; border-radius: 16px; }}
  .panel .ph {{ display: flex; align-items: center; gap: 8px; }}

  .steps3 {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 12px; }}
  .step {{ background: #fff; border: 1px solid {BORDER}; border-radius: 16px; padding: 16px 18px; display: flex;
           flex-direction: column; gap: 8px; box-shadow: 0 1px 2px rgba(15,23,42,.04); }}
  .step .top {{ display: flex; align-items: center; justify-content: space-between; }}
  .step .n {{ font-family: {SERIF}; font-size: 1.5rem; font-weight: 600; color: {GOLD}; line-height: 1; }}
  .step .t {{ font-weight: 700; color: {INK}; font-size: 1rem; line-height: 1.35; }}
  .step .how {{ color: {INK_2}; font-size: .88rem; line-height: 1.55; }}
  .more {{ color: {INK_2}; font-size: .86rem; margin: 10px 2px 0; }}
  .more b {{ color: {NAVY}; }}

  .obs-h {{ display: flex; align-items: center; gap: 8px; font-size: .74rem; letter-spacing: .1em; text-transform: uppercase;
            font-weight: 800; margin: 6px 0 10px; color: {INK_2}; }}
  .obs-h .c {{ background: {IVORY}; border-radius: 99px; padding: 1px 8px; font-size: .72rem; letter-spacing: 0; }}
  .obs {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 12px; margin-bottom: 14px; }}
  .obs .point {{ margin: 0; border-left: 4px solid var(--dot); border-radius: 12px; background: #fff; }}
  .obs .point b::before {{ display: none; }}
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Formatting helpers
# -----------------------------------------------------------------------------
def pct(x, d=1):
    return "–" if x is None or pd.isna(x) else f"{x*100:.{d}f}%"


CUR = "$"  # set after the holdings are known
CURRENCY_BY_SUFFIX = {".NS": "₹", ".BO": "₹", ".L": "£", ".DE": "€", ".PA": "€", ".AS": "€", ".MI": "€",
                      ".TO": "C$", ".AX": "A$", ".HK": "HK$", ".T": "¥"}


def _indian(n):
    """1234567 -> '12,34,567'."""
    s_ = f"{int(round(n))}"
    if len(s_) <= 3:
        return s_
    head, tail = s_[:-3], s_[-3:]
    groups = []
    while len(head) > 2:
        groups.insert(0, head[-2:])
        head = head[:-2]
    return ",".join(([head] if head else []) + groups) + "," + tail


def money(x, currency=None):
    if x is None or pd.isna(x):
        return "–"
    c = currency or CUR
    body = _indian(abs(x)) if c == "₹" else f"{abs(x):,.0f}"
    return f"{'-' if x < 0 else ''}{c}{body}"


def short_money(x, currency=None):
    """Compact amounts for headlines: ₹58.5 lakh, ₹1.2 crore, $3.4M."""
    if x is None or pd.isna(x):
        return "–"
    c, a, sign = currency or CUR, abs(x), "-" if x < 0 else ""
    if c == "₹":
        if a >= 1e7:
            return f"{sign}₹{a/1e7:.2f}".rstrip("0").rstrip(".") + " crore"
        if a >= 1e5:
            return f"{sign}₹{a/1e5:.1f}".removesuffix(".0") + " lakh"
        return money(x, c)
    if a >= 1e6:
        return f"{sign}{c}{a/1e6:.2f}M"
    if a >= 1e4:
        return f"{sign}{c}{a/1e3:.0f}K"
    return money(x, c)


def num(x, d=2):
    return "–" if x is None or pd.isna(x) else f"{x:.{d}f}"


def style_fig(fig, height=380, yfmt=None, xfmt=None, legend=True, cur=None):
    fig.update_layout(
        title=dict(text=""), height=height, margin=dict(l=8, r=8, t=48 if legend else 8, b=8),
        paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
        font=dict(family="Plus Jakarta Sans, -apple-system, Segoe UI, sans-serif", size=13, color=INK_2),
        hoverlabel=dict(bgcolor="white", font_size=13, font_color=INK, bordercolor=BORDER),
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, title=None,
                    bgcolor="rgba(0,0,0,0)",
                    font=dict(size=13, color=INK_2)),
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False, linecolor=GRID, tickformat=(xfmt or "").lstrip("$") or None,
                     tickprefix=(cur or CUR) if (xfmt or "").startswith("$") else None)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor=GRID, tickformat=(yfmt or "").lstrip("$") or None,
                     tickprefix=(cur or CUR) if (yfmt or "").startswith("$") else None)
    if (cur or CUR) == "₹" and (yfmt or "").startswith("$"):
        _inr_ticks(fig)
    return fig


def _inr_ticks(fig):
    """Rupee y-axis in lakh / crore (₹10 lakh, ₹1.5 crore) instead of ₹1,000,000."""
    ys = []
    for tr in fig.data:
        y = getattr(tr, "y", None)
        if y is not None and getattr(tr, "orientation", None) != "h":
            ys.extend(v for v in np.asarray(y, dtype=float).ravel() if np.isfinite(v))
    if not ys:
        return
    lo, hi = min(ys), max(ys)
    if hi - lo <= 0:
        return
    raw = (hi - lo) / 5
    mag = 10 ** np.floor(np.log10(raw))
    step = next(m_ * mag for m_ in (1, 2, 2.5, 5, 10) if m_ * mag >= raw)
    vals = np.arange(np.floor(lo / step) * step, hi + step, step)
    fig.update_yaxes(tickvals=vals, ticktext=[short_money(v, "₹") if abs(v) >= 1e5 else money(v, "₹") for v in vals],
                     tickprefix=None, tickformat=None)


def md(text, **kwargs):
    """st.markdown that shows $ literally (Streamlit otherwise reads $...$ as a formula)."""
    safe = "&#36;" if kwargs.get("unsafe_allow_html") else "\\$"
    st.markdown(text.replace("$", safe), **kwargs)


def explain(title, body):
    with st.expander(title, icon=":material/lightbulb:"):
        md(body)


def show(fig, title=None):
    if title:
        md(f"<div class='chart-title'>{title}</div>", unsafe_allow_html=True)
    st.plotly_chart(fig, width="stretch", config={"displaylogo": False})


# -----------------------------------------------------------------------------
# Cached calculations
# -----------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def load_report(tickers, weights, benchmark, start, end, rf, level):
    return rk.portfolio_report(list(tickers), list(weights), benchmark or None,
                               start=str(start), end=str(end), riskfree_rate=rf, level=level)


@st.cache_data(ttl=3600, show_spinner=False)
def optimizer_results(rets, rf, max_weight, shrink):
    er = rk.annualize_rets(rets, 252)
    cov = (rk.cov_shrinkage(rets, 0.5) if shrink else rets.cov()) * 252
    ef = rk.efficient_frontier(40, er, cov, 0.0, max_weight)
    names = rets.columns
    weights = {
        "Max Sharpe": pd.Series(rk.msr(rf, er, cov, 0.0, max_weight), index=names),
        "Min volatility": pd.Series(rk.gmv(cov, 0.0, max_weight), index=names),
        "Risk parity": pd.Series(rk.risk_parity(cov, max_weight=max_weight).values, index=names),
        "Equal weight": rk.equal_weights(names),
    }
    return er, cov, ef, weights


@st.cache_data(ttl=3600, show_spinner=False)
def future_bands(port_returns, rf, max_years=10, n_paths=2000):
    """Simulated wealth (start = 1) at monthly steps: percentiles and probabilities per horizon."""
    paths = rk.bootstrap_paths(port_returns, 252 * max_years, n_paths=n_paths, seed=7)
    monthly = paths[:, ::21]
    q = {p: np.percentile(monthly, p, axis=0) for p in (10, 25, 50, 75, 90)}
    yearly = paths[:, [252 * y for y in range(1, max_years + 1)]]
    loss = (yearly < 1).mean(axis=0)
    beat_cash = (yearly > np.array([(1 + rf) ** y for y in range(1, max_years + 1)])).mean(axis=0)
    return q, loss, beat_cash


@st.cache_data(ttl=1800, show_spinner=False)
def market_data(name):
    data = market.download(tuple(market.all_tickers(name)))
    main = market.MARKETS[name]["main"]
    if main not in data or data[main].dropna().empty:
        raise RuntimeError("Market data unavailable")  # not cached, so the next visit retries
    return data


def market_reading(name, data):
    cfg = market.MARKETS[name]
    main = market.stats(data[cfg["main"]])
    vix = data[cfg["vix"]] if cfg["vix"] in data else None
    return market.regime(main, vix, cfg["headline"].get(cfg["main"], cfg["main"]).split(" (")[0])


@st.cache_data(ttl=86400, show_spinner=False)
def regular_plan_flags(codes):
    """{"MF:<code>": True if regular plan, False if direct} from the AMFI scheme list."""
    try:
        schemes = mf.load_schemes().set_index("code")["plan"]
    except Exception:
        return {}
    out = {}
    for c in codes:
        code = int(c[len(rk.MF_PREFIX):])
        if code in schemes.index:
            out[c] = schemes[code] == "regular"
    return out


def escape_dollars(chunks):
    for c in chunks:
        yield c.replace("$", "\\$")


def recovery_info(wealth):
    """Dates of the worst fall and how long it took to get back to the previous high."""
    dd = wealth / wealth.cummax() - 1
    trough = dd.idxmin()
    peak = wealth.loc[:trough].idxmax()
    after = wealth.loc[trough:]
    back = after[after >= wealth.loc[peak]]
    if len(back):
        months = (back.index[0] - peak).days / 30.4
        text = (f"about {months:.0f} months from the peak ({peak:%b %Y}) back to a new high ({back.index[0]:%b %Y})"
                if months >= 1.5 else "less than two months")
        return peak, trough, back.index[0], text
    return peak, trough, None, f"it has not fully recovered yet (the fall started in {peak:%b %Y})"


@st.cache_data(ttl=3600, show_spinner=False)
def strategy_backtest(rets, my_weights, rebalance, cost_bps, rf):
    strategies = {
        "Your portfolio": my_weights,
        "Equal weight": rk.equal_weights(rets.columns),
        "Min volatility (re-estimated)": rk.weight_gmv,
        "Risk parity (re-estimated)": rk.weight_risk_parity,
    }
    return rk.compare_strategies(rets, strategies, rebalance=rebalance, cost_bps=cost_bps,
                                 estimation_window=252, riskfree_rate=rf, periods_per_year=252)


@st.cache_data(show_spinner=False)
def read_upload(data, name, password=None):
    return hi.read_table(data, name, password)


@st.cache_data(ttl=3600, show_spinner=False)
def read_cas_cached(data, password):
    return txm.read_cas(data, password)


def pdf_first_text(data, password):
    import pdfplumber
    with pdfplumber.open(io.BytesIO(data), password=password or "") as pdf:
        return pdf.pages[0].extract_text() or "" if pdf.pages else ""


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def schemes_master():
    return peers.amfi_schemes()


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def nav_cached(code):
    return peers.nav_history(int(code))


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def peer_cached(code, rf_):
    rep_ = peers.peer_report(int(code), schemes_master(), rf_)
    return rep_


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def gap_cached(code):
    return peers.plan_gap(int(code), schemes_master())


@st.cache_data(ttl=3 * 3600, show_spinner=False)
def live_rf(cur):
    """(rate, source text) for the risk-free rate: India from direct liquid funds, US from 13-week T-bills."""
    try:
        if cur == "₹":
            r, d = peers.liquid_yield(schemes_master())
            return round(r, 4), f"3-month yield of direct liquid funds (≈ 91-day T-bill), {d:%d %b %Y}"
        if cur == "$":
            irx = market.download(("^IRX",), period="1mo")["^IRX"].dropna()
            return round(irx.iloc[-1] / 100, 4), f"13-week US T-bill yield, {irx.index[-1]:%d %b %Y}"
    except Exception:
        pass
    return None, None


def import_cas(upload, password):
    """A CAMS / KFintech CAS: holdings with exact AMFI codes, plus every transaction."""
    fid = upload.file_id
    try:
        with st.spinner(f"Reading the CAS statement {upload.name}..."):
            cas_hold, cas_tx = read_cas_cached(upload.getvalue(), password or "")
    except txm.CASError as e:
        st.warning(f"**{upload.name}**: {e}")
        if "password" in str(e).lower():
            st.text_input("CAS password", type="password", key=f"pdfpw_{fid}",
                          help="Usually your PAN in capitals. Used only to open the file in this session.")
        return None
    if cas_hold.empty:
        st.error(f"**{upload.name}**: no current holdings found in this statement.")
        return None
    missing = cas_hold["code"].isna()
    if missing.any():
        found = match_funds_cached(tuple(zip(cas_hold.loc[missing, "scheme"], cas_hold.loc[missing, "units"],
                                             cas_hold.loc[missing, "value"])))
        cas_hold.loc[missing, "code"] = [m["code"] for m in found]
        codes = dict(zip(cas_hold.loc[missing, "scheme"], [m["code"] for m in found]))
        cas_tx.loc[cas_tx["code"].isna(), "code"] = cas_tx.loc[cas_tx["code"].isna(), "scheme"].map(codes)
        cas_hold = cas_hold[cas_hold["code"].notna()]
    holdings = pd.DataFrame({"Ticker": [f"{rk.MF_PREFIX}{int(c)}" for c in cas_hold["code"]],
                             "Name": [mf.short_label(n) for n in cas_hold["scheme"]],
                             "Shares": cas_hold["units"].values, "Value": cas_hold["value"].values})
    holdings = holdings.groupby("Ticker", sort=False).agg(Name=("Name", "first"), Shares=("Shares", "sum"),
                                                          Value=("Value", "sum")).reset_index()
    holdings["Weight %"] = 100 * holdings["Value"] / holdings["Value"].sum()
    st.session_state[f"tx_{fid}"] = (cas_tx, cas_hold)
    total = holdings["Value"].sum()
    st.success(f"**{upload.name}**: CAS with **{len(holdings)} funds** worth **{money(total, '₹')}** "
               f"and **{len(cas_tx):,} transactions** since {cas_tx['date'].min():%b %Y}.")
    return holdings, total, "₹", f"cas_{fid}"


def import_transactions(upload, table):
    """A transaction history file (CSV/Excel): rebuilds holdings and keeps every transaction."""
    fid = upload.file_id
    try:
        t = txm.transactions_from_table(table)
    except ValueError as e:
        st.error(f"**{upload.name}**: {e}")
        return None
    if t["code"].isna().any():
        names = t.loc[t["code"].isna(), "scheme"].unique()
        with st.spinner("Matching fund names to AMFI schemes..."):
            found = match_funds_cached(tuple((n, None, None) for n in names))
        t.loc[t["code"].isna(), "code"] = t.loc[t["code"].isna(), "scheme"].map(dict(zip(names, [m["code"] for m in found])))
        t = t[t["code"].notna()]
    t["code"] = t["code"].astype(int)

    def lookup(code):
        s_ = nav_cached(int(code))
        return float(s_.iloc[-1]), s_.index[-1]
    try:
        with st.spinner("Valuing your funds at the latest NAV..."):
            h = txm.holdings_from_transactions(t, lookup)
    except Exception:
        st.error("Couldn't fetch NAVs from mfapi.in. Check your connection and try again.")
        return None
    if h.empty:
        st.error(f"**{upload.name}**: these transactions leave no units held today.")
        return None
    st.session_state[f"tx_{fid}"] = (t, h)
    holdings = pd.DataFrame({"Ticker": [f"{rk.MF_PREFIX}{int(c)}" for c in h["code"]],
                             "Name": [mf.short_label(n) for n in h["scheme"]], "Shares": h["units"].values,
                             "Value": h["value"].values})
    holdings = holdings.groupby("Ticker", sort=False).agg(Name=("Name", "first"), Shares=("Shares", "sum"),
                                                          Value=("Value", "sum")).reset_index()
    holdings["Weight %"] = 100 * holdings["Value"] / holdings["Value"].sum()
    total = holdings["Value"].sum()
    st.success(f"**{upload.name}**: {len(t):,} transactions → **{len(holdings)} funds** worth **{money(total, '₹')}**.")
    return holdings, total, "₹", f"tx_{fid}"


@st.cache_data(ttl=7 * 86400, show_spinner=False)
def _name_cached(ticker):
    return stocks.lookup_name(ticker)  # errors are not cached, so a rate limit is retried later


def ticker_names(tickers):
    """Readable names for tickers typed without one (e.g. AAPL -> Apple)."""
    out = {}
    for t in tickers:
        try:
            n = _name_cached(t)
        except Exception:
            n = None
        if n:
            out[t] = n
    return out


@st.cache_data(ttl=900, show_spinner=False)
def latest_prices(tickers):
    """Most recent closing price for each ticker (used when a file has shares but no values)"""
    try:
        px = rk.get_prices(list(tickers), period="1mo", use_cache=False)
    except ValueError:
        return {}
    return px.ffill().iloc[-1].dropna().to_dict()


@st.cache_data(ttl=86400, show_spinner=False)
def match_funds_cached(funds):
    return mf.match_funds(list(funds))


def import_holdings(upload):
    """
    Sidebar UI for one uploaded file.
    Returns (holdings with Ticker/Name/Value/Weight %, total value or None, currency, key) or None.
    """
    is_pdf = upload.name.lower().endswith(".pdf")
    fid = upload.file_id
    password = st.session_state.get(f"pdfpw_{fid}") or None if is_pdf else None
    st.session_state.pop(f"tx_{fid}", None)
    if is_pdf:
        try:
            first_text = pdf_first_text(upload.getvalue(), password)
        except Exception:
            first_text = ""
        if txm.looks_like_cas(first_text):
            return import_cas(upload, password)
    try:
        with st.spinner(f"Reading {upload.name}..."):
            table = read_upload(upload.getvalue(), upload.name, password)
    except hi.PasswordRequired as e:
        st.warning(f"🔒 **{upload.name}**: {e} Statements sent by email are often locked, "
                   "for example with your PAN or date of birth.")
        st.text_input("PDF password", type="password", key=f"pdfpw_{fid}",
                      help="Used only to open the file in this session. It is never stored.")
        return None
    except ValueError as e:
        st.error(f"Couldn't read **{upload.name}**: {e}")
        return None
    except Exception:
        st.error(f"Couldn't read **{upload.name}**. Is it a CSV, Excel or PDF file?")
        return None

    low_cols = [str(c).lower() for c in table.columns]
    if (any("date" in c for c in low_cols) and any(k in c for c in low_cols for k in ("unit", "qty", "quantity"))
            and any(k in c for c in low_cols for k in ("amount", "invest")) and not is_pdf):
        return import_transactions(upload, table)

    guess = hi.guess_columns(table)
    needs_help = ((guess["ticker_col"] is None and guess["name_col"] is None)
                  or not any(guess[k] for k in ("value_col", "shares_col", "weight_col")))
    options = ["(none)"] + list(table.columns)
    with st.expander(f"Check the columns in {upload.name}", expanded=needs_help):
        def pick(label, role, help_text):
            choice = st.selectbox(label, options, help=help_text, key=f"map_{role}_{fid}",
                                  index=options.index(guess[role]) if guess[role] else 0)
            return None if choice == "(none)" else choice
        ticker_col = pick("Ticker column", "ticker_col", "The column with symbols like AAPL, VTI or RELIANCE.")
        name_col = pick("Fund name column (Indian mutual funds)", "name_col",
                        "Used when there is no ticker column: fund names are matched to AMFI schemes.")
        value_col = pick("Market value column", "value_col", "Best choice: today's value of each position.")
        shares_col = pick("Shares / units column", "shares_col",
                          "Used with today's prices if there is no value column.")
        weight_col = pick("Weight % column", "weight_col", "Used if there is no value or shares column.")
        labels = list(hi.SUFFIXES)
        auto = hi.guess_suffix(table)
        suffix_label = st.selectbox("Exchange (adds the Yahoo suffix)", labels, key=f"sfx_{fid}",
                                    index=[hi.SUFFIXES[l] for l in labels].index(auto),
                                    help="Yahoo needs a suffix for non-US stocks, e.g. RELIANCE.NS on India's NSE.")
        st.caption("Table we read from your PDF (PDF reading is best effort, so please check it):" if is_pdf
                   else "First rows we read from your file:")
        st.dataframe(table.head(8), hide_index=True, width="stretch")
    suffix = hi.SUFFIXES[suffix_label]
    by_name = ticker_col is None and name_col is not None

    prices = None
    if value_col is None and shares_col and ticker_col:
        tickers = sorted({hi.clean_ticker(t, suffix) for t in table[ticker_col]} - {""})
        tickers = [t for t in tickers if hi.TICKER_PATTERN.match(t)]
        with st.spinner("Looking up today's prices..."):
            prices = latest_prices(tuple(tickers))
    try:
        holdings, skipped = hi.build_holdings(table, ticker_col, shares_col, value_col, weight_col,
                                              prices=prices, suffix=suffix, name_col=name_col)
    except ValueError as e:
        st.error(f"**{upload.name}**: {e}")
        return None

    currency = CURRENCY_BY_SUFFIX.get(suffix, "$")
    if by_name:
        currency = "₹"
        funds = tuple(zip(holdings["Name"], holdings["Shares"].fillna(0), holdings["Value"].fillna(0)))
        try:
            with st.spinner("Matching funds to AMFI schemes and checking NAVs..."):
                matches = match_funds_cached(funds)
        except Exception:
            st.error("Couldn't reach the mutual fund database (mfapi.in). Check your connection and try again.")
            return None
        holdings["Ticker"] = [f"{rk.MF_PREFIX}{m['code']}" if m["code"] else "" for m in matches]
        holdings["Name"] = [mf.short_label(m["scheme"]) if m["scheme"] else n
                            for m, n in zip(matches, holdings["Name"])]
        icons = {"verified": "✅ NAV verified", "name only": "⚠️ check this", "not found": "❌ not found"}
        report = pd.DataFrame({"Check": [icons[m["status"]] for m in matches],
                               "Matched AMFI scheme": [m["scheme"] or "–" for m in matches],
                               "In your file": [f[0] for f in funds],
                               "Why": [m["detail"] for m in matches]})
        n_ok = sum(m["status"] == "verified" for m in matches)
        with st.expander(f"Fund matches: {n_ok} of {len(matches)} verified", expanded=n_ok < len(matches)):
            st.dataframe(report, hide_index=True, width="stretch")
            st.caption("Each fund's value ÷ units on your statement is compared with the NAV published by AMFI. "
                       "A match confirms the exact scheme, plan and option.")
        missing = holdings["Ticker"] == ""
        if missing.any():
            st.warning("Left out (no matching scheme found): " + ", ".join(holdings.loc[missing, "Name"]))
            holdings = holdings[~missing]
            holdings["Weight %"] = 100 * holdings["Weight %"] / holdings["Weight %"].sum()

    total = holdings["Value"].sum() if holdings["Value"].notna().all() else None
    kind = "funds" if by_name else "holdings"
    st.success(f"**{upload.name}**: loaded **{len(holdings)} {kind}**"
               + (f" worth **{money(total, currency)}**" if total else "") + ".")
    if len(skipped):
        with st.expander(f"{len(skipped)} rows not included"):
            st.dataframe(skipped, hide_index=True, width="stretch")
    key = f"{fid}_{ticker_col}_{name_col}_{value_col}_{shares_col}_{weight_col}_{suffix}"
    return holdings, total, currency, key


def combine_files(results):
    """Merges holdings from several files into one portfolio (by value when every file has values)."""
    if len(results) == 1:
        h, total, currency, key = results[0]
        return h, total, currency, key
    frames, totals = [], [r[1] for r in results]
    by_value = all(t for t in totals)
    for h, total, _, _ in results:
        part = h[["Ticker", "Name", "Value", "Weight %"]].copy()
        part["Amount"] = part["Value"] if by_value else part["Weight %"] / len(results)
        frames.append(part)
    allh = pd.concat(frames)
    merged = allh.groupby("Ticker", sort=False).agg(Name=("Name", "first"), Amount=("Amount", "sum"))
    merged["Weight %"] = 100 * merged["Amount"] / merged["Amount"].sum()
    merged = merged.reset_index().sort_values("Weight %", ascending=False, ignore_index=True)
    if not by_value:
        st.info("Not every file has market values, so each file was given equal weight.")
    total = sum(totals) if by_value else None
    return merged, total, results[0][2], "+".join(r[3] for r in results)


# -----------------------------------------------------------------------------
# Sidebar: the portfolio
# -----------------------------------------------------------------------------
with st.sidebar:
    md("<div class='side-brand'>Risk Kit</div><div class='side-title'>Your portfolio</div>", unsafe_allow_html=True)
    uploads = st.file_uploader(
        "Upload a statement", type=["csv", "xlsx", "xls", "pdf"], accept_multiple_files=True,
        help="A CAMS / KFintech CAS (best: adds your real returns and tax), a broker holdings export, or a "
             "transaction file. Upload several to combine accounts. Files are used only in this session.")
    results = [r for r in (import_holdings(u) for u in uploads or []) if r is not None]
    imported = combine_files(results) if results else None
    if imported is None:
        preset = st.selectbox("Or try an example", list(PRESETS) + ["Start from scratch"])
        base = PRESETS.get(preset, {"MF:120716": 100})
        start_table = pd.DataFrame({"Name": [PRESET_NAMES.get(t) or stocks.name_of(t) or "" for t in base],
                                    "Weight %": list(base.values()), "Ticker": list(base)})
        editor_key, default_value = f"holdings_{preset}", 1_000_000 if preset.startswith("India") else 100_000
        if any(t.startswith(rk.MF_PREFIX) or t.endswith((".NS", ".BO")) for t in base):
            CUR = "₹"
    else:
        file_holdings, file_total, CUR, editor_key = imported
        start_table = file_holdings[["Name", "Weight %", "Ticker"]].copy()
        start_table["Name"] = start_table["Name"].fillna("")
        start_table["Weight %"] = start_table["Weight %"].round(2)
        default_value = int(round(file_total)) if file_total else 100_000
    with st.form("portfolio", border=False):
        holdings = st.data_editor(
            start_table,
            num_rows="dynamic", hide_index=True, key=f"editor_{editor_key}", width="stretch",
            column_config={
                "Name": st.column_config.TextColumn("Holding", help="Any name you like"),
                "Weight %": st.column_config.NumberColumn("Weight %", min_value=0.0, max_value=100.0, step=0.1,
                                                          format="%.1f", width="small"),
                "Ticker": st.column_config.TextColumn(
                    "Symbol", width="small",
                    help="To add a holding: its Yahoo symbol (RELIANCE.NS, AAPL) or MF:<AMFI code> for a mutual fund. "
                         "Easier: use 'Try ideas → Build a new one' to search by name."),
            })
        value = st.number_input(f"Portfolio value ({CUR})", min_value=100, value=max(default_value, 100), step=1_000,
                                key=f"value_{editor_key}", help="Used to turn percentages into money.")
        with st.expander("Settings", icon=":material/tune:"):
            bench_default = {"₹": peers.NIFTY50_TR, "£": "^FTSE", "€": "^STOXX50E", "C$": "^GSPTSE", "A$": "^AXJO",
                             "HK$": "^HSI", "¥": "^N225"}.get(CUR, "SPY")
            benchmark = st.text_input("Compare with (symbol)", bench_default, key=f"bench_{editor_key}",
                                      help="What counts as 'the market' for you. Default for India: Nifty 50 "
                                           f"total return ({peers.NIFTY50_TR}); for the US: S&P 500 (SPY).")
            c1, c2 = st.columns(2)
            start = c1.date_input("From", dt.date(2007, 1, 1), min_value=dt.date(1990, 1, 1))
            end = c2.date_input("To", dt.date.today())
            rf_live, rf_src = live_rf(CUR)
            rf_default = round(rf_live * 100, 2) if rf_live else (6.0 if CUR == "₹" else 3.0)
            rf = st.number_input("Safe interest rate (% a year)", 0.0, 15.0, rf_default, 0.05,
                                 key=f"rf_{editor_key}_{rf_default}",
                                 help="What a riskless deposit pays now. " + (f"Live: {rf_src}." if rf_src else "")) / 100
            conf = st.radio("Bad-day measure", ["95%", "99%"], horizontal=True,
                            help="Value at Risk confidence: 95% = the worst 1 day in 20; 99% = 1 in 100.")
        submitted = st.form_submit_button("Update analysis", type="primary", width="stretch")

level = 5 if conf == "95%" else 1

_tx_parts = [st.session_state.get(f"tx_{u.file_id}") for u in (uploads or [])] if imported is not None else []
_tx_parts = [p for p in _tx_parts if p is not None]
TX = pd.concat([p[0] for p in _tx_parts], ignore_index=True).sort_values("date", kind="stable") if _tx_parts else None
TX_HOLD = pd.concat([p[1] for p in _tx_parts], ignore_index=True) if _tx_parts else None

rows = holdings.dropna(subset=["Ticker"]).copy()
rows["Ticker"] = rows["Ticker"].astype(str).str.strip().str.upper()
rows["Weight %"] = pd.to_numeric(rows["Weight %"], errors="coerce").fillna(0)
rows = rows[(rows["Ticker"] != "") & (rows["Weight %"] > 0)].groupby("Ticker", sort=False)["Weight %"].sum()
NAMES = {}
if "Name" in holdings.columns:
    for t, n in zip(holdings["Ticker"], holdings["Name"]):
        if isinstance(t, str) and isinstance(n, str) and n.strip():
            NAMES[t.strip().upper()] = n.strip()
_unnamed = tuple(t for t in rows.index if t not in NAMES and not t.startswith(rk.MF_PREFIX))
if _unnamed:
    NAMES.update(ticker_names(_unnamed))
_suffixes = pd.Series([".NS" if t.startswith(rk.MF_PREFIX) else "." + t.rsplit(".", 1)[1] if "." in t else ""
                       for t in rows.index], dtype=str)
if len(_suffixes):
    CUR = CURRENCY_BY_SUFFIX.get(_suffixes.mode().iloc[0], "$")

def hero(sub="Upload a statement or pick an example in the sidebar.", kpis=()):
    items = "".join(f"<div class='kpi'><div class='l'>{l}</div><div class='v {c}'>{v}</div></div>" for l, v, c in kpis)
    return (f"<div class='hero'><div><div class='eyebrow'>Risk Kit · Portfolio analytics</div>"
            f"<div class='title'>Portfolio risk report</div><div class='sub'>{sub}</div></div>"
            f"<div class='kpis'>{items}</div></div>").replace("$", "&#36;")


hero_slot = st.empty()
hero_slot.markdown(hero(), unsafe_allow_html=True)

if rows.empty:
    st.info("Add at least one ticker with a weight above 0 in the sidebar.")
    st.stop()
if start >= end:
    st.error("The start date must be before the end date.")
    st.stop()
if abs(rows.sum() - 100) > 0.5:
    st.sidebar.info(f"Your weights add up to {rows.sum():.1f}%. They were scaled to 100%.")

try:
    with st.spinner("Downloading prices from Yahoo Finance and crunching the numbers..."):
        rep = load_report(tuple(rows.index), tuple(rows.values), benchmark.strip().upper(),
                          start, end, rf, level)
except ValueError as e:
    st.error(f"Couldn't analyze this portfolio: {e}")
    st.stop()
except Exception as e:  # network problems, Yahoo outages
    st.error("Something went wrong while downloading data from Yahoo Finance. "
             "Check the tickers and your connection, then try again.")
    with st.expander("Technical details"):
        st.code(repr(e))
    st.stop()

# -----------------------------------------------------------------------------
# Shared values
# -----------------------------------------------------------------------------
def relabel(obj, names):
    """Renames tickers to display names throughout the report (index, columns, '<ticker> Return' columns)."""
    if not names:
        return obj
    def f(x):
        return names.get(x, x) if isinstance(x, str) else x
    if isinstance(obj, dict):
        return {k: relabel(v, names) for k, v in obj.items()}
    if isinstance(obj, pd.DataFrame):
        out = obj.copy()
        out.columns = [f(c) for c in out.columns]
        if out.index.dtype == object:
            out.index = [f(i) for i in out.index]
        out.attrs = {k: relabel(v, names) for k, v in obj.attrs.items()}
        return out
    if isinstance(obj, pd.Series):
        out = obj.copy()
        if out.index.dtype == object:
            out.index = [f(i) for i in out.index]
        out.name = f(obj.name)
        return out
    if isinstance(obj, (list, tuple)):
        return type(obj)(f(x) for x in obj)
    return obj


labels, used = {}, set()
for t in rows.index:
    name = NAMES.get(t)
    if name and name not in used and name not in rows.index:
        labels[t] = name
        used.add(name)
bench_raw = benchmark.strip().upper()
bench_label = peers.BENCH_LABELS.get(bench_raw) or stocks.name_of(bench_raw) or ticker_names((bench_raw,)).get(bench_raw, bench_raw)
if bench_label != bench_raw and bench_raw not in rows.index:
    labels[bench_raw] = bench_label
bench_key = labels.get(bench_raw, bench_raw)  # the benchmark's column in the returns table
rep = relabel(rep, {**labels, **{f"{t} Return": f"{n} Return" for t, n in labels.items()},
                    f"Benchmark ({bench_raw})": f"Benchmark ({bench_label})"})

w = rep["weights"]
tickers = list(w.index)
rets = rep["returns"]
port = rep["portfolio"]
bench_name = bench_label
bench_col = f"Benchmark ({bench_name})"
summary = rep["summary"]
has_bench = bench_col in summary.index
bench = rets[bench_key].reindex(port.index) if has_bench else None
p = summary.loc["Portfolio"]
b = summary.loc[bench_col] if has_bench else None
t0, t1 = rep["period"]
years = (t1 - t0).days / 365.25
var_key, cvar_key = f"Historic VaR ({level}%)", f"Historic CVaR ({level}%)"
days_in = round(100 / level)

if rep["dropped_tickers"]:
    st.warning(f"No data found for: {', '.join(rep['dropped_tickers'])}. "
               "They were left out and the remaining weights rescaled.")
if pd.Timestamp(start) < t0 - pd.Timedelta(days=10):
    st.caption(f"Analysis from {t0:%d %b %Y}, the first date all your holdings have prices.")

# Facts shared by the commentary and the explanations
final = value * rep["wealth"].iloc[-1]
bench_wealth = (1 + bench.fillna(0)).cumprod() if has_bench else None
bench_final = value * bench_wealth.iloc[-1] if has_bench else None
peak_d, trough_d, recov_d, recovery_text = recovery_info(rep["wealth"])
corr_m = rep["correlation"]
avg_corr = (corr_m.values.sum() - len(corr_m)) / (len(corr_m) * (len(corr_m) - 1)) if len(corr_m) > 1 else None
vol_now = rep["ewma_vol"].dropna().iloc[-1] if rep["ewma_vol"].notna().any() else None
rc_all = rep["risk_contribution"]
stress_all = rep["stress"]
covered_all = stress_all[stress_all["Covered"] == True].copy()  # noqa: E712
crisis_port = crisis_bench = None
if has_bench and len(covered_all):
    _bs = rk.stress_test(rets[[bench_key]], [1.0])
    if "Portfolio Return" in _bs.columns:
        crisis_port = covered_all["Portfolio Return"].astype(float).mean()
        crisis_bench = _bs["Portfolio Return"].reindex(covered_all.index).astype(float).mean()
capm_p = rep["capm"].loc["Portfolio"] if has_bench and "capm" in rep else None
down_cap = rk.up_down_capture(port, bench).iloc[0]["Down Capture"] if has_bench else None
facts = cmt.PortfolioFacts(
    years=years, value=value, final_value=final, ret=p["Annualized Return"], vol=p["Annualized Vol"],
    sharpe=p["Sharpe Ratio"], max_dd=p["Max Drawdown"], recovery=recovery_text, rf=rf,
    n_holdings=len(tickers), top_risk_name=rc_all.idxmax(), top_risk_share=rc_all.max(),
    top_risk_weight=w[rc_all.idxmax()], avg_corr=avg_corr, vol_now=vol_now,
    bench=bench_name if has_bench else None,
    bench_ret=b["Annualized Return"] if has_bench else None, bench_vol=b["Annualized Vol"] if has_bench else None,
    bench_sharpe=b["Sharpe Ratio"] if has_bench else None, bench_max_dd=b["Max Drawdown"] if has_bench else None,
    bench_final_value=bench_final,
    beta=capm_p["Beta"] if capm_p is not None else None,
    alpha=capm_p["Alpha (annual)"] if capm_p is not None else None,
    alpha_t=capm_p["Alpha t-stat"] if capm_p is not None else None,
    down_capture=down_cap, crisis_port=crisis_port, crisis_bench=crisis_bench,
    n_crises=len(covered_all) if crisis_port is not None else 0,
    var_ok=bool(rep["kupiec"]["Model OK"]) if "kupiec" in rep else None,
    breach_rate=rep["kupiec"]["Breach Rate"] if "kupiec" in rep else None, level=level, money=money)
verdict = cmt.build_commentary(facts)
ICON = {"good": "✅", "ok": "⚠️", "bad": "❌"}

# Inputs for the adviser: per-holding table and simulated outcomes
label_of = {t: labels.get(t, t) for t in rows.index}
reg_flags = regular_plan_flags(tuple(t for t in rows.index if t.startswith(rk.MF_PREFIX)))
hold = pd.DataFrame({"weight": w, "risk_share": rc_all.reindex(tickers)})
for col, key in [("ret", "Annualized Return"), ("vol", "Annualized Vol"), ("sharpe", "Sharpe Ratio")]:
    hold[col] = [summary.loc[t, key] if t in summary.index else np.nan for t in tickers]
hold["regular"] = [next((reg_flags[r] for r, l in label_of.items() if l == t and r in reg_flags), None) for t in tickers]
future_rows, bad_case = {}, {}
if len(port.dropna()) >= 260:
    _q, _loss, _ = future_bands(port.dropna(), rf)
    for y in range(1, 11):
        bad_case[y] = _q[10][y * 12]
        if y in (1, 3, 5, 10):
            future_rows[y] = {"bad": _q[10][y * 12] * value, "mid": _q[50][y * 12] * value,
                              "good": _q[90][y * 12] * value, "loss": _loss[y - 1]}
# Fund intelligence: peer ranks, measured plan costs, actual returns and tax lots
MF_HELD = [t for t in rows.index if t.startswith(rk.MF_PREFIX)]
PEER, GAP = {}, {}
if MF_HELD:
    try:
        with st.spinner("Ranking your funds against their category peers (AMFI data)..."):
            sc_master = schemes_master()
            for t in MF_HELD:
                code = int(t[len(rk.MF_PREFIX):])
                pr = peer_cached(code, round(rf, 4))
                if pr:
                    PEER[label_of[t]] = pr
                row_ = sc_master[sc_master["code"] == code]
                if len(row_) and not bool(row_.iloc[0]["direct"]):
                    g = gap_cached(code)
                    if g and g["gap"] > 0:
                        GAP[label_of[t]] = g
    except Exception:
        st.toast("Couldn't load category data from AMFI right now; fund rankings are skipped.")

TAXSET = txm.TaxSettings(slab=st.session_state.get("tax_slab", 30) / 100,
                         ltcg_used=float(st.session_state.get("tax_ltcg_used", 0) or 0))
LOTS, REALISED, RET, RET_TOTAL, TAX = {}, [], None, None, None
if TX is not None and len(TX):
    _lots, REALISED = txm.build_lots(TX)
    for (code, _), ls in _lots.items():
        if ls and code == code and code is not None:
            LOTS.setdefault(int(code), []).extend(ls)
    for c in LOTS:
        LOTS[c].sort(key=lambda l: l.date)
    RET, RET_TOTAL = txm.returns_summary(TX, TX_HOLD)
    try:
        bench_tr = nav_cached(int(peers.NIFTY50_TR[len(rk.MF_PREFIX):]))
        RET_TOTAL["bench_value"], RET_TOTAL["bench_xirr"] = txm.same_flows_into(TX, bench_tr, RET_TOTAL["as_of"])
    except Exception:
        RET_TOTAL["bench_value"] = RET_TOTAL["bench_xirr"] = None

    # Tax position if everything were sold today
    sc_cat = {}
    try:
        _m = schemes_master().set_index("code")
        sc_cat = {c: (_m.loc[c, "category"], _m.loc[c, "name"]) if c in _m.index else ("", "") for c in LOTS}
    except Exception:
        sc_cat = {c: ("", "") for c in LOTS}
    today = pd.Timestamp.today().normalize()
    fy = txm.financial_year(today)
    klass = {c: st.session_state.get(f"taxclass_{c}", txm.tax_class(*sc_cat.get(c, ("", "")))) for c in LOTS}
    TAXSET.classes = {(r.code, r.scheme): klass.get(int(r.code), "other") for r in REALISED if r.code == r.code}
    booked_eq_lt = txm.realised_eq_ltcg(REALISED, TAXSET.classes, fy)
    rows_tax, all_pieces = [], []
    fmv_cache = {}
    for code, ls in LOTS.items():
        units = sum(l.units for l in ls)
        if units <= 1e-6:
            continue
        try:
            navs = nav_cached(code)
            nav_now = float(navs.iloc[-1])
            fmv = navs[navs.index <= txm.FMV_DATE]
            fmv_cache[code] = float(fmv.iloc[-1]) if len(fmv) else None
        except Exception:
            continue
        k = klass[code]
        pieces = txm.gains_on_sale(ls, units, nav_now, today, k, fmv_cache[code])
        for p_ in pieces:
            p_["klass"] = k
        all_pieces += pieces
        name_ = label_of.get(f"{rk.MF_PREFIX}{code}", sc_cat.get(code, ("", ""))[1] or str(code))
        rows_tax.append({"code": code, "Fund": name_, "Tax type": k, "Value": units * nav_now,
                         "Short-term gain": sum(p_["gain"] for p_ in pieces if p_["term"] == "short"),
                         "Long-term gain": sum(p_["gain"] for p_ in pieces if p_["term"] == "long"),
                         "Exit-load window": txm.exit_load_units(ls, today, k) * nav_now,
                         "Locked (ELSS)": txm.locked_units(ls, today, *sc_cat.get(code, ("", ""))) * nav_now,
                         "lt_lots": [l for l in ls if (today - l.date).days > 365], "nav": nav_now})
    if rows_tax:
        TAX = {"rows": pd.DataFrame(rows_tax), "all": txm.tax_on_pieces(all_pieces, TAXSET, today, booked_eq_lt),
               "booked_eq_lt": booked_eq_lt, "fy": fy, "klass": klass, "fmv": fmv_cache, "today": today}
        # Tax-free harvesting: long-term equity gains that fit in this year's exemption
        room = max(txm.LTCG_EXEMPTION - TAXSET.ltcg_used - booked_eq_lt, 0.0)
        harvest_gain = harvest_value = 0.0
        for r_ in rows_tax:
            if r_["Tax type"] != "equity" or room - harvest_gain <= 0:
                continue
            for l in r_["lt_lots"]:
                g_ = l.units * r_["nav"] - l.cost
                if g_ <= 0:
                    continue
                take = min(1.0, (room - harvest_gain) / g_)
                harvest_gain += g_ * take
                harvest_value += l.units * r_["nav"] * take
                if room - harvest_gain <= 0.5:
                    break
        TAX.update(room=room, harvest_gain=harvest_gain, harvest_value=harvest_value)

hold["gap"] = [GAP[t]["gap"] if t in GAP else np.nan for t in hold.index]
hold["peer"] = [PEER[t]["label"] if t in PEER else None for t in hold.index]
hold["peer_rank"] = [f"#{PEER[t]['ranks']['3Y']['rank']} of {PEER[t]['ranks']['3Y']['n']} in "
                     f"{PEER[t]['category'].split(' - ')[-1]} over 3 years"
                     if t in PEER and PEER[t]["ranks"].get("3Y") else None for t in hold.index]
EXTRA = {}
if RET_TOTAL is not None:
    EXTRA["xirr"] = RET_TOTAL["xirr"]
    EXTRA["bench_xirr"] = RET_TOTAL.get("bench_xirr")
if TAX is not None:
    EXTRA.update(tax_all=TAX["all"]["tax"], harvest_gain=TAX["harvest_gain"], harvest_value=TAX["harvest_value"])

home_market = market.detect_market(rows.index)
try:
    mkt_reg = market_reading(home_market, market_data(home_market))
    mkt_reg["implications"] = market.implications(mkt_reg, capm_p["Beta"] if capm_p is not None else None, value, money)
except Exception:
    mkt_reg = None
# What each holding is (asset class; fund, ETF or stock) and the investor's own answers from the Action plan tab
_cat = {}
if MF_HELD:
    try:
        _mst = schemes_master().drop_duplicates("code").set_index("code")
        for _t in MF_HELD:
            _c = int(_t[len(rk.MF_PREFIX):])
            if _c in _mst.index:
                _cat[_t] = (_mst.loc[_c, "name"], _mst.loc[_c, "category"])
    except Exception:
        pass
CLASS_OF, KIND_OF = {}, {}
for _t in rows.index:
    _nm, _ct = _cat.get(_t, (NAMES.get(_t) or label_of[_t], ""))
    CLASS_OF[label_of[_t]] = advisor.asset_class(_t, _nm, _ct, CUR)
    KIND_OF[label_of[_t]] = advisor.holding_kind(_t, _nm, _ct)
hold["kind"] = [KIND_OF.get(n, "stock") for n in hold.index]
PROFILE = {"horizon": st.session_state.get("pf_horizon") or "5–10 years",
           "comfort": st.session_state.get("pf_comfort") or "Balanced",
           "emergency": st.session_state.get("pf_emergency") or "Not sure",
           "monthly": float(st.session_state.get("pf_monthly") or 0)}
actions = advisor.action_plan(facts, hold, rep["correlation"], bad_case, CUR, market=mkt_reg, extra=EXTRA,
                              profile=PROFILE, classes=CLASS_OF)

ret_cls = "pos" if p["Annualized Return"] >= 0 else "neg"
hero_slot.markdown(hero(
    f"{len(tickers)} holding{'s' if len(tickers) > 1 else ''} · {t0:%b %Y} – {t1:%b %Y} · compared with {bench_name}",
    [("Value", money(value), ""),
     ("Your XIRR", pct(RET_TOTAL["xirr"]), "pos" if RET_TOTAL["xirr"] >= 0 else "neg") if RET_TOTAL is not None
     else ("Return / yr", pct(p["Annualized Return"]), ret_cls),
     ("Worst fall", pct(p["Max Drawdown"]), "neg"), ("Verdict", verdict["grade"], ""),
     (f"{mkt_reg['name']} today" if mkt_reg else "Market today", mkt_reg["mood"] if mkt_reg else "–", "")]),
    unsafe_allow_html=True)

SHOW_FUNDS = bool(MF_HELD) or TX is not None
_tab_names = ([":material/insights: Summary", ":material/task_alt: Action plan", ":material/trending_up: Future"]
              + ([":material/account_balance: My funds"] if SHOW_FUNDS else [])
              + [":material/public: Markets", ":material/science: Try ideas", ":material/analytics: Details"])
_tabs = st.tabs(_tab_names)
t_over, t_plan, t_future = _tabs[:3]
t_funds = _tabs[3] if SHOW_FUNDS else None
t_mkt, t_tools, t_details = _tabs[-3:]
with t_tools:
    t_what, t_build = st.tabs([":material/tune: Change my portfolio", ":material/add_chart: Build a new one"])
with t_details:
    t_risk, t_div, t_stress, t_stats = st.tabs(["Risk", "Diversification", "Crises", "All statistics"])

# =============================================================================
# OVERVIEW
# =============================================================================
with t_over:
    pts = verdict["points"]
    gclass = {"Strong": "good", "Solid": "good", "Mixed": "ok", "Weak": "bad"}[verdict["grade"]]
    gi = ["Weak", "Mixed", "Solid", "Strong"].index(verdict["grade"])
    tally = {k: sum(pt["status"] == k for pt in pts) for k in ("good", "ok", "bad")}
    tally_html = "".join(f"<span class='{k}'>{n} {lbl if n == 1 else lbl + 's'}</span>" for k, lbl in
                         [("good", "strength"), ("ok", "thing to watch"), ("bad", "concern")]
                         for n in [tally[k]] if n).replace("things to watchs", "things to watch").replace(
                         "thing to watchs", "things to watch")
    md(f"<div class='verdict {gclass}'><div class='vl'><div class='eyebrow'>Overall verdict</div>"
       f"<div class='word'>{verdict['grade']}</div><div class='meter'>"
       + "".join(f"<span class='{'on' if i <= gi else ''}'></span>" for i in range(4))
       + "</div><div class='meter-l'><span>Weak</span><span>Strong</span></div></div>"
       f"<div class='vr'><p>{verdict['summary']}</p><div class='tally'>{tally_html}</div></div></div>",
       unsafe_allow_html=True)

    def delta(key, fmt):
        if b is None:
            return None
        return f"{fmt(p[key] - b[key])} vs market"

    def stat(label, value_, diff=None, better=None, foot=""):
        chip = ""
        if diff is not None:
            cls = "flat" if better is None else ("up" if better else "down")
            chip = f"<span class='d {cls}'>{diff}</span>"
        return (f"<div class='stat'><div class='l'>{label}</div><div class='v'>{value_}</div>{chip}"
                f"<div class='f'>{foot}</div></div>")

    def vs(key, words, higher_is_better=True):
        """Chip text like '1.8 pts ahead of market', and whether that's good for the investor."""
        if b is None:
            return None, None
        d_ = p[key] - b[key]
        if abs(d_) < 0.0005:
            return "Same as market", None
        word = words[0] if d_ > 0 else words[1]
        return f"{abs(d_)*100:.1f} pts {word} than market", (d_ > 0) == higher_is_better

    d1, g1 = vs("Annualized Return", ("ahead", "behind"))
    d1 = d1.replace("ahead than", "ahead of").replace("behind than", "behind") if d1 else d1
    d2, g2 = vs("Annualized Vol", ("bumpier", "calmer"), higher_is_better=False)
    d3, g3 = vs("Max Drawdown", ("shallower", "deeper"))
    md("<div class='stats'>"
       + stat("Growth per year", pct(p["Annualized Return"]), d1, g1,
              f"{bench_name}: {pct(b['Annualized Return'])}" if b is not None else f"Over {years:.1f} years")
       + stat("Ups and downs", pct(p["Annualized Vol"]), d2, g2,
              "Typical yearly swing · lower is calmer")
       + stat("Worst fall", pct(p["Max Drawdown"]), d3, g3,
              f"Peak {peak_d:%b %Y} → low {trough_d:%b %Y}")
       + stat(f"{short_money(value)} became", short_money(final), f"× {final / value:.1f}", True if final > value else False,
              f"{bench_name}: {short_money(bench_final)}" if has_bench else f"Since {t0:%b %Y}")
       + "</div>", unsafe_allow_html=True)

    fig = go.Figure()
    if has_bench:
        bw = value * (1 + bench.fillna(0)).cumprod()
        fig.add_scatter(x=bw.index, y=bw, name=bench_name, line=dict(color="#a7adba", width=1.6, dash="dot"),
                        hovertemplate=CUR + "%{y:,.0f}")
    fig.add_scatter(x=port.index, y=value * rep["wealth"], name="Your portfolio", line=dict(color=NAVY_2, width=2.4),
                    hovertemplate=CUR + "%{y:,.0f}")
    fig.update_layout(hovermode="x unified")
    fig = style_fig(fig, 340, yfmt="$,.0f")
    show(fig, f"How {short_money(value)} grew since {t0:%b %Y}")

    goods = [pt for pt in pts if pt["status"] == "good"][:3]
    watch = sorted([pt for pt in pts if pt["status"] != "good"], key=lambda pt: pt["status"] != "bad")[:3]
    panels = ""
    for title, items, cls in [("What's working", goods, "good"), ("Watch out for", watch, "bad")]:
        body = "".join(f"<li><b>{pt['title']}</b><span>{pt['text']}</span></li>" for pt in items) or \
            "<li><span>Nothing stands out.</span></li>"
        panels += f"<div class='panel {cls}'><div class='ph'>{title}</div><ul>{body}</ul></div>"
    md(f"<div class='sec'><span class='t'>At a glance</span></div><div class='duo'>{panels}</div>",
       unsafe_allow_html=True)

    steps_html = "".join(
        f"<div class='step'><div class='top'><span class='n'>{i:02d}</span><span class='prio {a_['priority']}'>"
        f"{a_['priority']}</span></div><div class='t'>{a_['title']}</div><div class='how'>{a_['how']}</div></div>"
        for i, a_ in enumerate(actions[:3], 1))
    md(f"<div class='sec'><span class='t'>What to do next</span><span class='s'>Most important first</span></div>"
       f"<div class='steps3'>{steps_html}</div>"
       f"<div class='more'>{len(actions)} steps in all, with how-to guides, your target mix and an adviser chat, in "
       f"<b>Action plan</b>. Judged against the market and common rules of thumb; describes the past, not advice.</div>",
       unsafe_allow_html=True)

    md(f"<div class='sec'><span class='t'>Every observation</span><span class='s'>{len(pts)} in total</span></div>",
       unsafe_allow_html=True)
    with st.expander(f"All {len(pts)} observations, grouped", expanded=True, icon=":material/fact_check:"):
        html = ""
        for k, lbl, col in [("bad", "Concerns", NEG), ("ok", "Worth watching", "#b7791f"), ("good", "Strengths", POS)]:
            grp = [pt for pt in pts if pt["status"] == k]
            if not grp:
                continue
            html += (f"<div class='obs-h' style='color:{col}'>{lbl} <span class='c'>{len(grp)}</span></div><div class='obs'>"
                     + "".join(f"<div class='point {k}'><b>{pt['title']}</b>{pt['text']}</div>" for pt in grp) + "</div>")
        md(html, unsafe_allow_html=True)


with t_stats:
    st.subheader("Key numbers")
    m = st.columns(5)
    m[0].metric("Return per year", pct(p["Annualized Return"]), delta("Annualized Return", lambda d: f"{d*100:+.1f} pts"))
    m[1].metric("Volatility per year", pct(p["Annualized Vol"]), delta("Annualized Vol", lambda d: f"{d*100:+.1f} pts"),
                delta_color="inverse")
    m[2].metric("Sharpe ratio", num(p["Sharpe Ratio"]), delta("Sharpe Ratio", lambda d: f"{d:+.2f}"))
    m[3].metric("Max drawdown", pct(p["Max Drawdown"]), delta("Max Drawdown", lambda d: f"{d*100:+.1f} pts"))
    m[4].metric(f"1-day VaR ({conf})", money(value * p[var_key]), delta(var_key, lambda d: f"{d*100:+.2f} pts"),
                delta_color="inverse")
    lo_y = value * (p["Annualized Return"] - p["Annualized Vol"])
    hi_y = value * (p["Annualized Return"] + p["Annualized Vol"])
    explain("What these numbers mean for your portfolio", f"""
- **Return {pct(p['Annualized Return'])} a year:** {money(value)} became **{money(final)}** over {years:.1f} years{f", while {bench_name} turned the same amount into **{money(bench_final)}**" if has_bench else ""}.
- **Volatility {pct(p['Annualized Vol'])}:** in a typical year (2 years out of 3) your result would land between **{'a loss of ' + money(-lo_y) if lo_y < 0 else 'a gain of ' + money(lo_y)}** and **a gain of {money(hi_y)}**. Roughly 1 year in 6 is worse than that.
- **Sharpe {num(p['Sharpe Ratio'])}:** {cmt.sharpe_label(p['Sharpe Ratio'])}. You earned {num(p['Sharpe Ratio'])} units of return above a {pct(rf)} risk-free deposit for each unit of ups and downs{f" ({bench_name}: {num(b['Sharpe Ratio'])})" if has_bench else ""}.
- **Max drawdown {pct(p['Max Drawdown'])}:** at the worst point you'd have been sitting on a paper loss of **{money(-value * p['Max Drawdown'])}**. Recovery took {recovery_text}. Be honest about whether you'd have held on.
- **1-day VaR {money(value * p[var_key])}:** on about 1 trading day in {days_in} (roughly {252 // days_in} days a year), expect to lose this much or more.

The small coloured numbers compare with {bench_name}; green means better for you (for volatility, drawdown and VaR, lower is better).""")


    # Drawdown chart
    dd = rep["drawdown"]
    fig = go.Figure()
    fig.add_scatter(x=dd.index, y=dd, name="Your portfolio", fill="tozeroy",
                    line=dict(color=BLUE, width=1.5), fillcolor="rgba(42,120,214,0.18)",
                    hovertemplate="%{y:.1%}")
    if has_bench:
        bdd = rk.drawdown(bench.dropna(), start=1)["Drawdown"]
        fig.add_scatter(x=bdd.index, y=bdd, name=bench_name, line=dict(color=ORANGE, width=1.5),
                        hovertemplate="%{y:.1%}")
    fig.update_layout(hovermode="x unified")
    show(style_fig(fig, 320, yfmt=".0%"), "Drawdown: how far below its last high")
    longest = rk.drawdown_duration(port)["Longest"]
    explain("What this means for your portfolio", f"""
Your worst stretch began in **{peak_d:%b %Y}** and hit bottom in **{trough_d:%b %Y}**: {money(value)} would have shrunk to about **{money(value*(1+p['Max Drawdown']))}** ({pct(p['Max Drawdown'])}){f", while {bench_name} fell {pct(b['Max Drawdown'])}" if has_bench else ""}. Recovery took {recovery_text}.
The longest time spent below a previous high was **{longest:,} trading days (about {longest/252:.1f} years)**: that's how long you might wait without seeing a new high.

(The line sits at 0% at every new high; each dip shows how far the portfolio was below its best value so far.)""")

    st.subheader("All statistics")
    show_rows = ["Annualized Return", "Annualized Vol", "Sharpe Ratio", "Sortino Ratio", "Calmar Ratio",
                 "Max Drawdown", var_key, f"Cornish-Fisher VaR ({level}%)", cvar_key, "Skewness",
                 "Kurtosis", "Best Period", "Worst Period", "% Positive Periods"]
    fmt = {k: pct for k in show_rows}
    for k in ["Sharpe Ratio", "Sortino Ratio", "Calmar Ratio", "Skewness", "Kurtosis"]:
        fmt[k] = num
    table = pd.DataFrame({k: summary[k].map(fmt[k]) for k in show_rows}).T
    table.index = [k.replace("Annualized", "Annual").replace("Periods", "days").replace("Period", "day")
                   for k in table.index]
    st.dataframe(table, width="stretch", height=36 * (len(table) + 1) + 3)
    pos_days = p["% Positive Periods"]
    explain("What this table says about your portfolio", f"""
- Your best day was **{pct(p['Best Period'])}** ({money(value*p['Best Period'])}) and your worst **{pct(p['Worst Period'])}** ({money(value*p['Worst Period'])}). **{pct(pos_days, 0)}** of days were up days.
- Skewness {num(p['Skewness'])}: {"big down days have been more common than big up days" if p['Skewness'] < -0.1 else "big up days have been at least as common as big down days" if p['Skewness'] > 0.1 else "up and down days have been fairly balanced"}.
- Kurtosis {num(p['Kurtosis'], 1)}: {"extreme days happen much more often than a bell curve predicts (normal is 3), so plan for surprises" if p['Kurtosis'] > 5 else "extreme days are a bit more common than a bell curve predicts (normal is 3)" if p['Kurtosis'] > 3.5 else "returns have been close to a bell curve"}.
- CVaR {pct(p[cvar_key])}: when a bad day (1 in {days_in}) does happen, the average loss has been about **{money(value*p[cvar_key])}**.

Glossary:
- **Sortino**: like Sharpe, but only counts *downside* swings as risk.
- **Calmar**: yearly return divided by the max drawdown.
- **Cornish-Fisher VaR**: VaR adjusted for fat tails and lopsided returns.
- **CVaR (Expected Shortfall)**: the *average* loss on the days that were worse than the VaR.
- **Skewness**: below 0 means big drops are more common than big jumps.
- **Kurtosis**: 3 for a normal bell curve; above 3 means extreme days happen more often than the bell curve predicts (true for almost every market).""")

# =============================================================================
# MY FUNDS: actual returns, category ranks, tax
# =============================================================================
def _funds_tab():
    if not MF_HELD and TX is None:
        st.info("This tab is for Indian mutual funds. Upload a mutual fund statement, or better, a **detailed CAS** "
                "(Consolidated Account Statement) from CAMS or KFintech: it adds your real returns (XIRR) and a "
                "capital-gains tax view.", icon=":material/info:")
    if RET_TOTAL is not None:
        st.subheader("Your actual returns")
        md(f"<span class='small-note'>From your {len(TX):,} transactions since {RET_TOTAL['first']:%b %Y}: every SIP, "
           "purchase and redemption on its real date.</span>", unsafe_allow_html=True)
        k = st.columns(5)
        k[0].metric("Invested", money(RET_TOTAL["invested"]))
        k[1].metric("Value today", money(RET_TOTAL["value"]))
        k[2].metric("Total gain", money(RET_TOTAL["gain"]),
                    help=f"Includes {money(RET_TOTAL['withdrawn'])} you have withdrawn." if RET_TOTAL["withdrawn"] else None)
        k[3].metric("Your XIRR", pct(RET_TOTAL["xirr"]))
        if RET_TOTAL.get("bench_xirr") is not None:
            d_ = RET_TOTAL["xirr"] - RET_TOTAL["bench_xirr"]
            k[4].metric("Nifty 50 XIRR", pct(RET_TOTAL["bench_xirr"]), f"{d_*100:+.1f} pts for you")
            md(f"<div class='backdrop {'good' if d_ > 0.005 else 'bad' if d_ < -0.005 else 'ok'}'>"
               f"If every rupee had gone into a Nifty 50 index fund on the same dates, you'd have "
               f"<b>{money(RET_TOTAL['bench_value'])}</b> today instead of <b>{money(RET_TOTAL['value'])}</b> "
               f"({'you are ahead by ' if d_ >= 0 else 'you are behind by '}{money(abs(RET_TOTAL['value'] - RET_TOTAL['bench_value']))}).</div>",
               unsafe_allow_html=True)
        rt = RET.copy()
        rt["Fund"] = [label_of.get(f"{rk.MF_PREFIX}{int(c)}", mf.short_label(s_)) if c == c else s_
                      for c, s_ in zip(rt["code"], rt["scheme"])]
        rt = rt[(rt["invested"] > 0)].sort_values("value", ascending=False)
        st.dataframe(pd.DataFrame({
            "Fund": rt["Fund"], "Since": rt["first"].dt.strftime("%b %Y"), "Invested": rt["invested"].map(money),
            "Withdrawn": rt["withdrawn"].map(money), "Value today": rt["value"].map(money),
            "Gain": rt["gain"].map(money), "XIRR": rt["xirr"].map(pct)}), hide_index=True, width="stretch")
        explain("XIRR vs 'return per year'", f"""
**XIRR** is the return on *your* money, given *when* you invested it: an SIP started at a market peak earns less than one started in a crash, even in the same fund. It's the number to compare with an FD or a loan rate.
The **return per year** on the Overview tab measures the funds themselves as if you had invested everything on day one. The two differ when your investments were spread out over time.
The Nifty 50 comparison replays your exact cash flows into a Nifty 50 index fund (UTI, direct plan; its NAV includes dividends).""")

    if PEER:
        st.subheader("How your funds rank in their category")
        md("<span class='small-note'>Each fund against every fund in the same SEBI category, plan (direct or regular) "
           "and option, from AMFI's daily NAVs. Rank 1 is best.</span>", unsafe_allow_html=True)

        def vs(r_, col, fmt=pct):
            x = r_["ranks"].get(col)
            return "–" if not x else f"{fmt(x['value'])} (#{x['rank']}/{x['n']})"
        tbl = pd.DataFrame([{
            "Fund": n_, "Category": pr["category"].split(" - ")[-1], "Verdict": pr["label"],
            "1Y": vs(pr, "1Y"), "3Y": vs(pr, "3Y"), "5Y": vs(pr, "5Y"),
            "Sharpe 3Y": vs(pr, "Sharpe 3Y", num), "Worst fall 3Y": vs(pr, "Worst fall 3Y"),
            "Category median 3Y": pct(pr["ranks"]["3Y"]["median"]) if pr["ranks"].get("3Y") else "–",
            "Extra cost vs direct": f"{GAP[n_]['gap']*100:.2f}% / yr" if n_ in GAP else "–"}
            for n_, pr in PEER.items()])
        st.dataframe(tbl, hide_index=True, width="stretch",
                     column_config={"Verdict": st.column_config.TextColumn(help="Average percentile of 3- and 5-year "
                                                                                 "return and 3-year Sharpe in the category.")})
        pick = st.selectbox("See the whole category for", list(PEER), key="peer_pick")
        pr = PEER[pick]
        met_all = pr["metrics"].dropna(subset=["3Y"]).sort_values("3Y")
        met = met_all
        if len(met_all) > 15:
            keep = set(met_all.index[-10:]) | set(met_all.index[:3]) | {pr["code"]}
            met = met_all[met_all.index.isin(keep)]
        if len(met):
            me = pr["code"]
            fig = go.Figure(go.Bar(
                x=met["3Y"], y=[mf.short_label(n) if isinstance(n, str) else str(c) for c, n in zip(met.index, met["name"])],
                orientation="h", marker=dict(color=[BLUE if c == me else "#c4cad4" for c in met.index], cornerradius=3),
                hovertemplate="%{y}: %{x:.1%} a year<extra></extra>"))
            fig.add_vline(x=met_all["3Y"].median(), line=dict(color=ORANGE, width=2, dash="dot"),
                          annotation_text="median", annotation_position="bottom right")
            show(style_fig(fig, 60 + 20 * len(met), xfmt=".0%", legend=False),
                 f"3-year return per year · {pr['category'].split(' - ')[-1]} · {pr['plan']} "
                 + (f"(top 10, yours and bottom 3 of {len(met_all)})" if len(met) < len(met_all) else f"({len(met)} funds)"))
        if GAP:
            cost_now = sum(value * w[n_] * g_["gap"] for n_, g_ in GAP.items())
            md(f"<div class='backdrop bad'><b>Regular plans cost you about {money(cost_now)} a year.</b> Measured from "
               "AMFI NAVs: the direct plan of each of these funds grew faster by the amount shown under "
               "'Extra cost vs direct' (that difference is the distributor commission).</div>", unsafe_allow_html=True)

    if TAX is not None:
        st.subheader("Tax if you sold today")
        c1, c2 = st.columns(2)
        c1.selectbox("Your income-tax slab", [0, 5, 10, 15, 20, 25, 30], index=6, key="tax_slab",
                     format_func=lambda x: f"{x}%", help="Used for short-term gains on debt funds and some others.")
        c2.number_input("Equity long-term gains already booked elsewhere this year (₹)", 0, 10_000_000, 0, 5_000,
                        key="tax_ltcg_used", help="Shares the ₹1.25 lakh yearly exemption.")
        a_ = TAX["all"]
        k = st.columns(4)
        k[0].metric("Short-term gains", money(a_["eq_st"] + a_["slab"]))
        k[1].metric("Long-term gains", money(a_["eq_lt"] + a_["other_lt"]))
        k[2].metric("Tax if all sold", money(a_["tax"]))
        k[3].metric("Tax-free room", money(TAX["room"]), help=f"Unused ₹1.25 lakh exemption for FY {TAX['fy']}-{str(TAX['fy'] + 1)[2:]}.",
                    delta=f"{money(TAX['booked_eq_lt'])} already booked" if TAX["booked_eq_lt"] else None, delta_color="off")
        if TAX["harvest_gain"] > 1000:
            md(f"<div class='backdrop good'><b>Harvest {money(TAX['harvest_gain'])} of gains tax-free before 31 March.</b> "
               f"Sell units worth about {money(TAX['harvest_value'])} from your oldest long-term equity units and buy "
               f"them back. Your cost goes up, saving about {money(TAX['harvest_gain'] * txm.EQ_LTCG * (1 + txm.CESS))} of "
               "future tax. Unused exemption is lost each year. Check exit loads first.</div>", unsafe_allow_html=True)
        tr = TAX["rows"]
        st.dataframe(pd.DataFrame({
            "Fund": tr["Fund"], "Tax type": tr["Tax type"].str.capitalize(), "Value": tr["Value"].map(money),
            "Short-term gain": tr["Short-term gain"].map(money), "Long-term gain": tr["Long-term gain"].map(money),
            "Bought < 1 year ago (exit load?)": tr["Exit-load window"].map(money),
            "Locked (ELSS)": tr["Locked (ELSS)"].map(money)}), hide_index=True, width="stretch")
        with st.expander("Tax type per fund (change if needed)", icon=":material/tune:"):
            for c_, k_ in TAX["klass"].items():
                st.selectbox(label_of.get(f"{rk.MF_PREFIX}{c_}", str(c_)), ["equity", "debt", "other"],
                             index=["equity", "debt", "other"].index(k_), key=f"taxclass_{c_}")
            st.caption("Equity: equity funds, aggressive hybrid, arbitrage, most balanced advantage funds. Debt: debt "
                       "funds and conservative hybrid. Other: gold, international, multi-asset and similar.")
        st.caption("Rules for sales from 23 July 2024: equity funds 20% short term (1 year or less), 12.5% long term "
                   "above ₹1.25 lakh a year; debt funds bought from April 2023 at your slab; others 12.5% after 2 years, "
                   "else slab; 4% cess; gains up to 31 Jan 2018 grandfathered. Surcharge, carried-forward losses and exit "
                   "loads aren't included. An estimate, not tax advice.")



if SHOW_FUNDS:
    with t_funds:
        _funds_tab()

# =============================================================================
# MARKETS TODAY
# =============================================================================
def fmt_level(t, v):
    if t in ("^TNX",):
        return f"{v:.2f}%"
    if t in ("^INDIAVIX", "^VIX", "DX-Y.NYB"):
        return f"{v:.1f}"
    if t == "INR=X":
        return f"₹{v:.2f}"
    return f"{v:,.0f}" if v >= 1000 else f"{v:,.2f}"


def metric_row(items, data, stats_by, per_row=5, period="1D"):
    cols = st.columns(min(per_row, len(items)))
    for i, (t, label) in enumerate(items):
        stt = stats_by.get(t)
        if stt is None:
            continue
        points = t in ("^TNX", "^INDIAVIX", "^VIX")
        if points:
            prev = stt["last"] / (1 + stt[period])
            delta = f"{stt['last'] - prev:+.2f} pts"
        else:
            delta = f"{stt[period]*100:+.2f}%"
        cols[i % len(cols)].metric(
            label, fmt_level(t, stt["last"]), f"{delta} ({period})",
            delta_color="inverse" if t in ("^INDIAVIX", "^VIX", "^TNX") else "off" if t in ("INR=X", "DX-Y.NYB") else "normal",
            chart_data=data[t].dropna().iloc[-63:].round(2).tolist(), chart_type="line")


def group_bars(group, stats_by, period):
    rows_ = [(label, stats_by[t][period]) for t, label in group.items() if stats_by.get(t) and pd.notna(stats_by[t][period])]
    if not rows_:
        return None
    df = pd.DataFrame(rows_, columns=["Name", "Return"]).sort_values("Return")
    fig = go.Figure(go.Bar(
        x=df["Return"], y=df["Name"], orientation="h",
        marker=dict(color=[BLUE if r >= 0 else ORANGE for r in df["Return"]], cornerradius=4),
        text=[f"{r*100:+.1f}%" for r in df["Return"]], textposition="outside", cliponaxis=False,
        textfont=dict(color=INK_2, size=12), hovertemplate="%{y}: %{x:+.1%}<extra></extra>"))
    lim = max(abs(df["Return"]).max() * 1.3, 0.01)
    fig = style_fig(fig, 34 * len(df) + 40, xfmt=".0%", legend=False)
    fig.update_xaxes(range=[-lim if df["Return"].min() < 0 else 0, lim if df["Return"].max() > 0 else 0.002],
                     zeroline=True, zerolinecolor=MUTED)
    return fig


with t_mkt:
    default_mkt = home_market if uploads else "India"
    mc = st.columns([3, 2])
    with mc[0]:
        choice = st.segmented_control("Market", ["India", "Global", "US"], default=default_mkt, key="mkt_choice",
                                      label_visibility="collapsed") or default_mkt
    with mc[1]:
        md(f"<div style='text-align:right' class='small-note'>"
           + (f"Matched to your holdings: <b>{home_market}</b>" if uploads else "Default view: India")
           + "</div>", unsafe_allow_html=True)
    try:
        with st.spinner("Loading market data..."):
            mdata = market_data(choice)
    except Exception:
        mdata = None
    if mdata is None or mdata.empty:
        st.warning("Market data from Yahoo Finance isn't available right now. Try again in a few minutes.")
    else:
        cfg = market.MARKETS[choice]
        stats_by = {t: market.stats(mdata[t]) for t in mdata.columns}
        reg = market_reading(choice, mdata)
        beta_here = capm_p["Beta"] if (capm_p is not None and choice == home_market) else None
        means = market.implications(reg, beta_here, value, money)
        as_of = stats_by[cfg["main"]]["date"]
        main_stats = stats_by[cfg["main"]]
        md(f"""<div class='mood'>
  <div><div class='eyebrow'>Market mood · {reg['name']} · {as_of:%d %b %Y}</div>
       <div class='label {reg['tone']}'>{reg['mood']}</div>
       <ul>{''.join(f'<li>{n}</li>' for n in reg['notes'])}</ul></div>
  <div class='tiles'>
    <div class='tile'><div class='l'>Trend</div><div class='v'>{reg['trend']}</div>
      <div class='d'>vs 50 & 200-day averages</div></div>
    <div class='tile'><div class='l'>From 52-week high</div><div class='v'>{reg['from_high']*100:.1f}%</div>
      <div class='d'>{reg['drawdown']}</div></div>
    <div class='tile'><div class='l'>Volatility</div><div class='v'>{reg['volatility']}</div>
      <div class='d'>{f"{reg['vix_ratio']:.1f}× usual" if reg['vix_ratio'] else ""}</div></div>
    <div class='tile'><div class='l'>1 month</div><div class='v'>{main_stats['1M']*100:+.1f}%</div><div class='d'>{reg['name']}</div></div>
    <div class='tile'><div class='l'>Year to date</div><div class='v'>{main_stats['YTD']*100:+.1f}%</div><div class='d'>{reg['name']}</div></div>
    <div class='tile'><div class='l'>1 year</div><div class='v'>{main_stats['1Y']*100:+.1f}%</div><div class='d'>{reg['name']}</div></div>
  </div>
  <div class='means'><b>What it means for you:</b> {' '.join(means)}</div>
</div>""", unsafe_allow_html=True)

        metric_row(list(cfg["headline"].items()), mdata, stats_by)

        st.subheader("How the market has moved")
        period = st.segmented_control("Period", list(market.PERIODS), default="1M", key="mkt_period",
                                      label_visibility="collapsed") or "1M"
        fig = go.Figure()
        for i, t in enumerate(cfg["chart"]):
            s_ = mdata[t].dropna().iloc[-252:]
            if len(s_) < 30:
                continue
            fig.add_scatter(x=s_.index, y=s_ / s_.iloc[0] * 100, name=cfg["headline"].get(t, t),
                            line=dict(color=SERIES[i], width=2), hovertemplate="%{y:.1f}")
        fig.add_hline(y=100, line=dict(color=MUTED, width=1, dash="dot"))
        fig.update_layout(hovermode="x unified")
        show(style_fig(fig, 340), "Last 12 months (start = 100)")

        gcols = st.columns(len(cfg["groups"]))
        for col, (gname, group) in zip(gcols, cfg["groups"].items()):
            with col:
                gfig = group_bars(group, stats_by, period)
                if gfig is not None:
                    show(gfig, f"{gname} · {period}")
        if "Mutual fund categories" in cfg["groups"]:
            st.caption("Fund categories are tracked with index ETFs as stand-ins (e.g. Nifty BeES for large cap, "
                       "gilt BeES for government bonds), so they show the category's market move before fund costs.")

        st.subheader("Rates, currency and commodities")
        metric_row(list(cfg["macro"].items()), mdata, stats_by, period="1M")
        if choice != "Global":
            st.subheader("Global backdrop")
            metric_row(list(market.GLOBAL_STRIP.items()), mdata, stats_by, per_row=7)

        with st.expander("All numbers", icon=":material/table:"):
            names = {**market.GLOBAL_STRIP, **cfg["macro"], **{k: v for g in cfg["groups"].values() for k, v in g.items()},
                     **cfg["headline"]}
            tbl = pd.DataFrame([{"Name": names[t], "Level": fmt_level(t, stats_by[t]["last"]),
                                 **{k: stats_by[t][k] for k in ["1D", "1W", "1M", "3M", "YTD", "1Y", "from_high"]}}
                                for t in names if stats_by.get(t)])
            pc = {k: st.column_config.NumberColumn(k, format="percent") for k in ["1D", "1W", "1M", "3M", "YTD", "1Y"]}
            pc["from_high"] = st.column_config.NumberColumn("From 52w high", format="percent")
            st.dataframe(tbl, hide_index=True, width="stretch", column_config=pc, height=min(36 * len(tbl) + 40, 600))
        st.caption(f"Data: Yahoo Finance daily closes up to {as_of:%d %b %Y} (delayed, refreshed every 30 minutes). "
                   "Indices are price indices (they exclude dividends). Mood rules: trend vs the 50- and 200-day "
                   "averages, distance from the 52-week high, and the volatility index vs its one-year median "
                   "(see market.py). Not a forecast.")


# =============================================================================
# FUTURE
# =============================================================================
with t_future:
    st.subheader("Where could this portfolio be in a few years?")
    md(f"If you leave **{money(value)}** in this exact mix and the future behaves like the past "
       f"**{years:.1f} years** did, here is the range of outcomes.")
    horizon = st.slider("Years from now", 1, 10, 5, help="Drag to see the outlook over 1 to 10 years.")
    if len(port.dropna()) < 260:
        st.info("Needs at least about a year of history to simulate the future.")
    else:
        with st.spinner("Simulating 2,000 possible futures..."):
            q, loss_p, cash_p = future_bands(port.dropna(), rf)
        i = horizon * 12
        good, mid, bad = q[90][i] * value, q[50][i] * value, q[10][i] * value
        cagr = lambda x: (x / value) ** (1 / horizon) - 1  # noqa: E731
        c1, c2, c3 = st.columns(3)
        with c1:
            md(f"<div class='case good'><span class='lbl'>Good case</span><span class='big'>{money(good)}</span>"
                        f"{pct(cagr(good))} a year · gain of {money(good - value)}<br>"
                        f"<span class='small-note'>1 in 10 futures did even better</span></div>", unsafe_allow_html=True)
        with c2:
            md(f"<div class='case mid'><span class='lbl'>Middle of the range</span><span class='big'>{money(mid)}</span>"
                        f"{pct(cagr(mid))} a year<br><span class='small-note'>half did better, half worse</span></div>",
                        unsafe_allow_html=True)
        with c3:
            md(f"<div class='case bad'><span class='lbl'>Bad case</span><span class='big'>{money(bad)}</span>"
                        f"{pct(cagr(bad))} a year · {'loss' if bad < value else 'gain'} of {money(abs(bad - value))}<br>"
                        f"<span class='small-note'>1 in 10 futures did even worse</span></div>", unsafe_allow_html=True)
        st.write("")
        md(f"After **{horizon} year{'s' if horizon > 1 else ''}**: a **{pct(loss_p[horizon-1], 0)}** chance of having "
           f"less than you started with, and a **{pct(cash_p[horizon-1], 0)}** chance of beating a risk-free "
           f"deposit at {pct(rf)} a year ({money(value * (1 + rf) ** horizon)}).")

        dates = pd.date_range(t1, periods=i + 1, freq="MS")
        fig = go.Figure()
        fig.add_scatter(x=dates, y=q[90][:i+1] * value, line=dict(width=0), hoverinfo="skip", showlegend=False)
        fig.add_scatter(x=dates, y=q[10][:i+1] * value, fill="tonexty", fillcolor="rgba(42,120,214,0.12)",
                        line=dict(width=0), name="Good to bad case (80% of futures)", hoverinfo="skip")
        fig.add_scatter(x=dates, y=q[75][:i+1] * value, line=dict(width=0), hoverinfo="skip", showlegend=False)
        fig.add_scatter(x=dates, y=q[25][:i+1] * value, fill="tonexty", fillcolor="rgba(42,120,214,0.25)",
                        line=dict(width=0), name="Middle half of futures", hoverinfo="skip")
        fig.add_scatter(x=dates, y=q[90][:i+1] * value, name="Good case", line=dict(color=GREEN, width=2, dash="dot"),
                        hovertemplate=CUR + "%{y:,.0f}")
        fig.add_scatter(x=dates, y=q[50][:i+1] * value, name="Middle", line=dict(color=BLUE, width=2.5),
                        hovertemplate=CUR + "%{y:,.0f}")
        fig.add_scatter(x=dates, y=q[10][:i+1] * value, name="Bad case", line=dict(color=CRITICAL, width=2, dash="dot"),
                        hovertemplate=CUR + "%{y:,.0f}")
        fig.add_scatter(x=[dates[0], dates[-1]], y=[value, value], name="What you put in", mode="lines",
                        line=dict(color=INK_2, width=1, dash="dash"), hoverinfo="skip")
        fig.update_layout(hovermode="x unified", legend=dict(traceorder="normal"))
        lo_y, hi_y = min(q[10][:i+1].min(), 1) * value, q[90][:i+1].max() * value
        fig = style_fig(fig, 420, yfmt="$,.0f")
        fig.update_yaxes(range=[lo_y - (hi_y - lo_y) * 0.05, hi_y + (hi_y - lo_y) * 0.14])
        show(fig, f"Possible value of {money(value)} over the next {horizon} years")

        explain("What this means for you", f"""
- **Good case ({money(good)}):** markets are kind and your mix grows about {pct(cagr(good))} a year. Don't count on it: only 1 in 10 simulated futures did better.
- **Bad case ({money(bad)}):** a rough patch like the worst stretches in your history. {f"You'd be down {money(value - bad)} after {horizon} year{'s' if horizon > 1 else ''}." if bad < value else f"Even here you'd still be up {money(bad - value)}."} 1 in 10 futures were worse still, so make sure you could live with this.
- **Time helps:** the chance of being below what you put in falls from **{pct(loss_p[0], 0)}** after 1 year to **{pct(loss_p[min(9, len(loss_p)-1)], 0)}** after 10 years, because good and bad years tend to even out.

**How it's calculated:** we replay random one-month stretches of this portfolio's real daily returns from {t0:%b %Y} to {t1:%b %Y}, end to end, 2,000 times. That keeps the real crashes and calm spells instead of assuming a bell curve. It assumes the mix doesn't change and the future looks like that past period.
{f"**Caution:** your history is only {years:.1f} years long, so these ranges lean heavily on how that short period went ({pct(p['Annualized Return'])} a year). Treat them as rough." if years < 5 else ""}
This is a simulation, not a forecast or a promise.""")

# =============================================================================
# WHAT SHOULD I DO?
# =============================================================================
with t_plan:
    st.subheader("Your action plan")
    md("<span class='small-note'>Rules of thumb applied to your numbers, your goal and today's market. General guidance, "
       "not personal financial advice: check costs, exit loads and tax before acting.</span>", unsafe_allow_html=True)

    with st.container(border=True):
        md("<b>About you</b> <span class='small-note'>· sets your target mix and tailors the steps. "
           "Stays in this browser session.</span>", unsafe_allow_html=True)
        pc = st.columns([1.55, 1.05, 0.95, 0.9])
        pc[0].segmented_control("When will you need most of this money?", advisor.HORIZONS, key="pf_horizon",
                                default="5–10 years")
        pc[1].segmented_control("How much ups and downs can you take?", advisor.COMFORTS, key="pf_comfort",
                                default="Balanced", help="Careful: a 15% fall would worry you. Balanced: you can sit "
                                "through 25–30%. Growth: you can hold on through a 40%+ crash.")
        pc[2].segmented_control("Emergency fund in place?", ["Yes", "No", "Not sure"], key="pf_emergency",
                                default="Not sure", help="6 months of expenses kept somewhere safe and easy to reach.")
        pc[3].number_input(f"Monthly investing ({CUR})", min_value=0, step=1000 if CUR == "₹" else 100,
                           key="pf_monthly", help="Your SIPs or regular contributions. Optional.")

    def _done_key(t):
        return "done_" + "".join(ch if ch.isalnum() else "_" for ch in t)[:60]

    n_done = sum(bool(st.session_state.get(_done_key(x["title"]))) for x in actions)
    saving = sum(x.get("yearly", 0) or 0 for x in actions)
    kc = st.columns(4)
    kc[0].metric("Steps", len(actions))
    kc[1].metric("Urgent", sum(x["priority"] == "High" for x in actions))
    kc[2].metric("Cost you can cut", f"{money(saving)}/yr" if saving else "–",
                 help="Yearly saving from moving regular plans to direct plans, at today's value.")
    kc[3].metric("Done", f"{n_done} of {len(actions)}")
    if actions:
        st.progress(n_done / len(actions))
    if mkt_reg:
        md(f"<div class='backdrop {mkt_reg['tone']}'><b>Market backdrop · {mkt_reg['name']}: {mkt_reg['mood']}.</b> "
           f"{mkt_reg['trend']}, {abs(mkt_reg['from_high'])*100:.1f}% below its 52-week high, volatility "
           f"{mkt_reg['volatility'].lower()}. {mkt_reg['implications'][0]}</div>", unsafe_allow_html=True)

    p_steps, p_mix, p_each, p_chat = st.tabs(["Steps", "Target mix", "Each holding", "Ask the adviser"])

    with p_steps:
        for when in advisor.WHEN_ORDER:
            group = [x for x in actions if x.get("when", "This month") == when]
            if not group:
                continue
            md(f"<div class='when'>{when}</div>", unsafe_allow_html=True)
            cols = st.columns(2)
            for i, x in enumerate(group):
                with cols[i % 2]:
                    k = _done_key(x["title"])
                    done = bool(st.session_state.get(k))
                    impact = f"<span class='impact'>{x['impact']}</span>" if x.get("impact") else ""
                    md(f"<div class='action{' done' if done else ''}'><div class='head'>"
                       f"<span class='prio {x['priority']}'>{x['priority']}</span><span class='t'>{x['title']}</span>"
                       f"{impact}</div><div class='why'>{x['why']}</div><div class='how'>{x['how']}</div></div>",
                       unsafe_allow_html=True)
                    if x.get("steps"):
                        with st.expander("How to do it", icon=":material/checklist:"):
                            md("\n".join(f"{j}. {stp}" for j, stp in enumerate(x["steps"], 1)))
                    st.checkbox("Done", key=k)
        good_pts = [pt["title"] for pt in verdict["points"] if pt["status"] == "good" and pt["scored"]]
        if good_pts:
            md("<div style='margin-top:6px'><b>Keep doing</b><br>" + "".join(f"<span class='keep'>{g}</span>" for g in good_pts)
               + "</div>", unsafe_allow_html=True)

    with p_mix:
        cls_lbl = advisor.class_labels(CUR)
        mix_now = advisor.current_mix(hold["weight"].to_dict(), CLASS_OF)
        mix_tgt = advisor.target_mix(PROFILE["horizon"], PROFILE["comfort"], CUR)
        gaps = advisor.mix_gaps(mix_now, mix_tgt, value)
        eq_now, eq_tgt = mix_now["home"] + mix_now["intl"], mix_tgt["home"] + mix_tgt["intl"]
        off = [g for g in gaps if abs(g["gap"]) >= 0.05]
        md(f"**For a {advisor.GOAL_PHRASE[PROFILE['horizon']]} goal with {PROFILE['comfort'].lower()} risk comfort**, a sensible "
           f"starting point is about **{eq_tgt*100:.0f}% in shares**. You have **{eq_now*100:.0f}%**. "
           + ("Your mix is close to target: no moves needed." if not off else
              f"{len(off)} part{'s are' if len(off) > 1 else ' is'} more than 5 points off."))
        order_ = list(advisor.CLASSES)[::-1]
        fig = go.Figure()
        fig.add_bar(y=[cls_lbl[c] for c in order_], x=[mix_now[c] for c in order_], name="You now",
                    orientation="h", marker_color=NAVY, text=[f"{mix_now[c]*100:.0f}%" for c in order_],
                    textposition="outside")
        fig.add_bar(y=[cls_lbl[c] for c in order_], x=[mix_tgt[c] for c in order_], name="Target",
                    orientation="h", marker_color="#9db4d9", text=[f"{mix_tgt[c]*100:.0f}%" for c in order_],
                    textposition="outside")
        fig.update_layout(barmode="group")
        fig = style_fig(fig, height=300, xfmt=".0%")
        fig.update_xaxes(range=[0, max(max(mix_now.values()), max(mix_tgt.values())) * 1.2])
        show(fig, "Your mix vs target")
        if off:
            md("<b>Moves to get there</b>", unsafe_allow_html=True)
            for g in off:
                verb = "Add about" if g["gap"] > 0 else "Reduce by about"
                md(f"- **{cls_lbl[g['cls']]}:** {verb} {money(abs(g['amount']))} "
                   f"({g['current']*100:.0f}% → {g['target']*100:.0f}%)")
            md("<span class='small-note'>Cheapest way first: send new " + ("SIPs" if CUR == "₹" else "money")
               + " to what's under target. Then switch from what's over, starting with units held over a year"
               + (" and gains inside the ₹1.25 lakh tax-free limit" if CUR == "₹" else "") + ". Test the result in "
               "<b>Try ideas → Change my portfolio</b>.</span>", unsafe_allow_html=True)
        with st.expander("What counts as what", icon=":material/category:"):
            cls_rows = []
            for n_, cl in CLASS_OF.items():
                cls_rows.append({"Holding": n_, "Counted as": " + ".join(
                    f"{cls_lbl[c]}" + (f" {v*100:.0f}%" if v < 1 else "") for c, v in cl.items()),
                    "% of money": hold.loc[n_, "weight"] * 100 if n_ in hold.index else np.nan})
            st.dataframe(pd.DataFrame(cls_rows), hide_index=True, width="stretch",
                         column_config={"% of money": st.column_config.NumberColumn(format="%.1f%%")})
            st.caption("From each fund's SEBI category and name. Hybrid funds are split by their usual mix; "
                       "flexi-cap funds' small overseas slice is ignored.")
        with st.expander("How the target is set", icon=":material/info:"):
            tbl = pd.DataFrame(advisor.EQUITY_BY, index=advisor.HORIZONS).T.map(lambda v: f"{v*100:.0f}%")
            md("Share in shares (home + international) by risk comfort and when you need the money:")
            st.dataframe(tbl, width="stretch")
            md("The rest goes to " + cls_lbl["debt"].lower() + " and " + cls_lbl["gold"].lower() + " ("
               + ("10%, or 5% for goals under 3 years" if CUR == "₹" else "5%") + "). International shares are "
               + ("15%" if CUR == "₹" else "30%") + " of the shares slice. These are common rules of thumb: the "
               "longer you can wait, the more a fall has time to recover.")

    with p_each:
        rev = advisor.holding_review(hold, rep["correlation"], years, CLASS_OF)
        tags = {"Switch": "🔁 Switch", "Merge": "🔗 Merge", "Review": "🔍 Review", "Watch": "👀 Watch", "Keep": "✅ Keep"}
        cnt = rev["Verdict"].value_counts()
        md(" · ".join(f"**{cnt[v_]}** {v_.lower()}" for v_ in tags if v_ in cnt))
        show_rev = rev.assign(Verdict=rev["Verdict"].map(tags), Money=rev["Money"] * 100, Risk=rev["Risk"] * 100,
                              **{"Return / yr": rev["Return / yr"] * 100})
        st.dataframe(show_rev, hide_index=True, width="stretch", column_config={
            "Money": st.column_config.NumberColumn("% of money", format="%.1f%%"),
            "Risk": st.column_config.NumberColumn("% of risk", format="%.1f%%"),
            "Return / yr": st.column_config.NumberColumn(format="%.1f%%"),
            "Sharpe": st.column_config.NumberColumn(format="%.2f"),
            "Why": st.column_config.TextColumn(width="large")})
        md("<span class='small-note'><b>Switch</b>: same fund, cheaper plan. <b>Merge</b>: a near-copy of a better "
           "holding. <b>Review</b>: weak against its peers or the safe rate. <b>Watch</b>: fine, but carries outsized "
           "risk. <b>Keep</b>: nothing to fix.</span>", unsafe_allow_html=True)

    with p_chat:
        md("<span class='small-note'>Chat with Claude acting as a senior portfolio adviser. It sees the holdings, "
           "statistics and your answers above, but not your name, PAN, account numbers or contact details, and "
           "sends them to Anthropic's API to answer. General guidance, not regulated financial advice.</span>",
           unsafe_allow_html=True)

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            try:
                api_key = st.secrets.get("ANTHROPIC_API_KEY")
            except Exception:
                api_key = None
        if not api_key:
            with st.expander("Get full, conversational answers from Claude (optional)", icon=":material/key:"):
                api_key = st.text_input("Anthropic API key", type="password", key="adv_api_key",
                                        help="Kept only in this browser session. Or set ANTHROPIC_API_KEY before starting the app.")
                st.caption("Without a key, the adviser answers common questions from your numbers using built-in rules. "
                           "Get a key at console.anthropic.com; each question costs about 1–3 US cents.")
        if True:  # the chat is always shown; it uses Claude when a key is set
            import anthropic
            brief = advisor.portfolio_brief(facts, hold, rep["correlation"], verdict, actions, future_rows, CUR,
                                            dt.date.today(), market=mkt_reg, extra=EXTRA,
                                            profile=PROFILE, classes=CLASS_OF)
            sys_prompt = advisor.system_prompt(brief, CUR)
            chat_id = str(hash((tuple(rows.index), tuple(np.round(rows.values, 2)), str(t0), round(value))))
            if st.session_state.get("adv_chat_id") != chat_id:
                st.session_state.adv_chat_id = chat_id
                st.session_state.adv_msgs = []
            msgs = st.session_state.adv_msgs

            suggestions = ["What should I do first?"]
            _titles = " ".join(x["title"] for x in actions)
            for _k, _q in [("direct", "Which regular fund should I switch first, given tax?"),
                           ("target mix", "How do I reach my target mix without paying much tax?"),
                           ("look-alike", "Which of my look-alike funds should I keep?"),
                           ("Review ", "Should I really replace my weakest fund?"),
                           ("single shares", "How should I cut my biggest share holdings?"),
                           ("emergency", "How big should my emergency fund be?")]:
                if _k in _titles and len(suggestions) < 3:
                    suggestions.append(_q)
            suggestions += ["How can I make this portfolio safer?", f"Is this right for a {advisor.GOAL_PHRASE[PROFILE['horizon']]} goal?"]
            suggestions = suggestions[:4]
            if not msgs:
                sc = st.columns(len(suggestions))
                for col, q in zip(sc, suggestions):
                    if col.button(q, use_container_width=True):
                        st.session_state.adv_pending = q
            for msg in msgs:
                with st.chat_message(msg["role"], avatar=":material/person:" if msg["role"] == "user" else ":material/support_agent:"):
                    st.markdown(msg["content"].replace("$", "\\$"))

            question = st.chat_input("Ask about your portfolio, e.g. 'Which fund would you sell?'", key="adv_input")
            question = question or st.session_state.pop("adv_pending", None)
            if question:
                msgs.append({"role": "user", "content": question})
                with st.chat_message("user", avatar=":material/person:"):
                    st.markdown(question.replace("$", "\\$"))
                with st.chat_message("assistant", avatar=":material/support_agent:"):
                    if not api_key:
                        answer = advisor.offline_reply(question, facts, hold, actions, verdict, future_rows, market=mkt_reg)
                        st.markdown(answer.replace("$", "\\$"))
                        msgs.append({"role": "assistant", "content": answer})
                    else:
                        try:
                            client = anthropic.Anthropic(api_key=api_key)
                            answer = st.write_stream(escape_dollars(advisor.stream_reply(client, sys_prompt, msgs)))
                            msgs.append({"role": "assistant", "content": answer.replace("\\$", "$")})
                        except anthropic.AuthenticationError:
                            msgs.pop()
                            st.error("That API key was rejected. Check it and try again.")
                        except anthropic.RateLimitError:
                            msgs.pop()
                            st.error("Too many requests right now. Wait a minute and ask again.")
                        except anthropic.APIConnectionError:
                            msgs.pop()
                            st.error("Couldn't reach Anthropic's API. Check your internet connection.")
                        except anthropic.APIStatusError as e:
                            msgs.pop()
                            st.error(f"The adviser is unavailable right now ({e.status_code}). Please try again.")
            if msgs and st.button("Clear chat"):
                st.session_state.adv_msgs = []
                st.rerun()

# =============================================================================
# RISK
# =============================================================================
with t_risk:
    st.subheader("How much could you lose in one day?")
    pr = port.dropna()
    mc = rk.var_monte_carlo(pr, level, 50_000, seed=42)
    methods = pd.DataFrame({
        "Method": ["Historic", "Normal (Gaussian)", "Cornish-Fisher", "Monte Carlo (bootstrap)", "CVaR / Expected Shortfall"],
        "1-day loss": [rk.var_historic(pr, level), rk.var_gaussian(pr, level),
                       rk.var_cornish_fisher(pr, level), mc["VaR"], rk.cvar_historic(pr, level)],
        "How it works": ["The actual worst days in your history",
                         "Assumes a bell curve (underestimates crashes)",
                         "Bell curve corrected for fat tails and skew",
                         "Re-samples 50,000 random past days",
                         "Average loss on the days worse than historic VaR"],
    })
    methods["Amount"] = methods["1-day loss"] * value
    methods["Over 10 days (≈)"] = rk.scale_var(methods["Amount"], 10)

    if True:
        fig = go.Figure(go.Bar(
            y=methods["Method"], x=methods["Amount"], orientation="h",
            marker=dict(color=[BLUE]*4 + [ORANGE], cornerradius=4),
            text=[money(v) for v in methods["Amount"]], textposition="outside",
            hovertemplate="%{y}: " + CUR + "%{x:,.0f}<extra></extra>"))
        fig.update_yaxes(autorange="reversed")
        fig.update_xaxes(range=[0, methods["Amount"].max() * 1.25])
        show(style_fig(fig, 280, xfmt="$,.0f", legend=False), f"1-day loss at {conf} confidence on {money(value)}")
        st.dataframe(methods.assign(**{
            "1-day loss": methods["1-day loss"].map(lambda x: pct(x, 2)),
            "Amount": methods["Amount"].map(money),
            "Over 10 days (≈)": methods["Over 10 days (≈)"].map(money)}),
            hide_index=True, width="stretch",
            column_order=["Method", "1-day loss", "Amount", "Over 10 days (≈)", "How it works"])
    _h, _n, _c = methods["Amount"].iloc[0], methods["Amount"].iloc[1], methods["Amount"].iloc[4]
    explain("What this means for your portfolio", f"""
- On roughly **{252 // days_in} trading days a year**, expect to lose **{money(_h)} or more** in a single day.
- When one of those bad days happens, the **average** loss is about **{money(_c)}** (CVaR, orange). That's the better number to plan around.
- The bell-curve method says {money(_n)}, which is {"**lower** than your real history: it understates your risk by about " + pct(1 - _n/_h, 0) if _n < _h * 0.97 else "**higher** than your real history, so it's a cautious estimate here" if _n > _h * 1.03 else "close to your real history"}.
- Over a bad **two weeks** (10 trading days), a loss of around **{money(methods['Over 10 days (≈)'].iloc[0])}** would not be unusual.

(VaR answers "how much could I lose on a bad day?". The five methods are different ways of estimating it.)""")

    # Distribution
    st.subheader("What normal and bad days look like")
    var_h = rk.var_historic(pr, level)
    fig = go.Figure()
    fig.add_histogram(x=pr, nbinsx=120, histnorm="probability density", name="Actual daily returns",
                      marker=dict(color=BLUE, line=dict(color=SURFACE, width=1)), opacity=0.85,
                      hovertemplate="%{x:.2%}<extra></extra>")
    xs = np.linspace(pr.quantile(0.001), pr.quantile(0.999), 300)
    fig.add_scatter(x=xs, y=norm.pdf(xs, pr.mean(), pr.std()), name="Bell curve (normal)",
                    line=dict(color=INK_2, width=2, dash="dash"), hoverinfo="skip")
    fig.add_vline(x=-var_h, line=dict(color=CRITICAL, width=2))
    fig.add_annotation(x=-var_h, y=1, yref="paper", text=f"VaR {pct(-var_h, 2)}", showarrow=False,
                       xanchor="right", xshift=-6, font=dict(color=CRITICAL, size=12))
    fig.update_xaxes(range=[pr.quantile(0.002), pr.quantile(0.998)], title="Daily return")
    fig.update_yaxes(showticklabels=False, title="How often")
    show(style_fig(fig, 340, xfmt=".1%"))
    explain("What this means for your portfolio", f"""
Most days are quiet: half of all days moved less than **{pct(pr.abs().median(), 2)}** ({money(value * pr.abs().median())}).
But the tails are real: your worst day was **{pct(pr.min())}** ({money(value * pr.min())}), and there were **{int((pr < -2 * pr.std()).sum())}** days with losses bigger than {pct(2 * pr.std())}, where a bell curve would expect about {int(len(pr) * 0.0228)}.
Everything left of the red line is the worst {level}% of your days.

(Each bar counts how often a daily return of that size happened; the dashed line is a perfect bell curve for comparison.)""")

    # Backtest
    st.subheader("Did VaR actually work? (backtest)")
    if "var_backtest" in rep:
        bt, ku = rep["var_backtest"], rep["kupiec"]
        fig = go.Figure()
        fig.add_scatter(x=bt.index, y=bt["Return"], mode="markers", name="Daily return",
                        marker=dict(color="#b9b8b2", size=3), hovertemplate="%{y:.2%}")
        fig.add_scatter(x=bt.index, y=-bt["VaR"], name="VaR limit (from previous 250 days)",
                        line=dict(color=BLUE, width=2), hovertemplate="%{y:.2%}")
        br = bt[bt["Breach"]]
        fig.add_scatter(x=br.index, y=br["Return"], mode="markers", name="Breach: loss worse than VaR",
                        marker=dict(color=CRITICAL, size=6, symbol="x"), hovertemplate="%{y:.2%}")
        fig.update_layout(hovermode="closest")
        show(style_fig(fig, 360, yfmt=".0%"))
        ok = bool(ku["Model OK"])
        md(
            f"<div class='verdict {'good' if ok else 'bad'}'>{'✅' if ok else '❌'} "
            f"<b>{'VaR held up' if ok else 'VaR failed the test'}.</b> Over {int(ku['Observations']):,} days we expected about "
            f"<b>{ku['Expected Breaches']:.0f}</b> breaches ({level}%) and saw <b>{int(ku['Actual Breaches'])}</b> "
            f"({pct(ku['Breach Rate'], 1)}). Kupiec test p-value: <b>{ku['p-value']:.3f}</b> "
            f"({'above' if ok else 'below'} 0.05, so the difference is "
            f"{'consistent with luck' if ok else 'too large to be luck'}).</div>", unsafe_allow_html=True)
        clusters = br.index.to_series().dt.year.value_counts().head(2) if len(br) else pd.Series(dtype=int)
        explain("What this means for your portfolio", f"""
{"Good news: the daily loss estimate has been **reliable** for this portfolio, so you can use the VaR figures on this page as a fair guide." if ok else f"The daily loss estimate has been **too optimistic** for this portfolio: losses beyond it happened {int(ku['Actual Breaches'])} times instead of about {ku['Expected Breaches']:.0f}. Treat the VaR figures on this page as a **minimum**, not a typical bad day."}
{"The breaches bunched up in " + " and ".join(str(y) for y in clusters.index) + ": risk models are slowest to react exactly when markets turn." if len(clusters) else ""}

(How it works: each day, VaR is estimated from only the previous 250 days, as you could have done at the time, then compared with what actually happened the next day.)""")
    else:
        st.info("Not enough history for a backtest: choose an earlier start date (needs about 13 months).")

    # Volatility over time
    st.subheader("Is risk rising or falling right now?")
    fig = go.Figure()
    fig.add_scatter(x=rep["rolling_vol"].index, y=rep["rolling_vol"], name="Rolling volatility (last 3 months)",
                    line=dict(color=BLUE, width=2), hovertemplate="%{y:.1%}")
    fig.add_scatter(x=rep["ewma_vol"].index, y=rep["ewma_vol"], name="EWMA volatility (reacts faster)",
                    line=dict(color=ORANGE, width=1.5), hovertemplate="%{y:.1%}")
    fig.add_scatter(x=[port.index[0], port.index[-1]], y=[p["Annualized Vol"]] * 2, name="Long-run average",
                    mode="lines", line=dict(color=INK_2, width=1.5, dash="dot"), hoverinfo="skip")
    fig.update_layout(hovermode="x unified")
    show(style_fig(fig, 340, yfmt=".0%"))
    now = rep["ewma_vol"].dropna().iloc[-1]
    md(f"Current EWMA volatility: **{pct(now)}** vs a long-run average of **{pct(p['Annualized Vol'])}**: "
                f"risk is currently **{'above' if now > p['Annualized Vol'] else 'below'} normal**.")
    peak_vol = rep["ewma_vol"].max()
    explain("What this means for your portfolio", f"""
Right now a bad day (1 in 20) would cost about **{money(value * 1.65 * now / 252 ** 0.5)}**, compared with about **{money(value * 1.65 * p['Annualized Vol'] / 252 ** 0.5)}** at normal risk levels.
At its stormiest ({rep['ewma_vol'].idxmax():%b %Y}) this portfolio was swinging at **{pct(peak_vol)}** a year, when a bad day could cost about **{money(value * 1.65 * peak_vol / 252 ** 0.5)}**.

(Risk comes in waves. EWMA, the method banks have used since 1994, weights recent days more, so it reacts faster than the 3-month rolling line.)""")

# =============================================================================
# DIVERSIFICATION
# =============================================================================
with t_div:
    st.subheader("Where your money is vs where your risk is")
    rc = rep["risk_contribution"]
    fig = go.Figure()
    fig.add_bar(x=tickers, y=w.values, name="Share of money", marker=dict(color=BLUE, cornerradius=4),
                hovertemplate="%{x}: %{y:.1%} of money<extra></extra>")
    fig.add_bar(x=tickers, y=rc.values, name="Share of risk", marker=dict(color=ORANGE, cornerradius=4),
                hovertemplate="%{x}: %{y:.1%} of risk<extra></extra>")
    fig.update_layout(barmode="group", bargap=0.3, bargroupgap=0.08)
    show(style_fig(fig, 360, yfmt=".0%"))
    top = rc.idxmax()
    md(f"**{top}** holds **{pct(w[top])}** of your money but drives **{pct(rc[top])}** of your risk. "
                f"Diversification ratio: **{num(rk.diversification_ratio(w.values, rets[tickers].cov()))}** "
                "(1.0 = no diversification benefit; higher is better).")
    calmers = [t for t in tickers if rc[t] < w[t] * 0.8]
    drivers = [t for t in tickers if rc[t] > w[t] * 1.2]
    explain("What this means for your portfolio", f"""
{"- **Adding risk beyond their size:** " + ", ".join(f"{t} ({pct(w[t], 0)} of money → {pct(rc[t], 0)} of risk)" for t in drivers) if drivers else "- No holding adds much more risk than its share of money."}
{"- **Calming the portfolio:** " + ", ".join(f"{t} ({pct(w[t], 0)} of money → {pct(rc[t], 0)} of risk)" for t in calmers) if calmers else "- No holding is doing much to calm the portfolio down."}
- Your diversification ratio of {num(rk.diversification_ratio(w.values, rets[tickers].cov()))} means mixing these holdings cut the swings by about **{pct(1 - 1/rk.diversification_ratio(w.values, rets[tickers].cov()), 0)}** compared with holding them separately.

(Volatile holdings move a portfolio more than calm ones with the same money; risk shares add up to 100%.)""")

    st.subheader("Do your holdings move together?")
    corr = rep["correlation"]
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.index, zmin=-1, zmax=1,
        colorscale=[[0, "#2a78d6"], [0.5, "#f0efec"], [1, "#e34948"]],
        text=corr.round(2).values, texttemplate="%{text}", textfont=dict(size=13),
        hovertemplate="%{y} vs %{x}: %{z:.2f}<extra></extra>", xgap=2, ygap=2,
        colorbar=dict(title="", tickvals=[-1, 0, 1], ticktext=["-1 opposite", "0 unrelated", "+1 together"])))
    fig.update_yaxes(autorange="reversed")
    show(style_fig(fig, 120 + 48 * len(tickers), legend=False))
    if len(tickers) > 1:
        pairs = corr.where(np.triu(np.ones(corr.shape, dtype=bool), 1)).stack()
        hi_pair, lo_pair = pairs.idxmax(), pairs.idxmin()
        corr_text = (f"- **Most alike:** {hi_pair[0]} and {hi_pair[1]} ({pairs.max():.2f}). "
                     f"{'They behave almost like the same holding.' if pairs.max() > 0.85 else 'They often move together.'}\n"
                     f"- **Best diversifiers:** {lo_pair[0]} and {lo_pair[1]} ({pairs.min():.2f}). "
                     f"{'When one falls, the other often holds up.' if pairs.min() < 0.3 else 'Even this pair moves together quite a lot.'}\n"
                     f"- **On average** your holdings have a correlation of {avg_corr:.2f}: "
                     f"{'little real diversification between them' if avg_corr > 0.8 else 'some diversification' if avg_corr > 0.5 else 'good diversification'}.")
    else:
        corr_text = "With a single holding there is nothing to diversify against."
    explain("What this means for your portfolio", f"""
{corr_text}

(Correlation runs from −1, always opposite, through 0, unrelated, to +1, always together. In crashes correlations tend to rise, so check the Stress tests tab.)""")

    if has_bench and "capm" in rep:
        st.subheader(f"How each holding behaves relative to {bench_name}")
        cp = rep["capm"].copy()
        cap = rk.up_down_capture(rets[tickers].assign(Portfolio=port), bench)
        cp = cp.join(cap)
        st.dataframe(pd.DataFrame({
            "Beta": cp["Beta"].map(num),
            "Alpha per year": cp["Alpha (annual)"].map(pct),
            "Alpha t-stat": cp["Alpha t-stat"].map(num),
            "R²": cp["R-squared"].map(lambda x: pct(x, 0)),
            "Up capture": cp["Up Capture"].map(lambda x: pct(x, 0)),
            "Down capture": cp["Down Capture"].map(lambda x: pct(x, 0)),
        }), width="stretch")
        md(f"Portfolio tracking error: **{pct(rep['tracking_error'])}** · "
                    f"Information ratio: **{num(rep['information_ratio'])}**")
        pr_row = cp.loc["Portfolio"]
        explain("What this means for your portfolio", f"""
- **If {bench_name} drops 10%**, your portfolio has typically dropped about **{abs(pr_row['Beta'])*10:.0f}%** (beta {num(pr_row['Beta'])}): roughly **{money(value * abs(pr_row['Beta']) * 0.10)}** on {money(value)}.
- On the market's **up days** you captured {pct(pr_row['Up Capture'], 0)} of the gains; on its **down days**, {pct(pr_row['Down Capture'], 0)} of the losses. {"That's a good trade-off: you keep more of the upside than the downside." if pr_row['Up Capture'] > pr_row['Down Capture'] + 0.05 else "You take losses about as fully as gains." if abs(pr_row['Up Capture'] - pr_row['Down Capture']) <= 0.05 else "You've been taking more of the losses than the gains."}
- **Alpha {pct(pr_row['Alpha (annual)'])} a year** (t-stat {num(pr_row['Alpha t-stat'])}): {"strong evidence of returns beyond what market exposure explains." if pr_row['Alpha t-stat'] >= 2 else "strong evidence of falling short of what market exposure predicts." if pr_row['Alpha t-stat'] <= -2 else "within what luck could explain, so no proven edge either way."}
- {pct(pr_row['R-squared'], 0)} of your portfolio's ups and downs are explained by {bench_name}{" — it's largely a market ride." if pr_row['R-squared'] > 0.8 else "; the rest comes from your own mix of holdings."}

(Beta: how strongly it follows the market. Alpha: return beyond that, trustworthy only with a t-stat above about 2. Capture: share of the market's up and down days you experienced.)""")

# =============================================================================
# STRESS TESTS
# =============================================================================
with t_stress:
    st.subheader("How would this portfolio have done in past crises?")
    stress = rep["stress"]
    covered = stress[stress["Covered"] == True].copy()  # noqa: E712
    if has_bench:
        bstress = rk.stress_test(rets[[bench_key]], [1.0])
        covered["Benchmark"] = (bstress["Portfolio Return"].reindex(covered.index)
                                if "Portfolio Return" in bstress.columns else np.nan)
    if covered.empty:
        if t0 > pd.Timestamp(max(e for _, e in rk.CRISES.values())):
            st.info(f"Your holdings only have a shared price history from {t0:%b %Y}, which is after all of "
                    "these crises (a recently launched fund limits how far back the analysis can go). "
                    "Use the scenario sliders below, or analyze older holdings separately.")
        else:
            st.info("Your date range doesn't cover any of the crisis windows. Choose an earlier start date (e.g. 2007).")
    else:
        fig = go.Figure()
        labels = [f"{n}<br><span style='font-size:11px;color:{MUTED}'>{s[:7]} → {e[:7]}</span>"
                  for n, s, e in zip(covered.index, covered["Start"], covered["End"])]
        fig.add_bar(y=labels, x=covered["Portfolio Return"].astype(float), orientation="h", name="Your portfolio",
                    marker=dict(color=BLUE, cornerradius=4), text=covered["Portfolio Return"].map(pct),
                    textposition="outside", hovertemplate="%{x:.1%}<extra>Your portfolio</extra>")
        if has_bench:
            fig.add_bar(y=labels, x=covered["Benchmark"].astype(float), orientation="h", name=bench_name,
                        marker=dict(color=ORANGE, cornerradius=4), text=covered["Benchmark"].map(pct),
                        textposition="outside", hovertemplate=f"%{{x:.1%}}<extra>{bench_name}</extra>")
        lo = min(covered["Portfolio Return"].min(), covered.get("Benchmark", covered["Portfolio Return"]).min())
        hi = max(covered["Portfolio Return"].max(), covered.get("Benchmark", covered["Portfolio Return"]).max(), 0)
        fig.update_xaxes(range=[lo * 1.3, max(hi * 1.3, 0.05)])
        fig.update_yaxes(autorange="reversed")
        fig.update_layout(barmode="group", bargap=0.28, bargroupgap=0.06)
        show(style_fig(fig, 110 + 70 * len(covered), xfmt=".0%"))
        worst = covered["Portfolio Return"].astype(float).idxmin()
        loss = covered.loc[worst, "Portfolio Return"]
        md(f"Worst crisis for this portfolio: **{worst}**, "
           + (f"a loss of **{pct(-loss)}**, about **{money(-value * loss)}** on {money(value)}." if loss < 0
              else f"and it still gained **{pct(loss)}**."))
    missing = stress[stress["Covered"] != True]  # noqa: E712
    if len(missing):
        st.caption("Not shown (your data doesn't fully cover these windows): " + ", ".join(missing.index))
    with st.expander("Per-holding results in each crisis"):
        cols = [c for c in stress.columns if c.endswith("Return") or c == "Portfolio Max Drawdown"]
        st.dataframe(covered[cols].astype(float).map(pct) if not covered.empty else covered, width="stretch")
    if not covered.empty:
        cover_text = (f"Across the {len(covered)} crises your data covers, this portfolio moved **{pct(covered['Portfolio Return'].astype(float).mean())}** on average"
                      + (f" against **{pct(covered['Benchmark'].astype(float).mean())}** for {bench_name}" if has_bench and covered['Benchmark'].notna().any() else "")
                      + f": about **{money(abs(covered['Portfolio Return'].astype(float).mean()) * value)}** on {money(value)}. "
                      "Ask yourself whether you'd have stayed invested through a loss like that.")
    else:
        cover_text = "None of the historical crises fall inside your data, so use the scenario sliders below instead."
    explain("What this means for your portfolio", f"""
{cover_text}

(Stress tests replay real disasters, peak to trough, on your exact mix. The next crisis won't look exactly like these, which is why there's also a custom scenario below.)""")

    st.subheader("Your own scenario")
    md("Move the sliders to imagine a sudden shock to each holding.")
    shocks = {}
    sc = st.columns(min(len(tickers), 4))
    defaults = {"AGG": 5, "TLT": 10, "IEF": 5, "GLD": 5, "BND": 5}
    for i, t in enumerate(tickers):
        shocks[t] = sc[i % len(sc)].slider(f"{t}", -60, 40, defaults.get(t, -30), 5, format="%d%%") / 100
    shock = rk.shock_test(w, shocks)
    total = shock["Portfolio Return"]
    md(f"<div class='verdict {'good' if total >= 0 else 'bad'}'>In this scenario your portfolio would "
                f"{'gain' if total >= 0 else 'lose'} <b>{pct(abs(total))}</b>, about <b>{money(abs(total) * value)}</b>.</div>",
                unsafe_allow_html=True)

# =============================================================================
# WHAT-IF: TRADE SIMULATOR
# =============================================================================
@st.cache_data(ttl=86400, show_spinner=False)
def scheme_table():
    sc = mf.load_schemes()
    return sc[["code", "name", "plan", "option"]]


def search_funds(query, limit=25):
    words = [x for x in query.lower().split() if x]
    if len(query.strip()) < 3:
        return []
    sc = scheme_table()
    hit = sc[sc["name"].str.lower().map(lambda n: all(x in n for x in words))]
    hit = hit.assign(rank=(hit["plan"] != "direct").astype(int) + (hit["option"] != "growth").astype(int))
    return hit.sort_values(["rank", "name"]).head(limit)[["code", "name"]].values.tolist()


def scenario_stats(r_port, rets_u, weights, val):
    """Headline statistics for one portfolio over the scenario window."""
    s = rk.summary_stats(r_port.to_frame("P"), riskfree_rate=rf, level=level).loc["P"]
    st_ = rk.stress_test(rets_u, weights)
    cov_ = st_[st_["Covered"] == True]  # noqa: E712
    out = {"ret": s["Annualized Return"], "vol": s["Annualized Vol"], "sharpe": s["Sharpe Ratio"],
           "mdd": s["Max Drawdown"], "var": val * s[var_key],
           "crisis": cov_["Portfolio Return"].astype(float).mean() if len(cov_) else np.nan, "bad5": np.nan}
    if len(r_port.dropna()) >= 260:
        q_, _, _ = future_bands(r_port.dropna(), rf, max_years=5)
        out["bad5"] = q_[10][60] * val
    return out


def trade_simulator_tax_pieces(book, units_after):
    pieces, exit_v, locked_v = [], 0.0, 0.0
    today = TAX["today"]
    for tk, held, after, price in zip(book["Ticker"], book["Units"], units_after, book["Price"]):
        sold = held - after
        if sold <= 1e-9 or not str(tk).startswith(rk.MF_PREFIX):
            continue
        code = int(str(tk)[len(rk.MF_PREFIX):])
        ls = LOTS.get(code)
        if not ls:
            continue
        sold = min(sold, sum(l.units for l in ls))
        k = TAX["klass"].get(code, "other")
        ps = txm.gains_on_sale(ls, sold, float(price), today, k, TAX["fmv"].get(code))
        for p_ in ps:
            p_["klass"] = k
            if p_["days"] < 365 and k != "debt":
                exit_v += p_["proceeds"]
        pieces += ps
        row_ = TAX["rows"][TAX["rows"]["code"] == code]
        if len(row_) and row_["Locked (ELSS)"].iloc[0] > 0:
            locked_v += min(float(row_["Locked (ELSS)"].iloc[0]), sold * float(price))
    return pieces, exit_v, locked_v


def trade_tax(book, units_after):
    """Capital-gains estimate for the sells in a what-if, from the investor's real tax lots (needs transactions)."""
    if TAX is None:
        return None
    pieces, exit_v, locked_v = trade_simulator_tax_pieces(book, units_after)
    if not pieces:
        return None
    out = txm.tax_on_pieces(pieces, TAXSET, TAX["today"], TAX["booked_eq_lt"])
    out.update(exit_load_value=exit_v, locked_value=locked_v)
    return out


def trade_simulator():
    st.subheader("Trade simulator")
    md("<span class='small-note'>Buy or sell units of what you own, or add something new, and see how the "
       "portfolio would have behaved. Nothing is traded; this is a sandbox.</span>", unsafe_allow_html=True)

    sim_id = str(hash((tuple(rows.index), tuple(np.round(rows.values, 2)), round(value))))
    if st.session_state.get("wi_id") != sim_id:
        st.session_state.wi_id, st.session_state.wi_new, st.session_state.wi_ver = sim_id, [], 0

    # Current position: units at the latest price
    px_now = rep["prices"].ffill().iloc[-1]
    raw_of = {label_of[t]: t for t in rows.index}
    book = pd.DataFrame({"Ticker": [raw_of.get(n, n) for n in tickers], "Holding": tickers,
                         "Price": [float(px_now[n]) for n in tickers]})
    book["Units"] = value * np.asarray(w[tickers], dtype=float) / book["Price"]

    # Add new holdings
    with st.expander("Add a new holding to the scenario", icon=":material/add_circle:"):
        a1, a2 = st.columns(2)
        with a1:
            q = st.text_input("Search Indian mutual funds", placeholder="e.g. parag parikh flexi, nifty 50 index",
                              key="wi_q")
            found = search_funds(q) if q else []
            if q and not found:
                st.caption("No scheme matches. Try fewer or shorter words.")
            if found:
                pick = st.selectbox("Scheme", found, format_func=lambda x: x[1], key="wi_pick")
                if st.button("Add fund", key="wi_add_fund", icon=":material/add:"):
                    t = f"{rk.MF_PREFIX}{pick[0]}"
                    if t not in book["Ticker"].values and t not in [n["ticker"] for n in st.session_state.wi_new]:
                        st.session_state.wi_new.append({"ticker": t, "name": mf.short_label(pick[1])})
                        st.session_state.wi_ver += 1
                        st.rerun()
        with a2:
            tk = st.text_input("Or a stock / ETF ticker (Yahoo symbol)", placeholder="e.g. GOLDBEES.NS, VXUS, TCS.NS",
                               key="wi_tk").strip().upper()
            if st.button("Add ticker", key="wi_add_tk", icon=":material/add:", disabled=not tk):
                price = latest_prices((tk,)).get(tk)
                if price is None:
                    st.error(f"No price found for {tk}. Indian stocks need .NS (e.g. TCS.NS).")
                elif tk not in book["Ticker"].values and tk not in [n["ticker"] for n in st.session_state.wi_new]:
                    st.session_state.wi_new.append({"ticker": tk, "name": tk})
                    st.session_state.wi_ver += 1
                    st.rerun()

    for n in st.session_state.wi_new:
        price = latest_prices((n["ticker"],)).get(n["ticker"])
        if price:
            book.loc[len(book)] = {"Ticker": n["ticker"], "Holding": n["name"], "Price": float(price), "Units": 0.0}

    mode = st.segmented_control("Trade in", ["Units", f"Amount ({CUR})"], default="Units", key="wi_mode")
    by_units = mode != f"Amount ({CUR})"
    ticket = book[["Holding", "Units", "Price"]].assign(Value=(book["Units"] * book["Price"]).round(0), Trade=0.0, **{"Sell all": False})
    ed = st.data_editor(
        ticket, hide_index=True, width="stretch", key=f"wi_ed_{st.session_state.wi_ver}_{by_units}",
        disabled=["Holding", "Units", "Price", "Value"],
        column_config={
            "Units": st.column_config.NumberColumn("Units held", format="%.3f"),
            "Price": st.column_config.NumberColumn("Price / NAV", format=f"{CUR}%.2f"),
            "Value": st.column_config.NumberColumn("Value now", format="localized"),
            "Trade": st.column_config.NumberColumn(
                "Buy (+) / sell (−) units" if by_units else f"Buy (+) / sell (−) {CUR}",
                format="%.3f" if by_units else "%.0f", step=0.001 if by_units else 100.0,
                help="Positive to buy, negative to sell."),
            "Sell all": st.column_config.CheckboxColumn("Sell all"),
        })
    trade_units = ed["Trade"].fillna(0).to_numpy() if by_units else ed["Trade"].fillna(0).to_numpy() / book["Price"].to_numpy()
    units_after = book["Units"].to_numpy() + trade_units
    oversold = units_after < -1e-9
    units_after = np.where(ed["Sell all"].to_numpy(), 0.0, np.clip(units_after, 0, None))
    if oversold.any():
        st.warning("You can't sell more than you hold: " + ", ".join(book["Holding"][oversold])
                   + " capped at selling everything.")
    flows = (units_after - book["Units"].to_numpy()) * book["Price"].to_numpy()
    if np.abs(flows).sum() < 0.5:
        st.info("Enter a trade above (for example −50 units of one fund and +100 of another) to see its effect.",
                icon=":material/edit_note:")
        return
    val_after = float((units_after * book["Price"]).sum())
    if val_after <= 0:
        st.warning("That would sell everything. Keep at least one holding to compare.")
        return
    bought, sold = flows[flows > 0].sum(), -flows[flows < 0].sum()
    net = bought - sold
    md(f"<div class='backdrop ok'><b>Trades:</b> sell {money(sold)}, buy {money(bought)}. "
       + (f"That needs <b>{money(net)}</b> of new money." if net > 0.5 else
          f"<b>{money(-net)}</b> comes out as cash (left out of the comparison)." if net < -0.5 else "Fully self-funded.")
       + f" Portfolio after: <b>{money(val_after)}</b>.</div>", unsafe_allow_html=True)
    sale_tax = trade_tax(book, units_after)
    if sale_tax is not None:
        t_ = sale_tax
        md(f"<div class='backdrop {'bad' if t_['tax'] > 0 else 'good'}'><b>Estimated tax on these sales: "
           f"{money(t_['tax'])}</b> (short-term gains {money(t_['eq_st'] + t_['slab'])}, long-term "
           f"{money(t_['eq_lt'] + t_['other_lt'])}, of which {money(max(t_['eq_lt'] - t_['taxable_eq_lt'], 0))} "
           "falls in this year's tax-free ₹1.25 lakh)."
           + (f" {money(t_['exit_load_value'])} of the units sold were bought under a year ago and may carry an "
              "exit load (often 1%)." if t_["exit_load_value"] > 1 else "")
           + (f" <b>{money(t_['locked_value'])} is still in the ELSS 3-year lock-in and can't be sold yet.</b>"
              if t_["locked_value"] > 1 else "") + "</div>", unsafe_allow_html=True)

    w_after = units_after * book["Price"].to_numpy() / val_after
    w_before = np.r_[np.asarray(w[tickers], dtype=float), np.zeros(len(book) - len(tickers))]
    try:
        with st.spinner("Re-running the analysis..."):
            rep2 = load_report(tuple(book["Ticker"]), tuple(np.round(w_after, 6)), benchmark.strip().upper(),
                               start, end, rf, level)
    except Exception as e:
        st.error(f"Couldn't analyse this scenario: {e}")
        return
    r_u = rep2["returns"][list(book["Ticker"])].rename(columns=dict(zip(book["Ticker"], book["Holding"])))
    port_b = (r_u * w_before).sum(axis=1)
    port_a = (r_u * w_after).sum(axis=1)
    sb = scenario_stats(port_b, r_u, pd.Series(w_before, index=r_u.columns), value)
    sa = scenario_stats(port_a, r_u, pd.Series(w_after, index=r_u.columns), val_after)
    win0, win1 = r_u.index[0], r_u.index[-1]
    if win0 > t0 + pd.Timedelta(days=10):
        st.caption(f"Compared over {win0:%b %Y} – {win1:%b %Y}, the period every holding in the scenario has prices.")

    # One-line verdict
    dv = sa["vol"] / sb["vol"] - 1
    bits = [f"risk {'down' if dv < 0 else 'up'} {abs(dv)*100:.0f}% (volatility {pct(sb['vol'])} → {pct(sa['vol'])})",
            f"past return {pct(sb['ret'])} → {pct(sa['ret'])} a year",
            f"worst fall {pct(sb['mdd'])} → {pct(sa['mdd'])}"]
    better = (sa["sharpe"] > sb["sharpe"] + 0.02, sa["sharpe"] < sb["sharpe"] - 0.02)
    tone = "good" if better[0] else "bad" if better[1] else "ok"
    md(f"<div class='backdrop {tone}'><b>{'Better' if better[0] else 'Worse' if better[1] else 'About the same'} "
       f"reward for risk</b> (Sharpe {num(sb['sharpe'])} → {num(sa['sharpe'])}): " + "; ".join(bits) + ".</div>",
       unsafe_allow_html=True)

    mc_ = st.columns(4)
    mc_[0].metric("Return per year", pct(sa["ret"]), f"{(sa['ret'] - sb['ret'])*100:+.1f} pts")
    mc_[1].metric("Volatility", pct(sa["vol"]), f"{(sa['vol'] - sb['vol'])*100:+.1f} pts", delta_color="inverse")
    mc_[2].metric("Sharpe ratio", num(sa["sharpe"]), f"{sa['sharpe'] - sb['sharpe']:+.2f}")
    mc_[3].metric("Max drawdown", pct(sa["mdd"]), f"{(sa['mdd'] - sb['mdd'])*100:+.1f} pts")
    mc_ = st.columns(3)
    mc_[0].metric(f"1-day VaR ({conf})", money(sa["var"]), f"{money(sa['var'] - sb['var'])}", delta_color="inverse")
    mc_[1].metric("Average crisis return", pct(sa["crisis"]),
                  None if pd.isna(sa["crisis"]) else f"{(sa['crisis'] - sb['crisis'])*100:+.1f} pts")
    mc_[2].metric("5-year bad case", money(sa["bad5"]),
                  None if pd.isna(sa["bad5"]) else f"{money(sa['bad5'] - sb['bad5'])} vs before")

    rc_b = pd.Series(0.0, index=r_u.columns)
    rc_a = pd.Series(0.0, index=r_u.columns)
    cov_u = r_u.cov()
    nz_b, nz_a = w_before > 0, w_after > 0
    rc_b[nz_b] = rk.risk_contribution(w_before[nz_b], cov_u.loc[nz_b, nz_b].values)
    rc_a[nz_a] = rk.risk_contribution(w_after[nz_a], cov_u.loc[nz_a, nz_a].values)
    cc = st.columns(2)
    for col, (title, b_, a_) in zip(cc, [("Share of money", w_before, w_after), ("Share of risk", rc_b.values, rc_a.values)]):
        f_ = go.Figure()
        f_.add_bar(y=r_u.columns, x=b_, name="Before", orientation="h", marker=dict(color="#c4cad4", cornerradius=3),
                   hovertemplate="%{y}: %{x:.1%}<extra>Before</extra>")
        f_.add_bar(y=r_u.columns, x=a_, name="After", orientation="h", marker=dict(color=BLUE, cornerradius=3),
                   hovertemplate="%{y}: %{x:.1%}<extra>After</extra>")
        f_.update_layout(barmode="group", bargap=0.3, yaxis=dict(autorange="reversed"))
        with col:
            show(style_fig(f_, 70 + 46 * len(r_u.columns), xfmt=".0%"), title)

    g_ = go.Figure()
    g_.add_scatter(x=port_b.index, y=value * (1 + port_b).cumprod(), name="Before", line=dict(color="#98a2b3", width=2),
                   hovertemplate=CUR + "%{y:,.0f}")
    g_.add_scatter(x=port_a.index, y=value * (1 + port_a).cumprod(), name="After", line=dict(color=BLUE, width=2.2),
                   hovertemplate=CUR + "%{y:,.0f}")
    g_.update_layout(hovermode="x unified")
    show(style_fig(g_, 340, yfmt="$,.0f"), f"Growth of {money(value)}, before vs after (past {win1.year - win0.year} years)")

    tr = book.assign(After=units_after, Change=units_after - book["Units"], Amount=flows)
    tr = tr[np.abs(tr["Amount"]) > 0.5]
    st.dataframe(pd.DataFrame({
        "Holding": tr["Holding"], "Action": np.where(tr["Change"] > 0, "Buy", "Sell"),
        "Units": tr["Change"].abs().round(3), "Amount": tr["Amount"].abs().map(money),
        "Units after": tr["After"].round(3)}), hide_index=True, width="stretch")
    st.caption("Based on past prices over the same period for both versions. "
               + ("Tax is estimated from your transaction lots (first in, first out); exit loads and brokerage are left out. "
                  if TAX is not None else "Leaves out exit loads, capital-gains tax and brokerage (upload a CAS or "
                  "transaction file to see tax). ") + "Not a recommendation.")


# =============================================================================
# WHAT-IF OPTIMIZER
# =============================================================================
with t_what:
    trade_simulator()
    st.subheader("Compare with textbook mixes")
    md("<span class='small-note'>The same holdings in different proportions, chosen by classic rules. Based on past "
       "returns, which are a poor guide to the future: use them to understand trade-offs, not to pick a portfolio."
       "</span>", unsafe_allow_html=True)
    if len(tickers) < 2:
        st.warning("Add at least two holdings to compare portfolio mixes.")
    else:
        ar = rets[tickers].dropna()
        c1, c2 = st.columns(2)
        max_w = c1.slider("Maximum weight in any one holding", int(np.ceil(100 / len(tickers))), 100, 100, 5,
                          format="%d%%", help="Caps stop the optimizer putting everything in one asset.") / 100
        shrink = c2.toggle("Stabilize estimates (covariance shrinkage)", True,
                           help="Blends the measured correlations with their average, which reduces the effect of noise.")
        try:
            er, cov, ef, alts = optimizer_results(ar, rf, max_w, shrink)
        except ValueError as e:
            st.error(str(e))
            st.stop()
        cands = {"Your portfolio": w[tickers], **alts}

        fig = go.Figure()
        fig.add_scatter(x=ef["Volatility"], y=ef["Returns"], name="Efficient frontier", mode="lines",
                        line=dict(color=INK_2, width=2),
                        hovertemplate="Vol %{x:.1%} · Return %{y:.1%}<extra>Frontier</extra>")
        av = np.sqrt(np.diag(cov))
        fig.add_scatter(x=av, y=er.values, mode="markers+text", text=tickers, textposition="middle right",
                        name="Single holdings", marker=dict(color="#c9c8c2", size=9),
                        textfont=dict(color=MUTED, size=11),
                        hovertemplate="%{text}: vol %{x:.1%}, return %{y:.1%}<extra></extra>")
        symbols = ["star", "diamond", "circle", "square", "triangle-up"]
        for i, (name, wt) in enumerate(cands.items()):
            wv = np.asarray(wt, dtype=float)
            fig.add_scatter(x=[rk.portfolio_vol(wv, cov.values)], y=[rk.portfolio_returns(wv, er.values)],
                            mode="markers", name=name,
                            marker=dict(color=SERIES[i], size=16 if i == 0 else 13, symbol=symbols[i],
                                        line=dict(color=SURFACE, width=2)),
                            hovertemplate=f"{name}: vol %{{x:.1%}}, return %{{y:.1%}}<extra></extra>")
        fig.update_layout(hovermode="closest")
        fig.update_xaxes(title="Volatility per year (risk)")
        fig.update_yaxes(title="Return per year")
        show(style_fig(fig, 470, xfmt=".0%", yfmt=".0%"), "Risk vs return of different mixes (based on the past)")
        my_vol = rk.portfolio_vol(np.asarray(w[tickers], dtype=float), cov.values)
        my_ret = rk.portfolio_returns(np.asarray(w[tickers], dtype=float), er.values)
        same_risk = ef.loc[ef["Volatility"] <= my_vol * 1.001, "Returns"].max()
        same_ret = ef.loc[ef["Returns"] >= my_ret * 0.999, "Volatility"].min()
        explain("What this means for your portfolio", f"""
{f"In hindsight, a different mix of the **same holdings** could have earned about **{pct(same_risk)}** a year for the same risk (yours: {pct(my_ret)}), or matched your return with **{pct(same_ret)}** volatility instead of {pct(my_vol)}." if pd.notna(same_risk) and same_risk > my_ret + 0.002 else "Your mix sits on or very close to the efficient frontier: in hindsight, you couldn't have done much better for the same risk with these holdings."}
Remember that the frontier is drawn with hindsight: past returns are a weak guide, so the gap is smaller than it looks.

**How to read it:** every point on the curve is the mix with the **lowest risk for a given return**. Mixes below or to the right of the curve take more risk than they need to for that return.
- **Max Sharpe**: the best return per unit of risk *in hindsight*. It is very sensitive to past returns, often putting most of the money in whatever did best recently.
- **Min volatility**: the calmest possible mix. It only needs risk estimates, which are more reliable than return estimates.
- **Risk parity**: every holding contributes the same amount of risk.
- **Equal weight**: the simple 1/N rule, which is surprisingly hard to beat in practice.

Real-life use: this is Harry Markowitz's 1952 idea (Nobel Prize 1990), and it's the starting point of almost every institutional asset-allocation process.""")

        wt_table = pd.DataFrame({k: pd.Series(np.asarray(v, dtype=float), index=tickers) for k, v in cands.items()})
        stats_rows = pd.DataFrame({k: [rk.portfolio_returns(wt_table[k].values, er.values),
                                       rk.portfolio_vol(wt_table[k].values, cov.values)]
                                   for k in wt_table}, index=["Expected return*", "Volatility"])
        md("**Weights of each mix**")
        st.dataframe(pd.concat([wt_table.map(lambda x: pct(x, 0)), stats_rows.map(pct)]), width="stretch")
        st.caption("*Expected return = historical average, a weak guide to the future.")

        st.subheader("Would these mixes have worked in real time?")
        md("The frontier above uses hindsight. This backtest re-estimates the mixes each period "
                    "using **only the past year of data**, and charges trading costs.")
        c1, c2 = st.columns(2)
        reb = c1.selectbox("Rebalance", ["Monthly", "Quarterly", "Yearly", "Never (buy and hold)"])
        cost = c2.slider("Trading cost (basis points per trade)", 0, 50, 10, 5,
                         help="10 bps = 0.10% of the amount bought or sold.")
        reb_code = {"Monthly": "M", "Quarterly": "Q", "Yearly": "Y", "Never (buy and hold)": None}[reb]
        if len(ar) < 300:
            st.info("Needs more than about 14 months of history.")
        else:
            with st.spinner("Running backtest..."):
                bstats, brets = strategy_backtest(ar, w[tickers], reb_code, cost, rf)
            fig = go.Figure()
            for i, col in enumerate(brets.columns):
                fig.add_scatter(x=brets.index, y=value * (1 + brets[col]).cumprod(), name=col,
                                line=dict(color=SERIES[i], width=2), hovertemplate=CUR + "%{y:,.0f}")
            fig.update_layout(hovermode="x unified")
            show(style_fig(fig, 380, yfmt="$,.0f"))
            st.dataframe(pd.DataFrame({
                "Return per year": bstats["Annualized Return"].map(pct),
                "Volatility": bstats["Annualized Vol"].map(pct),
                "Sharpe": bstats["Sharpe Ratio"].map(num),
                "Max drawdown": bstats["Max Drawdown"].map(pct),
                "Turnover per year": bstats["Annual Turnover"].map(lambda x: pct(x, 0)),
            }), width="stretch")
            rank = bstats["Sharpe Ratio"].rank(ascending=False)["Your portfolio"]
            explain("What this means for your portfolio", f"""
Run in real time with {reb.lower()} rebalancing and {cost} bps trading costs, your mix ranked **#{int(rank)} of {len(bstats)}** on reward for risk (Sharpe {num(bstats.loc['Your portfolio', 'Sharpe Ratio'])}). {"The simpler alternatives didn't beat it." if rank == 1 else f"The best was **{bstats['Sharpe Ratio'].idxmax()}** (Sharpe {num(bstats['Sharpe Ratio'].max())})."}
Rebalancing your mix would have meant trading about **{pct(bstats.loc['Your portfolio', 'Annual Turnover'], 0)}** of the portfolio a year, roughly **{money(value * bstats.loc['Your portfolio', 'Annual Turnover'] * cost / 10000)}** a year in costs on {money(value)}.

Prices move every day, so your weights drift. **Rebalancing** sells what went up and buys what went down to get back to target. More frequent rebalancing keeps risk on target but costs more in trading.
**Turnover** = how much of the portfolio is traded per year (100% = the whole portfolio once).
The re-estimated strategies only ever see data available at the time, avoiding *look-ahead bias*, the most common mistake in backtests.""")

# =============================================================================
# ABOUT
# =============================================================================
# =============================================================================
# BUILD A PORTFOLIO: pick stocks, ETFs and funds by name and simulate them
# =============================================================================
@st.cache_data(ttl=86400, show_spinner=False)
def _search_cached(query, prefer):
    return stocks.search(query, prefer=prefer)


def stock_search(query, prefer):
    try:
        return _search_cached(query, prefer)
    except Exception:
        return None


def _bld_add(ticker, name, kind):
    basket = st.session_state.bld_basket
    if ticker not in [b["ticker"] for b in basket]:
        default = 100_000 if (ticker.endswith((".NS", ".BO")) or ticker.startswith(rk.MF_PREFIX)) else 10_000
        basket.append({"ticker": ticker, "name": name, "kind": kind, "amount": float(default)})
        st.session_state.bld_ver += 1


def _bld_pick_popular():
    pick = st.session_state.get("bld_pop")
    if pick:
        grp = stocks.POPULAR[st.session_state.get("bld_mkt", "India")]
        _bld_add(grp[pick], pick, "ETF" if "ETF" in pick else "Stock")
    st.session_state.bld_pop = None


def portfolio_builder():
    ss = st.session_state
    ss.setdefault("bld_basket", [])
    ss.setdefault("bld_ver", 0)
    st.subheader("Build a portfolio and test it")
    md("<span class='small-note'>Pick companies, ETFs or Indian mutual funds by name, decide how much to put in each, "
       "and see how this mix would have done in the past and what the next 1–10 years could look like. "
       "Nothing is bought; this is a sandbox.</span>", unsafe_allow_html=True)

    # ---- choose holdings
    c1, c2 = st.columns([3, 2], gap="large")
    with c1:
        q = st.text_input("Search by name", placeholder="e.g. Reliance, HDFC Bank, Apple, Parag Parikh Flexi Cap",
                          key="bld_q", icon=":material/search:")
        if q and len(q.strip()) >= 2:
            found = stock_search(q.strip(), "India" if home_market == "India" else "US")
            options = [] if not found else [(f["ticker"], f["name"], f["kind"], f"{f['kind']} · {f['where']}")
                                            for f in found]
            options += [(f"{rk.MF_PREFIX}{code}", mf.short_label(name), "Mutual fund", f"Mutual fund · {name}")
                        for code, name in (search_funds(q, 6) if len(q.strip()) >= 3 else [])]
            if found is None:
                st.caption("Stock search is unavailable right now (Yahoo Finance didn't answer). Funds still work.")
            if options:
                choice = st.selectbox("Results", options, key="bld_res",
                                      format_func=lambda o: f"{o[1]}  —  {o[3]}" if o[2] != "Mutual fund" else f"{o[3]}")
                if st.button("Add to portfolio", icon=":material/add:", key="bld_add", type="primary"):
                    _bld_add(choice[0], choice[1], choice[2])
                    st.rerun()
            elif found is not None:
                st.caption("Nothing matches. Try the company's common name, e.g. 'Infosys' or 'Tata Steel'.")
    with c2:
        st.segmented_control("Popular picks", ["India", "US"], default="India" if home_market == "India" else "US",
                             key="bld_mkt")
        st.pills("Popular", list(stocks.POPULAR[ss.get("bld_mkt") or "India"]), key="bld_pop",
                 on_change=_bld_pick_popular, label_visibility="collapsed")

    basket = ss.bld_basket
    if not basket:
        st.info("Add at least one holding to start. Try a few large companies plus an index or gold ETF.",
                icon=":material/playlist_add:")
        return
    indian = [b["ticker"].endswith((".NS", ".BO")) or b["ticker"].startswith(rk.MF_PREFIX) for b in basket]
    BC = "₹" if sum(indian) >= len(basket) / 2 else "$"

    # ---- amounts
    ticket = pd.DataFrame({"Holding": [b["name"] for b in basket], "Type": [b["kind"] for b in basket],
                           "Amount": [b["amount"] for b in basket], "Remove": False})
    ed = st.data_editor(ticket, hide_index=True, width="stretch", key=f"bld_ed_{ss.bld_ver}",
                        disabled=["Holding", "Type"],
                        column_config={"Amount": st.column_config.NumberColumn(f"Amount to invest ({BC})", min_value=0.0,
                                                                               step=1000.0, format="localized"),
                                       "Remove": st.column_config.CheckboxColumn("Remove")})
    if ed["Remove"].any():
        ss.bld_basket = [b for b, r in zip(basket, ed["Remove"]) if not r]
        ss.bld_ver += 1
        st.rerun()
    for b, a in zip(basket, ed["Amount"].fillna(0)):
        b["amount"] = float(a)
    amounts = np.array([b["amount"] for b in basket])
    total = amounts.sum()
    if total <= 0:
        st.warning("Enter an amount for at least one holding.")
        return
    if sum(indian) not in (0, len(basket)):
        st.caption("This mix has Indian and US holdings. Each one's return is in its own currency; "
                   "rupee–dollar moves are left out.")

    s1, s2, s3 = st.columns([2, 2, 1])
    hist = s1.segmented_control("Learn from the past", ["3 years", "5 years", "10 years", "15 years"],
                                default="10 years", key="bld_hist")
    years_back = int((hist or "10 years").split()[0])
    bench_t, bench_lbl = (peers.NIFTY50_TR, "Nifty 50 TR") if BC == "₹" else ("SPY", "S&P 500")
    s2.markdown(f"<div class='small-note' style='padding-top:30px'>Compared with <b>{bench_lbl}</b> · total "
                f"<b>{money(total, BC)}</b> in {len(basket)} holding{'s' if len(basket) > 1 else ''}</div>",
                unsafe_allow_html=True)
    if s3.button("Clear all", icon=":material/delete:", key="bld_clear"):
        ss.bld_basket, ss.bld_ver = [], ss.bld_ver + 1
        st.rerun()

    live = [(b, a) for b, a in zip(basket, amounts) if a > 0]
    tick = tuple(b["ticker"] for b, _ in live)
    wts = tuple(np.round(np.array([a for _, a in live]) / total, 6))
    names = {b["ticker"]: b["name"] for b, _ in live}
    start_b = (pd.Timestamp.today() - pd.DateOffset(years=years_back)).date()
    try:
        with st.spinner("Fetching prices and testing the mix..."):
            rb = load_report(tick, wts, bench_t, start_b, dt.date.today(), rf, level)
    except Exception as e:
        st.error(f"Couldn't test this mix: {e}")
        return
    if rb["dropped_tickers"]:
        st.warning("No price history for: " + ", ".join(names.get(t, t) for t in rb["dropped_tickers"])
                   + ". Left out of the test.")
    bcol_raw = f"Benchmark ({bench_t})"
    rb = relabel(rb, {**names, bcol_raw: "Benchmark"})
    sm = rb["summary"]
    P = sm.loc["Portfolio"]
    B = sm.loc["Benchmark"] if "Benchmark" in sm.index else None
    b0, b1 = rb["period"]
    yrs = (b1 - b0).days / 365.25
    if b0 > pd.Timestamp(start_b) + pd.Timedelta(days=20):
        st.info(f"The test starts in {b0:%b %Y}, the first date every holding has prices.", icon=":material/info:")

    # ---- verdict
    port_b = rb["portfolio"].dropna()
    bench_r = rb["returns"][bench_t].reindex(port_b.index) if bench_t in rb["returns"] else None
    grow = total * (1 + port_b).cumprod()
    if B is not None:
        better_ret = P["Annualized Return"] > B["Annualized Return"]
        calmer = abs(P["Max Drawdown"]) < abs(B["Max Drawdown"])
        tone = "good" if better_ret and calmer else "bad" if not better_ret and not calmer else "ok"
        verdict_txt = (f"Over the past {yrs:.1f} years this mix returned <b>{pct(P['Annualized Return'])} a year</b> vs "
                       f"{pct(B['Annualized Return'])} for {bench_lbl}, with a worst fall of <b>{pct(P['Max Drawdown'])}</b> "
                       f"vs {pct(B['Max Drawdown'])}. "
                       + {"good": "More return with smaller falls: a strong mix, if the past is any guide.",
                          "ok": ("Higher return, but bigger falls: you are paid for extra risk." if better_ret else
                                 "Smoother than the index, but it earned less: a trade of return for calm."),
                          "bad": f"Less return and bigger falls than simply owning {bench_lbl}."}[tone])
        md(f"<div class='backdrop {tone}'>{verdict_txt}</div>", unsafe_allow_html=True)
    k = st.columns(4)
    dl = lambda a, b_: None if B is None else f"{(a - b_) * 100:+.1f} pts vs index"  # noqa: E731
    k[0].metric("Return per year", pct(P["Annualized Return"]), dl(P["Annualized Return"], B["Annualized Return"] if B is not None else 0))
    k[1].metric("Volatility per year", pct(P["Annualized Vol"]), dl(P["Annualized Vol"], B["Annualized Vol"] if B is not None else 0),
                delta_color="inverse")
    k[2].metric("Worst fall", pct(P["Max Drawdown"]), dl(P["Max Drawdown"], B["Max Drawdown"] if B is not None else 0))
    k[3].metric(f"{money(total, BC)} became", money(grow.iloc[-1], BC), help=f"Invested in {b0:%b %Y}, held to today.")

    # ---- past growth
    fig = go.Figure()
    fig.add_scatter(x=grow.index, y=grow, name="Your mix", line=dict(color=BLUE, width=2.2),
                    hovertemplate=BC + "%{y:,.0f}")
    if bench_r is not None:
        fig.add_scatter(x=grow.index, y=total * (1 + bench_r.fillna(0)).cumprod(), name=bench_lbl,
                        line=dict(color="#98a2b3", width=2), hovertemplate=BC + "%{y:,.0f}")
    fig.update_layout(hovermode="x unified")
    show(style_fig(fig, 340, yfmt="$,.0f", cur=BC), f"If you had invested {money(total, BC)} in {b0:%b %Y}")

    # ---- future
    st.markdown("##### The next few years")
    if len(port_b) < 260:
        st.info("Needs at least a year of shared price history to simulate the future.")
    else:
        hz = st.slider("Years from now", 1, 10, 5, key="bld_h")
        with st.spinner("Simulating 2,000 possible futures..."):
            qb, loss_b, cash_b = future_bands(port_b, rf)
        i = hz * 12
        good, mid, bad = qb[90][i] * total, qb[50][i] * total, qb[10][i] * total
        cg = lambda x: (x / total) ** (1 / hz) - 1  # noqa: E731
        cc = st.columns(3)
        cc[0].markdown(f"<div class='case good'><span class='lbl'>Good case</span><span class='big'>{money(good, BC)}</span>"
                       f"{pct(cg(good))} a year<br><span class='small-note'>1 in 10 futures did better</span></div>",
                       unsafe_allow_html=True)
        cc[1].markdown(f"<div class='case mid'><span class='lbl'>Middle</span><span class='big'>{money(mid, BC)}</span>"
                       f"{pct(cg(mid))} a year<br><span class='small-note'>half did better, half worse</span></div>",
                       unsafe_allow_html=True)
        cc[2].markdown(f"<div class='case bad'><span class='lbl'>Bad case</span><span class='big'>{money(bad, BC)}</span>"
                       f"{pct(cg(bad))} a year<br><span class='small-note'>1 in 10 futures did worse</span></div>",
                       unsafe_allow_html=True)
        st.write("")
        md(f"After **{hz} year{'s' if hz > 1 else ''}**: **{pct(loss_b[hz-1], 0)}** chance of ending below "
           f"{money(total, BC)}, and **{pct(cash_b[hz-1], 0)}** chance of beating a {pct(rf)} deposit "
           f"({money(total * (1 + rf) ** hz, BC)}).")
        dates = pd.date_range(b1, periods=i + 1, freq="MS")
        f2 = go.Figure()
        f2.add_scatter(x=dates, y=qb[90][:i+1] * total, line=dict(width=0), hoverinfo="skip", showlegend=False)
        f2.add_scatter(x=dates, y=qb[10][:i+1] * total, fill="tonexty", fillcolor="rgba(42,120,214,0.14)",
                       line=dict(width=0), name="Good to bad case", hoverinfo="skip")
        f2.add_scatter(x=dates, y=qb[90][:i+1] * total, name="Good case", line=dict(color=GREEN, width=2, dash="dot"),
                       hovertemplate=BC + "%{y:,.0f}")
        f2.add_scatter(x=dates, y=qb[50][:i+1] * total, name="Middle", line=dict(color=BLUE, width=2.5),
                       hovertemplate=BC + "%{y:,.0f}")
        f2.add_scatter(x=dates, y=qb[10][:i+1] * total, name="Bad case", line=dict(color=CRITICAL, width=2, dash="dot"),
                       hovertemplate=BC + "%{y:,.0f}")
        f2.add_scatter(x=[dates[0], dates[-1]], y=[total, total], name="What you put in", mode="lines",
                       line=dict(color=INK_2, width=1, dash="dash"), hoverinfo="skip")
        f2.update_layout(hovermode="x unified")
        show(style_fig(f2, 360, yfmt="$,.0f", cur=BC), f"Possible value of {money(total, BC)} over {hz} years")

    # ---- holdings and risk
    st.markdown("##### What's inside")
    hn = [names[t] for t in tick if names[t] in sm.index]
    wser = pd.Series(np.array(wts), index=[names[t] for t in tick])
    rc = rb["risk_contribution"]
    tbl = pd.DataFrame({"Holding": hn, "Money": [pct(wser[n]) for n in hn], "Share of risk": [pct(rc.get(n, np.nan)) for n in hn],
                        "Return / yr": [pct(sm.loc[n, "Annualized Return"]) for n in hn],
                        "Volatility": [pct(sm.loc[n, "Annualized Vol"]) for n in hn],
                        "Worst fall": [pct(sm.loc[n, "Max Drawdown"]) for n in hn]})
    l, r_ = st.columns([3, 2], gap="large")
    l.dataframe(tbl, hide_index=True, width="stretch")
    with r_:
        f3 = go.Figure()
        f3.add_bar(y=hn, x=[wser[n] for n in hn], name="Money", orientation="h", marker=dict(color=BLUE, cornerradius=3))
        f3.add_bar(y=hn, x=[rc.get(n, 0) for n in hn], name="Risk", orientation="h", marker=dict(color=ORANGE, cornerradius=3))
        f3.update_layout(barmode="group", bargap=0.3, bargroupgap=0.08, yaxis=dict(autorange="reversed"))
        show(style_fig(f3, 80 + 44 * len(hn), xfmt=".0%"), "Money vs share of risk")

    # ---- crises
    stb = rb["stress"]
    cov_b = stb[stb["Covered"] == True]  # noqa: E712
    if len(cov_b):
        bst = rk.stress_test(rb["returns"][[bench_t]], [1.0]) if bench_t in rb["returns"] else None
        crisis = pd.DataFrame({"Crisis": cov_b.index, "Your mix": cov_b["Portfolio Return"].astype(float).map(pct)})
        if bst is not None:
            crisis[bench_lbl] = bst["Portfolio Return"].reindex(cov_b.index).astype(float).map(pct).values
        st.markdown("##### In past market crises")
        st.dataframe(crisis, hide_index=True, width="stretch")

    # ---- crisp notes
    notes = []
    top = wser.idxmax()
    if wser.max() > 0.30 and len(wser) > 1:
        notes.append(f"**{top}** is {pct(wser.max(), 0)} of the money. Most advisers cap any single stock at 10–20%.")
    big_risk = rc.idxmax() if len(rc) else None
    if big_risk is not None and rc.max() > wser.get(big_risk, 0) + 0.10:
        notes.append(f"**{big_risk}** carries {pct(rc.max(), 0)} of the risk for {pct(wser.get(big_risk, 0), 0)} of the money.")
    corr_b = rb["correlation"].values
    if len(wser) >= 3:
        avg_c = corr_b[np.triu_indices_from(corr_b, 1)].mean()
        if avg_c > 0.6:
            notes.append(f"Your picks move together (average correlation {avg_c:.2f}): add something different, "
                         "like a bond fund or gold.")
    stock_only = all(b["kind"] == "Stock" for b, _ in live)
    if stock_only and len(live) < 8:
        notes.append(f"{len(live)} stock{'s' if len(live) > 1 else ''} is a concentrated bet. An index fund gives you "
                     "50+ companies in one go.")
    if yrs < 5:
        notes.append(f"Only {yrs:.1f} years of shared history: treat the numbers as rough.")
    if notes:
        md("<div class='backdrop ok'><b>Before you invest</b><ul style='margin:6px 0 0 18px;padding:0'>"
           + "".join(f"<li>{n}</li>" for n in notes).replace("**", "") + "</ul></div>", unsafe_allow_html=True)
    st.caption("Past prices replayed; the future can be very different. Returns include dividends for stocks and ETFs "
               "(Yahoo adjusted prices) and are before tax and costs. Not a recommendation.")


with t_build:
    portfolio_builder()

with st.sidebar.expander("How it works · data · disclaimer", icon=":material/info:"):
    md(f"""
#### About
Built on **risk_kit.py**, an open Python toolkit for portfolio risk. The beginner's guide (`Risk_Kit_Guide.pdf`) explains every concept on these pages from scratch.

**Data.** Daily adjusted closing prices from Yahoo Finance via the `yfinance` library. Adjusted prices include dividends and splits, so returns are *total returns*.
Yahoo's data is free but unofficial: it may contain errors, gaps or delays and can change without notice.
Your analysis covers **{t0:%d %b %Y} – {t1:%d %b %Y}** ({len(port):,} trading days).

**Method notes.**
- The portfolio is rebalanced to your target weights every day (a standard simplifying assumption). The What-if tab shows realistic rebalancing with costs.
- Statistics are annualized with 252 trading days per year.
- Indian mutual funds are valued at their daily NAV (AMFI data via mfapi.in). Growth-option NAVs include reinvested income, so they are total returns.
- Market indices such as ^NSEI (Nifty 50) or ^GSPC are *price* indices: they leave out dividends, so they understate the market's total return by roughly 1–2% a year. Comparing a fund or ETF against them flatters the fund slightly. A total-return ETF (for example SPY) avoids this.
- VaR is one-day and historical unless stated; the backtest uses a 250-day rolling window.
- Past performance does not predict future results. Every number here describes the **past**.

#### Disclaimer
This tool is for **education and information only**. It is not investment, tax or financial advice, and nothing here is a recommendation to buy or sell any security. Consider speaking to a qualified financial adviser before making investment decisions.
""")
