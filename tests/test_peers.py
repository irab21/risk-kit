"""Tests for the AMFI file parser and peer ranking (offline, synthetic NAVs)."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import peers  # noqa: E402

NAVALL = """Scheme Code;ISIN Div Payout/ ISIN Growth;ISIN Div Reinvestment;Scheme Name;Plan;Option;Net Asset Value;Date

Open Ended Schemes(Hybrid Scheme - Aggressive Hybrid Fund)

Alpha Mutual Fund

1;INF1;-;Alpha Hybrid Fund;Regular Plan;Growth;50.0;18-Nov-2026
2;INF2;-;Alpha Hybrid Fund;Direct Plan;Growth;55.0;18-Nov-2026
3;INF3;INF3B;Alpha Hybrid Fund;Regular Plan;IDCW;20.0;18-Nov-2026

Open Ended Schemes(Hybrid Schemes - Aggressive Hybrid Fund)

Beta Mutual Fund

4;INF4;-;Beta Equity Hybrid Fund - Regular - Growth;;;40.0;18-Nov-2026
5;INF5;-;Beta Equity Hybrid Fund - Direct Plan - Growth;;;44.0;18-Nov-2026
"""


def test_parse_navall_normalises_categories_and_blank_plans():
    df = peers.parse_navall(NAVALL)
    assert df["category"].nunique() == 1
    assert list(df["direct"]) == [False, True, False, False, True]
    assert list(df["growth"]) == [True, True, False, True, True]
    assert peers.sibling_plan(1, df) == 2


def nav(cagr, vol, seed, n=1500):
    r = np.random.default_rng(seed).normal(cagr / 252, vol / np.sqrt(252), n)
    return pd.Series(100 * np.exp(np.cumsum(r)), index=pd.bdate_range("2021-01-01", periods=n))


def test_peer_report_ranks_the_best_fund_on_top():
    df = peers.parse_navall(NAVALL)
    hist = {1: nav(0.20, 0.10, 1), 4: nav(0.02, 0.10, 2)}
    rep = peers.peer_report(1, df, rf=0.06, histories=hist)
    assert rep["n_peers"] == 2 and rep["ranks"]["3Y"]["rank"] == 1 and rep["label"] == "Top quartile"
    worse = peers.peer_report(4, df, rf=0.06, histories=hist)
    assert worse["label"] == "Bottom quartile"


def test_fund_metrics_windows():
    m = peers.fund_metrics(nav(0.10, 0.0001, 3, n=1000), rf=0.06)
    assert abs(m["1Y"] - 0.105) < 0.01 and np.isnan(m["5Y"]) and m["Vol 3Y"] < 0.01


def test_tracked_index_strips_fund_house():
    assert peers.tracked_index("UTI Nifty 50 Index Fund", "UTI Mutual Fund") == "nifty 50 index"
    assert peers.tracked_index("ICICI Prudential Nifty 50 Index Fund", "ICICI Prudential Mutual Fund") == "nifty 50 index"
    assert peers.tracked_index("Nippon India ETF Nifty 50 BeES", "Nippon India Mutual Fund") == "nifty 50 etf"
    assert peers.tracked_index("Motilal Oswal Nifty Next 50 Index Fund", "Motilal Oswal Mutual Fund") == "nifty next 50 index"
