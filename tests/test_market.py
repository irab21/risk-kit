"""Tests for market conditions (offline, synthetic prices)."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import market  # noqa: E402

DAYS = pd.bdate_range("2024-01-01", periods=520)


def series(path):
    return pd.Series(path, index=DAYS[:len(path)])


def test_detect_market():
    assert market.detect_market(["MF:102885", "MF:1"]) == "India"
    assert market.detect_market(["RELIANCE.NS", "SPY"]) == "India"
    assert market.detect_market(["SPY", "AGG"]) == "US"
    assert market.detect_market(["VOD.L", "SAP.DE"]) == "Global"
    assert market.detect_market([]) == "India"


def test_stats_and_bull_regime():
    rng = np.random.default_rng(0)
    up = series(100 * np.exp(np.cumsum(0.0008 + rng.normal(0, 0.006, 520))))
    st = market.stats(up)
    assert st["1Y"] > 0 and st["from_high"] <= 0 and st["1D"] == up.iloc[-1] / up.iloc[-2] - 1
    calm_vix = series(np.r_[np.full(400, 16.0), np.full(120, 12.0)])
    reg = market.regime(st, calm_vix, "Index")
    assert reg["trend"] == "Uptrend" and reg["volatility"] == "Calm" and reg["tone"] == "good"


def test_bear_regime_and_implications():
    path = np.r_[np.linspace(100, 130, 300), np.linspace(130, 95, 220)]
    reg = market.regime(market.stats(series(path)), series(np.r_[np.full(400, 14.0), np.full(120, 26.0)]), "Index")
    assert reg["drawdown"] == "Bear market" and reg["trend"] == "Downtrend" and reg["volatility"] == "Elevated"
    assert reg["mood"] == "Stressed"
    lines = market.implications(reg, beta=0.7, value=100_000)
    assert any("SIPs" in x for x in lines) and any("instalments" in x for x in lines) and any("7%" in x for x in lines)
