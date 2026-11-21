"""Tests for mutual fund matching and NAV prices, offline (network calls are replaced)."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import mutual_funds as mf  # noqa: E402
import risk_kit as rk  # noqa: E402

SCHEMES = [
    (100, "HDFC Aggressive Hybrid Fund - Regular Plan - Growth Option"),
    (101, "HDFC Aggressive Hybrid Fund - Direct Plan - Growth Option"),
    (102, "HDFC Aggressive Hybrid Fund - Regular Plan - IDCW"),
    (103, "HDFC Balanced Advantage Fund - Regular Plan - Growth"),
    (200, "BANDHAN MID CAP FUND - Regular Plan - Growth"),
    (201, "Bandhan Large & Mid Cap Fund - Regular Plan - Growth"),
    (300, "Old Fund - Segregated Portfolio 1 - Regular Plan - Growth"),
]


@pytest.fixture
def schemes(monkeypatch, tmp_path):
    monkeypatch.setattr(mf, "SCHEMES_CACHE", tmp_path / "schemes.json")
    monkeypatch.setattr(rk, "_http_json", lambda url, timeout=30: [{"schemeCode": c, "schemeName": n} for c, n in SCHEMES])
    return mf.load_schemes()


def test_parse_name_reads_abbreviations():
    assert mf.parse_name("HDFC AGGRESSIVE HYBRID FUND-REG - Gr") == (["hdfc", "aggressive", "hybrid"], "regular", "growth")
    assert mf.parse_name("Axis Bluechip - Dir - Div Payout")[1:] == ("direct", "idcw")


def test_candidates_respect_plan_and_option(schemes):
    assert mf.candidates("HDFC AGGRESSIVE HYBRID FUND-REG - Gr", schemes)["code"].iloc[0] == 100
    assert mf.candidates("HDFC Aggressive Hybrid - Direct - Growth", schemes)["code"].iloc[0] == 101
    assert mf.candidates("HDFC Aggressive Hybrid Reg IDCW", schemes)["code"].iloc[0] == 102
    assert mf.candidates("Bandhan Mid cap fund - Reg - G r", schemes)["code"].iloc[0] == 200
    assert 300 not in set(schemes["code"])  # segregated portfolios are dropped


def test_nav_check_picks_the_verified_scheme(schemes, monkeypatch):
    # name ranks 201 first for this made-up name, but only 200's NAV matches value / units
    monkeypatch.setattr(mf, "_nav_matches", lambda code, nav: pd.Timestamp("2026-11-10") if code == 200 else None)
    result = mf.match_fund("Bandhan Large Mid Cap - Reg - Gr", units=100, value=1896.3, schemes=schemes)
    assert result["code"] == 200 and result["status"] == "verified"


def test_no_nav_match_falls_back_to_name(schemes, monkeypatch):
    monkeypatch.setattr(mf, "_nav_matches", lambda code, nav: None)
    result = mf.match_fund("HDFC Aggressive Hybrid Reg Gr", units=10, value=1000, schemes=schemes)
    assert result["code"] == 100 and result["status"] == "name only"
    assert mf.match_fund("Totally Unknown Thing", 1, 1, schemes=schemes)["status"] in ("not found", "name only")


def test_short_label():
    assert mf.short_label("BANDHAN MID CAP FUND - Regular Plan - Growth") == "Bandhan Mid Cap"
    assert mf.short_label("HDFC Aggressive Hybrid Fund - Regular Plan - Growth Option") == "HDFC Aggressive Hybrid"


def test_get_prices_mixes_funds_and_yahoo(monkeypatch):
    dates = pd.bdate_range("2024-01-01", periods=60)
    monkeypatch.setattr(rk, "get_mf_nav", lambda code: pd.Series(np.linspace(10, 12, 60), index=dates,
                                                                  name=f"MF:{code}"))
    prices = rk.get_prices(["MF:123"], start="2024-01-15", end="2024-02-15", use_cache=False)
    assert list(prices.columns) == ["MF:123"]
    assert prices.index[0] >= pd.Timestamp("2024-01-15") and prices.index[-1] < pd.Timestamp("2024-02-15")
