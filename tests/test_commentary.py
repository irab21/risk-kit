"""Tests for the plain-English verdict and the future simulation."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import commentary as cmt  # noqa: E402
import risk_kit as rk  # noqa: E402


def facts(**over):
    base = dict(years=10, value=100_000, final_value=260_000, ret=0.10, vol=0.12, sharpe=0.6, max_dd=-0.25,
                recovery="about 12 months", rf=0.03, n_holdings=4, top_risk_name="SPY", top_risk_share=0.3,
                top_risk_weight=0.3, avg_corr=0.4, vol_now=0.12, bench="SPY", bench_ret=0.08, bench_vol=0.18,
                bench_sharpe=0.3, bench_max_dd=-0.50, bench_final_value=216_000, beta=0.6, alpha=0.02, alpha_t=2.5,
                down_capture=0.6, crisis_port=-0.10, crisis_bench=-0.25, n_crises=3, var_ok=True, breach_rate=0.05)
    base.update(over)
    return cmt.PortfolioFacts(**base)


def statuses(result):
    return {p["title"]: p["status"] for p in result["points"]}


def test_strong_portfolio():
    r = cmt.build_commentary(facts())
    assert r["grade"] == "Strong"
    assert "Beat the market by 2.0 points a year" in statuses(r)
    assert "Biggest strength: beat the market" in r["summary"]
    assert "$44,000" not in r["summary"]  # money formatting is the caller's


def test_weak_portfolio():
    r = cmt.build_commentary(facts(ret=0.02, final_value=122_000, sharpe=-0.1, vol=0.25, max_dd=-0.6,
                                   top_risk_share=0.8, avg_corr=0.9, crisis_port=-0.4, alpha=-0.03, alpha_t=-2.4))
    s = statuses(r)
    assert r["grade"] == "Weak"
    assert s["Did not beat a risk-free deposit"] == "bad"
    assert s["Concentrated risk"] == "bad"
    assert s["Hit harder than the market in crises"] == "bad"


def test_without_benchmark_and_short_history():
    r = cmt.build_commentary(facts(bench=None, bench_ret=None, years=1.5, n_crises=0, beta=None))
    titles = [p["title"] for p in r["points"]]
    assert "Short track record" in titles
    assert not any("market" in t.lower() and "beat" in t.lower() for t in titles)


def test_bootstrap_paths_shape_and_centre():
    r = pd.Series(np.random.default_rng(0).normal(0.0004, 0.01, 2000))
    paths = rk.bootstrap_paths(r, 252, n_paths=3000, seed=1)
    assert paths.shape == (3000, 253) and (paths[:, 0] == 1).all()
    expected = np.exp(np.log1p(r).mean() * 252)
    assert abs(np.median(paths[:, -1]) / expected - 1) < 0.03
