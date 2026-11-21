"""Tests for risk_kit. Run from the risk_kit folder with:  python -m pytest -q"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import risk_kit as rk  # noqa: E402


@pytest.fixture
def monthly():
    rng = np.random.default_rng(0)
    idx = pd.period_range("2000-01", periods=240, freq="M")
    return pd.DataFrame(rng.normal(0.007, 0.045, (240, 3)), index=idx, columns=["A", "B", "C"])


@pytest.fixture
def daily():
    rng = np.random.default_rng(1)
    idx = pd.bdate_range("2006-01-02", "2023-12-29")
    market = rng.normal(0.0004, 0.012, len(idx))
    data = {"MKT": market,
            "HIBETA": 1.5*market + rng.normal(0, 0.006, len(idx)),
            "BOND": rng.normal(0.0001, 0.003, len(idx))}
    return pd.DataFrame(data, index=idx)


# --- data helpers ---------------------------------------------------------------

def test_prices_to_returns_and_frequency():
    idx = pd.bdate_range("2024-01-01", "2024-12-31")
    prices = pd.DataFrame({"X": np.linspace(100, 120, len(idx))}, index=idx)
    d = rk.prices_to_returns(prices, "D")
    m = rk.prices_to_returns(prices, "M")
    assert len(d) == len(idx) - 1
    assert len(m) == 11
    assert rk.infer_periods_per_year(d) == 252
    assert rk.infer_periods_per_year(m) == 12
    total = (1 + m["X"]).prod() - 1
    assert total == pytest.approx(prices["X"].resample("ME").last().iloc[-1] / prices["X"].resample("ME").last().iloc[0] - 1)


def test_clean_returns_warns_on_extremes():
    r = pd.Series([0.01, 5.0, -0.02])
    with pytest.warns(UserWarning):
        rk.clean_returns(r)


# --- drawdowns and performance ----------------------------------------------------

def test_drawdown_worked_example():
    r = pd.Series([0.10, 0.05, -0.20, -0.10, 0.15, 0.25])
    dd = rk.drawdown(r)
    assert dd["Drawdown"].min() == pytest.approx(-0.28)
    assert rk.max_drawdown(r) == pytest.approx(-0.28)
    dur = rk.drawdown_duration(r)
    assert dur["Longest"] == 3 and dur["Current"] == 0


def test_annualize_and_sharpe():
    r = pd.Series([0.01]*12, index=pd.period_range("2020-01", periods=12, freq="M"))
    assert rk.annualize_rets(r) == pytest.approx(1.01**12 - 1)
    noisy = pd.Series([0.05, -0.03]*60, index=pd.period_range("2010-01", periods=120, freq="M"))
    assert rk.annualize_vol(noisy) == pytest.approx(noisy.std()*np.sqrt(12))
    assert rk.sharpe_ratio(noisy, 0.03) < rk.sharpe_ratio(noisy, 0.0)


def test_sortino_and_calmar(monthly):
    s = rk.sortino_ratio(monthly, 0.02)
    assert s.shape == (3,)
    assert np.all(np.isfinite(s))
    c = rk.calmar_ratio(monthly["A"])
    assert c == pytest.approx(rk.annualize_rets(monthly["A"]) / -rk.max_drawdown(monthly["A"]))


def test_summary_stats_shape(monthly):
    st = rk.summary_stats(monthly)
    assert list(st.index) == ["A", "B", "C"]
    assert "Sortino Ratio" in st.columns and "Max Drawdown" in st.columns


# --- VaR ------------------------------------------------------------------------

def test_gaussian_var_formula():
    r = pd.Series(np.random.default_rng(2).normal(0.01, 0.04, 100_000))
    assert rk.var_gaussian(r, 5) == pytest.approx(-(0.01 - 1.6449*0.04), abs=1e-3)
    assert rk.cvar_gaussian(r, 5) > rk.var_gaussian(r, 5)


def test_cvar_exceeds_var(monthly):
    assert (rk.cvar_historic(monthly, 5) >= rk.var_historic(monthly, 5)).all()


def test_monte_carlo_var_close_to_historic(monthly):
    mc = rk.var_monte_carlo(monthly, 5, n_sims=200_000, seed=0)
    hist = rk.var_historic(monthly, 5)
    assert np.allclose(mc["VaR"], hist, atol=0.004)
    single = rk.var_monte_carlo(monthly["A"], 5, method="normal", seed=0)
    assert set(single.index) == {"VaR", "CVaR"}


def test_var_backtest_and_kupiec():
    r = pd.Series(np.random.default_rng(3).normal(0, 0.01, 3000))
    bt = rk.var_backtest(r, level=5, window=250, method="gaussian")
    k = rk.kupiec_test(bt["Breach"], 5)
    assert 0.03 < k["Breach Rate"] < 0.07
    assert k["Model OK"]


# --- benchmark measures ------------------------------------------------------------

def test_beta_and_capm_recover_known_beta(daily):
    b = rk.beta(daily["HIBETA"], daily["MKT"])
    assert b == pytest.approx(1.5, abs=0.02)
    cp = rk.capm(daily[["HIBETA", "BOND"]], daily["MKT"], 0.0)
    assert cp.loc["HIBETA", "Beta"] == pytest.approx(1.5, abs=0.02)
    assert cp.loc["HIBETA", "R-squared"] > 0.9
    assert abs(cp.loc["BOND", "Beta"]) < 0.05


def test_tracking_error_and_ir(daily):
    assert rk.tracking_error(daily["MKT"], daily["MKT"]) == pytest.approx(0)
    te = rk.tracking_error(daily["HIBETA"], daily["MKT"])
    assert te > 0
    assert np.isfinite(rk.information_ratio(daily["HIBETA"], daily["MKT"]))
    cap = rk.up_down_capture(daily["HIBETA"], daily["MKT"])
    assert cap.loc["HIBETA", "Up Capture"] == pytest.approx(1.5, abs=0.05)


def test_capm_expected_return():
    assert rk.capm_expected_return(1.2, 0.03, 0.08) == pytest.approx(0.09)


# --- rolling and EWMA ----------------------------------------------------------------

def test_rolling_and_ewma(daily):
    rv = rk.rolling_vol(daily["MKT"], 63)
    assert rv.dropna().mean() == pytest.approx(0.012*np.sqrt(252), rel=0.1)
    ev = rk.ewma_vol(daily["MKT"])
    assert ev.dropna().iloc[-1] == pytest.approx(0.012*np.sqrt(252), rel=0.35)
    rb = rk.rolling_beta(daily["HIBETA"], daily["MKT"], 126)
    assert rb.dropna().mean() == pytest.approx(1.5, abs=0.05)
    cov = rk.ewma_cov(daily)
    assert cov.shape == (3, 3)


# --- portfolio construction ------------------------------------------------------------

def test_portfolio_vol_worked_example():
    s = np.array([0.2, 0.1])
    for rho, expected in [(1, 0.15), (0, 0.1118), (-0.5, 0.0866)]:
        cov = np.outer(s, s) * np.array([[1, rho], [rho, 1]])
        assert rk.portfolio_vol(np.array([0.5, 0.5]), cov) == pytest.approx(expected, abs=1e-4)


def test_gmv_and_msr_two_assets():
    er = pd.Series([0.10, 0.06], index=["A", "B"])
    s = np.array([0.2, 0.1])
    cov = pd.DataFrame(np.outer(s, s)*np.array([[1, .2], [.2, 1]]), index=er.index, columns=er.index)
    assert rk.gmv(cov)[0] == pytest.approx(0.143, abs=0.005)
    assert rk.msr(0.03, er, cov)[0] == pytest.approx(0.387, abs=0.005)


def test_weight_caps(monthly):
    er = rk.annualize_rets(monthly)
    cov = monthly.cov()*12
    w = rk.msr(0.0, er, cov, max_weight=0.4)
    assert w.max() <= 0.4 + 1e-6 and w.sum() == pytest.approx(1)
    with pytest.raises(ValueError):
        rk.gmv(cov, max_weight=0.2)


def test_risk_parity_equalizes_risk(monthly):
    cov = monthly.cov()
    cov.iloc[0, 0] *= 4
    w = rk.risk_parity(cov)
    rc = rk.risk_contribution(w, cov)
    assert np.allclose(rc, 1/3, atol=1e-4)
    assert w["A"] < w["B"]


def test_max_diversification(monthly):
    cov = monthly.cov()
    w = rk.max_diversification(cov)
    assert rk.diversification_ratio(w, cov) >= rk.diversification_ratio(np.repeat(1/3, 3), cov) - 1e-9


def test_shrinkage_endpoints(monthly):
    assert np.allclose(rk.cov_shrinkage(monthly, 0), monthly.cov())
    assert np.allclose(rk.cov_shrinkage(monthly, 1), rk.cov_constant_correlation(monthly))


def test_efficient_frontier_table(monthly):
    ef = rk.efficient_frontier(10, rk.annualize_rets(monthly), monthly.cov()*12)
    assert list(ef.columns[:2]) == ["Returns", "Volatility"]
    assert np.allclose(ef[["A", "B", "C"]].sum(axis=1), 1, atol=1e-4)


# --- backtesting ------------------------------------------------------------------------

def test_backtest_rebalanced_every_period_matches_simple_formula(monthly):
    w = [0.5, 0.3, 0.2]
    bt = rk.backtest_portfolio(monthly, w, rebalance="M")
    expected = rk.portfolio_return_series(monthly, w)
    assert np.allclose(bt["Return"], expected)


def test_backtest_costs_and_drift(monthly):
    w = [0.5, 0.3, 0.2]
    free = rk.backtest_portfolio(monthly, w, rebalance="Q", cost_bps=0)
    costly = rk.backtest_portfolio(monthly, w, rebalance="Q", cost_bps=50)
    assert costly["Wealth"].iloc[-1] < free["Wealth"].iloc[-1]
    hold = rk.backtest_portfolio(monthly, w, rebalance=None)
    assert hold["Turnover"].iloc[1:].sum() == 0
    expected_final = ((1 + monthly).prod() * pd.Series(w, index=monthly.columns)).sum()
    assert hold["Wealth"].iloc[-1] == pytest.approx(expected_final)


def test_backtest_with_weighting_function(monthly):
    bt = rk.backtest_portfolio(monthly, rk.weight_gmv, rebalance="Q", estimation_window=36,
                               cov_estimator=rk.cov_shrinkage, delta=0.5)
    assert len(bt) == len(monthly) - 36
    assert np.allclose(bt.attrs["weights"].sum(axis=1), 1, atol=1e-6)
    stats, rets = rk.compare_strategies(monthly, {"EW": rk.weight_ew, "Fixed": [0.4, 0.4, 0.2]},
                                        estimation_window=36)
    assert "Annual Turnover" in stats.columns


# --- CPPI and simulation ---------------------------------------------------------------

def test_cppi_keeps_floor_in_mild_markets(monthly):
    bt = rk.run_cppi(monthly[["A"]], m=3, floor=0.8)
    assert (bt["Wealth"]["A"] >= 0.8*1000 - 1e-9).all()
    assert "floor_value" in bt and bt["floor"] == 0.8


def test_gbm_seed_and_shape():
    a = rk.gbm(n_years=2, n_scenarios=5, seed=1)
    b = rk.gbm(n_years=2, n_scenarios=5, seed=1)
    assert a.shape == (25, 5) and a.equals(b)


def test_cir_shapes_and_positive_rates():
    rates, prices = rk.cir(n_years=5, n_scenarios=20, seed=0)
    assert rates.shape == (61, 20) and (rates.values >= 0).all()
    assert prices.iloc[-1].mean() == pytest.approx(1, abs=0.01)


# --- bonds and LDI -------------------------------------------------------------------------

def test_bond_math():
    assert rk.bond_price(10, 100, 0.05, 2, 0.05) == pytest.approx(100)
    assert rk.bond_price(10, 100, 0.05, 2, 0.06) < 100
    zero = pd.Series([100.0], index=[7.0])
    assert rk.macaulay_duration(zero, 0.04) == pytest.approx(7)
    liab = pd.Series([100.0, 100.0], index=[3.0, 5.0])
    expected_pv = 100/1.03**3 + 100/1.03**5
    assert rk.pv(liab, 0.03) == pytest.approx(expected_pv)
    assert rk.funding_ratio(180, liab, 0.03) == pytest.approx(180/expected_pv)


def test_match_durations():
    short = rk.bond_cash_flows(5, 1000, 0.05, 1)
    long = rk.bond_cash_flows(20, 1000, 0.05, 1)
    target = pd.Series([1000.0], index=[10.0])
    ws = rk.match_durations(target, short, long, 0.04)
    d_mix = ws*rk.macaulay_duration(short, 0.04) + (1-ws)*rk.macaulay_duration(long, 0.04)
    assert d_mix == pytest.approx(10)


# --- stress tests ------------------------------------------------------------------------------

def test_stress_test(daily):
    st = rk.stress_test(daily[["MKT", "BOND"]], [0.6, 0.4])
    assert not st.loc["Dot-com crash", "Covered"]
    assert st.loc["COVID crash", "Covered"]
    window = daily.loc["2020-02-19":"2020-03-23", ["MKT", "BOND"]]
    expected = (1 + window @ np.array([0.6, 0.4])).prod() - 1
    assert st.loc["COVID crash", "Portfolio Return"] == pytest.approx(expected)


def test_shock_test():
    out = rk.shock_test({"Stocks": 0.6, "Bonds": 0.4}, {"Stocks": -0.30, "Bonds": 0.05})
    assert out["Portfolio Return"] == pytest.approx(-0.16)


# --- dashboard report (Yahoo download replaced by synthetic prices) ---------------------------

def test_portfolio_report_offline(monkeypatch, daily):
    prices = 100 * (1 + daily).cumprod()

    def fake_get_prices(tickers, **kwargs):
        return prices[[t for t in tickers if t in prices.columns]]

    monkeypatch.setattr(rk, "get_prices", fake_get_prices)
    rep = rk.portfolio_report(["MKT", "BOND", "MISSING"], [0.5, 0.3, 0.2], benchmark="MKT")
    assert rep["dropped_tickers"] == ["MISSING"]
    assert rep["weights"].sum() == pytest.approx(1)
    assert rep["weights"]["MKT"] == pytest.approx(0.625)
    for key in ["summary", "capm", "stress", "var_backtest", "kupiec", "risk_contribution"]:
        assert key in rep
    assert rep["risk_contribution"].sum() == pytest.approx(1)
    with pytest.raises(ValueError):
        rk.portfolio_report(["MKT", "BOND"], [0.5], benchmark="MKT")


def test_clean_price_series_removes_glitch_and_fixes_split():
    idx = pd.bdate_range("2020-01-01", periods=10)
    glitch = pd.Series([33.4, 33.5, 33.6, 0.3355, 0.3365, 33.65, 33.7, 33.8, 34.0, 34.1], index=idx)
    out = rk.clean_price_series(glitch)
    assert out.iloc[3:5].isna().all() and out.dropna().pct_change().abs().max() < 0.01
    split = pd.Series([1000, 1010, 1020, 102, 103, 104, 105, 106, 107, 108], index=idx, dtype=float)
    fixed = rk.clean_price_series(split)
    assert fixed.iloc[0] == pytest.approx(100) and fixed.pct_change().abs().max() < 0.02
    crash = pd.Series([100, 101, 50, 51, 52, 53, 54, 55, 56, 57], index=idx, dtype=float)
    assert rk.clean_price_series(crash).iloc[0] == 100  # a real 50% fall is left alone
