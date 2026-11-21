"""
risk_kit: a toolkit for portfolio risk analysis and construction.

Sections
    1.  Data: Yahoo Finance prices, returns, course CSV files
    2.  Returns, drawdowns and performance measures
    3.  Distribution statistics (skewness, kurtosis, normality)
    4.  Value at Risk and Expected Shortfall, VaR backtesting
    5.  Risk against a benchmark (beta, alpha, CAPM, tracking error)
    6.  Risk over time (rolling statistics, EWMA)
    7.  Portfolio math, covariance estimation, optimization
    8.  Portfolio backtesting with rebalancing and costs
    9.  CPPI portfolio insurance
    10. Simulation (GBM, CIR interest rates)
    11. Bonds and liability-driven investing
    12. Stress testing
    13. One-call reports for dashboards

Conventions
    - Returns are simple returns in decimals: 0.05 means +5%.
    - `level` in VaR/CVaR functions is a percentage: level=5 means the worst 5%.
    - VaR and CVaR are reported as positive numbers for losses.
    - Functions accept a pandas Series (one asset) or DataFrame (one column per asset).
"""

from pathlib import Path
import time
import warnings

import numpy as np
import pandas as pd
import scipy.stats
from scipy.optimize import minimize
from scipy.stats import norm

# CSV data files for the EDHEC course datasets live in a "data" folder next to this script
DATA_DIR = Path(__file__).resolve().parent / "data"
CACHE_DIR = DATA_DIR / "cache"

PERIODS_PER_YEAR = {"D": 252, "W": 52, "M": 12, "Q": 4, "Y": 1}


# =============================================================================
# 1. DATA
# =============================================================================

MF_PREFIX = "MF:"  # Indian mutual funds: "MF:<AMFI scheme code>", e.g. "MF:150402"
MFAPI_URL = "https://api.mfapi.in/mf"


def _http_json(url, timeout=30):
    import json
    import urllib.request
    req = urllib.request.Request(url, headers={"User-Agent": "risk-kit/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def get_mf_nav(scheme_code):
    """
    Full daily NAV history of an Indian mutual fund from AMFI data (via mfapi.in).
    scheme_code: AMFI scheme code (int or str). Returns a Series indexed by date.
    NAVs of growth options include reinvested gains, so their changes are total returns.
    """
    data = _http_json(f"{MFAPI_URL}/{int(scheme_code)}")
    rows = data.get("data") or []
    if not rows:
        raise ValueError(f"No NAV data for scheme {scheme_code}")
    s = pd.Series({pd.to_datetime(r["date"], format="%d-%m-%Y"): float(r["nav"]) for r in rows})
    s = s[s > 0].sort_index()
    return s.rename(f"{MF_PREFIX}{int(scheme_code)}")


def _period_start(period, end=None):
    """'5y' -> the date five years before `end` (today if None); 'max' -> None"""
    if not period or period == "max":
        return None
    end = pd.Timestamp(end) if end else pd.Timestamp.today().normalize()
    n, unit = int("".join(c for c in period if c.isdigit()) or 1), "".join(c for c in period if c.isalpha())
    offsets = {"d": pd.DateOffset(days=n), "wk": pd.DateOffset(weeks=n), "mo": pd.DateOffset(months=n),
               "y": pd.DateOffset(years=n), "ytd": None}
    if unit == "ytd":
        return pd.Timestamp(end.year, 1, 1)
    return end - offsets.get(unit, pd.DateOffset(years=n))


SPLIT_FACTORS = (3, 4, 5, 8, 10, 20, 25, 50, 100)


def clean_price_series(s, max_glitch_days=10):
    """
    Repairs two common Yahoo data errors in one price series:
    - glitches: the price jumps by 45%+ and jumps back within `max_glitch_days` (e.g. a few days quoted
      at 1/100 of the real price); those days are removed;
    - unadjusted splits: a lasting jump by a split-like factor (3x, 10x, 100x, ...); earlier prices are rescaled.
    Returns the cleaned series (same index; removed days are NaN).
    """
    s = s.astype(float).copy()
    v = s.dropna()
    if len(v) < 3:
        return s
    vals, idx = v.to_numpy().copy(), v.index
    keep = np.ones(len(vals), dtype=bool)
    i = 1
    while i < len(vals):
        prev = vals[:i][keep[:i]][-1] if keep[:i].any() else vals[i - 1]
        f = vals[i] / prev
        if 0.55 < f < 1.8:
            i += 1
            continue
        back = next((j for j in range(i + 1, min(i + 1 + max_glitch_days, len(vals)))
                     if 0.8 < vals[j] / prev < 1.25), None)
        if back is not None:
            keep[i:back] = False
            i = back
            continue
        factor = f if f >= 1 else 1 / f
        if any(abs(factor / k - 1) < 0.06 for k in SPLIT_FACTORS):
            vals[:i] = vals[:i] * f
        i += 1
    out = pd.Series(np.where(keep, vals, np.nan), index=idx)
    return out.reindex(s.index).where(s.notna())


def get_prices(tickers, start=None, end=None, period="max", use_cache=True,
               max_cache_age_hours=24):
    """
    Downloads daily adjusted closing prices from Yahoo Finance.

    tickers: a ticker string ("SPY") or a list of tickers (["SPY", "AGG"])
    start, end: dates such as "2015-01-01"; if start is None, `period` is used ("max", "10y", "5y", ...)
    use_cache: saves downloads to data/cache and reuses them for `max_cache_age_hours`

    Indian mutual funds can be mixed in as "MF:<AMFI scheme code>" (NAVs from mfapi.in).

    Returns a DataFrame of prices, one column per ticker, indexed by date.
    Prices are adjusted for dividends and splits, so their changes are total returns.
    """
    if isinstance(tickers, str):
        tickers = [tickers]
    tickers = [t.strip().upper() for t in tickers]
    if not tickers:
        raise ValueError("Give at least one ticker")

    cache_file = None
    if use_cache:
        key = "_".join(sorted(tickers)) + f"__{start}_{end}_{period}"
        key = "".join(c if c.isalnum() or c in "_-." else "-" for c in key)
        cache_file = CACHE_DIR / f"{key}.csv"
        if cache_file.exists():
            age_hours = (time.time() - cache_file.stat().st_mtime) / 3600
            if age_hours < max_cache_age_hours:
                cached = pd.read_csv(cache_file, index_col=0, parse_dates=True)
                return cached[[t for t in tickers if t in cached.columns]]

    yahoo = [t for t in tickers if not t.startswith(MF_PREFIX)]
    funds = [t for t in tickers if t.startswith(MF_PREFIX)]
    frames = []
    if yahoo:
        try:
            import yfinance as yf
        except ImportError as exc:
            raise ImportError("get_prices needs yfinance: pip install yfinance") from exc
        kwargs = dict(auto_adjust=True, progress=False, group_by="column")
        if start is None:
            data = yf.download(yahoo, period=period, **kwargs)
        else:
            data = yf.download(yahoo, start=start, end=end, **kwargs)
        if data is not None and not data.empty:
            ydata = data["Close"]
            if isinstance(ydata, pd.Series):
                ydata = ydata.to_frame(yahoo[0])
            ydata.index = pd.to_datetime(ydata.index).tz_localize(None)
            ydata = ydata.apply(clean_price_series)
            frames.append(ydata)
    for t in funds:
        try:
            nav = get_mf_nav(t[len(MF_PREFIX):])
        except Exception as exc:  # network error or unknown scheme: reported as missing below
            warnings.warn(f"Couldn't download NAVs for {t}: {exc}")
            continue
        lo = pd.Timestamp(start) if start is not None else _period_start(period, end)
        hi = pd.Timestamp(end) - pd.Timedelta(days=1) if end is not None else None  # end is exclusive, as in Yahoo
        frames.append(nav.loc[lo:hi].to_frame(t))
    if not frames:
        raise ValueError(f"No price data returned for {tickers}")

    prices = pd.concat(frames, axis=1).sort_index()
    if yahoo and funds:  # exchanges and fund houses publish on slightly different holidays
        prices = prices.ffill(limit=3)
    prices = prices.dropna(how="all")

    missing = [t for t in tickers if t not in prices.columns or prices[t].isna().all()]
    if missing:
        warnings.warn(f"No data returned for: {', '.join(missing)}")
        prices = prices.drop(columns=[t for t in missing if t in prices.columns])
    prices = prices[[t for t in tickers if t in prices.columns]]

    if cache_file is not None:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        prices.to_csv(cache_file)
    return prices


def prices_to_returns(prices, freq="D", log=False):
    """
    Converts prices into returns.

    freq: "D" daily, "W" weekly, "M" monthly, "Q" quarterly, "Y" yearly.
          Lower frequencies use the last price of each period.
    log:  True gives log returns ln(P_t / P_t-1); False (default) gives simple returns.
    """
    freq = freq.upper()
    if freq not in PERIODS_PER_YEAR:
        raise ValueError(f"freq must be one of {list(PERIODS_PER_YEAR)}")
    if freq != "D":
        rule = {"W": "W-FRI", "M": "ME", "Q": "QE", "Y": "YE"}[freq]
        prices = prices.resample(rule).last()
    if log:
        rets = np.log(prices / prices.shift(1))
    else:
        rets = prices.pct_change(fill_method=None)
    rets = rets.iloc[1:]
    if freq != "D":
        rets.index = rets.index.to_period(freq if freq != "Y" else "Y")
    return rets


def get_yahoo_returns(tickers, start=None, end=None, freq="M", period="max", use_cache=True):
    """Downloads prices from Yahoo Finance and returns their returns at the chosen frequency."""
    prices = get_prices(tickers, start=start, end=end, period=period, use_cache=use_cache)
    return prices_to_returns(prices, freq=freq)


def infer_periods_per_year(r):
    """
    Guesses how many periods per year a return series has from its index:
    252 for daily, 52 weekly, 12 monthly, 4 quarterly, 1 yearly.
    """
    idx = r.index
    if isinstance(idx, pd.PeriodIndex):
        code = idx.freqstr.upper()[0]
        return {"D": 252, "B": 252, "W": 52, "M": 12, "Q": 4, "Y": 1, "A": 1}.get(code, 12)
    if isinstance(idx, pd.DatetimeIndex) and len(idx) > 2:
        days = np.median(np.diff(idx.values).astype("timedelta64[D]").astype(float))
        if days <= 4:
            return 252
        if days <= 10:
            return 52
        if days <= 40:
            return 12
        if days <= 120:
            return 4
        return 1
    raise ValueError("Cannot infer the frequency: pass periods_per_year explicitly")


def _ppy(r, periods_per_year):
    return periods_per_year if periods_per_year is not None else infer_periods_per_year(r)


def clean_returns(r, max_abs_return=1.0):
    """
    Basic data checks before analysis:
    - drops rows where every asset is missing
    - warns about returns larger than `max_abs_return` (100% by default), which are usually data errors
    """
    if not isinstance(r, (pd.Series, pd.DataFrame)):
        raise TypeError("Expected a pandas Series or DataFrame of returns")
    r = r.dropna(how="all") if isinstance(r, pd.DataFrame) else r.dropna()
    extreme = (r.abs() > max_abs_return)
    n_extreme = int(extreme.values.sum())
    if n_extreme:
        warnings.warn(f"{n_extreme} returns above {max_abs_return:.0%} in absolute value: check the data")
    return r


def load_returns_csv(path, date_format=None, percent=False):
    """Loads returns from any CSV whose first column is a date. Set percent=True if values are in %."""
    r = pd.read_csv(path, index_col=0)
    r.index = pd.to_datetime(r.index, format=date_format)
    return r / 100 if percent else r


# --- EDHEC course datasets (CSV files in the data folder) -----------------------

def get_returns():
    """Monthly returns of the bottom (SmallCap) and top (LargeCap) market-cap quintiles"""
    returns = pd.read_csv(DATA_DIR / "Portfolios_Formed_on_ME_monthly_EW.csv",
                          header=0, index_col=0, na_values=-99.99)
    returns = returns[['Lo 20', 'Hi 20']]
    returns.columns = ['SmallCap', 'LargeCap']
    returns = returns/100
    returns.index = pd.to_datetime(returns.index, format="%Y%m").to_period("M")
    return returns


def get_hfi_returns():
    """EDHEC hedge fund index returns"""
    hfi = pd.read_csv(DATA_DIR / "edhec-hedgefundindices.csv",
                      header=0, index_col=0, parse_dates=True, dayfirst=True)
    hfi = hfi/100
    hfi.index = hfi.index.to_period('M')
    return hfi


def _get_ind_file(filename):
    """Loads one of the Ken French 30 industry portfolio files"""
    ind = pd.read_csv(DATA_DIR / filename, header=0, index_col=0)
    ind.index = pd.to_datetime(ind.index, format="%Y%m").to_period("M")
    ind.columns = ind.columns.str.strip()  # remove spaces in the column names
    return ind


def get_ind_returns():
    """Monthly value-weighted returns of 30 industry portfolios"""
    return _get_ind_file("ind30_m_vw_rets.csv")/100


def get_ind_size():
    """Average firm size of the 30 industry portfolios"""
    return _get_ind_file("ind30_m_size.csv")


def get_ind_nfirms():
    """Number of firms in each of the 30 industry portfolios"""
    return _get_ind_file("ind30_m_nfirms.csv")


# =============================================================================
# 2. RETURNS, DRAWDOWNS AND PERFORMANCE MEASURES
# =============================================================================

def drawdown(return_series: pd.Series, start=1000):
    """
    Takes a time series of asset returns and returns a DataFrame with columns for
    the wealth index, the previous peaks, and the percentage drawdown
    """
    wealth_index = start*(1+return_series).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdowns = (wealth_index - previous_peaks)/previous_peaks
    return pd.DataFrame({"Wealth": wealth_index,
                         "Previous Peak": previous_peaks,
                         "Drawdown": drawdowns})


def max_drawdown(r):
    """Largest peak-to-trough loss, as a negative number (-0.25 = a 25% drawdown)"""
    if isinstance(r, pd.DataFrame):
        return r.aggregate(max_drawdown)
    return drawdown(r.dropna())["Drawdown"].min()


def drawdown_duration(r):
    """
    How long the investment spent below a previous peak, in periods.
    Returns a Series with the longest underwater spell and the current one
    (0 if the investment is at a new high).
    """
    if isinstance(r, pd.DataFrame):
        return r.apply(drawdown_duration).T
    underwater = drawdown(r.dropna())["Drawdown"] < 0
    longest = current = 0
    for is_under in underwater:
        current = current + 1 if is_under else 0
        longest = max(longest, current)
    return pd.Series({"Longest": longest, "Current": current})


def annualize_rets(r, periods_per_year=None):
    """
    Annualizes a set of returns by compounding.
    periods_per_year is inferred from the index when not given.
    """
    r = r.dropna() if isinstance(r, pd.Series) else r
    ppy = _ppy(r, periods_per_year)
    compounded_growth = (1+r).prod()
    n_periods = r.shape[0] if isinstance(r, pd.Series) else r.count()
    return compounded_growth**(ppy/n_periods)-1


def annualize_vol(r, periods_per_year=None):
    """Annualizes the volatility of a set of returns: vol x sqrt(periods per year)"""
    return r.std()*(_ppy(r, periods_per_year)**0.5)


def sharpe_ratio(r, riskfree_rate=0.0, periods_per_year=None):
    """Computes the annualized Sharpe ratio of a set of returns"""
    ppy = _ppy(r, periods_per_year)
    rf_per_period = (1+riskfree_rate)**(1/ppy)-1
    excess_ret = r - rf_per_period
    ann_ex_ret = annualize_rets(excess_ret, ppy)
    ann_vol = annualize_vol(r, ppy)
    return ann_ex_ret/ann_vol


def semideviation(r):
    """Standard deviation of the negative returns only"""
    is_negative = r < 0
    return r[is_negative].std(ddof=0)


def downside_deviation(r, target=0.0, periods_per_year=None):
    """
    Annualized downside deviation: the root mean square of returns below `target` (per period),
    counting returns above the target as zero. Used by the Sortino ratio.
    """
    shortfall = np.minimum(r - target, 0)
    return np.sqrt((shortfall**2).mean()) * np.sqrt(_ppy(r, periods_per_year))


def sortino_ratio(r, riskfree_rate=0.0, periods_per_year=None):
    """
    Like the Sharpe ratio, but divides by downside deviation instead of volatility,
    so upside surprises are not penalized.
    """
    ppy = _ppy(r, periods_per_year)
    rf_per_period = (1+riskfree_rate)**(1/ppy)-1
    ann_ex_ret = annualize_rets(r - rf_per_period, ppy)
    return ann_ex_ret / downside_deviation(r, target=rf_per_period, periods_per_year=ppy)


def calmar_ratio(r, periods_per_year=None):
    """Annualized return divided by the size of the maximum drawdown"""
    return annualize_rets(r, periods_per_year) / -max_drawdown(r)


def summary_stats(r, riskfree_rate=0.03, periods_per_year=None, level=5):
    """
    One table with the main statistics for each column of r:
    return, volatility, Sharpe, Sortino, Calmar, max drawdown, skewness, kurtosis, VaR, CVaR.
    """
    if isinstance(r, pd.Series):
        r = r.to_frame(r.name or "Asset")
    ppy = _ppy(r, periods_per_year)

    def stats(col):
        x = col.dropna()
        return pd.Series({
            "Annualized Return": annualize_rets(x, ppy),
            "Annualized Vol": annualize_vol(x, ppy),
            "Sharpe Ratio": sharpe_ratio(x, riskfree_rate, ppy),
            "Sortino Ratio": sortino_ratio(x, riskfree_rate, ppy),
            "Calmar Ratio": calmar_ratio(x, ppy),
            "Max Drawdown": max_drawdown(x),
            "Longest Drawdown (periods)": drawdown_duration(x)["Longest"],
            "Skewness": skewness(x),
            "Kurtosis": kurtosis(x),
            f"Historic VaR ({level}%)": var_historic(x, level),
            f"Cornish-Fisher VaR ({level}%)": var_cornish_fisher(x, level),
            f"Historic CVaR ({level}%)": cvar_historic(x, level),
            "Best Period": x.max(),
            "Worst Period": x.min(),
            "% Positive Periods": (x > 0).mean(),
            "Periods": len(x),
        })
    return r.apply(stats).T


# =============================================================================
# 3. DISTRIBUTION STATISTICS
# =============================================================================

def skewness(r):
    """Skewness: negative means large losses are more common than large gains"""
    demeaned_r = r-r.mean()
    sigma_r = r.std(ddof=0)
    exp = (demeaned_r**3).mean()
    return exp/sigma_r**3


def kurtosis(r):
    """Kurtosis (not excess kurtosis): 3 for a normal distribution, higher means fat tails"""
    demeaned_r = r-r.mean()
    sigma_r = r.std(ddof=0)
    exp = (demeaned_r**4).mean()
    return exp/sigma_r**4


def is_normal(r, level=0.01):
    """Jarque-Bera test: True if the hypothesis of normality is accepted at the given level"""
    if isinstance(r, pd.DataFrame):
        return r.aggregate(is_normal, level=level)
    statistic, p_value = scipy.stats.jarque_bera(r.dropna())
    return p_value > level


# =============================================================================
# 4. VALUE AT RISK AND EXPECTED SHORTFALL
# =============================================================================

def var_historic(r, level=5):
    """Historic VaR: the loss exceeded in only `level` percent of past periods"""
    if isinstance(r, pd.DataFrame):
        return r.aggregate(var_historic, level=level)
    elif isinstance(r, pd.Series):
        return -np.percentile(r.dropna(), level)
    else:
        raise TypeError("Expected r to be Series or DataFrame")


def var_gaussian(r, level=5):
    """Parametric VaR assuming normally distributed returns"""
    z = norm.ppf(level/100)
    return -(r.mean() + z*r.std(ddof=0))


def var_cornish_fisher(r, level=5):
    """Modified VaR: Gaussian VaR adjusted for the skewness and kurtosis of the returns"""
    z = norm.ppf(level/100)
    s = skewness(r)
    k = kurtosis(r)
    z = (z + (z**2 - 1)*s/6 + (z**3 - 3*z)*(k-3)/24 - (2*z**3 - 5*z)*(s**2)/36)
    return -(r.mean() + z*r.std(ddof=0))


def cvar_historic(r, level=5):
    """Historic CVaR (Expected Shortfall): average loss in the worst `level` percent of periods"""
    if isinstance(r, pd.Series):
        r = r.dropna()
        is_beyond = r <= -var_historic(r, level=level)
        return -r[is_beyond].mean()
    elif isinstance(r, pd.DataFrame):
        return r.aggregate(cvar_historic, level=level)
    else:
        raise TypeError("Expected r to be a Series or DataFrame")


def cvar_gaussian(r, level=5):
    """Parametric CVaR assuming normal returns: mean loss beyond the Gaussian VaR"""
    alpha = level/100
    z = norm.ppf(alpha)
    return -(r.mean() - r.std(ddof=0) * norm.pdf(z) / alpha)


def var_monte_carlo(r, level=5, n_sims=100_000, method="bootstrap", seed=None):
    """
    Monte Carlo VaR and CVaR for one period.
    method="bootstrap" resamples past returns (keeps fat tails, needs no distribution);
    method="normal" draws from a normal distribution with the same mean and volatility.
    For a DataFrame, rows are resampled together so correlations are kept.
    Returns a DataFrame (or Series for one asset) with VaR and CVaR.
    """
    rng = np.random.default_rng(seed)
    is_series = isinstance(r, pd.Series)
    df = r.to_frame() if is_series else r
    df = df.dropna()
    if method == "bootstrap":
        idx = rng.integers(0, len(df), size=n_sims)
        sims = df.values[idx]
    elif method == "normal":
        sims = rng.multivariate_normal(df.mean().values, np.atleast_2d(df.cov().values), size=n_sims)
    else:
        raise ValueError("method must be 'bootstrap' or 'normal'")
    sims = pd.DataFrame(sims, columns=df.columns)
    out = pd.DataFrame({"VaR": var_historic(sims, level), "CVaR": cvar_historic(sims, level)})
    return out.iloc[0] if is_series else out


def scale_var(var, horizon):
    """Scales a one-period VaR to `horizon` periods with the square-root-of-time rule"""
    return var * np.sqrt(horizon)


def var_backtest(r, level=5, window=250, method="historic"):
    """
    Checks whether VaR estimates held up in practice.
    Each period, VaR is estimated from the previous `window` returns only, then compared
    with what actually happened. A breach is a loss larger than the VaR.

    method: "historic", "gaussian" or "cornish_fisher"
    Returns a DataFrame with columns Return, VaR, Breach.
    """
    funcs = {"historic": var_historic, "gaussian": var_gaussian,
             "cornish_fisher": var_cornish_fisher}
    if method not in funcs:
        raise ValueError(f"method must be one of {list(funcs)}")
    r = r.dropna()
    if len(r) <= window:
        raise ValueError("Need more returns than the estimation window")
    var_est = r.rolling(window).apply(lambda x: funcs[method](pd.Series(x), level), raw=True)
    var_est = var_est.shift(1)  # use only information available before each period
    out = pd.DataFrame({"Return": r, "VaR": var_est}).dropna()
    out["Breach"] = out["Return"] < -out["VaR"]
    return out


def kupiec_test(breaches, level=5, significance=0.05):
    """
    Kupiec proportion-of-failures test: is the number of VaR breaches consistent with the level?
    breaches: boolean Series (for example var_backtest(...)["Breach"])
    """
    n = len(breaches)
    x = int(np.sum(breaches))
    p = level/100
    phat = x/n
    if x == 0:
        lr = -2*(n*np.log(1-p))
    elif x == n:
        lr = -2*(n*np.log(p))
    else:
        lr = -2*((n-x)*np.log(1-p) + x*np.log(p) - (n-x)*np.log(1-phat) - x*np.log(phat))
    p_value = 1 - scipy.stats.chi2.cdf(lr, df=1)
    return pd.Series({"Observations": n, "Expected Breaches": n*p, "Actual Breaches": x,
                      "Breach Rate": phat, "LR Statistic": lr, "p-value": p_value,
                      "Model OK": p_value > significance})


# =============================================================================
# 5. RISK AGAINST A BENCHMARK
# =============================================================================

def _align(r, benchmark):
    joined = pd.concat([r, benchmark.rename("__bench__")], axis=1).dropna()
    return joined.drop(columns="__bench__"), joined["__bench__"]


def beta(r, benchmark):
    """Beta: how much the asset moves, on average, when the benchmark moves 1%"""
    if isinstance(r, pd.DataFrame):
        return r.apply(lambda col: beta(col, benchmark))
    a, b = _align(r, benchmark)
    a = a.iloc[:, 0]
    return a.cov(b) / b.var()


def capm(r, benchmark, riskfree_rate=0.0, periods_per_year=None):
    """
    Regresses excess returns of each asset on excess returns of the benchmark:
        R - Rf = alpha + beta x (Rm - Rf) + noise
    Returns alpha (annualized), beta, R-squared and alpha's t-statistic per asset.
    """
    if isinstance(r, pd.Series):
        r = r.to_frame(r.name or "Asset")
    ppy = _ppy(r, periods_per_year)
    rf = (1+riskfree_rate)**(1/ppy)-1
    rows = {}
    for col in r.columns:
        a, b = _align(r[col], benchmark)
        y, x = a.iloc[:, 0] - rf, b - rf
        res = scipy.stats.linregress(x, y)
        n = len(x)
        resid = y - (res.intercept + res.slope*x)
        se_alpha = np.sqrt((resid**2).sum()/(n-2)) * np.sqrt(1/n + x.mean()**2/((x-x.mean())**2).sum())
        rows[col] = {"Alpha (annual)": (1+res.intercept)**ppy - 1,
                     "Beta": res.slope,
                     "R-squared": res.rvalue**2,
                     "Alpha t-stat": res.intercept/se_alpha if se_alpha > 0 else np.nan}
    return pd.DataFrame(rows).T


def capm_expected_return(beta_value, riskfree_rate, market_return):
    """CAPM: expected return = risk-free rate + beta x (market return - risk-free rate)"""
    return riskfree_rate + beta_value*(market_return - riskfree_rate)


def tracking_error(r, benchmark, periods_per_year=None):
    """Annualized volatility of the difference between the asset and the benchmark"""
    if isinstance(r, pd.DataFrame):
        return r.apply(lambda col: tracking_error(col, benchmark, periods_per_year))
    a, b = _align(r, benchmark)
    active = a.iloc[:, 0] - b
    return active.std() * np.sqrt(_ppy(r, periods_per_year))


def information_ratio(r, benchmark, periods_per_year=None):
    """Annualized active return (asset minus benchmark) divided by tracking error"""
    if isinstance(r, pd.DataFrame):
        return r.apply(lambda col: information_ratio(col, benchmark, periods_per_year))
    ppy = _ppy(r, periods_per_year)
    a, b = _align(r, benchmark)
    active = a.iloc[:, 0] - b
    return active.mean()*ppy / (active.std()*np.sqrt(ppy))


def up_down_capture(r, benchmark):
    """
    Up capture: the asset's average return in periods when the benchmark rose, as a share of the benchmark's.
    Down capture: the same for periods when the benchmark fell. Below 1 in down markets is good.
    """
    if isinstance(r, pd.Series):
        r = r.to_frame(r.name or "Asset")
    rows = {}
    for col in r.columns:
        a, b = _align(r[col], benchmark)
        a = a.iloc[:, 0]
        up, down = b > 0, b < 0
        rows[col] = {"Up Capture": a[up].mean()/b[up].mean(),
                     "Down Capture": a[down].mean()/b[down].mean()}
    return pd.DataFrame(rows).T


# =============================================================================
# 6. RISK OVER TIME
# =============================================================================

def rolling_vol(r, window=36, periods_per_year=None):
    """Annualized volatility over a moving window of `window` periods"""
    return r.rolling(window).std() * np.sqrt(_ppy(r, periods_per_year))


def rolling_sharpe(r, window=36, riskfree_rate=0.0, periods_per_year=None):
    """Annualized Sharpe ratio over a moving window (arithmetic approximation)"""
    ppy = _ppy(r, periods_per_year)
    rf = (1+riskfree_rate)**(1/ppy)-1
    ex = r - rf
    return ex.rolling(window).mean()*ppy / (r.rolling(window).std()*np.sqrt(ppy))


def rolling_beta(r, benchmark, window=36):
    """Beta over a moving window"""
    if isinstance(r, pd.DataFrame):
        return r.apply(lambda col: rolling_beta(col, benchmark, window))
    a, b = _align(r, benchmark)
    a = a.iloc[:, 0]
    return a.rolling(window).cov(b) / b.rolling(window).var()


def rolling_corr(r, benchmark, window=36):
    """Correlation with a benchmark over a moving window"""
    if isinstance(r, pd.DataFrame):
        return r.apply(lambda col: rolling_corr(col, benchmark, window))
    a, b = _align(r, benchmark)
    return a.iloc[:, 0].rolling(window).corr(b)


def rolling_max_drawdown(r, window=36):
    """Worst drawdown within each moving window"""
    def mdd(x):
        wealth = np.cumprod(1+x)
        return (wealth/np.maximum.accumulate(wealth) - 1).min()
    return r.rolling(window).apply(mdd, raw=True)


def ewma_vol(r, lam=0.94, periods_per_year=None):
    """
    Exponentially weighted volatility (RiskMetrics). Recent returns count more:
        var_t = lam x var_(t-1) + (1 - lam) x r_(t-1)^2
    lam = 0.94 is the RiskMetrics standard for daily data; use about 0.97 for monthly.
    Returns the annualized volatility forecast for each period.
    """
    ppy = _ppy(r, periods_per_year)
    var = (r**2).ewm(alpha=1-lam, adjust=False).mean().shift(1)
    return np.sqrt(var*ppy)


def ewma_cov(r, lam=0.94):
    """Exponentially weighted covariance matrix as of the last period (per-period units)"""
    r = r.dropna()
    n = len(r)
    weights = (1-lam) * lam**np.arange(n-1, -1, -1)
    weights /= weights.sum()
    demeaned = r - r.mean()
    cov = (demeaned.values * weights[:, None]).T @ demeaned.values
    return pd.DataFrame(cov, index=r.columns, columns=r.columns)


# =============================================================================
# 7. PORTFOLIO MATH, COVARIANCE ESTIMATION, OPTIMIZATION
# =============================================================================

def portfolio_returns(weights, returns):
    """Weighted portfolio return: weights' x returns"""
    return weights.T @ returns


def portfolio_vol(weights, covmat):
    """Portfolio volatility: sqrt(weights' x covariance x weights)"""
    return (weights.T @ covmat @ weights)**0.5


def portfolio_return_series(r, weights):
    """Time series of a portfolio's returns, rebalanced to `weights` every period"""
    w = pd.Series(weights, index=r.columns) if not isinstance(weights, pd.Series) else weights
    return (r[w.index] * w).sum(axis=1, min_count=1)


def risk_contribution(weights, cov):
    """
    Share of total portfolio risk coming from each asset. The shares add up to 1.
    An asset with 10% of the money can contribute 30% of the risk.
    """
    w = np.asarray(weights)
    cov_arr = np.asarray(cov)
    total_var = w @ cov_arr @ w
    contrib = w * (cov_arr @ w) / total_var
    if isinstance(cov, pd.DataFrame):
        return pd.Series(contrib, index=cov.index)
    return contrib


def cov_sample(r, **kwargs):
    """Plain sample covariance matrix"""
    return r.cov()


def cov_constant_correlation(r, **kwargs):
    """Covariance matrix assuming every pair of assets has the same (average) correlation"""
    rhos = r.corr()
    n = rhos.shape[0]
    rho_bar = (rhos.values.sum() - n) / (n*(n-1))
    sd = r.std()
    ccor = np.full_like(rhos, rho_bar)
    np.fill_diagonal(ccor, 1.)
    return pd.DataFrame(ccor * np.outer(sd, sd), index=r.columns, columns=r.columns)


def cov_shrinkage(r, delta=0.5, **kwargs):
    """
    Shrinks the sample covariance toward the constant-correlation covariance.
    delta = 0 gives the sample covariance, delta = 1 the constant-correlation one.
    Shrinkage reduces estimation error and makes optimized weights more stable.
    """
    return delta*cov_constant_correlation(r) + (1-delta)*cov_sample(r)


def _bounds(n, min_weight=0.0, max_weight=1.0):
    if n*max_weight < 1 - 1e-9 or n*min_weight > 1 + 1e-9:
        raise ValueError(f"Weight limits [{min_weight}, {max_weight}] cannot add up to 1 with {n} assets")
    return ((min_weight, max_weight),)*n


def _as_output(weights, like):
    if isinstance(like, (pd.Series, pd.DataFrame)):
        return pd.Series(weights, index=like.index)
    return weights


def minimize_vol(target_return, er, cov, min_weight=0.0, max_weight=1.0):
    """Weights with the lowest volatility that reach `target_return`"""
    n = er.shape[0]
    init_guess = np.repeat(1/n, n)
    bounds = _bounds(n, min_weight, max_weight)
    return_is_target = {'type': 'eq', 'args': (er,),
                        'fun': lambda weights, er: target_return - portfolio_returns(weights, er)}
    weights_sum_to_1 = {'type': 'eq', 'fun': lambda weights: np.sum(weights)-1}
    results = minimize(portfolio_vol, init_guess, args=(cov,), method="SLSQP",
                       options={'disp': False},
                       constraints=(return_is_target, weights_sum_to_1),
                       bounds=bounds)
    return results.x


def optimal_weights(n_points, er, cov, min_weight=0.0, max_weight=1.0):
    """Minimum-volatility weights for evenly spaced target returns"""
    target_rs = np.linspace(er.min(), er.max(), n_points)
    return [minimize_vol(t, er, cov, min_weight, max_weight) for t in target_rs]


def msr(riskfree_rate, er, cov, min_weight=0.0, max_weight=1.0):
    """Weights of the Maximum Sharpe Ratio portfolio"""
    n = er.shape[0]
    init_guess = np.repeat(1/n, n)
    bounds = _bounds(n, min_weight, max_weight)
    weights_sum_to_1 = {'type': 'eq', 'fun': lambda weights: np.sum(weights)-1}

    def neg_sharpe_ratio(weights, riskfree_rate, er, cov):
        r = portfolio_returns(weights, er)
        vol = portfolio_vol(weights, cov)
        return -(r-riskfree_rate)/vol

    results = minimize(neg_sharpe_ratio, init_guess, args=(riskfree_rate, er, cov), method="SLSQP",
                       options={'disp': False}, constraints=(weights_sum_to_1,), bounds=bounds)
    return results.x


def gmv(cov, min_weight=0.0, max_weight=1.0):
    """Weights of the Global Minimum Variance portfolio"""
    n = cov.shape[0]
    return msr(0, np.repeat(1, n), cov, min_weight, max_weight)


def equal_weights(n_or_assets):
    """Equal weights 1/N. Pass a number of assets or a list/Index of names."""
    if isinstance(n_or_assets, int):
        return np.repeat(1/n_or_assets, n_or_assets)
    return pd.Series(1/len(n_or_assets), index=n_or_assets)


def inverse_vol_weights(cov):
    """Weights proportional to 1 / volatility: calmer assets get more money"""
    inv = 1/np.sqrt(np.diag(np.asarray(cov)))
    return _as_output(inv/inv.sum(), cov)


def risk_parity(cov, target_risk=None, min_weight=0.0, max_weight=1.0):
    """
    Equal Risk Contribution weights: each asset contributes the same share of total risk
    (or the shares in `target_risk`, which must add up to 1).
    """
    cov_arr = np.asarray(cov)
    n = cov_arr.shape[0]
    target = np.repeat(1/n, n) if target_risk is None else np.asarray(target_risk)

    def deviation(w):
        return ((risk_contribution(w, cov_arr) - target)**2).sum()

    results = minimize(deviation, np.asarray(inverse_vol_weights(cov_arr)), method="SLSQP",
                       options={'disp': False, 'ftol': 1e-12, 'maxiter': 500},
                       constraints=({'type': 'eq', 'fun': lambda w: np.sum(w)-1},),
                       bounds=_bounds(n, max(min_weight, 1e-6), max_weight))
    return _as_output(results.x, cov)


def diversification_ratio(weights, cov):
    """Weighted average asset volatility divided by portfolio volatility (1 = no diversification)"""
    w = np.asarray(weights)
    vols = np.sqrt(np.diag(np.asarray(cov)))
    return (w @ vols) / portfolio_vol(w, np.asarray(cov))


def max_diversification(cov, min_weight=0.0, max_weight=1.0):
    """Weights that maximize the diversification ratio"""
    cov_arr = np.asarray(cov)
    n = cov_arr.shape[0]
    results = minimize(lambda w: -diversification_ratio(w, cov_arr), np.repeat(1/n, n),
                       method="SLSQP", options={'disp': False},
                       constraints=({'type': 'eq', 'fun': lambda w: np.sum(w)-1},),
                       bounds=_bounds(n, min_weight, max_weight))
    return _as_output(results.x, cov)


def plot_ef2(n_points, er, cov, style=".-"):
    """Plots the 2-asset efficient frontier"""
    if er.shape[0] != 2 or cov.shape[0] != 2:
        raise ValueError("plot_ef2 can only plot 2-asset frontiers")
    weights = [np.array([w, 1-w]) for w in np.linspace(0, 1, n_points)]
    rets = [portfolio_returns(w, er) for w in weights]
    vols = [portfolio_vol(w, cov) for w in weights]
    ef = pd.DataFrame({"Returns": rets, "Volatility": vols})
    return ef.plot.line(x="Volatility", y="Returns", style=style)


def efficient_frontier(n_points, er, cov, min_weight=0.0, max_weight=1.0):
    """The efficient frontier as data: a DataFrame of Returns, Volatility and the weights of each point"""
    weights = optimal_weights(n_points, er, cov, min_weight, max_weight)
    ef = pd.DataFrame({"Returns": [portfolio_returns(w, er) for w in weights],
                       "Volatility": [portfolio_vol(w, cov) for w in weights]})
    names = er.index if isinstance(er, pd.Series) else range(len(er))
    return pd.concat([ef, pd.DataFrame(weights, columns=names)], axis=1)


def plot_ef(n_points, er, cov, show_cml=False, style="--", riskfree_rate=0, show_ew=False,
            show_gmv=False):
    """Plots the N-asset efficient frontier"""
    ef = efficient_frontier(n_points, er, cov)
    ax = ef.plot.line(x="Volatility", y="Returns", style=style)
    if show_gmv:
        w_gmv = gmv(cov)
        ax.plot([portfolio_vol(w_gmv, cov)], [portfolio_returns(w_gmv, er)],
                color="midnightblue", marker="o", markersize=12)
    if show_ew:
        w_ew = np.repeat(1/er.shape[0], er.shape[0])
        ax.plot([portfolio_vol(w_ew, cov)], [portfolio_returns(w_ew, er)],
                color="goldenrod", marker="o", markersize=12)
    if show_cml:
        ax.set_xlim(left=0)
        w_msr = msr(riskfree_rate, er, cov)
        ax.plot([0, portfolio_vol(w_msr, cov)], [riskfree_rate, portfolio_returns(w_msr, er)],
                color="green", marker="o", linestyle="dashed", linewidth=2)
    return ax


# --- weighting schemes for backtests: each takes a window of returns, returns weights ---

def weight_ew(r, **kwargs):
    """Equal weights"""
    return equal_weights(r.columns)


def weight_inverse_vol(r, **kwargs):
    """Inverse-volatility weights"""
    return inverse_vol_weights(r.cov())


def weight_gmv(r, cov_estimator=cov_sample, **kwargs):
    """Global minimum variance weights, from the chosen covariance estimator"""
    return pd.Series(gmv(cov_estimator(r, **kwargs), kwargs.get("min_weight", 0.0),
                         kwargs.get("max_weight", 1.0)), index=r.columns)


def weight_risk_parity(r, cov_estimator=cov_sample, **kwargs):
    """Equal risk contribution weights"""
    return risk_parity(cov_estimator(r, **kwargs))


def weight_max_div(r, cov_estimator=cov_sample, **kwargs):
    """Maximum diversification weights"""
    return max_diversification(cov_estimator(r, **kwargs))


# =============================================================================
# 8. PORTFOLIO BACKTESTING WITH REBALANCING AND COSTS
# =============================================================================

def backtest_portfolio(r, weights, rebalance="M", cost_bps=0.0, estimation_window=None,
                       start_value=1.0, **weight_kwargs):
    """
    Backtests a portfolio with realistic rebalancing.

    r: returns, one column per asset (daily or monthly)
    weights: fixed target weights (list, array or Series), or a weighting function such as
             weight_gmv that receives the past `estimation_window` returns and returns weights
    rebalance: "D", "W", "M", "Q", "Y" for calendar rebalancing, or None to buy and hold
    cost_bps: trading cost in basis points of the amount traded (10 = 0.10%)

    Between rebalances, weights drift with the market. At each rebalance the portfolio trades
    back to target; turnover = sum of absolute weight changes, cost = turnover x cost_bps / 10,000.

    Returns a DataFrame with Return, Wealth, Turnover, Cost, plus a `weights` attribute
    holding the weights held each period (result.attrs["weights"]).
    """
    r = r.dropna(how="all").fillna(0.0)
    assets = r.columns
    if callable(weights):
        if estimation_window is None:
            raise ValueError("A weighting function needs an estimation_window")
        weight_fn = weights
    else:
        fixed = pd.Series(weights, index=assets) if not isinstance(weights, pd.Series) else weights[assets]
        if not np.isclose(fixed.sum(), 1):
            raise ValueError("Fixed weights must add up to 1")
        weight_fn = None

    start = estimation_window or 0
    idx = r.index
    if rebalance is None:
        rebal_flags = np.zeros(len(idx), dtype=bool)
    else:
        rebalance = rebalance.upper()
        if isinstance(idx, pd.PeriodIndex):
            keys = idx.asfreq(rebalance if rebalance != "Y" else "Y")
        else:
            keys = idx.to_period(rebalance if rebalance != "Y" else "Y")
        keys = pd.Series(keys, index=idx)
        rebal_flags = (keys != keys.shift(1)).values
    rebal_flags[start] = True  # always invest at the start

    current = np.zeros(len(assets))
    rows, held = [], []
    for i in range(start, len(idx)):
        turnover = 0.0
        if rebal_flags[i]:
            if weight_fn is not None:
                target = np.asarray(weight_fn(r.iloc[i-estimation_window:i], **weight_kwargs), dtype=float)
            else:
                target = fixed.values.astype(float)
            turnover = np.abs(target - current).sum() if i > start else np.abs(target).sum()
            current = target
        cost = turnover * cost_bps / 10_000
        period_ret = r.iloc[i].values
        port_ret = current @ period_ret - cost
        held.append(current.copy())
        rows.append((port_ret, turnover, cost))
        grown = current * (1 + period_ret)
        current = grown / grown.sum() if grown.sum() != 0 else current

    out = pd.DataFrame(rows, index=idx[start:], columns=["Return", "Turnover", "Cost"])
    out.insert(1, "Wealth", start_value*(1+out["Return"]).cumprod())
    out.attrs["weights"] = pd.DataFrame(held, index=idx[start:], columns=assets)
    return out


def compare_strategies(r, strategies, rebalance="M", cost_bps=0.0, estimation_window=None,
                       riskfree_rate=0.03, periods_per_year=None):
    """
    Backtests several strategies and returns (summary table, DataFrame of returns).
    strategies: dict of name -> fixed weights or weighting function
    """
    rets = {}
    turnover = {}
    for name, w in strategies.items():
        bt = backtest_portfolio(r, w, rebalance=rebalance, cost_bps=cost_bps,
                                estimation_window=estimation_window if callable(w) else None)
        rets[name] = bt["Return"]
        turnover[name] = bt["Turnover"].iloc[1:]  # excludes the initial purchase
    rets = pd.DataFrame(rets).dropna()
    stats = summary_stats(rets, riskfree_rate, _ppy(r, periods_per_year))
    years = len(rets) / _ppy(r, periods_per_year)
    stats["Annual Turnover"] = pd.Series({k: v.reindex(rets.index).sum() for k, v in turnover.items()}) / years
    return stats, rets


# =============================================================================
# 9. CPPI PORTFOLIO INSURANCE
# =============================================================================

def run_cppi(risky_r, safe_r=None, m=3, start=1000, floor=0.8, riskfree_rate=0.03, drawdown=None,
             periods_per_year=12):
    """
    Run a backtest of the CPPI strategy, given a set of returns for the risky asset
    Returns a dictionary containing: Asset Value History, Risk Budget History, Risky Weight History
    """
    dates = risky_r.index
    n_steps = len(dates)
    account_value = start
    floor_value = start*floor
    peak = account_value
    if isinstance(risky_r, pd.Series):
        risky_r = pd.DataFrame(risky_r, columns=["R"])

    if safe_r is None:
        safe_r = pd.DataFrame().reindex_like(risky_r)
        safe_r.values[:] = (1+riskfree_rate)**(1/periods_per_year) - 1
    account_history = pd.DataFrame().reindex_like(risky_r)
    risky_w_history = pd.DataFrame().reindex_like(risky_r)
    cushion_history = pd.DataFrame().reindex_like(risky_r)
    floorval_history = pd.DataFrame().reindex_like(risky_r)
    peak_history = pd.DataFrame().reindex_like(risky_r)

    for step in range(n_steps):
        if drawdown is not None:
            peak = np.maximum(peak, account_value)
            floor_value = peak*(1-drawdown)
        cushion = (account_value - floor_value)/account_value
        risky_w = m*cushion
        risky_w = np.minimum(risky_w, 1)
        risky_w = np.maximum(risky_w, 0)
        safe_w = 1-risky_w
        risky_alloc = account_value*risky_w
        safe_alloc = account_value*safe_w
        account_value = risky_alloc*(1+risky_r.iloc[step]) + safe_alloc*(1+safe_r.iloc[step])
        cushion_history.iloc[step] = cushion
        risky_w_history.iloc[step] = risky_w
        account_history.iloc[step] = account_value
        floorval_history.iloc[step] = floor_value
        peak_history.iloc[step] = peak
    risky_wealth = start*(1+risky_r).cumprod()
    return {
        "Wealth": account_history,
        "Risky Wealth": risky_wealth,
        "Risk Budget": cushion_history,
        "Risky Allocation": risky_w_history,
        "m": m,
        "start": start,
        "floor": floor,
        "risky_r": risky_r,
        "safe_r": safe_r,
        "drawdown": drawdown,
        "peak": peak_history,
        "floor_value": floorval_history,
    }


# =============================================================================
# 10. SIMULATION
# =============================================================================

def gbm(n_years=10, n_scenarios=1000, mu=0.07, sigma=0.15, steps_per_year=12, s_0=100.0,
        prices=True, seed=None):
    """
    Evolution of Geometric Brownian Motion trajectories, such as for Stock Prices through Monte Carlo
    :param n_years:  The number of years to generate data for
    :param n_scenarios: The number of scenarios/trajectories
    :param mu: Annualized Drift, e.g. Market Return
    :param sigma: Annualized Volatility
    :param steps_per_year: granularity of the simulation
    :param s_0: initial value
    :param seed: set for repeatable results
    :return: a DataFrame of prices (or a numpy array of returns if prices=False),
             n_years*steps_per_year+1 rows and n_scenarios columns
    """
    rng = np.random.default_rng(seed)
    dt = 1/steps_per_year
    n_steps = int(n_years*steps_per_year) + 1
    rets_plus_1 = rng.normal(loc=(1+mu)**dt, scale=(sigma*np.sqrt(dt)), size=(n_steps, n_scenarios))
    rets_plus_1[0] = 1
    return s_0*pd.DataFrame(rets_plus_1).cumprod() if prices else rets_plus_1-1


def bootstrap_paths(r, n_periods, n_paths=2000, block=21, start=1.0, seed=None):
    """
    Simulates future wealth paths by replaying blocks of past returns in random order
    (a "block bootstrap"). Using blocks of about a month keeps real-world features that a bell
    curve misses: fat tails and calm/stormy spells. No distribution is assumed.

    r: past returns (Series); n_periods: how many periods to simulate.
    Returns an array of shape (n_paths, n_periods + 1) of wealth, starting at `start`.
    """
    x = np.log1p(np.asarray(pd.Series(r).dropna(), dtype=float))
    if len(x) <= block:
        raise ValueError("Need more past returns than the block length to simulate")
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n_periods / block))
    starts = rng.integers(0, len(x) - block + 1, size=(n_paths, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)).reshape(n_paths, -1)[:, :n_periods]
    log_paths = np.cumsum(x[idx], axis=1)
    return start * np.exp(np.hstack([np.zeros((n_paths, 1)), log_paths]))


def inst_to_ann(r):
    """Converts a continuously compounded (instantaneous) rate to an annual rate"""
    return np.expm1(r)


def ann_to_inst(r):
    """Converts an annual rate to a continuously compounded (instantaneous) rate"""
    return np.log1p(r)


def cir(n_years=10, n_scenarios=1, a=0.05, b=0.03, sigma=0.05, steps_per_year=12, r_0=None, seed=None):
    """
    Cox-Ingersoll-Ross model of interest rates: rates are pulled toward a long-run level b
    at speed a, with random shocks that shrink as rates approach zero (so rates stay positive).
    Returns (annualized rates, prices of a zero-coupon bond maturing at n_years) as DataFrames.
    """
    rng = np.random.default_rng(seed)
    if r_0 is None:
        r_0 = b
    r_0 = ann_to_inst(r_0)
    dt = 1/steps_per_year
    num_steps = int(n_years*steps_per_year) + 1
    shock = rng.normal(0, scale=np.sqrt(dt), size=(num_steps, n_scenarios))
    rates = np.empty_like(shock)
    rates[0] = r_0

    h = np.sqrt(a**2 + 2*sigma**2)
    prices = np.empty_like(shock)

    def price(ttm, r):
        _A = ((2*h*np.exp((h+a)*ttm/2))/(2*h+(h+a)*(np.exp(h*ttm)-1)))**(2*a*b/sigma**2)
        _B = (2*(np.exp(h*ttm)-1))/(2*h + (h+a)*(np.exp(h*ttm)-1))
        return _A*np.exp(-_B*r)

    prices[0] = price(n_years, r_0)
    for step in range(1, num_steps):
        r_t = rates[step-1]
        d_r_t = a*(b-r_t)*dt + sigma*np.sqrt(np.maximum(r_t, 0))*shock[step]
        rates[step] = np.abs(r_t + d_r_t)
        prices[step] = price(n_years-step*dt, rates[step])

    return pd.DataFrame(inst_to_ann(rates)), pd.DataFrame(prices)


# =============================================================================
# 11. BONDS AND LIABILITY-DRIVEN INVESTING
# =============================================================================

def discount(t, r):
    """Price today of 1 paid at time t (years), at annual interest rate r"""
    return (1+r)**(-np.asarray(t, dtype=float))


def pv(flows, r):
    """Present value of cash flows: a Series indexed by time in years"""
    return (discount(flows.index, r) * flows.values).sum()


def funding_ratio(assets, liabilities, r):
    """Assets divided by the present value of liabilities. Above 1 means fully funded."""
    return assets / pv(liabilities, r)


def bond_cash_flows(maturity, principal=100, coupon_rate=0.03, coupons_per_year=12):
    """Cash flows of a regular coupon bond, as a Series indexed by time in years"""
    n_coupons = round(maturity*coupons_per_year)
    coupon_amt = principal*coupon_rate/coupons_per_year
    times = np.arange(1, n_coupons+1) / coupons_per_year
    flows = pd.Series(coupon_amt, index=times)
    flows.iloc[-1] += principal
    return flows


def bond_price(maturity, principal=100, coupon_rate=0.03, coupons_per_year=12, discount_rate=0.03):
    """Price of a coupon bond: present value of its cash flows at a yearly discount rate"""
    flows = bond_cash_flows(maturity, principal, coupon_rate, coupons_per_year)
    per_period = discount_rate/coupons_per_year
    periods = flows.index.values*coupons_per_year
    return (flows.values / (1+per_period)**periods).sum()


def macaulay_duration(flows, discount_rate):
    """
    Macaulay duration: the average time (years) until the cash flows arrive, weighted by
    their present values. Also roughly how much the price falls (%) when rates rise 1%.
    """
    w = discount(flows.index, discount_rate) * flows.values
    w = w / w.sum()
    return np.average(flows.index, weights=w)


def match_durations(cf_target, cf_short, cf_long, discount_rate):
    """
    Weight in the short bond so that a mix of a short and a long bond has the same
    duration as the target (for example a pension liability).
    """
    d_t = macaulay_duration(cf_target, discount_rate)
    d_s = macaulay_duration(cf_short, discount_rate)
    d_l = macaulay_duration(cf_long, discount_rate)
    return (d_l - d_t)/(d_l - d_s)


# =============================================================================
# 12. STRESS TESTING
# =============================================================================

# Peak-to-trough windows of major market stress (dates of the US stock market high and low)
CRISES = {
    "Dot-com crash": ("2000-03-24", "2002-10-09"),
    "Global financial crisis": ("2007-10-09", "2009-03-09"),
    "Euro debt crisis": ("2011-04-29", "2011-10-03"),
    "China / oil selloff": ("2015-07-20", "2016-02-11"),
    "Q4 2018 selloff": ("2018-09-20", "2018-12-24"),
    "COVID crash": ("2020-02-19", "2020-03-23"),
    "2022 rate shock": ("2022-01-03", "2022-10-12"),
}


def stress_test(r, weights, scenarios=None):
    """
    Replays historical crises on a portfolio.

    r: daily asset returns (for example prices_to_returns(get_prices([...])))
    weights: portfolio weights (list, array or Series matching r's columns)
    scenarios: dict of name -> (start date, end date); defaults to CRISES

    For each scenario: the portfolio's total return and maximum drawdown over the window,
    and each asset's total return. Scenarios that the data does not fully cover are marked
    "Covered" = False (some assets may not have existed yet).
    """
    scenarios = CRISES if scenarios is None else scenarios
    w = pd.Series(weights, index=r.columns) if not isinstance(weights, pd.Series) else weights[r.columns]
    rows = {}
    for name, (start, end) in scenarios.items():
        window = r.loc[start:end]
        covered = (len(window) > 0 and window.notna().all().all()
                   and r.index[0] <= pd.Timestamp(start) + pd.Timedelta(days=7))
        if len(window) == 0:
            rows[name] = {"Start": start, "End": end, "Covered": False}
            continue
        asset_tot = (1+window.fillna(0)).prod() - 1
        port = (window.fillna(0) * w).sum(axis=1)
        rows[name] = {"Start": start, "End": end, "Covered": bool(covered),
                      "Portfolio Return": (1+port).prod() - 1,
                      "Portfolio Max Drawdown": max_drawdown(port),
                      **{f"{a} Return": asset_tot[a] for a in r.columns}}
    return pd.DataFrame(rows).T


def shock_test(weights, shocks):
    """
    Hypothetical scenario: the portfolio's return if each asset moves by the given shock.
    weights, shocks: Series indexed by asset (shocks as decimals, -0.30 = a 30% fall)
    """
    w = pd.Series(weights)
    s = pd.Series(shocks).reindex(w.index).fillna(0)
    return pd.Series({"Portfolio Return": (w*s).sum(), **{f"{a} contribution": w[a]*s[a] for a in w.index}})


# =============================================================================
# 13. ONE-CALL REPORTS FOR DASHBOARDS
# =============================================================================

def portfolio_report(tickers, weights=None, benchmark="SPY", start=None, end=None, freq="D",
                     riskfree_rate=0.03, level=5, rolling_window=None, use_cache=True):
    """
    Everything a risk dashboard needs, from Yahoo Finance, in one call.

    tickers: list of holdings; weights: their weights (equal weights if None)
    benchmark: ticker to compare against (None to skip)
    freq: "D" daily or "M" monthly returns for the statistics

    Returns a dict of pandas objects:
      prices, returns, portfolio (portfolio returns), wealth, drawdown,
      summary (summary_stats of assets, portfolio and benchmark), capm, correlation,
      risk_contribution, rolling_vol, ewma_vol, var_backtest, kupiec, stress
    """
    tickers = [tickers] if isinstance(tickers, str) else list(tickers)
    tickers = [t.strip().upper() for t in tickers if t and t.strip()]
    benchmark = benchmark.strip().upper() if benchmark else None
    if weights is not None:
        if len(weights) != len(tickers):
            raise ValueError("Give one weight per ticker")
        weight_map = pd.Series(weights, index=tickers, dtype=float)
        if (weight_map < 0).any() or weight_map.sum() <= 0:
            raise ValueError("Weights must be non-negative and not all zero")
    all_tickers = tickers + ([benchmark] if benchmark and benchmark not in tickers else [])
    prices = get_prices(all_tickers, start=start, end=end, use_cache=use_cache)
    dropped = [t for t in tickers if t not in prices.columns]
    tickers = [t for t in tickers if t in prices.columns]
    if not tickers:
        raise ValueError("None of the tickers returned data")
    raw_prices = prices
    prices = prices.dropna(subset=tickers)
    if len(prices) < 30:
        first = raw_prices[tickers].apply(lambda c: c.first_valid_index()).dropna()
        late = first[first == first.max()].index.tolist() if len(first) else []
        hint = (f" {', '.join(late)} only has prices from {first.max():%d %b %Y}. "
                "Remove it, or move the start date later." if late else " Choose an earlier start date.")
        raise ValueError("Fewer than 30 days where every holding has a price." + hint)
    w = equal_weights(tickers) if weights is None else weight_map[tickers]
    w = w / w.sum()

    daily = prices_to_returns(prices, "D")
    rets = daily if freq.upper() == "D" else prices_to_returns(prices, freq)
    ppy = PERIODS_PER_YEAR[freq.upper()]
    window = rolling_window or (63 if ppy == 252 else 12)

    port = portfolio_return_series(rets[tickers], w).rename("Portfolio")
    table = pd.concat([rets[tickers], port], axis=1)
    bench = rets[benchmark] if benchmark and benchmark in rets.columns else None
    if bench is not None:
        table = pd.concat([table, bench.rename(f"Benchmark ({benchmark})")], axis=1)

    report = {
        "dropped_tickers": dropped,
        "period": (prices.index[0], prices.index[-1]),
        "weights": w,
        "prices": prices,
        "returns": rets,
        "portfolio": port,
        "wealth": (1+port).cumprod(),
        "drawdown": drawdown(port, start=1)["Drawdown"],
        "summary": summary_stats(table, riskfree_rate, ppy, level),
        "correlation": rets[tickers].corr(),
        "risk_contribution": risk_contribution(w.values, rets[tickers].cov()).rename("Risk Share"),
        "rolling_vol": rolling_vol(port, window, ppy),
        "ewma_vol": ewma_vol(port, 0.94 if ppy == 252 else 0.97, ppy),
        "stress": stress_test(daily[tickers], w),
    }
    report["risk_contribution"].index = tickers
    if bench is not None:
        report["capm"] = capm(table.drop(columns=f"Benchmark ({benchmark})"), bench, riskfree_rate, ppy)
        report["tracking_error"] = tracking_error(port, bench, ppy)
        report["information_ratio"] = information_ratio(port, bench, ppy)
    bt_window = 250 if ppy == 252 else 60
    if len(port.dropna()) > bt_window + 20:
        report["var_backtest"] = var_backtest(port, level, bt_window)
        report["kupiec"] = kupiec_test(report["var_backtest"]["Breach"], level)
    return report
