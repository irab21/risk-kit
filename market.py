"""
Today's market conditions: index levels, returns over several periods, trend, distance from the
52-week high and volatility, plus a one-word "mood" built from simple, visible rules.

Markets: India (default for Indian stocks and mutual funds), US and Global.
Data: Yahoo Finance daily closes (indices are price indices; ETFs proxy fund categories).
"""
import numpy as np
import pandas as pd
import yfinance as yf

MARKETS = {
    "India": {
        "main": "^NSEI", "vix": "^INDIAVIX", "cur": "₹",
        "headline": {"^NSEI": "Nifty 50", "^BSESN": "Sensex", "^NSEBANK": "Bank Nifty",
                     "^NSEMDCP50": "Nifty Midcap 50", "HDFCSML250.NS": "Smallcap 250"},
        "chart": ["^NSEI", "^NSEMDCP50", "HDFCSML250.NS", "^NSEBANK"],
        "groups": {
            "Sectors": {"^CNXIT": "IT", "^NSEBANK": "Banks", "^CNXPSUBANK": "PSU banks", "^CNXPHARMA": "Pharma",
                        "^CNXAUTO": "Auto", "^CNXFMCG": "FMCG", "^CNXMETAL": "Metals", "^CNXENERGY": "Energy",
                        "^CNXREALTY": "Realty", "^CNXINFRA": "Infrastructure"},
            "Mutual fund categories": {"NIFTYBEES.NS": "Large cap", "JUNIORBEES.NS": "Next 50",
                                       "^NSEMDCP50": "Mid cap", "HDFCSML250.NS": "Small cap",
                                       "BANKBEES.NS": "Banking", "GOLDBEES.NS": "Gold", "SILVERBEES.NS": "Silver",
                                       "LTGILTBEES.NS": "Long-term gilt", "GILT5YBEES.NS": "Medium-term gilt",
                                       "EBBETF0433.NS": "Corporate bond", "LIQUIDCASE.NS": "Liquid",
                                       "MON100.NS": "International (US tech)"},
        },
        "macro": {"^INDIAVIX": "India VIX", "INR=X": "USD / INR", "GOLDBEES.NS": "Gold (₹ ETF)", "BZ=F": "Brent crude ($)"},
    },
    "US": {
        "main": "^GSPC", "vix": "^VIX", "cur": "$",
        "headline": {"^GSPC": "S&P 500", "^IXIC": "Nasdaq", "^DJI": "Dow Jones", "^RUT": "Russell 2000"},
        "chart": ["^GSPC", "^IXIC", "^DJI", "^RUT"],
        "groups": {
            "Sectors": {"XLK": "Technology", "XLC": "Communication", "XLY": "Consumer discretionary",
                        "XLP": "Consumer staples", "XLF": "Financials", "XLV": "Health care", "XLI": "Industrials",
                        "XLE": "Energy", "XLB": "Materials", "XLU": "Utilities", "XLRE": "Real estate"},
        },
        "macro": {"^VIX": "VIX", "^TNX": "US 10-year yield", "DX-Y.NYB": "US dollar index", "GC=F": "Gold ($)",
                  "BZ=F": "Brent crude ($)"},
    },
    "Global": {
        "main": "^GSPC", "vix": "^VIX", "cur": "$",
        "headline": {"^GSPC": "S&P 500 (US)", "^STOXX50E": "Euro Stoxx 50", "^N225": "Nikkei 225 (Japan)",
                     "^NSEI": "Nifty 50 (India)", "EEM": "Emerging markets"},
        "chart": ["^GSPC", "^STOXX50E", "^N225", "^NSEI", "EEM"],
        "groups": {
            "Regions": {"^GSPC": "US", "^IXIC": "US tech", "^STOXX50E": "Europe", "^FTSE": "UK", "^N225": "Japan",
                        "^HSI": "Hong Kong", "EEM": "Emerging markets", "^NSEI": "India"},
        },
        "macro": {"^VIX": "VIX", "^TNX": "US 10-year yield", "DX-Y.NYB": "US dollar index", "GC=F": "Gold ($)",
                  "BZ=F": "Brent crude ($)"},
    },
}
GLOBAL_STRIP = {"^GSPC": "S&P 500", "^STOXX50E": "Euro Stoxx 50", "^N225": "Nikkei 225", "^VIX": "VIX",
                "^TNX": "US 10Y yield", "GC=F": "Gold", "BZ=F": "Brent"}
LEVEL_ONLY = {"^INDIAVIX", "^VIX", "^TNX", "INR=X", "DX-Y.NYB"}  # shown as levels / point changes
PERIODS = {"1W": 5, "1M": 21, "3M": 63, "YTD": None, "1Y": 252}


def detect_market(tickers):
    """India for Indian stocks / mutual funds, US for plain US tickers, Global otherwise."""
    tickers = [str(t).upper() for t in tickers]
    if not tickers:
        return "India"
    if any(t.startswith("MF:") or t.endswith((".NS", ".BO")) for t in tickers):
        return "India"
    if all("." not in t and not t.startswith("^") for t in tickers):
        return "US"
    return "Global"


def all_tickers(name):
    cfg = MARKETS[name]
    out = set(cfg["headline"]) | set(cfg["chart"]) | set(cfg["macro"]) | {cfg["main"], cfg["vix"]} | set(GLOBAL_STRIP)
    for g in cfg["groups"].values():
        out |= set(g)
    return sorted(out)


def download(tickers, period="2y"):
    data = yf.download(list(tickers), period=period, auto_adjust=True, progress=False, threads=True)
    close = data["Close"] if isinstance(data.columns, pd.MultiIndex) else data[["Close"]].rename(
        columns={"Close": tickers[0]})
    return close.dropna(how="all")


def stats(s):
    """Level, returns over standard periods, distance from 52-week high, trend and volatility for one series."""
    s = s.dropna()
    if len(s) < 30:
        return None
    last = s.iloc[-1]
    out = {"last": last, "date": s.index[-1]}
    for k, n in PERIODS.items():
        if n is None:
            base = s[s.index < pd.Timestamp(s.index[-1].year, 1, 1)]
            out[k] = last / base.iloc[-1] - 1 if len(base) else np.nan
        else:
            out[k] = last / s.iloc[-n - 1] - 1 if len(s) > n else np.nan
    out["1D"] = last / s.iloc[-2] - 1
    yr = s.iloc[-252:]
    out["high"] = yr.max()
    out["from_high"] = last / yr.max() - 1
    out["ma50"], out["ma200"] = s.iloc[-50:].mean(), s.iloc[-200:].mean() if len(s) >= 200 else np.nan
    r = np.log(s).diff().dropna()
    out["vol_now"] = r.iloc[-20:].std() * np.sqrt(252)
    out["vol_year"] = r.iloc[-252:].std() * np.sqrt(252)
    return out


def regime(main, vix_series=None, name="the market"):
    """
    Plain-English reading of market conditions from the main index (and its volatility index).
    Returns dict: mood, tone (good/ok/bad), trend, drawdown, volatility labels, score and bullet sentences.
    """
    above200 = bool(main["last"] > main["ma200"]) if pd.notna(main["ma200"]) else None
    above50 = bool(main["last"] > main["ma50"])
    if above200 and above50:
        trend, t_score = "Uptrend", 1
    elif above200 is False and not above50:
        trend, t_score = "Downtrend", -1
    else:
        trend, t_score = "Sideways", 0

    dd = main["from_high"]
    if dd > -0.03:
        dd_label, d_score = "Near its high", 1
    elif dd > -0.10:
        dd_label, d_score = "Pullback", 0
    elif dd > -0.20:
        dd_label, d_score = "Correction", -1
    else:
        dd_label, d_score = "Bear market", -2

    vix_now = vix_ratio = None
    if vix_series is not None and len(vix_series.dropna()) > 60:
        v = vix_series.dropna()
        vix_now, vix_ratio = v.iloc[-1], v.iloc[-1] / v.iloc[-252:].median()
    ratio = vix_ratio if vix_ratio is not None else main["vol_now"] / main["vol_year"]
    if ratio > 1.25:
        vol_label, v_score = "Elevated", -1
    elif ratio < 0.85:
        vol_label, v_score = "Calm", 1
    else:
        vol_label, v_score = "Normal", 0

    score = t_score + d_score + v_score
    mood, tone = (("Strong", "good") if score >= 2 else ("Steady", "good") if score == 1 else
                  ("Mixed", "ok") if score >= -1 else ("Cautious", "bad") if score >= -2 else ("Stressed", "bad"))

    notes = [f"{name} is {abs(dd)*100:.1f}% below its 52-week high"
             + (f" and up {main['1Y']*100:.1f}% over the past year." if main["1Y"] >= 0 else
                f" and down {abs(main['1Y'])*100:.1f}% over the past year."),
             {"Uptrend": "It trades above its 50- and 200-day averages: the trend is up.",
              "Downtrend": "It trades below its 50- and 200-day averages: the trend is down.",
              "Sideways": "It sits between its 50- and 200-day averages: no clear trend."}[trend],
             ({"Elevated": "Fear is above normal", "Calm": "Fear is below normal", "Normal": "Fear is around normal"}[vol_label]
              + (f" (volatility index {vix_now:.1f}, {ratio:.1f}× its one-year median)." if vix_now is not None
                 else f" (recent swings {main['vol_now']*100:.0f}% a year vs {main['vol_year']*100:.0f}% usual)."))]
    return {"mood": mood, "tone": tone, "trend": trend, "drawdown": dd_label, "volatility": vol_label,
            "score": score, "from_high": dd, "y1": main["1Y"], "vix": vix_now, "vix_ratio": ratio,
            "notes": notes, "name": name}


def implications(reg, beta=None, value=None, money=lambda x: f"{x:,.0f}"):
    """Two or three crisp 'what it means for you' lines."""
    out = []
    if reg["drawdown"] in ("Correction", "Bear market"):
        out.append("Prices are below their highs: a good time to keep SIPs running, not to stop them.")
    elif reg["drawdown"] == "Near its high" and reg["y1"] > 0.15:
        out.append("After a strong year, markets are near highs: rebalance back to your target mix rather than add more risk.")
    else:
        out.append("No extreme in prices either way: stick to your plan and keep investing regularly.")
    if reg["volatility"] == "Elevated":
        out.append("Volatility is high: invest new lump sums in 3–6 monthly instalments instead of all at once.")
    if beta is not None and value:
        out.append(f"If {reg['name']} drops 10%, your portfolio would typically fall about {abs(beta)*10:.0f}% "
                   f"({money(value * abs(beta) * 0.10)}).")
    return out
