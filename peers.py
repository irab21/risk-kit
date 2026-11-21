"""
Mutual fund peer comparison, built only from public AMFI data.

- amfi_schemes(): AMFI's daily NAV file (every open scheme with its SEBI category, plan and option).
- peer_report(code): ranks a fund against every fund in the same category, plan and option on
  1/3/5-year returns, 3-year volatility, Sharpe ratio and worst fall.
- plan_gap(code): the measured cost of a regular plan, i.e. how much faster the direct plan of the
  same fund grew a year (that difference is the distributor commission in the expense ratio).
- liquid_yield(): today's annualised 3-month return of direct-plan liquid funds, a close daily proxy
  for the 91-day Treasury bill rate (used as India's risk-free rate).
- NIFTY50_TR: a large Nifty 50 index fund (direct plan); its NAV tracks the Nifty 50 Total Return
  Index (dividends included) minus a small fee, making a fairer benchmark than the price index.
"""
import pickle
import re
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

import risk_kit as rk

NAVALL_URL = "https://www.amfiindia.com/spages/NAVAll.txt"
CACHE = rk.CACHE_DIR
NAV_CACHE = CACHE / "nav"
NIFTY50_TR = "MF:120716"  # UTI Nifty 50 Index Fund - Direct Plan - Growth
BENCH_LABELS = {NIFTY50_TR: "Nifty 50 TR"}
WINDOWS = {"1Y": 1, "3Y": 3, "5Y": 5}


# ---------------------------------------------------------------------------
# AMFI scheme master
# ---------------------------------------------------------------------------
def parse_navall(text):
    """AMFI NAVAll.txt -> DataFrame: code, name, plan, option, nav, date, category, scheme_type, amc."""
    rows, category, scheme_type, amc = [], "", "", ""
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("Scheme Code"):
            continue
        if ";" not in line:
            if "Schemes(" in line and line.endswith(")"):
                scheme_type, category = line.split("(", 1)
                category = category[:-1].strip()
                scheme_type = scheme_type.strip()
            else:
                amc = line
            continue
        f = [x.strip() for x in line.split(";")]
        if len(f) < 6 or not f[0].isdigit():
            continue
        if len(f) >= 8:
            code, name, plan, option, nav, date = f[0], f[3], f[4], f[5], f[6], f[7]
        else:  # older 6-field format: plan and option are inside the name
            code, name, nav, date = f[0], f[3], f[4], f[5]
            low = name.lower()
            plan = "Direct Plan" if "direct" in low else "Regular Plan"
            option = "IDCW" if any(x in low for x in ("idcw", "dividend", "payout")) else "Growth"
        rows.append((int(code), name, plan, option, pd.to_numeric(nav, errors="coerce"),
                     pd.to_datetime(date, format="%d-%b-%Y", errors="coerce"), category, scheme_type, amc))
    df = pd.DataFrame(rows, columns=["code", "name", "plan", "option", "nav", "date", "category", "scheme_type", "amc"])
    # AMFI spells some categories two ways ("Hybrid Scheme" / "Hybrid Schemes")
    df["category"] = (df["category"].str.replace(r"\bSchemes\b", "Scheme", regex=True)
                      .str.replace(r"\s+", " ", regex=True).str.strip())
    # Many rows leave plan / option blank and put them in the name instead
    name_low = df["name"].str.lower()
    plan_low = df["plan"].fillna("").str.lower()
    opt_low = df["option"].fillna("").str.lower()
    df["direct"] = plan_low.str.contains("direct") | ((plan_low == "") & name_low.str.contains(r"\bdirect\b"))
    text = opt_low.where(opt_low != "", name_low)
    df["growth"] = text.str.contains("growth") & ~text.str.contains("idcw|dividend|payout|bonus")
    df["plan"] = np.where(df["plan"].fillna("") == "", np.where(df["direct"], "Direct Plan", "Regular Plan"), df["plan"])
    return df


def amfi_schemes(max_age_hours=20):
    """All schemes in AMFI's daily file, cached on disk."""
    path = CACHE / "amfi_navall.txt"
    if path.exists() and time.time() - path.stat().st_mtime < max_age_hours * 3600:
        text = path.read_text()
    else:
        import urllib.request
        req = urllib.request.Request(NAVALL_URL, headers={"User-Agent": "risk-kit/1.0"})
        with urllib.request.urlopen(req, timeout=90) as r:
            text = r.read().decode("utf-8", errors="ignore")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return parse_navall(text)


def nav_history(code, max_age_hours=20):
    """Full NAV history for one scheme, cached on disk for a day."""
    path = NAV_CACHE / f"{int(code)}.pkl"
    if path.exists() and time.time() - path.stat().st_mtime < max_age_hours * 3600:
        return pickle.loads(path.read_bytes())
    s = rk.get_mf_nav(int(code))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(s))
    return s


def nav_histories(codes, workers=8):
    def one(c):
        try:
            return c, nav_history(c)
        except Exception:
            return c, None
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return {c: s for c, s in pool.map(one, list(codes)) if s is not None and len(s) > 30}


# ---------------------------------------------------------------------------
# Metrics and ranking
# ---------------------------------------------------------------------------
def fund_metrics(nav, rf=0.06, as_of=None):
    """Annualised 1/3/5-year returns, and 3-year volatility, Sharpe and worst fall, from a NAV series."""
    nav = nav.dropna()
    as_of = as_of or nav.index[-1]
    nav = nav[nav.index <= as_of]
    out = {}
    for k, yrs in WINDOWS.items():
        start = as_of - pd.DateOffset(years=yrs)
        past = nav[nav.index <= start]
        out[k] = (nav.iloc[-1] / past.iloc[-1]) ** (1 / yrs) - 1 if len(past) and past.index[-1] > start - pd.Timedelta(days=10) else np.nan
    w3 = nav[nav.index > as_of - pd.DateOffset(years=3)]
    if len(w3) > 500 and not np.isnan(out["3Y"]):
        r = w3.pct_change().dropna()
        out["Vol 3Y"] = r.std() * np.sqrt(252)
        out["Sharpe 3Y"] = (out["3Y"] - rf) / out["Vol 3Y"] if out["Vol 3Y"] > 0 else np.nan
        out["Worst fall 3Y"] = (w3 / w3.cummax() - 1).min()
    else:
        out["Vol 3Y"] = out["Sharpe 3Y"] = out["Worst fall 3Y"] = np.nan
    return out


PASSIVE = re.compile(r"index fund|\betf\b|fund of fund|fof", re.IGNORECASE)


def tracked_index(name, amc=""):
    """'UTI Nifty 50 Index Fund' -> 'nifty 50 index': the index a passive fund tracks (fund house removed)."""
    n = re.sub(r"[^a-z0-9& ]+", " ", str(name).lower())
    n = re.sub(r"\s+", " ", n).strip()
    amc_words = set(re.sub(r"[^a-z0-9& ]+", " ", str(amc).lower()).split()) - {"mutual", "fund"}
    words = n.split()
    while words and (words[0] in amc_words or words[0] in {"mutual", "fund"}):
        words.pop(0)
    if words[:1] == ["etf"]:  # "Nippon India ETF Nifty 50 BeES" -> "nifty 50 etf"
        return " ".join(w for w in words[1:] if w not in {"bees", "etf"}) + " etf"
    n = " ".join(words)
    m = re.search(r"^(.*?\b(index|etf))\b", n)
    return (m.group(1) if m else n).strip()


def peer_group(code, schemes):
    """
    Same category, plan and option; only schemes with a NAV in the last 10 days.
    Index funds and ETFs are compared only with funds tracking the same index (a Nifty 50 fund vs other
    Nifty 50 funds, not vs pharma or Nasdaq index funds).
    """
    row = schemes[schemes["code"] == int(code)]
    if row.empty:
        return row, None
    r = row.iloc[0]
    recent = schemes["date"] >= schemes["date"].max() - pd.Timedelta(days=10)
    grp = schemes[(schemes["category"] == r["category"]) & (schemes["direct"] == r["direct"])
                  & (schemes["growth"] == r["growth"]) & recent]
    if PASSIVE.search(r["category"]) or PASSIVE.search(r["name"]):
        idx = tracked_index(r["name"], r["amc"])
        pool = schemes[(schemes["direct"] == r["direct"]) & (schemes["growth"] == r["growth"]) & recent
                       & (schemes["category"].str.contains(PASSIVE) | schemes["name"].str.contains(PASSIVE))]
        same = pool[[tracked_index(n, a) == idx for n, a in zip(pool["name"], pool["amc"])]]
        if len(same) >= 3:
            r = r.copy()
            r["category"] = f"Index Funds - {idx.title().replace('Etf', 'ETF')} funds"
            return same, r
    return grp, r


def rank_in_group(metrics, code):
    """metrics: DataFrame indexed by code. Returns per-metric (rank, count, percentile) for `code`.
    Percentile 100 = best. Lower is better for volatility and worst fall magnitude."""
    out = {}
    for col in ["1Y", "3Y", "5Y", "Vol 3Y", "Sharpe 3Y", "Worst fall 3Y"]:
        s = metrics[col].dropna()
        if code not in s.index:
            out[col] = None
            continue
        asc = col == "Vol 3Y"  # low volatility ranks first
        ranks = s.rank(ascending=asc, method="min")
        n = len(s)
        rank = int(ranks[code])
        out[col] = {"rank": rank, "n": n, "pct": 100 * (n - rank) / max(n - 1, 1), "median": s.median(),
                    "value": s[code]}
    return out


def quartile_label(pct):
    if pct is None or np.isnan(pct):
        return "Not enough history"
    return ("Top quartile" if pct >= 75 else "Above average" if pct >= 50 else
            "Below average" if pct >= 25 else "Bottom quartile")


def peer_report(code, schemes=None, rf=0.06, histories=None):
    """
    Ranks one fund within its category. Returns dict with category, n_peers, ranks (rank_in_group),
    score (average percentile of 3Y return, 5Y return and 3Y Sharpe), label, and the peer metrics table.
    """
    schemes = amfi_schemes() if schemes is None else schemes
    grp, row = peer_group(code, schemes)
    if row is None:
        return None
    hist = histories if histories is not None else nav_histories(grp["code"])
    if int(code) not in hist:
        return None
    as_of = hist[int(code)].index[-1]
    metrics = pd.DataFrame({c: fund_metrics(s, rf, as_of) for c, s in hist.items() if c in set(grp["code"])}).T
    metrics["name"] = grp.set_index("code")["name"].reindex(metrics.index)
    ranks = rank_in_group(metrics, int(code))
    pcts = [ranks[k]["pct"] for k in ("3Y", "5Y", "Sharpe 3Y") if ranks.get(k)]
    score = float(np.mean(pcts)) if pcts else (ranks["1Y"]["pct"] if ranks.get("1Y") else np.nan)
    return {"code": int(code), "category": row["category"], "plan": row["plan"], "n_peers": len(metrics),
            "ranks": ranks, "score": score, "label": quartile_label(score), "metrics": metrics}


def sibling_plan(code, schemes):
    """The same fund's other plan (direct <-> regular), same option."""
    row = schemes[schemes["code"] == int(code)]
    if row.empty:
        return None
    r = row.iloc[0]
    sib = schemes[(schemes["name"] == r["name"]) & (schemes["amc"] == r["amc"]) & (schemes["direct"] != r["direct"])
                  & (schemes["growth"] == r["growth"]) & (schemes["option"] == r["option"])]
    if sib.empty:
        sib = schemes[(schemes["name"] == r["name"]) & (schemes["direct"] != r["direct"]) & (schemes["growth"] == r["growth"])]
    return None if sib.empty else int(sib.iloc[0]["code"])


def plan_gap(code, schemes=None, years=3):
    """
    Measured yearly cost of a regular plan: direct-plan growth minus regular-plan growth per year,
    over the last `years` (or the longest common period of at least a year).
    Returns dict(direct_code, regular_code, gap, years) or None.
    """
    schemes = amfi_schemes() if schemes is None else schemes
    sib = sibling_plan(code, schemes)
    if sib is None:
        return None
    me_direct = bool(schemes.loc[schemes["code"] == int(code), "direct"].iloc[0])
    d_code, r_code = (int(code), sib) if me_direct else (sib, int(code))
    try:
        d, r = nav_history(d_code), nav_history(r_code)
    except Exception:
        return None
    both = pd.concat([d, r], axis=1, join="inner").dropna()
    if len(both) < 250:
        return None
    end = both.index[-1]
    start = max(both.index[0], end - pd.DateOffset(years=years))
    w = both[both.index >= start]
    yrs = (w.index[-1] - w.index[0]).days / 365.25
    cagr = (w.iloc[-1] / w.iloc[0]) ** (1 / yrs) - 1
    return {"direct_code": d_code, "regular_code": r_code, "gap": float(cagr.iloc[0] - cagr.iloc[1]), "years": yrs}


def liquid_yield(schemes=None, days=91, top=25):
    """Median annualised return over the last `days` of direct-plan growth liquid funds (risk-free proxy)."""
    schemes = amfi_schemes() if schemes is None else schemes
    liq = schemes[schemes["category"].str.contains("Liquid Fund", case=False) & schemes["direct"] & schemes["growth"]]
    liq = liq[liq["date"] >= liq["date"].max() - pd.Timedelta(days=7)].head(top)
    hist = nav_histories(liq["code"])
    ys = []
    for s in hist.values():
        past = s[s.index <= s.index[-1] - pd.Timedelta(days=days)]
        if len(past):
            span = (s.index[-1] - past.index[-1]).days
            ys.append((s.iloc[-1] / past.iloc[-1]) ** (365 / span) - 1)
    if not ys:
        raise ValueError("No liquid fund data")
    return float(np.median(ys)), max(s.index[-1] for s in hist.values())
