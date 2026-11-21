"""
Matches Indian mutual fund names from statements to AMFI scheme codes, so their NAV history
can be downloaded (risk_kit.get_prices with "MF:<code>").

Statement names are abbreviated and inconsistent ("HDFC AGGRESSIVE HYBRID FUND-REG - Gr"), so
matching works in two steps:
  1. Name: compare the words of the name with all ~38,000 AMFI schemes, keeping only the same
     plan (Regular / Direct) and option (Growth / IDCW).
  2. NAV check: value / units on the statement is the fund's NAV on the statement date. The
     candidate whose published NAV matches it (within 0.5%) on a recent date is the right scheme.
"""
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

import risk_kit as rk

SCHEMES_CACHE = rk.CACHE_DIR / "mf_schemes.json"
ABBREVIATIONS = {"reg": "regular", "dir": "direct", "gr": "growth", "gth": "growth", "div": "idcw",
                 "dividend": "idcw", "pl": "plan", "opt": "option", "eq": "equity", "mf": "", "cap": "cap"}
PLAN_OPTION_WORDS = {"regular", "direct", "growth", "idcw", "payout", "reinvestment", "bonus", "plan", "option",
                     "fund", "scheme", "the", "of", "and", "mutual", "retail", "institutional"}
DEAD_WORDS = {"segregated", "unclaimed", "matured"}
NAV_TOLERANCE = 0.005
NAV_LOOKBACK_DAYS = 75


def _tokens(name):
    words = re.findall(r"[a-z0-9]+", str(name).lower().replace("&", " and "))
    return [ABBREVIATIONS.get(w, w) for w in words if ABBREVIATIONS.get(w, w)]


def parse_name(name):
    """Returns (core words, plan, option) for a scheme name"""
    toks = _tokens(name)
    plan = "direct" if "direct" in toks else "regular"
    option = "idcw" if {"idcw", "payout", "reinvestment", "bonus"} & set(toks) else "growth"
    core = [w for w in toks if w not in PLAN_OPTION_WORDS and len(w) > 1]
    return core, plan, option


def load_schemes(max_age_days=7):
    """All AMFI schemes (code and name), cached on disk for a week"""
    if SCHEMES_CACHE.exists() and time.time() - SCHEMES_CACHE.stat().st_mtime < max_age_days * 86400:
        data = json.loads(SCHEMES_CACHE.read_text())
    else:
        data = rk._http_json(rk.MFAPI_URL, timeout=90)
        SCHEMES_CACHE.parent.mkdir(parents=True, exist_ok=True)
        SCHEMES_CACHE.write_text(json.dumps(data))
    rows = []
    for d in data:
        toks = set(_tokens(d["schemeName"]))
        if toks & DEAD_WORDS:
            continue
        core, plan, option = parse_name(d["schemeName"])
        has_growth = "growth" in toks
        rows.append((int(d["schemeCode"]), d["schemeName"], frozenset(core), core[0] if core else "",
                     plan, "growth" if has_growth and option == "growth" else option if not has_growth else "mixed"))
    return pd.DataFrame(rows, columns=["code", "name", "core", "house", "plan", "option"])


def candidates(name, schemes, k=6):
    """Best name matches with the same plan and option, best first"""
    core, plan, option = parse_name(name)
    if not core:
        return schemes.iloc[0:0]
    pool = schemes[(schemes["plan"] == plan) & (schemes["option"] == option)]
    same_house = pool[pool["house"] == core[0]]
    pool = same_house if len(same_house) else pool
    target = set(core)
    score = pool["core"].map(lambda c: len(target & c) / len(target | c))
    out = pool.assign(score=score)
    return out[out["score"] > 0].sort_values("score", ascending=False).head(k)


def _nav_matches(code, implied_nav):
    """Date on which the scheme's NAV equals implied_nav (within tolerance), or None"""
    try:
        latest = rk._http_json(f"{rk.MFAPI_URL}/{code}/latest", timeout=20)["data"][0]
        if abs(float(latest["nav"]) / implied_nav - 1) > 0.15:  # quick filter before the full history
            return None
        nav = rk.get_mf_nav(code)
    except Exception:
        return None
    recent = nav[nav.index >= nav.index[-1] - pd.Timedelta(days=NAV_LOOKBACK_DAYS)]
    error = (recent / implied_nav - 1).abs()
    return error.idxmin() if len(error) and error.min() <= NAV_TOLERANCE else None


def match_fund(name, units=None, value=None, schemes=None):
    """
    Finds the AMFI scheme for a statement name.
    Returns a dict: code, scheme, status ("verified" | "name only" | "not found"), detail, alternatives.
    """
    schemes = load_schemes() if schemes is None else schemes
    cands = candidates(name, schemes)
    alternatives = [(int(c), n) for c, n in zip(cands["code"], cands["name"])]
    if cands.empty:
        return {"code": None, "scheme": None, "status": "not found",
                "detail": "No scheme with a similar name", "alternatives": []}
    implied = value / units if units and value and units > 0 else None
    if implied:
        for code, scheme in alternatives:
            date = _nav_matches(code, implied)
            if date is not None:
                return {"code": code, "scheme": scheme, "status": "verified",
                        "detail": f"NAV ₹{implied:,.2f} matches the published NAV on {date:%d %b %Y}",
                        "alternatives": alternatives}
    best = cands.iloc[0]
    detail = ("Closest name; the NAV on your statement didn't match a recent published NAV"
              if implied else "Closest name (no units/value to check the NAV)")
    return {"code": int(best["code"]), "scheme": best["name"], "status": "name only",
            "detail": detail, "alternatives": alternatives}


def match_funds(funds, schemes=None):
    """funds: list of (name, units, value). Matches them in parallel; returns a list of dicts."""
    schemes = load_schemes() if schemes is None else schemes
    with ThreadPoolExecutor(max_workers=6) as pool:
        return list(pool.map(lambda f: match_fund(*f, schemes=schemes), funds))


def short_label(scheme_name):
    """'BANDHAN MID CAP FUND - Regular Plan - Growth' -> 'Bandhan Mid Cap'"""
    base = re.split(r"\s+-\s+|\s*\((?:regular|direct|growth|idcw)", scheme_name, flags=re.I)[0]
    base = re.sub(r"\b(fund|scheme|plan|growth|option|regular|direct)\b", "", base, flags=re.I)
    base = re.sub(r"\s{2,}", " ", base).strip(" -")
    if base.isupper() or base.islower():
        base = " ".join(w if len(w) <= 3 and w.isupper() and w not in {"MID", "CAP"} else w.capitalize()
                        for w in base.split())
    return base
