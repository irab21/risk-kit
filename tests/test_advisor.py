"""Tests for the rule-based action plan and the chat brief."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import advisor  # noqa: E402
import commentary as cmt  # noqa: E402
from test_commentary import facts  # noqa: E402

NAMES = ["Fund A", "Fund B", "Fund C"]


def holdings(regular=True):
    return pd.DataFrame({"weight": [0.4, 0.3, 0.3], "risk_share": [0.4, 0.3, 0.3], "ret": [0.1, 0.09, 0.08],
                         "vol": [0.15, 0.15, 0.15], "sharpe": [0.5, 0.4, 0.3], "regular": [regular] * 3}, index=NAMES)


def lockstep():
    return pd.DataFrame(0.95, index=NAMES, columns=NAMES) + pd.DataFrame(
        [[0.05, 0, 0], [0, 0.05, 0], [0, 0, 0.05]], index=NAMES, columns=NAMES)


def titles(actions):
    return [a["title"] for a in actions]


def test_overlapping_regular_funds():
    acts = advisor.action_plan(facts(), holdings(), lockstep(), {y: 1.1 for y in range(1, 11)}, "₹")
    t = titles(acts)
    assert any("direct plans" in x for x in t)
    assert any("look-alike" in x for x in t)
    assert any("cushion" in x for x in t)
    assert [a["priority"] for a in acts] == sorted([a["priority"] for a in acts], key=["High", "Medium", "Low"].index)
    assert "Keep Fund A and Fund B" in next(a["how"] for a in acts if "look-alike" in a["title"])


def test_diversified_direct_portfolio_gets_few_actions():
    corr = pd.DataFrame([[1, 0.1, 0.2], [0.1, 1, 0.0], [0.2, 0.0, 1]], index=NAMES, columns=NAMES)
    acts = advisor.action_plan(facts(), holdings(regular=False), corr, {y: 0.9 if y < 3 else 1.2 for y in range(1, 11)})
    assert titles(acts) == ["Keep money needed within 3 years out", "Rebalance once a year"]


def test_lagging_market_and_never_safe():
    acts = advisor.action_plan(facts(ret=0.05, bench_ret=0.10, final_value=160_000, bench_final_value=260_000),
                               holdings(None), lockstep(), {y: 0.9 for y in range(1, 11)})
    t = titles(acts)
    assert "Make a low-cost index fund your core" in t and "Only long-term money belongs here" in t


def test_brief_and_prompt():
    f = facts()
    verdict = cmt.build_commentary(f)
    acts = advisor.action_plan(f, holdings(), lockstep(), {y: 1.1 for y in range(1, 11)}, "₹")
    future = {1: {"bad": 90_000, "mid": 110_000, "good": 130_000, "loss": 0.2}}
    brief = advisor.portfolio_brief(f, holdings(), lockstep(), verdict, acts, future, "₹", pd.Timestamp("2026-11-19"))
    assert "Fund A | 40.0% | 40.0%" in brief and "regular plan" in brief and "1y:" in brief
    prompt = advisor.system_prompt(brief, "₹")
    assert "SEBI-registered" in prompt and brief in prompt


def test_market_aware_actions():
    reg = {"name": "Nifty 50", "drawdown": "Correction", "from_high": -0.14, "y1": -0.05, "volatility": "Elevated",
           "vix_ratio": 1.6, "mood": "Cautious"}
    t = titles(advisor.action_plan(facts(), holdings(False), lockstep(), {y: 1.1 for y in range(1, 11)}, market=reg))
    assert "Keep investing through the dip" in t and "Stagger new money" in t
    hot = {**reg, "drawdown": "Near its high", "from_high": -0.01, "y1": 0.25, "volatility": "Calm"}
    t = titles(advisor.action_plan(facts(), holdings(False), lockstep(), {y: 1.1 for y in range(1, 11)}, market=hot))
    assert "Rebalance into strength" in t and "Stagger new money" not in t


def test_actions_are_crisp():
    acts = advisor.action_plan(facts(years=2), holdings(), lockstep(), {y: 0.9 for y in range(1, 11)}, "₹")
    for a in acts:
        assert len(a["why"].split()) <= 25 and len(a["how"].split()) <= 25, a


def test_measured_cost_gap_peer_rank_and_harvest():
    h = holdings()
    h["gap"] = [0.012, 0.008, np.nan]
    h["peer"] = ["Top quartile", "Above average", "Bottom quartile"]
    h["peer_rank"] = [None, None, "#29 of 29 in Aggressive Hybrid Fund over 3 years"]
    acts = advisor.action_plan(facts(), h, lockstep(), {}, "₹",
                               extra={"tax_all": 50000, "harvest_gain": 90000, "harvest_value": 300000})
    cost = next(a for a in acts if a["title"].startswith("Move 3"))
    assert "Measured from NAVs" in cost["why"] and "tax" in cost["how"]
    assert any(t.startswith(f"Review {NAMES[2]}") for t in titles(acts))
    assert any("tax-free" in t for t in titles(acts))
    brief = advisor.portfolio_brief(facts(), h, lockstep(), {"grade": "B", "summary": "", "points": []}, acts, {},
                                    "₹", __import__("datetime").date(2026, 11, 19), extra={"xirr": 0.11, "bench_xirr": 0.12})
    assert "Bottom quartile" in brief and "XIRR 11.0%" in brief


def test_no_trim_advice_for_broad_index_core():
    acts = advisor.action_plan(facts(top_risk_name="UTI Nifty 50 Index", top_risk_share=0.56, top_risk_weight=0.35),
                               holdings(False), lockstep(), {}, "₹")
    assert not any(t.startswith("Trim") for t in titles(acts))
    acts = advisor.action_plan(facts(top_risk_name="Adani Enterprises", top_risk_share=0.56, top_risk_weight=0.35),
                               holdings(False), lockstep(), {}, "₹")
    assert any(t.startswith("Trim Adani") for t in titles(acts))


def test_asset_classes():
    ac = advisor.asset_class
    assert ac("MF:1", "ICICI Pru Corporate Bond", "Debt Scheme - Corporate Bond Fund") == {"debt": 1.0}
    assert ac("MF:2", "Nippon India Gold Savings", "Other Scheme - FoF Domestic") == {"gold": 1.0}
    assert ac("MF:3", "Motilal Oswal Nasdaq 100 FOF", "Other Scheme - FoF Overseas") == {"intl": 1.0}
    assert ac("MF:4", "ICICI Pru Equity & Debt", "Hybrid Scheme - Aggressive Hybrid Fund") == {"home": 0.75, "debt": 0.25}
    assert ac("RELIANCE.NS", "Reliance Industries") == {"home": 1.0}
    assert ac("AAPL", "Apple", cur="₹") == {"intl": 1.0}
    assert ac("SPY", "S&P 500 ETF", cur="$") == {"home": 1.0}
    assert ac("BND", "Total bond market ETF", cur="$") == {"debt": 1.0}
    assert advisor.holding_kind("GOLDBEES.NS", "Gold ETF (Nippon)") == "etf"
    assert advisor.holding_kind("TCS.NS", "Tata Consultancy Services") == "stock"


def test_target_mix_and_profile_actions():
    for h in advisor.HORIZONS:
        for c in advisor.COMFORTS:
            assert abs(sum(advisor.target_mix(h, c, "₹").values()) - 1) < 1e-9
    assert advisor.target_mix("10+ years", "Growth")["debt"] < advisor.target_mix("Under 3 years", "Careful")["debt"]
    h = holdings(False)
    classes = {n: {"home": 1.0} for n in NAMES}
    prof = {"horizon": "Under 3 years", "comfort": "Careful", "emergency": "No", "monthly": 10_000}
    acts = advisor.action_plan(facts(), h, lockstep(), {y: 1.1 for y in range(1, 11)}, "₹", profile=prof, classes=classes)
    t = titles(acts)
    assert t[0] == "Build an emergency fund first" or "Build an emergency fund first" in t[:3]
    mix = next(a for a in acts if a["title"] == "Move towards your target mix")
    assert mix["priority"] == "High" and "debt" in mix["how"] and "Add a cushion for market falls" not in t
    assert any(a["title"].startswith("Raise your SIP") and a["impact"] for a in acts)
    assert all(a["when"] in advisor.WHEN_ORDER and a["steps"] for a in acts)
    for a in acts:
        assert len(a["why"].split()) <= 30 and len(a["how"].split()) <= 30, a


def test_sip_step_up_beats_flat():
    flat = advisor.sip_future(10_000, 10, 0.10)
    assert 1.9e6 < flat < 2.2e6  # ~₹20 lakh for ₹10k/month at 10% over 10 years
    assert advisor.sip_future(10_000, 10, 0.10, 0.10) > flat * 1.4


def test_holding_review_verdicts():
    h = holdings(True)
    h["peer"] = [None, None, "Bottom quartile"]
    h["peer_rank"] = [None, None, "#40 of 42"]
    rev = advisor.holding_review(h, lockstep()).set_index("Holding")
    assert set(rev["Verdict"]) == {"Switch"}  # regular plans come first
    rev = advisor.holding_review(holdings(False), lockstep()).set_index("Holding")
    assert rev.loc["Fund A", "Verdict"] == "Keep" and rev.loc["Fund C", "Verdict"] == "Merge"


def test_bond_cushion_not_flagged_for_low_return():
    h = holdings(False)
    h["sharpe"] = [0.5, 0.4, -0.2]
    corr = pd.DataFrame(np.eye(3), index=NAMES, columns=NAMES)
    rev = advisor.holding_review(h, corr, 5, {"Fund C": {"debt": 1.0}}).set_index("Holding")
    assert rev.loc["Fund C", "Verdict"] == "Keep" and "cushion" in rev.loc["Fund C", "Why"]
    assert advisor.holding_review(h, corr, 5).set_index("Holding").loc["Fund C", "Verdict"] == "Review"
