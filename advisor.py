"""
"What should I do?" for a portfolio.

action_plan() turns the analysis into a prioritised to-do list with simple, visible rules
(no AI needed). portfolio_brief() writes the same facts as plain text, and stream_reply()
lets Claude answer follow-up questions about them in a chat, acting as a senior adviser.
"""
import re

import numpy as np
import pandas as pd

from commentary import is_broad_index

MODEL = "claude-opus-5-5"

DIVERSIFIERS = {
    "₹": "a short-duration debt or gilt fund (for stability), a gold ETF or fund, and an international equity fund",
    "$": "a high-quality bond fund (e.g. AGG or BND), gold (e.g. GLD) and international stocks (e.g. VXUS)",
}


def _pairs(corr):
    """All holding pairs with their correlation, highest first."""
    if len(corr) < 2:
        return pd.Series(dtype=float)
    mask = np.triu(np.ones(corr.shape, dtype=bool), 1)
    return corr.where(mask).stack().sort_values(ascending=False)


# ---------------------------------------------------------------------------------------------
# Asset mix: what each holding is, and a sensible target for the investor's goal
# ---------------------------------------------------------------------------------------------
HORIZONS = ["Under 3 years", "3–5 years", "5–10 years", "10+ years"]
COMFORTS = ["Careful", "Balanced", "Growth"]
HORIZON_YEARS = {"Under 3 years": 2, "3–5 years": 4, "5–10 years": 7, "10+ years": 15}
# Share in shares (home + international) by risk comfort and horizon: common rules of thumb
EQUITY_BY = {"Careful": [0.10, 0.30, 0.45, 0.60], "Balanced": [0.20, 0.45, 0.60, 0.70],
             "Growth": [0.30, 0.60, 0.75, 0.85]}
CLASSES = ["home", "intl", "debt", "gold"]
# Rough long-run returns for the SIP projection only (before tax, after costs)
EXPECTED = {"home": 0.11, "intl": 0.09, "debt": 0.07, "gold": 0.07}

US_BONDS = {"BND", "AGG", "TLT", "IEF", "SHY", "LQD", "TIP", "BNDX", "VGIT", "VGSH", "VGLT", "SCHZ", "GOVT",
            "MUB", "HYG", "JNK", "BIL", "SGOV", "VCIT", "VCSH", "IEI", "SPTL"}
US_GOLD = {"GLD", "IAU", "SGOL", "GLDM", "DBC", "GSG", "PDBC", "SLV"}
US_INTL = {"VXUS", "VEA", "VWO", "EFA", "IEMG", "IXUS", "ACWX", "EEM", "SCHF", "SPDW", "IEFA", "VGK", "EWJ"}
_DEBT = re.compile(r"debt scheme|liquid|overnight|money market|gilt|\bbond|duration|treasur|\bsdl\b|banking and psu|"
                   r"credit risk|floater|arbitrage|t-bill|fixed maturity|target maturity", re.I)
_GOLD = re.compile(r"\bgold|silver|commodit", re.I)
_INTL_ANY = re.compile(r"overseas|international|global|world|emerging|ex-us|developed market", re.I)
_INTL_IN = re.compile(r"nasdaq|s&p 500|\bus equity|u\.s\.|fang|hang seng|japan|china|taiwan|\bus\b", re.I)
_FUNDISH = re.compile(r"fund|etf|bees|\bfof\b|index", re.I)
_HYBRID = [(r"aggressive hybrid", {"home": 0.75, "debt": 0.25}), (r"conservative hybrid", {"home": 0.25, "debt": 0.75}),
           (r"multi asset", {"home": 0.6, "debt": 0.25, "gold": 0.15}), (r"equity savings", {"home": 0.35, "debt": 0.65}),
           (r"balanced advantage|dynamic asset", {"home": 0.6, "debt": 0.4}), (r"balanced hybrid", {"home": 0.5, "debt": 0.5}),
           (r"hybrid", {"home": 0.6, "debt": 0.4})]


GOAL_PHRASE = {"Under 3 years": "short (under 3 years)", "3–5 years": "3–5 year", "5–10 years": "5–10 year",
               "10+ years": "10+ year"}


def lower_label(label):
    """'Debt & cash' -> 'debt & cash', but keeps 'Indian shares' and 'US shares'."""
    return label if label.split()[0] in ("Indian", "US") else label[0].lower() + label[1:]


def class_labels(cur):
    india = cur == "₹"
    return {"home": "Indian shares" if india else "US shares", "intl": "International shares",
            "debt": "Debt & cash" if india else "Bonds & cash", "gold": "Gold" if india else "Gold & commodities"}


def asset_class(ticker, name, category="", cur="₹"):
    """
    What a holding is, as {class: fraction}: home shares, international shares, debt or gold.
    Uses the AMFI category for mutual funds and the name / ticker for everything else. A rule of thumb:
    hybrid funds are split by their typical mix, and flexi-cap funds' small overseas slice is ignored.
    """
    t, text = str(ticker).upper(), f"{name or ''} {category or ''}"
    india = cur == "₹"
    is_mf = t.startswith("MF:")
    if t in US_GOLD or _GOLD.search(text):
        return {"gold": 1.0}
    if t in US_BONDS or _DEBT.search(text):
        return {"debt": 1.0}
    if re.search(r"hybrid|balanced advantage|dynamic asset|multi asset|equity savings", text, re.I):
        return next(dict(mix) for pat, mix in _HYBRID if re.search(pat, text, re.I))
    if t == "VT":
        return {"home": 0.6, "intl": 0.4} if not india else {"intl": 1.0}
    if t in US_INTL:
        return {"intl": 1.0}
    if is_mf or _FUNDISH.search(text):
        if _INTL_ANY.search(text) or (india and _INTL_IN.search(text)):
            return {"intl": 1.0}
        return {"home": 1.0}
    # a single stock
    if india and not t.startswith("^") and not t.endswith((".NS", ".BO")):
        return {"intl": 1.0}
    return {"home": 1.0}


def holding_kind(ticker, name, category=""):
    """'fund' (mutual fund), 'etf' or 'stock'."""
    t, text = str(ticker).upper(), f"{name or ''} {category or ''}"
    if t.startswith("MF:"):
        return "fund"
    if re.search(r"\betf\b|bees|index fund|\btrust\b", text, re.I) or t in US_BONDS | US_GOLD | US_INTL | {
            "SPY", "VOO", "IVV", "QQQ", "VTI", "VT", "DIA", "IWM", "SCHD", "VIG", "VUG", "VTV"}:
        return "etf"
    return "stock"


def current_mix(weights, classes):
    """weights: {name: weight}; classes: {name: {class: fraction}} -> {class: share of money}."""
    mix = dict.fromkeys(CLASSES, 0.0)
    for n, w in weights.items():
        for c, frac in classes.get(n, {"home": 1.0}).items():
            mix[c] += w * frac
    tot = sum(mix.values()) or 1.0
    return {c: v / tot for c, v in mix.items()}


def target_mix(horizon="5–10 years", comfort="Balanced", cur="₹"):
    """A sensible starting mix for a goal: rules of thumb, not a personal recommendation."""
    eq = EQUITY_BY[comfort][HORIZONS.index(horizon)]
    gold = (0.05 if horizon == "Under 3 years" else 0.10) if cur == "₹" else 0.05
    intl = round(eq * (0.15 if cur == "₹" else 0.30), 2)
    return {"home": eq - intl, "intl": intl, "debt": round(1 - eq - gold, 2), "gold": gold}


def mix_gaps(current, target, value):
    """Per class: current, target, gap (target - current) and the amount to move, biggest gaps first."""
    rows = [{"cls": c, "current": current.get(c, 0.0), "target": target[c], "gap": target[c] - current.get(c, 0.0),
             "amount": (target[c] - current.get(c, 0.0)) * value} for c in CLASSES]
    return sorted(rows, key=lambda r: -abs(r["gap"]))


def sip_future(monthly, years, rate, step_up=0.0):
    """Value of a monthly SIP after `years`, raised by `step_up` once a year."""
    r, total, amt = (1 + rate) ** (1 / 12) - 1, 0.0, monthly
    for month in range(int(years * 12)):
        if month and month % 12 == 0:
            amt *= 1 + step_up
        total = (total + amt) * (1 + r)
    return total


def action_plan(f, holdings, corr, bad_case, cur="$", market=None, extra=None, profile=None, classes=None,
                today=None):
    """
    f: commentary.PortfolioFacts. holdings: DataFrame indexed by name with columns
    weight, risk_share, ret, vol, sharpe, regular (True/False/None for non-funds), optionally gap, peer,
    peer_rank and kind ('fund' / 'etf' / 'stock').
    corr: correlation matrix. bad_case: {years: 10th-percentile wealth multiple}.
    market: market.regime() output for the portfolio's home market, or None.
    profile: {"horizon", "comfort", "emergency" ('Yes'/'No'/'Not sure'), "monthly"} from the investor, or None.
    classes: {name: asset_class(...)} to compare the mix with the target for the profile.
    Returns a list of actions (dicts with priority, title, why, how, when, steps, impact), most urgent first.
    `why` and `how` are kept to one line each; `steps` holds the detail.
    """
    m = f.money
    extra = extra or {}
    india = cur == "₹"
    today = today or pd.Timestamp.today()
    actions = []

    def add(priority, title, why, how, when="This month", steps=(), impact=None, yearly=0.0):
        actions.append({"priority": priority, "title": title, "why": why, "how": how, "when": when,
                        "steps": list(steps), "impact": impact, "yearly": yearly})

    # Safety first: emergency money
    if profile and profile.get("emergency") == "No":
        add("High", "Build an emergency fund first",
            "Without one, a job loss or hospital bill forces you to sell, often at the worst time.",
            "Set aside 6 months of expenses in " + ("a liquid fund or savings account." if india else "a savings or money-market account."),
            when="This week",
            steps=["Add up 6 months of essential spending: rent or EMI, bills, food, insurance premiums.",
                   "Park it in " + ("a liquid or overnight fund (money in 1 working day) or a sweep-in FD."
                                    if india else "a high-yield savings or money-market account."),
                   "Fund it from new savings first; only sell investments if it would take more than a year.",
                   "Check you also have health insurance and, if anyone depends on you, term life cover."])

    elif profile and profile.get("emergency") == "Not sure":
        add("Low", "Check your safety net",
            "Investing works best when nothing can force you to sell in a fall.",
            "Confirm 6 months of expenses in safe savings, plus health and term-life cover.",
            when="This week",
            steps=["Emergency fund: 6 months of essential spending, kept " + ("in a liquid fund or savings account."
                                                                              if india else "in savings or a money-market fund."),
                   "Health insurance that covers your family, beyond any employer policy.",
                   "Term life cover of 10–15× yearly income if anyone depends on you.",
                   "Once all three are in place, set 'Emergency fund in place?' to Yes above."])

    # Costs: regular plans
    reg = holdings[holdings["regular"] == True]  # noqa: E712
    if len(reg):
        amt = f.value * reg["weight"].sum()
        gaps = reg["gap"] if "gap" in reg else pd.Series(np.nan, index=reg.index)
        measured = gaps.notna().any()
        yearly = float((f.value * reg["weight"] * gaps.fillna(0.01)).sum())
        rate = yearly / amt if amt else 0.01
        drag10 = amt * ((1 + f.ret) ** 10 - (1 + f.ret - rate) ** 10)
        tax_note = (f" Selling all your funds today would cost about {m(extra['tax_all'])} in tax, so switch "
                    "units held over a year first.") if extra.get("tax_all") else " Check exit load and capital-gains tax first."
        add("High", f"Move {len(reg)} regular-plan fund{'s' if len(reg) > 1 else ''} to direct plans",
            (f"Measured from NAVs, the direct plans grew {rate*100:.2f}% a year faster: about {m(yearly)} a year now, "
             f"{m(drag10)} over 10 years." if measured else
             f"Distributor commissions cost ~1% a year: about {m(yearly)} now, {m(drag10)} over 10 years."),
            "Switch to the direct plan of the same fund." + tax_note,
            when="This month", impact=f"≈ {m(yearly)} a year", yearly=yearly,
            steps=[f"Funds affected: {', '.join(reg.index)}.",
                   "In My funds → Tax, note which units are older than 1 year and which are still in the exit-load window.",
                   "Switch the old, long-term units first: up to ₹1.25 lakh of equity gains a year is tax-free." if india
                   else "Switch the units with the smallest taxable gain first.",
                   "Use the fund house website, MF Central or your platform: choose 'Switch' to the Direct – Growth option "
                   "of the same scheme." if india else "Ask your platform to convert to the lowest-cost share class.",
                   "Move the rest as units pass 1 year or leave the exit-load window. Point new SIPs at the direct plan today."])

    # Weak funds vs their category
    if "peer" in holdings:
        weak = holdings[(holdings["peer"] == "Bottom quartile") & (holdings["weight"] >= 0.05)]
        for name, h in weak.iterrows():
            add("Medium", f"Review {name}",
                f"It ranks in the bottom quarter of its category ({h['peer_rank'] or 'on returns and risk'}).",
                "Give it one more year only if there's a clear reason; otherwise move to a top-half fund in the same category.",
                when="This month",
                steps=["Check its 1, 3 and 5-year ranks in My funds: one bad year is noise, three is a pattern.",
                       "Look for a reason: a new fund manager, a changed strategy, or a style (e.g. value) out of favour.",
                       "If there's no good reason, pick a fund in the top half of the same category with a 5+ year record and low cost.",
                       "Stop new SIPs into it now; move the existing money tax-efficiently over the next few months."])

    # Tax-free gains harvesting (India)
    if extra.get("harvest_gain", 0) > 5000:
        add("Medium", f"Book {m(extra['harvest_gain'])} of gains tax-free this year",
            "Up to ₹1.25 lakh of long-term equity gains a year is tax-free, and unused room is lost on 31 March.",
            f"Sell and rebuy about {m(extra['harvest_value'])} of your oldest units; this resets cost and cuts future tax.",
            when="This week" if today.month in (1, 2, 3) else "This month",
            impact=f"≈ {m(extra['harvest_gain'] * 0.125)} less tax later",
            steps=[f"Redeem about {m(extra['harvest_value'])} of units held over a year (oldest first: My funds → Tax lists them).",
                   "Reinvest the same day or next day in the same fund (direct plan) so you stay invested.",
                   "Check exit loads: units over a year old usually have none.",
                   "Keep the redemption statement for your tax return; repeat every year before 31 March."])

    # Overlap
    pairs = _pairs(corr)
    twins = pairs[pairs > 0.9]
    if len(twins) >= 2 or (len(holdings) >= 3 and len(pairs) and pairs.mean() > 0.85):
        involved = sorted({n for pair in twins.index for n in pair}) or list(holdings.index)
        best = holdings.loc[involved, "sharpe"].sort_values(ascending=False).head(2).index
        add("High" if len(involved) >= 3 else "Medium",
            f"Consolidate {len(involved)} look-alike holdings into 1–2",
            f"They move together (correlation {twins.min() if len(twins) else pairs.mean():.2f}+), so they add fees, not protection.",
            f"Keep {' and '.join(best)} (best reward for risk); exit the rest gradually.",
            when="Next 3–6 months",
            steps=[f"Look-alikes: {', '.join(involved)}.",
                   f"Keep {' and '.join(best)}: they earned the most per unit of risk.",
                   "Stop new SIPs into the others now and redirect them to the ones you keep." if india
                   else "Redirect new contributions to the ones you keep.",
                   "Move existing money in 2–3 steps, using tax-free gains and avoiding exit loads.",
                   "Aim for 4–6 funds in total; more rarely adds diversification."])
    elif "kind" in holdings and (holdings["kind"] == "fund").sum() > 8:
        add("Medium", f"Cut down from {(holdings['kind'] == 'fund').sum()} funds to 4–6",
            "Past six or so funds you mostly own the same shares several times over.",
            "Keep one fund per job: core index, flexi or mid cap, debt, gold, international.",
            when="Next 3–6 months",
            steps=["List each fund's job; two funds with the same job is one too many.",
                   "Keep the cheaper one with the better 5-year rank in its category.",
                   "Stop SIPs into the rest and move the money over a few months, tax-efficiently."])

    # Asset mix vs the investor's target
    mix_action = False
    if profile and classes:
        labels = class_labels(cur)
        cur_mix = current_mix(holdings["weight"].to_dict(), classes)
        tgt = target_mix(profile.get("horizon", "5–10 years"), profile.get("comfort", "Balanced"), cur)
        gaps = mix_gaps(cur_mix, tgt, f.value)
        if abs(gaps[0]["gap"]) >= 0.10:
            mix_action = True
            eq_now, eq_tgt = cur_mix["home"] + cur_mix["intl"], tgt["home"] + tgt["intl"]
            short = profile.get("horizon") in ("Under 3 years", "3–5 years")
            adds = [g for g in gaps if g["gap"] >= 0.05]
            cuts = [g for g in gaps if g["gap"] <= -0.05]
            add_txt = " and ".join(f"{m(g['amount'])} to {lower_label(labels[g['cls']])}" for g in adds[:2])
            cut_txt = " and ".join(lower_label(labels[g["cls"]]) for g in cuts[:2])
            add("High" if short and eq_now > eq_tgt + 0.15 else "Medium",
                "Move towards your target mix",
                f"Shares are {eq_now*100:.0f}% of your money vs about {eq_tgt*100:.0f}% for a "
                f"{GOAL_PHRASE.get(profile.get('horizon'), '5–10 year')} goal and a {profile.get('comfort', 'Balanced').lower()} risk profile.",
                (f"Add about {add_txt}" if add_txt else "Rebalance") + (f", taken from {cut_txt}." if cut_txt else "."),
                when="Next 3–6 months",
                steps=["See the Target mix tab for the full comparison and amounts.",
                       ("Point new SIPs at what's under target first: no tax, no exit load." if india
                        else "Point new contributions at what's under target first: no tax to pay."),
                       "Then switch from what's over target, starting with units held over a year"
                       + (" and inside the ₹1.25 lakh tax-free limit." if india else "."),
                       "For debt, a short-duration, corporate-bond or gilt index fund keeps it simple." if india
                       else "For bonds, a broad, low-cost bond index fund keeps it simple.",
                       "Re-check once a year; anything within ±5 points of target is fine."])

    # Diversifiers (skipped when the target mix already covers it)
    if not mix_action and (len(holdings) == 1 or (len(corr) > 1 and ((corr.sum() - 1) / (len(corr) - 1)).min() > 0.6)):
        add("Medium", "Add a cushion for market falls",
            "Everything here falls together"
            + (f": {abs(f.crisis_port)*100:.0f}% average loss in past crises." if f.crisis_port is not None else "."),
            f"Put 15–30% in {DIVERSIFIERS.get(cur, DIVERSIFIERS['$'])}.",
            when="Next 3–6 months",
            steps=["Start by sending new money to the cushion: it changes the mix without selling.",
                   "Debt and gold have tended to hold up or rise when shares fall; that's what makes them a cushion.",
                   "Check the Crises tab after: your average loss in past crises should be noticeably smaller.",
                   "Try the change first in Try ideas → Change my portfolio."])

    # Concentration
    # A broad, low-cost index fund is meant to be the core and to carry most of the equity risk
    broad_core = is_broad_index(f.top_risk_name) and f.top_risk_share < 0.7
    if len(holdings) > 1 and f.top_risk_share > 0.5 and not broad_core:
        add("Medium", f"Trim {f.top_risk_name}",
            f"{f.top_risk_weight*100:.0f}% of the money but {f.top_risk_share*100:.0f}% of the risk.",
            "Shift a slice into calmer holdings until its risk share is closer to its weight.",
            when="Next 3–6 months",
            steps=[f"A target: bring {f.top_risk_name} down so it's under 40% of the risk (see Details → Risk).",
                   "Sell in 2–3 steps rather than all at once, so you don't time it badly.",
                   "Use the money for what's under target in your mix, or for debt if you don't have a target."])
    if "kind" in holdings:
        big = holdings[(holdings["kind"] == "stock") & (holdings["weight"] > 0.10)]
        if len(big):
            add("Medium", f"Cap single shares at about 10% each ({len(big)} over)",
                f"{', '.join(big.index[:3])}: one bad result or scandal can wipe out years of gains.",
                "Keep any one company under 10%; move the excess into a broad index fund.",
                when="Next 3–6 months",
                steps=[f"Over 10%: " + ", ".join(f"{n} ({h['weight']*100:.0f}%)" for n, h in big.iterrows()) + ".",
                       "Trim the excess in 2–3 steps, using long-term gains first to keep tax low.",
                       "A " + ("Nifty 50 or Nifty 500" if india else "total-market") + " index fund gives the same "
                       "exposure spread over hundreds of companies."])

    # Performance
    if f.bench and f.bench_ret is not None and f.ret < f.bench_ret - 0.01 and f.years >= 3:
        add("High", "Make a low-cost index fund your core",
            f"{f.ret*100:.1f}% a year vs {f.bench_ret*100:.1f}% for {f.bench}: {m(f.bench_final_value - f.final_value)} behind.",
            f"Hold 50–70% in a {f.bench} index fund; keep active funds only if they've clearly earned it.",
            when="Next 3–6 months", impact=f"{m(f.bench_final_value - f.final_value)} gap so far",
            steps=["Pick a direct-plan index fund with a low expense ratio and small tracking error." if india
                   else "Pick a broad index fund or ETF with a very low expense ratio.",
                   "Send all new money there first.",
                   "Keep an active fund only if it beat its index over 5 years after costs.",
                   "Move the rest over 3–6 months, tax-efficiently."])
    elif f.sharpe < 0.3 and f.years >= 3:
        add("Medium", "Get paid better for the risk",
            f"Sharpe {f.sharpe:.2f}: little reward for the swings.",
            "Cut costs and add diversifiers: the two most reliable fixes.",
            when="Ongoing habits",
            steps=["Costs: direct plans and index funds where active funds haven't earned their fee.",
                   "Diversifiers: holdings that don't move with your biggest one (see Details → Diversification)."])

    # Horizon
    safe_after = next((y for y, mult in sorted(bad_case.items()) if mult >= 1), None)
    if safe_after is None:
        add("High", "Only long-term money belongs here",
            f"Even after 10 years the bad case is below {m(f.value)}.",
            "Keep emergency and near-term money in a liquid fund or savings account.",
            when="This week",
            steps=["List anything you'll need in the next 5 years: a house deposit, fees, a wedding, a car.",
                   "Keep that money out of this portfolio, in " + ("a liquid, short-duration or FD-like option." if india
                                                                   else "savings, money-market funds or short-term bonds."),
                   "See the Future tab for the bad, middle and good cases."])
    elif safe_after >= 2:
        add("High" if safe_after >= 4 else "Medium", f"Keep money needed within {safe_after} years out",
            f"In a bad case you'd be below your starting value for {safe_after - 1}+ year{'s' if safe_after > 2 else ''}.",
            "Hold 6–12 months of expenses and near-term goals in safe assets first.",
            when="This week",
            steps=[f"List goals due within {safe_after} years and their amounts.",
                   "Move that money to safe assets gradually as each goal gets within 2–3 years.",
                   "See the Future tab for how long a bad stretch could last."])

    # Market conditions
    if market:
        name = market["name"]
        if market["drawdown"] in ("Correction", "Bear market"):
            add("Medium", "Keep SIPs running through the dip" if india else "Keep investing through the dip",
                f"{name} is {abs(market['from_high'])*100:.0f}% below its high: regular investing now buys cheaper.",
                ("Don't pause SIPs" if india else "Don't pause regular investments") + "; add any lump sum in 3–6 monthly instalments.",
                when="This week",
                steps=["Falls are when SIPs buy the most units; pausing locks in the worst of both.",
                       "If you have spare cash, split it into 3–6 monthly parts" + (" (an STP from a liquid fund automates this)." if india else "."),
                       "Don't check the portfolio daily; review it once a quarter."])
        elif market["drawdown"] == "Near its high" and market["y1"] > 0.15:
            add("Medium", "Rebalance into strength",
                f"{name} is near its high after a {market['y1']*100:.0f}% year.",
                "Book gains back to your target mix; don't add risk at the top.",
                when="This month",
                steps=["Compare your mix with the target (Target mix tab).",
                       "If shares are 5+ points over, move the excess to debt, using tax-free gains first."])
        if market["volatility"] == "Elevated":
            add("Medium", "Stagger new money",
                f"Market volatility is {market['vix_ratio']:.1f}× its usual level.",
                "Invest lump sums over 3–6 months" + (" (an STP from a liquid fund does this automatically)." if india else "."),
                when="Next 3–6 months",
                steps=["Park the lump sum in a liquid fund." if india else "Park the lump sum in a money-market fund.",
                       "Move an equal amount into your chosen funds every week or month."])
    elif f.vol_now is not None and f.vol_now > 1.3 * f.vol:
        add("Medium", "Avoid sudden moves",
            f"Your portfolio's swings are {f.vol_now*100:.0f}% a year vs {f.vol*100:.0f}% usual.",
            "Make any changes in 2–4 steps over a few months.",
            when="Next 3–6 months",
            steps=["Split each change into 2–4 parts a few weeks apart.", "Avoid acting on a single day's news."])

    # Habits
    if profile and (profile.get("monthly") or 0) > 0:
        monthly, yrs = float(profile["monthly"]), HORIZON_YEARS.get(profile.get("horizon"), 7)
        tgt = target_mix(profile.get("horizon", "5–10 years"), profile.get("comfort", "Balanced"), cur)
        rate = sum(tgt[c] * EXPECTED[c] for c in CLASSES)
        flat, stepped = sip_future(monthly, yrs, rate), sip_future(monthly, yrs, rate, 0.10)
        add("Low", f"Raise your {'SIP' if india else 'monthly investment'} 10% every year",
            f"At about {rate*100:.0f}% a year, {m(monthly)} a month grows to {m(flat)} in {yrs} years; with yearly 10% step-ups, {m(stepped)}.",
            "Set an automatic yearly step-up" + (" (most platforms offer 'SIP top-up')." if india else "."),
            when="Ongoing habits", impact=f"≈ {m(stepped - flat)} more in {yrs} years",
            steps=["Time it with your salary increase so you don't feel it.",
                   "Send the extra to whatever is under target in your mix.",
                   "Returns are an assumption, not a promise: the point is the habit."])
    add("Low", "Rebalance once a year",
        "Your mix drifts towards whatever has risen, quietly adding risk.",
        "Rebalance yearly or at a 5-point drift, ideally using new " + ("SIPs." if india else "contributions."),
        when="Ongoing habits",
        steps=["Pick a fixed date each year: " + ("April, when the tax year starts." if india else "January, or your birthday."),
               "Compare your mix with the target; act only if a class is 5+ points off.",
               "Fix it with new money first, then with tax-efficient switches."])
    if f.years < 3:
        add("Low", "Judge funds on 5+ years",
            f"Only {f.years:.1f} years of shared history here.",
            "Check each fund's 5–10 year record and costs before replacing it.",
            when="Ongoing habits",
            steps=["Short records flatter or punish whatever style was in fashion.",
                   "Use category ranks over 3 and 5 years (My funds) before replacing anything."])

    order = {"High": 0, "Medium": 1, "Low": 2}
    actions.sort(key=lambda a: order[a["priority"]])
    return actions


WHEN_ORDER = ["This week", "This month", "Next 3–6 months", "Ongoing habits"]


def holding_review(holdings, corr, years=5.0, classes=None):
    """
    One verdict per holding: Switch, Merge, Review, Watch or Keep, with the reasons.
    holdings: the same table as action_plan(); classes: {name: asset_class(...)}. Returns a DataFrame.
    Debt and gold are judged as cushions, not on return.
    """
    pairs = _pairs(corr)
    classes = classes or {}
    rows = []
    for name, h in holdings.iterrows():
        verdict, why = None, []
        cushion = set(classes.get(name, {})) <= {"debt", "gold"} and bool(classes.get(name))
        if h.get("regular") == True:  # noqa: E712
            verdict = "Switch"
            gap = h.get("gap")
            why.append("regular plan: the direct plan of the same fund costs "
                       + (f"{gap*100:.2f}% a year less" if gap is not None and pd.notna(gap) else "about 1% a year less"))
        twin = None
        for (a, b), v in pairs.items():
            if v > 0.9 and name in (a, b):
                other = b if a == name else a
                if holdings.loc[other, "sharpe"] > h["sharpe"]:
                    twin = (other, v)
                    break
        if twin:
            verdict = verdict or "Merge"
            why.append(f"moves almost exactly like {twin[0]} ({twin[1]:.2f}), which earned more for its risk")
        peer = h.get("peer")
        if peer == "Bottom quartile":
            verdict = verdict or "Review"
            why.append(f"bottom quarter of its category ({h.get('peer_rank') or 'returns and risk'})")
        if years >= 3 and pd.notna(h["sharpe"]) and h["sharpe"] < 0 and not cushion:
            verdict = verdict or "Review"
            why.append("hasn't beaten the safe rate over this period")
        if h["weight"] >= 0.1 and h["risk_share"] > 1.6 * h["weight"] and not is_broad_index(name):
            verdict = verdict or "Watch"
            why.append(f"{h['weight']*100:.0f}% of the money but {h['risk_share']*100:.0f}% of the risk")
        if h.get("kind") == "stock" and h["weight"] > 0.10:
            verdict = verdict or "Watch"
            why.append(f"a single company at {h['weight']*100:.0f}% of the portfolio")
        if verdict is None:
            verdict = "Keep"
            if is_broad_index(name):
                why.append("low-cost index core")
            elif cushion and peer not in ("Top quartile", "Above average"):
                why.append("a cushion: judge it on how it holds up when shares fall, not on its return")
            elif peer == "Top quartile":
                why.append("top quarter of its category")
            elif peer:
                why.append(f"{str(peer).lower()} in its category")
            else:
                why.append("no problems found")
        elif peer == "Top quartile":
            why.append("top quarter of its category")
        rows.append({"Holding": name, "Verdict": verdict, "Money": h["weight"], "Risk": h["risk_share"],
                     "Return / yr": h["ret"], "Sharpe": h["sharpe"], "Why": "; ".join(why).capitalize() + "."})
    order = {"Switch": 0, "Merge": 1, "Review": 2, "Watch": 3, "Keep": 4}
    return pd.DataFrame(rows).sort_values(["Verdict", "Money"], key=lambda s: s.map(order) if s.name == "Verdict" else -s,
                                          kind="stable").reset_index(drop=True)


def portfolio_brief(f, holdings, corr, verdict, actions, future, cur, as_of, market=None, extra=None, profile=None,
                    classes=None):
    """The portfolio as plain text, for the chat model. No personal details, only holdings and statistics."""
    m = f.money
    pct = lambda x: "n/a" if x is None or pd.isna(x) else f"{x*100:.1f}%"  # noqa: E731
    lines = [
        f"Analysis date: {as_of:%d %b %Y}. Currency: {cur}. Portfolio value used: {m(f.value)}.",
        f"History analysed: {f.years:.1f} years. Risk-free rate assumed: {pct(f.rf)}. Benchmark: {f.bench or 'none'}.",
        "",
        "PORTFOLIO VS BENCHMARK (annualised)",
        f"- Return {pct(f.ret)} vs {pct(f.bench_ret)}; volatility {pct(f.vol)} vs {pct(f.bench_vol)}",
        f"- Sharpe {f.sharpe:.2f} vs {f.bench_sharpe if f.bench_sharpe is None else round(f.bench_sharpe, 2)}; "
        f"max drawdown {pct(f.max_dd)} vs {pct(f.bench_max_dd)}; recovery: {f.recovery}",
        f"- {m(f.value)} grew to {m(f.final_value)}"
        + (f" vs {m(f.bench_final_value)} in the benchmark" if f.bench_final_value else ""),
        f"- Beta {f.beta if f.beta is None else round(f.beta, 2)}, alpha {pct(f.alpha)} a year "
        f"(t-stat {f.alpha_t if f.alpha_t is None else round(f.alpha_t, 2)}), down-capture {pct(f.down_capture)}",
        f"- Average loss across {f.n_crises} historical crises covered: {pct(f.crisis_port)} vs {pct(f.crisis_bench)}",
        f"- Current volatility (EWMA) {pct(f.vol_now)} vs long-run {pct(f.vol)}",
        "",
        "HOLDINGS (name | % of money | % of risk | return/yr | vol/yr | Sharpe | plan)",
    ]
    for name, h in holdings.iterrows():
        plan = {True: "regular plan", False: "direct plan"}.get(h["regular"], "-")
        lines.append(f"- {name} | {h['weight']*100:.1f}% | {h['risk_share']*100:.1f}% | {pct(h['ret'])} | "
                     f"{pct(h['vol'])} | {h['sharpe']:.2f} | {plan}"
                     + (f" | category: {h['peer']}, {h['peer_rank']}" if "peer" in h and h["peer"] else "")
                     + (f" | direct plan grows {h['gap']*100:.2f}%/yr faster" if "gap" in h and pd.notna(h["gap"]) else ""))
    extra = extra or {}
    if extra.get("xirr") is not None:
        lines += ["", f"ACTUAL INVESTOR RETURN (from transactions): XIRR {pct(extra['xirr'])}"
                  + (f"; the same cash flows into a Nifty 50 index fund: {pct(extra['bench_xirr'])}" if extra.get("bench_xirr") is not None else "")]
    if extra.get("tax_all") is not None:
        lines += [f"TAX (India estimate): selling everything today ≈ {m(extra['tax_all'])} tax; tax-free long-term "
                  f"gains still available this year ≈ {m(extra.get('harvest_gain', 0))}."]
    pairs = _pairs(corr)
    if len(pairs):
        lines += ["", f"CORRELATIONS: average {pairs.mean():.2f}; most alike "
                  + ", ".join(f"{a} & {b} {v:.2f}" for (a, b), v in pairs.head(3).items())
                  + "; least alike " + ", ".join(f"{a} & {b} {v:.2f}" for (a, b), v in pairs.tail(2).items())]
    lines += ["", "SIMULATED FUTURE (block bootstrap of past daily returns, mix unchanged; value of "
              f"{m(f.value)}): years -> bad case (10th pct) / middle / good case (90th pct) / chance of a loss"]
    for y, row in future.items():
        lines.append(f"- {y}y: {m(row['bad'])} / {m(row['mid'])} / {m(row['good'])} / {row['loss']*100:.0f}%")
    if market:
        lines += ["", f"MARKET CONDITIONS TODAY ({market['name']}): mood {market['mood']}; trend {market['trend']}; "
                  f"{market['drawdown'].lower()} ({market['from_high']*100:.1f}% from 52-week high); 1-year "
                  f"{market['y1']*100:.1f}%; volatility {market['volatility'].lower()}."] + [f"- {n}" for n in market["notes"]]
    if profile:
        lines += ["", f"INVESTOR PROFILE (self-reported): goal horizon {profile.get('horizon')}; risk comfort "
                  f"{profile.get('comfort')}; emergency fund: {profile.get('emergency')}; monthly investing: "
                  + (m(profile['monthly']) if profile.get("monthly") else "not given")]
        if classes:
            labels = class_labels(cur)
            now = current_mix(holdings["weight"].to_dict(), classes)
            tgt = target_mix(profile.get("horizon", "5–10 years"), profile.get("comfort", "Balanced"), cur)
            lines.append("ASSET MIX now vs rule-of-thumb target: " + "; ".join(
                f"{labels[c]} {now[c]*100:.0f}% vs {tgt[c]*100:.0f}%" for c in CLASSES))
    lines += ["", f"DASHBOARD VERDICT: {verdict['grade']}. " + verdict["summary"]]
    lines += [f"- [{p['status']}] {p['title']}: {p['text']}" for p in verdict["points"]]
    lines += ["", "RULE-BASED ACTION PLAN SHOWN TO THE USER"]
    lines += [f"- ({a['priority']}) {a['title']}. Why: {a['why']} How: {a['how']}" for a in actions]
    return "\n".join(lines)


SYSTEM_PROMPT = """You are a senior portfolio adviser with twenty years in investment banking and private wealth \
management. The user is a private investor looking at their own portfolio in a risk dashboard. Below is everything \
the dashboard knows about it.

How to advise:
- Be direct and specific, like a trusted senior banker talking to a client: say what you would do, in what order, \
and why. Name the holdings. Put numbers in the user's currency ({cur}).
- Ground every claim in the data below. You don't have live prices, expense ratios, fund fact sheets, news or the \
user's tax position: when a recommendation depends on them, say what to check and where.
- Weigh trade-offs honestly (taxes and exit loads on switching, concentration, cost, behaviour in a crash). Past \
returns are a weak guide to the future; risk, cost and diversification are more dependable.
- When the right answer depends on the user's goals, time horizon, income stability, emergency fund or tax bracket, \
ask one short question, or give the answer for the most likely cases.
- Be crisp: lead with the answer in one line, then at most 3–5 bullets. Under about 150 words unless the user asks for depth. No filler, no generic lectures.
- Factor in today's market conditions (below) when timing matters: lump sums vs instalments, rebalancing, whether to keep SIPs going.
- You are not a registered investment adviser{reg} and this is general guidance based on the data shown. Mention \
that once, briefly, in your first answer, not in every reply. Never promise returns.

Portfolio data:
<portfolio>
{brief}
</portfolio>"""


def system_prompt(brief, cur):
    reg = " (SEBI-registered)" if cur == "₹" else ""
    return SYSTEM_PROMPT.format(brief=brief, cur=cur, reg=reg)


def stream_reply(client, system, history, effort="medium"):
    """
    Streams Claude's answer as text chunks. history: list of {"role", "content"} messages.
    Uses server-side fallbacks so a declined request is retried on another model automatically.
    """
    with client.beta.messages.stream(
        model=MODEL,
        max_tokens=16000,
        system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
        messages=history,
        output_config={"effort": effort},
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
    ) as stream:
        for text in stream.text_stream:
            yield text
        final = stream.get_final_message()
    if final.stop_reason == "refusal":
        yield "\n\n_I can't help with that request. Try asking about your portfolio in a different way._"
    elif final.stop_reason == "max_tokens":
        yield "\n\n_(Answer cut short. Ask me to continue.)_"


def offline_reply(question, f, holdings, actions, verdict, future, market=None):
    """
    Answers common questions without an AI model, by picking the relevant parts of the analysis.
    Used when no Anthropic API key is set.
    """
    q = question.lower()
    m = f.money

    def has(*words):
        return any(w in q for w in words)

    def fmt(a):
        return f"**{a['title']}** ({a['priority']} priority)\n\n{a['why']}\n\n*What to do:* {a['how']}"

    def find(*words):
        return [a for a in actions if any(w in a["title"].lower() for w in words)]

    if has("regular", "direct", "fee", "cost", "commission", "expense"):
        hits = find("direct")
        body = fmt(hits[0]) if hits else "All your funds are already direct plans (or aren't mutual funds), so there's no commission to save there."
    elif has("cut", "sell", "consolidat", "too many", "how many", "overlap", "which fund", "best fund", "worst fund", "keep"):
        ranked = holdings.sort_values("sharpe", ascending=False)
        table = "\n".join(f"{i}. {n}: {h['ret']*100:.1f}% a year, Sharpe {h['sharpe']:.2f}"
                          for i, (n, h) in enumerate(ranked.iterrows(), 1))
        hits = find("look-alike")
        body = (fmt(hits[0]) + "\n\n" if hits else "Your holdings don't overlap much, so there's no strong case for cutting any.\n\n") \
            + f"Ranked by reward for risk:\n\n{table}"
    elif has("safe", "risk", "crash", "lose", "loss", "fall", "protect", "volatil"):
        body = (f"At its worst this portfolio fell **{abs(f.max_dd)*100:.0f}%** ({m(-f.value * f.max_dd)} on {m(f.value)}); "
                f"recovery took {f.recovery}.\n\n")
        body += "\n\n".join(fmt(a) for a in find("cushion", "look-alike", "trim", "stagger", "sudden")) or \
            "It is already reasonably diversified; the main lever left is holding less in shares and more in bonds or cash."
    elif has("year", "goal", "future", "long", "retire", "horizon", "grow"):
        rows = "\n".join(f"- **{y} year{'s' if y > 1 else ''}:** bad case {m(r['bad'])}, middle {m(r['mid'])}, "
                         f"good case {m(r['good'])} ({r['loss']*100:.0f}% chance of a loss)" for y, r in future.items())
        hits = find("money needed", "long-term money")
        body = f"If nothing changes and the future looks like the past:\n\n{rows}" + ("\n\n" + fmt(hits[0]) if hits else "")
    elif has("market today", "market condition", "markets", "nifty", "sensex", "s&p", "today", "right now", "timing", "lump", "invest now"):
        if market:
            body = (f"**Market mood ({market['name']}): {market['mood']}.** " + " ".join(market["notes"]) + "\n\n"
                    + "\n".join(f"- {x}" for x in market.get("implications", [])))
        else:
            body = "Market data isn't available right now. Check the Markets tab."
    elif has("how did", "perform", "good", "bad", "doing", "verdict", "market", "benchmark"):
        body = f"**Overall: {verdict['grade']}.** {verdict['summary']}\n\n" + "\n".join(
            f"- {'✅' if p['status'] == 'good' else '❌' if p['status'] == 'bad' else '⚠️'} {p['title']}"
            for p in verdict["points"] if p["scored"])
    else:
        body = "Here's what I'd do, in order:\n\n" + "\n\n".join(
            f"{i}. " + fmt(a) for i, a in enumerate(actions[:3], 1))
    return body
