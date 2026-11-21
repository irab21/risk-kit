"""
Plain-English verdict on a portfolio: how it did against the market, cash and common rules of thumb.

build_commentary() turns numbers into a grade and a list of points. Each point has a status
("good", "ok", "bad"), a short title and a sentence written about this portfolio, in money terms
where possible. The rules are simple, transparent thresholds (listed next to each point), so a
reader can see exactly why a point is marked good or bad.
"""
import re
from dataclasses import dataclass, field
from typing import Callable, Optional


def pct(x, d=1):
    return f"{x*100:.{d}f}%"


BROAD_INDEX = re.compile(r"nifty 50|nifty 100|sensex|s&p 500|total (stock )?market|world|msci|\bspy\b|\bvti\b|"
                         r"\bvt\b|index", re.IGNORECASE)


def is_broad_index(name):
    """A broad market index fund or ETF (meant to be a portfolio's core), judged from its name."""
    return bool(BROAD_INDEX.search(str(name or "")))


@dataclass
class PortfolioFacts:
    years: float
    value: float                      # starting amount used for money examples
    final_value: float                # what `value` grew to
    ret: float                        # annualized return
    vol: float                        # annualized volatility
    sharpe: float
    max_dd: float                     # negative, e.g. -0.29
    recovery: str                     # e.g. "about 5 months" or "it has not fully recovered yet"
    rf: float                         # risk-free rate
    n_holdings: int
    top_risk_name: str
    top_risk_share: float             # share of total risk from the biggest contributor
    top_risk_weight: float            # that holding's share of money
    avg_corr: Optional[float]         # average correlation between holdings (None for one holding)
    vol_now: Optional[float]          # current EWMA volatility
    bench: Optional[str] = None
    bench_ret: Optional[float] = None
    bench_vol: Optional[float] = None
    bench_sharpe: Optional[float] = None
    bench_max_dd: Optional[float] = None
    bench_final_value: Optional[float] = None
    beta: Optional[float] = None
    alpha: Optional[float] = None
    alpha_t: Optional[float] = None
    down_capture: Optional[float] = None
    crisis_port: Optional[float] = None    # average return across covered crises
    crisis_bench: Optional[float] = None
    n_crises: int = 0
    var_ok: Optional[bool] = None
    breach_rate: Optional[float] = None
    level: int = 5
    money: Callable[[float], str] = field(default=lambda x: f"{x:,.0f}")


def sharpe_label(s):
    if s < 0:
        return "negative: risk was not rewarded at all"
    if s < 0.3:
        return "poor"
    if s < 0.6:
        return "decent"
    if s < 1.0:
        return "good"
    return "excellent"


def build_commentary(f: PortfolioFacts):
    m = f.money
    pts = []

    def add(status, title, text, scored=True, key=""):
        pts.append({"status": status, "title": title, "text": text, "scored": scored, "key": key})

    has_b = f.bench is not None and f.bench_ret is not None

    # 1. Return against the market (good: ahead by 1 point a year or more; bad: behind by 1 point or more)
    if has_b:
        d = f.ret - f.bench_ret
        gap = f.final_value - f.bench_final_value
        if d >= 0.01:
            add("good", f"Beat the market by {d*100:.1f} points a year",
                f"Your portfolio grew {pct(f.ret)} a year against {pct(f.bench_ret)} for {f.bench}. "
                f"On {m(f.value)}, that's {m(gap)} more than simply holding the market.")
        elif d <= -0.01:
            add("bad", f"Trailed the market by {-d*100:.1f} points a year",
                f"Your portfolio grew {pct(f.ret)} a year against {pct(f.bench_ret)} for {f.bench}. "
                f"On {m(f.value)}, you ended {m(-gap)} behind simply holding the market.")
        else:
            add("ok", "Kept pace with the market",
                f"{pct(f.ret)} a year against {pct(f.bench_ret)} for {f.bench}: within one point a year.")

    # 2. Return against a risk-free deposit (bad: did not beat it; good: beat it by 2+ points a year)
    excess = f.ret - f.rf
    if excess <= 0:
        add("bad", "Did not beat a risk-free deposit",
            f"A riskless deposit at {pct(f.rf)} would have earned more than this portfolio's {pct(f.ret)} a year, "
            "without any of the ups and downs.")
    elif excess < 0.02:
        add("ok", "Only just beat a risk-free deposit",
            f"{pct(f.ret)} a year against {pct(f.rf)} risk-free: an extra {pct(excess)} a year for taking market risk.")
    else:
        add("good", "Comfortably beat a risk-free deposit",
            f"{pct(f.ret)} a year against {pct(f.rf)} risk-free: {pct(excess)} a year extra for taking market risk.")

    # 3. Bumpiness against the market (good: 15%+ calmer; bad: 15%+ bumpier)
    swing_lo, swing_hi = f.value * (f.ret - f.vol), f.value * (f.ret + f.vol)
    typical = (f"In a typical year (2 years out of 3), {m(f.value)} would end somewhere between "
               f"{'a loss of ' + m(-swing_lo) if swing_lo < 0 else 'a gain of ' + m(swing_lo)} and a gain of {m(swing_hi)}.")
    if has_b and f.bench_vol:
        ratio = f.vol / f.bench_vol
        if ratio < 0.85:
            add("good", f"A smoother ride than the market ({(1-ratio)*100:.0f}% less bumpy)",
                f"Volatility of {pct(f.vol)} against {pct(f.bench_vol)} for {f.bench}. {typical}")
        elif ratio > 1.15:
            add("bad", f"A bumpier ride than the market ({(ratio-1)*100:.0f}% more)",
                f"Volatility of {pct(f.vol)} against {pct(f.bench_vol)} for {f.bench}. {typical}")
        else:
            add("ok", "About as bumpy as the market",
                f"Volatility of {pct(f.vol)} against {pct(f.bench_vol)} for {f.bench}. {typical}")
    else:
        add("ok", f"Volatility of {pct(f.vol)} a year", typical)

    # 4. Reward for risk (Sharpe; good: 0.5+ and ahead of the market; bad: below 0.2 or clearly behind)
    bs = f.bench_sharpe if has_b else None
    if f.sharpe >= 0.5 and (bs is None or f.sharpe >= bs + 0.1):
        status = "good"
    elif f.sharpe < 0.2 or (bs is not None and f.sharpe < bs - 0.15):
        status = "bad"
    else:
        status = "ok"
    comp = f" ({f.bench}: {bs:.2f})" if bs is not None else ""
    add(status, f"Reward for the risk taken: {sharpe_label(f.sharpe).split(':')[0]}",
        f"Sharpe ratio {f.sharpe:.2f}{comp}: for every unit of ups and downs, the portfolio earned {f.sharpe:.2f} "
        "units of return above cash. Rule of thumb: below 0.3 poor, 0.3–0.6 decent, 0.6–1 good, above 1 excellent.")

    # 5. Worst fall (good: at most 3/4 of the market's fall; bad: deeper than the market or than 40%)
    bottom = f.value * (1 + f.max_dd)
    text = (f"At its worst, {m(f.value)} would have dropped to {m(bottom)} ({pct(f.max_dd)})"
            + (f", against {pct(f.bench_max_dd)} for {f.bench}" if has_b and f.bench_max_dd is not None else "")
            + f". Recovery took {f.recovery}.")
    if has_b and f.bench_max_dd is not None and f.bench_max_dd < 0:
        rel = f.max_dd / f.bench_max_dd
        status = "good" if rel <= 0.75 else "bad" if rel > 1.1 or f.max_dd < -0.4 else "ok"
    else:
        status = "bad" if f.max_dd < -0.4 else "ok" if f.max_dd < -0.2 else "good"
    title = {"good": "Fell less than the market in its worst stretch",
             "bad": "Deep falls: needs strong nerves",
             "ok": "Worst fall similar to the market"}[status] if has_b else f"Worst fall: {pct(f.max_dd)}"
    add(status, title, text)

    # 6. Crises (good: lost at most 3/4 of what the market lost; bad: lost more)
    if f.n_crises and f.crisis_bench is not None and f.crisis_bench < 0:
        rel = f.crisis_port / f.crisis_bench
        status = "good" if rel <= 0.75 else "bad" if rel > 1.1 else "ok"
        add(status, {"good": "Held up better in crises", "bad": "Hit harder than the market in crises",
                     "ok": "Fell about as much as the market in crises"}[status],
            f"Across {f.n_crises} past {'crises' if f.n_crises > 1 else 'crisis'} in your data, it moved "
            f"{pct(f.crisis_port)} on average against {pct(f.crisis_bench)} for {f.bench}: "
            f"about {m(abs(f.crisis_port) * f.value)} on {m(f.value)}.")

    # 7. Diversification (bad: one holding, or one holding drives over half the risk, or holdings move in lockstep)
    if f.n_holdings == 1:
        add("bad", "No diversification", "Everything rides on a single holding.")
    else:
        status = "good"
        notes = [f"{f.top_risk_name} is {pct(f.top_risk_weight, 0)} of your money but {pct(f.top_risk_share, 0)} "
                 "of your risk."]
        if f.top_risk_share > 0.5 and is_broad_index(f.top_risk_name) and f.top_risk_share < 0.7:
            notes.append("That's expected: a broad index fund is the core of the portfolio, and it already "
                         "spreads its risk across many companies.")
        elif f.top_risk_share > 0.5:
            status = "bad"
            notes.append("When one holding drives most of the risk, the portfolio behaves a lot like that holding alone.")
        elif f.top_risk_share > 0.35:
            status = "ok"
        if f.avg_corr is not None:
            if f.avg_corr > 0.85:
                status = "bad"
                notes.append(f"Your holdings move almost in lockstep (average correlation {f.avg_corr:.2f}), "
                             "so owning several of them protects you very little.")
            elif f.avg_corr > 0.7:
                status = "ok" if status == "good" else status
                notes.append(f"Your holdings tend to move together (average correlation {f.avg_corr:.2f}).")
            else:
                notes.append(f"Your holdings often move differently (average correlation {f.avg_corr:.2f}), "
                             "which cushions falls.")
        add(status, {"good": "Well spread out", "ok": "Somewhat concentrated", "bad": "Concentrated risk"}[status],
            " ".join(notes))

    # 8. Market sensitivity and skill (not scored: describes behaviour)
    if has_b and f.beta is not None:
        status = "ok"
        text = (f"When {f.bench} has moved 10%, your portfolio has typically moved about {abs(f.beta)*10:.0f}% "
                "in the same direction")
        if f.down_capture is not None:
            text += f", and on the market's down days it took about {pct(f.down_capture, 0)} of the fall"
        text += ". "
        if f.alpha is not None and f.alpha_t is not None:
            if f.alpha > 0 and f.alpha_t >= 2:
                text += (f"It also earned about {pct(f.alpha)} a year beyond what that market exposure explains, "
                         "and the evidence for it is statistically strong.")
                status = "good"
            elif f.alpha < 0 and f.alpha_t <= -2:
                text += f"It lagged what its market exposure would predict by about {pct(-f.alpha)} a year (statistically strong)."
                status = "bad"
            else:
                text += ("Its results beyond that market exposure are within what luck could explain, "
                         "so there's no proven edge either way.")
                status = "ok"
        add(status, "Defensive: moves less than the market" if f.beta < 0.85 else
            "Aggressive: moves more than the market" if f.beta > 1.15 else "Moves roughly with the market",
            text, scored=False)

    # 9. Right now (not scored)
    if f.vol_now is not None and f.vol > 0:
        r = f.vol_now / f.vol
        day = f.value * 1.65 * f.vol_now / 252 ** 0.5
        if r > 1.3:
            add("bad", "Risk is higher than usual right now",
                f"Recent swings are running at {pct(f.vol_now)} a year against a usual {pct(f.vol)}. "
                f"A bad day (1 in 20) today could cost around {m(day)}.", scored=False)
        elif r < 0.8:
            add("good", "Markets are calmer than usual for this portfolio",
                f"Recent swings are running at {pct(f.vol_now)} a year against a usual {pct(f.vol)}. "
                f"A bad day (1 in 20) today would cost around {m(day)}. Calm spells don't last forever.", scored=False)
        else:
            add("ok", "Risk is around its normal level right now",
                f"Recent swings: {pct(f.vol_now)} a year, close to the usual {pct(f.vol)}.", scored=False)

    # 10. Reliability of the daily loss estimate (not scored)
    if f.var_ok is False:
        add("ok", "Daily loss estimates have been too optimistic",
            f"Losses beyond the 1-day VaR happened on {pct(f.breach_rate, 1)} of days instead of {f.level}%. "
            "Treat the 'bad day' figures in this report as a minimum, not a typical bad day.", scored=False)

    if f.years < 3:
        add("ok", "Short track record",
            f"These results cover only {f.years:.1f} years, which may not include a full market cycle. "
            "A good or bad couple of years can make any portfolio look better or worse than it really is.",
            scored=False)

    priority = ["market", "Reward", "fall", "crises", "ride", "bumpy", "spread", "Concentrated", "diversification",
                "deposit"]
    def rank(p):
        return next((i for i, k in enumerate(priority) if k.lower() in p["title"].lower()), len(priority))
    scored = sorted([p for p in pts if p["scored"]], key=rank)
    score = sum(p["status"] == "good" for p in scored) - sum(p["status"] == "bad" for p in scored)
    grade = "Strong" if score >= 3 else "Solid" if score >= 1 else "Mixed" if score >= -1 else "Weak"

    summary = (f"Over {f.years:.1f} years, {m(f.value)} in this portfolio grew to {m(f.final_value)}, "
               f"or {pct(f.ret)} a year")
    if has_b:
        summary += f", while {f.bench} returned {pct(f.bench_ret)} a year"
    summary += "."
    best = next((p["title"] for p in scored if p["status"] == "good"), None)
    worst = next((p["title"] for p in scored if p["status"] == "bad"), None)
    if best:
        summary += f" Biggest strength: {best[0].lower() + best[1:]}."
    if worst:
        summary += f" Main weakness: {worst[0].lower() + worst[1:]}."
    return {"grade": grade, "score": score, "summary": summary, "points": pts}
