"""
Your actual investing history: CAS import, XIRR, tax lots and Indian capital-gains estimates.

- read_cas(): reads a CAMS / KFintech Consolidated Account Statement (CAS) PDF with `casparser`
  and returns current holdings plus every transaction. Personal details (name, PAN, email,
  address) are never kept.
- xirr(): the money-weighted return of real cash flows (what you actually earned, given when you
  put money in and took it out).
- build_lots(): first-in-first-out tax lots per fund, carrying original dates through fund mergers.
- estimate_sale_tax(): capital-gains tax if units were sold today, under the rules for sales on or
  after 23 July 2024 (equity: 20% short term, 12.5% long term above the yearly exemption of
  ₹1.25 lakh; debt funds bought from April 2023: your income-tax slab; plus 4% cess).

Estimates only: surcharge, set-off of losses carried forward and special cases are left out.
"""
import io
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd
from scipy.optimize import brentq

BUY = {"PURCHASE", "PURCHASE_SIP", "SWITCH_IN", "DIVIDEND_REINVEST", "GIFT_IN"}
SELL = {"REDEMPTION", "SWITCH_OUT"}
TRANSFER_OUT = {"GIFT_OUT"}
MERGER_OUT, MERGER_IN = {"SWITCH_OUT_MERGER"}, {"SWITCH_IN_MERGER"}
PAYOUT = {"DIVIDEND_PAYOUT"}

LTCG_EXEMPTION = 125_000
EQ_STCG, EQ_LTCG, OTHER_LTCG, CESS = 0.20, 0.125, 0.125, 0.04
NEW_REGIME_DATE = pd.Timestamp("2024-07-23")
SPECIFIED_MF_DATE = pd.Timestamp("2023-04-01")
GRANDFATHER_DATE = pd.Timestamp("2018-02-01")
FMV_DATE = pd.Timestamp("2018-01-31")


class CASError(ValueError):
    pass


# ---------------------------------------------------------------------------
# Reading statements
# ---------------------------------------------------------------------------
def looks_like_cas(first_page_text):
    t = (first_page_text or "").lower()
    return "consolidated account statement" in t or ("cams" in t and "folio" in t) or ("kfintech" in t and "folio" in t)


def _f(x):
    try:
        return float(x) if x is not None else np.nan
    except (TypeError, ValueError):
        return np.nan


def cas_to_frames(cas):
    """casparser dict output -> (holdings, transactions). Investor details are dropped."""
    hold, txns = [], []
    for folio in cas.get("folios", []):
        for s in folio.get("schemes", []):
            code = int(s["amfi"]) if str(s.get("amfi") or "").isdigit() else None
            name = s.get("scheme", "")
            val = s.get("valuation") or {}
            hold.append({"code": code, "scheme": name, "isin": s.get("isin"), "units": _f(s.get("close")),
                         "nav": _f(val.get("nav")), "value": _f(val.get("value")),
                         "cost": _f(val.get("cost")), "as_of": pd.to_datetime(val.get("date"))})
            for t in s.get("transactions", []):
                ttype = str(t.get("type", "")).upper().split(".")[-1]
                txns.append({"code": code, "scheme": name, "date": pd.to_datetime(t.get("date")), "type": ttype,
                             "units": abs(_f(t.get("units"))), "amount": abs(_f(t.get("amount"))),
                             "nav": _f(t.get("nav")), "signed_units": _f(t.get("units"))})
    holdings = pd.DataFrame(hold)
    if len(holdings):
        holdings = (holdings.groupby(["code", "scheme"], dropna=False, sort=False)
                    .agg(isin=("isin", "first"), units=("units", "sum"), nav=("nav", "last"), value=("value", "sum"),
                         cost=("cost", "sum"), as_of=("as_of", "max")).reset_index())
        holdings = holdings[holdings["units"] > 1e-6]
    tx = pd.DataFrame(txns, columns=["code", "scheme", "date", "type", "units", "amount", "nav", "signed_units"])
    return holdings, tx.sort_values("date", kind="stable").reset_index(drop=True)


def read_cas(data, password=""):
    """Parses a CAS PDF (bytes). Raises CASError with a readable message."""
    import casparser
    try:
        cas = casparser.read_cas_pdf(io.BytesIO(data), password or "", output="dict")
    except Exception as e:  # casparser raises its own exception types
        msg = str(e).lower()
        if "password" in msg or "decrypt" in msg:
            raise CASError("The password didn't open this statement. CAS files use your PAN (in capitals) by default.")
        raise CASError(f"This looks like a CAS statement but couldn't be read ({type(e).__name__}).")
    if str(cas.get("cas_type", "")).upper().endswith("SUMMARY"):
        raise CASError("This is a summary CAS without transactions. Request a 'Detailed' CAS from CAMS or KFintech.")
    return cas_to_frames(cas)


TEMPLATE_CSV = """Date,Fund,AMFI code,Type,Units,Amount
2021-04-05,SBI Aggressive Hybrid Fund - Regular - Growth,102885,Purchase,120.512,20000
2022-04-05,SBI Aggressive Hybrid Fund - Regular - Growth,102885,SIP,95.250,20000
2024-06-10,SBI Aggressive Hybrid Fund - Regular - Growth,102885,Redemption,50.000,13500
"""


def transactions_from_table(df):
    """
    Generic transaction file (CSV/Excel export from an app or a spreadsheet). Needs columns for date,
    fund, units and amount; optional type and AMFI code. Sells are recognised from the type text
    (redeem, sell, switch out, SWP) or from negative units.
    """
    cols = {c: str(c).lower() for c in df.columns}
    def find(*keys):
        return next((c for c, l in cols.items() if any(k in l for k in keys)), None)
    c_date, c_fund, c_units = find("date"), find("fund", "scheme", "name"), find("unit", "quantity", "qty")
    c_amt, c_type, c_code = find("amount", "value", "invest"), find("type", "transaction", "description"), find("amfi", "code")
    if not all([c_date, c_fund, c_units, c_amt]):
        raise ValueError("A transaction file needs date, fund, units and amount columns.")
    units = pd.to_numeric(df[c_units].astype(str).str.replace(",", ""), errors="coerce")
    amount = pd.to_numeric(df[c_amt].astype(str).str.replace(r"[,₹$ ]", "", regex=True), errors="coerce")
    text = df[c_type].astype(str).str.lower() if c_type else pd.Series("", index=df.index)
    is_sell = text.str.contains("redeem|redemption|sell|switch out|switch-out|swp|withdraw") | (units < 0)
    is_payout = text.str.contains("payout") & ~text.str.contains("reinvest")
    ttype = np.where(is_payout, "DIVIDEND_PAYOUT", np.where(is_sell, "REDEMPTION", "PURCHASE"))
    code = pd.to_numeric(df[c_code], errors="coerce") if c_code else pd.Series(np.nan, index=df.index)
    tx = pd.DataFrame({"code": code.astype("Int64"), "scheme": df[c_fund].astype(str).str.strip(),
                       "date": pd.to_datetime(df[c_date], errors="coerce", dayfirst=True), "type": ttype,
                       "units": units.abs(), "amount": amount.abs(), "nav": amount.abs() / units.abs(),
                       "signed_units": np.where(is_sell, -units.abs(), units.abs())})
    tx = tx.dropna(subset=["date", "units"]).sort_values("date", kind="stable").reset_index(drop=True)
    return tx


def holdings_from_transactions(tx, nav_lookup):
    """Units held per fund from transactions, valued with nav_lookup(code) -> (nav, date)."""
    signed = np.where(tx["type"].isin(BUY | MERGER_IN), tx["units"],
                      np.where(tx["type"].isin(SELL | MERGER_OUT | TRANSFER_OUT), -tx["units"], 0.0))
    units = pd.Series(signed, index=tx.index).groupby([tx["code"], tx["scheme"]]).sum()
    rows = []
    for (code, scheme), u in units.items():
        if u > 1e-6:
            nav, as_of = nav_lookup(code)
            rows.append({"code": code, "scheme": scheme, "units": u, "nav": nav, "value": u * nav, "as_of": as_of})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Returns
# ---------------------------------------------------------------------------
def xirr(flows):
    """flows: iterable of (date, amount); money in is negative, money out (and today's value) positive.
    Returns the annual rate, or nan if it can't be solved."""
    f = [(pd.Timestamp(d), float(a)) for d, a in flows if a and not np.isnan(a)]
    if len(f) < 2 or not (any(a < 0 for _, a in f) and any(a > 0 for _, a in f)):
        return np.nan
    t0 = min(d for d, _ in f)
    yrs = np.array([(d - t0).days / 365.0 for d, _ in f])
    amt = np.array([a for _, a in f])
    def npv(r):
        return float(np.sum(amt / (1 + r) ** yrs))
    try:
        return brentq(npv, -0.9999, 100.0, maxiter=500)
    except ValueError:
        return np.nan


def cash_flows(tx, include_internal=False):
    """(date, amount) pairs from transactions: purchases negative, redemptions and payouts positive.
    Switches between funds and mergers are internal to the portfolio and dropped unless asked."""
    out = []
    for r in tx.itertuples():
        if r.type in {"PURCHASE", "PURCHASE_SIP", "GIFT_IN"} or (include_internal and r.type in {"SWITCH_IN"} | MERGER_IN):
            out.append((r.date, -(r.amount if r.amount == r.amount and r.amount else r.units * r.nav)))
        elif r.type in {"REDEMPTION", "DIVIDEND_PAYOUT", "GIFT_OUT"} or (include_internal and r.type in {"SWITCH_OUT"} | MERGER_OUT):
            out.append((r.date, r.amount if r.amount == r.amount and r.amount else r.units * r.nav))
    return out


def returns_summary(tx, holdings, as_of=None):
    """Per fund and in total: invested, withdrawn, value now, gain and XIRR."""
    rows = []
    for (code, scheme), g in tx.groupby(["code", "scheme"], dropna=False, sort=False):
        h = holdings[(holdings["scheme"] == scheme)]
        val = float(h["value"].sum()) if len(h) else 0.0
        when = h["as_of"].max() if len(h) and pd.notna(h["as_of"].max()) else (as_of or pd.Timestamp.today().normalize())
        flows = cash_flows(g, include_internal=True)
        invested = -sum(a for _, a in flows if a < 0)
        withdrawn = sum(a for _, a in flows if a > 0)
        rows.append({"code": code, "scheme": scheme, "invested": invested, "withdrawn": withdrawn, "value": val,
                     "gain": val + withdrawn - invested, "xirr": xirr(flows + [(when, val)]) if val or withdrawn else np.nan,
                     "first": g["date"].min()})
    per_fund = pd.DataFrame(rows)
    when = holdings["as_of"].max() if len(holdings) and "as_of" in holdings and pd.notna(holdings["as_of"].max()) \
        else (as_of or pd.Timestamp.today().normalize())
    flows = cash_flows(tx)
    total_val = float(holdings["value"].sum()) if len(holdings) else 0.0
    invested = -sum(a for _, a in flows if a < 0)
    withdrawn = sum(a for _, a in flows if a > 0)
    total = {"invested": invested, "withdrawn": withdrawn, "value": total_val,
             "gain": total_val + withdrawn - invested, "xirr": xirr(flows + [(when, total_val)]),
             "first": tx["date"].min(), "as_of": when}
    return per_fund, total


def same_flows_into(tx, nav, as_of=None):
    """
    'What if every rupee had gone into this instead?': replays your external cash flows (purchases,
    redemptions) into another fund or index with NAV series `nav`. Returns (value now, xirr).
    """
    nav = nav.dropna().sort_index()
    units = 0.0
    flows = cash_flows(tx)
    for d, a in sorted(flows):
        px = nav[nav.index <= d]
        if px.empty:
            px = nav.iloc[:1]
        units = max(units - a / px.iloc[-1], 0.0)  # a < 0 buys, a > 0 sells
    when = as_of or nav.index[-1]
    value = units * nav[nav.index <= when].iloc[-1]
    return value, xirr(flows + [(when, value)])


# ---------------------------------------------------------------------------
# Tax lots
# ---------------------------------------------------------------------------
@dataclass
class Lot:
    date: pd.Timestamp
    units: float
    cost: float  # total rupees paid for these units


@dataclass
class Realised:
    code: object
    scheme: str
    buy_date: pd.Timestamp
    sale_date: pd.Timestamp
    units: float
    cost: float
    proceeds: float


def _consume(lots, units):
    """Removes `units` first-in-first-out; returns the removed pieces as lots."""
    taken = []
    while units > 1e-9 and lots:
        lot = lots[0]
        q = min(units, lot.units)
        frac = q / lot.units
        taken.append(Lot(lot.date, q, lot.cost * frac))
        lot.units -= q
        lot.cost -= lot.cost * frac
        units -= q
        if lot.units <= 1e-9:
            lots.pop(0)
    return taken


def build_lots(tx):
    """
    FIFO lots per fund. Returns (lots: {(code, scheme): [Lot]}, realised: [Realised]).
    Fund mergers keep the original purchase dates and cost (they are not taxable).
    """
    lots, realised, merged = {}, [], []
    for r in tx.sort_values("date", kind="stable").itertuples():
        key = (r.code, r.scheme)
        book = lots.setdefault(key, [])
        amount = r.amount if r.amount == r.amount and r.amount else r.units * (r.nav if r.nav == r.nav else 0)
        if r.type in BUY:
            book.append(Lot(r.date, r.units, amount))
        elif r.type in SELL or r.type in TRANSFER_OUT:
            pieces = _consume(book, r.units)
            if r.type in SELL:
                total_u = sum(p.units for p in pieces) or 1
                for p in pieces:
                    realised.append(Realised(r.code, r.scheme, p.date, r.date, p.units, p.cost, amount * p.units / total_u))
        elif r.type in MERGER_OUT:
            merged.append((r.date, _consume(book, r.units)))
        elif r.type in MERGER_IN:
            src = next((m for m in merged if abs((m[0] - r.date).days) <= 10), None)
            if src:
                merged.remove(src)
                old_units = sum(p.units for p in src[1]) or 1
                for p in src[1]:
                    book.append(Lot(p.date, r.units * p.units / old_units, p.cost))
            else:
                book.append(Lot(r.date, r.units, amount))
    return lots, realised


# ---------------------------------------------------------------------------
# Tax rules
# ---------------------------------------------------------------------------
EQUITY_WORDS = ("equity scheme", "elss", "aggressive hybrid", "arbitrage", "balanced advantage",
                "dynamic asset allocation", "equity savings")
DEBT_WORDS = ("debt scheme", "liquid", "overnight", "money market", "gilt", "conservative hybrid", "income")
NON_EQUITY_NAME = ("gold", "silver", "gilt", "g-sec", "gsec", "sdl", "bond", "debt", "crisil", "liquid", "nasdaq",
                   "s&p", "international", "global", "overseas", "us ", "hang seng", "nyse", "msci", "fang", "china")


def tax_class(category, name=""):
    """'equity', 'debt' or 'other' for capital-gains purposes (best guess from the SEBI category)."""
    c, n = (category or "").lower(), (name or "").lower()
    if any(w in c for w in EQUITY_WORDS):
        return "equity"
    if any(w in c for w in DEBT_WORDS) or any(w in n for w in ("gilt", "g-sec", "sdl", "bond", "debt", "liquid")):
        return "debt"
    if ("index" in c or "etf" in c) and not any(w in n for w in NON_EQUITY_NAME):
        return "equity"
    return "other"


def financial_year(d):
    d = pd.Timestamp(d)
    return d.year if d.month >= 4 else d.year - 1


@dataclass
class TaxSettings:
    slab: float = 0.30               # your income-tax slab rate
    ltcg_used: float = 0.0           # equity long-term gains already booked this financial year elsewhere
    classes: dict = field(default_factory=dict)  # (code, scheme) -> "equity" / "debt" / "other"


def classify_piece(klass, buy_date, sale_date):
    """('short' | 'long', rate or 'slab') for one lot."""
    held = (sale_date - buy_date).days
    if klass == "equity":
        return ("long", EQ_LTCG) if held > 365 else ("short", EQ_STCG)
    if klass == "debt":
        if buy_date >= SPECIFIED_MF_DATE:
            return ("short", "slab")
        return ("long", OTHER_LTCG) if held > 730 else ("short", "slab")
    return ("long", OTHER_LTCG) if held > 730 else ("short", "slab")


def gains_on_sale(lots, units, nav, sale_date, klass, fmv_nav=None):
    """Gain pieces if `units` were sold at `nav` today (lots are not modified)."""
    copy = [Lot(l.date, l.units, l.cost) for l in lots]
    pieces = _consume(copy, units)
    out = []
    for p in pieces:
        proceeds = p.units * nav
        cost = p.cost
        if klass == "equity" and p.date < GRANDFATHER_DATE and fmv_nav:
            cost = max(cost, min(p.units * fmv_nav, proceeds))  # grandfathering of gains up to 31 Jan 2018
        term, rate = classify_piece(klass, p.date, sale_date)
        out.append({"buy_date": p.date, "units": p.units, "cost": cost, "proceeds": proceeds, "gain": proceeds - cost,
                    "term": term, "rate": rate, "days": (sale_date - p.date).days})
    return out


def tax_on_pieces(pieces, settings, sale_date=None, realised_eq_ltcg_this_fy=0.0):
    """
    Total estimated tax for gain pieces (from several funds), with the equity LTCG exemption shared.
    Returns dict: eq_st, eq_lt, other_st (slab), other_lt, taxable_eq_lt, tax (incl. 4% cess),
    exemption_left.
    """
    eq_st = sum(p["gain"] for p in pieces if p["klass"] == "equity" and p["term"] == "short")
    eq_lt = sum(p["gain"] for p in pieces if p["klass"] == "equity" and p["term"] == "long")
    slab = sum(p["gain"] for p in pieces if p["rate"] == "slab")
    oth_lt = sum(p["gain"] for p in pieces if p["klass"] != "equity" and p["term"] == "long")
    exemption_left = max(LTCG_EXEMPTION - settings.ltcg_used - realised_eq_ltcg_this_fy, 0.0)
    taxable_eq_lt = max(eq_lt - exemption_left, 0.0)
    tax = (EQ_STCG * max(eq_st, 0) + EQ_LTCG * taxable_eq_lt + settings.slab * max(slab, 0)
           + OTHER_LTCG * max(oth_lt, 0)) * (1 + CESS)
    return {"eq_st": eq_st, "eq_lt": eq_lt, "slab": slab, "other_lt": oth_lt, "taxable_eq_lt": taxable_eq_lt,
            "tax": tax, "exemption_left": max(exemption_left - max(eq_lt, 0), 0.0)}


def realised_eq_ltcg(realised, classes, fy):
    """Equity long-term gains already booked in financial year `fy` (from the statement)."""
    total = 0.0
    for r in realised:
        if classes.get((r.code, r.scheme), "other") != "equity" or financial_year(r.sale_date) != fy:
            continue
        if (r.sale_date - r.buy_date).days > 365:
            total += r.proceeds - r.cost
    return total


def exit_load_units(lots, today, klass, days=365):
    """Units bought within `days` (commonly charged ~1% exit load in equity and hybrid funds)."""
    if klass == "debt":
        return 0.0
    return sum(l.units for l in lots if (today - l.date).days < days)


def locked_units(lots, today, category, name=""):
    """ELSS units are locked for 3 years from each purchase."""
    if "elss" not in (category or "").lower() and "tax saver" not in (name or "").lower():
        return 0.0
    return sum(l.units for l in lots if (today - l.date).days < 3 * 365)
