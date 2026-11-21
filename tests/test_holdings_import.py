"""Tests for holdings_import, using files shaped like real broker exports."""
import io
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import holdings_import as hi  # noqa: E402

FIDELITY = b"""Account Number,Account Name,Symbol,Description,Quantity,Last Price,Last Price Change,Current Value,Percent Of Account
Z123,Individual,SPAXX**,HELD IN MONEY MARKET,,,,$1520.33,3.10%
Z123,Individual,VTI,VANGUARD TOTAL STOCK MKT,100,$290.10,+$1.20,$29010.00,59.20%
Z123,Individual,BND,VANGUARD TOTAL BOND MARKET,250,$73.00,-$0.10,$18250.00,37.24%
Z123,Individual,Pending Activity,,,,,-$200.00,
Z456,Roth IRA,VTI,VANGUARD TOTAL STOCK MKT,10,$290.10,+$1.20,$2901.00,100%

"The data and information in this spreadsheet is provided to you solely for your use."
"Date downloaded 11/19/2026"
"""

SCHWAB = b""""Positions for account Individual ...123 as of 09:30 AM ET, 11/19/2026"

"Symbol","Description","Quantity","Price","Price Change %","Market Value","Day Change %"
"AAPL","APPLE INC","50","$230.00","1.2%","$11,500.00","1.2%"
"BRK/B","BERKSHIRE HATHAWAY CL B","10","$480.00","0.1%","$4,800.00","0.1%"
"Cash & Cash Investments","--","--","--","--","$2,000.00","--"
"Account Total","--","--","--","--","$18,300.00","--"
"""

ZERODHA = b"""Instrument,Qty.,Avg. cost,LTP,Cur. val,P&L,Net chg.,Day chg.
RELIANCE,10,2400.5,2950.0,29500.0,5495.0,22.89,0.5
INFY,20,1400,1800,36000,8000,28.57,-0.2
"""

VANGUARD_SHARES_ONLY = b"""Account Number,Investment Name,Symbol,Shares
123,Total Stock,VTI,10
123,Total Intl,VXUS,20
"""


def load(data, name="file.csv"):
    table = hi.read_table(data, name)
    return table, hi.guess_columns(table)


def test_fidelity_skips_cash_pending_and_footer_and_merges_accounts():
    table, cols = load(FIDELITY)
    assert cols["ticker_col"] == "Symbol"
    assert cols["value_col"] == "Current Value"
    assert cols["shares_col"] == "Quantity"
    holdings, skipped = hi.build_holdings(table, **cols)
    vti = holdings.set_index("Ticker").loc["VTI"]
    assert vti["Value"] == pytest.approx(29010 + 2901)  # two accounts merged
    assert set(holdings["Ticker"]) == {"VTI", "BND"}
    assert "money market" in skipped.set_index("Row").loc["SPAXX**", "Reason"]
    assert holdings["Weight %"].sum() == pytest.approx(100)
    assert any("Pending" in r for r in skipped["Row"])


def test_schwab_preamble_totals_and_share_class():
    table, cols = load(SCHWAB)
    assert cols["value_col"] == "Market Value"
    holdings, skipped = hi.build_holdings(table, **cols)
    assert list(holdings["Ticker"]) == ["AAPL", "BRK-B"]
    assert holdings["Value"].sum() == pytest.approx(16300)
    assert {"Cash & Cash Investments", "Account Total"} <= set(skipped["Row"])


def test_zerodha_columns_and_nse_suffix():
    table, cols = load(ZERODHA)
    assert cols == {"ticker_col": "Instrument", "name_col": None, "shares_col": "Qty.", "value_col": "Cur. val",
                    "weight_col": None}
    assert hi.guess_suffix(table) == ".NS"
    holdings, _ = hi.build_holdings(table, **cols, suffix=".NS")
    assert list(holdings["Ticker"]) == ["INFY.NS", "RELIANCE.NS"]


def test_shares_only_uses_prices():
    table, cols = load(VANGUARD_SHARES_ONLY)
    assert cols["value_col"] is None and cols["shares_col"] == "Shares"
    holdings, _ = hi.build_holdings(table, **cols, prices={"VTI": 300.0, "VXUS": 60.0})
    w = holdings.set_index("Ticker")["Weight %"]
    assert w["VTI"] == pytest.approx(100 * 3000 / 4200)


def test_shares_only_without_prices_fails_clearly():
    table, cols = load(VANGUARD_SHARES_ONLY)
    with pytest.raises(ValueError, match="prices"):
        hi.build_holdings(table, **cols)


def test_excel_and_semicolon_files():
    buf = io.BytesIO()
    pd.DataFrame({"Ticker": ["SPY", "AGG"], "Weight": ["60%", "40%"]}).to_excel(buf, index=False)
    table, cols = load(buf.getvalue(), "h.xlsx")
    holdings, _ = hi.build_holdings(table, **cols)
    assert holdings.set_index("Ticker")["Weight %"]["SPY"] == pytest.approx(60)

    table, cols = load(b"Symbol;Market Value\nSPY;1.000\nQQQ;3.000\n")
    assert cols["value_col"] == "Market Value"


def test_numbers_and_tickers():
    s = hi.to_number(pd.Series(["$1,234.50", "(12.5)", "45%", "₹ 10,000", "--", "-3"]))
    assert s.iloc[0] == 1234.5 and s.iloc[1] == -12.5 and s.iloc[2] == 45 and s.iloc[3] == 10000
    assert pd.isna(s.iloc[4]) and s.iloc[5] == -3
    assert hi.clean_ticker(" nse:tcs ", ".NS") == "TCS.NS"
    assert hi.clean_ticker("VOD.L", ".NS") == "VOD.L"


def test_no_ticker_column():
    with pytest.raises(ValueError, match="header"):
        hi.read_table(b"a,b\n1,2\n", "x.csv")


# --- PDF statements (fixtures made by printing HTML statements to PDF) ---
FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


def load_pdf(name):
    return load(open(os.path.join(FIXTURES, name), "rb").read(), name)


@pytest.mark.parametrize("name", ["statement_ruled.pdf", "statement_borderless.pdf"])
def test_pdf_statement_with_and_without_table_lines(name):
    table, cols = load_pdf(name)
    assert cols["ticker_col"] == "Symbol" and cols["value_col"] == "Market Value"
    holdings, skipped = hi.build_holdings(table, **cols)
    values = holdings.set_index("Ticker")["Value"].to_dict()
    assert values == {"VTI": 34812.0, "BND": 21900.0, "AAPL": 9200.0, "MSFT": 6300.0}
    assert holdings.set_index("Ticker")["Shares"]["VTI"] == 120  # quantity and price kept apart
    assert {"SWVXX", "Total"} <= set(skipped["Row"])


def test_pdf_multipage_with_repeated_header():
    table, cols = load_pdf("statement_india_multipage.pdf")
    assert len(table) == 48  # every row on both pages, repeated header removed
    assert hi.guess_suffix(table) == ".NS"
    holdings, _ = hi.build_holdings(table, **cols, suffix=".NS")
    assert len(holdings) == 8 and holdings["Value"].sum() == pytest.approx(6 * 280_800)


def test_pdf_without_text():
    with pytest.raises(ValueError, match="scanned"):
        load_pdf("scanned_blank.pdf")


def test_password_protected_pdf():
    data = open(os.path.join(FIXTURES, "statement_locked.pdf"), "rb").read()  # password: ABCDE1234F
    with pytest.raises(hi.PasswordRequired, match="password-protected"):
        hi.read_table(data, "s.pdf")
    with pytest.raises(hi.PasswordRequired, match="didn't open"):
        hi.read_table(data, "s.pdf", "wrong")
    assert len(hi.read_table(data, "s.pdf", "ABCDE1234F")) == 6


def test_mutual_fund_statement_two_row_header_and_wrapped_names():
    table, cols = load_pdf("mf_statement_sample.pdf")
    assert cols["ticker_col"] is None
    assert (cols["name_col"], cols["shares_col"], cols["value_col"]) == ("Product Name", "Balance Units", "Current Value")
    holdings, _ = hi.build_holdings(table, **cols)
    assert list(holdings["Name"]) == ["Alpha Flexi Cap Fund - Reg - Gr", "Beta Liquid Fund - Dir - Gr",
                                      "Gamma Value Fund - Reg - IDCW"]
    assert holdings["Value"].sum() == pytest.approx(80_200)  # category, investor and total rows left out
    assert (holdings["Ticker"] == "").all()  # to be filled by mutual fund matching
