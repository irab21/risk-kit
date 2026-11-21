"""Tests for CAS import, XIRR, tax lots and capital-gains estimates (offline)."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import transactions as tx  # noqa: E402

T = pd.Timestamp


def cas_dict():
    """Shaped like casparser's dict output for a detailed CAMS statement."""
    def t(d, typ, units, amount):
        return {"date": d, "description": typ, "amount": amount, "units": units, "nav": abs(amount / units),
                "balance": 0, "type": typ}
    return {
        "statement_period": {"from": "01-Jan-2015", "to": "18-Nov-2026"},
        "investor_info": {"name": "TEST INVESTOR", "email": "x@example.com", "address": "-", "mobile": "-"},
        "cas_type": "DETAILED", "file_type": "CAMS",
        "folios": [{"folio": "123/45", "amc": "Alpha Mutual Fund", "PAN": "ABCDE1234F", "schemes": [
            {"scheme": "Alpha Flexi Cap Fund - Direct Growth", "amfi": "111111", "isin": "INF000A01", "rta_code": "A1",
             "rta": "CAMS", "open": 0, "close": 150.0, "close_calculated": 150.0,
             "valuation": {"date": "2026-11-18", "nav": 200.0, "value": 30000.0, "cost": 15000.0},
             "transactions": [t("2016-01-04", "PURCHASE", 100.0, 10000.0), t("2025-06-02", "PURCHASE_SIP", 100.0, 15000.0),
                              t("2026-01-05", "REDEMPTION", -50.0, -9000.0), t("2026-01-05", "STT_TAX", 0.0001, 0.1)]},
            {"scheme": "Beta Old Fund", "amfi": "222222", "rta_code": "B1", "rta": "CAMS", "open": 0, "close": 0,
             "close_calculated": 0, "valuation": {"date": "2024-01-01", "nav": 12.0, "value": 0.0},
             "transactions": [t("2019-03-01", "PURCHASE", 1000.0, 10000.0),
                              t("2024-01-01", "SWITCH_OUT_MERGER", -1000.0, -12000.0)]},
            {"scheme": "Beta Merged Fund", "amfi": "333333", "rta_code": "B2", "rta": "CAMS", "open": 0, "close": 600.0,
             "close_calculated": 600.0, "valuation": {"date": "2026-11-18", "nav": 25.0, "value": 15000.0},
             "transactions": [t("2024-01-01", "SWITCH_IN_MERGER", 600.0, 12000.0)]},
        ]}],
    }


def test_cas_to_frames_drops_personal_details_and_closed_funds():
    holdings, t = tx.cas_to_frames(cas_dict())
    assert list(holdings["code"]) == [111111, 333333]
    assert holdings["value"].sum() == 45000
    assert "PAN" not in t.columns and not t.astype(str).apply(lambda c: c.str.contains("ABCDE1234F")).any().any()
    assert set(t["type"]) >= {"PURCHASE", "REDEMPTION", "SWITCH_IN_MERGER", "STT_TAX"}


def test_xirr_simple_cases():
    assert tx.xirr([(T("2020-01-01"), -100), (T("2021-01-01"), 110)]) == pytest.approx(0.0997, abs=1e-3)
    assert np.isnan(tx.xirr([(T("2020-01-01"), -100)]))


def test_fifo_lots_merger_keeps_original_dates():
    holdings, t = tx.cas_to_frames(cas_dict())
    lots, realised = tx.build_lots(t)
    alpha = lots[(111111, "Alpha Flexi Cap Fund - Direct Growth")]
    assert [round(l.units) for l in alpha] == [50, 100] and alpha[0].date == T("2016-01-04")
    assert realised[0].units == 50 and realised[0].cost == pytest.approx(5000) and realised[0].proceeds == 9000
    merged = lots[(333333, "Beta Merged Fund")]
    assert merged[0].date == T("2019-03-01") and merged[0].units == 600 and merged[0].cost == pytest.approx(10000)


def test_returns_summary_and_same_flows():
    holdings, t = tx.cas_to_frames(cas_dict())
    per_fund, total = tx.returns_summary(t, holdings)
    assert total["invested"] == 35000 and total["withdrawn"] == 9000 and total["value"] == 45000
    assert total["gain"] == 19000 and 0 < total["xirr"] < 0.2
    nav = pd.Series(np.linspace(10, 30, 4000), index=pd.bdate_range("2015-06-01", periods=4000))
    value, rate = tx.same_flows_into(t, nav, as_of=T("2026-11-18"))
    assert value > 0 and rate > 0


def test_equity_tax_short_long_exemption_and_cess():
    lots = [tx.Lot(T("2024-01-10"), 100, 10000), tx.Lot(T("2026-06-01"), 100, 20000)]
    pieces = tx.gains_on_sale(lots, 200, 300.0, T("2026-11-19"), "equity")
    for p in pieces:
        p["klass"] = "equity"
    assert [p["term"] for p in pieces] == ["long", "short"]
    res = tx.tax_on_pieces(pieces, tx.TaxSettings(slab=0.3))
    assert res["eq_lt"] == 20000 and res["eq_st"] == 10000 and res["taxable_eq_lt"] == 0
    assert res["tax"] == pytest.approx(0.20 * 10000 * 1.04)
    big = tx.tax_on_pieces([{**pieces[0], "gain": 200000}], tx.TaxSettings(ltcg_used=25000))
    assert big["taxable_eq_lt"] == 100000 and big["tax"] == pytest.approx(0.125 * 100000 * 1.04)


def test_grandfathering_and_debt_slab():
    old = [tx.Lot(T("2015-01-01"), 100, 1000)]
    p = tx.gains_on_sale(old, 100, 50.0, T("2026-11-19"), "equity", fmv_nav=40.0)[0]
    assert p["cost"] == 4000 and p["gain"] == 1000
    new_debt = tx.gains_on_sale([tx.Lot(T("2023-05-01"), 10, 1000)], 10, 130.0, T("2026-11-19"), "debt")[0]
    assert new_debt["rate"] == "slab"
    old_debt = tx.gains_on_sale([tx.Lot(T("2022-05-01"), 10, 1000)], 10, 130.0, T("2026-11-19"), "debt")[0]
    assert old_debt["term"] == "long" and old_debt["rate"] == 0.125


def test_tax_class_and_locks():
    assert tx.tax_class("Hybrid Scheme - Aggressive Hybrid Fund") == "equity"
    assert tx.tax_class("Debt Scheme - Liquid Fund") == "debt"
    assert tx.tax_class("Other Scheme - Index Funds", "UTI Nifty 50 Index Fund") == "equity"
    assert tx.tax_class("Other Scheme - Index Funds", "Motilal Nasdaq 100 Index") == "other"
    assert tx.tax_class("Other Scheme - Gold ETF", "Nippon Gold BeES") == "other"
    lots = [tx.Lot(T("2024-12-01"), 10, 100), tx.Lot(T("2026-10-01"), 5, 60)]
    assert tx.locked_units(lots, T("2026-11-19"), "Equity Scheme - ELSS") == 15
    assert tx.exit_load_units(lots, T("2026-11-19"), "equity") == 5


def test_generic_transaction_file():
    t = tx.transactions_from_table(pd.read_csv(__import__("io").StringIO(tx.TEMPLATE_CSV)))
    assert list(t["type"]) == ["PURCHASE", "PURCHASE", "REDEMPTION"] and t["code"].iloc[0] == 102885
    lots, realised = tx.build_lots(t)
    assert len(realised) == 1 and realised[0].units == 50
