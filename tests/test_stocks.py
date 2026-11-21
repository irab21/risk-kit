"""Tests for name-based stock search (offline)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import stocks  # noqa: E402


def test_clean_name():
    assert stocks.clean_name("Reliance Industries Limited") == "Reliance Industries"
    assert stocks.clean_name("Apple Inc.") == "Apple"
    assert stocks.clean_name("TATA MOTORS LIMITED") == "Tata Motors"


def test_parse_quotes_filters_and_dedupes():
    quotes = [
        {"symbol": "INFY", "longname": "Infosys Limited", "exchange": "NYQ", "quoteType": "EQUITY"},
        {"symbol": "INFY.NS", "longname": "Infosys Limited", "exchange": "NSI", "quoteType": "EQUITY"},
        {"symbol": "INFY.BO", "longname": "Infosys Limited", "exchange": "BSE", "quoteType": "EQUITY"},
        {"symbol": "INFY.BA", "longname": "Infosys Limited", "exchange": "BUE", "quoteType": "EQUITY"},
        {"symbol": "SAAPL=F", "shortname": "Apple futures", "exchange": "CME", "quoteType": "FUTURE"},
        {"symbol": "NIFTYBEES.NS", "longname": "Nippon India ETF Nifty 50 BeES", "exchange": "NSI", "quoteType": "EQUITY"},
    ]
    out = stocks.parse_quotes(quotes)
    assert [o["ticker"] for o in out] == ["INFY", "INFY.NS", "NIFTYBEES.NS"]
    assert out[1]["where"] == "NSE India" and out[2]["kind"] == "ETF"


def test_known_names():
    assert stocks.name_of("aapl") == "Apple" and stocks.name_of("QQQ") == "Nasdaq 100 ETF"
    assert stocks.lookup_name("HDFCBANK.NS") == "HDFC Bank"  # from the built-in list, no network
